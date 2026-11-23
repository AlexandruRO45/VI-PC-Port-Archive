using System.Diagnostics;
using System.IO;
using System.Text;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Media;
using Leonida.Core;
using Microsoft.Win32;

namespace Leonida.Launcher.Views;

/// <summary>
/// GPU tests, for contributors: runs scripts/run_gpu_tests.py against the test programs built from this checkout,
/// shows each test's result and drafts the "Tested &lt;commit&gt; on &lt;GPU&gt;" report. It never runs during a build
/// (the two would share the GPU) and uses no game files.
/// </summary>
public partial class GpuTestView : UserControl
{
    readonly AppState state = AppState.Instance;
    BuildSession? run;
    readonly List<string> lines = [];
    List<GpuTestResult> results = [];
    HardwareInfo? hw;
    string commit = "?";
    bool loading;

    public sealed record Row(string Tag, Brush TagBrush, string Name, string Detail, GpuTestOutcome Outcome);

    public GpuTestView()
    {
        InitializeComponent();
        Loaded += async (_, _) => await ActivateAsync();
    }

    public async Task ActivateAsync()
    {
        loading = true;
        var saved = state.Settings.Current.GpuTestPrograms;
        ProgramsBox.Text = GpuTests.HasPrograms(saved) ? saved! : (state.Repo != null ? GpuTests.FindPrograms(state.Repo) : null) ?? saved ?? "";
        loading = false;
        ShowProgramsNote();
        commit = state.Repo != null ? await Task.Run(() => Git(state.Repo, "rev-parse --short=9 HEAD")) ?? "?" : "?";
        Commit.Text = Loc.F("Leonida at {0}", commit);
        hw ??= await Task.Run(HardwareDetector.Detect);
        var g = hw.PrimaryGpu;
        GpuName.Text = g?.Name ?? Loc.T("No graphics card found");
        GpuMore.Text = g == null ? "" : string.Join(" · ", new[] { g.Driver != null ? Loc.F("driver {0}", g.Driver) : null, hw.Os }.Where(x => x != null));
        UpdateButtons();
    }

    static string? Git(string repo, string args)
    {
        try
        {
            var p = Process.Start(new ProcessStartInfo("git", args) { WorkingDirectory = repo, RedirectStandardOutput = true, UseShellExecute = false, CreateNoWindow = true })!;
            var o = p.StandardOutput.ReadToEnd().Trim();
            p.WaitForExit(5000);
            return p.ExitCode == 0 && o.Length > 0 ? o : null;
        }
        catch (Exception) { return null; }
    }

    void ShowProgramsNote()
    {
        var ok = GpuTests.HasPrograms(ProgramsBox.Text);
        ProgramsNote.Foreground = ok ? (Brush)FindResource("Ok") : (Brush)FindResource("Bad");
        ProgramsNote.Text = ok
            ? Loc.T("Found. Rebuild them after pulling: an older build can report false failures.")
            : Loc.T("No vk_compute_test.exe here. Build the test programs first (docs/testing.md: scripts\\build_runtime.ps1), or choose the folder they are in.");
        UpdateButtons();
    }

    void OnProgramsChanged(object sender, TextChangedEventArgs e)
    {
        if (loading) return;
        if (GpuTests.HasPrograms(ProgramsBox.Text))
        {
            state.Settings.Current.GpuTestPrograms = ProgramsBox.Text.Trim();
            try { state.Settings.Save(); } catch (IOException) { }
        }
        ShowProgramsNote();
    }

    void OnBrowse(object sender, RoutedEventArgs e)
    {
        var d = new OpenFolderDialog { Title = Loc.T("The folder with vk_compute_test.exe") };
        if (Directory.Exists(ProgramsBox.Text)) d.InitialDirectory = ProgramsBox.Text;
        if (d.ShowDialog(Window.GetWindow(this)) == true) ProgramsBox.Text = d.FolderName;
    }

    void UpdateButtons()
    {
        var running = run is { IsRunning: true };
        RunButton.IsEnabled = !running && state.Python != null && state.HasRepo && GpuTests.HasPrograms(ProgramsBox.Text);
        RunButton.Content = Loc.T(running ? "RUNNING…" : "RUN THE TESTS");
        StopButton.Visibility = running ? Visibility.Visible : Visibility.Collapsed;
    }

    async void OnRun(object sender, RoutedEventArgs e)
    {
        if (Builds.Instance.Busy) { await Popup.Show("GPU tests", "A build or the port is running. Run the GPU tests when it has finished: they need the graphics card to themselves."); return; }
        if (state.Python == null || state.Repo == null) return;
        lines.Clear();
        results = [];
        Output.Clear();
        ResultCard.Visibility = ReportCard.Visibility = Visibility.Collapsed;
        var args = new List<string> { "-I", "-u", Path.Combine(state.Repo, "scripts", "run_gpu_tests.py"), "--build", ProgramsBox.Text.Trim() };
        if (!string.IsNullOrWhiteSpace(FilterBox.Text)) args.AddRange(["--filter", FilterBox.Text.Trim()]);
        var env = state.BuildEnvironment().ToDictionary(kv => kv.Key, kv => (string?)kv.Value);
        Directory.CreateDirectory(AppPaths.Logs);
        run = new BuildSession(state.Python.Executable, args, state.Repo, Path.Combine(AppPaths.Logs, $"gpu-tests-{DateTime.Now:yyyyMMdd-HHmmss}.log"),
                               new BuildMonitor([], null), env, state.OutputEncoding());
        var buffer = new StringBuilder();
        run.Line += l => { lock (buffer) buffer.AppendLine(l.Text); };
        run.Exited += s => Dispatcher.BeginInvoke(() => Finished(s));
        var tick = new System.Windows.Threading.DispatcherTimer { Interval = TimeSpan.FromMilliseconds(250) };
        tick.Tick += (_, _) =>
        {
            string add;
            lock (buffer) { add = buffer.ToString(); buffer.Clear(); }
            if (add.Length == 0) { if (run is not { IsRunning: true }) tick.Stop(); return; }
            lines.AddRange(add.Split('\n').Select(x => x.TrimEnd('\r')));
            Output.AppendText(add);
            Output.ScrollToEnd();
        };
        tick.Start();
        OutputExpander.IsExpanded = true;
        run.Start();
        UpdateButtons();
    }

    void Finished(BuildSession s)
    {
        // The last lines may still be in the buffer: let the timer flush them, then read the results.
        Dispatcher.BeginInvoke(() =>
        {
            results = GpuTests.Parse(lines);
            ShowResults(s);
            UpdateButtons();
        }, System.Windows.Threading.DispatcherPriority.ContextIdle);
    }

    void ShowResults(BuildSession s)
    {
        int pass = results.Count(r => r.Outcome == GpuTestOutcome.Pass), fail = results.Count(r => r.Outcome == GpuTestOutcome.Fail), skip = results.Count(r => r.Outcome == GpuTestOutcome.Skip);
        NPass.Text = pass.ToString(); NFail.Text = fail.ToString(); NSkip.Text = skip.ToString();
        NTime.Text = Format.Duration(s.Elapsed);
        ResultTitle.Text = results.Count == 0
            ? Loc.T(s.Outcome == SessionOutcome.Cancelled || s.Outcome == SessionOutcome.Stopped ? "Stopped before the results" : "No results: see the output below")
            : Loc.F("{0} of {1} passed", pass, results.Count);
        ResultCard.Visibility = Visibility.Visible;
        ShowRows();
        if (results.Count > 0)
        {
            var g = hw?.PrimaryGpu;
            ReportBox.Text = GpuTests.Report(commit, g?.Name ?? "?", g?.Driver, hw?.Os ?? "Windows", FilterBox.Text, results);
            ReportCard.Visibility = Visibility.Visible;
        }
        else OutputExpander.IsExpanded = true;
    }

    void ShowRows()
    {
        Brush ok = (Brush)FindResource("Ok"), bad = (Brush)FindResource("Bad"), skip = (Brush)FindResource("Faint");
        Results.ItemsSource = results
            .Where(r => ShowFailed.IsChecked != true || r.Outcome == GpuTestOutcome.Fail)
            .Select(r => new Row(r.Outcome.ToString().ToUpperInvariant(), r.Outcome switch { GpuTestOutcome.Pass => ok, GpuTestOutcome.Fail => bad, _ => skip }, r.Name, r.Detail, r.Outcome))
            .ToList();
    }

    void OnShowFilter(object sender, RoutedEventArgs e) { if (Results != null) ShowRows(); }

    async void OnStop(object sender, RoutedEventArgs e)
    {
        if (run is { IsRunning: true } r) await r.StopAsync(TimeSpan.FromSeconds(10));
    }

    void OnCopy(object sender, RoutedEventArgs e)
    {
        try { Clipboard.SetText(ReportBox.Text); } catch (System.Runtime.InteropServices.COMException) { }
    }

    /// <summary>Stops a run when the launcher closes.</summary>
    public void Shutdown() => run?.Dispose();
}
