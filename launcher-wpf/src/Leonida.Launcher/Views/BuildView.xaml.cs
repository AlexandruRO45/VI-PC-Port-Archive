using System.Collections.ObjectModel;
using System.IO;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Threading;
using Leonida.Core;
using Microsoft.Win32;

namespace Leonida.Launcher.Views;

/// <summary>The BUILD tab: what the running (or last) build is doing, measured, and its output.</summary>
public partial class BuildView : UserControl
{
    const int MaxShown = 60000;
    readonly AppState state = AppState.Instance;
    readonly List<LogLine> all = [];
    readonly ObservableCollection<LogLine> shown = [];
    readonly List<ResourceSample> samples = [];
    readonly DispatcherTimer flush = new() { Interval = TimeSpan.FromMilliseconds(150) };
    readonly DispatcherTimer tick = new() { Interval = TimeSpan.FromSeconds(1) };
    bool expanded;
    bool sizing;
    DateTime lastSize = DateTime.MinValue;
    string sizeText = "—";

    public event Action? BuildDialogRequested;

    public BuildView()
    {
        InitializeComponent();
        ConsoleList.ItemsSource = shown;
        flush.Tick += (_, _) => FlushLines();
        tick.Tick += (_, _) => Refresh();
        flush.Start();
        tick.Start();
        Builds.Instance.NewRun += OnNewRun;
        Builds.Instance.Changed += Refresh;
        Loaded += (_, _) => { AutoScroll.IsChecked = state.Settings.Current.AutoScroll; LoadRunningLog(); Refresh(); };
        // A window closed for a language change must stop taking lines from the queue the new one reads.
        Unloaded += (_, _) =>
        {
            flush.Stop();
            tick.Stop();
            Builds.Instance.NewRun -= OnNewRun;
            Builds.Instance.Changed -= Refresh;
        };
    }

    void OnNewRun() { ClearConsole(); samples.Clear(); sizeText = "—"; lastSize = DateTime.MinValue; }

    /// <summary>When this BUILD tab opens on a run already going (the window was reloaded), its output so far comes
    /// from the run's log file; new lines then come from the queue as usual.</summary>
    void LoadRunningLog()
    {
        if (all.Count > 0 || Builds.Instance.LastLogPath is not string path || !File.Exists(path)) return;
        if (state.Session is not { IsRunning: true } && Builds.Instance.Play is not { IsRunning: true }) return;
        try
        {
            using var f = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.ReadWrite | FileShare.Delete);
            using var r = new StreamReader(f);
            var monitor = new BuildMonitor([], null);
            string? line;
            while ((line = r.ReadLine()) != null)
            {
                if (line.StartsWith("# ")) continue;
                var l = monitor.Feed(line, false);
                all.Add(l);
                shown.Add(l);
            }
            Count();
        }
        catch (IOException) { }
    }

    void OnNewBuild(object sender, RoutedEventArgs e) => BuildDialogRequested?.Invoke();

    async void OnStop(object sender, RoutedEventArgs e)
    {
        StopButton.IsEnabled = false;
        await Builds.Instance.StopAsync();
    }

    async void OnCancel(object sender, RoutedEventArgs e)
    {
        if (await Popup.Ask("Cancel the build", "End the build and every process it started now?\n\nStop lets it finish its step cleanly.",
                            "Cancel the build", "Keep building", danger: true))
            Builds.Instance.Cancel();
    }

    // ================================================================ the dashboard

    sealed record StageView(string Name, string Detail, string Icon, Brush Background, Brush Foreground, Brush Border);

    void Refresh()
    {
        ConsoleList.FontSize = state.Settings.Current.ConsoleFontSize;
        var builds = Builds.Instance;
        var s = state.Session;
        var play = builds.Play;
        // The port's run shows here too, when it is newer than the last build.
        var showPlay = play != null && (s == null || play.Started > s.Started);
        var run = showPlay ? play : s;
        if (run == null)
        {
            Empty.Visibility = Visibility.Visible;
            Live.Visibility = Visibility.Collapsed;
            var last = state.History.Entries.FirstOrDefault();
            EmptyText.Text = last == null ? Loc.T("Start one and follow every stage here.")
                : Loc.F("The last build {0} on {1} after {2} (HISTORY has its log).", Loc.T(last.Outcome.ToString()).ToLowerInvariant(),
                        last.Started.ToString("d MMM"), Format.Duration(TimeSpan.FromSeconds(last.Seconds)));
            return;
        }
        Empty.Visibility = Visibility.Collapsed;
        Live.Visibility = Visibility.Visible;
        var running = run.IsRunning;
        StopButton.Visibility = CancelButton.Visibility = running ? Visibility.Visible : Visibility.Collapsed;
        AgainButton.Visibility = running ? Visibility.Collapsed : Visibility.Visible;
        if (!running) StopButton.IsEnabled = true;
        (string text, string brush) badge = run.Outcome switch
        {
            SessionOutcome.Running => (showPlay ? "RUNNING" : "BUILDING", "Pink"),
            SessionOutcome.Succeeded => (showPlay ? "ENDED" : "SUCCEEDED", "Ok"),
            SessionOutcome.Stopped => ("STOPPED", "Warn"),
            SessionOutcome.Cancelled => ("CANCELLED", "Warn"),
            SessionOutcome.CouldNotStart => ("COULD NOT START", "Bad"),
            _ => (showPlay ? Loc.F("EXIT {0}", run.ExitCode) : "FAILED", "Bad"),
        };
        BadgeText.Text = Loc.T(badge.text);
        Badge.Background = (Brush)FindResource(badge.brush);
        var m = run.Monitor;
        var stage = m.Stages.FirstOrDefault(x => x.State == StageState.Running);
        Title.Text = showPlay ? Loc.T("THE PORT") : stage != null ? Loc.F("STAGE {0} OF {1}: {2}", m.Stages.IndexOf(stage) + 1, m.Stages.Count, stage.Name.ToUpperInvariant())
            : running ? Loc.T("STARTING") : Loc.F("{0} OF {1} STAGES DONE", m.Stages.Count(x => x.State == StageState.Done), m.Stages.Count);
        Operation.Text = showPlay ? run.CommandLineText : $"{m.CurrentOperation}   ·   " + Loc.F("started {0}", run.Started.ToString("HH:mm:ss")) +
                                                          (run.Ended is DateTime e ? Loc.F(", ended {0}", e.ToString("HH:mm:ss")) : "");
        Operation.ToolTip = run.CommandLineText;

        StageList.Visibility = m.Stages.Count > 0 && !expanded ? Visibility.Visible : Visibility.Collapsed;
        StatsPanel.Visibility = expanded ? Visibility.Collapsed : Visibility.Visible;
        StageList.ItemsSource = m.Stages.Select(st =>
        {
            (string icon, Brush bg, Brush fg, Brush border) look = st.State switch
            {
                StageState.Running => ("", (Brush)FindResource("Sunset"), Brushes.White, Brushes.Transparent),
                StageState.Done => ("", Brushes.Black, Brushes.White, Brushes.Black),
                StageState.Failed => ("", (Brush)FindResource("Bad"), Brushes.White, Brushes.Transparent),
                StageState.Skipped => ("", Theme.Brush("CardBg"), Theme.Brush("Faint"), Theme.Brush("Line")),
                _ => ("", Theme.Brush("CardBg"), Theme.Brush("Muted"), Theme.Brush("Line")),
            };
            var detail = st.State switch
            {
                StageState.Running => (st.Step.Length > 0 ? st.Step + " · " : "") + Format.Duration(st.Duration ?? TimeSpan.Zero),
                StageState.Done or StageState.Failed => st.Duration is TimeSpan d ? Format.Duration(d) : "",
                StageState.Skipped => Loc.T("not run"),
                _ => Loc.T("waiting"),
            };
            return new StageView(st.Name.ToUpperInvariant(), detail, look.icon, look.bg, look.fg, look.border);
        }).ToList();

        NElapsed.Text = Format.Short(run.Elapsed);
        NWarn.Text = m.Warnings.ToString("N0");
        NErr.Text = m.Errors.ToString("N0");
        if (running && run.LastSample is ResourceSample r)
        {
            if (samples.Count == 0 || samples[^1].Time != r.Time) samples.Add(r);
            if (samples.Count > 300) samples.RemoveAt(0);
            NCpu.Text = $"{r.CpuPercent:0}%";
            NMem.Text = Format.Bytes(r.MemoryBytes);
            NSys.Text = r.SystemTotal > 0 ? $"{100.0 * (r.SystemTotal - r.SystemAvailable) / r.SystemTotal:0}%" : "—";
            NProc.Text = r.Processes.ToString();
            DrawChart();
        }
        else if (!running)
        {
            NCpu.Text = "—";
            NMem.Text = run.PeakMemory > 0 ? Loc.T("peak ") + Format.Bytes(run.PeakMemory) : "—";
            NProc.Text = "0";
        }
        if (!showPlay) _ = WalkSizes(run, force: !running && lastSize < (run.Ended ?? DateTime.MaxValue));
        NSize.Text = showPlay ? "—" : sizeText;

        // Shaders
        var hasShaders = m.Stages.Any(x => x.Name == "shaders");
        ShaderPanel.Visibility = hasShaders && !expanded ? Visibility.Visible : Visibility.Collapsed;
        if (!hasShaders) return;
        var sh = m.Shaders;
        var nums = new[] { ShTotal, ShValid, ShFailed, ShActive, ShRemaining, ShSpeed, ShAvg, ShEta };
        if (sh is { Total: > 0 })
        {
            var total = sh.Total!.Value;
            var done = sh.Done ?? 0;
            ShBar.Value = 100.0 * done / total;
            ShPercent.Text = $"{100.0 * done / total:0.0}%";
            ShTotal.Text = total.ToString("N0");
            ShValid.Text = (sh.Valid ?? 0).ToString("N0");
            ShFailed.Text = (sh.Failed ?? 0).ToString("N0");
            ShActive.Text = running ? sh.ActiveShards.ToString() : "0";
            ShRemaining.Text = (total - done).ToString("N0");
            ShSpeed.Text = sh.ShadersPerSecond is double sp && running ? $"{sp:0.0}/s" : "—";
            ShAvg.Text = sh.SecondsPerShader is double ps ? (ps < 1 ? $"{ps * 1000:0} ms" : $"{ps:0.00} s") : "—";
            ShEta.Text = sh.Eta is TimeSpan eta && running ? Format.Short(eta) : "—";
            ShNote.Text = Loc.F("{0} translator processes", sh.Shards) +
                          (sh.Current != null && running ? Loc.F(" · now {0}", sh.Current) : "") +
                          (m.KeptShards is int k ? Loc.F(" · {0} of {1} shards kept from the run before (not counted again)", k, m.ShardCount) : "") +
                          (sh.DisasmTotal is int dt ? Loc.F(" · disassembly {0}/{1}", (sh.DisasmDone ?? 0).ToString("N0"), dt.ToString("N0")) +
                              (sh.DisasmReused is > 0 ? Loc.F(" ({0} reused)", sh.DisasmReused.Value.ToString("N0")) : "") : "");
        }
        else
        {
            foreach (var t in nums) t.Text = "—";
            if (sh?.DisasmTotal is int dt2 && dt2 > 0)
            {
                // Before the translation: the disassembly (llvm-mc) is the shader step running, with its own numbers.
                var dd = sh.DisasmDone ?? 0;
                ShBar.Value = 100.0 * dd / dt2;
                ShPercent.Text = Loc.F("DISASSEMBLY {0}%", (100.0 * dd / dt2).ToString("0.0"));
                ShTotal.Text = dt2.ToString("N0");
                ShRemaining.Text = (dt2 - dd).ToString("N0");
                ShFailed.Text = (sh.DisasmFailed ?? 0).ToString("N0");
                ShSpeed.Text = sh.DisasmPerSecond is double ds && running ? $"{ds:0.0}/s" : "—";
                ShEta.Text = sh.DisasmEta is TimeSpan de && running ? Format.Short(de) : "—";
                ShNote.Text = Loc.F("Disassembling the native RDNA2 code with llvm-mc: {0} of {1}", dd.ToString("N0"), dt2.ToString("N0")) +
                              (sh.DisasmReused is > 0 ? Loc.F(" ({0} reused from the run before)", sh.DisasmReused.Value.ToString("N0")) : "") +
                              Loc.T(". The translation to SPIR-V follows; its numbers come from the translator.");
                return;
            }
            ShBar.Value = 0;
            ShPercent.Text = "";
            ShNote.Text = m.ShardSummaries > 0 ? Loc.F("From the translator's summaries: {0} shaders, {1} compiled.", m.SummaryShaders.ToString("N0"), m.SummaryValid.ToString("N0"))
                : m.Stages.First(x => x.Name == "shaders").State == StageState.Done ? Loc.T("Nothing to translate: the shaders from the run before were reused.")
                : Loc.T("The numbers appear when the translation starts; they come from the translator itself.");
        }
        if (m.PipelineSummaries > 0) ShNote.Text += Loc.F(" · pipelines {0} ok, {1} failed, {2} skipped", m.PipelinesOk.ToString("N0"), m.PipelinesFailed.ToString("N0"), m.PipelinesSkipped.ToString("N0"));
    }

    void DrawChart()
    {
        double w = Chart.ActualWidth, h = Chart.ActualHeight;
        if (w <= 0 || h <= 0 || samples.Count < 2) return;
        var max = Math.Max(1, samples.Max(x => x.MemoryBytes));
        var cpu = new PointCollection();
        var mem = new PointCollection();
        for (var i = 0; i < samples.Count; i++)
        {
            var x = w * i / 299.0;
            cpu.Add(new Point(x, h - h * samples[i].CpuPercent / 100.0));
            mem.Add(new Point(x, h - h * 0.95 * samples[i].MemoryBytes / max));
        }
        CpuLine.Points = cpu;
        MemLine.Points = mem;
    }

    /// <summary>The port's and the work folder's size, walked on a worker thread at most once a minute.</summary>
    async Task WalkSizes(BuildSession run, bool force)
    {
        if (sizing || (!force && DateTime.Now - lastSize < TimeSpan.FromMinutes(1))) return;
        sizing = true;
        lastSize = DateTime.Now;
        var o = Builds.Instance.LastOptions ?? state.Settings.Current.Build;
        var outF = run.Monitor.OutFolder ?? state.OutFolder(o);
        var workF = run.Monitor.WorkFolder ?? state.WorkFolder(o);
        try
        {
            var (ob, of, wb, wf) = await Task.Run(() =>
            {
                var (b1, f1) = Size(workF, null);
                var (b2, f2) = Size(outF, workF);
                return (b2, f2, b1, f1);
            });
            sizeText = Format.Bytes(ob + wb);
            SizeBox.ToolTip = $"Port: {Format.Bytes(ob)} in {of:N0} files ({outF})\nWork files: {Format.Bytes(wb)} in {wf:N0} files ({workF})";
        }
        finally { sizing = false; }
    }

    static (long, long) Size(string folder, string? skip)
    {
        long bytes = 0, files = 0;
        if (!Directory.Exists(folder)) return (0, 0);
        var full = skip == null ? null : Path.GetFullPath(skip).TrimEnd('\\');
        var stack = new Stack<string>([folder]);
        while (stack.Count > 0)
        {
            var dir = stack.Pop();
            if (full != null && string.Equals(Path.GetFullPath(dir).TrimEnd('\\'), full, StringComparison.OrdinalIgnoreCase)) continue;
            try
            {
                foreach (var e in new DirectoryInfo(dir).EnumerateFileSystemInfos())
                {
                    if (e.Attributes.HasFlag(FileAttributes.ReparsePoint)) continue;
                    if (e is DirectoryInfo) stack.Push(e.FullName);
                    else if (e is FileInfo f) { bytes += f.Length; files++; }
                }
            }
            catch (Exception) { }
        }
        return (bytes, files);
    }

    // ================================================================ the console

    void FlushLines()
    {
        var q = Builds.Instance.Lines;
        if (q.IsEmpty) return;
        var filter = Filter();
        var added = false;
        while (q.TryDequeue(out var l))
        {
            all.Add(l);
            if (filter(l)) { shown.Add(l); added = true; }
        }
        if (all.Count > MaxShown * 2) all.RemoveRange(0, all.Count - MaxShown);
        while (shown.Count > MaxShown) shown.RemoveAt(0);
        Count();
        if (added && AutoScroll.IsChecked == true) ConsoleList.ScrollIntoView(shown[^1]);
    }

    void Count() => ConsoleCount.Text = Loc.F("{0} lines", all.Count.ToString("N0")) + (shown.Count != all.Count ? Loc.F(" · {0} shown", shown.Count.ToString("N0")) : "");

    Func<LogLine, bool> Filter()
    {
        var text = SearchBox.Text.Trim();
        var mode = FilterBox.SelectedIndex;
        return l => (mode switch
        {
            1 => l.Kind is LineKind.Warning or LineKind.Error,
            2 => l.Kind == LineKind.Error,
            3 => l.Kind is LineKind.Info or LineKind.Success or LineKind.Launcher && !l.FromStderr,
            _ => true,
        }) && (text.Length == 0 || l.Text.Contains(text, StringComparison.OrdinalIgnoreCase));
    }

    void OnSearch(object sender, RoutedEventArgs e)
    {
        if (ConsoleList == null) return;
        var f = Filter();
        shown.Clear();
        foreach (var l in all.Where(f).TakeLast(MaxShown)) shown.Add(l);
        Count();
    }

    void ClearConsole() { all.Clear(); shown.Clear(); ConsoleCount.Text = ""; }
    void OnClearConsole(object sender, RoutedEventArgs e) => ClearConsole();
    void OnAutoScroll(object sender, RoutedEventArgs e) => state.Settings.Current.AutoScroll = AutoScroll.IsChecked == true;

    void OnConsoleMenu(object sender, RoutedEventArgs e) { ConsoleMenu.PlacementTarget = ConsoleMenuButton; ConsoleMenu.IsOpen = true; }

    void OnCopyLines(object sender, RoutedEventArgs e)
    {
        var lines = ConsoleList.SelectedItems.Count > 0 ? ConsoleList.SelectedItems.Cast<LogLine>().OrderBy(shown.IndexOf) : shown.AsEnumerable();
        Clipboard.SetText(string.Join(Environment.NewLine, lines.Select(l => l.Text)));
    }

    void OnConsoleKey(object sender, KeyEventArgs e)
    {
        if (e.Key == Key.C && Keyboard.Modifiers.HasFlag(ModifierKeys.Control)) { OnCopyLines(sender, e); e.Handled = true; }
    }

    void OnSaveLog(object sender, RoutedEventArgs e)
    {
        var d = new SaveFileDialog { Filter = "Log (*.log)|*.log|Text (*.txt)|*.txt", FileName = $"leonida-{DateTime.Now:yyyyMMdd-HHmmss}.log" };
        if (d.ShowDialog() != true) return;
        var path = Builds.Instance.LastLogPath;
        if (path != null && File.Exists(path))
        {
            using var src = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.ReadWrite);   // the whole run
            using var dst = File.Create(d.FileName);
            src.CopyTo(dst);
        }
        else File.WriteAllLines(d.FileName, all.Select(l => l.Text));
    }

    void OnOpenLog(object sender, RoutedEventArgs e)
    {
        var path = Builds.Instance.LastLogPath;
        if (path != null && File.Exists(path)) AppState.Open(path); else OnOpenLogs(sender, e);
    }

    void OnOpenLogs(object sender, RoutedEventArgs e) { Directory.CreateDirectory(AppPaths.Logs); AppState.Open(AppPaths.Logs); }

    void OnExpand(object sender, RoutedEventArgs e)
    {
        expanded = !expanded;
        ExpandButton.Content = expanded ? "" : "";
        Refresh();
    }
}
