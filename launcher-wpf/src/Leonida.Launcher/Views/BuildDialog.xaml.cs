using System.IO;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Input;
using Leonida.Core;
using Microsoft.Win32;

namespace Leonida.Launcher.Views;

/// <summary>
/// The build popup: the game copy, where the port goes and what to build up front; every other make_game.py option
/// under Advanced. START BUILD hands the options to <see cref="Builds"/>.
/// </summary>
public partial class BuildDialog : UserControl
{
    static readonly string[] RunStages = ["check", "shaders", "runtime"];
    readonly AppState state = AppState.Instance;
    bool loading = true;   // until the controls exist: the slider reports a value while the XAML loads

    public event Action? Closed;
    public event Action? Started;

    public BuildDialog()
    {
        InitializeComponent();
        loading = false;
    }

    public void Open(BuildOptions o, string? choice = null)
    {
        loading = true;
        GameBox.Text = o.Game ?? "";
        OutBox.ItemsSource = state.Settings.Current.RecentOutputs.ToList();
        OutBox.Text = o.Out ?? "";
        WorkBox.Text = o.Work ?? "";
        StageCheck.IsChecked = o.Stages.Contains("check");
        StageShaders.IsChecked = o.Stages.Contains("shaders");
        StageAssets.IsChecked = o.Stages.Contains("assets");
        StageRuntime.IsChecked = o.Stages.Contains("runtime");
        StagePipelines.IsChecked = o.Stages.Contains("pipelines");
        FullCheck.IsChecked = o.FullCheck;
        Wav.IsChecked = o.Wav;
        Wave64Box.SelectedIndex = Math.Max(0, Array.IndexOf(BuildOptions.Wave64Modes, o.Wave64Emulate));
        NoFloatControls.IsChecked = o.NoFloatControls;
        BaryFallback.IsChecked = o.BaryFallback;
        TexelOffsetFallback.IsChecked = o.TexelOffsetFallback;
        F16InterpFallback.IsChecked = o.F16InterpFallback;
        MeshCompute.IsChecked = o.MeshCompute;
        PcTailUbo.IsChecked = o.PcTailUbo;
        JobsSlider.Maximum = Math.Max(2, Environment.ProcessorCount);
        JobsDefault.IsChecked = o.Jobs == null;
        JobsSlider.Value = o.Jobs ?? state.Probe?.DefaultJobs ?? Math.Max(1, Environment.ProcessorCount / 2);
        RuntimeBuildBox.Text = o.RuntimeBuild ?? "";
        ExtraBox.Text = o.ExtraArguments ?? "";
        var stages = o.OrderedStages();
        choice ??= o.CheckOnly ? "check"
            : stages.SequenceEqual(RunStages) ? "run"
            : stages.Count == BuildOptions.AllStages.Length ? "all" : "custom";
        (choice switch { "check" => ChoiceCheck, "all" => ChoiceAll, "custom" => ChoiceCustom, _ => ChoiceRun }).IsChecked = true;
        StagePanel.Visibility = choice == "custom" ? Visibility.Visible : Visibility.Collapsed;
        Advanced.IsExpanded = false;
        ReloadPresets();
        loading = false;
        Refresh();
        Visibility = Visibility.Visible;
        if (string.IsNullOrWhiteSpace(GameBox.Text)) _ = FindAsync(auto: true);
    }

    BuildOptions Read()
    {
        static string? N(string? s) => string.IsNullOrWhiteSpace(s) ? null : s.Trim().Trim('"');
        List<string> stages;
        if (ChoiceRun.IsChecked == true) stages = [.. RunStages];
        else if (ChoiceAll.IsChecked == true || ChoiceCheck.IsChecked == true) stages = [.. BuildOptions.AllStages];
        else
        {
            stages = [];
            if (StageCheck.IsChecked == true) stages.Add("check");
            if (StageShaders.IsChecked == true) stages.Add("shaders");
            if (StageAssets.IsChecked == true) stages.Add("assets");
            if (StageRuntime.IsChecked == true) stages.Add("runtime");
            if (StagePipelines.IsChecked == true) stages.Add("pipelines");
        }
        return new BuildOptions
        {
            Game = N(GameBox.Text), Out = N(OutBox.Text), Work = N(WorkBox.Text), Stages = stages,
            CheckOnly = ChoiceCheck.IsChecked == true, FullCheck = FullCheck.IsChecked == true, Wav = Wav.IsChecked == true,
            Jobs = JobsDefault.IsChecked == true ? null : (int)JobsSlider.Value,
            RuntimeBuild = N(RuntimeBuildBox.Text),
            Wave64Emulate = (Wave64Box.SelectedItem as ComboBoxItem)?.Tag as string ?? "auto",
            NoFloatControls = NoFloatControls.IsChecked == true, BaryFallback = BaryFallback.IsChecked == true,
            TexelOffsetFallback = TexelOffsetFallback.IsChecked == true, F16InterpFallback = F16InterpFallback.IsChecked == true,
            MeshCompute = MeshCompute.IsChecked == true, PcTailUbo = PcTailUbo.IsChecked == true,
            ExtraArguments = N(ExtraBox.Text),
        };
    }

    void Refresh()
    {
        if (loading) return;
        var o = Read();
        JobsValue.Text = JobsDefault.IsChecked == true ? Loc.F("default ({0})", state.Probe?.DefaultJobs ?? Math.Max(1, Environment.ProcessorCount / 2)) : ((int)JobsSlider.Value).ToString();
        CommandPreview.Text = CommandLine.Join(["python", "-I", "make_game.py", .. o.ToArguments()]);
        var outF = state.OutFolder(o);
        var workF = state.WorkFolder(o);
        OutNote.Text = (o.Out == null ? Loc.T("Default: ") : "") + outF + Free(outF) + Need(o);
        // make_game.py's preflight decides (it counts what an earlier build already wrote); this only warns early.
        var short_ = !o.CheckOnly && state.Probe != null && FreeBytes(workF) is long free &&
                     free < (long)o.Stages.Sum(s => state.Probe.MinFreeGb.GetValueOrDefault(s)) << 30;
        OutNote.Foreground = (System.Windows.Media.Brush)FindResource(short_ ? "Warn" : "Faint");
        if (short_) OutNote.Text += Environment.NewLine + Loc.T("That drive looks too small for this choice: pick a folder on a bigger drive, or Ready to run.");
        WorkNote.Text = (o.Work == null ? Loc.T("Default: ") : "") + workF + Free(workF);
        var problems = Builds.Instance.Problems(o);
        Problems.ItemsSource = problems;
        StartButton.IsEnabled = problems.Count == 0;
        StartButton.Content = Loc.T(o.CheckOnly ? "CHECK FILES" : "START BUILD");
    }

    string Need(BuildOptions o)
    {
        if (o.CheckOnly || state.Probe == null) return "";
        var gb = o.Stages.Sum(s => state.Probe.MinFreeGb.GetValueOrDefault(s));
        return gb > 0 ? Loc.F("  ·  needs about {0} GB (less when an earlier build is reused)", gb) : "";
    }

    static long? FreeBytes(string path)
    {
        try { var d = new DriveInfo(Path.GetPathRoot(Path.GetFullPath(path))!); return d.IsReady ? d.AvailableFreeSpace : null; }
        catch (Exception) { return null; }
    }

    static string Free(string path)
    {
        try
        {
            var d = new DriveInfo(Path.GetPathRoot(Path.GetFullPath(path))!);
            return d.IsReady ? Loc.F("  ·  {0} free", Format.Bytes(d.AvailableFreeSpace)) : "";
        }
        catch (Exception) { return ""; }
    }

    void OnChanged(object sender, RoutedEventArgs e) => Refresh();

    void OnJobs(object sender, RoutedPropertyChangedEventArgs<double> e)
    {
        if (!loading && JobsDefault.IsChecked == true && Math.Abs(e.NewValue - e.OldValue) >= 1) JobsDefault.IsChecked = false;
        Refresh();
    }

    void OnChoice(object sender, RoutedEventArgs e)
    {
        StagePanel.Visibility = ChoiceCustom.IsChecked == true ? Visibility.Visible : Visibility.Collapsed;
        Refresh();
    }

    // ---- folders and copies ----

    static string? Pick(string title, string? start)
    {
        var d = new OpenFolderDialog { Title = title };
        if (!string.IsNullOrWhiteSpace(start) && Directory.Exists(start)) d.InitialDirectory = start;
        return d.ShowDialog() == true ? d.FolderName : null;
    }

    void OnBrowseGame(object sender, RoutedEventArgs e) { if (Pick("Your game copy (the folder with MicrosoftGame.Config)", GameBox.Text) is string f) GameBox.Text = f; }
    void OnBrowseOut(object sender, RoutedEventArgs e) { if (Pick("Where the port goes", OutBox.Text) is string f) OutBox.Text = f; }
    void OnBrowseWork(object sender, RoutedEventArgs e) { if (Pick("Folder for intermediate files", WorkBox.Text) is string f) WorkBox.Text = f; }
    void OnBrowseRuntime(object sender, RoutedEventArgs e) { if (Pick("A Windows runtime build", RuntimeBuildBox.Text) is string f) RuntimeBuildBox.Text = f; }

    async void OnFind(object sender, RoutedEventArgs e) => await FindAsync(auto: false);

    async Task FindAsync(bool auto)
    {
        if (state.Python == null || state.Repo == null) return;
        FindButton.IsEnabled = false;
        GameNote.Text = Loc.T("Looking for your copy where Build.bat looks…");
        try
        {
            var r = await Probe.RunAsync(state.Python, state.Repo, ["--find"], state.BuildEnvironment());
            var copies = r.Copies ?? [];
            CopiesBox.ItemsSource = copies.Select(c => $"{c.Path}   ({c.Title} {c.Version})").ToList();
            CopiesBox.Tag = copies;
            CopiesBox.Visibility = copies.Count > 1 ? Visibility.Visible : Visibility.Collapsed;
            GameNote.Text = copies.Count switch
            {
                0 => Loc.T("No copy found automatically: choose the folder that holds MicrosoftGame.Config."),
                1 => Loc.F("Found: {0} {1}.", copies[0].Title, copies[0].Version),
                _ => Loc.F("Found {0} copies: choose one below.", copies.Count),
            } + ((r.Unusable?.Count ?? 0) > 0 ? Loc.T(" Packaged containers (.xvc/.msixvc) can't be used: the build needs the unpacked folder.") : "");
            if (copies.Count >= 1 && (copies.Count == 1 || !auto) && string.IsNullOrWhiteSpace(GameBox.Text)) GameBox.Text = copies[0].Path;
        }
        catch (Exception ex) { GameNote.Text = Loc.T("The search failed: ") + ex.Message; }
        finally { FindButton.IsEnabled = true; }
    }

    void OnCopyChosen(object sender, SelectionChangedEventArgs e)
    {
        if (CopiesBox.Tag is List<GameCopy> c && CopiesBox.SelectedIndex >= 0) GameBox.Text = c[CopiesBox.SelectedIndex].Path;
    }

    // ---- presets ----

    void ReloadPresets()
    {
        var names = new List<string> { Loc.T("Load a preset…") };
        names.AddRange(state.Settings.Current.Presets.Select(p => p.Name));
        PresetBox.ItemsSource = names;
        PresetBox.SelectedIndex = 0;
    }

    void OnPreset(object sender, SelectionChangedEventArgs e)
    {
        if (loading || PresetBox.SelectedIndex <= 0) return;
        var keepOpen = Advanced.IsExpanded;
        Open(state.Settings.Current.Presets[PresetBox.SelectedIndex - 1].Options.Clone());
        Advanced.IsExpanded = keepOpen;
    }

    void OnSavePreset(object sender, RoutedEventArgs e)
    {
        var name = Prompt.Ask(Window.GetWindow(this)!, "Save as preset", "Name:", $"Preset {state.Settings.Current.Presets.Count + 1}");
        if (string.IsNullOrWhiteSpace(name)) return;
        state.Settings.Current.Presets.RemoveAll(p => p.Name == name.Trim());
        state.Settings.Current.Presets.Add(new Preset { Name = name.Trim(), Options = Read() });
        state.Settings.Save();
        ReloadPresets();
    }

    void OnDeletePreset(object sender, RoutedEventArgs e)
    {
        if (PresetBox.SelectedIndex <= 0) { MessageBox.Show(Loc.T("Choose the preset in the list first."), "Leonida Launcher"); return; }
        state.Settings.Current.Presets.RemoveAt(PresetBox.SelectedIndex - 1);
        state.Settings.Save();
        ReloadPresets();
    }

    void OnCopyCommand(object sender, RoutedEventArgs e) => Clipboard.SetText(CommandPreview.Text);

    // ---- open / close / start ----

    void OnStart(object sender, RoutedEventArgs e)
    {
        var o = Read();
        if (!Builds.Instance.Start(o)) { Refresh(); return; }
        Visibility = Visibility.Collapsed;
        Started?.Invoke();
    }

    void OnClose(object sender, RoutedEventArgs e) => Close();

    public void Close()
    {
        if (!loading) state.Settings.Current.Build = Read();   // keep what was chosen, even without building
        Visibility = Visibility.Collapsed;
        Closed?.Invoke();
    }

    void OnBackdrop(object sender, MouseButtonEventArgs e) { if (e.OriginalSource == Backdrop) Close(); }
    void OnPanelClick(object sender, MouseButtonEventArgs e) => e.Handled = true;
    void OnBackdropSize(object sender, SizeChangedEventArgs e) => Panel.MaxHeight = Math.Max(300, e.NewSize.Height - 60);

    protected override void OnKeyDown(KeyEventArgs e)
    {
        if (e.Key == Key.Escape) { Close(); e.Handled = true; }
        base.OnKeyDown(e);
    }
}
