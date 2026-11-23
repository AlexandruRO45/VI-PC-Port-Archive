using System.IO;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Media;
using Leonida.Core;
using Microsoft.Win32;

namespace Leonida.Launcher.Views;

/// <summary>
/// The settings page, opened from the avatar menu: Launcher settings (language, start page, avatar, online
/// features, build behaviour, logs) and Leonida settings (the setup check, Play's options, folders, environment).
/// Back closes it.
/// </summary>
public partial class SettingsView : UserControl
{
    static readonly (string Key, string Name)[] StartPages =
        [("last", "Where you left off"), ("home", "Home"), ("leonida", "Leonida"), ("caniru", "Can I Run Leonida")];
    readonly AppState state = AppState.Instance;
    bool loading = true;

    public event Action? BackRequested;
    public event Action? LanguageChanged;
    public event Action? AvatarChanged;

    public SettingsView()
    {
        InitializeComponent();
        SettingsPath.Text = AppPaths.Settings;
        var v = typeof(SettingsView).Assembly.GetName().Version?.ToString(3) ?? "?";
        SideVersion.Text = "v" + v;
        AboutVersion.Text = Loc.F("Leonida Launcher {0} · .NET {1} · Windows", v, Environment.Version.ToString(2));
    }

    /// <summary>The pages: (key, nav button, panel, title). "launcher" and "leonida" open each group's first page.</summary>
    (string Key, RadioButton Nav, FrameworkElement Page, string Title)[] AllPages =>
    [
        ("general", NavGeneral, PageGeneral, "General"), ("builds", NavBuilds, PageBuilds, "Builds"),
        ("online", NavOnline, PageOnline, "Online"), ("about", NavAbout, PageAbout, "About and files"),
        ("setup", NavSetup, PageSetup, "Your setup"), ("play", NavPlay, PagePlay, "Play"),
        ("folders", NavFolders, PageFolders, "Folders and Python"), ("disk", NavDisk, PageDisk, "Disk space"),
    ];

    bool switching;

    public void ShowSection(string section)
    {
        section = section switch { "launcher" => "general", "leonida" => "setup", _ => section };
        var pages = AllPages;
        if (!pages.Any(p => p.Key == section)) section = "general";
        switching = true;
        foreach (var p in pages)
        {
            bool on = p.Key == section;
            p.Page.Visibility = on ? Visibility.Visible : Visibility.Collapsed;
            if (on) { p.Nav.IsChecked = true; PageTitle.Text = Loc.T(p.Title); }
        }
        switching = false;
        Pages.ScrollToTop();
        if (section == "setup") ShowChecks();
        if (section == "disk") _ = ShowDiskAsync();
    }

    void OnSection(object sender, RoutedEventArgs e)
    {
        if (!switching && PageTitle != null && sender is RadioButton { Tag: string key }) ShowSection(key);
    }

    void OnBack(object sender, RoutedEventArgs e) => BackRequested?.Invoke();

    public void Initialize()
    {
        loading = true;
        var c = state.Settings.Current;
        // Launcher
        LanguageBox.ItemsSource = Loc.Languages.Select(l => l.Name).ToList();
        LanguageBox.SelectedIndex = Math.Max(0, Loc.Languages.ToList().FindIndex(l => l.Code == c.Language));
        ThemeBox.ItemsSource = Theme.Choices.Select(t => Loc.T(t.Name)).ToList();
        ThemeBox.SelectedIndex = Math.Max(0, Array.FindIndex(Theme.Choices, t => t.Code == c.Theme));
        StartPageBox.ItemsSource = StartPages.Select(p => Loc.T(p.Name)).ToList();
        StartPageBox.SelectedIndex = Math.Max(0, Array.FindIndex(StartPages, p => p.Key == c.StartPage));
        StartCollapsed.IsChecked = c.StartCollapsed;
        AllowGitHub.IsChecked = c.AllowGitHub;
        AllowChecker.IsChecked = c.AllowChecker;
        StopGracefully.IsChecked = c.StopGracefully;
        GraceBox.Text = c.StopGraceSeconds.ToString();
        NotifyDone.IsChecked = c.NotifyWhenDone;
        SoundDone.IsChecked = c.SoundWhenDone;
        ToastDone.IsChecked = c.ToastWhenDone;
        CloseToTray.IsChecked = c.CloseToTray;
        FontSlider.Value = c.ConsoleFontSize;
        KeepLogsBox.Text = c.KeepLogsDays.ToString();
        AvatarPreview.Content = Avatar.Current(44);
        // Leonida
        OverlayBox.Text = c.Play.Overlay ?? "";
        ModsBox.Text = c.Play.Mods ?? "";
        VulkanBackend.IsChecked = c.Play.VulkanBackend;
        TraceD3D12.IsChecked = c.Play.TraceD3D12;
        TraceFileBox.Text = c.Play.TraceFile ?? "";
        GameArgsBox.Text = c.Play.GameArguments ?? "";
        EnvBox.Text = string.Join(Environment.NewLine, c.Environment.Select(kv => $"{kv.Key}={kv.Value}"));
        PythonBox.Text = c.PythonOverride ?? "";
        RepoBox.Text = state.Repo ?? "—";
        ShowPython();
        ShowChecks();
        loading = false;
        OnFont(this, new RoutedPropertyChangedEventArgs<double>(0, FontSlider.Value));
    }

    // ================================================================ launcher settings

    void OnChanged(object sender, RoutedEventArgs e)
    {
        if (loading) return;
        var c = state.Settings.Current;
        c.StartPage = StartPages[Math.Max(0, StartPageBox.SelectedIndex)].Key;
        c.StartCollapsed = StartCollapsed.IsChecked == true;
        c.AllowGitHub = AllowGitHub.IsChecked == true;
        c.AllowChecker = AllowChecker.IsChecked == true;
        c.StopGracefully = StopGracefully.IsChecked == true;
        c.NotifyWhenDone = NotifyDone.IsChecked == true;
        c.SoundWhenDone = SoundDone.IsChecked == true;
        c.ToastWhenDone = ToastDone.IsChecked == true;
        c.CloseToTray = CloseToTray.IsChecked == true;
        if (int.TryParse(GraceBox.Text, out var g) && g is >= 5 and <= 600) c.StopGraceSeconds = g;
        if (int.TryParse(KeepLogsBox.Text, out var d) && d is >= 0 and <= 3650) c.KeepLogsDays = d;
        Save();
    }

    void OnLanguage(object sender, SelectionChangedEventArgs e)
    {
        if (loading || LanguageBox.SelectedIndex < 0) return;
        var code = Loc.Languages[LanguageBox.SelectedIndex].Code;
        if (code == state.Settings.Current.Language) return;
        state.Settings.Current.Language = code;
        Save();
        LanguageChanged?.Invoke();   // the window reloads in the new language
    }

    void OnTheme(object sender, SelectionChangedEventArgs e)
    {
        if (loading || ThemeBox.SelectedIndex < 0) return;
        var code = Theme.Choices[ThemeBox.SelectedIndex].Code;
        if (code == state.Settings.Current.Theme) return;
        state.Settings.Current.Theme = code;
        Save();
        Theme.Apply(code);
        LanguageChanged?.Invoke();   // the window reloads, so what code drew (README pages, lists) takes the new colours too
    }

    void OnFont(object sender, RoutedPropertyChangedEventArgs<double> e)
    {
        if (FontValue == null || FontPreview == null || FontSlider == null) return;   // the slider reports a value while the XAML loads
        FontValue.Text = FontSlider.Value.ToString("0.#");
        FontPreview.FontSize = FontSlider.Value;
        if (loading) return;
        state.Settings.Current.ConsoleFontSize = FontSlider.Value;
        Save();
    }

    void OnNewAvatar(object sender, RoutedEventArgs e)
    {
        state.Settings.Current.AvatarSeed = Random.Shared.Next();
        state.Settings.Current.AvatarCustom = false;
        Save();
        AvatarPreview.Content = Avatar.Current(44);
        AvatarChanged?.Invoke();
    }

    async void OnUploadAvatar(object sender, RoutedEventArgs e)
    {
        var d = new OpenFileDialog
        {
            Title = Loc.T("Choose a picture"),
            Filter = Loc.T("Pictures") + " (*.png;*.jpg;*.jpeg;*.bmp;*.gif)|*.png;*.jpg;*.jpeg;*.bmp;*.gif",
        };
        if (d.ShowDialog(Window.GetWindow(this)) != true) return;
        try { Avatar.SaveCustom(d.FileName); }
        catch (Exception ex)
        {
            await Popup.Show("Could not use that picture", ex.Message, problem: true);
            return;
        }
        state.Settings.Current.AvatarCustom = true;
        Save();
        AvatarPreview.Content = Avatar.Current(44);
        AvatarChanged?.Invoke();
    }

    void Save() { try { state.Settings.Save(); } catch (IOException) { } }

    // ================================================================ disk space

    /// <summary>One row of the Disk space page; the size fills in when it has been measured.</summary>
    public sealed class DiskRow(DiskEntry e) : System.ComponentModel.INotifyPropertyChanged
    {
        public DiskEntry Entry { get; } = e;
        public string Title => Loc.T(Entry.Title);
        public string Description => Loc.T(Entry.Description);
        public string Path => Entry.Path;
        public string Size => measured ? Format.Bytes(Entry.Bytes) : "…";
        public string FileCount => measured ? Loc.F("{0:N0} files", Entry.Files) : Loc.T("measuring");
        public string OpenText => Loc.T("Open");
        public string DeleteText => Loc.T("Delete");
        public Visibility DeleteVisibility => Entry.Deletable ? Visibility.Visible : Visibility.Collapsed;
        public bool CanDelete => measured && Entry.Bytes > 0 && !Builds.Instance.Busy;
        bool measured;
        public void Measured() { measured = true; Changed(); }
        public void Changed() => PropertyChanged?.Invoke(this, new(null));
        public event System.ComponentModel.PropertyChangedEventHandler? PropertyChanged;
    }

    CancellationTokenSource? diskCancel;

    async Task ShowDiskAsync()
    {
        diskCancel?.Cancel();
        var cancel = (diskCancel = new CancellationTokenSource()).Token;
        var o = state.Settings.Current.Build;
        string outFolder = state.OutFolder(o), work = state.WorkFolder(o);
        var rows = DiskUsage.Entries(outFolder, work, AppPaths.Logs, AppPaths.WebView).Select(e => new DiskRow(e)).ToList();
        DiskList.ItemsSource = rows;
        ShowDrive(outFolder, null);
        long total = 0;
        foreach (var r in rows)
        {
            try
            {
                var skip = r.Entry.Deletable ? null : work;   // the port's row leaves out the work folder, listed on its own
                var (b, f) = await Task.Run(() => DiskUsage.Measure(r.Entry.Path, skip, cancel), cancel);
                r.Entry.Bytes = b; r.Entry.Files = f;
                r.Measured();
                if (!r.Entry.Path.StartsWith(System.IO.Path.GetDirectoryName(AppPaths.Logs)!, StringComparison.OrdinalIgnoreCase)) total += b;
                ShowDrive(outFolder, total);
            }
            catch (OperationCanceledException) { return; }
        }
    }

    void ShowDrive(string folder, long? leonida)
    {
        try
        {
            var drive = new DriveInfo(System.IO.Path.GetPathRoot(System.IO.Path.GetFullPath(folder))!);
            long size = drive.TotalSize, free = drive.AvailableFreeSpace, used = size - free, mine = Math.Min(leonida ?? 0, used);
            DiskDrive.Text = Loc.F("{0} free of {1} on {2}", Format.Bytes(free), Format.Bytes(size), drive.Name.TrimEnd('\\'));
            DiskSummary.Text = leonida is long l ? Loc.F("The build's files use {0} of it. A full build needs about 30 GB (everything: about 330 GB).", Format.Bytes(l))
                                                 : Loc.T("Measuring what the build has written…");
            BarLeonida.Width = new GridLength(mine, GridUnitType.Star);
            BarOther.Width = new GridLength(used - mine, GridUnitType.Star);
            BarFree.Width = new GridLength(free, GridUnitType.Star);
        }
        catch (Exception e) when (e is IOException or ArgumentException or UnauthorizedAccessException)
        {
            DiskDrive.Text = folder;
            DiskSummary.Text = Loc.T("The drive could not be read.");
        }
    }

    void OnDiskRefresh(object sender, RoutedEventArgs e) => _ = ShowDiskAsync();

    void OnDiskOpen(object sender, RoutedEventArgs e)
    {
        if ((sender as FrameworkElement)?.Tag is DiskRow r && Directory.Exists(r.Path)) AppState.Open(r.Path);
    }

    async void OnDiskDelete(object sender, RoutedEventArgs e)
    {
        if ((sender as FrameworkElement)?.Tag is not DiskRow r) return;
        if (Builds.Instance.Busy) { await Popup.Show("Disk space", "Nothing can be deleted while a build or the port runs."); return; }
        var o = state.Settings.Current.Build;
        var roots = new[] { state.WorkFolder(o), AppPaths.Logs, AppPaths.WebView };
        if (DiskUsage.RefuseDelete(r.Path, roots, o.Game) is string why)
        {
            await Popup.Show("Not deleted", Loc.F("The launcher won't delete {0}: {1}.", r.Path, Loc.T(why)), problem: true);
            return;
        }
        var text = Loc.F("Delete {0} ({1})?", r.Title, Format.Bytes(r.Entry.Bytes)) + "\n\n" + (r.Entry.Warning != null ? Loc.T(r.Entry.Warning) : r.Description);
        if (!await Popup.Ask("Delete files", text, Loc.F("Delete {0}", Format.Bytes(r.Entry.Bytes)), danger: true)) return;
        var logs = r.Path.Equals(AppPaths.Logs, StringComparison.OrdinalIgnoreCase);
        var failed = await Task.Run(() => DeleteFolder(r.Path, keepFolder: logs));
        if (failed > 0) await Popup.Show("Disk space", Loc.F("{0} file(s) were in use and stay.", failed));
        await ShowDiskAsync();
    }

    /// <summary>Deletes what is in a folder (and the folder, unless asked to keep it); files in use stay. Links are
    /// removed, never followed.</summary>
    static int DeleteFolder(string folder, bool keepFolder)
    {
        int failed = 0;
        var dir = new DirectoryInfo(folder);
        foreach (var f in dir.EnumerateFileSystemInfos())
        {
            try
            {
                if (f is DirectoryInfo d && !d.Attributes.HasFlag(FileAttributes.ReparsePoint)) failed += DeleteFolder(d.FullName, keepFolder: false);
                else if (f is DirectoryInfo link) link.Delete();
                else { f.Attributes = FileAttributes.Normal; f.Delete(); }
            }
            catch (Exception e) when (e is IOException or UnauthorizedAccessException) { failed++; }
        }
        if (!keepFolder && failed == 0)
            try { dir.Delete(); } catch (Exception e) when (e is IOException or UnauthorizedAccessException) { failed++; }
        return failed;
    }

    async void OnReset(object sender, RoutedEventArgs e)
    {
        if (!await Popup.Ask("Restore defaults", "Restore every launcher setting and build option to its default? Presets and recent folders go too; the build history and logs stay.",
                             "Restore defaults", danger: true)) return;
        var language = state.Settings.Current.Language;
        state.Settings.Reset();
        Initialize();
        AvatarChanged?.Invoke();
        if (state.Settings.Current.Language != language) LanguageChanged?.Invoke();
    }

    void OnLogs(object sender, RoutedEventArgs e) { Directory.CreateDirectory(AppPaths.Logs); AppState.Open(AppPaths.Logs); }

    // ================================================================ Leonida settings

    void ShowPython() =>
        PythonText.Text = state.Python is PythonInfo p ? Loc.F("Python {0}: {1}\n(found as `{2}`, the way Build.bat finds it)", p.Version, p.Executable, p.Command)
                                                       : Loc.T("Not found: install Python 3.12 or newer.");

    sealed record ToolView(string Name, string For, string Detail, string Hint, string Glyph, Brush Color, Visibility HintVisibility);

    public void ShowChecks()
    {
        var p = state.Probe;
        if (p == null)
        {
            Summary.Text = Loc.T(state.ProbeError != null ? "Something is missing." : "Checking…");
            ChecksNote.Text = state.ProbeError ?? "";
            ToolList.ItemsSource = null;
            return;
        }
        var missing = p.Tools.Where(t => !t.Ok && !t.MadeByBuild).ToList();
        Summary.Text = missing.Count == 0 ? Loc.T("Everything the build needs is installed.") : Loc.F("{0} thing(s) to install: see the red lines.", missing.Count);
        Summary.Foreground = (Brush)FindResource(missing.Count == 0 ? "Ok" : "Bad");
        ChecksNote.Text = Loc.T("Asked from make_game.py itself: the checks its preflight makes before a build. Some are only needed for some stages.");
        ToolList.ItemsSource = p.Tools.OrderBy(t => t.Ok ? 2 : t.MadeByBuild ? 1 : 0).Select(t =>
        {
            var problem = !t.Ok && !t.MadeByBuild;
            return new ToolView(t.Name, Loc.T("   for ") + t.NeededFor, t.Detail, problem ? t.Hint : "",
                t.Ok ? "" : t.MadeByBuild ? "" : "",
                (Brush)FindResource(t.Ok ? "Ok" : t.MadeByBuild ? "Faint" : "Bad"), problem ? Visibility.Visible : Visibility.Collapsed);
        }).ToList();
        var o = state.Settings.Current.Build;
        DiskText.Text = Loc.F("Port: {0}{1}\nWork files: {2}{3}", state.OutFolder(o), Free(state.OutFolder(o)), state.WorkFolder(o), Free(state.WorkFolder(o)));
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

    async void OnRefresh(object sender, RoutedEventArgs e)
    {
        Summary.Text = Loc.T("Checking…");
        await state.RefreshProbeAsync();
        ShowChecks();
    }

    void OnPlayChanged(object sender, RoutedEventArgs e)
    {
        if (loading) return;
        static string? N(string s) => string.IsNullOrWhiteSpace(s) ? null : s.Trim();
        state.Settings.Current.Play = new PlayOptions
        {
            Overlay = N(OverlayBox.Text), Mods = N(ModsBox.Text), VulkanBackend = VulkanBackend.IsChecked == true,
            TraceD3D12 = TraceD3D12.IsChecked == true, TraceFile = N(TraceFileBox.Text), GameArguments = N(GameArgsBox.Text),
        };
    }

    static string? Pick(string title, string? start)
    {
        var d = new OpenFolderDialog { Title = title };
        if (!string.IsNullOrWhiteSpace(start) && Directory.Exists(start)) d.InitialDirectory = start;
        return d.ShowDialog() == true ? d.FolderName : null;
    }

    void OnBrowseOverlay(object sender, RoutedEventArgs e) { if (Pick("Overlay", OverlayBox.Text) is string f) OverlayBox.Text = f; }
    void OnBrowseMods(object sender, RoutedEventArgs e) { if (Pick("Mods", ModsBox.Text) is string f) ModsBox.Text = f; }

    void OnChooseRepo(object sender, RoutedEventArgs e)
    {
        if (Pick("Leonida", state.Repo) is not string f) return;
        if (!RepoLocator.IsRepo(f)) { _ = Popup.Show("Not a Leonida folder", Loc.F("{0} has no make_game.py. Choose the folder with Build.bat.", f), problem: true); return; }
        state.Settings.Current.RepoRoot = f;
        Save();
        _ = Popup.Show("Leonida folder", "Restart Leonida Launcher to use this folder.");
    }

    void OnBrowsePython(object sender, RoutedEventArgs e)
    {
        var d = new OpenFileDialog { Filter = "python.exe|python.exe|*.exe|*.exe" };
        if (d.ShowDialog() == true) PythonBox.Text = d.FileName;
    }

    async void OnApply(object sender, RoutedEventArgs e)
    {
        var env = new Dictionary<string, string>();
        foreach (var line in EnvBox.Text.Split('\n'))
        {
            var t = line.Trim();
            var eq = t.IndexOf('=');
            if (eq > 0) env[t[..eq].Trim()] = t[(eq + 1)..].Trim();
        }
        state.Settings.Current.Environment = env;
        state.Settings.Current.PythonOverride = string.IsNullOrWhiteSpace(PythonBox.Text) ? null : PythonBox.Text.Trim();
        Save();
        Summary.Text = Loc.T("Checking…");
        await state.InitializeAsync();
        ShowPython();
        ShowChecks();
    }
}
