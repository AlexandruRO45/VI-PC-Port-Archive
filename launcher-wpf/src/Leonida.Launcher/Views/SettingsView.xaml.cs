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
    }

    public void ShowSection(string section)
    {
        (section == "leonida" ? NavLeonida : NavLauncher).IsChecked = true;
        LauncherSection.Visibility = section == "leonida" ? Visibility.Collapsed : Visibility.Visible;
        LeonidaSection.Visibility = section == "leonida" ? Visibility.Visible : Visibility.Collapsed;
        if (section == "leonida") ShowChecks();
    }

    void OnSection(object sender, RoutedEventArgs e)
    {
        if (LauncherSection != null) ShowSection(sender == NavLeonida ? "leonida" : "launcher");
    }

    void OnBack(object sender, RoutedEventArgs e) => BackRequested?.Invoke();

    public void Initialize()
    {
        loading = true;
        var c = state.Settings.Current;
        // Launcher
        LanguageBox.ItemsSource = Loc.Languages.Select(l => l.Name).ToList();
        LanguageBox.SelectedIndex = Math.Max(0, Array.FindIndex(Loc.Languages, l => l.Code == c.Language));
        StartPageBox.ItemsSource = StartPages.Select(p => Loc.T(p.Name)).ToList();
        StartPageBox.SelectedIndex = Math.Max(0, Array.FindIndex(StartPages, p => p.Key == c.StartPage));
        StartCollapsed.IsChecked = c.StartCollapsed;
        AllowOnline.IsChecked = c.AllowOnline;
        StopGracefully.IsChecked = c.StopGracefully;
        GraceBox.Text = c.StopGraceSeconds.ToString();
        NotifyDone.IsChecked = c.NotifyWhenDone;
        SoundDone.IsChecked = c.SoundWhenDone;
        FontSlider.Value = c.ConsoleFontSize;
        KeepLogsBox.Text = c.KeepLogsDays.ToString();
        AvatarPreview.Content = Avatar.Make(c.AvatarSeed, 56);
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
        c.AllowOnline = AllowOnline.IsChecked == true;
        c.StopGracefully = StopGracefully.IsChecked == true;
        c.NotifyWhenDone = NotifyDone.IsChecked == true;
        c.SoundWhenDone = SoundDone.IsChecked == true;
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
        Save();
        AvatarPreview.Content = Avatar.Make(state.Settings.Current.AvatarSeed, 56);
        AvatarChanged?.Invoke();
    }

    void Save() { try { state.Settings.Save(); } catch (IOException) { } }

    void OnReset(object sender, RoutedEventArgs e)
    {
        if (MessageBox.Show(Window.GetWindow(this)!, Loc.T("Restore every launcher setting and build option to its default? Presets and recent folders go too; the build history and logs stay."),
                            "Leonida Launcher", MessageBoxButton.YesNo, MessageBoxImage.Question) != MessageBoxResult.Yes) return;
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
        if (!RepoLocator.IsRepo(f)) { MessageBox.Show($"{f}: make_game.py?", "Leonida Launcher"); return; }
        state.Settings.Current.RepoRoot = f;
        Save();
        MessageBox.Show(Loc.T("Restart Leonida Launcher to use this folder."), "Leonida Launcher");
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
