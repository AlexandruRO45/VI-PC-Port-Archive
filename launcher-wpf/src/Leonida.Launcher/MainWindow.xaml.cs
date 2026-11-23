using System.IO;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Interop;
using System.Windows.Media;
using System.Windows.Threading;
using Leonida.Core;

namespace Leonida.Launcher;

public partial class MainWindow : Window
{
    static readonly string[] Pages = ["home", "leonida", "build", "history", "achievements", "caniru", "gpu"];
    readonly AppState state = AppState.Instance;
    readonly DispatcherTimer timer = new() { Interval = TimeSpan.FromSeconds(1) };
    bool syncing, closingAfterStop, reloading;

    /// <param name="openSettings">Open the settings page on this section (after a language change reloads the window).</param>
    public MainWindow(string? openSettings = null)
    {
        InitializeComponent();
        UpdateAvatar();
        VersionText.Text = "v" + (typeof(MainWindow).Assembly.GetName().Version?.ToString(3) ?? "1.0.2");
        RestoreWindow();
        Loaded += async (_, _) =>
        {
            var c = state.Settings.Current;
            ApplySidebar(openSettings == null && c.StartCollapsed || c.SidebarCollapsed);
            ShowPage(openSettings != null || c.StartPage == "last" ? c.LastPage : c.StartPage);
            if (openSettings != null) OpenSettings(openSettings);
            await state.InitializeAsync();
            HomePage.Initialize();
            SettingsPage.Initialize();
            Refresh();
        };
        StateChanged += (_, _) =>
        {
            Root.Margin = WindowState == WindowState.Maximized ? new Thickness(7) : new Thickness(0);   // chromeless overhang
            MaxButton.Content = WindowState == WindowState.Maximized ? "" : "";
        };
        timer.Tick += (_, _) => Refresh();
        timer.Start();
        Builds.Instance.Changed += OnBuildsChanged;
        Builds.Instance.BuildEnded += OnBuildEnded;
        Builds.Instance.PortEnded += OnPortEnded;
        Tray.Instance.OpenRequested += ShowFromTray;
        Tray.Instance.StopRequested += OnTrayStop;
        Tray.Instance.QuitRequested += OnTrayQuit;
        TaskbarItemInfo = new System.Windows.Shell.TaskbarItemInfo();
        var homeTab = new RadioButton { Content = Loc.T("HOME"), Tag = "", GroupName = "home", Style = (Style)FindResource("SubNav"), Margin = new Thickness(11, 0, 11, 0) };
        homeTab.Checked += (_, _) => { if (!syncing) HomePage.ShowPage(""); };
        HomeTabs.Children.Add(homeTab);
        foreach (var p in Views.HomeView.Pages)
        {
            var tab = new RadioButton { Content = Loc.T(p.Tab), Tag = p.Key, GroupName = "home", Style = (Style)FindResource("SubNav"), Margin = new Thickness(11, 0, 11, 0) };
            tab.Checked += (_, _) => { if (!syncing) HomePage.ShowPage((string)tab.Tag); };
            HomeTabs.Children.Add(tab);
        }
        HomePage.PageChanged += key =>
        {
            syncing = true;
            foreach (RadioButton t in HomeTabs.Children) t.IsChecked = (string)t.Tag == key;
            syncing = false;
        };
        LeonidaPage.NavigateRequested += ShowPage;
        LeonidaPage.BuildDialogRequested += OpenBuildDialog;
        BuildPage.BuildDialogRequested += () => OpenBuildDialog(null);
        HistoryPage.ReuseRequested += o => { Dialog.Open(o); };
        Dialog.Started += () => ShowPage("build");
        Dialog.Closed += Refresh;
        SettingsPage.BackRequested += CloseSettings;
        SettingsPage.AvatarChanged += UpdateAvatar;
        SettingsPage.LanguageChanged += Reload;
        Closed += (_, _) => timer.Stop();
        Loc.Apply(this);
    }

    // ---------- avatar and settings ----------

    void UpdateAvatar()
    {
        AvatarButton.Content = Avatar.Current(36);   // the button is 44 with a 4 px ring margin
        MenuAvatar.Content = Avatar.Current(38);
    }

    void OnAvatar(object sender, RoutedEventArgs e)
    {
        MenuVersion.Text = VersionText.Text;
        AvatarMenu.PlacementTarget = AvatarButton;
        // Right edge under the avatar's right edge (the menu's 14 px shadow margin included).
        AvatarMenu.CustomPopupPlacementCallback = (popup, target, _) =>
            [new System.Windows.Controls.Primitives.CustomPopupPlacement(new Point(target.Width - popup.Width + 14, target.Height + 2),
                                                                         System.Windows.Controls.Primitives.PopupPrimaryAxis.Horizontal)];
        AvatarMenu.IsOpen = true;
    }

    void OnOpenSettings(object sender, RoutedEventArgs e) => OpenSettings("launcher");

    public void OpenSettings(string section)
    {
        SettingsPage.ShowSection(section);
        SettingsPage.Visibility = Visibility.Visible;
    }

    void CloseSettings()
    {
        SettingsPage.Visibility = Visibility.Collapsed;
        Refresh();
    }

    /// <summary>The language changed: a new window in that language takes this one's place. A running build is not
    /// the window's, so it carries on, and the new BUILD tab picks its output up from the log.</summary>
    void Reload()
    {
        SaveWindowBounds();
        try { state.Settings.Save(); } catch (IOException) { }
        Loc.Language = state.Settings.Current.Language;
        var next = new MainWindow("launcher");
        Application.Current.MainWindow = next;
        next.Show();
        reloading = true;
        Close();
    }

    void OpenBuildDialog(string? choice)
    {
        if (Builds.Instance.Busy) { ShowPage("build"); return; }
        Dialog.Open(state.Settings.Current.Build.Clone(), choice);
        Dialog.Focus();
    }

    // ---------- pages ----------

    public void ShowPage(string page)
    {
        if (page == "canirun") page = "caniru";
        if (page == "settings") { OpenSettings("leonida"); return; }
        if (!Pages.Contains(page)) page = "home";
        HomePage.Visibility = page == "home" ? Visibility.Visible : Visibility.Collapsed;
        LeonidaPage.Visibility = page == "leonida" ? Visibility.Visible : Visibility.Collapsed;
        BuildPage.Visibility = page == "build" ? Visibility.Visible : Visibility.Collapsed;
        HistoryPage.Visibility = page == "history" ? Visibility.Visible : Visibility.Collapsed;
        AchievementsPage.Visibility = page == "achievements" ? Visibility.Visible : Visibility.Collapsed;
        SettingsPage.Visibility = page == "settings" ? Visibility.Visible : Visibility.Collapsed;
        CanIRunPage.Visibility = page == "caniru" ? Visibility.Visible : Visibility.Collapsed;
        GpuPage.Visibility = page == "gpu" ? Visibility.Visible : Visibility.Collapsed;
        syncing = true;
        var leonida = page is "leonida" or "build" or "history" or "achievements";
        TopTabs.Visibility = leonida ? Visibility.Visible : Visibility.Collapsed;
        HomeTabs.Visibility = page == "home" ? Visibility.Visible : Visibility.Collapsed;
        // The tabs follow the page: the build pages show each other, Leonida and Achievements show each other.
        var buildPages = page is "build" or "history";
        TabAchievements.Visibility = buildPages ? Visibility.Collapsed : Visibility.Visible;
        TabBuild.Visibility = TabHistory.Visibility = buildPages ? Visibility.Visible : Visibility.Collapsed;
        TabLeonida.IsChecked = page == "leonida";
        TabAchievements.IsChecked = page == "achievements";
        TabBuild.IsChecked = page == "build";
        TabHistory.IsChecked = page == "history";
        LibHome.IsChecked = page == "home";
        LibLeonida.IsChecked = leonida;
        LibCanIRun.IsChecked = page == "caniru";
        LibGpu.IsChecked = page == "gpu";
        syncing = false;
        if (page == "caniru") CanIRunPage.Activate();
        if (page == "gpu" && GpuPage.IsLoaded) _ = GpuPage.ActivateAsync();
        if (page == "leonida") LeonidaPage.Refresh();
        state.Settings.Current.LastPage = page;
    }

    void OnTab(object sender, RoutedEventArgs e)
    {
        if (syncing) return;
        ShowPage(sender == TabBuild ? "build" : sender == TabHistory ? "history" : sender == TabAchievements ? "achievements" : "leonida");
    }

    // Checked (mouse, keyboard, screen readers) and Click (clicking the entry that is already selected) both navigate.
    void OnLibHome(object sender, RoutedEventArgs e) { if (!syncing) { ShowPage("home"); HomePage.ShowPage(""); } }
    void OnLibLeonida(object sender, RoutedEventArgs e) { if (!syncing) ShowPage("leonida"); }
    void OnLibCanIRun(object sender, RoutedEventArgs e) { if (!syncing) ShowPage("caniru"); }
    void OnLibGpu(object sender, RoutedEventArgs e) { if (!syncing) ShowPage("gpu"); }

    // ---------- sidebar ----------

    void OnToggleSidebar(object sender, RoutedEventArgs e) => ApplySidebar(!state.Settings.Current.SidebarCollapsed);

    void ApplySidebar(bool collapsed)
    {
        state.Settings.Current.SidebarCollapsed = collapsed;
        SideColumn.Width = new GridLength(collapsed ? 78 : 270);
        var v = collapsed ? Visibility.Collapsed : Visibility.Visible;
        BrandText.Visibility = LibraryHeader.Visibility = HomeText.Visibility = LibText.Visibility = CanText.Visibility = GpuText.Visibility = VersionText.Visibility = v;
        LibHome.ToolTip = collapsed ? Loc.T("Home") : null;
        LibLeonida.ToolTip = collapsed ? "Leonida" : null;
        LibCanIRun.ToolTip = collapsed ? Loc.T("Can I Run Leonida") : null;
        LibGpu.ToolTip = collapsed ? Loc.T("GPU tests") : null;
        CollapseButton.ApplyTemplate();
        if (CollapseButton.Template.FindName("g", CollapseButton) is TextBlock g) g.Text = collapsed ? "" : "";
    }

    // ---------- status ----------

    const double RingRadius = 15;   // the grey track in MainWindow.xaml uses the same circle
    System.Windows.Media.Animation.DoubleAnimation? ringSpin;

    /// <summary>The sidebar's build ring: an arc for <paramref name="pct"/> with the number inside, or a spinning
    /// quarter arc when the running stage has no percentage.</summary>
    void SetRing(double? pct, bool visible)
    {
        LibRing.Visibility = visible ? Visibility.Visible : Visibility.Collapsed;
        bool spin = visible && pct == null;
        if (spin && ringSpin == null)
        {
            ringSpin = new System.Windows.Media.Animation.DoubleAnimation(0, 360, TimeSpan.FromSeconds(1.1))
                { RepeatBehavior = System.Windows.Media.Animation.RepeatBehavior.Forever };
            LibRingSpin.BeginAnimation(RotateTransform.AngleProperty, ringSpin);
        }
        else if (!spin && ringSpin != null)
        {
            LibRingSpin.BeginAnimation(RotateTransform.AngleProperty, null);
            LibRingSpin.Angle = 0;
            ringSpin = null;
        }
        if (!visible) return;
        double frac = pct is double p ? Math.Clamp(p / 100.0, 0, 1) : 0.25;
        LibRingText.Text = pct is double q ? $"{Math.Floor(q):0}%" : "";
        LibRingArc.Data = Arc(18, 18, RingRadius, frac);
    }

    /// <summary>An arc from 12 o'clock clockwise over <paramref name="frac"/> of the circle.</summary>
    static Geometry Arc(double cx, double cy, double r, double frac)
    {
        if (frac >= 0.9999) return new EllipseGeometry(new Point(cx, cy), r, r);
        if (frac <= 0.0001) return Geometry.Empty;
        double a = frac * 2 * Math.PI;
        var start = new Point(cx, cy - r);
        var end = new Point(cx + r * Math.Sin(a), cy - r * Math.Cos(a));
        var fig = new PathFigure { StartPoint = start, IsClosed = false };
        fig.Segments.Add(new ArcSegment(end, new Size(r, r), 0, frac > 0.5, SweepDirection.Clockwise, true));
        var g = new PathGeometry([fig]);
        g.Freeze();
        return g;
    }

    void Refresh()
    {
        var s = state.Session;
        if (s is { IsRunning: true })
        {
            var sh = s.Monitor.Shaders;
            var stage = s.Monitor.Stages.FirstOrDefault(x => x.State == StageState.Running);
            double? pct = stage?.Name == "shaders" && sh is { Total: > 0, Done: int d, Total: int t } ? 100.0 * d / t : null;
            LibStatus.Text = Loc.F("Building · {0}", stage?.Name ?? Loc.T("Starting"));
            SetRing(pct, visible: !state.Settings.Current.SidebarCollapsed);
            if (TaskbarItemInfo is { } tb)
            {
                tb.ProgressState = pct == null ? System.Windows.Shell.TaskbarItemProgressState.Indeterminate : System.Windows.Shell.TaskbarItemProgressState.Normal;
                if (pct is double tp) tb.ProgressValue = tp / 100.0;
            }
            Tray.Instance.SetStatus(LibStatus.Text + (pct is double tt ? $" {tt:0}%" : ""));
        }
        else
        {
            SetRing(null, visible: false);
            if (TaskbarItemInfo is { } tb) tb.ProgressState = System.Windows.Shell.TaskbarItemProgressState.None;
            var install = Builds.Instance.Install();
            LibStatus.Text = Loc.T(!state.HasRepo ? "Leonida folder not found"
                : Builds.Instance.IsPlaying ? "Running"
                : install?.CanPlay == true ? "Ready to play"
                : install?.RuntimeBuilt == true ? "Built" : "Not built yet");
        }
        if (LeonidaPage.IsVisible) LeonidaPage.Refresh();
    }

    void OnBuildsChanged() => Refresh();

    /// <summary>The end of a build: taskbar flash, sound and a Windows notification, as Settings › Builds says.</summary>
    void OnBuildEnded(HistoryEntry e)
    {
        var c = state.Settings.Current;
        if (c.NotifyWhenDone) Flash();
        if (c.SoundWhenDone) (e.Outcome == SessionOutcome.Succeeded ? System.Media.SystemSounds.Asterisk : System.Media.SystemSounds.Exclamation).Play();
        if (!c.ToastWhenDone || (IsVisible && IsActive)) return;
        var took = Format.Duration(TimeSpan.FromSeconds(e.Seconds));
        var (title, text, problem) = e.Outcome switch
        {
            SessionOutcome.Succeeded => (Loc.T("Leonida is built"), Loc.F("The build finished in {0}.", took), false),
            SessionOutcome.Stopped => (Loc.T("Build stopped"), Loc.F("Stopped after {0}. Start it again to continue where it left off.", took), false),
            SessionOutcome.Cancelled => (Loc.T("Build cancelled"), Loc.F("Cancelled after {0}.", took), true),
            _ => (Loc.T("The build failed"), e.FailedStage is string st ? Loc.F("It failed in the {0} stage after {1}. The BUILD tab has the details.", st, took)
                                                                      : Loc.F("It failed after {0}. The BUILD tab has the details.", took), true),
        };
        Tray.Instance.Notify(title, text, problem);
    }

    void OnPortEnded(bool dryRun, TimeSpan took, SessionOutcome outcome)
    {
        if (!IsVisible) { Tray.Instance.Unpin(); Close(); }   // closed to the tray while it ran: nothing is left to wait for
    }

    void ShowFromTray()
    {
        Show();
        if (WindowState == WindowState.Minimized) WindowState = WindowState.Normal;
        Activate();
        Tray.Instance.Unpin();
    }

    async void OnTrayStop()
    {
        await Builds.Instance.StopAsync();
    }

    async void OnTrayQuit()
    {
        if (Builds.Instance.Busy)
        {
            closingAfterStop = true;
            await Builds.Instance.StopAsync();
            await Task.Delay(300);
        }
        quitting = true;
        Close();
    }

    void Flash()
    {
        if (IsActive) return;
        var f = new Native.FLASHWINFO { hwnd = new WindowInteropHelper(this).Handle, dwFlags = 3 | 12, uCount = 5 };
        f.cbSize = (uint)System.Runtime.InteropServices.Marshal.SizeOf(f);
        Native.FlashWindowEx(ref f);
    }

    // ---------- window ----------

    void OnMinimize(object sender, RoutedEventArgs e) => WindowState = WindowState.Minimized;
    void OnMaximize(object sender, RoutedEventArgs e) => WindowState = WindowState == WindowState.Maximized ? WindowState.Normal : WindowState.Maximized;
    void OnClose(object sender, RoutedEventArgs e) => Close();

    protected override void OnPreviewKeyDown(System.Windows.Input.KeyEventArgs e)
    {
        if (PopupHost.Children.Count > 0) { base.OnPreviewKeyDown(e); return; }   // Esc and Enter belong to the popup
        if (e.Key == System.Windows.Input.Key.Escape && Dialog.Visibility == Visibility.Visible) { Dialog.Close(); e.Handled = true; }
        else if (e.Key == System.Windows.Input.Key.Escape && SettingsPage.Visibility == Visibility.Visible) { CloseSettings(); e.Handled = true; }
        base.OnPreviewKeyDown(e);
    }

    void SaveWindowBounds()
    {
        var c = state.Settings.Current;
        c.WindowMaximized = WindowState == WindowState.Maximized;
        var r = WindowState == WindowState.Normal ? new Rect(Left, Top, Width, Height) : RestoreBounds;
        c.WindowLeft = r.Left; c.WindowTop = r.Top; c.WindowWidth = r.Width; c.WindowHeight = r.Height;
    }

    void RestoreWindow()
    {
        // Always the same size (1465 × 795, set in MainWindow.xaml); only the position is remembered.
        var c = state.Settings.Current;
        if (c.WindowLeft is double l && c.WindowTop is double t)
        {
            var screen = new Rect(SystemParameters.VirtualScreenLeft, SystemParameters.VirtualScreenTop,
                                  SystemParameters.VirtualScreenWidth, SystemParameters.VirtualScreenHeight);
            if (screen.Contains(new Point(l + 40, t + 20))) { WindowStartupLocation = WindowStartupLocation.Manual; Left = l; Top = t; }
        }
    }

    /// <summary>Rounded corners on every edge: Windows 11 rounds the window itself (DWMWA_WINDOW_CORNER_PREFERENCE),
    /// so resizing and snapping stay native. Older Windows ignores the request and keeps square corners.</summary>
    protected override void OnSourceInitialized(EventArgs e)
    {
        base.OnSourceInitialized(e);
        var hwnd = new WindowInteropHelper(this).Handle;
        var round = 2;   // DWMWCP_ROUND
        Native.DwmSetWindowAttribute(hwnd, 33, ref round, sizeof(int));
    }

    bool quitting;

    protected override async void OnClosing(System.ComponentModel.CancelEventArgs e)
    {
        if (reloading) { base.OnClosing(e); return; }   // the new window carries on; nothing is stopped or saved twice
        if (Builds.Instance.Busy && !closingAfterStop && !quitting && state.Settings.Current.CloseToTray)
        {
            // Keep building from the tray: the window hides, the icon shows what the build is doing.
            e.Cancel = true;
            SaveWindowBounds();
            Hide();
            Tray.Instance.Pin();
            Refresh();
            if (!state.Settings.Current.TrayExplained)
            {
                state.Settings.Current.TrayExplained = true;
                try { state.Settings.Save(); } catch (IOException) { }
                Tray.Instance.Notify(Loc.T("Leonida keeps going in the tray"),
                    Loc.T(state.IsBuilding ? "The build goes on. Click the icon to open the launcher, or right-click it to stop the build."
                                           : "The port goes on. Click the icon to open the launcher."), false);
            }
            return;
        }
        if (Builds.Instance.Busy && !closingAfterStop)
        {
            e.Cancel = true;
            if (!await Popup.Ask(state.IsBuilding ? "A build is running" : "The port is running",
                                 state.IsBuilding ? "A build is running. Stop it and close the launcher?\n\nA build stops the way Ctrl+C stops it and continues where it left off next time."
                                                  : "The port is running. Stop it and close the launcher?",
                                 "Stop and close", "Keep running")) return;
            closingAfterStop = true;
            await Builds.Instance.StopAsync();
            await Task.Delay(300);
            Close();
            return;
        }
        if (IsVisible) SaveWindowBounds();
        try { state.Settings.Save(); } catch (IOException) { }
        state.Session?.Dispose();
        Builds.Instance.Play?.Dispose();
        GpuPage.Shutdown();
        Tray.Instance.Dispose();
        base.OnClosing(e);
    }
}

static class Native
{
    [System.Runtime.InteropServices.StructLayout(System.Runtime.InteropServices.LayoutKind.Sequential)]
    public struct FLASHWINFO { public uint cbSize; public IntPtr hwnd; public uint dwFlags; public uint uCount; public uint dwTimeout; }
    [System.Runtime.InteropServices.DllImport("user32.dll")] public static extern bool FlashWindowEx(ref FLASHWINFO f);
    [System.Runtime.InteropServices.DllImport("dwmapi.dll")] public static extern int DwmSetWindowAttribute(IntPtr hwnd, int attribute, ref int value, int size);
}
