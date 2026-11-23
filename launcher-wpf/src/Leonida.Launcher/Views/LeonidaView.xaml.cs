using System.IO;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Input;
using Leonida.Core;

namespace Leonida.Launcher.Views;

/// <summary>
/// The game page. One button, whichever applies: PLAY when the port is built and can start, BUILD when it can't,
/// the build's progress while one runs (it opens the BUILD tab), STOP while the port runs. Everything else is in the
/// ⋯ menu.
/// </summary>
public partial class LeonidaView : UserControl
{
    enum MainAction { Build, Play, ViewBuild, Stop }
    readonly AppState state = AppState.Instance;
    MainAction action = MainAction.Build;

    public event Action<string>? NavigateRequested;
    public event Action<string?>? BuildDialogRequested;   // null: as last time; "check": the file check

    public LeonidaView() => InitializeComponent();

    public void Refresh()
    {
        var builds = Builds.Instance;
        var install = builds.Install();
        var s = state.Session;
        if (s is { IsRunning: true })
        {
            action = MainAction.ViewBuild;
            var sh = s.Monitor.Shaders;
            var stage = s.Monitor.Stages.FirstOrDefault(x => x.State == StageState.Running);
            var pct = stage?.Name == "shaders" && sh is { Total: > 0, Done: int d, Total: int t } ? $" {100.0 * d / t:0}%" : "";
            MainButton.Content = Loc.F("BUILDING{0}", pct);
            Status.Text = Loc.F("{0} · {1}. Press the button to follow it.", stage != null ? char.ToUpper(stage.Name[0]) + stage.Name[1..] : Loc.T("Starting"), Format.Duration(s.Elapsed));
        }
        else if (builds.IsPlaying)
        {
            action = MainAction.Stop;
            MainButton.Content = Loc.T("STOP");
            Status.Text = Loc.T("The port is running. Its output is in the BUILD tab.");
        }
        else if (install?.CanPlay == true)
        {
            action = MainAction.Play;
            MainButton.Content = Loc.T("PLAY");
            Status.Text = Loc.T("Built. The loader runs a title end to end only with a stand-in executable so far, so expect it to stop early.");
        }
        else
        {
            action = MainAction.Build;
            MainButton.Content = install?.RuntimeBuilt == true ? Loc.T("UPDATE BUILD") : Loc.BuildLabel;
            Status.Text = !state.HasRepo ? Loc.T("Leonida.exe needs to be next to Build.bat in your Leonida folder (or choose the folder in Settings).")
                : install?.RuntimeBuilt == true ? Why(install.WhyNotPlayable)
                : Loc.T("Builds the PC port from your own copy of the game, on this PC.");
        }
        MainButton.IsEnabled = state.HasRepo && state.Python != null || action != MainAction.Build;
        MenuRebuild.IsEnabled = MenuCheck.IsEnabled = MenuOptions.IsEnabled = !builds.Busy;
        MenuDryRun.IsEnabled = !builds.Busy && install?.CanPlay == true;
        MenuRebuild.Visibility = install?.RuntimeBuilt == true || install?.ShadersTranslated == true ? Visibility.Visible : Visibility.Collapsed;

        // The tile
        static string Yes(bool b) => Loc.T(b ? "ready" : "not yet");
        PortShaders.Text = Yes(install?.ShadersTranslated == true);
        PortRuntime.Text = Yes(install?.RuntimeBuilt == true);
        var last = state.History.Entries.FirstOrDefault();
        LastBuild.Text = last == null ? Loc.T("never") : $"{Loc.T(last.Outcome.ToString()).ToLowerInvariant()}, {Ago(last.Ended ?? last.Started)}";
        TileTitle.Text = Loc.T(s is { IsRunning: true } ? "Building…" : install?.CanPlay == true ? "Ready to play" : install?.RuntimeBuilt == true ? "Built" : "Not built yet");
        TileDetail.Text = install?.OutFolder ?? "";
    }

    static string Ago(DateTime t)
    {
        var d = DateTime.Now - t;
        return d.TotalMinutes < 1 ? Loc.T("just now") : d.TotalHours < 1 ? Loc.F("{0} min ago", (int)d.TotalMinutes)
             : d.TotalDays < 1 ? Loc.F("{0} h ago", (int)d.TotalHours) : Loc.F("{0} d ago", (int)d.TotalDays);
    }

    /// <summary>The port's "why not playable" message (from Leonida.Core) in the launcher's language.</summary>
    static string Why(string? s)
    {
        if (s == null) return "";
        const string missing = " is missing from your game copy, so there is nothing to run yet.";
        if (s.EndsWith(missing)) return Loc.F("{0} is missing from your game copy, so there is nothing to run yet.", s[..^missing.Length]);
        return Loc.T(s);
    }

    async void OnMain(object sender, RoutedEventArgs e)
    {
        switch (action)
        {
            case MainAction.ViewBuild: NavigateRequested?.Invoke("build"); break;
            case MainAction.Stop: await Builds.Instance.StopAsync(); break;
            case MainAction.Play: Play(false); break;
            default: BuildDialogRequested?.Invoke(null); break;
        }
        Refresh();
    }

    void Play(bool dryRun)
    {
        var problems = Builds.Instance.PlayProblems(dryRun);
        if (problems.Count > 0)
        {
            MessageBox.Show(Window.GetWindow(this)!, Loc.T("Not ready to start:") + "\n\n• " + string.Join("\n• ", problems.Select(Why)), "Leonida Launcher");
            return;
        }
        Builds.Instance.StartPort(dryRun);
        NavigateRequested?.Invoke("build");
    }

    void OnMore(object sender, RoutedEventArgs e)
    {
        MoreMenu.PlacementTarget = MoreButton;
        MoreMenu.IsOpen = true;
    }

    void OnBuildOptions(object sender, RoutedEventArgs e) => BuildDialogRequested?.Invoke(null);
    void OnRebuild(object sender, RoutedEventArgs e) => BuildDialogRequested?.Invoke(null);
    void OnCheckFiles(object sender, RoutedEventArgs e) => BuildDialogRequested?.Invoke("check");
    void OnDryRun(object sender, RoutedEventArgs e) => Play(true);
    void OnTrouble(object sender, RoutedEventArgs e) => NavigateRequested?.Invoke("settings");
    void OnTile(object sender, MouseButtonEventArgs e) => OnOpenOutput(sender, e);

    void OnOpenOutput(object sender, RoutedEventArgs e)
    {
        var folder = Builds.Instance.Install()?.OutFolder;
        if (folder != null && Directory.Exists(folder)) AppState.Open(folder);
        else MessageBox.Show(Loc.F("{0} does not exist yet: the first build makes it.", folder), "Leonida Launcher");
    }

    void OnOpenLogs(object sender, RoutedEventArgs e) { Directory.CreateDirectory(AppPaths.Logs); AppState.Open(AppPaths.Logs); }
    void OnOpenRepo(object sender, RoutedEventArgs e) { if (state.Repo != null) AppState.Open(state.Repo); }
}
