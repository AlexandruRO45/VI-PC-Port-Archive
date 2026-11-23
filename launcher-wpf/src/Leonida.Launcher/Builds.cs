using System.Collections.Concurrent;
using System.IO;
using System.Text;
using System.Windows;
using Leonida.Core;

namespace Leonida.Launcher;

/// <summary>
/// Starts and stops builds and the port for every page: the LEONIDA button, the BUILD tab and the build popup all go
/// through here, so there is only ever one build. Output lines are queued for the BUILD tab's console.
/// </summary>
public sealed class Builds
{
    public static Builds Instance { get; } = new();
    readonly AppState state = AppState.Instance;

    public ConcurrentQueue<LogLine> Lines { get; } = new();
    public BuildSession? Play { get; private set; }
    public BuildOptions? LastOptions { get; private set; }
    public string? LastLogPath { get; private set; }
    public bool IsPlaying => Play?.IsRunning == true;
    public bool Busy => state.IsBuilding || IsPlaying;

    /// <summary>Raised on the UI thread when a build or the port starts or ends.</summary>
    public event Action? Changed;
    /// <summary>Raised on the UI thread before a new run's lines start (the console clears).</summary>
    public event Action? NewRun;

    public List<string> Problems(BuildOptions o)
    {
        var problems = o.Validate();
        if (state.Python == null) problems.Insert(0, "Python 3.12 or newer was not found (Settings).");
        if (!state.HasRepo) problems.Insert(0, "The Leonida folder was not found (Settings).");
        if (IsPlaying) problems.Add("The port is running; close it before building.");
        if (state.IsBuilding) problems.Add("A build is already running.");
        return problems.Select(Loc.T).ToList();
    }

    /// <summary>Runs make_game.py with these options, exactly as Build.bat would with the same arguments.</summary>
    public bool Start(BuildOptions o)
    {
        if (Problems(o).Count > 0) return false;
        state.Settings.Current.Build = o.Clone();
        state.Settings.RememberOutput(o.Out);
        try { state.Settings.Save(); } catch (IOException) { }

        var id = DateTime.Now.ToString("yyyyMMdd-HHmmss");
        var progress = Path.Combine(AppPaths.Progress, id);
        Directory.CreateDirectory(progress);
        // make_game.py always checks the copy first, whatever --stages says.
        var stages = o.CheckOnly ? ["check"] : new[] { "check" }.Concat(o.OrderedStages().Where(s => s != "check")).ToList();
        var monitor = new BuildMonitor(stages, progress);
        var env = state.BuildEnvironment().ToDictionary(kv => kv.Key, kv => (string?)kv.Value);
        env["LEONIDA_PROGRESS_DIR"] = progress;   // tools/progress_report.py: live shader counts; nothing else changes
        var args = new List<string> { "-I", "-u", Path.Combine(state.Repo!, "make_game.py") };
        args.AddRange(o.ToArguments());
        var session = new BuildSession(state.Python!.Executable, args, state.Repo!, Path.Combine(AppPaths.Logs, $"build-{id}.log"),
                                       monitor, env, state.OutputEncoding());
        LastOptions = o.Clone();
        LastLogPath = session.LogPath;
        NewRun?.Invoke();
        session.Line += Lines.Enqueue;
        session.Exited += s => Application.Current.Dispatcher.BeginInvoke(() => Finished(s, o, id, progress));
        state.Session = session;
        session.Start();
        Changed?.Invoke();
        return true;
    }

    void Finished(BuildSession s, BuildOptions o, string id, string progress)
    {
        var m = s.Monitor;
        var sh = m.Shaders;
        state.History.Add(new HistoryEntry
        {
            Id = id, Started = s.Started, Ended = s.Ended, Seconds = s.Elapsed.TotalSeconds, Outcome = s.Outcome, ExitCode = s.ExitCode,
            CommandLine = s.CommandLineText, Options = o.Clone(),
            StagesDone = m.Stages.Where(x => x.State == StageState.Done).Select(x => x.Name).ToList(),
            FailedStage = m.Stages.FirstOrDefault(x => x.State == StageState.Failed)?.Name,
            Warnings = m.Warnings, Errors = m.Errors, FirstError = m.FirstError,
            Shaders = sh?.Total ?? (m.ShardSummaries > 0 ? m.SummaryShaders : null),
            ShadersTranslated = sh?.Translated ?? (m.ShardSummaries > 0 ? m.SummaryTranslated : null),
            ShadersValid = sh?.Valid ?? (m.ShardSummaries > 0 ? m.SummaryValid : null),
            TranslateSeconds = m.TranslateStarted is DateTime t0
                ? ((m.Stages.FirstOrDefault(x => x.Name == "shaders")?.Ended ?? s.Ended ?? DateTime.Now) - t0).TotalSeconds : null,
            PeakMemoryBytes = s.PeakMemory, LogPath = s.LogPath, OutFolder = m.OutFolder,
        });
        try { Directory.Delete(progress, true); } catch (Exception) { }   // counts only; the log is the record
        state.NotifyBuildFinished();
        Changed?.Invoke();
    }

    public async Task StopAsync()
    {
        if (state.Session is { IsRunning: true } s)
        {
            if (state.Settings.Current.StopGracefully) await s.StopAsync(TimeSpan.FromSeconds(Math.Max(5, state.Settings.Current.StopGraceSeconds)));
            else s.Cancel();
        }
        else if (Play is { IsRunning: true } p) await p.StopAsync(TimeSpan.FromSeconds(10));
    }

    public void Cancel()
    {
        if (state.Session is { IsRunning: true } s) s.Cancel();
        else Play?.Cancel();
    }

    // ---------------------------------------------------------------- the port

    public PortInstall? Install()
    {
        try
        {
            var o = state.Settings.Current.Build;
            return PortInstall.Inspect(state.OutFolder(o), state.WorkFolder(o));
        }
        catch (Exception) { return null; }
    }

    public List<string> PlayProblems(bool dryRun)
    {
        var install = Install();
        var p = install == null ? ["The output folder could not be read."] : install.PlayProblems(state.Settings.Current.Play);
        if (!dryRun && install is { ShadersTranslated: false }) p.Add("The shaders are not translated yet (shaders stage).");
        if (Busy) p.Add("A build or the port is already running.");
        return p;
    }

    /// <summary>gta6rt.exe --run (or --dry-run) on the game's executable, with Settings › Play's options.</summary>
    public bool StartPort(bool dryRun)
    {
        if (PlayProblems(dryRun).Count > 0) return false;
        var install = Install()!;
        var o = state.Settings.Current.Play;
        var env = state.BuildEnvironment().ToDictionary(kv => kv.Key, kv => (string?)kv.Value);
        if (o.VulkanBackend) env["LEONIDA_D3D12"] = "vulkan";
        var id = DateTime.Now.ToString("yyyyMMdd-HHmmss");
        var session = new BuildSession(install.Runtime, install.PlayArguments(o, dryRun), install.GameFolder!,
                                       Path.Combine(AppPaths.Logs, $"{(dryRun ? "dryrun" : "play")}-{id}.log"),
                                       new BuildMonitor([], null), env, new UTF8Encoding(false));
        LastLogPath = session.LogPath;
        NewRun?.Invoke();
        session.Line += Lines.Enqueue;
        session.Exited += _ => Application.Current.Dispatcher.BeginInvoke(() => Changed?.Invoke());
        Play = session;
        session.Start();
        Changed?.Invoke();
        return true;
    }
}
