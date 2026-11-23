using System.Diagnostics;
using System.Text;

namespace Leonida.Core;

public enum SessionOutcome { Running, Succeeded, Failed, Stopped, Cancelled, CouldNotStart }

/// <summary>CPU and memory of the whole build tree, from its job object.</summary>
public sealed record ResourceSample(DateTime Time, double CpuPercent, long MemoryBytes, int Processes,
                                    ulong SystemTotal, ulong SystemAvailable);

/// <summary>
/// One run of make_game.py (or of any command, for Play): started in a job object, its stdout and stderr read line
/// by line on background threads, every line written to a log file as it arrives and handed to the monitor.
/// Stop sends Ctrl+C (make_game.py stops cleanly and resumes next time), Cancel ends the whole tree at once.
/// </summary>
public sealed class BuildSession : IDisposable
{
    readonly Process process = new();
    readonly JobObject job = new();
    readonly StreamWriter log;
    readonly object logGate = new();
    readonly CancellationTokenSource sampling = new();
    TimeSpan lastCpu = TimeSpan.Zero;
    DateTime lastSample = DateTime.Now;
    int stopRequested;   // 0 none, 1 stop (Ctrl+C), 2 cancel (kill)

    public BuildSession(string exe, IReadOnlyList<string> args, string workingDir, string logPath,
                        BuildMonitor monitor, IDictionary<string, string?> environment, Encoding outputEncoding)
    {
        Monitor = monitor;
        LogPath = logPath;
        CommandLineText = CommandLine.Join([exe, .. args]);
        Directory.CreateDirectory(Path.GetDirectoryName(logPath)!);
        log = new StreamWriter(new FileStream(logPath, FileMode.Create, FileAccess.Write, FileShare.Read), new UTF8Encoding(false))
            { AutoFlush = true };
        var psi = new ProcessStartInfo(exe)
        {
            UseShellExecute = false,
            CreateNoWindow = true,            // a console without a window: Ctrl+C can still reach it
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            RedirectStandardInput = true,     // closed at once: a prompt reads end-of-file instead of waiting forever
            StandardOutputEncoding = outputEncoding,
            StandardErrorEncoding = outputEncoding,
            WorkingDirectory = workingDir,
        };
        foreach (var a in args) psi.ArgumentList.Add(a);
        foreach (var (k, v) in environment)
        {
            if (v == null) psi.Environment.Remove(k);
            else psi.Environment[k] = v;
        }
        process.StartInfo = psi;
        process.EnableRaisingEvents = true;
    }

    public BuildMonitor Monitor { get; }
    public string LogPath { get; }
    public string CommandLineText { get; }
    public DateTime Started { get; private set; }
    public DateTime? Ended { get; private set; }
    public int? ExitCode { get; private set; }
    public SessionOutcome Outcome { get; private set; } = SessionOutcome.Running;
    public ResourceSample? LastSample { get; private set; }
    public long PeakMemory { get; private set; }
    public TimeSpan Elapsed => (Ended ?? DateTime.Now) - Started;
    public bool IsRunning => Outcome == SessionOutcome.Running;

    public event Action<LogLine>? Line;
    public event Action<BuildSession>? Exited;

    public void Start()
    {
        Started = DateTime.Now;
        WriteLog($"# Leonida Launcher {Started:yyyy-MM-dd HH:mm:ss}");
        WriteLog($"# {CommandLineText}");
        Emit(new LogLine(DateTime.Now, LineKind.Launcher, "> " + CommandLineText, false));
        try
        {
            ConsoleControl.AllowCtrlCInChildren();
            process.Start();
        }
        catch (Exception e)
        {
            Ended = DateTime.Now;
            Outcome = SessionOutcome.CouldNotStart;
            Emit(new LogLine(DateTime.Now, LineKind.Error, $"Could not start {process.StartInfo.FileName}: {e.Message}", true));
            Exited?.Invoke(this);
            return;
        }
        try { job.Assign(process); }
        catch (Exception e) { Emit(new LogLine(DateTime.Now, LineKind.Warning, $"(launcher) could not track child processes: {e.Message}", true)); }
        process.StandardInput.Close();
        var outReader = Task.Run(() => Pump(process.StandardOutput, false));
        var errReader = Task.Run(() => Pump(process.StandardError, true));
        _ = Task.Run(() => SampleLoop(sampling.Token));
        _ = Task.Run(async () =>
        {
            await process.WaitForExitAsync();
            await Task.WhenAll(outReader, errReader);   // every line is in before the end is reported
            Finish();
        });
    }

    void Pump(StreamReader reader, bool stderr)
    {
        string? text;
        while ((text = reader.ReadLine()) != null) Emit(Monitor.Feed(text, stderr));
    }

    void Emit(LogLine line)
    {
        WriteLog(line.Text);
        Line?.Invoke(line);
    }

    void WriteLog(string text)
    {
        lock (logGate)
        {
            try { log.WriteLine(text); } catch (ObjectDisposedException) { }
        }
    }

    async Task SampleLoop(CancellationToken ct)
    {
        while (!ct.IsCancellationRequested)
        {
            Sample();
            try { await Task.Delay(1000, ct); } catch (OperationCanceledException) { break; }
        }
    }

    void Sample()
    {
        var now = DateTime.Now;
        var (cpu, _) = job.Accounting();
        var wall = (now - lastSample).TotalSeconds;
        var percent = wall > 0 ? (cpu - lastCpu).TotalSeconds / (wall * Environment.ProcessorCount) * 100 : 0;
        lastCpu = cpu;
        lastSample = now;
        long memory = 0;
        var ids = job.ProcessIds();
        foreach (var id in ids)
        {
            try { using var p = Process.GetProcessById(id); memory += p.WorkingSet64; }
            catch (ArgumentException) { }          // exited since the list was taken
            catch (InvalidOperationException) { }
        }
        PeakMemory = Math.Max(PeakMemory, memory);
        var (total, available) = SystemMemory.Status();
        LastSample = new ResourceSample(now, Math.Clamp(percent, 0, 100), memory, ids.Count, total, available);
        Monitor.PollProgress();
    }

    void Finish()
    {
        sampling.Cancel();
        Monitor.PollProgress();
        Ended = DateTime.Now;
        ExitCode = process.ExitCode;
        Outcome = stopRequested switch
        {
            2 => SessionOutcome.Cancelled,
            1 => SessionOutcome.Stopped,
            _ => ExitCode == 0 ? SessionOutcome.Succeeded : Monitor.StoppedLineSeen ? SessionOutcome.Stopped : SessionOutcome.Failed,
        };
        Monitor.Complete(ExitCode ?? -1, Outcome is SessionOutcome.Stopped or SessionOutcome.Cancelled);
        // Whatever the build left running in its tree (it shouldn't) ends with it.
        job.Terminate();
        var note = Outcome switch
        {
            SessionOutcome.Succeeded => $"Build finished in {Format.Duration(Elapsed)}.",
            SessionOutcome.Stopped => "Build stopped. Start it again to continue where it left off.",
            SessionOutcome.Cancelled => "Build cancelled; every process it started was ended.",
            _ => $"Build failed with exit code {ExitCode}. Read the messages above.",
        };
        Emit(new LogLine(DateTime.Now, Outcome == SessionOutcome.Succeeded ? LineKind.Success : LineKind.Launcher, note, false));
        Exited?.Invoke(this);
    }

    /// <summary>Ctrl+C, as in a terminal; if the build is still running after the grace period, it is ended.</summary>
    public async Task StopAsync(TimeSpan grace)
    {
        if (!IsRunning || Interlocked.CompareExchange(ref stopRequested, 1, 0) != 0) return;
        Emit(new LogLine(DateTime.Now, LineKind.Launcher, "Stopping (Ctrl+C)…", false));
        if (!ConsoleControl.SendCtrlC(process.Id))
            Emit(new LogLine(DateTime.Now, LineKind.Warning, "(launcher) Ctrl+C could not be delivered; ending the build instead", false));
        else
        {
            try { await process.WaitForExitAsync(new CancellationTokenSource(grace).Token); return; }
            catch (OperationCanceledException) { }
            Emit(new LogLine(DateTime.Now, LineKind.Warning, $"(launcher) still running after {grace.TotalSeconds:0} s; ending it", false));
        }
        job.Terminate();
    }

    /// <summary>Ends the build and every process it started, now.</summary>
    public void Cancel()
    {
        if (!IsRunning) return;
        Interlocked.Exchange(ref stopRequested, 2);
        Emit(new LogLine(DateTime.Now, LineKind.Launcher, "Cancelling: ending the build and all its processes…", false));
        job.Terminate();
    }

    public void Dispose()
    {
        sampling.Cancel();
        if (IsRunning) job.Terminate();
        job.Dispose();
        process.Dispose();
        lock (logGate) log.Dispose();
    }
}

public static class Format
{
    public static string Duration(TimeSpan t) =>
        t.TotalHours >= 1 ? $"{(int)t.TotalHours} h {t.Minutes:00} min" :
        t.TotalMinutes >= 1 ? $"{(int)t.TotalMinutes} min {t.Seconds:00} s" : $"{t.TotalSeconds:0} s";

    /// <summary>Short form for a stat tile: 12s, 4m 32s, 1h 02m.</summary>
    public static string Short(TimeSpan t) =>
        t.TotalHours >= 1 ? $"{(int)t.TotalHours}h {t.Minutes:00}m" : t.TotalMinutes >= 1 ? $"{(int)t.TotalMinutes}m {t.Seconds:00}s" : $"{t.TotalSeconds:0}s";

    public static string Bytes(double b)
    {
        string[] u = ["B", "KB", "MB", "GB", "TB"];
        var i = 0;
        while (b >= 1024 && i < u.Length - 1) { b /= 1024; i++; }
        return i == 0 ? $"{b:0} {u[i]}" : $"{b:0.0} {u[i]}";
    }
}
