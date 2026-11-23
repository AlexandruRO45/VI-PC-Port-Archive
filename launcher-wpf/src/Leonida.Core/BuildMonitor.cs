using System.Text.Json;
using System.Text.RegularExpressions;

namespace Leonida.Core;

public enum LineKind { Info, Command, Warning, Error, Success, Launcher }

public sealed record LogLine(DateTime Time, LineKind Kind, string Text, bool FromStderr);

public enum StageState { Pending, Running, Done, Failed, Skipped }

public sealed class StageStatus
{
    public required string Name { get; init; }
    public StageState State { get; set; } = StageState.Pending;
    public string Step { get; set; } = "";
    public DateTime? Started { get; set; }
    public DateTime? Ended { get; set; }
    public TimeSpan? Duration => Started is DateTime s ? (Ended ?? DateTime.Now) - s : null;
}

/// <summary>Counts from tools/progress_report.py files (null where nothing reported them).</summary>
public sealed record ShaderCounts(
    int? DisasmTotal, int? DisasmDone, int? DisasmReused, int? DisasmFailed,
    int? Total, int? Done, int? Translated, int? Valid, int? Failed, int ActiveShards, int Shards,
    double? ShadersPerSecond, double? SecondsPerShader, TimeSpan? Eta, string? Current,
    double? DisasmPerSecond = null, TimeSpan? DisasmEta = null);

/// <summary>
/// Reads make_game.py's output line by line and keeps what can be measured from it: the stage and step it is on,
/// warnings and errors, and the summaries each tool prints (shard results, disassembly, pipelines, the file check).
/// The live shader counts come from the opt-in reports in the progress folder (tools/progress_report.py), read by
/// <see cref="PollProgress"/>. Nothing here is estimated from a model of how long a build takes.
/// </summary>
public sealed partial class BuildMonitor
{
    public static readonly string[] KnownStages = BuildOptions.AllStages;

    readonly object gate = new();
    readonly string? progressDir;
    readonly List<(DateTime t, int done)> speedSamples = [];
    readonly List<(DateTime t, int done)> disasmSamples = [];
    DateTime? translateStarted;

    public BuildMonitor(IEnumerable<string> stages, string? progressDir)
    {
        this.progressDir = progressDir;
        Stages = stages.Select(s => new StageStatus { Name = s }).ToList();
    }

    public List<StageStatus> Stages { get; }
    public string CurrentOperation { get; private set; } = "Starting";
    public int Warnings { get; private set; }
    public int Errors { get; private set; }
    public int LinesSeen { get; private set; }
    public string? OutFolder { get; private set; }
    public string? WorkFolder { get; private set; }
    public bool FinishedLineSeen { get; private set; }
    public bool StoppedLineSeen { get; private set; }
    public string? FirstError { get; private set; }

    // Summaries the tools print when they finish (summed over shards and pipeline batches).
    public int? CheckFiles { get; private set; }
    public int CheckProblems { get; private set; }
    public int ShardSummaries { get; private set; }
    public int SummaryShaders { get; private set; }
    public int SummaryTranslated { get; private set; }
    public int SummaryValid { get; private set; }
    public int? KeptShards { get; private set; }
    public int? ShardCount { get; private set; }
    public int AssetSteps { get; private set; }
    public int PipelineModulesOk { get; private set; }
    public int PipelineModulesFailed { get; private set; }
    public int PipelinesOk { get; private set; }
    public int PipelinesFailed { get; private set; }
    public int PipelinesSkipped { get; private set; }
    public int PipelineSummaries { get; private set; }
    public ShaderCounts? Shaders { get; private set; }

    [GeneratedRegex(@"^(\d\d:\d\d:\d\d)  (.*)$")] private static partial Regex Stamped();
    [GeneratedRegex(@"^(check|shaders|assets|runtime|pipelines|preflight): (.*)$")] private static partial Regex StageMsg();
    [GeneratedRegex(@"^(\d+) shaders: (\d+) translated, (\d+) pass spirv-val")] private static partial Regex ShardSummary();
    [GeneratedRegex(@"^check: .*?, (\d+) files$")] private static partial Regex CheckCount();
    [GeneratedRegex(@"^  (missing|wrong size)  ")] private static partial Regex CheckProblem();
    [GeneratedRegex(@"keeping (\d+) of (\d+) shards")] private static partial Regex Keeping();
    [GeneratedRegex(@"--shard [0-9.,]+/(\d+)")] private static partial Regex ShardArg();
    [GeneratedRegex(@"^(\S+): (\d+) modules ok, (\d+) failed")] private static partial Regex Modules();
    [GeneratedRegex(@"^(\S+) pipelines: (\d+) ok, (\d+) failed, (\d+) skipped")] private static partial Regex Pipes();
    [GeneratedRegex(@"^building into (.+) \(work files in (.+)\)$")] private static partial Regex BuildingInto();
    [GeneratedRegex(@"\b(?:error|fatal error)\b|Traceback \(most recent call last\)|\bFAILED\b|MODULE CRASH", RegexOptions.IgnoreCase)]
    private static partial Regex ErrorWords();
    [GeneratedRegex(@"\bwarning\b|couldn't|could not|cannot|not usable|\blimited\b", RegexOptions.IgnoreCase)]
    private static partial Regex WarningWords();

    /// <summary>Classifies one line of output and updates the counts. Thread-safe.</summary>
    public LogLine Feed(string raw, bool stderr)
    {
        lock (gate)
        {
            LinesSeen++;
            var kind = Classify(raw, stderr, out var message);
            if (kind == LineKind.Error)
            {
                Errors++;
                FirstError ??= raw.Trim();
            }
            else if (kind == LineKind.Warning) Warnings++;
            if (message != null) Interpret(message.TrimEnd());
            else InterpretToolLine(raw);
            return new LogLine(DateTime.Now, kind, raw, stderr);
        }
    }

    LineKind Classify(string raw, bool stderr, out string? message)
    {
        message = null;
        var m = Stamped().Match(raw);
        if (m.Success)
        {
            message = m.Groups[2].Value;
            if (message.StartsWith("  ")) return LineKind.Command;          // run() echoes each command it starts
            if (message.StartsWith("finished in ")) return LineKind.Success;
            if (message.EndsWith(": done")) return LineKind.Success;
            return WarningWords().IsMatch(message) ? LineKind.Warning : LineKind.Info;
        }
        var t = raw.TrimStart();
        if (t.StartsWith("ERROR", StringComparison.Ordinal)) return LineKind.Error;
        if (t.StartsWith("WARNING", StringComparison.Ordinal)) return LineKind.Warning;
        if (CheckProblem().IsMatch(raw)) return LineKind.Error;
        if (t.StartsWith("Stopped.", StringComparison.Ordinal)) { StoppedLineSeen = true; return LineKind.Warning; }
        if (ErrorWords().IsMatch(t) && !t.Contains(" 0 failed") && !Regex.IsMatch(t, @"\b0 errors?\b")) return LineKind.Error;
        if (WarningWords().IsMatch(t) && !Regex.IsMatch(t, @"\b0 warnings?\b")) return LineKind.Warning;
        return LineKind.Info;
    }

    void Interpret(string message)
    {
        var into = BuildingInto().Match(message);
        if (into.Success) { OutFolder = into.Groups[1].Value; WorkFolder = into.Groups[2].Value; }
        if (message.StartsWith("finished in "))
        {
            FinishedLineSeen = true;
            foreach (var s in Stages.Where(s => s.State == StageState.Running)) { s.State = StageState.Done; s.Ended = DateTime.Now; }
            CurrentOperation = message;
            return;
        }
        var keep = Keeping().Match(message);
        if (keep.Success) { KeptShards = int.Parse(keep.Groups[1].Value); ShardCount = int.Parse(keep.Groups[2].Value); }
        var shardArg = ShardArg().Match(message);
        if (shardArg.Success) ShardCount = int.Parse(shardArg.Groups[1].Value);

        var sm = StageMsg().Match(message);
        if (!sm.Success)
        {
            if (!message.StartsWith("  ")) CurrentOperation = message;
            return;
        }
        var stage = sm.Groups[1].Value;
        var text = sm.Groups[2].Value;
        CurrentOperation = message;
        if (stage == "preflight") return;
        var cc = CheckCount().Match(message);
        if (cc.Success) CheckFiles = int.Parse(cc.Groups[1].Value);
        if (stage == "assets" && (text.Contains(" from ") || text.StartsWith("texture pool") || text.StartsWith("audio")))
            AssetSteps++;
        if (stage == "shaders" && text.StartsWith("translating RDNA2")) translateStarted = DateTime.Now;
        Enter(stage, text);
    }

    /// <summary>A stage line: that stage runs now, and the stages before it in the list are finished.</summary>
    void Enter(string stage, string step)
    {
        var index = Stages.FindIndex(s => s.Name == stage);
        if (index < 0) return;
        for (var i = 0; i < index; i++)
            if (Stages[i].State == StageState.Running) { Stages[i].State = StageState.Done; Stages[i].Ended = DateTime.Now; }
        var s = Stages[index];
        if (s.State == StageState.Pending) { s.State = StageState.Running; s.Started = DateTime.Now; }
        s.Step = step;
        if (step == "done" || step.StartsWith("built") || step.StartsWith("all files present"))
        {
            s.State = StageState.Done;
            s.Ended = DateTime.Now;
        }
    }

    void InterpretToolLine(string raw)
    {
        var t = raw.Trim();
        var shard = ShardSummary().Match(t);
        if (shard.Success)
        {
            ShardSummaries++;
            SummaryShaders += int.Parse(shard.Groups[1].Value);
            SummaryTranslated += int.Parse(shard.Groups[2].Value);
            SummaryValid += int.Parse(shard.Groups[3].Value);
            return;
        }
        if (CheckProblem().IsMatch(raw)) { CheckProblems++; return; }
        var mods = Modules().Match(t);
        if (mods.Success)
        {
            PipelineModulesOk += int.Parse(mods.Groups[2].Value);
            PipelineModulesFailed += int.Parse(mods.Groups[3].Value);
            return;
        }
        var pipes = Pipes().Match(t);
        if (pipes.Success)
        {
            PipelineSummaries++;
            PipelinesOk += int.Parse(pipes.Groups[2].Value);
            PipelinesFailed += int.Parse(pipes.Groups[3].Value);
            PipelinesSkipped += int.Parse(pipes.Groups[4].Value);
        }
    }

    /// <summary>Marks the run's end: a running stage either finished (exit 0) or is where it failed or stopped.</summary>
    public void Complete(int exitCode, bool stopped)
    {
        lock (gate)
        {
            foreach (var s in Stages.Where(s => s.State == StageState.Running))
            {
                s.State = exitCode == 0 ? StageState.Done : StageState.Failed;
                s.Ended = DateTime.Now;
            }
            if (exitCode == 0)
                foreach (var s in Stages.Where(s => s.State == StageState.Pending)) s.State = StageState.Skipped;
            CurrentOperation = exitCode == 0 ? "Finished" : stopped ? "Stopped" : $"Stopped with exit code {exitCode}";
        }
    }

    /// <summary>Reads the progress reports the shader tools write while they run (no-op without a folder).</summary>
    public void PollProgress()
    {
        if (progressDir == null || !Directory.Exists(progressDir)) return;
        Dictionary<string, JsonElement>? disasm = null;
        var shards = new List<Dictionary<string, JsonElement>>();
        foreach (var file in Directory.EnumerateFiles(progressDir, "*.json"))
        {
            var report = ReadReport(file);
            if (report == null) continue;
            var name = Path.GetFileNameWithoutExtension(file);
            if (name == "disasm") disasm = report;
            else if (name.StartsWith("translate")) shards.Add(report);
        }
        lock (gate)
        {
            int? I(Dictionary<string, JsonElement>? d, string k) =>
                d != null && d.TryGetValue(k, out var v) && v.ValueKind == JsonValueKind.Number ? v.GetInt32() : null;
            int? Sum(string k) => shards.Count == 0 ? null : shards.Sum(s => I(s, k) ?? 0);
            bool Finished(Dictionary<string, JsonElement> s) => s.TryGetValue("finished", out var f) && f.ValueKind == JsonValueKind.True;
            var total = Sum("total");
            var done = Sum("done");
            double? speed = null, perShader = null;
            TimeSpan? eta = null;
            if (done is int d && shards.Count > 0)
            {
                var now = DateTime.Now;
                speedSamples.Add((now, d));
                speedSamples.RemoveAll(x => now - x.t > TimeSpan.FromSeconds(60));
                var first = speedSamples[0];
                var span = (now - first.t).TotalSeconds;
                if (span >= 5 && d > first.done) speed = (d - first.done) / span;
                // Process time per shader: each shard translates one shader at a time, so the time the shards have
                // run, summed, over the shaders they finished.
                double busy = 0;
                foreach (var s in shards)
                {
                    if (s.TryGetValue("started", out var st) && s.TryGetValue("updated", out var up) &&
                        st.ValueKind == JsonValueKind.Number && up.ValueKind == JsonValueKind.Number)
                        busy += up.GetDouble() - st.GetDouble();
                }
                if (d > 0 && busy > 0) perShader = busy / d;
                if (speed is double sp && sp > 0 && total is int tt) eta = TimeSpan.FromSeconds((tt - d) / sp);
            }
            // The disassembly's own speed, the same way: shaders finished over the last minute.
            double? dSpeed = null;
            TimeSpan? dEta = null;
            if (I(disasm, "done") is int dd && I(disasm, "total") is int dtot)
            {
                var now = DateTime.Now;
                disasmSamples.Add((now, dd));
                disasmSamples.RemoveAll(x => now - x.t > TimeSpan.FromSeconds(60));
                var f = disasmSamples[0];
                var span = (now - f.t).TotalSeconds;
                if (span >= 5 && dd > f.done) dSpeed = (dd - f.done) / span;
                if (dSpeed is double ds && ds > 0 && dd < dtot) dEta = TimeSpan.FromSeconds((dtot - dd) / ds);
            }
            string? current = shards.Where(s => !Finished(s) && s.ContainsKey("current"))
                                    .Select(s => s["current"].GetString()).LastOrDefault();
            Shaders = new ShaderCounts(
                I(disasm, "total"), I(disasm, "done"), I(disasm, "reused"), I(disasm, "failed"),
                total, done, Sum("translated"), Sum("valid"), Sum("failed"),
                shards.Count(s => !Finished(s)), shards.Count, speed, perShader, eta, current, dSpeed, dEta);
        }
    }

    static Dictionary<string, JsonElement>? ReadReport(string file)
    {
        try
        {
            // Share delete: the tool replaces the report while it is open here (os.replace).
            using var fs = new FileStream(file, FileMode.Open, FileAccess.Read, FileShare.ReadWrite | FileShare.Delete);
            return JsonSerializer.Deserialize<Dictionary<string, JsonElement>>(fs);
        }
        catch (Exception e) when (e is IOException or JsonException or UnauthorizedAccessException) { return null; }
    }

    public DateTime? TranslateStarted => translateStarted;
}
