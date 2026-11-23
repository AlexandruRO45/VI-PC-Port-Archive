using System.Diagnostics;
using System.Text;
using System.Text.RegularExpressions;
using Leonida.Core;

// Tests for Leonida.Core. Each check prints PASS/FAIL; the exit code is the number of failures.
var failures = 0;
var checks = 0;
void Check(bool ok, string what)
{
    checks++;
    if (!ok) failures++;
    Console.WriteLine($"{(ok ? "PASS" : "FAIL")} {what}");
}

var repo = RepoLocator.Find(null) ?? throw new InvalidOperationException("run from inside the Leonida checkout");
var temp = Path.Combine(Path.GetTempPath(), "leonida-launcher-tests-" + Environment.ProcessId);
Directory.CreateDirectory(temp);

// ---- every make_game.py option has a control, and every argument the launcher writes is one make_game.py takes ----
{
    var source = File.ReadAllText(Path.Combine(repo, "make_game.py"));
    var flags = Regex.Matches(source, @"ap\.add_argument\(""(--[a-z0-9-]+)""").Select(m => m.Groups[1].Value).ToHashSet();
    var handledOutside = new HashSet<string> { "--pick", "--find" };   // the launcher's folder dialog and copy finder
    var all = new BuildOptions
    {
        Game = @"C:\Games\Package", Out = @"D:\Leonida Out", Work = @"E:\work", Stages = ["shaders", "check"], CheckOnly = true,
        FullCheck = true, Wav = true, Jobs = 3, RuntimeBuild = @"D:\rt", Wave64Emulate = "on", NoFloatControls = true, BaryFallback = true,
        TexelOffsetFallback = true, F16InterpFallback = true, MeshCompute = true, PcTailUbo = true,
    };
    var written = all.ToArguments().Where(a => a.StartsWith("--")).Select(a => a.Split('=')[0]).ToHashSet();
    Check(written.All(flags.Contains), "every flag the launcher writes is one make_game.py defines: " + string.Join(" ", written.Where(f => !flags.Contains(f))));
    var missing = flags.Where(f => !written.Contains(f) && !handledOutside.Contains(f)).ToList();
    Check(missing.Count == 0, "every make_game.py option is covered by the launcher" + (missing.Count > 0 ? ": missing " + string.Join(" ", missing) : ""));
    Check(BuildOptions.AllStages.SequenceEqual(Regex.Match(source, @"STAGES = \[(.*?)\]").Groups[1].Value.Split(',').Select(s => s.Trim().Trim('"'))),
          "the launcher's stage list is make_game.py's STAGES, in order");
    var allArgs = all.ToArguments();
    Check(allArgs.Contains("--stages=check,shaders"), "stages are written in make_game.py's order, whatever order they were ticked in");
    Check(allArgs[allArgs.IndexOf("--out") + 1] == @"D:\Leonida Out", "a path with a space stays one argument");
    Check(allArgs[allArgs.IndexOf("--wave64-emulate") + 1] == "on" && allArgs[allArgs.IndexOf("--jobs") + 1] == "3", "values follow their flags");
}
{
    var defaults = new BuildOptions { Game = @"C:\g" }.ToArguments();
    Check(defaults.SequenceEqual(["--game", @"C:\g"]), "defaults write nothing but --game, so make_game.py's own defaults apply");
    var extra = new BuildOptions { Game = "g", ExtraArguments = "--future-flag \"a b\" x" }.ToArguments();
    Check(extra.SkipWhile(a => a != "--future-flag").SequenceEqual(["--future-flag", "a b", "x"]), "extra arguments pass through, quotes grouping");
}

// ---- validation ----
{
    var game = Path.Combine(temp, "package");
    Directory.CreateDirectory(game);
    var o = new BuildOptions { Game = game };
    Check(o.Validate().Any(p => p.Contains("MicrosoftGame.Config")), "a folder without MicrosoftGame.Config is refused");
    File.WriteAllText(Path.Combine(game, "MicrosoftGame.Config"), "<Game/>");
    Check(o.Validate().Count == 0, "a package folder with defaults is accepted");
    o.Out = Path.Combine(game, "out");
    Check(o.Validate().Any(p => p.Contains("outside the game folder")), "an output folder inside the game folder is refused (as make_game.py does)");
    var file = Path.Combine(temp, "afile");
    File.WriteAllText(file, "x");
    o.Out = Path.Combine(file, "sub");
    Check(o.Validate().Any(p => p.Contains("is a file")), "an output path under a file is refused (require_folder)");
    o.Out = null;
    o.Stages = [];
    Check(o.Validate().Any(p => p.Contains("at least one stage")), "no stage selected is refused");
    o.Stages = ["runtime"];
    o.RuntimeBuild = temp;
    Check(o.Validate().Any(p => p.Contains("gta6rt.exe")), "--runtime-build without gta6rt.exe is refused");
}

// ---- command lines ----
Check(CommandLine.Split("a \"b c\" \"\" d").SequenceEqual(["a", "b c", "", "d"]), "command-line split keeps quoted spaces and empty quotes");
Check(CommandLine.Join(["python", @"C:\a b\x.py", "--out", @"D:\x"]) == "python \"C:\\a b\\x.py\" --out D:\\x", "command-line join quotes what needs it");

// ---- the monitor, on make_game.py's own line formats ----
{
    var m = new BuildMonitor(["check", "shaders", "runtime"], null);
    string[] lines =
    [
        "10:00:00  check: Xbox Series package 1.0.2.1, 108 files",
        "10:00:01  check: all files present. Building from it.",
        "10:00:01  preflight: the shaders in D:\\w can be reused; not counting their 30 GB",
        "10:00:02  building into D:\\Games\\Leonida (work files in D:\\Games\\Leonida\\work)",
        "10:00:02  shaders: carving shaders.rpf",
        "10:00:03    shader_carve.py C:\\g D:\\w",
        "10:01:00  shaders: translating RDNA2 to SPIR-V on 8 processes (the longest step)",
        "10:01:00    keeping 2 of 8 shards finished by the run before",
        "3000 shaders: 2990 translated, 2985 pass spirv-val",
        "3000 shaders: 3000 translated, 3000 pass spirv-val",
        "WARNING: cannot remove the stale index",
        "10:30:00  shaders: done",
        "10:30:01  runtime: building",
        "D:\\x.cpp(1): warning C4996: 'u8path' deprecated",
        "LINK : fatal error LNK1104: cannot open file 'gta6rt.exe'",
        "ERROR: runtime: build did not produce D:\\rt\\gta6rt.exe",
    ];
    var kinds = lines.Select(l => m.Feed(l, l.StartsWith("ERROR")).Kind).ToList();
    Check(m.CheckFiles == 108, "the check stage's file count is read");
    Check(m.OutFolder == @"D:\Games\Leonida" && m.WorkFolder == @"D:\Games\Leonida\work", "the output and work folders are read from 'building into'");
    Check(m.Stages[0].State == StageState.Done && m.Stages[1].State == StageState.Done, "check and shaders are done after their last lines");
    Check(m.Stages[2].State == StageState.Running && m.Stages[2].Step == "building", "runtime is running, on its step");
    Check(m.ShardSummaries == 2 && m.SummaryShaders == 6000 && m.SummaryValid == 5985, "shard summaries are summed");
    Check(m.KeptShards == 2 && m.ShardCount == 8, "kept shards are read");
    Check(kinds[5] == LineKind.Command, "a command run() echoes is a command line");
    Check(m.Warnings == 2 && m.Errors == 2, $"warnings and errors are counted (got {m.Warnings} warnings, {m.Errors} errors)");
    Check(m.FirstError!.StartsWith("LINK : fatal error"), "the first error is kept");
    m.Complete(1, stopped: false);
    Check(m.Stages[2].State == StageState.Failed, "a non-zero exit fails the stage that was running");
    var ok = new BuildMonitor(["check", "shaders"], null);
    ok.Feed("10:00:00  check: Xbox Series package 1.0.2.1, 108 files", false);
    ok.Feed("10:00:05  finished in 0 min", false);
    ok.Complete(0, false);
    Check(ok.Stages.All(s => s.State is StageState.Done or StageState.Skipped) && ok.FinishedLineSeen, "a finished run ends every stage");
    var noisy = new BuildMonitor([], null);
    Check(noisy.Feed("cs: 120 modules ok, 0 failed", false).Kind == LineKind.Info && noisy.PipelineModulesOk == 120, "'0 failed' is not an error");
    noisy.Feed("graphics pipelines: 900 ok, 3 failed, 12 skipped", false);
    noisy.Feed("graphics pipelines: 100 ok, 0 failed, 0 skipped", false);
    Check(noisy.PipelinesOk == 1000 && noisy.PipelinesFailed == 3 && noisy.PipelinesSkipped == 12, "pipeline summaries are summed over batches");
    Check(noisy.Feed("  missing  shaders.rpf", false).Kind == LineKind.Error && noisy.CheckProblems == 1, "a missing file in the check is an error");
}

// ---- live shader counts from tools/progress_report.py files ----
{
    var dir = Path.Combine(temp, "progress");
    Directory.CreateDirectory(dir);
    var now = (DateTimeOffset.UtcNow.ToUnixTimeMilliseconds() / 1000.0).ToString("0.###", System.Globalization.CultureInfo.InvariantCulture);
    var now60 = (double.Parse(now, System.Globalization.CultureInfo.InvariantCulture) - 60).ToString("0.###", System.Globalization.CultureInfo.InvariantCulture);
    var now20 = (double.Parse(now, System.Globalization.CultureInfo.InvariantCulture) - 20).ToString("0.###", System.Globalization.CultureInfo.InvariantCulture);
    var now5 = (double.Parse(now, System.Globalization.CultureInfo.InvariantCulture) - 5).ToString("0.###", System.Globalization.CultureInfo.InvariantCulture);
    File.WriteAllText(Path.Combine(dir, "disasm.json"), $"{{\"total\": 26503, \"done\": 26503, \"reused\": 100, \"failed\": 0, \"finished\": true, \"started\": {now60}, \"updated\": {now}}}");
    File.WriteAllText(Path.Combine(dir, "translate.shard0.json"), $"{{\"total\": 100, \"done\": 40, \"translated\": 39, \"valid\": 38, \"failed\": 1, \"current\": \"ps_1\", \"started\": {now20}, \"updated\": {now}}}");
    File.WriteAllText(Path.Combine(dir, "translate.shard1.json"), $"{{\"total\": 100, \"done\": 100, \"translated\": 99, \"valid\": 99, \"failed\": 1, \"finished\": true, \"started\": {now20}, \"updated\": {now5}}}");
    File.WriteAllText(Path.Combine(dir, "translate.shard2.json"), "{\"total\": 1");   // half written: skipped, never a crash
    var m = new BuildMonitor(["shaders"], dir);
    m.PollProgress();
    var s = m.Shaders!;
    Check(s.Total == 200 && s.Done == 140 && s.Valid == 137 && s.Failed == 2, "shard reports are summed");
    Check(s.ActiveShards == 1 && s.Shards == 2 && s.Current == "ps_1", "running shards and the current shader are reported");
    Check(s.DisasmTotal == 26503 && s.DisasmReused == 100, "the disassembly report is read");
    Check(s.SecondsPerShader is double p && Math.Abs(p - 35.0 / 140) < 0.01, "process time per shader is the shards' run time over shaders done");
    Check(s.ShadersPerSecond == null && s.Eta == null, "no speed or ETA from a single sample");
}

// ---- tools/progress_report.py itself, run by Python (the same module the shader tools import) ----
{
    var py = PythonLocator.Find();
    if (py == null) Console.WriteLine("SKIP progress_report.py: no Python 3.12+");
    else
    {
        var dir = Path.Combine(temp, "pyprogress");
        Directory.CreateDirectory(dir);
        var psi = new ProcessStartInfo(py.Executable) { UseShellExecute = false, CreateNoWindow = true, WorkingDirectory = Path.Combine(repo, "tools") };
        foreach (var a in new[] { "-c", "import progress_report as p; r = p.Progress('x'); r.update(total=5, done=1); r.update(total=5, done=2); r.update(force=True, total=5, done=5, finished=True)" })
            psi.ArgumentList.Add(a);
        psi.Environment["LEONIDA_PROGRESS_DIR"] = dir;
        using (var p = Process.Start(psi)!) p.WaitForExit();
        var text = File.ReadAllText(Path.Combine(dir, "x.json"));
        Check(text.Contains("\"done\": 5") && text.Contains("\"finished\": true") && Directory.GetFiles(dir).Length == 1,
              "progress_report.py writes the last report whole and leaves no temporary file");
        var off = new ProcessStartInfo(py.Executable) { UseShellExecute = false, CreateNoWindow = true, WorkingDirectory = Path.Combine(repo, "tools") };
        foreach (var a in new[] { "-c", "import progress_report as p; r = p.Progress('y'); r.update(force=True, done=1); assert not r" }) off.ArgumentList.Add(a);
        off.Environment.Remove("LEONIDA_PROGRESS_DIR");
        using (var p = Process.Start(off)!) { p.WaitForExit(); Check(p.ExitCode == 0, "without LEONIDA_PROGRESS_DIR nothing is written (Build.bat unchanged)"); }
    }
}

// ---- contributors ----
{
    var list = Contributors.Parse(
        "  3308\tItsNotRexed <a@x>\n   14\tClaude <noreply@anthropic.com>\n    2\tClaude Code <noreply@anthropic.com>\n" +
        "    1\tdependabot[bot] <49699333+dependabot[bot]@users.noreply.github.com>\n    5\tDominik <d@x>\n    3\tDominik <other@x>\n" +
        "    1\tDevin AI <158243242+devin-ai-integration[bot]@users.noreply.github.com>\n");
    Check(!list.Any(c => c.Name.StartsWith("Claude")), "Claude and Claude Code are not listed");
    Check(!list.Any(c => c.Name.Contains("[bot]") || c.Name.StartsWith("Devin")), "bots are not listed");
    Check(list.Count == 2 && list[1] == new Contributor("Dominik", 8), "one name with two addresses is counted once");
}

// ---- Play ----
{
    var outF = Path.Combine(temp, "port");
    var game = Path.Combine(temp, "game dir");
    Directory.CreateDirectory(outF);
    Directory.CreateDirectory(game);
    Check(PortInstall.Inspect(outF).WhyNotPlayable!.Contains("runtime stage"), "an empty output folder is not playable");
    File.WriteAllText(Path.Combine(outF, "gta6rt.exe"), "");
    File.WriteAllText(Path.Combine(outF, "gta6rt_tls_host.dll"), "");
    File.WriteAllText(Path.Combine(outF, "game_path.txt"), game + "\n");
    var i = PortInstall.Inspect(outF);
    Check(!i.CanPlay && i.WhyNotPlayable!.Contains(PortInstall.ExpectedExe), "without the game's executable Play explains why");
    File.WriteAllText(Path.Combine(game, PortInstall.ExpectedExe), "");
    i = PortInstall.Inspect(outF);
    Check(i.CanPlay, "the runtime plus the game's executable is playable");
    var a = i.PlayArguments(new PlayOptions { Overlay = @"D:\ov", TraceD3D12 = true, GameArguments = "-x 1" });
    Check(a.SequenceEqual(["--run", Path.Combine(game, PortInstall.ExpectedExe), "--game-dir", game, "--overlay", @"D:\ov", "--trace-d3d12", "--", "-x", "1"]),
          "Play runs gta6rt.exe --run with the options runtime/src/main.cpp takes");
    var dry = i.PlayArguments(new PlayOptions { TraceD3D12 = true, GameArguments = "-x" }, dryRun: true);
    Check(dry[0] == "--dry-run" && !dry.Contains("--trace-d3d12") && !dry.Contains("--"), "the dry run takes only the options --dry-run accepts");
    var trace = Path.Combine(temp, "t.jsonl");
    File.WriteAllText(trace, "");
    Check(i.PlayProblems(new PlayOptions { TraceFile = trace }).Any(p => p.Contains("already exists")), "an existing trace file is refused, as gta6rt refuses it");
}

// ---- settings survive a damaged file ----
{
    var path = Path.Combine(temp, "settings.json");
    File.WriteAllText(path, "{ not json");
    var s = Json.Load<LauncherSettings>(path);
    Check(s.Build.Stages.Count == 5 && File.Exists(path + ".damaged"), "a damaged settings file gives defaults and is kept aside");
    s.Build.Out = @"D:\Games\Leonida";
    Json.Save(path, s);
    Check(Json.Load<LauncherSettings>(path).Build.Out == @"D:\Games\Leonida", "settings round-trip");
}

// ---- hardware ----
Check(HardwareDetector.MemoryTypeName(26) == "DDR4" && HardwareDetector.MemoryTypeName(34) == "DDR5" && HardwareDetector.MemoryTypeName(0) == null,
      "SMBIOS memory type codes");
{
    var h = HardwareDetector.Detect();
    Check(h.Cpu.Length > 0 && h.Threads > 0 && h.RamUsableBytes > 0, $"this PC is detected: {h.Cpu}, {h.Threads} threads, {Format.Bytes(h.RamUsableBytes)}");
    Console.WriteLine($"     GPUs: {string.Join("; ", h.Gpus.Select(g => $"{g.Name} {Format.Bytes(g.MemoryBytes)}{(g.Integrated ? " (integrated)" : "")}"))}; RAM {h.RamGb} GB {h.RamType} {h.RamSpeedMts} MT/s");
}

// ---- a real build process: output, Ctrl+C, and Cancel ending the whole tree ----
{
    var py = PythonLocator.Find();
    if (py == null) Console.WriteLine("SKIP process tests: no Python 3.12+");
    else
    {
        // Lines arrive, in order, in the log and through the event; the exit code is kept; Polish text survives cp1250.
        Encoding.RegisterProvider(CodePagesEncodingProvider.Instance);
        var enc = Encoding.GetEncoding(System.Globalization.CultureInfo.CurrentCulture.TextInfo.ANSICodePage);
        var lines = new List<string>();
        var done = new TaskCompletionSource();
        var s = new BuildSession(py.Executable, ["-I", "-u", "-c", "import sys; print('10:00:00  check: x, 3 files'); print('ERROR: boom', file=sys.stderr); sys.exit(3)"],
                                 temp, Path.Combine(temp, "run1.log"), new BuildMonitor(["check"], null), new Dictionary<string, string?>(), enc);
        s.Line += l => { lock (lines) lines.Add(l.Text); };
        s.Exited += _ => done.TrySetResult();
        s.Start();
        await done.Task.WaitAsync(TimeSpan.FromSeconds(30));
        Check(s.ExitCode == 3 && s.Outcome == SessionOutcome.Failed, "a failing build is reported as failed with its exit code");
        Check(lines.Contains("ERROR: boom") && s.Monitor.Errors == 1 && s.Monitor.CheckFiles == 3, "stdout and stderr lines both arrive and are parsed");
        s.Dispose();
        Check(File.ReadAllText(Path.Combine(temp, "run1.log")).Contains("ERROR: boom"), "every line is in the log file");

        // Ctrl+C: Python raises KeyboardInterrupt, as make_game.py's cli() expects, and prints its own message.
        var stopped = new TaskCompletionSource();
        var stopLines = new List<string>();
        var script = "import time, sys\ntry:\n    print('ready', flush=True)\n    time.sleep(60)\nexcept KeyboardInterrupt:\n    print('Stopped. Run it again to continue where it left off.', file=sys.stderr)\n    sys.exit(130)\n";
        var stop = new BuildSession(py.Executable, ["-I", "-u", "-c", script], temp, Path.Combine(temp, "run2.log"),
                                    new BuildMonitor([], null), new Dictionary<string, string?>(), enc);
        var ready = new TaskCompletionSource();
        stop.Line += l => { lock (stopLines) stopLines.Add(l.Text); if (l.Text == "ready") ready.TrySetResult(); };
        stop.Exited += _ => stopped.TrySetResult();
        stop.Start();
        await ready.Task.WaitAsync(TimeSpan.FromSeconds(30));
        var t0 = DateTime.Now;
        await stop.StopAsync(TimeSpan.FromSeconds(15));
        await stopped.Task.WaitAsync(TimeSpan.FromSeconds(30));
        Check(stop.ExitCode == 130 && stopLines.Any(l => l.StartsWith("Stopped.")) && DateTime.Now - t0 < TimeSpan.FromSeconds(10),
              $"Stop delivers Ctrl+C and the build ends itself cleanly (exit {stop.ExitCode})");
        Check(stop.Outcome == SessionOutcome.Stopped, "a stopped build is reported as stopped");
        stop.Dispose();

        // Cancel: a child and a grandchild started by the build both end with it.
        var pidFile = Path.Combine(temp, "pids.txt");
        var tree = "import subprocess, sys, time\n" +
                   "c = subprocess.Popen([sys.executable, '-c', 'import subprocess,sys,time; g = subprocess.Popen([sys.executable, \"-c\", \"import time; time.sleep(120)\"]); open(sys.argv[1], \"a\").write(str(g.pid) + \"\\\\n\"); time.sleep(120)', sys.argv[1]])\n" +
                   "open(sys.argv[1], 'a').write(str(c.pid) + '\\n')\nprint('ready', flush=True)\ntime.sleep(120)\n";
        var cancelled = new TaskCompletionSource();
        var cancel = new BuildSession(py.Executable, ["-I", "-u", "-c", tree, pidFile], temp, Path.Combine(temp, "run3.log"),
                                      new BuildMonitor([], null), new Dictionary<string, string?>(), enc);
        var up = new TaskCompletionSource();
        cancel.Line += l => { if (l.Text == "ready") up.TrySetResult(); };
        cancel.Exited += _ => cancelled.TrySetResult();
        cancel.Start();
        await up.Task.WaitAsync(TimeSpan.FromSeconds(30));
        for (var k = 0; k < 50 && (!File.Exists(pidFile) || File.ReadAllLines(pidFile).Length < 2); k++) await Task.Delay(100);
        var pids = File.ReadAllLines(pidFile).Where(l => l.Trim().Length > 0).Select(int.Parse).ToList();
        cancel.Cancel();
        await cancelled.Task.WaitAsync(TimeSpan.FromSeconds(30));
        await Task.Delay(500);
        bool Alive(int pid) { try { using var p = Process.GetProcessById(pid); return !p.HasExited; } catch (ArgumentException) { return false; } }
        Check(pids.Count == 2 && pids.All(p => !Alive(p)), $"Cancel ends the build's child and grandchild too ({pids.Count} found, alive: {pids.Count(Alive)})");
        Check(cancel.Outcome == SessionOutcome.Cancelled, "a cancelled build is reported as cancelled");
        cancel.Dispose();
    }
}

// ---- settings: the one Online switch of 1.0.0/1.0.1 becomes GitHub + Can I Run checker
{
    var o = new System.Text.Json.JsonSerializerOptions { PropertyNameCaseInsensitive = true };
    var oldOff = System.Text.Json.JsonSerializer.Deserialize<LauncherSettings>("{\"AllowOnline\": false}", o)!;
    Check(!oldOff.AllowGitHub && !oldOff.AllowChecker, "an old settings file with online features off turns GitHub and the checker off");
    var oldOn = System.Text.Json.JsonSerializer.Deserialize<LauncherSettings>("{\"AllowOnline\": true}", o)!;
    Check(oldOn.AllowGitHub && oldOn.AllowChecker, "an old settings file with online features on keeps both on");
    var split = System.Text.Json.JsonSerializer.Deserialize<LauncherSettings>("{\"AllowGitHub\": false, \"AllowChecker\": true}", o)!;
    Check(!split.AllowGitHub && split.AllowChecker, "the two online switches are read on their own");
    Check(!System.Text.Json.JsonSerializer.Serialize(new LauncherSettings()).Contains("AllowOnline"), "the old switch is never written back");
}

// ---- Disk space: what the launcher may delete
{
    var root = Path.Combine(temp, "disk");
    var work = Path.Combine(root, "out", "work");
    var game = Path.Combine(root, "game");
    Directory.CreateDirectory(Path.Combine(work, "shaders", "dxbc"));
    Directory.CreateDirectory(Path.Combine(work, "pkg"));
    File.WriteAllText(Path.Combine(work, "pkg", "MicrosoftGame.Config"), "<Game/>");
    Directory.CreateDirectory(game);
    Directory.CreateDirectory(Path.Combine(root, "elsewhere"));
    string[] roots = [work];
    Check(DiskUsage.RefuseDelete(Path.Combine(work, "shaders", "dxbc"), roots, game) == null, "a folder in the work folder may be deleted");
    Check(DiskUsage.RefuseDelete(Path.Combine(root, "elsewhere"), roots, game) != null, "a folder outside the work folder is refused");
    Check(DiskUsage.RefuseDelete(Path.Combine(root, "out"), roots, game) != null, "the folder holding the work folder is refused");
    Check(DiskUsage.RefuseDelete(Path.Combine(work, "pkg"), roots, game) != null, "a folder with MicrosoftGame.Config in it is refused");
    Check(DiskUsage.RefuseDelete(work, roots, work) != null, "the game copy is refused even when it is inside the work folder");
    Check(DiskUsage.RefuseDelete(Path.Combine(work, "shaders"), roots, Path.Combine(work, "shaders", "dxbc")) != null, "a folder holding the game copy is refused");
    Check(DiskUsage.RefuseDelete(Path.Combine(work + "-other"), roots, game) != null, "a sibling whose name starts with the work folder's is refused");
    File.WriteAllBytes(Path.Combine(work, "shaders", "dxbc", "a.txt"), new byte[1000]);
    var (bytes, files) = DiskUsage.Measure(Path.Combine(root, "out"), skip: work);
    Check(bytes == 0 && files == 0, "the port's size leaves out the work folder inside it");
    Check(DiskUsage.Measure(work).Bytes >= 1000, "the work folder's files are counted");
}

// ---- Language files
{
    var dir = Path.GetFullPath(Path.Combine(AppContext.BaseDirectory, "..", "..", "..", "..", "..", "src", "Leonida.Launcher", "Languages"));
    var files = Directory.Exists(dir) ? Directory.GetFiles(dir, "*.json") : [];
    Check(files.Length > 0, "the launcher has language files (" + dir + ")");
    foreach (var f in files)
    {
        LanguageFile? lf = null;
        try { lf = LanguageFile.Parse(File.ReadAllText(f)); } catch (FormatException e) { Check(false, $"{Path.GetFileName(f)} reads: {e.Message}"); continue; }
        Check(lf.Code == Path.GetFileNameWithoutExtension(f), $"{Path.GetFileName(f)}: the code matches the file name");
        var bad = lf.PlaceholderMismatches().ToList();
        Check(bad.Count == 0, $"{Path.GetFileName(f)}: every translation keeps its {{0}} placeholders" + (bad.Count > 0 ? ": " + string.Join(" | ", bad.Take(5)) : ""));
        Check(System.Globalization.CultureInfo.GetCultures(System.Globalization.CultureTypes.AllCultures).Any(c => c.Name == lf.Culture), $"{Path.GetFileName(f)}: culture {lf.Culture} exists");
    }
    bool Fails(string json) { try { LanguageFile.Parse(json); return false; } catch (FormatException) { return true; } }
    Check(Fails("{\"code\":\"x y\",\"name\":\"a\",\"culture\":\"en-US\",\"strings\":{}}"), "a bad language code is refused");
    Check(Fails("{\"code\":\"de\",\"name\":\"Deutsch\",\"culture\":\"de-DE\",\"strings\":{\"a\":1}}"), "a translation that is not text is refused");
    Check(Fails("not json"), "a file that is not JSON is refused");
    Check(LanguageFile.Placeholders("{0} of {1:N0}, {{literal}}").SequenceEqual([0, 1]), "placeholders are found, escaped braces are not");
}

// ---- GPU tests: the results block and the report
{
    string[] output =
    [
        "=== rdna2spirv.py ===", "  PASS  not_a_result  (before the block)",
        "", "=== GPU TEST RESULTS ===",
        "  PASS  cs_basic              ok", "  FAIL  cs_wave32_ballot      word 3: 0x1 != 0x0 (C:\\Users\\someone\\work\\x.spv)",
        "  SKIP  ms_mesh               no VK_EXT_mesh_shader", "", "2/3 passed, 1 skipped",
    ];
    var r = GpuTests.Parse(output);
    Check(r.Count == 3 && r[0] is { Name: "cs_basic", Outcome: GpuTestOutcome.Pass } && r[1].Outcome == GpuTestOutcome.Fail && r[2].Outcome == GpuTestOutcome.Skip,
          "the GPU results block is read, and only it");
    var report = GpuTests.Report("abc123def", "AMD Radeon RX 9070", "32.0.1", "Windows 11", null, r);
    Check(report.StartsWith("Tested `abc123def` on AMD Radeon RX 9070, driver 32.0.1, Windows 11: **1 failed**"), "the report names the commit, card, driver and verdict");
    Check(report.Contains("1/3 passed, 1 skipped") && report.Contains("`cs_wave32_ballot`"), "the report has the counts and each failure");
    Check(!report.Contains("someone") && report.Contains("<path>"), "the report carries no local path");
    var dir = Path.Combine(temp, "gpu", "runtime", "build", "Release");
    Directory.CreateDirectory(dir);
    File.WriteAllText(Path.Combine(dir, GpuTests.Marker), "");
    Check(GpuTests.FindPrograms(Path.Combine(temp, "gpu")) == dir, "the test programs are found where docs/testing.md builds them");
}

try { Directory.Delete(temp, true); } catch (IOException) { }
Console.WriteLine($"Leonida.Core.Tests: {checks} checks, {failures} failures");
return failures;
