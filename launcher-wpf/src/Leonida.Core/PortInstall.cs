namespace Leonida.Core;

/// <summary>
/// What a finished build left in the output folder, and how Play starts it. The runtime stage copies gta6rt.exe and
/// gta6rt_tls_host.dll into --out and writes game_path.txt with the game folder; the port runs the package's
/// executable (game/xbox-1.0.2.1.json "expected_exe") through gta6rt.exe --run, as runtime/README.md shows.
/// </summary>
public sealed class PortInstall
{
    public const string ExpectedExe = "game_scarlett_master.exe";
    static readonly string[] RuntimeFiles = ["gta6rt.exe", "gta6rt_tls_host.dll"];

    public required string OutFolder { get; init; }
    public string Runtime => Path.Combine(OutFolder, "gta6rt.exe");
    public string? GameFolder { get; init; }
    public string? GameExe => GameFolder == null ? null : Path.Combine(GameFolder, ExpectedExe);
    public bool RuntimeBuilt { get; init; }
    public bool GameExePresent { get; init; }
    public bool ShadersTranslated { get; init; }
    public bool PipelineCache { get; init; }
    public bool CanPlay => RuntimeBuilt && GameExePresent;

    /// <summary>Why Play is not available, or null when it is.</summary>
    public string? WhyNotPlayable =>
        !RuntimeBuilt ? "Build the runtime stage first: it puts gta6rt.exe in the output folder." :
        GameFolder == null ? "The output folder has no game_path.txt; build the runtime stage again." :
        !GameExePresent ? $"{ExpectedExe} is missing from your game copy, so there is nothing to run yet." : null;

    public static PortInstall Inspect(string outFolder, string? workFolder = null)
    {
        string? game = null;
        try
        {
            var gp = Path.Combine(outFolder, "game_path.txt");
            if (File.Exists(gp)) game = File.ReadAllText(gp).Trim();
        }
        catch (IOException) { }
        var work = workFolder ?? Path.Combine(outFolder, "work");
        return new PortInstall
        {
            OutFolder = outFolder,
            GameFolder = string.IsNullOrWhiteSpace(game) ? null : game,
            RuntimeBuilt = RuntimeFiles.All(f => File.Exists(Path.Combine(outFolder, f))),
            GameExePresent = !string.IsNullOrWhiteSpace(game) && File.Exists(Path.Combine(game, ExpectedExe)),
            ShadersTranslated = File.Exists(Path.Combine(work, "shaders", "spirv", "index.json")),
            PipelineCache = File.Exists(Path.Combine(outFolder, "pipeline_cache.bin")),
        };
    }

    /// <summary>gta6rt.exe's arguments for Play (or for the imports check with dryRun).</summary>
    public List<string> PlayArguments(PlayOptions o, bool dryRun = false)
    {
        var a = new List<string> { dryRun ? "--dry-run" : "--run", GameExe ?? throw new InvalidOperationException("no game folder") };
        a.Add("--game-dir");
        a.Add(GameFolder!);
        if (!string.IsNullOrWhiteSpace(o.Overlay)) { a.Add("--overlay"); a.Add(o.Overlay.Trim()); }
        if (!string.IsNullOrWhiteSpace(o.Mods)) { a.Add("--mods"); a.Add(o.Mods.Trim()); }
        if (o.TraceD3D12 && !dryRun) a.Add("--trace-d3d12");
        if (!string.IsNullOrWhiteSpace(o.TraceFile)) { a.Add("--trace"); a.Add(o.TraceFile.Trim()); }
        if (!dryRun && !string.IsNullOrWhiteSpace(o.GameArguments)) { a.Add("--"); a.AddRange(CommandLine.Split(o.GameArguments)); }
        return a;
    }

    public List<string> PlayProblems(PlayOptions o)
    {
        var p = new List<string>();
        if (WhyNotPlayable is string why) p.Add(why);
        if (!string.IsNullOrWhiteSpace(o.TraceFile) && File.Exists(o.TraceFile))
            p.Add("The trace file already exists; gta6rt refuses it so an input is never overwritten. Choose a new name.");
        foreach (var (n, v) in new[] { ("overlay", o.Overlay), ("mods", o.Mods) })
            if (!string.IsNullOrWhiteSpace(v) && PathChecks.DestinationProblem(v) is string bad) p.Add($"The {n} folder: {bad}");
        return p;
    }
}
