namespace Leonida.Core;

/// <summary>
/// Every option make_game.py takes (Build.bat passes its arguments straight through), as the launcher keeps them.
/// <see cref="ToArguments"/> turns them into the same command line a user would type after Build.bat, so the
/// builder's own defaults apply wherever an option is left unset. --pick and --find are interactive in the
/// terminal; the launcher does both itself (a folder dialog, and tools/launcher_probe.py --find, which calls the
/// same find_dumps) and always passes --game, so make_game.py never waits for keyboard input.
/// </summary>
public sealed class BuildOptions
{
    /// <summary>make_game.py's stages, in the order it runs them (STAGES).</summary>
    public static readonly string[] AllStages = ["check", "shaders", "assets", "runtime", "pipelines"];
    public static readonly string[] Wave64Modes = ["auto", "on", "off"];

    public string? Game { get; set; }                 // --game
    public string? Out { get; set; }                  // --out    (unset: Documents\Leonida)
    public string? Work { get; set; }                 // --work   (unset: <out>\work)
    public List<string> Stages { get; set; } = [.. AllStages];   // --stages (all = left out)
    public bool CheckOnly { get; set; }               // --check
    public bool FullCheck { get; set; }               // --full-check
    public bool Wav { get; set; }                     // --wav
    public int? Jobs { get; set; }                    // --jobs   (unset: half the CPU threads)
    public string? RuntimeBuild { get; set; }         // --runtime-build DIR
    public string Wave64Emulate { get; set; } = "auto";   // --wave64-emulate
    public bool NoFloatControls { get; set; }         // --no-float-controls
    public bool BaryFallback { get; set; }            // --bary-fallback
    public bool TexelOffsetFallback { get; set; }     // --texel-offset-fallback
    public bool F16InterpFallback { get; set; }       // --f16-interp-fallback
    public bool MeshCompute { get; set; }             // --mesh-compute
    public bool PcTailUbo { get; set; }               // --pc-tail-ubo
    /// <summary>Anything else, passed through unchanged after the options above (for options added later).</summary>
    public string? ExtraArguments { get; set; }

    public BuildOptions Clone()
    {
        var c = (BuildOptions)MemberwiseClone();
        c.Stages = [.. Stages];
        return c;
    }

    /// <summary>The selected stages in make_game.py's order, whatever order they were ticked in.</summary>
    public IReadOnlyList<string> OrderedStages() => AllStages.Where(s => Stages.Contains(s)).ToList();

    public List<string> ToArguments()
    {
        var a = new List<string>();
        void Path(string flag, string? value)
        {
            if (!string.IsNullOrWhiteSpace(value)) { a.Add(flag); a.Add(value.Trim()); }
        }
        Path("--game", Game);
        Path("--out", Out);
        Path("--work", Work);
        var stages = OrderedStages();
        if (stages.Count != AllStages.Length) a.Add("--stages=" + string.Join(",", stages));   // '=': never read as a flag
        if (CheckOnly) a.Add("--check");
        if (FullCheck) a.Add("--full-check");
        if (Wav) a.Add("--wav");
        if (Jobs is int j) { a.Add("--jobs"); a.Add(j.ToString(System.Globalization.CultureInfo.InvariantCulture)); }
        Path("--runtime-build", RuntimeBuild);
        if (Wave64Emulate != "auto") { a.Add("--wave64-emulate"); a.Add(Wave64Emulate); }
        if (NoFloatControls) a.Add("--no-float-controls");
        if (BaryFallback) a.Add("--bary-fallback");
        if (TexelOffsetFallback) a.Add("--texel-offset-fallback");
        if (F16InterpFallback) a.Add("--f16-interp-fallback");
        if (MeshCompute) a.Add("--mesh-compute");
        if (PcTailUbo) a.Add("--pc-tail-ubo");
        if (!string.IsNullOrWhiteSpace(ExtraArguments)) a.AddRange(CommandLine.Split(ExtraArguments));
        return a;
    }

    /// <summary>Problems that would make make_game.py stop at once; an empty list means it may start.</summary>
    public List<string> Validate()
    {
        var problems = new List<string>();
        if (string.IsNullOrWhiteSpace(Game))
            problems.Add("Choose your game folder (the one that holds MicrosoftGame.Config), or use Find copies.");
        else if (!Directory.Exists(Game))
            problems.Add($"The game folder does not exist: {Game}");
        else if (!File.Exists(System.IO.Path.Combine(Game, "MicrosoftGame.Config")))
            problems.Add($"{Game} has no MicrosoftGame.Config: choose the package folder itself.");
        if (!CheckOnly && Stages.Count == 0) problems.Add("Select at least one stage.");
        foreach (var (name, value) in new[] { ("output", Out), ("work", Work) })
        {
            if (string.IsNullOrWhiteSpace(value)) continue;
            var why = PathChecks.DestinationProblem(value);
            if (why != null) problems.Add($"The {name} folder: {why}");
            else if (!string.IsNullOrWhiteSpace(Game) && PathChecks.IsWithin(value, Game))
                problems.Add($"The {name} folder must be outside the game folder.");
        }
        if (Jobs is int j && j < 1) problems.Add("Parallel jobs must be 1 or more.");
        if (!Wave64Modes.Contains(Wave64Emulate)) problems.Add($"Unknown wave64 mode '{Wave64Emulate}'.");
        if (!string.IsNullOrWhiteSpace(RuntimeBuild) && Stages.Contains("runtime") && !CheckOnly)
        {
            foreach (var f in new[] { "gta6rt.exe", "gta6rt_tls_host.dll" })
                if (!File.Exists(System.IO.Path.Combine(RuntimeBuild, f)))
                    problems.Add($"The runtime build folder has no {f}.");
        }
        return problems;
    }
}

public static class PathChecks
{
    /// <summary>Why a destination folder can't be used, or null: make_game.py's require_folder, before it runs.</summary>
    public static string? DestinationProblem(string path)
    {
        string full;
        try { full = Path.GetFullPath(path.Trim()); }
        catch (Exception e) { return $"'{path}' is not a valid path ({e.Message})"; }
        var probe = full;
        while (!Directory.Exists(probe))
        {
            if (File.Exists(probe)) return $"{probe} is a file, not a folder";
            var parent = Path.GetDirectoryName(probe);
            if (parent == null) return $"the drive of {full} does not exist";
            probe = parent;
        }
        return null;
    }

    public static bool IsWithin(string path, string folder)
    {
        try
        {
            var p = Path.GetFullPath(path).TrimEnd('\\', '/') + "\\";
            var f = Path.GetFullPath(folder).TrimEnd('\\', '/') + "\\";
            return p.StartsWith(f, StringComparison.OrdinalIgnoreCase);
        }
        catch { return false; }
    }
}

/// <summary>Windows command-line splitting for the "extra arguments" box (quotes group, backslashes kept).</summary>
public static class CommandLine
{
    public static List<string> Split(string text)
    {
        var parts = new List<string>();
        var cur = new System.Text.StringBuilder();
        bool quoted = false, any = false;
        foreach (var ch in text)
        {
            if (ch == '"') { quoted = !quoted; any = true; continue; }
            if (char.IsWhiteSpace(ch) && !quoted)
            {
                if (any || cur.Length > 0) parts.Add(cur.ToString());
                cur.Clear(); any = false;
                continue;
            }
            cur.Append(ch);
        }
        if (any || cur.Length > 0) parts.Add(cur.ToString());
        return parts;
    }

    /// <summary>A command line as a user would type it, for display and the console's first line.</summary>
    public static string Join(IEnumerable<string> args) =>
        string.Join(" ", args.Select(a => a.Length == 0 || a.Any(c => char.IsWhiteSpace(c) || c == '"') ? "\"" + a.Replace("\"", "\\\"") + "\"" : a));
}
