using System.Diagnostics;
using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;

namespace Leonida.Core;

/// <summary>The Leonida checkout the launcher drives: the folder with make_game.py and Build.bat.</summary>
public static class RepoLocator
{
    public static bool IsRepo(string? dir) =>
        !string.IsNullOrWhiteSpace(dir) && File.Exists(Path.Combine(dir, "make_game.py")) &&
        Directory.Exists(Path.Combine(dir, "tools"));

    /// <summary>The configured folder if it is a checkout, else the nearest one above the launcher's own folder
    /// (Leonida.exe beside Build.bat, or in launcher\dist), else null and the user chooses one.</summary>
    public static string? Find(string? configured)
    {
        if (IsRepo(configured)) return Path.GetFullPath(configured!);
        for (var dir = new DirectoryInfo(AppContext.BaseDirectory); dir != null; dir = dir.Parent)
            if (IsRepo(dir.FullName)) return dir.FullName;
        return null;
    }
}

public sealed record PythonInfo(string Executable, string Version, string Command);

/// <summary>
/// Python as Build.bat finds it: `py -3.14`, -3.13, -3.12, -3 (3.12 or newer), else `python` on PATH. The launcher
/// then runs the interpreter by its full path (sys.executable): the WindowsApps `python` alias does not see folders
/// other programs make under %LOCALAPPDATA%, such as the llvm-mc that scripts\build_llvm_mc.ps1 used to install
/// there (make_game.py's appdata_hint), and the full path does.
/// </summary>
public static class PythonLocator
{
    const string Probe = "import sys; print(sys.executable); print('%d.%d.%d' % sys.version_info[:3]); " +
                         "sys.exit(0 if sys.version_info >= (3, 12) else 1)";

    public static PythonInfo? Find(string? overridePath = null)
    {
        if (!string.IsNullOrWhiteSpace(overridePath))
            return Ask(overridePath, [], overridePath);
        foreach (var v in new[] { "3.14", "3.13", "3.12", "3" })
        {
            var found = Ask("py", ["-" + v], "py -" + v);
            if (found != null) return found;
        }
        return Ask("python", [], "python");
    }

    static PythonInfo? Ask(string exe, string[] pre, string command)
    {
        try
        {
            var psi = new ProcessStartInfo(exe) { UseShellExecute = false, CreateNoWindow = true,
                RedirectStandardOutput = true, RedirectStandardError = true };
            foreach (var a in pre) psi.ArgumentList.Add(a);
            psi.ArgumentList.Add("-I");
            psi.ArgumentList.Add("-c");
            psi.ArgumentList.Add(Probe);
            using var p = Process.Start(psi);
            if (p == null) return null;
            var output = p.StandardOutput.ReadToEnd();
            p.StandardError.ReadToEnd();
            if (!p.WaitForExit(20000)) { try { p.Kill(true); } catch { } return null; }
            var lines = output.Split('\n', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries);
            if (p.ExitCode != 0 || lines.Length < 2 || !File.Exists(lines[0])) return null;
            return new PythonInfo(lines[0], lines[1], command);
        }
        catch (System.ComponentModel.Win32Exception) { return null; }   // not installed / not on PATH
    }
}

/// <summary>tools/launcher_probe.py's answer: make_game.py's defaults and checks, as JSON.</summary>
public sealed class ProbeResult
{
    [JsonPropertyName("root")] public string Root { get; set; } = "";
    [JsonPropertyName("python")] public string Python { get; set; } = "";
    [JsonPropertyName("stdout_encoding")] public string? StdoutEncoding { get; set; }
    [JsonPropertyName("stages")] public List<string> Stages { get; set; } = [];
    [JsonPropertyName("wave64_modes")] public List<string> Wave64Modes { get; set; } = [];
    [JsonPropertyName("default_out")] public string DefaultOut { get; set; } = "";
    [JsonPropertyName("default_jobs")] public int DefaultJobs { get; set; }
    [JsonPropertyName("cpu_count")] public int CpuCount { get; set; }
    [JsonPropertyName("min_free_gb")] public Dictionary<string, int> MinFreeGb { get; set; } = [];
    [JsonPropertyName("versions")] public List<GameVersion> Versions { get; set; } = [];
    [JsonPropertyName("tools")] public List<ToolCheck> Tools { get; set; } = [];
    [JsonPropertyName("copies")] public List<GameCopy>? Copies { get; set; }
    [JsonPropertyName("unusable")] public List<UnusableCopy>? Unusable { get; set; }
    [JsonPropertyName("gpu")] public JsonElement? Gpu { get; set; }
    [JsonPropertyName("free_bytes")] public long? FreeBytes { get; set; }
}

public sealed class GameVersion
{
    [JsonPropertyName("file")] public string File { get; set; } = "";
    [JsonPropertyName("title")] public string? Title { get; set; }
    [JsonPropertyName("version")] public string? Version { get; set; }
    [JsonPropertyName("files")] public int Files { get; set; }
}

public sealed class ToolCheck
{
    [JsonPropertyName("name")] public string Name { get; set; } = "";
    [JsonPropertyName("ok")] public bool Ok { get; set; }
    [JsonPropertyName("detail")] public string Detail { get; set; } = "";
    [JsonPropertyName("needed_for")] public string NeededFor { get; set; } = "";
    [JsonPropertyName("hint")] public string Hint { get; set; } = "";
    /// <summary>Missing now, but the build makes it when it gets there: nothing to install.</summary>
    [JsonPropertyName("made_by_build")] public bool MadeByBuild { get; set; }
}

public sealed class GameCopy
{
    [JsonPropertyName("path")] public string Path { get; set; } = "";
    [JsonPropertyName("title")] public string? Title { get; set; }
    [JsonPropertyName("version")] public string? Version { get; set; }
}

public sealed class UnusableCopy
{
    [JsonPropertyName("path")] public string Path { get; set; } = "";
    [JsonPropertyName("kind")] public string? Kind { get; set; }
    [JsonPropertyName("version")] public string? Version { get; set; }
}

public static class Probe
{
    /// <summary>Runs tools/launcher_probe.py with the given flags. Throws with the script's own error text when it
    /// fails, so the user sees why (a missing module, a Python too old) instead of an empty page.</summary>
    public static async Task<ProbeResult> RunAsync(PythonInfo python, string repo, IEnumerable<string> flags,
                                                   IDictionary<string, string>? env = null,
                                                   CancellationToken ct = default)
    {
        var psi = new ProcessStartInfo(python.Executable) { UseShellExecute = false, CreateNoWindow = true,
            RedirectStandardOutput = true, RedirectStandardError = true, WorkingDirectory = repo,
            StandardOutputEncoding = Encoding.UTF8 };
        psi.ArgumentList.Add("-I");
        psi.ArgumentList.Add(Path.Combine(repo, "tools", "launcher_probe.py"));
        foreach (var f in flags) psi.ArgumentList.Add(f);
        if (env != null) foreach (var (k, v) in env) psi.Environment[k] = v;
        using var p = Process.Start(psi) ?? throw new InvalidOperationException("could not start Python");
        var stdout = p.StandardOutput.ReadToEndAsync(ct);
        var stderr = p.StandardError.ReadToEndAsync(ct);
        try { await p.WaitForExitAsync(ct); }
        catch (OperationCanceledException) { try { p.Kill(true); } catch { } throw; }
        var text = await stdout;
        if (p.ExitCode != 0)
            throw new InvalidOperationException($"launcher_probe.py exited with {p.ExitCode}: {(await stderr).Trim()}");
        // The JSON is ASCII-escaped by json.dump, so the pipe's code page cannot garble it.
        var last = text.Split('\n', StringSplitOptions.RemoveEmptyEntries).LastOrDefault()
                   ?? throw new InvalidOperationException("launcher_probe.py printed nothing");
        return JsonSerializer.Deserialize<ProbeResult>(last)
               ?? throw new InvalidOperationException("launcher_probe.py printed no JSON object");
    }
}
