using System.Text;
using System.Text.RegularExpressions;

namespace Leonida.Core;

public enum GpuTestOutcome { Pass, Fail, Skip }

public sealed record GpuTestResult(string Name, GpuTestOutcome Outcome, string Detail);

/// <summary>
/// The repository's GPU test suite, scripts/run_gpu_tests.py, for the GPU tests page: where its test programs are,
/// what its results block says, and the "Tested … on …" line the coordination issue asks testers to post.
/// </summary>
public static partial class GpuTests
{
    /// <summary>The program every run needs; a folder with it is a test-program folder.</summary>
    public const string Marker = "vk_compute_test.exe";

    /// <summary>Where docs/testing.md's build commands put the test programs, most likely first.</summary>
    public static readonly string[] Candidates =
        [@"runtime\build", @"runtime\build\Release", @"runtime\build\RelWithDebInfo", @"runtime\build\vulkan", @"runtime\build\windows"];

    public static string? FindPrograms(string repo) =>
        Candidates.Select(c => Path.Combine(repo, c)).FirstOrDefault(d => File.Exists(Path.Combine(d, Marker)));

    public static bool HasPrograms(string? folder) => !string.IsNullOrWhiteSpace(folder) && File.Exists(Path.Combine(folder, Marker));

    /// <summary>The results block after "=== GPU TEST RESULTS ===": one "  PASS  name  detail" line per test.</summary>
    public static List<GpuTestResult> Parse(IEnumerable<string> lines)
    {
        var results = new List<GpuTestResult>();
        var inBlock = false;
        foreach (var line in lines)
        {
            if (line.Contains("=== GPU TEST RESULTS ===")) { inBlock = true; results.Clear(); continue; }
            if (!inBlock) continue;
            var m = ResultLine().Match(line);
            if (!m.Success) continue;
            var outcome = m.Groups[1].Value switch { "PASS" => GpuTestOutcome.Pass, "FAIL" => GpuTestOutcome.Fail, _ => GpuTestOutcome.Skip };
            results.Add(new GpuTestResult(m.Groups[2].Value, outcome, m.Groups[3].Value.Trim()));
        }
        return results;
    }

    /// <summary>
    /// The comment for a pull request or the coordination issue: commit, GPU, driver, OS, the command and the
    /// counts, then each failure. Only names go in, never a local path.
    /// </summary>
    public static string Report(string commit, string gpu, string? driver, string os, string? filter, IReadOnlyList<GpuTestResult> results)
    {
        int pass = results.Count(r => r.Outcome == GpuTestOutcome.Pass), skip = results.Count(r => r.Outcome == GpuTestOutcome.Skip);
        var failed = results.Where(r => r.Outcome == GpuTestOutcome.Fail).ToList();
        var verdict = failed.Count == 0 && pass > 0 ? "pass" : failed.Count > 0 ? $"{failed.Count} failed" : "nothing passed";
        var cmd = "python -I scripts/run_gpu_tests.py" + (string.IsNullOrWhiteSpace(filter) ? "" : $" --filter {filter.Trim()}");
        var sb = new StringBuilder();
        sb.Append($"Tested `{commit}` on {gpu}{(string.IsNullOrWhiteSpace(driver) ? "" : $", driver {driver}")}, {os}: **{verdict}**.\n\n");
        sb.Append($"`{cmd}`: {pass}/{results.Count} passed{(skip > 0 ? $", {skip} skipped" : "")}.\n");
        if (failed.Count > 0)
        {
            sb.Append("\nFailed:\n\n");
            foreach (var f in failed) sb.Append($"- `{f.Name}`{(f.Detail.Length > 0 ? ": " + Scrub(f.Detail) : "")}\n");
        }
        return sb.ToString();
    }

    /// <summary>A detail line without anything that looks like a local path (C:\…, /home/…).</summary>
    public static string Scrub(string text) => PathLike().Replace(text, "<path>");

    [GeneratedRegex(@"^\s+(PASS|FAIL|SKIP)\s+(\S+)\s*(.*)$")] private static partial Regex ResultLine();
    [GeneratedRegex(@"(?:[A-Za-z]:[\\/]|\\\\|/(?:home|Users|mnt|tmp)/)[^\s""'`,;)]*")] private static partial Regex PathLike();
}
