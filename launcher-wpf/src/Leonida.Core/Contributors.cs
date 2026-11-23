using System.Diagnostics;
using System.Net.Http;
using System.Text;
using System.Text.Json;

namespace Leonida.Core;

/// <param name="Avatar">The GitHub profile picture, when the list came from GitHub.</param>
/// <param name="Profile">The GitHub profile page, when known.</param>
public sealed record Contributor(string Name, int Commits, string? Avatar = null, string? Profile = null);

public enum ContributorSource { GitHub, GitHubCache, GitHistory }

/// <summary>
/// The people who wrote Leonida. From GitHub's contributor list of ItsNotRexed/Leonida when it can be fetched (names,
/// profile pictures, commits; cached for a day), else from the checkout's own history (git shortlog). AI assistants
/// and bots are not people, so they are left out either way: Claude and Claude Code (anthropic.com addresses, names
/// starting with "Claude"), accounts GitHub marks as bots, [bot] accounts and Devin.
/// </summary>
public static class Contributors
{
    public const string Repo = "ItsNotRexed/Leonida";

    public static bool IsExcluded(string name, string email) =>
        email.EndsWith("@anthropic.com", StringComparison.OrdinalIgnoreCase) ||
        name.StartsWith("Claude", StringComparison.OrdinalIgnoreCase) ||
        name.StartsWith("Devin", StringComparison.OrdinalIgnoreCase) ||
        name.Contains("[bot]", StringComparison.OrdinalIgnoreCase) ||
        email.Contains("[bot]", StringComparison.OrdinalIgnoreCase);

    // ---------------------------------------------------------------- GitHub

    /// <summary>GitHub's contributors list (a public API, no sign-in), from the cache when it is under a day old.
    /// Null when it can't be had (offline, rate-limited) and there is no cache.</summary>
    public static async Task<(List<Contributor> list, ContributorSource source, DateTime when)?> LoadGitHubAsync(
        string cachePath, CancellationToken ct = default)
    {
        string? cached = null;
        DateTime cachedAt = DateTime.MinValue;
        try
        {
            if (File.Exists(cachePath)) { cached = await File.ReadAllTextAsync(cachePath, ct); cachedAt = File.GetLastWriteTime(cachePath); }
        }
        catch (IOException) { }
        if (cached != null && DateTime.Now - cachedAt < TimeSpan.FromDays(1))
            return (ParseGitHub(cached), ContributorSource.GitHubCache, cachedAt);
        try
        {
            using var http = new HttpClient { Timeout = TimeSpan.FromSeconds(15) };
            http.DefaultRequestHeaders.UserAgent.ParseAdd("LeonidaLauncher/1.0");   // a generic client name, nothing personal
            http.DefaultRequestHeaders.Accept.ParseAdd("application/vnd.github+json");
            var all = new List<JsonElement>();
            for (var page = 1; page <= 10; page++)
            {
                var text = await http.GetStringAsync($"https://api.github.com/repos/{Repo}/contributors?per_page=100&page={page}", ct);
                using var doc = JsonDocument.Parse(text);
                var items = doc.RootElement.EnumerateArray().Select(e => e.Clone()).ToList();
                all.AddRange(items);
                if (items.Count < 100) break;
            }
            var json = JsonSerializer.Serialize(all);
            Directory.CreateDirectory(Path.GetDirectoryName(cachePath)!);
            await File.WriteAllTextAsync(cachePath, json, ct);
            return (ParseGitHub(json), ContributorSource.GitHub, DateTime.Now);
        }
        catch (Exception e) when (e is HttpRequestException or TaskCanceledException or JsonException or IOException)
        {
            return cached != null ? (ParseGitHub(cached), ContributorSource.GitHubCache, cachedAt) : null;
        }
    }

    /// <summary>GitHub's contributors JSON (an array of {login, avatar_url, html_url, contributions, type}).</summary>
    public static List<Contributor> ParseGitHub(string json)
    {
        var list = new List<Contributor>();
        using var doc = JsonDocument.Parse(json);
        foreach (var e in doc.RootElement.EnumerateArray())
        {
            var login = e.TryGetProperty("login", out var l) ? l.GetString() ?? "" : "";
            var type = e.TryGetProperty("type", out var t) ? t.GetString() ?? "" : "";
            if (login.Length == 0 || type.Equals("Bot", StringComparison.OrdinalIgnoreCase) || IsExcluded(login, "")) continue;
            var commits = e.TryGetProperty("contributions", out var c) && c.TryGetInt32(out var n) ? n : 0;
            var avatar = e.TryGetProperty("avatar_url", out var a) ? a.GetString() : null;
            var profile = e.TryGetProperty("html_url", out var h) ? h.GetString() : null;
            list.Add(new Contributor(login, commits, avatar, profile));
        }
        return list.OrderByDescending(x => x.Commits).ThenBy(x => x.Name, StringComparer.OrdinalIgnoreCase).ToList();
    }

    // ---------------------------------------------------------------- git history

    /// <summary>Parses `git shortlog -sne` output ("  123\tName &lt;email&gt;").</summary>
    public static List<Contributor> Parse(string shortlog)
    {
        var byName = new Dictionary<string, int>(StringComparer.OrdinalIgnoreCase);
        var display = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase);
        foreach (var raw in shortlog.Split('\n'))
        {
            var line = raw.Trim();
            var tab = line.IndexOf('\t');
            if (tab < 0 || !int.TryParse(line[..tab].Trim(), out var commits)) continue;
            var who = line[(tab + 1)..].Trim();
            var lt = who.LastIndexOf('<');
            var name = (lt >= 0 ? who[..lt] : who).Trim();
            var email = lt >= 0 ? who[(lt + 1)..].TrimEnd('>').Trim() : "";
            if (name.Length == 0 || IsExcluded(name, email)) continue;
            byName[name] = byName.GetValueOrDefault(name) + commits;
            display.TryAdd(name, name);
        }
        return byName.Select(kv => new Contributor(display[kv.Key], kv.Value))
                     .OrderByDescending(c => c.Commits).ThenBy(c => c.Name, StringComparer.OrdinalIgnoreCase).ToList();
    }

    public static async Task<List<Contributor>> LoadAsync(string repo, CancellationToken ct = default)
    {
        var psi = new ProcessStartInfo("git") { UseShellExecute = false, CreateNoWindow = true, RedirectStandardOutput = true,
            RedirectStandardError = true, StandardOutputEncoding = Encoding.UTF8, WorkingDirectory = repo };
        foreach (var a in new[] { "-c", "i18n.logOutputEncoding=UTF-8", "shortlog", "-sne", "HEAD" }) psi.ArgumentList.Add(a);
        using var p = Process.Start(psi) ?? throw new InvalidOperationException("git did not start");
        var output = await p.StandardOutput.ReadToEndAsync(ct);
        await p.WaitForExitAsync(ct);
        if (p.ExitCode != 0) throw new InvalidOperationException((await p.StandardError.ReadToEndAsync(ct)).Trim());
        return Parse(output);
    }
}
