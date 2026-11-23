using System.Text.Json;

namespace Leonida.Core;

public sealed record Achievement(int Id, string Name, string Description, int Gamerscore, bool Secret);

/// <summary>
/// GTA VI's achievement list, from github.com/Dom-in/Gta6AchivmentsList (the game downloads its own from Xbox Live,
/// which the launcher never contacts). Downloaded when the launcher starts and kept in a cache, so the page works
/// offline; a list in %APPDATA%\Leonida\Launcher\achievements.json wins over both, for testing a list by hand.
/// </summary>
public static class AchievementList
{
    public const string Url = "https://raw.githubusercontent.com/Dom-in/Gta6AchivmentsList/main/achievements.json";
    public const string Page = "https://github.com/Dom-in/Gta6AchivmentsList";

    public static string CachePath => Path.Combine(AppPaths.Local, "achievements.json");
    public static string OverridePath => Path.Combine(AppPaths.Roaming, "achievements.json");

    /// <summary>Raised (on any thread) when a newer list has been saved to the cache.</summary>
    public static event Action? Updated;

    /// <summary>The list's achievements: { "achievements": [ … ] } or a bare [ … ]. FormatException when it isn't one.</summary>
    public static List<Achievement> Parse(string json)
    {
        try
        {
            using var doc = JsonDocument.Parse(json, new JsonDocumentOptions { AllowTrailingCommas = true, CommentHandling = JsonCommentHandling.Skip });
            var root = doc.RootElement;
            var arr = root.ValueKind == JsonValueKind.Array ? root
                : root.ValueKind == JsonValueKind.Object && root.TryGetProperty("achievements", out var a) && a.ValueKind == JsonValueKind.Array ? a
                : throw new FormatException("no achievements array");
            var list = new List<Achievement>();
            foreach (var e in arr.EnumerateArray())
            {
                if (e.ValueKind != JsonValueKind.Object) continue;
                string S(string k) => e.TryGetProperty(k, out var v) && v.ValueKind == JsonValueKind.String ? v.GetString()!.Trim() : "";
                int I(string k) => e.TryGetProperty(k, out var v) && v.TryGetInt32(out var n) ? n : 0;
                bool B(string k) => e.TryGetProperty(k, out var v) && v.ValueKind == JsonValueKind.True;
                var name = S("name");
                if (name.Length == 0) continue;
                list.Add(new Achievement(I("id") is > 0 and var id ? id : list.Count + 1, name, S("description"), Math.Max(0, I("gamerscore")), B("secret") || B("hidden")));
            }
            return list;
        }
        catch (JsonException e) { throw new FormatException(e.Message); }
    }

    /// <summary>The list to show: the hand-made one if there is one, else the downloaded copy; null when neither.</summary>
    public static (List<Achievement> List, string Source, DateTime When)? Load()
    {
        foreach (var (path, source) in new[] { (OverridePath, "file"), (CachePath, "github") })
        {
            try
            {
                if (File.Exists(path)) return (Parse(File.ReadAllText(path)), source, File.GetLastWriteTime(path));
            }
            catch (Exception e) when (e is IOException or FormatException or UnauthorizedAccessException) { }
        }
        return null;
    }

    /// <summary>Downloads the list and keeps it when it reads; false when offline or the file is not a list.</summary>
    public static async Task<bool> RefreshAsync(CancellationToken ct = default)
    {
        try
        {
            using var http = new HttpClient { Timeout = TimeSpan.FromSeconds(15) };
            http.DefaultRequestHeaders.UserAgent.ParseAdd("LeonidaLauncher/1.0");   // a generic client name, nothing personal
            var json = await http.GetStringAsync(Url, ct);
            if (Parse(json).Count == 0) return false;
            var old = File.Exists(CachePath) ? await File.ReadAllTextAsync(CachePath, ct) : null;
            if (old != json)
            {
                Directory.CreateDirectory(Path.GetDirectoryName(CachePath)!);
                var tmp = CachePath + ".tmp";
                await File.WriteAllTextAsync(tmp, json, ct);
                File.Move(tmp, CachePath, true);
            }
            else File.SetLastWriteTime(CachePath, DateTime.Now);
            Updated?.Invoke();
            return true;
        }
        catch (Exception e) when (e is HttpRequestException or TaskCanceledException or IOException or FormatException or UnauthorizedAccessException)
        {
            return false;
        }
    }
}
