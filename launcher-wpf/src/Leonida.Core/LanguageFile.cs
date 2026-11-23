using System.Text.Json;
using System.Text.RegularExpressions;

namespace Leonida.Core;

/// <summary>
/// One translation of the launcher: a JSON file with the language's code, its name in that language, the culture for
/// numbers and dates, and the strings, from the English text as written to the translation.
/// <code>
/// { "code": "pl", "name": "Polski", "culture": "pl-PL", "strings": { "Home": "Strona główna", … } }
/// </code>
/// A string left out (or empty) shows in English. See launcher-wpf/src/Leonida.Launcher/Languages/README.md.
/// </summary>
public sealed partial class LanguageFile
{
    public required string Code { get; init; }
    public required string Name { get; init; }
    public required string Culture { get; init; }
    public required IReadOnlyDictionary<string, string> Strings { get; init; }

    public static LanguageFile English { get; } = new() { Code = "en", Name = "English", Culture = "en-US", Strings = new Dictionary<string, string>() };

    /// <summary>Reads a language file; FormatException says what is wrong with it.</summary>
    public static LanguageFile Parse(string json)
    {
        JsonDocument doc;
        try { doc = JsonDocument.Parse(json, new JsonDocumentOptions { CommentHandling = JsonCommentHandling.Skip, AllowTrailingCommas = true }); }
        catch (JsonException e) { throw new FormatException("not valid JSON: " + e.Message); }
        using (doc)
        {
            var root = doc.RootElement;
            if (root.ValueKind != JsonValueKind.Object) throw new FormatException("the file must be one JSON object");
            string Text(string name)
            {
                if (!root.TryGetProperty(name, out var v) || v.ValueKind != JsonValueKind.String || string.IsNullOrWhiteSpace(v.GetString()))
                    throw new FormatException($"\"{name}\" must be a non-empty string");
                return v.GetString()!.Trim();
            }
            var code = Text("code");
            if (!CodePattern().IsMatch(code)) throw new FormatException($"\"code\" must look like \"pl\" or \"pt-BR\", not \"{code}\"");
            if (!root.TryGetProperty("strings", out var s) || s.ValueKind != JsonValueKind.Object)
                throw new FormatException("\"strings\" must be an object");
            var strings = new Dictionary<string, string>(StringComparer.Ordinal);
            foreach (var p in s.EnumerateObject())
            {
                if (p.Value.ValueKind != JsonValueKind.String) throw new FormatException($"the translation of \"{p.Name}\" must be a string");
                strings[p.Name] = p.Value.GetString()!;
            }
            return new LanguageFile { Code = code, Name = Text("name"), Culture = Text("culture"), Strings = strings };
        }
    }

    /// <summary>The {0}, {1:0.0} … placeholders of a format string, as a sorted list of their indexes.</summary>
    public static List<int> Placeholders(string text) =>
        PlaceholderPattern().Matches(text.Replace("{{", "").Replace("}}", "")).Select(m => int.Parse(m.Groups[1].Value)).Distinct().Order().ToList();

    /// <summary>Translations whose placeholders differ from their English text's: they would show the wrong value or throw.</summary>
    public IEnumerable<string> PlaceholderMismatches() =>
        Strings.Where(kv => kv.Value.Length > 0 && !Placeholders(kv.Key).SequenceEqual(Placeholders(kv.Value))).Select(kv => kv.Key);

    [GeneratedRegex(@"^[a-z]{2,3}(-[A-Za-z0-9]{2,8})*$")] private static partial Regex CodePattern();
    [GeneratedRegex(@"\{(\d+)(?:[,:][^}]*)?\}")] private static partial Regex PlaceholderPattern();
}
