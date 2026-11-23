using System.IO;
using System.Text.Json;
using System.Windows;
using System.Windows.Controls;

namespace Leonida.Launcher.Views;

/// <summary>
/// Achievements, as a test page: the list comes from a file on this PC (the launcher ships none of the game's text),
/// and every achievement is locked, since nothing can unlock one until Leonida can take over the game's own unlocks.
/// </summary>
public partial class AchievementsView : UserControl
{
    /// <summary>[{ "name": …, "description": …, "gamerscore": 10, "hidden": false }, …]</summary>
    public static string ListPath => Path.Combine(Path.GetDirectoryName(Leonida.Core.AppPaths.Settings)!, "achievements.json");

    public sealed record Item(string Name, string Description, int Gamerscore, string State);

    public AchievementsView()
    {
        InitializeComponent();
        Loaded += (_, _) => Load();
    }

    public void Load()
    {
        FilePath.Text = ListPath;
        FileProblem.Visibility = Visibility.Collapsed;
        var items = new List<Item>();
        if (File.Exists(ListPath))
        {
            try
            {
                using var doc = JsonDocument.Parse(File.ReadAllText(ListPath), new JsonDocumentOptions { AllowTrailingCommas = true, CommentHandling = JsonCommentHandling.Skip });
                foreach (var e in doc.RootElement.EnumerateArray())
                {
                    string S(string k) => e.TryGetProperty(k, out var v) && v.ValueKind == JsonValueKind.String ? v.GetString()! : "";
                    var hidden = e.TryGetProperty("hidden", out var h) && h.ValueKind == JsonValueKind.True;
                    var g = e.TryGetProperty("gamerscore", out var gs) && gs.TryGetInt32(out var n) ? n : 0;
                    items.Add(new Item(hidden ? Loc.T("Secret achievement") : S("name"),
                                       hidden ? Loc.T("Details are hidden until it unlocks.") : S("description"), g, Loc.T("LOCKED")));
                }
            }
            catch (Exception ex) when (ex is JsonException or InvalidOperationException or IOException)
            {
                FileProblem.Text = Loc.F("The file could not be read: {0}", ex.Message);
                FileProblem.Visibility = Visibility.Visible;
                items.Clear();
            }
        }
        List.ItemsSource = items;
        Empty.Visibility = items.Count == 0 ? Visibility.Visible : Visibility.Collapsed;
        Count.Text = Loc.F("0 / {0}", items.Count);
        Score.Text = Loc.F("0 of {0} G unlocked", items.Sum(i => i.Gamerscore));
    }

    void OnOpenFolder(object sender, RoutedEventArgs e)
    {
        Directory.CreateDirectory(Path.GetDirectoryName(ListPath)!);
        AppState.Open(Path.GetDirectoryName(ListPath)!);
    }

    void OnReload(object sender, RoutedEventArgs e) => Load();
}
