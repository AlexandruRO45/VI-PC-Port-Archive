using System.Windows;
using System.Windows.Controls;
using System.Windows.Input;
using Leonida.Core;

namespace Leonida.Launcher.Views;

/// <summary>
/// Achievements, as a test page: GTA VI's list from github.com/Dom-in/Gta6AchivmentsList (downloaded when the
/// launcher starts, kept for offline), every achievement locked, since nothing can unlock one until Leonida can take
/// over the game's own unlocks. A secret achievement's description shows when you click it.
/// </summary>
public partial class AchievementsView : UserControl
{
    readonly AppState state = AppState.Instance;

    /// <summary>One card. Revealed is per session; secrets start hidden.</summary>
    public sealed class Item(Achievement a) : System.ComponentModel.INotifyPropertyChanged
    {
        public Achievement A { get; } = a;
        bool revealed;
        public string Name => A.Name;
        public string Gamerscore => A.Gamerscore.ToString();
        public string Description => A.Secret && !revealed ? Loc.T("Secret achievement: click to show what it is for.") : A.Description;
        public string State => Loc.T(A.Secret ? "LOCKED · SECRET" : "LOCKED");
        public Cursor Cursor => A.Secret && !revealed ? Cursors.Hand : Cursors.Arrow;
        /// <summary>The first letter or digit of the name ("T**s Ass…" → T).</summary>
        public string Initial => A.Name.FirstOrDefault(char.IsLetterOrDigit) is var c and not '\0' ? char.ToUpperInvariant(c).ToString() : "?";
        /// <summary>A gradient from the logo's colours, the same one for an achievement every time.</summary>
        public System.Windows.Media.Brush Art
        {
            get
            {
                System.Windows.Media.Color[] c =
                [
                    System.Windows.Media.Color.FromRgb(0x6B, 0x2F, 0xD6), System.Windows.Media.Color.FromRgb(0xEC, 0x2F, 0x8C),
                    System.Windows.Media.Color.FromRgb(0xFF, 0x8A, 0x3D), System.Windows.Media.Color.FromRgb(0x2A, 0x14, 0x52),
                    System.Windows.Media.Color.FromRgb(0xB4, 0x2A, 0x78), System.Windows.Media.Color.FromRgb(0x14, 0x8F, 0xB8),
                ];
                var h = A.Name.Aggregate(17, (x, ch) => unchecked(x * 31 + ch)) & 0x7fffffff;
                var b = new System.Windows.Media.LinearGradientBrush(c[h % c.Length], c[(h / 7 + 1 + h % c.Length) % c.Length], 45 + h % 90);
                b.Freeze();
                return b;
            }
        }
        public void Reveal() { revealed = true; PropertyChanged?.Invoke(this, new(null)); }
        public event System.ComponentModel.PropertyChangedEventHandler? PropertyChanged;
    }

    public AchievementsView()
    {
        InitializeComponent();
        Loaded += (_, _) => Load();
        AchievementList.Updated += () => Dispatcher.BeginInvoke(() => { if (IsLoaded) Load(); });
    }

    public void Load()
    {
        var got = AchievementList.Load();
        var items = got?.List.Select(a => new Item(a)).ToList() ?? [];
        List.ItemsSource = items;
        Empty.Visibility = items.Count == 0 ? Visibility.Visible : Visibility.Collapsed;
        Count.Text = Loc.F("0 / {0}", items.Count);
        Score.Text = Loc.F("0 of {0} G unlocked", items.Sum(i => i.A.Gamerscore));
        Source.Text = got is { } g
            ? g.Source == "file" ? Loc.F("From your own list: {0}", AchievementList.OverridePath)
                                 : Loc.F("List from {0} · downloaded {1:g}", "github.com/Dom-in/Gta6AchivmentsList", g.When)
            : "";
        EmptyText.Text = Loc.T(state.Settings.Current.AllowAchievements
            ? "The list hasn't been downloaded yet. Are you online? It downloads when the launcher starts, or now with the button below."
            : "Downloading the achievement list is off (Settings › Online). Turn it on, or download it once with the button below.");
    }

    void OnCard(object sender, MouseButtonEventArgs e)
    {
        if ((sender as FrameworkElement)?.DataContext is Item { A.Secret: true } i) i.Reveal();
    }

    async void OnDownload(object sender, RoutedEventArgs e)
    {
        DownloadButton.IsEnabled = false;
        var ok = await AchievementList.RefreshAsync();
        DownloadButton.IsEnabled = true;
        if (!ok) await Popup.Show("Achievements", "The list could not be downloaded. Check your connection and try again.", problem: true);
        Load();
    }

    void OnOpenList(object sender, RoutedEventArgs e) => AppState.Open(AchievementList.Page);
}
