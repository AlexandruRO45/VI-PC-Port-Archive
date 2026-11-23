using System.Diagnostics;
using System.IO;
using System.Text.RegularExpressions;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Documents;
using System.Windows.Media;
using Leonida.Core;

namespace Leonida.Launcher.Views;

/// <summary>
/// HOME: a landing page, and one page per README section (Project status, Platforms, How to build, How it works,
/// Want to help, Credits and license), reached from Home's own tabs. Everything comes from README.md of the
/// checkout, read when the launcher opens, so it is never older than the code it describes.
/// </summary>
public partial class HomeView : UserControl
{
    /// <summary>Home's pages: tab key, tab label, the README headings they show, an icon and a colour.</summary>
    public static readonly (string Key, string Tab, string[] Headings, string Glyph, string Brush)[] Pages =
    [
        ("status", "PROJECT STATUS", ["Project status"], "", "Pink"),
        ("platforms", "PLATFORMS", ["Platforms"], "", "Purple"),
        ("works", "HOW IT WORKS", ["How it works"], "", "Orange"),
        ("credits", "CREDITS", [], "", "Pink"),
    ];

    readonly AppState state = AppState.Instance;
    readonly Dictionary<string, MarkdownSection> sections = new(StringComparer.OrdinalIgnoreCase);   // by README anchor
    readonly Dictionary<string, string> anchorToPage = new(StringComparer.OrdinalIgnoreCase);
    List<Contributor> contributors = [];
    string current = "";

    /// <summary>The Home page now shown ("" for the landing page), for the tabs to follow.</summary>
    public event Action<string>? PageChanged;

    public HomeView() => InitializeComponent();

    public async void Initialize()
    {
        LoadReadme();
        await LoadFactsAsync();
    }

    // ---------------------------------------------------------------- pages

    /// <summary>"" or null: the landing page; otherwise one of <see cref="Pages"/>' keys.</summary>
    public void ShowPage(string? key)
    {
        key ??= "";
        current = key;
        Landing.Visibility = key == "" ? Visibility.Visible : Visibility.Collapsed;
        CreditsPage.Visibility = key == "credits" ? Visibility.Visible : Visibility.Collapsed;
        SectionPage.Visibility = key != "" && key != "credits" ? Visibility.Visible : Visibility.Collapsed;
        if (key != "" && key != "credits")
        {
            var page = Pages.First(p => p.Key == key);
            SectionTitle.Text = Loc.T(page.Tab);
            var lines = page.Headings.Select(h => sections.GetValueOrDefault(Markdown.Anchor(h))).FirstOrDefault(s => s != null)?.Lines;
            SectionBody.Document = Markdown.Render(lines ?? [Loc.T("This section is not in your checkout's README.md.")], OnLink);
            SectionPage.ScrollToTop();
        }
        if (key == "credits") { if (People.Items.Count > 0) People.ScrollIntoView(People.Items[0]); _ = LoadPeopleAsync(); }
        if (key == "") Landing.ScrollToTop();
        PageChanged?.Invoke(key);
    }

    public string CurrentPage => current;

    sealed record Tile(string Key, string Title, string Glyph, Brush Brush);

    void LoadReadme()
    {
        sections.Clear();
        anchorToPage.Clear();
        var path = state.Repo == null ? null : Path.Combine(state.Repo, "README.md");
        if (path == null || !File.Exists(path))
        {
            ReadmeError.Text = Loc.T("README.md was not found. Put Leonida.exe next to Build.bat in your Leonida folder, or choose the folder in Settings.");
            ReadmeError.Visibility = Visibility.Visible;
            Tagline.Text = "Grand Theft Auto VI (Xbox Series) running natively on Windows, built on your PC from your own copy.";
            return;
        }
        ReadmeError.Visibility = Visibility.Collapsed;
        var (pre, list) = Markdown.Split(File.ReadAllText(path));

        // The centred HTML header holds the tagline (<h4>) and the header links (<a href>).
        var raw = string.Join("\n", pre);
        var h4 = Regex.Match(raw, @"<h4[^>]*>(.*?)</h4>", RegexOptions.Singleline);
        Tagline.Text = h4.Success ? Markdown.InlineText(h4.Groups[1].Value.Replace("\n", " ")) : "";
        Links.Children.Clear();
        foreach (Match a in Regex.Matches(raw, @"<a href=""(https?://[^""]+)"">([^<]+)</a>"))
            Links.Children.Add(LinkButton(a.Groups[2].Value.Trim(), a.Groups[1].Value));
        Links.Children.Add(LinkButton("GitHub", Markdown.RepoUrl));
        Links.Children.Add(LinkButton("Issues", Markdown.RepoUrl + "/issues"));
        // The intro: the preamble without its plain "Discord: <url>" line (Discord is in the links).
        Intro.Document = Markdown.Render(pre.Where(l => !Regex.IsMatch(l.Trim(), @"^Discord:\s*https?://")), OnLink, 14.5);

        foreach (var s in list)
        {
            sections[s.Anchor] = s;
            var page = Pages.FirstOrDefault(p => p.Headings.Any(h => Markdown.Anchor(h) == s.Anchor)).Key;
            if (page != null)
            {
                anchorToPage[s.Anchor] = page;
                foreach (var l in s.Lines.Where(l => l.StartsWith("### "))) anchorToPage.TryAdd(Markdown.Anchor(l[4..]), page);
            }
        }
        SectionTiles.ItemsSource = Pages.Select(p => { var t = Loc.T(p.Tab); return new Tile(p.Key, char.ToUpper(t[0]) + t[1..].ToLowerInvariant(), p.Glyph, (Brush)FindResource(p.Brush)); }).ToList();
    }

    void OnTile(object sender, System.Windows.Input.MouseButtonEventArgs e)
    {
        if (sender is FrameworkElement { Tag: string key }) ShowPage(key);
    }

    Button LinkButton(string text, string url)
    {
        var b = new Button { Content = text, Style = (Style)FindResource("Link"), Margin = new Thickness(0, 0, 16, 6), ToolTip = url };
        b.Click += (_, _) => AppState.Open(url);
        return b;
    }

    /// <summary>A README link: an anchor opens the page that holds it, a repository path opens on GitHub.</summary>
    void OnLink(string url)
    {
        if (url.StartsWith('#'))
        {
            if (anchorToPage.TryGetValue(url[1..], out var page)) ShowPage(page);
            return;
        }
        AppState.Open(Markdown.Resolve(url));
    }

    // ---------------------------------------------------------------- facts and people

    async Task LoadFactsAsync()
    {
        var versions = state.Probe?.Versions;
        GameVersion.Text = versions is { Count: > 0 }
            ? string.Join("\n", versions.Select(v => $"{v.Title} {v.Version} ({v.Files} files)"))
            : state.ProbeError ?? Loc.T("Unknown");
        if (state.Repo == null) { Checkout.Text = Loc.T("No Leonida folder"); return; }
        Checkout.Text = await Git(state.Repo, "log", "-1", "--format=%h · %cd · %s", "--date=format:%Y-%m-%d %H:%M") is string head
            ? $"{await Git(state.Repo, "rev-parse", "--abbrev-ref", "HEAD") ?? "?"} @ {head}"
            : Loc.T("Not a git checkout (a source download)");
        await LoadPeopleAsync();
    }

    bool peopleLoaded;
    ContributorSource source;

    /// <summary>GitHub's contributor list (with profile pictures; cached a day), else the checkout's git history.</summary>
    async Task LoadPeopleAsync()
    {
        if (peopleLoaded) return;
        peopleLoaded = true;
        PeopleNote.Text = Loc.T("Loading…");
        var cache = Path.Combine(AppPaths.Local, "contributors.json");
        var online = state.Settings.Current.AllowGitHub;
        var fromGitHub = online ? await Contributors.LoadGitHubAsync(cache) : null;
        if (fromGitHub is { } g && g.list.Count > 0)
        {
            contributors = g.list;
            source = g.source;
            PeopleNote.Text = Loc.F("From GitHub, as of {0}: everyone with commits in {1}. AI assistants and bots (Claude, Claude Code, Devin, Dependabot) are tools, not contributors, and are not listed.",
                                    g.when.ToString("d MMM HH:mm"), Contributors.Repo);
        }
        else if (state.Repo != null)
        {
            try
            {
                contributors = await Contributors.LoadAsync(state.Repo);
                source = ContributorSource.GitHistory;
                PeopleNote.Text = Loc.T(online ? "GitHub can't be reached, so this is your checkout's git history (no profile pictures). AI assistants and bots are not listed."
                                               : "GitHub is off (Settings › Online), so this is your checkout's git history, without profile pictures.");
            }
            catch (Exception e) { PeopleNote.Text = "The contributor list could not be read: " + e.Message; peopleLoaded = false; }
        }
        ContributorCount.Text = contributors.Count > 0 ? Loc.F("{0} people, {1} commits", contributors.Count, contributors.Sum(c => c.Commits).ToString("N0")) : Loc.T("Unavailable");
        PeopleCount.Text = contributors.Count.ToString("N0");
        CommitCount.Text = contributors.Sum(c => c.Commits).ToString("N0");
        ShowPeople(PeopleSearch.Text.Trim());
    }

    static readonly Color[] Palette = [Color.FromRgb(0x6B, 0x2F, 0xD6), Color.FromRgb(0xEC, 0x2F, 0x8C), Color.FromRgb(0xFF, 0x8A, 0x3D),
                                       Color.FromRgb(0x2A, 0x14, 0x52), Color.FromRgb(0xB4, 0x2A, 0x78), Color.FromRgb(0x1E, 0x9E, 0x58)];

    /// <summary>One row: the picture loads from GitHub on its own when the row comes into view (rows are virtualized).</summary>
    sealed class Person
    {
        public required string Rank { get; init; }
        public required Brush RankBrush { get; init; }
        public required string Name { get; init; }
        public required string Handle { get; init; }
        public required string Commits { get; init; }
        public required double BarWidth { get; init; }
        public required string Initial { get; init; }
        public required Brush Color { get; init; }
        public string? AvatarUrl { get; init; }
        ImageSource? picture;
        public ImageSource? Picture
        {
            get
            {
                if (picture != null || AvatarUrl == null) return picture;
                try
                {
                    var b = new System.Windows.Media.Imaging.BitmapImage();
                    b.BeginInit();
                    b.UriSource = new Uri(AvatarUrl + (AvatarUrl.Contains('?') ? "&" : "?") + "s=96");
                    b.DecodePixelWidth = 84;
                    b.CacheOption = System.Windows.Media.Imaging.BitmapCacheOption.OnLoad;
                    b.EndInit();
                    picture = b;
                }
                catch (Exception) { }
                return picture;
            }
        }
    }

    void ShowPeople(string filter)
    {
        var max = Math.Max(1, contributors.FirstOrDefault()?.Commits ?? 1);
        var gold = (Brush)FindResource("Pink");
        var grey = (Brush)FindResource("Faint");
        People.ItemsSource = contributors.Select((c, i) => (c, rank: i + 1))
            .Where(x => filter.Length == 0 || x.c.Name.Contains(filter, StringComparison.OrdinalIgnoreCase))
            .Select(x => new Person
            {
                Rank = $"#{x.rank}",
                RankBrush = x.rank <= 3 ? gold : grey,
                Name = x.c.Name,
                Handle = x.c.Profile != null ? x.c.Profile.Replace("https://", "") : Loc.T("commit author"),
                Commits = x.c.Commits.ToString("N0"),
                BarWidth = Math.Max(3, 220.0 * x.c.Commits / max),
                Initial = x.c.Name.Length > 0 ? char.ToUpperInvariant(x.c.Name[0]).ToString() : "?",
                Color = new SolidColorBrush(Palette[Math.Abs(x.c.Name.GetHashCode(StringComparison.OrdinalIgnoreCase)) % Palette.Length]),
                AvatarUrl = x.c.Avatar,
            }).ToList();
    }

    void OnPeopleSearch(object sender, TextChangedEventArgs e) => ShowPeople(PeopleSearch.Text.Trim());

    static async Task<string?> Git(string repo, params string[] args)
    {
        try
        {
            var psi = new ProcessStartInfo("git") { UseShellExecute = false, CreateNoWindow = true, RedirectStandardOutput = true,
                RedirectStandardError = true, WorkingDirectory = repo, StandardOutputEncoding = System.Text.Encoding.UTF8 };
            foreach (var a in args) psi.ArgumentList.Add(a);
            using var p = Process.Start(psi)!;
            var o = await p.StandardOutput.ReadToEndAsync();
            await p.WaitForExitAsync();
            return p.ExitCode == 0 ? o.Trim() : null;
        }
        catch (Exception) { return null; }
    }
}
