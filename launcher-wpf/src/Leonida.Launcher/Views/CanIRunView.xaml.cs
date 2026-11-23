using System.Runtime.InteropServices;
using System.Text.Json;
using System.Text.RegularExpressions;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Media;
using Leonida.Core;
using Microsoft.Web.WebView2.Core;
using Microsoft.Web.WebView2.Wpf;

namespace Leonida.Launcher.Views;

/// <summary>
/// "Can I run Leonida?", automatic: this PC's CPU, GPU and memory are read from Windows, then run through the
/// community checker at assttro.github.io/can-i-run-leonida for every reference on its list. The checker's code has
/// no license, so none of it is copied: the launcher loads the live page in a hidden WebView2 window, fills in its
/// form as a visitor would, picks each reference in turn and reads the verdict the page shows. The page runs on this
/// PC; the specs are typed into it, not sent anywhere.
/// </summary>
public partial class CanIRunView : UserControl
{
    public const string CheckerUrl = "https://assttro.github.io/can-i-run-leonida/";
    readonly AppState state = AppState.Instance;
    HardwareInfo? detected;
    WebView2? web;
    Window? host;
    bool started, pageReady;

    public CanIRunView()
    {
        InitializeComponent();
        Attribution.Text = "Verdicts and reference ranges: the community checker by assttro, run live. They compare benchmark " +
                           "bands, not a Leonida requirement; nothing is playable yet.";
    }

    /// <summary>The page was shown: detect and compare, once (Detect again or a restart runs it again).</summary>
    public async void Activate()
    {
        if (started) return;
        started = true;
        Headline.Text = Loc.T("Checking your PC…");
        detected = await Task.Run(HardwareDetector.Detect);
        ShowSpecs();
        if (!state.Settings.Current.AllowOnline) { OnlineOff(); return; }
        await LoadCheckerAsync();
    }

    void OnlineOff()
    {
        Headline.Text = Loc.T("Online features are off in Launcher settings, so only your hardware is shown.");
        HeadDot.Fill = (Brush)FindResource("Faint");
        CompareStatus.Text = "";
        Tiles.ItemsSource = null;
    }

    // ---------------------------------------------------------------- your PC

    /// <summary>What is compared: what Windows reports for this PC (DDR4 when the firmware does not say).</summary>
    (string cpu, string gpu, double ram, string ramType) Values()
    {
        var h = detected;
        return (h != null ? Clean(h.Cpu) : "",
                h?.PrimaryGpu is GpuInfo g ? GpuForChecker(g) : "",
                h?.RamGb ?? 16,
                h?.CheckerRamType ?? "ddr4");
    }

    void ShowSpecs()
    {
        var (cpu, gpu, ram, type) = Values();
        var h = detected;
        CpuText.Text = cpu;
        CpuDetail.Text = h == null ? "" : h.Cores > 0 ? Loc.F("{0} cores · {1} threads", h.Cores, h.Threads) : Loc.F("{0} threads", h.Threads);
        GpuText.Text = gpu;
        var g = h?.PrimaryGpu;
        GpuDetail.Text = g == null ? "" : $"{(g.MemoryBytes > 0 ? Format.Bytes(g.MemoryBytes) + Loc.T(" video memory") : "")}{(g.Driver != null ? Loc.F(" · driver {0}", g.Driver) : "")}";
        GpuDetail.ToolTip = h == null ? null : string.Join("\n", h.Gpus.Select(x => $"{x.Name} ({Format.Bytes(x.MemoryBytes)}{(x.Integrated ? ", integrated" : "")})"));
        RamText.Text = $"{ram:0.#} GB {(type == "ddr5" ? "DDR5" : "DDR4")}";
        RamDetail.Text = h == null ? "" : (h.RamModules > 0 ? Loc.F("{0} module(s)", h.RamModules) : "") + (h.RamSpeedMts is uint s ? $" · {s} MT/s" : "") +
                                          (h.RamType == null ? Loc.T(" · type not reported, DDR4 assumed") : "");
        Subline.Text = h?.Os ?? "";
    }

    /// <summary>The GPU as the checker reads it: the name, plus its memory for models sold in two sizes.</summary>
    static string GpuForChecker(GpuInfo g) =>
        Clean(g.Name) + (!g.Integrated && g.MemoryBytes >= 1UL << 30 ? $" {Math.Round(g.MemoryBytes / 1073741824.0)} GB" : "");

    /// <summary>Windows' marketing name without (R), (TM), "@ 3.6GHz" and "8-Core Processor".</summary>
    public static string Clean(string name)
    {
        var s = Regex.Replace(name, @"\((?:R|TM|tm|r)\)|®|™", " ");
        s = Regex.Replace(s, @"\s+@\s*[\d.]+\s*GHz", "", RegexOptions.IgnoreCase);
        s = Regex.Replace(s, @"\b(?:\d+|Six|Eight|Twelve|Sixteen)-Core Processor\b|\bProcessor\b|\bCPU\b|\bwith Radeon Graphics\b|\b\d+(?:st|nd|rd|th) Gen\b", " ", RegexOptions.IgnoreCase);
        return Regex.Replace(s, @"\s+", " ").Trim();
    }

    // ---------------------------------------------------------------- the checker, hidden

    async Task LoadCheckerAsync()
    {
        CompareStatus.Text = Loc.T("Asking the checker…");
        try
        {
            // A WebView2 needs a window: a small one off screen, never shown in the taskbar, closed with the launcher.
            web = new WebView2();
            host = new Window
            {
                Width = 1000, Height = 900, Left = -32000, Top = -32000, ShowInTaskbar = false, ShowActivated = false,
                WindowStyle = WindowStyle.None, Content = web, Owner = Window.GetWindow(this), Title = "Leonida checker",
            };
            host.Show();
            var env = await CoreWebView2Environment.CreateAsync(null, AppPaths.WebView);   // its data stays in %LOCALAPPDATA%
            await web.EnsureCoreWebView2Async(env);
            web.CoreWebView2.Settings.AreDevToolsEnabled = false;
            web.CoreWebView2.NewWindowRequested += (_, e) => e.Handled = true;
            web.NavigationCompleted += async (_, e) =>
            {
                pageReady = e.IsSuccess;
                if (e.IsSuccess) await CompareAsync();
                else Offline($"the page could not be loaded ({e.WebErrorStatus})");
            };
            web.Source = new Uri(CheckerUrl);
        }
        catch (Exception e) when (e is WebView2RuntimeNotFoundException or COMException or InvalidOperationException)
        {
            Offline("the Microsoft Edge WebView2 Runtime is missing (it comes with Windows 11; install it from " +
                    "https://go.microsoft.com/fwlink/p/?LinkId=2124703)");
        }
    }

    void Offline(string why)
    {
        Headline.Text = Loc.T("The checker can't be reached right now.");
        HeadDot.Fill = (Brush)FindResource("Faint");
        CompareStatus.Text = Loc.F("Your PC is detected above, but {0}. Are you online? Open the checker in your browser to compare by hand.", why);
        Tiles.ItemsSource = null;
    }

    sealed record PartView(string Part, string Status, string Reference, Brush Brush);
    sealed record TileView(string Label, string Verdict, Brush VerdictBrush, List<PartView> Parts, bool Console, bool AtOrAbove);

    async Task CompareAsync()
    {
        if (!pageReady || web?.CoreWebView2 == null) return;
        CompareStatus.Text = Loc.T("Comparing…");
        var (cpu, gpu, ram, type) = Values();
        var script = Script.Replace("__VALUES__", JsonSerializer.Serialize(new { cpu, gpu, ram, ramType = type }));
        string raw;
        try { raw = await web.CoreWebView2.ExecuteScriptAsync(script); }
        catch (Exception e) { Offline("it did not answer: " + e.Message); return; }
        using var doc = JsonDocument.Parse(JsonSerializer.Deserialize<string>(raw) ?? "{}");
        if (doc.RootElement.TryGetProperty("error", out var err))
        {
            Offline("its page has changed and could not be read (" + err.GetString() + ")");
            return;
        }
        var tiles = new List<TileView>();
        foreach (var item in doc.RootElement.GetProperty("results").EnumerateArray())
        {
            var statuses = item.GetProperty("parts").EnumerateArray().Select(p => p.GetProperty("status").GetString() ?? "").ToList();
            var parts = item.GetProperty("parts").EnumerateArray().Select(p => new PartView(
                p.GetProperty("part").GetString()!.ToUpperInvariant(), Loc.T(p.GetProperty("label").GetString() ?? ""),
                p.GetProperty("reference").GetString() ?? "", Status(p.GetProperty("status").GetString()))).ToList();
            var label = Loc.T(item.GetProperty("label").GetString() ?? "");
            var verdict = Verdict(item.GetProperty("verdict").GetString() ?? "");
            var reason = item.GetProperty("reason").GetString() ?? "";
            var bad = statuses.Contains("below") || statuses.Contains("outside");
            var unsure = statuses.Contains("unknown") || statuses.Contains("uncertain") || statuses.Contains("empty");
            tiles.Add(new TileView(label, verdict + (reason.Length > 0 ? " — " + reason : ""),
                (Brush)FindResource(bad ? "Bad" : unsure ? "Warn" : "Ok"), parts,
                !(item.GetProperty("label").GetString() ?? "").Contains("estimate", StringComparison.OrdinalIgnoreCase), !bad && !unsure));
        }
        Tiles.ItemsSource = tiles;
        var consoles = tiles.Where(t => t.Console).ToList();
        var ok = consoles.Where(t => t.AtOrAbove).Select(t => t.Label).ToList();
        if (consoles.Count > 0 && ok.Count == consoles.Count)
        {
            Headline.Text = Loc.T("Your PC is at or above every console on the list.");
            HeadDot.Fill = (Brush)FindResource("Ok");
        }
        else if (ok.Count > 0)
        {
            Headline.Text = Loc.F("Your PC matches the {0}.", string.Join(", ", ok));
            HeadDot.Fill = (Brush)FindResource("Ok");
        }
        else
        {
            Headline.Text = tiles.FirstOrDefault(t => !t.Console)?.Verdict ?? Loc.T("Your PC is below the consoles on the list.");
            HeadDot.Fill = (Brush)FindResource("Warn");
        }
        CompareStatus.Text = Loc.F("{0} references, as the checker lists them. Hover a line for the reference hardware.", tiles.Count);
    }

    /// <summary>The checker's verdict ("Above the Series X hardware range") in the launcher's language.</summary>
    static string Verdict(string v)
    {
        var m = Regex.Match(v, @"^(Above|Around|Below) the (.+) hardware range$");
        if (m.Success) return Loc.F(m.Groups[1].Value + " the {0} hardware range", m.Groups[2].Value);
        m = Regex.Match(v, @"^(Partial|Uncertain) (.+) comparison$");
        if (m.Success) return Loc.F(m.Groups[1].Value + " {0} comparison", m.Groups[2].Value);
        return Loc.T(v);
    }

    Brush Status(string? s) => (Brush)FindResource(s switch
    {
        "above" or "near" => "Ok",
        "below" or "outside" => "Bad",
        "uncertain" or "unknown" => "Warn",
        _ => "Faint",
    });

    // The page's form (#settings) runs its update() on every input event: set the fields as a visitor would, pick
    // each <option> of the reference list in turn, and read what the page then shows. Returns a JSON string.
    const string Script = """
        (() => {
          try {
            const v = __VALUES__;
            const form = document.getElementById('settings');
            if (!form) return JSON.stringify({ error: 'no #settings form' });
            const el = form.elements;
            const fire = (x) => x.dispatchEvent(new Event('input', { bubbles: true }));
            const pick = (select, input, value) => {
              const match = [...select.options].find(o => o.value.toLowerCase() === value.toLowerCase());
              select.value = match ? match.value : 'custom'; fire(select);
              if (!match) { input.value = value; fire(input); }
            };
            pick(el.cpu, el.cpuCustom, v.cpu);
            pick(el.gpu, el.gpuCustom, v.gpu);
            el.ramType.value = v.ramType; fire(el.ramType);
            el.ram.value = String(v.ram); fire(el.ram);
            const text = (id) => { const n = document.getElementById(id); return n && !n.hidden ? n.textContent.trim() : ''; };
            const refs = [...el.reference.options].map(o => ({ value: o.value, label: o.textContent.trim() }));
            const first = el.reference.value;
            const results = refs.map(r => {
              el.reference.value = r.value; fire(el.reference);
              const parts = ['cpu', 'gpu', 'ram'].map(p => {
                const n = document.getElementById(p + '-result');
                return { part: p, status: n ? (n.dataset.status || '') : '', label: n ? n.textContent.trim() : '', reference: text(p + '-reference') };
              });
              return { value: r.value, label: r.label, verdict: text('verdict'), reason: text('verdict-reason'), parts };
            });
            el.reference.value = first; fire(el.reference);
            return JSON.stringify({ results });
          } catch (e) { return JSON.stringify({ error: String(e) }); }
        })()
        """;

    // ---------------------------------------------------------------- detect again

    async void OnDetect(object sender, RoutedEventArgs e)
    {
        DetectButton.IsEnabled = false;
        Headline.Text = Loc.T("Checking your PC…");
        HeadDot.Fill = (Brush)FindResource("Faint");
        try
        {
            detected = await Task.Run(HardwareDetector.Detect);
            ShowSpecs();
            if (!state.Settings.Current.AllowOnline) { OnlineOff(); return; }
            if (pageReady) await CompareAsync();
            else if (web == null) await LoadCheckerAsync();
            else web.CoreWebView2?.Reload();   // the page failed to load before: try it again
        }
        finally { DetectButton.IsEnabled = true; }
    }

}
