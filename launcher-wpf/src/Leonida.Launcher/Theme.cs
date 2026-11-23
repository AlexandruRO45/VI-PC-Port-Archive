using System.Windows;
using System.Windows.Media;
using Microsoft.Win32;

namespace Leonida.Launcher;

/// <summary>
/// Light, dark and the Leonida Theme. Every colour that changes with the theme is a brush in Application.Resources
/// under one of <see cref="Keys"/>, used as a DynamicResource; <see cref="Apply"/> swaps the set. The sunset accents
/// and the white BUILD/PLAY button are the same in all of them.
/// <para>A value is a colour (<c>#RRGGBB</c>) or a gradient: <c>v:#top:#bottom</c> (top to bottom) or
/// <c>h:#left:#right</c>.</para>
/// </summary>
public static class Theme
{
    public static readonly (string Code, string Name)[] Choices =
        [("system", "Same as Windows"), ("light", "Light"), ("dark", "Dark"), ("leonida", "Leonida Theme")];

    // Light, dark, and Leonida: Leonida after dark, a violet night over the sea with a sunset still in the sky.
    static readonly Dictionary<string, (string Light, string Dark, string Leonida)> Palette = new()
    {
        ["Paper"] = ("#F2F2F2", "#121212", "v:#1A0B2E:#2E0F38"),        // the page
        ["CardBg"] = ("#FFFFFF", "#1C1C1C", "#221034"),                 // cards, panels, menus
        ["CardAlt"] = ("#F6F6F6", "#232323", "#2B1641"),                // a card's second shade (tiles, read-only boxes)
        ["CardBorder"] = ("#1A1A1A", "#3A3A3A", "#5A2F7E"),             // a card's outline
        ["Line"] = ("#D4D4D4", "#333333", "#3C2356"),                   // hairlines and dividers
        ["LineStrong"] = ("#9C9C9C", "#555555", "#7A55A3"),             // outlines of buttons and boxes
        ["Text"] = ("#141414", "#E8E8E8", "#F4EAFF"),
        ["Heading"] = ("#000000", "#FFFFFF", "#FFFFFF"),
        ["SubText"] = ("#555555", "#B4B4B4", "#D2BEEA"),
        ["Muted"] = ("#666666", "#A0A0A0", "#B9A0D6"),
        ["Faint"] = ("#8E8E8E", "#7C7C7C", "#9479B3"),
        ["Hover"] = ("#F2F2F2", "#2A2A2A", "#341D4E"),
        ["InputBg"] = ("#FFFFFF", "#1A1A1A", "#1B0D2B"),
        ["InputBorder"] = ("#9C9C9C", "#4C4C4C", "#6A4492"),
        ["MenuBg"] = ("#FFFFFF", "#1E1E1E", "#22113A"),
        ["MenuBorder"] = ("#2E2E2E", "#3E3E3E", "#8A3FA8"),
        ["MenuHover"] = ("#F2F2F2", "#2E2E2E", "#3A1F58"),
        ["Art1"] = ("#E3E3E3", "#1B1B1B", "v:#6A1F5C:#2A1040"),         // the background art: sun
        ["Art2"] = ("#DCDCDC", "#181818", "#170A28"),                   // skyline
        ["Art3"] = ("#CFCFCF", "#1F1F1F", "#14081F"),                   // near palm
        ["Art4"] = ("#D6D6D6", "#1D1D1D", "#1B0C2B"),                   // far palm
        ["Ink"] = ("#0B0B0B", "#0B0B0B", "v:#140823:#1F0A2A"),          // the sidebar and the window buttons
        ["SideHover"] = ("#1C1C1C", "#1C1C1C", "#2A1342"),
        ["SideMuted"] = ("#9B9B9B", "#9B9B9B", "#A98BC9"),
        ["CaptionBg"] = ("#000000", "#000000", "h:#2A0F4A:#6E1A5E"),    // the black bars: card captions, popup headers
    };

    public static IEnumerable<string> Keys => Palette.Keys;

    /// <summary>"light", "dark" or "leonida" for a setting ("system" asks Windows' app theme).</summary>
    public static string Resolve(string setting)
    {
        if (setting is "light" or "dark" or "leonida") return setting;
        try
        {
            using var k = Registry.CurrentUser.OpenSubKey(@"Software\Microsoft\Windows\CurrentVersion\Themes\Personalize");
            return k?.GetValue("AppsUseLightTheme") is int v && v == 0 ? "dark" : "light";
        }
        catch (Exception) { return "light"; }
    }

    /// <summary>Dark backgrounds (dark and Leonida).</summary>
    public static bool IsDark { get; private set; }

    public static void Apply(string setting)
    {
        var mode = Resolve(setting);
        IsDark = mode != "light";
        var res = Application.Current.Resources;
        foreach (var (key, (light, dark, leonida)) in Palette)
            res[key] = Make(mode switch { "dark" => dark, "leonida" => leonida, _ => light });
    }

    static Brush Make(string value)
    {
        Color C(string hex) => (Color)ColorConverter.ConvertFromString(hex);
        Brush b;
        if (value.StartsWith("v:") || value.StartsWith("h:"))
        {
            var parts = value.Split(':');
            b = new LinearGradientBrush(C(parts[1]), C(parts[2]), value[0] == 'v' ? 90 : 0);
        }
        else b = new SolidColorBrush(C(value));
        b.Freeze();
        return b;
    }

    /// <summary>A theme brush for code that draws (README pages, list chips).</summary>
    public static Brush Brush(string key) => (Brush)Application.Current.Resources[key];
}
