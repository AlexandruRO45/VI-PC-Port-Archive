using System.Windows;
using System.Windows.Media;
using Microsoft.Win32;

namespace Leonida.Launcher;

/// <summary>
/// Light and dark. Every colour that changes with the theme is a brush in Application.Resources under one of
/// <see cref="Keys"/>, used as a DynamicResource; <see cref="Apply"/> swaps the set. The black sidebar, captions and
/// console, the sunset accents and the white BUILD/PLAY button are the same in both.
/// </summary>
public static class Theme
{
    public static readonly (string Code, string Name)[] Choices = [("system", "Same as Windows"), ("light", "Light"), ("dark", "Dark")];

    static readonly Dictionary<string, (string Light, string Dark)> Palette = new()
    {
        ["Paper"] = ("#F2F2F2", "#121212"),        // the page
        ["CardBg"] = ("#FFFFFF", "#1C1C1C"),       // cards, panels, menus
        ["CardAlt"] = ("#F6F6F6", "#232323"),      // a card's second shade (tiles, read-only boxes)
        ["CardBorder"] = ("#1A1A1A", "#3A3A3A"),   // a card's outline
        ["Line"] = ("#D4D4D4", "#333333"),         // hairlines and dividers
        ["LineStrong"] = ("#9C9C9C", "#555555"),   // outlines of buttons and boxes
        ["Text"] = ("#141414", "#E8E8E8"),
        ["Heading"] = ("#000000", "#FFFFFF"),
        ["SubText"] = ("#555555", "#B4B4B4"),
        ["Muted"] = ("#666666", "#A0A0A0"),
        ["Faint"] = ("#8E8E8E", "#7C7C7C"),
        ["Hover"] = ("#F2F2F2", "#2A2A2A"),
        ["InputBg"] = ("#FFFFFF", "#1A1A1A"),
        ["InputBorder"] = ("#9C9C9C", "#4C4C4C"),
        ["MenuBg"] = ("#FFFFFF", "#1E1E1E"),
        ["MenuBorder"] = ("#2E2E2E", "#3E3E3E"),
        ["MenuHover"] = ("#F2F2F2", "#2E2E2E"),
        ["Art1"] = ("#E3E3E3", "#1B1B1B"),         // the background art: sun
        ["Art2"] = ("#DCDCDC", "#181818"),         // skyline
        ["Art3"] = ("#CFCFCF", "#1F1F1F"),         // near palm
        ["Art4"] = ("#D6D6D6", "#1D1D1D"),         // far palm
    };

    public static IEnumerable<string> Keys => Palette.Keys;

    /// <summary>"light" or "dark" for a setting ("system" asks Windows' app theme).</summary>
    public static string Resolve(string setting)
    {
        if (setting is "light" or "dark") return setting;
        try
        {
            using var k = Registry.CurrentUser.OpenSubKey(@"Software\Microsoft\Windows\CurrentVersion\Themes\Personalize");
            return k?.GetValue("AppsUseLightTheme") is int v && v == 0 ? "dark" : "light";
        }
        catch (Exception) { return "light"; }
    }

    public static bool IsDark { get; private set; }

    public static void Apply(string setting)
    {
        IsDark = Resolve(setting) == "dark";
        var res = Application.Current.Resources;
        foreach (var (key, (light, dark)) in Palette)
        {
            var b = new SolidColorBrush((Color)ColorConverter.ConvertFromString(IsDark ? dark : light));
            b.Freeze();
            res[key] = b;
        }
    }

    /// <summary>A theme brush for code that draws (README pages, list chips).</summary>
    public static Brush Brush(string key) => (Brush)Application.Current.Resources[key];
}
