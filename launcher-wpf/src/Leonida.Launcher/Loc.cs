using Leonida.Core;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Documents;
using System.Windows.Markup;

namespace Leonida.Launcher;

/// <summary>
/// The launcher's languages: English (the text as written) and one JSON file per translation. Static text in the XAML is translated when a
/// window loads (<see cref="Apply"/> walks it and swaps any string it has a translation for; bound values are left
/// alone); text made in code goes through <see cref="T"/> and <see cref="F"/>. Changing the language reloads the
/// window. README pages, make_game.py's output and other people's names stay as they are.
/// </summary>
public static class Loc
{
    /// <summary>English (the text as written) and every language file: the ones built in (Languages/*.json in the
    /// source) and any in %APPDATA%\Leonida\Launcher\Languages, which win over a built-in one with the same code, so a
    /// translation can be tried without building the launcher.</summary>
    public static IReadOnlyList<LanguageFile> Languages => languages ??= LoadLanguages();
    static IReadOnlyList<LanguageFile>? languages;
    static string language = "en";
    static IReadOnlyDictionary<string, string> strings = new Dictionary<string, string>();

    public static string UserLanguagesFolder => System.IO.Path.Combine(System.IO.Path.GetDirectoryName(Leonida.Core.AppPaths.Settings)!, "Languages");

    static List<LanguageFile> LoadLanguages()
    {
        var all = new Dictionary<string, LanguageFile> { ["en"] = LanguageFile.English };
        var asm = typeof(Loc).Assembly;
        foreach (var name in asm.GetManifestResourceNames().Where(n => n.StartsWith("Languages.") && n.EndsWith(".json")).Order())
        {
            using var s = asm.GetManifestResourceStream(name)!;
            using var r = new System.IO.StreamReader(s);
            try { var f = LanguageFile.Parse(r.ReadToEnd()); all[f.Code] = f; } catch (FormatException) { }
        }
        try
        {
            if (System.IO.Directory.Exists(UserLanguagesFolder))
                foreach (var path in System.IO.Directory.EnumerateFiles(UserLanguagesFolder, "*.json").Order())
                    try { var f = LanguageFile.Parse(System.IO.File.ReadAllText(path)); all[f.Code] = f; }
                    catch (Exception e) when (e is FormatException or System.IO.IOException) { }
        }
        catch (System.IO.IOException) { }
        return [all["en"], .. all.Values.Where(l => l.Code != "en").OrderBy(l => l.Name, StringComparer.CurrentCultureIgnoreCase)];
    }

    /// <summary>"en" or "pl". Numbers, dates and sizes follow it too (1.0 GB / 1,0 GB), not the Windows locale.</summary>
    public static string Language
    {
        get => language;
        set
        {
            var file = Languages.FirstOrDefault(l => l.Code == value) ?? LanguageFile.English;
            language = file.Code;
            strings = file.Strings;
            System.Globalization.CultureInfo culture;
            try { culture = System.Globalization.CultureInfo.GetCultureInfo(file.Culture); }
            catch (System.Globalization.CultureNotFoundException) { culture = System.Globalization.CultureInfo.GetCultureInfo("en-US"); }
            System.Globalization.CultureInfo.DefaultThreadCurrentCulture = culture;
            System.Globalization.CultureInfo.DefaultThreadCurrentUICulture = culture;
            System.Globalization.CultureInfo.CurrentCulture = culture;
            System.Globalization.CultureInfo.CurrentUICulture = culture;
        }
    }
    static bool Translated => strings.Count > 0;
    public static System.Globalization.CultureInfo Culture => System.Globalization.CultureInfo.CurrentUICulture;

    /// <summary>The text in the current language. A key may carry a context after '|' ("BUILD|button") for a word that
    /// translates two ways; untranslated, the part before the '|' is shown.</summary>
    public static string T(string en) => strings.TryGetValue(en, out var p) && p.Length > 0 ? p : Plain(en);
    static string Plain(string key) { var i = key.IndexOf('|'); return i < 0 ? key : key[..i]; }
    static bool Lookup(string en, out string t) { if (strings.TryGetValue(en, out t!) && t.Length > 0) return true; t = en; return false; }
    public static string F(string en, params object?[] args) => string.Format(T(en), args);

    /// <summary>Translates every static string under <paramref name="root"/> (logical tree, tooltips, context menus,
    /// list columns).</summary>
    public static void Apply(DependencyObject root)
    {
        if (!Translated) return;
        var seen = new HashSet<DependencyObject>();
        Walk(root, seen);
    }

    static void Walk(DependencyObject d, HashSet<DependencyObject> seen)
    {
        if (!seen.Add(d)) return;
        switch (d)
        {
            case TextBlock tb when !BindingOperations.IsDataBound(tb, TextBlock.TextProperty):
                // Text set as an attribute leaves Inlines empty, so the whole text is looked up first; text made of
                // several runs (some bold, some bound) is translated run by run.
                if (Lookup(tb.Text, out var t)) tb.Text = t;
                else foreach (var i in tb.Inlines.ToList()) Walk(i, seen);
                break;
            case Run r when !BindingOperations.IsDataBound(r, Run.TextProperty):
                if (Lookup(r.Text, out var rt)) r.Text = rt;
                break;
            case HeaderedItemsControl h when h.Header is string hs && !BindingOperations.IsDataBound(h, HeaderedItemsControl.HeaderProperty):
                if (Lookup(hs, out var ht)) h.Header = ht;
                break;
            case HeaderedContentControl hc when hc.Header is string hcs:
                if (Lookup(hcs, out var hct)) hc.Header = hct;
                break;
            case ContentControl cc when cc.Content is string cs && !BindingOperations.IsDataBound(cc, ContentControl.ContentProperty):
                if (Lookup(cs, out var ct)) cc.Content = ct;
                break;
            case Window w when Lookup(w.Title, out var wt):
                w.Title = wt;
                break;
        }
        if (d is FrameworkElement fe)
        {
            if (fe.ToolTip is string tip && Lookup(tip, out var tt)) fe.ToolTip = tt;
            if (fe.ContextMenu != null) Walk(fe.ContextMenu, seen);
            if (fe is ListView { View: GridView gv })
                foreach (var col in gv.Columns)
                    if (col.Header is string ch && Lookup(ch, out var cht)) col.Header = cht;
        }
        foreach (var child in LogicalTreeHelper.GetChildren(d))
            if (child is DependencyObject c) Walk(c, seen);
    }

    /// <summary>{l:T text} for text inside templates, which the walk can't reach.</summary>
    [MarkupExtensionReturnType(typeof(string))]
    public sealed class TExtension(string text) : MarkupExtension
    {
        public override object ProvideValue(IServiceProvider serviceProvider) => T(text);
    }
}
