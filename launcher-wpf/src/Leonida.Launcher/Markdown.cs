using System.Net;
using System.Text.RegularExpressions;
using System.Windows;
using System.Windows.Documents;
using System.Windows.Media;

namespace Leonida.Launcher;

public sealed record MarkdownSection(string Title, string Anchor, List<string> Lines);

/// <summary>
/// A small CommonMark subset for README.md and the docs it links: headings, paragraphs, lists, tables, fenced code,
/// quotes and inline bold, italics, code and links. HTML blocks (the README's centred header) are left out; inline
/// tags are reduced to their text. Enough for the project's own pages, not a general Markdown engine.
/// </summary>
public static partial class Markdown
{
    public const string RepoUrl = "https://github.com/ItsNotRexed/Leonida";

    /// <summary>The text before the first "## " heading, and one section per "## " heading.</summary>
    public static (List<string> preamble, List<MarkdownSection> sections) Split(string text)
    {
        var lines = text.Replace("\r\n", "\n").Split('\n').ToList();
        var pre = new List<string>();
        var sections = new List<MarkdownSection>();
        var inFence = false;
        foreach (var line in lines)
        {
            if (line.TrimStart().StartsWith("```")) inFence = !inFence;
            if (!inFence && line.StartsWith("## "))
            {
                var title = line[3..].Trim();
                sections.Add(new MarkdownSection(title, Anchor(title), []));
                continue;
            }
            if (sections.Count == 0) pre.Add(line); else sections[^1].Lines.Add(line);
        }
        return (pre, sections);
    }

    /// <summary>GitHub's heading anchor: lower case, spaces to '-', punctuation dropped.</summary>
    public static string Anchor(string heading)
    {
        var s = InlineText(heading).ToLowerInvariant();
        s = Regex.Replace(s, @"[^\p{L}\p{N}\s_-]", "");
        return Regex.Replace(s, @"\s", "-");
    }

    /// <summary>Plain text of inline Markdown (for titles and anchors).</summary>
    public static string InlineText(string s)
    {
        s = LinkRe().Replace(s, "$1");
        s = Regex.Replace(s, @"<[^>]+>", "");
        s = s.Replace("**", "").Replace("`", "");
        return WebUtility.HtmlDecode(s).Trim();
    }

    public static FlowDocument Render(IEnumerable<string> lines, Action<string> onLink, double fontSize = 14)
    {
        var doc = new FlowDocument
        {
            FontFamily = new FontFamily("Segoe UI"),
            FontSize = fontSize,
            PagePadding = new Thickness(0),
            Foreground = Theme.Brush("Text"),
            TextAlignment = TextAlignment.Left,
            LineHeight = fontSize * 1.5,
        };
        var list = lines.ToList();
        var i = 0;
        while (i < list.Count)
        {
            var line = list[i];
            var t = line.Trim();
            if (t.Length == 0) { i++; continue; }
            if (t.StartsWith("```"))
            {
                var code = new List<string>();
                i++;
                while (i < list.Count && !list[i].Trim().StartsWith("```")) code.Add(list[i++]);
                i++;
                doc.Blocks.Add(CodeBlock(string.Join("\n", code)));
                continue;
            }
            if (t.StartsWith('<'))   // an HTML block: skip to the blank line that ends it
            {
                while (i < list.Count && list[i].Trim().Length > 0) i++;
                continue;
            }
            var h = HeadingRe().Match(t);
            if (h.Success)
            {
                var level = h.Groups[1].Value.Length;
                var p = new Paragraph { Margin = new Thickness(0, 16, 0, 6), FontWeight = FontWeights.SemiBold,
                    FontSize = level <= 2 ? fontSize + 7 : level == 3 ? fontSize + 4 : fontSize + 1,
                    FontFamily = (FontFamily)Application.Current.FindResource("Display") };
                AddInlines(p.Inlines, h.Groups[2].Value, onLink);
                doc.Blocks.Add(p);
                i++;
                continue;
            }
            if (t.StartsWith('|'))
            {
                var rows = new List<string>();
                while (i < list.Count && list[i].Trim().StartsWith('|')) rows.Add(list[i++].Trim());
                doc.Blocks.Add(Table(rows, onLink, fontSize));
                continue;
            }
            if (t.StartsWith('>'))
            {
                var quote = new List<string>();
                while (i < list.Count && list[i].Trim().StartsWith('>')) quote.Add(list[i++].Trim().TrimStart('>').Trim());
                // GitHub alerts: "> [!IMPORTANT]" on the first line names the callout.
                var alert = quote.Count > 0 ? Regex.Match(quote[0], @"^\[!(NOTE|TIP|IMPORTANT|WARNING|CAUTION)\]\s*(.*)$") : Match.Empty;
                if (alert.Success) quote[0] = "**" + alert.Groups[1].Value[0] + alert.Groups[1].Value[1..].ToLowerInvariant() + ".** " + alert.Groups[2].Value;
                var p = new Paragraph { Margin = new Thickness(0, 4, 0, 10), Padding = new Thickness(12, 4, 0, 4),
                    BorderBrush = (Brush)Application.Current.FindResource("Pink"), BorderThickness = new Thickness(3, 0, 0, 0),
                    Foreground = Theme.Brush("SubText") };
                AddInlines(p.Inlines, string.Join(" ", quote), onLink);
                doc.Blocks.Add(p);
                continue;
            }
            if (ListItemRe().IsMatch(line))
            {
                doc.Blocks.Add(ListBlock(list, ref i, onLink));
                continue;
            }
            // A paragraph: lines up to a blank line or the start of another block.
            var text = new List<string>();
            while (i < list.Count)
            {
                var l = list[i];
                var lt = l.Trim();
                if (lt.Length == 0 || lt.StartsWith("```") || lt.StartsWith('|') || lt.StartsWith('>') ||
                    HeadingRe().IsMatch(lt) || (text.Count > 0 && ListItemRe().IsMatch(l))) break;
                text.Add(lt);
                i++;
            }
            var para = new Paragraph { Margin = new Thickness(0, 0, 0, 10) };
            AddInlines(para.Inlines, string.Join(" ", text), onLink);
            doc.Blocks.Add(para);
        }
        return doc;
    }

    static Block ListBlock(List<string> lines, ref int i, Action<string> onLink)
    {
        var first = ListItemRe().Match(lines[i]);
        var indent = first.Groups[1].Value.Length;
        var ordered = char.IsDigit(first.Groups[2].Value[0]);
        var list = new List { MarkerStyle = ordered ? TextMarkerStyle.Decimal : TextMarkerStyle.Disc, Margin = new Thickness(0, 0, 0, 10),
                              Padding = new Thickness(22, 0, 0, 0) };
        while (i < lines.Count)
        {
            var m = ListItemRe().Match(lines[i]);
            if (!m.Success)
            {
                // A continuation line of the item before (indented text), or the end of the list.
                if (lines[i].Trim().Length > 0 && lines[i].StartsWith(new string(' ', indent + 2)) && list.ListItems.Count > 0 &&
                    list.ListItems.LastListItem.Blocks.LastBlock is Paragraph lp)
                {
                    lp.Inlines.Add(new Run(" "));
                    AddInlines(lp.Inlines, lines[i].Trim(), onLink);
                    i++;
                    continue;
                }
                break;
            }
            var depth = m.Groups[1].Value.Length;
            if (depth > indent && list.ListItems.Count > 0)
            {
                list.ListItems.LastListItem.Blocks.Add(ListBlock(lines, ref i, onLink));
                continue;
            }
            if (depth < indent) break;
            var p = new Paragraph { Margin = new Thickness(0, 0, 0, 3) };
            AddInlines(p.Inlines, m.Groups[3].Value, onLink);
            list.ListItems.Add(new ListItem(p));
            i++;
        }
        return list;
    }

    static Block Table(List<string> rows, Action<string> onLink, double fontSize)
    {
        static List<string> Cells(string row)
        {
            var r = row.Trim().Trim('|');
            var cells = new List<string>();
            var cur = new System.Text.StringBuilder();
            var code = false;
            for (var k = 0; k < r.Length; k++)
            {
                if (r[k] == '`') code = !code;
                if (r[k] == '\\' && k + 1 < r.Length && r[k + 1] == '|') { cur.Append('|'); k++; continue; }
                if (r[k] == '|' && !code) { cells.Add(cur.ToString().Trim()); cur.Clear(); continue; }
                cur.Append(r[k]);
            }
            cells.Add(cur.ToString().Trim());
            return cells;
        }
        var header = Cells(rows[0]);
        var body = rows.Skip(1).Where(r => !Regex.IsMatch(r, @"^\|?\s*:?-{3,}")).Select(Cells).ToList();
        var table = new Table { CellSpacing = 0, Margin = new Thickness(0, 4, 0, 14), BorderBrush = Theme.Brush("Line"),
                                BorderThickness = new Thickness(1), FontSize = fontSize - 1 };
        for (var c = 0; c < header.Count; c++)
            table.Columns.Add(new TableColumn { Width = c == 0 && header.Count > 1 ? new GridLength(1, GridUnitType.Star) : new GridLength(header.Count > 2 ? 1.6 : 3, GridUnitType.Star) });
        var group = new TableRowGroup();
        void AddRow(List<string> cells, bool head, bool shade)
        {
            var row = new TableRow { Background = head ? Brushes.Black : shade ? Theme.Brush("CardAlt") : Theme.Brush("CardBg") };
            for (var c = 0; c < header.Count; c++)
            {
                var p = new Paragraph { Margin = new Thickness(0) };
                AddInlines(p.Inlines, c < cells.Count ? cells[c] : "", onLink);
                if (head) { p.Foreground = Brushes.White; p.FontWeight = FontWeights.SemiBold; }
                row.Cells.Add(new TableCell(p) { Padding = new Thickness(10, 7, 10, 7), BorderBrush = Theme.Brush("Line"),
                                                 BorderThickness = new Thickness(0, 0, c < header.Count - 1 ? 1 : 0, 1) });
            }
            group.Rows.Add(row);
        }
        AddRow(header, true, false);
        for (var r = 0; r < body.Count; r++) AddRow(body[r], false, r % 2 == 1);
        table.RowGroups.Add(group);
        return table;
    }

    static Block CodeBlock(string code) => new Paragraph(new Run(code))
    {
        FontFamily = (FontFamily)Application.Current.FindResource("Mono"),
        FontSize = 12.5,
        Background = new SolidColorBrush(Color.FromRgb(0x14, 0x14, 0x14)),
        Foreground = new SolidColorBrush(Color.FromRgb(0xE6, 0xE6, 0xE6)),
        Padding = new Thickness(14, 10, 14, 10),
        Margin = new Thickness(0, 2, 0, 12),
        LineHeight = 18,
    };

    /// <summary>Inline Markdown into runs: **bold**, *italic*, `code`, [links](...), &lt;br&gt;, HTML entities.</summary>
    public static void AddInlines(InlineCollection target, string text, Action<string> onLink)
    {
        text = Regex.Replace(text, @"<br\s*/?>", "\n", RegexOptions.IgnoreCase);
        text = Regex.Replace(text, @"!\[[^\]]*\]\([^)]*\)", "");                 // images
        text = Regex.Replace(text, @"<a\s+href=""([^""]+)""[^>]*>(.*?)</a>", "[$2]($1)", RegexOptions.IgnoreCase);
        text = Regex.Replace(text, @"</?(?:b|strong)>", "**", RegexOptions.IgnoreCase);
        text = Regex.Replace(text, @"<[^>]+>", "");
        var pos = 0;
        foreach (Match m in TokenRe().Matches(text))
        {
            if (m.Index > pos) target.Add(new Run(Decode(text[pos..m.Index])));
            if (m.Groups["code"].Success)
                target.Add(new Run(m.Groups["code"].Value) { FontFamily = (FontFamily)Application.Current.FindResource("Mono"),
                    FontSize = 12.5, Background = Theme.Brush("Hover") });
            else if (m.Groups["bold"].Success)
            {
                var b = new Bold();
                AddInlines(b.Inlines, m.Groups["bold"].Value, onLink);
                target.Add(b);
            }
            else if (m.Groups["italic"].Success)
            {
                var it = new Italic();
                AddInlines(it.Inlines, m.Groups["italic"].Value, onLink);
                target.Add(it);
            }
            else if (m.Groups["ltext"].Success)
            {
                var url = m.Groups["url"].Value;
                var link = new Hyperlink { Foreground = (Brush)Application.Current.FindResource("Purple"), Cursor = System.Windows.Input.Cursors.Hand,
                                           ToolTip = url };
                AddInlines(link.Inlines, m.Groups["ltext"].Value, onLink);
                link.Click += (_, _) => onLink(url);
                target.Add(link);
            }
            pos = m.Index + m.Length;
        }
        if (pos < text.Length) target.Add(new Run(Decode(text[pos..])));
    }

    static string Decode(string s) => WebUtility.HtmlDecode(s.Replace("\\_", "_").Replace("\\*", "*"));

    /// <summary>Where a README link goes: an anchor stays in the page, a repo path opens on GitHub.</summary>
    public static string Resolve(string url, string? basePath = null)
    {
        if (url.StartsWith("http://") || url.StartsWith("https://") || url.StartsWith("mailto:")) return url;
        if (url.StartsWith('#')) return url;
        var path = basePath == null ? url : System.IO.Path.Combine(System.IO.Path.GetDirectoryName(basePath) ?? "", url).Replace('\\', '/');
        return $"{RepoUrl}/blob/main/{path.TrimStart('/')}";
    }

    [GeneratedRegex(@"^(#{1,6})\s+(.*)$")] private static partial Regex HeadingRe();
    [GeneratedRegex(@"^(\s*)([-*+]|\d+[.)])\s+(.*)$")] private static partial Regex ListItemRe();
    [GeneratedRegex(@"\[([^\]]+)\]\(([^)]+)\)")] private static partial Regex LinkRe();
    [GeneratedRegex(@"`(?<code>[^`]+)`|\*\*(?<bold>.+?)\*\*|(?<![\w*])\*(?<italic>[^*\s][^*]*?)\*(?!\w)|\[(?<ltext>[^\]]+)\]\((?<url>[^)\s]+)\)")]
    private static partial Regex TokenRe();
}
