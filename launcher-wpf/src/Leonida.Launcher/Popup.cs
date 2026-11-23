using System.Windows;
using System.Windows.Controls;
using System.Windows.Input;
using System.Windows.Media;

namespace Leonida.Launcher;

/// <summary>
/// The launcher's own message boxes: a card in the middle of the window over a dimmed backdrop, in the same style as
/// the build popup (black header, the sunset line, buttons at the bottom right). Esc is the cancel button, Enter the
/// main one. When the window is hidden (in the tray) or not open yet, Windows' own message box is used instead.
/// </summary>
public static class Popup
{
    /// <summary>A question; true when the main button was chosen.</summary>
    public static Task<bool> Ask(string title, string text, string yes, string no = "Cancel", bool danger = false, IEnumerable<string>? bullets = null) =>
        Open(title, text, bullets, yes, no, danger, problem: false, input: null).ContinueWith(t => t.Result.Ok, TaskScheduler.FromCurrentSynchronizationContext());

    /// <summary>A notice with one button.</summary>
    public static Task Show(string title, string text, IEnumerable<string>? bullets = null, bool problem = false) =>
        Open(title, text, bullets, "OK", null, false, problem, input: null);

    /// <summary>One line of text (a preset's name); null when cancelled or left empty.</summary>
    public static async Task<string?> Input(string title, string label, string initial, string ok)
    {
        var r = await Open(title, label, null, ok, "Cancel", false, false, input: initial);
        return r.Ok && !string.IsNullOrWhiteSpace(r.Text) ? r.Text.Trim() : null;
    }

    static Grid? Host => Application.Current?.MainWindow is MainWindow { IsVisible: true } w ? w.PopupHost : null;

    static Task<(bool Ok, string? Text)> Open(string title, string text, IEnumerable<string>? bullets, string yes, string? no, bool danger, bool problem, string? input)
    {
        var host = Host;
        if (host == null) return Task.FromResult(Fallback(title, text, bullets, no != null, problem));

        var done = new TaskCompletionSource<(bool, string?)>();
        var res = Application.Current.Resources;

        // Body
        var body = new StackPanel { Margin = new Thickness(26, 20, 26, 22) };
        body.Children.Add(new TextBlock { Text = Loc.T(text), TextWrapping = TextWrapping.Wrap, FontSize = 14, Foreground = Theme.Brush("Text"), LineHeight = 21 });
        foreach (var b in bullets ?? [])
        {
            var row = new DockPanel { Margin = new Thickness(0, 8, 0, 0) };
            var dot = new Border { Width = 6, Height = 6, CornerRadius = new CornerRadius(3), Background = (Brush)res["Sunset"], Margin = new Thickness(2, 7, 12, 0), VerticalAlignment = VerticalAlignment.Top };
            DockPanel.SetDock(dot, Dock.Left);
            row.Children.Add(dot);
            row.Children.Add(new TextBlock { Text = b, TextWrapping = TextWrapping.Wrap, FontSize = 13, Foreground = Theme.Brush("SubText") });
            body.Children.Add(row);
        }
        TextBox? box = null;
        if (input != null)
        {
            box = new TextBox { Text = input, Margin = new Thickness(0, 12, 0, 0), Height = 38, FontSize = 14, VerticalContentAlignment = VerticalAlignment.Center, Padding = new Thickness(8, 0, 8, 0) };
            body.Children.Add(box);
        }

        // Footer
        var main = new Button
        {
            Content = Loc.T(yes), Style = (Style)res[danger ? "Danger" : "Bevel"], MinWidth = 150, Height = danger ? double.NaN : 44,
            Padding = danger ? new Thickness(22, 10, 22, 10) : new Thickness(24, 0, 24, 0), FontSize = danger ? 13 : 18, VerticalAlignment = VerticalAlignment.Center,
        };
        var buttons = new StackPanel { Orientation = Orientation.Horizontal, HorizontalAlignment = HorizontalAlignment.Right };
        Button? cancel = null;
        if (no != null)
        {
            cancel = new Button { Content = Loc.T(no), Style = (Style)res["Outline"], Padding = new Thickness(18, 9, 18, 9), Margin = new Thickness(0, 0, 14, 0), VerticalAlignment = VerticalAlignment.Center };
            buttons.Children.Add(cancel);
        }
        buttons.Children.Add(main);
        var footer = new Border
        {
            Background = Theme.Brush("CardAlt"), BorderBrush = Theme.Brush("Line"), BorderThickness = new Thickness(0, 1, 0, 0),
            Padding = new Thickness(26, 14, 26, 14), Child = buttons,
        };

        // Header: black, the title in the display face, a mark for problems
        var head = new DockPanel();
        if (problem)
        {
            var mark = new TextBlock { Text = "", FontFamily = (FontFamily)res["Icons"], FontSize = 22, Foreground = (Brush)res["Pink"], Margin = new Thickness(0, 2, 14, 0), VerticalAlignment = VerticalAlignment.Center };
            DockPanel.SetDock(mark, Dock.Left);
            head.Children.Add(mark);
        }
        head.Children.Add(new TextBlock { Text = Loc.T(title).ToUpper(Loc.Culture), Foreground = Brushes.White, FontFamily = (FontFamily)res["DisplayBold"], FontSize = 26, TextTrimming = TextTrimming.CharacterEllipsis });
        var header = new Border { Background = Theme.Brush("CaptionBg"), Padding = new Thickness(26, 14, 26, 14), Child = head };

        var card = new DockPanel();
        DockPanel.SetDock(header, Dock.Top);
        var line = new System.Windows.Shapes.Rectangle { Height = 4, Fill = (Brush)res["Sunset"] };
        DockPanel.SetDock(line, Dock.Top);
        DockPanel.SetDock(footer, Dock.Bottom);
        card.Children.Add(header);
        card.Children.Add(line);
        card.Children.Add(footer);
        card.Children.Add(body);
        var panel = new Border
        {
            Width = 540, Background = Theme.Brush("CardBg"), BorderBrush = Brushes.Black, BorderThickness = new Thickness(1),
            HorizontalAlignment = HorizontalAlignment.Center, VerticalAlignment = VerticalAlignment.Center,
            Effect = (System.Windows.Media.Effects.Effect)res["SoftShadow"], Child = card,
        };
        var layer = new Grid { Background = new SolidColorBrush(Color.FromArgb(0xB3, 0, 0, 0)), Focusable = false };
        layer.Children.Add(panel);
        layer.MouseLeftButtonDown += (_, e) => e.Handled = true;   // a click outside does nothing: the question needs an answer

        var focusBefore = Keyboard.FocusedElement;
        var window = Window.GetWindow(host)!;
        void Finish(bool ok)
        {
            if (done.Task.IsCompleted) return;
            window.PreviewKeyDown -= OnKey;
            host.Children.Remove(layer);
            if (host.Children.Count == 0) host.Visibility = Visibility.Collapsed;
            (focusBefore as IInputElement)?.Focus();
            done.SetResult((ok, box?.Text));
        }
        void OnKey(object sender, KeyEventArgs e)
        {
            if (host.Children.Count == 0 || host.Children[^1] != layer) return;   // only the top popup answers keys
            if (e.Key == Key.Escape) { Finish(cancel == null); e.Handled = true; }
            else if (e.Key == Key.Enter) { Finish(true); e.Handled = true; }
        }
        main.Click += (_, _) => Finish(true);
        if (cancel != null) cancel.Click += (_, _) => Finish(false);
        window.PreviewKeyDown += OnKey;

        host.Children.Add(layer);
        host.Visibility = Visibility.Visible;
        layer.Loaded += (_, _) =>
        {
            if (box != null) { box.Focus(); box.SelectAll(); }
            else main.Focus();
        };
        return done.Task;
    }

    static (bool, string?) Fallback(string title, string text, IEnumerable<string>? bullets, bool question, bool problem)
    {
        var all = Loc.T(text) + (bullets?.Any() == true ? "\n\n• " + string.Join("\n• ", bullets) : "");
        var r = MessageBox.Show(all, Loc.T(title), question ? MessageBoxButton.YesNo : MessageBoxButton.OK,
                                problem ? MessageBoxImage.Warning : question ? MessageBoxImage.Question : MessageBoxImage.Information);
        return (r is MessageBoxResult.Yes or MessageBoxResult.OK, null);
    }
}
