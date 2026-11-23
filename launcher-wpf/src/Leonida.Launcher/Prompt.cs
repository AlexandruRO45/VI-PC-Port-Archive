using System.Windows;
using System.Windows.Controls;

namespace Leonida.Launcher;

/// <summary>A one-line text prompt (a preset's name), built in code so it needs no XAML of its own.</summary>
public static class Prompt
{
    public static string? Ask(Window owner, string title, string label, string initial = "")
    {
        var box = new TextBox { Text = initial, Margin = new Thickness(0, 6, 0, 14), MinWidth = 320 };
        var ok = new Button { Content = Loc.T("Save"), IsDefault = true, Style = (Style)Application.Current.FindResource("Dark"), Margin = new Thickness(0, 0, 8, 0) };
        var cancel = new Button { Content = Loc.T("Cancel"), IsCancel = true, Style = (Style)Application.Current.FindResource("Outline"), Margin = new Thickness(0) };
        var buttons = new StackPanel { Orientation = Orientation.Horizontal, HorizontalAlignment = HorizontalAlignment.Right };
        buttons.Children.Add(ok);
        buttons.Children.Add(cancel);
        var panel = new StackPanel { Margin = new Thickness(18) };
        panel.Children.Add(new TextBlock { Text = Loc.T(label), FontWeight = FontWeights.SemiBold });
        panel.Children.Add(box);
        panel.Children.Add(buttons);
        var w = new Window
        {
            Title = Loc.T(title), Owner = owner, Content = panel, SizeToContent = SizeToContent.WidthAndHeight, ResizeMode = ResizeMode.NoResize,
            WindowStartupLocation = WindowStartupLocation.CenterOwner, ShowInTaskbar = false,
        };
        ok.Click += (_, _) => w.DialogResult = true;
        w.Loaded += (_, _) => { box.Focus(); box.SelectAll(); };
        return w.ShowDialog() == true ? box.Text : null;
    }
}
