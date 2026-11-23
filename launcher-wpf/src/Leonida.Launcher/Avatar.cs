using System.Windows;
using System.Windows.Controls;
using System.Windows.Media;
using System.Windows.Shapes;

namespace Leonida.Launcher;

/// <summary>
/// The navbar avatar: a random picture made from a seed (kept in the settings, so it stays the same until
/// "New random avatar"): a mirrored 5×5 pattern, like a GitHub identicon, on a gradient from the logo's colours.
/// Nothing is downloaded and nothing identifies the user.
/// </summary>
public static class Avatar
{
    static readonly Color[] Colors =
    [
        Color.FromRgb(0x6B, 0x2F, 0xD6), Color.FromRgb(0xEC, 0x2F, 0x8C), Color.FromRgb(0xFF, 0x8A, 0x3D),
        Color.FromRgb(0x2A, 0x14, 0x52), Color.FromRgb(0xB4, 0x2A, 0x78), Color.FromRgb(0x14, 0x8F, 0xB8),
        Color.FromRgb(0x1E, 0x9E, 0x58), Color.FromRgb(0xE8, 0x4A, 0x3A),
    ];

    public static FrameworkElement Make(int seed, double size)
    {
        var rnd = new Random(seed);
        var a = Colors[rnd.Next(Colors.Length)];
        Color b;
        do b = Colors[rnd.Next(Colors.Length)]; while (b == a);
        var grid = new Grid { Width = size, Height = size, Clip = new EllipseGeometry(new Point(size / 2, size / 2), size / 2, size / 2) };
        grid.Children.Add(new Rectangle
        {
            Fill = new LinearGradientBrush(a, b, rnd.NextDouble() * 360),
        });
        // The pattern: 5×5 cells, the left three columns random, mirrored to the right; inset so the circle shows them.
        var cell = size * 0.62 / 5;
        var origin = (size - cell * 5) / 2;
        var canvas = new Canvas { Width = size, Height = size };
        var white = new SolidColorBrush(Color.FromArgb(0xE6, 0xFF, 0xFF, 0xFF));
        for (var y = 0; y < 5; y++)
        for (var x = 0; x < 3; x++)
        {
            if (rnd.Next(2) == 0) continue;
            foreach (var cx in x == 2 ? new[] { 2 } : new[] { x, 4 - x })
            {
                var r = new Rectangle { Width = cell, Height = cell, Fill = white, RadiusX = cell * 0.18, RadiusY = cell * 0.18 };
                Canvas.SetLeft(r, origin + cx * cell);
                Canvas.SetTop(r, origin + y * cell);
                canvas.Children.Add(r);
            }
        }
        grid.Children.Add(canvas);
        grid.Children.Add(new Ellipse { Stroke = new SolidColorBrush(Color.FromArgb(0x40, 0, 0, 0)), StrokeThickness = 1 });
        return grid;
    }
}
