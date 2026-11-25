# Draws icon.ico: a pink-to-orange gear with "VI" in its hub. Run: powershell -File make-icon.ps1
Add-Type -AssemblyName System.Drawing
$sizes = 16, 24, 32, 48, 64, 128, 256
$pngs = foreach ($n in $sizes) {
    $bmp = New-Object Drawing.Bitmap $n, $n
    $g = [Drawing.Graphics]::FromImage($bmp)
    $g.SmoothingMode = 'AntiAlias'; $g.TextRenderingHint = 'AntiAliasGridFit'; $g.PixelOffsetMode = 'HighQuality'
    $c = $n / 2; $outer = $n * 0.49; $body = $n * 0.39; $hub = $n * 0.29; $teeth = 8
    # Gear outline: body circle plus trapezoid teeth.
    $path = New-Object Drawing.Drawing2D.GraphicsPath
    $pts = New-Object 'System.Collections.Generic.List[System.Drawing.PointF]'
    for ($i = 0; $i -lt $teeth; $i++) {
        $a = 2 * [Math]::PI * $i / $teeth
        $half = [Math]::PI / $teeth
        foreach ($p in @(@(-0.55, $body), @(-0.36, $outer), @(0.36, $outer), @(0.55, $body), @(1.45, $body))) {
            $ang = $a + $p[0] * $half
            $pts.Add((New-Object Drawing.PointF ($c + $p[1] * [Math]::Cos($ang)), ($c + $p[1] * [Math]::Sin($ang))))
        }
    }
    $path.AddPolygon($pts.ToArray())
    $rect = New-Object Drawing.RectangleF 0, 0, $n, $n
    $brush = New-Object Drawing.Drawing2D.LinearGradientBrush $rect, ([Drawing.Color]::FromArgb(255, 255, 61, 127)), ([Drawing.Color]::FromArgb(255, 255, 160, 64)), 45
    $g.FillPath($brush, $path)
    $dark = New-Object Drawing.SolidBrush ([Drawing.Color]::FromArgb(255, 20, 14, 32))
    $g.FillEllipse($dark, $c - $hub, $c - $hub, 2 * $hub, 2 * $hub)
    # "VI" filling the hub.
    $font = New-Object Drawing.Font 'Bahnschrift', ($n * 0.30), ([Drawing.FontStyle]::Bold), ([Drawing.GraphicsUnit]::Pixel)
    $fmt = New-Object Drawing.StringFormat; $fmt.Alignment = 'Center'; $fmt.LineAlignment = 'Center'
    $text = New-Object Drawing.RectangleF 0, ($n * 0.02), $n, $n
    $g.DrawString('VI', $font, [Drawing.Brushes]::White, $text, $fmt)
    $g.Dispose()
    $ms = New-Object IO.MemoryStream; $bmp.Save($ms, [Drawing.Imaging.ImageFormat]::Png); $bmp.Dispose()
    , $ms.ToArray()
}
# ICO container with PNG entries.
$out = New-Object IO.MemoryStream; $w = New-Object IO.BinaryWriter $out
$w.Write([uint16]0); $w.Write([uint16]1); $w.Write([uint16]$sizes.Count)
$offset = 6 + 16 * $sizes.Count
for ($i = 0; $i -lt $sizes.Count; $i++) {
    $s = $sizes[$i] % 256
    $w.Write([byte]$s); $w.Write([byte]$s); $w.Write([byte]0); $w.Write([byte]0)
    $w.Write([uint16]1); $w.Write([uint16]32); $w.Write([uint32]$pngs[$i].Length); $w.Write([uint32]$offset)
    $offset += $pngs[$i].Length
}
foreach ($p in $pngs) { $w.Write($p) }
[IO.File]::WriteAllBytes((Join-Path $PSScriptRoot 'icon.ico'), $out.ToArray())
[IO.File]::WriteAllBytes((Join-Path $PSScriptRoot 'icon-256.png'), $pngs[-1])
