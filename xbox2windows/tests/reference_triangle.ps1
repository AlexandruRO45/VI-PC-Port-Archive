param([string]$Sample, [string]$Compiler, [string]$Renderer, [string]$Adapter)
$ErrorActionPreference = 'Stop'
$hashes = @{
    'VertexShader.hlsl' = 'E27F0718574544EE093D65A5118D771CBAFB180465734E531A40CC093086AE7F'
    'PixelShader.hlsl' = 'ADBCA57C5709070DEEB2D5CFE87FF67054B684E4318DC55AC1B1D95113EC2940'
    'media/image1.png' = '7F154D7B06DBC92013E3E8C909BE2CB25E7620F48A4C9D921B68F05F2E872AE1'
}
# The hashes are of the files as stored in Xbox-GDK-Samples. Its .gitattributes sets *.hlsl eol=crlf,
# so a git checkout writes CRLF shaders. Hash those with CRLF folded to LF; the PNG is compared as is.
function Get-ReferenceHash([string]$Path) {
    $bytes = [IO.File]::ReadAllBytes($Path)
    if ($Path.EndsWith('.hlsl')) {
        $lf = [Collections.Generic.List[byte]]::new($bytes.Length)
        for ($i = 0; $i -lt $bytes.Length; ++$i) {
            if ($bytes[$i] -ne 13 -or $i + 1 -ge $bytes.Length -or $bytes[$i + 1] -ne 10) { $lf.Add($bytes[$i]) }
        }
        $bytes = $lf.ToArray()
    }
    $sha = [Security.Cryptography.SHA256]::Create()
    try { -join ($sha.ComputeHash($bytes) | ForEach-Object { $_.ToString('X2') }) } finally { $sha.Dispose() }
}
foreach ($name in $hashes.Keys) {
    if ((Get-ReferenceHash (Join-Path $Sample $name)) -ne $hashes[$name]) {
        throw "Unexpected reference version: $name"
    }
}
$prefix = Join-Path (Get-Location) "reference_$Adapter"
foreach ($stage in @('Vertex', 'Pixel')) {
    $profile = if ($stage -eq 'Vertex') {'vs_6_0'} else {'ps_6_0'}
    & $Compiler -T $profile -E main -Fo "$prefix.$stage.cso" (Join-Path $Sample "$($stage)Shader.hlsl")
    if ($LASTEXITCODE) { exit $LASTEXITCODE }
}
& $Renderer $Adapter "$prefix.bmp" "$prefix.Vertex.cso" "$prefix.Pixel.cso"
if ($LASTEXITCODE) { exit $LASTEXITCODE }

# Compare pixel centers in normalized coordinates, away from the resized image edges.
Add-Type -AssemblyName System.Drawing
$screen = [Drawing.Bitmap]::new((Join-Path $Sample 'media/image1.png'))
$actual = [Drawing.Bitmap]::new("$prefix.bmp")
$count = 0
$frameErrors = 0
$photoErrors = 0
$sumPhotoError = 0
$maxPhotoError = 0
try {
    for ($y = 0; $y -lt $actual.Height; ++$y) {
        for ($x = 0; $x -lt $actual.Width; ++$x) {
            $u = ($x + 0.5) / $actual.Width
            $v = ($y + 0.5) / $actual.Height
            $red = 1.5 - 2 * $v
            $green = (1 - $red) / 2 + 2 * ($u - 0.5)
            $blue = 1 - $red - $green
            if ([Math]::Abs($red) -lt .04 -or [Math]::Abs($green) -lt .04 -or [Math]::Abs($blue) -lt .04) { continue }
            $linear = @($red, $green, $blue)
            if ($red -lt 0 -or $green -lt 0 -or $blue -lt 0) { $linear = @(.052860655, .052860655, .052860655) }
            $gpu = $actual.GetPixel($x, $y)
            $ref = $screen.GetPixel([Math]::Floor($u * $screen.Width), [Math]::Floor($v * $screen.Height))
            $channels = @($gpu.R, $gpu.G, $gpu.B)
            $photo = @($ref.R, $ref.G, $ref.B)
            for ($i = 0; $i -lt 3; ++$i) {
                $c = $linear[$i]
                $srgb = if ($c -le .0031308) { 12.92 * $c } else { 1.055 * [Math]::Pow($c, 1/2.4) - .055 }
                if ([Math]::Abs($channels[$i] - [Math]::Round($srgb * 255)) -gt 2) { ++$frameErrors }
                $errorValue = [Math]::Abs($channels[$i] - $photo[$i])
                $sumPhotoError += $errorValue
                $maxPhotoError = [Math]::Max($maxPhotoError, $errorValue)
                if ($errorValue -gt 24) { ++$photoErrors }
                ++$count
            }
        }
    }
    "channels=$count source_errors_gt2=$frameErrors published_image_errors_gt24=$photoErrors mean_abs=$([Math]::Round($sumPhotoError/$count,2)) max_abs=$maxPhotoError"
    if ($count -eq 0 -or $frameErrors -or $photoErrors -gt 0.01 * $count) { throw 'Reference comparison failed' }
} finally {
    $actual.Dispose()
    $screen.Dispose()
}
