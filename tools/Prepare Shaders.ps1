param([string]$Package, [string]$Repo, [string]$Out, [string]$Work)
$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.Windows.Forms
Add-Type -AssemblyName System.Drawing
$form = New-Object Windows.Forms.Form
$form.BackColor = [Drawing.Color]::Black; $form.FormBorderStyle = 'None'; $form.StartPosition = 'CenterScreen'; $form.ClientSize = New-Object Drawing.Size(620,180); $form.TopMost = $true
$title = New-Object Windows.Forms.Label; $title.Text = 'LEONIDA'; $title.ForeColor = [Drawing.Color]::White; $title.Font = New-Object Drawing.Font('Segoe UI',18,[Drawing.FontStyle]::Bold); $title.AutoSize = $true; $title.Location = New-Object Drawing.Point(260,30)
$status = New-Object Windows.Forms.Label; $status.Text = 'Unpacking shaders...'; $status.ForeColor = [Drawing.Color]::Silver; $status.AutoSize = $true; $status.Location = New-Object Drawing.Point(245,76)
$bar = New-Object Windows.Forms.ProgressBar; $bar.Style = 'Marquee'; $bar.MarqueeAnimationSpeed = 25; $bar.Width = 360; $bar.Height = 8; $bar.Location = New-Object Drawing.Point(130,112)
$form.Controls.AddRange(@($title,$status,$bar))
$log = Join-Path $Work 'shader-stage.log'; New-Item -ItemType Directory -Force -Path $Work | Out-Null
$p = Start-Process python.exe -ArgumentList @('-u','make_game.py','--game',$Package,'--out',$Out,'--work',$Work,'--stages','shaders','--no-tui') -WorkingDirectory $Repo -RedirectStandardOutput $log -RedirectStandardError (Join-Path $Work 'shader-stage.err') -PassThru -WindowStyle Hidden
$timer = New-Object Windows.Forms.Timer; $timer.Interval = 500
$timer.Add_Tick({ if (Test-Path $log) { $tail = Get-Content $log -Tail 1 -ErrorAction SilentlyContinue; if ($tail) { $status.Text = $tail.Trim(); $status.Refresh() } }; if ($p.HasExited) { $timer.Stop(); $bar.Style='Continuous'; $bar.Value=100; $status.Text = if ($p.ExitCode -eq 0) {'Shaders ready'} else {'Shader preparation failed'}; $status.Refresh(); Start-Sleep -Milliseconds 900; $form.Close() } })
$form.Add_Shown({$timer.Start()}); [void]$form.ShowDialog(); exit $p.ExitCode
