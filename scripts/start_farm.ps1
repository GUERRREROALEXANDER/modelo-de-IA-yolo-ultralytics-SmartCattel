# Runs SmartCattle on a farm's PC: installs what is missing, links the PC to an account the first time, then keeps it
# running and restarts it if it stops. The dashboard is on the internet; this PC only opens the farm's cameras.
# Usage: double click "Iniciar SmartCattle.bat"
param(
    [string]$Backend = '',
    [switch]$NoAutoStart
)
$ErrorActionPreference = 'Stop'
$root = Resolve-Path (Join-Path $PSScriptRoot '..')
Set-Location $root
$venv = Join-Path $root '.venv-farm'
$python = Join-Path $venv 'Scripts\python.exe'

function Say($text) { Write-Host "`n$text" -ForegroundColor Green }

function Find-Python {
    foreach ($name in 'python', 'py') {
        $found = Get-Command $name -ErrorAction SilentlyContinue
        # Windows ships a "python" that only opens the Store.
        if ($found -and $found.Source -notmatch 'WindowsApps') { return $found.Source }
    }
    return $null
}

function Find-Cloudflared {
    $candidates = @(
        (Get-Command cloudflared -ErrorAction SilentlyContinue).Source,
        (Join-Path $env:LOCALAPPDATA 'cloudflared\cloudflared.exe'),
        (Join-Path ${env:ProgramFiles(x86)} 'cloudflared\cloudflared.exe'),
        (Join-Path $env:ProgramFiles 'cloudflared\cloudflared.exe')
    )
    $candidates += Get-ChildItem (Join-Path $env:LOCALAPPDATA 'Microsoft\WinGet\Packages') -Filter 'cloudflared*.exe' -Recurse -ErrorAction SilentlyContinue | ForEach-Object FullName
    return $candidates | Where-Object { $_ -and (Test-Path $_) } | Select-Object -First 1
}

function Install-WithWinget($id) {
    if (-not (Get-Command winget -ErrorAction SilentlyContinue)) {
        throw "Falta winget en este Windows. Instala 'App Installer' desde la Microsoft Store y vuelve a abrir esto."
    }
    winget install --id $id --exact --silent --accept-package-agreements --accept-source-agreements | Out-Host
}

if (-not (Test-Path $python)) {
    $system = Find-Python
    if (-not $system) {
        Say 'Instalando Python...'
        Install-WithWinget 'Python.Python.3.13'
        $env:Path = [Environment]::GetEnvironmentVariable('Path', 'User') + ';' + [Environment]::GetEnvironmentVariable('Path', 'Machine')
        $system = Find-Python
        if (-not $system) { throw 'Python se instalo pero aun no aparece. Cierra esta ventana y vuelve a abrir el programa.' }
    }
    Say 'Preparando el programa. La primera vez tarda varios minutos y necesita internet.'
    & $system -m venv $venv
    & $python -m pip install --quiet --upgrade pip
    # CPU-only PyTorch first: the default download bundles CUDA and weighs several GB.
    & $python -m pip install --quiet torch==2.14.0 torchvision==0.29.0 --index-url https://download.pytorch.org/whl/cpu
    & $python -m pip install --quiet -c constraints.txt -r requirements.txt
    if ($LASTEXITCODE -ne 0) { Remove-Item -Recurse -Force $venv; throw 'No se pudo instalar. Revisa internet y vuelve a abrir el programa.' }
}

$cloudflared = Find-Cloudflared
if (-not $cloudflared) {
    Say 'Instalando la conexion segura hacia internet (cloudflared)...'
    # The official single file, into this user's folder: it needs no administrator, so nobody has to be there to approve.
    $folder = Join-Path $env:LOCALAPPDATA 'cloudflared'
    New-Item -ItemType Directory -Force $folder | Out-Null
    $cloudflared = Join-Path $folder 'cloudflared.exe'
    Invoke-WebRequest 'https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-windows-amd64.exe' -OutFile $cloudflared -UseBasicParsing
}
$env:SMARTCATTLE_CLOUDFLARED = $cloudflared

$linked = (Test-Path '.env') -and (Select-String -Path '.env' -Pattern '^SMARTCATTLE_API_KEY=farm-' -Quiet)
while (-not $linked) {
    Say 'Entra a SmartCattle con tu cuenta, ve a Camaras y pulsa "Mostrar codigo".'
    $code = Read-Host 'Escribe aqui el codigo'
    if ($Backend) { & $python pair_farm.py $code --backend $Backend } else { & $python pair_farm.py $code }
    $linked = $LASTEXITCODE -eq 0
}

if (-not $NoAutoStart) {
    # So that it comes back by itself after a power cut.
    $startup = [Environment]::GetFolderPath('Startup')
    $link = (New-Object -ComObject WScript.Shell).CreateShortcut((Join-Path $startup 'SmartCattle.lnk'))
    $link.TargetPath = Join-Path $root 'Iniciar SmartCattle.bat'
    $link.WorkingDirectory = "$root"
    $link.Save()
}

Say 'SmartCattle esta encendido. Deja esta ventana abierta y conecta las camaras desde la pagina.'
while ($true) {
    & $python -u live.py
    Write-Host 'El programa se detuvo. Se reinicia en 10 segundos...' -ForegroundColor Yellow
    Start-Sleep -Seconds 10
}
