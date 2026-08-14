<#
.SYNOPSIS
    Build the portable Windows app: one folder holding sbs3d.exe, the Python
    runtime it needs and the depth model's weights.  No installer, no Python on
    the machine it runs on.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File packaging\windows\build.ps1

.EXAMPLE
    # A build for machines with an NVIDIA card: three times the size, and a
    # tenth of a second a photo rather than twenty.
    powershell -ExecutionPolicy Bypass -File packaging\windows\build.ps1 -Cuda

.NOTES
    Wants about 4 GB of free space to work in, or 12 GB with -Cuda.  Everything
    it leaves behind can go afterwards except <Work>\models, which saves the
    next build a 1.3 GB download.
#>
[CmdletBinding()]
param(
    # Python to build with; anything from 3.10 to 3.14 that Torch publishes
    # wheels for.  The build's Python is the app's Python, so this is the only
    # one that ever has to exist.
    [string]$Python = "",
    [string]$Work = "$env:USERPROFILE\sbs3d-build",
    [ValidateSet("small", "base", "large")]
    [string[]]$Models = @("large"),
    [switch]$Cuda,
    # Which CUDA build of Torch to take when -Cuda is given.  cu126 by default:
    # it runs on any driver from 525 up, where cu130 wants 580 or newer.
    [string]$TorchIndex = "",
    [switch]$SkipZip
)

$ErrorActionPreference = "Stop"
$root = (Get-Item $PSScriptRoot).Parent.Parent.FullName

function Invoke-Tool {
    param([string]$Exe, [string[]]$Arguments)
    Write-Host "`n> $Exe $($Arguments -join ' ')" -ForegroundColor DarkCyan
    & $Exe @Arguments
    if ($LASTEXITCODE -ne 0) { throw "$Exe failed with exit code $LASTEXITCODE" }
}

function Resolve-Python {
    if ($Python) { return $Python }
    foreach ($candidate in @("py", "python")) {
        $found = Get-Command $candidate -ErrorAction SilentlyContinue
        if ($found) {
            # `py` is a launcher rather than an interpreter; ask it which one it means.
            $exe = & $found.Source -c "import sys; print(sys.executable)"
            if ($LASTEXITCODE -eq 0 -and $exe) { return $exe.Trim() }
        }
    }
    throw "No Python found.  Install one from python.org and pass -Python <path to python.exe>."
}

$version = (Select-String -Path (Join-Path $root "pyproject.toml") -Pattern '^version\s*=\s*"(.+)"').Matches[0].Groups[1].Value
$flavour = if ($Cuda) { "cuda" } else { "cpu" }
$stage = Join-Path $Work "src"
# One tree per flavour: `pip install torch` leaves an already-satisfied Torch
# alone, so a CPU environment reused for a CUDA build would quietly ship the
# wrong one.
$venv = Join-Path $Work "venv-$flavour"
$vpy = Join-Path $venv "Scripts\python.exe"
$dist = Join-Path $Work "dist-$flavour"
$app = Join-Path $dist "sbs3d"

Write-Host "sbs3d $version -- portable Windows build ($flavour)" -ForegroundColor Green
Write-Host "  source : $root"
Write-Host "  build  : $Work"

# --- a copy of the source to build from ------------------------------------
# PyInstaller reads and writes a great deal; keeping all of it on a local disk
# matters when the checkout itself lives somewhere slower, WSL included.
if (Test-Path $stage) { Remove-Item $stage -Recurse -Force }
New-Item -ItemType Directory -Path $stage -Force | Out-Null
foreach ($item in @("sbs3d", "packaging", "pyproject.toml", "requirements.txt", "README.md", "LICENSE")) {
    Copy-Item (Join-Path $root $item) -Destination $stage -Recurse -Force
}
Get-ChildItem $stage -Recurse -Force -Filter "__pycache__" | Remove-Item -Recurse -Force

# --- the environment the app is frozen out of ------------------------------
if (-not (Test-Path $vpy)) {
    Invoke-Tool (Resolve-Python) @("-m", "venv", $venv)
}
Invoke-Tool $vpy @("-m", "pip", "install", "--upgrade", "--quiet", "pip", "wheel")

# Torch comes from its own index: PyPI's Windows wheel carries the CUDA
# runtime, which is gigabytes of no use to a machine without an NVIDIA card.
if (-not $TorchIndex) { $TorchIndex = if ($Cuda) { "cu126" } else { "cpu" } }
Invoke-Tool $vpy @("-m", "pip", "install", "--index-url", "https://download.pytorch.org/whl/$TorchIndex", "torch")
Invoke-Tool $vpy @("-m", "pip", "install", "-r", (Join-Path $stage "requirements.txt"), "pyinstaller")

# --- weights ---------------------------------------------------------------
$weights = Join-Path $Work "models"
foreach ($model in $Models) {
    if (-not (Test-Path (Join-Path $weights "$model\config.json"))) {
        Invoke-Tool $vpy (@((Join-Path $stage "packaging\windows\fetch_models.py")) + $Models + @("-o", $weights))
        break
    }
}

# --- freeze ----------------------------------------------------------------
Invoke-Tool $vpy @(
    "-m", "PyInstaller", "--noconfirm", "--clean",
    "--distpath", $dist, "--workpath", (Join-Path $Work "work-$flavour"),
    (Join-Path $stage "packaging\windows\sbs3d.spec")
)

# --- everything that goes beside the exe rather than inside it -------------
foreach ($model in $Models) {
    $to = Join-Path $app "models\$model"
    New-Item -ItemType Directory -Path $to -Force | Out-Null
    Copy-Item (Join-Path $weights "$model\*") -Destination $to -Recurse -Force
}
Copy-Item (Join-Path $root "README.md") -Destination $app -Force
Copy-Item (Join-Path $root "LICENSE") -Destination $app -Force
Set-Content -Path (Join-Path $app "Read me first.txt") -Encoding UTF8 -Value @"
sbs3d $version -- side-by-side 3D photos

Double-click sbs3d.exe.  Nothing to install: the whole app is this folder, so
it can live on a USB stick or anywhere else you like, as long as it stays
together.  sbs3d-cli.exe is the same program for the command line -- run it
from a terminal with --help to see what it takes.

This is the $flavour build. $(if ($Cuda) { "It uses an NVIDIA card when there is one, and falls back to the processor." } else { "It runs on the processor, so a large photo takes a little while." })
"@

$size = "{0:N0} MB" -f ((Get-ChildItem $app -Recurse -File | Measure-Object Length -Sum).Sum / 1MB)
Write-Host "`nBuilt $app ($size)" -ForegroundColor Green

if (-not $SkipZip) {
    $zip = Join-Path $Work "sbs3d-$version-win64-$flavour.zip"
    if (Test-Path $zip) { Remove-Item $zip -Force }
    Write-Host "Zipping to $zip ..."
    Add-Type -AssemblyName System.IO.Compression.FileSystem
    [System.IO.Compression.ZipFile]::CreateFromDirectory($app, $zip, [System.IO.Compression.CompressionLevel]::Optimal, $true)
    $zipsize = "{0:N0} MB" -f ((Get-Item $zip).Length / 1MB)
    Write-Host "`nDistribute $zip ($zipsize)" -ForegroundColor Green
}
