# Builds the frontend (if needed), starts the FastAPI backend, and opens a
# Cloudflare quick tunnel to it. Prints the public URL once ready.
#
# Usage: powershell -ExecutionPolicy Bypass -File deploy\start_demo.ps1
#
# Note: the quick tunnel URL is random and changes every time this runs —
# that's the trade-off for not needing a Cloudflare account / owned domain.
# Stop with Ctrl+C; this leaves the uvicorn process running in the
# background, so also run `Get-Job | Stop-Job` or close the terminal.

$repoRoot = Split-Path -Parent $PSScriptRoot
$frontendDist = Join-Path $repoRoot "frontend\dist"
$cloudflaredPath = "$env:USERPROFILE\bin\cloudflared.exe"

if (-not (Test-Path $frontendDist)) {
    Write-Host "Building frontend..."
    Push-Location (Join-Path $repoRoot "frontend")
    npm run build
    Pop-Location
}

if (-not (Test-Path $cloudflaredPath)) {
    Write-Error "cloudflared not found at $cloudflaredPath. Download it from https://github.com/cloudflare/cloudflared/releases"
    exit 1
}

Write-Host "Starting backend on port 8010..."
Push-Location (Join-Path $repoRoot "backend")
$backendJob = Start-Process -PassThru -NoNewWindow uvicorn -ArgumentList "main:app", "--host", "0.0.0.0", "--port", "8010"
Pop-Location

Start-Sleep -Seconds 3

Write-Host "Starting Cloudflare quick tunnel..."
& $cloudflaredPath tunnel --url http://localhost:8010
