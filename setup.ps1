# One-time setup on Windows (PowerShell).  Usage: powershell -ExecutionPolicy Bypass -File setup.ps1
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
$py = "py"
try { & py -3.12 --version | Out-Null; $pyArgs = @("-3.12") } catch {
  try { & py -3.11 --version | Out-Null; $pyArgs = @("-3.11") } catch { $py = "python"; $pyArgs = @() } }
& $py @pyArgs -m venv .venv
& .\.venv\Scripts\python.exe -m pip install --upgrade pip
& .\.venv\Scripts\python.exe -m pip install -r requirements.txt
if (Get-Command ollama -ErrorAction SilentlyContinue) { ollama pull qwen3.5:4b } else { Write-Host "Optional: install Ollama (https://ollama.com) for the AI assistant" }
Write-Host "Done. Start with: .\run.bat"
