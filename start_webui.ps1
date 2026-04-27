$ErrorActionPreference = "Stop"

$RepoRoot = $PSScriptRoot

Write-Host "=== Starting llama-webui ===" -ForegroundColor Cyan
Write-Host "Model discovery: project models, ~/models, ~/llama-rpc/models, and Hugging Face cache" -ForegroundColor Gray
Write-Host ""

$Launcher = Join-Path $RepoRoot ".venv\Scripts\llama-webui.exe"
& $Launcher
