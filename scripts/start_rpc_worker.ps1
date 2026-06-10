#!/usr/bin/env pwsh
<#
.SYNOPSIS
    Start llama.cpp rpc-server on the worker host.
.DESCRIPTION
    Run this on the remote GPU/CPU worker machine, not on the WebUI coordinator.
    The script finds rpc-server.exe, optionally opens the Windows firewall port,
    prints the worker IP addresses, and starts rpc-server on 0.0.0.0.
.PARAMETER Port
    TCP port for rpc-server. Default: 50052.
.PARAMETER LlamaBinDir
    Directory containing rpc-server.exe. If omitted, common locations are searched.
.PARAMETER AddFirewallRule
    Add an inbound Windows firewall rule for the selected port. Requires elevated PowerShell.
.PARAMETER Background
    Start rpc-server in the background and write logs under data\rpc-worker-logs.
.EXAMPLE
    .\scripts\start_rpc_worker.ps1
.EXAMPLE
    .\scripts\start_rpc_worker.ps1 -Port 50052 -LlamaBinDir C:\Users\sergi\llama-rpc\bin -AddFirewallRule
#>
[CmdletBinding()]
param(
    [int]$Port = 50052,
    [string]$LlamaBinDir = "",
    [switch]$AddFirewallRule,
    [switch]$Background,
    [string]$LogDir = ""
)

$ErrorActionPreference = "Stop"

function Resolve-RpcServer {
    param([string]$ExplicitBinDir)

    $candidates = @()
    if ($ExplicitBinDir) {
        $candidates += (Join-Path $ExplicitBinDir "rpc-server.exe")
    }
    if ($env:LLAMA_CPP_BIN_DIR) {
        $candidates += (Join-Path $env:LLAMA_CPP_BIN_DIR "rpc-server.exe")
    }

    $scriptDir = Split-Path -Parent $MyInvocation.ScriptName
    $repoRoot = Split-Path -Parent $scriptDir
    $candidates += (Join-Path $repoRoot "third_party\llama.cpp\prebuilt\rpc-server.exe")
    $candidates += (Join-Path $repoRoot "third_party\llama.cpp\build-cuda\bin\rpc-server.exe")
    $candidates += (Join-Path $repoRoot "third_party\llama.cpp\build\bin\rpc-server.exe")
    $candidates += "C:\Users\sergi\llama-rpc\bin\rpc-server.exe"
    $candidates += "C:\Users\Sergiio\llama-rpc\bin\rpc-server.exe"

    foreach ($candidate in $candidates) {
        if ($candidate -and (Test-Path $candidate)) {
            return (Resolve-Path $candidate).Path
        }
    }

    $fromPath = Get-Command "rpc-server.exe" -ErrorAction SilentlyContinue
    if ($fromPath) {
        return $fromPath.Source
    }

    throw "rpc-server.exe not found. Pass -LlamaBinDir C:\path\to\llama.cpp\bin."
}

function Test-IsAdmin {
    $identity = [Security.Principal.WindowsIdentity]::GetCurrent()
    $principal = [Security.Principal.WindowsPrincipal]::new($identity)
    return $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
}

function Add-RpcFirewallRule {
    param([int]$RulePort)

    if (-not (Test-IsAdmin)) {
        throw "AddFirewallRule requires an elevated PowerShell session."
    }

    $displayName = "llama.cpp RPC $RulePort"
    $existing = Get-NetFirewallRule -DisplayName $displayName -ErrorAction SilentlyContinue
    if (-not $existing) {
        New-NetFirewallRule `
            -DisplayName $displayName `
            -Direction Inbound `
            -Action Allow `
            -Protocol TCP `
            -LocalPort $RulePort `
            -Profile Private | Out-Null
    }
}

$rpcServer = Resolve-RpcServer -ExplicitBinDir $LlamaBinDir
$rpcDir = Split-Path -Parent $rpcServer
$argsList = @("-H", "0.0.0.0", "-p", "$Port", "-c")

Write-Host "llama.cpp RPC worker" -ForegroundColor Cyan
Write-Host "rpc-server: $rpcServer"
Write-Host "port      : $Port"

$gpu = Get-Command "nvidia-smi.exe" -ErrorAction SilentlyContinue
if ($gpu) {
    Write-Host ""
    & $gpu.Source --query-gpu=name,memory.free,memory.total,pstate --format=csv,noheader,nounits
} else {
    Write-Warning "nvidia-smi.exe not found. Continue only if this worker is intentionally CPU-only or uses another backend."
}

if ($AddFirewallRule) {
    Add-RpcFirewallRule -RulePort $Port
    Write-Host "Firewall rule is present for TCP $Port." -ForegroundColor Green
}

Write-Host ""
Write-Host "Worker IP addresses to use in WebUI:" -ForegroundColor Yellow
Get-NetIPAddress -AddressFamily IPv4 |
    Where-Object { $_.IPAddress -notlike "127.*" -and $_.PrefixOrigin -ne "WellKnown" } |
    Select-Object InterfaceAlias, IPAddress |
    Format-Table -AutoSize

Write-Host "Coordinator test command:" -ForegroundColor Yellow
Write-Host "  .\scripts\test_rpc_endpoint.ps1 -HostIp <worker-ip> -Port $Port"
Write-Host ""

if ($Background) {
    if (-not $LogDir) {
        $scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
        $repoRoot = Split-Path -Parent $scriptDir
        $LogDir = Join-Path $repoRoot "data\rpc-worker-logs"
    }
    New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
    $stdout = Join-Path $LogDir "rpc-worker.out.log"
    $stderr = Join-Path $LogDir "rpc-worker.err.log"
    $proc = Start-Process `
        -FilePath $rpcServer `
        -WorkingDirectory $rpcDir `
        -ArgumentList $argsList `
        -RedirectStandardOutput $stdout `
        -RedirectStandardError $stderr `
        -PassThru `
        -WindowStyle Hidden
    Write-Host "Started rpc-server in background. PID: $($proc.Id)" -ForegroundColor Green
    Write-Host "stdout: $stdout"
    Write-Host "stderr: $stderr"
    Start-Sleep -Seconds 2
    Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue |
        Select-Object LocalAddress, LocalPort, State, OwningProcess |
        Format-Table -AutoSize
    exit 0
}

Write-Host "Starting rpc-server in the foreground. Keep this window open." -ForegroundColor Green
Set-Location $rpcDir
& $rpcServer @argsList
