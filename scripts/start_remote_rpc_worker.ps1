#!/usr/bin/env pwsh
<#
.SYNOPSIS
    Start a Windows RPC worker over SSH from the coordinator host.
.DESCRIPTION
    This is optional automation for the Windows-to-Windows setup. It connects
    to the worker via SSH, checks the GPU, stops old rpc-server processes, and
    launches rpc-server in the background on the worker.
.PARAMETER HostIp
    Worker IP address. Current laptop example: 192.168.1.146.
.PARAMETER User
    SSH user on the worker.
.PARAMETER KeyPath
    SSH private key path.
.PARAMETER RemoteBinDir
    Directory on the worker containing rpc-server.exe.
.PARAMETER Port
    RPC TCP port. Default: 50052.
.PARAMETER AddFirewallRule
    Add the Windows firewall rule on the worker. The SSH user must be elevated or allowed to create rules.
.EXAMPLE
    .\scripts\start_remote_rpc_worker.ps1 -HostIp 192.168.1.146 -User sergi -KeyPath $HOME\.ssh\laptop
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$HostIp,
    [string]$User = "sergi",
    [string]$KeyPath = "$HOME\.ssh\laptop",
    [string]$RemoteBinDir = "C:\Users\sergi\llama-rpc\bin",
    [int]$Port = 50052,
    [switch]$AddFirewallRule
)

$ErrorActionPreference = "Stop"

if (-not (Get-Command "ssh.exe" -ErrorAction SilentlyContinue)) {
    throw "ssh.exe not found. Install OpenSSH Client or run the worker script directly on the laptop."
}
if (-not (Test-Path $KeyPath)) {
    throw "SSH key not found: $KeyPath"
}

$remote = "$User@$HostIp"
$sshBaseArgs = @(
    "-o", "BatchMode=yes",
    "-o", "ConnectTimeout=10",
    "-o", "ServerAliveInterval=15",
    "-o", "ServerAliveCountMax=2",
    "-o", "StrictHostKeyChecking=accept-new",
    "-i", $KeyPath
)

function Invoke-RemotePowerShell {
    param([string]$Script)

    $encoded = [Convert]::ToBase64String([Text.Encoding]::Unicode.GetBytes($Script))
    & ssh.exe @sshBaseArgs $remote "powershell -NoProfile -ExecutionPolicy Bypass -EncodedCommand $encoded"
    if ($LASTEXITCODE -ne 0) {
        throw "Remote PowerShell failed on $remote"
    }
}

$firewallBlock = ""
if ($AddFirewallRule) {
    $firewallBlock = @"
`$ruleName = 'llama.cpp RPC $Port'
if (-not (Get-NetFirewallRule -DisplayName `$ruleName -ErrorAction SilentlyContinue)) {
    New-NetFirewallRule -DisplayName `$ruleName -Direction Inbound -Action Allow -Protocol TCP -LocalPort $Port -Profile Private | Out-Null
}
"@
}

$remoteScript = @"
`$ErrorActionPreference = 'Stop'
`$bin = '$RemoteBinDir'
`$rpc = Join-Path `$bin 'rpc-server.exe'
if (-not (Test-Path `$rpc)) {
    throw "rpc-server.exe not found at `$rpc"
}
Write-Host 'GPU check:'
& nvidia-smi --query-gpu=name,memory.free,memory.total,pstate --format=csv,noheader,nounits
if (`$LASTEXITCODE -ne 0) {
    throw 'nvidia-smi failed; refusing to start an unverified RPC worker.'
}
$firewallBlock
Get-Process rpc-server -ErrorAction SilentlyContinue | Stop-Process -Force
`$logDir = Join-Path `$env:TEMP 'llama-rpc-worker'
New-Item -ItemType Directory -Force -Path `$logDir | Out-Null
`$stdout = Join-Path `$logDir 'rpc-server.out.log'
`$stderr = Join-Path `$logDir 'rpc-server.err.log'
`$args = @('-H', '0.0.0.0', '-p', '$Port', '-c')
`$proc = Start-Process -FilePath `$rpc -WorkingDirectory `$bin -ArgumentList `$args -RedirectStandardOutput `$stdout -RedirectStandardError `$stderr -WindowStyle Hidden -PassThru
Start-Sleep -Seconds 2
`$listener = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue | Select-Object -First 1
if (-not `$listener) {
    Get-Content `$stderr -ErrorAction SilentlyContinue
    throw "rpc-server did not open TCP port $Port"
}
Write-Host "rpc-server started. PID=`$(`$proc.Id), endpoint=0.0.0.0:$Port"
Write-Host "stdout=`$stdout"
Write-Host "stderr=`$stderr"
"@

Write-Host "Starting RPC worker on $remote..." -ForegroundColor Cyan
Invoke-RemotePowerShell -Script $remoteScript
Write-Host ""
Write-Host "Testing coordinator -> worker TCP reachability..." -ForegroundColor Cyan
& (Join-Path $PSScriptRoot "test_rpc_endpoint.ps1") -HostIp $HostIp -Port $Port
