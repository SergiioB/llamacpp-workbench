#!/usr/bin/env pwsh
<#
.SYNOPSIS
    Test connectivity from the WebUI coordinator to a llama.cpp RPC worker.
.DESCRIPTION
    Run this on the coordinator host before pressing Start in the WebUI.
    A passing TCP test means the WebUI RPC Check should also pass.
.PARAMETER HostIp
    RPC worker IP or DNS name. Current laptop example: 192.168.1.146.
.PARAMETER Port
    RPC worker TCP port. Default: 50052.
.EXAMPLE
    .\scripts\test_rpc_endpoint.ps1 -HostIp 192.168.1.146
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$HostIp,
    [int]$Port = 50052
)

$ErrorActionPreference = "Stop"

Write-Host "Testing llama.cpp RPC endpoint $HostIp`:$Port" -ForegroundColor Cyan

$ping = Test-Connection -ComputerName $HostIp -Count 1 -Quiet -ErrorAction SilentlyContinue
if ($ping) {
    Write-Host "Ping: reachable" -ForegroundColor Green
} else {
    Write-Warning "Ping failed. TCP can still work if ICMP is blocked, but check the host/IP first."
}

$tcp = Test-NetConnection -ComputerName $HostIp -Port $Port -InformationLevel Detailed
$sourceAddress = if ($tcp.SourceAddress -and $tcp.SourceAddress.IPAddress) {
    $tcp.SourceAddress.IPAddress
} else {
    [string]$tcp.SourceAddress
}
Write-Host ""
Write-Host "Source address: $sourceAddress"
Write-Host "Interface     : $($tcp.InterfaceAlias)"
Write-Host "TCP result    : $($tcp.TcpTestSucceeded)"

if ($tcp.TcpTestSucceeded) {
    Write-Host ""
    Write-Host "RPC endpoint is reachable." -ForegroundColor Green
    Write-Host "Use these WebUI settings:" -ForegroundColor Yellow
    Write-Host "  Runtime -> Mode : RPC Split"
    Write-Host "  RPC Host        : $HostIp"
    Write-Host "  Port            : $Port"
    Write-Host "  Tensor Split    : 34,66"
    exit 0
}

Write-Host ""
Write-Host "RPC endpoint is not reachable." -ForegroundColor Red
Write-Host "On the worker host, verify:"
Write-Host "  Get-NetTCPConnection -LocalPort $Port -State Listen"
Write-Host "  .\scripts\start_rpc_worker.ps1 -Port $Port -AddFirewallRule"
Write-Host ""
Write-Host "If ping works but TCP fails, the usual causes are:"
Write-Host "  - rpc-server is not running"
Write-Host "  - rpc-server is bound to 127.0.0.1 instead of 0.0.0.0"
Write-Host "  - Windows firewall is blocking inbound TCP $Port"
Write-Host "  - the laptop changed networks or is on a public network profile"
Write-Host ""
Write-Host "If a model is already loaded and chat works, stop llama-server before"
Write-Host "using this standalone TCP probe. An active coordinator connection can"
Write-Host "make extra rpc-server connection probes fail."
exit 1
