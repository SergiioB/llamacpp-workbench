# llama.cpp RPC Remote Host Runbook

This runbook is the repeatable Windows-to-Windows RPC launch process for
llama-webui.

Current known-good topology:

- Coordinator: this desktop, running llama-webui at `http://localhost:8095`
- RPC worker: laptop at `192.168.1.146`
- RPC port: `50052`
- Tensor split: `34,66`
- Worker command: `rpc-server.exe -H 0.0.0.0 -p 50052 -c`

Use the IP shown in the WebUI `Remote RPC Device` fields. If the laptop gets a
new address, update the WebUI field and use that same address in the test
script. Do not rely on a previously saved address after the laptop changes IPs.

## Success Criteria

Before pressing `Start` in the WebUI, all of these should be true:

1. The worker laptop is awake and on the same trusted LAN.
2. The worker has the same compatible llama.cpp build as the coordinator.
3. `nvidia-smi` works on the worker if the worker is CUDA-backed.
4. `rpc-server.exe` is listening on `0.0.0.0:50052`.
5. From the coordinator, `Test-NetConnection <worker-ip> -Port 50052` succeeds.
6. In the WebUI, `Settings -> Runtime -> Remote RPC Device -> Check` says reachable.

If chat already works, the setup is good even if an old `Server Control` message
mentions a previous failed endpoint. The Launch Readiness panel and chat result
reflect the current state.

## Option A: Start RPC Manually On The Worker

On the laptop, open PowerShell in the llama.cpp binary directory:

```powershell
cd C:\Users\sergi\llama-rpc\bin
.\rpc-server.exe -H 0.0.0.0 -p 50052 -c
```

Keep that PowerShell window open. A healthy worker prints visible devices and:

```text
Starting RPC server
endpoint       : 0.0.0.0:50052
transport      : TCP
```

If Windows firewall blocks the port, run an elevated PowerShell on the worker:

```powershell
New-NetFirewallRule -DisplayName "llama.cpp RPC 50052" -Direction Inbound -Action Allow -Protocol TCP -LocalPort 50052 -Profile Private
```

## Option B: Use The Worker Script On The Worker

Copy or run the repository script on the laptop:

```powershell
.\scripts\start_rpc_worker.ps1 -Port 50052 -LlamaBinDir C:\Users\sergi\llama-rpc\bin -AddFirewallRule
```

For background mode:

```powershell
.\scripts\start_rpc_worker.ps1 -Port 50052 -LlamaBinDir C:\Users\sergi\llama-rpc\bin -AddFirewallRule -Background
```

The script prints the worker IPv4 addresses. Put the active LAN address into
the WebUI RPC Host field.

## Option C: Start The Worker Over SSH From The Coordinator

This requires OpenSSH Server on the laptop and an SSH key from the coordinator.

From the coordinator:

```powershell
.\scripts\start_remote_rpc_worker.ps1 -HostIp 192.168.1.146 -User sergi -KeyPath $HOME\.ssh\laptop -RemoteBinDir C:\Users\sergi\llama-rpc\bin -AddFirewallRule
```

If the laptop IP changes, replace `192.168.1.146` with the new worker IP.

## Verify From The Coordinator

Run this from the WebUI repository on the coordinator:

```powershell
.\scripts\test_rpc_endpoint.ps1 -HostIp 192.168.1.146 -Port 50052
```

Passing output means the WebUI RPC Check should also pass.

You can also use the raw Windows command:

```powershell
Test-NetConnection 192.168.1.146 -Port 50052
```

The important line is:

```text
TcpTestSucceeded : True
```

## Configure The WebUI

Open `http://localhost:8095`, then:

1. Open `Settings`.
2. Set `Runtime -> Mode` to `RPC Split`.
3. In `Remote RPC Device`, set:
   - `RPC Host`: `192.168.1.146`
   - `Port`: `50052`
   - `Tensor Split`: `34,66`
4. Click `Check`.
5. Click `Save Config` if you edited fields manually.
6. Select the model or preset.
7. Click `Start`, `Load Preset`, or `Load Selected`.

During model load, the worker logs many `set_tensor` lines. That is normal. It
is the one-time tensor upload to the remote backend.

## Troubleshooting

Ping works but TCP fails:

- `rpc-server.exe` is not running.
- The worker command used `-H 127.0.0.1` instead of `-H 0.0.0.0`.
- Windows firewall is blocking inbound TCP `50052`.
- The worker is on a public network profile or a different network segment.

TCP worked earlier but now fails:

- The laptop slept.
- The laptop IP changed.
- The RPC worker crashed or was closed.
- A previous `llama-server` process is still holding resources.

Stop stale processes on the worker:

```powershell
Get-Process rpc-server -ErrorAction SilentlyContinue | Stop-Process -Force
```

Stop stale coordinator processes from the WebUI with `Stop`, or from PowerShell:

```powershell
Get-Process llama-server -ErrorAction SilentlyContinue | Stop-Process -Force
```

If the worker logs `recv failed`, restart `rpc-server`, stop the coordinator
`llama-server`, and retry with the known-good `34,66` split before changing
other settings.
