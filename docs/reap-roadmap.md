# REAP Roadmap

This repository is the serving and benchmarking layer for local `llama.cpp` deployments. It is not yet an end-to-end personalized pruning system.

## Target End State

The intended end state is:

1. collect personal AI interaction data
2. normalize it into a calibration and retrieval corpus
3. prune or otherwise adapt a large MoE checkpoint off-device
4. quantize and convert the result for local serving
5. serve the artifact here through `llama.cpp`
6. optionally add retrieval on top of the same corpus

## Where Personal Data Fits

Personal exports should live outside the repository.

One practical layout is:

```text
external-data/
├── extracted/
│   ├── codex_conversations.jsonl
│   ├── gemini_conversations.jsonl
│   ├── opencode_conversations.jsonl
│   ├── pi_mono_conversations.jsonl
│   └── droid_conversations.jsonl
├── raw/
│   ├── google-takeout.zip
│   ├── codex_raw.tar.gz
│   └── other-raw-archives.tar.gz
└── normalized/
    ├── normalized_conversations.jsonl
    ├── reap_calibration.jsonl
    └── summary.json
```

Nothing in this repository assumes those paths anymore. The corpus can live anywhere as long as the downstream tools can see it.

## What This Repo Already Covers

- browsing, loading, and serving GGUF models
- persisting local chats and runtime settings
- downloading GGUFs from direct URLs
- benchmarking model candidates on the local board
- documenting which configs actually worked on RK3588

## What This Repo Does Not Yet Do

- ingest personal exports directly in the UI
- build the normalized corpus
- compute REAP saliency
- prune experts itself
- quantize a freshly pruned checkpoint into GGUF
- schedule retrieval over the corpus during generation

## Practical Personalized REAP Workflow

### Step 1: Extract And Normalize Elsewhere

Use a separate extraction tool or pipeline to gather data from:

- local coding agents
- CLI assistants
- exported web-chat histories
- any other relevant prompt/response logs

Normalize that into:

- clean conversations for retrieval
- shorter calibration samples for pruning analysis

### Step 2: Run Pruning Off-Device

For a personalized `Qwen3.5-35B-A3B` path, pruning should be treated as workstation or server work, not RK3588 work.

The RK3588 board is the deployment target, not the right place to do heavy calibration passes over a large native MoE checkpoint.

### Step 3: Quantize For Deployment

Once pruning is done:

- convert or quantize to GGUF
- choose quantization based on the deployment board
- benchmark both prompt and generation speed
- validate RAM and swap behavior

### Step 4: Serve And Compare

Bring the resulting GGUF back to this repository and compare it against the known-good presets in the web UI.

## Compute Expectations

What seems realistic:

- RK3588: good for serving and benchmarking compact GGUFs
- desktop GPU with `12 GiB` VRAM: useful for dataset work, quantized evaluation, and experimentation
- larger native pruning jobs on `35B` MoE checkpoints: likely still better suited to a stronger GPU box

This repository keeps that boundary explicit so the deployment UI stays simple and robust.
