# Artifact Registry (§5.3)

SHA-256 hashes of every model file and dataset manifest used in experiments. `.onnx` files are not committed (see `.gitignore`); this registry is the record that pins them.

Compute a hash:

- PowerShell: `Get-FileHash -Algorithm SHA256 <path>`
- Python: `python -c "import hashlib,sys; print(hashlib.sha256(open(sys.argv[1],'rb').read()).hexdigest())" <path>`

Record lowercase hex digests. Any change to a file requires a new row; never edit an existing hash.

## Environment Lockfile Note (§2.7, §5.3)

`requirements.txt` was compiled with `pip-compile` on Windows (Python 3.11) and therefore pins Windows-specific transitive dependencies such as `colorama`. Google Colab runs on Linux: L3 fine-tuning notebooks on Colab install from `requirements.in` on top of Colab's Linux GPU runtime and export the exact environment, config, and seeds alongside the `.onnx` artifact (§2.7).

## ONNX Models

Fetched by `scripts/fetch_models.py`, which pins the same commit and SHA-256 values and refuses a mismatch; `tests/test_artifacts.py` keeps the script and this table identical. Both run on CPU through onnxruntime with `intra_op_num_threads=1`, `inter_op_num_threads=1` (AGENTS.md Rule 8).

| Layer | Model ID | Source revision | Local path | Tokenizer limit | SHA-256 | Recorded (date) |
| --- | --- | --- | --- | --- | --- | --- |
| L2, L4 X3 | `sentence-transformers/all-MiniLM-L6-v2` (repository ONNX export `onnx/model.onnx`, fp32) | `1110a243fdf4706b3f48f1d95db1a4f5529b4d41` | `models/minilm/model.onnx` | 256 | `6fd5d72fe4589f189f8ebc006442dbb529bb7ce38f8082112682524616046452` | 2026-11-27 |
| L3 | `protectai/deberta-v3-base-prompt-injection-v2` (repository ONNX export `onnx/model.onnx`, fp32, off-the-shelf, not fine-tuned) | `90c9989b1a342275dd0d1a95aad283c04e075671` | `models/deberta/model.onnx` | 512 | `f0ea7f239f765aedbde7c9e163a7cb38a79c5b8853d3f76db5152172047b228c` | 2026-11-27 |

Companion files (their hashes are part of each layer version, so a change re-keys the IP-5 layer cache):

| Layer | File | Source path | Local path | SHA-256 | Recorded (date) |
| --- | --- | --- | --- | --- | --- |
| L2, L4 X3 | MiniLM tokenizer | `tokenizer.json` | `models/minilm/tokenizer.json` | `be50c3628f2bf5bb5e3a7f17b1f74611b2561a3a27eeab05e5aa30f411572037` | 2026-11-27 |
| L3 | DeBERTa tokenizer | `onnx/tokenizer.json` | `models/deberta/tokenizer.json` | `752fe5f0d5678ad563e1bd2ecc1ddf7a3ba7e2024d0ac1dba1a72975e26dff2f` | 2026-11-27 |
| L3 | DeBERTa label config (`id2label`: 0 `SAFE`, 1 `INJECTION`) | `onnx/config.json` | `models/deberta/config.json` | `3093743035223c46b1497a72e939e56fa0a50afbd7bafbf7eb8aad060b8d23f8` | 2026-11-27 |

## Dataset Manifests

| Manifest | Path | Source | Rows | SHA-256 | Recorded (date) |
| --- | --- | --- | --- | --- | --- |
| L2 bootstrap seed (development and demo only, not experiment data; measured runs populate L2 from S1-train, §2.4) | `data/l2_seed/bootstrap.jsonl` | hand-written | 24 | `f3b5225e50650720614a52459b196d63c28f90661594eef21cb328d2668860e9` | 2026-11-27 |
