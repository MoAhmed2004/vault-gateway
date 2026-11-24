# Artifact Registry (§5.3)

SHA-256 hashes of every model file and dataset manifest used in experiments. `.onnx` files are not committed (see `.gitignore`); this registry is the record that pins them.

Compute a hash:

- PowerShell: `Get-FileHash -Algorithm SHA256 <path>`
- Python: `python -c "import hashlib,sys; print(hashlib.sha256(open(sys.argv[1],'rb').read()).hexdigest())" <path>`

Record lowercase hex digests. Any change to a file requires a new row; never edit an existing hash.

## Environment Lockfile Note (§2.7, §5.3)

`requirements.txt` was compiled with `pip-compile` on Windows (Python 3.11) and therefore pins Windows-specific transitive dependencies such as `colorama`. Google Colab runs on Linux: L3 fine-tuning notebooks on Colab install from `requirements.in` on top of Colab's Linux GPU runtime and export the exact environment, config, and seeds alongside the `.onnx` artifact (§2.7).

## ONNX Models

| Layer | Model ID | Source revision | Local path | Tokenizer limit | SHA-256 | Recorded (date) |
| --- | --- | --- | --- | --- | --- | --- |
| | | | | | | |

## Dataset Manifests

| Manifest | Path | Source | Rows | SHA-256 | Recorded (date) |
| --- | --- | --- | --- | --- | --- |
| | | | | | |
