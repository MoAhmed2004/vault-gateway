# Changelog

All changes to frozen W2 interfaces (§5.2) are recorded here. After the W2 interface freeze, a frozen interface changes only with the agreement of all five roles, an entry in this file, and a spec revision before the code changes (AGENTS.md Rule 10).

## [Unreleased] - W2 Interface Freeze (Steps 4 and 4.5)

The upcoming freeze tag marks "interfaces frozen". It does NOT mark "W2 exit gate passed": the exit-gate items listed under "Open W2 exit-gate items" below are still pending.

### Frozen at Step 4

- Layer interface and `LayerResult` (`contracts/layer_result.py`).
- Log schema: `RequestLogRecord`, `ShadowRecord`, `EgressVariantRecord` (`contracts/records.py`), SQLite `SCHEMA_VERSION = 1` (`telemetry/writer.py`).
- `InspectionBundle` schema (`contracts/inspection.py`, IP-1, IP-2) and its builder (`app/inspection.py`).
- IP-5 cache key formats (`contracts/cache_keys.py`).
- Objective file, translation table, dataset manifest and `CanaryPIIManifest` schemas (`contracts/datasets.py`).
- System-prompt, manifest and message builder (`app/prompting.py`, PS-2, PS-3, PS-4).
- Canonical request schema (`app/schemas.py`, PS-1) and the uniform refusal (`app/errors.py`, TM-R6).
- Package layout: `contracts/`, `app/`, `attacker/`, `harness/`, `telemetry/`.

### Deviation from the planned layout

- IP-5 cache keys live in `contracts/cache_keys.py` instead of the planned `app/cache_keys.py`. Reason: `telemetry/` and `harness/` caches must share the key format, and neither may import `app` (AGENTS.md package boundaries), while both may import `contracts`.

### Step 4.5 - pre-tag closure of the Step 4 audit findings

- F1 (IP-1): Base64 joining in `app/inspection.py` now tries contiguous runs of adjacent candidates longest first from each position, so one neighbouring 20+ character word (or path) no longer makes the whole-group join fail and leaves the split payload undecoded. The first version of this fix capped runs at `MAX_JOIN_TOKENS = 16` (`BUNDLE_ALGORITHM_REVISION = 2`); that cap introduced the R1 regression below and was removed in Step 4.6. Revision 2 never left the working tree.
- F2 (TM-R6): the app is built with `redirect_slashes=False`; `POST /v1/chat/`, `GET /healthz/` and `POST /v1/chat//` get the uniform 503 instead of a 307 redirect.
- F3 (TM-R1, §3.2): `Content-Length` must be 1 to 20 ASCII digits. Latin-1 superscript digits (`\xb2`) and 5,000-digit values are refused as `body_cap` (`Blocked-ingress`) instead of raising `ValueError` and being logged as `internal_error`.
- F4 (IP-2): `TokenizerLimit` checks its name against `contracts.inspection.TokenizerName` at construction, so a bad tokenizer name fails at startup rather than on every request.
- F5 (IP-5, IP-1): the `InspectionBundle` schema requires `prompt_hash == canonical_hash({"raw_view": raw_view})`.
- F6 (PS-3): every `derive_*` function in `app/secret_derivation.py` refuses `|` in each HMAC message field, so `derive_sanitizer_nonce(s, "O1|d", "x")` can no longer collide with `("O1", "d|x")`.
- F7 (RT-V6): config field-cap refusals in `app/routes.py` log the parsed prompt's `prompt_hash` instead of `UNPARSED_PROMPT_HASH`.
- F8 (§5.2, §5.4): this changelog was added; `AGENTS.md` and `app/middleware.py` now state that Starlette's `ServerErrorMiddleware` sits outside `FailClosedMiddleware` (the registered global `Exception` handler still guarantees the uniform 503), and that uvicorn's own plain-text 400 for unparseable HTTP never reaches the ASGI app and is outside TM-R6.

### Step 4.6 - R1 regression in joined Base64 decoding

- R1 (IP-1, TM-R1, TM-R2): capping joined runs at 16 pieces let a payload split into more than 16 pieces fall off a 4-character block boundary at piece 16, decode only its prefix, and misalign every remaining piece, hiding the end of the payload (52/300 English and 63/300 Arabic trials in the audit). Re-decoding every longer run from every start without a cap fixed that but cost about 110 ms on 181 junk candidates in 4,000 characters.
- Resolution: joined runs now use incremental forward-scan decoding with no 16-piece cutoff. From each start, Base64 (and hex) decode only newly completed blocks into a strict incremental UTF-8 decoder, so no token is re-decoded as part of a longer run; a start stops scanning as soon as its bytes are not UTF-8, a token breaks the encoding rules (`=` before the end), or the 85% printable ratio can no longer be reached even if every remaining character were printable. The longest valid run wins. A stop inside a group must end on a block boundary or `=` padding, so the tokens after it start on a fresh block; when no such stop exists the longest run is used anyway, and its continuation is joined without the whitespace gap (an unpadded payload next to an ordinary word would otherwise be left undecoded or split mid-word).
- `MAX_JOIN_TOKENS` is removed. `BUNDLE_ALGORITHM_REVISION` is now 3 and the `bundle_version` payload carries `join_scan = "incremental-uncapped"` in its place, so every IP-5 layer cache key built from an earlier bundle is invalidated.
- Measured (fixed seeds 0 to 299, payloads of 520+ characters split into random 20 to 27 character pieces, more than 16 pieces): 0/300 hidden in each of English without a decoy word, English with a leading or trailing decoy word, Arabic without a decoy word, and Arabic with a leading or trailing decoy word; also 0/300 with padding stripped. Worst case on 181 candidates in 3,800 characters: 1.1 to 1.7 ms for random junk and 4.1 to 6.9 ms for valid UTF-8 non-printable tokens (best of 20 calls, range over several runs on one developer machine), against the 400 ms TM-R2 budget. The worst case is bounded by the pruning rule, not by a piece limit; `tests/test_inspection.py` asserts the scan work (at most 6,000 token feeds on both floods, against about 16,400 for the naive design) rather than wall-clock time.

### Open W2 exit-gate items

- Per-role backups (§5.2).
- `TARGET_MODEL` startup check against Groq's active model list (§2.7, §3.9, AGENTS.md Rule 2; planned `app/groq_client.py`, `tests/test_model_pinning.py`).
- Commit the `S1` and `D-BEN-1` manifest hashes under `data/manifests/`.
