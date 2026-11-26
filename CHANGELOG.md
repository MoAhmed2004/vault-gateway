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
- Measured (fixed seeds 0 to 299, payloads of 520+ characters split into random 20 to 27 character pieces, more than 16 pieces): 0/300 hidden in each of English without a decoy word, English with a leading or trailing decoy word, Arabic without a decoy word, and Arabic with a leading or trailing decoy word; also 0/300 with padding stripped. Worst case on 181 candidates in 3,800 characters: 1.1 to 1.7 ms for random junk and 4.1 to 6.9 ms for valid UTF-8 non-printable tokens (best of 20 calls, range over several runs on one developer machine), against the 400 ms TM-R2 budget. The worst case is bounded by the pruning rule, not by a piece limit; `tests/test_inspection.py` asserts the scan work rather than wall-clock time: at most 6,000 token feeds on both floods above (against about 16,400 for the naive design), and at most 16,000 on the near-threshold flood below.
- Note (Step 5A, closes audit finding L1): the scan work is quadratic in the worst case, not linear, so the Step 4.6 test name and this entry overstated it. The worst case is a near-threshold flood: 190 identical 15-byte tokens, each with 3 non-printable bytes (20% non-printable against the 15% the 85% rule allows), keep every start alive for most of the run. Measured: 13,913 token feeds against 18,145 for the naive uncapped design (about 77%); 21 to 26 ms on one developer machine against about 1.3 s for the naive re-decode (the audit measured about 13 ms against about 346 ms), well inside the 400 ms TM-R2 budget. The test is renamed `test_scan_work_is_pruned_on_hostile_candidate_floods`, and `test_scan_work_is_quadratic_but_bounded_on_a_near_threshold_flood` asserts the 16,000-feed bound. No code change, `BUNDLE_ALGORITHM_REVISION` stays 3, and `bundle_version` is unchanged.

### Open W2 exit-gate items

- Per-role backups (§5.2).
- Commit the `S1` and `D-BEN-1` manifest hashes under `data/manifests/`.
- Closed at Step 5A: the `TARGET_MODEL` startup check against Groq's active model list (§2.7, §3.9, AGENTS.md Rule 2), in `app/groq_client.py` and the `app/main.py` lifespan, tested in `tests/test_upstream_groq.py` and `tests/test_gateway_shell.py`.

## [Unreleased] - Step 5A (W3 to W5 scope: L1, Groq client, IP-5 caches, L4, live pipeline)

No frozen W2 interface changed: `contracts/` is untouched, the IP-5 keys come from `contracts/cache_keys.py` as frozen, and every log row is a frozen `RequestLogRecord`.

- IP-5 caches (`app/cache.py`): layer verdicts, egress verdicts and target responses in one local SQLite file (`CACHE_DB_PATH`, default `data/cache/vault_cache.db`, git-ignored), WAL, `synchronous=NORMAL`, `busy_timeout=5000`, one dedicated thread owning the only connection, fail-fast schema check, first write wins, writes bypassable for E4 (IP-8). Egress rows also store the L4 version and a different version is a miss, because the frozen egress key carries no L4 version [Project Decision].
- Groq client (`app/groq_client.py`): one module, pinned model, no retries, no redirects, whole-call timeout (`UPSTREAM_TIMEOUT_S`), a completion whose `model` differs from the pin is refused (`ALLOW_MODEL_FAILOVER=false`), every failure maps to `Error` / `upstream_error` and the uniform 503. Startup refuses to run unless `TARGET_MODEL` and every `PROBE_CANDIDATES` id are on Groq's active model list (§3.9 says each candidate id is validated at startup).
- L1 (`app/layers/l1_regex.py`): an 11-rule deterministic set on both views plus a ROT13 pass on the decoded view [Project Decision: the spec names a "regex set" but not its rules]. Measured worst case on a crafted 4,000-character input that carries every rule anchor: about 3.4 ms against the 5 ms L1 budget.
- L4 (`app/layers/l4_egress.py`): X0 no scan; X1 pattern-only Presidio with masking as `<ENTITY_TYPE>`; X2 adds the exact canary block and the planted-PII manifest scan (separator- and case-insensitive, masked). X3 is refused at startup until Step 5B. Masking is local because `presidio-anonymizer` is not in the lock file. Email domain validation uses the bundled suffix snapshot, so L4 makes no network call.
- Pipeline (`app/pipeline.py`, `app/routes.py`, `app/main.py`): bundle, L1, L2 and L3 hooks (disabled until Step 5B), TM-R3 policy (`app/timeout_policy.py`), cached target call, cached L4, one log row per request. A layer is timed in its worker thread against its TM-R2 budget, because asyncio timers on Windows have about 15.6 ms resolution and cannot enforce a 5 ms deadline. `L4_BUDGET_MS = 150` is new [Project Decision]; the L1 to L4 budgets must fit `TOTAL_BUDGET_MS`.
- Fail-closed gaps until later steps: `aegis_context` documents are refused (`internal_error`) until the sanitizer exists; `UPSTREAM_MODE` `stub` and `ollama` have no client, so admitted requests are `upstream_error`; shadow modes are refused at app creation.
- Outcome labelling: the gateway runs no oracle, so a delivered unaltered response is logged `Success` with `oracle_result=None`; the harness relabels it from the oracle (§3.2).
- New settings (`app/config.py`, `.env.example`): `GROQ_BASE_URL`, `UPSTREAM_TIMEOUT_S`, `TARGET_MAX_TOKENS`, `TARGET_TEMPERATURE`, `TARGET_SEED`, `TARGET_REASONING_EFFORT`, `EGRESS_VARIANT`, `L4_BUDGET_MS`, `CACHE_DB_PATH`, `CACHE_ENABLED`, `CACHE_WRITES_ENABLED`, `OBJECTIVE_ID` (one objective per gateway process; in production and demo the canary and planted PII are fresh per process, which is the PS-3 session).
- Test file names: `tests/test_layer_l4.py` and `tests/test_upstream_groq.py` replace the planned `tests/test_egress_l4.py` and `tests/test_model_pinning.py` (AGENTS.md map updated). Module paths are as planned.
