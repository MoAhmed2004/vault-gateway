# AGENTS.md

V.A.U.L.T. gateway, Specification Draft v3.3.2. This file is the single source of truth for project invariants, package boundaries, and PS-5 rules. Every human or AI contributor must read it before changing code, tests, docs, or UI.

Scope at W2 (Step 2.5): Step 1 audit findings closed, locked environment, neutral `contracts/` package (IP-9, §3.2), attacker isolation guard (PS-2, §5.2), and the permanent PS-5 linter. No `app/`, `telemetry/`, or `harness/` runtime logic exists yet.

Label convention: `[Spec ID]` marks a rule taken from the spec text. `[Project Decision, extends <ID>]` marks a stricter choice made by this project that the spec permits but does not state verbatim.

## Invariant Rules

1. [IP-7, IP-8] Runtime & Storage: Python 3.11, FastAPI, Pydantic v2, httpx. Run a single uvicorn worker (`--workers 1`) [Project Decision, extends IP-7: IP-7 itself specifies a bounded threadpool with a semaphore and queue limit, default one worker per physical core; the single process keeps one in-memory queue and one SQLite writer]. Store SQLite databases on local disk (never synced folders) with `journal_mode=WAL`, `synchronous=NORMAL`, `busy_timeout=5000`, written by a single writer thread fed from an in-memory queue.

2. [§2.7, §3.9] Target Model Pinning: `TARGET_MODEL` is checked against Groq's active model list at startup; fail fast if missing. `PROBE_CANDIDATES` are `openai/gpt-oss-20b`, `qwen/qwen3.8-27b` (non-thinking instruct mode), and `openai/gpt-oss-120b`. Set `ALLOW_MODEL_FAILOVER=false` so runtime failover never mixes models mid-experiment. All Groq calls must go through a single client module.

3. [PS-1, CA-1, CA-2, CA-3, TM-R1] Single-Turn Request Schema & Streaming Body Cap: Use one canonical request schema (`extra="forbid"`): `user_prompt: str` (max 4,000 chars), optional `aegis_context: list[ContextDocument]` (config defaults: max 4 docs, max 1,500 chars/doc), and `stream: bool = False`. Reject `stream=true`, multi-turn history, extra fields, client-supplied `system`/`tool` roles, or oversized payloads with uniform HTTP 503. Cap raw HTTP body size at 128 KiB (131,072 bytes) by counting bytes as chunks arrive in ASGI middleware BEFORE JSON parsing.

4. [IP-1, IP-2, TM-R1] InspectionBundle & Per-View Window Limit: `raw_view` is untouched. `decoded_view` applies Unicode NFKC, strips Unicode category `Cf`, and decodes Base64/hex (depth 2, >=20 chars or padded, >=85% printable UTF-8, including whitespace-joined adjacent candidates). Cap `decoded_view` at 4,000 chars (reuses `MAX_PROMPT_CHARS`). Slice both views into character windows (600 chars, stride 300). If a window exceeds a layer's tokenizer limit (256 for MiniLM, 512 for DeBERTa), split it into sub-windows BEFORE checking the 16-window per-view limit so `truncation_count` remains `0`. Refuse with uniform HTTP 503 ONLY if `decoded_view` > 4,000 chars or windows in EITHER view exceed 16 after sub-window splitting. Never truncate.

5. [IP-9, §3.2] Neutral Contracts Package: Shared types live in `contracts/` and import only stdlib and pydantic (zero project imports): frozen `LayerResult` (`layer`, `view`, `per_window_scores` as an immutable tuple, `overall_score`, `verdict` in `allow|block|timeout|error`, `latency_ms`, `truncation_count` == 0, `error_message`), `Outcome` enum (`Blocked-ingress`, `Admitted-refused`, `Egress-saved`, `Success`, `Error`), `canonical_hash()` (deterministic JSON + SHA-256 via `hashlib`; never use Python `hash()`), and the `TelemetrySink` Protocol (`record_request`, `record_shadow`) that `app/` layers, middleware, and request handlers call instead of importing `telemetry/`.

6. [PS-2, PS-3, TM-S3, D-PII] Domain-Separated Secrets: In `eval` mode, refuse to start if `HMAC_SECRET` equals the exact placeholder `vault_dev_secret_change_me` from `.env.example`. Derive secrets with domain separation: canary via `HMAC-SHA256(secret, f"canary|{objective_id}")`, synthetic PII via `HMAC-SHA256(secret, f"pii|{objective_id}|{entity_type}")` seeding pinned `Faker` (`faker==30.1.0`), and sanitizer nonce (`<EXTERNAL_DATA_{nonce}>`) via `HMAC-SHA256(secret, f"nonce|{objective_id}|{doc_id}")`. In `production`/`demo` mode, use `secrets.token_hex(16)`. Keep shared defensive system-prompt text first and per-objective secrets last.

7. [TM-R6] Global Uniform 503 Refusal: Register global FastAPI exception handlers for `RequestValidationError` (422), `StarletteHTTPException` (404/405), and unhandled exceptions so every block or error returns identical status `503`, identical headers, and body `{"detail": "Service Unavailable. Request blocked by security policy."}`. Read `X-Trial-ID` for server logging only; NEVER echo `X-Trial-ID` in responses. Every client sets `max_retries=0`.

8. [IP-6, IP-7, TM-R4] Threadpool, Atomic ChromaDB & Egress/Reasoning Rule: Run all layer calls in a bounded threadpool and shed overload with 503 (IP-7). Set ONNX `intra_op_num_threads=1` and `inter_op_num_threads=1` [Project Decision, extends IP-7: keeps per-call CPU use predictable inside the bounded pool; not verbatim IP-7 text]. Read the immutable ChromaDB collection version pointer once per request. Cache raw provider JSON including `reasoning`. L4 egress scanning (IP-6) runs ONLY in egress variants `X1` and above: `X0` performs no egress scan; `X1` runs pattern-only Presidio; `X2` adds exact canary matching; `X3` adds the decoded-text and similarity leak check (§2.1). Where it runs, L4 scans both `content` and `reasoning` using pattern-only Presidio (`EMAIL_ADDRESS`, `PHONE_NUMBER`, `CREDIT_CARD`, `US_SSN`, `IBAN_CODE`, no spaCy NER) and, from `X2`, canary matching; `reasoning` findings are logged and `reasoning` is stripped, while L4's block verdict is determined solely by `content`. [TM-R4] Any L4 timeout or error ALWAYS fails closed with uniform HTTP 503, independent of `TIMEOUT_POLICY`. Return client-visible responses strictly as `{"content": str}`. E4 uses a local fixed-delay stub upstream.

9. [PS-2, §5.2] Strict Attacker Isolation: Oracles and evaluation runners live in `harness/`. `attacker/` talks only to the public HTTP endpoint and must NEVER import `app`, `telemetry`, `harness`, `HMAC_SECRET`, manifests (`data/manifests`), or caches (`data/cache`) (enforced by AST unit test).

10. [§5.5] Verification Discipline: Every implemented Spec ID must have a unit test. Frozen interfaces (end of W2, including package layout) change only via changelog and spec update. `scripts/ps5_lint.py` must exit 0 before every commit.

11. [PS-5, Anti-Vibecode Engineering Standard] Explicitly ban all 20 AI-cliche patterns across code, comments, logs, docs, and the W15 SOC Admin Monitor UI: (1) purple-to-blue gradient, (2) gradient hero text, (3) emojis in headings/logs/comments, (4) Inter font everywhere, (5) colored border cards, (6) glassmorphism cards, (7) low-contrast dark mode, (8) 3 icon boxes in a row, (9) badge above the headline, (10) Lucide icons everywhere, (11) untouched shadcn UI, (12) fade-in on scroll, (13) cursor-following beam, (14) buttons fade on hover, (15) inconsistent spacing, (16) em dashes everywhere, (17) generic buzzword copy, (18) serif italic accents, (19) Space Grotesk + Instrument Serif, (20) grain over a gradient. Build the W15 monitor as a flat, high-contrast utilitarian SOC console (1px solid borders, tabular monospace metrics, strict green/amber/red status indicators).
    - [Project Decision, extends PS-5] Spec pattern 16 names em dashes only; this project also bans en dashes (U+2013) everywhere. `scripts/ps5_lint.py` enforces zero emojis, zero em dashes, and zero en dashes.
    - Plain utilitarian status glyphs (U+2713, U+2717, U+2715, U+25CF, U+25A0, box-drawing, ASCII and arrow symbols) are allowed in the W15 SOC console and are not flagged. True pictographs and emoji (including U+2705, U+274C, U+26A0, and the emoji presentation selector) remain banned.
    - The linter skips `data/raw/`, `data/cache/`, `data/translations/`, `results/`, `.venv/`, `.git/`, `chroma_db/`, `__pycache__/`, binary and PDF files, and any file larger than 5 MiB. `data/translations/` is excluded because it holds hand-curated multilingual text (Arabic, French, Spanish).
    - Text-level UI pattern checks (`linear-gradient`, `backdrop-filter`, `Inter`, `Space Grotesk`, `Instrument Serif`, `lucide`) will be added to `scripts/ps5_lint.py` before W15.

## Package Boundaries (§5.2, §5.3)

| Package | Owns | May import |
| --- | --- | --- |
| `contracts/` | Frozen shared types, `Outcome`, `canonical_hash()`, `TelemetrySink` Protocol | stdlib, pydantic only (zero project imports) |
| `app/` (startup composition: `app/main.py`, `app/wiring.py`) | Process startup, dependency wiring of a concrete sink | `contracts`, `telemetry` (wiring/startup composition ONLY) |
| `app/` (all other modules) | Gateway: config, schema, middleware, inspection, client, refusal handlers, request handlers | `contracts` (log through `contracts.TelemetrySink`, never `telemetry`) |
| `app/layers/` | Detection layers L1 to L4 | `contracts`, `app` (never `telemetry`; log through `contracts.TelemetrySink`) |
| `telemetry/` | SQLite single-writer logging; concrete `TelemetrySink` implementation | `contracts` |
| `harness/` | Oracles, evaluation runners, manifests | `contracts`, `app`, `telemetry` |
| `attacker/` | Black-box attack client over public HTTP only | `contracts`, httpx, stdlib |
| `scripts/` | Operator utilities (hashing, model export, PS-5 lint) | any |
| `tests/` | Unit tests, one or more per implemented Spec ID | any |

## Spec-ID-to-Module-and-Test Map

Module and test paths are planned targets. They are frozen with the package layout at end of W2. Rows marked "implemented" exist and pass at Step 2.

| Spec ID(s) | Rule | Topic | Module | Unit test |
| --- | --- | --- | --- | --- |
| IP-9, §3.2 | 5 | Neutral contracts, `TelemetrySink` taking typed records (implemented) | `contracts/layer_result.py`, `contracts/outcome.py`, `contracts/hashing.py`, `contracts/telemetry_sink.py` | `tests/test_contracts.py` |
| RT-V6, IP-4 | 5 | Frozen `RequestLogRecord` (RT-V6 fields) and `ShadowRecord` / `EgressVariantRecord` (IP-4 fields) (implemented) | `contracts/records.py` | `tests/test_contracts.py` |
| §5.2, §5.3 | 9, 10 | Repo-wide package boundary table enforced by AST (implemented) | `contracts/`, `telemetry/`, `app/` | `tests/test_package_boundaries.py` |
| §2.7, §3.9 | 2 | Target model pinning | `app/config.py`, `app/groq_client.py` | `tests/test_model_pinning.py` |
| PS-1, CA-1, CA-2, CA-3 | 3 | Single-turn request schema | `app/schemas.py` | `tests/test_request_schema.py` |
| TM-R1 | 3, 4 | Body cap and window limit | `app/middleware.py`, `app/inspection.py` | `tests/test_body_cap.py`, `tests/test_window_limit.py` |
| IP-1, IP-2 | 4 | InspectionBundle views and windows | `app/inspection.py` | `tests/test_inspection_bundle.py` |
| IP-3 | 4, 7 | Production mode: L1, L2, L3 in order, short-circuit on block | `app/pipeline.py` | `tests/test_pipeline_modes.py` |
| IP-4 | 4 | Shadow modes `shadow_ingress_only` and `shadow_full_e2e` (no short-circuit) | `app/pipeline.py` | `tests/test_pipeline_modes.py` |
| IP-5 | 5 | Cache key formats: layer verdicts (prompt hash, layer version, bundle version, ChromaDB version); egress verdicts (response hash, manifest ID, variant); target responses (model ID, parameters, full message list) | `app/cache_keys.py` | `tests/test_cache_keys.py` |
| PS-2, PS-3, TM-S3, D-PII | 6 | Domain-separated secret derivation | `app/secret_derivation.py` | `tests/test_secret_derivation.py` |
| D-PII | 6, 8 | Pattern-only Presidio detects 100% of planted synthetic PII | `app/layers/l4_egress.py` | `tests/test_egress_l4.py` |
| PS-4 | 6 | Fair C0: same system prompt, parameters, and message builder as every configuration | `app/prompting.py` | `tests/test_fair_c0.py` |
| TM-R2 | 1, 8 | Time budgets: L1 5 ms, L2 50 ms, L3 150 ms, total 400 ms | `app/config.py`, `app/pipeline.py` | `tests/test_time_budgets.py` |
| TM-R3 | 7 | Ingress timeout policy P-open, P-degrade (L2 threshold minus 0.05), P-closed | `app/timeout_policy.py` | `tests/test_timeout_policy.py` |
| TM-R4 | 7, 8 | L4 timeout or error always fails closed with uniform 503 | `app/layers/l4_egress.py` | `tests/test_egress_l4.py` |
| TM-R5 | 1, 8 | E4 evaluates all three ingress policies under load (timeout-induction) | `harness/e4_load.py`, `app/upstream_stub.py` | `tests/test_e4_load.py` |
| TM-R6 | 7 | Uniform 503 refusal | `app/errors.py` | `tests/test_uniform_refusal.py` |
| TM-S1 | 4 | L1 and L3 run per `aegis_context` document; flagged document dropped and logged | `app/sanitizer.py` | `tests/test_sanitizer.py` |
| TM-S3 | 6 | Per-document sanitizer nonce wrapping `<EXTERNAL_DATA_{nonce}>` (nonce from `app/secret_derivation.py`) | `app/sanitizer.py` | `tests/test_sanitizer.py` |
| TM-S2 | 4 | Delimiter hygiene: NFKC, strip zero-width, escape `<EXTERNAL_DATA` family case-insensitively | `app/sanitizer.py` | `tests/test_sanitizer.py` |
| TM-S4 | 6 | Defensive system prompt identical in every configuration including C0 and D0 | `app/prompting.py` | `tests/test_fair_c0.py` |
| TM-S5 | 6 | E5 wrapping variants D0 to D3 (datamarking in D3) | `app/sanitizer.py` | `tests/test_sanitizer.py` |
| IP-6 | 8 | L4 egress scope, variants X0 to X3, reasoning strip | `app/layers/l4_egress.py` | `tests/test_egress_l4.py` |
| IP-7 | 1, 8 | Bounded threadpool, load shedding, atomic ChromaDB pointer | `app/executor.py`, `app/vector_store.py` | `tests/test_threadpool_shedding.py` |
| IP-7, IP-8 | 1 | SQLite single-writer telemetry | `telemetry/writer.py` | `tests/test_telemetry_writer.py` |
| PS-2, §5.2 | 9 | Attacker isolation guard, hardened at Step 2.5 (implemented) | `attacker/client.py`, `harness/oracles.py`, `harness/runner.py` | `tests/test_attacker_isolation.py` |
| §5.5 | 10 | Spec coverage and AI use log | `tests/`, `docs/AI_USE_LOG.md`, `docs/artifacts.md` | `tests/test_spec_coverage.py` |
| PS-5 | 11 | Permanent PS-5 linter with dataset skips and emoji range fix (implemented) | `scripts/ps5_lint.py` | `tests/test_ps5_lint.py` |
