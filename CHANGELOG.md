# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **Playbooks** (`playbook`, `rule`, `playbook_eval`, `Playbook`): business rules evaluated against *many rows at once*, the third fan-out shape alongside `typesafe_eval` (one row, N questions) and `typesafe_eval_each` (one contract per segment). A playbook judges a **group of rows together**, which is the only shape that can answer a rule no single row can violate -- "no employee may claim more than $5,000 in total", "every escalated ticket must get a support reply". Pure Python over existing primitives (`polar_llama/playbook.py`): rows are grouped, each group's rows become a `List[Struct]` that `typesafe_eval`'s existing structured-state handling sends as a JSON array of records, and every rule in the playbook is answered in the SAME request per group -- so N rules cost no extra requests. Groups go out in parallel under the existing `POLAR_LLAMA_MAX_CONCURRENCY` bound, inheriting the TypeSafe client's 429/529/5xx backoff-with-jitter retry and per-group `_error` isolation. **Zero new Rust.** `rule(statement, passes=..., fails=...)` frames a noul as an adherence question (the answer is the probability the rows *adhere*, so violations threshold at `< 0.5`); `playbook(name, **rules)` is a named, reusable rule set; `playbook_eval(df, playbook, by=..., records=..., compute=..., context=..., usage=..., max_rows_per_group=...)` returns one row per group with the group keys, `row_count`, anything computed, one Float64 column per rule, and `_error`. Two API decisions are driven by measurement against the live API rather than assumption: (1) **signal dilutes with group size** -- on a planted violation, clean vs violating separated at 6 rows (0.33/0.98), was marginal at 20 (0.40/0.49) and *indistinguishable* by 60 (0.20/0.39), 150 (0.18/0.20) and 300 (0.18/0.22), and this is NOT specific to arithmetic (a purely semantic rule, one un-answered ticket among many, held at 11 rows and was missed at 59), so `max_rows_per_group` (default 25) warns rather than silently returning noise; (2) **Polars should do the arithmetic** -- judging a pre-computed aggregate instead of raw rows to add up was exact at every size tested (0.03 vs 0.98 from 20 through 1,000 rows summarised) because the state stays small however many rows it covers, which is what `compute=` exists for. The state a group receives is always a named object (`records`, `row_count`, plus anything from `compute`/`context`); `row_count` is always sent partly so the shape cannot depend on whether `compute` happened to be passed, and both names are reserved. DataFrame-level orchestration like `quality_report` / `induce_codebook`, deliberately NOT added to the `.llama` expression namespace. See `docs/PLAYBOOKS.md`.

## [0.9.0] - 2026-09-19

### Added
- **TypeSafe System One inference layer** (`typesafe_eval`, `noul`, `choice`, `score`, `typesafe_models`): a native Rust client for the [TypeSafe API](https://docs.typesafe.ai/api) (`src/model_client/typesafe.rs`) plus a Polars plugin expression over it (`src/typesafe_expr.rs`), exposed to Python as `polar_llama/typesafe.py`. TypeSafe is deliberately *not* a chat-completions provider, so it intentionally does **not** implement the `ModelClient` trait the OpenAI/Anthropic/Gemini/Groq/Bedrock clients share and is **not** added to the `Provider` enum: there is no prompt and no free-text completion, and forcing it through a messages-in/text-out trait would have meant faking both. One request carries a single `state` plus a map of typed `questions` and returns one typed `answer` each -- `noul` (yes/no, a probability in `[0, 1]`), `choice` (pick one of a closed set, returning the pick, a probability per option, and a confidence), `score` (rate against ordered levels, returning a probability-weighted value that can land *between* levels, a probability per level, and a confidence). That shape is exactly a DataFrame row, which is the whole reason it belongs here: the answers are typed and calibrated rather than parsed out of prose, so they land as ordinary Float64/String columns you can filter, sort, threshold and join on. **Every question rides in one request per row** -- TypeSafe's own "speculative fan-out" guidance -- and the API is shaped so a caller cannot accidentally pay for the state once per question. Output is a Struct the caller `.unnest()`s: per question `<id>` (Float64 for noul/score, String for choice) plus `<id>_confidence` for choice/score (a noul gets none, because TypeSafe returns none -- a single probability already *is* the distribution); `probabilities=True` adds `<id>_p_<option>` / `<id>_p_<level index>`; `usage=True` adds `_model` (the *resolved* version that ran, e.g. `jev-1.13.0`, not the `jev-latest` alias), `_input_tokens`, `_output_tokens` and `_latency_ms` (total wall clock including retries). The declared output dtype is derived from the questions via `output_type_func_with_kwargs`, so `.collect_schema()` on a LazyFrame resolves the full shape **without spending a token**; a single `ColumnPlan` is the sole source of truth for both the schema and the values written into it, so the two cannot drift. Ordering is preserved end to end by passing questions as a JSON *array* rather than an object (`serde_json::Map` is a `BTreeMap` in this build and would silently re-sort them). State assembly: one input expression is sent as that bare value, several are sent as a JSON object keyed by column name (the shape TypeSafe recommends, since each part keeps a descriptive name) with numbers and booleans keeping their JSON type; nested Polars dtypes convert structurally and recursively (a `List` column becomes a JSON array, a `Struct` column a JSON object) rather than shipping an `AnyValue` debug repr to the model, since TypeSafe's `state` explicitly accepts arrays and objects; a length-1 input broadcasts across the frame so a constant (a shared policy, a schema, the document every row is scored against) can ride along with per-row state; and `state_json=True` reinterprets string inputs as pre-encoded JSON documents. Structured `instructions`/`criteria` (`string | object | array | null`) pass through untouched, and raw TypeSafe question JSON is accepted alongside the builders so an existing payload works unchanged. **Failure is per row, never per frame**: `_error` is always present and null on success; a row whose state is entirely null is never sent and returns all-null with a *null* `_error` (a missing input is not a failure); a question TypeSafe declines to answer leaves its columns null rather than erroring; and locally-checkable mistakes (no questions, a one-option choice, an unknown question type, colliding output field names) raise in Python or at schema time, before anything is billed. `429`/`529`/transient 5xx retry with exponential backoff and jitter honouring `Retry-After`, with the jitter *subtracted* so the 30s cap is a real ceiling; `401`/`422` are not retried, since a bad key or malformed question will not fix itself and retrying only bills for it again. Batches reuse the existing shared pooled HTTP client and the same `POLAR_LLAMA_MAX_CONCURRENCY` bound as every other provider; `POLAR_LLAMA_TYPESAFE_MAX_RETRIES` (default 3) tunes the retry budget, and `TYPESAFE_API_KEY`/`TYPESAFE_BASE_URL` follow the existing per-provider env conventions. Available as `typesafe_eval(...)` and `.llama.typesafe_eval(...)`, with `typesafe_models()` listing the model catalogue. Zero new dependencies. See `docs/TYPESAFE.md`.

- **Contracts and the per-line dimension for TypeSafe** (`typesafe_eval_each`, `contract_questions`, `choice_field`, `score_field`, plus `contract=` on `typesafe_eval`): where `typesafe_eval` fans out over *questions* for one state, `typesafe_eval_each` fans out over *segments* -- a `List` column of lines/clauses/passages/chunks goes in, and every segment comes back with the same contract answered for it. All of a row's segments are evaluated in **one request**, not one per segment: measured against the live API on an 8-clause contract that is 3.2x fewer input tokens (743 vs 2,367) and 8x fewer round trips, and it is the only shape that preserves context -- a clause reading "renews automatically unless either party gives 60 days notice" is unreadable in isolation, and batched the model sees its neighbours. The segments go into `state` once keyed by their global index and each question names the index it applies to (`QuestionSpec::retarget_to_line`, preserving the original instructions verbatim under `question`); repeating the segment text inside every question instead measured only ~11% more expensive, so the id indirection buys context rather than bytes. **Chunking** is automatic and necessary: the request ceiling is token-based, not count-based (640 questions / ~40k input tokens succeeded live, 1,200 returned `400 max_tokens_exceeded`, an error type absent from TypeSafe's documented table and correctly treated as non-retryable), so `max_questions` (default 200) caps questions per request and segments are split to respect it -- each segment costing one question *per contract field*, so chunk size is `max_questions / len(contract)` -- with chunks fired concurrently under the existing `POLAR_LLAMA_MAX_CONCURRENCY` bound via a new `evaluate_batch_varying` (each chunk carries its own question set, unlike `evaluate_batch` where every row shares one). `line_id` is global, so a chunk boundary never renumbers a line; a failing chunk marks only its own segments in `_error` while the rest of the document resolves; a null segments list is never sent. Output is `List[Struct{line_id, line, <answers>, _error}]` -- a Polars expression cannot change row cardinality, so `.explode()` is the documented step to one row per segment (`include_segment=False` drops the `line` column). Further positional arguments become shared context folded into every chunk's state (a title, a policy, the query being matched against), length-1 broadcasting. **Contracts**: a Pydantic model whose fields name the features to extract, with the Python type picking the question type -- `bool` -> Noul, `Literal[...]`/`Enum` -> Choice, numeric + `score_field(levels=...)` -> Score -- usable via `contract=` on both expressions. Deliberately does NOT reuse `_pydantic_to_json_schema`: that helper strips every sibling of a `$ref` for OpenAI strict mode, which would silently drop the `description` on an Enum-typed field, and the description is exactly what becomes the question's instructions; contracts resolve `$ref`/`allOf`/`anyOf` themselves and keep outer keys. A `bool` field returns a *probability*, not a boolean -- the calibrated value is the product, and thresholding is the caller's decision. A plain `str` field is rejected with an explanation rather than silently mishandled: TypeSafe answers are typed over a closed set and have no free-text primitive. `'#'` is reserved in question ids (the per-line key separator) and rejected. The segments argument is dtype-checked in the output-type function, so passing a String column instead of a List fails at schema resolution -- naming `str.split` as the fix -- before a request is billed. Available as `typesafe_eval_each(...)` and `.llama.typesafe_eval_each(...)`. Zero new dependencies. See `docs/TYPESAFE.md`.

### Changed
- **pyo3's `extension-module` is now a default Cargo feature** rather than an unconditional one (`default = ["extension-module"]`). `extension-module` leaves the CPython symbols undefined, which is required for the importable wheel but makes the crate impossible to link into a test binary -- so `cargo test` could not run *any* of the crate's `#[cfg(test)]` modules, and CI never invoked it. `cargo test --lib --no-default-features` now links against libpython and runs them (109 tests). Default features are unchanged, so `cargo build`, `maturin develop` and the wheel build behave exactly as before.

## [0.8.3] - 2026-07-14

### Added
- **Deterministic run manifests** (`RunManifest`, issue #85, pure Python -- zero Rust changes): a small, JSON-serializable audit record of *what a run asked for* -- provider, model, prompt/schema content (as sha256 hashes, never plaintext, unless opted in), sampling params, and the exact fingerprint the runtime itself uses for dedupe/checkpointing -- with a `manifest_id` computed deterministically from those fields, so two runs with identical configuration (different `created_at` timestamps, different token counts) get the same `manifest_id`. New `polar_llama/manifest.py`: `build_manifest(df=..., symbol=..., provider=..., model=..., system_prompt=..., response_model=..., prompt_template=..., params=..., seed=..., usage_column=..., checkpoint=..., response_cache=..., dedupe_stats=..., store_texts=...)` builds a `RunManifest`; `with_manifest_id(df, manifest)` attaches a constant `manifest_id` column to a result DataFrame; `save_manifest`/`load_manifest` (also `RunManifest.save`/`.load`) round-trip a manifest through an atomically-written, sorted-keys JSON sidecar, with `load_manifest` recomputing the hash over the loaded determinism fields and raising `ManifestIntegrityError` on any mismatch (tamper/corruption detection); `replay(manifest, df, input_column, ...)` re-issues the request a manifest describes against re-supplied row data, verifying (by hash) that any re-supplied `system_prompt`/`response_model`/`prompt_template` matches what the manifest recorded before running (`ManifestMismatchError` if not; `verify=False` to skip), auto-filling from `RunManifest.texts` when `store_texts=True` was used at build time. **Reuses, never reimplements**: `config_fingerprint` is computed by moving `_request_fingerprint` out of `polar_llama/__init__.py` into `polar_llama/keys.py` as the public `request_fingerprint` (aliased back as `_request_fingerprint` so the existing #75 checkpoint / #77 dedupe call sites are untouched) -- a manifest's `config_fingerprint` is therefore *guaranteed* identical to the runtime fingerprint for the same configuration, not a parallel hash that could drift. `checkpoint_id`/`response_cache_id` are read straight out of a `Checkpoint`/`ResponseCache` store's `_meta.json` (issues #75/#77); `aggregate_usage` sums a `usage=True` output column's `USAGE_DTYPE` struct (issue #76) unmodified; `dedupe_stats` snapshots a `polar_llama.DedupeStats` (issue #77) verbatim. The determinism hash excludes `manifest_id` itself, `created_at` (the one field two identical runs may differ on), `aggregate_usage`/`dedupe_stats` (nondeterministic outputs), and `checkpoint_id`/`response_cache_id` (identify a resume mechanism, not the request) -- but DOES include `polar_llama_version` and `endpoint` (a library upgrade or different provider base-URL is a config change for audit purposes). Documented caveat: hosted `inference_async`/`inference_messages` forward no sampling params today, so `params={}` on a manifest means "provider server-side defaults apply" -- the manifest guarantees what the *client* asked for, not what the provider actually used. See `docs/RUN_MANIFESTS.md`.

## [0.8.2] - 2026-07-14

### Added
- **Cross-platform local server docs + CI coverage for llama.cpp** (issue #84, Tier 1 -- docs/CI/tests only, zero library-code changes, zero new dependencies): `engine="server"` (`polar_llama/local/server_backend.py`) was already a thin, provider-agnostic adapter over the existing async fan-out -- it works with *any* OpenAI-compatible local server, not just Apple-Silicon-only `mlx_lm.server`/`vllm-mlx`, but that wasn't documented or exercised in CI against a real non-Apple server. This release adds: **`docs/local_llamacpp_backend.md`**, install/run instructions for [llama.cpp](https://github.com/ggml-org/llama.cpp)'s `llama-server` (prebuilt Linux/Windows release binaries, CUDA/ROCm/Vulkan builds, `brew install llama.cpp` on macOS) with the sampling-parameter caveat front and center -- `max_tokens`/`temperature`/`top_p`/`stop` are NOT forwarded to the request body by the Rust `OpenAIClient` (`src/model_client/openai.rs::format_request_body` intentionally omits them, since some hosted models reject them), so sampling must be set via `llama-server`'s own CLI flags (`--temp`, `-n`, `--top-p`, ...) -- plus a feature matrix (platform, structured outputs, sampling-param forwarding, batching, collapsed prefill, batched quantized KV, `usage=True`, streaming `on_token`) across `server`+`llama-server` / `server`+`mlx_lm.server` / `in_process`(mlx); a new **`llamacpp_server_test`** Linux CI job (`.github/workflows/CI.yml`) that downloads a pinned llama.cpp release build (`b10004`) and a pinned tiny GGUF (`bartowski/SmolLM2-135M-Instruct-GGUF`, cached by filename via `actions/cache`), starts a real `llama-server`, polls `/health` until ready, and runs a gated live test against it; and **`tests/test_llamacpp_server_live.py`**, skipped everywhere except that CI job (gated on `POLAR_LLAMA_LLAMACPP_URL`), asserting response *shape* (correct row count, non-null/non-empty String completions, none matching the `{"_error": ...}` in-band error envelope, a `system=` case, and a `usage=True` case decoding llama-server's real `usage` block via the existing `parse_usage`) rather than content, since the pinned model is a tiny (~135M-param) instruct model with no strong output guarantees. `docs/local_mlx_backend.md` is updated to de-scope its Apple-only framing: it's now clearly labeled as covering the MLX-specific engines (`engine="in_process"`, and `engine="server"` pointed at `mlx_lm.server`/`vllm-mlx`), with `engine="server"` itself called out as cross-platform and linked to the new doc. A Tier 2 **in-process** llama.cpp binding (e.g. `llama-cpp-python` or ONNX Runtime, giving non-Apple machines a no-server-process option with sampling parameters that *are* forwarded, analogous to `engine="in_process"` for MLX) is explicitly deferred to a separate follow-up issue.

## [0.8.1] - 2026-07-14

### Added
- **Local (offline) embeddings** (`embedding_local`, issue #83): the fully-offline counterpart of `embedding_async` -- runs an embedding model in-process via `mlx_embeddings` instead of calling a hosted provider API, so a document/query pipeline can go text -> `List[Float64]` embeddings -> `HnswIndex`/`cosine_similarity`/`knn_hnsw`/`cluster_embeddings` with zero network calls and no API key. Output dtype is `List[Float64]`, identical to `embedding_async` (verified: `embedding_async`'s Rust plugin, `src/expressions.rs::embedding_async`, only nulls a `None` input row and sends `""` through to the model like any other text -- `embedding_local` matches that null-only convention). Mirrors the `polar_llama/local/engine.py` design (precedent: issue #56's `inference_local`) with a SEPARATE registry (`polar_llama/local/embed.py`): `LocalEmbeddingEngine` Protocol, a deterministic dependency-free `FakeEmbeddingEngine` (SHA-256-derived per-text vectors, so a duplicated document is its own nearest neighbor in tests -- no `mlx` import, CI-safe), and `MlxEmbeddingEngine` (lazy-loaded singleton wrapping `mlx_embeddings.load`/`generate`, reading pooled vectors off `output.text_embeds`; API shape verified on-device against `mlx-embeddings==0.1.0` with the default model, `mlx-community/bge-small-en-v1.5-bf16` -- output is already L2-normalized, ~1330 docs/sec warmed up on an M-series Mac). `register_embedding_engine`/`get_embedding_engine`/`clear_embedding_registry` provide the same test-injection seam as the generation side, and the SAME `POLAR_LLAMA_LOCAL_ENGINE=fake` environment override flips the whole local stack (generation + embeddings) to the dependency-free fake engine. `batch_size` chunks large columns (default 64); `normalize=True` (default) L2-normalizes every output vector; a per-row engine failure degrades to a null embedding, never aborting the batch. Available as `embedding_local(expr, ...)` and `.llama.embedding_local(...)`. `mlx-embeddings>=0.1.0` is a new line in the existing optional `[local]` extra (Apple-silicon-only via its own `mlx` dependency, same story as `mlx-lm`) -- never imported at `import polar_llama`/`import polar_llama.local` time (CI runs the FakeEmbeddingEngine path with zero mlx). See `docs/LOCAL_EMBEDDINGS.md`.

## [0.8.0] - 2026-07-14

### Added
- **Persistent, incrementally updatable HNSW index** (`HnswIndex`, issue #82): a DataFrame-native, stateful, serializable ANN index (precedent: `Checkpoint` #75, `Codebook` #78) built on top of `instant-distance`'s `HnswMap` -- which is immutable once built, so this is a small LSM-like layer around it, not a new HNSW implementation. `HnswIndex.build(df, id_col, embedding_col)` builds a fresh index; `.add(df, id_col, embedding_col)` upserts into a brute-force staging buffer, queryable immediately without touching the immutable main graph; `.remove(ids)` soft-deletes via a tombstone set. `.query(query_df, embedding_col, k)` returns `DataFrame(query_id|neighbor_id|distance|rank)` (`query_id` is the query row's 0-based index); `.query_one(vector, k)` is the single-vector convenience form; `.knn(expr, k) -> pl.Expr` bridges the same batch core into a `map_batches` expression (`List[Struct{neighbor_id, distance, rank}]` per row). Queries over-fetch from the main graph (`k' = min(k + tombstone_count, ef_search)`) to compensate for tombstoned hits, merge with the staging buffer, and filter dead ids. **Compaction** (automatic via `auto_compact=True`, default on, once the staging buffer or tombstone count crosses `max(threshold_min, threshold_ratio * len(index))`; or manual via `.compact()`) rebuilds the main graph from every currently-live point and clears staging/tombstones. External ids are arbitrary caller strings, mapped to a stable internal-id space that survives compaction. `.save(path)`/`HnswIndex.load(path)` round-trip the whole index -- main graph, pending staging/tombstones, id maps, dimension, and the pinned build seed (for deterministic compaction) -- through a small `bincode`-encoded file with a magic+version header; `bincode` is the only new Rust dependency this feature adds (pure-Rust, serde-only -- `instant-distance`'s already-enabled `with-serde` feature supplies `Serialize`/`Deserialize` for the graph itself). Implemented as a new `#[pyclass]` (`src/index.rs::PyHnswIndex`, Python name `_HnswIndexCore`) taking/returning `pyo3-polars` `PySeries`/`PyDataFrame` directly (zero-copy Arrow, no JSON shuttle) with batch queries releasing the GIL; `src/ann.rs` and the stateless `knn_hnsw` expression are untouched -- `HnswIndex` reuses `ann::EmbeddingPoint`'s cosine-distance metric verbatim. See `docs/VECTOR_SIMILARITY_AND_ANN.md`.

### Notes
- Query latency at 100k points is single-digit milliseconds after compaction; loading a 100k-point index from disk is a low-single-digit-second `bincode` deserialization (see `scripts/bench_hnsw_index.py`). `instant-distance` has no mmap/lazy-load path, so `.load()` always deserializes the full graph into memory up front -- there is no partial/streaming load.

## [0.7.3] - 2026-07-14

### Added
- **Human-in-the-loop review loop** (`export_review_sample`, `import_corrections`, `corrections_to_trainset`, `retune_from_corrections`, `CorrectionResult`, issue #81): a DataFrame-shaped loop for turning a sample of LLM output into human corrections and feeding them back into the prompt optimizer, with zero reimplementation of existing machinery. `export_review_sample(df, n=, strata=, allocation="proportional"|"equal", confidence_column=, oversample_low_confidence=, seed=, path=, format=)` draws a deterministic, stratified sample without replacement (largest-remainder or equal-split quotas, with a min-1-per-stratum guarantee and cap-and-redistribute for undersized strata) using a hand-rolled Efraimidis-Spirakis weighted sampler (pure stdlib `random`, no numpy) -- `oversample_low_confidence` biases the sample toward rows with low (or null, treated as `0.0`) model confidence. Writes CSV out of the box; XLSX (`format="xlsx"`) needs the new optional `[excel]` extra (`pip install polar-llama[excel]`, pulling in `xlsxwriter`) and raises a clear `ImportError` naming that extra when it's missing -- never a required dependency. `import_corrections(df, corrections, code_column=, corrected_column="corrected_code")` joins reviewed corrections back onto the full DataFrame (`_review_id` join key; a blank/null cell means "not reviewed", not "the code is empty"; duplicate ids raise; unmatched correction ids warn/raise/ignore per `on_unmatched=`) and returns `CorrectionResult(df, kappa, agreement_rate, n_reviewed, n_changed, n_unmatched_corrections, n_unreviewed)` -- `kappa`/`agreement_rate` are computed by reusing `polar_llama.reliability.cohens_kappa` verbatim, including its `NaN`-on-zero-chance-agreement convention. `corrections_to_trainset`/`retune_from_corrections` map corrections onto the exact column contract `BootstrapFewShot.compile` requires (one column per signature input/output field name) and hand the result straight to the existing `polar_llama.optimize.BootstrapFewShot(...).compile(...)` -- no bootstrap or demo-selection logic is duplicated. Defaults (`threshold=0.0`, `use_gold_outputs=True`) deliberately diverge from `BootstrapFewShot`'s own defaults, since corrections are by definition the rows the model got wrong; classic bootstrap semantics are reachable via kwargs. Like `induce_codebook`/`quality_report`, these are DataFrame-level orchestration functions, not added to the `.llama` namespace. See `docs/HITL_WORKFLOW.md`.

## [0.7.2] - 2026-07-14

### Added
- **Survey data-quality flags** (`quality_report`, `QualityConfig`, issue #80): a DataFrame-shaped pipeline that scores every respondent for common survey data-quality problems and returns a per-flag summary, without ever dropping a row. Every score is a graded `Float64` in `[0, 1]` (higher = more suspicious); booleans are derived by thresholding, and a null score (not enough signal) always resolves to `flag = False`. Heuristic tier (zero API calls): `straightlining_score`, `gibberish_score`, and `duplicate_answer_score` are new Rust plugin expressions (`src/quality.rs` pure functions, unit-tested there, mirroring the `src/metrics.rs`/issue #79 split); `response_length_score` (robust z-score of answer length) and `speeder_score` (percentile- or median-fraction-based) are pure Polars. All five are available standalone and via the `.llama` namespace (`pl.col("g1").llama.straightlining_score([...])`, etc.). Embedding/LLM tier (opt-in, `QualityConfig(llm_tier=True)`, default off): cross-respondent `near_duplicate` detection (`embedding_async` + `knn_hnsw` + `cosine_similarity`) and a `likely_ai` stylistic-heuristic flag (`inference_messages(..., response_model=...)`, also exposed standalone as `ai_likelihood`) -- both carry a mandatory "flag, not verdict" framing; `likely_ai` additionally documents the false-positive risk of AI-text detectors, including their documented bias against non-native English speakers, and states it must never be sole grounds for exclusion or panelist sanction. `quality_report(df, config)` returns `QualityReport(df, summary)`: `.df` is `df` plus a `quality` struct column (schema reflects which inputs were configured -- unconfigured sub-structs are omitted, not null-filled); `.summary` is one row per active flag plus a final `any_flag` row. All default thresholds are documented, subjective conventions -- see `docs/QUALITY_FLAGS.md`.

## [0.7.1] - 2026-07-14

### Added
- **Inter-rater reliability metrics** (`cohens_kappa`, `krippendorffs_alpha`, issue #79): DataFrame-shaped aggregation expressions -- `df.select(kappa=cohens_kappa("llm", "human"))`, `df.group_by("topic").agg(alpha=krippendorffs_alpha(["r1", "r2", "r3"]))` -- backed by pure, unit-tested Rust (`src/metrics.rs`, zero new dependencies). `cohens_kappa(a, b, weights=None|"linear"|"quadratic")` reproduces `sklearn.metrics.cohen_kappa_score` bit-for-bit (pairwise-complete rows, sklearn's label-*index* weighting convention, and its `NaN`-not-null convention when expected-by-chance agreement is zero). `krippendorffs_alpha(cols, level="nominal"|"ordinal"|"interval"|"ratio")` reproduces the `krippendorff` PyPI package's coincidence-matrix algorithm bit-for-bit on its own published fixtures, with nulls treated as missing ratings (units with fewer than 2 non-null ratings are excluded). Both accept `n_bootstrap=`/`ci=`/`seed=` for a nonparametric case-resampling bootstrap CI, returned as `Struct{value, ci_low, ci_high}` (`.struct.unnest()`); `n_bootstrap=None` (default) returns a plain `Float64`. Available as `.llama.cohens_kappa(...)` / `.llama.krippendorffs_alpha(...)` on expressions too. This is the first aggregation-shaped plugin expression in the package (`returns_scalar=True`, new in `polar_llama/utils.py::register_plugin`), so it composes with `.select()`, `group_by().agg()`, and lazy frames for free. Multi-label/set-valued codes (MASI) are deferred -- `docs/RELIABILITY_METRICS.md` documents two supported strategies (per-label binary alpha, exact-set nominal alpha) per the issue's "or documented strategy" acceptance arm. See `docs/RELIABILITY_METRICS.md`.

## [0.7.0] - 2026-07-14

### Added
- **Codebook induction** (`cluster_embeddings`, `induce_codebook`, `apply_codebook`, `codebook_to_taxonomy`, issue #78): a DataFrame-shaped pipeline for building and applying a qualitative-coding codebook from a text column, with no external clustering dependency. `cluster_embeddings(embeddings_col, k=..., k_min=..., k_max=..., seed=...)` is a whole-column Rust plugin expression (same pattern as `knn_hnsw`) backed by a hand-rolled k-means++ / Lloyd's algorithm (`src/kmeans.rs`): a splitmix64 PRNG, cosine-distance spherical k-means, and a sampled-silhouette heuristic for automatic `k` selection when `k` is omitted -- zero new dependencies (no `linfa`/`ndarray`/`rand`/scikit-learn/numpy). `induce_codebook(df, column, ...)` embeds (via `embedding_async`, unless `embedding_column=` is given), clusters, picks per-cluster exemplars with ordinary `sort` + `group_by().agg(...head(n))`, and asks the LLM to name and define each cluster (`inference_messages` + a strict-schema response model), merging codes that independently collide across clusters. It returns the input DataFrame plus `cluster_id`/`cluster_distance` columns (same row count and order -- no reshaping) and a `Codebook`. `apply_codebook(text_col, codebook)` multi-label-codes a text column against a codebook in one `inference_messages` call per document, using a response model shaped as a fixed-length `List[{code, applies, confidence, evidence}]` (never a `Dict`/dynamic-key object -- the issue #51 strict-mode lesson), and composes directly back onto the same DataFrame `induce_codebook` returned via a plain `with_columns` (no join/explode). `codebook_to_taxonomy(codebook)` bridges an induced codebook into the `tag_taxonomy`/`_create_taxonomy_pydantic_model` taxonomy shape for callers who want single-label (mutually exclusive) classification instead of `apply_codebook`'s multi-label evaluation. See `docs/CODEBOOK_INDUCTION.md`.

## [0.6.3] - 2026-07-14

### Added
- **In-run duplicate collapsing and a persistent, cross-job response cache** (`inference_async(..., dedupe=True)` / `response_cache=...`, and `inference_messages`, issue #77): `dedupe=True` sends each unique row in a batch to the backend once and fans the result back out to every row that shared it, at zero extra I/O. `response_cache="path"` (or a `ResponseCache(path, ttl=..., on_mismatch=...)`) adds a persistent store on top -- reusing `polar_llama/checkpoint.py`'s Parquet-part store, atomic writes, and fingerprint-based invalidation verbatim -- so identical requests from a *different* run or process reuse a prior result. `response_cache=` implies `dedupe=True` automatically. Only successful results are ever persisted; a failed row always recomputes. Pass `dedupe_stats=DedupeStats()` to get `rows_total`/`rows_null`/`cache_hits`/`rows_collapsed`/`calls_made`/`hit_rate` back after a collect. `ResponseCache(...).prune()`/`.clear()` compact/invalidate the store manually. Both kwargs default to off/`None` -- byte-identical to the pre-#77 code path. See `docs/design/RESPONSE_CACHE.md`.
- Extracted the hit/pending/fan-out grouping bookkeeping shared by checkpointing and dedupe into `polar_llama.keys.plan_collapse`/`fan_out`; `checkpoint.checkpointed_expr` now calls these instead of duplicating the logic (verified behavior-identical -- `tests/test_checkpointing.py` passes unmodified against the refactor).

### Notes
- Not currently supported together (raise `ValueError`): `response_cache=` + `checkpoint=` (two persistent stores for one expression); `response_cache=` + `usage=True` (the usage envelope's failed-row shape isn't recognized by the store's ok/fail classification -- same reason as `checkpoint=` + `usage=True`). `dedupe=True` alongside `checkpoint=` is allowed and silently subsumed (checkpoint already collapses duplicates per batch). `dedupe=True` + `usage=True` (no store) is allowed; duplicate rows carry the same `usage` struct as the row actually computed, so `SUM(cost_usd)` over-counts by the collapse factor -- `dedupe_stats.calls_made` is the true-spend signal.

## [0.6.2] - 2026-07-14

### Added
- **Per-row usage & cost accounting** (`inference_async(..., usage=True)` / `inference_messages`, issue #76): returns a `Struct{response, usage: Struct{input_tokens, output_tokens, cached_tokens, latency_ms, cost_usd}}` column. `usage=False` (default) is byte-identical to before. Provider usage metadata is parsed for OpenAI, Groq, Anthropic, Gemini, and Bedrock (Anthropic's split cache tokens are normalized so `cached_tokens ⊆ input_tokens`), latency is measured around the HTTP call in Rust, and `cost_usd` is computed from a packaged, overridable price table (`polar_llama/pricing.py` + `polar_llama/data/`; pass `price_table=` or `pricing.register_model(...)`). Unknown models yield `cost_usd = null` with a one-time warning rather than an error. The MLX in-process local engine reports `input_tokens`/`output_tokens`/`latency_ms` with `cost_usd = 0.0`.

### Notes
- Caveats (documented): Anthropic cache-*write* tokens are folded into `input_tokens` at the base input rate, so `cost_usd` slightly undercounts when prompt-cache writes occur. `usage=True` combined with `checkpoint=` currently raises `ValueError` (envelope-wrapped error rows would be misclassified by the checkpoint store); supporting both together is deferred (see the issue #76 design notes).

## [0.6.1] - 2026-07-14

### Added
- **Resumable batch runs / checkpointing** (`inference_async(..., checkpoint="path")` and `inference_messages`, issue #75): completed rows are persisted to a sidecar Parquet store as a run progresses, keyed by a sha256 content hash over the row input plus the full run configuration (provider, model, system prompt, response schema, and the endpoint base-URL override). Re-running resumes and skips completed rows, so a job killed at 50% costs ≈ one full pass on resume. Changing any config input invalidates old entries automatically; failed rows are stored as failed and retried by default (`Checkpoint(path, retry_failed=False)` to keep the stored error). The store is crash-durable (atomic append-only Parquet parts) and the API stays a lazy, DataFrame-shaped Polars expression that composes in `with_columns`/`LazyFrame` pipelines. The hashing primitives live in `polar_llama/keys.py` for reuse by content-hash caching (#77). See `docs/design/CHECKPOINTING.md`.

## [0.6.0] - 2026-07-13

### Added
- **Streaming inference** (`inference_stream()`, issue #74): stream completions token-by-token with an `on_token(row_index, delta)` callback, returning a DataFrame-only `Struct{text: Utf8, finished: Boolean}` column (`STREAM_RESPONSE_DTYPE`) so the DataFrame stays rectangular even when a stream is cut off. `finished=False` covers every non-terminal outcome: a mid-stream provider `error` event, a transport drop, EOF without a completion marker, an `on_token` callback that raises, or Ctrl-C — all of these are reported as a `RuntimeWarning` with partial text returned rather than an exception propagating out of the expression. Native SSE parsing for OpenAI, Groq (OpenAI-compatible), and Anthropic; Gemini and Bedrock stream via a buffered fallback (one full-text delta then completion). `response_model`/`response_format` are rejected immediately with a clear `ValueError` — streaming is text-only.
- `GROQ_BASE_URL` environment variable override for the Groq endpoint, matching the existing `OPENAI_BASE_URL` / `ANTHROPIC_BASE_URL` pattern (also needed so streaming's mock-server tests can point Groq at a local server).
- `reqwest`'s `stream` feature, enabling chunked body streaming (`bytes_stream()`) for the new SSE drivers.

## [0.5.3] - 2026-07-13

### Fixed
- `tag_taxonomy()` no longer fails on OpenAI (and other strict-schema providers) with `invalid_request_error: 'required' ... Extra required key 'thinking'` (#51). Two stacked OpenAI-strict-mode schema violations are fixed: (1) the per-field `thinking` reasoning is now `List[{value, reasoning}]` instead of a `Dict[str, str]` dynamic-key map (which strict mode rejects for lacking `properties`/`required`); (2) `_pydantic_to_json_schema` now strips sibling keywords from `$ref` nodes (pydantic emits taxonomy fields as `{"$ref": ..., "description": ...}`, which strict mode rejects with "$ref cannot have keywords"). Verified end-to-end with live OpenAI (`gpt-4o-mini`), Anthropic (`claude-haiku-4-5`), and Groq (`llama-4-scout`) calls. `_validate_strict_mode_schema` now also warns when a user-supplied response model uses a `Dict`-typed field.

## [0.5.2] - 2026-07-12

### Fixed
- `inference_local(engine="in_process")` on Gemma 3n no longer crashes end-to-end (#70). The mlx-lm #1384 batched shared-KV fix was only applied on the prompt-tuning bridge/benchmark paths, never on the `inference_local` load path, so `MlxBatchEngine` loaded gemma-3n unpatched and batched generation raised `ValueError: too many values to unpack (expected 2)`. Both this patch and a new guard are now applied automatically at model load (`polar_llama/local/engine.py::_apply_mlx_patches`).
- Unmasked the real in-process error: `mlx_lm.generate.BatchGenerator.stats` divided `prompt_tokens / prompt_time` with `prompt_time == 0` in its teardown, raising `ZeroDivisionError` *during* exception handling and replacing the underlying error. A new guarded, idempotent patch (`apply_batchgen_stats_zerodiv_patch`) wraps the context manager so a body exception is never masked and a zero-time exit yields `tps = 0.0`.

## [0.5.1] - 2026-07-05

### Added
- **Local prompt-tuning bridge** — `polar_llama.local.make_local_inference_fn(model, ...)` returns an `inference_fn` that drives `polar_llama.optimize`'s DSPy-style `Predict` / `BootstrapFewShot` / `InstructionOptimizer` against on-device gemma-3n (mlx-lm), with no cloud/Rust path. It reuses the singleton-loaded weights and applies the mlx-lm #1384 batched fix automatically.
- **Collapsed-prefill opt-in for `inference_local(engine="in_process")`** via `POLAR_LLAMA_LOCAL_COLLAPSE=1` — shares the common prompt prefix across rows (also the bridge's default). Measured **~2.8× faster on a full prompt-tuning schedule** (3.4× on a demo-laden eval) at identical, parity-verified output; the dominant speedup whenever rows share a long prefix (a shared `system` prompt, or few-shot demos during tuning). Mutually exclusive with `POLAR_LLAMA_LOCAL_KV_BITS` (that path takes precedence).
- `MlxBatchEngine.get_model_and_tokenizer()` to reuse the singleton-loaded weights.

### Fixed
- `InstructionOptimizer` no longer crashes with `TypeError: the truth value of a Series is ambiguous` when the proposer model returns the `instructions` field as a JSON array instead of a newline-delimited string — list/Series values are flattened to newline-delimited text (`polar_llama/optimize.py`). This surfaces with small local models that emit `{"instructions": [...]}`.

## [0.5.0] - 2026-07-04

### Added
- **Local MLX inference backend** — `col(...).llama.inference_local(...)` for on-device batched generation on Apple Silicon, with two engines:
  - `engine="server"` (default): points the existing async fan-out at a local OpenAI-compatible endpoint (`mlx_lm.server` / vllm-mlx) via `OPENAI_BASE_URL` — no Rust changes.
  - `engine="in_process"`: a `map_batches` UDF wrapping `mlx_lm`'s `BatchGenerator`, behind a `LocalEngine` protocol with a `FakeEngine` seam so the batching/ordering/error-isolation logic is testable on CPU/CI without a GPU. Optional extra: `pip install polar-llama[local]`.
- **Collapsed prefix prefill** (`polar_llama/local/collapsed_prefill.py`): computes a shared prompt prefix once instead of re-prefilling it per row (token-level longest-common-prefix + suffix-only batching). Measured **10.36× vs sequential** on gemma-3n E4B (32 rows, 5 KB shared prompt) at 32/32 exact greedy parity.
- **Batched quantized KV cache** (`BatchQuantizedKVCache`; opt-in via `POLAR_LLAMA_LOCAL_KV_BITS=4`): closes mlx-lm's "quantized KV × batching" gap. ~47% KV-memory cut (Q4) at parity with fp16 batched output, roughly doubling the batch/context that fits in 24 GB (memory/capacity win; not a throughput speedup with the current unfused attention path).
- **mlx-lm #1384 fix** (runtime monkeypatch, `polar_llama/local/_mlx_patches.py`): corrects a RoPE offset-aliasing bug that garbled *batched* generation on hybrid Gemma 3n / Gemma 4 models; verified token-identical to sequential. Ready-to-post upstream PR in `patches/PR_1384.md`.
- Build-vs-buy gate benchmark (`benchmarks/local_mlx_gate.py`), parity/throughput validators, and design/decision docs under `docs/`.

### Notes
- The local backend and its Apple-GPU tests are opt-in; CI runs the CPU-safe logic tests (`-m "not local_gpu"`, FakeEngine, no mlx). The `[local]` extra requires Python ≥ 3.10.

## [0.3.0] - 2026-06-10

### Added
- **Tool use / MCP integration** (`tools_to_response_model`, `mcp_tools`, `execute_tool_calls`, `tool_results_to_message`, and `.llama` namespace methods): LLMs emit tool calls as structured output and `execute_tool_calls` runs every call of every row batch-parallel against an MCP server (`tools/call`) or a Python `executor` callable; per-call failures are data. Guide: `docs/TOOL_USE.md`; example: `examples/tool_use_calorie_tracker.py`.
- **Provider-native prompt caching** (`cache=True` / `CacheConfig`): shares a cached system prefix across rows via Anthropic `cache_control` content blocks, with 5-minute and 1-hour (`ttl="1h"`, `extended-cache-ttl` beta) TTLs; `inference_messages` now also accepts List(Struct) input in addition to JSON strings.
- **DSPy-style prompt optimization engine** (`polar_llama.optimize`):
  - `Signature` — declarative task specs (`"question -> answer"` shorthand or explicit `InputField`/`OutputField` with types and descriptions)
  - `Predict` — executable LLM module that runs one parallel, batched inference per DataFrame and returns `pred_<field>` columns
  - `evaluate` — metric-based scoring of a module against a labeled DataFrame
  - `BootstrapFewShot` — mines few-shot demonstrations from training rows the module already answers correctly (akin to `dspy.BootstrapFewShot`)
  - `InstructionOptimizer` — COPRO-style instruction search: an LLM proposes instruction rewrites, every candidate is evaluated on the trainset, best one wins
  - Fully testable offline via an injectable `inference_fn` backend
- `OPENAI_BASE_URL` / `ANTHROPIC_BASE_URL` environment overrides for proxies and gateways
- `POLAR_LLAMA_MAX_CONCURRENCY` environment variable to bound concurrent in-flight requests per batch (default 64; previously unbounded)
- Gemini: native `system_instruction` support and native JSON-schema structured outputs (`response_json_schema`)
- Pricing data for current models (GPT-5/4.1/o-series, Claude 4.x/Fable 5, Gemini 2.5, Bedrock Claude 4.5) and o200k tokenizer detection for GPT-4.1/GPT-5/o3/o4

### Changed
- **Updated default models** (previous defaults were retired/decommissioned):
  - OpenAI: `gpt-4-turbo` → `gpt-4o-mini`
  - Anthropic: `claude-3-opus-20240229` (retired) → `claude-opus-4-8`
  - Gemini: `gemini-1.5-pro` → `gemini-2.5-flash`
  - Groq: `llama3-70b-8192` (decommissioned) → `llama-3.3-70b-versatile`
  - Bedrock: `anthropic.claude-3-haiku-20240307-v1:0` → `us.anthropic.claude-haiku-4-5-20251001-v1:0`
- OpenAI/Groq requests no longer hardcode `temperature`/`max_tokens` (newer models such as the o-series and GPT-5 reject those parameters)
- Bedrock region now respects `AWS_REGION`/`AWS_DEFAULT_REGION` before falling back to `us-east-1`
- Removed import-time debug printing from the Python package and the native module
- Minimum supported Python version is now 3.9 (abi3-py39 wheels)
- Removed deprecated `new()`/`with_model()` Rust constructors (deprecated since 0.2.0); use `new_with_model()`

### Fixed
- **Gemini structured outputs** previously sent OpenAI-style Bearer auth and always failed; Gemini now authenticates via the `x-goog-api-key` header on all paths
- **Bedrock structured outputs** previously attempted a raw HTTP POST to the Bedrock endpoint; they now route through the AWS SDK like plain requests
- Bedrock now works from the synchronous `inference` expression (previously returned an error)
- TLS verification is no longer disabled (`danger_accept_invalid_certs` removed); the shared client uses rustls with the OS certificate store
- **Bedrock prompt caching** now actually emits a `CachePoint` block on the Converse request when `cache_control` is set (previously the marker was injected but never sent). Bedrock supports only the default ~5-minute cache type.
- **Prompt-cache grouping**: rows with no cacheable system prefix are no longer lumped into a single serial cache group (they are processed as individual rows).
- **Windows wheels build again**: the AWS SDK now uses the modern `ring` rustls provider (`aws-smithy-http-client/rustls-ring`) instead of `aws-lc-rs`, whose `aws-lc-sys` native build fails under MSVC ("C atomics require C11 or later"). This also drops the legacy rustls 0.21 path.
- Bumped vulnerable transitive crates flagged by `cargo audit` (bytes, quinn-proto, rustls-webpki, time, anyhow, memmap2). The remaining pyo3 < 0.29 advisories are pinned by pyo3-polars 0.26 and are explicitly ignored in CI until the bridge supports pyo3 0.29.

### Performance
- `inference_messages` now accepts `List(Struct{role, content})` input natively in Rust, so the default path no longer wraps every call in a Python `map_batches` UDF (keeps queries lazy/streaming)
- Single shared HTTP client with connection pooling (previously a new client per batch)
- Bounded request concurrency via buffered streams instead of unbounded `join_all`
- JSON schemas are compiled once per batch instead of once per row
- AWS Bedrock client/credential chain is cached per region instead of being rebuilt per request
- Vector similarity ops (`cosine_similarity`, `dot_product`, `euclidean_distance`) use a contiguous-slice fast path
- Tokenizer/pricing lookups hoisted out of per-row loops in cost expressions
- Removed redundant per-row clones of schemas, models, and message batches in the expression layer (~400 lines of duplicated dispatch removed)

### Security
- **Fixed RUSTSEC-2025-0020** (pyo3 buffer overflow): upgraded pyo3 0.23 → 0.27 via pyo3-polars 0.26
- Removed `ureq` 2.x and `once_cell` dependencies (sync path now reuses the async clients; `std::sync::LazyLock` replaces `once_cell`)
- Updated polars 0.46 → 0.53, jsonschema 0.28 → 0.46, tiktoken-rs 0.6 → 0.12, aws-sdk-bedrockruntime to latest

## [0.2.2] - 2025-12-17

### Added
- **LLM cost calculation** with tiktoken tokenization for accurate token counting and cost estimation
- **Vector embeddings** (`embedding_async`) - Parallelized, memory-efficient embedding generation
  - Support for OpenAI, Gemini, and AWS Bedrock embedding models
  - Streaming approach for minimal memory footprint
- **Vector similarity functions** for high-performance vector operations:
  - `cosine_similarity` - Measure angle between vectors
  - `dot_product` - Calculate dot product
  - `euclidean_distance` - Calculate straight-line distance
- **Approximate Nearest Neighbor (ANN) search** via HNSW algorithm (`knn_hnsw`)
  - Sub-linear O(log N) search time
  - High recall rates (>95% typical)
  - Scalable to millions of vectors
- Comprehensive repository grading rubric and documentation improvements
- CODE_OF_CONDUCT.md for community guidelines
- SECURITY.md for vulnerability reporting
- Complete API documentation for all Polars expressions
- Architecture diagram in documentation
- Dependency scanning with Dependabot and cargo-audit
- Code coverage reporting in CI pipeline
- `.cargo/audit.toml` configuration for documented security exceptions

### Fixed
- Struct schema inference for Pydantic models with Optional fields
- Mermaid flowchart rendering in documentation

### Security
- **Fixed RUSTSEC-2025-0024**: Updated crossbeam-channel from 0.5.14 to 0.5.15 (double free on Drop)
- **Fixed RUSTSEC-2024-0421**: Updated idna dependency via url crate upgrade (Punycode label issue)
- **Fixed RUSTSEC-2025-0009**: Updated ring from 0.17.11 to 0.17.14 (AES panic issue)
- Updated ureq from 0.11 to 2.x to fix rustls 0.16 vulnerabilities and webpki issues (RUSTSEC-2024-0336, RUSTSEC-2023-0052)
- Updated tokio from 1.37 to 1.48 to fix unsound broadcast channel issue (RUSTSEC-2025-0023)
- Updated reqwest from 0.11 to 0.12 to get newer rustls versions
- Updated futures from 0.3.30 to 0.3.31 to avoid yanked version
- **Documented RUSTSEC-2025-0020** (pyo3 buffer overflow): Cannot be fixed yet as pyo3-polars 0.20.0 requires pyo3 0.23
  - The vulnerability is in PyString::from_object which this codebase doesn't directly use
  - Risk assessed as LOW for our use case
  - Will update when polars 0.52+ stabilizes and pyo3-polars supports pyo3 0.24+
  - Documented in audit.toml with justification
- **3 out of 4 critical vulnerabilities resolved**, 1 documented and accepted with risk assessment

## [0.2.1] - 2025-11-19

### Added
- Link-Time Optimization (LTO) for release builds to improve performance
- Single codegen unit for release builds

### Changed
- Added `llama` namespace for better Python package organization
- Performance optimizations in release configuration

### Fixed
- Additional error handling improvements

## [0.2.0] - 2025-11-12

### Added
- **Taxonomy-based tagging feature** with detailed reasoning and confidence scores
  - Support for hierarchical taxonomies
  - Multi-category classification
  - Confidence scoring for each tag
  - Reasoning explanation for tag assignments
  - Comprehensive documentation in `docs/TAXONOMY_TAGGING.md`
- **Structured outputs with Pydantic integration**
  - Support for Pydantic models as response schemas
  - Automatic validation of LLM responses
  - Polars struct-based output for structured data
  - Parallel validation for batch operations
- Structured output validation tests
- Comprehensive test suite for taxonomy tagging
- Python 3.8 compatibility for structured outputs

### Changed
- Updated documentation for Pydantic structured outputs feature
- Enhanced error handling for API responses

### Fixed
- Clippy `needless_question_mark` lint warning
- Dead code warning in `AnthropicContent` struct
- Python 3.8 compatibility issues in test suite

## [0.1.6] - 2024-03-08

### Added
- Comprehensive test suite for LLM inference interfaces
- Support for all providers in synchronous inference
- GitHub workflow dispatch for manual CI triggers
- Expanded test coverage

### Changed
- Refactored test organization and structure
- Professionalized README and PyPI configuration
- Cleaned up expression declarations

### Fixed
- Synchronous inference now supports all providers (not just async)
- AI inference error handling
- Import paths and Python layer structure

## [0.1.5] - 2024-03-08

### Added
- AWS Bedrock provider support
- Message history support for multi-turn conversations
- PyPI package publishing support

### Changed
- Improved import structure and Python abstraction layer
- Cleaned up module exports

### Fixed
- Polars expression registration issues
- Import errors in tests
- Non-existent feature flags removed

## [0.1.0] - 2024-03-08

### Added
- Initial release of Polar Llama
- OpenAI provider support
- Anthropic (Claude) provider support
- Google Gemini provider support
- Groq provider support
- Parallel asynchronous inference via Polars expressions
- Multi-message conversation support
- PyO3-based Python bindings
- Tokio async runtime integration
- Basic test suite
- MIT License
- Initial documentation

### Features
- `inference()` - Synchronous LLM inference
- `inference_async()` - Parallel asynchronous inference
- `inference_messages()` - Multi-message conversations
- `string_to_message()` - Message formatting helper
- `combine_messages()` - Message array handling
- Provider abstraction via ModelClient trait

[Unreleased]: https://github.com/daviddrummond95/polar_llama/compare/v0.2.2...HEAD
[0.2.2]: https://github.com/daviddrummond95/polar_llama/compare/v0.2.1...v0.2.2
[0.2.1]: https://github.com/daviddrummond95/polar_llama/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/daviddrummond95/polar_llama/compare/v0.1.6...v0.2.0
[0.1.6]: https://github.com/daviddrummond95/polar_llama/compare/v0.1.5...v0.1.6
[0.1.5]: https://github.com/daviddrummond95/polar_llama/compare/v0.1.0...v0.1.5
[0.1.0]: https://github.com/daviddrummond95/polar_llama/releases/tag/v0.1.0
