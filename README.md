### Polar Llama

![Logo](https://raw.githubusercontent.com/daviddrummond95/polar_llama/refs/heads/main/PolarLlama.webp)

#### Overview

Polar Llama is a Python library designed to enhance the efficiency of making parallel inference calls to the ChatGPT API using the Polars dataframe tool. This library enables users to manage multiple API requests simultaneously, significantly speeding up the process compared to serial request handling.

#### Key Features

- **Parallel Inference**: Send multiple inference requests in parallel to the ChatGPT API without waiting for each individual request to complete.
- **Integration with Polars**: Utilizes the Polars dataframe for organizing and handling requests, leveraging its efficient data processing capabilities.
- **Easy to Use**: Simplifies the process of sending queries and retrieving responses from the ChatGPT API through a clean and straightforward interface.
- **Multi-Message Support**: Create and process conversations with multiple messages in context, supporting complex multi-turn interactions.
- **Multiple Provider Support**: Works with OpenAI, Anthropic, Gemini, Groq, and AWS Bedrock models, giving you flexibility in your AI infrastructure.
- **Structured Outputs**: Define response schemas using Pydantic models for type-safe, validated LLM outputs returned as Polars Structs with direct field access.
- **Vector Similarity**: Rust-powered similarity metrics (cosine, dot product, Euclidean distance) for high-performance vector operations.
- **Approximate Nearest Neighbor Search**: HNSW algorithm for fast semantic search and recommendations at scale.
- **Prompt Optimization**: A DSPy-style optimization engine (`Signature`, `Predict`, `BootstrapFewShot`, `InstructionOptimizer`) that tunes instructions and few-shot demos against your labeled data using parallel batched evaluation.
- **Tool Use / MCP**: Dataframe-native tool calling — LLMs emit tool calls as structured output, and `execute_tool_calls` runs every call of every row in parallel against an MCP server or a Python callable. See [docs/TOOL_USE.md](docs/TOOL_USE.md).
- **Playbooks & consistency checks**: Business rules evaluated across *many rows at once*, and — more usefully — self-consistency checks for the rules you can't write down in advance ("three years employed and never took leave"). Polars does the arithmetic; the model finds the edge cases you didn't predict. See [docs/PLAYBOOKS.md](docs/PLAYBOOKS.md) and [docs/CONSISTENCY_PLAYBOOK.md](docs/CONSISTENCY_PLAYBOOK.md).
- **TypeSafe System One**: A native Rust inference layer for the [TypeSafe API](https://docs.typesafe.ai/api) — typed, calibrated yes/no, choice and score questions answered per row, or one Pydantic **contract** applied to every line of a document in a single request, landing as ordinary typed columns with confidence you can threshold on. See [docs/TYPESAFE.md](docs/TYPESAFE.md).
- **Prompt Caching**: Provider-native prompt caching (Anthropic 5m/1h `cache_control`) to share a cached system prefix across rows — pass `cache=True` with a `system_prompt`.

#### Installation

To install Polar Llama, you can use pip:

```bash
pip install polar-llama
```

Alternatively, for development purposes, you can install from source:

```bash
maturin develop
```

#### Example Usage

Here's how you can use Polar Llama to send multiple inference requests in parallel:

```python
import polars as pl
from polar_llama import Provider
import dotenv

dotenv.load_dotenv()

# Example questions
questions = [
    'What is the capital of France?',
    'What is the difference between polars and pandas?'
]

# Creating a dataframe with questions
df = pl.DataFrame({'Questions': questions})

# Using the fluent .llama namespace (recommended)
df = df.with_columns(
    answer=pl.col('Questions').llama.inference_async(
        provider=Provider.OPENAI,
        model='gpt-4o-mini'
    )
)

# Alternative: Using functional API
from polar_llama import string_to_message, inference_async

df = df.with_columns(
    prompt=string_to_message(pl.col("Questions"), message_type='user')
)
df = df.with_columns(
    answer=inference_async(pl.col('prompt'), provider=Provider.OPENAI, model='gpt-4o-mini')
)
```

#### Multi-Message Conversations

Polar Llama supports multi-message conversations, allowing you to maintain context across multiple turns:

```python
import polars as pl
from polar_llama import combine_messages, inference_messages
import dotenv

dotenv.load_dotenv()

# Create a dataframe with system prompts and user questions
df = pl.DataFrame({
    "system_prompt": [
        "You are a helpful assistant.",
        "You are a math expert."
    ],
    "user_question": [
        "What's the weather like today?",
        "Solve x^2 + 5x + 6 = 0"
    ]
})

# Using .llama namespace (recommended)
df = df.with_columns([
    pl.col("system_prompt").llama.to_message(role="system").alias("system_message"),
    pl.col("user_question").llama.to_message(role="user").alias("user_message")
])

# Combine into conversations
df = df.with_columns(
    conversation=combine_messages(pl.col("system_message"), pl.col("user_message"))
)

# Send to model and get responses
df = df.with_columns(
    response=inference_messages(pl.col("conversation"), provider="openai", model="gpt-4")
)
```

#### AWS Bedrock Support

Polar Llama now supports AWS Bedrock models. To use Bedrock, ensure you have AWS credentials configured (via AWS CLI, environment variables, or IAM roles):

```python
import polars as pl
from polar_llama import string_to_message, inference_async
import dotenv

dotenv.load_dotenv()

# Example questions
questions = [
    'What is the capital of France?',
    'Explain quantum computing in simple terms.'
]

# Creating a dataframe with questions
df = pl.DataFrame({'Questions': questions})

# Adding prompts to the dataframe
df = df.with_columns(
    prompt=string_to_message(pl.col("Questions"), message_type='user')
)

# Using AWS Bedrock with Claude model
df = df.with_columns(
    answer=inference_async(pl.col('prompt'), provider='bedrock', model='anthropic.claude-3-haiku-20240307-v1:0')
)
```

#### Structured Outputs with Pydantic

Polar Llama supports structured outputs using Pydantic models. Define your response schema as a Pydantic `BaseModel`, and the LLM will return validated, type-safe data as a Polars Struct:

```python
import polars as pl
from polar_llama import inference_async, Provider
from pydantic import BaseModel

# Define your response schema
class MovieRecommendation(BaseModel):
    title: str
    genre: str
    year: int
    reason: str

# Create a dataframe
df = pl.DataFrame({
    'prompt': ['Recommend a great sci-fi movie from the 2010s']
})

# Get structured output
df = df.with_columns(
    recommendation=inference_async(
        pl.col('prompt'),
        provider=Provider.OPENAI,
        model='gpt-4o-mini',
        response_model=MovieRecommendation
    )
)

# Access struct fields directly!
print(df['recommendation'].struct.field('title')[0])  # "Interstellar"
print(df['recommendation'].struct.field('year')[0])   # 2014
```

**Key Features:**
- **Type Safety**: Responses are validated against your Pydantic schema
- **Direct Field Access**: Use `.struct.field('field_name')` to access individual fields
- **Error Handling**: Built-in `_error`, `_details`, and `_raw` fields for graceful error handling
- **Works Everywhere**: Compatible with `inference_async()`, `inference()`, and `inference_messages()`
- **Multi-Provider**: Works with OpenAI, Anthropic, Groq, Gemini, and Bedrock

**Error Handling:**
```python
# Check for errors in responses
error = df['recommendation'].struct.field('_error')[0]
if error:
    print(f"Error: {error}")
    print(f"Details: {df['recommendation'].struct.field('_details')[0]}")
    print(f"Raw response: {df['recommendation'].struct.field('_raw')[0]}")
```

#### Streaming Responses

`inference_stream()` streams completions token-by-token via an `on_token(row_index, delta)` callback, while still returning a plain DataFrame column — there's no separate iterator to manage. Each row's result is a `Struct{text: Utf8, finished: Boolean}`: `finished=True` means the stream completed normally; `finished=False` covers any interruption (a provider error mid-stream, a dropped connection, Ctrl-C, or an `on_token` callback that raises) and comes with a `RuntimeWarning` plus whatever partial text had arrived — the DataFrame itself is never left in a broken state.

```python
import polars as pl
from polar_llama import inference_stream, Provider

df = pl.DataFrame({
    'prompt': ["Tell me a short joke", "Say hello in French"]
})

df = df.with_columns(
    response=inference_stream(
        pl.col('prompt'),
        provider=Provider.OPENAI,
        model='gpt-4o-mini',
        on_token=lambda i, d: print(d, end=""),
    )
)

print(df.select(
    pl.col('response').struct.field('text'),
    pl.col('response').struct.field('finished'),
))
```

Streaming is text-only: pass `response_model` and you'll get a clear `ValueError` telling you to use `inference_async()` for structured output instead. OpenAI, Groq, and Anthropic stream natively over SSE; Gemini and Bedrock fall back to a single "full response, then done" delta so every provider works with the same API.

#### Resumable Runs (Checkpointing)

Large jobs fail mid-run — rate limits, network blips, provider outages. Pass `checkpoint="path"` and completed rows are persisted to a sidecar store as the run progresses; re-run the same code and it **resumes**, skipping rows that already finished so you pay for roughly one full pass, not two.

```python
from polar_llama import inference_async, Provider

out = df.with_columns(
    answer=inference_async(
        pl.col("prompt"),
        provider=Provider.OPENAI,
        model="gpt-4o-mini",
        checkpoint="run1.ckpt",   # directory of append-only Parquet parts
    )
)
# If this dies at 50%, just run it again — the first 50% is served from
# the checkpoint and only the remaining rows hit the API.
```

The checkpoint key hashes the row input **and** the run config (provider, model, prompt, response schema, endpoint), so changing any of them invalidates old entries automatically — you never get a stale result from a different configuration. Failed rows are stored as failed and retried on resume by default (`Checkpoint(path, retry_failed=False)` to return the stored error instead). Works with structured outputs and tool-use passes. See `docs/design/CHECKPOINTING.md`.

#### Per-Row Usage & Cost

Pass `usage=True` to get token counts, latency, and estimated cost per row as a struct column — useful for invoicing against LLM spend or surfacing usage to users.

```python
out = df.with_columns(
    r=inference_async(pl.col("prompt"), provider=Provider.OPENAI,
                      model="gpt-4o-mini", usage=True)
)
# r is Struct{response, usage: Struct{input_tokens, output_tokens,
#                                     cached_tokens, latency_ms, cost_usd}}
totals = out.select(
    cost=pl.col("r").struct.field("usage").struct.field("cost_usd").sum(),
    in_tok=pl.col("r").struct.field("usage").struct.field("input_tokens").sum(),
)
```

`cost_usd` is computed from a packaged, overridable price table (`polar_llama/pricing.py`); pass `price_table=` (a dict or path) or call `register_model_price(...)` to add/override models. Unknown models yield `cost_usd = null` with a one-time warning. Usage is captured for OpenAI, Groq, Anthropic, Gemini, and Bedrock; the MLX local engine reports tokens + latency with `cost_usd = 0.0`. `usage=False` (the default) is unchanged.

#### Duplicate Collapsing & a Persistent Response Cache

Two related, separately-gated features. `dedupe=True` collapses exact-duplicate rows **within one run** — each unique input is sent once and the result is fanned back out to every row that asked for it, zero extra I/O:

```python
out = df.with_columns(answer=inference_async(pl.col("prompt"), model="gpt-4o-mini", dedupe=True))
```

`response_cache="path"` adds a persistent, **cross-job** store (implies `dedupe=True` automatically): identical requests from a different run, even a different process, reuse a prior result instead of re-calling the backend.

```python
from polar_llama import ResponseCache, DedupeStats

stats = DedupeStats()
out = df.with_columns(
    answer=inference_async(
        pl.col("prompt"),
        model="gpt-4o-mini",
        response_cache=ResponseCache("responses.cache", ttl="24h"),
        dedupe_stats=stats,
    )
)
print(stats.hit_rate, stats.rows_collapsed, stats.calls_made)
```

Only successful results are ever persisted — a failed row always recomputes on the next run. `ttl=` expires entries lazily on read; call `ResponseCache(...).prune()` to compact the store and reclaim expired entries, or `.clear()` to invalidate everything. Not currently supported together with `checkpoint=` or `usage=True` (raises `ValueError` — see `docs/design/RESPONSE_CACHE.md` for the full interop rules and why).

#### Run Manifests (Reproducibility & Audit)

`build_manifest(...)` produces a small, JSON-serializable audit record of *what a run asked for* — provider, model, prompt/schema content (as sha256 hashes, never plaintext, unless you opt in), sampling params — with a `manifest_id` computed deterministically so two runs with identical configuration get the same id even if their timestamps or token counts differ:

```python
from polar_llama import build_manifest, with_manifest_id

out = df.with_columns(response=inference_async(pl.col("prompt"), model="gpt-4o-mini", system_prompt=SYS))

manifest = build_manifest(
    out, symbol="inference_async", provider="openai", model="gpt-4o-mini", system_prompt=SYS,
)
manifest.save("runs/2026-07-14.manifest.json")   # atomic JSON sidecar
out = with_manifest_id(out, manifest)            # attach a manifest_id column to the results
```

Later, verify a manifest and re-run it against fresh row data:

```python
from polar_llama import load_manifest, replay

manifest = load_manifest("runs/2026-07-14.manifest.json")  # raises ManifestIntegrityError if tampered
out = replay(manifest, new_df, "prompt", system_prompt=SYS)  # raises ManifestMismatchError if SYS drifted
```

`config_fingerprint` is computed via the same `request_fingerprint` the #75 checkpoint / #77 dedupe paths use — a manifest's fingerprint is guaranteed identical to the runtime one, not a parallel hash. Manifests store *hashes* of prompts/schemas, not their text (safe to share without leaking prompt IP); pass `build_manifest(..., store_texts=True)` to embed plaintext for fully self-contained replay. Caveat: hosted `inference_async`/`inference_messages` don't forward sampling params (`temperature`, `max_tokens`, ...) today, so `params={}` on a manifest means "provider defaults apply" — the manifest guarantees what the client asked for, not what the provider used. See `docs/RUN_MANIFESTS.md`.

#### Local Inference (Apple Silicon / MLX)

Run inference **on-device** on Apple Silicon instead of a provider API — no keys, no network — via [mlx-lm](https://github.com/ml-explore/mlx-lm). Install the extra (Apple Silicon, Python ≥ 3.10):

```bash
pip install "polar-llama[local]"
```

```python
import polars as pl
import polar_llama  # registers the .llama namespace

df = pl.DataFrame({"ticket": [
    "My laptop won't turn on even when it's plugged in.",
    "I forgot my password and I'm locked out of my account.",
]})

tagged = df.with_columns(
    tag=pl.col("ticket").llama.inference_local(
        model="mlx-community/gemma-3n-E4B-it-lm-4bit",
        system="Classify the ticket as exactly one of: Hardware, Account, Billing.",
        engine="in_process",   # on-device mlx-lm, batched over the column
        max_tokens=16,
        temperature=0.0,
    )
)
```

Two engines are available:

- **`engine="in_process"`** — batched generation directly via `mlx-lm` (`BatchGenerator` / `batch_generate`); nothing to run separately.
- **`engine="server"`** (default) — routes the existing async fan-out to a local OpenAI-compatible endpoint (`mlx_lm.server`) via `OPENAI_BASE_URL`.

**Not on Apple Silicon?** `engine="server"` isn't MLX-specific — it works with any OpenAI-compatible local server, including [llama.cpp](https://github.com/ggml-org/llama.cpp)'s `llama-server` on Linux, Windows, or macOS (CPU or CUDA/ROCm/Vulkan/Metal GPU). See [`docs/local_llamacpp_backend.md`](docs/local_llamacpp_backend.md) for install/run instructions, the sampling-parameter caveat (`max_tokens`/`temperature`/`top_p`/`stop` are set via `llama-server` CLI flags, not forwarded from Python — same limitation as `mlx_lm.server`), and a full feature matrix across `server`+`llama-server` / `server`+`mlx_lm.server` / `in_process`.

**Collapsed prefill.** When rows share a long common prefix (a shared `system` prompt, or few-shot demos), set `POLAR_LLAMA_LOCAL_COLLAPSE=1` to compute that prefix once instead of re-prefilling it per row — a large speedup for prefill-bound workloads, at parity-verified output. Hybrid Gemma 3n / Gemma 4 models need the mlx-lm [#1384](https://github.com/ml-explore/mlx-lm/issues/1384) batched-RoPE fix, which Polar Llama applies automatically. See [`docs/local_mlx_backend.md`](docs/local_mlx_backend.md).

#### Prompt Optimization (DSPy-style)

Polar Llama ships a lightweight prompt optimization engine inspired by [DSPy](https://github.com/stanfordnlp/dspy). Declare your task as a `Signature`, wrap it in a `Predict` module, and let an optimizer tune the prompt against your labeled DataFrame — every candidate is evaluated with one parallel, batched inference call:

```python
import polars as pl
from polar_llama import Predict, Signature, BootstrapFewShot, InstructionOptimizer, evaluate

# 1. Declare the task
module = Predict(
    Signature("question -> answer", instructions="Answer concisely."),
    provider="openai",
    model="gpt-4o-mini",
)

# 2. Labeled training data
trainset = pl.DataFrame({
    "question": ["What is 2+2?", "Capital of France?", "Largest planet?"],
    "answer": ["4", "Paris", "Jupiter"],
})

# 3. A metric: (gold row, prediction) -> bool | float
def exact_match(example, prediction):
    return example["answer"].strip().lower() == (prediction["answer"] or "").strip().lower()

# 4a. Bootstrap few-shot demos from rows the model already gets right
compiled = BootstrapFewShot(metric=exact_match, max_demos=4).compile(module, trainset)

# 4b. Or search for better instructions (COPRO-style)
optimizer = InstructionOptimizer(metric=exact_match, n_candidates=4)
compiled = optimizer.compile(module, trainset)
print(optimizer.history)  # [(instructions, score), ...]

# 5. Run the optimized module on new data — outputs land in pred_* columns
result = compiled(pl.DataFrame({"question": ["What is 3+3?"]}))
print(result["pred_answer"])

# Score any module against a labeled set
print(evaluate(compiled, trainset, exact_match).score)
```

Output fields can be typed and described for stronger structured outputs:

```python
from polar_llama import OutputField

sig = Signature(
    "review -> sentiment, confidence",
    instructions="Classify the sentiment of the review.",
    outputs={
        "sentiment": OutputField(desc="one of: positive, negative, neutral"),
        "confidence": OutputField(desc="confidence from 0.0 to 1.0", dtype=float),
    },
)
```

**Optimize entirely on-device.** Drive the optimizer with a local model instead of a provider by passing an `inference_fn` — no API keys, no network:

```python
from polar_llama.local import make_local_inference_fn

# Backs Predict/BootstrapFewShot/InstructionOptimizer with on-device gemma-3n
# (mlx-lm). Collapsed prefill (on by default) shares the system+demos prefix
# across rows — the dominant speedup during tuning, where few-shot demos make
# that prefix ~80% of every prompt (~2.8x on a full tuning schedule, at
# identical output). Requires the [local] extra.
fn = make_local_inference_fn("mlx-community/gemma-3n-E4B-it-lm-4bit")
module = Predict(Signature("note -> category", instructions="Classify the note."), inference_fn=fn)
compiled = InstructionOptimizer(metric=exact_match).compile(module, trainset)
```

#### Human-in-the-loop Review Loop

Turn a sample of an LLM's output into a spreadsheet for a human reviewer, then feed their corrections back into the optimizer: code → export a sample for review → a human corrects it → import the corrections (with agreement stats) → retune. See [`docs/HITL_WORKFLOW.md`](docs/HITL_WORKFLOW.md) for the full loop.

```python
from polar_llama import export_review_sample, import_corrections, retune_from_corrections

# 1. Export a stratified sample for review, oversampling low-confidence rows
sample = export_review_sample(
    coded_df,               # has a "code" column and, optionally, a "confidence" column
    n=100,
    strata="code",
    confidence_column="confidence",
    oversample_low_confidence=4.0,
    path="review_batch.csv",   # a human opens this, fills in corrected_code
)

# 2. Join the (reviewed) corrections back on and score human/LLM agreement
result = import_corrections(coded_df, reviewed_df, code_column="code")
print(result.kappa, result.agreement_rate, result.n_reviewed, result.n_changed)

# 3. Mine few-shot demos from the corrections and compile an improved module
tuned = retune_from_corrections(
    result.df.filter(pl.col("was_reviewed")),
    Signature("text -> code"),
    provider="openai",
    model="gpt-4o-mini",
)
```

`export_review_sample` writes CSV out of the box; XLSX export needs the optional `[excel]` extra (`pip install "polar-llama[excel]"`), never a required dependency. `import_corrections` reuses `cohens_kappa` (see [`docs/RELIABILITY_METRICS.md`](docs/RELIABILITY_METRICS.md)); `retune_from_corrections` reuses `BootstrapFewShot` above — no logic is duplicated between the two features.

#### Vector Embeddings

Polar Llama provides parallelized, memory-efficient embedding generation for converting text into vector representations. This is useful for semantic search, clustering, and similarity analysis:

```python
import polars as pl
from polar_llama import embedding_async, Provider

# Create a dataframe with text
df = pl.DataFrame({
    'text': [
        'Machine learning is transforming AI',
        'Natural language processing enables text understanding',
        'Deep learning uses neural networks'
    ]
})

# Generate embeddings in parallel
df = df.with_columns(
    embeddings=embedding_async(
        pl.col('text'),
        provider=Provider.OPENAI,
        model='text-embedding-3-small'  # 1536 dimensions
    )
)

# Access embedding dimensions
df = df.with_columns(
    dimensions=pl.col('embeddings').list.len()
)

print(df['dimensions'][0])  # 1536
```

**Using the .llama namespace:**
```python
df = df.with_columns(
    embeddings=pl.col('text').llama.embedding(
        provider=Provider.OPENAI,
        model='text-embedding-3-small'
    )
)
```

**Supported Models:**
- **OpenAI**: `text-embedding-3-small` (1536 dims), `text-embedding-3-large` (3072 dims)
- **Gemini**: `text-embedding-004` (768 dims)
- **AWS Bedrock**: `amazon.titan-embed-text-v1` (1536 dims)

**Key Features:**
- **Parallel Processing**: All embeddings are generated concurrently using async/await for maximum performance
- **Memory Efficient**: Streaming approach minimizes memory footprint
- **Type Safety**: Returns `List[Float64]` for seamless integration with Polars operations
- **Null Handling**: Gracefully handles null values in input

**Practical Example - Semantic Similarity:**
```python
from polar_llama import cosine_similarity

# Generate embeddings
df = df.with_columns(
    embeddings=embedding_async(pl.col('text'), provider=Provider.OPENAI)
)

# Calculate similarity between first document and all others
query_emb = df["embeddings"][0]
df = df.with_columns(
    similarity=cosine_similarity(
        pl.lit([query_emb]),
        pl.col('embeddings')
    )
)

print(df.select(['text', 'similarity']))
```

#### Local (Offline) Embeddings

The fully-offline counterpart of `embedding_async` (issue #83): runs an embedding model **in this process** via `mlx_embeddings`, instead of calling a hosted provider API. No API key, no network call. Output dtype is `List[Float64]`, identical to `embedding_async` — a drop-in input to `cosine_similarity`, `knn_hnsw`, `HnswIndex`, and `cluster_embeddings`. See [`docs/LOCAL_EMBEDDINGS.md`](docs/LOCAL_EMBEDDINGS.md) for the full guide (download/caching, throughput numbers, the CI-safe fake-engine seam).

```python
from polar_llama import embedding_local

# Requires the [local] extra on Apple Silicon: pip install "polar-llama[local]"
# Downloads the default model (~65MB) once, cached under ~/.cache/huggingface;
# every call after that is fully offline.
df = df.with_columns(
    embeddings=pl.col('text').llama.embedding_local()
)
```

#### Vector Similarity and Approximate Nearest Neighbor Search

Polar Llama includes high-performance Rust-powered vector similarity operations and approximate nearest neighbor (ANN) search capabilities for semantic search, recommendations, and clustering:

**Similarity Metrics:**
```python
from polar_llama import cosine_similarity, dot_product, euclidean_distance

df = pl.DataFrame({
    "vec1": [[1.0, 0.0, 0.0], [1.0, 2.0, 3.0]],
    "vec2": [[1.0, 0.0, 0.0], [2.0, 4.0, 6.0]]
})

# Calculate different similarity metrics
df = df.with_columns([
    cosine_similarity(pl.col("vec1"), pl.col("vec2")).alias("cosine_sim"),
    dot_product(pl.col("vec1"), pl.col("vec2")).alias("dot_prod"),
    euclidean_distance(pl.col("vec1"), pl.col("vec2")).alias("distance")
])

# Using .llama namespace (alternative)
df = df.with_columns(
    cosine_sim=pl.col("vec1").llama.cosine_similarity(pl.col("vec2"))
)
```

**HNSW Approximate Nearest Neighbor Search:**

Fast k-nearest neighbor search using the HNSW (Hierarchical Navigable Small World) algorithm:

```python
from polar_llama import knn_hnsw, embedding_async, Provider

# Create corpus of documents
corpus = pl.DataFrame({
    "doc": ["AI research", "cooking tips", "machine learning", "recipes"]
}).with_columns(
    embedding=embedding_async(pl.col("doc"), provider=Provider.OPENAI)
)

# Create query
query = pl.DataFrame({
    "query": ["artificial intelligence"]
}).with_columns(
    query_emb=embedding_async(pl.col("query"), provider=Provider.OPENAI),
    corpus_emb=pl.lit([corpus["embedding"].to_list()])
).with_columns(
    neighbors=knn_hnsw(
        pl.col("query_emb"),
        pl.col("corpus_emb").list.first(),
        k=2  # Find 2 nearest neighbors
    )
)

# Get nearest neighbor documents
indices = query["neighbors"][0]
print(corpus[indices]["doc"])  # ['AI research', 'machine learning']
```

**Metadata-Enhanced Search:**

Combine taxonomy filtering with vector search for precise, context-aware results:

```python
from polar_llama import tag_taxonomy, embedding_async, knn_hnsw, Provider

# Define taxonomy for metadata
taxonomy = {
    "category": {
        "description": "Content category",
        "values": {
            "technology": "Tech and programming content",
            "cooking": "Food and recipe content"
        }
    }
}

# Use with_columns() for parallel execution
corpus = corpus.with_columns([
    tag_taxonomy(pl.col("content"), taxonomy, provider=Provider.ANTHROPIC)
        .alias("tags"),
    embedding_async(pl.col("content"), provider=Provider.OPENAI)
        .alias("embedding")
])

# Note: with_columns() runs operations in parallel
# Speedup depends on operation durations (best when similar duration)
# Each operation also internally parallelizes all API calls across all documents

# Extract category
corpus = corpus.with_columns(
    category=pl.col("tags").struct.field("category").struct.field("value")
)

# STEP 1: Filter by metadata (category = technology)
tech_docs = corpus.filter(pl.col("category") == "technology")

# STEP 2: Semantic search within filtered subset
query = query.with_columns(
    tech_corpus=pl.lit([tech_docs["embedding"].to_list()])
).with_columns(
    tech_neighbors=knn_hnsw(pl.col("query_emb"), pl.col("tech_corpus").list.first(), k=3)
)

# Results are guaranteed to be technology-related AND semantically relevant
```

**Key Features:**
- **Blazing Fast**: Rust-powered similarity calculations with zero-copy operations
- **Multiple Metrics**: Cosine similarity, dot product, and Euclidean distance
- **HNSW Algorithm**: State-of-the-art approximate nearest neighbor search
- **Parallel Processing**: Vectorized operations for maximum performance
- **Metadata Filtering**: Combine structured and semantic search

See `docs/VECTOR_SIMILARITY_AND_ANN.md` for complete documentation and advanced examples.

#### Persistent HNSW Index

`knn_hnsw` above rebuilds its graph on every call — fine for a one-off query, wasteful if you query the same corpus repeatedly or want to grow it over time. `HnswIndex` (issue #82) is a persistent, incrementally updatable index: build it once, `.add()`/`.remove()` points into it (queryable immediately, via a brute-force staging buffer + soft-delete tombstones on top of `instant-distance`'s otherwise-immutable HNSW graph, auto-compacted periodically), and `.save()`/`.load()` it to/from disk.

```python
from polar_llama import HnswIndex

index = HnswIndex.build(corpus, id_col="doc_id", embedding_col="embedding")
index.add(new_docs, id_col="doc_id", embedding_col="embedding")  # incremental, queryable immediately
index.remove(["doc-2"])                                          # soft-delete

results = index.query(queries, embedding_col="embedding", k=5)   # query_id | neighbor_id | distance | rank
index.save("my_index.bin")
index = HnswIndex.load("my_index.bin")
```

See `docs/VECTOR_SIMILARITY_AND_ANN.md#persistent-hnsw-index-hnswindex` for the full API (`.query_one`, `.knn()` expression bridge, compaction policy, performance characteristics).

#### Codebook Induction

Build a qualitative-coding codebook from a text column — embed, cluster, and let the LLM name and define each cluster — then apply it back as multi-label tags. No external clustering dependency: `cluster_embeddings` is a hand-rolled k-means (k-means++ / Lloyd's, in Rust) with automatic `k` selection via sampled silhouette.

```python
from polar_llama import induce_codebook, apply_codebook, Provider

tickets = pl.DataFrame({"text": [...]})  # a text column to discover themes in

# 1. Embed + cluster + name each cluster with an LLM (auto-picks k)
result = induce_codebook(
    tickets, "text", provider=Provider.OPENAI, model="gpt-4o-mini"
)
print(result.codebook)              # Codebook(5 codes: billing_issue, login_failure, ...)
print(result.df["cluster_id"])      # same rows as `tickets`, plus cluster_id/cluster_distance

# 2. Apply the induced codebook back as multi-label tags — composes directly
#    onto result.df, no join or explode needed
coded = result.df.with_columns(
    labels=apply_codebook(pl.col("text"), result.codebook, provider=Provider.OPENAI)
)
# labels: List[Struct{code, applies, confidence, evidence}], one entry per code
```

**Key Features:**
- **Zero clustering dependencies**: k-means++ / Lloyd's, a splitmix64 PRNG, and sampled silhouette are hand-rolled in Rust (`src/kmeans.rs`) — no `scikit-learn`/`numpy`/`umap`.
- **DataFrame-shaped**: `induce_codebook` is DataFrame-in/DataFrame-out (same row count and order back, plus `cluster_id`/`cluster_distance`); `apply_codebook` is expression-in/expression-out.
- **Multi-label by design**: `apply_codebook` evaluates every candidate code per document in one structured-output call — several codes can apply to the same document.
- **Strict-mode safe**: the apply model is a fixed-length `List[{code, applies, confidence, evidence}]`, never a `Dict` keyed by code (the same lesson as taxonomy tagging's `thinking` field).
- **Bridges to `tag_taxonomy`**: `codebook_to_taxonomy(result.codebook)` turns an induced codebook into a taxonomy for single-label (mutually exclusive) classification instead.

See `docs/CODEBOOK_INDUCTION.md` for the full walkthrough.

#### Survey Data-Quality Flags

Score every respondent in a survey DataFrame for common data-quality problems — straightlining, gibberish open-ends, duplicate answers, response-length outliers, speeding — and get a per-flag summary back, without ever dropping a row. Every score is graded (`Float64` in `[0, 1]`, higher = more suspicious); a null score (not enough signal) always resolves to `flag = False`. The heuristic tier is zero-API; an opt-in embedding/LLM tier adds cross-respondent near-duplicate detection and a stylistic likely-AI-generated-text flag.

```python
from polar_llama import QualityConfig, quality_report

config = QualityConfig(
    id_column="respondent_id",
    grid_columns=["g1", "g2", "g3", "g4", "g5"],   # Likert grid
    text_columns=["oe1", "oe2"],                    # open-ends
    duration_column="duration_s",                   # seconds
)
result = quality_report(survey_df, config)

result.df.height == survey_df.height   # True — quality_report never drops a row
result.df.select("respondent_id", "quality").unnest("quality")
result.summary   # one row per flag: flag, n_scored, n_flagged, rate, threshold, mean_score
```

**Key Features:**
- **Never drops a row**: every flag is a threshold on a graded `[0, 1]` score; missing data scores null, which always resolves to `flag = False` — never dropped, never force-flagged.
- **Zero-API heuristic tier**: `straightlining_score`, `gibberish_score`, and `duplicate_answer_score` are Rust plugin expressions (`src/quality.rs`); `response_length_score` (robust z-score of length) and `speeder_score` (percentile- or median-fraction-based) are pure Polars. All five also work standalone and via `.llama` (`pl.col("g1").llama.straightlining_score([...])`).
- **Opt-in embedding/LLM tier**: `QualityConfig(llm_tier=True)` adds `near_duplicate` (cross-respondent, via `embedding_async` + `knn_hnsw` + `cosine_similarity`) and `likely_ai` (via `inference_messages(..., response_model=...)`, also exposed standalone as `ai_likelihood`) — off by default, zero network calls unless requested.
- **Flag, not verdict**: `likely_ai` carries a mandatory limitation warning — AI-text detectors are unreliable and biased against non-native English speakers; treat it as a review-prioritization signal only, never as sole grounds for exclusion or panelist sanction.
- **Documented, tunable thresholds**: every default threshold is a subjective convention, not a validated cutoff — see `docs/QUALITY_FLAGS.md` section 7.

See `docs/QUALITY_FLAGS.md` for the full formulas, every default threshold, and the AI-detection limitation discussion.

#### TypeSafe System One (Typed, Calibrated Decisions)

TypeSafe is not a chat-completions provider: instead of a prompt and free text back, one request carries a `state` plus a map of typed **questions**, and returns one typed **answer** each — a yes/no probability (`noul`), a pick from a closed set with a full probability distribution (`choice`), or a probability-weighted rating over ordered levels (`score`). Because the answers are typed and calibrated rather than parsed out of prose, they land in the dataframe as ordinary numeric and string columns.

Every question rides in the *same* request per row, which is TypeSafe's own "speculative fan-out" guidance — ask everything you might want up front, and let your code decide afterwards what to read.

```python
import polars as pl
from polar_llama import typesafe_eval, noul, choice, score

df = pl.DataFrame({"message": [
    "Help! My payouts have been failing for 3 days.",
    "Hi, just wondering what your enterprise pricing looks like.",
]})

out = df.with_columns(
    ts=typesafe_eval(
        pl.col("message"),
        questions={
            "is_urgent": noul("Does this convey urgency?",
                              true="Explicitly time-sensitive",
                              false="No urgency expressed"),
            "department": choice("Which team should handle this?", {
                "billing": "Payments, invoicing, refunds",
                "technical": "Bugs, outages, integrations",
                "sales": "Pricing, upgrades, new accounts",
            }),
            "frustration": score("How frustrated is the customer?",
                                 ["Calm", "Frustrated", "Very angry"]),
        },
    )
).unnest("ts")

# message                 is_urgent  department  department_confidence  frustration  _error
# "Help! My payouts…"     0.95       billing     0.79                   1.04         null
# "Hi, just wondering…"   0.06       sales       1.0                    0.0          null

# Confidence is the lever: act automatically, or route to a human.
auto  = out.filter(pl.col("department_confidence") > 0.9)
human = out.filter(pl.col("department_confidence") < 0.5)
```

- **Schema without a request**: output dtypes are derived from the questions, so `.collect_schema()` on a LazyFrame resolves the full shape without spending a token.
- **Per-row failure isolation**: `_error` is always present and null on success; an all-null state is never sent at all. `429`/`529`/5xx retry with exponential backoff and jitter, while `401`/`422` fail fast rather than billing for a retry that cannot help.
- **Opt-in detail**: `probabilities=True` adds the full distribution per question; `usage=True` adds the resolved model version plus token and latency accounting.

Set `TYPESAFE_API_KEY` (and optionally `TYPESAFE_BASE_URL`). See [docs/TYPESAFE.md](docs/TYPESAFE.md) for the full output schema, structured state, and configuration.

Define the struct you want as a Pydantic **contract** and apply it to every line of a document — all of a row's lines are answered in one request, so the model sees the surrounding lines as context and you don't pay for the document N times:

```python
from typing import Literal
from pydantic import BaseModel, Field
from polar_llama import typesafe_eval_each, score_field

class ClauseFeatures(BaseModel):
    is_payment: bool = Field(description="Does this clause create a payment obligation?")
    category: Literal["fees", "term", "liability", "other"] = Field(description="What kind of clause?")
    severity: float = score_field("How onerous for the Customer?", ["Benign", "Notable", "Onerous"])

clauses = (
    df.with_columns(clause=pl.col("contract_text").str.split("\n"))
      .with_columns(f=typesafe_eval_each(pl.col("clause"), contract=ClauseFeatures))
      .explode("f").unnest("f")
)

# doc  line_id  line                                is_payment  category  severity
# msa  0        "Definitions. 'Services' means…"    0.03        other     0.00
# msa  1        "Fees. Customer shall pay all…"     0.99        fees      0.12
# msa  4        "Limitation of Liability. In no…"   0.04        liability 1.52

clauses.filter(pl.col("is_payment") > 0.8)
```

Measured against the live API on an 8-clause contract, batching a document's lines into one request is **3.2x fewer input tokens and 8x fewer round trips** than a request per line — and it is the only version that keeps context. Documents longer than `max_questions` are chunked automatically and fired concurrently, with `line_id` staying global.

Field types pick the question type: `bool` → Noul (a probability, which you threshold), `Literal[...]`/`Enum` → Choice, numeric + `score_field(levels=...)` → Score. A plain `str` field is rejected with an explanation — TypeSafe answers are typed over a closed set and there is no free-text primitive.

#### Playbooks and Self-Consistency Checks

`typesafe_eval` judges one row and `typesafe_eval_each` judges each segment of a document. A **playbook** judges a *group of rows together* — the shape you need for a rule no single row can violate.

The most useful form needs no rules at all. Some problems can't be written down in advance — *"does it make sense that someone has worked here three years and never taken leave?"* Nobody writes that rule, and if you did you'd then need one for the intern on the top pay band, the rep with record commission and no customer meetings, the employee on leave who closed a full year of tickets. The list has no end. So don't write rules — ask whether the record hangs together:

```python
from polar_llama import self_consistency, playbook_eval

flags = playbook_eval(df, self_consistency("employee record"), by="employee_id")
flags.sort("review_priority", descending=True).head(20)   # a triage queue
```

Three things measured rather than assumed, all in [docs/CONSISTENCY_PLAYBOOK.md](docs/CONSISTENCY_PLAYBOOK.md):

- **Use a score, not a yes/no.** "Is this coherent?" caught 2 of 4 planted issues; "how strongly does this warrant review?" caught 4 of 4 with no false positives (clean 0.07–0.18, planted 0.98–1.82).
- **Record width dominates.** The same people with 5 fields ranked *clean above planted*; with 12 fields the ranking was correct. Consistency is a relationship between fields — give it few fields and there's little to be wrong.
- **Column names are part of the prompt.** Renaming `pto_days` to `pto_days_TAKEN_last_12_months` roughly doubled the separation.

For rules you *can* state, a playbook takes them explicitly:

```python
from polar_llama import playbook, rule, playbook_eval

policy = playbook(
    "expense_policy",
    within_limit = rule("No employee may claim more than $5,000 in total."),
    receipts     = rule("Every claim over $75 must reference a receipt."),
)

verdicts = playbook_eval(
    df, policy,
    by="employee",                                     # one verdict per employee
    records=["date", "amount_usd", "category"],        # columns -> a record per row
    compute={"total_usd": pl.col("amount_usd").sum()}, # Polars does the arithmetic
    context={"limit_usd": 5000},
)

verdicts.filter(pl.col("within_limit") < 0.5)          # the violations
```

```
# employee  total_usd  row_count  within_limit  receipts  _error
# alice     5400       3          0.04          0.88      null     <- flagged
# bob       950        3          0.98          0.91      null
# frank     6100       3          0.03          0.79      null     <- flagged
```

Each rule answers the **probability the rows adhere**, so violations are `< 0.5`. Every rule rides in the same request per group, and groups go out in parallel with the usual retry and per-group error isolation.

Two things worth knowing, both measured rather than assumed:

- **Signal dilutes with group size.** On a planted violation, clean and violating data separated cleanly at 6 rows (0.33 vs 0.98), were marginal at 20 (0.40 vs 0.49), and were *indistinguishable* by 60 (0.20 vs 0.39). This is not specific to arithmetic — a semantic rule held at 11 rows and was missed at 59. `playbook_eval` warns above `max_rows_per_group` (default 25).
- **Let Polars do the arithmetic.** Judging a pre-computed aggregate instead of raw rows was exact at every size tested up to 1,000 rows summarised (0.03 vs 0.98), because the state stays small however many rows it covers. That is what `compute=` is for; reserve the model for judgment Polars cannot express.

#### Tool Use and MCP

Polar Llama supports tool calling without hiding an agent loop inside your rows. The loop is unrolled into the dataframe: the LLM *emits* tool calls as structured output, `execute_tool_calls` runs every call of every row concurrently (against an MCP server or any Python callable), and a second inference pass synthesizes the results — every intermediate step is an ordinary column.

```python
from polar_llama import (
    mcp_tools, tools_to_response_model, execute_tool_calls,
    tool_results_to_message, combine_messages, inference_messages, Provider,
)

tools = mcp_tools("http://localhost:8811/mcp")     # tools/list introspection
ToolCalls = tools_to_response_model(tools)          # emission schema

df = (
    df
    # 1. Emit: the LLM parameterizes N calls per row (structured output)
    .with_columns(calls=pl.col("meal").llama.inference_async(
        provider=Provider.OPENAI, model="gpt-4o-mini", response_model=ToolCalls))
    # 2. Execute: all calls across all rows, in parallel, errors as data
    .with_columns(results=execute_tool_calls(
        pl.col("calls"), transport="http://localhost:8811/mcp", tools=tools))
    # 3. Synthesize: fold results back through a second inference pass
    .with_columns(answer=inference_messages(
        combine_messages(
            pl.col("meal").llama.to_message(role="user"),
            tool_results_to_message(pl.col("results")),
        ),
        provider=Provider.OPENAI, model="gpt-4o-mini"))
)
```

See [docs/TOOL_USE.md](docs/TOOL_USE.md) for the full guide and [docs/design/MCP_TOOL_INTEGRATION.md](docs/design/MCP_TOOL_INTEGRATION.md) for the design rationale.

#### Benefits

- **Speed**: Processes multiple queries in parallel, drastically reducing the time required for bulk query handling.
- **Scalability**: Scales efficiently with the increase in number of queries, ideal for high-demand applications.
- **Ease of Integration**: Integrates seamlessly into existing Python projects that utilize Polars, making it easy to add parallel processing capabilities.
- **Context Preservation**: Maintain conversation context with multi-message support for more natural interactions.
- **Provider Flexibility**: Choose from multiple LLM providers based on your needs and access.
- **Type Safety**: Get validated, structured outputs using Pydantic schemas for reliable data extraction.

#### Testing

Polar Llama includes a comprehensive test suite that validates parallel execution, provider support, and core functionality.

**Setup:**

1. Copy `.env.example` to `.env` and add your API keys:
   ```bash
   cp .env.example .env
   # Edit .env and add your provider API keys
   ```

2. Install test dependencies:
   ```bash
   pip install -r tests/requirements.txt
   ```

**Run Python tests:**
```bash
pytest tests/ -v
```

**Run Rust tests:**
```bash
cargo test --test model_client_tests -- --nocapture
```

Tests automatically detect configured providers and only run tests for those with valid API keys. See [tests/README.md](tests/README.md) for detailed testing documentation.

#### Contributing

We welcome contributions to Polar Llama! If you're interested in improving the library or adding new features, please feel free to fork the repository and submit a pull request.

#### License

Polar Llama is released under the MIT license. For more details, see the LICENSE file in the repository.

#### Roadmap

- [x] **Multi-Message Support**: Support for multi-message conversations to maintain context.
- [x] **Multiple Provider Support**: Support for different LLM providers (OpenAI, Anthropic, Gemini, Groq, AWS Bedrock).
- [x] **Structured Data Outputs**: Add support for structured data outputs using Pydantic models with type validation and Polars Struct returns.
- [x] **Tool Use / MCP**: Batch-parallel tool-call emission and execution with MCP support (see [docs/TOOL_USE.md](docs/TOOL_USE.md)).
- [x] **Streaming Responses**: Support for streaming responses from LLM providers (`inference_stream()`, see above).
