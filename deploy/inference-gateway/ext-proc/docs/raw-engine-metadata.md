# Raw Engine Metadata for External EPP

This note scopes the raw-engine metadata discussion to the external EPP path in
this package. It does not claim that engine-native HTTP metadata can replace the
full Dynamo `ModelDeploymentCard` or `ModelRuntimeConfig` contract.

For raw aggregated workers behind InferencePool, the EPP needs only a small
MDC/MRC-like subset:

- enough model identity to initialize the KV router and optional served indexer
- KV block geometry so request-side block hashes match worker-side KV events
- worker endpoint and stable worker attribution
- DP rank topology for per-rank KV-event listeners
- KV-event connectivity for precise prefix routing
- capacity hints for logging and future load/reservation integration

Full frontend, planner, parser, media, LoRA, topology, and disaggregated-serving
metadata remain outside this first raw aggregated-worker scope.

## SGLang Example Payloads

These are trimmed representative payloads from SGLang's HTTP surfaces. The
actual `/server_info` response contains many more `ServerArgs` fields because
SGLang returns `dataclasses.asdict(server_args)` plus scheduler information,
internal states, the server version, and the structured `kv_events` descriptor.

`GET /v1/models`:

```json
{
  "object": "list",
  "data": [
    {
      "id": "Qwen/Qwen3-0.6B",
      "object": "model",
      "created": 1760000000,
      "owned_by": "sglang",
      "root": "Qwen/Qwen3-0.6B",
      "parent": null,
      "max_model_len": 32768
    }
  ]
}
```

`GET /model_info`:

```json
{
  "model_path": "Qwen/Qwen3-0.6B",
  "tokenizer_path": "Qwen/Qwen3-0.6B",
  "is_generation": true,
  "preferred_sampling_params": null,
  "weight_version": null,
  "has_image_understanding": false,
  "has_audio_understanding": false,
  "model_type": "qwen3",
  "architectures": ["Qwen3ForCausalLM"]
}
```

`GET /server_info` with KV-event publishing enabled:

```json
{
  "model_path": "Qwen/Qwen3-0.6B",
  "served_model_name": "Qwen/Qwen3-0.6B",
  "tokenizer_path": "Qwen/Qwen3-0.6B",
  "page_size": 16,
  "dp_size": 4,
  "enable_dp_attention": true,
  "nnodes": 2,
  "node_rank": 1,
  "max_running_requests": 16,
  "max_prefill_tokens": 1024,
  "max_total_num_tokens": 2048,
  "internal_states": [],
  "version": "0.5.0",
  "kv_events": {
    "publisher": "zmq",
    "endpoint_host": "*",
    "endpoint_port_base": 5557,
    "topic": "",
    "block_size": 16,
    "dp_size": 4
  }
}
```

Important details for the EPP adapter:

- `endpoint_host` can be a ZMQ wildcard such as `*`; the subscriber should use
  the Pod IP or worker URL host when dialing.
- Per-rank ZMQ endpoints are `tcp://<pod-ip>:<endpoint_port_base + dp_rank>`.
- `kv_events` is `null` when KV-event publishing is disabled or the configured
  endpoint is not a routable TCP endpoint.
- Capacity fields can be absent depending on SGLang version and configuration,
  so the EPP treats them as hints rather than required routing inputs.

## Required vs Optional Fields

For precise KV-aware routing, the external EPP must know the fields that affect
hash compatibility and KV-event attribution before it initializes the router.
The hard requirements are:

- `model_name` or a configured model name fallback, used to bind router/indexer
  state to the served model
- `block_size` / `page_size`, used to compute request-side KV block hashes
- worker endpoint identity, from Kubernetes Pod IP plus InferencePool target port
- worker id, derived from `hash_pod_name(pod_name)`
- KV-event endpoint topology: `kv_events.endpoint_port_base` and
  `kv_events.dp_size`

The following fields are useful but optional in this PR:

- `kv_events.topic`, because an empty topic is a valid subscribe-all filter
- `max_total_num_tokens`, `max_running_requests`, and `max_prefill_tokens`
- derived capacity values such as `total_kv_blocks`, `max_num_seqs`, and
  `max_num_batched_tokens`
- `context_length`

Optional capacity fields are logged and kept available for future
slot/reservation integration, but current external EPP routing correctness does
not depend on them.

## Failure Behavior

The adapter must fail closed for fields that would make precise routing silently
wrong:

- `DYN_EPP_ENGINE=sglang`: wait and retry until a ready SGLang Pod exposes a
  usable `block_size` / `page_size`. Do not fall back to the default block size,
  because the `KvRouter` block size is fixed for the process lifetime.
- `DYN_EPP_ENGINE=auto`: probe SGLang first. If no usable SGLang metadata is
  found, preserve the vLLM/env fallback path so existing raw vLLM deployments do
  not block on SGLang-specific metadata.
- `kv_events=null` or invalid `kv_events`: register no precise KV listener for
  that pod. The deployment can still route load-aware or use predict-on-route
  bookkeeping, but it should not claim precise KV-event-backed routing for that
  pod.
- missing optional capacity fields: continue with routing and log only the
  fields that were observed.

## Known Gap

The router's block size is chosen during EPP startup and is not updated after
the `KvRouter` is constructed. If startup guessed `16` while SGLang workers use
`--page-size 64`, request-side hashes and worker KV-event hashes would be
computed at different granularities. The result is silent precise-routing
degradation: listeners may connect and receive events, but overlap scores will
not match the request blocks.

For that reason, explicit SGLang mode waits for real metadata before
initializing the router. Future work could make block-size/model bootstrap
fully reconciled, but this PR intentionally keeps the router immutable and
prevents the bad initial value instead.

## EPP-Required Subset

| EPP metadata need | Why EPP needs it | SGLang raw source | vLLM raw source today | Current handling / gap |
| --- | --- | --- | --- | --- |
| Model name / served model id | Initializes `KvRouter`; required by remote/served indexer modes. | Prefer `/v1/models`; fall back to `DYN_MODEL_NAME`; then `/model_info.model_path`. | `/v1/models` exposes served model ids. `DYN_MODEL_NAME` is still a useful override when the request-plane model name differs. | Covered for both engines. |
| KV block size / page size | Request-side token blocks must hash with the same block size as worker KV events. | Prefer `/server_info.kv_events.block_size`; fall back to `/server_info.page_size`. | Dynamo-wrapped vLLM reads `kv_event_block_size` from `VllmConfig`; raw vLLM external mode currently relies on `DYN_KV_CACHE_BLOCK_SIZE` or startup config. Some vLLM metrics expose cache config, but this is not yet a stable normalized runtime-info contract. | Covered for SGLang. vLLM works with the fixed env/static contract from the vLLM external EPP path, but full self-discovery should use a stable upstream runtime-info endpoint or thin adapter. |
| Worker inference endpoint | EPP forwards through Gateway-selected endpoints and uses pod IP for direct KV-event listeners. | Kubernetes Pod IP plus InferencePool target port. | Kubernetes Pod IP plus InferencePool target port. | Covered outside engine metadata. |
| Worker id for KV attribution | KV events must be stamped with the same worker id the router selects. | `hash_pod_name(pod_name)`. | `hash_pod_name(pod_name)`. | Covered outside engine metadata. |
| DP size and local DP rank range | Per-rank KV indexers/listeners need the rank range owned by this pod. | `/server_info.kv_events.dp_size`, plus `/server_info` fields such as `dp_size`, `enable_dp_attention`, `nnodes`, and `node_rank`. | Dynamo-wrapped vLLM derives this from `VllmConfig.parallel_config`. Raw vLLM external mode does not currently probe a stable runtime endpoint for the resolved DP range. | Covered for SGLang aggregated workers. vLLM fixed-port mode is preserved; full DP self-discovery needs a stable runtime-info endpoint or adapter. |
| KV-event publisher endpoint and topic | Precise KV-aware routing needs direct access to each worker/rank event stream. | `/server_info.kv_events.endpoint_port_base` plus `dp_rank`; topic from `/server_info.kv_events.topic` or `DYN_EPP_KV_EVENT_TOPIC`. | The vLLM external EPP path uses the fixed-port contract `DYN_EPP_KV_EVENT_PORT` and optional `DYN_EPP_KV_EVENT_TOPIC`. | Covered for SGLang with per-rank endpoints. vLLM behavior remains fixed-port to avoid changing the existing contract. |
| Capacity hints | Useful for logs and future slot/reservation integration: total KV blocks, sequence limits, batched-token limits. | `/server_info` server args and scheduler-derived fields, when present. This PR parses common top-level, `server_args`, `scheduler_info`, and `scheduler_infos[0]` shapes. | Dynamo-wrapped vLLM reads `num_gpu_blocks`, `max_num_seqs`, and `max_num_batched_tokens` from `VllmConfig`. Raw vLLM may expose related values through metrics/startup config, but not as a stable normalized EPP metadata contract. | Parsed for SGLang as hints only. Not required for current external EPP routing correctness. |
| Exact tokenization / chat template | Precise prefix routing needs token ids matching the worker-side cache keys. | `/model_info` helps identify the model, but it is not treated as a complete tokenizer/template contract. | `/v1/tokenize` can tokenize on a worker, but binding EPP to a selected worker for tokenization adds an extra worker hop and load coupling. | This PR keeps precise mode on `DYN_EPP_TOKENIZE_URL`, normally a local tokenizer sidecar. Load-aware mode does not require tokenizer metadata. |
| Disaggregated-serving endpoint and role | Needed for runtime-less P/D disaggregated routing, not for aggregated workers. | SGLang runtime metadata can expose some disaggregation config, but this PR does not consume it. | Dynamo-wrapped vLLM/SGLang publish richer role and endpoint metadata through Dynamo registration. Raw vLLM does not have a normalized external contract here. | Out of scope for this PR. Aggregated raw workers only. |

## Interpretation

For raw SGLang aggregated workers, `/model_info` plus `/server_info` appear
sufficient for the external EPP subset needed by this PR: model identity,
block/page size, DP rank topology, KV-event connectivity, and capacity hints.
The adapter should still normalize and validate those engine-native responses
instead of treating the raw SGLang payload as the Dynamo contract.

For raw vLLM, the existing external EPP path remains intentionally conservative:
model identity can come from `/v1/models`, while block size and KV-event
connectivity come from the fixed env/static contract added for the vLLM path.
That is enough to preserve the current vLLM behavior, but it is not the same as
full runtime metadata self-discovery. A stable upstream vLLM runtime-info
endpoint, or a thin engine-local adapter that normalizes `VllmConfig`, would be
the cleaner long-term equivalent to the SGLang `/server_info` path.

## Non-Goals

The following MDC/MRC fields are important elsewhere in Dynamo but are not
required by this raw aggregated external EPP path:

- tool-call, reasoning, and structural-tag parser configuration
- media decoder/fetcher configuration
- LoRA cards and adapter-specific routing metadata
- worker `needs` graphs for disaggregated prefill/decode or encode stages
- topology domains, taints, and KV-transfer policy
- worker-local indexer advertisement through Dynamo runtime discovery
- high-frequency load, queue, slot, or KV state

Dynamic state should continue to come from request-plane accounting, slot or
reservation services, metrics, and KV-event streams rather than from startup
metadata probes.
