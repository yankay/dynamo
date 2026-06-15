// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use anyhow::Result;
use dynamo_llm::kv_router::KvRouter;
use dynamo_runtime::discovery::hash_pod_name;

use super::{pod_endpoint_address, pod_is_ready};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum ExternalEngine {
    Auto,
    Vllm,
    Sglang,
}

impl ExternalEngine {
    pub(super) fn from_env() -> Self {
        match std::env::var("DYN_EPP_ENGINE")
            .unwrap_or_else(|_| "auto".to_string())
            .trim()
            .to_lowercase()
            .as_str()
        {
            "vllm" => Self::Vllm,
            "sglang" => Self::Sglang,
            _ => Self::Auto,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct ExternalBootstrap {
    pub(super) block_size: u32,
    pub(super) model_name: String,
}

const SGLANG_BOOTSTRAP_INITIAL_RETRY: Duration = Duration::from_millis(500);
const SGLANG_BOOTSTRAP_MAX_RETRY: Duration = Duration::from_secs(5);

#[derive(Debug, Clone, PartialEq, Eq)]
struct KvListenerSpec {
    worker_id: u64,
    dp_rank: u32,
    endpoint: String,
    topic: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct KvListenerKey {
    worker_id: u64,
    dp_rank: u32,
}

struct RegisteredKvListener {
    endpoint: String,
    topic: String,
    token: tokio_util::sync::CancellationToken,
}

#[derive(Debug, Clone)]
struct CachedPodKvListeners {
    fingerprint: String,
    specs: Vec<KvListenerSpec>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct SglangKvEvents {
    endpoint_port_base: u16,
    block_size: u32,
    dp_size: u32,
    topic: String,
}

#[derive(Debug, Clone)]
struct SglangWorkerMetadata {
    model_name: Option<String>,
    block_size: Option<u32>,
    dp_size: u32,
    capacity: SglangWorkerCapacity,
    kv_events: Option<SglangKvEvents>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
struct SglangWorkerCapacity {
    context_length: Option<u32>,
    max_total_num_tokens: Option<u64>,
    max_running_requests: Option<u64>,
    max_prefill_tokens: Option<u64>,
    total_kv_blocks: Option<u64>,
    max_num_seqs: Option<u64>,
    max_num_batched_tokens: Option<u64>,
    enable_dp_attention: bool,
    nnodes: u32,
    node_rank: u32,
    data_parallel_start_rank: u32,
    data_parallel_size: u32,
}

pub(super) async fn resolve_external_bootstrap(
    engine: ExternalEngine,
    client: &reqwest::Client,
    pod_store: &kube::runtime::reflector::Store<k8s_openapi::api::core::v1::Pod>,
    target_port: Option<i32>,
) -> ExternalBootstrap {
    match engine {
        ExternalEngine::Sglang => {
            return resolve_sglang_bootstrap_with_retry(client, pod_store, target_port).await;
        }
        ExternalEngine::Auto => {
            if let Some(metadata) =
                fetch_first_sglang_metadata(client, pod_store, target_port).await
            {
                if let Some(bootstrap) = sglang_bootstrap_from_metadata(metadata.clone()) {
                    log_sglang_bootstrap(&metadata, &bootstrap);
                    return bootstrap;
                }
                tracing::warn!(
                    ?metadata,
                    "SGLang metadata probe succeeded but did not include a usable block size; using external env fallback"
                );
            }
        }
        ExternalEngine::Vllm => {}
    }

    external_bootstrap_from_env(match engine {
        ExternalEngine::Sglang => "sglang",
        _ => "vllm",
    })
}

async fn resolve_sglang_bootstrap_with_retry(
    client: &reqwest::Client,
    pod_store: &kube::runtime::reflector::Store<k8s_openapi::api::core::v1::Pod>,
    target_port: Option<i32>,
) -> ExternalBootstrap {
    let mut attempt: u64 = 0;
    let mut retry_after = SGLANG_BOOTSTRAP_INITIAL_RETRY;
    loop {
        attempt += 1;
        match fetch_first_sglang_metadata(client, pod_store, target_port).await {
            Some(metadata) => {
                if let Some(bootstrap) = sglang_bootstrap_from_metadata(metadata.clone()) {
                    log_sglang_bootstrap(&metadata, &bootstrap);
                    return bootstrap;
                }
                tracing::warn!(
                    ?metadata,
                    attempt,
                    retry_after_ms = retry_after.as_millis(),
                    "DYN_EPP_ENGINE=sglang but metadata did not include a usable block size; retrying before initializing router"
                );
            }
            None => {
                if attempt == 1 || attempt % 12 == 0 {
                    tracing::warn!(
                        attempt,
                        retry_after_ms = retry_after.as_millis(),
                        "DYN_EPP_ENGINE=sglang but no ready SGLang pod metadata is available; retrying before initializing router"
                    );
                }
            }
        }

        tokio::time::sleep(retry_after).await;
        retry_after = retry_after
            .saturating_mul(2)
            .min(SGLANG_BOOTSTRAP_MAX_RETRY);
    }
}

fn sglang_bootstrap_from_metadata(metadata: SglangWorkerMetadata) -> Option<ExternalBootstrap> {
    let block_size = metadata.block_size?;
    let model_name = metadata
        .model_name
        .unwrap_or_else(|| default_external_model_name("sglang"));
    Some(ExternalBootstrap {
        block_size,
        model_name,
    })
}

fn log_sglang_bootstrap(metadata: &SglangWorkerMetadata, bootstrap: &ExternalBootstrap) {
    tracing::info!(
        block_size = bootstrap.block_size,
        model_name = %bootstrap.model_name,
        dp_size = metadata.dp_size,
        context_length = ?metadata.capacity.context_length,
        total_kv_blocks = ?metadata.capacity.total_kv_blocks,
        max_num_seqs = ?metadata.capacity.max_num_seqs,
        max_num_batched_tokens = ?metadata.capacity.max_num_batched_tokens,
        data_parallel_start_rank = metadata.capacity.data_parallel_start_rank,
        data_parallel_size = metadata.capacity.data_parallel_size,
        kv_events = metadata.kv_events.is_some(),
        "External SGLang metadata resolved"
    );
}

pub(super) fn spawn_kv_event_reconciler(
    decode_router: Arc<KvRouter>,
    pod_store: kube::runtime::reflector::Store<k8s_openapi::api::core::v1::Pod>,
    target_port: Option<i32>,
    engine: ExternalEngine,
    client: reqwest::Client,
    vllm_kv_event_port: i32,
    fallback_kv_event_topic: String,
) {
    tokio::spawn(async move {
        let mut registered: HashMap<KvListenerKey, RegisteredKvListener> = HashMap::new();
        let mut metadata_cache: HashMap<String, CachedPodKvListeners> = HashMap::new();
        loop {
            let mut desired: HashMap<KvListenerKey, KvListenerSpec> = HashMap::new();
            for pod in pod_store.state() {
                if pod.metadata.name.is_none() || !pod_is_ready(&pod) {
                    continue;
                }
                let specs = resolve_pod_kv_listener_specs(
                    &client,
                    &pod,
                    target_port,
                    engine,
                    vllm_kv_event_port,
                    &fallback_kv_event_topic,
                    &mut metadata_cache,
                )
                .await;
                for spec in specs {
                    desired.insert(
                        KvListenerKey {
                            worker_id: spec.worker_id,
                            dp_rank: spec.dp_rank,
                        },
                        spec,
                    );
                }
            }

            for (key, spec) in &desired {
                let should_replace = registered
                    .get(key)
                    .map(|listener| {
                        listener.endpoint != spec.endpoint || listener.topic != spec.topic
                    })
                    .unwrap_or(false);
                if should_replace && let Some(listener) = registered.remove(key) {
                    listener.token.cancel();
                }

                if let std::collections::hash_map::Entry::Vacant(slot) = registered.entry(*key) {
                    let token = decode_router.register_worker_kv_events(
                        spec.worker_id,
                        spec.endpoint.clone(),
                        spec.topic.clone(),
                        Some(spec.dp_rank),
                    );
                    slot.insert(RegisteredKvListener {
                        endpoint: spec.endpoint.clone(),
                        topic: spec.topic.clone(),
                        token,
                    });
                    tracing::info!(
                        endpoint = %spec.endpoint,
                        topic = %spec.topic,
                        worker_id = spec.worker_id,
                        dp_rank = spec.dp_rank,
                        "Registered worker KV-event listener"
                    );
                }
            }

            registered.retain(|key, listener| {
                if desired.contains_key(key) {
                    true
                } else {
                    tracing::info!(
                        worker_id = key.worker_id,
                        dp_rank = key.dp_rank,
                        endpoint = %listener.endpoint,
                        "Worker pod/rank gone; cancelling KV-event listener"
                    );
                    listener.token.cancel();
                    false
                }
            });
            tokio::time::sleep(std::time::Duration::from_secs(5)).await;
        }
    });
}

fn external_bootstrap_from_env(default_model_name: &str) -> ExternalBootstrap {
    let block_size = std::env::var("DYN_KV_CACHE_BLOCK_SIZE")
        .ok()
        .and_then(|v| v.trim().parse::<u32>().ok())
        .unwrap_or(16);
    let model_name = default_external_model_name(default_model_name);
    ExternalBootstrap {
        block_size,
        model_name,
    }
}

fn configured_external_model_name() -> Option<String> {
    std::env::var("DYN_MODEL_NAME")
        .ok()
        .map(|v| v.trim().to_string())
        .filter(|v| !v.is_empty())
}

fn default_external_model_name(default_model_name: &str) -> String {
    configured_external_model_name().unwrap_or_else(|| default_model_name.to_string())
}

async fn fetch_first_sglang_metadata(
    client: &reqwest::Client,
    pod_store: &kube::runtime::reflector::Store<k8s_openapi::api::core::v1::Pod>,
    target_port: Option<i32>,
) -> Option<SglangWorkerMetadata> {
    for pod in pod_store.state() {
        if !pod_is_ready(&pod) {
            continue;
        }
        let Some(endpoint) = pod_endpoint_address(&pod, target_port) else {
            continue;
        };
        let Some(pod_name) = pod.metadata.name.as_deref() else {
            continue;
        };

        let fallback_topic = std::env::var("DYN_EPP_KV_EVENT_TOPIC").unwrap_or_default();
        match fetch_sglang_worker_metadata(client, &endpoint, &fallback_topic).await {
            Ok(metadata) => {
                tracing::info!(
                    pod = %pod_name,
                    endpoint = %endpoint,
                    "Resolved SGLang metadata from ready pod"
                );
                return Some(metadata);
            }
            Err(error) => {
                tracing::debug!(
                    pod = %pod_name,
                    endpoint = %endpoint,
                    error = %error,
                    "Ready pod did not expose SGLang metadata"
                );
            }
        }
    }
    None
}

async fn fetch_sglang_worker_metadata(
    client: &reqwest::Client,
    endpoint: &str,
    fallback_topic: &str,
) -> Result<SglangWorkerMetadata> {
    let base = format!("http://{endpoint}");
    let models = fetch_json(client, &format!("{base}/v1/models")).await.ok();
    let model_info = fetch_json(client, &format!("{base}/model_info")).await?;
    let server_info = fetch_json(client, &format!("{base}/server_info")).await?;

    Ok(parse_sglang_worker_metadata(
        models.as_ref(),
        &model_info,
        &server_info,
        fallback_topic,
        configured_external_model_name().as_deref(),
    ))
}

async fn fetch_json(client: &reqwest::Client, url: &str) -> Result<serde_json::Value> {
    let resp = client
        .get(url)
        .send()
        .await
        .map_err(|e| anyhow::anyhow!("GET {url} failed: {e}"))?;
    let status = resp.status();
    let value = resp
        .json::<serde_json::Value>()
        .await
        .map_err(|e| anyhow::anyhow!("GET {url} response parse failed: {e}"))?;
    if !status.is_success() {
        anyhow::bail!("GET {url} returned {status}: {value}");
    }
    Ok(value)
}

fn parse_sglang_worker_metadata(
    models: Option<&serde_json::Value>,
    model_info: &serde_json::Value,
    server_info: &serde_json::Value,
    fallback_topic: &str,
    configured_model_name: Option<&str>,
) -> SglangWorkerMetadata {
    let kv_events = parse_sglang_kv_events(server_info, fallback_topic);
    let block_size = kv_events.as_ref().map(|k| k.block_size).or_else(|| {
        json_u32_any(
            server_info,
            &[&["page_size"], &["server_args", "page_size"]],
        )
    });
    let dp_size = kv_events
        .as_ref()
        .map(|k| k.dp_size)
        .or_else(|| json_u32_any(server_info, &[&["dp_size"], &["server_args", "dp_size"]]))
        .unwrap_or(1);
    let capacity = parse_sglang_worker_capacity(server_info, block_size, dp_size);
    let model_name = parse_openai_model_name(models)
        .or_else(|| configured_model_name.map(str::to_string))
        .or_else(|| {
            model_info
                .get("model_path")
                .and_then(|v| v.as_str())
                .map(str::to_string)
        });

    SglangWorkerMetadata {
        model_name,
        block_size,
        dp_size,
        capacity,
        kv_events,
    }
}

fn parse_openai_model_name(models: Option<&serde_json::Value>) -> Option<String> {
    models?
        .get("data")?
        .as_array()?
        .first()?
        .get("id")?
        .as_str()
        .map(str::to_string)
}

fn parse_sglang_kv_events(
    server_info: &serde_json::Value,
    fallback_topic: &str,
) -> Option<SglangKvEvents> {
    let kv_events = server_info.get("kv_events")?.as_object()?;
    if kv_events
        .get("publisher")
        .and_then(|v| v.as_str())
        .is_some_and(|publisher| publisher != "zmq")
    {
        return None;
    }
    if kv_events
        .get("endpoint")
        .and_then(|v| v.as_str())
        .is_some_and(|endpoint| !endpoint.starts_with("tcp://"))
    {
        return None;
    }

    let endpoint_port_base = kv_events
        .get("endpoint_port_base")?
        .as_u64()
        .and_then(|n| u16::try_from(n).ok())
        .filter(|n| *n > 0)?;
    let block_size = kv_events
        .get("block_size")?
        .as_u64()
        .and_then(|n| u32::try_from(n).ok())
        .filter(|n| *n > 0)?;
    let dp_size = kv_events
        .get("dp_size")?
        .as_u64()
        .and_then(|n| u32::try_from(n).ok())
        .filter(|n| *n > 0)?;
    let topic = kv_events
        .get("topic")
        .and_then(|v| v.as_str())
        .unwrap_or(fallback_topic)
        .to_string();

    Some(SglangKvEvents {
        endpoint_port_base,
        block_size,
        dp_size,
        topic,
    })
}

fn parse_sglang_worker_capacity(
    server_info: &serde_json::Value,
    block_size: Option<u32>,
    dp_size: u32,
) -> SglangWorkerCapacity {
    let context_length = json_u32_any(
        server_info,
        &[
            &["context_length"],
            &["server_args", "context_length"],
            &["model_config", "context_length"],
            &["max_model_len"],
            &["server_args", "max_model_len"],
        ],
    );
    let max_total_num_tokens = json_u64_with_scheduler_info(server_info, "max_total_num_tokens");
    let max_running_requests = json_u64_any(
        server_info,
        &[
            &["max_running_requests"],
            &["server_args", "max_running_requests"],
        ],
    );
    let max_prefill_tokens = json_u64_any(
        server_info,
        &[
            &["max_prefill_tokens"],
            &["server_args", "max_prefill_tokens"],
        ],
    );
    let enable_dp_attention = json_bool_any(
        server_info,
        &[
            &["enable_dp_attention"],
            &["server_args", "enable_dp_attention"],
        ],
    )
    .unwrap_or(false);
    let nnodes = json_u32_any(server_info, &[&["nnodes"], &["server_args", "nnodes"]])
        .unwrap_or(1)
        .max(1);
    let node_rank = json_u32_any(
        server_info,
        &[&["node_rank"], &["server_args", "node_rank"]],
    )
    .unwrap_or(0);
    let (data_parallel_start_rank, data_parallel_size) =
        sglang_local_dp_rank_bounds(dp_size, enable_dp_attention, nnodes, node_rank);
    let total_kv_blocks = json_u64_any(
        server_info,
        &[
            &["total_kv_blocks"],
            &["scheduler_info", "total_kv_blocks"],
            &["runtime_config", "total_kv_blocks"],
        ],
    )
    .or_else(|| {
        let block_size = u64::from(block_size?);
        let tokens = max_total_num_tokens?;
        Some(tokens.div_ceil(block_size))
    });
    let max_num_seqs = json_u64_any(
        server_info,
        &[
            &["max_num_seqs"],
            &["runtime_config", "max_num_seqs"],
            &["scheduler_info", "max_num_seqs"],
        ],
    )
    .or_else(|| {
        let requests = max_running_requests?;
        Some(if dp_size <= 1 {
            requests
        } else {
            requests / u64::from(dp_size)
        })
    });
    let max_num_batched_tokens = json_u64_any(
        server_info,
        &[
            &["max_num_batched_tokens"],
            &["runtime_config", "max_num_batched_tokens"],
            &["scheduler_info", "max_num_batched_tokens"],
        ],
    )
    .or(max_prefill_tokens)
    .or(max_total_num_tokens);

    SglangWorkerCapacity {
        context_length,
        max_total_num_tokens,
        max_running_requests,
        max_prefill_tokens,
        total_kv_blocks,
        max_num_seqs,
        max_num_batched_tokens,
        enable_dp_attention,
        nnodes,
        node_rank,
        data_parallel_start_rank,
        data_parallel_size,
    }
}

fn sglang_local_dp_rank_bounds(
    dp_size: u32,
    enable_dp_attention: bool,
    nnodes: u32,
    node_rank: u32,
) -> (u32, u32) {
    if enable_dp_attention && dp_size > 1 {
        let local_dp_size = dp_size / nnodes.max(1);
        if local_dp_size > 0 {
            let start = node_rank.saturating_mul(local_dp_size);
            let end = start.saturating_add(local_dp_size).min(dp_size);
            if start < end {
                return (start, end - start);
            }
        }
    }

    (0, 1)
}

fn json_u32_any(value: &serde_json::Value, paths: &[&[&str]]) -> Option<u32> {
    json_u64_any(value, paths).and_then(|n| u32::try_from(n).ok())
}

fn json_u64_any(value: &serde_json::Value, paths: &[&[&str]]) -> Option<u64> {
    paths.iter().find_map(|path| json_u64(value, path))
}

fn json_u64(value: &serde_json::Value, path: &[&str]) -> Option<u64> {
    let mut current = value;
    for key in path {
        current = current.get(*key)?;
    }
    current.as_u64()
}

fn json_bool_any(value: &serde_json::Value, paths: &[&[&str]]) -> Option<bool> {
    paths.iter().find_map(|path| json_bool(value, path))
}

fn json_bool(value: &serde_json::Value, path: &[&str]) -> Option<bool> {
    let mut current = value;
    for key in path {
        current = current.get(*key)?;
    }
    current.as_bool()
}

fn json_u64_with_scheduler_info(value: &serde_json::Value, key: &str) -> Option<u64> {
    json_u64_any(value, &[&[key], &["scheduler_info", key]]).or_else(|| {
        value
            .get("scheduler_infos")
            .and_then(|infos| infos.as_array())
            .and_then(|infos| infos.first())
            .and_then(|info| json_u64(info, &[key]))
    })
}

async fn resolve_pod_kv_listener_specs(
    client: &reqwest::Client,
    pod: &k8s_openapi::api::core::v1::Pod,
    target_port: Option<i32>,
    engine: ExternalEngine,
    vllm_kv_event_port: i32,
    fallback_topic: &str,
    metadata_cache: &mut HashMap<String, CachedPodKvListeners>,
) -> Vec<KvListenerSpec> {
    match engine {
        ExternalEngine::Vllm => vllm_kv_listener_specs(pod, vllm_kv_event_port, fallback_topic),
        ExternalEngine::Sglang => {
            sglang_kv_listener_specs(client, pod, target_port, fallback_topic, metadata_cache).await
        }
        ExternalEngine::Auto => {
            let sglang =
                sglang_kv_listener_specs(client, pod, target_port, fallback_topic, metadata_cache)
                    .await;
            if sglang.is_empty() {
                vllm_kv_listener_specs(pod, vllm_kv_event_port, fallback_topic)
            } else {
                sglang
            }
        }
    }
}

async fn sglang_kv_listener_specs(
    client: &reqwest::Client,
    pod: &k8s_openapi::api::core::v1::Pod,
    target_port: Option<i32>,
    fallback_topic: &str,
    metadata_cache: &mut HashMap<String, CachedPodKvListeners>,
) -> Vec<KvListenerSpec> {
    let Some(pod_name) = pod.metadata.name.as_deref() else {
        return Vec::new();
    };
    let Some(ip) = pod.status.as_ref().and_then(|s| s.pod_ip.as_deref()) else {
        return Vec::new();
    };
    let Some(endpoint) = pod_endpoint_address(pod, target_port) else {
        return Vec::new();
    };
    let cache_key = pod_uid_or_name(pod);
    let fingerprint = pod_metadata_fingerprint(pod, &endpoint);
    if let Some(cached) = metadata_cache.get(&cache_key)
        && cached.fingerprint == fingerprint
    {
        return cached.specs.clone();
    }

    let worker_id = hash_pod_name(pod_name);
    match fetch_sglang_worker_metadata(client, &endpoint, fallback_topic).await {
        Ok(metadata) => {
            let specs = build_sglang_kv_listener_specs(worker_id, ip, &metadata);
            if specs.is_empty() {
                tracing::warn!(
                    pod = %pod_name,
                    endpoint = %endpoint,
                    "SGLang metadata did not include usable KV-event endpoints; skipping precise KV listener for this pod"
                );
            } else {
                tracing::info!(
                    pod = %pod_name,
                    endpoint = %endpoint,
                    worker_id,
                    listener_count = specs.len(),
                    context_length = ?metadata.capacity.context_length,
                    total_kv_blocks = ?metadata.capacity.total_kv_blocks,
                    max_num_seqs = ?metadata.capacity.max_num_seqs,
                    max_num_batched_tokens = ?metadata.capacity.max_num_batched_tokens,
                    data_parallel_start_rank = metadata.capacity.data_parallel_start_rank,
                    data_parallel_size = metadata.capacity.data_parallel_size,
                    "Resolved SGLang worker KV-event metadata"
                );
            }
            metadata_cache.insert(
                cache_key,
                CachedPodKvListeners {
                    fingerprint,
                    specs: specs.clone(),
                },
            );
            specs
        }
        Err(error) => {
            tracing::debug!(
                pod = %pod_name,
                endpoint = %endpoint,
                error = %error,
                "SGLang metadata probe failed; falling back or skipping according to engine mode"
            );
            Vec::new()
        }
    }
}

fn vllm_kv_listener_specs(
    pod: &k8s_openapi::api::core::v1::Pod,
    kv_event_port: i32,
    topic: &str,
) -> Vec<KvListenerSpec> {
    let Some(name) = pod.metadata.name.as_deref() else {
        return Vec::new();
    };
    let Some(ip) = pod.status.as_ref().and_then(|s| s.pod_ip.as_deref()) else {
        return Vec::new();
    };
    vec![KvListenerSpec {
        worker_id: hash_pod_name(name),
        dp_rank: 0,
        endpoint: format!("tcp://{ip}:{kv_event_port}"),
        topic: topic.to_string(),
    }]
}

fn build_sglang_kv_listener_specs(
    worker_id: u64,
    pod_ip: &str,
    metadata: &SglangWorkerMetadata,
) -> Vec<KvListenerSpec> {
    let Some(kv_events) = metadata.kv_events.as_ref() else {
        return Vec::new();
    };
    let rank_range = sglang_listener_dp_rank_range(metadata, kv_events.dp_size);
    rank_range
        .filter_map(|dp_rank| {
            let port = u32::from(kv_events.endpoint_port_base).checked_add(dp_rank)?;
            if port > u32::from(u16::MAX) {
                return None;
            }
            Some(KvListenerSpec {
                worker_id,
                dp_rank,
                endpoint: format!("tcp://{pod_ip}:{port}"),
                topic: kv_events.topic.clone(),
            })
        })
        .collect()
}

fn sglang_listener_dp_rank_range(
    metadata: &SglangWorkerMetadata,
    kv_events_dp_size: u32,
) -> std::ops::Range<u32> {
    if metadata.capacity.enable_dp_attention {
        let start = metadata.capacity.data_parallel_start_rank;
        let end = start
            .saturating_add(metadata.capacity.data_parallel_size)
            .min(kv_events_dp_size);
        if start < end {
            return start..end;
        }
    }

    0..kv_events_dp_size
}

fn pod_uid_or_name(pod: &k8s_openapi::api::core::v1::Pod) -> String {
    pod.metadata
        .uid
        .clone()
        .or_else(|| pod.metadata.name.clone())
        .unwrap_or_default()
}

fn pod_metadata_fingerprint(pod: &k8s_openapi::api::core::v1::Pod, endpoint: &str) -> String {
    format!(
        "{}:{}:{}",
        pod.metadata.resource_version.as_deref().unwrap_or_default(),
        pod.status
            .as_ref()
            .and_then(|s| s.pod_ip.as_deref())
            .unwrap_or_default(),
        endpoint
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn sglang_metadata_maps_kv_events_to_worker_specs() {
        let models = json!({
            "data": [{"id": "Qwen/Qwen3-8B"}]
        });
        let model_info = json!({
            "model_path": "fallback-model"
        });
        let server_info = json!({
            "page_size": 16,
            "dp_size": 1,
            "kv_events": {
                "publisher": "zmq",
                "endpoint_host": "*",
                "endpoint_port_base": 5557,
                "topic": "",
                "block_size": 64,
                "dp_size": 4
            }
        });

        let metadata =
            parse_sglang_worker_metadata(Some(&models), &model_info, &server_info, "ignored", None);
        assert_eq!(metadata.model_name.as_deref(), Some("Qwen/Qwen3-8B"));
        assert_eq!(metadata.block_size, Some(64));
        assert_eq!(metadata.dp_size, 4);
        assert_eq!(metadata.capacity.data_parallel_start_rank, 0);
        assert_eq!(metadata.capacity.data_parallel_size, 1);

        let specs = build_sglang_kv_listener_specs(7, "10.0.0.9", &metadata);
        assert_eq!(specs.len(), 4);
        assert_eq!(specs[0].endpoint, "tcp://10.0.0.9:5557");
        assert_eq!(specs[0].dp_rank, 0);
        assert_eq!(specs[0].topic, "");
        assert_eq!(specs[3].endpoint, "tcp://10.0.0.9:5560");
        assert_eq!(specs[3].dp_rank, 3);
    }

    #[test]
    fn sglang_metadata_uses_local_dp_range_for_dp_attention() {
        let model_info = json!({
            "model_path": "Qwen/Qwen3-8B"
        });
        let server_info = json!({
            "server_args": {
                "page_size": 16,
                "dp_size": 8,
                "enable_dp_attention": true,
                "nnodes": 2,
                "node_rank": 1
            },
            "kv_events": {
                "publisher": "zmq",
                "endpoint_port_base": 5557,
                "topic": "kv-events",
                "block_size": 16,
                "dp_size": 8
            }
        });

        let metadata = parse_sglang_worker_metadata(None, &model_info, &server_info, "", None);
        assert!(metadata.capacity.enable_dp_attention);
        assert_eq!(metadata.capacity.data_parallel_start_rank, 4);
        assert_eq!(metadata.capacity.data_parallel_size, 4);

        let specs = build_sglang_kv_listener_specs(9, "10.0.0.9", &metadata);
        assert_eq!(specs.len(), 4);
        assert_eq!(specs[0].dp_rank, 4);
        assert_eq!(specs[0].endpoint, "tcp://10.0.0.9:5561");
        assert_eq!(specs[3].dp_rank, 7);
        assert_eq!(specs[3].endpoint, "tcp://10.0.0.9:5564");
    }

    #[test]
    fn sglang_metadata_derives_worker_capacity() {
        let model_info = json!({
            "model_path": "Qwen/Qwen3-8B"
        });
        let server_info = json!({
            "server_args": {
                "context_length": 8192,
                "page_size": 16,
                "dp_size": 4,
                "max_running_requests": 16,
                "max_prefill_tokens": 1024
            },
            "scheduler_infos": [{
                "max_total_num_tokens": 2048
            }],
            "kv_events": {
                "publisher": "zmq",
                "endpoint_port_base": 5557,
                "topic": "kv-events",
                "block_size": 16,
                "dp_size": 4
            }
        });

        let metadata = parse_sglang_worker_metadata(None, &model_info, &server_info, "", None);
        assert_eq!(metadata.capacity.context_length, Some(8192));
        assert_eq!(metadata.capacity.max_total_num_tokens, Some(2048));
        assert_eq!(metadata.capacity.max_running_requests, Some(16));
        assert_eq!(metadata.capacity.max_prefill_tokens, Some(1024));
        assert_eq!(metadata.capacity.total_kv_blocks, Some(128));
        assert_eq!(metadata.capacity.max_num_seqs, Some(4));
        assert_eq!(metadata.capacity.max_num_batched_tokens, Some(1024));
    }

    #[test]
    fn sglang_bootstrap_requires_resolved_block_size() {
        let model_info = json!({
            "model_path": "Qwen/Qwen3-0.6B"
        });
        let server_info = json!({
            "dp_size": 1,
            "kv_events": null
        });

        let metadata = parse_sglang_worker_metadata(None, &model_info, &server_info, "topic", None);
        assert_eq!(metadata.model_name.as_deref(), Some("Qwen/Qwen3-0.6B"));
        assert_eq!(metadata.block_size, None);
        assert_eq!(sglang_bootstrap_from_metadata(metadata), None);
    }

    #[test]
    fn sglang_metadata_without_kv_events_keeps_capacity_but_skips_listeners() {
        let model_info = json!({
            "model_path": "Qwen/Qwen3-0.6B"
        });
        let server_info = json!({
            "page_size": 32,
            "dp_size": 2,
            "kv_events": null
        });

        let metadata = parse_sglang_worker_metadata(None, &model_info, &server_info, "topic", None);
        assert_eq!(metadata.model_name.as_deref(), Some("Qwen/Qwen3-0.6B"));
        assert_eq!(metadata.block_size, Some(32));
        assert_eq!(metadata.dp_size, 2);
        assert!(metadata.kv_events.is_none());
        assert!(build_sglang_kv_listener_specs(1, "10.0.0.2", &metadata).is_empty());
    }

    #[test]
    fn sglang_model_name_prefers_configured_name_before_model_path() {
        let model_info = json!({
            "model_path": "fallback-model-path"
        });
        let server_info = json!({
            "page_size": 16,
            "dp_size": 1,
            "kv_events": null
        });

        let metadata = parse_sglang_worker_metadata(
            None,
            &model_info,
            &server_info,
            "topic",
            Some("configured-model"),
        );
        assert_eq!(metadata.model_name.as_deref(), Some("configured-model"));
    }

    #[test]
    fn sglang_kv_events_reject_invalid_descriptor() {
        let server_info = json!({
            "kv_events": {
                "publisher": "zmq",
                "endpoint": "ipc:///tmp/kv-events",
                "endpoint_port_base": 5557,
                "block_size": 64,
                "dp_size": 1
            }
        });
        assert!(parse_sglang_kv_events(&server_info, "").is_none());

        let server_info = json!({
            "kv_events": {
                "publisher": "zmq",
                "block_size": 64,
                "dp_size": 1
            }
        });
        assert!(parse_sglang_kv_events(&server_info, "").is_none());
    }
}
