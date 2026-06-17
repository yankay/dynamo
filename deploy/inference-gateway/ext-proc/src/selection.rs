// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Thin EndpointPicker adapter for the runtime-free selection service.

use std::collections::HashSet;
use std::time::Duration;

use reqwest::StatusCode;
use serde::{Deserialize, Serialize};

use crate::picker::{Endpoint, EndpointPicker, PickError, PickResult, RequestInfo};

const DEFAULT_TENANT_ID: &str = "default";
const DEFAULT_MODEL_NAME: &str = "default";

#[derive(Debug, Clone)]
pub struct SelectionPickerConfig {
    pub selection_service_url: String,
    pub tenant_id: String,
    pub default_model_name: String,
    pub timeout: Duration,
}

impl SelectionPickerConfig {
    pub fn new(selection_service_url: impl Into<String>) -> Self {
        Self {
            selection_service_url: selection_service_url.into(),
            tenant_id: DEFAULT_TENANT_ID.to_string(),
            default_model_name: DEFAULT_MODEL_NAME.to_string(),
            timeout: Duration::from_millis(2000),
        }
    }
}

#[derive(Debug, Clone)]
pub struct SelectionPicker {
    client: SelectionClient,
    tenant_id: String,
    default_model_name: String,
}

impl SelectionPicker {
    pub fn new(config: SelectionPickerConfig) -> Result<Self, SelectionClientError> {
        Ok(Self {
            client: SelectionClient::new(&config.selection_service_url, config.timeout)?,
            tenant_id: default_if_empty(config.tenant_id, DEFAULT_TENANT_ID),
            default_model_name: default_if_empty(config.default_model_name, DEFAULT_MODEL_NAME),
        })
    }

    fn build_selection_request(
        &self,
        req: &RequestInfo,
    ) -> Result<SelectAndReserveRequest, PickError> {
        let body: serde_json::Value = serde_json::from_slice(&req.body).map_err(|error| {
            PickError::TokenizationFailed(format!("request body must be valid JSON: {error}"))
        })?;

        let token_ids = extract_token_data(&body)?;
        let model_name = body
            .get("model")
            .and_then(serde_json::Value::as_str)
            .filter(|model| !model.is_empty())
            .unwrap_or(self.default_model_name.as_str())
            .to_string();
        let allowed_worker_ids = parse_candidate_subset(&req.candidate_subset)?;

        Ok(SelectAndReserveRequest {
            model_name,
            tenant_id: self.tenant_id.clone(),
            selection_id: Some(req.request_id.clone()),
            reservation_id: Some(req.request_id.clone()),
            token_ids,
            expected_output_tokens: extract_expected_output_tokens(&body),
            priority_jump: body
                .pointer("/nvext/agent_hints/priority")
                .and_then(serde_json::Value::as_f64),
            strict_priority: body
                .pointer("/nvext/agent_hints/strict_priority")
                .and_then(value_as_u32),
            allowed_worker_ids,
        })
    }

    async fn pick_header_only(&self) -> Result<PickResult, PickError> {
        let workers = self
            .client
            .list_workers(&self.default_model_name, &self.tenant_id)
            .await
            .map_err(client_error_to_pick_error)?;
        let worker = workers
            .into_iter()
            .find(|worker| worker.lifecycle == WorkerLifecycle::Schedulable)
            .ok_or(PickError::NoEndpoints)?;
        let endpoint = worker
            .endpoint
            .ok_or_else(|| PickError::RoutingFailed("schedulable worker has no endpoint".into()))?;
        Ok(PickResult {
            endpoint: normalize_endpoint_for_envoy(&endpoint)
                .map_err(client_error_to_pick_error)?,
            ..Default::default()
        })
    }
}

#[tonic::async_trait]
impl EndpointPicker for SelectionPicker {
    async fn pick(
        &self,
        req: &RequestInfo,
        _endpoints: &[Endpoint],
    ) -> Result<PickResult, PickError> {
        if req.body.is_empty() {
            return self.pick_header_only().await;
        }

        let selection_request = self.build_selection_request(req)?;
        let token_ids = selection_request.token_ids.clone();
        let response = self
            .client
            .select_and_reserve(&selection_request)
            .await
            .map_err(client_error_to_pick_error)?;
        let endpoint =
            normalize_endpoint_for_envoy(&response.endpoint).map_err(client_error_to_pick_error)?;

        Ok(PickResult {
            endpoint,
            fallbacks: vec![],
            headers: vec![
                (
                    "x-worker-instance-id".to_string(),
                    response.worker_id.to_string(),
                ),
                ("x-dp-rank".to_string(), response.dp_rank.to_string()),
                (
                    "x-dynamo-routing-mode".to_string(),
                    "aggregated".to_string(),
                ),
            ],
            token_ids: Some(token_ids),
        })
    }

    async fn on_prefill_complete(&self, request_id: &str) {
        if request_id.is_empty() {
            return;
        }
        if let Err(error) = self.client.mark_prefill_complete(request_id).await {
            tracing::warn!(
                request_id = %request_id,
                error = %error,
                "Failed to mark selection-service reservation prefill-complete"
            );
        }
    }

    async fn on_request_complete(&self, request_id: &str) {
        if request_id.is_empty() {
            return;
        }
        if let Err(error) = self.client.delete_reservation(request_id).await {
            tracing::warn!(
                request_id = %request_id,
                error = %error,
                "Failed to delete selection-service reservation"
            );
        }
    }
}

#[derive(Debug, Clone)]
struct SelectionClient {
    base_url: reqwest::Url,
    http: reqwest::Client,
}

impl SelectionClient {
    fn new(base_url: &str, timeout: Duration) -> Result<Self, SelectionClientError> {
        let mut parsed =
            reqwest::Url::parse(base_url).map_err(|source| SelectionClientError::InvalidUrl {
                url: base_url.to_string(),
                message: source.to_string(),
            })?;
        if !parsed.path().ends_with('/') {
            parsed.set_path(&format!("{}/", parsed.path().trim_end_matches('/')));
        }
        let http = reqwest::Client::builder()
            .timeout(timeout)
            .build()
            .map_err(SelectionClientError::BuildClient)?;
        Ok(Self {
            base_url: parsed,
            http,
        })
    }

    async fn select_and_reserve(
        &self,
        request: &SelectAndReserveRequest,
    ) -> Result<SelectResponse, SelectionClientError> {
        let response = self
            .http
            .post(self.url("select_and_reserve")?)
            .json(request)
            .send()
            .await
            .map_err(SelectionClientError::Request)?;
        decode_json_response(response, "POST", "/select_and_reserve").await
    }

    async fn list_workers(
        &self,
        model_name: &str,
        tenant_id: &str,
    ) -> Result<Vec<WorkerRecord>, SelectionClientError> {
        let response = self
            .http
            .get(self.url("workers")?)
            .query(&[("model_name", model_name), ("tenant_id", tenant_id)])
            .send()
            .await
            .map_err(SelectionClientError::Request)?;
        decode_json_response(response, "GET", "/workers").await
    }

    async fn mark_prefill_complete(
        &self,
        reservation_id: &str,
    ) -> Result<(), SelectionClientError> {
        let path = format!("reservations/{reservation_id}/prefill_complete");
        self.empty_response("POST", &path).await
    }

    async fn delete_reservation(&self, reservation_id: &str) -> Result<(), SelectionClientError> {
        let path = format!("reservations/{reservation_id}");
        self.empty_response("DELETE", &path).await
    }

    async fn empty_response(
        &self,
        method: &'static str,
        path: &str,
    ) -> Result<(), SelectionClientError> {
        let url = self.url(path)?;
        let request = match method {
            "POST" => self.http.post(url),
            "DELETE" => self.http.delete(url),
            _ => unreachable!("unsupported selection client method"),
        };
        let response = request
            .send()
            .await
            .map_err(SelectionClientError::Request)?;
        ensure_success(response, method, path).await
    }

    fn url(&self, path: &str) -> Result<reqwest::Url, SelectionClientError> {
        self.base_url
            .join(path)
            .map_err(|source| SelectionClientError::JoinUrl {
                base_url: self.base_url.to_string(),
                path: path.to_string(),
                message: source.to_string(),
            })
    }
}

async fn decode_json_response<T: for<'de> Deserialize<'de>>(
    response: reqwest::Response,
    method: &'static str,
    path: &'static str,
) -> Result<T, SelectionClientError> {
    let response = ensure_success_response(response, method, path).await?;
    response
        .json::<T>()
        .await
        .map_err(SelectionClientError::DecodeJson)
}

async fn ensure_success(
    response: reqwest::Response,
    method: &'static str,
    path: &str,
) -> Result<(), SelectionClientError> {
    ensure_success_response(response, method, path).await?;
    Ok(())
}

async fn ensure_success_response(
    response: reqwest::Response,
    method: &'static str,
    path: &str,
) -> Result<reqwest::Response, SelectionClientError> {
    let status = response.status();
    if status.is_success() {
        return Ok(response);
    }
    let body = response.text().await.unwrap_or_else(|error| {
        format!("failed to read selection-service error response: {error}")
    });
    Err(SelectionClientError::HttpStatus {
        method,
        path: path.to_string(),
        status,
        body,
    })
}

fn extract_token_data(body: &serde_json::Value) -> Result<Vec<u32>, PickError> {
    let values = body
        .pointer("/nvext/token_data")
        .and_then(serde_json::Value::as_array)
        .ok_or_else(|| {
            PickError::TokenizationFailed(
                "nvext.token_data is required for selection-service EPP mode".to_string(),
            )
        })?;
    if values.is_empty() {
        return Err(PickError::TokenizationFailed(
            "nvext.token_data must not be empty".to_string(),
        ));
    }
    values
        .iter()
        .map(|value| {
            value_as_u32(value).ok_or_else(|| {
                PickError::TokenizationFailed(
                    "nvext.token_data must contain only unsigned 32-bit integers".to_string(),
                )
            })
        })
        .collect()
}

fn extract_expected_output_tokens(body: &serde_json::Value) -> Option<u32> {
    body.get("max_completion_tokens")
        .and_then(value_as_u32)
        .or_else(|| body.get("max_tokens").and_then(value_as_u32))
}

fn parse_candidate_subset(values: &[String]) -> Result<Option<HashSet<u64>>, PickError> {
    if values.is_empty() {
        return Ok(None);
    }
    let mut allowed = HashSet::with_capacity(values.len());
    for value in values {
        let worker_id = value.parse::<u64>().map_err(|error| {
            PickError::RoutingFailed(format!(
                "candidate subset value {value:?} is not a numeric worker_id: {error}"
            ))
        })?;
        allowed.insert(worker_id);
    }
    Ok(Some(allowed))
}

fn value_as_u32(value: &serde_json::Value) -> Option<u32> {
    let raw = value.as_u64()?;
    u32::try_from(raw).ok()
}

fn normalize_endpoint_for_envoy(endpoint: &str) -> Result<String, SelectionClientError> {
    if endpoint.is_empty() {
        return Err(SelectionClientError::InvalidEndpoint(
            "selection response endpoint is empty".to_string(),
        ));
    }
    let Ok(url) = reqwest::Url::parse(endpoint) else {
        return Ok(endpoint.to_string());
    };
    match url.scheme() {
        "http" | "https" => {
            let host = url.host_str().ok_or_else(|| {
                SelectionClientError::InvalidEndpoint(format!(
                    "selection response endpoint {endpoint:?} has no host"
                ))
            })?;
            let port = url.port_or_known_default().ok_or_else(|| {
                SelectionClientError::InvalidEndpoint(format!(
                    "selection response endpoint {endpoint:?} has no port"
                ))
            })?;
            Ok(format!("{host}:{port}"))
        }
        _ => Ok(endpoint.to_string()),
    }
}

fn default_if_empty(value: String, default: &str) -> String {
    if value.is_empty() {
        default.to_string()
    } else {
        value
    }
}

fn client_error_to_pick_error(error: SelectionClientError) -> PickError {
    match error {
        SelectionClientError::NoEndpoints => PickError::NoEndpoints,
        other => PickError::RoutingFailed(other.to_string()),
    }
}

#[derive(Debug, thiserror::Error)]
pub enum SelectionClientError {
    #[error("invalid selection service URL {url:?}: {message}")]
    InvalidUrl { url: String, message: String },
    #[error("failed to join selection service URL {base_url:?} with path {path:?}: {message}")]
    JoinUrl {
        base_url: String,
        path: String,
        message: String,
    },
    #[error("failed to build selection service HTTP client: {0}")]
    BuildClient(reqwest::Error),
    #[error("selection service request failed: {0}")]
    Request(reqwest::Error),
    #[error("selection service {method} {path} failed with HTTP {status}: {body}")]
    HttpStatus {
        method: &'static str,
        path: String,
        status: StatusCode,
        body: String,
    },
    #[error("failed to decode selection service JSON response: {0}")]
    DecodeJson(reqwest::Error),
    #[error("invalid selection service endpoint: {0}")]
    InvalidEndpoint(String),
    #[error("no schedulable selection service workers")]
    NoEndpoints,
}

#[derive(Debug, Serialize)]
struct SelectAndReserveRequest {
    model_name: String,
    tenant_id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    selection_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    reservation_id: Option<String>,
    token_ids: Vec<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    expected_output_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    priority_jump: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    strict_priority: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    allowed_worker_ids: Option<HashSet<u64>>,
}

#[derive(Debug, Deserialize)]
struct SelectResponse {
    worker_id: u64,
    dp_rank: u32,
    endpoint: String,
}

#[derive(Debug, Deserialize)]
struct WorkerRecord {
    lifecycle: WorkerLifecycle,
    endpoint: Option<String>,
}

#[derive(Debug, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum WorkerLifecycle {
    Schedulable,
    Incomplete,
    Draining,
    Unschedulable,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    use axum::extract::{Path, State};
    use axum::http::StatusCode as AxumStatusCode;
    use axum::routing::{delete, get, post};
    use axum::{Json, Router};
    use tokio::sync::Mutex;

    #[derive(Clone, Default)]
    struct FakeState {
        select_requests: Arc<Mutex<Vec<serde_json::Value>>>,
        prefill_complete: Arc<Mutex<Vec<String>>>,
        deleted: Arc<Mutex<Vec<String>>>,
        select_status: Arc<Mutex<AxumStatusCode>>,
    }

    async fn fake_selection_service(state: FakeState) -> String {
        let app = Router::new()
            .route("/select_and_reserve", post(fake_select_and_reserve))
            .route("/workers", get(fake_workers))
            .route(
                "/reservations/{reservation_id}/prefill_complete",
                post(fake_prefill_complete),
            )
            .route("/reservations/{reservation_id}", delete(fake_delete))
            .with_state(state);
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });
        format!("http://{addr}")
    }

    async fn fake_select_and_reserve(
        State(state): State<FakeState>,
        Json(body): Json<serde_json::Value>,
    ) -> (AxumStatusCode, Json<serde_json::Value>) {
        state.select_requests.lock().await.push(body);
        let status = *state.select_status.lock().await;
        if !status.is_success() {
            return (
                status,
                Json(serde_json::json!({"error": "selection unavailable"})),
            );
        }
        (
            AxumStatusCode::OK,
            Json(serde_json::json!({
                "worker_id": 42,
                "dp_rank": 3,
                "endpoint": "http://10.0.0.42:8000",
                "block_size": 16,
                "overlap": {"longest_matched": 0, "gpu": 0, "dp": {"3": 0}, "cpu": 0, "disk": 0},
                "effective_prefill_tokens": 4,
                "model_name": "model",
                "tenant_id": "tenant",
                "reservation_id": "req-1"
            })),
        )
    }

    async fn fake_workers() -> Json<serde_json::Value> {
        Json(serde_json::json!([
            {"worker_id": 1, "lifecycle": "incomplete", "endpoint": "http://10.0.0.1:8000"},
            {"worker_id": 42, "lifecycle": "schedulable", "endpoint": "http://10.0.0.42:8000"}
        ]))
    }

    async fn fake_prefill_complete(
        State(state): State<FakeState>,
        Path(reservation_id): Path<String>,
    ) -> Json<serde_json::Value> {
        state.prefill_complete.lock().await.push(reservation_id);
        Json(serde_json::json!({"ok": true}))
    }

    async fn fake_delete(
        State(state): State<FakeState>,
        Path(reservation_id): Path<String>,
    ) -> Json<serde_json::Value> {
        state.deleted.lock().await.push(reservation_id);
        Json(serde_json::json!({"ok": true}))
    }

    #[tokio::test]
    async fn client_calls_select_and_lifecycle_endpoints() {
        let state = FakeState::default();
        let base_url = fake_selection_service(state.clone()).await;
        let client = SelectionClient::new(&base_url, Duration::from_secs(1)).unwrap();
        let response = client
            .select_and_reserve(&SelectAndReserveRequest {
                model_name: "model".into(),
                tenant_id: "tenant".into(),
                selection_id: Some("req-1".into()),
                reservation_id: Some("req-1".into()),
                token_ids: vec![1, 2, 3, 4],
                expected_output_tokens: Some(32),
                priority_jump: Some(5.0),
                strict_priority: Some(9),
                allowed_worker_ids: Some(HashSet::from([42])),
            })
            .await
            .unwrap();
        assert_eq!(response.worker_id, 42);
        assert_eq!(response.dp_rank, 3);

        client.mark_prefill_complete("req-1").await.unwrap();
        client.delete_reservation("req-1").await.unwrap();

        let requests = state.select_requests.lock().await;
        assert_eq!(requests[0]["selection_id"], "req-1");
        assert_eq!(requests[0]["reservation_id"], "req-1");
        assert_eq!(requests[0]["expected_output_tokens"], 32);
        assert_eq!(requests[0]["priority_jump"], 5.0);
        assert_eq!(requests[0]["strict_priority"], 9);
        assert_eq!(requests[0]["allowed_worker_ids"][0], 42);
        assert_eq!(*state.prefill_complete.lock().await, vec!["req-1"]);
        assert_eq!(*state.deleted.lock().await, vec!["req-1"]);
    }

    #[tokio::test]
    async fn client_surfaces_http_status_failures() {
        let state = FakeState::default();
        *state.select_status.lock().await = AxumStatusCode::SERVICE_UNAVAILABLE;
        let base_url = fake_selection_service(state).await;
        let client = SelectionClient::new(&base_url, Duration::from_secs(1)).unwrap();
        let error = client
            .select_and_reserve(&SelectAndReserveRequest {
                model_name: "model".into(),
                tenant_id: "tenant".into(),
                selection_id: None,
                reservation_id: None,
                token_ids: vec![1],
                expected_output_tokens: None,
                priority_jump: None,
                strict_priority: None,
                allowed_worker_ids: None,
            })
            .await
            .unwrap_err();
        assert!(error.to_string().contains("HTTP 503"));
    }

    #[tokio::test]
    async fn picker_maps_selection_response_to_pick_result() {
        let state = FakeState::default();
        let base_url = fake_selection_service(state.clone()).await;
        let picker = SelectionPicker::new(SelectionPickerConfig {
            selection_service_url: base_url,
            tenant_id: "tenant".into(),
            default_model_name: "fallback-model".into(),
            timeout: Duration::from_secs(1),
        })
        .unwrap();
        let result = picker
            .pick(
                &RequestInfo {
                    request_id: "req-1".into(),
                    headers: vec![],
                    body: br#"{
                        "model": "model",
                        "max_completion_tokens": 32,
                        "nvext": {
                            "token_data": [1, 2, 3, 4],
                            "agent_hints": {"priority": 5, "strict_priority": 9}
                        }
                    }"#
                    .to_vec(),
                    model: "model".into(),
                    candidate_subset: vec!["42".into()],
                },
                &[],
            )
            .await
            .unwrap();
        assert_eq!(result.endpoint, "10.0.0.42:8000");
        assert_eq!(
            result.headers,
            vec![
                ("x-worker-instance-id".into(), "42".into()),
                ("x-dp-rank".into(), "3".into()),
                ("x-dynamo-routing-mode".into(), "aggregated".into()),
            ]
        );
        assert_eq!(result.token_ids, Some(vec![1, 2, 3, 4]));
    }

    #[tokio::test]
    async fn picker_rejects_missing_token_data() {
        let picker = SelectionPicker {
            client: SelectionClient::new("http://127.0.0.1:1", Duration::from_secs(1)).unwrap(),
            tenant_id: "tenant".into(),
            default_model_name: "model".into(),
        };
        let error = picker
            .pick(
                &RequestInfo {
                    request_id: "req-1".into(),
                    headers: vec![],
                    body: br#"{"model":"model","messages":[]}"#.to_vec(),
                    model: "model".into(),
                    candidate_subset: vec![],
                },
                &[],
            )
            .await
            .unwrap_err();
        assert!(matches!(error, PickError::TokenizationFailed(_)));
    }

    #[tokio::test]
    async fn picker_routes_header_only_to_first_schedulable_worker() {
        let state = FakeState::default();
        let base_url = fake_selection_service(state).await;
        let picker = SelectionPicker::new(SelectionPickerConfig {
            selection_service_url: base_url,
            tenant_id: "tenant".into(),
            default_model_name: "model".into(),
            timeout: Duration::from_secs(1),
        })
        .unwrap();
        let result = picker
            .pick(
                &RequestInfo {
                    request_id: "req-1".into(),
                    headers: vec![],
                    body: vec![],
                    model: String::new(),
                    candidate_subset: vec![],
                },
                &[],
            )
            .await
            .unwrap();
        assert_eq!(result.endpoint, "10.0.0.42:8000");
        assert!(result.headers.is_empty());
        assert!(result.token_ids.is_none());
    }

    #[test]
    fn normalize_endpoint_strips_http_scheme_and_path() {
        assert_eq!(
            normalize_endpoint_for_envoy("http://worker.default.svc:8000/v1").unwrap(),
            "worker.default.svc:8000"
        );
        assert_eq!(
            normalize_endpoint_for_envoy("https://worker.default.svc").unwrap(),
            "worker.default.svc:443"
        );
        assert_eq!(
            normalize_endpoint_for_envoy("10.0.0.42:8000").unwrap(),
            "10.0.0.42:8000"
        );
    }

    #[test]
    fn candidate_subset_must_be_numeric() {
        let error = parse_candidate_subset(&["pod-a".to_string()]).unwrap_err();
        assert!(matches!(error, PickError::RoutingFailed(_)));
    }
}
