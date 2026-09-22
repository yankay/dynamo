/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package v1alpha1

import (
	"encoding/json"
	"reflect"
	"testing"

	"github.com/stretchr/testify/require"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	"k8s.io/utils/ptr"
)

func TestExtraPodSpec_MarshalJSON(t *testing.T) {
	t.Log("Define the wire contract for direct ExtraPodSpec serialization")
	tests := []struct {
		name     string
		spec     ExtraPodSpec
		wantJSON string
	}{
		{
			name: "nil PodSpec with mainContainer",
			spec: ExtraPodSpec{
				PodSpec:       nil,
				MainContainer: &corev1.Container{Name: "main"},
			},
			wantJSON: `{"mainContainer":{"name":"main"}}`,
		},
		{
			name: "mainContainer omits empty name and resources",
			spec: ExtraPodSpec{
				MainContainer: &corev1.Container{Image: "worker:1"},
			},
			wantJSON: `{"mainContainer":{"image":"worker:1"}}`,
		},
		{
			name: "empty mainContainer remains present",
			spec: ExtraPodSpec{
				MainContainer: &corev1.Container{},
			},
			wantJSON: `{"mainContainer":{}}`,
		},
		{
			name: "nil Containers omits containers key entirely",
			spec: ExtraPodSpec{
				PodSpec: &corev1.PodSpec{
					NodeSelector: map[string]string{"gpu": annotationTrue},
				},
			},
			wantJSON: `{"nodeSelector":{"gpu":"true"}}`,
		},
		{
			name: "empty Containers omits containers key",
			spec: ExtraPodSpec{
				PodSpec: &corev1.PodSpec{
					Containers: []corev1.Container{},
				},
			},
			wantJSON: `{}`,
		},
		{
			name: "populated Containers are serialized",
			spec: ExtraPodSpec{
				PodSpec: &corev1.PodSpec{
					Containers: []corev1.Container{{Name: "sidecar"}},
				},
			},
			wantJSON: `{"containers":[{"name":"sidecar"}]}`,
		},
		{
			name: "required names in container lists are not normalized",
			spec: ExtraPodSpec{
				PodSpec: &corev1.PodSpec{
					Containers: []corev1.Container{{Image: "sidecar:1"}},
				},
			},
			wantJSON: `{"containers":[{"name":"","image":"sidecar:1"}]}`,
		},
		{
			name: "init and ephemeral containers omit empty resources",
			spec: ExtraPodSpec{
				PodSpec: &corev1.PodSpec{
					InitContainers: []corev1.Container{{Name: "init"}},
					EphemeralContainers: []corev1.EphemeralContainer{{
						EphemeralContainerCommon: corev1.EphemeralContainerCommon{Name: "debug"},
					}},
				},
			},
			wantJSON: `{"initContainers":[{"name":"init"}],"ephemeralContainers":[{"name":"debug"}]}`,
		},
		{
			name: "non-empty container resources are preserved",
			spec: ExtraPodSpec{
				MainContainer: &corev1.Container{
					Resources: corev1.ResourceRequirements{
						Requests: corev1.ResourceList{corev1.ResourceCPU: resource.MustParse("1")},
					},
				},
				PodSpec: &corev1.PodSpec{
					Containers: []corev1.Container{{
						Name: "sidecar",
						Resources: corev1.ResourceRequirements{
							Claims: []corev1.ResourceClaim{{Name: "gpu"}},
						},
					}},
					InitContainers: []corev1.Container{{
						Name: "init",
						Resources: corev1.ResourceRequirements{
							Limits: corev1.ResourceList{corev1.ResourceMemory: resource.MustParse("1Gi")},
						},
					}},
					EphemeralContainers: []corev1.EphemeralContainer{{
						EphemeralContainerCommon: corev1.EphemeralContainerCommon{
							Name: "debug",
							Resources: corev1.ResourceRequirements{
								Requests: corev1.ResourceList{corev1.ResourceCPU: resource.MustParse("1")},
							},
						},
					}},
				},
			},
			wantJSON: `{"mainContainer":{"resources":{"requests":{"cpu":"1"}}},"containers":[{"name":"sidecar","resources":{"claims":[{"name":"gpu"}]}}],"initContainers":[{"name":"init","resources":{"limits":{"memory":"1Gi"}}}],"ephemeralContainers":[{"name":"debug","resources":{"requests":{"cpu":"1"}}}]}`,
		},
		{
			name: "unrelated empty pod fields are preserved",
			spec: ExtraPodSpec{
				PodSpec: &corev1.PodSpec{
					Volumes: []corev1.Volume{{
						Name:         "scratch",
						VolumeSource: corev1.VolumeSource{EmptyDir: &corev1.EmptyDirVolumeSource{}},
					}},
				},
			},
			wantJSON: `{"volumes":[{"name":"scratch","emptyDir":{}}]}`,
		},
		{
			name: "tolerations preserved without containers",
			spec: ExtraPodSpec{
				PodSpec: &corev1.PodSpec{
					Tolerations: []corev1.Toleration{{Key: "nvidia.com/gpu"}},
				},
			},
			wantJSON: `{"tolerations":[{"key":"nvidia.com/gpu"}]}`,
		},
		{
			name: "mainContainer alongside PodSpec fields",
			spec: ExtraPodSpec{
				PodSpec: &corev1.PodSpec{
					NodeSelector: map[string]string{"zone": "us-east"},
				},
				MainContainer: &corev1.Container{Name: "main"},
			},
			wantJSON: `{"nodeSelector":{"zone":"us-east"},"mainContainer":{"name":"main"}}`,
		},
		{
			name:     "nil PodSpec and nil mainContainer",
			spec:     ExtraPodSpec{},
			wantJSON: `{}`,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Log("Marshal both a value and a pointer without changing the input")
			before := tt.spec.DeepCopy()
			for _, input := range []any{tt.spec, &tt.spec} {
				got, err := json.Marshal(input)
				require.NoError(t, err)
				require.JSONEq(t, tt.wantJSON, string(got))
			}

			t.Log("Verify marshaling left the caller-owned pod and containers unchanged")
			if !reflect.DeepEqual(before, &tt.spec) {
				t.Fatal("MarshalJSON mutated its input")
			}
		})
	}
}

func TestExtraPodSpec_MarshalJSONPreservesLargeIntegers(t *testing.T) {
	t.Log("Build a pod with integers beyond float64's exact range")
	const large int64 = 9007199254740993
	spec := ExtraPodSpec{
		PodSpec: &corev1.PodSpec{
			ActiveDeadlineSeconds:         ptr.To(large),
			TerminationGracePeriodSeconds: ptr.To(large),
		},
		MainContainer: &corev1.Container{Image: "worker:1"},
	}

	t.Log("Marshal through a map value to exercise recursive value-receiver dispatch")
	raw, err := json.Marshal(map[string]ExtraPodSpec{"pod": spec})
	require.NoError(t, err)
	var restored map[string]ExtraPodSpec
	require.NoError(t, json.Unmarshal(raw, &restored))

	t.Log("Verify both integers survive exactly and the container is normalized")
	pod := restored["pod"]
	require.NotNil(t, pod.PodSpec)
	require.Equal(t, ptr.To(large), pod.ActiveDeadlineSeconds)
	require.Equal(t, ptr.To(large), pod.TerminationGracePeriodSeconds)
	require.NotContains(t, string(raw), `"resources"`)
	require.NotContains(t, string(raw), `"name"`)
}

func TestExtraPodSpec_MarshalJSON_RoundTrip(t *testing.T) {
	original := ExtraPodSpec{
		PodSpec: &corev1.PodSpec{
			Containers:   []corev1.Container{{Name: "main", Image: "nginx"}},
			NodeSelector: map[string]string{"gpu": annotationTrue},
			Tolerations:  []corev1.Toleration{{Key: "nvidia.com/gpu", Operator: corev1.TolerationOpExists}},
		},
		MainContainer: &corev1.Container{Name: "override", Image: "custom"},
	}

	data, err := json.Marshal(original)
	if err != nil {
		t.Fatalf("MarshalJSON() error = %v", err)
	}

	var restored ExtraPodSpec
	if err := json.Unmarshal(data, &restored); err != nil {
		t.Fatalf("UnmarshalJSON() error = %v", err)
	}

	if !reflect.DeepEqual(original.PodSpec, restored.PodSpec) {
		t.Errorf("round-trip: PodSpec mismatch\n got: %+v\nwant: %+v", restored.PodSpec, original.PodSpec)
	}

	if !reflect.DeepEqual(original.MainContainer, restored.MainContainer) {
		t.Errorf("round-trip: MainContainer mismatch\n got: %+v\nwant: %+v", restored.MainContainer, original.MainContainer)
	}
}
