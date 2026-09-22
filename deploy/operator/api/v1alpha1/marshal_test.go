/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
	"testing"

	"github.com/google/go-cmp/cmp"
	"github.com/stretchr/testify/require"
	"sigs.k8s.io/controller-runtime/pkg/conversion"

	v1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
)

func TestV1alpha1WireShapeSurvivesV1beta1StorageMigration(t *testing.T) {
	const podJSON = `{"mainContainer":{"image":"busybox"},"containers":[{"name":"empty-sidecar"},{"name":"resource-sidecar","resources":{"claims":[{"name":"sidecar-gpu"}]}}],"initContainers":[{"name":"init"}],"ephemeralContainers":[{"name":"debug"}]}`
	tests := []struct {
		name       string
		rawAlpha   string
		alpha      conversion.Convertible
		hub        conversion.Hub
		storedHub  conversion.Hub
		afterAlpha conversion.Convertible
	}{
		{
			name:       "DynamoGraphDeployment",
			rawAlpha:   `{"apiVersion":"nvidia.com/v1alpha1","kind":"DynamoGraphDeployment","metadata":{"name":"preupgrade-spec-probe","namespace":"dynamo-cloud"},"spec":{"services":{"Frontend":{"replicas":0,"componentType":"frontend","extraPodSpec":{"mainContainer":{"image":"busybox:1.36"}}}}}}`,
			alpha:      &DynamoGraphDeployment{},
			hub:        &v1beta1.DynamoGraphDeployment{},
			storedHub:  &v1beta1.DynamoGraphDeployment{},
			afterAlpha: &DynamoGraphDeployment{},
		},
		{
			name:       "DynamoComponentDeployment",
			rawAlpha:   `{"apiVersion":"nvidia.com/v1alpha1","kind":"DynamoComponentDeployment","metadata":{"name":"preupgrade-component","namespace":"dynamo-cloud"},"spec":{"componentType":"frontend","extraPodSpec":{"mainContainer":{"image":"busybox:1.36"}},"replicas":0}}`,
			alpha:      &DynamoComponentDeployment{},
			hub:        &v1beta1.DynamoComponentDeployment{},
			storedHub:  &v1beta1.DynamoComponentDeployment{},
			afterAlpha: &DynamoComponentDeployment{},
		},
		{
			name:       "DynamoGraphDeployment with all container lists and multiple services",
			rawAlpha:   `{"spec":{"services":{"frontend":{"extraPodSpec":` + podJSON + `},"worker":{"extraPodSpec":` + podJSON + `}}}}`,
			alpha:      &DynamoGraphDeployment{},
			hub:        &v1beta1.DynamoGraphDeployment{},
			storedHub:  &v1beta1.DynamoGraphDeployment{},
			afterAlpha: &DynamoGraphDeployment{},
		},
		{
			name:       "DynamoComponentDeployment with all container lists",
			rawAlpha:   `{"spec":{"extraPodSpec":` + podJSON + `}}`,
			alpha:      &DynamoComponentDeployment{},
			hub:        &v1beta1.DynamoComponentDeployment{},
			storedHub:  &v1beta1.DynamoComponentDeployment{},
			afterAlpha: &DynamoComponentDeployment{},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Log("Decode the sparse v1alpha1 object and retain its original wire-level spec")
			var before map[string]any
			if err := json.Unmarshal([]byte(tt.rawAlpha), &before); err != nil {
				t.Fatalf("unmarshal raw v1alpha1 object: %v", err)
			}
			if err := json.Unmarshal([]byte(tt.rawAlpha), tt.alpha); err != nil {
				t.Fatalf("unmarshal typed v1alpha1 object: %v", err)
			}

			t.Log("Verify recursive marshaling preserves the original sparse spec before conversion")
			directJSON, err := json.Marshal(tt.alpha)
			require.NoError(t, err)
			var direct map[string]any
			require.NoError(t, json.Unmarshal(directJSON, &direct))
			require.Equal(t, before["spec"], direct["spec"])

			t.Log("Convert the object to v1beta1 and round-trip it through storage JSON")
			if err := tt.alpha.ConvertTo(tt.hub); err != nil {
				t.Fatalf("ConvertTo() error = %v", err)
			}
			hubJSON, err := json.Marshal(tt.hub)
			if err != nil {
				t.Fatalf("marshal v1beta1 object: %v", err)
			}
			if err := json.Unmarshal(hubJSON, tt.storedHub); err != nil {
				t.Fatalf("unmarshal stored v1beta1 object: %v", err)
			}

			t.Log("Convert the stored object back to the v1alpha1 API wire representation")
			if err := tt.afterAlpha.ConvertFrom(tt.storedHub); err != nil {
				t.Fatalf("ConvertFrom() error = %v", err)
			}
			afterJSON, err := json.Marshal(tt.afterAlpha)
			if err != nil {
				t.Fatalf("marshal converted v1alpha1 object: %v", err)
			}
			var after map[string]any
			if err := json.Unmarshal(afterJSON, &after); err != nil {
				t.Fatalf("unmarshal converted v1alpha1 object: %v", err)
			}

			t.Log("Verify storage migration preserves the original sparse spec shape")
			if diff := cmp.Diff(before["spec"], after["spec"]); diff != "" {
				t.Fatalf("v1alpha1 spec changed across v1beta1 storage migration (-want +got):\n%s", diff)
			}
		})
	}
}
