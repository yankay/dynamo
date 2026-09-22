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
	"bytes"
	"encoding/json"

	corev1 "k8s.io/api/core/v1"
)

// MarshalJSON preserves sparse container fields, including after v1beta1 storage
// conversion. A value receiver also normalizes ExtraPodSpec values in maps and
// recursively marshaled DGD/DCD specs without root-level traversal.
func (e ExtraPodSpec) MarshalJSON() ([]byte, error) {
	// Shadow PodSpec.Containers: its required field otherwise emits null, which
	// the CRD schema rejects. The alias avoids inheriting PodSpec marshal methods.
	type PodSpecAlias corev1.PodSpec
	aux := struct {
		*PodSpecAlias `json:",inline"`
		Containers    []corev1.Container `json:"containers,omitempty"`
		MainContainer *corev1.Container  `json:"mainContainer,omitempty"`
	}{
		MainContainer: e.MainContainer,
	}
	if e.PodSpec != nil {
		a := PodSpecAlias(*e.PodSpec)
		aux.PodSpecAlias = &a
		aux.Containers = e.PodSpec.Containers
	}

	// Normalize serialized fields without mutating the caller-owned pod or containers.
	raw, err := json.Marshal(aux)
	if err != nil {
		return nil, err
	}

	// PodSpec contains int64 fields that must not lose precision through float64.
	decoder := json.NewDecoder(bytes.NewReader(raw))
	decoder.UseNumber()
	var pod map[string]any
	if err := decoder.Decode(&pod); err != nil {
		return nil, err
	}

	// MainContainer has no required name and conversion clears its synthetic one.
	if mainContainer, ok := pod["mainContainer"].(map[string]any); ok {
		if name, ok := mainContainer["name"].(string); ok && name == "" {
			delete(mainContainer, "name")
		}
		removeEmptyContainerResourcesJSON(mainContainer)
	}

	// Native container lists also materialize zero ResourceRequirements as {}.
	for _, field := range []string{"containers", "initContainers", "ephemeralContainers"} {
		containers, ok := pod[field].([]any)
		if !ok {
			continue
		}
		for _, container := range containers {
			if container, ok := container.(map[string]any); ok {
				removeEmptyContainerResourcesJSON(container)
			}
		}
	}

	return json.Marshal(pod)
}

func removeEmptyContainerResourcesJSON(container map[string]any) {
	resources, ok := container["resources"].(map[string]any)
	if ok && len(resources) == 0 {
		delete(container, "resources")
	}
}
