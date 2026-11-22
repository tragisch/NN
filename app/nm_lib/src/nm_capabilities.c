/*
 * Copyright (c) 2026 @tragisch <https://github.com/tragisch>
 * SPDX-License-Identifier: MIT
 *
 * This file is part of a project licensed under the MIT License.
 * See the LICENSE file in the root directory for details.
 */

#include "nm_capabilities.h"

/* Keep this in step with what nm_import_graphir.c, the execution plan and the
 * interpreter (nm_graph_interp.c) accept. int64 and bool values run on the
 * interpreter; the rule "float_inputs" marks the operations that take float
 * values only (arithmetic, activations, MatMul, normalization, pooling,
 * ReduceMax). tests/test_nm_conformance.c checks it against the corpus: every
 * "supported" case must run, every "unsupported" case must be rejected. */
static const char PROFILE[] =
    "{\n"
    "  \"profile_version\": 1,\n"
    "  \"backend\": \"nm_lib\",\n"
    "  \"graphir\": {\"format_versions\": [3, 4], \"opset_versions\": [1, 2, 3]},\n"
    "  \"dtypes\": [\"float32\", \"bfloat16\", \"float16\", \"int64\", \"bool\"],\n"
    "  \"promoted_dtypes\": [\"bfloat16\", \"float16\"],\n"
    "  \"supports_quantization\": false,\n"
    "  \"supports_external_weights\": true,\n"
    "  \"requires_constant_data\": true,\n"
    "  \"operations\": [\n"
    "    {\"name\": \"Input\"},\n"
    "    {\"name\": \"Constant\"},\n"
    "    {\"name\": \"MatMul\", \"rules\": [\"float_inputs\"]},\n"
    "    {\"name\": \"Add\", \"rules\": [\"float_inputs\"]},\n"
    "    {\"name\": \"Sub\", \"rules\": [\"float_inputs\"]},\n"
    "    {\"name\": \"Mul\", \"rules\": [\"float_inputs\"]},\n"
    "    {\"name\": \"Div\", \"rules\": [\"float_inputs\"]},\n"
    "    {\"name\": \"ReLU\", \"rules\": [\"float_inputs\"]},\n"
    "    {\"name\": \"Sigmoid\", \"rules\": [\"float_inputs\"]},\n"
    "    {\"name\": \"Tanh\", \"rules\": [\"float_inputs\"]},\n"
    "    {\"name\": \"GELU\", \"rules\": [\"float_inputs\"]},\n"
    "    {\"name\": \"Softmax\", \"rules\": [\"last_axis\", \"float_inputs\"]},\n"
    "    {\"name\": \"LayerNorm\", \"rules\": [\"float_inputs\"]},\n"
    "    {\"name\": \"Conv2D\",\n"
    "     \"attributes\": {\"dilations\": {\"allowed\": [[1, 1]]},\n"
    "                     \"groups\": {\"allowed\": [1]}},\n"
    "     \"rules\": [\"symmetric_pads\", \"float_inputs\"]},\n"
    "    {\"name\": \"MaxPool2D\", \"rules\": [\"symmetric_pads\", \"float_inputs\"]},\n"
    "    {\"name\": \"Reshape\"},\n"
    "    {\"name\": \"Transpose\"},\n"
    "    {\"name\": \"Gather\"},\n"
    "    {\"name\": \"Equal\"},\n"
    "    {\"name\": \"Not\"},\n"
    "    {\"name\": \"Where\"},\n"
    "    {\"name\": \"Cast\"},\n"
    "    {\"name\": \"Expand\"},\n"
    "    {\"name\": \"Slice\"},\n"
    "    {\"name\": \"ReduceMax\", \"rules\": [\"float_inputs\"]},\n"
    "    {\"name\": \"Range\"},\n"
    "    {\"name\": \"Squeeze\"},\n"
    "    {\"name\": \"Unsqueeze\"}\n"
    "  ]\n"
    "}\n";

const char *nm_capabilities_json(void) { return PROFILE; }
