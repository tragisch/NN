/*
 * Copyright (c) 2026 @tragisch <https://github.com/tragisch>
 * SPDX-License-Identifier: MIT
 *
 * This file is part of a project licensed under the MIT License.
 * See the LICENSE file in the root directory for details.
 */

/* Weight sets: the weights of a graph as one message that can travel through
 * shared memory, a file or any other transport, and be bound to a graph on the
 * receiving side without copying.
 *
 * Message layout (little endian, 64-byte aligned buffer):
 *   [64-byte header][.girw version 2 package]
 * The package comes from nm_girw.h, so blobs are 64-byte aligned, checksummed
 * and byte-compatible with GraphIR's weight packages. The header names the
 * message, a weights version and the id of the graph structure the weights
 * belong to; a receiver rejects weights for another structure. */
#ifndef NM_WEIGHTSET_H
#define NM_WEIGHTSET_H

#include "nm_girw.h"
#include "nm_types.h"
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define NM_WEIGHTSET_ALIGNMENT 64U

typedef void (*NmWeightRelease)(void *context);
typedef struct NmWeightSet NmWeightSet;

/* Identifies the structure of a graph (values, shapes, nodes and their
 * attributes) and ignores the weights. Equal ids mean weights fit. */
uint64_t nm_graph_structure_id(const NmGraphModel *graph);

/* Size of the message for the graph's current weights (0 if there are none). */
size_t nm_weightset_message_size(const NmGraphModel *graph);

/* Writes the message into `out` (at least nm_weightset_message_size bytes,
 * 64-byte aligned). `epoch` identifies one publisher run and `version` counts
 * its sets (it should increase with every set). A receiver treats a changed
 * epoch as a restarted publisher, so versions may start over. */
NmStatus nm_weightset_write(const NmGraphModel *graph, uint64_t epoch,
                            uint64_t version, void *out, size_t out_size);

/* Opens a received message without copying. `release(release_context)` is
 * called once when the last reference is dropped, to hand the memory back to
 * its transport. The message must be 64-byte aligned and stay unchanged until
 * then. NM_STATUS_FORMAT_ERROR for malformed messages and checksum mismatches.
 * On failure the release callback is not called. */
NmStatus nm_weightset_open(const void *message, size_t size, NmGirwVerify verify,
                           NmWeightRelease release, void *release_context,
                           NmWeightSet **out_set);

uint64_t nm_weightset_epoch(const NmWeightSet *set);
uint64_t nm_weightset_version(const NmWeightSet *set);
uint64_t nm_weightset_structure_id(const NmWeightSet *set);

void nm_weightset_retain(NmWeightSet *set);
void nm_weightset_release(NmWeightSet *set); /* drops one reference */

/* Makes the graph's constants use the set's memory. Fails without changing the
 * graph if the structure id differs or a tensor does not match its constant. The
 * graph keeps its own reference to the set; a graph bound this way is read-only
 * for training. Weights of a graph that already was bound are replaced. */
NmStatus nm_graph_bind_weights(NmGraphModel *graph, NmWeightSet *set);

/* Version and epoch of the bound set, 0 if the graph owns its weights. */
uint64_t nm_graph_bound_version(const NmGraphModel *graph);
uint64_t nm_graph_bound_epoch(const NmGraphModel *graph);

#ifdef __cplusplus
}
#endif

#endif /* NM_WEIGHTSET_H */
