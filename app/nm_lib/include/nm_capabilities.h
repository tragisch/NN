/*
 * Copyright (c) 2026 @tragisch <https://github.com/tragisch>
 * SPDX-License-Identifier: MIT
 *
 * This file is part of a project licensed under the MIT License.
 * See the LICENSE file in the root directory for details.
 */

/* What nm_lib can run, as a GraphIR backend capability profile.
 *
 * The profile is plain JSON so that GraphIR (or any other tool) can read it
 * without linking nm_lib. GraphIR checks graphs against it before handing them
 * over; the conformance corpus tests that nm_lib and the profile agree. */
#ifndef NM_CAPABILITIES_H
#define NM_CAPABILITIES_H

#ifdef __cplusplus
extern "C" {
#endif

/* NUL-terminated JSON, valid for the lifetime of the program. */
const char *nm_capabilities_json(void);

#ifdef __cplusplus
}
#endif

#endif /* NM_CAPABILITIES_H */
