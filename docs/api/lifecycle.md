# API: Describing and Building a Network

## Ownership

| Object | Created by | Freed by | Notes |
| --- | --- | --- | --- |
| `NmSpec` | `nm_spec_new` | `nm_spec_free` | recipe only; reusable |
| `NmNetwork` | `nm_network_build`, `nm_network_load`, `nm_network_import_graphir` | `nm_network_free` | owns weights and engine |
| `NmTensor` data you filled | you | you | borrowed by the library |
| `NmTensor` from `predict` | the library | `nm_tensor_free` | |

`nm_spec_free(NULL)` and `nm_network_free(NULL)` are no-ops.

## Spec

```c
NmSpec *nm_spec_new(NmShape input);
NmStatus nm_spec_dense(NmSpec *spec, size_t units, NmActivation activation);
NmStatus nm_spec_conv2d(NmSpec *spec, size_t filters, size_t kernel,
                        size_t stride, size_t padding, NmActivation activation);
NmStatus nm_spec_maxpool2d(NmSpec *spec, size_t kernel, size_t stride);
NmStatus nm_spec_flatten(NmSpec *spec);
```

- `nm_spec_new` returns `NULL` unless the input has 1 to 3 non-zero
  dimensions.
- `nm_spec_dense` needs `units > 0`; `nm_spec_conv2d` needs positive
  `filters`, `kernel`, `stride`; `nm_spec_maxpool2d` needs positive `kernel`
  and `stride`. They return `NM_STATUS_INVALID_ARGUMENT` otherwise and set
  `nm_last_error()`.
- Whether the layers fit together (a convolution needs a
  `{channels, height, width}` input, a dense layer after a convolution needs a
  `nm_spec_flatten` first) is decided in `nm_network_build`.
- Not available: `AvgPool2D`, `BatchNorm2D`, dropout, recurrent layers.

## Build options

```c
typedef struct {
  uint64_t seed;        /* 0: keep the library's random state */
  bool deterministic;   /* same seed and data give the same weights */
  NmBackendType backend;
} NmBuildOptions;
```

`nm_build_options_default()` is `seed = 0`, `deterministic = true`,
`backend = NM_BACKEND_AUTO`. Passing `NULL` to `nm_network_build` means the
defaults.

A non-zero seed calls the library-wide `nm_set_seed`, so initial weights of
networks built later depend on the seed of the last non-zero build. The seed
also drives the shuffling in `nm_network_fit` of that network.

## Build

```c
NmNetwork *nm_network_build(const NmSpec *spec, const NmBuildOptions *options);
```

Validates the layers, draws the weights and prepares the engine. Failures
return `NULL`, for example:

- `nm_network_build: the spec has no layers`
- `nm_network_build: layer 1 rejected (invalid argument); check shapes`

For a shape problem the logger prints the precise cause on stderr (for
example `Conv2D requires spatial input (ndim==3, C×H×W)`).

## Inspecting

```c
size_t nm_network_parameter_count(const NmNetwork *network);
size_t nm_network_summary(const NmNetwork *network, char *buffer, size_t size);
```

`nm_network_summary` writes a table of layers, output shapes (without the
batch dimension) and parameter counts. Like `snprintf` it returns the length
the full text needs, and `buffer` may be `NULL` to ask for the size:

```c
size_t need = nm_network_summary(net, NULL, 0);
char *text = malloc(need + 1);
nm_network_summary(net, text, need + 1);
puts(text);
free(text);
```

```text
Layer                    Output               Params
Dense 32 + Tanh          [32]                     64
Dense 32 + Tanh          [32]                   1056
Dense 1                  [1]                      33
Total parameters: 1153
```
