# API: Components

`nm_api.h` is built from a few public components. Use them directly when you
need more control than `nm_api.h` offers, for example to share weights between
processes or to train an imported graph with your own engine settings.

```text
nm_frontend  topology, shapes, initial weights  ->  graph
nm_engine_c  train and run a graph
nm_import_graphir / nm_export_graphir   GraphIR JSON  <->  graph
nm_girw      .girw weight packages
nm_weightset weights as versioned messages
nm_connector publisher / subscriber on a transport
nm_capabilities  what can run, as JSON
```

Bazel targets are `//app/nm_lib:nm_frontend`, `nm_engine_c`,
`nm_import_graphir`, `nm_export_graphir`, `nm_weights`, `nm_connector` and
`nm_capabilities`. All of them work on the `FloatMatrix` type from mMatrix,
so their programs include the mMatrix header directory as well.

## Frontend (`nm_frontend.h`)

```c
NmFrontendModel *model = nm_frontend_create((NmShape){.ndim = 1, .dims = {4}});
nm_frontend_add_dense(model, &(NmDenseConfig){.units = 8, .activation = NM_ACT_RELU});
nm_frontend_add_dense(model, &(NmDenseConfig){.units = 3, .activation = NM_ACT_SOFTMAX});
nm_frontend_finalize(model);          /* validates, draws weights, builds the graph */
NmGraphModel *graph = nm_frontend_graph(model);   /* owned by the model */
```

Also `nm_frontend_add_conv2d`, `nm_frontend_add_maxpool2d`,
`nm_frontend_add_flatten` and `nm_frontend_import_legacy`. Finalizing never
prepares a runtime.

## Engine (`nm_engine_c.h`)

```c
NmEngineCConfig cfg = nm_engine_c_defaults();   /* loss, optimizer, backend, ... */
NmEngineC *engine = nm_engine_c_create(&cfg);
nm_engine_c_prepare(engine, model);              /* or _prepare_graph(engine, graph) */
nm_engine_c_fit_graph(engine, graph, X, Y, &train_config);
nm_engine_c_predict_graph(engine, graph, X, &prediction);
```

- `nm_engine_c_prepare_graphir_json` imports a GraphIR file and prepares in one
  step.
- Training works for graphs made of Dense, Conv2D, MaxPool2D and Reshape
  layers, including imported ones. Weights are updated in place.
- `nm_engine_c_predict_inputs` and `nm_engine_c_predict_batches` take named,
  typed `NmInput` values (float32, int64, bool).
- `nm_engine_c_capabilities` returns a static description of the backend
  (determinism, training support, fused paths, prepared inference).
- The `NmEngineCConfig` carries optimizer hyperparameters that `NmFitOptions`
  does not expose (momentum, Adam betas, epsilon), `autograd_mode` and
  `batch_parallel` settings.

## Weight sets and connector

A weight set is the weights of a graph as one message: a 64-byte header plus a
`.girw` v2 package, so it can travel through shared memory, a file or any other
transport and be bound to a graph on the receiving side without copying.

```c
size_t size = nm_weightset_message_size(graph);
nm_weightset_write(graph, /*epoch*/ 1, /*version*/ 1, buffer, size);
/* receiver: */
nm_weightset_open(message, size, NM_GIRW_VERIFY_ALL, release, ctx, &set);
nm_engine_c_bind_weights(engine, graph, set);   /* between two predicts */
```

Rules: the structure id of the graph must equal the one in the message
(`nm_graph_structure_id`), a bound graph is read-only for `fit`, and a
changed epoch means the publisher restarted.

`nm_connector.h` adds publisher and subscriber objects with versioning,
structure checks and staleness detection on top of a pluggable `NmTransport`.
Shipped transports: an in-process queue (`nm_transport_memory_pair`) and a
directory of files (`nm_transport_file_sender` / `_receiver`).

```c
NmConnectorPublisher *pub = nm_connector_publisher_create(sender, graph);
nm_connector_publisher_publish(pub);               /* after training steps */

NmConnectorSubscriber *sub =
    nm_connector_subscriber_create(receiver, engine, graph, NULL);
nm_connector_subscriber_poll(sub);                 /* between predicts */
NmConnectorState state = nm_connector_subscriber_state(sub);
```

States: `WAITING`, `BOUND`, `STALE` (last weights stay), `REJECTING` (the last
message did not fit; previous weights stay).

## Capabilities

```c
const char *json = nm_capabilities_json();   /* valid for the program's lifetime */
```

A GraphIR backend profile as JSON: format versions 3 and 4, opsets 1 to 3,
data types (`float32`, `int64`, `bool`, plus promoted `bfloat16` and
`float16`), external weights supported, quantization not supported, and the
list of supported operations: `Input`, `Constant`, `MatMul`, `Add`, `Sub`,
`Mul`, `Div`, `ReLU`, `Sigmoid`, `Tanh`, `GELU`, `Softmax` (last axis),
`LayerNorm`, `Conv2D` (no dilation, one group, symmetric padding), `MaxPool2D`,
`Reshape`, `Transpose`, `Gather`, `Equal`, `Not`, `Where`, `Cast`, `Expand`,
`Slice`, `ReduceMax`, `Range`, `Squeeze`, `Unsqueeze`. The tool
`//app/nm_lib:print_capabilities` prints it; the conformance corpus
(`tests/fixtures/conformance`) checks that nm_lib and the profile agree.
