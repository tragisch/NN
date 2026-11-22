load("@mkdocs//:defs.bzl", "mkdocs_build", "mkdocs_config", "mkdocs_serve")
load("@rules_docs//docgen:defs.bzl", "docs")
load("@rules_python//python:pip.bzl", "compile_pip_requirements")

alias(
	name = "nm",
	actual = "//app/nm_lib:nm",
)

compile_pip_requirements(
	name = "requirements",
	src = "docs/requirements.in",
	requirements_txt = "docs/requirements.txt",
	tags = ["manual"],
)

docs(
	name = "docs_content",
	title = "NN Library Documentation",
	test = False,
	nav = {
		"docs/index.md": "Home",
		"docs/getting-started/installation.md": "Getting Started / Installation",
		"docs/getting-started/first-network.md": "Getting Started / First Network",
		"docs/getting-started/library-usage.md": "Getting Started / Library Usage",
		"docs/api/overview.md": "API / Overview",
		"docs/api/lifecycle.md": "API / Describing and Building",
		"docs/api/training.md": "API / Training & Inference",
		"docs/api/activations.md": "API / Activations",
		"docs/api/losses.md": "API / Losses",
		"docs/api/metrics.md": "API / Metrics",
		"docs/api/serialization.md": "API / Saving, Loading, GraphIR",
		"docs/api/components.md": "API / Components",
		"docs/api/optimization.md": "API / Optimization",
		"docs/examples/mnist.md": "Examples / MNIST",
		"docs/examples/regression.md": "Examples / Regression",
		"docs/performance/backends.md": "Performance / Backends",
		"docs/legacy/index.md": "Legacy / nm.h Overview",
		"docs/legacy/lifecycle.md": "Legacy / Lifecycle",
		"docs/legacy/training.md": "Legacy / Training & Inference",
		"docs/legacy/activations.md": "Legacy / Activations",
		"docs/legacy/losses.md": "Legacy / Losses",
		"docs/legacy/metrics.md": "Legacy / Metrics",
		"docs/legacy/serialization-nms3.md": "Legacy / Serialization (NMS3)",
		"docs/legacy/mnist.md": "Legacy / MNIST",
		"docs/faq.md": "FAQ",
	},
	visibility = ["//visibility:public"],
)

mkdocs_config(
	name = "mkdocs_config",
	docs = ":docs_content",
	mkdocs_base = "docs/mkdocs.tpl.yaml",
)

mkdocs_build(
	name = "mkdocs",
	config = ":mkdocs_config",
	docs = [":docs_content"],
	site_dir = "site",
	visibility = ["//visibility:public"],
)

mkdocs_serve(
	name = "mkdocs.serve",
	config = ":mkdocs_config",
	docs = [":docs_content"],
	visibility = ["//visibility:public"],
)

exports_files([
    "mnist_test_inputs.bin",
    "mnist_test_labels.bin",
    "mnist_cnn.graphir.json",
])
