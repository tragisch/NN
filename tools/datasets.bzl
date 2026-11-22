"""Macros to decompress http_file-fetched .gz datasets into raw IDX files."""

_GUNZIP = "gunzip -c $(location {src}) > $@"

def gunzip_file(name, repo, src_file, out = None):
    """Decompress @repo//:<src_file> into <out>.

    Args:
      name: genrule target name (must differ from out to avoid rule/file clash)
      repo: http_file repo name, e.g. "mnist_train_images_gz"
      src_file: filename inside the repo (the http_file downloaded_file_path)
      out: output filename (runfiles path component)
    """
    if out == None:
        out = src_file[:-3]  # strip .gz
    native.genrule(
        name = name,
        srcs = ["@" + repo + "//:" + src_file],
        outs = [out],
        cmd = _GUNZIP.format(src = "@" + repo + "//:" + src_file),
        visibility = ["//visibility:public"],
    )
