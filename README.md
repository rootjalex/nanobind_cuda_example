nanobind_cuda_example
================

```bash
module load cuda
pip install .
python tests/test_basic.py
```

This repository contains a tiny project showing how to create C++ bindings
using [nanobind](https://github.com/wjakob/nanobind) and
[scikit-build-core](https://scikit-build-core.readthedocs.io/en/latest/index.html) with CUDA. It is derived from [nanobind_example](https://github.com/wjakob/nanobind_example), and specifically set to build on a NVIDIA GeForce RTX 4090 (`-arch=sm_89`).

Installation
------------

1. Clone this repository
2. Run `pip install ./nanobind_cuda_example`

Afterwards, you should be able to issue the following commands (shown in an
interactive Python session):

```pycon
>>> import nanobind_cuda_example
>>> nanobind_cuda_example.add(1, 2)
3
```

See `tests/test_basic.py` for an example of using the GPU functionality via pytorch tensors.

CI Examples
-----------

The `.github/workflows` directory contains two continuous integration workflows
for GitHub Actions. The first one (`pip`) runs automatically after each commit
and ensures that packages can be built successfully and that tests pass.

The `wheels` workflow uses
[cibuildwheel](https://cibuildwheel.readthedocs.io/en/stable/) to automatically
produce binary wheels for a large variety of platforms. If a `pypi_password`
token is provided using GitHub Action's _secrets_ feature, this workflow can
even automatically upload packages on PyPI.


License
-------

_nanobind_ and this example repository are both provided under a BSD-style
license that can be found in the [LICENSE](./LICENSE) file. By using,
distributing, or contributing to this project, you agree to the terms and
conditions of this license.
