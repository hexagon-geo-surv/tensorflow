# TensorFlow CPU No-AVX Builder

Dockerized environment and automated scripts to build non-AVX CPU TensorFlow
pip wheels from source.

## Overview

Compiles TensorFlow targeting the Intel **Westmere** architecture
(`-march=westmere -Wno-sign-compare`). Westmere supports instruction sets up
to **SSE4.2** and **AES-NI**, but pre-dates **AVX**, ensuring full compatibility
with older CPUs and virtualized environments without AVX support.

## Files

- `Dockerfile`: Ubuntu 22.04 base with Clang-17, Python 3.11, Bazelisk, and
  required build toolchains for Linux.
- `build_tf_noavx.sh`: Linux in-container configuration and build script.
- `run_build.sh`: Host helper script for Linux builds with volume mounts.
- `build_tf_windows_noavx.bat`: Windows batch script for building wheels.
- `../windows/build_tf_windows_noavx.sh`: Windows Bash script for MSYS2/CI.
- `../windows/test_tf_windows_noavx.sh`: Script to test the built Windows wheel.

## Usage

### Linux (Dockerized)

From the root of the TensorFlow repository:

```bash
./tensorflow/tools/ci_build/noavx/run_build.sh
```

Or manually:

```bash
docker build -t tf-cpu-noavx-builder tensorflow/tools/ci_build/noavx
docker run --rm \
  --name tf-cpu-noavx-build-run \
  -v $(pwd):/tensorflow \
  -v /tmp/tf_wheel:/tf_wheel \
  -v /tmp/tf_bazel_cache:/root/.cache \
  tf-cpu-noavx-builder
```

### Windows (Dockerized)

From Windows Command Prompt (`cmd.exe`) or PowerShell with Docker Desktop running:

```bat
tensorflow\tools\ci_build\noavx\run_build.bat
```

This builds the Docker image, compiles the No-AVX wheel inside the container, verifies it with Python tests, and outputs the wheel to `build_output\`.

### Windows (Native without Docker)

From PowerShell or MSYS2 Bash at the repository root:

```bash
./tensorflow/tools/ci_build/windows/build_tf_windows_noavx.sh
./tensorflow/tools/ci_build/windows/test_tf_windows_noavx.sh
```

Or from Windows Command Prompt (`cmd.exe`):

```bat
tensorflow\tools\ci_build\windows\build_tf_windows_noavx.bat
```

The resulting `.whl` package will be placed in `build_output/` or `/tmp/tf_wheel/`.
