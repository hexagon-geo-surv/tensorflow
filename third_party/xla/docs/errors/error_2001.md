# Error code: E201

**Category:** Unsupported RHS DataType on Hardware

This error occurs when the data type used for the **Right-Hand Side (RHS)**
operand in a matrix multiplication (e.g., `jax.lax.dot_general`, `jax.lax.conv`,
`jax.numpy.matmul`, or the `@` operator) is not natively supported by the
specific TPU generation being used.

**Sample Error Messages:**

```
INTERNAL: Mosaic failed to compile TPU kernel: Unsupported matmul RHS type on target: 'vector<256x256xi8>' 20

...

The MLIR operation involved:
%13440 = "tpu.matmul"(%13435, %13437, %13439) <dimension_numbers = #tpu.dot_dimension_numbers<...

```

**XLA Backends:** TPU

## Overview

The TPU's Matrix Multiply Unit (MXU) natively supports BFloat16 and Float32
operations on all hardware generations.

However, native support for quantized data types (e.g., Int4, Int8, or Float8)
varies by hardware generation. This error is triggered when your kernel attempts
to map a matrix multiplication to the MXU using a data type that your specific
TPU generation does not have the physical circuitry to execute.

This error typically indicates that the compiler's **Canonicalization** pass—
which attempts to automatically convert unsupported types into supported ones
(e.g., via software emulation)—was unable to find a valid conversion rule or
was prevented from doing so because **Compatibility Mode** was disabled.

## Debugging

To resolve this error, you must align your data types with the capabilities of
your hardware. You have following options:

### 1. Cast to Native Types {#cast-native}
The most reliable fix is to manually cast your operands to a hardware-supported
datatype (like `BFloat16` or `Float32`) inside your kernel code before the
matmul operation.

* **Why:** `BFloat16` is the universal data type supported natively by the
MXU on all TPU generations.
* **Trade-off:** This consumes extra memory bandwidth and VPU (Vector
Processing Unit) cycles to perform the cast, but it guarantees your kernel
will run on the current hardware.

### 2. Enable Compatibility Mode {#compat-mode}
You can allow the compiler to automatically handle these type issues by enabling
**Compatibility Mode** via the `--xla_mosaic_compat_mode=true` flag. This acts
as a "polyfill," injecting software emulation sequences for operations your
hardware does not natively support.

**What Compatibility Mode enables:**

* **Mixed-Precision MatMuls:** Allows mixing Integer operands with Float
accumulators by automatically inserting cast operations (e.g., extending
integers to `Float32` before the matmul).
* **Low-Precision Emulation:** Emulates unsupported types like `4-bit`
floating point (`4E2M1FN`) or `8-bit` floating point (`8E4M3FN`) by
extending them to supported types like `BFloat16` or `Float32` before
execution.
* **Scalar & Vector Fixes:** Enables complex integer arithmetic (like
`vector<i16>` addition) and `select` operations on narrow types (e.g.,
8-bit or 16-bit) by routing them through 32-bit intermediate logic.

This mode prioritizes compatibility over peak performance. Emulation requires
additional instructions to convert data formats before the MXU can operate on
them.

### 3. Upgrade Hardware or Request Support {#upgrade}

If your algorithm strictly requires native performance for types like `Int4` or
`Float8` without the overhead of casting or emulation, you will need to run on
a newer TPU generation with native support.

**Feature Request:** If you believe your hardware supports this operation, or
if the compiler is missing a valid emulation path even in Compatibility Mode,
please file a feature request.
