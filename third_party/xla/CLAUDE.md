# AI Assistant Guidelines for OpenXLA (`openxla/xla`)

Welcome to the OpenXLA (`openxla/xla`) repository. This guide aggregates the
code review, readability, compiler architecture, compatibility, and testing
standards used across the OpenXLA team. Follow these rules when writing code,
reviewing changes, and preparing Pull Requests (PRs).

## 1. Repository Layout, Build, and Dependencies

*   **Directory structure**:
    *   [`xla/`](xla/): Core XLA compiler, runtime, and backends (`xla/hlo/`,
        `xla/service/`, `xla/backends/gpu/`, `xla/backends/cpu/`,
        `xla/stream_executor/`, `xla/pjrt/`, `xla/codegen/`, `xla/python/`,
        `xla/tsl/`). See [`xla/AGENTS.md`](xla/AGENTS.md) for directory-level
        details.
    *   [`build_tools/`](build_tools/): CI scripts, build configurations, and
        developer tooling.
    *   [`docs/`](docs/): User-facing and developer documentation.
    *   [`third_party/`](third_party/): Vendored dependencies, patches (such as
        `third_party/shardy/temporary.patch` and
        `third_party/stablehlo/temporary.patch`), and Bazel repository rules.
*   **Include paths**: Always include headers relative to the repository root
    (for example, `#include "xla/hlo/ir/hlo_module.h"`).
*   **Building and testing with Bazel**:
    *   Run unit tests with `bazel test //xla/...` (for example,
        `bazel test //xla/service:hlo_verifier_test`).
    *   Pass `--config=cuda` or `--config=rocm` when building or testing GPU
        targets, and `-c opt` when running performance-sensitive tests or
        benchmarks.
*   **XLA Bazel rules**:
    *   Use `xla_cc_test` (loaded from `//xla:xla.default.bzl`) for
        hardware-independent C++ unit tests. Raw `cc_test` is disallowed in
        `xla/`.
    *   Use `xla_test` (loaded from `//xla/tests:build_defs.bzl`) for execution
        tests that require a configured hardware backend (GPU or CPU).
    *   Use `tf_proto_library` (loaded from
        `//xla/tsl/platform:build_config.bzl`) for `.proto` definitions instead
        of `proto_library` or `cc_proto_library`.
    *   Keep `BUILD` dependencies minimal (include what you use; remove unused
        `deps`) and restrict target visibility to the narrowest scope needed.
*   **License headers**: Start every new `.h`, `.cc`, `.td`, `.py`, `.bzl`, and
    `BUILD` file with the standard OpenXLA Apache 2.0 license header using the
    current 4-digit year (`Copyright 2026 The OpenXLA Authors.`).

## 2. PR Scoping, Minimality, and Description Standards

*   **Solve a real problem with falsifiable verification**: Every PR must
    address a concrete bug, performance bottleneck, or feature requirement and
    include a test or benchmark that fails (or regresses) without the change and
    passes with it.
*   **One logical change per PR**:
    *   Keep PRs small and self-contained.
    *   Never mix pure refactorings, formatting sweeps, file moves, or
        tangential cleanups with behavioral changes. Land prerequisite
        refactorings or helper utilities in separate PRs first.
    *   Do not modify unrelated surrounding code, comments, or formatting.
*   **Reproducer-first bugfix workflow**:
    1.  Write a minimal unit test that reproduces the bug.
    2.  Verify that the test fails on the unpatched codebase for the expected
        reason.
    3.  Implement the minimal root-cause fix and verify the test passes.
    4.  Never hide bugs by weakening assertions, widening numerical tolerances
        without mathematical justification, or adding `GTEST_SKIP()`.
*   **Minimality and anti-bloat discipline**:
    *   Every added line, branch, parameter, and helper function must earn its
        place (YAGNI).
    *   Search the current module, `xla/`, `xla/tsl/`, and Abseil for existing
        utilities before writing new helpers.
    *   Inline single-use trivial wrapper functions and single-use temporary
        variables unless the variable name documents a complex expression.
    *   Do not add speculative configuration flags, unused abstractions, or
        backward-compatibility shims for internal APIs with no external callers.
    *   Do not add silent fallback branches or defensive runtime checks for
        invariants already guaranteed by the type system or upstream passes
        (such as `HloVerifier`). Enforce true internal invariants with
        `TF_RET_CHECK` or `CHECK`.
    *   Remove all debug logging, commented-out code, and resolved `TODO`s
        before opening a PR. Any remaining `TODO` must reference a tracking
        issue.
*   **PR title and description**:
    *   **Title**: Prefix with the component tag in brackets followed by a
        concise, imperative summary (for example,
        `[XLA:GPU] Skip Triton dot emitter when contracting dimension is zero`
        or `[XLA:CPU] Preserve frontend attributes in YNNPACK fusions`).
    *   **Description**: State clearly:
        1.  **Status quo and problem**: What the system currently does and why
            it fails or is insufficient.
        2.  **Design choices**: The technical approach taken and why.
        3.  **Trade-offs and alternatives**: Any compromises accepted and why
            alternative approaches were rejected.
    *   Explain *why* the change is made rather than listing a file-by-file
        summary of the diff. Omit obsolete history from earlier draft iterations
        and avoid AI filler or exaggerated language.
    *   **Performance changes**: When modifying cost models, heuristic
        thresholds, tiling search spaces, or kernels, include baseline vs.
        candidate benchmark measurements and specify the target hardware.

## 3. C++17 Standard, Coding Style, and Readability

OpenXLA follows the
[Google C++ Style Guide](https://google.github.io/styleguide/cppguide.html).

*   **Strict C++17 compatibility (`-std=c++17`)**:
    *   The codebase compiles with `-std=c++17` and `-Wc++17-compat-pedantic`.
        Do **not** use C++20 or C++23 features:
        *   Use `absl::Span` (`absl/types/span.h`), never `std::span`.
        *   Do not call `.contains()` on `std::map`, `std::set`,
            `std::unordered_map`, or `std::unordered_set`. Prefer
            `absl::flat_hash_map` and `absl::flat_hash_set` (which provide
            `.contains()` in C++17), or use `.find(k) != .end()`.
        *   Use `absl::StartsWith` / `absl::EndsWith` (`absl/strings/match.h`),
            never `std::string::starts_with` / `ends_with`.
        *   Use `absl::StrCat` / `absl::StrFormat`, never `std::format`.
        *   Use `absl::bit_cast` (`absl/base/casts.h`), never `std::bit_cast`.
        *   Do not use C++20 `concept`s, `requires` clauses, `<ranges>`,
            `std::erase_if`, `using enum`, or non-C++17 designated initializer
            extensions.
*   **Control flow and braces**:
    *   Always enclose `if`, `for`, and `while` bodies in `{}` braces, even for
        single-line statements (this prevents macro expansion bugs with
        `ABSL_RETURN_IF_ERROR`).
    *   Prefer early returns, guard clauses, and `continue` to keep the main
        logic at the top indentation level. Never write `else` after an `if`
        block that ends with `return`, `continue`, or `break`.
*   **Explicit types vs. `auto`**:
    *   Spell out types explicitly when the type is not immediately obvious from
        the right-hand side—especially for `absl::StatusOr<T>` unwrapping,
        pointer types, and integer widths (`int64_t` vs. `int32_t`).
    *   Use `auto` only when the type is duplicated on the right-hand side
        (`std::make_unique<Foo>()`, `static_cast<Bar>(...)`), for iterators and
        lambdas, or in structured bindings over map entries
        (`for (const auto& [key, value] : map)`).
*   **Error handling and invariant checks**:
    *   Include `absl/status/status.h`, `absl/status/statusor.h`, and
        `absl/status/status_macros.h`.
    *   Propagate errors with `ABSL_RETURN_IF_ERROR(expr)` and unwrap
        `absl::StatusOr<T>` with `ABSL_ASSIGN_OR_RETURN(lhs, expr)`.
    *   Never call `.value()` or dereference `*status_or` without first
        confirming `.ok()` or unwrapping via `ABSL_ASSIGN_OR_RETURN`.
    *   Use `TF_RET_CHECK(condition)` (from `xla/status_macros.h`) instead of
        `DCHECK` or `LOG(DFATAL)` for internal invariant checks inside functions
        returning `absl::Status` or `absl::StatusOr<T>`.
*   **Rich decision types (`xla::Decision`)**:
    *   In precondition, support, fusion, or autotuning eligibility functions,
        prefer returning `xla::Decision` (`Decision::Allow()` /
        `Decision::Forbid("reason")` from `xla/service/decision.h`, aliased as
        `FusionDecision` in fusion passes) over `bool` or `std::optional` so
        that callers, debug logs, and tests can inspect the exact rejection
        reason.
    *   Do not store a redundant `bool` flag alongside an `std::optional<T>` or
        nullable pointer in result structs.
*   **Containers, algorithms, and deterministic iteration**:
    *   Prefer Abseil container algorithms (`absl::c_find`, `absl::c_any_of`,
        `absl::c_all_of`, `absl::c_count_if`, `absl::c_sort`,
        `absl::c_transform` from `absl/algorithm/container.h`) over
        iterator-pair `std::*` algorithms when operating on an entire container.
    *   Prefer `absl::flat_hash_map` and `absl::flat_hash_set` over
        `std::unordered_map` and `std::unordered_set`, and `std::vector` indexed
        by dense ID over `std::map` when keys are dense integers.
    *   **Compiler determinism**: Never iterate directly over pointer-keyed or
        unordered hash containers (`absl::flat_hash_map`, `absl::flat_hash_set`)
        when creating instructions, assigning IDs, or emitting code. Sort keys
        by a deterministic key (such as `HloInstruction::unique_id()` or name)
        or use a deterministic container (`HloInstructionMap`) so compilation
        output is reproducible across runs.
*   **Namespaces and qualification**:
    *   Use C++17 concatenated namespaces (`namespace xla::gpu {`).
    *   Always qualify `std::` and `absl::` explicitly at call sites (never
        write `using std::...;` or `using absl::...;`).
    *   Avoid unnecessary leading global-scope `::` qualifiers.
    *   Never put namespace-level `using` declarations in `.h` files. In `.cc`
        and `*_test.cc` files, place `using ::testing::...;` declarations inside
        the anonymous `namespace {` block.
*   **Function signatures, lifetimes, and concurrency**:
    *   Return values (`T`, `std::optional<T>`, `absl::StatusOr<T>`, or named
        structs) instead of using non-const pointer or reference output
        parameters.
    *   Pass read-only strings as `absl::string_view` and contiguous sequences
        as `absl::Span<const T>`. Never bind an `absl::string_view` or
        `absl::Span` to a temporary that is destroyed before the view is
        consumed.
    *   Do not wrap arguments in `std::move(...)` when passing to `const T&`
        parameters.
    *   Avoid deeply nested lambdas (2+ levels); extract named helper functions
        in an anonymous namespace instead. Ensure lambdas that outlive the
        enclosing scope do not capture stack locals by reference (`[&]`).
    *   Use `int64_t` when computing element counts, byte sizes, or strides to
        prevent 32-bit signed integer overflow.
    *   Use `absl::Mutex` and `absl::MutexLock` with `ABSL_GUARDED_BY`
        annotations instead of `std::mutex` or `tensorflow::mutex`.
*   **Comments and naming**:
    *   Start implementation files with a concise top-level comment describing
        the file's purpose.
    *   Name files, classes, and functions after domain concepts rather than
        temporary project phases or milestone numbers.
    *   Only write code comments that explain non-obvious invariants, formulas,
        or *why* something is done—never restate what the code literally does.

## 4. Avoiding `#ifdef`s Policy

Preprocessor conditionals (`#if`, `#ifdef`, `#ifndef`) fragment build
configurations, hide uncompiled code from `clang-tidy` and IDE indexers, and
cause cross-platform CI breakages across CUDA, ROCm, SYCL, and CPU builds.
Follow [`docs/avoiding_ifdefs.md`](docs/avoiding_ifdefs.md):

1.  **Never use `#if` / `#ifdef` to skip unit tests in `*_test.cc` (Category
    I)**:
    *   Do not wrap tests or assertions in `#ifdef TENSORFLOW_USE_ROCM`,
        `#if GOOGLE_CUDA`, or `#if CUDA_VERSION < ...`.
    *   Query `se::DeviceDescription` at runtime and call `GTEST_SKIP()`:
        *   `device_desc.gpu_compute_capability().IsRocm()` / `.IsCuda()`
        *   `device_desc.gpu_compute_capability().cuda_compute_capability()->IsAtLeastHopper()`
        *   `device_desc.runtime_version() < se::SemanticVersion{12, 3, 0}`,
            `device_desc.driver_version()`, or `device_desc.dnn_version()`.
2.  **Code that compiles in every configuration (Category II)**:
    *   Use normal runtime `if` statements inspecting `se::DeviceDescription`
        for platform or version branching.
    *   For build-mode checks where both branches are valid C++ on all platforms
        (such as `#ifndef NDEBUG`), use `if constexpr (tsl::kIsDebugBuild)`
        (from `xla/tsl/platform/platform.h`) so the compiler type-checks both
        branches in every build.
3.  **Platform-specific headers and symbols (Category III)**:
    *   Push low-level CUDA/ROCm/SYCL driver and library calls into
        `xla/stream_executor/` behind a platform-neutral interface.
    *   When platform-specific code is required outside `stream_executor`, split
        implementations into separate `.cc` files (`*_cuda.cc`, `*_rocm.cc`,
        `*_stub.cc`) selected in `BUILD` via `if_cuda_is_configured`,
        `if_rocm_is_configured`, or `if_cuda_newer_than`.
    *   **Public headers (`.h`) must never conditionally `#include` CUDA, ROCm,
        or cuDNN headers.**

## 5. Compiler Architecture, Phase Ordering, and IR Invariants

*   **Pass placement and pipeline stage boundaries**:
    *   Place transformations in the pass that owns that abstraction level. Do
        not embed hardware-specific tiling or layout heuristics inside canonical
        algebraic simplification passes.
    *   Do not read or depend on physical layouts (`shape.layout()`) in passes
        that run before Layout Assignment.
    *   Any HLO pass that runs *after* Layout Assignment or Memory Space
        Assignment must preserve physical layouts (`minor_to_major`) and memory
        spaces (`Layout::memory_space()`), or explicitly propagate legal layouts
        to newly created instructions.
*   **HLO graph mutations and metadata preservation**:
    *   Return an accurate `bool changed` from `HloPassInterface::RunImpl`
        (`true` if and only if the module was mutated).
    *   When replacing, cloning, or decomposing HLO instructions, preserve
        `sharding()`, `metadata()`, `frontend_attributes()`, and control
        dependencies (`control_predecessors()` / `control_successors()`).
    *   Propagate `original_value()` (`OriginalValue`) when the replacement
        computes the same semantic value, and keep its `TupleTree` structure
        synchronized whenever instruction shapes or tuple nesting change.
*   **Corner-case shapes, types, and numerical semantics**:
    *   Always check how transformations behave on:
        *   Dynamic dimensions (`shape.is_dynamic()`), unbounded dynamic shapes,
            and dynamic padding.
        *   Zero-element arrays (`ShapeUtil::IsZeroElementArray(shape)`), rank-0
            scalars, and degenerate size-1 dimensions.
        *   Tuple shapes, token shapes, and sub-byte element types (`s4`, `u4`,
            `s2`, `u2`, `pred`).
    *   Ensure floating-point rewrites respect IEEE-754 edge cases (`NaN`
        propagation, `±Inf`, signed zero `-0.0`, and non-associative reductions)
        unless guarded by fast-math / precision options.
    *   Guard integer index and dimension arithmetic against division or modulo
        by zero and signed overflow.
    *   When reasoning about `DynamicSlice` and `DynamicUpdateSlice` offsets or
        buffer aliasing, remember that XLA clamps out-of-bounds runtime slice
        start indices to `[0, max(0, operand_dim - slice_dim)]`.
*   **SPMD sharding and collectives**:
    *   Account for uneven shard padding when `dimension_size % num_shards != 0`
        as well as replicated, partially replicated, subgroup, and unreduced
        shardings.
    *   Preserve valid `replica_groups`, `channel_id`, `use_global_device_ids`,
        and topologically paired `async-start` / `async-done` instructions.
*   **MLIR and Triton codegen**:
    *   Always construct MLIR operations using the static builder
        `OpTy::create(rewriter, loc, ...)` (the `rewriter.create<OpTy>(...)`
        overload is deprecated).
    *   Create top-level MLIR modules with
        `xla::llvm_ir::CreateMlirModuleOp(loc)` (from
        `xla/service/llvm_ir/llvm_util.h`) rather than
        `mlir::ModuleOp::create(loc)` so `MLIRContext` diagnostic handlers are
        wired up.
    *   Include Triton headers without a `third_party/` prefix
        (`#include "triton/..."`).
    *   Validate Triton tile element counts (`<= 1,048,576`) and device shared
        memory limits from `se::DeviceDescription` in cost models and search
        spaces before invoking the emitter.
    *   Do not construct `mlir::SymbolTable(module_op)` inside per-operation
        rewrite patterns ($O(N^2)$ compile-time overhead); pass a cached
        `mlir::SymbolTableCollection` instead.

## 6. Backend Compatibility and Runtime Rules

### XLA:GPU Ahead-of-Time (AOT) Compatibility

Serialized XLA:GPU executables (`GpuExecutableProto`, `ThunkProto`, and all
transitively referenced protobuf messages) must maintain **6 months (180 days)
backward compatibility** (new runtime loading an old serialized executable) and
**2 weeks (14 days) forward compatibility** (up-to-2-week-older runtime loading
a newly compiled executable). See
[`docs/gpu_aot_compatibility.md`](docs/gpu_aot_compatibility.md):

*   **Protobuf wire safety**:
    *   Never change the number, type, or semantics of an existing protobuf
        field.
    *   When removing a field, reserve both its field number and field name
        (`reserved 42; reserved "old_field_name";`).
    *   `FromProto` deserializers must handle missing optional fields by
        restoring the previous default runtime behavior.
*   **Staged rollout for breaking runtime changes**:
    *   When introducing a new `Thunk` kind, a new custom-call / FFI target, or
        a new required proto field that an older runtime cannot execute:
        1.  **Phase 1 (Runtime first)**: Land runtime support (`Thunk`,
            `FromProto`, FFI handler) and keep compiler emission disabled by
            default behind a flag.
        2.  **Phase 2 (Enable emission)**: Enable compiler emission by default
            only after the forward-compatibility window (at least 2 weeks) has
            elapsed.
        3.  **Phase 3 (Legacy cleanup)**: Keep the legacy `FromProto`
            deserialization path for at least 6 months after compiler emission
            of the legacy format stops.
*   **Runtime `DebugOptions` registration**:
    *   If a `DebugOptions` flag in `xla/xla.proto` is read *after* compilation
        (for example, during executable loading, thunk initialization, or
        runtime execution), register its field number in `kRuntimeFields` in
        `xla/debug_options_runtime_fields.cc`.
*   **Golden files**: Never hand-edit serialized AOT test golden files under
    `xla/tests/aot_compatibility/gpu/`.

### XLA:GPU Runtime, FFI, and CUDA Toolkit Compatibility

*   **CUDA and cuDNN dynamic version checks (`xla/stream_executor/cuda/`)**:
    *   Binaries compiled with a given CUDA major version `X` must compile
        against XLA's minimum supported CUDA/cuDNN headers and **run against any
        `X.0+` driver and toolkit library at execution time** (`libcuda.so` and
        toolkit symbols in `xla/tsl/cuda/*.symbols` are loaded dynamically).
    *   Any CUDA driver/runtime API function, enum value, attribute, or struct
        field introduced after `X.0` (such as CUDA `12.1+` in CUDA 12 builds or
        `13.1+` in CUDA 13 builds) **must** be guarded by an explicit runtime
        version check (`device_description.driver_version() >= ...`,
        `runtime_version()`, or library version check) with a safe fallback or
        `absl::Status` error—even if it compiles without `#if` and even if gated
        by GPU compute capability.
    *   Symbols introduced after the minimum supported compile-time header
        version must also be guarded at compile time (`if_cuda_newer_than`
        target split or `#if CUDA_VERSION >= ...` after explicitly including
        `#include "third_party/gpus/cuda/include/cuda.h"`).
    *   Register every newly called CUDA or cuDNN C API symbol (including
        `_ptsz` and `_ptds` per-thread default stream aliases) in the
        corresponding `xla/tsl/cuda/*.symbols` file.
*   **FFI and device topology hygiene**:
    *   In XLA FFI handlers, prefer typed `ffi::Buffer<T>` over untyped
        `ffi::AnyBuffer` whenever element types are statically known; explicitly
        validate element type and rank when `AnyBuffer` is required.
    *   Never hardcode GPU device counts (such as `8` GPUs per host) or warp
        sizes (`32`); query `se::DeviceDescription`.
    *   In `xla/service/gpu/thunk_emitter.cc`, never call `Await` outside
        `EmitHloEntryComputation`.

### XLA:CPU ABI Compatibility

*   Every XLA:CPU compiler and runtime sharing the same
    `kHostExecutionAbiVersion` (`xla/backends/cpu/runtime/host_execution_abi.h`)
    must remain mutually compatible in **both directions** and produce
    numerically identical outputs across `CompilationResultProto`, `ThunkProto`,
    `XLA_CPU_KernelCallFrame` (`xla/backends/cpu/runtime/kernel_c_api.h`), and
    linked math libraries.
*   Evolve CPU runtime protos additively with fallback deserialization, or gate
    new compiler emissions on `abi_version > kLastAbiBeforeFeature` rather than
    making uncoordinated breaking changes.

### Cross-Repository Dependencies (Shardy and StableHLO)

*   OpenXLA pins external commits in `workspace0.bzl` / `workspace.bzl`. If a PR
    depends on a simultaneous change in Shardy or StableHLO, update
    `third_party/shardy/temporary.patch` or
    `third_party/stablehlo/temporary.patch` in the same PR so the build remains
    self-contained.

## 7. Testing Standards

*   **Fixtures and hermeticity**:
    *   Use `HloHardwareIndependentTestBase` for compiler pass and IR unit tests
        that do not execute on a device.
    *   Use `HloInterpreterReferenceMixin<HloTestBase>` (or `HloPjRtTestBase`)
        when comparing execution results against the reference interpreter.
        Never use `HloTestBaseLegacy` in new tests.
    *   Configure test flags by overriding `GetDebugOptionsForTest()` rather
        than constructing a fresh `DebugOptions` proto.
    *   Tests must be deterministic and hermetic: never use `absl::SleepFor`,
        wall-clock timeouts, or unseeded random number generators.
*   **Readable HLO and MLIR tests**:
    *   Embed HLO modules directly in C++ unit tests using `R"hlo(...)hlo"` raw
        string literals rather than imperative `HloInstruction::Create*` builder
        chains or external `.hlo` files.
    *   Keep test HLO minimal and readable: rename auto-generated dump names
        (`%fusion.123.4`) to semantic names (`%input`, `%weights`), omit default
        attributes (`size=0`), and include brief ASCII before/after graph
        diagrams in comments for non-trivial graph rewrites.
    *   When testing `OriginalValue` propagation, write `origin={...}` directly
        in the input HLO string and assert with `FileCheck` or pattern matchers.
    *   In `FileCheck` tests, anchor functions with `CHECK-LABEL`, use
        `CHECK-NEXT` for ordered sequences, and `CHECK-DAG` for independent
        instructions.
*   **Assertions and GoogleMock matchers**:
    *   Include `<gmock/gmock.h>`, `<gtest/gtest.h>`, and
        `absl/status/status_matchers.h`.
    *   Use `ASSERT_OK`, `EXPECT_OK`, `ASSERT_OK_AND_ASSIGN`,
        `absl_testing::IsOk()`, `absl_testing::IsOkAndHolds(...)`, and
        `absl_testing::StatusIs(absl::StatusCode::k..., HasSubstr("..."))`.
    *   **Never** use legacy `TF_ASSERT_OK_AND_ASSIGN`, `TF_ASSERT_OK`,
        `TF_EXPECT_OK`, or `testing::status::*`.
    *   **Never** write weak boolean error checks like
        `EXPECT_FALSE(status.ok())`; always verify both the error code and a
        meaningful message substring with `absl_testing::StatusIs`.
    *   When testing functions returning `xla::Decision`, assert on the
        rejection explanation
        (`EXPECT_THAT(decision.Explain(), HasSubstr(...))`) whenever a case is
        forbidden.
    *   Use GoogleMock matchers (`ElementsAre`, `UnorderedElementsAre`,
        `IsEmpty`, `FieldsAre`, `Optional`) instead of manual indexing loops or
        `EXPECT_EQ(vec.size(), ...)`.
    *   Test helper functions must return `absl::Status` or `absl::StatusOr<T>`
        to the `TEST` body rather than calling `CHECK(...)` (which crashes the
        test binary with `SIGABRT`) or hiding `ASSERT_*` / `EXPECT_*` macros
        inside void helpers.
*   **Test structure (DAMP over DRY)**:
    *   Organize each test into visually separated **Arrange**, **Act**, and
        **Assert** sections.
    *   Keep tests straight-line: avoid loops, branching, or recomputing
        expected outputs using the same formula as the production code.
    *   Use parameterized tests (`TEST_P` / `INSTANTIATE_TEST_SUITE_P` with
        meaningful test-name generators in C++,
        `@parameterized.named_parameters` in Python) or explicit support tables
        (`kSupported`, `kNotSupported`) instead of duplicating test bodies or
        creating huge Cartesian `::testing::Combine` suites.
    *   Prefer real objects or lightweight fakes over mocks; never mock value
        types, protobufs, or the class under test.

## 8. Pre-PR Self-Review Checklist

Before finalizing a PR, verify every item below:

1.  **Scope**: Only files and lines directly required for the task are modified;
    no unrelated formatting changes, dead code, debug logs, or echo comments
    remain.
2.  **C++17 & Style**: No C++20 APIs (`std::span`, `std::*map::contains`,
    `std::string::starts_with`); braces on all single-line control statements;
    `ABSL_RETURN_IF_ERROR` and `ABSL_ASSIGN_OR_RETURN` used consistently; no
    unchecked `.value()` calls.
3.  **Compiler Invariants**: Dynamic, zero-sized, scalar, tuple, and sub-byte
    shapes handled; `int64_t` used for dimension products; layouts, memory
    spaces, shardings, metadata, and `OriginalValue` preserved; no unordered
    hash iteration affecting IR output.
4.  **Compatibility**:
    *   No new `#ifdef`s in tests or platform-independent code
        ([`docs/avoiding_ifdefs.md`](docs/avoiding_ifdefs.md)).
    *   GPU AOT proto compatibility (6-month backward, 2-week forward), staged
        rollout for new thunks/targets, and `kRuntimeFields` registration
        preserved
        ([`docs/gpu_aot_compatibility.md`](docs/gpu_aot_compatibility.md)).
    *   Runtime version guards and `xla/tsl/cuda/*.symbols` entries added for
        any post-`.0` CUDA/cuDNN APIs.
    *   CPU `kHostExecutionAbiVersion` bidirectional compatibility preserved.
5.  **Tests & Build**: Targeted unit tests fail without the fix and pass with it
    (`bazel test`); `BUILD` dependencies and license headers are clean; code is
    formatted with `clang-format` and `buildifier`.
