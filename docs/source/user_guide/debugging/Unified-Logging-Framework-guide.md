# Spyre Unified Logging Framework (ULF) Guide

<!-- markdownlint-disable MD024 -->

## Table of Contents

- [Spyre Unified Logging Framework (ULF) Guide](#spyre-unified-logging-framework-ulf-guide)
  - [Table of Contents](#table-of-contents)
  - [Quick Start](#quick-start)
  - [1. Overview](#1-overview)
    - [Migrating from Legacy Variables](#migrating-from-legacy-variables)
  - [2. TORCH\_LOGS Syntax](#2-torch_logs-syntax)
    - [Syntax Forms](#syntax-forms)
    - [Hierarchy Rules](#hierarchy-rules)
    - [Common Recipes](#common-recipes)
    - [Bare-Name Warning](#bare-name-warning)
  - [3. The Naming Rule: torch\_spyre.\* vs spyre.\*](#3-the-naming-rule-torch_spyre-vs-spyre)
    - [The Rule](#the-rule)
    - [Where Internal Names Surface](#where-internal-names-surface)
  - [4. Available Components](#4-available-components)
    - [Direct TORCH\_LOGS Targets](#direct-torch_logs-targets)
    - [Child Components (controlled via parent inheritance)](#child-components-controlled-via-parent-inheritance)
  - [5. Python Usage](#5-python-usage)
    - [Getting a Logger](#getting-a-logger)
    - [Adding Logging to New Code](#adding-logging-to-new-code)
    - [Log Format](#log-format)
    - [File Output](#file-output)
  - [6. C++ Usage](#6-c-usage)
    - [Available Macros](#available-macros)
    - [Adding Logging to New C++ Code](#adding-logging-to-new-c-code)
    - [Using the Macros](#using-the-macros)
    - [Legacy DEBUGINFO](#legacy-debuginfo)
    - [Log Format (C++)](#log-format-c)
    - [File Output (C++)](#file-output-c)
  - [7. Troubleshooting](#7-troubleshooting)
  - [Appendix A: Python Programmatic API](#appendix-a-python-programmatic-api)
    - [Setting Levels](#setting-levels)
    - [Introspection](#introspection)
    - [File Output](#file-output-1)
    - [Reset (for testing)](#reset-for-testing)
  - [Appendix B: C++ Direct API](#appendix-b-c-direct-api)
    - [Checking if a Level is Enabled](#checking-if-a-level-is-enabled)
    - [Querying Configuration](#querying-configuration)
    - [Setting Levels (testing only)](#setting-levels-testing-only)
    - [Thread Safety](#thread-safety)
  - [Appendix C: Design Rationale — Naming](#appendix-c-design-rationale--naming)
    - [How PyTorch Does It (for contrast)](#how-pytorch-does-it-for-contrast)
    - [How ULF Does It (prefix normalization, no alias table)](#how-ulf-does-it-prefix-normalization-no-alias-table)
    - [Why `spyre.*` Internally](#why-spyre-internally)

---

## Quick Start

```bash
# Enable all spyre logging at INFO
export TORCH_LOGS="+torch_spyre"

# Debug a specific component (e.g., lowering)
export TORCH_LOGS="torch_spyre.inductor.lowering:DEBUG"

# Full debug — Python inductor + C++ runtime
export TORCH_LOGS="torch_spyre:DEBUG"
```

See [Common Recipes](#common-recipes) for detailed level resolution across
multiple components.

---

## 1. Overview

The **Unified Logging Framework (ULF)** consolidates all torch-spyre logging
(Python and C++) behind the standard **`TORCH_LOGS`** environment variable —
the same mechanism PyTorch itself uses. ULF replaces the legacy
`SPYRE_INDUCTOR_LOG`, `SPYRE_INDUCTOR_LOG_LEVEL`, and `TORCH_SPYRE_DEBUG`
variables with a single, hierarchical, per-component configuration that is
consistent across Python and C++ code paths.

A **component** is a dot-separated identifier (e.g.,
`torch_spyre.inductor.lowering`) that names a logical subsystem within
torch-spyre. Components form a hierarchy: setting a level on a parent
propagates to all its children unless a more-specific child entry overrides
it.

Configuration priority (highest wins):

1. `TORCH_LOGS` environment variable
2. Legacy env vars (deprecated, emit warnings)
3. Programmatic API (`logging_config.set_log_level(...)`)
4. Defaults (all components at WARNING)

### Migrating from Legacy Variables

The following legacy environment variables are deprecated and emit warnings
on use. Migrate to their `TORCH_LOGS` equivalents:

| Old variable | New equivalent |
| --- | --- |
| `SPYRE_INDUCTOR_LOG=1` | `TORCH_LOGS="+torch_spyre.inductor"` |
| `SPYRE_INDUCTOR_LOG_LEVEL=DEBUG` | `TORCH_LOGS="torch_spyre.inductor:DEBUG"` |
| `TORCH_SPYRE_DEBUG=1` | `TORCH_LOGS="torch_spyre:DEBUG"` |
| `SPYRE_LOG_FILE=/path` | `logging_config.set_log_file("/path")` |

---

## 2. TORCH_LOGS Syntax

### Syntax Forms

`TORCH_LOGS` accepts a **comma-separated list** of entries. Each entry
targets a component and takes one of three forms:

| Syntax | Effect | Example |
| --- | --- | --- |
| `+<component>` | Enable at **INFO** | `+torch_spyre.inductor` |
| `-<component>` | **Disable** (level = DISABLED/60) | `-torch_spyre.inductor.passes` |
| `<component>:<LEVEL>` | Set explicit level | `torch_spyre.runtime:DEBUG` |

Valid levels: `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`, `DISABLED`

### Hierarchy Rules

A parent setting propagates to children unless a more-specific entry
overrides. Setting `+torch_spyre.inductor` cascades INFO down to all child
components like `spyre.inductor.codegen`, `spyre.inductor.lowering`, etc.

Non-spyre entries (e.g., `+inductor`, `dynamo`) are passed through to
PyTorch's own logger system — ULF only processes entries where the component
normalizes to a name starting with `spyre`.

### Common Recipes

| TORCH_LOGS value | Component | Effective level |
| --- | --- | --- |
| `+torch_spyre` | all `spyre.*` | INFO |
| `+torch_spyre.inductor` | `spyre.inductor.*` | INFO |
| | `spyre.runtime` | WARNING (unaffected) |
| `torch_spyre:DEBUG` | all `spyre.*` | DEBUG |
| `+torch_spyre,-torch_spyre.inductor.passes` | `spyre.inductor.lowering` | INFO |
| | `spyre.inductor.passes` | DISABLED |
| | `spyre.runtime` | INFO |
| `torch_spyre.runtime:DEBUG,+torch_spyre.inductor` | `spyre.runtime` | DEBUG |
| | `spyre.inductor.lowering` | INFO |
| | `spyre.inductor.codegen.superdsc` | INFO |
| `+torch_spyre.inductor,torch_spyre.inductor.lowering:DEBUG` | `spyre.inductor.lowering` | DEBUG (most specific) |
| | `spyre.inductor.codegen` | INFO (from parent) |

**Multi-component example:**

```bash
export TORCH_LOGS="+torch_spyre.inductor.lowering,torch_spyre.runtime:DEBUG,-torch_spyre.inductor.passes,+inductor"
```

| Component | Level | Reason |
| --- | --- | --- |
| `spyre.inductor.lowering` | INFO | `+` prefix |
| `spyre.runtime` | DEBUG | explicit `:DEBUG` |
| `spyre.inductor.passes` | DISABLED | `-` prefix |
| `spyre.inductor.codegen` | WARNING | no entry → default |
| `inductor` (PyTorch native) | handled by PyTorch | not a `torch_spyre` component, passed through |

### Bare-Name Warning

> **A bare component name without `+`, `-`, or `:LEVEL` does NOTHING.**

Given:

```bash
TORCH_LOGS="+torch_spyre.inductor.lowering,torch_spyre.inductor.codegen,-torch_spyre.inductor.codegen.superdsc"
```

| Entry | Parsed as | Resulting level |
| --- | --- | --- |
| `+torch_spyre.inductor.lowering` | `+` prefix → INFO | **INFO** |
| `torch_spyre.inductor.codegen` | **No prefix, no colon** → **NOT PARSED** | **WARNING** (unchanged default) |
| `-torch_spyre.inductor.codegen.superdsc` | `-` prefix → DISABLED | **DISABLED** |

This is **inconsistent with PyTorch's own TORCH_LOGS behavior** where a bare
component name typically enables DEBUG-level logging. In ULF, the parser only
recognizes three forms: `+component`, `-component`, or `component:LEVEL`. A
bare component name is silently ignored.

Fix: use `+torch_spyre.inductor.codegen` (INFO) or
`torch_spyre.inductor.codegen:DEBUG` (explicit level).

---

## 3. The Naming Rule: torch_spyre.\* vs spyre.\*

### The Rule

| Context | Use | Example |
| --- | --- | --- |
| `TORCH_LOGS` environment variable | `torch_spyre.*` | `TORCH_LOGS="+torch_spyre.inductor"` |
| Everything else | `spyre.*` | `logging.getLogger("spyre.inductor")` |

**Why:** PyTorch validates every `TORCH_LOGS` target with
`importlib.util.find_spec()` at `import torch` time. Only `torch_spyre.*` is
an importable package that passes this check. Bare `spyre.*` causes torch to
raise "Invalid log settings" before any Spyre code runs. ULF's parser
normalizes `torch_spyre.*` → `spyre.*` internally so the configured level
lands on the correct logger.

### Where Internal Names Surface

The internal `spyre.*` name appears in these places during everyday
development:

1. **Log output** — every emitted line displays `spyre.*`, not `torch_spyre.*`
2. **Source code** — `get_inductor_logger("lowering")` returns a logger named `spyre.inductor.lowering`
3. **Programmatic API** — `logging_config.set_log_level("spyre.inductor", "DEBUG")`
4. **Python's logging module** — `logging.getLogger("spyre.inductor.lowering")`
5. **C++ macros** — `SPYRE_LOG("spyre.runtime", DEBUG)`
6. **Test assertions** — `capture_logs("spyre.inductor.passes", level="INFO")`

For design rationale on this naming split, see
[Appendix C](#appendix-c-design-rationale--naming).

---

## 4. Available Components

### Direct TORCH_LOGS Targets

These components have stub namespace packages and can be used directly in
`TORCH_LOGS`:

| TORCH_LOGS name | Internal logger | What it controls | Primary source |
| --- | --- | --- | --- |
| `torch_spyre` | `spyre` | Root — all Spyre logging | — |
| `torch_spyre.inductor` | `spyre.inductor` | All Inductor compiler passes and codegen | `torch_spyre/_inductor/` |
| `torch_spyre.inductor.lowering` | `spyre.inductor.lowering` | Op lowering (ATen → Spyre IR) | `torch_spyre/_inductor/lowering.py` |
| `torch_spyre.inductor.codegen` | `spyre.inductor.codegen` | Code generation (parent) | `torch_spyre/_inductor/codegen/bundle.py` |
| `torch_spyre.inductor.stickify` | `spyre.inductor.stickify` | Tensor stickification passes | `torch_spyre/_inductor/insert_restickify.py` |
| `torch_spyre.inductor.passes` | `spyre.inductor.passes` | General compiler passes | `torch_spyre/_inductor/passes.py` |
| `torch_spyre.runtime` | `spyre.runtime` | C++ runtime (allocator, streams, distributed) | `torch_spyre/csrc/` |

### Child Components (controlled via parent inheritance)

These do **not** have stub packages and cannot be named directly in
`TORCH_LOGS`. They inherit their level from the nearest configured ancestor.
Use the programmatic API to target them individually.

| Internal logger | What it controls | Controlled by parent | Source file |
| --- | --- | --- | --- |
| `spyre.inductor.codegen.superdsc` | SuperDSC kernel compiler | `torch_spyre.inductor.codegen` | `_inductor/codegen/superdsc.py` |
| `spyre.inductor.work_division` | Multi-core work division planning | `torch_spyre.inductor` | `_inductor/work_division.py` |
| `spyre.inductor.scratchpad.allocator` | LX scratchpad allocation | `torch_spyre.inductor` | `_inductor/scratchpad/allocator.py` |
| `spyre.inductor.propagate_layouts` | Layout propagation | `torch_spyre.inductor` | `_inductor/propagate_layouts.py` |
| `spyre.inductor.sdsc_compile` | SDSC bundle compilation | `torch_spyre.inductor.codegen` | `_inductor/codegen/bundle.py` |
| `spyre.inductor.spyre_kernel` | Kernel wrapper scheduling | `torch_spyre.inductor` | `_inductor/spyre_kernel.py` |
| `spyre.inductor.padding` | Tensor padding | `torch_spyre.inductor` | `_inductor/padding.py` |
| `spyre.inductor.scheduler` | Node scheduling | `torch_spyre.inductor` | `_inductor/scheduler.py` |
| `spyre.inductor.ir` | Spyre IR nodes | `torch_spyre.inductor` | `_inductor/ir.py` |

Full list: see `DEFAULT_LOG_LEVELS` in `torch_spyre/logging_config.py` and
`get_inductor_logger()` calls throughout `torch_spyre/_inductor/`.

---

## 5. Python Usage

### Getting a Logger

```python
from torch_spyre._inductor.logging_utils import get_inductor_logger

logger = get_inductor_logger("lowering")  # → "spyre.inductor.lowering"
logger.info("mm: x%s @ y%s -> %s", x.shape, y.shape, out.shape)
```

### Adding Logging to New Code

```python
# In torch_spyre/_inductor/my_new_pass.py:
from torch_spyre._inductor.logging_utils import get_inductor_logger

logger = get_inductor_logger("my_new_pass")

# Use standard Python logging methods:
logger.debug("Detailed trace: %s", detail)
logger.warning("Unexpected condition: %s", condition)
```

The logger `spyre.inductor.my_new_pass` is automatically controlled by
`TORCH_LOGS="+torch_spyre.inductor"` via parent inheritance.

### Log Format

```text
[INFO] [spyre.inductor.lowering] mm: x[2,3] @ y[3,4] -> [2,4]
```

### File Output

Applies to both Python and C++ logging (C++ sink follows Python config):

```bash
SPYRE_LOG_FILE=/tmp/spyre.log  # legacy, deprecated
```

Or programmatically:

```python
from torch_spyre import logging_config
logging_config.set_log_file("/tmp/spyre.log")
```

This configures the top-level `spyre` logger's file handler (Python) and
calls `LoggingConfig::set_log_file()` on the C++ side — both languages write
to the same file.

---

## 6. C++ Usage

### Available Macros

| Macro | Component | Level |
| --- | --- | --- |
| `SPYRE_RUNTIME_DEBUG()` | `spyre.runtime` | DEBUG |
| `SPYRE_RUNTIME_INFO()` | `spyre.runtime` | INFO |
| `SPYRE_RUNTIME_WARNING()` | `spyre.runtime` | WARNING |
| `SPYRE_RUNTIME_ERROR()` | `spyre.runtime` | ERROR |
| `SPYRE_RUNTIME_CRITICAL()` | `spyre.runtime` | CRITICAL |
| `SPYRE_LOG(component, LEVEL)` | any | any |
| `SPYRE_LOG_ENABLED(component, level)` | any | any (returns bool) |
| `DEBUGINFO(...)` | `spyre.runtime` | DEBUG (legacy) |

### Adding Logging to New C++ Code

```cpp
#include "logging_config.h"

// Use SPYRE_LOG with your component name:
SPYRE_LOG("spyre.runtime", INFO) << "New feature initialized";

// For conditional expensive computation:
if (SPYRE_LOG_ENABLED("spyre.runtime", torch_spyre::logging::LogLevel::DEBUG)) {
    auto stats = compute_expensive_stats();
    SPYRE_RUNTIME_DEBUG() << "Stats: " << stats;
}
```

### Using the Macros

```cpp
#include "logging_config.h"

SPYRE_RUNTIME_DEBUG() << "Allocated " << nbytes << " bytes";
SPYRE_RUNTIME_INFO() << "Kernel launched on device " << dev_id;

// Generic form for any component:
SPYRE_LOG("spyre.inductor.codegen", DEBUG) << "Generating op: " << op_name;
```

Example with `TORCH_LOGS`:

```bash
export TORCH_LOGS="+torch_spyre.runtime,torch_spyre.inductor.codegen:DEBUG"
```

```cpp
#include "logging_config.h"

// With the above TORCH_LOGS setting:
SPYRE_RUNTIME_INFO() << "This prints (INFO enabled by +)";
SPYRE_RUNTIME_DEBUG() << "This does NOT print (+ only enables INFO)";
SPYRE_LOG("spyre.inductor.codegen", DEBUG) << "This prints (DEBUG set explicitly)";
SPYRE_LOG("spyre.inductor.lowering", INFO) << "This does NOT print (no entry, defaults to WARNING)";
```

### Legacy DEBUGINFO

```cpp
#include "logging.h"

DEBUGINFO("Allocating ", nbytes, " bytes on Spyre", device);
// Equivalent to: SPYRE_RUNTIME_DEBUG() << __func__ << ": Allocating " << ...
```

`DEBUGINFO` maps to component `spyre.runtime` at DEBUG level.

### Log Format (C++)

```text
[DEBUG] [spyre.runtime] 2026-08-06 14:30:22 allocate_tensor: Allocated 1024 bytes
```

### File Output (C++)

C++ log output follows the file path configured from Python. There is no
separate C++ env var — call `logging_config.set_log_file("/path")` from
Python (or use the legacy `SPYRE_LOG_FILE`) and C++ output goes to the same
file. See [File Output in section 5](#file-output).

---

## 7. Troubleshooting

**Logs don't appear:**

- Check that your `TORCH_LOGS` entry has a `+` prefix or `:LEVEL` suffix.
  A bare component name (e.g., `torch_spyre.inductor.codegen`) is silently
  ignored — use `+torch_spyre.inductor.codegen` instead.
- Verify the component's effective level is at or below the log call level.
  `+` sets INFO, so `logger.debug(...)` calls still won't appear — use
  `torch_spyre.inductor.codegen:DEBUG` for full verbosity.

**"Invalid log settings" error at import time:**

- You used bare `spyre.*` in `TORCH_LOGS`. PyTorch's validator requires an
  importable package name. Use `torch_spyre.*` instead.

**Child component not responding to TORCH_LOGS:**

- Components like `spyre.inductor.work_division` have no stub package and
  cannot be named directly in `TORCH_LOGS`. Enable their parent instead
  (`+torch_spyre.inductor`) or use the programmatic API:
  `logging_config.set_log_level("spyre.inductor.work_division", "DEBUG")`

**Legacy env var deprecation warnings:**

- See [Migrating from Legacy Variables](#migrating-from-legacy-variables) in
  section 1 for the equivalent `TORCH_LOGS` settings.

---

## Appendix A: Python Programmatic API

The `torch_spyre.logging_config` module provides runtime control over logging
configuration. Most developers will not need this — `TORCH_LOGS` is
sufficient for typical use.

### Setting Levels

```python
from torch_spyre import logging_config

logging_config.set_log_level("spyre.inductor.lowering", "DEBUG")
logging_config.enable("spyre.runtime")      # shorthand for INFO
logging_config.disable("spyre.inductor.passes")
```

### Introspection

```python
# Get effective config for all components
logging_config.get_effective_config()
# → {"spyre": "WARNING", "spyre.inductor": "WARNING", ...}

# Get source of a component's config
logging_config.get_config_source("spyre.inductor")
# → "TORCH_LOGS" | "legacy:SPYRE_INDUCTOR_LOG" | "programmatic" | "default"

# List all known components
logging_config.list_components()
# → ["spyre", "spyre.inductor", "spyre.inductor.lowering", ...]
```

### File Output

For file output configuration, see [File Output in section 5](#file-output).
The query API:

```python
logging_config.get_log_file()  # → "/tmp/spyre.log" or None
```

### Reset (for testing)

```python
# Re-read environment variables and reinitialize all state
logging_config.reset()
```

---

## Appendix B: C++ Direct API

The C++ logging API is defined in `torch_spyre/csrc/logging_config.h`. Most
developers will use the `SPYRE_LOG` / `SPYRE_RUNTIME_*` macros (section 6).
The direct API below is for advanced use cases like conditional expensive
computation.

### Checking if a Level is Enabled

```cpp
#include "logging_config.h"

using torch_spyre::logging::LoggingConfig;
using torch_spyre::logging::LogLevel;

if (LoggingConfig::instance().is_enabled("spyre.runtime", LogLevel::DEBUG)) {
    // Only compute expensive diagnostics when DEBUG is active
    auto stats = compute_memory_stats();
    SPYRE_RUNTIME_DEBUG() << "Memory: " << stats;
}
```

### Querying Configuration

```cpp
// Get the effective level for a component
LogLevel level = LoggingConfig::instance().get_log_level("spyre.runtime");

// List all configured components
std::vector<std::string> components = LoggingConfig::instance().get_components();
```

### Setting Levels (testing only)

```cpp
// Override a component's level at runtime
LoggingConfig::instance().set_log_level("spyre.runtime", LogLevel::DEBUG);
```

### Thread Safety

- Config reads: lock-free via generation-validated thread_local cache
- Config writes: exclusive lock (only at init or programmatic change)
- Log output: single-write per record; safe on POSIX via FILE* flockfile

---

## Appendix C: Design Rationale — Naming

This appendix explains why ULF uses two different name forms. Most developers
only need the rule from [section 3](#3-the-naming-rule-torch_spyre-vs-spyre);
this background is for contributors working on the logging infrastructure
itself.

### How PyTorch Does It (for contrast)

In upstream PyTorch, `TORCH_LOGS` uses **short aliases** that are distinct
from the internal Python logger names:

| TORCH_LOGS alias | Internal Python logger |
| --- | --- |
| `inductor` | `torch._inductor` |
| `dynamo` | `torch._dynamo` |
| `aot` | `torch._functorch.aot_autograd` |

PyTorch maintains a registration table that maps these short aliases to their
full logger paths.

### How ULF Does It (prefix normalization, no alias table)

ULF does **not** use an alias-registration table. Instead, it applies a
simple prefix normalization: `torch_spyre.*` → `spyre.*`. The function
`_normalize_component()` in `torch_spyre/logging_config.py` performs this
mapping.

ULF's parser will accept bare `spyre.*` as input, but PyTorch's `find_spec()`
validator rejects it at `import torch` time before ULF ever runs. Therefore
`torch_spyre.*` is required in practice.

### Why `spyre.*` Internally

The Python **package** is named `torch_spyre` (installed as `torch-spyre`),
and source files live under `torch_spyre/_inductor/...`. The internal
logger namespace uses `spyre.*` deliberately:

| Layer | Namespace | Example |
| --- | --- | --- |
| Python package (import path) | `torch_spyre` | `from torch_spyre._inductor.lowering import ...` |
| TORCH_LOGS component | `torch_spyre` | `TORCH_LOGS="+torch_spyre.inductor.lowering"` |
| Python `logging` logger | `spyre` | `logging.getLogger("spyre.inductor.lowering")` |
| C++ component string | `spyre` | `SPYRE_LOG("spyre.runtime", DEBUG)` |

Rationale:

1. **Brevity** — `spyre.inductor` is shorter than `torch_spyre._inductor`
   and appears in every log line.
2. **Clean hierarchy** — the `_inductor` internal package path and leading
   `torch_` prefix are implementation details that don't belong in log
   output.
3. **Consistency** — PyTorch itself uses short, clean names (`inductor`,
   `dynamo`) rather than exposing `torch._inductor`. ULF follows the same
   philosophy.
