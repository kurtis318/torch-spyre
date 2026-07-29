# Copyright 2026 The Torch-Spyre Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unified logging configuration for torch-spyre.

This module provides a centralized logging configuration system that:
1. Parses SPYRE_LOGS environment variable for spyre.* namespaces
2. Maintains backward compatibility with legacy environment variables
3. Exposes configuration to C++ via pybind11
4. Provides programmatic API for runtime configuration
5. Configures hierarchical Python logging handlers for the spyre namespace
"""

import logging
import os
import threading
import warnings
from enum import IntEnum
from typing import Dict, List, Optional, Tuple


class LogLevel(IntEnum):
    """Standard Python logging levels."""

    NOTSET = 0
    DEBUG = 10
    INFO = 20
    WARNING = 30
    ERROR = 40
    CRITICAL = 50
    DISABLED = 60


DEFAULT_LOG_LEVELS = {
    "spyre": LogLevel.WARNING,
    "spyre.inductor": LogLevel.WARNING,
    "spyre.inductor.lowering": LogLevel.WARNING,
    "spyre.inductor.stickify": LogLevel.WARNING,
    "spyre.inductor.codegen": LogLevel.WARNING,
    "spyre.inductor.passes": LogLevel.WARNING,
    "spyre.runtime": LogLevel.WARNING,
    "spyre.execution": LogLevel.WARNING,
    "spyre.device": LogLevel.WARNING,
}

_config: Dict[str, LogLevel] = {}
_config_source: Dict[str, str] = {}
_log_file_path: Optional[str] = None
_log_file_source: str = "default"
_initialized = False
_python_logging_configured = False
_lock = threading.RLock()
_cpp_logging_module = None  # None = not attempted, False = unavailable


def _parse_entries(
    entries: List[str], source_label: str
) -> Tuple[Dict[str, LogLevel], Dict[str, str]]:
    """Parse a list of log-config entries with +/-/: syntax.

    Each entry is one of:
    - "+spyre.inductor"  → INFO
    - "-spyre.inductor"  → DISABLED
    - "spyre.inductor:DEBUG" → explicit level

    Non-spyre entries are silently ignored.

    Returns:
        Tuple of (config dict, sources dict)
    """
    config: Dict[str, LogLevel] = {}
    sources: Dict[str, str] = {}

    for entry in entries:
        if entry.startswith("+"):
            component = entry[1:]
            if component.startswith("spyre"):
                config[component] = LogLevel.INFO
                sources[component] = source_label
        elif entry.startswith("-"):
            component = entry[1:]
            if component.startswith("spyre"):
                config[component] = LogLevel.DISABLED
                sources[component] = source_label
        elif ":" in entry:
            component, level_str = entry.split(":", 1)
            component = component.strip()
            level_str = level_str.strip()
            if component.startswith("spyre"):
                try:
                    level = getattr(LogLevel, level_str.upper())
                    config[component] = level
                    sources[component] = source_label
                except AttributeError:
                    warnings.warn(
                        f"Invalid log level '{level_str}' for {component}",
                        stacklevel=3,
                    )

    return config, sources


def _parse_spyre_logs() -> Tuple[Dict[str, LogLevel], Dict[str, str]]:
    """Parse SPYRE_LOGS environment variable for spyre namespaces.

    Supported formats:
    - SPYRE_LOGS="spyre.inductor:DEBUG"
    - SPYRE_LOGS="+spyre.inductor"  (enables at INFO)
    - SPYRE_LOGS="-spyre.inductor"  (disables)
    - SPYRE_LOGS="spyre:INFO,spyre.inductor:DEBUG"

    Returns:
        Tuple of (config dict mapping component names to log levels,
                  sources dict mapping component names to source labels)
    """
    spyre_logs = os.environ.get("SPYRE_LOGS", "")
    if not spyre_logs:
        return {}, {}

    entries = [e.strip() for e in spyre_logs.split(",") if e.strip()]
    return _parse_entries(entries, "SPYRE_LOGS")


def _parse_legacy_vars() -> Tuple[Dict[str, LogLevel], Dict[str, str]]:
    """Parse legacy environment variables with deprecation warnings.

    Legacy variables:
    - SPYRE_INDUCTOR_LOG=1
    - SPYRE_INDUCTOR_LOG_LEVEL=DEBUG
    - TORCH_SPYRE_DEBUG=1
    - SPYRE_LOG_FILE=/path/to/file.log

    Returns:
        Tuple of (config dict mapping component names to log levels,
                  sources dict mapping component names to source labels)
    """
    global _log_file_path, _log_file_source

    config: Dict[str, LogLevel] = {}
    sources: Dict[str, str] = {}

    if os.environ.get("SPYRE_INDUCTOR_LOG") == "1":
        warnings.warn(
            "SPYRE_INDUCTOR_LOG is deprecated. "
            "Use SPYRE_LOGS='spyre.inductor:INFO' instead.",
            DeprecationWarning,
            stacklevel=3,
        )
        level_str = os.environ.get("SPYRE_INDUCTOR_LOG_LEVEL", "INFO")
        try:
            level = getattr(LogLevel, level_str.upper())
            config["spyre.inductor"] = level
            sources["spyre.inductor"] = "legacy:SPYRE_INDUCTOR_LOG"
        except AttributeError:
            config["spyre.inductor"] = LogLevel.INFO
            sources["spyre.inductor"] = "legacy:SPYRE_INDUCTOR_LOG"

    if os.environ.get("TORCH_SPYRE_DEBUG") == "1":
        warnings.warn(
            "TORCH_SPYRE_DEBUG is deprecated. Use SPYRE_LOGS='spyre:DEBUG' instead.",
            DeprecationWarning,
            stacklevel=3,
        )
        for component in DEFAULT_LOG_LEVELS:
            if component not in config:
                config[component] = LogLevel.DEBUG
                sources[component] = "legacy:TORCH_SPYRE_DEBUG"

    legacy_log_file = os.environ.get("SPYRE_LOG_FILE")
    if legacy_log_file:
        warnings.warn(
            "SPYRE_LOG_FILE is deprecated. It is mapped to the top-level "
            "'spyre' logger file handler for backward compatibility.",
            DeprecationWarning,
            stacklevel=3,
        )
        _log_file_path = legacy_log_file
        _log_file_source = "legacy:SPYRE_LOG_FILE"

    torch_logs = os.environ.get("TORCH_LOGS", "")
    if torch_logs:
        spyre_entries = [
            entry.strip()
            for entry in torch_logs.split(",")
            if entry.strip() and entry.strip().lstrip("+-").startswith("spyre")
        ]
        if spyre_entries:
            found = ", ".join(spyre_entries)
            warnings.warn(
                f"TORCH_LOGS contains spyre.* entries ({found}) which should "
                f"be moved to SPYRE_LOGS. "
                f"Non-spyre TORCH_LOGS entries (e.g. '+inductor') are "
                f"unaffected. Replace with: SPYRE_LOGS='{','.join(spyre_entries)}'",
                DeprecationWarning,
                stacklevel=3,
            )
            tl_config, tl_sources = _parse_entries(spyre_entries, "legacy:TORCH_LOGS")
            config.update(tl_config)
            sources.update(tl_sources)

    return config, sources


def _resolve_config() -> Dict[str, LogLevel]:
    """Resolve final configuration from all sources.

    Priority order:
    1. SPYRE_LOGS
    2. Legacy environment variables (TORCH_LOGS spyre.* entries,
       SPYRE_INDUCTOR_LOG, TORCH_SPYRE_DEBUG)
    3. Programmatic API (applied later)
    4. Defaults

    Returns:
        Resolved configuration dictionary
    """
    config = DEFAULT_LOG_LEVELS.copy()

    legacy_config, legacy_sources = _parse_legacy_vars()
    config.update(legacy_config)
    _config_source.update(legacy_sources)

    spyre_logs_config, spyre_sources = _parse_spyre_logs()
    config.update(spyre_logs_config)
    _config_source.update(spyre_sources)

    # When a user explicitly configures a parent component, propagate that
    # level to any more-specific defaults that would otherwise shadow it.
    # For example, SPYRE_LOGS='+spyre.inductor' should override the default
    # WARNING entry for 'spyre.inductor.codegen' so that child loggers like
    # 'spyre.inductor.codegen.superdsc' resolve to the user-specified level.
    explicit_sources = {
        "SPYRE_LOGS",
        "legacy:TORCH_LOGS",
        "legacy:SPYRE_INDUCTOR_LOG",
        "legacy:TORCH_SPYRE_DEBUG",
    }
    for component in list(config):
        if _config_source.get(component, "default") not in explicit_sources:
            # Check if a less-specific ancestor was explicitly configured
            parts = component.split(".")
            for i in range(len(parts) - 1, 0, -1):
                parent = ".".join(parts[:i])
                if _config_source.get(parent, "default") in explicit_sources:
                    config[component] = config[parent]
                    _config_source[component] = _config_source[parent]
                    break

    for component in config:
        if component not in _config_source:
            _config_source[component] = "default"

    return config


def _make_formatter() -> logging.Formatter:
    """Create the default formatter for spyre loggers."""
    return logging.Formatter("[%(levelname)s] [%(name)s] %(message)s")


def _ensure_initialized_locked():
    """Ensure module is initialized. Caller must hold _lock."""
    global _config, _initialized
    if not _initialized:
        _config = _resolve_config()
        _initialized = True
        _configure_python_logging_locked()


def _configure_python_logging_locked():
    """Configure Python logging for spyre. Caller must hold _lock."""
    global _python_logging_configured

    if _python_logging_configured:
        return

    spyre_logger = logging.getLogger("spyre")
    spyre_logger.setLevel(int(_config.get("spyre", LogLevel.WARNING)))

    desired_file = _log_file_path
    formatter = _make_formatter()

    existing_file_handlers = [
        handler
        for handler in spyre_logger.handlers
        if isinstance(handler, logging.FileHandler)
    ]
    existing_stream_handlers = [
        handler
        for handler in spyre_logger.handlers
        if isinstance(handler, logging.StreamHandler)
        and not isinstance(handler, logging.FileHandler)
    ]

    if desired_file:
        file_handler_present = any(
            getattr(handler, "baseFilename", None) == os.path.abspath(desired_file)
            for handler in existing_file_handlers
        )
        if not file_handler_present:
            handler = logging.FileHandler(desired_file)
            handler.setFormatter(formatter)
            spyre_logger.addHandler(handler)

    if not existing_stream_handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        spyre_logger.addHandler(handler)

    _python_logging_configured = True


def configure_python_logging():
    """Configure the top-level hierarchical Python logger for spyre.

    This is idempotent and safe to call multiple times.
    """
    with _lock:
        _ensure_initialized_locked()
        _configure_python_logging_locked()


def initialize():
    """Initialize logging configuration from environment variables.

    This should be called once during module initialization.
    Thread-safe and idempotent.
    """
    with _lock:
        _ensure_initialized_locked()


def _sync_cpp_config(cpp_config: List[Tuple[str, int]], log_file: str):
    """Push a config snapshot to the C++ LoggingConfig singleton.

    Args:
        cpp_config: List of (component, level_int) tuples, snapshotted
            under _lock by the caller.
        log_file: Log file path (empty string for none).
    """
    global _cpp_logging_module
    if _cpp_logging_module is False:
        return
    if _cpp_logging_module is None:
        try:
            from torch_spyre._C import _logging  # type: ignore[attr-defined]

            _cpp_logging_module = _logging
        except (ImportError, ModuleNotFoundError):
            _cpp_logging_module = False
            return

    config = _cpp_logging_module.LoggingConfig.instance()
    config.initialize_from_python(cpp_config)
    config.set_log_file(log_file)


def reset():
    """Reset logging state and re-initialize from current environment variables.

    Thread-safe. Intended for test isolation where environment variables
    are modified between calls.
    """
    global _config, _config_source, _log_file_path, _log_file_source
    global _initialized, _python_logging_configured

    with _lock:
        _config = {}
        _config_source = {}
        _log_file_path = None
        _log_file_source = "default"
        _initialized = False
        _python_logging_configured = False
        _ensure_initialized_locked()
        cpp_config = [(comp, int(lvl)) for comp, lvl in _config.items()]
        log_file = _log_file_path or ""

    _sync_cpp_config(cpp_config, log_file)


def get_log_level(component: str) -> LogLevel:
    """Get effective log level for a component.

    Args:
        component: Component name (e.g., "spyre.inductor")

    Returns:
        Effective log level for the component
    """
    with _lock:
        _ensure_initialized_locked()
        if component in _config:
            return _config[component]

        parts = component.split(".")
        for i in range(len(parts) - 1, 0, -1):
            parent = ".".join(parts[:i])
            if parent in _config:
                return _config[parent]

    return LogLevel.WARNING


def set_log_level(component: str, level: str):
    """Set log level for a component programmatically.

    Args:
        component: Component name (e.g., "spyre.inductor")
        level: Log level name (DEBUG, INFO, WARNING, ERROR, CRITICAL, DISABLED)
    """
    try:
        level_enum = getattr(LogLevel, level.upper())
    except AttributeError as exc:
        raise ValueError(f"Invalid log level: {level}") from exc

    with _lock:
        _ensure_initialized_locked()
        _config[component] = level_enum
        _config_source[component] = "programmatic"

        logger = logging.getLogger(component)
        logger.setLevel(int(level_enum))

        cpp_config = [(comp, int(lvl)) for comp, lvl in _config.items()]
        log_file = _log_file_path or ""

    _sync_cpp_config(cpp_config, log_file)


def enable(component: str):
    """Enable logging for a component at INFO level.

    Args:
        component: Component name (e.g., "spyre.inductor")
    """
    set_log_level(component, "INFO")


def disable(component: str):
    """Disable logging for a component.

    Args:
        component: Component name (e.g., "spyre.inductor")
    """
    set_log_level(component, "DISABLED")


def get_log_file() -> Optional[str]:
    """Get the configured log file path, if any."""
    with _lock:
        _ensure_initialized_locked()
        return _log_file_path


def set_log_file(path: Optional[str]):
    """Set the log file path programmatically.

    Thread-safety note: on the C++ side the old stream is destroyed
    immediately, so this must not be called while C++ threads are
    actively emitting log records.  In normal usage this is safe
    because configuration happens at import time or under the GIL
    before compiled workloads spawn threads.
    """
    global _log_file_path, _log_file_source, _python_logging_configured

    with _lock:
        _ensure_initialized_locked()
        _log_file_path = path
        _log_file_source = "programmatic" if path else "default"
        _python_logging_configured = False
        _configure_python_logging_locked()
        cpp_config = [(comp, int(lvl)) for comp, lvl in _config.items()]
        log_file = _log_file_path or ""

    _sync_cpp_config(cpp_config, log_file)


def get_effective_config() -> Dict[str, str]:
    """Get effective configuration for all components.

    Returns:
        Dictionary mapping component names to level names
    """
    with _lock:
        _ensure_initialized_locked()
        return {component: level.name for component, level in _config.items()}


def get_output_config() -> Dict[str, Optional[str]]:
    """Get effective output configuration."""
    with _lock:
        _ensure_initialized_locked()
        return {
            "log_file": _log_file_path,
            "log_file_source": _log_file_source,
        }


def get_config_source(component: str) -> str:
    """Get configuration source for a component.

    Args:
        component: Component name

    Returns:
        Source name: "SPYRE_LOGS", "legacy:TORCH_LOGS", "legacy:...",
        "programmatic", or "default"
    """
    with _lock:
        _ensure_initialized_locked()
        return _config_source.get(component, "default")


def list_components() -> List[str]:
    """List all available logging components.

    Returns:
        List of component names
    """
    return list(DEFAULT_LOG_LEVELS.keys())


def get_config_for_cpp() -> List[Tuple[str, int]]:
    """Get configuration in format suitable for C++.

    Returns:
        List of (component, level) tuples with integer levels
    """
    with _lock:
        _ensure_initialized_locked()
        return [(comp, int(level)) for comp, level in _config.items()]


initialize()
