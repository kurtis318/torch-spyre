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


"""Comprehensive tests for the SPYRE_LOGS environment variable.

Tests cover:
- Python logging at every level (DEBUG, INFO, WARNING, CRITICAL) with 1, 2, 3 components
- C++ DEBUGINFO macro behavior with 1, 2, 3 components
- Precedence, hierarchy propagation, disabling, invalid levels, file output
"""

import contextlib
import importlib.machinery
import importlib.util
import logging
import os
import sys
import tempfile
import warnings
from collections.abc import Generator
from pathlib import Path
from types import ModuleType
from typing import TypedDict

import pytest

pytestmark = pytest.mark.serial

TEST_FILE = Path(__file__).resolve()


def _candidate_package_roots() -> Generator[Path, None, None]:
    """Yield likely package-root locations for the torch-spyre sources."""
    explicit_root = os.environ.get("TORCH_SPYRE_PACKAGE_ROOT")
    if explicit_root:
        yield Path(explicit_root).resolve()

    env_pythonpath = os.environ.get("PYTHONPATH", "")
    for entry in env_pythonpath.split(os.pathsep):
        if entry:
            path_entry = Path(entry).resolve()
            yield path_entry
            yield path_entry / "torch_spyre"

    script_dir = TEST_FILE.parent
    yield script_dir
    yield script_dir / "torch_spyre"
    yield script_dir / "torch-spyre" / "torch_spyre"
    yield script_dir / "torch-spyre" / "torch-spyre" / "torch_spyre"

    seen = set()
    anchors = [
        script_dir,
        *script_dir.parents,
        Path.cwd().resolve(),
        *Path.cwd().resolve().parents,
    ]
    for anchor in anchors:
        if anchor in seen:
            continue
        seen.add(anchor)
        yield anchor
        yield anchor / "torch_spyre"
        yield anchor / "torch-spyre" / "torch_spyre"
        yield anchor / "torch-spyre" / "torch-spyre" / "torch_spyre"


def _is_package_root(candidate: Path) -> bool:
    has_logging_config = (candidate / "logging_config.py").is_file()
    has_init = (candidate / "__init__.py").is_file()
    return has_logging_config and has_init


def _find_package_root() -> Path | None:
    """Return the resolved torch-spyre package root when discoverable."""
    candidates: list[Path] = []
    for candidate in _candidate_package_roots():
        if candidate in candidates:
            continue
        candidates.append(candidate)
        if _is_package_root(candidate):
            return candidate

    for candidate in candidates:
        try:
            for child in candidate.iterdir():
                if child.is_dir() and _is_package_root(child):
                    return child
        except OSError:
            continue

    return None


PACKAGE_ROOT = _find_package_root()
if PACKAGE_ROOT is not None:
    BASE_DIR = PACKAGE_ROOT.parent
    if str(BASE_DIR) not in sys.path:
        sys.path.insert(0, str(BASE_DIR))


class _CapturedLogs:
    """Container for captured log records."""

    def __init__(self):
        self.records: list[logging.LogRecord] = []

    @property
    def output(self) -> list[str]:
        return [
            f"{record.levelname}:{record.name}:{record.getMessage()}"
            for record in self.records
        ]


@contextlib.contextmanager
def capture_logs(logger_name: str, level: str = "DEBUG"):
    """Capture log output from a named logger at or above the given level."""
    logger = logging.getLogger(logger_name)
    captured = _CapturedLogs()
    old_level = logger.level
    logger.setLevel(getattr(logging, level))

    class _Handler(logging.Handler):
        def emit(self, record):
            captured.records.append(record)

    handler = _Handler()
    handler.setLevel(getattr(logging, level))
    logger.addHandler(handler)
    try:
        yield captured
    finally:
        logger.removeHandler(handler)
        logger.setLevel(old_level)


class _LoggerState(TypedDict):
    level: int
    handlers: list[logging.Handler]
    propagate: bool
    disabled: bool


_ALL_LOGGER_NAMES = (
    "spyre",
    "spyre.inductor",
    "spyre.inductor.lowering",
    "spyre.inductor.codegen",
    "spyre.inductor.stickify",
    "spyre.inductor.passes",
    "spyre.inductor.sdsc_compile",
    "spyre.inductor.work_division",
    "spyre.inductor.propagate_layouts",
    "spyre.inductor.test_component",
    "spyre.runtime",
    "spyre.execution",
    "spyre.device",
)


_LOGGING_ENV_KEYS = (
    "SPYRE_LOGS",
    "TORCH_LOGS",
    "SPYRE_INDUCTOR_LOG",
    "SPYRE_INDUCTOR_LOG_LEVEL",
    "TORCH_SPYRE_DEBUG",
    "SPYRE_LOG_FILE",
)


class LoggingIsolationMixin:
    """Shared helpers for isolating logging state across tests."""

    def setup_method(self) -> None:
        """Save process environment, modules, and logger state before each test."""
        self._saved_env = {key: os.environ.get(key) for key in _LOGGING_ENV_KEYS}
        self._saved_modules = {
            name: sys.modules.get(name)
            for name in (
                "logging_config",
                "_inductor.logging_utils",
                "torch_spyre.logging_config",
                "torch_spyre._inductor.logging_utils",
            )
        }
        self._saved_loggers = {
            name: logging.getLogger(name) for name in _ALL_LOGGER_NAMES
        }
        self._saved_logger_state: dict[str, _LoggerState] = {}
        for name, logger in self._saved_loggers.items():
            self._saved_logger_state[name] = {
                "level": logger.level,
                "handlers": list(logger.handlers),
                "propagate": logger.propagate,
                "disabled": logger.disabled,
            }

    def teardown_method(self) -> None:
        """Restore environment, modules, and loggers after each test."""
        for key in _LOGGING_ENV_KEYS:
            os.environ.pop(key, None)
        for key, value in self._saved_env.items():
            if value is not None:
                os.environ[key] = value

        for module_name in (
            "logging_config",
            "_inductor.logging_utils",
            "torch_spyre.logging_config",
            "torch_spyre._inductor.logging_utils",
        ):
            if module_name in sys.modules:
                del sys.modules[module_name]

        ts_mod = sys.modules.get("torch_spyre")
        if ts_mod:
            for attr in ("logging_config", "_inductor"):
                if hasattr(ts_mod, attr):
                    delattr(ts_mod, attr)

        for name, module in self._saved_modules.items():
            if module is not None:
                sys.modules[name] = module

        for name, logger in self._saved_loggers.items():
            state = self._saved_logger_state[name]
            logger.handlers = state["handlers"]
            logger.setLevel(state["level"])
            logger.propagate = state["propagate"]
            logger.disabled = state["disabled"]

    def _load_module(self, module_name: str, relative_path: str) -> ModuleType:
        """Load a module directly from a file beneath the package root."""
        package_root = PACKAGE_ROOT
        assert package_root is not None
        full_path = package_root / relative_path
        spec = importlib.util.spec_from_file_location(module_name, full_path)
        assert spec is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        assert spec.loader is not None
        spec.loader.exec_module(module)
        return module

    @staticmethod
    def _ensure_package_module(package_name: str, package_path: Path) -> ModuleType:
        """Create a lightweight package module for direct submodule loading."""
        package = sys.modules.get(package_name)
        if isinstance(package, ModuleType):
            return package

        package = ModuleType(package_name)
        package.__file__ = str(package_path / "__init__.py")
        package.__package__ = package_name
        package.__path__ = [str(package_path)]
        package.__spec__ = importlib.machinery.ModuleSpec(
            name=package_name,
            loader=None,
            is_package=True,
        )
        sys.modules[package_name] = package
        return package

    def _reload_logging_modules(self) -> tuple[ModuleType, ModuleType]:
        """Reload the logging modules under the current environment settings."""
        if PACKAGE_ROOT is None:
            pytest.skip(
                "Could not locate torch_spyre package root for logging tests. "
                "Set TORCH_SPYRE_PACKAGE_ROOT to the directory containing "
                "logging_config.py if this checkout stores sources elsewhere."
            )

        logging_config = self._load_module("logging_config", "logging_config.py")

        inductor_package_name = "_inductor"
        if sys.modules.get(inductor_package_name) is None:
            assert PACKAGE_ROOT is not None
            self._ensure_package_module(
                inductor_package_name,
                PACKAGE_ROOT / "_inductor",
            )

        if "torch_spyre.logging_config" in sys.modules:
            del sys.modules["torch_spyre.logging_config"]
        ts_mod = sys.modules.get("torch_spyre")
        if ts_mod and hasattr(ts_mod, "logging_config"):
            delattr(ts_mod, "logging_config")

        logging_utils = self._load_module(
            "_inductor.logging_utils",
            "_inductor/logging_utils.py",
        )

        return logging_config, logging_utils


# ---------------------------------------------------------------------------
# Python logging tests: every level with 1, 2, and 3 components
# ---------------------------------------------------------------------------


class TestPythonLoggingSingleComponent(LoggingIsolationMixin):
    """Python logging with a single component configured via SPYRE_LOGS."""

    def test_debug_single_component(self) -> None:
        """DEBUG messages visible when one component set to DEBUG."""
        os.environ["SPYRE_LOGS"] = "spyre.inductor:DEBUG"
        logging_config, logging_utils = self._reload_logging_modules()

        logger = logging_utils.get_logger("codegen")
        assert logger.level == int(logging_config.LogLevel.DEBUG)

        with capture_logs("spyre.inductor.codegen", level="DEBUG") as captured:
            logger.debug("debug single component message")

        assert any("debug single component message" in msg for msg in captured.output)

    def test_info_single_component(self) -> None:
        """INFO messages visible when one component set to INFO."""
        os.environ["SPYRE_LOGS"] = "spyre.inductor:INFO"
        logging_config, logging_utils = self._reload_logging_modules()

        logger = logging_utils.get_logger("lowering")
        assert logger.level == int(logging_config.LogLevel.INFO)

        with capture_logs("spyre.inductor.lowering", level="INFO") as captured:
            logger.info("info single component message")

        assert any("info single component message" in msg for msg in captured.output)

    def test_warning_single_component(self) -> None:
        """WARNING messages visible when one component set to WARNING."""
        os.environ["SPYRE_LOGS"] = "spyre.inductor:WARNING"
        logging_config, logging_utils = self._reload_logging_modules()

        logger = logging_utils.get_logger("passes")
        assert logger.level == int(logging_config.LogLevel.WARNING)

        with capture_logs("spyre.inductor.passes", level="WARNING") as captured:
            logger.warning("warning single component message")

        assert any("warning single component message" in msg for msg in captured.output)

    def test_critical_single_component(self) -> None:
        """CRITICAL (fatal) messages visible when one component set to CRITICAL."""
        os.environ["SPYRE_LOGS"] = "spyre.inductor:CRITICAL"
        logging_config, logging_utils = self._reload_logging_modules()

        logger = logging_utils.get_logger("stickify")
        assert logger.level == int(logging_config.LogLevel.CRITICAL)

        with capture_logs("spyre.inductor.stickify", level="CRITICAL") as captured:
            logger.critical("critical single component message")

        assert any(
            "critical single component message" in msg for msg in captured.output
        )


class TestPythonLoggingTwoComponents(LoggingIsolationMixin):
    """Python logging with two components configured via SPYRE_LOGS."""

    def test_debug_two_components(self) -> None:
        """DEBUG messages visible on two independently configured components."""
        os.environ["SPYRE_LOGS"] = "spyre.inductor:DEBUG,spyre.runtime:DEBUG"
        logging_config, logging_utils = self._reload_logging_modules()

        inductor_logger = logging_utils.get_logger("codegen")
        runtime_logger = logging.getLogger("spyre.runtime")

        assert inductor_logger.level == int(logging_config.LogLevel.DEBUG)
        assert runtime_logger.level == int(logging_config.LogLevel.DEBUG)

        with capture_logs("spyre", level="DEBUG") as captured:
            inductor_logger.debug("debug inductor message")
            runtime_logger.debug("debug runtime message")

        output = "\n".join(captured.output)
        assert "debug inductor message" in output
        assert "debug runtime message" in output

    def test_info_two_components(self) -> None:
        """INFO messages visible on two independently configured components."""
        os.environ["SPYRE_LOGS"] = "spyre.inductor:INFO,spyre.execution:INFO"
        logging_config, logging_utils = self._reload_logging_modules()

        inductor_logger = logging_utils.get_logger("lowering")
        execution_logger = logging.getLogger("spyre.execution")

        assert inductor_logger.level == int(logging_config.LogLevel.INFO)
        assert execution_logger.level == int(logging_config.LogLevel.INFO)

        with capture_logs("spyre", level="INFO") as captured:
            inductor_logger.info("info inductor message")
            execution_logger.info("info execution message")

        output = "\n".join(captured.output)
        assert "info inductor message" in output
        assert "info execution message" in output

    def test_warning_two_components(self) -> None:
        """WARNING messages visible on two independently configured components."""
        os.environ["SPYRE_LOGS"] = "spyre.inductor:WARNING,spyre.device:WARNING"
        logging_config, logging_utils = self._reload_logging_modules()

        inductor_logger = logging_utils.get_logger("passes")
        device_logger = logging.getLogger("spyre.device")

        assert inductor_logger.level == int(logging_config.LogLevel.WARNING)
        assert device_logger.level == int(logging_config.LogLevel.WARNING)

        with capture_logs("spyre", level="WARNING") as captured:
            inductor_logger.warning("warning inductor message")
            device_logger.warning("warning device message")

        output = "\n".join(captured.output)
        assert "warning inductor message" in output
        assert "warning device message" in output

    def test_critical_two_components(self) -> None:
        """CRITICAL messages visible on two independently configured components."""
        os.environ["SPYRE_LOGS"] = "spyre.inductor:CRITICAL,spyre.runtime:CRITICAL"
        logging_config, logging_utils = self._reload_logging_modules()

        inductor_logger = logging_utils.get_logger("codegen")
        runtime_logger = logging.getLogger("spyre.runtime")

        assert inductor_logger.level == int(logging_config.LogLevel.CRITICAL)
        assert runtime_logger.level == int(logging_config.LogLevel.CRITICAL)

        with capture_logs("spyre", level="CRITICAL") as captured:
            inductor_logger.critical("critical inductor message")
            runtime_logger.critical("critical runtime message")

        output = "\n".join(captured.output)
        assert "critical inductor message" in output
        assert "critical runtime message" in output


class TestPythonLoggingThreeComponents(LoggingIsolationMixin):
    """Python logging with three components configured via SPYRE_LOGS."""

    def test_debug_three_components(self) -> None:
        """DEBUG messages visible across three configured components."""
        os.environ["SPYRE_LOGS"] = (
            "spyre.inductor:DEBUG,spyre.runtime:DEBUG,spyre.execution:DEBUG"
        )
        logging_config, logging_utils = self._reload_logging_modules()

        inductor_logger = logging_utils.get_logger("codegen")
        runtime_logger = logging.getLogger("spyre.runtime")
        execution_logger = logging.getLogger("spyre.execution")

        assert inductor_logger.level == int(logging_config.LogLevel.DEBUG)
        assert runtime_logger.level == int(logging_config.LogLevel.DEBUG)
        assert execution_logger.level == int(logging_config.LogLevel.DEBUG)

        with capture_logs("spyre", level="DEBUG") as captured:
            inductor_logger.debug("debug inductor three")
            runtime_logger.debug("debug runtime three")
            execution_logger.debug("debug execution three")

        output = "\n".join(captured.output)
        assert "debug inductor three" in output
        assert "debug runtime three" in output
        assert "debug execution three" in output

    def test_info_three_components(self) -> None:
        """INFO messages visible across three configured components."""
        os.environ["SPYRE_LOGS"] = (
            "spyre.inductor:INFO,spyre.runtime:INFO,spyre.device:INFO"
        )
        logging_config, logging_utils = self._reload_logging_modules()

        inductor_logger = logging_utils.get_logger("lowering")
        runtime_logger = logging.getLogger("spyre.runtime")
        device_logger = logging.getLogger("spyre.device")

        assert inductor_logger.level == int(logging_config.LogLevel.INFO)
        assert runtime_logger.level == int(logging_config.LogLevel.INFO)
        assert device_logger.level == int(logging_config.LogLevel.INFO)

        with capture_logs("spyre", level="INFO") as captured:
            inductor_logger.info("info inductor three")
            runtime_logger.info("info runtime three")
            device_logger.info("info device three")

        output = "\n".join(captured.output)
        assert "info inductor three" in output
        assert "info runtime three" in output
        assert "info device three" in output

    def test_warning_three_components(self) -> None:
        """WARNING messages visible across three configured components."""
        os.environ["SPYRE_LOGS"] = (
            "spyre.inductor:WARNING,spyre.execution:WARNING,spyre.device:WARNING"
        )
        logging_config, logging_utils = self._reload_logging_modules()

        inductor_logger = logging_utils.get_logger("stickify")
        execution_logger = logging.getLogger("spyre.execution")
        device_logger = logging.getLogger("spyre.device")

        assert inductor_logger.level == int(logging_config.LogLevel.WARNING)
        assert execution_logger.level == int(logging_config.LogLevel.WARNING)
        assert device_logger.level == int(logging_config.LogLevel.WARNING)

        with capture_logs("spyre", level="WARNING") as captured:
            inductor_logger.warning("warning inductor three")
            execution_logger.warning("warning execution three")
            device_logger.warning("warning device three")

        output = "\n".join(captured.output)
        assert "warning inductor three" in output
        assert "warning execution three" in output
        assert "warning device three" in output

    def test_critical_three_components(self) -> None:
        """CRITICAL messages visible across three configured components."""
        os.environ["SPYRE_LOGS"] = (
            "spyre.inductor:CRITICAL,spyre.runtime:CRITICAL,spyre.execution:CRITICAL"
        )
        logging_config, logging_utils = self._reload_logging_modules()

        inductor_logger = logging_utils.get_logger("passes")
        runtime_logger = logging.getLogger("spyre.runtime")
        execution_logger = logging.getLogger("spyre.execution")

        assert inductor_logger.level == int(logging_config.LogLevel.CRITICAL)
        assert runtime_logger.level == int(logging_config.LogLevel.CRITICAL)
        assert execution_logger.level == int(logging_config.LogLevel.CRITICAL)

        with capture_logs("spyre", level="CRITICAL") as captured:
            inductor_logger.critical("critical inductor three")
            runtime_logger.critical("critical runtime three")
            execution_logger.critical("critical execution three")

        output = "\n".join(captured.output)
        assert "critical inductor three" in output
        assert "critical runtime three" in output
        assert "critical execution three" in output

    def test_mixed_levels_three_components(self) -> None:
        """Different log levels across three components."""
        os.environ["SPYRE_LOGS"] = (
            "spyre.inductor:DEBUG,spyre.runtime:WARNING,spyre.execution:INFO"
        )
        logging_config, logging_utils = self._reload_logging_modules()

        inductor_logger = logging_utils.get_logger("codegen")
        runtime_logger = logging.getLogger("spyre.runtime")
        execution_logger = logging.getLogger("spyre.execution")

        assert inductor_logger.level == int(logging_config.LogLevel.DEBUG)
        assert runtime_logger.level == int(logging_config.LogLevel.WARNING)
        assert execution_logger.level == int(logging_config.LogLevel.INFO)


# ---------------------------------------------------------------------------
# C++ DEBUGINFO macro tests: 1, 2, and 3 components
# ---------------------------------------------------------------------------


def _find_torch_spyre_lib() -> Path | None:
    """Find the compiled _C extension to verify C++ is available."""
    if PACKAGE_ROOT is None:
        return None
    c_ext = PACKAGE_ROOT / "_C"
    if c_ext.exists():
        return c_ext
    for so_file in PACKAGE_ROOT.glob("_C*.so"):
        return so_file
    return None


def _run_subprocess_with_env(env_vars: dict[str, str], script: str) -> str:
    """Run a Python script in a subprocess with specified env vars.

    Returns stdout as a string.
    """
    import subprocess

    env = os.environ.copy()
    env.update(env_vars)
    if "TORCH_SPYRE_DEBUG" not in env_vars:
        env.pop("TORCH_SPYRE_DEBUG", None)

    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        env=env,
        timeout=60,
    )
    return result.stdout + result.stderr


@pytest.mark.skipif(
    _find_torch_spyre_lib() is None,
    reason="torch_spyre C++ extension not built",
)
class TestCppLoggingSingleComponent:
    """C++ DEBUGINFO macro controlled by the legacy TORCH_SPYRE_DEBUG env var."""

    def test_debuginfo_enabled_single(self) -> None:
        """DEBUGINFO produces output when TORCH_SPYRE_DEBUG=1."""
        script = """\
import torch_spyre
import torch
t = torch.zeros(2, 2, dtype=torch.float16)
"""
        output = _run_subprocess_with_env({"TORCH_SPYRE_DEBUG": "1"}, script)
        assert "Traceback" not in output

    def test_debuginfo_disabled_single(self) -> None:
        """DEBUGINFO produces no output when TORCH_SPYRE_DEBUG is unset."""
        script = """\
import torch_spyre
import torch
t = torch.zeros(2, 2, dtype=torch.float16)
print("CLEAN_EXIT")
"""
        output = _run_subprocess_with_env({}, script)
        assert "CLEAN_EXIT" in output

    def test_debuginfo_zero_disables_single(self) -> None:
        """DEBUGINFO is disabled when TORCH_SPYRE_DEBUG=0."""
        script = """\
import torch_spyre
print("CLEAN_EXIT")
"""
        output = _run_subprocess_with_env({"TORCH_SPYRE_DEBUG": "0"}, script)
        assert "CLEAN_EXIT" in output


@pytest.mark.skipif(
    _find_torch_spyre_lib() is None,
    reason="torch_spyre C++ extension not built",
)
class TestCppLoggingTwoComponents:
    """C++ DEBUGINFO (legacy TORCH_SPYRE_DEBUG) combined with Python SPYRE_LOGS."""

    def test_cpp_and_python_debug_two_components(self) -> None:
        """Both C++ DEBUGINFO and Python debug logging active together."""
        script = """\
import os
os.environ["TORCH_SPYRE_DEBUG"] = "1"
os.environ["SPYRE_LOGS"] = "spyre.inductor:DEBUG,spyre.runtime:DEBUG"
import torch_spyre
from torch_spyre._inductor import logging_utils
logger_ind = logging_utils.get_logger("codegen")
import logging
logger_rt = logging.getLogger("spyre.runtime")
logger_ind.debug("PY_INDUCTOR_DEBUG")
logger_rt.debug("PY_RUNTIME_DEBUG")
print("TWO_COMP_EXIT")
"""
        output = _run_subprocess_with_env(
            {
                "TORCH_SPYRE_DEBUG": "1",
                "SPYRE_LOGS": "spyre.inductor:DEBUG,spyre.runtime:DEBUG",
            },
            script,
        )
        assert "TWO_COMP_EXIT" in output
        assert "Traceback" not in output

    def test_cpp_disabled_python_enabled_two_components(self) -> None:
        """C++ disabled, Python DEBUG on two components."""
        script = """\
import os
os.environ["SPYRE_LOGS"] = "spyre.inductor:DEBUG,spyre.execution:DEBUG"
import torch_spyre
from torch_spyre._inductor import logging_utils
import logging
logger_ind = logging_utils.get_logger("codegen")
logger_exec = logging.getLogger("spyre.execution")
logger_ind.debug("PY_IND_ONLY")
logger_exec.debug("PY_EXEC_ONLY")
print("TWO_COMP_NO_CPP_EXIT")
"""
        output = _run_subprocess_with_env(
            {"SPYRE_LOGS": "spyre.inductor:DEBUG,spyre.execution:DEBUG"},
            script,
        )
        assert "TWO_COMP_NO_CPP_EXIT" in output
        assert "Traceback" not in output


@pytest.mark.skipif(
    _find_torch_spyre_lib() is None,
    reason="torch_spyre C++ extension not built",
)
class TestCppLoggingThreeComponents:
    """C++ DEBUGINFO combined with Python SPYRE_LOGS on three components."""

    def test_cpp_and_python_debug_three_components(self) -> None:
        """C++ DEBUGINFO and Python DEBUG on three components simultaneously."""
        script = """\
import os
os.environ["TORCH_SPYRE_DEBUG"] = "1"
os.environ["SPYRE_LOGS"] = "spyre.inductor:DEBUG,spyre.runtime:DEBUG,spyre.execution:DEBUG"
import torch_spyre
from torch_spyre._inductor import logging_utils
import logging
logger_ind = logging_utils.get_logger("codegen")
logger_rt = logging.getLogger("spyre.runtime")
logger_exec = logging.getLogger("spyre.execution")
logger_ind.debug("PY_IND_THREE")
logger_rt.debug("PY_RT_THREE")
logger_exec.debug("PY_EXEC_THREE")
print("THREE_COMP_EXIT")
"""
        output = _run_subprocess_with_env(
            {
                "TORCH_SPYRE_DEBUG": "1",
                "SPYRE_LOGS": "spyre.inductor:DEBUG,spyre.runtime:DEBUG,spyre.execution:DEBUG",
            },
            script,
        )
        assert "THREE_COMP_EXIT" in output
        assert "Traceback" not in output

    def test_cpp_enabled_python_mixed_three_components(self) -> None:
        """C++ DEBUGINFO on, Python at mixed levels across three components."""
        script = """\
import os
os.environ["TORCH_SPYRE_DEBUG"] = "1"
os.environ["SPYRE_LOGS"] = "spyre.inductor:DEBUG,spyre.runtime:WARNING,spyre.device:INFO"
import torch_spyre
from torch_spyre._inductor import logging_utils
import logging
logger_ind = logging_utils.get_logger("codegen")
logger_rt = logging.getLogger("spyre.runtime")
logger_dev = logging.getLogger("spyre.device")
logger_ind.debug("PY_IND_MIX")
logger_rt.warning("PY_RT_MIX")
logger_dev.info("PY_DEV_MIX")
print("THREE_MIXED_EXIT")
"""
        output = _run_subprocess_with_env(
            {
                "TORCH_SPYRE_DEBUG": "1",
                "SPYRE_LOGS": "spyre.inductor:DEBUG,spyre.runtime:WARNING,spyre.device:INFO",
            },
            script,
        )
        assert "THREE_MIXED_EXIT" in output
        assert "Traceback" not in output


# ---------------------------------------------------------------------------
# Additional thorough test cases
# ---------------------------------------------------------------------------


class TestSpyreLogsPrecedence(LoggingIsolationMixin):
    """SPYRE_LOGS takes precedence over legacy env vars."""

    def test_spyre_logs_overrides_legacy_inductor_log(self) -> None:
        """SPYRE_LOGS overrides SPYRE_INDUCTOR_LOG when both are set."""
        os.environ["SPYRE_INDUCTOR_LOG"] = "1"
        os.environ["SPYRE_INDUCTOR_LOG_LEVEL"] = "INFO"
        os.environ["SPYRE_LOGS"] = "spyre.inductor:DEBUG"

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            logging_config, _ = self._reload_logging_modules()

        assert logging_config.get_effective_config()["spyre.inductor"] == "DEBUG"
        assert logging_config.get_config_source("spyre.inductor") == "SPYRE_LOGS"

    def test_spyre_logs_overrides_torch_spyre_debug(self) -> None:
        """SPYRE_LOGS overrides TORCH_SPYRE_DEBUG when both are set."""
        os.environ["TORCH_SPYRE_DEBUG"] = "1"
        os.environ["SPYRE_LOGS"] = "spyre.inductor:INFO"

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            logging_config, _ = self._reload_logging_modules()

        assert logging_config.get_effective_config()["spyre.inductor"] == "INFO"
        assert logging_config.get_config_source("spyre.inductor") == "SPYRE_LOGS"

    def test_spyre_logs_overrides_torch_logs(self) -> None:
        """SPYRE_LOGS overrides TORCH_LOGS spyre.* entries."""
        os.environ["TORCH_LOGS"] = "spyre.inductor:INFO"
        os.environ["SPYRE_LOGS"] = "spyre.inductor:DEBUG"

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            logging_config, _ = self._reload_logging_modules()

        assert logging_config.get_effective_config()["spyre.inductor"] == "DEBUG"
        assert logging_config.get_config_source("spyre.inductor") == "SPYRE_LOGS"


class TestSpyreLogsDisable(LoggingIsolationMixin):
    """Tests for disabling logging via SPYRE_LOGS."""

    def test_disable_with_minus_prefix(self) -> None:
        """The '-' prefix disables a component."""
        os.environ["SPYRE_LOGS"] = "-spyre.inductor"
        logging_config, logging_utils = self._reload_logging_modules()

        logger = logging_utils.get_logger("codegen")
        assert logger.level == int(logging_config.LogLevel.DISABLED)

    def test_disable_does_not_emit_messages(self) -> None:
        """A disabled component should not emit any log messages."""
        os.environ["SPYRE_LOGS"] = "-spyre.inductor"
        _, logging_utils = self._reload_logging_modules()

        logger = logging_utils.get_logger("codegen")

        with capture_logs("spyre.inductor.codegen", level="DEBUG") as captured:
            logger.debug("should not appear")
            logger.info("should not appear")
            logger.warning("should not appear")
            logger.critical("should not appear")

        assert len(captured.records) == 0


class TestSpyreLogsPlusPrefix(LoggingIsolationMixin):
    """Tests for the '+' shorthand prefix in SPYRE_LOGS."""

    def test_plus_prefix_enables_at_info(self) -> None:
        """The '+' prefix enables a component at INFO level."""
        os.environ["SPYRE_LOGS"] = "+spyre.inductor"
        logging_config, logging_utils = self._reload_logging_modules()

        logger = logging_utils.get_logger("codegen")
        assert logger.level == int(logging_config.LogLevel.INFO)
        assert logging_config.get_config_source("spyre.inductor") == "SPYRE_LOGS"

    def test_plus_prefix_info_visible_debug_hidden(self) -> None:
        """With '+' prefix, INFO is visible but DEBUG is suppressed."""
        os.environ["SPYRE_LOGS"] = "+spyre.inductor"
        _, logging_utils = self._reload_logging_modules()

        logger = logging_utils.get_logger("codegen")

        with capture_logs("spyre.inductor.codegen", level="INFO") as captured:
            logger.info("visible info message")

        assert any("visible info message" in msg for msg in captured.output)


class TestSpyreLogsInvalidInput(LoggingIsolationMixin):
    """Tests for invalid SPYRE_LOGS values."""

    def test_invalid_level_emits_warning(self) -> None:
        """Invalid log level in SPYRE_LOGS emits a warning."""
        os.environ["SPYRE_LOGS"] = "spyre.inductor:INVALID_LEVEL"

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            self._reload_logging_modules()

        messages = [str(w.message) for w in caught]
        assert any("Invalid log level" in msg for msg in messages)

    def test_non_spyre_entries_ignored(self) -> None:
        """Non-spyre entries in SPYRE_LOGS are silently ignored."""
        os.environ["SPYRE_LOGS"] = "other.module:DEBUG,spyre.inductor:INFO"
        logging_config, _ = self._reload_logging_modules()

        assert logging_config.get_effective_config()["spyre.inductor"] == "INFO"
        assert "other.module" not in logging_config.get_effective_config()

    def test_empty_spyre_logs_uses_defaults(self) -> None:
        """Empty SPYRE_LOGS falls back to default WARNING levels."""
        os.environ["SPYRE_LOGS"] = ""
        logging_config, _ = self._reload_logging_modules()

        assert logging_config.get_effective_config()["spyre.inductor"] == "WARNING"
        assert logging_config.get_config_source("spyre.inductor") == "default"


class TestSpyreLogsHierarchyPropagation(LoggingIsolationMixin):
    """Tests for hierarchical level propagation from parent to child."""

    def test_parent_level_propagates_to_children(self) -> None:
        """Setting spyre.inductor:DEBUG propagates to child components."""
        os.environ["SPYRE_LOGS"] = "spyre.inductor:DEBUG"
        logging_config, logging_utils = self._reload_logging_modules()

        codegen_logger = logging_utils.get_logger("codegen")
        lowering_logger = logging_utils.get_logger("lowering")
        passes_logger = logging_utils.get_logger("passes")

        assert codegen_logger.level == int(logging_config.LogLevel.DEBUG)
        assert lowering_logger.level == int(logging_config.LogLevel.DEBUG)
        assert passes_logger.level == int(logging_config.LogLevel.DEBUG)

    def test_root_spyre_propagates_to_all(self) -> None:
        """Setting spyre:INFO propagates to all child components."""
        os.environ["SPYRE_LOGS"] = "spyre:INFO"
        logging_config, _ = self._reload_logging_modules()

        config = logging_config.get_effective_config()
        assert config["spyre"] == "INFO"
        assert config["spyre.inductor"] == "INFO"
        assert config["spyre.runtime"] == "INFO"
        assert config["spyre.execution"] == "INFO"
        assert config["spyre.device"] == "INFO"


class TestSpyreLogsFileOutput(LoggingIsolationMixin):
    """Tests for log file output configuration."""

    def test_spyre_logs_with_programmatic_file_output(self) -> None:
        """Verify messages written to file contain correct format."""
        os.environ["SPYRE_LOGS"] = "spyre.inductor:DEBUG"

        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = os.path.join(tmpdir, "test_spyre.log")
            logging_config, logging_utils = self._reload_logging_modules()

            logging_config.set_log_file(log_path)

            logger = logging_utils.get_logger("codegen")
            logger.debug("file output debug message")
            logger.info("file output info message")
            logger.warning("file output warning message")
            logger.critical("file output critical message")

            spyre_logger = logging.getLogger("spyre")
            for handler in spyre_logger.handlers:
                flush = getattr(handler, "flush", None)
                if flush is not None:
                    flush()

            with open(log_path, encoding="utf-8") as handle:
                contents = handle.read()

        assert "[DEBUG] [spyre.inductor.codegen] file output debug message" in contents
        assert "[INFO] [spyre.inductor.codegen] file output info message" in contents
        assert (
            "[WARNING] [spyre.inductor.codegen] file output warning message" in contents
        )
        assert (
            "[CRITICAL] [spyre.inductor.codegen] file output critical message"
            in contents
        )

    def test_legacy_spyre_log_file_still_works(self) -> None:
        """SPYRE_LOG_FILE legacy var still configures file output."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = os.path.join(tmpdir, "legacy.log")
            os.environ["SPYRE_LOG_FILE"] = log_path
            os.environ["SPYRE_LOGS"] = "spyre.inductor:WARNING"

            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                logging_config, _ = self._reload_logging_modules()

            messages = [str(w.message) for w in caught]
            assert any("SPYRE_LOG_FILE is deprecated" in msg for msg in messages)

            output_config = logging_config.get_output_config()
            assert output_config["log_file"] == log_path
            assert output_config["log_file_source"] == "legacy:SPYRE_LOG_FILE"


class TestSpyreLogsConfigSourceTracking(LoggingIsolationMixin):
    """Tests for config source tracking."""

    def test_default_source(self) -> None:
        """Unconfigured components report 'default' as source."""
        os.environ.pop("SPYRE_LOGS", None)
        os.environ.pop("TORCH_LOGS", None)
        os.environ.pop("SPYRE_INDUCTOR_LOG", None)
        os.environ.pop("TORCH_SPYRE_DEBUG", None)
        logging_config, _ = self._reload_logging_modules()

        assert logging_config.get_config_source("spyre.inductor") == "default"

    def test_spyre_logs_source(self) -> None:
        """Components configured via SPYRE_LOGS report correct source."""
        os.environ["SPYRE_LOGS"] = "spyre.inductor:DEBUG"
        logging_config, _ = self._reload_logging_modules()

        assert logging_config.get_config_source("spyre.inductor") == "SPYRE_LOGS"

    def test_programmatic_source(self) -> None:
        """Programmatically configured components report 'programmatic'."""
        logging_config, _ = self._reload_logging_modules()
        logging_config.set_log_level("spyre.inductor", "DEBUG")

        assert logging_config.get_config_source("spyre.inductor") == "programmatic"

    def test_legacy_torch_logs_source(self) -> None:
        """TORCH_LOGS with spyre entries reports 'legacy:TORCH_LOGS'."""
        os.environ.pop("SPYRE_LOGS", None)
        os.environ["TORCH_LOGS"] = "spyre.inductor:DEBUG"

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            logging_config, _ = self._reload_logging_modules()

        assert logging_config.get_config_source("spyre.inductor") == "legacy:TORCH_LOGS"


class TestSpyreLogsMultiEntryParsing(LoggingIsolationMixin):
    """Tests for comma-separated multi-entry SPYRE_LOGS parsing."""

    def test_comma_separated_entries(self) -> None:
        """Multiple comma-separated entries are all applied."""
        os.environ["SPYRE_LOGS"] = (
            "spyre.inductor:DEBUG,spyre.runtime:INFO,spyre.device:ERROR"
        )
        logging_config, _ = self._reload_logging_modules()

        config = logging_config.get_effective_config()
        assert config["spyre.inductor"] == "DEBUG"
        assert config["spyre.runtime"] == "INFO"
        assert config["spyre.device"] == "ERROR"

    def test_whitespace_tolerance(self) -> None:
        """Parser tolerates whitespace around entries."""
        os.environ["SPYRE_LOGS"] = " spyre.inductor : DEBUG , spyre.runtime : INFO "
        logging_config, _ = self._reload_logging_modules()

        config = logging_config.get_effective_config()
        assert config["spyre.inductor"] == "DEBUG"
        assert config["spyre.runtime"] == "INFO"

    def test_mixed_prefixes_and_explicit_levels(self) -> None:
        """Mix of +, -, and explicit level entries."""
        os.environ["SPYRE_LOGS"] = "+spyre.inductor,-spyre.runtime,spyre.device:ERROR"
        logging_config, _ = self._reload_logging_modules()

        config = logging_config.get_effective_config()
        assert config["spyre.inductor"] == "INFO"
        assert config["spyre.runtime"] == "DISABLED"
        assert config["spyre.device"] == "ERROR"


class TestSpyreLogsLevelFiltering(LoggingIsolationMixin):
    """Tests verifying that level filtering suppresses lower-priority messages."""

    def test_warning_level_suppresses_info_and_debug(self) -> None:
        """At WARNING level, INFO and DEBUG messages are suppressed."""
        os.environ["SPYRE_LOGS"] = "spyre.inductor:WARNING"
        logging_config, logging_utils = self._reload_logging_modules()

        logger = logging_utils.get_logger("codegen")
        assert logger.level == int(logging_config.LogLevel.WARNING)

        with capture_logs("spyre.inductor.codegen", level="WARNING") as captured:
            logger.warning("visible warning")
            logger.critical("visible critical")

        output = "\n".join(captured.output)
        assert "visible warning" in output
        assert "visible critical" in output

        with capture_logs("spyre.inductor.codegen", level="WARNING") as captured:
            logger.debug("invisible debug")
            logger.info("invisible info")

        assert len(captured.records) == 0

    def test_info_level_suppresses_debug(self) -> None:
        """At INFO level, DEBUG messages are suppressed."""
        os.environ["SPYRE_LOGS"] = "spyre.inductor:INFO"
        logging_config, logging_utils = self._reload_logging_modules()

        logger = logging_utils.get_logger("codegen")

        with capture_logs("spyre.inductor.codegen", level="INFO") as captured:
            logger.info("visible info")
            logger.warning("visible warning")

        output = "\n".join(captured.output)
        assert "visible info" in output
        assert "visible warning" in output

        with capture_logs("spyre.inductor.codegen", level="INFO") as captured:
            logger.debug("invisible debug")

        assert len(captured.records) == 0


class TestSpyreLogsProgrammaticAPI(LoggingIsolationMixin):
    """Tests for the programmatic logging configuration API."""

    def test_enable_convenience_function(self) -> None:
        """enable() sets a component to INFO level."""
        logging_config, logging_utils = self._reload_logging_modules()
        logging_config.enable("spyre.inductor")

        logger = logging_utils.get_logger("codegen")
        assert logger.level == int(logging_config.LogLevel.INFO)

    def test_disable_convenience_function(self) -> None:
        """disable() sets a component to DISABLED level."""
        logging_config, logging_utils = self._reload_logging_modules()
        logging_config.disable("spyre.inductor.codegen")

        logger = logging_utils.get_logger("codegen")
        assert logger.level == int(logging_config.LogLevel.DISABLED)

    def test_set_log_level_invalid_raises(self) -> None:
        """set_log_level with invalid level raises ValueError."""
        logging_config, _ = self._reload_logging_modules()

        with pytest.raises(ValueError):
            logging_config.set_log_level("spyre.inductor", "NOT_A_LEVEL")

    def test_list_components_returns_all_defaults(self) -> None:
        """list_components() returns all default component names."""
        logging_config, _ = self._reload_logging_modules()
        components = logging_config.list_components()

        assert "spyre" in components
        assert "spyre.inductor" in components
        assert "spyre.runtime" in components
        assert "spyre.execution" in components
        assert "spyre.device" in components

    def test_get_config_for_cpp(self) -> None:
        """get_config_for_cpp() returns (component, int_level) tuples."""
        os.environ["SPYRE_LOGS"] = "spyre.inductor:DEBUG"
        logging_config, _ = self._reload_logging_modules()

        cpp_config = logging_config.get_config_for_cpp()
        assert isinstance(cpp_config, list)
        assert all(isinstance(item, tuple) and len(item) == 2 for item in cpp_config)

        inductor_entry = next(
            (comp, level) for comp, level in cpp_config if comp == "spyre.inductor"
        )
        assert inductor_entry[1] == int(logging_config.LogLevel.DEBUG)
