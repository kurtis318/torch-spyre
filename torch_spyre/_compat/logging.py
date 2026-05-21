# Copyright 2025-2026 The Torch-Spyre Authors.
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

"""Backward compatibility layer for legacy logging configuration.

This module provides compatibility shims to support legacy logging
patterns during the transition period. It will be removed in a future release.
"""

import os
import warnings
from typing import Optional


def check_legacy_env_vars() -> bool:
    """Check if any legacy environment variables are set.

    Returns:
        True if legacy variables detected, False otherwise
    """
    legacy_vars = [
        "SPYRE_INDUCTOR_LOG",
        "SPYRE_INDUCTOR_LOG_LEVEL",
        "TORCH_SPYRE_DEBUG",
        "SPYRE_LOG_FILE",
    ]

    return any(os.environ.get(var) for var in legacy_vars)


def emit_migration_warning():
    """Emit a one-time migration warning if legacy variables are used."""
    if not check_legacy_env_vars():
        return

    warnings.warn(
        "\n"
        + "=" * 70
        + "\n"
        + "DEPRECATION WARNING: Legacy logging environment variables detected.\n"
        + "\n"
        + "Please migrate to TORCH_LOGS / unified logging configuration:\n"
        + "  Old: SPYRE_INDUCTOR_LOG=1 SPYRE_INDUCTOR_LOG_LEVEL=DEBUG\n"
        + "  New: TORCH_LOGS='spyre.inductor:DEBUG'\n"
        + "\n"
        + "  Old: TORCH_SPYRE_DEBUG=1\n"
        + "  New: TORCH_LOGS='spyre.runtime:DEBUG'\n"
        + "\n"
        + "  Old: SPYRE_LOG_FILE=/tmp/spyre.log\n"
        + "  New: use the unified logging configuration API\n"
        + "       (legacy value is still mapped for backward compatibility)\n"
        + "\n"
        + "Legacy variables will be removed in version 2.0.0.\n"
        + "See documentation: https://docs.example.com/logging-migration\n"
        + "=" * 70,
        DeprecationWarning,
        stacklevel=2,
    )


def get_legacy_log_file() -> Optional[str]:
    """Get log file path from legacy SPYRE_LOG_FILE variable.

    Returns:
        Log file path if set, None otherwise
    """
    log_file = os.environ.get("SPYRE_LOG_FILE")
    if log_file:
        warnings.warn(
            "SPYRE_LOG_FILE is deprecated and is mapped to the top-level "
            "'spyre' logger file handler for backward compatibility. "
            "Prefer the unified logging configuration API.",
            DeprecationWarning,
            stacklevel=2,
        )
    return log_file


emit_migration_warning()
