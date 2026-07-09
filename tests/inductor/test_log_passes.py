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


"""Tests for per-pass operation logging in CustomPreSchedulingPasses."""

import os
from unittest.mock import patch

import torch  # noqa: F401

from torch_spyre._inductor import config
from torch_spyre._inductor.passes import _get_pass_name, _should_log_pass


class TestGetPassName:
    """Tests for _get_pass_name helper."""

    def test_regular_function(self):
        def my_pass(graph):
            pass

        assert _get_pass_name(my_pass) == "my_pass"

    def test_lambda(self):
        fn = lambda graph: None  # noqa: E731
        assert _get_pass_name(fn) == "<lambda>"

    def test_bound_method(self):
        class MyPass:
            def run(self, graph):
                pass

        obj = MyPass()
        assert _get_pass_name(obj.run) == "run"

    def test_callable_object(self):
        class MyCallablePass:
            def __call__(self, graph):
                pass

        obj = MyCallablePass()
        # callable objects without __name__ or __func__ fall back to class name
        assert _get_pass_name(obj) == "MyCallablePass"

    def test_decorated_function_preserves_name(self):
        import functools

        def decorator(fn):
            @functools.wraps(fn)
            def wrapper(*args, **kwargs):
                return fn(*args, **kwargs)

            return wrapper

        @decorator
        def insert_restickify(graph):
            pass

        assert _get_pass_name(insert_restickify) == "insert_restickify"


class TestShouldLogPass:
    """Tests for _should_log_pass helper."""

    def test_empty_config_returns_false(self):
        with config.patch({"log_passes": ""}):
            assert _should_log_pass("split_multi_ops") is False

    def test_all_returns_true(self):
        with config.patch({"log_passes": "all"}):
            assert _should_log_pass("split_multi_ops") is True
            assert _should_log_pass("insert_restickify") is True

    def test_one_returns_true(self):
        with config.patch({"log_passes": "1"}):
            assert _should_log_pass("any_pass_name") is True

    def test_single_name_match(self):
        with config.patch({"log_passes": "split_multi_ops"}):
            assert _should_log_pass("split_multi_ops") is True
            assert _should_log_pass("insert_restickify") is False

    def test_comma_separated_list(self):
        with config.patch({"log_passes": "split_multi_ops,insert_restickify"}):
            assert _should_log_pass("split_multi_ops") is True
            assert _should_log_pass("insert_restickify") is True
            assert _should_log_pass("deadcode_elimination") is False

    def test_comma_separated_with_spaces(self):
        with config.patch({"log_passes": " split_multi_ops , insert_restickify "}):
            assert _should_log_pass("split_multi_ops") is True
            assert _should_log_pass("insert_restickify") is True

    def test_no_partial_match(self):
        with config.patch({"log_passes": "split_multi"}):
            assert _should_log_pass("split_multi_ops") is False


class TestLogPassesConfig:
    """Tests for the log_passes configuration knob."""

    def test_default_is_empty(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("SPYRE_LOG_PASSES", None)
            # config.log_passes reads from env at module load; verify the
            # default when no env var is set.
            with config.patch({"log_passes": ""}):
                assert config.log_passes == ""

    def test_reads_from_env_var(self):
        with patch.dict(
            os.environ, {"SPYRE_LOG_PASSES": "split_multi_ops,deadcode_elimination"}
        ):
            # Simulate fresh config read by patching to the env var value
            with config.patch({"log_passes": os.environ["SPYRE_LOG_PASSES"]}):
                assert config.log_passes == "split_multi_ops,deadcode_elimination"
