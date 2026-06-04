/*
 * Copyright 2026 The Torch-Spyre Authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "logging_config.h"

namespace py = pybind11;

namespace torch_spyre {
namespace logging {

void init_logging_bindings(py::module& m) {
  // Create logging submodule
  auto logging_module = m.def_submodule("_logging", "C++ logging interface");

  // Expose LogLevel enum
  py::enum_<LogLevel>(logging_module, "LogLevel")
      .value("NOTSET", LogLevel::NOTSET)
      .value("DEBUG", LogLevel::DEBUG)
      .value("INFO", LogLevel::INFO)
      .value("WARNING", LogLevel::WARNING)
      .value("ERROR", LogLevel::ERROR)
      .value("CRITICAL", LogLevel::CRITICAL)
      .export_values();

  // Expose LoggingConfig class
  py::class_<LoggingConfig>(logging_module, "LoggingConfig")
      .def_static("instance", &LoggingConfig::instance,
                  py::return_value_policy::reference)
      .def("initialize_from_python", &LoggingConfig::initialize_from_python,
           py::arg("config"),
           "Initialize C++ logging from Python configuration")
      .def("get_log_level", &LoggingConfig::get_log_level, py::arg("component"),
           "Get log level for a component")
      .def("set_log_level", &LoggingConfig::set_log_level, py::arg("component"),
           py::arg("level"), "Set log level for a component")
      .def("is_enabled", &LoggingConfig::is_enabled, py::arg("component"),
           py::arg("level"),
           "Check if logging is enabled for component at level")
      .def("get_components", &LoggingConfig::get_components,
           "Get list of all configured components");

  // Utility functions
  logging_module.def("log_level_to_string", &log_level_to_string,
                     py::arg("level"), "Convert log level to string");
  logging_module.def("string_to_log_level", &string_to_log_level,
                     py::arg("level_str"), "Convert string to log level");
}

}  // namespace logging
}  // namespace torch_spyre
