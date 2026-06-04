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

#pragma once

#include <atomic>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

namespace torch_spyre {
namespace logging {

// Log levels matching Python's logging module
enum class LogLevel : int {
  NOTSET = 0,
  DEBUG = 10,
  INFO = 20,
  WARNING = 30,
  ERROR = 40,
  CRITICAL = 50
};

// Convert log level to string
const char* log_level_to_string(LogLevel level);

// Convert string to log level
LogLevel string_to_log_level(const std::string& level_str);

/**
 * Unified logging configuration manager for C++ components.
 *
 * This class provides:
 * - Thread-safe access to logging configuration
 * - Integration with Python logging_config module
 * - Hierarchical component lookup
 * - Zero-overhead when logging is disabled
 */
class LoggingConfig {
 public:
  // Get singleton instance
  static LoggingConfig& instance();

  // Initialize from Python configuration
  // Called once during module initialization via pybind11
  void initialize_from_python(
      const std::vector<std::pair<std::string, int>>& config);

  // Get log level for a component
  // Thread-safe, lock-free read after initialization
  LogLevel get_log_level(const std::string& component) const;

  // Check if logging is enabled for a component at a given level
  // Optimized for fast path (disabled logging)
  inline bool is_enabled(const std::string& component, LogLevel level) const {
    return get_log_level(component) <= level;
  }

  // Set log level programmatically (for testing)
  void set_log_level(const std::string& component, LogLevel level);

  // Get all configured components
  std::vector<std::string> get_components() const;

 private:
  LoggingConfig() = default;
  ~LoggingConfig() = default;

  // Prevent copying
  LoggingConfig(const LoggingConfig&) = delete;
  LoggingConfig& operator=(const LoggingConfig&) = delete;

  // Configuration storage
  std::unordered_map<std::string, LogLevel> config_;

  // Mutex for configuration updates (not needed for reads after init)
  mutable std::mutex mutex_;

  // Initialization flag
  std::atomic<bool> initialized_{false};

  // Resolve log level with hierarchical lookup
  LogLevel resolve_log_level(const std::string& component) const;
};

/**
 * RAII logger class for structured logging.
 *
 * Usage:
 *   Logger log("spyre.runtime", LogLevel::DEBUG);
 *   if (log.is_enabled()) {
 *       log.debug() << "Message: " << value;
 *   }
 */
class Logger {
 public:
  Logger(const std::string& component, LogLevel level);

  // Check if logging is enabled at the requested level
  bool is_enabled() const;

  // Stream-based logging
  class LogStream {
   public:
    LogStream(const std::string& component, LogLevel level, bool enabled);
    ~LogStream();

    template <typename T>
    LogStream& operator<<(const T& value) {
      if (enabled_) {
        stream_ << value;
      }
      return *this;
    }

   private:
    std::string component_;
    LogLevel level_;
    bool enabled_;
    std::ostringstream stream_;
  };

  LogStream debug();
  LogStream info();
  LogStream warning();
  LogStream error();

 private:
  std::string component_;
  LogLevel requested_level_;
  LogLevel min_level_;
};

// Convenience macros for logging
#define SPYRE_LOG_ENABLED(component, level) \
  torch_spyre::logging::LoggingConfig::instance().is_enabled(component, level)

#define SPYRE_LOG(component, level)                                        \
  if (SPYRE_LOG_ENABLED(component, torch_spyre::logging::LogLevel::level)) \
  torch_spyre::logging::Logger(component,                                  \
                               torch_spyre::logging::LogLevel::level)      \
      .level()

// Component-specific macros
#define SPYRE_RUNTIME_DEBUG() SPYRE_LOG("spyre.runtime", DEBUG)
#define SPYRE_RUNTIME_INFO() SPYRE_LOG("spyre.runtime", INFO)
#define SPYRE_RUNTIME_WARNING() SPYRE_LOG("spyre.runtime", WARNING)
#define SPYRE_RUNTIME_ERROR() SPYRE_LOG("spyre.runtime", ERROR)

}  // namespace logging
}  // namespace torch_spyre
