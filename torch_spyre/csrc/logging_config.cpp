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

#include "logging_config.h"

#include <ctime>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace torch_spyre {
namespace logging {

// Log level string conversion
const char* log_level_to_string(LogLevel level) {
  switch (level) {
    case LogLevel::DEBUG:
      return "DEBUG";
    case LogLevel::INFO:
      return "INFO";
    case LogLevel::WARNING:
      return "WARNING";
    case LogLevel::ERROR:
      return "ERROR";
    case LogLevel::CRITICAL:
      return "CRITICAL";
    default:
      return "NOTSET";
  }
}

LogLevel string_to_log_level(const std::string& level_str) {
  if (level_str == "DEBUG") return LogLevel::DEBUG;
  if (level_str == "INFO") return LogLevel::INFO;
  if (level_str == "WARNING") return LogLevel::WARNING;
  if (level_str == "ERROR") return LogLevel::ERROR;
  if (level_str == "CRITICAL") return LogLevel::CRITICAL;
  return LogLevel::NOTSET;
}

// LoggingConfig implementation
LoggingConfig& LoggingConfig::instance() {
  static LoggingConfig instance;
  return instance;
}

void LoggingConfig::initialize_from_python(
    const std::vector<std::pair<std::string, int>>& config) {
  std::lock_guard<std::mutex> lock(mutex_);

  config_.clear();
  for (const auto& [component, level] : config) {
    config_[component] = static_cast<LogLevel>(level);
  }

  initialized_.store(true, std::memory_order_release);
}

LogLevel LoggingConfig::get_log_level(const std::string& component) const {
  // Fast path: check if initialized
  if (!initialized_.load(std::memory_order_acquire)) {
    return LogLevel::WARNING;  // Default before initialization
  }

  return resolve_log_level(component);
}

LogLevel LoggingConfig::resolve_log_level(const std::string& component) const {
  // Exact match
  auto it = config_.find(component);
  if (it != config_.end()) {
    return it->second;
  }

  // Walk up hierarchy
  std::string current = component;
  while (true) {
    auto pos = current.rfind('.');
    if (pos == std::string::npos) {
      break;
    }

    current = current.substr(0, pos);
    it = config_.find(current);
    if (it != config_.end()) {
      return it->second;
    }
  }

  return LogLevel::WARNING;  // Ultimate fallback
}

void LoggingConfig::set_log_level(const std::string& component,
                                  LogLevel level) {
  std::lock_guard<std::mutex> lock(mutex_);
  config_[component] = level;
}

std::vector<std::string> LoggingConfig::get_components() const {
  std::lock_guard<std::mutex> lock(mutex_);

  std::vector<std::string> components;
  components.reserve(config_.size());

  for (const auto& [component, _] : config_) {
    components.push_back(component);
  }

  return components;
}

// Logger implementation
Logger::Logger(const std::string& component, LogLevel level)
    : component_(component),
      requested_level_(level),
      min_level_(LoggingConfig::instance().get_log_level(component)) {}

bool Logger::is_enabled() const {
  return min_level_ <= requested_level_;
}

Logger::LogStream Logger::debug() {
  return LogStream(component_, LogLevel::DEBUG, min_level_ <= LogLevel::DEBUG);
}

Logger::LogStream Logger::info() {
  return LogStream(component_, LogLevel::INFO, min_level_ <= LogLevel::INFO);
}

Logger::LogStream Logger::warning() {
  return LogStream(component_, LogLevel::WARNING,
                   min_level_ <= LogLevel::WARNING);
}

Logger::LogStream Logger::error() {
  return LogStream(component_, LogLevel::ERROR, min_level_ <= LogLevel::ERROR);
}

// LogStream implementation
Logger::LogStream::LogStream(const std::string& component, LogLevel level,
                             bool enabled)
    : component_(component), level_(level), enabled_(enabled) {}

Logger::LogStream::~LogStream() {
  if (enabled_ && !stream_.str().empty()) {
    // Get current time
    auto now = std::time(nullptr);
    char time_buf[32];
    std::strftime(time_buf, sizeof(time_buf), "%Y-%m-%d %H:%M:%S",
                  std::localtime(&now));

    // Output formatted log message
    std::cerr << "[" << log_level_to_string(level_) << "] "
              << "[" << component_ << "] " << time_buf << " " << stream_.str()
              << std::endl;
  }
}

}  // namespace logging
}  // namespace torch_spyre
