#include <gtest/gtest.h>

#include <chrono>
#include <cstdint>
#include <memory>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"
#include "seq/khovansky_d_rectangles_integral/include/ops_seq.hpp"

TEST(khovansky_d_rectangles_integral_seq, test_pipeline_run_seq) {
  const int num_dimensions = 3;
  std::vector<double> lower_limits = {0.0, 0.0, 0.0};
  std::vector<double> upper_limits = {1.0, 1.0, 1.0};
  int num_partitions = 100;
  double integral_result = 0.0;

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs_count.emplace_back(num_dimensions);
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(lower_limits.data()));
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(upper_limits.data()));
  task_data->inputs_count.emplace_back(num_partitions);
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(&integral_result));

  auto test_task = std::make_shared<khovansky_d_rectangles_integral_seq::RectanglesSeq>(task_data);
  test_task->integrand_function = [](const std::vector<double> &point) { return point[0] * point[1] * point[2]; };

  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 50;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perf_results = std::make_shared<ppc::core::PerfResults>();

  auto perf_analyzer = std::make_shared<ppc::core::Perf>(test_task);
  perf_analyzer->PipelineRun(perf_attr, perf_results);
  ppc::core::Perf::PrintPerfStatistic(perf_results);
  ASSERT_NEAR(integral_result, 1.0 / 8.0, 1e-1);
}

TEST(khovansky_d_rectangles_integral_seq, test_task_run_seq) {
  const int num_dimensions = 3;
  std::vector<double> lower_limits = {0.0, 0.0, 0.0};
  std::vector<double> upper_limits = {1.0, 1.0, 1.0};
  int num_partitions = 100;
  double integral_result = 0.0;

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs_count.emplace_back(num_dimensions);
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(lower_limits.data()));
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(upper_limits.data()));
  task_data->inputs_count.emplace_back(num_partitions);
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(&integral_result));

  auto test_task = std::make_shared<khovansky_d_rectangles_integral_seq::RectanglesSeq>(task_data);
  test_task->integrand_function = [](const std::vector<double> &point) { return point[0] * point[1] * point[2]; };

  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 50;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perf_results = std::make_shared<ppc::core::PerfResults>();

  auto perf_analyzer = std::make_shared<ppc::core::Perf>(test_task);
  perf_analyzer->TaskRun(perf_attr, perf_results);
  ppc::core::Perf::PrintPerfStatistic(perf_results);
  ASSERT_NEAR(integral_result, 1.0 / 8.0, 1e-1);
}