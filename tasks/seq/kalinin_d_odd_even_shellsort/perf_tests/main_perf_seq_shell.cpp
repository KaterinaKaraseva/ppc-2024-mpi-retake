#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <memory>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"
#include "seq/kalinin_d_odd_even_shellsort/include/header_seq_odd_even_shell.hpp"

TEST(kalinin_d_odd_even_shell_seq, test_pipline_run_seq) {
  const int n = 3000000;
  // Create data
  std::vector<int> arr(n);
  std::vector<int> out(n);
  kalinin_d_odd_even_shell_seq::GimmeRandVec(arr);
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(arr.data()));
  task_data_seq->inputs_count.emplace_back(arr.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  // crate task
  auto test_task_seq = std::make_shared<kalinin_d_odd_even_shell_seq::OddEvenShellSeq>(task_data_seq);

  // create perf attrib
  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  // Create and init perf results
  auto perf_results = std::make_shared<ppc::core::PerfResults>();

  auto perf_analyzer = std::make_shared<ppc::core::Perf>(test_task_seq);
  perf_analyzer->PipelineRun(perf_attr, perf_results);
  ppc::core::Perf::PrintPerfStatistic(perf_results);
  std::ranges::sort(arr.begin(), arr.end());
  ASSERT_EQ(arr, out);
}

TEST(kalinin_d_odd_even_shell_seq, test_task_run_seq) {
  const int n = 3000000;
  // Create data
  std::vector<int> arr(n);
  std::vector<int> out(n);
  kalinin_d_odd_even_shell_seq::GimmeRandVec(arr);
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(arr.data()));
  task_data_seq->inputs_count.emplace_back(arr.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  // crate task
  auto test_task_seq = std::make_shared<kalinin_d_odd_even_shell_seq::OddEvenShellSeq>(task_data_seq);

  // create perf attrib
  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  // Create and init perf results
  auto perf_results = std::make_shared<ppc::core::PerfResults>();

  // Create Perf analyzer
  auto perf_analyzer = std::make_shared<ppc::core::Perf>(test_task_seq);
  perf_analyzer->TaskRun(perf_attr, perf_results);
  ppc::core::Perf::PrintPerfStatistic(perf_results);
  std::ranges::sort(arr.begin(), arr.end());
  ASSERT_EQ(arr, out);
}