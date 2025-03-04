#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <memory>
#include <random>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"
#include "seq/kavtorev_d_radix_double_sort/include/ops_seq.hpp"

using kavtorev_d_radix_double_sort::RadixSortSequential;

TEST(kavtorev_d_radix_double_sort_seq, test_pipeline_run) {
  int n = 10000000;
  std::vector<double> input_data(n);
  std::vector<double> output_data(n, 0.0);

  {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dist(-1e9, 1e9);
    for (int i = 0; i < n; ++i) {
      input_data[i] = dist(gen);
    }
  }

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&n));
  task_data_seq->inputs_count.emplace_back(1);

  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_data.data()));
  task_data_seq->inputs_count.emplace_back(n);

  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(output_data.data()));
  task_data_seq->outputs_count.emplace_back(n);

  auto test_task_sequential = std::make_shared<kavtorev_d_radix_double_sort::RadixSortSequential>(task_data_seq);
  ASSERT_TRUE(test_task_sequential->ValidationImpl());
  test_task_sequential->PreProcessingImpl();
  test_task_sequential->RunImpl();
  test_task_sequential->PostProcessingImpl();

  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 5;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perf_results = std::make_shared<ppc::core::PerfResults>();

  auto perf_analyzer = std::make_shared<ppc::core::Perf>(test_task_sequential);
  perf_analyzer->PipelineRun(perf_attr, perf_results);

  ppc::core::Perf::PrintPerfStatistic(perf_results);

  std::vector<double> ref_data = input_data;
  std::ranges::sort(ref_data.begin(), ref_data.end());
  for (int i = 0; i < n; ++i) {
    ASSERT_NEAR(ref_data[i], output_data[i], 1e-12);
  }
}

TEST(kavtorev_d_radix_double_sort_seq, test_task_run) {
  int n = 1000000;
  std::vector<double> input_data(n);
  std::vector<double> output_data(n, 0.0);

  {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dist(-1e9, 1e9);
    for (int i = 0; i < n; ++i) {
      input_data[i] = dist(gen);
    }
  }

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&n));
  task_data_seq->inputs_count.emplace_back(1);

  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_data.data()));
  task_data_seq->inputs_count.emplace_back(n);

  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(output_data.data()));
  task_data_seq->outputs_count.emplace_back(n);

  auto test_task_sequential = std::make_shared<kavtorev_d_radix_double_sort::RadixSortSequential>(task_data_seq);
  ASSERT_TRUE(test_task_sequential->ValidationImpl());
  test_task_sequential->PreProcessingImpl();
  test_task_sequential->RunImpl();
  test_task_sequential->PostProcessingImpl();

  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 5;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perf_results = std::make_shared<ppc::core::PerfResults>();

  auto perf_analyzer = std::make_shared<ppc::core::Perf>(test_task_sequential);
  perf_analyzer->PipelineRun(perf_attr, perf_results);

  ppc::core::Perf::PrintPerfStatistic(perf_results);

  std::vector<double> ref_data = input_data;
  std::ranges::sort(ref_data.begin(), ref_data.end());
  for (int i = 0; i < n; ++i) {
    ASSERT_NEAR(ref_data[i], output_data[i], 1e-12);
  }
}