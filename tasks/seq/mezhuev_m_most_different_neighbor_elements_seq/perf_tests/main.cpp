#include <gtest/gtest.h>

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <random>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"
#include "seq/mezhuev_m_most_different_neighbor_elements_seq/include/seq.hpp"

TEST(mezhuev_m_most_different_neighbor_elements_seq, PipelineRunPerformance) {
  constexpr int kTestSize = 10000000;

  std::vector<int> input_data(kTestSize);
  std::vector<int> output_data(2);

  std::mt19937 rng(42);
  std::uniform_int_distribution<int> dist(-1000000, 1000000);
  for (int& val : input_data) {
    val = dist(rng);
  }

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.push_back(reinterpret_cast<uint8_t*>(input_data.data()));
  task_data->inputs_count.push_back(static_cast<uint32_t>(input_data.size()));
  task_data->outputs.push_back(reinterpret_cast<uint8_t*>(output_data.data()));
  task_data->outputs_count.push_back(2);

  auto task =
      std::make_shared<mezhuev_m_most_different_neighbor_elements_seq::MostDifferentNeighborElements>(task_data);

  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 20;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perf_results = std::make_shared<ppc::core::PerfResults>();

  auto perf_analyzer = std::make_shared<ppc::core::Perf>(task);
  perf_analyzer->PipelineRun(perf_attr, perf_results);

  ppc::core::Perf::PrintPerfStatistic(perf_results);
  EXPECT_EQ(output_data.size(), static_cast<size_t>(2));
}

TEST(mezhuev_m_most_different_neighbor_elements_seq, TaskRunPerformance) {
  constexpr int kTestSize = 10000000;

  std::vector<int> input_data(kTestSize);
  std::vector<int> output_data(2);

  std::mt19937 rng(42);
  std::uniform_int_distribution<int> dist(-1000000, 1000000);
  for (int& val : input_data) {
    val = dist(rng);
  }

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.push_back(reinterpret_cast<uint8_t*>(input_data.data()));
  task_data->inputs_count.push_back(static_cast<uint32_t>(input_data.size()));
  task_data->outputs.push_back(reinterpret_cast<uint8_t*>(output_data.data()));
  task_data->outputs_count.push_back(2);

  auto task =
      std::make_shared<mezhuev_m_most_different_neighbor_elements_seq::MostDifferentNeighborElements>(task_data);

  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 30;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perf_results = std::make_shared<ppc::core::PerfResults>();

  auto perf_analyzer = std::make_shared<ppc::core::Perf>(task);
  perf_analyzer->TaskRun(perf_attr, perf_results);

  ppc::core::Perf::PrintPerfStatistic(perf_results);

  EXPECT_EQ(output_data.size(), static_cast<size_t>(2));
}