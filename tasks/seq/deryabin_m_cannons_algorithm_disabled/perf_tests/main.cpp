#include <gtest/gtest.h>

#include <chrono>
#include <cstdint>
#include <memory>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"
#include "seq/deryabin_m_cannons_algorithm/include/ops_seq.hpp"

TEST(deryabin_m_cannons_algorithm_seq, test_pipeline_run_Seq) {
  std::vector<double> input_matrix_a = std::vector<double>(10000, 0);
  std::vector<double> input_matrix_b = std::vector<double>(10000, 0);
  std::vector<double> output_matrix_c = std::vector<double>(10000, 0);
  for (unsigned short dimension = 0; dimension < 100; dimension++) {
    input_matrix_a[dimension * 101] = 1;
    input_matrix_b[dimension * 101] = 1;
  }
  std::vector<std::vector<double>> in_matrix_a(1, input_matrix_a);
  std::vector<std::vector<double>> in_matrix_b(1, input_matrix_b);
  std::vector<std::vector<double>> out_matrix_c(1, output_matrix_c);

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in_matrix_a.data()));
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in_matrix_b.data()));
  task_data_seq->inputs_count.emplace_back(input_matrix_a.size());
  task_data_seq->inputs_count.emplace_back(input_matrix_b.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out_matrix_c.data()));
  task_data_seq->outputs_count.emplace_back(out_matrix_c.size());

  auto cannons_algorithm_task_sequential =
      std::make_shared<deryabin_m_cannons_algorithm_seq::CannonsAlgorithmTaskSequential>(task_data_seq);

  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perf_results = std::make_shared<ppc::core::PerfResults>();

  auto perf_analyzer = std::make_shared<ppc::core::Perf>(cannons_algorithm_task_sequential);
  perf_analyzer->PipelineRun(perf_attr, perf_results);
  ppc::core::Perf::PrintPerfStatistic(perf_results);
  ASSERT_EQ(in_matrix_a[0], out_matrix_c[0]);
}

TEST(deryabin_m_cannons_algorithm_seq, test_task_run_Seq) {
  std::vector<double> input_matrix_a = std::vector<double>(10000, 0);
  std::vector<double> input_matrix_b = std::vector<double>(10000, 0);
  std::vector<double> output_matrix_c = std::vector<double>(10000, 0);
  for (unsigned short dimension = 0; dimension < 100; dimension++) {
    input_matrix_a[dimension * 101] = 1;
    input_matrix_b[dimension * 101] = 1;
  }
  std::vector<std::vector<double>> in_matrix_a(1, input_matrix_a);
  std::vector<std::vector<double>> in_matrix_b(1, input_matrix_b);
  std::vector<std::vector<double>> out_matrix_c(1, output_matrix_c);

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in_matrix_a.data()));
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in_matrix_b.data()));
  task_data_seq->inputs_count.emplace_back(input_matrix_a.size());
  task_data_seq->inputs_count.emplace_back(input_matrix_b.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out_matrix_c.data()));
  task_data_seq->outputs_count.emplace_back(out_matrix_c.size());

  auto cannons_algorithm_task_sequential =
      std::make_shared<deryabin_m_cannons_algorithm_seq::CannonsAlgorithmTaskSequential>(task_data_seq);

  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perf_results = std::make_shared<ppc::core::PerfResults>();

  auto perf_analyzer = std::make_shared<ppc::core::Perf>(cannons_algorithm_task_sequential);
  perf_analyzer->TaskRun(perf_attr, perf_results);
  ppc::core::Perf::PrintPerfStatistic(perf_results);
  ASSERT_EQ(in_matrix_a[0], out_matrix_c[0]);
}
