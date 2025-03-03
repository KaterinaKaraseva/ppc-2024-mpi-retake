#include <gtest/gtest.h>

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"
#include "seq/Konstantinov_I_sum_of_vector_elements/include/ops_seq.hpp"

namespace konstantinov_i_sum_of_vector_elements_seq {
namespace {
std::vector<int> GenerateRandVector(int size, int lower_bound = 0, int upper_bound = 50) {
  std::vector<int> result(size);
  for (int i = 0; i < size; i++) {
    result[i] = lower_bound + rand() % (upper_bound - lower_bound + 1);
  }
  return result;
}

std::vector<std::vector<int>> GenerateRandMatrix(int rows, int columns, int lower_bound = 0, int upper_bound = 50) {
  std::vector<std::vector<int>> result(rows);
  for (int i = 0; i < rows; i++) {
    result[i] = konstantinov_i_sum_of_vector_elements_seq::GenerateRandVector(columns, lower_bound, upper_bound);
  }
  return result;
}
}  // namespace
}  // namespace konstantinov_i_sum_of_vector_elements_seq

TEST(Konstantinov_I_sum_of_vector_seq, test_pipeline_run) {
  int rows = 10000;
  int columns = 10000;
  int result = 0;
  std::vector<std::vector<int>> input = konstantinov_i_sum_of_vector_elements_seq::GenerateRandMatrix(rows, columns);
  int sum = 0;
  for (const std::vector<int> &vec : input) {
    for (int elem : vec) {
      sum += elem;
    }
  }
  std::shared_ptr<ppc::core::TaskData> task_data_par = std::make_shared<ppc::core::TaskData>();

  task_data_par->inputs_count.emplace_back(rows);
  task_data_par->inputs_count.emplace_back(columns);
  for (auto &row : input) {
    task_data_par->inputs.emplace_back(reinterpret_cast<uint8_t *>(row.data()));
  }
  task_data_par->outputs_count.emplace_back(1);
  task_data_par->outputs.emplace_back(reinterpret_cast<uint8_t *>(&result));

  // Create Task
  auto test = std::make_shared<konstantinov_i_sum_of_vector_elements_seq::SumVecElemSequential>(task_data_par);

  // Create Perf attributes
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
  auto perf_analyzer = std::make_shared<ppc::core::Perf>(test);
  perf_analyzer->PipelineRun(perf_attr, perf_results);
  ppc::core::Perf::PrintPerfStatistic(perf_results);
  ASSERT_EQ(sum, result);
}

TEST(Konstantinov_I_sum_of_vector_seq, test_task_run) {
  int rows = 10000;
  int columns = 10000;
  int result = 0;
  std::vector<std::vector<int>> input = konstantinov_i_sum_of_vector_elements_seq::GenerateRandMatrix(rows, columns);
  int sum = 0;
  for (const std::vector<int> &vec : input) {
    for (int elem : vec) {
      sum += elem;
    }
  }
  std::shared_ptr<ppc::core::TaskData> task_data_par = std::make_shared<ppc::core::TaskData>();

  task_data_par->inputs_count.emplace_back(rows);
  task_data_par->inputs_count.emplace_back(columns);
  for (auto &row : input) {
    task_data_par->inputs.emplace_back(reinterpret_cast<uint8_t *>(row.data()));
  }
  task_data_par->outputs_count.emplace_back(1);
  task_data_par->outputs.emplace_back(reinterpret_cast<uint8_t *>(&result));

  // Create Task
  auto test = std::make_shared<konstantinov_i_sum_of_vector_elements_seq::SumVecElemSequential>(task_data_par);
  // Create Perf attributes
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
  auto perf_analyzer = std::make_shared<ppc::core::Perf>(test);
  perf_analyzer->TaskRun(perf_attr, perf_results);
  ppc::core::Perf::PrintPerfStatistic(perf_results);
  ASSERT_EQ(sum, result);
}