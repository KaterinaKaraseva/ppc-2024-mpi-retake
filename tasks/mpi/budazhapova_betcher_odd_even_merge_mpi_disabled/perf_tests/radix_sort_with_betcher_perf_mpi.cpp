#include <gtest/gtest.h>

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <ctime>
#include <memory>
#include <random>
#include <vector>

#include "boost/mpi/communicator.hpp"
#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"
#include "mpi/budazhapova_betcher_odd_even_merge_mpi/include/radix_sort_with_betcher.h"

namespace budazhapova_betcher_odd_even_merge_mpi {
namespace {
std::vector<int> GenerateRandomVector(int size, int min_value, int max_value) {
  std::vector<int> random_vector(size);
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> dis(min_value, max_value);
  for (int i = 0; i < size; ++i) {
    random_vector[i] = dis(gen);
  }

  return random_vector;
}
}  // namespace
}  // namespace budazhapova_betcher_odd_even_merge_mpi

TEST(budazhapova_betcher_odd_even_merge_mpi, test_pipeline_run) {
  boost::mpi::communicator world;
  std::vector<int> input_vector = budazhapova_betcher_odd_even_merge_mpi::GenerateRandomVector(12000000, 5, 100);
  std::vector<int> out(12000000, 0);

  std::shared_ptr<ppc::core::TaskData> task_data_par = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    task_data_par->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_vector.data()));
    task_data_par->inputs_count.emplace_back(input_vector.size());
    task_data_par->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
    task_data_par->outputs_count.emplace_back(out.size());
  }

  auto test_mpi_task_par = std::make_shared<budazhapova_betcher_odd_even_merge_mpi::MergeParallel>(task_data_par);
  ASSERT_EQ(test_mpi_task_par->ValidationImpl(), true);
  test_mpi_task_par->PreProcessingImpl();
  test_mpi_task_par->RunImpl();
  test_mpi_task_par->PostProcessingImpl();

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

  auto perf_analyzer = std::make_shared<ppc::core::Perf>(test_mpi_task_par);
  perf_analyzer->PipelineRun(perf_attr, perf_results);
  // Create Perf analyzer
  if (world.rank() == 0) {
    ppc::core::Perf::PrintPerfStatistic(perf_results);
    ASSERT_EQ(input_vector, out);
  }
}
TEST(budazhapova_betcher_odd_even_merge_mpi, test_task_run) {
  boost::mpi::communicator world;
  std::vector<int> input_vector = budazhapova_betcher_odd_even_merge_mpi::GenerateRandomVector(12000000, 5, 100);
  std::vector<int> out(12000000, 0);
  std::shared_ptr<ppc::core::TaskData> task_data_par = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    task_data_par->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_vector.data()));
    task_data_par->inputs_count.emplace_back(input_vector.size());
    task_data_par->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
    task_data_par->outputs_count.emplace_back(out.size());
  }

  auto test_mpi_task_par = std::make_shared<budazhapova_betcher_odd_even_merge_mpi::MergeParallel>(task_data_par);
  ASSERT_EQ(test_mpi_task_par->ValidationImpl(), true);
  test_mpi_task_par->PreProcessingImpl();
  test_mpi_task_par->RunImpl();
  test_mpi_task_par->PostProcessingImpl();

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
  auto perf_analyzer = std::make_shared<ppc::core::Perf>(test_mpi_task_par);
  perf_analyzer->TaskRun(perf_attr, perf_results);
  // Create Perf analyzer
  if (world.rank() == 0) {
    ppc::core::Perf::PrintPerfStatistic(perf_results);
    ASSERT_EQ(input_vector, out);
  }
}