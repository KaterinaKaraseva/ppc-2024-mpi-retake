#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/timer.hpp>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"
#include "mpi/Konstantinov_I_sum_of_vector_elements/include/ops_mpi.hpp"

namespace konstantinov_i_sum_of_vector_elements_mpi {
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
    result[i] = konstantinov_i_sum_of_vector_elements_mpi::GenerateRandVector(columns, lower_bound, upper_bound);
  }
  return result;
}
}  // namespace
}  // namespace konstantinov_i_sum_of_vector_elements_mpi

TEST(Konstantinov_I_sum_of_vector_mpi, test_pipeline_run) {
  boost::mpi::communicator world;
  int rows = 10000;
  int columns = 10000;
  int result = 0;
  std::vector<std::vector<int>> input =
      konstantinov_i_sum_of_vector_elements_mpi::GenerateRandMatrix(rows, columns, 1, 1);
  std::shared_ptr<ppc::core::TaskData> task_data_par = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    task_data_par->inputs_count.emplace_back(rows);
    task_data_par->inputs_count.emplace_back(columns);
    for (long unsigned int i = 0; i < input.size(); i++) {
      task_data_par->inputs.emplace_back(reinterpret_cast<uint8_t *>(input[i].data()));
    }
    task_data_par->outputs_count.emplace_back(1);
    task_data_par->outputs.emplace_back(reinterpret_cast<uint8_t *>(&result));
  }
  auto test = std::make_shared<konstantinov_i_sum_of_vector_elements_mpi::SumVecElemParallel>(task_data_par);

  test->ValidationImpl();
  test->PreProcessingImpl();
  test->RunImpl();
  test->PostProcessingImpl();

  // Create Perf attributes
  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 10;
  const boost::mpi::timer current_timer;
  perf_attr->current_timer = [&] { return current_timer.elapsed(); };

  // Create and init perf results
  auto perf_results = std::make_shared<ppc::core::PerfResults>();

  // Create Perf analyzer
  auto perf_analyzer = std::make_shared<ppc::core::Perf>(test);
  perf_analyzer->PipelineRun(perf_attr, perf_results);

  if (world.rank() == 0) {
    ppc::core::Perf::PrintPerfStatistic(perf_results);
    ASSERT_EQ(rows * columns, result);
  }
}

TEST(Konstantinov_I_sum_of_vector_mpi, test_task_run) {
  boost::mpi::communicator world;
  int rows = 15000;
  int columns = 15000;
  int result = 0;
  std::vector<std::vector<int>> input =
      konstantinov_i_sum_of_vector_elements_mpi::GenerateRandMatrix(rows, columns, 1, 1);
  std::shared_ptr<ppc::core::TaskData> task_data_par = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    task_data_par->inputs_count.emplace_back(rows);
    task_data_par->inputs_count.emplace_back(columns);
    for (long unsigned int i = 0; i < input.size(); i++) {
      task_data_par->inputs.emplace_back(reinterpret_cast<uint8_t *>(input[i].data()));
    }
    task_data_par->outputs_count.emplace_back(1);
    task_data_par->outputs.emplace_back(reinterpret_cast<uint8_t *>(&result));
  }
  auto test = std::make_shared<konstantinov_i_sum_of_vector_elements_mpi::SumVecElemParallel>(task_data_par);

  test->ValidationImpl();
  test->PreProcessingImpl();
  test->RunImpl();
  test->PostProcessingImpl();

  // Create Perf attributes
  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 10;
  const boost::mpi::timer current_timer;
  perf_attr->current_timer = [&] { return current_timer.elapsed(); };

  // Create and init perf results
  auto perf_results = std::make_shared<ppc::core::PerfResults>();

  // Create Perf analyzer
  auto perf_analyzer = std::make_shared<ppc::core::Perf>(test);
  perf_analyzer->TaskRun(perf_attr, perf_results);

  if (world.rank() == 0) {
    ppc::core::Perf::PrintPerfStatistic(perf_results);
    ASSERT_EQ(rows * columns, result);
  }
}