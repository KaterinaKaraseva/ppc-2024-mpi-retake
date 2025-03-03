#include <gtest/gtest.h>

#include <chrono>
#include <cstdint>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "boost/mpi/communicator.hpp"
#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"
#include "mpi/strakhov_a_char_freq_counter/include/ops_mpi.hpp"

namespace strakhov_a_char_freq_counter_mpi {
namespace {
std::vector<char> FillRandomChars(int size, const std::string &charset) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> dist(0, static_cast<int>(charset.size()) - 1);
  std::vector<char> result(size);
  for (char &c : result) {
    c = charset[dist(gen)];
  }
  return result;
}
}  // namespace
}  // namespace strakhov_a_char_freq_counter_mpi

TEST(strakhov_a_char_freq_counter_mpi, test_pipeline_run) {
  boost::mpi::communicator world;

  // Create data
  std::vector<char> in_string = strakhov_a_char_freq_counter_mpi::FillRandomChars(
      30000000, "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789!@#$%^&*");
  std::vector<int> out_par(1, 0);
  std::vector<char> in_target = strakhov_a_char_freq_counter_mpi::FillRandomChars(
      1, "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789!@#$%^&*");

  // Create taskdata
  auto task_data_mpi_par = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    task_data_mpi_par->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_string.data()));
    task_data_mpi_par->inputs_count.emplace_back(in_string.size());
    task_data_mpi_par->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_target.data()));
    task_data_mpi_par->inputs_count.emplace_back(in_target.size());
    task_data_mpi_par->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_par.data()));
    task_data_mpi_par->outputs_count.emplace_back(out_par.size());
  }
  // Create Task
  auto test_task_mpi = std::make_shared<strakhov_a_char_freq_counter_mpi::CharFreqCounterPar>(task_data_mpi_par);
  ASSERT_EQ(test_task_mpi->ValidationImpl(), true);
  test_task_mpi->PreProcessingImpl();
  test_task_mpi->RunImpl();
  test_task_mpi->PostProcessingImpl();

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
  auto perf_analyzer = std::make_shared<ppc::core::Perf>(test_task_mpi);
  perf_analyzer->PipelineRun(perf_attr, perf_results);

  if (world.rank() == 0) {
    ppc::core::Perf::PrintPerfStatistic(perf_results);
  }
}

TEST(strakhov_a_char_freq_counter_mpi, test_task_run) {
  boost::mpi::communicator world;

  // Create data
  std::vector<char> in_string = strakhov_a_char_freq_counter_mpi::FillRandomChars(
      30000000, "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789!@#$%^&*");
  std::vector<int> out_par(1, 0);
  std::vector<char> in_target = strakhov_a_char_freq_counter_mpi::FillRandomChars(
      1, "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789!@#$%^&*");

  // Create task_data
  auto task_data_mpi_par = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    task_data_mpi_par->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_string.data()));
    task_data_mpi_par->inputs_count.emplace_back(in_string.size());
    task_data_mpi_par->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_target.data()));
    task_data_mpi_par->inputs_count.emplace_back(in_target.size());
    task_data_mpi_par->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_par.data()));
    task_data_mpi_par->outputs_count.emplace_back(out_par.size());
  }
  // Create Task
  auto test_task_mpi = std::make_shared<strakhov_a_char_freq_counter_mpi::CharFreqCounterPar>(task_data_mpi_par);
  ASSERT_EQ(test_task_mpi->ValidationImpl(), true);
  test_task_mpi->PreProcessingImpl();
  test_task_mpi->RunImpl();
  test_task_mpi->PostProcessingImpl();

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
  auto perf_analyzer = std::make_shared<ppc::core::Perf>(test_task_mpi);
  perf_analyzer->TaskRun(perf_attr, perf_results);

  if (world.rank() == 0) {
    ppc::core::Perf::PrintPerfStatistic(perf_results);
  }
}
