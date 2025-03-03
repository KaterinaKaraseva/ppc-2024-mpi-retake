// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/timer.hpp>
#include <cstdint>
#include <cstdlib>
#include <ctime>
#include <memory>
#include <string>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"
#include "mpi/budazhapova_e_count_freq_character/include/count_freq_chart_mpi_header.hpp"

namespace budazhapova_e_count_freq_chart_mpi {
namespace {
std::string GetRandomString(long long length) {
  static std::string charset = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890";
  std::string result;
  result.resize(length);

  srand(time(nullptr));
  for (int i = 0; i < length; i++) {
    result[i] = charset[rand() % charset.size()];
  }
  return result;
}
}  // namespace
}  // namespace budazhapova_e_count_freq_chart_mpi

TEST(budazhapova_e_count_freq_chart_mpi, test_pipeline_run) {
  boost::mpi::communicator world;
  std::string global_str;
  long long size_string = 123456789;
  global_str = budazhapova_e_count_freq_chart_mpi::GetRandomString(size_string);
  std::vector<int> global_out(1, 0);
  char symb = 'a';
  // Create TaskData

  std::shared_ptr<ppc::core::TaskData> task_data_par = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    task_data_par->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_str.data()));
    task_data_par->inputs_count.emplace_back(global_str.size());
    task_data_par->inputs.emplace_back(reinterpret_cast<uint8_t*>(&symb));
    task_data_par->inputs_count.emplace_back(1);
    task_data_par->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_out.data()));
    task_data_par->outputs_count.emplace_back(global_out.size());
  }
  auto test_mpi_task_parallel =
      std::make_shared<budazhapova_e_count_freq_chart_mpi::TestMPITaskParallel>(task_data_par);
  ASSERT_EQ(test_mpi_task_parallel->Validation(), true);
  test_mpi_task_parallel->PreProcessing();
  test_mpi_task_parallel->Run();
  test_mpi_task_parallel->PostProcessing();
  // Create Perf attributes
  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 10;
  const boost::mpi::timer current_timer;
  perf_attr->current_timer = [&] { return current_timer.elapsed(); };
  // Create and init perf results
  auto perf_results = std::make_shared<ppc::core::PerfResults>();
  // Create Perf analyzer
  auto perf_analyzer = std::make_shared<ppc::core::Perf>(test_mpi_task_parallel);
  perf_analyzer->PipelineRun(perf_attr, perf_results);
  if (world.rank() == 0) {
    ppc::core::Perf::PrintPerfStatistic(perf_results);
    // ASSERT_EQ(, global_out[0]);
  }
}

TEST(budazhapova_e_count_freq_chart_mpi, test_task_run) {
  boost::mpi::communicator world;
  std::string global_str;
  long long size_string = 123456789;
  global_str = budazhapova_e_count_freq_chart_mpi ::GetRandomString(size_string);
  std::vector<int> global_out(1, 0);
  char symb = 'a';

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> task_data_par = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    task_data_par->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_str.data()));
    task_data_par->inputs_count.emplace_back(global_str.size());
    task_data_par->inputs.emplace_back(reinterpret_cast<uint8_t*>(&symb));
    task_data_par->inputs_count.emplace_back(1);
    task_data_par->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_out.data()));
    task_data_par->outputs_count.emplace_back(global_out.size());
  }
  auto test_mpi_task_parallel =
      std::make_shared<budazhapova_e_count_freq_chart_mpi::TestMPITaskParallel>(task_data_par);
  ASSERT_EQ(test_mpi_task_parallel->Validation(), true);
  test_mpi_task_parallel->PreProcessing();
  test_mpi_task_parallel->Run();
  test_mpi_task_parallel->PostProcessing();
  // Create Perf attributes
  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 10;
  const boost::mpi::timer current_timer;
  perf_attr->current_timer = [&] { return current_timer.elapsed(); };
  // Create and init perf results
  auto perf_results = std::make_shared<ppc::core::PerfResults>();
  // Create Perf analyzer
  auto perf_analyzer = std::make_shared<ppc::core::Perf>(test_mpi_task_parallel);
  perf_analyzer->TaskRun(perf_attr, perf_results);
  if (world.rank() == 0) {
    ppc::core::Perf::PrintPerfStatistic(perf_results);
  }
}