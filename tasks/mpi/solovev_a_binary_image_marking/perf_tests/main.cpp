#include <gtest/gtest.h>

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"
#include "mpi/solovev_a_binary_image_marking/include/ops_mpi.hpp"

TEST(solovev_a_binary_image_marking_mpi, pipeline_run) {
  boost::mpi::communicator world;
  const int m = 2500;
  const int n = 2500;

  std::vector<int> data(m * n, 1);
  std::vector<int> labled_image(m * n);

  std::shared_ptr<ppc::core::TaskData> task_data_par = std::make_shared<ppc::core::TaskData>();

  task_data_par->inputs.emplace_back(reinterpret_cast<uint8_t*>(const_cast<int*>(&m)));
  task_data_par->inputs_count.emplace_back(1);

  task_data_par->inputs.emplace_back(reinterpret_cast<uint8_t*>(const_cast<int*>(&n)));
  task_data_par->inputs_count.emplace_back(1);

  task_data_par->inputs.emplace_back(reinterpret_cast<uint8_t*>(const_cast<int*>(data.data())));
  task_data_par->inputs_count.emplace_back(data.size());

  task_data_par->outputs.emplace_back(reinterpret_cast<uint8_t*>(labled_image.data()));
  task_data_par->outputs_count.emplace_back(labled_image.size());

  auto task_parallel = std::make_shared<solovev_a_binary_image_marking::TestMPITaskParallel>(task_data_par);

  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perf_results = std::make_shared<ppc::core::PerfResults>();

  auto perf_analyzer = std::make_shared<ppc::core::Perf>(task_parallel);
  perf_analyzer->PipelineRun(perf_attr, perf_results);
  ppc::core::Perf::PrintPerfStatistic(perf_results);

  if (world.rank() == 0) {
    for (size_t i = 0; i < labled_image.size(); ++i) {
      ASSERT_EQ(data[i], labled_image[i]);
    }
  }
}

TEST(solovev_a_binary_image_marking_mpi, task_run) {
  boost::mpi::communicator world;
  const int m = 2500;
  const int n = 2500;

  std::vector<int> data(m * n, 1);
  std::vector<int> labled_image(m * n);

  std::shared_ptr<ppc::core::TaskData> task_data_par = std::make_shared<ppc::core::TaskData>();

  task_data_par->inputs.emplace_back(reinterpret_cast<uint8_t*>(const_cast<int*>(&m)));
  task_data_par->inputs_count.emplace_back(1);

  task_data_par->inputs.emplace_back(reinterpret_cast<uint8_t*>(const_cast<int*>(&n)));
  task_data_par->inputs_count.emplace_back(1);

  task_data_par->inputs.emplace_back(reinterpret_cast<uint8_t*>(const_cast<int*>(data.data())));
  task_data_par->inputs_count.emplace_back(data.size());

  task_data_par->outputs.emplace_back(reinterpret_cast<uint8_t*>(labled_image.data()));
  task_data_par->outputs_count.emplace_back(labled_image.size());

  auto task_parallel = std::make_shared<solovev_a_binary_image_marking::TestMPITaskParallel>(task_data_par);

  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perf_results = std::make_shared<ppc::core::PerfResults>();

  auto perf_analyzer = std::make_shared<ppc::core::Perf>(task_parallel);
  perf_analyzer->TaskRun(perf_attr, perf_results);
  ppc::core::Perf::PrintPerfStatistic(perf_results);

  if (world.rank() == 0) {
    for (size_t i = 0; i < labled_image.size(); ++i) {
      ASSERT_EQ(data[i], labled_image[i]);
    }
  }
}