#include <gtest/gtest.h>

#include <boost/mpi/timer.hpp>
#include <cstdint>
#include <memory>
#include <vector>

#include "boost/mpi/communicator.hpp"
#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"
#include "mpi/kavtorev_d_most_different_neighbor_elements/include/ops_mpi.hpp"

TEST(kavtorev_d_most_different_neighbor_elements_mpi, test_pipeline_run) {
  boost::mpi::communicator world;
  std::vector<int> in(10000000, 0);
  std::vector<int32_t> out(1, 0);

  auto task_data_mpi = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
    task_data_mpi->inputs_count.emplace_back(in.size());
    task_data_mpi->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
    task_data_mpi->outputs_count.emplace_back(out.size());
  }

  auto test_task_mpi =
      std::make_shared<kavtorev_d_most_different_neighbor_elements_mpi::MostDifferentNeighborElementsMpi>(
          task_data_mpi);
  ASSERT_EQ(test_task_mpi->ValidationImpl(), true);
  test_task_mpi->PreProcessingImpl();
  test_task_mpi->RunImpl();
  test_task_mpi->PostProcessingImpl();

  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 10;
  const boost::mpi::timer current_timer;
  perf_attr->current_timer = [&] { return current_timer.elapsed(); };

  auto perf_results = std::make_shared<ppc::core::PerfResults>();

  auto perf_analyzer = std::make_shared<ppc::core::Perf>(test_task_mpi);
  perf_analyzer->PipelineRun(perf_attr, perf_results);
  if (world.rank() == 0) {
    ppc::core::Perf::PrintPerfStatistic(perf_results);
    ASSERT_EQ(0, out[0]);
  }
}

TEST(kavtorev_d_most_different_neighbor_elements_mpi, test_task_run) {
  boost::mpi::communicator world;
  std::vector<int> in(10000000, 0);
  std::vector<int32_t> out(1, 0);

  auto task_data_mpi = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
    task_data_mpi->inputs_count.emplace_back(in.size());
    task_data_mpi->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
    task_data_mpi->outputs_count.emplace_back(out.size());
  }

  auto test_task_mpi =
      std::make_shared<kavtorev_d_most_different_neighbor_elements_mpi::MostDifferentNeighborElementsMpi>(
          task_data_mpi);
  ASSERT_EQ(test_task_mpi->ValidationImpl(), true);
  test_task_mpi->PreProcessingImpl();
  test_task_mpi->RunImpl();
  test_task_mpi->PostProcessingImpl();

  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 10;
  const boost::mpi::timer current_timer;
  perf_attr->current_timer = [&] { return current_timer.elapsed(); };

  auto perf_results = std::make_shared<ppc::core::PerfResults>();

  auto perf_analyzer = std::make_shared<ppc::core::Perf>(test_task_mpi);
  perf_analyzer->TaskRun(perf_attr, perf_results);
  if (world.rank() == 0) {
    ppc::core::Perf::PrintPerfStatistic(perf_results);
    ASSERT_EQ(0, out[0]);
  }
}
