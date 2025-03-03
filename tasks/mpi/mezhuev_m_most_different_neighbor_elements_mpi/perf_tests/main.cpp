#include <gtest/gtest.h>

#include <boost/mpi/timer.hpp>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <vector>

#include "boost/mpi/communicator.hpp"
#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"
#include "mpi/mezhuev_m_most_different_neighbor_elements_mpi/include/mpi.hpp"

namespace {
void GenerateRandomData(boost::mpi::communicator& world, std::vector<int>& in) {
  if (world.rank() == 0) {
    for (size_t i = 0; i < in.size(); i++) {
      in[i] = rand() % 1000000;
    }
  }
}
}  // namespace

TEST(mezhuev_m_most_different_neighbor_elements_mpi, test_pipeline_run) {
  boost::mpi::communicator world;
  std::vector<int> in(10'000'000, 0);
  std::vector<int> out1(2, 0);
  std::vector<int> out2(2, 0);

  if (in.size() < 2) {
    FAIL();
  }

  if (out1.size() < 2 || out2.size() < 2) {
    FAIL();
  }

  GenerateRandomData(world, in);

  auto task_data_mpi = std::make_shared<ppc::core::TaskData>();
  task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
  task_data_mpi->inputs_count.emplace_back(in.size());

  task_data_mpi->outputs.emplace_back(reinterpret_cast<uint8_t*>(out1.data()));
  task_data_mpi->outputs.emplace_back(reinterpret_cast<uint8_t*>(out2.data()));
  task_data_mpi->outputs_count.emplace_back(out1.size());
  task_data_mpi->outputs_count.emplace_back(out2.size());

  if (task_data_mpi->inputs.empty() || task_data_mpi->outputs.empty() || task_data_mpi->inputs[0] == nullptr ||
      task_data_mpi->outputs[0] == nullptr || task_data_mpi->outputs[1] == nullptr) {
    FAIL();
  }

  auto test_task_mpi = std::make_shared<mezhuev_m_most_different_neighbor_elements_mpi::MostDifferentNeighborElements>(
      world, task_data_mpi);

  ASSERT_EQ(test_task_mpi->ValidationImpl(), true);
  test_task_mpi->PreProcessingImpl();
  test_task_mpi->RunImpl();
  test_task_mpi->PostProcessingImpl();

  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 20;
  const boost::mpi::timer current_timer;
  perf_attr->current_timer = [&] { return current_timer.elapsed(); };

  auto perf_results = std::make_shared<ppc::core::PerfResults>();
  auto perf_analyzer = std::make_shared<ppc::core::Perf>(test_task_mpi);
  perf_analyzer->PipelineRun(perf_attr, perf_results);

  if (world.rank() == 0) {
    ppc::core::Perf::PrintPerfStatistic(perf_results);
    ASSERT_NE(out1[0], out2[0]);
    ASSERT_GT(std::abs(out1[0] - out2[0]), 0);
  }
}

TEST(mezhuev_m_most_different_neighbor_elements_mpi, test_task_run) {
  boost::mpi::communicator world;
  std::vector<int> in(10000000);
  std::vector<int> out1(2, 0);
  std::vector<int> out2(2, 0);

  GenerateRandomData(world, in);

  auto task_data_mpi = std::make_shared<ppc::core::TaskData>();
  task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
  task_data_mpi->inputs_count.emplace_back(in.size());

  task_data_mpi->outputs.emplace_back(reinterpret_cast<uint8_t*>(out1.data()));
  task_data_mpi->outputs.emplace_back(reinterpret_cast<uint8_t*>(out2.data()));
  task_data_mpi->outputs_count.emplace_back(out1.size());
  task_data_mpi->outputs_count.emplace_back(out2.size());

  auto test_task_mpi = std::make_shared<mezhuev_m_most_different_neighbor_elements_mpi::MostDifferentNeighborElements>(
      world, task_data_mpi);

  ASSERT_EQ(test_task_mpi->ValidationImpl(), true);
  test_task_mpi->PreProcessingImpl();
  test_task_mpi->RunImpl();
  test_task_mpi->PostProcessingImpl();

  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 20;
  const boost::mpi::timer current_timer;
  perf_attr->current_timer = [&] { return current_timer.elapsed(); };

  auto perf_results = std::make_shared<ppc::core::PerfResults>();
  auto perf_analyzer = std::make_shared<ppc::core::Perf>(test_task_mpi);
  perf_analyzer->TaskRun(perf_attr, perf_results);
  if (world.rank() == 0) {
    ppc::core::Perf::PrintPerfStatistic(perf_results);
    ASSERT_NE(out1[0], out2[0]);
    ASSERT_GT(std::abs(out1[0] - out2[0]), 0);
  }
}
