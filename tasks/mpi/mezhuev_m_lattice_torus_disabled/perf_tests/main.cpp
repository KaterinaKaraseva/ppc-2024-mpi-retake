#include <gtest/gtest.h>

#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "boost/mpi/communicator.hpp"
#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"
#include "mpi/mezhuev_m_lattice_torus/include/mpi.hpp"

TEST(mezhuev_m_lattice_torus_mpi, test_pipeline_run) {
  constexpr int kCount = 5000;

  std::vector<uint8_t> in(kCount * kCount, 0);
  std::vector<uint8_t> out(kCount * kCount, 0);

  for (size_t i = 0; i < kCount; i++) {
    in[(i * kCount) + i] = 1;
  }

  auto task_data_mpi = std::make_shared<ppc::core::TaskData>();
  task_data_mpi->inputs.emplace_back(in.data());
  task_data_mpi->inputs_count.emplace_back(in.size());
  task_data_mpi->outputs.emplace_back(out.data());
  task_data_mpi->outputs_count.emplace_back(out.size());

  auto test_task_mpi = std::make_shared<mezhuev_m_lattice_torus_mpi::GridTorusTopologyParallel>(task_data_mpi);

  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 30;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perf_results = std::make_shared<ppc::core::PerfResults>();

  auto perf_analyzer = std::make_shared<ppc::core::Perf>(test_task_mpi);
  perf_analyzer->PipelineRun(perf_attr, perf_results);

  boost::mpi::communicator world;
  if (world.rank() == 0) {
    ppc::core::Perf::PrintPerfStatistic(perf_results);
  }

  ASSERT_EQ(in, out);
}

TEST(mezhuev_m_lattice_torus_mpi, test_task_run) {
  constexpr int kCount = 6000;

  std::vector<uint8_t> in(kCount * kCount, 0);
  std::vector<uint8_t> out(kCount * kCount, 0);

  for (size_t i = 0; i < kCount; i++) {
    in[(i * kCount) + i] = 1;
  }

  auto task_data_mpi = std::make_shared<ppc::core::TaskData>();
  task_data_mpi->inputs.emplace_back(in.data());
  task_data_mpi->inputs_count.emplace_back(in.size());
  task_data_mpi->outputs.emplace_back(out.data());
  task_data_mpi->outputs_count.emplace_back(out.size());

  auto test_task_mpi = std::make_shared<mezhuev_m_lattice_torus_mpi::GridTorusTopologyParallel>(task_data_mpi);

  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 40;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perf_results = std::make_shared<ppc::core::PerfResults>();

  auto perf_analyzer = std::make_shared<ppc::core::Perf>(test_task_mpi);
  perf_analyzer->TaskRun(perf_attr, perf_results);

  boost::mpi::communicator world;
  if (world.rank() == 0) {
    ppc::core::Perf::PrintPerfStatistic(perf_results);
  }

  ASSERT_EQ(in, out);
}