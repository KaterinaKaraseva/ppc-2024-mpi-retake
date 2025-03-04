// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/timer.hpp>
#include <memory>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"
#include "mpi/stroganov_m_dining_philosophers/include/ops_mpi.hpp"

TEST(stroganov_m_dining_philosophers_mpi, test_pipeline_run) {
  boost::mpi::communicator world;

  std::shared_ptr<ppc::core::TaskData> task_data = std::make_shared<ppc::core::TaskData>();
  int count_philosophers = world.size();
  if (world.rank() == 0) {
    task_data->inputs_count.push_back(count_philosophers);
  }

  auto task_data_par = std::make_shared<stroganov_m_dining_philosophers_mpi::DiningPhilosophersMPI>(task_data);

  if (world.size() < 2) {
    GTEST_SKIP() << "Skipping test due to failed validation";
  }

  task_data_par->ValidationImpl();
  task_data_par->PreProcessingImpl();
  task_data_par->RunImpl();
  task_data_par->PostProcessingImpl();

  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 10;
  const boost::mpi::timer current_timer;
  perf_attr->current_timer = [&] { return current_timer.elapsed(); };

  auto perf_results = std::make_shared<ppc::core::PerfResults>();

  auto perf_analyzer = std::make_shared<ppc::core::Perf>(task_data_par);
  perf_analyzer->PipelineRun(perf_attr, perf_results);

  if (world.rank() == 0) {
    ppc::core::Perf::PrintPerfStatistic(perf_results);
    int expected_philosophers_finished = world.size();
    ASSERT_EQ(expected_philosophers_finished, world.size());
  }
}

TEST(stroganov_m_dining_philosophers_mpi, test_task_run) {
  boost::mpi::communicator world;

  std::shared_ptr<ppc::core::TaskData> task_data = std::make_shared<ppc::core::TaskData>();
  int count_philosophers = world.size();
  if (world.rank() == 0) {
    task_data->inputs_count.push_back(count_philosophers);
  }

  auto task_data_par = std::make_shared<stroganov_m_dining_philosophers_mpi::DiningPhilosophersMPI>(task_data);

  if (world.size() < 2) {
    GTEST_SKIP() << "Skipping test due to failed validation";
  }

  task_data_par->ValidationImpl();
  task_data_par->PreProcessingImpl();
  task_data_par->RunImpl();
  task_data_par->PostProcessingImpl();

  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 10;
  const boost::mpi::timer current_timer;
  perf_attr->current_timer = [&] { return current_timer.elapsed(); };

  auto perf_results = std::make_shared<ppc::core::PerfResults>();

  auto perf_analyzer = std::make_shared<ppc::core::Perf>(task_data_par);
  perf_analyzer->TaskRun(perf_attr, perf_results);

  if (world.rank() == 0) {
    ppc::core::Perf::PrintPerfStatistic(perf_results);
    int expected_philosophers_finished = world.size();
    ASSERT_EQ(expected_philosophers_finished, world.size());
  }
}