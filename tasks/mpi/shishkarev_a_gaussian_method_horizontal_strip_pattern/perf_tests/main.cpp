#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/timer.hpp>
#include <cmath>
#include <cstdint>
#include <memory>
#include <random>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"
#include "mpi/shishkarev_a_gaussian_method_horizontal_strip_pattern/include/ops_mpi.hpp"

namespace shishkarev_a_gaussian_method_horizontal_strip_pattern_mpi {

std::vector<double> GetRandomMatrix(int sz) {
  std::random_device dev;
  std::mt19937 gen(dev());
  std::uniform_real_distribution<double> dis(-1000, 1000);
  std::vector<double> matrix(sz);
  for (int i = 0; i < sz; ++i) {
    matrix[i] = dis(gen);
  }
  return matrix;
}

double AxB(int n, int m, std::vector<double> a, std::vector<double> res) {
  std::vector<double> tmp(m, 0);

  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n - 1; ++j) {
      tmp[i] += a[(i * n) + j] * res[j];
    }
    tmp[i] -= a[(i * n) + m];
  }

  double tmp_norm = 0;
  for (int i = 0; i < m; i++) {
    tmp_norm += tmp[i] * tmp[i];
  }
  return sqrt(tmp_norm);
}

}  // namespace shishkarev_a_gaussian_method_horizontal_strip_pattern_mpi

TEST(shishkarev_a_gaussian_method_horizontal_strip_pattern_mpi, test_pipeline_run) {
  boost::mpi::communicator world;

  const int cols = 101;
  const int rows = 100;
  std::vector<double> global_matrix(cols * rows);
  std::vector<double> global_res(cols - 1, 0);

  std::shared_ptr<ppc::core::TaskData> task_data_par = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    global_matrix = shishkarev_a_gaussian_method_horizontal_strip_pattern_mpi::GetRandomMatrix(cols * rows);
    task_data_par->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matrix.data()));
    task_data_par->inputs_count.emplace_back(global_matrix.size());
    task_data_par->inputs_count.emplace_back(cols);
    task_data_par->inputs_count.emplace_back(rows);
    task_data_par->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_res.data()));
    task_data_par->outputs_count.emplace_back(global_res.size());
  }

  auto mpi_gauss_horizontal_parallel =
      std::make_shared<shishkarev_a_gaussian_method_horizontal_strip_pattern_mpi::MPIGaussHorizontalParallel>(
          task_data_par);
  ASSERT_EQ(mpi_gauss_horizontal_parallel->ValidationImpl(), true);
  mpi_gauss_horizontal_parallel->PreProcessingImpl();
  mpi_gauss_horizontal_parallel->RunImpl();
  mpi_gauss_horizontal_parallel->PostProcessingImpl();

  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 10;
  const boost::mpi::timer current_timer;
  perf_attr->current_timer = [&] { return current_timer.elapsed(); };

  auto perf_results = std::make_shared<ppc::core::PerfResults>();

  auto perf_analyzer = std::make_shared<ppc::core::Perf>(mpi_gauss_horizontal_parallel);
  perf_analyzer->PipelineRun(perf_attr, perf_results);
  if (world.rank() == 0) {
    ppc::core::Perf::PrintPerfStatistic(perf_results);
    ASSERT_NEAR(shishkarev_a_gaussian_method_horizontal_strip_pattern_mpi::AxB(cols, rows, global_matrix, global_res),
                0, 1e-6);
  }
}

TEST(shishkarev_a_gaussian_method_horizontal_strip_pattern_mpi, test_task_run) {
  boost::mpi::communicator world;

  const int cols = 101;
  const int rows = 100;
  std::vector<double> global_matrix(cols * rows);
  std::vector<double> global_res(cols - 1, 0);

  std::shared_ptr<ppc::core::TaskData> task_data_par = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    global_matrix = shishkarev_a_gaussian_method_horizontal_strip_pattern_mpi::GetRandomMatrix(cols * rows);
    task_data_par->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matrix.data()));
    task_data_par->inputs_count.emplace_back(global_matrix.size());
    task_data_par->inputs_count.emplace_back(cols);
    task_data_par->inputs_count.emplace_back(rows);
    task_data_par->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_res.data()));
    task_data_par->outputs_count.emplace_back(global_res.size());
  }

  auto mpi_gauss_horizontal_parallel =
      std::make_shared<shishkarev_a_gaussian_method_horizontal_strip_pattern_mpi::MPIGaussHorizontalParallel>(
          task_data_par);
  ASSERT_EQ(mpi_gauss_horizontal_parallel->ValidationImpl(), true);
  mpi_gauss_horizontal_parallel->PreProcessingImpl();
  mpi_gauss_horizontal_parallel->RunImpl();
  mpi_gauss_horizontal_parallel->PostProcessingImpl();

  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 10;
  const boost::mpi::timer current_timer;
  perf_attr->current_timer = [&] { return current_timer.elapsed(); };

  auto perf_results = std::make_shared<ppc::core::PerfResults>();

  auto perf_analyzer = std::make_shared<ppc::core::Perf>(mpi_gauss_horizontal_parallel);
  perf_analyzer->TaskRun(perf_attr, perf_results);
  if (world.rank() == 0) {
    ppc::core::Perf::PrintPerfStatistic(perf_results);
    ASSERT_NEAR(shishkarev_a_gaussian_method_horizontal_strip_pattern_mpi::AxB(cols, rows, global_matrix, global_res),
                0, 1e-6);
  }
}