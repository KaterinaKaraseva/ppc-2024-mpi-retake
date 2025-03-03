#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <cstdint>
#include <memory>
#include <random>
#include <vector>

#include "core/task/include/task.hpp"
#include "mpi/shishkarev_a_gaussian_method_horizontal_strip_pattern/include/ops_mpi.hpp"

namespace shishkarev_a_gaussian_method_horizontal_strip_pattern_mpi {

std::vector<double> GetRandomMatrix(int sz) {
  std::random_device dev;
  std::mt19937 gen(dev());
  std::uniform_real_distribution<double> dis(-1000, 1000);
  std::vector<double> mat(sz);
  for (int i = 0; i < sz; ++i) {
    mat[i] = dis(gen);
  }
  return mat;
}

bool IsSingular(const std::vector<double>& matrix, Matrix mat) { return Determinant(mat, matrix) == 0; }

}  // namespace shishkarev_a_gaussian_method_horizontal_strip_pattern_mpi

TEST(shishkarev_a_gaussian_method_horizontal_strip_pattern_mpi, test_empty_matrix) {
  boost::mpi::communicator world;

  const int cols = 0;
  const int rows = 0;

  std::vector<double> global_matrix;
  std::vector<double> global_res;
  std::shared_ptr<ppc::core::TaskData> task_data_par = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    task_data_par->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matrix.data()));
    task_data_par->inputs_count.emplace_back(global_matrix.size());
    task_data_par->inputs_count.emplace_back(cols);
    task_data_par->inputs_count.emplace_back(rows);
    task_data_par->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_res.data()));
    task_data_par->outputs_count.emplace_back(global_res.size());
    shishkarev_a_gaussian_method_horizontal_strip_pattern_mpi::MPIGaussHorizontalParallel mpi_gauss_horizontal_parallel(
        task_data_par);
    ASSERT_FALSE(mpi_gauss_horizontal_parallel.ValidationImpl());
  }
}

TEST(shishkarev_a_gaussian_method_horizontal_strip_pattern_mpi, test_matrix_with_one_element) {
  boost::mpi::communicator world;

  const int cols = 1;
  const int rows = 1;

  std::vector<double> global_matrix;
  std::vector<double> global_res;
  std::shared_ptr<ppc::core::TaskData> task_data_par = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    global_matrix = {1};
    task_data_par->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matrix.data()));
    task_data_par->inputs_count.emplace_back(global_matrix.size());
    task_data_par->inputs_count.emplace_back(cols);
    task_data_par->inputs_count.emplace_back(rows);
    task_data_par->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_res.data()));
    task_data_par->outputs_count.emplace_back(global_res.size());
    shishkarev_a_gaussian_method_horizontal_strip_pattern_mpi::MPIGaussHorizontalParallel mpi_gauss_horizontal_parallel(
        task_data_par);
    ASSERT_FALSE(mpi_gauss_horizontal_parallel.ValidationImpl());
  }
}

TEST(shishkarev_a_gaussian_method_horizontal_strip_pattern_mpi, test_not_square_matrix) {
  boost::mpi::communicator world;

  const int cols = 5;
  const int rows = 2;

  std::vector<double> global_matrix;
  std::vector<double> global_res(cols - 1, 0);
  std::shared_ptr<ppc::core::TaskData> task_data_par = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    global_matrix = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    task_data_par->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matrix.data()));
    task_data_par->inputs_count.emplace_back(global_matrix.size());
    task_data_par->inputs_count.emplace_back(cols);
    task_data_par->inputs_count.emplace_back(rows);
    task_data_par->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_res.data()));
    task_data_par->outputs_count.emplace_back(global_res.size());
    shishkarev_a_gaussian_method_horizontal_strip_pattern_mpi::MPIGaussHorizontalParallel mpi_gauss_horizontal_parallel(
        task_data_par);
    ASSERT_FALSE(mpi_gauss_horizontal_parallel.ValidationImpl());
  }
}
