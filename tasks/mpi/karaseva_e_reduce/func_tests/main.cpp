#define OMPI_SKIP_MPICXX

#include <gtest/gtest.h>
#include <mpi.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <numeric>
#include <random>
#include <vector>

#include "core/task/include/task.hpp"
#include "mpi/karaseva_e_reduce/include/ops_mpi.hpp"

namespace {

// Utility function to generate random data
template <typename T>
std::vector<T> GetRandom(int size) {
  std::random_device dev;
  std::mt19937 gen(dev());
  std::uniform_int_distribution<> distrib(-100, 100);
  std::vector<T> vec(size);
  for (int i = 0; i < size; i++) {
    vec[i] = static_cast<T>(distrib(gen));
  }
  return vec;
}

}  // namespace

TEST(karaseva_e_reduce_mpi, test_reduce_int) {
  MPI_Comm comm = MPI_COMM_WORLD;
  int rank = 0;
  MPI_Comm_rank(comm, &rank);

  constexpr size_t kCount = 50;
  std::vector<int> in;
  std::vector<int> out(1, 0);

  if (rank == 0) {
    in = GetRandom<int>(kCount);
  } else {
    in.resize(kCount);
  }

  MPI_Bcast(in.data(), kCount, MPI_INT, 0, comm);

  auto task_data_mpi = std::make_shared<ppc::core::TaskData>();
  task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
  task_data_mpi->inputs_count.emplace_back(in.size());
  task_data_mpi->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  task_data_mpi->outputs_count.emplace_back(out.size());

  karaseva_e_reduce_mpi::TestTaskMPI<int> test_task_mpi(task_data_mpi);

  ASSERT_TRUE(test_task_mpi.ValidationImpl());
  test_task_mpi.PreProcessingImpl();
  test_task_mpi.RunImpl();
  test_task_mpi.PostProcessingImpl();

  if (rank == 0) {
    int expected_result = std::accumulate(in.begin(), in.end(), 0);
    EXPECT_EQ(out[0], expected_result);
  }
}

TEST(karaseva_e_reduce_mpi, test_reduce_double) {
  MPI_Comm comm = MPI_COMM_WORLD;
  int rank = 0;
  MPI_Comm_rank(comm, &rank);

  constexpr size_t kCount = 50;
  std::vector<double> in;
  std::vector<double> out(1, 0.0);

  if (rank == 0) {
    in = GetRandom<double>(kCount);
  } else {
    in.resize(kCount);
  }

  MPI_Bcast(in.data(), kCount, MPI_DOUBLE, 0, comm);

  auto task_data_mpi = std::make_shared<ppc::core::TaskData>();
  task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
  task_data_mpi->inputs_count.emplace_back(in.size());
  task_data_mpi->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  task_data_mpi->outputs_count.emplace_back(out.size());

  karaseva_e_reduce_mpi::TestTaskMPI<double> test_task_mpi(task_data_mpi);

  ASSERT_TRUE(test_task_mpi.ValidationImpl());
  test_task_mpi.PreProcessingImpl();
  test_task_mpi.RunImpl();
  test_task_mpi.PostProcessingImpl();

  if (rank == 0) {
    double expected_result = std::accumulate(in.begin(), in.end(), 0.0);
    EXPECT_DOUBLE_EQ(out[0], expected_result);
  }
}

TEST(karaseva_e_reduce_mpi, test_reduce_float) {
  MPI_Comm comm = MPI_COMM_WORLD;
  int rank = 0;
  MPI_Comm_rank(comm, &rank);

  constexpr size_t kCount = 50;
  std::vector<float> in;
  std::vector<float> out(1, 0.0F);

  if (rank == 0) {
    in = GetRandom<float>(kCount);
  } else {
    in.resize(kCount);
  }

  MPI_Bcast(in.data(), kCount, MPI_FLOAT, 0, comm);

  auto task_data_mpi = std::make_shared<ppc::core::TaskData>();
  task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
  task_data_mpi->inputs_count.emplace_back(in.size());
  task_data_mpi->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  task_data_mpi->outputs_count.emplace_back(out.size());

  karaseva_e_reduce_mpi::TestTaskMPI<float> test_task_mpi(task_data_mpi);

  ASSERT_TRUE(test_task_mpi.ValidationImpl());
  test_task_mpi.PreProcessingImpl();
  test_task_mpi.RunImpl();
  test_task_mpi.PostProcessingImpl();

  if (rank == 0) {
    float expected_result = std::accumulate(in.begin(), in.end(), 0.0F);
    EXPECT_FLOAT_EQ(out[0], expected_result);
  }
}