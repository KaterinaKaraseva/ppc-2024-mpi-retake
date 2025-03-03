#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <numeric>
#include <random>
#include <vector>

#include "core/task/include/task.hpp"
#include "mpi/karaseva_e_reduce/include/ops_mpi.hpp"

namespace {

// Utility function to generate random dat
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
  constexpr size_t kCount = 50;

  std::vector<int> in = GetRandom<int>(kCount);
  std::vector<int> out(1, 0);

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

  int expected_result = std::accumulate(in.begin(), in.end(), 0);

  EXPECT_EQ(out[0], expected_result);
}

TEST(karaseva_e_reduce_mpi, test_reduce_double) {
  constexpr size_t kCount = 50;

  std::vector<double> in = GetRandom<double>(kCount);
  std::vector<double> out(1, 0.0);

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

  double expected_result = std::accumulate(in.begin(), in.end(), 0.0);

  EXPECT_DOUBLE_EQ(out[0], expected_result);
}

TEST(karaseva_e_reduce_mpi, test_reduce_float) {
  constexpr size_t kCount = 50;

  std::vector<float> in = GetRandom<float>(kCount);
  std::vector<float> out(1, 0.0F);

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

  float expected_result = std::accumulate(in.begin(), in.end(), 0.0F);

  EXPECT_FLOAT_EQ(out[0], expected_result);
}