#include <gtest/gtest.h>

#include <algorithm>
#include <boost/mpi/communicator.hpp>
#include <cstdint>
#include <memory>
#include <vector>

#include "core/task/include/task.hpp"
#include "mpi/kalinin_d_odd_even_shellsort/include/header_mpi_odd_even_shell.hpp"

TEST(kalinin_d_odd_even_shell_mpi, Test_odd_even_sort_0) {
  const int n = 0;

  boost::mpi::communicator world;

  // Create data
  std::vector<int> arr;
  std::vector<int> out;
  auto task_data_mpi = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    arr.resize(n);
    out.resize(n);
    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(arr.data()));
    task_data_mpi->inputs_count.emplace_back(arr.size());
    task_data_mpi->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    task_data_mpi->outputs_count.emplace_back(out.size());
  }
  // Create Task}
  kalinin_d_odd_even_shell_mpi::OddEvenShellMpi task_mpi(task_data_mpi);
  if (world.rank() == 0) {
    ASSERT_EQ(task_mpi.Validation(), false);
  }
}

TEST(kalinin_d_odd_even_shell_mpi, Test_odd_even_sort_1000) {
  const int n = 10;
  boost::mpi::communicator world;
  // Create data
  std::vector<int> arr;
  std::vector<int> out;

  std::shared_ptr<ppc::core::TaskData> task_data_mpi = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    arr.resize(n);
    out.resize(n);
    kalinin_d_odd_even_shell_mpi::GimmeRandVec(arr);
    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(arr.data()));
    task_data_mpi->inputs_count.emplace_back(arr.size());
    task_data_mpi->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    task_data_mpi->outputs_count.emplace_back(out.size());
  }
  // Create Task
  kalinin_d_odd_even_shell_mpi::OddEvenShellMpi task_mpi(task_data_mpi);
  ASSERT_EQ(task_mpi.ValidationImpl(), true);
  task_mpi.PreProcessingImpl();

  task_mpi.RunImpl();
  task_mpi.PostProcessingImpl();
  if (world.rank() == 0) {
    std::ranges::sort(arr.begin(), arr.end());
    ASSERT_EQ(arr, out);
  }
}

TEST(kalinin_d_odd_even_shell_mpi, Test_odd_even_sort_999) {
  const int n = 999;
  boost::mpi::communicator world;
  // Create data
  std::vector<int> arr;
  std::vector<int> out;
  auto task_data_mpi = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    arr.resize(n);
    out.resize(n);
    kalinin_d_odd_even_shell_mpi::GimmeRandVec(arr);
    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(arr.data()));
    task_data_mpi->inputs_count.emplace_back(arr.size());
    task_data_mpi->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    task_data_mpi->outputs_count.emplace_back(out.size());
  }
  // Create Task
  kalinin_d_odd_even_shell_mpi::OddEvenShellMpi task_mpi(task_data_mpi);
  ASSERT_EQ(task_mpi.Validation(), true);
  task_mpi.PreProcessing();
  task_mpi.Run();

  task_mpi.PostProcessing();
  if (world.rank() == 0) {
    std::ranges::sort(arr.begin(), arr.end());
    ASSERT_EQ(arr, out);
  }
}

TEST(kalinin_d_odd_even_shell_mpi, Test_odd_even_sort_9999) {
  const int n = 9999;
  boost::mpi::communicator world;
  // Create data
  std::vector<int> arr;
  std::vector<int> out;
  auto task_data_mpi = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    arr.resize(n);
    out.resize(n);
    kalinin_d_odd_even_shell_mpi::GimmeRandVec(arr);
    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(arr.data()));
    task_data_mpi->inputs_count.emplace_back(arr.size());
    task_data_mpi->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    task_data_mpi->outputs_count.emplace_back(out.size());
  }
  // Create Task
  kalinin_d_odd_even_shell_mpi::OddEvenShellMpi task_mpi(task_data_mpi);
  ASSERT_EQ(task_mpi.Validation(), true);
  task_mpi.PreProcessing();
  task_mpi.Run();

  task_mpi.PostProcessing();
  if (world.rank() == 0) {
    std::ranges::sort(arr.begin(), arr.end());
    ASSERT_EQ(arr, out);
  }
}

TEST(kalinin_d_odd_even_shell_mpi, Test_odd_even_sort_1021) {
  const int n = 29;
  boost::mpi::communicator world;
  // Create data
  std::vector<int> arr;
  std::vector<int> out;
  auto task_data_mpi = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    arr.resize(n);
    out.resize(n);
    kalinin_d_odd_even_shell_mpi::GimmeRandVec(arr);
    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(arr.data()));
    task_data_mpi->inputs_count.emplace_back(arr.size());
    task_data_mpi->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    task_data_mpi->outputs_count.emplace_back(out.size());
  }
  // Create Task
  kalinin_d_odd_even_shell_mpi::OddEvenShellMpi task_mpi(task_data_mpi);
  ASSERT_EQ(task_mpi.Validation(), true);
  task_mpi.PreProcessing();
  task_mpi.Run();

  task_mpi.PostProcessing();
  if (world.rank() == 0) {
    std::ranges::sort(arr.begin(), arr.end());
    ASSERT_EQ(arr, out);
  }
}