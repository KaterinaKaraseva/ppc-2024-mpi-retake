#include <gtest/gtest.h>

#include <algorithm>
#include <boost/mpi/communicator.hpp>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <numeric>
#include <vector>

#include "core/task/include/task.hpp"
#include "mpi/mezhuev_m_most_different_neighbor_elements_mpi/include/mpi.hpp"

TEST(MostDifferentNeighborElementsMPI, HandlesLargeInput) {
  boost::mpi::communicator world;
  std::vector<int> input(10000);
  std::ranges::generate(input, []() { return rand() % 10000; });
  std::vector<int> output1(2, 0);
  std::vector<int> output2(2, 0);

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(input.data()));
  task_data->inputs_count.emplace_back(input.size());

  task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(output1.data()));
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(output2.data()));
  task_data->outputs_count.emplace_back(output1.size());
  task_data->outputs_count.emplace_back(output2.size());

  auto test_task =
      std::make_shared<mezhuev_m_most_different_neighbor_elements_mpi::MostDifferentNeighborElements>(world, task_data);

  ASSERT_TRUE(test_task->ValidationImpl());
  ASSERT_TRUE(test_task->PreProcessingImpl());
  ASSERT_TRUE(test_task->RunImpl());
  ASSERT_TRUE(test_task->PostProcessingImpl());
}

TEST(mezhuev_m_most_different_neighbor_elements_mpi, HandlesSingleElement) {
  boost::mpi::communicator world;
  std::vector<int> input = {5};
  std::vector<int> output1(1, 0);
  std::vector<int> output2(1, 0);

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(input.data()));
  task_data->inputs_count.emplace_back(input.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(output1.data()));
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(output2.data()));
  task_data->outputs_count.emplace_back(1);
  task_data->outputs_count.emplace_back(1);

  auto test_task =
      std::make_shared<mezhuev_m_most_different_neighbor_elements_mpi::MostDifferentNeighborElements>(world, task_data);
  ASSERT_FALSE(test_task->ValidationImpl());
}

TEST(mezhuev_m_most_different_neighbor_elements_mpi, HandlesEmptyInput) {
  boost::mpi::communicator world;
  std::vector<int> input;
  std::vector<int> output1(1, 0);
  std::vector<int> output2(1, 0);

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(input.data()));
  task_data->inputs_count.emplace_back(input.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(output1.data()));
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(output2.data()));
  task_data->outputs_count.emplace_back(1);
  task_data->outputs_count.emplace_back(1);

  auto test_task =
      std::make_shared<mezhuev_m_most_different_neighbor_elements_mpi::MostDifferentNeighborElements>(world, task_data);
  ASSERT_FALSE(test_task->ValidationImpl());
}

TEST(mezhuev_m_most_different_neighbor_elements_mpi, HandlesNegativeNumbers) {
  boost::mpi::communicator world;
  std::vector<int> input = {-10, -50, -30, -80, -5, -100};
  std::vector<int> output1(2, 0);
  std::vector<int> output2(2, 0);

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(input.data()));
  task_data->inputs_count.emplace_back(input.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(output1.data()));
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(output2.data()));
  task_data->outputs_count.emplace_back(2);
  task_data->outputs_count.emplace_back(2);

  auto test_task =
      std::make_shared<mezhuev_m_most_different_neighbor_elements_mpi::MostDifferentNeighborElements>(world, task_data);

  ASSERT_TRUE(test_task->ValidationImpl());
  ASSERT_TRUE(test_task->PreProcessingImpl());
  ASSERT_TRUE(test_task->RunImpl());
  ASSERT_TRUE(test_task->PostProcessingImpl());

  if (world.rank() == 0) {
    ASSERT_GT(std::abs(output1[0] - output2[0]), 0);
  }
}

TEST(mezhuev_m_most_different_neighbor_elements_mpi, HandlesSequentialNumbers) {
  boost::mpi::communicator world;
  std::vector<int> input(1000);
  std::iota(input.begin(), input.end(), 1);  // 1, 2, 3, ..., 1000
  std::vector<int> output1(2, 0);
  std::vector<int> output2(2, 0);

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(input.data()));
  task_data->inputs_count.emplace_back(input.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(output1.data()));
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(output2.data()));
  task_data->outputs_count.emplace_back(2);
  task_data->outputs_count.emplace_back(2);

  auto test_task =
      std::make_shared<mezhuev_m_most_different_neighbor_elements_mpi::MostDifferentNeighborElements>(world, task_data);

  ASSERT_TRUE(test_task->ValidationImpl());
  ASSERT_TRUE(test_task->PreProcessingImpl());
  ASSERT_TRUE(test_task->RunImpl());
  ASSERT_TRUE(test_task->PostProcessingImpl());

  if (world.rank() == 0) {
    ASSERT_EQ(std::abs(output1[0] - output2[0]), 1);
  }
}

TEST(mezhuev_m_most_different_neighbor_elements_mpi, HandlesLargeRandomInput) {
  boost::mpi::communicator world;
  std::vector<int> input(100000);
  std::ranges::generate(input.begin(), input.end(), []() { return rand() % 100000; });
  std::vector<int> output1(2, 0);
  std::vector<int> output2(2, 0);

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(input.data()));
  task_data->inputs_count.emplace_back(input.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(output1.data()));
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(output2.data()));
  task_data->outputs_count.emplace_back(2);
  task_data->outputs_count.emplace_back(2);

  auto test_task =
      std::make_shared<mezhuev_m_most_different_neighbor_elements_mpi::MostDifferentNeighborElements>(world, task_data);

  ASSERT_TRUE(test_task->ValidationImpl());
  ASSERT_TRUE(test_task->PreProcessingImpl());
  ASSERT_TRUE(test_task->RunImpl());
  ASSERT_TRUE(test_task->PostProcessingImpl());

  if (world.rank() == 0) {
    ASSERT_GT(std::abs(output1[0] - output2[0]), 0);
  }
}
