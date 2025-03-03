#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <memory>
#include <vector>

#include "core/task/include/task.hpp"
#include "seq/mezhuev_m_most_different_neighbor_elements_seq/include/seq.hpp"

TEST(mezhuev_m_most_different_neighbor_elements_seq, ValidationEmptyTaskData) {
  auto task_data = std::make_shared<ppc::core::TaskData>();
  mezhuev_m_most_different_neighbor_elements_seq::MostDifferentNeighborElements task(task_data);
  EXPECT_FALSE(task.ValidationImpl());
}

TEST(mezhuev_m_most_different_neighbor_elements_seq, ValidationMissingInputs) {
  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs_count.push_back(3);
  mezhuev_m_most_different_neighbor_elements_seq::MostDifferentNeighborElements task(task_data);
  EXPECT_FALSE(task.ValidationImpl());
}

TEST(mezhuev_m_most_different_neighbor_elements_seq, ValidationSmallInputSize) {
  auto task_data = std::make_shared<ppc::core::TaskData>();
  std::vector<int> input_data = {1};
  task_data->inputs.push_back(reinterpret_cast<uint8_t*>(input_data.data()));
  task_data->inputs_count.push_back(static_cast<uint32_t>(input_data.size()));
  task_data->outputs.push_back(reinterpret_cast<uint8_t*>(input_data.data()));
  task_data->outputs_count.push_back(2);

  mezhuev_m_most_different_neighbor_elements_seq::MostDifferentNeighborElements task(task_data);
  EXPECT_FALSE(task.ValidationImpl());
}

TEST(mezhuev_m_most_different_neighbor_elements_seq, ValidationCorrectInputs) {
  auto task_data = std::make_shared<ppc::core::TaskData>();
  std::vector<int> input_data = {1, 2, 3};
  task_data->inputs.push_back(reinterpret_cast<uint8_t*>(input_data.data()));
  task_data->inputs_count.push_back(static_cast<uint32_t>(input_data.size()));
  task_data->outputs.push_back(reinterpret_cast<uint8_t*>(input_data.data()));
  task_data->outputs_count.push_back(2);

  mezhuev_m_most_different_neighbor_elements_seq::MostDifferentNeighborElements task(task_data);
  EXPECT_TRUE(task.ValidationImpl());
}

TEST(mezhuev_m_most_different_neighbor_elements_seq, PreProcessingEmptyInput) {
  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs = {};
  mezhuev_m_most_different_neighbor_elements_seq::MostDifferentNeighborElements task(task_data);
  EXPECT_FALSE(task.PreProcessingImpl());
}

TEST(mezhuev_m_most_different_neighbor_elements_seq, PreProcessingValidInput) {
  auto task_data = std::make_shared<ppc::core::TaskData>();
  std::vector<int> input_data = {1, 2, 3};
  task_data->inputs.push_back(reinterpret_cast<uint8_t*>(input_data.data()));
  task_data->inputs_count.push_back(static_cast<uint32_t>(input_data.size()));

  mezhuev_m_most_different_neighbor_elements_seq::MostDifferentNeighborElements task(task_data);
  EXPECT_TRUE(task.PreProcessingImpl());
}

TEST(mezhuev_m_most_different_neighbor_elements_seq, PreProcessingInvalidSizeInput) {
  auto task_data = std::make_shared<ppc::core::TaskData>();
  std::vector<int> input_data = {1};
  task_data->inputs.push_back(reinterpret_cast<uint8_t*>(input_data.data()));
  task_data->inputs_count.push_back(static_cast<uint32_t>(input_data.size()));

  mezhuev_m_most_different_neighbor_elements_seq::MostDifferentNeighborElements task(task_data);
  EXPECT_FALSE(task.PreProcessingImpl());
}

TEST(mezhuev_m_most_different_neighbor_elements_seq, RunImplCorrectInput) {
  auto task_data = std::make_shared<ppc::core::TaskData>();
  std::vector<int> input_data = {1, 3, 2, 7};
  task_data->inputs.push_back(reinterpret_cast<uint8_t*>(input_data.data()));
  task_data->inputs_count.push_back(static_cast<uint32_t>(input_data.size()));
  task_data->outputs_count.push_back(2);

  mezhuev_m_most_different_neighbor_elements_seq::MostDifferentNeighborElements task(task_data);
  ASSERT_TRUE(task.PreProcessingImpl());
  ASSERT_TRUE(task.RunImpl());

  std::vector<int> expected_result = {2, 7};
  EXPECT_EQ(task.GetResult(), expected_result);
}

TEST(mezhuev_m_most_different_neighbor_elements_seq, RunImplEqualNeighbors) {
  auto task_data = std::make_shared<ppc::core::TaskData>();
  std::vector<int> input_data = {5, 5, 5, 5};
  task_data->inputs.push_back(reinterpret_cast<uint8_t*>(input_data.data()));
  task_data->inputs_count.push_back(static_cast<uint32_t>(input_data.size()));
  task_data->outputs_count.push_back(2);

  mezhuev_m_most_different_neighbor_elements_seq::MostDifferentNeighborElements task(task_data);
  ASSERT_TRUE(task.PreProcessingImpl());
  ASSERT_TRUE(task.RunImpl());

  std::vector<int> expected_result = {5, 5};
  EXPECT_EQ(task.GetResult(), expected_result);
}

TEST(mezhuev_m_most_different_neighbor_elements_seq, PostProcessingValidOutput) {
  auto task_data = std::make_shared<ppc::core::TaskData>();
  std::vector<int> input_data = {10, 20, 30};
  task_data->inputs.push_back(reinterpret_cast<uint8_t*>(input_data.data()));
  task_data->inputs_count.push_back(static_cast<uint32_t>(input_data.size()));

  std::vector<int> output_data(2);
  task_data->outputs.push_back(reinterpret_cast<uint8_t*>(output_data.data()));
  task_data->outputs_count.push_back(2);

  mezhuev_m_most_different_neighbor_elements_seq::MostDifferentNeighborElements task(task_data);

  ASSERT_TRUE(task.PreProcessingImpl());
  ASSERT_TRUE(task.RunImpl());
  EXPECT_TRUE(task.PostProcessingImpl());

  std::vector<int> expected_output = {10, 20};
  EXPECT_EQ(output_data, expected_output);
}

TEST(mezhuev_m_most_different_neighbor_elements_seq, PostProcessingEmptyOutput) {
  auto task_data = std::make_shared<ppc::core::TaskData>();

  std::vector<int> input_data = {10, 20};
  task_data->inputs.push_back(reinterpret_cast<uint8_t*>(input_data.data()));
  task_data->inputs_count.push_back(static_cast<uint32_t>(input_data.size()));

  task_data->outputs.clear();
  mezhuev_m_most_different_neighbor_elements_seq::MostDifferentNeighborElements task(task_data);

  ASSERT_TRUE(task.PreProcessingImpl());
  ASSERT_TRUE(task.RunImpl());
  EXPECT_FALSE(task.PostProcessingImpl());
}

TEST(mezhuev_m_most_different_neighbor_elements_seq, RunImplInsufficientInput) {
  auto task_data = std::make_shared<ppc::core::TaskData>();
  std::vector<int> input_data = {10};
  task_data->inputs.push_back(reinterpret_cast<uint8_t*>(input_data.data()));
  task_data->inputs_count.push_back(static_cast<uint32_t>(input_data.size()));

  mezhuev_m_most_different_neighbor_elements_seq::MostDifferentNeighborElements task(task_data);
  ASSERT_FALSE(task.RunImpl());
}

TEST(mezhuev_m_most_different_neighbor_elements_seq, ValidationWithNoOutputSpace) {
  auto task_data = std::make_shared<ppc::core::TaskData>();
  std::vector<int> input_data = {1, 2, 3};
  task_data->inputs.push_back(reinterpret_cast<uint8_t*>(input_data.data()));
  task_data->inputs_count.push_back(static_cast<uint32_t>(input_data.size()));
  task_data->outputs.push_back(reinterpret_cast<uint8_t*>(input_data.data()));
  task_data->outputs_count.push_back(1);

  mezhuev_m_most_different_neighbor_elements_seq::MostDifferentNeighborElements task(task_data);
  EXPECT_FALSE(task.ValidationImpl());
}
