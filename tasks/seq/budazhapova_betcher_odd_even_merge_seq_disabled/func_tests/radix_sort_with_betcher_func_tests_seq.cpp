
#include <gtest/gtest.h>

#include <cstdint>
#include <cstdlib>
#include <ctime>
#include <memory>
#include <random>
#include <vector>

#include "core/task/include/task.hpp"
#include "seq/budazhapova_betcher_odd_even_merge_seq/include/radix_sort_with_betcher_seq.h"

namespace budazhapova_betcher_odd_even_merge_seq {
namespace {
std::vector<int> GenerateRandomVector(int size, int min_value, int max_value) {
  std::vector<int> random_vector(size);
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> dis(min_value, max_value);
  for (int i = 0; i < size; ++i) {
    random_vector[i] = dis(gen);
  }

  return random_vector;
}
}  // namespace
}  // namespace budazhapova_betcher_odd_even_merge_seq
TEST(budazhapova_betcher_odd_even_merge_seq, ordinary_test) {
  std::vector<int> input_vector = {34, 12, 5, 78, 23, 45, 67, 89, 10, 2, 56, 43, 91, 15, 30};
  std::vector<int> out(15, 0);
  std::vector<int> sorted_vector = {2, 5, 10, 12, 15, 23, 30, 34, 43, 45, 56, 67, 78, 89, 91};

  std::shared_ptr<ppc::core::TaskData> task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_vector.data()));
  task_data_seq->inputs_count.emplace_back(input_vector.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  budazhapova_betcher_odd_even_merge_seq::MergeSequential test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  ASSERT_EQ(out, sorted_vector);
}

TEST(budazhapova_betcher_odd_even_merge_seq, random_vector_test) {
  std::vector<int> input_vector = budazhapova_betcher_odd_even_merge_seq::GenerateRandomVector(100, 5, 100);
  std::vector<int> out(100, 0);

  std::shared_ptr<ppc::core::TaskData> task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_vector.data()));
  task_data_seq->inputs_count.emplace_back(input_vector.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  budazhapova_betcher_odd_even_merge_seq::MergeSequential test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
}

TEST(budazhapova_betcher_odd_even_merge_seq, Validation_test) {
  std::vector<int> input_vector = {};
  std::vector<int> out(1, 0);

  std::shared_ptr<ppc::core::TaskData> task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_vector.data()));
  task_data_seq->inputs_count.emplace_back(input_vector.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  budazhapova_betcher_odd_even_merge_seq::MergeSequential test_task_sequential(task_data_seq);
  ASSERT_FALSE(test_task_sequential.Validation());
}