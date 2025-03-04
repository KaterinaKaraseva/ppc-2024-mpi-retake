#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <memory>
#include <vector>

#include "core/task/include/task.hpp"
#include "seq/kalinin_d_odd_even_shellsort/include/header_seq_odd_even_shell.hpp"
TEST(kalinin_d_odd_even_shell_seq, Test_odd_even_sort_0) {
  const int n = 0;
  // Create data
  std::vector<int> arr(n);
  std::vector<int> out(n);
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(arr.data()));
  task_data_seq->inputs_count.emplace_back(arr.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  // Create Task
  kalinin_d_odd_even_shell_seq::OddEvenShellSeq test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), false);
}

TEST(kalinin_d_odd_even_shell_seq, Test_odd_even_sort_1000) {
  const int n = 1000;
  // Create data
  std::vector<int> arr(n);
  std::vector<int> out(n);
  kalinin_d_odd_even_shell_seq::GimmeRandVec(arr);
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(arr.data()));
  task_data_seq->inputs_count.emplace_back(arr.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  // Create Task
  kalinin_d_odd_even_shell_seq::OddEvenShellSeq test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();

  test_task_sequential.PostProcessing();
  std::ranges::sort(arr.begin(), arr.end());
  ASSERT_EQ(arr, out);
}

TEST(kalinin_d_odd_even_shell_seq, Test_odd_even_sort_999) {
  const int n = 999;
  // Create data
  std::vector<int> arr(n);
  std::vector<int> out(n);
  kalinin_d_odd_even_shell_seq::GimmeRandVec(arr);
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(arr.data()));
  task_data_seq->inputs_count.emplace_back(arr.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  // Create Task
  kalinin_d_odd_even_shell_seq::OddEvenShellSeq test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();

  test_task_sequential.PostProcessing();
  std::ranges::sort(arr.begin(), arr.end());
  ASSERT_EQ(arr, out);
}

TEST(kalinin_d_odd_even_shell_seq, Test_odd_even_sort_9999) {
  const int n = 9999;
  // Create data
  std::vector<int> arr(n);
  std::vector<int> out(n);
  kalinin_d_odd_even_shell_seq::GimmeRandVec(arr);
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(arr.data()));
  task_data_seq->inputs_count.emplace_back(arr.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  // Create Task
  kalinin_d_odd_even_shell_seq::OddEvenShellSeq test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();

  test_task_sequential.PostProcessing();
  std::ranges::sort(arr.begin(), arr.end());
  ASSERT_EQ(arr, out);
}

TEST(kalinin_d_odd_even_shell_seq, Test_odd_even_sort_1021) {
  const int n = 1021;
  // Create data
  std::vector<int> arr(n);
  std::vector<int> out(n);
  kalinin_d_odd_even_shell_seq::GimmeRandVec(arr);
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(arr.data()));
  task_data_seq->inputs_count.emplace_back(arr.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  // Create Task
  kalinin_d_odd_even_shell_seq::OddEvenShellSeq test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();

  test_task_sequential.PostProcessing();
  std::ranges::sort(arr.begin(), arr.end());
  ASSERT_EQ(arr, out);
}