#include <gtest/gtest.h>

#include <cstdint>
#include <cstdlib>
#include <ctime>
#include <memory>
#include <string>
#include <vector>

#include "core/task/include/task.hpp"
#include "seq/budazhapova_e_count_freq_character/include/count_freq_counter_header.h"

namespace budazhapova_e_count_freq_chart_seq {
namespace {
std::string GetRandomString(int length) {
  static std::string charset = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890";
  std::string result;
  result.resize(length);

  srand(time(nullptr));
  for (int i = 0; i < length; i++) {
    result[i] = charset[rand() % charset.size()];
  }
  return result;
}
}  // namespace
}  // namespace budazhapova_e_count_freq_chart_seq

TEST(budazhapova_e_count_freq_chart_seq, ordinary_test) {
  std::string line = "dsdasdasdsadsadsadsxzcacsdvfdggregfgdgwdvfsdfdvvbvbvbvbvbvbvbvbdsfdsfdsfsdfbcbfbvbvbv";
  std::vector<std::string> in(1, line);
  std::vector<int> out(1, 0);

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  // Create Task
  budazhapova_e_count_freq_chart_seq::TestTaskSequential test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  ASSERT_EQ(out[0], 17);
}

TEST(budazhapova_e_count_freq_chart_seq, test_if_character_is_not_in_line) {
  std::string line = "aaaaaaa pochemu tak neponyatno ya hochu spat!!!!";
  std::vector<std::string> in(1, line);
  std::vector<int> out(1, 0);

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  // Create Task
  budazhapova_e_count_freq_chart_seq::TestTaskSequential test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  ASSERT_EQ(out[0], 11);
}

TEST(budazhapova_e_count_freq_chart_seq, test_if_character_is_one) {
  std::string line = " ";
  std::vector<std::string> in(1, line);
  std::vector<int> out(1, 0);

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  // Create Task
  budazhapova_e_count_freq_chart_seq::TestTaskSequential test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  ASSERT_EQ(out[0], 1);
}

TEST(budazhapova_e_count_freq_chart_seq, test_with_random_string) {
  std::string line;
  const int size_string = 10;
  line = budazhapova_e_count_freq_chart_seq::GetRandomString(size_string);
  std::vector<std::string> in(1, line);
  std::vector<int> out(1, 0);

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  // Create Task
  budazhapova_e_count_freq_chart_seq::TestTaskSequential test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
}