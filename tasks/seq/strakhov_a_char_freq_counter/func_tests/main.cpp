#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "core/task/include/task.hpp"
#include "seq/strakhov_a_char_freq_counter/include/ops_seq.hpp"

namespace strakhov_a_char_freq_counter_seq {
namespace {
std::vector<char> FillRandomChars(int size, const std::string &charset) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> dist(0, static_cast<int>(charset.size()) - 1);

  std::vector<char> result(size);
  for (char &c : result) {
    c = charset[dist(gen)];
  }
  return result;
}
}  // namespace
}  // namespace strakhov_a_char_freq_counter_seq

TEST(strakhov_a_char_freq_counter_seq, test_same_characters) {
  std::vector<char> in_string;
  int expectation = 1000;
  std::vector<int> out_seq(1, 0);
  std::vector<char> in_target(1, 'a');

  // Sequential

  // Create task_data

  in_string = std::vector<char>(expectation, 'a');
  auto task_data_mpi_seq = std::make_shared<ppc::core::TaskData>();

  task_data_mpi_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_string.data()));
  task_data_mpi_seq->inputs_count.emplace_back(in_string.size());
  task_data_mpi_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_target.data()));
  task_data_mpi_seq->inputs_count.emplace_back(in_target.size());
  task_data_mpi_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_seq.data()));
  task_data_mpi_seq->outputs_count.emplace_back(out_seq.size());

  strakhov_a_char_freq_counter_seq::CharFreqCounterSeq test_task_seq(task_data_mpi_seq);
  ASSERT_EQ(test_task_seq.ValidationImpl(), true);
  test_task_seq.PreProcessingImpl();
  test_task_seq.RunImpl();
  test_task_seq.PostProcessingImpl();

  ASSERT_EQ(out_seq[0], expectation);
}

TEST(strakhov_a_char_freq_counter_seq, test_no_characters) {
  std::vector<char> in_string;
  int expectation = 0;
  std::vector<int> out_seq(1, 0);
  std::vector<char> in_target(1, 'a');

  // Sequential

  // Create task_data

  in_string = std::vector<char>(1000, 'b');
  auto task_data_mpi_seq = std::make_shared<ppc::core::TaskData>();

  task_data_mpi_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_string.data()));
  task_data_mpi_seq->inputs_count.emplace_back(in_string.size());
  task_data_mpi_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_target.data()));
  task_data_mpi_seq->inputs_count.emplace_back(in_target.size());
  task_data_mpi_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_seq.data()));
  task_data_mpi_seq->outputs_count.emplace_back(out_seq.size());

  strakhov_a_char_freq_counter_seq::CharFreqCounterSeq test_task_seq(task_data_mpi_seq);
  ASSERT_EQ(test_task_seq.ValidationImpl(), true);
  test_task_seq.PreProcessingImpl();
  test_task_seq.RunImpl();
  test_task_seq.PostProcessingImpl();

  ASSERT_EQ(out_seq[0], expectation);
}

TEST(strakhov_a_char_freq_counter_seq, test_empty_string) {
  std::vector<char> in_string{};
  int expectation = 0;
  std::vector<int> out_seq(1, 0);
  std::vector<char> in_target(1, 'a');

  // Sequential

  // Create task_data

  auto task_data_mpi_seq = std::make_shared<ppc::core::TaskData>();

  task_data_mpi_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_string.data()));
  task_data_mpi_seq->inputs_count.emplace_back(in_string.size());
  task_data_mpi_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_target.data()));
  task_data_mpi_seq->inputs_count.emplace_back(in_target.size());
  task_data_mpi_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_seq.data()));
  task_data_mpi_seq->outputs_count.emplace_back(out_seq.size());

  strakhov_a_char_freq_counter_seq::CharFreqCounterSeq test_task_seq(task_data_mpi_seq);
  ASSERT_EQ(test_task_seq.ValidationImpl(), true);
  test_task_seq.PreProcessingImpl();
  test_task_seq.RunImpl();
  test_task_seq.PostProcessingImpl();

  ASSERT_EQ(out_seq[0], expectation);
}

TEST(strakhov_a_char_freq_counter_seq, test_single_character) {
  std::vector<char> in_string{};

  std::vector<int> out_seq(1, 0);
  std::vector<char> in_target(1, 'b');

  // Sequential

  // Create task_data
  int expectation = 1;
  in_string = std::vector<char>(1000, 'a');
  in_string[500] = 'b';
  auto task_data_mpi_seq = std::make_shared<ppc::core::TaskData>();

  task_data_mpi_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_string.data()));
  task_data_mpi_seq->inputs_count.emplace_back(in_string.size());
  task_data_mpi_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_target.data()));
  task_data_mpi_seq->inputs_count.emplace_back(in_target.size());
  task_data_mpi_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_seq.data()));
  task_data_mpi_seq->outputs_count.emplace_back(out_seq.size());

  strakhov_a_char_freq_counter_seq::CharFreqCounterSeq test_task_seq(task_data_mpi_seq);
  ASSERT_EQ(test_task_seq.ValidationImpl(), true);
  test_task_seq.PreProcessingImpl();
  test_task_seq.RunImpl();
  test_task_seq.PostProcessingImpl();

  ASSERT_EQ(out_seq[0], expectation);
}

TEST(strakhov_a_char_freq_counter_seq, simple_test_1) {
  std::vector<char> in_string = {'H', 'e', 'l', 'l', 'o'};
  int expectation = 1;
  std::vector<int> out_seq(1, 0);
  std::vector<char> in_target(1, 'H');

  // Sequential

  // Create task_data

  auto task_data_mpi_seq = std::make_shared<ppc::core::TaskData>();
  task_data_mpi_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_string.data()));
  task_data_mpi_seq->inputs_count.emplace_back(in_string.size());
  task_data_mpi_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_target.data()));
  task_data_mpi_seq->inputs_count.emplace_back(in_target.size());
  task_data_mpi_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_seq.data()));
  task_data_mpi_seq->outputs_count.emplace_back(out_seq.size());

  strakhov_a_char_freq_counter_seq::CharFreqCounterSeq test_task_seq(task_data_mpi_seq);
  ASSERT_EQ(test_task_seq.ValidationImpl(), true);
  test_task_seq.PreProcessingImpl();
  test_task_seq.RunImpl();
  test_task_seq.PostProcessingImpl();

  ASSERT_EQ(out_seq[0], expectation);
}
TEST(strakhov_a_char_freq_counter_seq, simple_test_2) {
  std::vector<char> in_string = {'H', 'e', 'l', 'l', 'o'};
  int expectation = 0;
  std::vector<int> out_seq(1, 0);
  std::vector<char> in_target(1, 'h');

  // Sequential

  // Create task_data

  auto task_data_mpi_seq = std::make_shared<ppc::core::TaskData>();
  task_data_mpi_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_string.data()));
  task_data_mpi_seq->inputs_count.emplace_back(in_string.size());
  task_data_mpi_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_target.data()));
  task_data_mpi_seq->inputs_count.emplace_back(in_target.size());
  task_data_mpi_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_seq.data()));
  task_data_mpi_seq->outputs_count.emplace_back(out_seq.size());

  strakhov_a_char_freq_counter_seq::CharFreqCounterSeq test_task_seq(task_data_mpi_seq);
  ASSERT_EQ(test_task_seq.ValidationImpl(), true);
  test_task_seq.PreProcessingImpl();
  test_task_seq.RunImpl();
  test_task_seq.PostProcessingImpl();

  ASSERT_EQ(out_seq[0], expectation);
}
TEST(strakhov_a_char_freq_counter_seq, simple_test_3) {
  std::vector<char> in_string = {'H', 'e', 'l', 'l', 'o'};
  int expectation = 2;
  std::vector<int> out_seq(1, 0);
  std::vector<char> in_target(1, 'l');

  // Sequential

  // Create task_data
  auto task_data_mpi_seq = std::make_shared<ppc::core::TaskData>();

  task_data_mpi_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_string.data()));
  task_data_mpi_seq->inputs_count.emplace_back(in_string.size());
  task_data_mpi_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_target.data()));
  task_data_mpi_seq->inputs_count.emplace_back(in_target.size());
  task_data_mpi_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_seq.data()));
  task_data_mpi_seq->outputs_count.emplace_back(out_seq.size());

  strakhov_a_char_freq_counter_seq::CharFreqCounterSeq test_task_seq(task_data_mpi_seq);
  ASSERT_EQ(test_task_seq.ValidationImpl(), true);
  test_task_seq.PreProcessingImpl();
  test_task_seq.RunImpl();
  test_task_seq.PostProcessingImpl();

  ASSERT_EQ(out_seq[0], expectation);
}

TEST(strakhov_a_char_freq_counter_seq, random_string) {
  std::vector<char> in_target = strakhov_a_char_freq_counter_seq::FillRandomChars(
      1, "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789!@#$%^&*");
  std::vector<char> in_string = strakhov_a_char_freq_counter_seq::FillRandomChars(
      1000, "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789!@#$%^&*");
  std::vector<int> out_par(1, 0);
  std::vector<int> out_seq(1, 0);
  int expectance_r = static_cast<int>(std::count(in_string.begin(), in_string.end(), in_target[0]));
  // Sequential

  // Create task_data
  auto task_data_mpi_seq = std::make_shared<ppc::core::TaskData>();

  task_data_mpi_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_string.data()));
  task_data_mpi_seq->inputs_count.emplace_back(in_string.size());
  task_data_mpi_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_target.data()));
  task_data_mpi_seq->inputs_count.emplace_back(in_target.size());
  task_data_mpi_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_seq.data()));
  task_data_mpi_seq->outputs_count.emplace_back(out_seq.size());

  strakhov_a_char_freq_counter_seq::CharFreqCounterSeq test_task_seq(task_data_mpi_seq);
  ASSERT_EQ(test_task_seq.ValidationImpl(), true);
  test_task_seq.PreProcessingImpl();
  test_task_seq.RunImpl();
  test_task_seq.PostProcessingImpl();

  ASSERT_EQ(out_seq[0], expectance_r);
}
