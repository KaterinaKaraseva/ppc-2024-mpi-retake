#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <string>
#include <vector>

#include "core/task/include/task.hpp"
#include "seq/chernova_n_word_count/include/ops_seq.hpp"

std::vector<char> chernova_n_word_count_seq::GenerateWords(int k) {
  const std::string words[] = {"one", "two", "three"};

  std::string result;
  size_t j = words->size();

  for (int i = 0; i < k; ++i) {
    result += words[rand() % (j)];
    if (i < k - 1) {
      result += ' ';
    }
  }

  return {result.begin(), result.end()};
}

const int k_ = 50;
namespace {
std::vector<char> test_data_seq = chernova_n_word_count_seq::GenerateWords(k_);
}  // namespace

TEST(chernova_n_word_count_seq, Test_empty_string) {
  std::vector<char> in = {};
  std::vector<int> out(1, 0);

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  chernova_n_word_count_seq::TestTaskSequential test_task_sequential(task_data_seq);
  ASSERT_FALSE(test_task_sequential.ValidationImpl());
}

TEST(chernova_n_word_count_seq, Test_five_words) {
  std::vector<char> in;
  std::string test_string = "This is a test phrase";
  in.resize(test_string.size());
  std::ranges::copy(test_string, in.begin());
  std::vector<int> out(1, 0);

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  chernova_n_word_count_seq::TestTaskSequential test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  ASSERT_EQ(out[0], 5);
}

TEST(chernova_n_word_count_seq, Test_five_words_with_space_and_hyphen) {
  std::vector<char> in;
  std::string test_string = "This   is a - test phrase";
  in.resize(test_string.size());
  std::ranges::copy(test_string, in.begin());
  std::vector<int> out(1, 0);

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  chernova_n_word_count_seq::TestTaskSequential test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  ASSERT_EQ(out[0], 5);
}

TEST(chernova_n_word_count_seq, Test_ten_words) {
  std::vector<char> in;
  std::string test_string = "This is a test phrase, I really love this phrase";
  in.resize(test_string.size());
  std::ranges::copy(test_string, in.begin());
  std::vector<int> out(1, 0);

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  chernova_n_word_count_seq::TestTaskSequential test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  ASSERT_EQ(out[0], 10);
}

TEST(chernova_n_word_count_seq, Test_five_words_with_a_lot_of_space) {
  std::vector<char> in;
  std::string test_string = "This               is           a             test                phrase";
  in.resize(test_string.size());
  std::ranges::copy(test_string, in.begin());
  std::vector<int> out(1, 0);

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  chernova_n_word_count_seq::TestTaskSequential test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  ASSERT_EQ(out[0], 5);
}

TEST(chernova_n_word_count_seq, Test_twenty_words) {
  std::vector<char> in;
  std::string test_string =
      "This is a test phrase, I really love this phrase. This is a test phrase, I really love this phrase";
  in.resize(test_string.size());
  std::ranges::copy(test_string, in.begin());
  std::vector<int> out(1, 0);

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  chernova_n_word_count_seq::TestTaskSequential test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  ASSERT_EQ(out[0], 20);
}

TEST(chernova_n_word_count_seq, Test_five_words_with_space_in_the_end) {
  std::vector<char> in;
  std::string test_string = "This is a test phrase           ";
  in.resize(test_string.size());
  std::ranges::copy(test_string, in.begin());
  std::vector<int> out(1, 0);

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  chernova_n_word_count_seq::TestTaskSequential test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  ASSERT_EQ(out[0], 5);
}

TEST(chernova_n_word_count_seq, Test_random_fifty_words) {
  std::vector<char> in = test_data_seq;
  std::vector<int> out(1, 0);

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  chernova_n_word_count_seq::TestTaskSequential test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  ASSERT_EQ(out[0], 50);
}