#include <gtest/gtest.h>

#include <algorithm>
#include <boost/mpi/communicator.hpp>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "core/task/include/task.hpp"
#include "mpi/chernova_n_word_count/include/ops_mpi.hpp"

std::vector<char> chernova_n_word_count_mpi::GenerateWords(int k) {
  const std::string words[] = {"one", "two", "three"};
  const int word_array_size = sizeof(words) / sizeof(words[0]);
  std::string result;

  for (int i = 0; i < k; ++i) {
    result += words[i % word_array_size];
    if (i < k - 1) {
      result += ' ';
    }
  }

  return {result.begin(), result.end()};
}

const int k_ = 50;
namespace {
std::vector<char> test_data_parallel = chernova_n_word_count_mpi::GenerateWords(k_);
}  // namespace

TEST(chernova_n_word_count_mpi, Test_empty_string) {
  boost::mpi::communicator world;
  std::vector<char> in = {};
  std::vector<int> out(1, 0);

  auto task_data_mpi = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
    task_data_mpi->inputs_count.emplace_back(in.size());
    task_data_mpi->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
    task_data_mpi->outputs_count.emplace_back(out.size());
    //}

    chernova_n_word_count_mpi::TestMPITaskParallel test_task_mpi(task_data_mpi);
    ASSERT_FALSE(test_task_mpi.ValidationImpl());
  }
}

TEST(chernova_n_word_count_mpi, Test_five_words) {
  boost::mpi::communicator world;
  std::vector<char> in;
  std::string test_string = "This is a test phrase";
  in.resize(test_string.size());
  std::ranges::copy(test_string, in.begin());
  std::vector<int> out(1, 0);

  auto task_data_mpi = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
    task_data_mpi->inputs_count.emplace_back(in.size());
    task_data_mpi->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
    task_data_mpi->outputs_count.emplace_back(out.size());
  }

  chernova_n_word_count_mpi::TestMPITaskParallel test_task_mpi(task_data_mpi);
  ASSERT_EQ(test_task_mpi.Validation(), true);
  test_task_mpi.PreProcessing();
  test_task_mpi.Run();
  test_task_mpi.PostProcessing();

  if (world.rank() == 0) {
    std::vector<int> reference_word_count(1, 0);
    auto task_data_seq = std::make_shared<ppc::core::TaskData>();
    task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
    task_data_seq->inputs_count.emplace_back(in.size());
    task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_word_count.data()));
    task_data_seq->outputs_count.emplace_back(reference_word_count.size());

    chernova_n_word_count_mpi::TestMPITaskSequential test_task_seq(task_data_seq);
    ASSERT_EQ(test_task_seq.Validation(), true);
    test_task_seq.PreProcessing();
    test_task_seq.Run();
    test_task_seq.PostProcessing();

    ASSERT_EQ(out[0], reference_word_count[0]);
  }
}

TEST(chernova_n_word_count_mpi, Test_five_words_with_space_and_hyphen) {
  boost::mpi::communicator world;
  std::vector<char> in;
  std::string test_string = "This   is a - test phrase";
  in.resize(test_string.size());
  std::ranges::copy(test_string, in.begin());
  std::vector<int> out(1, 0);

  auto task_data_mpi = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
    task_data_mpi->inputs_count.emplace_back(in.size());
    task_data_mpi->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
    task_data_mpi->outputs_count.emplace_back(out.size());
  }

  chernova_n_word_count_mpi::TestMPITaskParallel test_task_mpi(task_data_mpi);
  ASSERT_EQ(test_task_mpi.Validation(), true);
  test_task_mpi.PreProcessing();
  test_task_mpi.Run();
  test_task_mpi.PostProcessing();

  if (world.rank() == 0) {
    std::vector<int> reference_word_count(1, 0);
    auto task_data_seq = std::make_shared<ppc::core::TaskData>();
    task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
    task_data_seq->inputs_count.emplace_back(in.size());
    task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_word_count.data()));
    task_data_seq->outputs_count.emplace_back(reference_word_count.size());

    chernova_n_word_count_mpi::TestMPITaskSequential test_task_seq(task_data_seq);
    ASSERT_EQ(test_task_seq.Validation(), true);
    test_task_seq.PreProcessing();
    test_task_seq.Run();
    test_task_seq.PostProcessing();

    ASSERT_EQ(out[0], reference_word_count[0]);
  }
}

TEST(chernova_n_word_count_mpi, Test_ten_words) {
  boost::mpi::communicator world;
  std::vector<char> in;
  std::string test_string = "This is a test phrase, I really love this phrase";
  in.resize(test_string.size());
  std::ranges::copy(test_string, in.begin());
  std::vector<int> out(1, 0);

  auto task_data_mpi = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
    task_data_mpi->inputs_count.emplace_back(in.size());
    task_data_mpi->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
    task_data_mpi->outputs_count.emplace_back(out.size());
  }

  chernova_n_word_count_mpi::TestMPITaskParallel test_task_mpi(task_data_mpi);
  ASSERT_EQ(test_task_mpi.Validation(), true);
  test_task_mpi.PreProcessing();
  test_task_mpi.Run();
  test_task_mpi.PostProcessing();

  if (world.rank() == 0) {
    std::vector<int> reference_word_count(1, 0);
    auto task_data_seq = std::make_shared<ppc::core::TaskData>();
    task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
    task_data_seq->inputs_count.emplace_back(in.size());
    task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_word_count.data()));
    task_data_seq->outputs_count.emplace_back(reference_word_count.size());

    chernova_n_word_count_mpi::TestMPITaskSequential test_task_seq(task_data_seq);
    ASSERT_EQ(test_task_seq.Validation(), true);
    test_task_seq.PreProcessing();
    test_task_seq.Run();
    test_task_seq.PostProcessing();

    ASSERT_EQ(out[0], reference_word_count[0]);
  }
}

TEST(chernova_n_word_count_mpi, Test_five_words_with_a_lot_of_space) {
  boost::mpi::communicator world;
  std::vector<char> in;
  std::string test_string = "This               is           a             test                phrase";
  in.resize(test_string.size());
  std::ranges::copy(test_string, in.begin());
  std::vector<int> out(1, 0);

  auto task_data_mpi = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
    task_data_mpi->inputs_count.emplace_back(in.size());
    task_data_mpi->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
    task_data_mpi->outputs_count.emplace_back(out.size());
  }

  chernova_n_word_count_mpi::TestMPITaskParallel test_task_mpi(task_data_mpi);
  ASSERT_EQ(test_task_mpi.Validation(), true);
  test_task_mpi.PreProcessing();
  test_task_mpi.Run();
  test_task_mpi.PostProcessing();

  if (world.rank() == 0) {
    std::vector<int> reference_word_count(1, 0);
    auto task_data_seq = std::make_shared<ppc::core::TaskData>();
    task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
    task_data_seq->inputs_count.emplace_back(in.size());
    task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_word_count.data()));
    task_data_seq->outputs_count.emplace_back(reference_word_count.size());

    chernova_n_word_count_mpi::TestMPITaskSequential test_task_seq(task_data_seq);
    ASSERT_EQ(test_task_seq.Validation(), true);
    test_task_seq.PreProcessing();
    test_task_seq.Run();
    test_task_seq.PostProcessing();

    ASSERT_EQ(out[0], reference_word_count[0]);
  }
}

TEST(chernova_n_word_count_mpi, Test_twenty_words) {
  boost::mpi::communicator world;
  std::vector<char> in;
  std::string test_string =
      "This is a test phrase, I really love this phrase. This is a test phrase, I really love this phrase";
  in.resize(test_string.size());
  std::ranges::copy(test_string, in.begin());
  std::vector<int> out(1, 0);

  auto task_data_mpi = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
    task_data_mpi->inputs_count.emplace_back(in.size());
    task_data_mpi->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
    task_data_mpi->outputs_count.emplace_back(out.size());
  }

  chernova_n_word_count_mpi::TestMPITaskParallel test_task_mpi(task_data_mpi);
  ASSERT_EQ(test_task_mpi.Validation(), true);
  test_task_mpi.PreProcessing();
  test_task_mpi.Run();
  test_task_mpi.PostProcessing();

  if (world.rank() == 0) {
    std::vector<int> reference_word_count(1, 0);
    auto task_data_seq = std::make_shared<ppc::core::TaskData>();
    task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
    task_data_seq->inputs_count.emplace_back(in.size());
    task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_word_count.data()));
    task_data_seq->outputs_count.emplace_back(reference_word_count.size());

    chernova_n_word_count_mpi::TestMPITaskSequential test_task_seq(task_data_seq);
    ASSERT_EQ(test_task_seq.Validation(), true);
    test_task_seq.PreProcessing();
    test_task_seq.Run();
    test_task_seq.PostProcessing();

    ASSERT_EQ(out[0], reference_word_count[0]);
  }
}

TEST(chernova_n_word_count_mpi, Test_five_words_with_space_in_the_end) {
  boost::mpi::communicator world;
  std::vector<char> in;
  std::string test_string = "This is a test phrase           ";
  in.resize(test_string.size());
  std::ranges::copy(test_string, in.begin());
  std::vector<int> out(1, 0);

  auto task_data_mpi = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
    task_data_mpi->inputs_count.emplace_back(in.size());
    task_data_mpi->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
    task_data_mpi->outputs_count.emplace_back(out.size());
  }

  chernova_n_word_count_mpi::TestMPITaskParallel test_task_mpi(task_data_mpi);
  ASSERT_EQ(test_task_mpi.Validation(), true);
  test_task_mpi.PreProcessing();
  test_task_mpi.Run();
  test_task_mpi.PostProcessing();

  if (world.rank() == 0) {
    std::vector<int> reference_word_count(1, 0);
    auto task_data_seq = std::make_shared<ppc::core::TaskData>();
    task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
    task_data_seq->inputs_count.emplace_back(in.size());
    task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_word_count.data()));
    task_data_seq->outputs_count.emplace_back(reference_word_count.size());

    chernova_n_word_count_mpi::TestMPITaskSequential test_task_seq(task_data_seq);
    ASSERT_EQ(test_task_seq.Validation(), true);
    test_task_seq.PreProcessing();
    test_task_seq.Run();
    test_task_seq.PostProcessing();

    ASSERT_EQ(out[0], reference_word_count[0]);
  }
}

TEST(chernova_n_word_count_mpi, Test_random_fifty_words) {
  boost::mpi::communicator world;
  std::vector<char> in = test_data_parallel;
  std::vector<int> out(1, 0);

  auto task_data_mpi = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
    task_data_mpi->inputs_count.emplace_back(in.size());
    task_data_mpi->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
    task_data_mpi->outputs_count.emplace_back(out.size());
  }

  chernova_n_word_count_mpi::TestMPITaskParallel test_task_mpi(task_data_mpi);
  ASSERT_EQ(test_task_mpi.Validation(), true);
  test_task_mpi.PreProcessing();
  test_task_mpi.Run();
  test_task_mpi.PostProcessing();

  if (world.rank() == 0) {
    std::vector<int> reference_word_count(1, 0);
    auto task_data_seq = std::make_shared<ppc::core::TaskData>();
    task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
    task_data_seq->inputs_count.emplace_back(in.size());
    task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_word_count.data()));
    task_data_seq->outputs_count.emplace_back(reference_word_count.size());

    chernova_n_word_count_mpi::TestMPITaskSequential test_task_seq(task_data_seq);
    ASSERT_EQ(test_task_seq.Validation(), true);
    test_task_seq.PreProcessing();
    test_task_seq.Run();
    test_task_seq.PostProcessing();

    ASSERT_EQ(out[0], reference_word_count[0]);
  }
}