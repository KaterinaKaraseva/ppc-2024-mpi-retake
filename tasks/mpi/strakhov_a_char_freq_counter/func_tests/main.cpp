#include <gtest/gtest.h>

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <cstdint>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "core/task/include/task.hpp"
#include "mpi/strakhov_a_char_freq_counter/include/ops_mpi.hpp"

namespace strakhov_a_char_freq_counter_mpi {
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
}  // namespace strakhov_a_char_freq_counter_mpi

TEST(strakhov_a_char_freq_counter_mpi, test_same_characters) {
  boost::mpi::communicator world;
  std::vector<char> in_string;
  std::vector<int> out_par(1, 0);
  std::vector<int> out_seq(1, 0);
  std::vector<char> in_target(1, 'a');

  // Parallel

  // Create task_data
  auto task_data_mpi_par = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    in_string = std::vector<char>(1000, 'a');

    task_data_mpi_par->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_string.data()));
    task_data_mpi_par->inputs_count.emplace_back(in_string.size());
    task_data_mpi_par->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_target.data()));
    task_data_mpi_par->inputs_count.emplace_back(in_target.size());
    task_data_mpi_par->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_par.data()));
    task_data_mpi_par->outputs_count.emplace_back(out_par.size());
  }

  // Create Task
  strakhov_a_char_freq_counter_mpi::CharFreqCounterPar test_task_par(task_data_mpi_par);
  ASSERT_EQ(test_task_par.Validation(), true);
  test_task_par.PreProcessing();
  test_task_par.Run();
  test_task_par.PostProcessing();

  // Sequential

  // Create task_data

  if (world.rank() == 0) {
    in_string = std::vector<char>(1000, 'a');
    auto task_data_mpi_seq = std::make_shared<ppc::core::TaskData>();

    task_data_mpi_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_string.data()));
    task_data_mpi_seq->inputs_count.emplace_back(in_string.size());
    task_data_mpi_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_target.data()));
    task_data_mpi_seq->inputs_count.emplace_back(in_target.size());
    task_data_mpi_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_seq.data()));
    task_data_mpi_seq->outputs_count.emplace_back(out_seq.size());

    strakhov_a_char_freq_counter_mpi::CharFreqCounterSeq test_task_seq(task_data_mpi_seq);
    ASSERT_EQ(test_task_seq.Validation(), true);
    test_task_seq.PreProcessing();
    test_task_seq.Run();
    test_task_seq.PostProcessing();

    ASSERT_EQ(out_par[0], out_seq[0]);
  }
}

TEST(strakhov_a_char_freq_counter_mpi, test_no_characters) {
  boost::mpi::communicator world;
  std::vector<char> in_string;
  std::vector<int> out_par(1, 0);
  std::vector<int> out_seq(1, 0);
  std::vector<char> in_target(1, 'a');

  // Parallel

  // Create task_data
  auto task_data_mpi_par = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    in_string = std::vector<char>(1000, 'b');

    task_data_mpi_par->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_string.data()));
    task_data_mpi_par->inputs_count.emplace_back(in_string.size());
    task_data_mpi_par->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_target.data()));
    task_data_mpi_par->inputs_count.emplace_back(in_target.size());
    task_data_mpi_par->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_par.data()));
    task_data_mpi_par->outputs_count.emplace_back(out_par.size());
  }

  // Create Task
  strakhov_a_char_freq_counter_mpi::CharFreqCounterPar test_task_par(task_data_mpi_par);
  ASSERT_EQ(test_task_par.Validation(), true);
  test_task_par.PreProcessing();
  test_task_par.Run();
  test_task_par.PostProcessing();

  // Sequential

  // Create task_data

  if (world.rank() == 0) {
    in_string = std::vector<char>(1000, 'b');
    auto task_data_mpi_seq = std::make_shared<ppc::core::TaskData>();

    task_data_mpi_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_string.data()));
    task_data_mpi_seq->inputs_count.emplace_back(in_string.size());
    task_data_mpi_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_target.data()));
    task_data_mpi_seq->inputs_count.emplace_back(in_target.size());
    task_data_mpi_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_seq.data()));
    task_data_mpi_seq->outputs_count.emplace_back(out_seq.size());

    strakhov_a_char_freq_counter_mpi::CharFreqCounterSeq test_task_seq(task_data_mpi_seq);
    ASSERT_EQ(test_task_seq.Validation(), true);
    test_task_seq.PreProcessing();
    test_task_seq.Run();
    test_task_seq.PostProcessing();

    ASSERT_EQ(out_par[0], out_seq[0]);
  }
}

TEST(strakhov_a_char_freq_counter_mpi, test_empty_string) {
  boost::mpi::communicator world;
  std::vector<char> in_string{};
  std::vector<int> out_par(1, 0);
  std::vector<int> out_seq(1, 0);
  std::vector<char> in_target(1, 'a');

  // Parallel

  // Create task_data
  auto task_data_mpi_par = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    task_data_mpi_par->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_string.data()));
    task_data_mpi_par->inputs_count.emplace_back(in_string.size());
    task_data_mpi_par->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_target.data()));
    task_data_mpi_par->inputs_count.emplace_back(in_target.size());
    task_data_mpi_par->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_par.data()));
    task_data_mpi_par->outputs_count.emplace_back(out_par.size());
  }

  // Create Task
  strakhov_a_char_freq_counter_mpi::CharFreqCounterPar test_task_par(task_data_mpi_par);
  ASSERT_EQ(test_task_par.Validation(), true);
  test_task_par.PreProcessing();
  test_task_par.Run();
  test_task_par.PostProcessing();

  // Sequential

  // Create task_data

  if (world.rank() == 0) {
    auto task_data_mpi_seq = std::make_shared<ppc::core::TaskData>();

    task_data_mpi_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_string.data()));
    task_data_mpi_seq->inputs_count.emplace_back(in_string.size());
    task_data_mpi_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_target.data()));
    task_data_mpi_seq->inputs_count.emplace_back(in_target.size());
    task_data_mpi_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_seq.data()));
    task_data_mpi_seq->outputs_count.emplace_back(out_seq.size());

    strakhov_a_char_freq_counter_mpi::CharFreqCounterSeq test_task_seq(task_data_mpi_seq);
    ASSERT_EQ(test_task_seq.Validation(), true);
    test_task_seq.PreProcessing();
    test_task_seq.Run();
    test_task_seq.PostProcessing();

    ASSERT_EQ(out_par[0], out_seq[0]);
  }
}

TEST(strakhov_a_char_freq_counter_mpi, test_single_character) {
  boost::mpi::communicator world;
  std::vector<char> in_string{};
  std::vector<int> out_par(1, 0);
  std::vector<int> out_seq(1, 0);
  std::vector<char> in_target(1, 'b');

  // Parallel

  // Create task_data
  auto task_data_mpi_par = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    in_string = std::vector<char>(1000, 'a');
    in_string[500] = 'b';
    task_data_mpi_par->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_string.data()));
    task_data_mpi_par->inputs_count.emplace_back(in_string.size());
    task_data_mpi_par->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_target.data()));
    task_data_mpi_par->inputs_count.emplace_back(in_target.size());
    task_data_mpi_par->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_par.data()));
    task_data_mpi_par->outputs_count.emplace_back(out_par.size());
  }

  // Create Task
  strakhov_a_char_freq_counter_mpi::CharFreqCounterPar test_task_par(task_data_mpi_par);
  ASSERT_EQ(test_task_par.Validation(), true);
  test_task_par.PreProcessing();
  test_task_par.Run();
  test_task_par.PostProcessing();

  // Sequential

  // Create task_data

  if (world.rank() == 0) {
    in_string = std::vector<char>(1000, 'a');
    in_string[500] = 'b';
    auto task_data_mpi_seq = std::make_shared<ppc::core::TaskData>();

    task_data_mpi_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_string.data()));
    task_data_mpi_seq->inputs_count.emplace_back(in_string.size());
    task_data_mpi_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_target.data()));
    task_data_mpi_seq->inputs_count.emplace_back(in_target.size());
    task_data_mpi_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_seq.data()));
    task_data_mpi_seq->outputs_count.emplace_back(out_seq.size());

    strakhov_a_char_freq_counter_mpi::CharFreqCounterSeq test_task_seq(task_data_mpi_seq);
    ASSERT_EQ(test_task_seq.Validation(), true);
    test_task_seq.PreProcessing();
    test_task_seq.Run();
    test_task_seq.PostProcessing();

    ASSERT_EQ(out_par[0], out_seq[0]);
  }
}

TEST(strakhov_a_char_freq_counter_mpi, random_string) {
  // random generator

  boost::mpi::communicator world;
  std::vector<char> in_string = strakhov_a_char_freq_counter_mpi::FillRandomChars(
      300, "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789!@#$%^&*");
  std::vector<int> out_par(1, 0);
  std::vector<int> out_seq(1, 0);
  std::vector<char> in_target = strakhov_a_char_freq_counter_mpi::FillRandomChars(
      1, "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789!@#$%^&*");

  // Parallel

  // Create task_data
  auto task_data_mpi_par = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    task_data_mpi_par->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_string.data()));
    task_data_mpi_par->inputs_count.emplace_back(in_string.size());
    task_data_mpi_par->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_target.data()));
    task_data_mpi_par->inputs_count.emplace_back(in_target.size());
    task_data_mpi_par->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_par.data()));
    task_data_mpi_par->outputs_count.emplace_back(out_par.size());
  }

  // Create Task
  strakhov_a_char_freq_counter_mpi::CharFreqCounterPar test_task_par(task_data_mpi_par);
  ASSERT_EQ(test_task_par.Validation(), true);
  test_task_par.PreProcessing();
  test_task_par.Run();
  test_task_par.PostProcessing();

  // Sequential

  // Create task_data

  if (world.rank() == 0) {
    auto task_data_mpi_seq = std::make_shared<ppc::core::TaskData>();
    task_data_mpi_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_string.data()));
    task_data_mpi_seq->inputs_count.emplace_back(in_string.size());
    task_data_mpi_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_target.data()));
    task_data_mpi_seq->inputs_count.emplace_back(in_target.size());
    task_data_mpi_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_seq.data()));
    task_data_mpi_seq->outputs_count.emplace_back(out_seq.size());

    strakhov_a_char_freq_counter_mpi::CharFreqCounterSeq test_task_seq(task_data_mpi_seq);
    ASSERT_EQ(test_task_seq.Validation(), true);
    test_task_seq.PreProcessing();
    test_task_seq.Run();
    test_task_seq.PostProcessing();

    ASSERT_EQ(out_par[0], out_seq[0]);
  }
}

TEST(strakhov_a_char_freq_counter_mpi, simple_test_1) {
  boost::mpi::communicator world;
  std::vector<char> in_string = {'H', 'e', 'l', 'l', 'o'};
  std::vector<int> out_par(1, 0);
  std::vector<int> out_seq(1, 0);
  std::vector<char> in_target(1, 'H');
  // Parallel

  // Create task_data
  auto task_data_mpi_par = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    task_data_mpi_par->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_string.data()));
    task_data_mpi_par->inputs_count.emplace_back(in_string.size());
    task_data_mpi_par->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_target.data()));
    task_data_mpi_par->inputs_count.emplace_back(in_target.size());
    task_data_mpi_par->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_par.data()));
    task_data_mpi_par->outputs_count.emplace_back(out_par.size());
  }

  // Create Task
  strakhov_a_char_freq_counter_mpi::CharFreqCounterPar test_task_par(task_data_mpi_par);
  ASSERT_EQ(test_task_par.Validation(), true);
  test_task_par.PreProcessing();
  test_task_par.Run();
  test_task_par.PostProcessing();

  // Sequential

  // Create task_data

  if (world.rank() == 0) {
    auto task_data_mpi_seq = std::make_shared<ppc::core::TaskData>();
    task_data_mpi_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_string.data()));
    task_data_mpi_seq->inputs_count.emplace_back(in_string.size());
    task_data_mpi_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_target.data()));
    task_data_mpi_seq->inputs_count.emplace_back(in_target.size());
    task_data_mpi_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_seq.data()));
    task_data_mpi_seq->outputs_count.emplace_back(out_seq.size());

    strakhov_a_char_freq_counter_mpi::CharFreqCounterSeq test_task_seq(task_data_mpi_seq);
    ASSERT_EQ(test_task_seq.Validation(), true);
    test_task_seq.PreProcessing();
    test_task_seq.Run();
    test_task_seq.PostProcessing();

    ASSERT_EQ(out_par[0], out_seq[0]);
  }
}
TEST(strakhov_a_char_freq_counter_mpi, simple_test_2) {
  boost::mpi::communicator world;
  std::vector<char> in_string = {'H', 'e', 'l', 'l', 'o'};
  std::vector<int> out_par(1, 0);
  std::vector<int> out_seq(1, 0);
  std::vector<char> in_target(1, 'h');
  // Parallel

  // Create task_data
  auto task_data_mpi_par = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    task_data_mpi_par->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_string.data()));
    task_data_mpi_par->inputs_count.emplace_back(in_string.size());
    task_data_mpi_par->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_target.data()));
    task_data_mpi_par->inputs_count.emplace_back(in_target.size());
    task_data_mpi_par->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_par.data()));
    task_data_mpi_par->outputs_count.emplace_back(out_par.size());
  }

  // Create Task
  strakhov_a_char_freq_counter_mpi::CharFreqCounterPar test_task_par(task_data_mpi_par);
  ASSERT_EQ(test_task_par.Validation(), true);
  test_task_par.PreProcessing();
  test_task_par.Run();
  test_task_par.PostProcessing();

  // Sequential

  // Create task_data

  if (world.rank() == 0) {
    auto task_data_mpi_seq = std::make_shared<ppc::core::TaskData>();
    task_data_mpi_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_string.data()));
    task_data_mpi_seq->inputs_count.emplace_back(in_string.size());
    task_data_mpi_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_target.data()));
    task_data_mpi_seq->inputs_count.emplace_back(in_target.size());
    task_data_mpi_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_seq.data()));
    task_data_mpi_seq->outputs_count.emplace_back(out_seq.size());

    strakhov_a_char_freq_counter_mpi::CharFreqCounterSeq test_task_seq(task_data_mpi_seq);
    ASSERT_EQ(test_task_seq.Validation(), true);
    test_task_seq.PreProcessing();
    test_task_seq.Run();
    test_task_seq.PostProcessing();

    ASSERT_EQ(out_par[0], out_seq[0]);
  }
}
TEST(strakhov_a_char_freq_counter_mpi, simple_test_3) {
  boost::mpi::communicator world;
  std::vector<char> in_string = {'H', 'e', 'l', 'l', 'o'};
  std::vector<int> out_par(1, 0);
  std::vector<int> out_seq(1, 0);
  std::vector<char> in_target(1, 'l');
  // Parallel

  // Create task_data
  auto task_data_mpi_par = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    task_data_mpi_par->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_string.data()));
    task_data_mpi_par->inputs_count.emplace_back(in_string.size());
    task_data_mpi_par->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_target.data()));
    task_data_mpi_par->inputs_count.emplace_back(in_target.size());
    task_data_mpi_par->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_par.data()));
    task_data_mpi_par->outputs_count.emplace_back(out_par.size());
  }

  // Create Task
  strakhov_a_char_freq_counter_mpi::CharFreqCounterPar test_task_par(task_data_mpi_par);
  ASSERT_EQ(test_task_par.Validation(), true);
  test_task_par.PreProcessing();
  test_task_par.Run();
  test_task_par.PostProcessing();

  // Sequential

  // Create task_data

  if (world.rank() == 0) {
    auto task_data_mpi_seq = std::make_shared<ppc::core::TaskData>();
    task_data_mpi_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_string.data()));
    task_data_mpi_seq->inputs_count.emplace_back(in_string.size());
    task_data_mpi_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_target.data()));
    task_data_mpi_seq->inputs_count.emplace_back(in_target.size());
    task_data_mpi_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_seq.data()));
    task_data_mpi_seq->outputs_count.emplace_back(out_seq.size());

    strakhov_a_char_freq_counter_mpi::CharFreqCounterSeq test_task_seq(task_data_mpi_seq);
    ASSERT_EQ(test_task_seq.Validation(), true);
    test_task_seq.PreProcessing();
    test_task_seq.Run();
    test_task_seq.PostProcessing();

    ASSERT_EQ(out_par[0], out_seq[0]);
  }
}