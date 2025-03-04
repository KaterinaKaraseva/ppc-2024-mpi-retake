#include <gtest/gtest.h>

#include <cstdint>
#include <memory>
#include <vector>

#include "core/task/include/task.hpp"
#include "seq/shishkarev_a_gaussian_method_horizontal_strip_pattern/include/ops_seq.hpp"

TEST(shishkarev_a_gaussian_method_horizontal_strip_pattern_seq, test_for_empty_matrix) {
  const int cols = 0;
  const int rows = 0;

  std::vector<double> matrix;
  std::vector<double> res;

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrix.data()));
  task_data_seq->inputs_count = {static_cast<unsigned int>(matrix.size()), cols, rows};
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(res.data()));
  task_data_seq->outputs_count.emplace_back(res.size());

  auto task =
      std::make_shared<shishkarev_a_gaussian_method_horizontal_strip_pattern_seq::MPIGaussHorizontalSequential<double>>(
          task_data_seq);
  ASSERT_FALSE(task->ValidationImpl());
}

TEST(shishkarev_a_gaussian_method_horizontal_strip_pattern_seq, test_for_matrix_with_one_element) {
  const int cols = 1;
  const int rows = 1;

  std::vector<double> matrix = {1};
  std::vector<double> res;

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrix.data()));
  task_data_seq->inputs_count = {static_cast<unsigned int>(matrix.size()), cols, rows};
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(res.data()));
  task_data_seq->outputs_count.emplace_back(res.size());

  auto task =
      std::make_shared<shishkarev_a_gaussian_method_horizontal_strip_pattern_seq::MPIGaussHorizontalSequential<double>>(
          task_data_seq);
  ASSERT_FALSE(task->ValidationImpl());
}

TEST(shishkarev_a_gaussian_method_horizontal_strip_pattern_seq, test_not_square_matrix) {
  const int cols = 5;
  const int rows = 2;

  std::vector<double> matrix = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  std::vector<double> res(cols - 1, 0);

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrix.data()));
  task_data_seq->inputs_count = {static_cast<unsigned int>(matrix.size()), cols, rows};
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(res.data()));
  task_data_seq->outputs_count.emplace_back(res.size());

  auto task =
      std::make_shared<shishkarev_a_gaussian_method_horizontal_strip_pattern_seq::MPIGaussHorizontalSequential<double>>(
          task_data_seq);
}
