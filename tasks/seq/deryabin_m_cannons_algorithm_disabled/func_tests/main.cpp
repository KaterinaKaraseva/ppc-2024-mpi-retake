#include <gtest/gtest.h>

#include <cstdint>
#include <memory>
#include <vector>

#include "core/task/include/task.hpp"
#include "seq/deryabin_m_cannons_algorithm/include/ops_seq.hpp"

TEST(deryabin_m_cannons_algorithm_seq, test_simple_matrix) {
  // Create data
  std::vector<double> input_matrix_a{1, 2, 3, 4, 5, 6, 7, 8, 9};
  std::vector<double> input_matrix_b{1, 2, 3, 4, 5, 6, 7, 8, 9};
  std::vector<double> output_matrix_c(9, 0);
  std::vector<double> true_solution{30, 36, 42, 66, 81, 96, 102, 126, 150};

  std::vector<std::vector<double>> in_matrix_a(1, input_matrix_a);
  std::vector<std::vector<double>> in_matrix_b(1, input_matrix_b);
  std::vector<std::vector<double>> out_matrix_c(1, output_matrix_c);

  // Create TaskData
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in_matrix_a.data()));
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in_matrix_b.data()));
  task_data_seq->inputs_count.emplace_back(input_matrix_a.size());
  task_data_seq->inputs_count.emplace_back(input_matrix_b.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out_matrix_c.data()));
  task_data_seq->outputs_count.emplace_back(out_matrix_c.size());

  // Create Task
  deryabin_m_cannons_algorithm_seq::CannonsAlgorithmTaskSequential cannons_algorithm_task_sequential(task_data_seq);
  ASSERT_EQ(cannons_algorithm_task_sequential.Validation(), true);
  cannons_algorithm_task_sequential.PreProcessing();
  cannons_algorithm_task_sequential.Run();
  cannons_algorithm_task_sequential.PostProcessing();
  ASSERT_EQ(true_solution, out_matrix_c[0]);
}

TEST(deryabin_m_cannons_algorithm_seq, test_triangular_matrix) {
  // Create data
  std::vector<double> input_matrix_a{1, 2, 3, 0, 5, 6, 0, 0, 9};
  std::vector<double> input_matrix_b{1, 0, 0, 4, 5, 0, 7, 8, 9};
  std::vector<double> output_matrix_c(9, 0);
  std::vector<double> true_solution{30, 34, 27, 62, 73, 54, 63, 72, 81};

  std::vector<std::vector<double>> in_matrix_a(1, input_matrix_a);
  std::vector<std::vector<double>> in_matrix_b(1, input_matrix_b);
  std::vector<std::vector<double>> out_matrix_c(1, output_matrix_c);

  // Create TaskData
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in_matrix_a.data()));
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in_matrix_b.data()));
  task_data_seq->inputs_count.emplace_back(input_matrix_a.size());
  task_data_seq->inputs_count.emplace_back(input_matrix_b.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out_matrix_c.data()));
  task_data_seq->outputs_count.emplace_back(out_matrix_c.size());

  // Create Task
  deryabin_m_cannons_algorithm_seq::CannonsAlgorithmTaskSequential cannons_algorithm_task_sequential(task_data_seq);
  ASSERT_EQ(cannons_algorithm_task_sequential.Validation(), true);
  cannons_algorithm_task_sequential.PreProcessing();
  cannons_algorithm_task_sequential.Run();
  cannons_algorithm_task_sequential.PostProcessing();
  ASSERT_EQ(true_solution, out_matrix_c[0]);
}

TEST(deryabin_m_cannons_algorithm_seq, test_null_matrix) {
  // Create data
  std::vector<double> input_matrix_a{1, 2, 3, 4, 5, 6, 7, 8, 9};
  std::vector<double> input_matrix_b(9, 0);
  std::vector<double> output_matrix_c(9, 0);

  std::vector<std::vector<double>> in_matrix_a(1, input_matrix_a);
  std::vector<std::vector<double>> in_matrix_b(1, input_matrix_b);
  std::vector<std::vector<double>> out_matrix_c(1, output_matrix_c);

  // Create TaskData
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in_matrix_a.data()));
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in_matrix_b.data()));
  task_data_seq->inputs_count.emplace_back(input_matrix_a.size());
  task_data_seq->inputs_count.emplace_back(input_matrix_b.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out_matrix_c.data()));
  task_data_seq->outputs_count.emplace_back(out_matrix_c.size());

  // Create Task
  deryabin_m_cannons_algorithm_seq::CannonsAlgorithmTaskSequential cannons_algorithm_task_sequential(task_data_seq);
  ASSERT_EQ(cannons_algorithm_task_sequential.Validation(), true);
  cannons_algorithm_task_sequential.PreProcessing();
  cannons_algorithm_task_sequential.Run();
  cannons_algorithm_task_sequential.PostProcessing();
  ASSERT_EQ(in_matrix_b[0], out_matrix_c[0]);
}

TEST(deryabin_m_cannons_algorithm_seq, test_identity_matrix) {
  // Create data
  std::vector<double> input_matrix_a{1, 0, 0, 0, 1, 0, 0, 0, 1};
  std::vector<double> input_matrix_b{1, 2, 3, 4, 5, 6, 7, 8, 9};
  std::vector<double> output_matrix_c(9, 0);

  std::vector<std::vector<double>> in_matrix_a(1, input_matrix_a);
  std::vector<std::vector<double>> in_matrix_b(1, input_matrix_b);
  std::vector<std::vector<double>> out_matrix_c(1, output_matrix_c);

  // Create TaskData
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in_matrix_a.data()));
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in_matrix_b.data()));
  task_data_seq->inputs_count.emplace_back(input_matrix_a.size());
  task_data_seq->inputs_count.emplace_back(input_matrix_b.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out_matrix_c.data()));
  task_data_seq->outputs_count.emplace_back(out_matrix_c.size());

  // Create Task
  deryabin_m_cannons_algorithm_seq::CannonsAlgorithmTaskSequential cannons_algorithm_task_sequential(task_data_seq);
  ASSERT_EQ(cannons_algorithm_task_sequential.Validation(), true);
  cannons_algorithm_task_sequential.PreProcessing();
  cannons_algorithm_task_sequential.Run();
  cannons_algorithm_task_sequential.PostProcessing();
  ASSERT_EQ(in_matrix_b[0], out_matrix_c[0]);
}

TEST(deryabin_m_cannons_algorithm_seq, test_matrices_of_different_dimensions) {
  // Create data
  std::vector<double> input_matrix_a{1, 2, 3, 4, 5, 6, 7, 8, 9};
  std::vector<double> input_matrix_b{1, 2, 3, 4};
  std::vector<double> output_matrix_c(9, 0);

  std::vector<std::vector<double>> in_matrix_a(1, input_matrix_a);
  std::vector<std::vector<double>> in_matrix_b(1, input_matrix_b);
  std::vector<std::vector<double>> out_matrix_c(1, output_matrix_c);

  // Create TaskData
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in_matrix_a.data()));
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in_matrix_b.data()));
  task_data_seq->inputs_count.emplace_back(input_matrix_a.size());
  task_data_seq->inputs_count.emplace_back(input_matrix_b.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out_matrix_c.data()));
  task_data_seq->outputs_count.emplace_back(out_matrix_c.size());

  // Create Task
  deryabin_m_cannons_algorithm_seq::CannonsAlgorithmTaskSequential cannons_algorithm_task_sequential(task_data_seq);
  ASSERT_EQ(cannons_algorithm_task_sequential.Validation(), false);
}

TEST(deryabin_m_cannons_algorithm_seq, test_non_square_matrices) {
  // Create data
  std::vector<double> input_matrix_a{1, 2, 3, 4, 5};
  std::vector<double> input_matrix_b{1, 2, 3, 4, 5};
  std::vector<double> output_matrix_c(5, 0);

  std::vector<std::vector<double>> in_matrix_a(1, input_matrix_a);
  std::vector<std::vector<double>> in_matrix_b(1, input_matrix_b);
  std::vector<std::vector<double>> out_matrix_c(1, output_matrix_c);

  // Create TaskData
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in_matrix_a.data()));
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in_matrix_b.data()));
  task_data_seq->inputs_count.emplace_back(input_matrix_a.size());
  task_data_seq->inputs_count.emplace_back(input_matrix_b.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out_matrix_c.data()));
  task_data_seq->outputs_count.emplace_back(out_matrix_c.size());

  // Create Task
  deryabin_m_cannons_algorithm_seq::CannonsAlgorithmTaskSequential cannons_algorithm_task_sequential(task_data_seq);
  ASSERT_EQ(cannons_algorithm_task_sequential.Validation(), false);
}
