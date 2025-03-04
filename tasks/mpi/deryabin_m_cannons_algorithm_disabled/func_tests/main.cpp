#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <cstdint>
#include <memory>
#include <random>
#include <vector>

#include "core/task/include/task.hpp"
#include "mpi/deryabin_m_cannons_algorithm/include/ops_mpi.hpp"

TEST(deryabin_m_cannons_algorithm_mpi, test_simple_matrix) {
  boost::mpi::communicator world;
  std::vector<double> input_matrix_a{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
  std::vector<double> input_matrix_b{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
  std::vector<double> output_matrix_c(16, 0);
  std::vector<std::vector<double>> out_matrix_c(1, output_matrix_c);
  std::vector<double> true_solution{90, 100, 110, 120, 202, 228, 254, 280, 314, 356, 398, 440, 426, 484, 542, 600};
  std::shared_ptr<ppc::core::TaskData> task_data_mpi = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_matrix_a.data()));
    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_matrix_b.data()));
    task_data_mpi->inputs_count.emplace_back(input_matrix_a.size());
    task_data_mpi->inputs_count.emplace_back(input_matrix_b.size());
    task_data_mpi->outputs.emplace_back(reinterpret_cast<uint8_t*>(out_matrix_c.data()));
    task_data_mpi->outputs_count.emplace_back(out_matrix_c.size());
  }
  deryabin_m_cannons_algorithm_mpi::CannonsAlgorithmMPITaskParallel test_mpi_task_parallel(task_data_mpi);
  ASSERT_EQ(test_mpi_task_parallel.Validation(), true);
  test_mpi_task_parallel.PreProcessing();
  test_mpi_task_parallel.Run();
  test_mpi_task_parallel.PostProcessing();
  if (world.rank() == 0) {
    ASSERT_EQ(true_solution, out_matrix_c[0]);
  }
}

TEST(deryabin_m_cannons_algorithm_mpi, test_random_matrix) {
  boost::mpi::communicator world;
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> distribution(-100, 100);
  std::vector<double> input_matrix_a(16, distribution(gen));
  std::vector<double> input_matrix_b(16, distribution(gen));
  std::vector<double> output_matrix_c(16, 0);
  std::vector<std::vector<double>> out_matrix_c(1, output_matrix_c);

  std::shared_ptr<ppc::core::TaskData> task_data_mpi = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_matrix_a.data()));
    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_matrix_b.data()));
    task_data_mpi->inputs_count.emplace_back(input_matrix_a.size());
    task_data_mpi->inputs_count.emplace_back(input_matrix_b.size());
    task_data_mpi->outputs.emplace_back(reinterpret_cast<uint8_t*>(out_matrix_c.data()));
    task_data_mpi->outputs_count.emplace_back(out_matrix_c.size());
  }

  deryabin_m_cannons_algorithm_mpi::CannonsAlgorithmMPITaskParallel test_mpi_task_parallel(task_data_mpi);
  ASSERT_EQ(test_mpi_task_parallel.Validation(), true);
  test_mpi_task_parallel.PreProcessing();
  test_mpi_task_parallel.Run();
  test_mpi_task_parallel.PostProcessing();

  if (world.rank() == 0) {
    std::vector<std::vector<double>> reference_out_matrix_c(1, output_matrix_c);

    std::shared_ptr<ppc::core::TaskData> task_data_seq = std::make_shared<ppc::core::TaskData>();
    task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_matrix_a.data()));
    task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_matrix_b.data()));
    task_data_seq->inputs_count.emplace_back(input_matrix_a.size());
    task_data_seq->inputs_count.emplace_back(input_matrix_b.size());
    task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_out_matrix_c.data()));
    task_data_seq->outputs_count.emplace_back(out_matrix_c.size());

    deryabin_m_cannons_algorithm_mpi::CannonsAlgorithmMPITaskSequential test_mpi_task_sequential(task_data_seq);
    ASSERT_EQ(test_mpi_task_sequential.Validation(), true);
    test_mpi_task_sequential.PreProcessing();
    test_mpi_task_sequential.Run();
    test_mpi_task_sequential.PostProcessing();

    ASSERT_EQ(reference_out_matrix_c[0], out_matrix_c[0]);
  }
}

TEST(deryabin_m_cannons_algorithm_mpi, test_gigantic_random_matrix) {
  boost::mpi::communicator world;
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> distribution(-1000, 1000);
  std::vector<double> input_matrix_a(1600, distribution(gen));
  std::vector<double> input_matrix_b(1600, distribution(gen));
  std::vector<double> output_matrix_c(1600, 0);
  std::vector<std::vector<double>> out_matrix_c(1, output_matrix_c);

  std::shared_ptr<ppc::core::TaskData> task_data_mpi = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_matrix_a.data()));
    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_matrix_b.data()));
    task_data_mpi->inputs_count.emplace_back(input_matrix_a.size());
    task_data_mpi->inputs_count.emplace_back(input_matrix_b.size());
    task_data_mpi->outputs.emplace_back(reinterpret_cast<uint8_t*>(out_matrix_c.data()));
    task_data_mpi->outputs_count.emplace_back(out_matrix_c.size());
  }

  deryabin_m_cannons_algorithm_mpi::CannonsAlgorithmMPITaskParallel test_mpi_task_parallel(task_data_mpi);
  ASSERT_EQ(test_mpi_task_parallel.Validation(), true);
  test_mpi_task_parallel.PreProcessing();
  test_mpi_task_parallel.Run();
  test_mpi_task_parallel.PostProcessing();

  if (world.rank() == 0) {
    std::vector<std::vector<double>> reference_out_matrix_c(1, output_matrix_c);

    std::shared_ptr<ppc::core::TaskData> task_data_seq = std::make_shared<ppc::core::TaskData>();
    task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_matrix_a.data()));
    task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_matrix_b.data()));
    task_data_seq->inputs_count.emplace_back(input_matrix_a.size());
    task_data_seq->inputs_count.emplace_back(input_matrix_b.size());
    task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_out_matrix_c.data()));
    task_data_seq->outputs_count.emplace_back(out_matrix_c.size());

    deryabin_m_cannons_algorithm_mpi::CannonsAlgorithmMPITaskSequential test_mpi_task_sequential(task_data_seq);
    ASSERT_EQ(test_mpi_task_sequential.Validation(), true);
    test_mpi_task_sequential.PreProcessing();
    test_mpi_task_sequential.Run();
    test_mpi_task_sequential.PostProcessing();

    ASSERT_EQ(reference_out_matrix_c[0], out_matrix_c[0]);
  }
}

TEST(deryabin_m_cannons_algorithm_mpi, test_matrices_of_different_dimensions) {
  boost::mpi::communicator world;
  std::vector<double> input_matrix_a{1, 2, 3, 4, 5, 6, 7, 8, 9};
  std::vector<double> input_matrix_b{1, 2, 3, 4};

  std::shared_ptr<ppc::core::TaskData> task_data_mpi = std::make_shared<ppc::core::TaskData>();
  deryabin_m_cannons_algorithm_mpi::CannonsAlgorithmMPITaskParallel test_mpi_task_parallel(task_data_mpi);
  if (world.rank() == 0) {
    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_matrix_a.data()));
    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_matrix_b.data()));
    task_data_mpi->inputs_count.emplace_back(input_matrix_a.size());
    task_data_mpi->inputs_count.emplace_back(input_matrix_b.size());
    ASSERT_EQ(test_mpi_task_parallel.Validation(), false);
  }
  if (world.rank() == 0) {
    std::shared_ptr<ppc::core::TaskData> task_data_seq = std::make_shared<ppc::core::TaskData>();
    task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_matrix_a.data()));
    task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_matrix_b.data()));
    task_data_seq->inputs_count.emplace_back(input_matrix_a.size());
    task_data_seq->inputs_count.emplace_back(input_matrix_b.size());

    deryabin_m_cannons_algorithm_mpi::CannonsAlgorithmMPITaskSequential test_mpi_task_sequential(task_data_seq);
    ASSERT_EQ(test_mpi_task_sequential.Validation(), false);
  }
}

TEST(deryabin_m_cannons_algorithm_mpi, test_non_square_matrices) {
  boost::mpi::communicator world;
  std::vector<double> input_matrix_a{1, 2, 3, 4, 5, 6};
  std::vector<double> input_matrix_b{1, 2, 3, 4, 5, 6};

  std::shared_ptr<ppc::core::TaskData> task_data_mpi = std::make_shared<ppc::core::TaskData>();
  deryabin_m_cannons_algorithm_mpi::CannonsAlgorithmMPITaskParallel test_mpi_task_parallel(task_data_mpi);
  if (world.rank() == 0) {
    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_matrix_a.data()));
    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_matrix_b.data()));
    task_data_mpi->inputs_count.emplace_back(input_matrix_a.size());
    task_data_mpi->inputs_count.emplace_back(input_matrix_b.size());
    ASSERT_EQ(test_mpi_task_parallel.Validation(), false);
  }
  if (world.rank() == 0) {
    std::shared_ptr<ppc::core::TaskData> task_data_seq = std::make_shared<ppc::core::TaskData>();
    task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_matrix_a.data()));
    task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_matrix_b.data()));
    task_data_seq->inputs_count.emplace_back(input_matrix_a.size());
    task_data_seq->inputs_count.emplace_back(input_matrix_b.size());

    deryabin_m_cannons_algorithm_mpi::CannonsAlgorithmMPITaskSequential test_mpi_task_sequential(task_data_seq);
    ASSERT_EQ(test_mpi_task_sequential.Validation(), false);
  }
}

TEST(deryabin_m_cannons_algorithm_mpi, test_small_size_matrix) {
  boost::mpi::communicator world;
  std::vector<double> input_matrix_a{1, 2, 3, 4};
  std::vector<double> input_matrix_b{4, 3, 2, 1};
  std::vector<double> output_matrix_c(4, 0);
  std::vector<std::vector<double>> out_matrix_c(1, output_matrix_c);
  std::vector<double> true_solution{8, 5, 20, 13};
  std::shared_ptr<ppc::core::TaskData> task_data_mpi = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_matrix_a.data()));
    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_matrix_b.data()));
    task_data_mpi->inputs_count.emplace_back(input_matrix_a.size());
    task_data_mpi->inputs_count.emplace_back(input_matrix_b.size());
    task_data_mpi->outputs.emplace_back(reinterpret_cast<uint8_t*>(out_matrix_c.data()));
    task_data_mpi->outputs_count.emplace_back(out_matrix_c.size());
  }
  deryabin_m_cannons_algorithm_mpi::CannonsAlgorithmMPITaskParallel test_mpi_task_parallel(task_data_mpi);
  ASSERT_EQ(test_mpi_task_parallel.Validation(), true);
  test_mpi_task_parallel.PreProcessing();
  test_mpi_task_parallel.Run();
  test_mpi_task_parallel.PostProcessing();
  if (world.rank() == 0) {
    ASSERT_EQ(true_solution, out_matrix_c[0]);
  }
}

TEST(deryabin_m_cannons_algorithm_mpi, test_empty_matrix) {
  boost::mpi::communicator world;
  std::vector<double> input_matrix_a;
  std::vector<double> input_matrix_b;
  std::vector<double> output_matrix_c;
  std::vector<std::vector<double>> out_matrix_c(1, output_matrix_c);
  std::vector<double> true_solution;
  std::shared_ptr<ppc::core::TaskData> task_data_mpi = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_matrix_a.data()));
    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_matrix_b.data()));
    task_data_mpi->inputs_count.emplace_back(input_matrix_a.size());
    task_data_mpi->inputs_count.emplace_back(input_matrix_b.size());
    task_data_mpi->outputs.emplace_back(reinterpret_cast<uint8_t*>(out_matrix_c.data()));
    task_data_mpi->outputs_count.emplace_back(out_matrix_c.size());
  }
  deryabin_m_cannons_algorithm_mpi::CannonsAlgorithmMPITaskParallel test_mpi_task_parallel(task_data_mpi);
  ASSERT_EQ(test_mpi_task_parallel.Validation(), true);
  test_mpi_task_parallel.PreProcessing();
  test_mpi_task_parallel.Run();
  test_mpi_task_parallel.PostProcessing();
  if (world.rank() == 0) {
    ASSERT_EQ(true_solution, out_matrix_c[0]);
  }
}

TEST(deryabin_m_cannons_algorithm_mpi, test_square_and_non_square_matrices) {
  boost::mpi::communicator world;
  std::vector<double> input_matrix_a{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
  std::vector<double> input_matrix_b{1, 2, 3, 4, 5, 6};

  std::shared_ptr<ppc::core::TaskData> task_data_mpi = std::make_shared<ppc::core::TaskData>();
  deryabin_m_cannons_algorithm_mpi::CannonsAlgorithmMPITaskParallel test_mpi_task_parallel(task_data_mpi);
  if (world.rank() == 0) {
    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_matrix_a.data()));
    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_matrix_b.data()));
    task_data_mpi->inputs_count.emplace_back(input_matrix_a.size());
    task_data_mpi->inputs_count.emplace_back(input_matrix_b.size());
    ASSERT_EQ(test_mpi_task_parallel.Validation(), false);
  }
  if (world.rank() == 0) {
    std::shared_ptr<ppc::core::TaskData> task_data_seq = std::make_shared<ppc::core::TaskData>();
    task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_matrix_a.data()));
    task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_matrix_b.data()));
    task_data_seq->inputs_count.emplace_back(input_matrix_a.size());
    task_data_seq->inputs_count.emplace_back(input_matrix_b.size());

    deryabin_m_cannons_algorithm_mpi::CannonsAlgorithmMPITaskSequential test_mpi_task_sequential(task_data_seq);
    ASSERT_EQ(test_mpi_task_sequential.Validation(), false);
  }
}
