#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <vector>

#include "core/task/include/task.hpp"
#include "mpi/Konstantinov_I_sum_of_vector_elements/include/ops_mpi.hpp"

namespace konstantinov_i_sum_of_vector_elements_mpi {
namespace {
std::vector<int> GenerateRandVector(int size, int lower_bound = 0, int upper_bound = 50) {
  std::vector<int> result(size);
  for (int i = 0; i < size; i++) {
    result[i] = lower_bound + rand() % (upper_bound - lower_bound + 1);
  }
  return result;
}

std::vector<std::vector<int>> GenerateRandMatrix(int rows, int columns, int lower_bound = 0, int upper_bound = 50) {
  std::vector<std::vector<int>> result(rows);
  for (int i = 0; i < rows; i++) {
    result[i] = konstantinov_i_sum_of_vector_elements_mpi::GenerateRandVector(columns, lower_bound, upper_bound);
  }
  return result;
}
}  // namespace
}  // namespace konstantinov_i_sum_of_vector_elements_mpi

TEST(Konstantinov_I_sum_of_vector_elements_parallel, EmptyInput) {
  boost::mpi::communicator world;
  std::shared_ptr<ppc::core::TaskData> task_data_par = std::make_shared<ppc::core::TaskData>();
  konstantinov_i_sum_of_vector_elements_mpi::SumVecElemParallel test(task_data_par);
  if (world.rank() == 0) {
    ASSERT_FALSE(test.ValidationImpl());
  }
}

TEST(Konstantinov_I_sum_of_vector_elements_parallel, EmptyOutput) {
  boost::mpi::communicator world;
  int rows = 10;
  int columns = 10;
  std::vector<std::vector<int>> input = konstantinov_i_sum_of_vector_elements_mpi::GenerateRandMatrix(rows, columns);
  std::shared_ptr<ppc::core::TaskData> task_data_par = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    task_data_par->inputs_count.emplace_back(rows);
    task_data_par->inputs_count.emplace_back(columns);
    for (long unsigned int i = 0; i < input.size(); i++) {
      task_data_par->inputs.emplace_back(reinterpret_cast<uint8_t *>(input[i].data()));
    }
  }
  konstantinov_i_sum_of_vector_elements_mpi::SumVecElemParallel test(task_data_par);
  if (world.rank() == 0) {
    ASSERT_FALSE(test.ValidationImpl());
  }
}

TEST(Konstantinov_I_sum_of_vector_elements_parallel, Matrix1x1) {
  boost::mpi::communicator world;

  int rows = 1;
  int columns = 1;
  int result = 0;
  std::vector<std::vector<int>> input = konstantinov_i_sum_of_vector_elements_mpi::GenerateRandMatrix(rows, columns);
  std::shared_ptr<ppc::core::TaskData> task_data_par = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    task_data_par->inputs_count.emplace_back(rows);
    task_data_par->inputs_count.emplace_back(columns);
    for (long unsigned int i = 0; i < input.size(); i++) {
      task_data_par->inputs.emplace_back(reinterpret_cast<uint8_t *>(input[i].data()));
    }
    task_data_par->outputs_count.emplace_back(1);
    task_data_par->outputs.emplace_back(reinterpret_cast<uint8_t *>(&result));
  }

  konstantinov_i_sum_of_vector_elements_mpi::SumVecElemParallel test(task_data_par);
  test.ValidationImpl();
  test.PreProcessingImpl();
  test.RunImpl();
  test.PostProcessingImpl();

  if (world.rank() == 0) {
    int respar = result;
    konstantinov_i_sum_of_vector_elements_mpi::SumVecElemSequential testseq(task_data_par);
    testseq.ValidationImpl();
    testseq.PreProcessingImpl();
    testseq.RunImpl();
    testseq.PostProcessingImpl();
    ASSERT_EQ(respar, result);
  }
}

TEST(Konstantinov_I_sum_of_vector_elements_parallel, Matrix5x1) {
  boost::mpi::communicator world;

  int rows = 5;
  int columns = 1;
  int result = 0;
  std::vector<std::vector<int>> input = konstantinov_i_sum_of_vector_elements_mpi::GenerateRandMatrix(rows, columns);
  std::shared_ptr<ppc::core::TaskData> task_data_par = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    task_data_par->inputs_count.emplace_back(rows);
    task_data_par->inputs_count.emplace_back(columns);
    for (long unsigned int i = 0; i < input.size(); i++) {
      task_data_par->inputs.emplace_back(reinterpret_cast<uint8_t *>(input[i].data()));
    }
    task_data_par->outputs_count.emplace_back(1);
    task_data_par->outputs.emplace_back(reinterpret_cast<uint8_t *>(&result));
  }

  konstantinov_i_sum_of_vector_elements_mpi::SumVecElemParallel test(task_data_par);
  test.ValidationImpl();
  test.PreProcessingImpl();
  test.RunImpl();
  test.PostProcessingImpl();

  if (world.rank() == 0) {
    int respar = result;
    konstantinov_i_sum_of_vector_elements_mpi::SumVecElemSequential testseq(task_data_par);
    testseq.ValidationImpl();
    testseq.PreProcessingImpl();
    testseq.RunImpl();
    testseq.PostProcessingImpl();
    ASSERT_EQ(respar, result);
  }
}

TEST(Konstantinov_I_sum_of_vector_elements_parallel, Matrix10x10) {
  boost::mpi::communicator world;

  int rows = 10;
  int columns = 10;
  int result = 0;
  std::vector<std::vector<int>> input = konstantinov_i_sum_of_vector_elements_mpi::GenerateRandMatrix(rows, columns);
  std::shared_ptr<ppc::core::TaskData> task_data_par = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    task_data_par->inputs_count.emplace_back(rows);
    task_data_par->inputs_count.emplace_back(columns);
    for (long unsigned int i = 0; i < input.size(); i++) {
      task_data_par->inputs.emplace_back(reinterpret_cast<uint8_t *>(input[i].data()));
    }
    task_data_par->outputs_count.emplace_back(1);
    task_data_par->outputs.emplace_back(reinterpret_cast<uint8_t *>(&result));
  }

  konstantinov_i_sum_of_vector_elements_mpi::SumVecElemParallel test(task_data_par);
  test.ValidationImpl();
  test.PreProcessingImpl();
  test.RunImpl();
  test.PostProcessingImpl();

  if (world.rank() == 0) {
    int respar = result;
    konstantinov_i_sum_of_vector_elements_mpi::SumVecElemSequential testseq(task_data_par);
    testseq.ValidationImpl();
    testseq.PreProcessingImpl();
    testseq.RunImpl();
    testseq.PostProcessingImpl();
    ASSERT_EQ(respar, result);
  }
}

TEST(Konstantinov_I_sum_of_vector_elements_parallel, Matrix100x100) {
  boost::mpi::communicator world;

  int rows = 100;
  int columns = 100;
  int result = 0;
  std::vector<std::vector<int>> input = konstantinov_i_sum_of_vector_elements_mpi::GenerateRandMatrix(rows, columns);
  std::shared_ptr<ppc::core::TaskData> task_data_par = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    task_data_par->inputs_count.emplace_back(rows);
    task_data_par->inputs_count.emplace_back(columns);
    for (long unsigned int i = 0; i < input.size(); i++) {
      task_data_par->inputs.emplace_back(reinterpret_cast<uint8_t *>(input[i].data()));
    }
    task_data_par->outputs_count.emplace_back(1);
    task_data_par->outputs.emplace_back(reinterpret_cast<uint8_t *>(&result));
  }

  konstantinov_i_sum_of_vector_elements_mpi::SumVecElemParallel test(task_data_par);
  test.ValidationImpl();
  test.PreProcessingImpl();
  test.RunImpl();
  test.PostProcessingImpl();

  if (world.rank() == 0) {
    int respar = result;
    konstantinov_i_sum_of_vector_elements_mpi::SumVecElemSequential testseq(task_data_par);
    testseq.ValidationImpl();
    testseq.PreProcessingImpl();
    testseq.RunImpl();
    testseq.PostProcessingImpl();
    ASSERT_EQ(respar, result);
  }
}

TEST(Konstantinov_I_sum_of_vector_elements_parallel, Matrix100x10) {
  boost::mpi::communicator world;

  int rows = 100;
  int columns = 10;
  int result = 0;
  std::vector<std::vector<int>> input = konstantinov_i_sum_of_vector_elements_mpi::GenerateRandMatrix(rows, columns);
  std::shared_ptr<ppc::core::TaskData> task_data_par = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    task_data_par->inputs_count.emplace_back(rows);
    task_data_par->inputs_count.emplace_back(columns);
    for (long unsigned int i = 0; i < input.size(); i++) {
      task_data_par->inputs.emplace_back(reinterpret_cast<uint8_t *>(input[i].data()));
    }
    task_data_par->outputs_count.emplace_back(1);
    task_data_par->outputs.emplace_back(reinterpret_cast<uint8_t *>(&result));
  }

  konstantinov_i_sum_of_vector_elements_mpi::SumVecElemParallel test(task_data_par);
  test.ValidationImpl();
  test.PreProcessingImpl();
  test.RunImpl();
  test.PostProcessingImpl();

  if (world.rank() == 0) {
    int respar = result;
    konstantinov_i_sum_of_vector_elements_mpi::SumVecElemSequential testseq(task_data_par);
    testseq.ValidationImpl();
    testseq.PreProcessingImpl();
    testseq.RunImpl();
    testseq.PostProcessingImpl();
    ASSERT_EQ(respar, result);
  }
}

TEST(Konstantinov_I_sum_of_vector_elements_parallel, Matrix10x100) {
  boost::mpi::communicator world;

  int rows = 10;
  int columns = 100;
  int result = 0;
  std::vector<std::vector<int>> input = konstantinov_i_sum_of_vector_elements_mpi::GenerateRandMatrix(rows, columns);
  std::shared_ptr<ppc::core::TaskData> task_data_par = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    task_data_par->inputs_count.emplace_back(rows);
    task_data_par->inputs_count.emplace_back(columns);
    for (long unsigned int i = 0; i < input.size(); i++) {
      task_data_par->inputs.emplace_back(reinterpret_cast<uint8_t *>(input[i].data()));
    }
    task_data_par->outputs_count.emplace_back(1);
    task_data_par->outputs.emplace_back(reinterpret_cast<uint8_t *>(&result));
  }

  konstantinov_i_sum_of_vector_elements_mpi::SumVecElemParallel test(task_data_par);
  test.ValidationImpl();
  test.PreProcessingImpl();
  test.RunImpl();
  test.PostProcessingImpl();

  if (world.rank() == 0) {
    int respar = result;
    konstantinov_i_sum_of_vector_elements_mpi::SumVecElemSequential testseq(task_data_par);
    testseq.ValidationImpl();
    testseq.PreProcessingImpl();
    testseq.RunImpl();
    testseq.PostProcessingImpl();
    ASSERT_EQ(respar, result);
  }
}