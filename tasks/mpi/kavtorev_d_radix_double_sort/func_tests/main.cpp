#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <boost/serialization/vector.hpp>  //NOLINT
#include <cstdint>
#include <memory>
#include <random>
#include <vector>

#include "core/task/include/task.hpp"
#include "mpi/kavtorev_d_radix_double_sort/include/ops_mpi.hpp"

namespace mpi = boost::mpi;
using namespace kavtorev_d_radix_double_sort;

TEST(kavtorev_d_radix_double_sort_mpi, SimpleData) {
  mpi::environment env;
  mpi::communicator world;

  auto task_data_mpi = std::make_shared<ppc::core::TaskData>();
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();

  int n = 8;
  std::vector<double> input_data = {3.5, -2.1, 0.0, 1.1, -3.3, 2.2, -1.4, 5.6};
  std::vector<double> x_par(n, 0.0);
  std::vector<double> x_seq(n, 0.0);

  if (world.rank() == 0) {
    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t*>(&n));
    task_data_mpi->inputs_count.emplace_back(1);

    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_data.data()));
    task_data_mpi->inputs_count.emplace_back(n);

    task_data_mpi->outputs.emplace_back(reinterpret_cast<uint8_t*>(x_par.data()));
    task_data_mpi->outputs_count.emplace_back(n);

    task_data_seq->inputs = task_data_mpi->inputs;
    task_data_seq->inputs_count = task_data_mpi->inputs_count;

    task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(x_seq.data()));
    task_data_seq->outputs_count.emplace_back(n);
  }

  RadixSortParallel test_task_parallel(task_data_mpi);
  ASSERT_TRUE(test_task_parallel.ValidationImpl());
  test_task_parallel.PreProcessingImpl();
  test_task_parallel.RunImpl();
  test_task_parallel.PostProcessingImpl();

  if (world.rank() == 0) {
    kavtorev_d_radix_double_sort::RadixSortSequential test_task_sequential(task_data_seq);
    ASSERT_TRUE(test_task_sequential.ValidationImpl());
    test_task_sequential.PreProcessingImpl();
    test_task_sequential.RunImpl();
    test_task_sequential.PostProcessingImpl();

    auto* result_par = reinterpret_cast<double*>(task_data_mpi->outputs[0]);
    auto* result_seq = reinterpret_cast<double*>(task_data_seq->outputs[0]);

    for (int i = 0; i < n; ++i) {
      ASSERT_NEAR(result_par[i], result_seq[i], 1e-12);
    }
  }
}

TEST(kavtorev_d_radix_double_sort_mpi, ValidationFailureTestSize) {
  mpi::environment env;
  mpi::communicator world;

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();

  int n = 5;
  std::vector<double> input_data = {3.5, -2.1, 0.0};
  std::vector<double> x_seq(n, 0.0);

  if (world.rank() == 0) {
    task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&n));
    task_data_seq->inputs_count.emplace_back(1);

    task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_data.data()));
    task_data_seq->inputs_count.emplace_back(3);

    task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(x_seq.data()));
    task_data_seq->outputs_count.emplace_back(n);

    kavtorev_d_radix_double_sort::RadixSortSequential test_task_sequential(task_data_seq);
    ASSERT_FALSE(test_task_sequential.ValidationImpl());
  }
}

TEST(kavtorev_d_radix_double_sort_mpi, RandomDataSmall) {
  mpi::environment env;
  mpi::communicator world;

  int n = 20;
  std::vector<double> input_data(n);
  if (world.rank() == 0) {
    std::mt19937 gen(42);
    std::uniform_real_distribution<double> dist(-100.0, 100.0);
    for (int i = 0; i < n; ++i) {
      input_data[i] = dist(gen);
    }
  }

  std::vector<double> x_par(n, 0.0);
  std::vector<double> x_seq(n, 0.0);

  auto task_data_mpi = std::make_shared<ppc::core::TaskData>();
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t*>(&n));
    task_data_mpi->inputs_count.emplace_back(1);

    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_data.data()));
    task_data_mpi->inputs_count.emplace_back(n);

    task_data_mpi->outputs.emplace_back(reinterpret_cast<uint8_t*>(x_par.data()));
    task_data_mpi->outputs_count.emplace_back(n);

    task_data_seq->inputs = task_data_mpi->inputs;
    task_data_seq->inputs_count = task_data_mpi->inputs_count;

    task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(x_seq.data()));
    task_data_seq->outputs_count.emplace_back(n);
  }
  kavtorev_d_radix_double_sort::RadixSortParallel test_task_parallel(task_data_mpi);
  ASSERT_TRUE(test_task_parallel.ValidationImpl());
  test_task_parallel.PreProcessingImpl();
  test_task_parallel.RunImpl();
  test_task_parallel.PostProcessingImpl();

  if (world.rank() == 0) {
    kavtorev_d_radix_double_sort::RadixSortSequential test_task_sequential(task_data_seq);
    ASSERT_TRUE(test_task_sequential.ValidationImpl());
    test_task_sequential.PreProcessingImpl();
    test_task_sequential.RunImpl();
    test_task_sequential.PostProcessingImpl();

    auto* result_par = reinterpret_cast<double*>(task_data_mpi->outputs[0]);
    auto* result_seq = reinterpret_cast<double*>(task_data_seq->outputs[0]);

    for (int i = 0; i < n; ++i) {
      ASSERT_NEAR(result_par[i], result_seq[i], 1e-12);
    }
  }
}

TEST(kavtorev_d_radix_double_sort_mpi, RandomDataLarge) {
  mpi::environment env;
  mpi::communicator world;

  int n = 10000;
  std::vector<double> input_data;
  if (world.rank() == 0) {
    input_data.resize(n);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dist(-1e6, 1e6);
    for (int i = 0; i < n; ++i) {
      input_data[i] = dist(gen);
    }
  }

  std::vector<double> x_par(n, 0.0);
  std::vector<double> x_seq(n, 0.0);

  auto task_data_mpi = std::make_shared<ppc::core::TaskData>();
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t*>(&n));
    task_data_mpi->inputs_count.emplace_back(1);

    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_data.data()));
    task_data_mpi->inputs_count.emplace_back(n);

    task_data_mpi->outputs.emplace_back(reinterpret_cast<uint8_t*>(x_par.data()));
    task_data_mpi->outputs_count.emplace_back(n);

    task_data_seq->inputs = task_data_mpi->inputs;
    task_data_seq->inputs_count = task_data_mpi->inputs_count;

    task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(x_seq.data()));
    task_data_seq->outputs_count.emplace_back(n);
  }

  RadixSortParallel test_task_parallel(task_data_mpi);
  ASSERT_TRUE(test_task_parallel.ValidationImpl());
  test_task_parallel.PreProcessingImpl();
  test_task_parallel.RunImpl();
  test_task_parallel.PostProcessingImpl();

  if (world.rank() == 0) {
    kavtorev_d_radix_double_sort::RadixSortSequential test_task_sequential(task_data_seq);
    ASSERT_TRUE(test_task_sequential.ValidationImpl());
    test_task_sequential.PreProcessingImpl();
    test_task_sequential.RunImpl();
    test_task_sequential.PostProcessingImpl();

    auto* result_par = reinterpret_cast<double*>(task_data_mpi->outputs[0]);
    auto* result_seq = reinterpret_cast<double*>(task_data_seq->outputs[0]);

    for (int i = 0; i < n; ++i) {
      ASSERT_NEAR(result_par[i], result_seq[i], 1e-12);
    }
  }
}

TEST(kavtorev_d_radix_double_sort_mpi, AlreadySortedData) {
  mpi::environment env;
  mpi::communicator world;

  int n = 10;
  std::vector<double> input_data;
  if (world.rank() == 0) {
    input_data = {-5.4, -3.3, -1.0, 0.0, 0.1, 1.2, 2.3, 2.4, 3.5, 10.0};
  }

  std::vector<double> x_par(n, 0.0);
  std::vector<double> x_seq(n, 0.0);

  auto task_data_mpi = std::make_shared<ppc::core::TaskData>();
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t*>(&n));
    task_data_mpi->inputs_count.emplace_back(1);

    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_data.data()));
    task_data_mpi->inputs_count.emplace_back(n);

    task_data_mpi->outputs.emplace_back(reinterpret_cast<uint8_t*>(x_par.data()));
    task_data_mpi->outputs_count.emplace_back(n);

    task_data_seq->inputs = task_data_mpi->inputs;
    task_data_seq->inputs_count = task_data_mpi->inputs_count;

    task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(x_seq.data()));
    task_data_seq->outputs_count.emplace_back(n);
  }

  RadixSortParallel test_task_parallel(task_data_mpi);
  ASSERT_TRUE(test_task_parallel.ValidationImpl());
  test_task_parallel.PreProcessingImpl();
  test_task_parallel.RunImpl();
  test_task_parallel.PostProcessingImpl();

  if (world.rank() == 0) {
    kavtorev_d_radix_double_sort::RadixSortSequential test_task_sequential(task_data_seq);
    ASSERT_TRUE(test_task_sequential.ValidationImpl());
    test_task_sequential.PreProcessingImpl();
    test_task_sequential.RunImpl();
    test_task_sequential.PostProcessingImpl();

    auto* result_par = reinterpret_cast<double*>(task_data_mpi->outputs[0]);
    auto* result_seq = reinterpret_cast<double*>(task_data_seq->outputs[0]);

    for (int i = 0; i < n; ++i) {
      ASSERT_NEAR(result_par[i], result_seq[i], 1e-12);
    }
  }
}

TEST(kavtorev_d_radix_double_sort_mpi, ReverseSortedData) {
  mpi::environment env;
  mpi::communicator world;

  int n = 10;
  std::vector<double> input_data;
  if (world.rank() == 0) {
    input_data = {10.0, 3.5, 2.4, 2.3, 1.2, 0.1, 0.0, -1.0, -3.3, -5.4};
  }

  std::vector<double> x_par(n, 0.0);
  std::vector<double> x_seq(n, 0.0);

  auto task_data_mpi = std::make_shared<ppc::core::TaskData>();
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t*>(&n));
    task_data_mpi->inputs_count.emplace_back(1);

    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_data.data()));
    task_data_mpi->inputs_count.emplace_back(n);

    task_data_mpi->outputs.emplace_back(reinterpret_cast<uint8_t*>(x_par.data()));
    task_data_mpi->outputs_count.emplace_back(n);

    task_data_seq->inputs = task_data_mpi->inputs;
    task_data_seq->inputs_count = task_data_mpi->inputs_count;

    task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(x_seq.data()));
    task_data_seq->outputs_count.emplace_back(n);
  }

  RadixSortParallel test_task_parallel(task_data_mpi);
  ASSERT_TRUE(test_task_parallel.ValidationImpl());
  test_task_parallel.PreProcessingImpl();
  test_task_parallel.RunImpl();
  test_task_parallel.PostProcessingImpl();

  if (world.rank() == 0) {
    kavtorev_d_radix_double_sort::RadixSortSequential test_task_sequential(task_data_seq);
    ASSERT_TRUE(test_task_sequential.ValidationImpl());
    test_task_sequential.PreProcessingImpl();
    test_task_sequential.RunImpl();
    test_task_sequential.PostProcessingImpl();

    auto* result_par = reinterpret_cast<double*>(task_data_mpi->outputs[0]);
    auto* result_seq = reinterpret_cast<double*>(task_data_seq->outputs[0]);

    for (int i = 0; i < n; ++i) {
      ASSERT_NEAR(result_par[i], result_seq[i], 1e-12);
    }
  }
}