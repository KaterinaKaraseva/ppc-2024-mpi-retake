#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <memory>
#include <random>
#include <vector>

#include "core/task/include/task.hpp"
#include "seq/kavtorev_d_radix_double_sort/include/ops_seq.hpp"

using namespace kavtorev_d_radix_double_sort;

TEST(kavtorev_d_radix_double_sort_seq, SimpleData) {
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();

  int n = 8;
  std::vector<double> input_data = {3.5, -2.1, 0.0, 1.1, -3.3, 2.2, -1.4, 5.6};
  std::vector<double> x_seq(n, 0.0);

  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&n));
  task_data_seq->inputs_count.emplace_back(1);

  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_data.data()));
  task_data_seq->inputs_count.emplace_back(n);

  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(x_seq.data()));
  task_data_seq->outputs_count.emplace_back(n);

  kavtorev_d_radix_double_sort::RadixSortSequential test_task_sequential(task_data_seq);
  ASSERT_TRUE(test_task_sequential.ValidationImpl());
  test_task_sequential.PreProcessingImpl();
  test_task_sequential.RunImpl();
  test_task_sequential.PostProcessingImpl();

  auto* result_seq = reinterpret_cast<double*>(task_data_seq->outputs[0]);
  std::ranges::sort(input_data.begin(), input_data.end());

  for (int i = 0; i < n; ++i) {
    ASSERT_NEAR(input_data[i], result_seq[i], 1e-12);
  }
}

TEST(kavtorev_d_radix_double_sort_seq, ValidationFailureTestSize) {
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();

  int n = 5;
  std::vector<double> input_data = {3.5, -2.1, 0.0};
  std::vector<double> x_seq(n, 0.0);

  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&n));
  task_data_seq->inputs_count.emplace_back(1);

  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_data.data()));
  task_data_seq->inputs_count.emplace_back(3);

  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(x_seq.data()));
  task_data_seq->outputs_count.emplace_back(n);

  kavtorev_d_radix_double_sort::RadixSortSequential test_task_sequential(task_data_seq);
  ASSERT_FALSE(test_task_sequential.ValidationImpl());
}

TEST(kavtorev_d_radix_double_sort_seq, RandomDataSmall) {
  int n = 20;
  std::vector<double> input_data(n);
  std::mt19937 gen(42);
  std::uniform_real_distribution<double> dist(-100.0, 100.0);
  for (int i = 0; i < n; ++i) {
    input_data[i] = dist(gen);
  }

  std::vector<double> x_seq(n, 0.0);

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();

  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&n));
  task_data_seq->inputs_count.emplace_back(1);

  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_data.data()));
  task_data_seq->inputs_count.emplace_back(n);

  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(x_seq.data()));
  task_data_seq->outputs_count.emplace_back(n);

  kavtorev_d_radix_double_sort::RadixSortSequential test_task_sequential(task_data_seq);
  ASSERT_TRUE(test_task_sequential.ValidationImpl());
  test_task_sequential.PreProcessingImpl();
  test_task_sequential.RunImpl();
  test_task_sequential.PostProcessingImpl();

  auto* result_seq = reinterpret_cast<double*>(task_data_seq->outputs[0]);

  std::ranges::sort(input_data.begin(), input_data.end());

  for (int i = 0; i < n; ++i) {
    ASSERT_NEAR(input_data[i], result_seq[i], 1e-12);
  }
}

TEST(kavtorev_d_radix_double_sort_seq, RandomDataLarge) {
  int n = 10000;
  std::vector<double> input_data(n);
  std::mt19937 gen(42);
  std::uniform_real_distribution<double> dist(-100.0, 100.0);
  for (int i = 0; i < n; ++i) {
    input_data[i] = dist(gen);
  }

  std::vector<double> x_seq(n, 0.0);

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();

  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&n));
  task_data_seq->inputs_count.emplace_back(1);

  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_data.data()));
  task_data_seq->inputs_count.emplace_back(n);

  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(x_seq.data()));
  task_data_seq->outputs_count.emplace_back(n);

  kavtorev_d_radix_double_sort::RadixSortSequential test_task_sequential(task_data_seq);
  ASSERT_TRUE(test_task_sequential.ValidationImpl());
  test_task_sequential.PreProcessingImpl();
  test_task_sequential.RunImpl();
  test_task_sequential.PostProcessingImpl();

  auto* result_seq = reinterpret_cast<double*>(task_data_seq->outputs[0]);

  std::ranges::sort(input_data.begin(), input_data.end());

  for (int i = 0; i < n; ++i) {
    ASSERT_NEAR(input_data[i], result_seq[i], 1e-12);
  }
}

TEST(kavtorev_d_radix_double_sort_seq, AlreadySortedData) {
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();

  int n = 10;
  std::vector<double> input_data = {-5.4, -3.3, -1.0, 0.0, 0.1, 1.2, 2.3, 2.4, 3.5, 10.0};
  std::vector<double> x_seq(n, 0.0);

  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&n));
  task_data_seq->inputs_count.emplace_back(1);

  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_data.data()));
  task_data_seq->inputs_count.emplace_back(n);

  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(x_seq.data()));
  task_data_seq->outputs_count.emplace_back(n);

  kavtorev_d_radix_double_sort::RadixSortSequential test_task_sequential(task_data_seq);
  ASSERT_TRUE(test_task_sequential.ValidationImpl());
  test_task_sequential.PreProcessingImpl();
  test_task_sequential.RunImpl();
  test_task_sequential.PostProcessingImpl();

  auto* result_seq = reinterpret_cast<double*>(task_data_seq->outputs[0]);
  std::ranges::sort(input_data.begin(), input_data.end());

  for (int i = 0; i < n; ++i) {
    ASSERT_NEAR(input_data[i], result_seq[i], 1e-12);
  }
}

TEST(kavtorev_d_radix_double_sort_seq, ReverseSortedData) {
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();

  int n = 10;
  std::vector<double> input_data = {10.0, 3.5, 2.4, 2.3, 1.2, 0.1, 0.0, -1.0, -3.3, -5.4};
  std::vector<double> x_seq(n, 0.0);

  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&n));
  task_data_seq->inputs_count.emplace_back(1);

  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_data.data()));
  task_data_seq->inputs_count.emplace_back(n);

  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(x_seq.data()));
  task_data_seq->outputs_count.emplace_back(n);

  kavtorev_d_radix_double_sort::RadixSortSequential test_task_sequential(task_data_seq);
  ASSERT_TRUE(test_task_sequential.ValidationImpl());
  test_task_sequential.PreProcessingImpl();
  test_task_sequential.RunImpl();
  test_task_sequential.PostProcessingImpl();

  auto* result_seq = reinterpret_cast<double*>(task_data_seq->outputs[0]);
  std::ranges::sort(input_data.begin(), input_data.end());

  for (int i = 0; i < n; ++i) {
    ASSERT_NEAR(input_data[i], result_seq[i], 1e-12);
  }
}