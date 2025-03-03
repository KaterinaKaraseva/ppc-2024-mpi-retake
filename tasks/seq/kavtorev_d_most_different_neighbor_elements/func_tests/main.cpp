#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <memory>
#include <random>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"
#include "seq/kavtorev_d_most_different_neighbor_elements/include/ops_seq.hpp"

namespace kavtorev_d_most_different_neighbor_elements_seq {
namespace {
std::vector<int> Generator(int sz) {
  std::random_device dev;
  std::mt19937 gen(dev());

  std::vector<int> ans(sz);
  for (int i = 0; i < sz; ++i) {
    ans[i] = static_cast<int>(gen() % 1000);
    int x = static_cast<int>(gen() % 2);
    if (x == 0) {
      ans[i] *= -1;
    }
  }

  return ans;
}
}  // namespace
}  // namespace kavtorev_d_most_different_neighbor_elements_seq

TEST(kavtorev_d_most_different_neighbor_elements_seq, LargePositiveNumbers_ReturnsCorrectPair) {
  std::vector<int> in = {1000, 2000, 3000, 4000, 5000};
  std::vector<std::pair<int, int>> out(1);
  std::pair<int, int> ans = {1000, 2000};

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  kavtorev_d_most_different_neighbor_elements_seq::MostDifferentNeighborElementsSeq test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.ValidationImpl(), true);
  test_task_sequential.PreProcessingImpl();
  test_task_sequential.RunImpl();
  test_task_sequential.PostProcessingImpl();
  ASSERT_EQ(ans, out[0]);
}

TEST(kavtorev_d_most_different_neighbor_elements_seq, MixedPositiveAndNegativeNumbers_ReturnsCorrectPair) {
  std::vector<int> in = {-10, 20, -30, 40, -50};
  std::vector<std::pair<int, int>> out(1);
  std::pair<int, int> ans = {-50, 40};

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  kavtorev_d_most_different_neighbor_elements_seq::MostDifferentNeighborElementsSeq test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.ValidationImpl(), true);
  test_task_sequential.PreProcessingImpl();
  test_task_sequential.RunImpl();
  test_task_sequential.PostProcessingImpl();
  ASSERT_EQ(ans, out[0]);
}

TEST(kavtorev_d_most_different_neighbor_elements_seq, AlternatingPositiveAndNegativeNumbers_ReturnsCorrectPair) {
  std::vector<int> in = {1, -1, 2, -2, 3, -3, 4, -4, 5, -5};
  std::vector<std::pair<int, int>> out(1);
  std::pair<int, int> ans = {-5, 5};

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  kavtorev_d_most_different_neighbor_elements_seq::MostDifferentNeighborElementsSeq test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.ValidationImpl(), true);
  test_task_sequential.PreProcessingImpl();
  test_task_sequential.RunImpl();
  test_task_sequential.PostProcessingImpl();
  ASSERT_EQ(ans, out[0]);
}

TEST(kavtorev_d_most_different_neighbor_elements_seq, RepeatedNumbers_ReturnsCorrectPair) {
  std::vector<int> in = {5, 5, 5, 5, 5};
  std::vector<std::pair<int, int>> out(1);
  std::pair<int, int> ans = {5, 5};

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  kavtorev_d_most_different_neighbor_elements_seq::MostDifferentNeighborElementsSeq test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.ValidationImpl(), true);
  test_task_sequential.PreProcessingImpl();
  test_task_sequential.RunImpl();
  test_task_sequential.PostProcessingImpl();
  ASSERT_EQ(ans, out[0]);
}

TEST(kavtorev_d_most_different_neighbor_elements_seq, LargeRangeNumbers_ReturnsCorrectPair) {
  std::vector<int> in = {-1000000, 1000000};
  std::vector<std::pair<int, int>> out(1);
  std::pair<int, int> ans = {-1000000, 1000000};

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  kavtorev_d_most_different_neighbor_elements_seq::MostDifferentNeighborElementsSeq test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.ValidationImpl(), true);
  test_task_sequential.PreProcessingImpl();
  test_task_sequential.RunImpl();
  test_task_sequential.PostProcessingImpl();
  ASSERT_EQ(ans, out[0]);
}

TEST(kavtorev_d_most_different_neighbor_elements_seq, SmallRangeNumbers_ReturnsCorrectPair) {
  std::vector<int> in = {1, 2, 3, 4, 5};
  std::vector<std::pair<int, int>> out(1);
  std::pair<int, int> ans = {1, 2};

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  kavtorev_d_most_different_neighbor_elements_seq::MostDifferentNeighborElementsSeq test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.ValidationImpl(), true);
  test_task_sequential.PreProcessingImpl();
  test_task_sequential.RunImpl();
  test_task_sequential.PostProcessingImpl();
  ASSERT_EQ(ans, out[0]);
}

TEST(kavtorev_d_most_different_neighbor_elements_seq, EmptyInput_ReturnsFalse) {
  std::vector<int> in = {};
  std::vector<std::vector<int>> out(1);

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  kavtorev_d_most_different_neighbor_elements_seq::MostDifferentNeighborElementsSeq test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.ValidationImpl(), false);
}

TEST(kavtorev_d_most_different_neighbor_elements_seq, InputSizeTwo_ReturnsCorrectPair) {
  std::vector<int> in = kavtorev_d_most_different_neighbor_elements_seq::Generator(2);
  std::vector<std::pair<int, int>> out(1);
  std::pair<int, int> ans = {std::min(in[0], in[1]), std::max(in[0], in[1])};

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  kavtorev_d_most_different_neighbor_elements_seq::MostDifferentNeighborElementsSeq test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.ValidationImpl(), true);
  test_task_sequential.PreProcessingImpl();
  test_task_sequential.RunImpl();
  test_task_sequential.PostProcessingImpl();
  ASSERT_EQ(ans, out[0]);
}

TEST(kavtorev_d_most_different_neighbor_elements_seq, SequentialInput_ReturnsFirstTwoElements) {
  std::vector<int> in = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  std::vector<std::pair<int, int>> out(1);
  std::pair<int, int> ans = {1, 2};

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  kavtorev_d_most_different_neighbor_elements_seq::MostDifferentNeighborElementsSeq test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.ValidationImpl(), true);
  test_task_sequential.PreProcessingImpl();
  test_task_sequential.RunImpl();
  test_task_sequential.PostProcessingImpl();
  ASSERT_EQ(ans, out[0]);
}

TEST(kavtorev_d_most_different_neighbor_elements_seq, MostlyZerosInput_ReturnsZeroAndLargest) {
  std::vector<int> in = {0, 0, 0, 0, 0, 0, 0, 0, 0, 12};
  std::vector<std::pair<int, int>> out(1);
  std::pair<int, int> ans = {0, 12};

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  kavtorev_d_most_different_neighbor_elements_seq::MostDifferentNeighborElementsSeq test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.ValidationImpl(), true);
  test_task_sequential.PreProcessingImpl();
  test_task_sequential.RunImpl();
  test_task_sequential.PostProcessingImpl();
  ASSERT_EQ(ans, out[0]);
}

TEST(kavtorev_d_most_different_neighbor_elements_seq, AllZerosInput_ReturnsZeroZero) {
  std::vector<int> in(100, 0);
  std::vector<std::pair<int, int>> out(1);
  std::pair<int, int> ans = {0, 0};

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  kavtorev_d_most_different_neighbor_elements_seq::MostDifferentNeighborElementsSeq test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.ValidationImpl(), true);
  test_task_sequential.PreProcessingImpl();
  test_task_sequential.RunImpl();
  test_task_sequential.PostProcessingImpl();
  ASSERT_EQ(ans, out[0]);
}

TEST(kavtorev_d_most_different_neighbor_elements_seq, CloseNegativeNumbers_ReturnsCorrectPair) {
  std::vector<int> in = {-1, -2, -3, -4, -1000};
  std::vector<std::pair<int, int>> out(1);
  std::pair<int, int> ans = {-1000, -4};

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  kavtorev_d_most_different_neighbor_elements_seq::MostDifferentNeighborElementsSeq test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.ValidationImpl(), true);
  test_task_sequential.PreProcessingImpl();
  test_task_sequential.RunImpl();
  test_task_sequential.PostProcessingImpl();
  ASSERT_EQ(ans, out[0]);
}