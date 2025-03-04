#include <gtest/gtest.h>

#include <cstddef>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

#include "core/task/include/task.hpp"
#include "core/util/include/util.hpp"
#include "seq/karaseva_e_reduce/include/ops_seq.hpp"

TEST(karaseva_e_reduce_seq, test_reduce_50) {
  constexpr size_t kCount = 50;

  std::vector<int> in(kCount * kCount, 0);
  std::vector<int> out(1, 0);

  for (size_t i = 0; i < kCount; i++) {
    in[(i * kCount) + i] = 1;
  }

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<unsigned char *>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<unsigned char *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  karaseva_e_reduce_seq::TestTaskSequential test_task_sequential(task_data_seq);
  ASSERT_TRUE(test_task_sequential.Validation());
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();

  EXPECT_EQ(static_cast<size_t>(out[0]), kCount);
}

TEST(karaseva_e_reduce_seq, test_reduce_100_from_file) {
  std::string line;
  std::ifstream test_file(ppc::util::GetAbsolutePath("seq/karaseva_e_reduce/data/test.txt"));
  if (!test_file.is_open()) {
    FAIL() << "Could not open test file";
  }

  getline(test_file, line);
  test_file.close();

  const size_t count = std::stoul(line);

  std::vector<int> in(count * count, 0);
  std::vector<int> out(1, 0);

  for (size_t i = 0; i < count; i++) {
    in[(i * count) + i] = 1;
  }

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<unsigned char *>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<unsigned char *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  karaseva_e_reduce_seq::TestTaskSequential test_task_sequential(task_data_seq);
  ASSERT_TRUE(test_task_sequential.Validation());
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();

  EXPECT_EQ(static_cast<size_t>(out[0]), count);
}