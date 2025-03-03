#pragma once

#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace chernova_n_word_count_seq {

std::vector<char> CleanString(const std::vector<char>& input);
std::vector<char> GenerateWords(int k);
std::vector<char> GenerateWordsPerf(int k);

class TestTaskSequential : public ppc::core::Task {
 public:
  explicit TestTaskSequential(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  std::vector<char> input_;
  int spaceCount_{};
};

}  // namespace chernova_n_word_count_seq