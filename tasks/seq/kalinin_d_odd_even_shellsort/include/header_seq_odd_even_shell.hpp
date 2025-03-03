#pragma once

#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace kalinin_d_odd_even_shell_seq {
class OddEvenShellSeq : public ppc::core::Task {
 public:
  explicit OddEvenShellSeq(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  static void ShellSort(std::vector<int>& vec);

 private:
  std::vector<int> input_;
  std::vector<int> output_;
};
void GimmeRandVec(std::vector<int>& vec);

}  // namespace kalinin_d_odd_even_shell_seq