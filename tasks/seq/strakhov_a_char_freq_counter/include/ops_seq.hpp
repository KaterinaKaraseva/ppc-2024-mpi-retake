#pragma once

#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace strakhov_a_char_freq_counter_seq {

class CharFreqCounterSeq : public ppc::core::Task {
 public:
  explicit CharFreqCounterSeq(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  std::vector<signed char> input_;
  int result_{};
  char target_{};
};

}  // namespace strakhov_a_char_freq_counter_seq