#pragma once

#include <string>
#include <utility>

#include "core/task/include/task.hpp"

namespace budazhapova_e_count_freq_chart_seq {

class TestTaskSequential : public ppc::core::Task {
 public:
  explicit TestTaskSequential(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  std::string input_;
  char symb_{};
  int res_{};
};

}  // namespace budazhapova_e_count_freq_chart_seq