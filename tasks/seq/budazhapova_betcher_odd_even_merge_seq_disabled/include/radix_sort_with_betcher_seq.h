#pragma once

#include <memory>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace budazhapova_betcher_odd_even_merge_seq {

class MergeSequential : public ppc::core::Task {
 public:
  explicit MergeSequential(std::shared_ptr<ppc::core::TaskData> task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  std::vector<int> res_;
};
}  // namespace budazhapova_betcher_odd_even_merge_seq