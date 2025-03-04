#pragma once
#include <memory>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace konstantinov_i_sum_of_vector_elements_seq {

int VecElemSum(const std::vector<int>& vec);

class SumVecElemSequential : public ppc::core::Task {
 public:
  explicit SumVecElemSequential(std::shared_ptr<ppc::core::TaskData> task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  std::vector<int> input_;
  int result_{};
};
}  // namespace konstantinov_i_sum_of_vector_elements_seq