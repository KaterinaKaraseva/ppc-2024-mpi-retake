#pragma once

#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace kavtorev_d_most_different_neighbor_elements_seq {

class MostDifferentNeighborElementsSeq : public ppc::core::Task {
 public:
  explicit MostDifferentNeighborElementsSeq(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  std::vector<std::pair<int, int>> input_;
  std::pair<int, int> res_;
};

}  // namespace kavtorev_d_most_different_neighbor_elements_seq