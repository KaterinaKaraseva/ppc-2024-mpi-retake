#pragma once

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <climits>
#include <cstddef>
#include <memory>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace kavtorev_d_most_different_neighbor_elements_mpi {

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

class MostDifferentNeighborElementsMpi : public ppc::core::Task {
 public:
  explicit MostDifferentNeighborElementsMpi(std::shared_ptr<ppc::core::TaskData> task_data)
      : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  std::vector<int> input_, local_input_;
  std::pair<int, int> res_;
  size_t size_;
  size_t st_;
  boost::mpi::communicator world_;
};

}  // namespace kavtorev_d_most_different_neighbor_elements_mpi