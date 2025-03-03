#pragma once

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <climits>
#include <memory>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace mezhuev_m_most_different_neighbor_elements_mpi {

class MostDifferentNeighborElements : public ppc::core::Task {
 public:
  explicit MostDifferentNeighborElements(boost::mpi::communicator& world,
                                         std::shared_ptr<ppc::core::TaskData> task_data)
      : Task(std::move(task_data)), world_(world) {}

  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  boost::mpi::communicator world_;
  std::vector<int> input_;
  std::pair<int, int> result_;
};

}  // namespace mezhuev_m_most_different_neighbor_elements_mpi