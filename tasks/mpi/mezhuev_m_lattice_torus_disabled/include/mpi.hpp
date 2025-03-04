#pragma once
#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <cmath>
#include <memory>
#include <utility>

#include "core/task/include/task.hpp"

namespace mezhuev_m_lattice_torus_mpi {

class GridTorusTopologyParallel : public ppc::core::Task {
 public:
  explicit GridTorusTopologyParallel(std::shared_ptr<ppc::core::TaskData> task_data) : Task(std::move(task_data)) {}

  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  boost::mpi::communicator world_;
};

}  // namespace mezhuev_m_lattice_torus_mpi