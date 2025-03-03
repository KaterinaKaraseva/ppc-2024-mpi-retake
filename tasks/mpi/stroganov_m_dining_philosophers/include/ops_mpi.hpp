// Copyright 2023 Nesterov Alexander
#pragma once

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <utility>

#include "core/task/include/task.hpp"

namespace stroganov_m_dining_philosophers_mpi {

class DiningPhilosophersMPI : public ppc::core::Task {
 public:
  explicit DiningPhilosophersMPI(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  void Eat();
  void Think();
  bool DistributionForks();
  void ReleaseForks();
  bool CheckDeadlock();
  void ResolveDeadlock();
  bool CheckAllThink();

 private:
  boost::mpi::communicator world_;
  int status_;
  int l_philosopher_;
  int r_philosopher_;
  int count_philosophers_;
};

}  // namespace stroganov_m_dining_philosophers_mpi