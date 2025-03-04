#pragma once

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <string>
#include <utility>

#include "core/task/include/task.hpp"

namespace budazhapova_e_count_freq_chart_mpi {

int CountingFreq(std::string str, char symb);

class TestMPITaskSequential : public ppc::core::Task {
 public:
  explicit TestMPITaskSequential(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  std::string input_;
  int res_ = 0;
  char symb_;
};

class TestMPITaskParallel : public ppc::core::Task {
 public:
  explicit TestMPITaskParallel(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  std::string input_, local_input_;
  int res_ = 0, local_res_{};
  char symb_;
  boost::mpi::communicator world_;
};
}  // namespace budazhapova_e_count_freq_chart_mpi