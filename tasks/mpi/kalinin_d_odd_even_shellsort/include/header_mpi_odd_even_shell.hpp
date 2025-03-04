#pragma once

#include <boost/mpi/communicator.hpp>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace kalinin_d_odd_even_shell_mpi {
class OddEvenShellMpi : public ppc::core::Task {
 public:
  explicit OddEvenShellMpi(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  void ExchangeAndMerge(std::vector<int>& local_vec, int neighbour);
  void GatherResults(std::vector<int>& local_vec, int local_sz, int id);
  bool PostProcessingImpl() override;

  static void ShellSort(std::vector<int>& vec);

 private:
  std::vector<int> input_;
  std::vector<int> output_;
  boost::mpi::communicator world_;
};
void GimmeRandVec(std::vector<int>& vec);

}  // namespace kalinin_d_odd_even_shell_mpi
