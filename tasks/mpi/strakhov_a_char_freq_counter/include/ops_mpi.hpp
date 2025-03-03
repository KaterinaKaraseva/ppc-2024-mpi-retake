#pragma once

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace strakhov_a_char_freq_counter_mpi {

class CharFreqCounterSeq : public ppc::core::Task {
 public:
  explicit CharFreqCounterSeq(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  std::vector<signed char> input_;
  int result_{};
  char target_{};
  boost::mpi::communicator world_;
};

class CharFreqCounterPar : public ppc::core::Task {
 public:
  explicit CharFreqCounterPar(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  std::vector<signed char> input_, local_input_;
  int result_{}, local_result_{};
  char target_{};
  boost::mpi::communicator world_;
};

}  // namespace strakhov_a_char_freq_counter_mpi