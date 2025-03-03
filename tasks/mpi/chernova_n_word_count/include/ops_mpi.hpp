#pragma once

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace chernova_n_word_count_mpi {

std::vector<char> CleanString(const std::vector<char>& input);
std::vector<char> GenerateWords(int k);
std::vector<char> GenerateWordsPerf(int k);

class TestMPITaskSequential : public ppc::core::Task {
 public:
  explicit TestMPITaskSequential(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  std::vector<char> input_;
  int space_count_{};
};

class TestMPITaskParallel : public ppc::core::Task {
 public:
  explicit TestMPITaskParallel(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  std::vector<char> input_, local_input_;
  int space_count_{};
  int local_space_count_{};
  int part_size_{};
  boost::mpi::communicator world_;
};

}  // namespace chernova_n_word_count_mpi