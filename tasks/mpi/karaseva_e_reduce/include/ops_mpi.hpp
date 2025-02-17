#pragma once

#include <mpi.h>

#include <numeric>  // ��� std::accumulate
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace karaseva_e_reduce_mpi {

template <typename T>
class TestTaskMPI : public ppc::core::Task {
 public:
  explicit TestTaskMPI(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  std::vector<T> input_, output_;
  int rc_size_{};  // ������ ������

  void ReduceBinaryTree(T* local_data, T& global_data, int root);
};

}  // namespace karaseva_e_reduce_mpi
