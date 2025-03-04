#ifndef MODULES_TASK_3_KARASEVA_E_REDUCE_OPS_MPI_HPP
#define MODULES_TASK_3_KARASEVA_E_REDUCE_OPS_MPI_HPP

#define OMPI_SKIP_MPICXX

#include <mpi.h>

#include <boost/mpi/collectives.hpp>
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
  int rc_size_{};
  int input_size_;
  int local_size_;
  int remel_;
  std::vector<T> local_input_;
  T result_;
  MPI_Op op_;
  int size_;
  int rank_;
  int root_;
  MPI_Datatype mpi_type_;
};

}  // namespace karaseva_e_reduce_mpi

#endif  // MODULES_TASK_3_KARASEVA_E_REDUCE_OPS_MPI_HPP