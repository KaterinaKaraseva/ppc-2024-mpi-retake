#define OMPI_SKIP_MPICXX

#include "mpi/karaseva_e_reduce/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cstdint>
#include <numeric>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace karaseva_e_reduce_mpi {

template <typename T>
bool TestTaskMPI<T>::PreProcessingImpl() {
  MPI_Comm_rank(MPI_COMM_WORLD, &rank_);
  MPI_Comm_size(MPI_COMM_WORLD, &size_);

  if constexpr (std::is_same_v<T, int>) {
    mpi_type_ = MPI_INT;
  } else if constexpr (std::is_same_v<T, float>) {
    mpi_type_ = MPI_FLOAT;
  } else if constexpr (std::is_same_v<T, double>) {
    mpi_type_ = MPI_DOUBLE;
  }

  if (task_data->inputs_count.size() > 1) {
    root_ = *reinterpret_cast<int*>(task_data->inputs[1]);
    root_ = root_ % size_;
  } else {
    root_ = 0;
  }

  if (rank_ == root_) {
    input_size_ = task_data->inputs_count[0];
    if (input_size_ <= 0) {
      throw std::runtime_error("Input size must be positive");
    }

    input_.resize(input_size_);
    auto* input_data = reinterpret_cast<T*>(task_data->inputs[0]);
    std::copy(input_data, input_data + input_size_, input_.begin());

    const int base_chunk = input_size_ / size_;
    const int remainder = input_size_ % size_;
    std::vector<int> counts(size_, base_chunk);
    for (int i = 0; i < remainder; ++i) {
      counts[i]++;
    }

    int offset = 0;
    for (int proc = 0; proc < size_; ++proc) {
      if (proc == rank_) {
        local_input_.assign(input_.begin() + offset, input_.begin() + offset + counts[proc]);
      } else {
        MPI_Send(&counts[proc], 1, MPI_INT, proc, 0, MPI_COMM_WORLD);
        MPI_Send(input_.data() + offset, counts[proc], mpi_type_, proc, 0, MPI_COMM_WORLD);
      }
      offset += counts[proc];
    }
  } else {
    MPI_Recv(&local_size_, 1, MPI_INT, root_, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    local_input_.resize(local_size_);
    MPI_Recv(local_input_.data(), local_size_, mpi_type_, root_, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  }

  return true;
}

template <typename T>
bool TestTaskMPI<T>::ValidationImpl() {
  return !task_data->inputs_count.empty() && task_data->inputs_count[0] > 0 && !task_data->outputs_count.empty() &&
         task_data->outputs_count[0] >= 1;
}

template <typename T>
bool TestTaskMPI<T>::RunImpl() {
  T local_sum = std::accumulate(local_input_.begin(), local_input_.end(), T{0});
  T result = local_sum;

  int step = 1;
  int vr = (rank_ - root_ + size_) % size_;

  while (step < size_) {
    if (vr % (2 * step) == 0) {
      int partner_vr = vr + step;
      if (partner_vr >= size_) {
        step *= 2;
        continue;
      }
      int partner_rank = (partner_vr + root_) % size_;

      T recv_data;
      MPI_Recv(&recv_data, 1, mpi_type_, partner_rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      result += recv_data;
    } else {
      int partner_vr = vr - step;
      int partner_rank = (partner_vr + root_) % size_;

      MPI_Send(&result, 1, mpi_type_, partner_rank, 0, MPI_COMM_WORLD);
      break;
    }
    step *= 2;
  }

  if (rank_ == root_) {
    result_ = result;
  }

  return true;
}

template <typename T>
bool TestTaskMPI<T>::PostProcessingImpl() {
  if (rank_ == root_) {
    if (task_data->outputs[0] == nullptr) {
      task_data->outputs[0] = new uint8_t[sizeof(T)];
    }
    *reinterpret_cast<T*>(task_data->outputs[0]) = result_;
  }
  return true;
}

template class TestTaskMPI<int>;
template class TestTaskMPI<float>;
template class TestTaskMPI<double>;

}  // namespace karaseva_e_reduce_mpi