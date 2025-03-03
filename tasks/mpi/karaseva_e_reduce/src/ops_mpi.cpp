#define OMPI_SKIP_MPICXX

#include "mpi/karaseva_e_reduce/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace karaseva_e_reduce_mpi {

template <typename T>
bool TestTaskMPI<T>::PreProcessingImpl() {
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  if constexpr (std::is_same_v<T, int>) {
    mpi_type = MPI_INT;
  } else if constexpr (std::is_same_v<T, float>) {
    mpi_type = MPI_FLOAT;
  } else if constexpr (std::is_same_v<T, double>) {
    mpi_type = MPI_DOUBLE;
  }

  if (task_data->inputs_count.size() > 1) {
    root = *reinterpret_cast<int*>(task_data->inputs[1]);
    root = root % size;
  } else {
    root = 0;
  }

  if (rank == root) {
    input_size_ = task_data->inputs_count[0];
    if (input_size_ <= 0) {
      throw std::runtime_error("Input size must be positive");
    }

    input_.resize(input_size_);
    auto* input_data = reinterpret_cast<T*>(task_data->inputs[0]);
    std::copy(input_data, input_data + input_size_, input_.begin());

    const int base_chunk = input_size_ / size;
    const int remainder = input_size_ % size;
    std::vector<int> counts(size, base_chunk);
    for (int i = 0; i < remainder; ++i) counts[i]++;

    int offset = 0;
    for (int proc = 0; proc < size; ++proc) {
      if (proc == rank) {
        local_input_.assign(input_.begin() + offset, input_.begin() + offset + counts[proc]);
      } else {
        MPI_Send(&counts[proc], 1, MPI_INT, proc, 0, MPI_COMM_WORLD);
        MPI_Send(input_.data() + offset, counts[proc], mpi_type, proc, 0, MPI_COMM_WORLD);
      }
      offset += counts[proc];
    }
  } else {
    MPI_Recv(&local_size, 1, MPI_INT, root, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    local_input_.resize(local_size);
    MPI_Recv(local_input_.data(), local_size, mpi_type, root, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  }

  return true;
}

template <typename T>
bool TestTaskMPI<T>::ValidationImpl() {
  return task_data->inputs_count.size() > 0 && task_data->inputs_count[0] > 0 && task_data->outputs_count.size() > 0 &&
         task_data->outputs_count[0] >= 1;
}

template <typename T>
bool TestTaskMPI<T>::RunImpl() {
  T local_sum = std::accumulate(local_input_.begin(), local_input_.end(), T{0});
  T result = local_sum;

  int step = 1;
  while (step < size) {
    int distance = (rank - root + size) % size;
    if (distance % (2 * step) == 0) {
      int partner_rank = rank + step;
      if (partner_rank >= size) {
        partner_rank -= size;
      }
      if ((distance / step) % 2 == 0) {
        T recv_data;
        MPI_Recv(&recv_data, 1, mpi_type, partner_rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        result += recv_data;
      } else {
        MPI_Send(&result, 1, mpi_type, partner_rank, 0, MPI_COMM_WORLD);
        break;
      }
    }
    step *= 2;
  }

  if (rank == root) {
    result_ = result;
  }

  return true;
}

template <typename T>
bool TestTaskMPI<T>::PostProcessingImpl() {
  if (rank == root) {
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