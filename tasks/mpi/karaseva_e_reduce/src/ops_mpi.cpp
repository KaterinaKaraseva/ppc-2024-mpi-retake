#include "mpi/karaseva_e_reduce/include/ops_mpi.hpp"

#include <cmath>
#include <cstddef>
#include <numeric>  // ��� std::accumulate
#include <vector>

template <typename T>
bool karaseva_e_reduce_mpi::TestTaskMPI<T>::PreProcessingImpl() {
  // ��������� ������� ������ ��� T (int, float ��� double)
  unsigned int input_size = task_data->inputs_count[0];
  auto* in_ptr = reinterpret_cast<T*>(task_data->inputs[0]);
  input_ = std::vector<T>(in_ptr, in_ptr + input_size);

  // �������� ������ ���������� ��� T
  unsigned int output_size = task_data->outputs_count[0];
  output_ = std::vector<T>(output_size, 0.0);  // ���������� T ��� output_

  rc_size_ = static_cast<int>(std::sqrt(input_size));  // ������ ����������, ���� �� ���� �������
  return true;
}

template <typename T>
bool karaseva_e_reduce_mpi::TestTaskMPI<T>::ValidationImpl() {
  // ��� �������� ������ ������� ������ ������ ���� ������ 1, � �������� ������ 1
  return task_data->inputs_count[0] > 1 && task_data->outputs_count[0] == 1;
}

template <typename T>
bool karaseva_e_reduce_mpi::TestTaskMPI<T>::RunImpl() {
  // ��������� ������������
  T local_sum = std::accumulate(input_.begin(), input_.end(), T(0));  // ���������� T ��� ��������

  // ���������� ��� ���������� �����
  T global_sum = 0;

  // �������� ������ ��� ��������
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);  // �������� ����� �������� ��������
  MPI_Comm_size(MPI_COMM_WORLD, &size);  // �������� ����� ���������� ���������

  int partner_rank;
  for (int step = 1; step < size; step *= 2) {
    partner_rank = rank ^ step;  // ������� �������� ��� �������� ������

    if (rank < partner_rank) {
      // ������� ������� ���������� ������
      MPI_Send(&local_sum, 1, MPI_DOUBLE, partner_rank, 0, MPI_COMM_WORLD);  // �������� ��� MPI_DOUBLE
      break;
    } else if (rank > partner_rank) {
      // ������� ������� �������� ������
      T recv_data = 0;
      MPI_Recv(&recv_data, 1, MPI_DOUBLE, partner_rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      local_sum += recv_data;
    }
  }

  // �������� ������� �������� ���������
  if (rank == 0) {
    global_sum = local_sum;
    output_ = {global_sum};  // ��������� ��� T
  }

  MPI_Barrier(MPI_COMM_WORLD);  // ������������� ���� ���������
  return true;
}

template <typename T>
bool karaseva_e_reduce_mpi::TestTaskMPI<T>::PostProcessingImpl() {
  // ����������� ���������� � �������� ������
  // ����������� �������� ������, ����� �� �������������� ���� T
  for (size_t i = 0; i < output_.size(); i++) {
    reinterpret_cast<T*>(task_data->outputs[0])[i] = output_[i];  // ���������� T ��� ������
  }
  return true;
}

// ����� ����������� ��������� �������
template bool karaseva_e_reduce_mpi::TestTaskMPI<int>::PreProcessingImpl();
template bool karaseva_e_reduce_mpi::TestTaskMPI<int>::ValidationImpl();
template bool karaseva_e_reduce_mpi::TestTaskMPI<int>::RunImpl();
template bool karaseva_e_reduce_mpi::TestTaskMPI<int>::PostProcessingImpl();

template bool karaseva_e_reduce_mpi::TestTaskMPI<float>::PreProcessingImpl();
template bool karaseva_e_reduce_mpi::TestTaskMPI<float>::ValidationImpl();
template bool karaseva_e_reduce_mpi::TestTaskMPI<float>::RunImpl();
template bool karaseva_e_reduce_mpi::TestTaskMPI<float>::PostProcessingImpl();

template bool karaseva_e_reduce_mpi::TestTaskMPI<double>::PreProcessingImpl();
template bool karaseva_e_reduce_mpi::TestTaskMPI<double>::ValidationImpl();
template bool karaseva_e_reduce_mpi::TestTaskMPI<double>::RunImpl();
template bool karaseva_e_reduce_mpi::TestTaskMPI<double>::PostProcessingImpl();
