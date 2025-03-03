#include "seq/deryabin_m_cannons_algorithm/include/ops_seq.hpp"

#include <cmath>
#include <vector>

bool deryabin_m_cannons_algorithm_seq::CannonsAlgorithmTaskSequential::PreProcessingImpl() {
  input_matrix_A_ = reinterpret_cast<std::vector<double> *>(task_data->inputs[0])[0];
  input_matrix_B_ = reinterpret_cast<std::vector<double> *>(task_data->inputs[1])[0];
  output_matrix_C_ = std::vector<double>(input_matrix_A_.size());
  return true;
}

bool deryabin_m_cannons_algorithm_seq::CannonsAlgorithmTaskSequential::ValidationImpl() {
  return task_data->inputs_count[0] == task_data->inputs_count[1] &&
         task_data->inputs_count[1] == pow((unsigned short)sqrt(task_data->inputs_count[0]), 2) &&
         task_data->outputs_count[0] == 1;
}

bool deryabin_m_cannons_algorithm_seq::CannonsAlgorithmTaskSequential::RunImpl() {
  unsigned short i = 0;
  unsigned short j = 0;
  unsigned short count = 0;
  auto dimension = (unsigned short)sqrt(static_cast<unsigned short>(input_matrix_A_.size()));
  while (i != dimension) {
    j = 0;
    while (j != dimension) {
      count = 0;
      while (count != dimension) {
        output_matrix_C_[(i * dimension) + j] +=
            input_matrix_A_[(i * dimension) + count] * input_matrix_B_[(count * dimension) + j];
        count++;
      }
      j++;
    }
    i++;
  }
  return true;
}

bool deryabin_m_cannons_algorithm_seq::CannonsAlgorithmTaskSequential::PostProcessingImpl() {
  reinterpret_cast<std::vector<double> *>(task_data->outputs[0])[0] = output_matrix_C_;
  return true;
}
