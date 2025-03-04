#include "seq/Konstantinov_I_sum_of_vector_elements/include/ops_seq.hpp"

#include <cmath>
#include <vector>

int konstantinov_i_sum_of_vector_elements_seq::VecElemSum(const std::vector<int>& vec) {
  int result = 0;
  for (int elem : vec) {
    result += elem;
  }
  return result;
}

bool konstantinov_i_sum_of_vector_elements_seq::SumVecElemSequential::PreProcessingImpl() {
  int rows = static_cast<int>(task_data->inputs_count[0]);
  int columns = static_cast<int>(task_data->inputs_count[1]);

  input_ = std::vector<int>(rows * columns);

  for (int i = 0; i < rows; i++) {
    auto* el = reinterpret_cast<int*>(task_data->inputs[i]);
    for (int j = 0; j < columns; j++) {
      input_[(i * columns) + j] = el[j];
    }
  }

  return true;
}

bool konstantinov_i_sum_of_vector_elements_seq::SumVecElemSequential::ValidationImpl() {
  return (task_data->inputs_count.size() == 2 && task_data->inputs_count[0] > 0 && task_data->inputs_count[1] > 0 &&
          task_data->outputs_count.size() == 1 && task_data->outputs_count[0] == 1);
}

bool konstantinov_i_sum_of_vector_elements_seq::SumVecElemSequential::RunImpl() {
  result_ = VecElemSum(input_);
  return true;
}

bool konstantinov_i_sum_of_vector_elements_seq::SumVecElemSequential::PostProcessingImpl() {
  reinterpret_cast<int*>(task_data->outputs[0])[0] = result_;
  return true;
}