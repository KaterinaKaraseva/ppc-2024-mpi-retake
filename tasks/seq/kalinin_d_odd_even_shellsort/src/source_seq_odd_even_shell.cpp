#include <algorithm>
#include <random>
#include <vector>

#include "seq/kalinin_d_odd_even_shellsort/include/header_seq_odd_even_shell.hpp"
namespace kalinin_d_odd_even_shell_seq {

void OddEvenShellSeq::ShellSort(std::vector<int>& vec) {
  int n = static_cast<int>(vec.size());
  for (int gap = n / 2; gap > 0; gap /= 2) {
    for (int i = gap; i < n; i++) {
      int temp = vec[i];
      int j = 0;
      for (j = i; j >= gap && vec[j - gap] > temp; j -= gap) {
        vec[j] = vec[j - gap];
      }
      vec[j] = temp;
    }
  }
}

bool OddEvenShellSeq::PreProcessingImpl() {
  // Init vectors
  int n = static_cast<int>(task_data->inputs_count[0]);
  input_ = std::vector<int>(n);
  std::ranges::copy(reinterpret_cast<int*>(task_data->inputs[0]), reinterpret_cast<int*>(task_data->inputs[0]) + n,
                    input_.begin());
  return true;
}

bool OddEvenShellSeq::ValidationImpl() { return (task_data->outputs_count[0] > 0 && task_data->inputs_count[0] > 1); }

bool OddEvenShellSeq::RunImpl() {
  output_ = input_;
  ShellSort(output_);
  return true;
}

bool OddEvenShellSeq::PostProcessingImpl() {
  std::ranges::copy(output_, reinterpret_cast<int*>(task_data->outputs[0]));
  return true;
}

void GimmeRandVec(std::vector<int>& vec) {
  std::random_device rd;
  std::default_random_engine reng(rd());
  std::uniform_int_distribution<int> dist(0, static_cast<int>(vec.size()));
  std::ranges::generate(vec.begin(), vec.end(), [&dist, &reng] { return dist(reng); });
}
}  // namespace kalinin_d_odd_even_shell_seq