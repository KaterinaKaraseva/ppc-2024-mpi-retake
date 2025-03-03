#include <algorithm>
#include <cstddef>
#include <vector>

#include "seq/budazhapova_betcher_odd_even_merge_seq/include/radix_sort_with_betcher_seq.h"

namespace budazhapova_betcher_odd_even_merge_seq {
namespace {
void CountingSort(std::vector<int>& arr, int exp) {
  size_t n = arr.size();
  std::vector<int> output(n);
  std::vector<int> count(10, 0);

  for (size_t i = 0; i < n; i++) {
    int index = (arr[i] / exp) % 10;
    count[index]++;
  }
  for (size_t i = 1; i < 10; i++) {
    count[i] += count[i - 1];
  }
  for (size_t i = n; i-- > 0;) {
    int index = (arr[i] / exp) % 10;
    output[count[index] - 1] = arr[i];
    count[index]--;
  }
  for (size_t i = 0; i < n; i++) {
    arr[i] = output[i];
  }
}

void RadixSort(std::vector<int>& arr) {
  auto max_num_iter = std::ranges::max_element(arr);
  int max_num = *max_num_iter;
  for (int exp = 1; (max_num / exp > 0); exp *= 10) {
    CountingSort(arr, exp);
  }
}
}  // namespace
}  // namespace budazhapova_betcher_odd_even_merge_seq
bool budazhapova_betcher_odd_even_merge_seq::MergeSequential::PreProcessingImpl() {
  res_ = std::vector<int>(reinterpret_cast<int*>(task_data->inputs[0]),
                          reinterpret_cast<int*>(task_data->inputs[0]) + task_data->inputs_count[0]);
  return true;
}

bool budazhapova_betcher_odd_even_merge_seq::MergeSequential::ValidationImpl() {
  return task_data->inputs_count[0] > 0;
}

bool budazhapova_betcher_odd_even_merge_seq::MergeSequential::RunImpl() {
  RadixSort(res_);
  return true;
}

bool budazhapova_betcher_odd_even_merge_seq::MergeSequential::PostProcessingImpl() {
  int* output = reinterpret_cast<int*>(task_data->outputs[0]);
  for (size_t i = 0; i < res_.size(); i++) {
    output[i] = res_[i];
  }
  return true;
}