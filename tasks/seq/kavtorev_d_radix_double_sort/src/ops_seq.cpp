#include "seq/kavtorev_d_radix_double_sort/include/ops_seq.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

using namespace kavtorev_d_radix_double_sort;

bool RadixSortSequential::PreProcessingImpl() {
  data_.resize(n_);
  auto* arr = reinterpret_cast<double*>(task_data->inputs[1]);
  std::copy(arr, arr + n_, data_.begin());

  return true;
}

bool RadixSortSequential::ValidationImpl() {
  bool is_valid = true;
  n_ = *(reinterpret_cast<int*>(task_data->inputs[0]));
  if (task_data->inputs_count[0] != 1 || task_data->inputs_count[1] != static_cast<size_t>(n_) ||
      task_data->outputs_count[0] != static_cast<size_t>(n_)) {
    is_valid = false;
  }

  return is_valid;
}

bool RadixSortSequential::RunImpl() {
  RadixSortDoubles(data_);
  return true;
}

bool RadixSortSequential::PostProcessingImpl() {
  auto* out = reinterpret_cast<double*>(task_data->outputs[0]);
  std::ranges::copy(data_.begin(), data_.end(), out);
  return true;
}

void RadixSortSequential::RadixSortDoubles(std::vector<double>& data) {
  size_t n = data.size();
  std::vector<uint64_t> keys(n);
  for (size_t i = 0; i < n; ++i) {
    uint64_t u = 0;
    std::memcpy(&u, &data[i], sizeof(double));
    if ((u & 0x8000000000000000ULL) != 0) {
      u = ~u;
    } else {
      u |= 0x8000000000000000ULL;
    }
    keys[i] = u;
  }

  RadixSortUint64(keys);

  for (size_t i = 0; i < n; ++i) {
    uint64_t u = keys[i];
    if ((u & 0x8000000000000000ULL) != 0) {
      u &= ~0x8000000000000000ULL;
    } else {
      u = ~u;
    }
    std::memcpy(&data[i], &u, sizeof(double));
  }
}

void RadixSortSequential::RadixSortUint64(std::vector<uint64_t>& keys) {
  const int bits = 64;
  const int radix = 256;
  std::vector<uint64_t> temp(keys.size());

  for (int shift = 0; shift < bits; shift += 8) {
    size_t count[radix + 1] = {0};
    for (size_t i = 0; i < keys.size(); ++i) {
      uint8_t byte = (keys[i] >> shift) & 0xFF;
      ++count[byte + 1];
    }
    for (int i = 0; i < radix; ++i) {
      count[i + 1] += count[i];
    }
    for (size_t i = 0; i < keys.size(); ++i) {
      uint8_t byte = (keys[i] >> shift) & 0xFF;
      temp[count[byte]++] = keys[i];
    }
    keys.swap(temp);
  }
}