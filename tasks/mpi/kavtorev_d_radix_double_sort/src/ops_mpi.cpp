#include "mpi/kavtorev_d_radix_double_sort/include/ops_mpi.hpp"

#include <algorithm>
#include <boost/mpi/collectives.hpp>
#include <boost/mpi/collectives/broadcast.hpp>
#include <boost/mpi/collectives/scatterv.hpp>
#include <boost/serialization/vector.hpp>  //NOLINT
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iterator>
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

bool RadixSortParallel::PreProcessingImpl() {
  if (world_.rank() == 0) {
    data_.resize(n_);
    auto* arr = reinterpret_cast<double*>(task_data->inputs[1]);
    std::copy(arr, arr + n_, data_.begin());
  }

  return true;
}

bool RadixSortParallel::ValidationImpl() {
  bool is_valid = true;
  if (world_.rank() == 0) {
    n_ = *(reinterpret_cast<int*>(task_data->inputs[0]));
    if (task_data->inputs_count[0] != 1 || task_data->inputs_count[1] != static_cast<size_t>(n_) ||
        task_data->outputs_count[0] != static_cast<size_t>(n_)) {
      is_valid = false;
    }
  }
  boost::mpi::broadcast(world_, is_valid, 0);
  boost::mpi::broadcast(world_, n_, 0);
  return is_valid;
}

bool RadixSortParallel::RunImpl() {
  int rank = world_.rank();
  int size = world_.size();
  int local_n = n_ / size;
  int remainder = n_ % size;

  std::vector<int> counts(size);
  std::vector<int> displs(size, 0);
  if (rank == 0) {
    for (int i = 0; i < size; ++i) {
      counts[i] = local_n + (i < remainder ? 1 : 0);
    }
    for (int i = 1; i < size; ++i) {
      displs[i] = displs[i - 1] + counts[i - 1];
    }
  }

  boost::mpi::broadcast(world_, counts, 0);
  boost::mpi::broadcast(world_, displs, 0);

  std::vector<double> local_data(counts[rank]);
  boost::mpi::scatterv(world_, (rank == 0 ? data_.data() : (double*)nullptr), counts, displs, local_data.data(),
                       counts[rank], 0);
  RadixSortDoubles(local_data);

  int steps = 0;
  {
    int tmp = size;
    while (tmp > 1) {
      tmp = (tmp + 1) / 2;
      steps++;
    }
  }

  int group_size = 1;
  for (int step = 0; step < steps; ++step) {
    int partner_rank = rank + group_size;
    int group_step_size = group_size * 2;
    bool is_merger = (rank % group_step_size == 0);
    bool has_partner = (partner_rank < size);

    if (is_merger && has_partner) {
      int partner_size = 0;
      world_.recv(partner_rank, 0, partner_size);

      std::vector<double> partner_data(partner_size);
      world_.recv(partner_rank, 1, partner_data.data(), partner_size);

      std::vector<double> merged;
      merged.reserve(local_data.size() + partner_data.size());
      std::ranges::merge(local_data.begin(), local_data.end(), partner_data.begin(), partner_data.end(),
                         std::back_inserter(merged));
      local_data.swap(merged);
    } else if (!is_merger && (rank % group_step_size == group_size)) {
      int receiver = rank - group_size;
      int my_size = (int)local_data.size();
      world_.send(receiver, 0, my_size);
      world_.send(receiver, 1, local_data.data(), my_size);
      local_data.clear();
    }

    group_size *= 2;
  }

  if (rank == 0) {
    data_.swap(local_data);
  }

  return true;
}

bool RadixSortParallel::PostProcessingImpl() {
  if (world_.rank() == 0) {
    auto* out = reinterpret_cast<double*>(task_data->outputs[0]);
    std::ranges::copy(data_, out);
  }

  return true;
}

void RadixSortParallel::RadixSortDoubles(std::vector<double>& data) {
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

void RadixSortParallel::RadixSortUint64(std::vector<uint64_t>& keys) {
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