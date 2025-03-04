#include <algorithm>
#include <boost/mpi/collectives.hpp>
#include <boost/mpi/collectives/broadcast.hpp>
#include <boost/mpi/collectives/gatherv.hpp>
#include <boost/mpi/communicator.hpp>
#include <boost/serialization/vector.hpp>  // NOLINT
#include <cstddef>
#include <vector>

#include "mpi/budazhapova_betcher_odd_even_merge_mpi/include/radix_sort_with_betcher.h"

namespace budazhapova_betcher_odd_even_merge_mpi {
namespace {

void CountingSort(std::vector<int>& arr, int exp) {
  size_t n = arr.size();
  std::vector<int> output(n);
  std::vector<int> count(10, 0);

  for (size_t i = 0; i < n; i++) {
    int index = (arr[i] / exp) % 10;
    count[index]++;
  }
  for (int i = 1; i < 10; i++) {
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
void OddEvenMerge(std::vector<int>& local_data, std::vector<int>& received_data) {
  std::vector<int> merged(local_data.size() + received_data.size());
  std::ranges::copy(local_data, merged.begin());
  std::ranges::copy(received_data, merged.begin() + static_cast<long>(local_data.size()));
  budazhapova_betcher_odd_even_merge_mpi::RadixSort(merged);
  received_data.assign(merged.begin(), merged.begin() + static_cast<long>(received_data.size()));
  local_data.assign(merged.begin() + static_cast<long>(received_data.size()), merged.end());
}

void PerformOddEvenMerge(int neighbor_rank, std::vector<int>& local_data, const boost::mpi::communicator& world) {
  std::vector<int> received_data;
  world.recv(neighbor_rank, neighbor_rank, received_data);
  OddEvenMerge(local_data, received_data);
  world.send(neighbor_rank, world.rank(), received_data);
}

void OddEvenSortPhase(int phase, std::vector<int>& local_data, const boost::mpi::communicator& world) {
  int next_rank = world.rank() + 1;
  int prev_rank = world.rank() - 1;
  if (phase % 2 == 0) {
    if (world.rank() % 2 == 0 && next_rank < world.size()) {
      world.send(next_rank, world.rank(), local_data);
    } else if (world.rank() % 2 == 1) {
      PerformOddEvenMerge(prev_rank, local_data, world);
    }
    if (world.rank() % 2 == 0 && next_rank < world.size()) {
      world.recv(next_rank, next_rank, local_data);
    }
  } else {
    if (world.rank() % 2 == 1 && next_rank < world.size()) {
      world.send(next_rank, world.rank(), local_data);
    } else if (world.rank() % 2 == 0 && world.rank() > 0) {
      PerformOddEvenMerge(prev_rank, local_data, world);
    }
    if (world.rank() % 2 == 1 && next_rank < world.size()) {
      world.recv(next_rank, next_rank, local_data);
    }
  }
}

void DistributeData(int& n_of_send_elements, int& n_of_proc_with_extra_elements, int& start, int& end,
                    std::vector<int>& recv_counts, std::vector<int>& displacements, std::vector<int>& local_data,
                    const boost::mpi::communicator& world, std::vector<int>& res_data) {
  int world_size = 0;
  int world_rank = 0;
  int res_size = 0;

  world_size = world.size();
  world_rank = world.rank();

  if (!res_data.empty()) {
    res_size = static_cast<int>(res_data.size());
  }

  n_of_send_elements = res_size / world_size;
  n_of_proc_with_extra_elements = res_size % world_size;

  for (int i = 0; i < world_size; i++) {
    start = i * n_of_send_elements + std::min(i, n_of_proc_with_extra_elements);
    end = start + n_of_send_elements + (i < n_of_proc_with_extra_elements ? 1 : 0);
    recv_counts[i] = end - start;
    displacements[i] = (i == 0) ? 0 : displacements[i - 1] + recv_counts[i - 1];
  }

  start = world_rank * n_of_send_elements + std::min(world_rank, n_of_proc_with_extra_elements);
  end = start + n_of_send_elements + (world_rank < n_of_proc_with_extra_elements ? 1 : 0);

  local_data.resize(end - start);
  for (int i = start; i < end; i++) {
    local_data[i - start] = res_data[i];
  }
}
}  // namespace
}  // namespace budazhapova_betcher_odd_even_merge_mpi
bool budazhapova_betcher_odd_even_merge_mpi::MergeSequential::PreProcessingImpl() {
  res_ = std::vector<int>(reinterpret_cast<int*>(task_data->inputs[0]),
                          reinterpret_cast<int*>(task_data->inputs[0]) + task_data->inputs_count[0]);
  return true;
}

bool budazhapova_betcher_odd_even_merge_mpi::MergeSequential::ValidationImpl() {
  return task_data->inputs_count[0] > 0;
}

bool budazhapova_betcher_odd_even_merge_mpi::MergeSequential::RunImpl() {
  budazhapova_betcher_odd_even_merge_mpi::RadixSort(res_);
  return true;
}

bool budazhapova_betcher_odd_even_merge_mpi::MergeSequential::PostProcessingImpl() {
  int* output = reinterpret_cast<int*>(task_data->outputs[0]);
  for (size_t i = 0; i < res_.size(); i++) {
    output[i] = res_[i];
  }
  return true;
}

bool budazhapova_betcher_odd_even_merge_mpi::MergeParallel::PreProcessingImpl() {
  if (world_.rank() == 0) {
    res_ = std::vector<int>(reinterpret_cast<int*>(task_data->inputs[0]),
                            reinterpret_cast<int*>(task_data->inputs[0]) + task_data->inputs_count[0]);
  }
  return true;
}

bool budazhapova_betcher_odd_even_merge_mpi::MergeParallel::ValidationImpl() {
  if (world_.rank() == 0) {
    return task_data->inputs_count[0] > 0;
  }
  return true;
}
bool budazhapova_betcher_odd_even_merge_mpi::MergeParallel::RunImpl() {
  std::vector<int> recv_counts(world_.size(), 0);
  std::vector<int> displacements(world_.size(), 0);

  broadcast(world_, res_, 0);

  int n_of_send_elements = 0;
  int n_of_proc_with_extra_elements = 0;
  int start = 0;
  int end = 0;

  DistributeData(n_of_send_elements, n_of_proc_with_extra_elements, start, end, recv_counts, displacements, local_res_,
                 world_, res_);

  int world_size = world_.size();
  if (world_size > 1) {
    for (int phase = 0; phase < world_size; phase++) {
      OddEvenSortPhase(phase, local_res_, world_);
    }
  } else {
    budazhapova_betcher_odd_even_merge_mpi::RadixSort(local_res_);
  }

  boost::mpi::gatherv(world_, local_res_.data(), static_cast<int>(local_res_.size()), res_.data(), recv_counts,
                      displacements, 0);

  return true;
}

bool budazhapova_betcher_odd_even_merge_mpi::MergeParallel::PostProcessingImpl() {
  if (world_.rank() == 0) {
    int* output = reinterpret_cast<int*>(task_data->outputs[0]);
    for (size_t i = 0; i < res_.size(); i++) {
      output[i] = res_[i];
    }
  }
  return true;
}
