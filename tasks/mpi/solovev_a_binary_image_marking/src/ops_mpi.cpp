#include "mpi/solovev_a_binary_image_marking/include/ops_mpi.hpp"

#include <algorithm>
#include <boost/mpi/collectives.hpp>
#include <boost/mpi/collectives/broadcast.hpp>
#include <boost/mpi/collectives/gatherv.hpp>
#include <boost/mpi/collectives/scatterv.hpp>
#include <boost/mpi/communicator.hpp>
#include <queue>
#include <unordered_map>
#include <utility>
#include <vector>

bool solovev_a_binary_image_marking::ShouldProcess(int i, int j, const Matrix& data_tmp, const Matrix& labels_tmp,
                                                   int n_tmp) {
  return data_tmp[(i * n_tmp) + j] == 1 && labels_tmp[(i * n_tmp) + j] == 0;
}

bool solovev_a_binary_image_marking::IsValid(int x, int y, int m_tmp, int n_tmp) {
  return x >= 0 && x < m_tmp && y >= 0 && y < n_tmp;
}

void solovev_a_binary_image_marking::ProcessNeighbor(std::queue<Point>& q, int new_x, int new_y, Matrix& labels_tmp,
                                                     const Matrix& data_tmp, int label, int n_tmp) {
  int new_idx = (new_x * n_tmp) + new_y;
  if (data_tmp[new_idx] == 1 && labels_tmp[new_idx] == 0) {
    labels_tmp[new_idx] = label;
    q.push({new_x, new_y});
  }
}

void solovev_a_binary_image_marking::Bfs(int i, int j, int label, Matrix& labels_tmp, const Matrix& data_tmp, int m_tmp,
                                         int n_tmp, const Directions& directions) {
  std::queue<Point> q;
  q.push({i, j});
  labels_tmp[(i * n_tmp) + j] = label;

  while (!q.empty()) {
    Point current = q.front();
    q.pop();

    for (const Point& dir : directions) {
      int new_x = current.x + dir.x;
      int new_y = current.y + dir.y;

      if (IsValid(new_x, new_y, m_tmp, n_tmp)) {
        ProcessNeighbor(q, new_x, new_y, labels_tmp, data_tmp, label, n_tmp);
      }
    }
  }
}

bool solovev_a_binary_image_marking::TestMPITaskSequential::PreProcessingImpl() {
  int m_tmp = *reinterpret_cast<int*>(task_data->inputs[0]);
  int n_tmp = *reinterpret_cast<int*>(task_data->inputs[1]);
  auto* tmp_data = reinterpret_cast<int*>(task_data->inputs[2]);
  data_seq_.assign(tmp_data, tmp_data + task_data->inputs_count[2]);
  m_seq_ = m_tmp;
  n_seq_ = n_tmp;
  labels_seq_.resize(m_seq_ * n_seq_);
  return true;
}

bool solovev_a_binary_image_marking::TestMPITaskSequential::ValidationImpl() {
  int rows_check = *reinterpret_cast<int*>(task_data->inputs[0]);
  int coloms_check = *reinterpret_cast<int*>(task_data->inputs[1]);

  std::vector<int> input_check;

  int* input_check_data = reinterpret_cast<int*>(task_data->inputs[2]);
  int input_check_size = static_cast<int>(task_data->inputs_count[2]);
  input_check.assign(input_check_data, input_check_data + input_check_size);

  return (rows_check > 0 && coloms_check > 0 && !input_check.empty());
}

bool solovev_a_binary_image_marking::TestMPITaskSequential::RunImpl() {
  Directions directions = {{.x = -1, .y = 0}, {.x = 1, .y = 0}, {.x = 0, .y = -1}, {.x = 0, .y = 1}};
  int label = 1;

  for (int i = 0; i < m_seq_; ++i) {
    for (int j = 0; j < n_seq_; ++j) {
      if (ShouldProcess(i, j, data_seq_, labels_seq_, n_seq_)) {
        Bfs(i, j, label, labels_seq_, data_seq_, m_seq_, n_seq_, directions);
        ++label;
      }
    }
  }
  return true;
}

bool solovev_a_binary_image_marking::TestMPITaskSequential::PostProcessingImpl() {
  int* output = reinterpret_cast<int*>(task_data->outputs[0]);
  std::ranges::copy(labels_seq_, output);

  return true;
}

bool solovev_a_binary_image_marking::TestMPITaskParallel::PreProcessingImpl() {
  if (world_.rank() == 0) {
    int m_count = *reinterpret_cast<int*>(task_data->inputs[0]);
    int n_count = *reinterpret_cast<int*>(task_data->inputs[1]);
    auto* data_tmp = reinterpret_cast<int*>(task_data->inputs[2]);
    data_.assign(data_tmp, data_tmp + task_data->inputs_count[2]);
    m_ = m_count;
    n_ = n_count;
  }
  return true;
}

bool solovev_a_binary_image_marking::TestMPITaskParallel::ValidationImpl() {
  if (world_.rank() == 0) {
    int m_check = *reinterpret_cast<int*>(task_data->inputs[0]);
    int n_check = *reinterpret_cast<int*>(task_data->inputs[1]);
    int input_check_size = static_cast<int>(task_data->inputs_count[2]);
    return (m_check > 0 && n_check > 0 && input_check_size > 0);
  }
  return true;
}

bool solovev_a_binary_image_marking::TestMPITaskParallel::IsValidMPI(int nr, int local_pixel_count, int nc) const {
  return (nr >= 0 && nr < (local_pixel_count / n_) && nc >= 0 && nc < n_);
}

void solovev_a_binary_image_marking::TestMPITaskParallel::BFSCheck(const int* p_local_image, int curr_label,
                                                                   int* p_local_labels, int local_pixel_count, Point cp,
                                                                   std::queue<Point> bfs_queue) const {
  std::vector<Point> directions = {{.x = -1, .y = 0}, {.x = 1, .y = 0}, {.x = 0, .y = -1}, {.x = 0, .y = 1}};
  for (const auto& step : directions) {
    int nr = cp.x + step.x;
    int nc = cp.y + step.y;
    if (IsValidMPI(nr, local_pixel_count, nc)) {
      int ni = (nr * n_) + nc;
      if (p_local_image[ni] == 1 && p_local_labels[ni] == 0) {
        p_local_labels[ni] = curr_label;
        bfs_queue.push({nr, nc});
      }
    }
  }
}

void solovev_a_binary_image_marking::TestMPITaskParallel::MPIBfs(int* p_local_image, int local_pixel_count,
                                                                 int* p_local_labels, int curr_label) {
  auto to_coordinates = [this](int idx) -> std::pair<int, int> { return {idx / this->n_, idx % this->n_}; };
  for (int i = 0; i < local_pixel_count; ++i) {
    if (p_local_image[i] == 1 && p_local_labels[i] == 0) {
      std::queue<Point> bfs_queue;
      auto coord = to_coordinates(i);
      bfs_queue.push({coord.first, coord.second});
      p_local_labels[i] = curr_label;
      while (!bfs_queue.empty()) {
        Point cp = bfs_queue.front();
        bfs_queue.pop();
        BFSCheck(p_local_image, curr_label, p_local_labels, local_pixel_count, cp, bfs_queue);
      }
      ++curr_label;
    }
  }
}
namespace {
void MakeNorm(int total, std::vector<int> parent, int* p_global) {
  auto find_rep = [&parent](int x) -> int {
    while (x != parent[x]) {
      x = parent[x] = parent[parent[x]];
    }
    return x;
  };

  for (int i = 0; i < total; ++i) {
    if (p_global[i] != 0) {
      p_global[i] = find_rep(p_global[i]);
    }
  }
  std::unordered_map<int, int> norm;
  int next_label = 1;
  for (int i = 0; i < total; ++i) {
    if (p_global[i] != 0) {
      int rep = p_global[i];
      if (norm.find(rep) == norm.end()) {
        norm[rep] = next_label++;
      }
      p_global[i] = norm[rep];
    }
  }
}
}  // namespace

std::vector<int> solovev_a_binary_image_marking::TestMPITaskParallel::MakeMPIResult(
    std::vector<int> global_labels) const {
  int total = m_ * n_;
  int* p_global = global_labels.data();
  std::vector<int> parent(total + 1);
  for (int i = 1; i <= total; ++i) {
    parent[i] = i;
  }
  auto find_rep = [&parent](int x) -> int {
    while (x != parent[x]) {
      x = parent[x] = parent[parent[x]];
    }
    return x;
  };
  auto union_rep = [&find_rep, &parent](int a, int b) {
    int ra = find_rep(a);
    int rb = find_rep(b);
    if (ra != rb) {
      int new_rep = (ra < rb) ? ra : rb;
      int obsolete = (ra < rb) ? rb : ra;
      parent[obsolete] = new_rep;
    }
  };
  for (int row = 0; row < m_; ++row) {
    for (int col = 0; col < n_; ++col) {
      int idx = (row * n_) + col;
      if (p_global[idx] == 0) {
        continue;
      }
      int label = p_global[idx];
      if (col > 0 && p_global[(row * n_ + col) - 1] != 0) {
        union_rep(label, p_global[(row * n_) + col - 1]);
      }

      if (row > 0 && p_global[((row - 1) * n_) + col] != 0) {
        union_rep(label, p_global[((row - 1) * n_) + col]);
      }
    }
  }

  MakeNorm(total, parent, p_global);

  return global_labels;
}

bool solovev_a_binary_image_marking::TestMPITaskParallel::RunImpl() {
  boost::mpi::broadcast(world_, m_, 0);
  boost::mpi::broadcast(world_, n_, 0);

  int proc_rank = world_.rank();
  int proc_count = world_.size();

  std::vector<int> counts(proc_count, 0);
  std::vector<int> displacements(proc_count, 0);

  int current_row_offset = 0;

  for (int proc = 0; proc < proc_count; ++proc) {
    int proc_rows = m_ / proc_count;
    counts[proc] = proc_rows * n_;
    displacements[proc] = current_row_offset * n_;
    current_row_offset += proc_rows;
    if (proc == proc_count - 1) {
      counts[proc] += (n_ * m_ - (m_ / proc_count) * n_ * proc_count);
    }
  }

  int local_pixel_count = counts[proc_rank];
  std::vector<int> local_image(local_pixel_count);
  boost::mpi::scatterv(world_, data_.data(), counts, displacements, local_image.data(), local_pixel_count, 0);
  std::vector<int> local_labels(local_pixel_count, 0);
  int base_label = displacements[proc_rank] + 1;
  int curr_label = base_label;
  int* p_local_image = local_image.data();
  int* p_local_labels = local_labels.data();

  MPIBfs(p_local_image, local_pixel_count, p_local_labels, curr_label);

  std::vector<int> global_labels;

  if (proc_rank == 0) {
    global_labels.resize(m_ * n_);
  }

  boost::mpi::gatherv(world_, local_labels, global_labels.data(), counts, displacements, 0);

  if (proc_rank == 0) {
    labels_ = MakeMPIResult(global_labels);
  }
  return true;
}

bool solovev_a_binary_image_marking::TestMPITaskParallel::PostProcessingImpl() {
  if (world_.rank() == 0) {
    int* output = reinterpret_cast<int*>(task_data->outputs[0]);
    std::ranges::copy(labels_, output);
  }
  return true;
}