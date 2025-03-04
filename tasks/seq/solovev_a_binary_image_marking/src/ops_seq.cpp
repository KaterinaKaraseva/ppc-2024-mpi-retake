#include <algorithm>
#include <queue>
#include <vector>

#include "seq/solovev_a_binary_image_marking/include/ops_sec.hpp"

bool solovev_a_binary_image_marking::ShouldProcess(int i, int j, const Matrix &data_tmp, const Matrix &labels_tmp,
                                                   int n_tmp) {
  return data_tmp[(i * n_tmp) + j] == 1 && labels_tmp[(i * n_tmp) + j] == 0;
}

bool solovev_a_binary_image_marking::IsValid(int x, int y, int m_tmp, int n_tmp) {
  return x >= 0 && x < m_tmp && y >= 0 && y < n_tmp;
}

void solovev_a_binary_image_marking::ProcessNeighbor(std::queue<Point> &q, int new_x, int new_y, Matrix &labels_tmp,
                                                     const Matrix &data_tmp, int label, int n_tmp) {
  int new_idx = (new_x * n_tmp) + new_y;
  if (data_tmp[new_idx] == 1 && labels_tmp[new_idx] == 0) {
    labels_tmp[new_idx] = label;
    q.push({new_x, new_y});
  }
}

void solovev_a_binary_image_marking::Bfs(int i, int j, int label, Matrix &labels_tmp, const Matrix &data_tmp, int m_tmp,
                                         int n_tmp, const Directions &directions) {
  std::queue<Point> q;
  q.push({i, j});
  labels_tmp[(i * n_tmp) + j] = label;

  while (!q.empty()) {
    Point current = q.front();
    q.pop();

    for (const Point &dir : directions) {
      int new_x = current.x + dir.x;
      int new_y = current.y + dir.y;

      if (IsValid(new_x, new_y, m_tmp, n_tmp)) {
        ProcessNeighbor(q, new_x, new_y, labels_tmp, data_tmp, label, n_tmp);
      }
    }
  }
}

bool solovev_a_binary_image_marking::TestTaskSequential::PreProcessingImpl() {
  int m_tmp = *reinterpret_cast<int *>(task_data->inputs[0]);
  int n_tmp = *reinterpret_cast<int *>(task_data->inputs[1]);

  std::vector<int> input_tmp;

  int *tmp_data = reinterpret_cast<int *>(task_data->inputs[2]);
  int tmp_size = static_cast<int>(task_data->inputs_count[2]);
  input_tmp.assign(tmp_data, tmp_data + tmp_size);

  data_.resize(tmp_size);
  labels_.resize(tmp_size);

  data_.assign(input_tmp.begin(), input_tmp.end());

  m_ = m_tmp;
  n_ = n_tmp;

  labels_.assign(m_ * n_, 0);

  return true;
}

bool solovev_a_binary_image_marking::TestTaskSequential::ValidationImpl() {
  int m_check = *reinterpret_cast<int *>(task_data->inputs[0]);
  int n_check = *reinterpret_cast<int *>(task_data->inputs[1]);

  std::vector<int> input_check;

  int *input_check_data = reinterpret_cast<int *>(task_data->inputs[2]);
  int input_check_size = static_cast<int>(task_data->inputs_count[2]);
  input_check.assign(input_check_data, input_check_data + input_check_size);

  return (m_check > 0 && n_check > 0 && !input_check.empty());
}

bool solovev_a_binary_image_marking::TestTaskSequential::RunImpl() {
  Directions directions = {{.x = -1, .y = 0}, {.x = 1, .y = 0}, {.x = 0, .y = -1}, {.x = 0, .y = 1}};
  int label = 1;

  for (int i = 0; i < m_; ++i) {
    for (int j = 0; j < n_; ++j) {
      if (ShouldProcess(i, j, data_, labels_, n_)) {
        Bfs(i, j, label, labels_, data_, m_, n_, directions);
        ++label;
      }
    }
  }

  return true;
}

bool solovev_a_binary_image_marking::TestTaskSequential::PostProcessingImpl() {
  int *output = reinterpret_cast<int *>(task_data->outputs[0]);
  std::ranges::copy(labels_, output);

  return true;
}