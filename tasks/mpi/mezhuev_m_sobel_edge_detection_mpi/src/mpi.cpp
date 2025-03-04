#include "mpi/mezhuev_m_sobel_edge_detection_mpi/include/mpi.hpp"

#include <algorithm>
#include <boost/mpi/communicator.hpp>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace mezhuev_m_sobel_edge_detection_mpi {

bool SobelEdgeDetection::PreProcessingImpl() {
  if (!task_data || task_data->inputs.empty() || task_data->outputs.empty()) {
    return false;
  }

  if (task_data->inputs_count.empty() || task_data->inputs_count[0] == 0 || task_data->outputs_count.empty() ||
      task_data->outputs_count[0] == 0) {
    return false;
  }

  if (task_data->inputs[0] == nullptr || task_data->outputs[0] == nullptr) {
    return false;
  }

  size_t data_size = task_data->inputs_count[0];
  auto width = static_cast<size_t>(std::sqrt(data_size));
  size_t height = width;

  if (width < 3 || height < 3) {
    return false;
  }

  gradient_x_.resize(data_size);
  gradient_y_.resize(data_size);
  return true;
}

bool SobelEdgeDetection::ValidationImpl() {
  if (!task_data || task_data->inputs.empty() || task_data->outputs.empty()) {
    return false;
  }
  if (task_data->inputs.size() != 1 || task_data->outputs.size() != 1) {
    return false;
  }
  if (task_data->inputs[0] == nullptr || task_data->outputs[0] == nullptr) {
    return false;
  }
  if (task_data->inputs_count.empty() || task_data->outputs_count.empty()) {
    return false;
  }
  return task_data->inputs_count[0] == task_data->outputs_count[0];
}

bool SobelEdgeDetection::RunImpl() {
  if (!task_data || task_data->inputs.empty() || task_data->outputs.empty() || task_data->inputs_count.empty() ||
      task_data->outputs_count.empty()) {
    return false;
  }
  int rank = world_.rank();
  int size = world_.size();
  auto width = static_cast<size_t>(std::sqrt(task_data->inputs_count[0]));
  auto height = width;
  if (height < 3 || width < 3) {
    return false;
  }

  uint8_t* input = task_data->inputs[0];
  uint8_t* output = task_data->outputs[0];

  size_t rows_per_proc = height / size;
  size_t extra_rows = height % size;
  size_t start_row = (rank * rows_per_proc) + std::min(rank, static_cast<int>(extra_rows));
  size_t end_row = ((rank + 1) * rows_per_proc) + std::min(rank + 1, static_cast<int>(extra_rows));

  auto exchange_boundaries = [&](int rank, int size) {
    if (rank > 0) {
      world_.send(rank - 1, 0, input + (start_row * width), static_cast<int>(width));
      world_.recv(rank - 1, 0, input + ((start_row - 1) * width), static_cast<int>(width));
    }
    if (rank < size - 1) {
      world_.recv(rank + 1, 0, input + (end_row * width), static_cast<int>(width));
      world_.send(rank + 1, 0, input + ((end_row - 1) * width), static_cast<int>(width));
    }
  };
  exchange_boundaries(rank, size);

  auto apply_sobel = [&](size_t y, size_t x) -> uint8_t {
    static constexpr int kSobelX[3][3] = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};
    static constexpr int kSobelY[3][3] = {{1, 2, 1}, {0, 0, 0}, {-1, -2, -1}};

    int gx = 0;
    int gy = 0;
    for (int ky = -1; ky <= 1; ++ky) {
      for (int kx = -1; kx <= 1; ++kx) {
        uint8_t pixel = input[((y + ky) * width) + (x + kx)];
        gx += kSobelX[ky + 1][kx + 1] * pixel;
        gy += kSobelY[ky + 1][kx + 1] * pixel;
      }
    }
    return static_cast<uint8_t>(std::min(std::sqrt((gx * gx) + (gy * gy)), 255.0));
  };

  for (size_t y = std::max(start_row, size_t(1)); y < std::min(end_row, height - 1); ++y) {
    for (size_t x = 1; x < width - 1; ++x) {
      output[(y * width) + x] = apply_sobel(y, x);
    }
  }

  if (rank == 0) {
    for (int i = 1; i < size; ++i) {
      size_t ws = (i * rows_per_proc) + std::min(i, static_cast<int>(extra_rows));
      size_t we = ((i + 1) * rows_per_proc) + std::min(i + 1, static_cast<int>(extra_rows));
      world_.recv(i, 0, output + (ws * width), static_cast<int>((we - ws) * width));
    }
  } else {
    world_.send(0, 0, output + (start_row * width), static_cast<int>((end_row - start_row) * width));
  }

  return true;
}

bool SobelEdgeDetection::PostProcessingImpl() {
  if (!task_data || task_data->outputs.empty() || task_data->outputs[0] == nullptr) {
    return false;
  }

  size_t output_size = task_data->outputs_count[0];

  for (size_t i = 0; i < output_size; ++i) {
    if (task_data->outputs[0][i] > 0) {
      return true;
    }
  }

  return true;
}

}  // namespace mezhuev_m_sobel_edge_detection_mpi
