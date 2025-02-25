#include "mpi/karaseva_e_binaryimage/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <unordered_map>
#include <vector>

// Function to get the root of a label with path compression
int karaseva_e_binaryimage_mpi::TestTaskMPI::GetRootLabel(std::unordered_map<int, int>& label_parent, int label) {
  if (!label_parent.contains(label)) {
    label_parent[label] = label;  // If label is not in the set, it is its own parent
  } else if (label_parent[label] != label) {
    label_parent[label] = GetRootLabel(label_parent, label_parent[label]);  // Path compression
  }
  return label_parent[label];
}

// Function to union two labels
void karaseva_e_binaryimage_mpi::TestTaskMPI::UnionLabels(std::unordered_map<int, int>& label_parent, int label1,
                                                          int label2) {
  int root1 = GetRootLabel(label_parent, label1);
  int root2 = GetRootLabel(label_parent, label2);
  if (root1 != root2) {
    label_parent[root2] = root1;  // Union: make root1 the parent of root2
  }
}

// Function to process neighbors of a pixel and add them to a list
void karaseva_e_binaryimage_mpi::TestTaskMPI::ProcessNeighbors(int x, int y, int rows, int cols,
                                                               const std::vector<int>& labeled_image,
                                                               std::vector<int>& neighbors) {
  int dx[] = {-1, 0, -1};
  int dy[] = {0, -1, 1};

  for (int i = 0; i < 3; ++i) {
    int nx = x + dx[i];
    int ny = y + dy[i];
    // Check if the neighbor is within bounds and has a label >= 2 (indicating it's part of a label)
    if (nx >= 0 && nx < rows && ny >= 0 && ny < cols && labeled_image[(nx * cols) + ny] >= 2) {
      neighbors.push_back(labeled_image[(nx * cols) + ny]);  // Add the label of the neighbor
    }
  }
}

// Function to assign label to a pixel and perform union of labels
void karaseva_e_binaryimage_mpi::TestTaskMPI::AssignLabelToPixel(int pos, std::vector<int>& labeled_image,
                                                                 std::unordered_map<int, int>& label_parent,
                                                                 int& label_counter,
                                                                 const std::vector<int>& neighbors) {
  if (neighbors.empty()) {
    labeled_image[pos] = label_counter++;  // No neighbors, assign a new label
  } else {
    int min_neighbor = *std::ranges::min_element(neighbors);  // Find the smallest label from neighbors
    labeled_image[pos] = min_neighbor;
    // Perform union with all neighbors to ensure they share the same label
    for (int n : neighbors) {
      UnionLabels(label_parent, min_neighbor, n);
    }
  }
}

// Main labeling function
void karaseva_e_binaryimage_mpi::TestTaskMPI::Labeling(std::vector<int>& image, std::vector<int>& labeled_image,
                                                       int rows, int cols, int min_label,
                                                       std::unordered_map<int, int>& label_parent, int start_row,
                                                       int end_row) {
  int label_counter = min_label;

  if (start_row >= end_row) {
    return;
  }

  for (int x = start_row; x < end_row; ++x) {
    for (int y = 0; y < cols; ++y) {
      int pos = (x * cols) + y;
      // Skip pixels that are background or already labeled
      if (image[pos] == 0 || labeled_image[pos] >= 2) {
        std::vector<int> neighbors;
        ProcessNeighbors(x, y, rows, cols, labeled_image, neighbors);  // Find neighbors
        AssignLabelToPixel(pos, labeled_image, label_parent, label_counter,
                           neighbors);  // Assign label and union with neighbors
      }
    }
  }

  // Second pass: Ensure all pixels with labels are assigned to their root label
  for (int x = start_row; x < end_row; ++x) {
    for (int y = 0; y < cols; ++y) {
      int pos = (x * cols) + y;
      if (labeled_image[pos] >= 2) {
        labeled_image[pos] = GetRootLabel(label_parent, labeled_image[pos]);  // Path compression
      }
    }
  }
}

bool karaseva_e_binaryimage_mpi::TestTaskMPI::PreProcessingImpl() {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  bool is_root = (rank == 0);

  if (is_root && (task_data->inputs.empty() || task_data->inputs_count.empty())) {
    std::cerr << "[ERROR] Root process has empty inputs or inputs_count.\n";
    return false;
  }

  unsigned int rows = 0;
  unsigned int cols = 0;
  if (is_root) {
    rows = task_data->inputs_count[0];
    cols = task_data->inputs_count[1];
  }

  // Broadcasting the size and image dimensions to all processes
  MPI_Bcast(&rows, 1, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
  MPI_Bcast(&cols, 1, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
  std::cout << "[Rank " << rank << "] Received image dimensions: " << rows << "x" << cols << '\n';

  int input_size = static_cast<int>(rows * cols);

  // Broadcasting the image data
  if (is_root) {
    auto* in_ptr = reinterpret_cast<int*>(task_data->inputs[0]);
    input_ = std::vector<int>(in_ptr, in_ptr + input_size);
    std::cout << "[Rank 0] Broadcasting image data of size: " << input_size << '\n';
  } else {
    input_.resize(input_size);
  }

  int result = MPI_Bcast(input_.data(), input_size, MPI_INT, 0, MPI_COMM_WORLD);
  if (result != MPI_SUCCESS) {
    std::cerr << "[Rank " << rank << "] Error broadcasting image data.\n";
    return false;
  }

  std::cout << "[Rank " << rank << "] Image data broadcasted successfully.\n";

  return true;
}

bool karaseva_e_binaryimage_mpi::TestTaskMPI::ValidationImpl() {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  unsigned int input_count = 0;
  unsigned int output_count = 0;

  if (!task_data->inputs_count.empty()) {
    input_count = task_data->inputs_count[0];
    output_count = task_data->outputs_count[0];
  }

  if (input_count == 0 || output_count == 0 || input_count != output_count) {
    std::cerr << "[ERROR] Invalid input/output dimensions.\n";
    return false;
  }

  return true;
}

bool karaseva_e_binaryimage_mpi::TestTaskMPI::RunImpl() {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  unsigned int rows = task_data->inputs_count[0];
  unsigned int cols = task_data->inputs_count[1];
  int num_processes = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &num_processes);

  unsigned int local_rows = rows / num_processes;

  std::unordered_map<int, int> label_parent;
  local_labeled_image_.resize(local_rows * cols, 0);
  std::vector<int> neighbors;

  // Perform labeling for the local region assigned to the current process
  Labeling(input_, local_labeled_image_, rows, cols, 2, label_parent, rank * local_rows, (rank + 1) * local_rows);

  int result = MPI_Gather(local_labeled_image_.data(), static_cast<int>(local_rows * cols), MPI_INT,
                          task_data->outputs[0], static_cast<int>(local_rows * cols), MPI_INT, 0, MPI_COMM_WORLD);

  if (result != MPI_SUCCESS) {
    std::cerr << "[Rank " << rank << "] Error gathering labeled image data.\n";
    return false;
  }

  return true;
}

bool karaseva_e_binaryimage_mpi::TestTaskMPI::PostProcessingImpl() {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if (rank == 0) {
    auto* output_ptr = reinterpret_cast<int*>(task_data->outputs[0]);
    std::ranges::copy(local_labeled_image_.begin(), local_labeled_image_.end(), output_ptr);
  }

  return true;
}