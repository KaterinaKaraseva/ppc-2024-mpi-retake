#include <gtest/gtest.h>

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <cmath>
#include <cstdint>
#include <memory>
#include <numeric>
#include <vector>

#include "core/task/include/task.hpp"
#include "mpi/mezhuev_m_lattice_torus/include/mpi.hpp"

TEST(mezhuev_m_lattice_torus_mpi, DataTransferTest) {
  boost::mpi::communicator world;
  if (world.size() < 4) {
    GTEST_SKIP();
  }

  std::vector<uint8_t> input_data(4);
  std::iota(input_data.begin(), input_data.end(), 9);
  std::vector<uint8_t> output_data(4);

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(input_data.data());
  task_data->inputs_count.emplace_back(input_data.size());
  task_data->outputs.emplace_back(output_data.data());
  task_data->outputs_count.emplace_back(output_data.size());

  mezhuev_m_lattice_torus_mpi::GridTorusTopologyParallel task(task_data);

  EXPECT_TRUE(task.ValidationImpl());
  EXPECT_TRUE(task.PreProcessingImpl());
}

TEST(mezhuev_m_lattice_torus_mpi, TestPreProcessing) {
  boost::mpi::communicator world;

  std::vector<uint8_t> input_data(4);
  std::iota(input_data.begin(), input_data.end(), 9);
  std::vector<uint8_t> output_data(4);

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(input_data.data());
  task_data->inputs_count.emplace_back(input_data.size());
  task_data->outputs.emplace_back(output_data.data());
  task_data->outputs_count.emplace_back(output_data.size());

  mezhuev_m_lattice_torus_mpi::GridTorusTopologyParallel task(task_data);

  EXPECT_TRUE(task.PreProcessingImpl());
}

TEST(mezhuev_m_lattice_torus_mpi, TestLargeGridProcessing) {
  boost::mpi::communicator world;
  if (world.size() < 16) {
    GTEST_SKIP();
  }

  std::vector<uint8_t> input_data(16);
  std::iota(input_data.begin(), input_data.end(), 9);
  std::vector<uint8_t> output_data(16);

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(input_data.data());
  task_data->inputs_count.emplace_back(input_data.size());
  task_data->outputs.emplace_back(output_data.data());
  task_data->outputs_count.emplace_back(output_data.size());

  mezhuev_m_lattice_torus_mpi::GridTorusTopologyParallel task(task_data);

  EXPECT_TRUE(task.ValidationImpl());
  EXPECT_TRUE(task.PreProcessingImpl());
  EXPECT_TRUE(task.RunImpl());
  EXPECT_TRUE(task.PostProcessingImpl());
}

TEST(mezhuev_m_lattice_torus_mpi, TestIterationOnMaxGridSize) {
  boost::mpi::communicator world;
  if (world.size() < 16) {
    GTEST_SKIP();
  }

  int max_size = 256;
  std::vector<uint8_t> input_data(max_size);
  std::iota(input_data.begin(), input_data.end(), 9);
  std::vector<uint8_t> output_data(max_size);

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(input_data.data());
  task_data->inputs_count.emplace_back(input_data.size());
  task_data->outputs.emplace_back(output_data.data());
  task_data->outputs_count.emplace_back(output_data.size());

  mezhuev_m_lattice_torus_mpi::GridTorusTopologyParallel task(task_data);

  EXPECT_TRUE(task.ValidationImpl());
  EXPECT_TRUE(task.PreProcessingImpl());
  EXPECT_TRUE(task.RunImpl());
  EXPECT_TRUE(task.PostProcessingImpl());
}

TEST(mezhuev_m_lattice_torus_mpi, InvalidGridDimensions) {
  boost::mpi::communicator world;
  if (world.size() == 6) {
    std::vector<uint8_t> input_data(4);
    std::iota(input_data.begin(), input_data.end(), 9);
    std::vector<uint8_t> output_data(4);

    auto task_data = std::make_shared<ppc::core::TaskData>();
    task_data->inputs.emplace_back(input_data.data());
    task_data->inputs_count.emplace_back(input_data.size());
    task_data->outputs.emplace_back(output_data.data());
    task_data->outputs_count.emplace_back(output_data.size());

    mezhuev_m_lattice_torus_mpi::GridTorusTopologyParallel task(task_data);
    EXPECT_FALSE(task.ValidationImpl());
  }
}

TEST(mezhuev_m_lattice_torus_mpi, TestPreProcessingSuccess) {
  boost::mpi::communicator world;

  std::vector<uint8_t> input_data(4);
  std::iota(input_data.begin(), input_data.end(), 9);
  std::vector<uint8_t> output_data(4);

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(input_data.data());
  task_data->inputs_count.emplace_back(input_data.size());
  task_data->outputs.emplace_back(output_data.data());
  task_data->outputs_count.emplace_back(output_data.size());

  mezhuev_m_lattice_torus_mpi::GridTorusTopologyParallel task(task_data);

  EXPECT_TRUE(task.PreProcessingImpl());
}

TEST(mezhuev_m_lattice_torus_mpi, RunImpl_SingleProcess) {
  boost::mpi::communicator world;
  if (world.size() != 1) {
    GTEST_SKIP();
  }

  std::vector<uint8_t> input_data(10);
  std::iota(input_data.begin(), input_data.end(), 100);
  std::vector<uint8_t> output_data(10, 0);

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(input_data.data());
  task_data->inputs_count.emplace_back(input_data.size());
  task_data->outputs.emplace_back(output_data.data());
  task_data->outputs_count.emplace_back(output_data.size());

  mezhuev_m_lattice_torus_mpi::GridTorusTopologyParallel task(task_data);
  EXPECT_TRUE(task.ValidationImpl());
  EXPECT_TRUE(task.PreProcessingImpl());
  EXPECT_TRUE(task.RunImpl());
  EXPECT_TRUE(task.PostProcessingImpl());
  EXPECT_EQ(input_data, output_data);
}

TEST(mezhuev_m_lattice_torus_mpi, FullPipeline_SmallGrid) {
  boost::mpi::communicator world;
  if (world.size() < 2) {
    GTEST_SKIP();
  }

  std::vector<uint8_t> input_data(8, 55);
  std::vector<uint8_t> output_data(8, 0);

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(input_data.data());
  task_data->inputs_count.emplace_back(input_data.size());
  task_data->outputs.emplace_back(output_data.data());
  task_data->outputs_count.emplace_back(output_data.size());

  mezhuev_m_lattice_torus_mpi::GridTorusTopologyParallel task(task_data);
  EXPECT_TRUE(task.ValidationImpl());
  EXPECT_TRUE(task.PreProcessingImpl());
  EXPECT_TRUE(task.RunImpl());
  EXPECT_TRUE(task.PostProcessingImpl());
  EXPECT_EQ(input_data, output_data);
}

TEST(mezhuev_m_lattice_torus_mpi, PostProcessing_MultipleBuffers) {
  boost::mpi::communicator world;

  std::vector<uint8_t> input_data1(5);
  std::iota(input_data1.begin(), input_data1.end(), 10);
  std::vector<uint8_t> output_data1 = input_data1;

  std::vector<uint8_t> input_data2(5);
  std::iota(input_data2.begin(), input_data2.end(), 20);
  std::vector<uint8_t> output_data2 = input_data2;

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.push_back(input_data1.data());
  task_data->inputs.push_back(input_data2.data());
  task_data->inputs_count.push_back(input_data1.size());
  task_data->inputs_count.push_back(input_data2.size());

  task_data->outputs.push_back(output_data1.data());
  task_data->outputs.push_back(output_data2.data());
  task_data->outputs_count.push_back(output_data1.size());
  task_data->outputs_count.push_back(output_data2.size());

  mezhuev_m_lattice_torus_mpi::GridTorusTopologyParallel task(task_data);

  EXPECT_TRUE(task.PostProcessingImpl());

  output_data1[0] = 0;
  EXPECT_FALSE(task.PostProcessingImpl());
}
