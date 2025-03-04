#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <random>
#include <vector>

#include "core/task/include/task.hpp"
#include "mpi/komshina_d_num_of_alternating_signs_of_values/include/ops_mpi.hpp"

namespace {
std::vector<int> GenerateRandomVector(size_t size, int min_val = -100, int max_val = 100) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<int> dis(min_val, max_val);
  std::vector<int> vec(size);
  for (auto &v : vec) {
    v = dis(gen);
  }
  return vec;
}

int CountSignAlternations(const std::vector<int> &vec) {
  if (vec.size() < 2) {
    return 0;
  }
  int count = 0;
  for (size_t i = 1; i < vec.size(); ++i) {
    if (vec[i - 1] * vec[i] < 0) {
      ++count;
    }
  }
  return count;
}
}  // namespace

TEST(komshina_d_num_of_alternations_signs_mpi, NormalCase) {
  boost::mpi::communicator world;

  std::vector<int> in = {1, -1, 1, -1};
  std::vector<int32_t> out(1, 0);

  auto task_data_mpi = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    task_data_mpi->inputs_count.emplace_back(in.size());
    task_data_mpi->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    task_data_mpi->outputs_count.emplace_back(out.size());
  }

  komshina_d_num_of_alternations_signs_mpi::TestTaskMPI test_task_mpi(task_data_mpi);
  ASSERT_EQ(test_task_mpi.Validation(), true);
  test_task_mpi.PreProcessing();
  test_task_mpi.Run();
  test_task_mpi.PostProcessing();

  if (world.rank() == 0) {
    ASSERT_EQ(out[0], 3);
  }
}

TEST(komshina_d_num_of_alternations_signs_mpi, NormalCase2) {
  boost::mpi::communicator world;

  std::vector<int> in = {1, -2, 3, 4, 5, 6, -10, -5, 11, -1};
  std::vector<int32_t> out(1, 0);

  auto task_data_mpi = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    task_data_mpi->inputs_count.emplace_back(in.size());
    task_data_mpi->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    task_data_mpi->outputs_count.emplace_back(out.size());
  }

  komshina_d_num_of_alternations_signs_mpi::TestTaskMPI test_task_mpi(task_data_mpi);
  ASSERT_EQ(test_task_mpi.Validation(), true);
  test_task_mpi.PreProcessing();
  test_task_mpi.Run();
  test_task_mpi.PostProcessing();

  if (world.rank() == 0) {
    ASSERT_EQ(out[0], 5);
  }
}

TEST(komshina_d_num_of_alternations_signs_mpi, NCase) {
  boost::mpi::communicator world;

  std::vector<int> in = {1, -1, -1, -1, 1};
  std::vector<int32_t> out(1, 0);

  auto task_data_mpi = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    task_data_mpi->inputs_count.emplace_back(in.size());
    task_data_mpi->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    task_data_mpi->outputs_count.emplace_back(out.size());
  }

  komshina_d_num_of_alternations_signs_mpi::TestTaskMPI test_task_mpi(task_data_mpi);
  ASSERT_EQ(test_task_mpi.Validation(), true);
  test_task_mpi.PreProcessing();
  test_task_mpi.Run();
  test_task_mpi.PostProcessing();

  if (world.rank() == 0) {
    ASSERT_EQ(out[0], 2);
  }
}

TEST(komshina_d_num_of_alternations_signs_mpi, InvalidInputs) {
  boost::mpi::communicator world;

  std::vector<int> in = {1};
  std::vector<int32_t> out(1, 0);

  auto task_data_mpi = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    task_data_mpi->inputs_count.emplace_back(in.size());
    task_data_mpi->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    task_data_mpi->outputs_count.emplace_back(out.size());
  }

  komshina_d_num_of_alternations_signs_mpi::TestTaskMPI test_task_mpi(task_data_mpi);

  if (world.rank() == 0) {
    ASSERT_EQ(test_task_mpi.Validation(), false);
  }
}

TEST(komshina_d_num_of_alternations_signs_mpi, MixedSigns) {
  boost::mpi::communicator world;

  std::vector<int> in = {1, 1, 1, 1, 1, 1};
  std::vector<int32_t> out(1, 0);

  auto task_data_mpi = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    task_data_mpi->inputs_count.emplace_back(in.size());
    task_data_mpi->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    task_data_mpi->outputs_count.emplace_back(out.size());
  }

  komshina_d_num_of_alternations_signs_mpi::TestTaskMPI test_task_mpi(task_data_mpi);
  ASSERT_EQ(test_task_mpi.Validation(), true);
  test_task_mpi.PreProcessing();
  test_task_mpi.Run();
  test_task_mpi.PostProcessing();

  if (world.rank() == 0) {
    ASSERT_EQ(out[0], 0);
  }
}

TEST(komshina_d_num_of_alternations_signs_mpi, InvalidOutputSize) {
  boost::mpi::communicator world;

  std::vector<int> in = {1, -1, 1, -1, 1};
  std::vector<int32_t> out(0, 0);

  auto task_data_mpi = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    task_data_mpi->inputs_count.emplace_back(in.size());
    task_data_mpi->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    task_data_mpi->outputs_count.emplace_back(out.size());
  }

  komshina_d_num_of_alternations_signs_mpi::TestTaskMPI test_task_mpi(task_data_mpi);

  if (world.rank() == 0) {
    ASSERT_EQ(test_task_mpi.Validation(), false);
  }
}

TEST(komshina_d_num_of_alternations_signs_mpi, EmptyLocalInput) {
  boost::mpi::communicator world;

  std::vector<int> in = {1, -1};
  std::vector<int32_t> out(1, 0);

  auto task_data_mpi = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    task_data_mpi->inputs_count.emplace_back(in.size());
    task_data_mpi->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    task_data_mpi->outputs_count.emplace_back(out.size());
  }

  komshina_d_num_of_alternations_signs_mpi::TestTaskMPI test_task_mpi(task_data_mpi);

  ASSERT_EQ(test_task_mpi.Validation(), true);
  test_task_mpi.PreProcessing();
  test_task_mpi.Run();
  test_task_mpi.PostProcessing();

  if (world.rank() == 0) {
    ASSERT_EQ(out[0], 1);
  }
}

TEST(komshina_d_num_of_alternations_signs_mpi, CrossProcessSignChange) {
  boost::mpi::communicator world;

  std::vector<int> in = {1, 1, 1, -1, -1, -1};
  std::vector<int32_t> out(1, 0);

  auto task_data_mpi = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    task_data_mpi->inputs_count.emplace_back(in.size());
    task_data_mpi->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    task_data_mpi->outputs_count.emplace_back(out.size());
  }

  komshina_d_num_of_alternations_signs_mpi::TestTaskMPI test_task_mpi(task_data_mpi);

  ASSERT_EQ(test_task_mpi.Validation(), true);
  test_task_mpi.PreProcessing();
  test_task_mpi.Run();
  test_task_mpi.PostProcessing();

  if (world.rank() == 0) {
    ASSERT_EQ(out[0], 1);
  }
}

TEST(komshina_d_num_of_alternations_signs_mpi, RandomTestSmall) {
  boost::mpi::communicator world;

  std::vector<int> in = GenerateRandomVector(10);
  std::vector<int32_t> out(1, 0);

  auto task_data_mpi = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    task_data_mpi->inputs_count.emplace_back(in.size());
    task_data_mpi->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    task_data_mpi->outputs_count.emplace_back(out.size());
  }

  komshina_d_num_of_alternations_signs_mpi::TestTaskMPI test_task_mpi(task_data_mpi);
  ASSERT_EQ(test_task_mpi.Validation(), true);
  test_task_mpi.PreProcessing();
  test_task_mpi.Run();
  test_task_mpi.PostProcessing();

  if (world.rank() == 0) {
    ASSERT_EQ(out[0], CountSignAlternations(in));
  }
}

TEST(komshina_d_num_of_alternations_signs_mpi, RandomTestLarge) {
  boost::mpi::communicator world;

  std::vector<int> in = GenerateRandomVector(1000);
  std::vector<int32_t> out(1, 0);

  auto task_data_mpi = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    task_data_mpi->inputs_count.emplace_back(in.size());
    task_data_mpi->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    task_data_mpi->outputs_count.emplace_back(out.size());
  }

  komshina_d_num_of_alternations_signs_mpi::TestTaskMPI test_task_mpi(task_data_mpi);
  ASSERT_EQ(test_task_mpi.Validation(), true);
  test_task_mpi.PreProcessing();
  test_task_mpi.Run();
  test_task_mpi.PostProcessing();

  if (world.rank() == 0) {
    ASSERT_EQ(out[0], CountSignAlternations(in));
  }
}