#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <cstdint>
#include <memory>
#include <vector>

#include "core/task/include/task.hpp"
#include "mpi/khovansky_d_rectangles_integral/include/ops_mpi.hpp"

TEST(khovansky_d_rectangles_integral_mpi, test_rectangles_validation_failure) {
  boost::mpi::communicator world;

  auto task_data_mpi = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    task_data_mpi->inputs_count.emplace_back(0);
    task_data_mpi->inputs_count.emplace_back(0);
  }

  khovansky_d_rectangles_integral_mpi::RectanglesMpi rectangles(task_data_mpi);
  if (world.rank() == 0) {
    ASSERT_FALSE(rectangles.ValidationImpl());
  }
}

TEST(khovansky_d_rectangles_integral_mpi, test_integral_1d_x_squared) {
  boost::mpi::communicator world;

  auto task_data_mpi = std::make_shared<ppc::core::TaskData>();
  std::vector<double> lower_limits = {0.0};
  std::vector<double> upper_limits = {1.0};
  int num_partitions = 100;
  double integral_result = 0.0;

  if (world.rank() == 0) {
    task_data_mpi->inputs_count.emplace_back(1);
    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(lower_limits.data()));
    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(upper_limits.data()));
    task_data_mpi->inputs_count.emplace_back(num_partitions);
    task_data_mpi->outputs.emplace_back(reinterpret_cast<uint8_t *>(&integral_result));
  }

  khovansky_d_rectangles_integral_mpi::RectanglesMpi rectangles(task_data_mpi);
  rectangles.integrand_function = [](const std::vector<double> &point) { return point[0] * point[0]; };
  rectangles.PreProcessingImpl();
  rectangles.RunImpl();
  rectangles.PostProcessingImpl();

  if (world.rank() == 0) {
    ASSERT_NEAR(integral_result, 1.0 / 3.0, 1e-2);
  }
}

TEST(khovansky_d_rectangles_integral_mpi, test_integral_2d_xy) {
  boost::mpi::communicator world;

  auto task_data_mpi = std::make_shared<ppc::core::TaskData>();
  std::vector<double> lower_limits = {0.0, 0.0};
  std::vector<double> upper_limits = {1.0, 1.0};
  int num_partitions = 100;
  double integral_result = 0.0;

  if (world.rank() == 0) {
    task_data_mpi->inputs_count.emplace_back(2);
    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(lower_limits.data()));
    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(upper_limits.data()));
    task_data_mpi->inputs_count.emplace_back(num_partitions);
    task_data_mpi->outputs.emplace_back(reinterpret_cast<uint8_t *>(&integral_result));
  }

  khovansky_d_rectangles_integral_mpi::RectanglesMpi rectangles(task_data_mpi);
  rectangles.integrand_function = [](const std::vector<double> &point) { return point[0] * point[1]; };
  ASSERT_TRUE(rectangles.ValidationImpl());
  rectangles.PreProcessingImpl();
  rectangles.RunImpl();
  rectangles.PostProcessingImpl();

  if (world.rank() == 0) {
    ASSERT_NEAR(integral_result, 0.25, 1e-2);
  }
}

TEST(khovansky_d_rectangles_integral_mpi, test_invalid_partition_count) {
  boost::mpi::communicator world;
  auto task_data_mpi = std::make_shared<ppc::core::TaskData>();
  std::vector<double> lower_limits = {0.0};
  std::vector<double> upper_limits = {1.0};
  int num_partitions = 0;
  double integral_result = 0.0;

  if (world.rank() == 0) {
    task_data_mpi->inputs_count.emplace_back(1);
    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(lower_limits.data()));
    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(upper_limits.data()));
    task_data_mpi->inputs_count.emplace_back(num_partitions);
    task_data_mpi->outputs.emplace_back(reinterpret_cast<uint8_t *>(&integral_result));
  }

  khovansky_d_rectangles_integral_mpi::RectanglesMpi rectangles(task_data_mpi);
  if (world.rank() == 0) {
    ASSERT_FALSE(rectangles.ValidationImpl());
  }
}

TEST(khovansky_d_rectangles_integral_mpi, test_missing_bounds_input) {
  boost::mpi::communicator world;
  auto task_data_mpi = std::make_shared<ppc::core::TaskData>();
  std::vector<double> lower_limits;
  std::vector<double> upper_limits;
  int num_partitions = 100;
  double integral_result = 0.0;

  if (world.rank() == 0) {
    task_data_mpi->inputs_count.emplace_back(1);
    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(lower_limits.data()));
    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(upper_limits.data()));
    task_data_mpi->inputs_count.emplace_back(num_partitions);
    task_data_mpi->outputs.emplace_back(reinterpret_cast<uint8_t *>(&integral_result));
  }

  khovansky_d_rectangles_integral_mpi::RectanglesMpi rectangles(task_data_mpi);
  if (world.rank() == 0) {
    ASSERT_FALSE(rectangles.ValidationImpl());
  }
}

TEST(khovansky_d_rectangles_integral_mpi, test_inverted_bounds) {
  boost::mpi::communicator world;
  auto task_data_mpi = std::make_shared<ppc::core::TaskData>();
  std::vector<double> lower_limits = {10.0};
  std::vector<double> upper_limits = {0.0};
  int num_partitions = 100;
  double integral_result = 0.0;

  if (world.rank() == 0) {
    task_data_mpi->inputs_count.emplace_back(1);
    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(lower_limits.data()));
    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(upper_limits.data()));
    task_data_mpi->inputs_count.emplace_back(num_partitions);
    task_data_mpi->outputs.emplace_back(reinterpret_cast<uint8_t *>(&integral_result));
  }

  khovansky_d_rectangles_integral_mpi::RectanglesMpi rectangles(task_data_mpi);
  if (world.rank() == 0) {
    ASSERT_FALSE(rectangles.ValidationImpl());
  }
}

TEST(khovansky_d_rectangles_integral_mpi, test_integral_3d_xyz) {
  boost::mpi::communicator world;
  auto task_data_mpi = std::make_shared<ppc::core::TaskData>();
  std::vector<double> lower_limits = {0.0, 0.0, 0.0};
  std::vector<double> upper_limits = {1.0, 1.0, 1.0};
  int num_partitions = 100;
  double integral_result = 0.0;

  if (world.rank() == 0) {
    task_data_mpi->inputs_count.emplace_back(3);
    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(lower_limits.data()));
    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(upper_limits.data()));
    task_data_mpi->inputs_count.emplace_back(num_partitions);
    task_data_mpi->outputs.emplace_back(reinterpret_cast<uint8_t *>(&integral_result));
  }

  khovansky_d_rectangles_integral_mpi::RectanglesMpi rectangles(task_data_mpi);
  rectangles.integrand_function = [](const std::vector<double> &point) { return point[0] * point[1] * point[2]; };
  ASSERT_TRUE(rectangles.ValidationImpl());
  rectangles.PreProcessingImpl();
  rectangles.RunImpl();
  rectangles.PostProcessingImpl();

  if (world.rank() == 0) {
    ASSERT_NEAR(integral_result, 1.0 / 8.0, 1e-2);
  }
}

TEST(khovansky_d_rectangles_integral_mpi, test_integral_4d_xyzt) {
  boost::mpi::communicator world;
  auto task_data_mpi = std::make_shared<ppc::core::TaskData>();
  std::vector<double> lower_limits = {0.0, 0.0, 0.0, 0.0};
  std::vector<double> upper_limits = {1.0, 1.0, 1.0, 1.0};
  int num_partitions = 100;
  double integral_result = 0.0;

  if (world.rank() == 0) {
    task_data_mpi->inputs_count.emplace_back(4);
    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(lower_limits.data()));
    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(upper_limits.data()));
    task_data_mpi->inputs_count.emplace_back(num_partitions);
    task_data_mpi->outputs.emplace_back(reinterpret_cast<uint8_t *>(&integral_result));
  }

  khovansky_d_rectangles_integral_mpi::RectanglesMpi rectangles(task_data_mpi);
  rectangles.integrand_function = [](const std::vector<double> &point) {
    return point[0] * point[1] * point[2] * point[3];
  };
  ASSERT_TRUE(rectangles.ValidationImpl());
  rectangles.PreProcessingImpl();
  rectangles.RunImpl();
  rectangles.PostProcessingImpl();

  if (world.rank() == 0) {
    ASSERT_NEAR(integral_result, 1.0 / 16.0, 1e-2);
  }
}

TEST(khovansky_d_rectangles_integral_mpi, test_integral_3d_sum) {
  boost::mpi::communicator world;
  auto task_data_mpi = std::make_shared<ppc::core::TaskData>();
  std::vector<double> lower_limits = {0.0, 0.0, 0.0};
  std::vector<double> upper_limits = {1.0, 1.0, 1.0};
  int num_partitions = 100;
  double integral_result = 0.0;

  if (world.rank() == 0) {
    task_data_mpi->inputs_count.emplace_back(3);
    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(lower_limits.data()));
    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(upper_limits.data()));
    task_data_mpi->inputs_count.emplace_back(num_partitions);
    task_data_mpi->outputs.emplace_back(reinterpret_cast<uint8_t *>(&integral_result));
  }

  khovansky_d_rectangles_integral_mpi::RectanglesMpi rectangles(task_data_mpi);
  rectangles.integrand_function = [](const std::vector<double> &point) { return point[0] + point[1] + point[2]; };
  ASSERT_TRUE(rectangles.ValidationImpl());
  rectangles.PreProcessingImpl();
  rectangles.RunImpl();
  rectangles.PostProcessingImpl();

  if (world.rank() == 0) {
    ASSERT_NEAR(integral_result, 1.5, 1e-2);
  }
}