#include <gtest/gtest.h>

#include <cstdint>
#include <memory>
#include <vector>

#include "core/task/include/task.hpp"
#include "seq/khovansky_d_rectangles_integral/include/ops_seq.hpp"

TEST(khovansky_d_rectangles_integral_seq, test_rectangles_validation_failure) {
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();

  task_data_seq->inputs_count.emplace_back(0);
  task_data_seq->inputs_count.emplace_back(0);

  khovansky_d_rectangles_integral_seq::RectanglesSeq rectangles(task_data_seq);

  ASSERT_FALSE(rectangles.ValidationImpl());
}

TEST(khovansky_d_rectangles_integral_seq, test_integral_1d_x_squared) {
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  std::vector<double> lower_limits = {0.0};
  std::vector<double> upper_limits = {1.0};
  int num_partitions = 100;
  double integral_result = 0.0;

  task_data_seq->inputs_count.emplace_back(1);
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(lower_limits.data()));
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(upper_limits.data()));
  task_data_seq->inputs_count.emplace_back(num_partitions);
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(&integral_result));

  khovansky_d_rectangles_integral_seq::RectanglesSeq rectangles(task_data_seq);
  rectangles.integrand_function = [](const std::vector<double> &point) { return point[0] * point[0]; };
  rectangles.PreProcessingImpl();
  rectangles.RunImpl();
  rectangles.PostProcessingImpl();

  ASSERT_NEAR(integral_result, 1.0 / 3.0, 1e-2);
}

TEST(khovansky_d_rectangles_integral_seq, test_integral_2d_xy) {
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  std::vector<double> lower_limits = {0.0, 0.0};
  std::vector<double> upper_limits = {1.0, 1.0};
  int num_partitions = 100;
  double integral_result = 0.0;

  task_data_seq->inputs_count.emplace_back(2);
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(lower_limits.data()));
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(upper_limits.data()));
  task_data_seq->inputs_count.emplace_back(num_partitions);
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(&integral_result));

  khovansky_d_rectangles_integral_seq::RectanglesSeq rectangles(task_data_seq);
  rectangles.integrand_function = [](const std::vector<double> &point) { return point[0] * point[1]; };
  ASSERT_TRUE(rectangles.ValidationImpl());
  rectangles.PreProcessingImpl();
  rectangles.RunImpl();
  rectangles.PostProcessingImpl();

  ASSERT_NEAR(integral_result, 0.25, 1e-2);
}

TEST(khovansky_d_rectangles_integral_seq, test_invalid_partition_count) {
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  std::vector<double> lower_limits = {0.0};
  std::vector<double> upper_limits = {1.0};
  int num_partitions = 0;
  double integral_result = 0.0;

  task_data_seq->inputs_count.emplace_back(1);
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(lower_limits.data()));
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(upper_limits.data()));
  task_data_seq->inputs_count.emplace_back(num_partitions);
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(&integral_result));

  khovansky_d_rectangles_integral_seq::RectanglesSeq rectangles(task_data_seq);
  ASSERT_FALSE(rectangles.ValidationImpl());
}

TEST(khovansky_d_rectangles_integral_seq, test_missing_bounds_input) {
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  std::vector<double> lower_limits;
  std::vector<double> upper_limits;
  int num_partitions = 100;
  double integral_result = 0.0;

  task_data_seq->inputs_count.emplace_back(1);
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(lower_limits.data()));
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(upper_limits.data()));
  task_data_seq->inputs_count.emplace_back(num_partitions);
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(&integral_result));

  khovansky_d_rectangles_integral_seq::RectanglesSeq rectangles(task_data_seq);
  ASSERT_FALSE(rectangles.ValidationImpl());
}

TEST(khovansky_d_rectangles_integral_seq, test_inverted_bounds) {
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  std::vector<double> lower_limits = {10.0};
  std::vector<double> upper_limits = {0.0};
  int num_partitions = 100;
  double integral_result = 0.0;

  task_data_seq->inputs_count.emplace_back(1);
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(lower_limits.data()));
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(upper_limits.data()));
  task_data_seq->inputs_count.emplace_back(num_partitions);
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(&integral_result));

  khovansky_d_rectangles_integral_seq::RectanglesSeq rectangles(task_data_seq);
  ASSERT_FALSE(rectangles.ValidationImpl());
}

TEST(khovansky_d_rectangles_integral_seq, test_integral_3d_xyz) {
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  std::vector<double> lower_limits = {0.0, 0.0, 0.0};
  std::vector<double> upper_limits = {1.0, 1.0, 1.0};
  int num_partitions = 100;
  double integral_result = 0.0;

  task_data_seq->inputs_count.emplace_back(3);
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(lower_limits.data()));
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(upper_limits.data()));
  task_data_seq->inputs_count.emplace_back(num_partitions);
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(&integral_result));

  khovansky_d_rectangles_integral_seq::RectanglesSeq rectangles(task_data_seq);
  rectangles.integrand_function = [](const std::vector<double> &point) { return point[0] * point[1] * point[2]; };
  ASSERT_TRUE(rectangles.ValidationImpl());
  rectangles.PreProcessingImpl();
  rectangles.RunImpl();
  rectangles.PostProcessingImpl();

  ASSERT_NEAR(integral_result, 1.0 / 8.0, 1e-2);
}

TEST(khovansky_d_rectangles_integral_seq, test_integral_4d_xyzt) {
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  std::vector<double> lower_limits = {0.0, 0.0, 0.0, 0.0};
  std::vector<double> upper_limits = {1.0, 1.0, 1.0, 1.0};
  int num_partitions = 100;
  double integral_result = 0.0;

  task_data_seq->inputs_count.emplace_back(4);
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(lower_limits.data()));
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(upper_limits.data()));
  task_data_seq->inputs_count.emplace_back(num_partitions);
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(&integral_result));

  khovansky_d_rectangles_integral_seq::RectanglesSeq rectangles(task_data_seq);
  rectangles.integrand_function = [](const std::vector<double> &point) {
    return point[0] * point[1] * point[2] * point[3];
  };
  ASSERT_TRUE(rectangles.ValidationImpl());
  rectangles.PreProcessingImpl();
  rectangles.RunImpl();
  rectangles.PostProcessingImpl();

  ASSERT_NEAR(integral_result, 1.0 / 16.0, 1e-2);
}

TEST(khovansky_d_rectangles_integral_seq, test_integral_3d_sum) {
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  std::vector<double> lower_limits = {0.0, 0.0, 0.0};
  std::vector<double> upper_limits = {1.0, 1.0, 1.0};
  int num_partitions = 100;
  double integral_result = 0.0;

  task_data_seq->inputs_count.emplace_back(3);
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(lower_limits.data()));
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(upper_limits.data()));
  task_data_seq->inputs_count.emplace_back(num_partitions);
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(&integral_result));

  khovansky_d_rectangles_integral_seq::RectanglesSeq rectangles(task_data_seq);
  rectangles.integrand_function = [](const std::vector<double> &point) { return point[0] + point[1] + point[2]; };
  ASSERT_TRUE(rectangles.ValidationImpl());
  rectangles.PreProcessingImpl();
  rectangles.RunImpl();
  rectangles.PostProcessingImpl();

  ASSERT_NEAR(integral_result, 1.5, 1e-2);
}