#include <gtest/gtest.h>

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <random>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"
#include "mpi/solovev_a_binary_image_marking/include/ops_mpi.hpp"

namespace {
std::vector<int> RandomImg(int height, int width) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> dis(0, 1);

  std::vector<int> img(height * width);

  for (int& pixel : img) {
    pixel = dis(gen);
  }

  return img;
}

void TestBodyFunction(int height, int width) {
  boost::mpi::communicator world;

  std::vector<int> result_mpi(height * width);
  std::vector<int> result_seq(height * width);

  std::vector<int> expected_result(height * width, 0);

  std::shared_ptr<ppc::core::TaskData> task_data_par = std::make_shared<ppc::core::TaskData>();
  std::shared_ptr<ppc::core::TaskData> task_data_seq = std::make_shared<ppc::core::TaskData>();

  std::vector<int> img = RandomImg(height, width);
  task_data_par->inputs.emplace_back(reinterpret_cast<uint8_t*>(&height));
  task_data_par->inputs_count.emplace_back(1);
  task_data_par->inputs.emplace_back(reinterpret_cast<uint8_t*>(&width));
  task_data_par->inputs_count.emplace_back(1);
  task_data_par->inputs.emplace_back(reinterpret_cast<uint8_t*>(img.data()));
  task_data_par->inputs_count.emplace_back(img.size());

  task_data_par->outputs.emplace_back(reinterpret_cast<uint8_t*>(result_mpi.data()));
  task_data_par->outputs_count.emplace_back(result_mpi.size());

  if (world.rank() == 0) {
    task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&height));
    task_data_seq->inputs_count.emplace_back(1);
    task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&width));
    task_data_seq->inputs_count.emplace_back(1);
    task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(img.data()));
    task_data_seq->inputs_count.emplace_back(img.size());

    task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(result_seq.data()));
    task_data_seq->outputs_count.emplace_back(result_seq.size());

    solovev_a_binary_image_marking::TestMPITaskSequential binary_marker_seq(task_data_seq);
    ASSERT_EQ(binary_marker_seq.ValidationImpl(), true);
    binary_marker_seq.PreProcessingImpl();
    binary_marker_seq.RunImpl();
    binary_marker_seq.PostProcessingImpl();

    expected_result = std::move(result_seq);
  }

  solovev_a_binary_image_marking::TestMPITaskParallel binary_marker_mpi(task_data_par);
  ASSERT_EQ(binary_marker_mpi.ValidationImpl(), true);
  binary_marker_mpi.PreProcessingImpl();
  binary_marker_mpi.RunImpl();
  binary_marker_mpi.PostProcessingImpl();

  if (world.rank() == 0) {
    for (size_t i = 0; i < result_mpi.size(); ++i) {
      ASSERT_EQ(result_mpi[i], expected_result[i]);
    }
  }
}

void ValidationFalseTest(int height, int width) {
  boost::mpi::communicator world;

  std::vector<int> img{};
  std::vector<int> result_mpi;

  std::shared_ptr<ppc::core::TaskData> task_data_par = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    task_data_par->inputs.emplace_back(reinterpret_cast<uint8_t*>(&height));
    task_data_par->inputs_count.emplace_back(1);
    task_data_par->inputs.emplace_back(reinterpret_cast<uint8_t*>(&width));
    task_data_par->inputs_count.emplace_back(1);
    task_data_par->inputs.emplace_back(reinterpret_cast<uint8_t*>(img.data()));
    task_data_par->inputs_count.emplace_back(img.size());

    task_data_par->outputs.emplace_back(reinterpret_cast<uint8_t*>(result_mpi.data()));
    task_data_par->outputs_count.emplace_back(result_mpi.size());
  }

  solovev_a_binary_image_marking::TestMPITaskParallel binary_marker_mpi(task_data_par);
  if (world.rank() == 0) {
    ASSERT_EQ(binary_marker_mpi.ValidationImpl(), false);
  }
}

}  // namespace

TEST(solovev_a_binary_image_marking, Test_image_random_5X5) { TestBodyFunction(5, 5); }

TEST(solovev_a_binary_image_marking, Test_image_random_11X11) { TestBodyFunction(11, 11); }

TEST(solovev_a_binary_image_marking, Test_image_random_16X16) { TestBodyFunction(16, 16); }

TEST(solovev_a_binary_image_marking, Test_image_random_32X32) { TestBodyFunction(32, 32); }

TEST(solovev_a_binary_image_marking, Test_image_random_23X31) { TestBodyFunction(23, 31); }

TEST(solovev_a_binary_image_marking, Test_image_random_31X23) { TestBodyFunction(31, 23); }

TEST(solovev_a_binary_image_marking, Test_image_random_50X50) { TestBodyFunction(50, 50); }

TEST(solovev_a_binary_image_marking, Test_image_random_75X75) { TestBodyFunction(75, 75); }

TEST(solovev_a_binary_image_marking, Whole_image) {
  boost::mpi::communicator world;

  int height = 77;
  int width = 77;

  std::vector<int> result_mpi(height * width);
  std::vector<int> result_seq(height * width);

  std::vector<int> expected_result(height * width, 0);

  std::shared_ptr<ppc::core::TaskData> task_data_par = std::make_shared<ppc::core::TaskData>();
  std::shared_ptr<ppc::core::TaskData> task_data_seq = std::make_shared<ppc::core::TaskData>();

  std::vector<int> img(height * height, 1);
  task_data_par->inputs.emplace_back(reinterpret_cast<uint8_t*>(&height));
  task_data_par->inputs_count.emplace_back(1);
  task_data_par->inputs.emplace_back(reinterpret_cast<uint8_t*>(&width));
  task_data_par->inputs_count.emplace_back(1);
  task_data_par->inputs.emplace_back(reinterpret_cast<uint8_t*>(img.data()));
  task_data_par->inputs_count.emplace_back(img.size());

  task_data_par->outputs.emplace_back(reinterpret_cast<uint8_t*>(result_mpi.data()));
  task_data_par->outputs_count.emplace_back(result_mpi.size());

  if (world.rank() == 0) {
    task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&height));
    task_data_seq->inputs_count.emplace_back(1);
    task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&width));
    task_data_seq->inputs_count.emplace_back(1);
    task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(img.data()));
    task_data_seq->inputs_count.emplace_back(img.size());

    task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(result_seq.data()));
    task_data_seq->outputs_count.emplace_back(result_seq.size());

    solovev_a_binary_image_marking::TestMPITaskSequential binary_marker_seq(task_data_seq);
    ASSERT_EQ(binary_marker_seq.ValidationImpl(), true);
    binary_marker_seq.PreProcessingImpl();
    binary_marker_seq.RunImpl();
    binary_marker_seq.PostProcessingImpl();

    expected_result = std::move(result_seq);
  }

  solovev_a_binary_image_marking::TestMPITaskParallel binary_marker_mpi(task_data_par);
  ASSERT_EQ(binary_marker_mpi.ValidationImpl(), true);
  binary_marker_mpi.PreProcessingImpl();
  binary_marker_mpi.RunImpl();
  binary_marker_mpi.PostProcessingImpl();

  if (world.rank() == 0) {
    for (size_t i = 0; i < result_mpi.size(); ++i) {
      ASSERT_EQ(result_mpi[i], expected_result[i]);
    }
  }
}

TEST(solovev_a_binary_image_marking, Validation_false_1) { ValidationFalseTest(-1, 10); }

TEST(solovev_a_binary_image_marking, Validation_false_2) { ValidationFalseTest(10, -1); }

TEST(solovev_a_binary_image_marking, Validation_false_3) { ValidationFalseTest(10, 10); }
