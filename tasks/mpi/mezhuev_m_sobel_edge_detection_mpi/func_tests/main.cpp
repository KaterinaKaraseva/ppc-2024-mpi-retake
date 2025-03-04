#include <gtest/gtest.h>

#include <algorithm>
#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "core/task/include/task.hpp"
#include "mpi/mezhuev_m_sobel_edge_detection_mpi/include/mpi.hpp"

TEST(mezhuev_m_sobel_edge_detection_mpi, test_basic_case) {
  boost::mpi::environment env;
  boost::mpi::communicator world;

  constexpr int kWidth = 5;
  constexpr int kHeight = 5;
  constexpr int kImageSize = kWidth * kHeight;
  std::vector<uint8_t> in(kImageSize, 0);
  std::vector<uint8_t> out(kImageSize, 0);

  for (size_t y = 0; y < kHeight; ++y) {
    for (size_t x = 0; x < kWidth; ++x) {
      in[(y * kWidth) + x] = static_cast<uint8_t>(x * 50);
    }
  }
  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs = {in.data()};
  task_data->inputs_count = {kImageSize};
  task_data->outputs = {out.data()};
  task_data->outputs_count = {kImageSize};

  auto sobel_task = std::make_shared<mezhuev_m_sobel_edge_detection_mpi::SobelEdgeDetection>(world, task_data);

  ASSERT_TRUE(sobel_task->PreProcessingImpl() && sobel_task->RunImpl() && sobel_task->ValidationImpl() &&
              sobel_task->PostProcessingImpl());

  if (world.rank() == 0) {
    ASSERT_TRUE(std::ranges::any_of(out.begin(), out.end(), [](uint8_t val) { return val > 0; }));
  }
}

TEST(mezhuev_m_sobel_edge_detection_mpi, test_small_image_3x3) {
  boost::mpi::environment env;
  boost::mpi::communicator world;

  std::vector<uint8_t> in(9, 255);
  std::vector<uint8_t> out(9, 0);
  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs = {in.data()};
  task_data->inputs_count = {static_cast<uint32_t>(in.size())};
  task_data->outputs = {out.data()};
  task_data->outputs_count = {static_cast<uint32_t>(out.size())};

  auto sobel_task = std::make_shared<mezhuev_m_sobel_edge_detection_mpi::SobelEdgeDetection>(world, task_data);
  ASSERT_TRUE(sobel_task->PreProcessingImpl() && sobel_task->RunImpl() && sobel_task->ValidationImpl() &&
              sobel_task->PostProcessingImpl());

  if (world.rank() == 0) {
    ASSERT_TRUE(std::ranges::all_of(out.begin(), out.end(), [](uint8_t val) { return val == 0; }));
  }
}

TEST(mezhuev_m_sobel_edge_detection_mpi, test_empty_input) {
  boost::mpi::environment env;
  boost::mpi::communicator world;

  auto task_data = std::make_shared<ppc::core::TaskData>();
  auto sobel_task = std::make_shared<mezhuev_m_sobel_edge_detection_mpi::SobelEdgeDetection>(world, task_data);
  ASSERT_FALSE(sobel_task->PreProcessingImpl() || sobel_task->RunImpl() || sobel_task->ValidationImpl() ||
               sobel_task->PostProcessingImpl());
}

TEST(mezhuev_m_sobel_edge_detection_mpi, test_single_bright_pixel) {
  boost::mpi::environment env;
  boost::mpi::communicator world;

  std::vector<uint8_t> in(25, 0);
  std::vector<uint8_t> out(25, 0);
  in[12] = 255;

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs = {in.data()};
  task_data->inputs_count = {static_cast<uint32_t>(in.size())};
  task_data->outputs = {out.data()};
  task_data->outputs_count = {static_cast<uint32_t>(out.size())};

  auto sobel_task = std::make_shared<mezhuev_m_sobel_edge_detection_mpi::SobelEdgeDetection>(world, task_data);
  ASSERT_TRUE(sobel_task->PreProcessingImpl() && sobel_task->RunImpl() && sobel_task->ValidationImpl() &&
              sobel_task->PostProcessingImpl());

  if (world.rank() == 0) {
    ASSERT_TRUE(std::ranges::any_of(out.begin(), out.end(), [](uint8_t val) { return val > 0; }));
  }
}

TEST(mezhuev_m_sobel_edge_detection_mpi, test_large_image) {
  boost::mpi::environment env;
  boost::mpi::communicator world;

  std::vector<uint8_t> in(1024 * 1024, 128);
  std::vector<uint8_t> out(1024 * 1024, 0);

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs = {in.data()};
  task_data->inputs_count = {static_cast<uint32_t>(in.size())};
  task_data->outputs = {out.data()};
  task_data->outputs_count = {static_cast<uint32_t>(out.size())};

  auto sobel_task = std::make_shared<mezhuev_m_sobel_edge_detection_mpi::SobelEdgeDetection>(world, task_data);
  ASSERT_TRUE(sobel_task->PreProcessingImpl() && sobel_task->RunImpl() && sobel_task->ValidationImpl() &&
              sobel_task->PostProcessingImpl());
}

TEST(mezhuev_m_sobel_edge_detection_mpi, test_vertical_gradient) {
  boost::mpi::environment env;
  boost::mpi::communicator world;

  std::vector<uint8_t> in(25, 0);
  std::vector<uint8_t> out(25, 0);
  for (size_t y = 0; y < 5; ++y) {
    for (size_t x = 0; x < 5; ++x) {
      in[(y * 5) + x] = static_cast<uint8_t>(y * 50);
    }
  }
  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs = {in.data()};
  task_data->inputs_count = {static_cast<uint32_t>(in.size())};
  task_data->outputs = {out.data()};
  task_data->outputs_count = {static_cast<uint32_t>(out.size())};

  auto sobel_task = std::make_shared<mezhuev_m_sobel_edge_detection_mpi::SobelEdgeDetection>(world, task_data);
  ASSERT_TRUE(sobel_task->PreProcessingImpl() && sobel_task->RunImpl() && sobel_task->ValidationImpl() &&
              sobel_task->PostProcessingImpl());

  if (world.rank() == 0) {
    ASSERT_TRUE(std::ranges::any_of(out.begin(), out.end(), [](uint8_t val) { return val > 0; }));
  }
}

TEST(mezhuev_m_sobel_edge_detection_mpi, test_black_image) {
  boost::mpi::environment env;
  boost::mpi::communicator world;

  std::vector<uint8_t> in(25, 0);
  std::vector<uint8_t> out(25, 0);

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs = {in.data()};
  task_data->inputs_count = {static_cast<uint32_t>(in.size())};
  task_data->outputs = {out.data()};
  task_data->outputs_count = {static_cast<uint32_t>(out.size())};

  auto sobel_task = std::make_shared<mezhuev_m_sobel_edge_detection_mpi::SobelEdgeDetection>(world, task_data);
  ASSERT_TRUE(sobel_task->PreProcessingImpl() && sobel_task->RunImpl() && sobel_task->ValidationImpl() &&
              sobel_task->PostProcessingImpl());

  if (world.rank() == 0) {
    ASSERT_TRUE(std::ranges::all_of(out.begin(), out.end(), [](uint8_t val) { return val == 0; }));
  }
}

TEST(mezhuev_m_sobel_edge_detection_mpi, test_diagonal_gradient) {
  boost::mpi::environment env;
  boost::mpi::communicator world;

  std::vector<uint8_t> in(25, 0);
  std::vector<uint8_t> out(25, 0);
  for (size_t y = 0; y < 5; ++y) {
    for (size_t x = 0; x < 5; ++x) {
      in[(y * 5) + x] = static_cast<uint8_t>((x + y) * 50);
    }
  }

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs = {in.data()};
  task_data->inputs_count = {static_cast<uint32_t>(in.size())};
  task_data->outputs = {out.data()};
  task_data->outputs_count = {static_cast<uint32_t>(out.size())};

  auto sobel_task = std::make_shared<mezhuev_m_sobel_edge_detection_mpi::SobelEdgeDetection>(world, task_data);
  ASSERT_TRUE(sobel_task->PreProcessingImpl() && sobel_task->RunImpl() && sobel_task->ValidationImpl() &&
              sobel_task->PostProcessingImpl());

  if (world.rank() == 0) {
    ASSERT_TRUE(std::ranges::any_of(out.begin(), out.end(), [](uint8_t val) { return val > 0; }));
  }
}

TEST(mezhuev_m_sobel_edge_detection_mpi, test_horizontal_gradient) {
  boost::mpi::environment env;
  boost::mpi::communicator world;

  std::vector<uint8_t> in(25, 0);
  std::vector<uint8_t> out(25, 0);
  for (size_t y = 0; y < 5; ++y) {
    for (size_t x = 0; x < 5; ++x) {
      in[(y * 5) + x] = static_cast<uint8_t>(x * 50);
    }
  }

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs = {in.data()};
  task_data->inputs_count = {static_cast<uint32_t>(in.size())};
  task_data->outputs = {out.data()};
  task_data->outputs_count = {static_cast<uint32_t>(out.size())};

  auto sobel_task = std::make_shared<mezhuev_m_sobel_edge_detection_mpi::SobelEdgeDetection>(world, task_data);
  ASSERT_TRUE(sobel_task->PreProcessingImpl() && sobel_task->RunImpl() && sobel_task->ValidationImpl() &&
              sobel_task->PostProcessingImpl());

  if (world.rank() == 0) {
    ASSERT_TRUE(std::ranges::any_of(out.begin(), out.end(), [](uint8_t val) { return val > 0; }));
  }
}

TEST(mezhuev_m_sobel_edge_detection_mpi, test_constant_image) {
  boost::mpi::environment env;
  boost::mpi::communicator world;

  std::vector<uint8_t> in(25, 128);
  std::vector<uint8_t> out(25, 0);

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs = {in.data()};
  task_data->inputs_count = {static_cast<uint32_t>(in.size())};
  task_data->outputs = {out.data()};
  task_data->outputs_count = {static_cast<uint32_t>(out.size())};

  auto sobel_task = std::make_shared<mezhuev_m_sobel_edge_detection_mpi::SobelEdgeDetection>(world, task_data);
  ASSERT_TRUE(sobel_task->PreProcessingImpl() && sobel_task->RunImpl() && sobel_task->ValidationImpl() &&
              sobel_task->PostProcessingImpl());

  if (world.rank() == 0) {
    ASSERT_TRUE(std::ranges::all_of(out.begin(), out.end(), [](uint8_t val) { return val == 0; }));
  }
}
