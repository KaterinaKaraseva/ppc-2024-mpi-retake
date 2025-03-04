#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <cstdint>
#include <cstdlib>
#include <ctime>
#include <memory>
#include <string>
#include <vector>

#include "core/task/include/task.hpp"
#include "mpi/budazhapova_e_count_freq_character/include/count_freq_chart_mpi_header.hpp"

namespace budazhapova_e_count_freq_chart_mpi {
namespace {
std::string GetRandomString(int length) {
  static std::string charset = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890";
  std::string result;
  result.resize(length);

  srand(time(nullptr));
  for (int i = 0; i < length; i++) {
    result[i] = charset[rand() % charset.size()];
  }
  return result;
}
}  // namespace
}  // namespace budazhapova_e_count_freq_chart_mpi

TEST(budazhapova_e_count_freq_chart_mpi, test_with_random_string) {
  boost::mpi::communicator world;

  std::string global_str;
  std::vector<int> global_out(1, 0);
  char symb = '1';
  const int size_string = 10;
  global_str = budazhapova_e_count_freq_chart_mpi::GetRandomString(size_string);

  // Create task_data
  std::shared_ptr<ppc::core::TaskData> task_data_par = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    task_data_par->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_str.data()));
    task_data_par->inputs_count.emplace_back(global_str.size());
    task_data_par->inputs.emplace_back(reinterpret_cast<uint8_t*>(&symb));
    task_data_par->inputs_count.emplace_back(1);
    task_data_par->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_out.data()));
    task_data_par->outputs_count.emplace_back(global_out.size());
  }

  budazhapova_e_count_freq_chart_mpi::TestMPITaskParallel test_task_mpi(task_data_par);
  ASSERT_EQ(test_task_mpi.Validation(), true);
  test_task_mpi.PreProcessing();
  test_task_mpi.Run();
  test_task_mpi.PostProcessing();

  if (world.rank() == 0) {
    // Create data
    std::vector<int> reference_out(1, 0);
    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> task_data_seq = std::make_shared<ppc::core::TaskData>();
    task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_str.data()));
    task_data_seq->inputs_count.emplace_back(global_str.size());
    task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&symb));
    task_data_seq->inputs_count.emplace_back(1);
    task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_out.data()));
    task_data_seq->outputs_count.emplace_back(reference_out.size());

    // Create Task
    budazhapova_e_count_freq_chart_mpi::TestMPITaskSequential test_task_seq(task_data_seq);
    ASSERT_EQ(test_task_seq.Validation(), true);
    test_task_seq.PreProcessing();
    test_task_seq.Run();
    test_task_seq.PostProcessing();
    ASSERT_EQ(reference_out[0], global_out[0]);
  }
}
TEST(budazhapova_e_count_freq_chart_mpi, test_with_one_symb) {
  boost::mpi::communicator world;
  std::string global_str = "3";
  std::vector<int> global_out(1, 0);
  char symb = '3';
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> task_data_par = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    task_data_par->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_str.data()));
    task_data_par->inputs_count.emplace_back(global_str.size());
    task_data_par->inputs.emplace_back(reinterpret_cast<uint8_t*>(&symb));
    task_data_par->inputs_count.emplace_back(1);
    task_data_par->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_out.data()));
    task_data_par->outputs_count.emplace_back(global_out.size());
  }
  budazhapova_e_count_freq_chart_mpi::TestMPITaskParallel test_mpi_task_parallel(task_data_par);
  ASSERT_EQ(test_mpi_task_parallel.Validation(), true);
  test_mpi_task_parallel.PreProcessing();
  test_mpi_task_parallel.Run();
  test_mpi_task_parallel.PostProcessing();
  if (world.rank() == 0) {
    // Create data
    std::vector<int> reference_out(1, 0);
    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> task_data_seq = std::make_shared<ppc::core::TaskData>();
    task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_str.data()));
    task_data_seq->inputs_count.emplace_back(global_str.size());
    task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&symb));
    task_data_seq->inputs_count.emplace_back(1);
    task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_out.data()));
    task_data_seq->outputs_count.emplace_back(reference_out.size());

    // Create Task
    budazhapova_e_count_freq_chart_mpi::TestMPITaskSequential test_mpi_task_sequential(task_data_seq);
    ASSERT_EQ(test_mpi_task_sequential.Validation(), true);
    test_mpi_task_sequential.PreProcessing();
    test_mpi_task_sequential.Run();
    test_mpi_task_sequential.PostProcessing();
    ASSERT_EQ(reference_out[0], global_out[0]);
  }
}
TEST(budazhapova_e_count_freq_chart_mpi, all_same_symb) {
  boost::mpi::communicator world;
  std::string global_str(100, 'a');
  std::vector<int> global_out(1, 0);
  char symb = 'a';
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> task_data_par = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    task_data_par->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_str.data()));
    task_data_par->inputs_count.emplace_back(global_str.size());
    task_data_par->inputs.emplace_back(reinterpret_cast<uint8_t*>(&symb));
    task_data_par->inputs_count.emplace_back(1);
    task_data_par->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_out.data()));
    task_data_par->outputs_count.emplace_back(global_out.size());
  }
  budazhapova_e_count_freq_chart_mpi::TestMPITaskParallel test_mpi_task_parallel(task_data_par);
  ASSERT_EQ(test_mpi_task_parallel.Validation(), true);
  test_mpi_task_parallel.PreProcessing();
  test_mpi_task_parallel.Run();
  test_mpi_task_parallel.PostProcessing();
  if (world.rank() == 0) {
    // Create data
    std::vector<int> reference_out(1, 0);
    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> task_data_seq = std::make_shared<ppc::core::TaskData>();
    task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_str.data()));
    task_data_seq->inputs_count.emplace_back(global_str.size());
    task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&symb));
    task_data_seq->inputs_count.emplace_back(1);
    task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_out.data()));
    task_data_seq->outputs_count.emplace_back(reference_out.size());

    // Create Task
    budazhapova_e_count_freq_chart_mpi::TestMPITaskSequential test_mpi_task_sequential(task_data_seq);
    ASSERT_EQ(test_mpi_task_sequential.Validation(), true);
    test_mpi_task_sequential.PreProcessing();
    test_mpi_task_sequential.Run();
    test_mpi_task_sequential.PostProcessing();
    ASSERT_EQ(reference_out[0], global_out[0]);
  }
}
TEST(budazhapova_e_count_freq_chart_mpi, empty_string) {
  boost::mpi::communicator world;
  std::string global_str;
  std::vector<int> global_out(1, 0);
  char symb = 'a';
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> task_data_par = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    task_data_par->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_str.data()));
    task_data_par->inputs_count.emplace_back(global_str.size());
    task_data_par->inputs.emplace_back(reinterpret_cast<uint8_t*>(&symb));
    task_data_par->inputs_count.emplace_back(1);
    task_data_par->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_out.data()));
    task_data_par->outputs_count.emplace_back(global_out.size());
  }
  budazhapova_e_count_freq_chart_mpi::TestMPITaskParallel test_mpi_task_parallel(task_data_par);
  ASSERT_EQ(test_mpi_task_parallel.Validation(), true);
  test_mpi_task_parallel.PreProcessing();
  test_mpi_task_parallel.Run();
  test_mpi_task_parallel.PostProcessing();

  if (world.rank() == 0) {
    ASSERT_EQ(0, global_out[0]);
  }
}

TEST(budazhapova_e_count_freq_chart_mpi, no_symb_in_string) {
  boost::mpi::communicator world;
  std::string global_str(10, 'b');
  std::vector<int> global_out(1, 0);
  char symb = 'a';
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> task_data_par = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    task_data_par->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_str.data()));
    task_data_par->inputs_count.emplace_back(global_str.size());
    task_data_par->inputs.emplace_back(reinterpret_cast<uint8_t*>(&symb));
    task_data_par->inputs_count.emplace_back(1);
    task_data_par->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_out.data()));
    task_data_par->outputs_count.emplace_back(global_out.size());
  }
  budazhapova_e_count_freq_chart_mpi::TestMPITaskParallel test_mpi_task_parallel(task_data_par);
  ASSERT_EQ(test_mpi_task_parallel.Validation(), true);
  test_mpi_task_parallel.PreProcessing();
  test_mpi_task_parallel.Run();
  test_mpi_task_parallel.PostProcessing();
  if (world.rank() == 0) {
    // Create data
    std::vector<int> reference_out(1, 0);
    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> task_data_seq = std::make_shared<ppc::core::TaskData>();
    task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_str.data()));
    task_data_seq->inputs_count.emplace_back(global_str.size());
    task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&symb));
    task_data_seq->inputs_count.emplace_back(1);
    task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_out.data()));
    task_data_seq->outputs_count.emplace_back(reference_out.size());

    // Create Task
    budazhapova_e_count_freq_chart_mpi::TestMPITaskSequential test_mpi_task_sequential(task_data_seq);
    ASSERT_EQ(test_mpi_task_sequential.Validation(), true);
    test_mpi_task_sequential.PreProcessing();
    test_mpi_task_sequential.Run();
    test_mpi_task_sequential.PostProcessing();
    ASSERT_EQ(reference_out[0], global_out[0]);
  }
}
TEST(budazhapova_e_count_freq_chart_mpi, ordinary_string) {
  boost::mpi::communicator world;
  std::string global_str = "gdfhfsef";
  std::vector<int> global_out(1, 0);
  char symb = 'f';
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> task_data_par = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    task_data_par->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_str.data()));
    task_data_par->inputs_count.emplace_back(global_str.size());
    task_data_par->inputs.emplace_back(reinterpret_cast<uint8_t*>(&symb));
    task_data_par->inputs_count.emplace_back(1);
    task_data_par->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_out.data()));
    task_data_par->outputs_count.emplace_back(global_out.size());
  }
  budazhapova_e_count_freq_chart_mpi::TestMPITaskParallel test_mpi_task_parallel(task_data_par);
  ASSERT_EQ(test_mpi_task_parallel.Validation(), true);
  test_mpi_task_parallel.PreProcessing();
  test_mpi_task_parallel.Run();
  test_mpi_task_parallel.PostProcessing();
  if (world.rank() == 0) {
    // Create data
    std::vector<int> reference_out(1, 0);
    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> task_data_seq = std::make_shared<ppc::core::TaskData>();
    task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_str.data()));
    task_data_seq->inputs_count.emplace_back(global_str.size());
    task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&symb));
    task_data_seq->inputs_count.emplace_back(1);
    task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_out.data()));
    task_data_seq->outputs_count.emplace_back(reference_out.size());

    // Create Task
    budazhapova_e_count_freq_chart_mpi::TestMPITaskSequential test_mpi_task_sequential(task_data_seq);
    ASSERT_EQ(test_mpi_task_sequential.Validation(), true);
    test_mpi_task_sequential.PreProcessing();
    test_mpi_task_sequential.Run();
    test_mpi_task_sequential.PostProcessing();
    ASSERT_EQ(reference_out[0], global_out[0]);
  }
}
TEST(budazhapova_e_count_freq_chart_mpi, ordinary_string_2) {
  boost::mpi::communicator world;
  std::string global_str = "ergtergdfgdgesr";
  std::vector<int> global_out(1, 0);
  char symb = 'r';
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> task_data_par = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    task_data_par->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_str.data()));
    task_data_par->inputs_count.emplace_back(global_str.size());
    task_data_par->inputs.emplace_back(reinterpret_cast<uint8_t*>(&symb));
    task_data_par->inputs_count.emplace_back(1);
    task_data_par->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_out.data()));
    task_data_par->outputs_count.emplace_back(global_out.size());
  }
  budazhapova_e_count_freq_chart_mpi::TestMPITaskParallel test_mpi_task_parallel(task_data_par);
  ASSERT_EQ(test_mpi_task_parallel.Validation(), true);
  test_mpi_task_parallel.PreProcessing();
  test_mpi_task_parallel.Run();
  test_mpi_task_parallel.PostProcessing();
  if (world.rank() == 0) {
    // Create data
    std::vector<int> reference_out(1, 0);
    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> task_data_seq = std::make_shared<ppc::core::TaskData>();
    task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_str.data()));
    task_data_seq->inputs_count.emplace_back(global_str.size());
    task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&symb));
    task_data_seq->inputs_count.emplace_back(1);
    task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_out.data()));
    task_data_seq->outputs_count.emplace_back(reference_out.size());

    // Create Task
    budazhapova_e_count_freq_chart_mpi::TestMPITaskSequential test_mpi_task_sequential(task_data_seq);
    ASSERT_EQ(test_mpi_task_sequential.Validation(), true);
    test_mpi_task_sequential.PreProcessing();
    test_mpi_task_sequential.Run();
    test_mpi_task_sequential.PostProcessing();
    ASSERT_EQ(reference_out[0], global_out[0]);
  }
}
TEST(budazhapova_e_count_freq_chart_mpi, ordinary_string_3) {
  boost::mpi::communicator world;
  std::string global_str = "gdfhjdsfghbehjfbhjsef gejhfgse jf segfsgefyusg df gefgsugfdjhfsef";
  std::vector<int> global_out(1, 0);
  char symb = 'f';
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> task_data_par = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    task_data_par->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_str.data()));
    task_data_par->inputs_count.emplace_back(global_str.size());
    task_data_par->inputs.emplace_back(reinterpret_cast<uint8_t*>(&symb));
    task_data_par->inputs_count.emplace_back(1);
    task_data_par->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_out.data()));
    task_data_par->outputs_count.emplace_back(global_out.size());
  }
  budazhapova_e_count_freq_chart_mpi::TestMPITaskParallel test_mpi_task_parallel(task_data_par);
  ASSERT_EQ(test_mpi_task_parallel.Validation(), true);
  test_mpi_task_parallel.PreProcessing();
  test_mpi_task_parallel.Run();
  test_mpi_task_parallel.PostProcessing();
  if (world.rank() == 0) {
    // Create data
    std::vector<int> reference_out(1, 0);
    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> task_data_seq = std::make_shared<ppc::core::TaskData>();
    task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_str.data()));
    task_data_seq->inputs_count.emplace_back(global_str.size());
    task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&symb));
    task_data_seq->inputs_count.emplace_back(1);
    task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_out.data()));
    task_data_seq->outputs_count.emplace_back(reference_out.size());

    // Create Task
    budazhapova_e_count_freq_chart_mpi::TestMPITaskSequential test_mpi_task_sequential(task_data_seq);
    ASSERT_EQ(test_mpi_task_sequential.Validation(), true);
    test_mpi_task_sequential.PreProcessing();
    test_mpi_task_sequential.Run();
    test_mpi_task_sequential.PostProcessing();
    ASSERT_EQ(reference_out[0], global_out[0]);
  }
}