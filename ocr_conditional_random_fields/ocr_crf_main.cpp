#include <cstdio>
#include <iostream>

#include <unordered_map>
#include <random>
#include <chrono>

#include <cpputil/cpputil.hpp>
#include "dataset_util.h"
#include "crf_typedef.h"
#include "crf_inference.h"
#include "crf_cost_func.h"
#include "crf_learning.h"

#ifdef _DEBUG
#include <cppyplot.hpp>
#endif

int main()
{
#ifdef _DEBUG
  Cppyplot::cppyplot pyp;
#endif
  const auto root = (std::filesystem::path(__FILE__)).parent_path().string();
  auto dat_files = files_with_extension(root + "/dataset/data/processed/breta/words_gaplines", ".dat");
  
  // TODO: remove this once everything looks good
  // test algorithm correctness with a small batch
  size_t n = dat_files.size();

  crf::Words_t images(n);
  vector<vector<size_t>> labels_int(n);
  vector<string> labels_str(n);

  std::unordered_map<char, size_t> letter_map = letters2int();

  // Read Dataset 
  std::cout << "reading dataset ...\n";
  for (size_t i = 0u; i < n; i++)
  {
    tie (labels_str[i], images[i]) = read_png_data<15, 25>(dat_files[i]);
    
    labels_int[i] = vector<size_t>(labels_str[i].size());

    for (size_t j = 0u; j < labels_str[i].size(); j++)
    { labels_int[i][j] = letter_map.at(labels_str[i][j]); }

    printf("\r\tpercent complete: %0.2f", (float)(i)/(float)n);
  }

#if defined(_DEBUG) && (0)
  
  // show sample image
  auto row = 10u;
  for (unsigned int k = 0u; k < images[row].size(); k++)
  {
    auto& elem = images[row][k];
    auto& elem_label = labels_str[row][k];
    size_t elem_label_int = labels_int[row][k];
    pyp.raw(R"pyp(
    plt.figure(figsize=(6,5))
    plt.imshow(elem.reshape((25, 15)))
    plt.title(elem_label + '-' + str(elem_label_int), fontsize=14)
    plt.show()
    )pyp", _p(elem), _p(elem_label), _p(elem_label_int));
  }
#endif

  const size_t n_states = 26u;
  crf::NodeWeights_t node_weights;
  crf::TransWeights_t transition_weights; 

  // initialize node weights and transition weights
  std::random_device seed;
  std::mt19937 gen(seed());
  std::uniform_real_distribution<float> uniform(0.0F, 0.01F);
  
  for (auto row : node_weights.rowwise())
  { std::for_each(row.begin(), row.end(), [&](auto& elem){elem = uniform(gen); }); }

  for (auto row : transition_weights.rowwise())
  { std::for_each(row.begin(), row.end(), [&](auto& elem){elem = uniform(gen); }); }

  // Split Dataset
  auto[train_indices, test_indices] = util::train_test_split(images.size(), 0.7F);

  auto start_time = std::chrono::system_clock::now();

  // For each dataset call forward and backward algorithm for inference
  auto [cost_trend, weights, bias] = steepest_descent(images, labels_int, 
                                                      node_weights, transition_weights, 
                                                      log_conditional_prob, log_conditional_prime, 
                                                      50u, 0.0001F);
  auto end_time = std::chrono::system_clock::now();

  std::cout << "Elapsed: " << std::chrono::duration<double>(end_time - start_time).count() << '\n';

#ifdef _DEBUG
  pyp.raw(R"pyp(
    plt.figure(figsize=(6,5))
    plt.plot(cost_trend, 'g--x', markersize=2, linewidth=1.2)
    plt.grid(True)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.show()
  )pyp", _p(cost_trend));

  pyp.raw(R"pyp(
    plt.figure(figsize=(8,5))
    plt.imshow(weights[0, :].reshape(25, 15))
    plt.show()

    plt.figure(figsize=(8,5))
    plt.imshow(weights[1, :].reshape(25, 15))
    plt.show()
  )pyp", _p(weights));
#endif

  // Update Weights

  // write out parameters
  write_to_file("estimated_state_weights.txt", weights, {(size_t)weights.rows(), (size_t)weights.cols()});
  write_to_file("estimated_transition_weights.txt", bias, {(size_t)bias.rows(), (size_t)bias.cols()});

  return EXIT_SUCCESS;
}
