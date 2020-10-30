#include <cstdio>
#include <iostream>

#include <unordered_map>
#include <random>
#include <chrono>
#include <algorithm>

#include <cpputil/cpputil.hpp>
#include "dataset_util.h"
#include "crf_typedef.h"
#include "crf_inference.h"
#include "crf_cost_func.h"
#include "crf_learning.h"

#ifndef _DEBUG
#define _DEBUG
#endif
#ifdef _DEBUG
#include <cppyplot.hpp>
#endif

int main()
{
#ifdef _DEBUG
  Cppyplot::cppyplot pyp;
#endif
  const auto root = (std::filesystem::path(__FILE__)).parent_path().string();
  std::random_device seed;
  std::mt19937 gen(seed());

  auto dat_files = files_with_extension(root + "/dataset/data/processed/breta/words_gaplines", ".dat");
  std::shuffle(dat_files.begin(), dat_files.end(), gen);
  
  // TODO: remove this once everything looks good
  // test algorithm correctness with a small batch
  size_t n = 50u;

  crf::Words_t images(n);
  vector<vector<size_t>> labels_int(n);
  vector<string> labels_str(n);

  std::unordered_map<char, size_t> letter_map = letters2int();

  // Read Dataset 
  std::cout << "reading dataset ...\n";
  
  #pragma omp parallel for num_threads(2)
  for (int i = 0u; i < n; i++)
  {
    tie (labels_str[i], images[i]) = read_png_data<15, 25>(dat_files[i]);
    
    labels_int[i] = vector<size_t>(labels_str[i].size());

    for (int j = 0u; j < labels_str[i].size(); j++)
    { labels_int[i][j] = letter_map.at(labels_str[i][j]); }
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
  std::uniform_real_distribution<float> uniform(0.0F, 0.01F);

  for (auto row: node_weights.rowwise())
  { std::for_each(row.begin(), row.end(), [&](float& elem){ elem = uniform(gen);}); }

  for (auto row: transition_weights.rowwise())
  { std::for_each(row.begin(), row.end(), [&](float& elem){ elem = uniform(gen);}); }
  
  // Split Dataset
  auto[train_indices, test_indices] = util::train_test_split(images.size(), 0.7F);

  auto start_time = std::chrono::system_clock::now();

  // For each dataset call forward and backward algorithm for inference
  auto [cost_trend, weights, bias] = steepest_descent(images, labels_int, 
                                                      node_weights, transition_weights, 
                                                      log_conditional_prob, log_conditional_prime, 
                                                      25u, 0.005F); // TODO: 0.01 produces nan. Fix it!
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
    fig, ax = plt.subplots(2, 5, figsize=(8,5))
    ax = np.ravel(ax)
    titles = ''.join(['%c'%(k) for k in range(97, 97+ax.shape[0])])
    for i in range(0, ax.shape[0]):
      ax[i].imshow(weights[i, :].reshape(25, 15))
      ax[i].set_title(titles[i], fontsize=14)
    plt.show()
  )pyp", _p(weights));
#endif

  // write out estimated parameters
  write_to_file("estimated_state_weights.txt", weights, {(size_t)weights.rows(), (size_t)weights.cols()});
  write_to_file("initial_state_weights.txt", node_weights, {(size_t)node_weights.rows(), (size_t)node_weights.cols()});

  write_to_file("estimated_transition_weights.txt", bias, {(size_t)bias.rows(), (size_t)bias.cols()});
  write_to_file("initial_transition_weights.txt", transition_weights, {(size_t)transition_weights.rows(), (size_t)transition_weights.cols()});

  return EXIT_SUCCESS;
}
