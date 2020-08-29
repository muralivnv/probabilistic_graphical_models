#include <iostream>
#include <vector>
#include <array>
#include <string>
#include <cassert>
#include <chrono>

#include "hmm_utils.h"
#include "custom_utils.h"

using std::vector;
using std::string;
using std::size_t;

int main()
{
  float train_frac = 0.6F;

  const size_t n_hidden_states = 12u;
  VEC<string, n_hidden_states> tags;
  DMAT_STR sentences, labels;

  read_tags("data/tags-universal.txt", tags);
  read_text_corpus("data/brown-universal.txt", sentences, labels);
  
  auto start_time = std::chrono::system_clock::now();

  // split the data
  DVEC_UINT train_indices, test_indices;

  train_test_split(sentences, train_frac, train_indices, test_indices);

  // calculate count of each tag and pair of tags
  MAT<uint_t, n_hidden_states, n_hidden_states> tag_bigram_count;
  VEC<uint_t, n_hidden_states> tag_unigram_count;

  unigram_count(labels, tags, train_indices, tag_unigram_count);
  bigram_count(labels, tags, train_indices, tag_bigram_count);

  VEC<uint_t, n_hidden_states> starting_tag_count;
  VEC<uint_t, n_hidden_states> ending_tag_count;

  starting_ending_tag_count(labels, tags, train_indices, starting_tag_count, ending_tag_count);

  // calculate transition probabilities
  // $P(tag_2|tag_1) = \frac{Count(tag_2, tag_1)}{Count(tag_1)}$
  MAT<float, n_hidden_states, n_hidden_states> state_transition_prob;
  calc_transition_probabilities(tag_unigram_count, tag_bigram_count, state_transition_prob);

  // calculate observation probabilities
  HASH_MAP<std::string,                /* observation */
           VEC<float, n_hidden_states> /* probability w.r.to each state */
           >  observation_prob;
  calc_observation_probabilities(sentences, labels, tags, train_indices, tag_unigram_count, observation_prob);

  MAT<float, 2u, n_hidden_states> start_end_prob;
  calc_start_end_probabilities(starting_tag_count, ending_tag_count, train_indices.size(), tag_unigram_count, start_end_prob);
  
  DMAT_UINT predicted_pos_tags;
  viterbi_decode(sentences, test_indices, state_transition_prob, observation_prob, start_end_prob, predicted_pos_tags);

  float accuracy = evaluate_accuracy(labels, test_indices, predicted_pos_tags);
  std::cout << "prediction_accuracy: " << accuracy << '\n';

  auto end_time = std::chrono::system_clock::now();
  std::chrono::duration<double> elapsed = end_time - start_time;
  std::cout << "total elapsed time: " << elapsed.count() << " sec";

  return EXIT_SUCCESS;
}