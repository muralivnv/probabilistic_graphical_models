#ifndef _HMM_UTILS_H_
#define _HMM_UTILS_H_

#include <string>
#include <vector>
#include <array>
#include <unordered_map>

#include <algorithm>

#include "custom_typedefs.h"
#include "custom_utils.h"

using std::string;
using std::vector;
using std::size_t;
using std::unordered_map;
using std::begin;
using std::end;

static HASH_MAP<string, uint_t> common_tags_map;

template<size_t NTags>
void unigram_count(const DMAT_STR&            sentence_labels,
                   const VEC<string, NTags>&  common_tags,
                   const DVEC_UINT&           train_data_indices,
                         VEC<uint_t, NTags>&  tag_unigrams_count)
{
  tag_unigrams_count.fill(0u);

  // create a map between common tags to int if this map is not initialized before
  if (common_tags_map.size() != common_tags.size())
  {  
    for (uint_t iter = 0u; iter < common_tags.size(); iter++)
    {  common_tags_map.insert(std::make_pair(common_tags[iter], iter));  }
  }

  for (size_t train_iter = 0u; train_iter < train_data_indices.size(); train_iter++)
  {
    const DVEC_STR& this_training_labels = sentence_labels[train_data_indices[train_iter]];

    for (size_t word_iter = 0u; word_iter < this_training_labels.size(); word_iter++)
    {
      uint_t this_label_idx    = common_tags_map.at(this_training_labels[word_iter]);
      tag_unigrams_count[this_label_idx]++;
    }
  }
}


template <size_t NTags>
void bigram_count(const DMAT_STR&                  sentence_labels,
                  const VEC<string, NTags>&        common_tags,
                  const DVEC_UINT&                 train_data_indices,
                        MAT<uint_t, NTags, NTags>& tag_bigrams_count)
{
  tag_bigrams_count.fill({});

  // create a map between common tags to int if this map is not initialized before
  if (common_tags_map.size() != common_tags.size())
  {  
    for (uint_t iter = 0u; iter < common_tags.size(); iter++)
    {  common_tags_map.insert(std::make_pair(common_tags[iter], iter));  }
  }

  for (size_t train_iter = 0u; train_iter < train_data_indices.size(); train_iter++)
  {
    const DVEC_STR& this_training_labels = sentence_labels[train_data_indices[train_iter]];

    for (size_t word_iter = 1u; word_iter < this_training_labels.size(); word_iter++)
    {
      uint_t prev_label_idx    = common_tags_map.at(this_training_labels[word_iter-1u]);
      uint_t current_label_idx = common_tags_map.at(this_training_labels[word_iter]);
      tag_bigrams_count[prev_label_idx][current_label_idx]++;
    }
  }
}


template<size_t NTags>
void starting_ending_tag_count(const DMAT_STR&             sentence_labels,
                               const VEC<string, NTags>&   common_tags, 
                               const DVEC_UINT&            train_data_indices,
                                     VEC<uint_t, NTags>&   starting_tag_counts,
                                     VEC<uint_t, NTags>&   ending_tag_counts)
{

  starting_tag_counts.fill(0u);
  ending_tag_counts.fill(0u);

  // create a map between common tags to int if this map is not initialized before
  if (common_tags_map.size() != common_tags.size())
  {  
    for (uint_t iter = 0u; iter < common_tags.size(); iter++)
    {  common_tags_map.insert(std::make_pair(common_tags[iter], iter));  }
  }

  for (size_t train_iter = 0u; train_iter < train_data_indices.size(); train_iter++)
  {
    const DVEC_STR& this_training_labels = sentence_labels[train_data_indices[train_iter]];
    if (this_training_labels.size() > 0u)
    { 
      uint_t starting_tag_idx = common_tags_map.at(this_training_labels.front());
      uint_t ending_tag_idx   = common_tags_map.at(this_training_labels.back());
    
      starting_tag_counts[starting_tag_idx]++;
      ending_tag_counts[ending_tag_idx]++;
    }
  }
}


template <size_t NTags>
void calc_transition_probabilities(const VEC<uint_t, NTags>&        tag_unigram_count, 
                                   const MAT<uint_t, NTags, NTags>& tag_bigram_count, 
                                         MAT<float, NTags, NTags>&  state_transition_prob)
{
  // initialize state-transition probabilities to 0.0F
  state_transition_prob.fill({});

  for (uint_t tag_iter_outer = 0u; tag_iter_outer < tag_unigram_count.size(); tag_iter_outer++)
  {
    for (uint_t tag_iter_inner = 0u; tag_iter_inner < tag_unigram_count.size(); tag_iter_inner++)
    {
      // $P(tag_2|tag_1) = \frac{Count(tag_2, tag_1)}{Count(tag_1)}$
      state_transition_prob[tag_iter_outer][tag_iter_inner] = ((float)tag_bigram_count[tag_iter_outer][tag_iter_inner])
                                                              /(float)tag_unigram_count[tag_iter_outer];
    }
  }
}


template <size_t NTags>
void calc_observation_probabilities(const DMAT_STR&                            sentences, 
                                    const DMAT_STR&                            labels, 
                                    const VEC<string, NTags>&                  common_tags,
                                    const DVEC_UINT&                           train_indices,
                                    const VEC<uint_t, NTags>&                  tag_unigram_count,
                                          HASH_MAP<string, VEC<float, NTags>>& observation_prob)
{
  // create a map between common tags to int if this map is not initialized before
  if (common_tags_map.size() != common_tags.size())
  {  
    for (uint_t iter = 0u; iter < common_tags.size(); iter++)
    {  common_tags_map.insert(std::make_pair(common_tags[iter], iter));  }
  }

  for (size_t train_index = 0u; train_index < train_indices.size(); train_index++)
  {
    const DVEC_STR& this_sentence = sentences[train_indices[train_index]];
    for (size_t word_iter = 0u; word_iter < this_sentence.size(); word_iter++)
    {
      string this_word {this_sentence[word_iter]};

      const string& this_word_tag = labels[train_indices[train_index]][word_iter];
      const uint_t this_word_tag_idx = common_tags_map.at(this_word_tag);
      const float this_tag_inverse_count = 1.0F/((float)tag_unigram_count[this_word_tag_idx]);
      
      // to_lower_str(this_word);
      insert_if_not_exist(observation_prob, this_word, {});

      observation_prob.at(this_word)[this_word_tag_idx] += this_tag_inverse_count;
    }
  }
}


template <size_t NTags>
void calc_start_end_probabilities(const VEC<uint_t, NTags>&    starting_tag_counts, 
                                  const VEC<uint_t, NTags>&    ending_tag_counts,
                                  const size_t                 training_set_len,
                                  const VEC<uint_t, NTags>&    tag_unigram_count,
                                        MAT<float, 2u, NTags>& start_end_prob)
{
  const float train_len_inverse = 1.0F/(float)(training_set_len);

  start_end_prob.fill({});

  for (size_t tag_iter = 0u; tag_iter < starting_tag_counts.size(); tag_iter++)
  {
    start_end_prob[0u][tag_iter] = (float)(starting_tag_counts[tag_iter])*train_len_inverse;
    start_end_prob[1u][tag_iter] = (float)(ending_tag_counts[tag_iter])/tag_unigram_count[tag_iter];
  }
}


template<size_t NTags>
void viterbi_decode(const DMAT_STR&                            sentences, 
                    const DVEC_UINT&                           test_indices, 
                    const MAT<float, NTags, NTags>&            transition_prob,
                    const HASH_MAP<string, VEC<float, NTags>>& observation_prob, 
                    const MAT<float, 2u, NTags>&               start_end_prob,
                          DMAT_UINT&                           predicted_pos_tags)
{
  predicted_pos_tags = DMAT_UINT(test_indices.size());

  for (size_t test_index = 0u; test_index < test_indices.size(); test_index++)
  {
    const DVEC_STR& this_sentence = sentences[test_indices[test_index]];
    const size_t n_obs = this_sentence.size();
    if (n_obs > 0u)
    { 
      predicted_pos_tags[test_index] = DVEC_UINT(n_obs, 0u);
      
      vector<vector<ViterbiPathNode>> viterbi_path(n_obs, vector<ViterbiPathNode>(NTags));

      // calculate probabilities for the first variable using starting probabilities
      VEC<float, NTags> prev_state_prob;
      VEC<float, NTags> this_state_prob;
      typename VEC<float, NTags>::iterator prev_state_prob_iterator = begin(prev_state_prob);
      typename VEC<float, NTags>::iterator this_state_prob_iterator = begin(this_state_prob);

      float cur_obs_max_state_prob   = -10.0F;
      uint_t max_prob_prev_state_idx = 0u;

      string this_word = this_sentence[0];

      // starting state
      for (size_t tag_index = 0u; tag_index < NTags; tag_index++)
      {
        // unknown word
        if (observation_prob.find(this_word) != observation_prob.end())
        { 
          viterbi_path[0u][tag_index].prev_state = 0u;
          viterbi_path[0u][tag_index].max_prob =  start_end_prob[0u][tag_index]               /* P (Y   | start) */
                                                 *(observation_prob.at(this_word)[tag_index]); /* P (X_1 | Y)     */
        }
      }

      // transition states
      for (size_t word_iter = 1u; word_iter < this_sentence.size(); word_iter++)
      {
        this_word = this_sentence[word_iter];
      
        // unknown word
        if (observation_prob.find(this_word) == observation_prob.end())
        { continue; }

        for (uint_t cur_state_idx = 0u; cur_state_idx < NTags; cur_state_idx++)
        {
          cur_obs_max_state_prob  = viterbi_path[word_iter-1u][0].max_prob 
                                    * transition_prob[0][cur_state_idx];
          max_prob_prev_state_idx = 0u;

          for (uint_t prev_state_idx = 1u; prev_state_idx < NTags; prev_state_idx++)
          {
            float cur_state_prob = viterbi_path[word_iter-1u][prev_state_idx].max_prob 
                                  * transition_prob[prev_state_idx][cur_state_idx];

            if (cur_obs_max_state_prob < cur_state_prob)
            {
              cur_obs_max_state_prob  = cur_state_prob;
              max_prob_prev_state_idx = prev_state_idx;
            }
          }

          viterbi_path[word_iter][cur_state_idx].prev_state = max_prob_prev_state_idx;
          viterbi_path[word_iter][cur_state_idx].max_prob   = cur_obs_max_state_prob * observation_prob.at(this_word)[cur_state_idx];  
        }
      }

      // ending state
      typename vector<ViterbiPathNode>::iterator row_iterator = viterbi_path.back().begin();
      float final_best_prob  = -10.0F;
      uint_t best_last_state = 0u; 
      for (uint_t tag_index = 0u; tag_index < NTags; tag_index++)
      {
        (row_iterator + tag_index)->max_prob *= start_end_prob[1u][tag_index];
        if (final_best_prob < (row_iterator + tag_index)->max_prob)
        {
          final_best_prob = (row_iterator + tag_index)->max_prob;
          best_last_state = tag_index;
        }
      }

      predicted_pos_tags[test_index].back() = best_last_state;

      // back track viterbi path based on the last best state
      for (int state_iter = (int)n_obs-2; state_iter >= 0; state_iter--)
      {
        predicted_pos_tags[test_index][state_iter] = viterbi_path[state_iter+1][best_last_state].prev_state;
        best_last_state = predicted_pos_tags[test_index][state_iter];
      }
    }
  }
}


float evaluate_accuracy(const DMAT_STR& labels,
                        const DVEC_UINT& test_indices, 
                        const DMAT_UINT& predicted_pos_tags)
{
  size_t correct_prediction_count = 0u;
  size_t total_tags = 0u;

  for (size_t test_index = 0u; test_index < test_indices.size(); test_index++)
  {
    const DVEC_STR& this_sentence_tags = labels[test_indices[test_index]];
    for (size_t tag_index = 0u; tag_index < this_sentence_tags.size(); tag_index++)
    {
      total_tags++;
      const uint_t this_tag_idx = common_tags_map.at(this_sentence_tags[tag_index]);
      if (this_tag_idx == predicted_pos_tags[test_index][tag_index])
      {
        correct_prediction_count++;
      }
    }
  }

  float accuracy = (float)(correct_prediction_count)/(float)(total_tags);
  return accuracy;
}

#endif
