#ifndef _HMM_UTILS_H_
#define _HMM_UTILS_H_

#include <string>
#include <vector>
#include <array>
#include <unordered_map>

#include <algorithm>
#include <numeric>

#include "custom_typedefs.h"
#include "custom_utils.h"

using std::string;
using std::vector;
using std::size_t;
using std::unordered_map;
using std::pair;
using std::begin;
using std::end;

static HASH_MAP<string, uint_t> common_tags_map;

/*! 
*  \brief This function counts the number of times a word-tag appeared in the given dataset 
*         using the indices that were supplied to this function
*
*  \param[in]   dataset_tags         tags for the corpus-data 
*  \param[in]   common_tags          unique common tags over the entire dataset 
*  \param[in]   train_data_indices   data indices to be used to do tag count  
*  \param[out]  tag_unigram_count    calculated tag counts, size 1 x n_tags
*
*/
template<size_t NStates>
void unigram_count(const DMAT_STR&              dataset_tags,
                   const VEC<string, NStates>&  common_tags,
                   const DVEC_UINT&             train_data_indices,
                         VEC<uint_t, NStates>&  tag_unigrams_count)
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
    const DVEC_STR& this_training_labels = dataset_tags[train_data_indices[train_iter]];

    for (size_t word_iter = 0u; word_iter < this_training_labels.size(); word_iter++)
    {
      uint_t this_label_idx    = common_tags_map.at(this_training_labels[word_iter]);
      tag_unigrams_count[this_label_idx]++;
    }
  }
}


/*! 
*  \brief This function will count number of times word B appeared after word A (bigram)
*
*  \param[in]   dataset_tags         tags for the corpus-data 
*  \param[in]   common_tags          unique common tags over the entire dataset 
*  \param[in]   train_data_indices   data indices to be used to do tag count  
*  \param[out]  tag_bigram_count     calculated bigram tag count, size n_tags x n_tags
*
*/
template <size_t NStates>
void bigram_count(const DMAT_STR&                      dataset_tags,
                  const VEC<string, NStates>&          common_tags,
                  const DVEC_UINT&                     train_data_indices,
                        MAT<uint_t, NStates, NStates>& tag_bigram_count)
{
  tag_bigram_count.fill({});

  // create a map between common tags to int if this map is not initialized before
  if (common_tags_map.size() != common_tags.size())
  {  
    for (uint_t iter = 0u; iter < common_tags.size(); iter++)
    {  common_tags_map.insert(std::make_pair(common_tags[iter], iter));  }
  }

  for (size_t train_iter = 0u; train_iter < train_data_indices.size(); train_iter++)
  {
    const DVEC_STR& this_training_labels = dataset_tags[train_data_indices[train_iter]];

    for (size_t word_iter = 1u; word_iter < this_training_labels.size(); word_iter++)
    {
      uint_t prev_label_idx    = common_tags_map.at(this_training_labels[word_iter-1u]);
      uint_t current_label_idx = common_tags_map.at(this_training_labels[word_iter]);
      tag_bigram_count[prev_label_idx][current_label_idx]++;
    }
  }
}


/*! 
*  \brief This function counts number of times a tag has appeared at the beginning 
*         and at the end of a sentence
*
*  \param[in]   dataset_tags         tags for the corpus-data 
*  \param[in]   common_tags          unique common tags over the entire dataset 
*  \param[in]   train_data_indices   data indices to be used to do tag count
*  \param[out]  starting_tag_counts  output of the overall tag count that appear at the beginning of sentence, size 1 x n_tags
*  \param[out]  ending_tag_counts    output of the overall tag count that appear at the end of sentence, size 1 x n_tags
*
*/
template<size_t NStates>
void starting_ending_tag_count(const DMAT_STR&               dataset_tags,
                               const VEC<string, NStates>&   common_tags, 
                               const DVEC_UINT&              train_data_indices,
                                     VEC<uint_t, NStates>&   starting_tag_counts,
                                     VEC<uint_t, NStates>&   ending_tag_counts)
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
    const DVEC_STR& this_training_labels = dataset_tags[train_data_indices[train_iter]];
    if (this_training_labels.size() > 0u)
    { 
      uint_t starting_tag_idx = common_tags_map.at(this_training_labels.front());
      uint_t ending_tag_idx   = common_tags_map.at(this_training_labels.back());
    
      starting_tag_counts[starting_tag_idx]++;
      ending_tag_counts[ending_tag_idx]++;
    }
  }
}


/*! 
*  \brief This function calculates the probability of transition from one state to another 
*
*  \param[in]   tag_unigram_count       unigram tag counts over a training data
*  \param[in]   tag_bigram_count        bigram tag count over a training data
*  \param[out]  state_transition_prob   calculated state transition probabilities of size n_tags x n_tags
*
*/
template <size_t NStates>
void calc_transition_probabilities(const VEC<uint_t, NStates>&          tag_unigram_count, 
                                   const MAT<uint_t, NStates, NStates>& tag_bigram_count, 
                                         MAT<float, NStates, NStates>&  state_transition_prob)
{
  // initialize state-transition probabilities to 0.0F
  state_transition_prob.fill({});

  for (uint_t tag_iter_outer = 0u; tag_iter_outer < tag_unigram_count.size(); tag_iter_outer++)
  {
    for (uint_t tag_iter_inner = 0u; tag_iter_inner < tag_unigram_count.size(); tag_iter_inner++)
    {
      // $P(tag_2|tag_1) = \frac{Count(tag_2, tag_1)}{Count(tag_1)}$
      state_transition_prob[tag_iter_outer][tag_iter_inner] = ((float)tag_bigram_count[tag_iter_outer][tag_iter_inner])
                                                              /((float)tag_unigram_count[tag_iter_outer]);
    }
  }
}


/*! 
*  \brief This function calculates the observation probability given the state (in this part-of-speech tag)
*
*  \param[in]   sentences              dataset of sentences
*  \param[in]   labels                 dataset of labels(tags) for the sentences
*  \param[in]   common_tags            unique common tags over the entire dataset 
*  \param[in]   train_indices          data indices used to do unigram-tag count
*  \param[in]   tag_unigram_count      precalculated uni tag count
*  \param[out]  observation_prob       calculated observation probabilities, P (Obs | State)
*
*/
template <size_t NStates>
void calc_observation_probabilities(const DMAT_STR&                              sentences, 
                                    const DMAT_STR&                              labels, 
                                    const VEC<string, NStates>&                  common_tags,
                                    const DVEC_UINT&                             train_indices,
                                    const VEC<uint_t, NStates>&                  tag_unigram_count,
                                          HASH_MAP<string, VEC<float, NStates>>& observation_prob)
{
  // create a map between common tags to int if this map is not initialized before
  if (common_tags_map.size() != common_tags.size())
  {
    for (uint_t iter = 0u; iter < common_tags.size(); iter++)
    {  common_tags_map.insert(std::make_pair(common_tags[iter], iter));  }
  }

  for (size_t train_idx = 0u; train_idx < train_indices.size(); train_idx++)
  {
    const DVEC_STR& this_sentence = sentences[train_indices[train_idx]];
    for (size_t word_idx = 0u; word_idx < this_sentence.size(); word_idx++)
    {
      string this_word {this_sentence[word_idx]};

      const string& this_word_tag = labels[train_indices[train_idx]][word_idx];
      const uint_t this_word_tag_idx = common_tags_map.at(this_word_tag);
      const float this_tag_inverse_count = 1.0F/((float)tag_unigram_count[this_word_tag_idx]);
      
      to_lower_str(this_word);
      insert_if_not_exist(observation_prob, this_word, {});

      observation_prob.at(this_word)[this_word_tag_idx] += this_tag_inverse_count;
    }
  }

  // do normalization for each unique word so as to have total sum of prob to 1.0
  for (auto& observation : observation_prob)
  {
    auto& tag_prob = observation.second;
    float observation_sum = std::accumulate(tag_prob.begin(), tag_prob.end(), 0.0F);
    std::for_each(tag_prob.begin(), tag_prob.end(), [observation_sum](float& prob){ prob /= observation_sum; });
  }
}


/*! 
*  \brief This function calculates the initial tag probabilities and the final tag probabilities
*
*  \param[in]   starting_tag_counts    num times a tag starts in training dataset
*  \param[in]   ending_tag_counts      num times a tag ends in training dataset
*  \param[in]   training_set_len       length of the training set used to do the tag count 
*  \param[in]   tag_unigram_count      precalculated uni-tag count
*  \param[out]  start_end_prob         calculate start/end probabilities, size 2 x n_tags 
*                                 (row 0: start probabilities, row 1: end probablities)
*
*/
template <size_t NStates>
void calc_start_end_probabilities(const VEC<uint_t, NStates>&    starting_tag_counts, 
                                  const VEC<uint_t, NStates>&    ending_tag_counts,
                                  const size_t                   training_set_len,
                                  const VEC<uint_t, NStates>&    tag_unigram_count,
                                        MAT<float, 2u, NStates>& start_end_prob)
{
  const float train_len_inverse = 1.0F/(float)(training_set_len);

  start_end_prob.fill({});

  for (size_t tag_iter = 0u; tag_iter < starting_tag_counts.size(); tag_iter++)
  {
    start_end_prob[0u][tag_iter] = ((float)starting_tag_counts[tag_iter])*train_len_inverse;
    start_end_prob[1u][tag_iter] = ((float)ending_tag_counts[tag_iter])/(float)tag_unigram_count[tag_iter];
  }
}


/*! 
*  \brief This function predicts part-of-speech of a given dataset using Viterbi-algorithm
*
*  \param[in]   sentences               dataset of sentences
*  \param[in]   test_indices            index of the sentence whose part-of-speech tags need to be predicted
*  \param[in]   transition_prob         precalculate state-transition probabilities
*  \param[in]   observation_prob        precalculated observation probabilities
*  \param[in]   start_end_prob          precalculated start/end state probabilities
*  \param[out]  predicted_pos_tags      predicted part-of-speech tags for the given data indices using viterbi
*
*/
template<size_t NStates>
void viterbi_decode(const DMAT_STR&                              sentences, 
                    const DVEC_UINT&                             test_indices, 
                    const MAT<float, NStates, NStates>&          transition_prob,
                    const HASH_MAP<string, VEC<float, NStates>>& observation_prob, 
                    const MAT<float, 2u, NStates>&               start_end_prob,
                          DMAT_UINT&                             predicted_pos_tags)
{
  predicted_pos_tags = DMAT_UINT(test_indices.size());

  for (size_t test_idx = 0u; test_idx < test_indices.size(); test_idx++)
  {
    const DVEC_STR& this_sentence = sentences[test_indices[test_idx]];
    const size_t n_obs = this_sentence.size();
    if (n_obs > 0u)
    { 
      predicted_pos_tags[test_idx] = DVEC_UINT(n_obs, 0u);
      
      vector<vector<ViterbiPathNode>> viterbi_path(n_obs, vector<ViterbiPathNode>(NStates));

      float max_obs_state_prob   = -10.0F;
      uint_t prev_best_state     = 0u;

      string this_word = this_sentence[0];

      // starting state
      for (size_t tag_index = 0u; tag_index < NStates; tag_index++)
      {
        // unknown word
        to_lower_str(this_word);
        if (observation_prob.find(this_word) != observation_prob.end())
        { 
          viterbi_path[0u][tag_index].prev_state = 0u;
          viterbi_path[0u][tag_index].max_prob =   fast_log2f(start_end_prob[0u][tag_index])               /* P (Y   | start) */
                                                 + fast_log2f(observation_prob.at(this_word)[tag_index]); /* P (X_1 | Y)     */
        }
      }

      // transition states
      for (size_t word_idx = 1u; word_idx < this_sentence.size(); word_idx++)
      {
        this_word = this_sentence[word_idx];
        to_lower_str(this_word);

        // unknown word
        if (observation_prob.find(this_word) == observation_prob.end())
        { continue; }

        for (uint_t cur_state_idx = 0u; cur_state_idx < NStates; cur_state_idx++)
        {
          max_obs_state_prob  = viterbi_path[word_idx-1u][0].max_prob 
                                    + fast_log2f(transition_prob[0][cur_state_idx]);
          prev_best_state     = 0u;

          for (uint_t prev_state_idx = 1u; prev_state_idx < NStates; prev_state_idx++)
          {
            float cur_state_prob = viterbi_path[word_idx-1u][prev_state_idx].max_prob 
                                  + fast_log2f(transition_prob[prev_state_idx][cur_state_idx]);

            if (max_obs_state_prob < cur_state_prob)
            {
              max_obs_state_prob  = cur_state_prob;
              prev_best_state     = prev_state_idx;
            }
          }

          viterbi_path[word_idx][cur_state_idx].prev_state = prev_best_state;
          viterbi_path[word_idx][cur_state_idx].max_prob   = max_obs_state_prob + fast_log2f(observation_prob.at(this_word)[cur_state_idx]);  
        }
      }

      // ending state
      auto row_iterator = viterbi_path.back().begin();
      float final_best_prob  = -10.0F;
      uint_t best_last_state = 0u; 
      for (uint_t tag_idx = 0u; tag_idx < NStates; tag_idx++)
      {
        (row_iterator + tag_idx)->max_prob += fast_log2f(start_end_prob[1u][tag_idx]);
        if (final_best_prob < (row_iterator + tag_idx)->max_prob)
        {
          final_best_prob = (row_iterator + tag_idx)->max_prob;
          best_last_state = tag_idx;
        }
      }

      predicted_pos_tags[test_idx].back() = best_last_state;

      // back track viterbi path based on the last best state
      for (int state_iter = (int)n_obs-2; state_iter >= 0; state_iter--)
      {
        predicted_pos_tags[test_idx][state_iter] = viterbi_path[state_iter+1][best_last_state].prev_state;
        best_last_state = predicted_pos_tags[test_idx][state_iter];
      }
    }
  }
}


/*! 
*  \brief This function predicts part-of-speech of a given dataset using Viterbi-algorithm
*
*  \param[in]  dataset_tags            tags for the corpus-data 
*  \param[in]  test_indices            indices that were used to do part-of-speech prediction on
*  \param[in]  predicted_pos_tags      predicted part-of-speech on the dataset using the test_indices
*
*  \return accuracy                calculate accuracy of the prediction
*/
float evaluate_accuracy(const DMAT_STR&  dataset_tags,
                        const DVEC_UINT& test_indices, 
                        const DMAT_UINT& predicted_pos_tags)
{
  size_t correct_prediction_count = 0u;
  size_t total_tags = 0u;

  for (size_t test_idx = 0u; test_idx < test_indices.size(); test_idx++)
  {
    const DVEC_STR& this_sentence_tags = dataset_tags[test_indices[test_idx]];
    for (size_t tag_index = 0u; tag_index < this_sentence_tags.size(); tag_index++)
    {
      total_tags++;
      const uint_t this_tag_idx = common_tags_map.at(this_sentence_tags[tag_index]);
      if (this_tag_idx == predicted_pos_tags[test_idx][tag_index])
      {
        correct_prediction_count++;
      }
    }
  }

  float accuracy = (float)(correct_prediction_count)/(float)(total_tags);
  return accuracy;
}

#endif
