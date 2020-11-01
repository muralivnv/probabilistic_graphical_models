#ifndef _CRF_EVALUATE_H_
#define _CRF_EVALUATE_H_

#include <iostream>
#include "crf_typedef.h"
#include "crf_inference.h"


vector<vector<size_t>> predict_sequence(const crf::Words_t&        features, 
                                        const vector<size_t>&      indices_to_use,
                                        const crf::NodeWeights_t&  node_weights,
                                        const crf::TransWeights_t& trans_weights)
{
  vector<vector<size_t>> prediction_result(indices_to_use.size(), vector<size_t>{});
  for (size_t i = 0u; i < indices_to_use.size(); i++)
  {
    size_t data_idx = indices_to_use[i];
    prediction_result[i] = viterbi_decode(features[data_idx], node_weights, trans_weights);
  }

  return prediction_result;
}


float calc_accuracy(const crf::Words_t&           features,
                    const vector<vector<size_t>>& labels,
                    const vector<size_t>&         indices_to_use,
                    const crf::NodeWeights_t&     node_weights,
                    const crf::TransWeights_t&    trans_weights)
{
  float accuracy = 0.0F;
  float n_letters = 0.0F;
  for (size_t i = 0u; i < indices_to_use.size(); i++)
  {
    size_t data_idx   = indices_to_use[i];
    const auto& label = labels[data_idx];

    auto predicted_label = viterbi_decode(features[data_idx], node_weights, trans_weights);
    // int not_matched = 0;
    for (size_t t = 0u; t < predicted_label.size(); t++)
    {
      // if (label[t] != predicted_label[t])
      // {
      //   not_matched |= 1u;
      // }
      if (label[t] == predicted_label[t])
      {
        accuracy += 1.0F;
      }
      n_letters += 1.0F;
    }

    // if (not_matched == 0)
    // {
    //   accuracy += 1.0F;
    // }
  }

  // return accuracy/static_cast<float>(indices_to_use.size());
  return accuracy/n_letters;
}

#endif
