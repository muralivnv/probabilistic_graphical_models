#include <iostream>
#include <vector>
#include <string>

#include "custom_utils.h"

int main()
{
  SMS_DATASET_TYPE sms_dataset_input;
  std::vector<std::vector<int>> train_data_indices, test_data_indices;
  const float train_frac = 0.7F;

  FEATURE_PROBABILITY_TYPE feature_probabilities;
  read_data("data/sms_spam_collection/SMSSpamCollection.txt", sms_dataset_input);

  // ## Pre-Process data
  // preprocess data by removing punctuation and numbers from the text
  process_data(sms_dataset_input);

  train_test_split(sms_dataset_input, 
                   train_frac, 
                   train_data_indices, 
                   test_data_indices);

  // ## Process 'bag of words' and create probabilities based on frequency of words in a given class
  // Bag of words is nothing but generating unique words out of the dataset and calculating
  // frequencies of words in each message
  // in this example it is good to remove stop_words {at, an, and, is..} as they generally tend to skew the distribution
  // In some-other context this stop_words might be useful as it will help us to model a person speech pattern

  // calculate probabilities using just the training data
  calc_probabilities(sms_dataset_input,
                     train_data_indices,
                     feature_probabilities);

  std::vector<std::vector<int>> predicted_labels;
  predictionMetric mnb_prediction_metric;

  // now process test data and classify each document
  predict_class(sms_dataset_input,
                test_data_indices,
                feature_probabilities,
                predicted_labels);

  evaluate_result(sms_dataset_input,
                  test_data_indices,
                  predicted_labels,
                  mnb_prediction_metric);

  std::cout << mnb_prediction_metric;

  return EXIT_SUCCESS;
}