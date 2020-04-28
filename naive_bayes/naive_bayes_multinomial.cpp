#include <iostream>
#include <vector>
#include <string>
#include <chrono>

#include "custom_utils.h"

int main()
{
  SMS_DATASET_TYPE sms_dataset_input;
  std::vector<std::vector<int>> train_data_indices, test_data_indices;
  const float train_frac = 0.7F;

  auto start_time = std::chrono::system_clock::now();

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

  // calculate probabilities using just the training data based on term frequency {TF}
  FEATURE_PROBABILITY_TYPE feature_probabilities;
  std::vector<std::vector<int>> predicted_labels;
  predictionMetric mnb_TF_prediction_metric;
  predictionMetric mnb_TFIDF_prediction_metric;

  // calculate probabilities using just term-frequency
  calc_TF(sms_dataset_input,
          train_data_indices,
          feature_probabilities);
  
  calc_word_weights(sms_dataset_input,
                    train_data_indices,
                    feature_probabilities);

  // predict labels for the test dataset
  predict_class(sms_dataset_input,
                test_data_indices,
                feature_probabilities,
                predicted_labels);

  // evaluate metrics
  evaluate_result(sms_dataset_input,
                  test_data_indices,
                  predicted_labels,
                  mnb_TF_prediction_metric);

  // calculate probabilities of word using term-frequency_inverse-document-frequency
  calc_TF_IDF(sms_dataset_input,
              train_data_indices,
              feature_probabilities);

  calc_word_weights(sms_dataset_input,
                    train_data_indices,
                    feature_probabilities);

  // predict labels for the test dataset
  predict_class(sms_dataset_input,
                test_data_indices,
                feature_probabilities,
                predicted_labels);

  // evaluate metrics
  evaluate_result(sms_dataset_input,
                  test_data_indices,
                  predicted_labels,
                  mnb_TFIDF_prediction_metric);

  auto end_time = std::chrono::system_clock::now();
  std::chrono::duration<double> elapsed = end_time - start_time;

  std::cout << "\nMultinomial-naiveBayes (using TermFrequency)\n";
  std::cout << std::string(20, '=') << '\n';
  std::cout << mnb_TF_prediction_metric;

  std::cout << "\nMultinomial-naiveBayes (using TermFrequency-Invese Document Frequency)\n";
  std::cout << std::string(20, '=') << '\n';
  std::cout << mnb_TFIDF_prediction_metric;

  std::cout << "\n total time elapsed: " << elapsed.count() << " sec\n";
  return EXIT_SUCCESS;
}