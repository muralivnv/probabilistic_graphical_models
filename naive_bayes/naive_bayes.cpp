#include <iostream>
#include <vector>
#include <string>
#include <map>
#include <fstream>
#include <algorithm>

#include <Eigen/Eigen/Dense>
#include <Eigen/Eigen/Core>
#include "custom_utils.h"

using namespace Eigen;

int main()
{

  SMS_DATASET_TYPE sms_dataset_input;
  read_data("data/sms_spam_collection/SMSSpamCollection.txt", sms_dataset_input);

  // ## Pre-Process data
  // preprocess data by removing punctuation and numbers from the text
  process_data(sms_dataset_input);

  MatrixXi frequency_matrix_class1;
  MatrixXi frequency_matrix_class2;

  // ## Process 'bag of words' and create frequency matrix for each message
  // Bag of words is nothing but generating unique words out of the dataset and calculating
  // frequencies of words in each message
  // in this example it is good to remove stop_words {at, an, and, is..} as they generally tend to skew the distribution
  // In some-other context this stop_words might be useful as it will help us to model a person speech pattern
  tokenize_strings(sms_dataset_input,
                   frequency_matrix_class1,
                   frequency_matrix_class2);

  // ## Naive Bayes
  // Naive bayes uses bayes-rule to calculate posterior probability densities using prior probability densities
  // The main assumption that naive bayes rules takes is that the features are conditionally independent of each other
  // for example P(rain|cloudy,windy) = P(rain|cloudy)*P(rain|windy) even though cloudy and windy will generally give us
  // higher probability of rain. Neverthless, this assumption generally works pretty good.

  // $$ P(y|x_1,x_2,x_3,...x_n) = \frac{P(y) P(x_1,x_2,...x_n|y)}{P(x_1,x_2,...,x_n)} $$
  // because of conditional independence assumption across features, the above equation can be written as
  // $$ P(y|x_1,x_2,x_3,...x_n) = \frac{P(y) \prod_{i=1}^nP(x_i|y)}{P(x_1,x_2,....,x_n)} $$
  // $$ 

  return EXIT_SUCCESS;
}