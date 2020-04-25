#ifndef _CUSTOM_UTILS_H_
#define _CUSTOM_UTILS_H_

#include <vector>
#include <map>
#include <string>
#include <sstream>
#include <fstream>
#include <algorithm>

#include <cstdlib>
#include <ctime>
#include <cmath>

struct wordProbability{
  std::vector<float> probabilities;
  wordProbability(int n=2, const float init_val=0.0F):probabilities(std::vector<float>(n, init_val)){}
  ~wordProbability()
  { probabilities.clear(); }
};

struct predictionMetric{
  float accuracy;
  float f1_score;
  float precision;
  float recall;
  predictionMetric():accuracy(0.0F), f1_score(0.0F), precision(0.0F), recall(0.0F){}
  void reset()
  { this->accuracy = 0.0F; this->f1_score = 0.0F; this->precision = 0.0F; this->recall = 0.0F;  }
};

typedef std::map<int, std::vector<std::string>> SMS_DATASET_TYPE;
typedef std::map<std::string, wordProbability> FEATURE_PROBABILITY_TYPE;
typedef std::vector<std::vector<int>>           MAT_INT_2D;

std::map<std::string, int> identifiers_to_int{{"ham", 0}, {"spam", 1}};

void read_data(const std::string& data_filename, SMS_DATASET_TYPE& container_to_fill)
{
  std::ifstream sms_dataset_obj(data_filename, std::ios_base::in);

  // read in the data
  if (sms_dataset_obj.is_open())
  {
    std::string current_line;
    while(!sms_dataset_obj.eof())
    { 
      std::getline(sms_dataset_obj, current_line);

      std::size_t first_space_index = 0;
      first_space_index = current_line.find_first_of('\t');
      std::string this_data_key = current_line.substr(0, first_space_index);
      int this_data_key_int = identifiers_to_int[this_data_key];
      if (container_to_fill.find(this_data_key_int) == container_to_fill.end())
      {
        container_to_fill[this_data_key_int] = std::vector<std::string>{};
        container_to_fill[this_data_key_int].reserve(1000);
      }
      container_to_fill[this_data_key_int].push_back(current_line.substr(first_space_index+1, std::string::npos));
      if ((container_to_fill[this_data_key_int].size() - container_to_fill[this_data_key_int].capacity()) < 10u)
      {
        container_to_fill[this_data_key_int].reserve(1000);
      }
    }
  }
  sms_dataset_obj.close();

  for (std::pair<const int, std::vector<std::string>>&  row: container_to_fill)
  { row.second.shrink_to_fit();  }
}


void process_data(SMS_DATASET_TYPE& dataset)
{
  auto chars_to_remove = [](char c){return (std::ispunct(static_cast<unsigned char>(c)) || std::isdigit(c));};

  // remove punctuation marks, full-stops, commas, ...
  for (std::pair<const int, std::vector<std::string>>& row: dataset)
  {
    for (std::size_t iter = 0u; iter < row.second.size(); iter++)
    {
      std::string& this_string = row.second[iter];
      this_string.erase(std::remove_if(this_string.begin(), this_string.end(), chars_to_remove), this_string.end());
    }
  }
}


void generate_permutations(std::vector<int>& permutations_out, const std::size_t len)
{
  // first fill permutations_out 
  permutations_out = std::vector<int> (len, 0);
  int n = -1;
  std::generate(permutations_out.begin(), permutations_out.end(), [n=-1]() mutable {return n++;});

  n = len;
  
  srand(time(NULL));
  for (std::size_t iter = 0u; iter < len; iter++)
  {
    int rand_index = rand() % n;

    // swap first and random index
    int temp = permutations_out[len-n];
    permutations_out[len-n] = permutations_out[(len-n)+rand_index];
    permutations_out[rand_index] = temp;
    n--;
  }
}


void train_test_split(const SMS_DATASET_TYPE& dataset_container, 
                      const float train_frac, 
                      MAT_INT_2D& train_data_indices, 
                      MAT_INT_2D& test_data_indices)
{
  train_data_indices = MAT_INT_2D(dataset_container.size());
  test_data_indices  = MAT_INT_2D(dataset_container.size());

  for (auto& class_type : dataset_container)
  {
    int train_len = (int)((float)class_type.second.size()*train_frac);
    int test_len  = class_type.second.size() - train_len;
    std::vector<int> permutations;

    generate_permutations(permutations, class_type.second.size());

    train_data_indices[class_type.first] = std::vector<int>(train_len, 0);
    test_data_indices[class_type.first] = std::vector<int>(test_len, 0);

    std::copy_n(permutations.begin(), train_len, train_data_indices[class_type.first].begin());
    std::copy(permutations.begin()+train_len, permutations.end(), test_data_indices[class_type.first].begin());
  }
}


void calc_probabilities(const SMS_DATASET_TYPE&         input_dataset, 
                        const MAT_INT_2D&               data_indices_to_use,
                              FEATURE_PROBABILITY_TYPE& feature_probabilities)
{
  auto is_space = [](char c){ return (c == ' ');};
  std::vector<std::size_t> total_word_count(input_dataset.size(), 0);

  for (const auto& row: input_dataset)
  {
    auto& data_indices_this_class = data_indices_to_use[row.first];

    for (std::size_t iter = 0u; iter < data_indices_this_class.size(); iter++)
    {
      std::string current_string = row.second[data_indices_this_class[iter]];
      std::istringstream token_stream(current_string);
      std::string word;

      while(std::getline(token_stream, word, ' '))
      {
        word.erase(std::remove_if(word.begin(), word.end(), is_space), word.end());
        if (word.length() > 1u)
        {
          // change this word to lower case
          std::transform(word.begin(), word.end(), word.begin(), std::tolower);

          // if this word is not in the Bag-of-words container then add and initialize
          if (feature_probabilities.find(word) == feature_probabilities.end())
          {
            feature_probabilities[word] = wordProbability(input_dataset.size());
          }

          feature_probabilities[word].probabilities[row.first] += 1.0F;
          total_word_count[row.first]++;
        }
      }
    }
  }

  // now divide each word in each class with the total number of words in it's class
  // there is a chance we might encounter absolutely zero probabilities for a word in any class
  // for this use laplace smoothing
  const float smoothing_constant = 1.0F;
  for (std::pair<const std::string, wordProbability>& this_word : feature_probabilities)
  {
    for (std::size_t class_iter = 0u; class_iter < input_dataset.size(); class_iter++)
    {
      this_word.second.probabilities[class_iter] += smoothing_constant;
      float denominator = (float)total_word_count[class_iter] + smoothing_constant*((float)data_indices_to_use[class_iter].size());
      this_word.second.probabilities[class_iter] /= denominator;
    }
  }
}


void predict_class(const SMS_DATASET_TYPE&          input_dataset,
                   const MAT_INT_2D&                data_indices_to_use,
                   const FEATURE_PROBABILITY_TYPE&  feature_probabilities,
                         MAT_INT_2D&                predicted_labels)
{
  auto is_space = [](char c){ return (c == ' ');};
  std::vector<float> class_probabilities(input_dataset.size());
  predicted_labels = MAT_INT_2D(input_dataset.size());

  std::size_t total_docs = 0u;

  // calculate class_probabilities
  for (auto& class_type: input_dataset)
  { 
    total_docs += class_type.second.size(); 
    predicted_labels[class_type.first] = std::vector<int>(data_indices_to_use[class_type.first].size(), 0);
  }

  for (int class_iter = 0u; class_iter < data_indices_to_use.size(); class_iter++)
  {
    auto& data_indices_this_class = data_indices_to_use[class_iter];
    auto& this_class_dataset      = input_dataset.at(class_iter);

    class_probabilities[class_iter] = (float)(this_class_dataset.size())/(float)(total_docs);
    for (std::size_t iter = 0u; iter < data_indices_this_class.size(); iter++)
    {
      wordProbability this_document_probs(input_dataset.size(), 1.0F);
      std::string current_string = this_class_dataset[data_indices_this_class[iter]];
      std::istringstream token_stream(current_string);
      std::string word;

      while(std::getline(token_stream, word, ' '))
      {
        word.erase(std::remove_if(word.begin(), word.end(), is_space), word.end());
        if (word.length() > 1u)
        {
          // change this word to lower case
          std::transform(word.begin(), word.end(), word.begin(), std::tolower);

          // if this word is not in the Bag-of-words container then add and initialize
          if (feature_probabilities.find(word) != feature_probabilities.end())
          {
            auto& this_word_prob = feature_probabilities.at(word).probabilities;
            for (std::size_t word_prob_iter = 0u; word_prob_iter < this_word_prob.size(); word_prob_iter++)
            {
              this_document_probs.probabilities[word_prob_iter] += std::log(this_word_prob[word_prob_iter]);
            }
          }
        }
      }

      // now flag class with maximum probability as the predicted class
      float max_prob = 0.0F;
      for (std::size_t doc_prob_iter = 0u; doc_prob_iter < this_document_probs.probabilities.size(); doc_prob_iter++)
      {
        this_document_probs.probabilities[doc_prob_iter] += std::log(class_probabilities[class_iter]);
        if (this_document_probs.probabilities[doc_prob_iter] > max_prob)
        {
          predicted_labels[class_iter][iter] = doc_prob_iter;
          max_prob = this_document_probs.probabilities[doc_prob_iter];
        }
      }
    }
  }
}


void evaluate_result(const SMS_DATASET_TYPE& input_dataset,
                     const MAT_INT_2D&        data_indices_to_use,
                     const MAT_INT_2D&        predicted_labels,
                           predictionMetric&  evaluated_metric)
{
  float true_positive_count  = 0.0F;
  float false_positive_count = 0.0F;
  float true_negative_count  = 0.0F;
  float false_negative_count = 0.0F;
  evaluated_metric.reset();

  for (auto& class_input : input_dataset)
  {
    for (std::size_t index_iter = 0u; index_iter < data_indices_to_use[class_input.first].size(); index_iter++)
    {
      true_positive_count  += (class_input.first & predicted_labels[class_input.first][index_iter]) > 0?1.0F:0.0F;
      false_positive_count += (1-class_input.first) & (predicted_labels[class_input.first][index_iter]) > 0?1.0F:0.0F;
      true_negative_count  += (1-class_input.first) & (1-predicted_labels[class_input.first][index_iter]) >0?1.0F:0.0F;
      false_negative_count += (class_input.first) & (1-predicted_labels[class_input.first][index_iter]) >0?1.0F:0.0F;
    }
  }

  evaluated_metric.recall    = true_positive_count / (true_positive_count + false_negative_count);
  evaluated_metric.precision = true_positive_count / (true_positive_count + false_positive_count);
  evaluated_metric.accuracy  = (true_positive_count + true_negative_count) / (true_positive_count + false_negative_count + false_positive_count + true_negative_count);
  evaluated_metric.f1_score  = 2.0F*evaluated_metric.recall*evaluated_metric.precision/(evaluated_metric.recall + evaluated_metric.precision);
}


std::ostream& operator<<(std::ostream& os, const predictionMetric& metric_out)
{
  os << "accuracy: "  << metric_out.accuracy << '\n';
  os << "precision: " << metric_out.precision << '\n';
  os << "recall: "    << metric_out.recall << '\n';
  os << "f1_score: "  << metric_out.f1_score << '\n';

  return os;
}

#endif
