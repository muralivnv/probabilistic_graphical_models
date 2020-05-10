#ifndef _CUSTOM_UTILS_H_
#define _CUSTOM_UTILS_H_

#include <vector>
#include <unordered_map>
#include <string>
#include <sstream>
#include <fstream>
#include <algorithm>
#include <random>
#include <chrono>

#include <cmath>
#include <cstdint>

struct wordProbability{
  std::vector<float> probabilities;
  wordProbability(std::size_t n=2, const float init_val=0.0F):probabilities(std::vector<float>(n, init_val)){}
  ~wordProbability()
  { probabilities.clear(); probabilities.shrink_to_fit();}
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

typedef std::unordered_map<int, std::vector<std::string>> SMS_DATASET_TYPE;
typedef std::unordered_map<std::string, wordProbability>  FEATURE_PROBABILITY_TYPE;
typedef std::vector<std::vector<int>>           MAT_INT_2D;
typedef uint8_t                                 BYTE;

#define PREDICTION_TYPE_MULTINOMIAL     (BYTE)(0)
#define PREDICTION_TYPE_COMPLEMENT      (BYTE)(1)

static std::unordered_map<std::string, int> identifiers_to_int{{"ham", 0}, {"spam", 1}};


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
  auto punct_digit = [](char c){return (std::ispunct(static_cast<unsigned char>(c)) || std::isdigit(c));};

  auto invalid_char  = [](char c){return    !(std::isalpha(static_cast<unsigned char>(c))) 
                                         && !(std::isspace(static_cast<unsigned char>(c))); };

  // remove punctuation marks, full-stops, commas, ...
  for (std::pair<const int, std::vector<std::string>>& row: dataset)
  {
    for (std::size_t iter = 0u; iter < row.second.size(); iter++)
    {
      std::string& this_string = row.second[iter];
      this_string.erase(std::remove_if(this_string.begin(), this_string.end(), invalid_char), this_string.end());
      this_string.erase(std::remove_if(this_string.begin(), this_string.end(), punct_digit), this_string.end());
      std::transform(this_string.begin(), this_string.end(), this_string.begin(), [](char c) {return (char)std::tolower(c);}); 
    }
  }
}


void generate_permutations(std::vector<int>& permutations_out, const std::size_t len)
{
  // first fill permutations_out 
  permutations_out = std::vector<int> (len, 0);
  int n = -1;
  unsigned int seed  = static_cast<unsigned int>(std::chrono::system_clock::now().time_since_epoch().count());
  std::mt19937 random_generator(seed);
  std::generate(permutations_out.begin(), permutations_out.end(), [&n]() mutable {return ++n;});
  std::shuffle(permutations_out.begin(), permutations_out.end(), random_generator);
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
    std::size_t train_len = (std::size_t)((float)class_type.second.size()*train_frac);
    std::size_t test_len  = class_type.second.size() - train_len;
    std::vector<int> permutations;

    generate_permutations(permutations, class_type.second.size());

    train_data_indices[class_type.first] = std::vector<int>(train_len, 0);
    test_data_indices[class_type.first] = std::vector<int>(test_len, 0);

    std::copy_n(permutations.begin(), train_len, train_data_indices[class_type.first].begin());
    std::copy(permutations.begin()+train_len, permutations.end(), test_data_indices[class_type.first].begin());
  }
}


void calc_TF(const SMS_DATASET_TYPE&      input_dataset,
             const MAT_INT_2D&            data_indices_to_use,
                   FEATURE_PROBABILITY_TYPE& feature_probabilities)
{
  auto is_space = [](char c){ return (c == ' ');};

  feature_probabilities.clear();
  FEATURE_PROBABILITY_TYPE().swap(feature_probabilities);

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
          // if this word is not in the Bag-of-words container then add and initialize
          if (feature_probabilities.find(word) == feature_probabilities.end())
          {  feature_probabilities[word] = wordProbability(input_dataset.size());  }

          feature_probabilities[word].probabilities[row.first] += 1.0F;
        }
      }
    }
  }
}


void calc_TF_IDF(const SMS_DATASET_TYPE&  input_dataset,
                 const MAT_INT_2D&        data_indices_to_use,
                       FEATURE_PROBABILITY_TYPE& feature_probabilities)
{
  std::size_t number_doc = 0u;
  std::unordered_map<std::string, int> word_in_doc_count;
  
  feature_probabilities.clear();
  FEATURE_PROBABILITY_TYPE().swap(feature_probabilities);

  for (const auto& row: input_dataset)
  {
    auto& data_indices_this_class = data_indices_to_use[row.first];
    number_doc += row.second.size();

    for (std::size_t iter = 0u; iter < data_indices_this_class.size(); iter++)
    {
      std::string current_string = row.second[data_indices_this_class[iter]];
      std::istringstream token_stream(current_string);
      std::string word;

      std::unordered_map<std::string, int> local_map;
      while(std::getline(token_stream, word, ' '))
      {
        word.erase(std::remove_if(word.begin(), word.end(), std::isspace), word.end());
        if (word.length() > 1u)
        {
          // if this word is not in the Bag-of-words container then add and initialize
          if (feature_probabilities.find(word) == feature_probabilities.end())
          {  feature_probabilities[word] = wordProbability(input_dataset.size());  }

          feature_probabilities[word].probabilities[row.first] += 1.0F;
          local_map[word] = 0;
        }
      }

      for (const auto& word_occurance : local_map)
      {
        if(word_in_doc_count.find(word_occurance.first) == word_in_doc_count.end())
        {
          word_in_doc_count[word_occurance.first] = 1;
        }
        else
        { word_in_doc_count[word_occurance.first]++; }
      }
    }
  }

  for (std::pair<const std::string, wordProbability>& this_word : feature_probabilities)
  {
    const float IDF_weight = std::logf((float)number_doc/(float)(word_in_doc_count.at(this_word.first)));

    for (std::size_t class_iter = 0u; class_iter < input_dataset.size(); class_iter++)
    {
      this_word.second.probabilities[class_iter] = feature_probabilities[this_word.first].probabilities[class_iter] * IDF_weight;
    }
  }
}


void calc_word_weights_mnb(const SMS_DATASET_TYPE&         input_dataset, 
                           const MAT_INT_2D&               data_indices_to_use,
                                 FEATURE_PROBABILITY_TYPE& feature_probabilities)
{
  std::vector<float> total_freq_count(input_dataset.size(), 0.0F);

  for (const auto& word: feature_probabilities)
  {
    for (std::size_t class_iter = 0u; class_iter < word.second.probabilities.size(); class_iter++)
    {
      total_freq_count[class_iter] += word.second.probabilities[class_iter];
    }
  }

  // now divide each word in each class with the total number of words in it's class
  // there is a chance we might encounter absolutely zero probabilities for a word in any class
  // for this use laplace smoothing
  const float smoothing_constant = 1e-4F;
  for (std::pair<const std::string, wordProbability>& this_word : feature_probabilities)
  {
    for (std::size_t class_iter = 0u; class_iter < input_dataset.size(); class_iter++)
    {
      this_word.second.probabilities[class_iter] += smoothing_constant;
      float denominator = total_freq_count[class_iter] + (smoothing_constant*((float)data_indices_to_use[class_iter].size()));
      this_word.second.probabilities[class_iter] /= denominator;
    }
  }
}


void calc_word_weights_cnb(const SMS_DATASET_TYPE&         input_dataset, 
                           const MAT_INT_2D&               data_indices_to_use,
                                 FEATURE_PROBABILITY_TYPE& feature_probabilities)
{
  std::vector<float> per_class_word_weight_sum(input_dataset.size(), 0.0F);

  for (const auto& word: feature_probabilities)
  {
    for (std::size_t class_iter = 0u; class_iter < word.second.probabilities.size(); class_iter++)
    {
      per_class_word_weight_sum[class_iter] += word.second.probabilities[class_iter];
    }
  }

  // now divide each word in each class with the total number of words in it's class
  // there is a chance we might encounter absolutely zero probabilities for a word in any class
  // for this use laplace smoothing
  const float smoothing_constant = 1e-4F;
  for (std::pair<const std::string, wordProbability>& this_word : feature_probabilities)
  {
    const std::vector<float> initial_word_weight = this_word.second.probabilities;

    for (std::size_t class_iter = 0u; class_iter < input_dataset.size(); class_iter++)
    {
      std::size_t complement_class_idx = 1u-class_iter;

      this_word.second.probabilities[class_iter] = initial_word_weight[complement_class_idx] + smoothing_constant;
      float denominator = per_class_word_weight_sum[complement_class_idx] + (smoothing_constant*((float)data_indices_to_use[complement_class_idx].size()));
      this_word.second.probabilities[class_iter] /= denominator;
    }
  }
}


void predict_class(const SMS_DATASET_TYPE&          input_dataset,
                   const MAT_INT_2D&                data_indices_to_use,
                   const FEATURE_PROBABILITY_TYPE&  feature_probabilities,
                   const BYTE                       prediction_type,     
                         MAT_INT_2D&                predicted_labels)
{
  auto is_space = [](char c){ return (c == ' ');};
  std::vector<float> class_probabilities(input_dataset.size());
  predicted_labels = MAT_INT_2D(input_dataset.size());

  // calculate class_probabilities
  std::size_t total_docs = data_indices_to_use[0].size() + data_indices_to_use[1].size();
  class_probabilities[0] = std::logf((float)data_indices_to_use[0].size() / (float)total_docs);
  class_probabilities[1] = std::logf((float)data_indices_to_use[1].size() / (float)total_docs);

  for (int class_iter = 0u; class_iter < data_indices_to_use.size(); class_iter++)
  {
    auto& data_indices_this_class = data_indices_to_use[class_iter];
    auto& this_class_dataset      = input_dataset.at(class_iter);
    predicted_labels[class_iter]  = std::vector<int>(data_indices_to_use[class_iter].size(), 0);

    for (std::size_t iter = 0u; iter < data_indices_this_class.size(); iter++)
    {
      wordProbability this_document_probs(input_dataset.size(), 0.0F);
      std::string current_string = this_class_dataset[data_indices_this_class[iter]];
      std::istringstream token_stream(current_string);
      std::string word;

      while(std::getline(token_stream, word, ' '))
      {
        word.erase(std::remove_if(word.begin(), word.end(), is_space), word.end());
        if (word.length() > 1u)
        {
          // if this word is not in the Bag-of-words container then add and initialize
          if (feature_probabilities.find(word) != feature_probabilities.end())
          {
            auto& this_word_prob = feature_probabilities.at(word).probabilities;
            for (std::size_t word_prob_iter = 0u; word_prob_iter < this_word_prob.size(); word_prob_iter++)
            {
              this_document_probs.probabilities[word_prob_iter] += std::logf(this_word_prob[word_prob_iter]);
            }
          }
        }
      }

      // now flag class with maximum probability as the predicted class, initialize class with Spam
      float best_prob = -10000.0F;

      // for the complement class we need to select the class that has least probability in the end for a document
      if (prediction_type == PREDICTION_TYPE_COMPLEMENT)
      { best_prob = 1000.0F; }

      predicted_labels[class_iter][iter] = identifiers_to_int["spam"];
      for (int doc_prob_iter = 0u; doc_prob_iter < this_document_probs.probabilities.size(); doc_prob_iter++)
      {
        this_document_probs.probabilities[doc_prob_iter] += class_probabilities[doc_prob_iter];

        bool select_this_class = false;
        switch(prediction_type)
        {
          case (PREDICTION_TYPE_MULTINOMIAL):
          {
            if (this_document_probs.probabilities[doc_prob_iter] > best_prob)
            { select_this_class = true; }
          }
          break;
          case(PREDICTION_TYPE_COMPLEMENT):
          {
            if (this_document_probs.probabilities[doc_prob_iter] < best_prob)
            { select_this_class = true; }
          }
          break;
        }

        if (select_this_class == true)
        {
          predicted_labels[class_iter][iter] = doc_prob_iter;
          best_prob = this_document_probs.probabilities[doc_prob_iter];
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
      true_positive_count  += (class_input.first     & predicted_labels[class_input.first][index_iter])      > 0?1.0F:0.0F;
      false_positive_count += ((1-class_input.first) & (predicted_labels[class_input.first][index_iter]))    > 0?1.0F:0.0F;
      true_negative_count  += ((1-class_input.first) & (1-predicted_labels[class_input.first][index_iter]))  > 0?1.0F:0.0F;
      false_negative_count += ((class_input.first)   & (1-predicted_labels[class_input.first][index_iter]))  > 0?1.0F:0.0F;
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
