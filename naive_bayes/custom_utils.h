#ifndef _CUSTOM_UTILS_H_
#define _CUSTOM_UTILS_H_

// eigen library specific
#include <Eigen/Eigen/Dense>

using namespace Eigen;
typedef std::map<int, std::vector<std::string>> SMS_DATASET_TYPE;
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


void tokenize_strings(const SMS_DATASET_TYPE& dataset, 
                      MatrixXi& frequency_matrix_class1, 
                      MatrixXi& frequency_matrix_class2)
{
  auto is_space = [](char c){ return (c == ' ');};

  std::vector<std::vector<std::map<std::string, int>>> word_count_num(dataset.size());

  // now combine every key in each row and create frequency matrix as a whole
  std::map<std::string, int> combined_unique_words;

  for (const auto& row: dataset)
  {
    word_count_num[row.first].reserve(row.second.size());

    for (std::size_t iter = 0u; iter < row.second.size(); iter++)
    {
      std::string current_string = row.second[iter];
      std::istringstream token_stream(current_string);
      std::string word;
      word_count_num[row.first].push_back(std::map<std::string, int>{});

      while(std::getline(token_stream, word, ' '))
      {
        word.erase(std::remove_if(word.begin(), word.end(), is_space), word.end());
        if (word.length() > 1u)
        {
          // change this word to lower case
          std::transform(word.begin(), word.end(), word.begin(), std::tolower);
          if (word_count_num[row.first][iter].find(word) == word_count_num[row.first][iter].end())
          { word_count_num[row.first][iter][word] = 0; }

          word_count_num[row.first][iter][word]++;
          combined_unique_words[word] = 0;
        }
      }
    }
  }

  // frequency_matrix_class1.resize(class1_rows, combined_unique_words.size());
  frequency_matrix_class1.resize(word_count_num[0].size(), combined_unique_words.size());
  frequency_matrix_class2.resize(word_count_num[1].size(), combined_unique_words.size());
  // frequency_matrix_class2.resize(class2_rows combined_unique_words.size());

  std::size_t col_iter = 0u;
  // first do for class 1
  for (std::pair<const std::string, int>& elem : combined_unique_words)
  {
    for (std::size_t row_iter = 0u; row_iter < word_count_num[0].size(); row_iter++)
    {
      std::map<std::string, int> & row = word_count_num[0][row_iter];

      if (row.find(elem.first) != row.end())
      {
        frequency_matrix_class1(row_iter, col_iter) = row[elem.first];
      }
      else
      {
        frequency_matrix_class1(row_iter, col_iter) = 0;
      }
    }
    col_iter++;
  }

  // now for class 2
  col_iter = 0u;

  for (std::pair<const std::string, int>& elem : combined_unique_words)
  {
    for (std::size_t row_iter = 0u; row_iter < word_count_num[1].size(); row_iter++)
    {
      std::map<std::string, int> & row = word_count_num[1][row_iter];

      if (row.find(elem.first) != row.end())
      {
        frequency_matrix_class2(row_iter, col_iter) = row[elem.first];
      }
      else
      {
        frequency_matrix_class2(row_iter, col_iter) = 0;
      }
    }
    col_iter++;
  }
}

#endif
