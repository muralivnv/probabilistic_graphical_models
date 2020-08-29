#ifndef _CUSTOM_UTILS_H_
#define _CUSTOM_UTILS_H_

#include <string>

#include <vector>
#include <array>
#include <unordered_map>

#include <sstream>
#include <fstream>

#include <random>
#include <chrono>

#include <algorithm>

#include "custom_typedefs.h"

using std::vector;
using std::string;

void strip_str(std::string& string_to_str)
{
  static const char* whitespace = " \t\n\f\v";

  string_to_str.erase(0, string_to_str.find_first_not_of(whitespace));
  string_to_str.erase(string_to_str.find_last_not_of(whitespace) + 1);
}


template <std::size_t TagLen>
void read_tags(const std::string&         tag_filename,
                     VEC<string, TagLen>& tags)
{
  std::cout << "\nreading tags from file --- " << tag_filename << '\n';

  std::ifstream tag_file_obj(tag_filename);
  std::size_t tag_iter = 0u;
  if (tag_file_obj.is_open())
  {
    std::string current_tag;
    while(std::getline(tag_file_obj, current_tag) && tag_iter < TagLen)
    {
      strip_str(current_tag);
      if (current_tag.length() > 0u)
      { tags[tag_iter++] = current_tag; }
    }
  }
  tag_file_obj.close();

  std::cout << "finished reading tags\n\n";
}


void read_text_corpus(const std::string&  filename, 
                            DMAT_STR&      dataset,
                            DMAT_STR&      labels)
{
  std::cout << "\nreading corpus data from file --- " << filename << '\n';
  std::ifstream corpus_file_obj(filename);
  dataset.reserve(500);
  labels.reserve(500);

  if (corpus_file_obj.is_open())
  {
    std::string current_line;
    std::size_t sentence_idx = 0u;
    dataset.push_back(DVEC_STR{});
    labels.push_back(DVEC_STR{});

    bool skip_next_line = true;

    while(std::getline(corpus_file_obj, current_line))
    {
      if (skip_next_line == false)
      {
        if (current_line.length() == 0u)
        { 
          skip_next_line = true; 
          sentence_idx++; 
          continue;
          }

        std::istringstream tokenizer(current_line);
        
        std::getline(tokenizer, current_line, '\t');
        dataset[sentence_idx].insert(dataset[sentence_idx].end(), current_line);
        
        std::getline(tokenizer, current_line, '\n');
        labels[sentence_idx].insert(labels[sentence_idx].end(), current_line);
      }
      else
      { 
        skip_next_line = false;           
        dataset.push_back(DVEC_STR{});
        labels.push_back(DVEC_STR{});
      }
      
      if ((dataset.capacity() - dataset.size()) < 10u)
      {  
        dataset.reserve(dataset.size() + 500); 
        labels.reserve(labels.size() + 500);
      }
    }
    
  }
  corpus_file_obj.close();
  dataset.shrink_to_fit();
  labels.shrink_to_fit();

  std::cout << "finished reading corpus data\n"; 
  std::cout << "total dataset size: " << dataset.size() << "\n\n";
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


template<typename ContainerType>
void train_test_split(const ContainerType& dataset_container, 
                      const float          train_frac, 
                            DVEC_UINT&      train_data_indices, 
                            DVEC_UINT&      test_data_indices)
{
  std::size_t train_len = (std::size_t)((float)dataset_container.size()*train_frac);
  std::size_t test_len  = dataset_container.size() - train_len;
  DVEC_INT permutations;

  generate_permutations(permutations, dataset_container.size());
  train_data_indices = DVEC_UINT(train_len, 0u);
  test_data_indices  = DVEC_UINT(test_len, 0u);

  std::copy_n(permutations.begin(), train_len, train_data_indices.begin());
  std::copy(permutations.begin()+train_len, permutations.end(), test_data_indices.begin());
}


template<typename KeyType, typename ValueType>
void insert_if_not_exist(HASH_MAP<KeyType, ValueType>& container, 
                         const KeyType&                key, 
                               ValueType&&             value_to_insert)
{
  if (container.find(key) == container.end())
  { container.insert(std::make_pair(key, std::forward<ValueType>(value_to_insert))); }
}


inline void to_lower_str(std::string& str)
{
  std::transform(str.begin(), str.end(), str.begin(), [](char c){ return (char)std::tolower(c);});
}


// implementation taken from
// https://stackoverflow.com/questions/9411823/fast-log2float-x-implementation-c/29045715
inline float fast_log2f(float val)
{
  union { float val; int32_t x; } u = { val };
  float log_2 = (float)(((u.x >> 23) & 255) - 128);              
  u.x   &= ~(255 << 23);
  u.x   += 127 << 23;
  log_2 += ((-0.3358287811f) * u.val + 2.0f) * u.val  -0.65871759316667f; 
  return log_2;
} 


#endif
