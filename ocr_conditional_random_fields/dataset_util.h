#ifndef _DATASET_UTIL_H_
#define _DATASET_UTIL_H_

#include <vector>
#include <string>
#include <unordered_map>
#include <sstream>
#include <fstream>
#include <charconv>
#include <filesystem>

#include "crf_typedef.h"

vector<string> files_with_extension(const std::string_view  directory, 
                                    const std::string_view  extension)
{
  namespace fs = std::filesystem;

  vector<string> result;

  result.reserve(100);
  size_t capacity = 100u;
  size_t size     = 0u;
  auto extension_hash = fs::hash_value(extension);
  for (auto& path: fs::directory_iterator(directory))
  {
    if (fs::hash_value(path.path().extension()) == extension_hash)
    {
      result.push_back(path.path().string());
      size++;
      if ((capacity - size) <= 10u)
      {
        capacity+= 100u;
        result.reserve(capacity);
      }
    }
  }
  result.shrink_to_fit();

  return result;
}


template<size_t ImageWidth, size_t ImageHeight>
tuple<string, crf::Word_t> read_png_data(const string& filename)
{
  tuple retval(""s, crf::Word_t{});
  auto& [label, word] = retval;

  std::ifstream in_file(filename);
  word.reserve(10u);
  if (in_file.is_open())
  {
    std::stringstream file_buf;
    file_buf << in_file.rdbuf();
    string this_char;
    while(std::getline(file_buf, this_char))
    {
      if (this_char.empty())
      { continue; }
      size_t data_counter = 0u;
      label.append(this_char.substr(0, 1));
      word.emplace_back();
      auto& character = word.back();
      character.resize(ImageHeight*ImageWidth, 1u);

      for (size_t i = 0u; i < ImageHeight; i++)
      {
        string line;
        for (size_t j = 0u; j < ImageWidth - 1u; j++)
        {
          std::getline(file_buf, line, ' ');
          str2float(line, character(data_counter, 0u));
          data_counter++;
        }
        std::getline(file_buf, line, '\n');
        str2float(line, character(data_counter, 0u));
        data_counter++;
      }
    }
    in_file.close();
  }
  word.shrink_to_fit();

  return retval;
}


std::unordered_map<char, size_t> letters2int()
{
  int counter = 0u;
  std::unordered_map<char, size_t> letter_map;
  for (char i = 'A'; i <= 'Z'; i++)
  {  letter_map[i] = counter++;  }

  counter = 0u;
  for (char i = 'a'; i <= 'z'; i++)
  {  letter_map[i] = counter++;  }

  return letter_map;
}


void read_data(const vector<std::string>& files, 
                  crf::Words_t&           images,
                  vector<vector<size_t>>& labels)
{
  const auto letter_map = letters2int();

  #pragma omp parallel for num_threads(2)
  for (int i = 0u; i < files.size(); i++)
  {
    std::string label_str;
    tie (label_str, images[i]) = read_png_data<15, 25>(files[i]);
    
    labels[i] = vector<size_t>(label_str.size());
    for (int j = 0u; j < label_str.size(); j++)
    { labels[i][j] = letter_map.at(label_str[j]); }
  }
}


template<typename Container_t>
void write_to_file(std::string_view           filename, 
                   const Container_t&         container,
                   const std::vector<size_t>& shape)
{
  std::ofstream out_file(filename);
  if (out_file.is_open())
  {
    for (size_t i = 0u; i < shape[0]; i++)
    {
      for(size_t j = 0u; j < shape[1]; j++)
      {
        out_file << *(container.data() + (i*shape[1] + j)) << ' ';
      }
      out_file << '\n';
    }
  }
  out_file.close();

}

#endif
