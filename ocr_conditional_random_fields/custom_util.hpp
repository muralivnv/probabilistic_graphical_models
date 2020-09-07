#ifndef _OCR_CRF_UTIL_HPP_
#define _OCR_CRF_UTIL_HPP_

#include <sstream>
#include <vector>
#include <array>
#include <fstream>
#include <filesystem>

using std::vector, std::array, std::size_t, std::string;

void files_with_extension(const std::string&        directory, 
                          const std::string&        extension, 
                                std::vector<std::string>& results)
{
  namespace fs = std::filesystem;

  results.reserve(100);
  size_t capacity = 100u;
  size_t size     = 0u;
  auto extension_hash = fs::hash_value(extension);
  for (auto& path: fs::directory_iterator(directory))
  {
    if (fs::hash_value(path.path().extension()) == extension_hash)
    {
      results.push_back(path.path().string());
      size++;
      if ((capacity - size) <= 10u)
      {
        capacity+= 100u;
        results.reserve(capacity);
      }
    }
  }
  results.shrink_to_fit();
}


template<typename T, size_t ImageWidth, size_t ImageHeight>
void read_png_data(const string& filename, string& label, vector<array<T, ImageWidth*ImageHeight>>& word)
{
  std::ifstream in_file(filename);
  word.reserve(10u);
  if (in_file.is_open())
  {
    string this_char;
    while(std::getline(in_file, this_char))
    {
      if (this_char == "")
      { continue; }
      this_char = this_char.substr(0, this_char.length()-1);
      label.append(this_char);
      word.push_back(std::array<T, ImageWidth*ImageHeight>{});
      auto& character = word.back();
      for (size_t i = 0u; i < ImageHeight; i++)
      {
        string line;
        std::istringstream stream;
        std::getline(in_file, line);
        stream.str(line);
        for (size_t j = (i*ImageWidth); j < (i+1u)*ImageWidth; j++)
        {
          std::getline(stream, line, ' ');
          character[j] = std::stof(line);
        }
      }
    }
    in_file.close();
  }
  word.shrink_to_fit();
}

#endif
