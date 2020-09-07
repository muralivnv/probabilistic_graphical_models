#include <cstdio>
#include <iostream>
#include <array>
#include <vector>
#include <string>
#include <chrono>

#include <cppitertools/range.hpp>
#include <cppitertools/zip.hpp>

#include <cpputil/cpputil.hpp>
#include "custom_util.hpp"

using std::vector; 
using std::array;
using std::string;

#define CHAR_IMG_HEIGHT (60u)
#define CHAR_IMG_WIDTH  (30u)
namespace chrono = std::chrono;
namespace it=iter;

int main()
{
  string dataset_dir = "dataset/data/processed/breta/words_gaplines";

  vector<string> dat_files;
  files_with_extension(dataset_dir, ".dat", dat_files);

  vector<vector<std::array<float, CHAR_IMG_HEIGHT*CHAR_IMG_WIDTH>>> images(dat_files.size());
  vector<string> image_labels(dat_files.size());

// Read Dataset 
  auto start_time = chrono::system_clock::now();
  std::cout << "reading dataset ...\n";
  for (auto&& [i, dat_file, image, label]: it::zip(it::range(dat_files.size()), dat_files, images, image_labels))
  {
    read_png_data<float, CHAR_IMG_WIDTH, CHAR_IMG_HEIGHT>(dat_file, label, image);
    printf("\r\tpercent complete: %0.2f", (float)(i)/(float)images.size());
  }
  auto end_time = chrono::system_clock::now();
  std::cout << "\n\tElapsed: " << chrono::duration<double>(end_time - start_time).count() << " sec\n";

// Split Dataset

// Setup weights

// For each dataset call forward and backward algorithm for inference

// Calculate weight gradients

// Update Weights


  return EXIT_SUCCESS;
}