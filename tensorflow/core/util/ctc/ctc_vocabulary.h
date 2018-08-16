/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

/* Code adapted from https://github.com/timediv/tensorflow-with-kenlm */

#ifndef TENSORFLOW_CORE_UTIL_CTC_CTC_VOCABULARY_H_
#define TENSORFLOW_CORE_UTIL_CTC_CTC_VOCABULARY_H_

#include <fstream>
#include <istream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace tensorflow {
namespace ctc {

class Vocabulary {
  public:
    Vocabulary(std::vector<std::vector<int>> vocab_list, int alpha_size)
    : alphabet_size(alpha_size),
      vocab_size(vocab_list.size()),
      vocabulary(vocab_list) {}

    Vocabulary(const char *vocab_path, int alpha_size)
    : alphabet_size(alpha_size) {
      std::ifstream in(vocab_path);
      ReadFromFile(in);
      in.close();
    }

    ~Vocabulary() {
      vocabulary.clear();
    }

    bool inline IsBlankLabel(int label) const {
      return label == alphabet_size;
    }

    int GetVocabSize() {
      return vocab_size;
    }

    std::vector<std::vector<int>> GetVocabList() {
      return vocabulary;
    }

    void PrintVocab() {
      for (std::vector<int> word : vocabulary) {
        for (int wChar : word) {
          if (wChar >= 0 && wChar < alphabet_size)
            std::cout << wChar;
	  std::cout << std::endl;
        }
      }
    }

  private:
    int alphabet_size;
    int vocab_size;
    std::vector<std::vector<int>> vocabulary;

    void ReadFromFile(std::ifstream& in) {
      vocab_size = 0;
      std::string str;
      while (in >> str) {
        std::vector<int> ret;
        for (int i=0; i<str.length(); ++i) {
          ret.push_back(str.at(i) - 'a');
        }
        vocabulary.push_back(ret);
        vocab_size++;
      }
    }
};

} // namespace ctc
} // namespace tensorflow

#endif // TENSORFLOW_CORE_UTIL_CTC_CTC_VOCABULARY_H
