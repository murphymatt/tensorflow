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

// This test illustrates how to make use of the CTCBeamSearchDecoder using a
// custom BeamScorer and BeamState based on a dictionary with a few artificial
// words.
#include "tensorflow/core/util/ctc/ctc_beam_search.h"

#include <cmath>
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/test.h"

namespace {

typedef std::vector<std::vector<std::vector<float>>> TestData;
using tensorflow::ctc::CTCBeamSearchDecoder;
using tensorflow::ctc::CTCDecoder;
using tensorflow::ctc::TrieBeamScorer;
using tensorflow::ctc::TrieBeamState;

const char *dictionary_path = "./tensorflow/core/util/ctc/testdata/vocab";
// "the quick brown fox jumped over the lazy dog"
const int test_labels[] = {19, 7, 4, 16, 20, 8, 2, 10, 1, 17, 14, 22, 13, 5, 14,
  23, 9, 20, 12, 15, 18, 14, 21, 4, 17, 19, 7, 4, 11, 0, 25, 24, 3, 14, 6};
const int test_label_count = 35;

TEST(CtcBeamSearch, ScoreState) {
  const int batch_size = 1;
  const int timesteps = 3;
  const int top_paths = 1;
  const int num_classes = 4;

  std::vector<std::vector<int>> dictionary {{0, 1, 2}};
  TrieBeamScorer scorer(dictionary, num_classes, false);
  CTCBeamSearchDecoder<TrieBeamState> decoder(
      num_classes, 10 * top_paths, &scorer);

  int sequence_lengths[batch_size] = {timesteps};
  float input_data_mat[timesteps][batch_size][num_classes] = {
    {{1, 0, 0, 0}},
    {{0, 1, 0, 0}},
    {{0, 0, 1, 0}}};

    for (int t = 0; t < timesteps; ++t) {
      for (int b = 0; b < batch_size; ++b) {
        for (int c = 0; c < num_classes; ++c) {
          input_data_mat[t][b][c] = std::log(input_data_mat[t][b][c]);
        }
      }
    }

    std::vector<CTCDecoder::Output> expected_output = {
      {{0, 1, 2}},
    };

    Eigen::Map<const Eigen::ArrayXi> seq_len(&sequence_lengths[0], batch_size);
    std::vector<Eigen::Map<const Eigen::MatrixXf>> inputs;
    inputs.reserve(timesteps);
    for (int t = 0; t < timesteps; ++t) {
      inputs.emplace_back(&input_data_mat[t][0][0], batch_size, num_classes);
    }

    // Prepare containers for output and scores.
    std::vector<CTCDecoder::Output> outputs(top_paths);
    for (CTCDecoder::Output& output : outputs) {
      output.resize(batch_size);
    }
    float score[batch_size][top_paths] = {{0.0}};
    Eigen::Map<Eigen::MatrixXf> scores(&score[0][0], batch_size, top_paths);

    EXPECT_TRUE(decoder.Decode(seq_len, inputs, &outputs, &scores).ok());
    for (int path = 0; path < top_paths; ++path) {
      EXPECT_EQ(outputs[path][0], expected_output[0][path]);
    }
}

TEST(CtcBeamSearch, ScoreStateRepeatLabels) {
  const int batch_size = 1;
  const int timesteps = 5;
  const int top_paths = 1;
  const int num_classes = 4;

  std::vector<std::vector<int>> dictionary {{0, 1, 2}};
  TrieBeamScorer scorer(dictionary, num_classes, false);
  CTCBeamSearchDecoder<TrieBeamState> decoder(
      num_classes, 10 * top_paths, &scorer);

  int sequence_lengths[batch_size] = {timesteps};
  float input_data_mat[timesteps][batch_size][num_classes] = {
    {{1, 0, 0, 0}},
    {{1, 0, 0, 0}},
    {{0, 1, 0, 0}},
    {{0, 1, 0, 0}},
    {{0, 0, 1, 0}}};

    for (int t = 0; t < timesteps; ++t) {
      for (int b = 0; b < batch_size; ++b) {
        for (int c = 0; c < num_classes; ++c) {
          input_data_mat[t][b][c] = std::log(input_data_mat[t][b][c]);
        }
      }
    }

    std::vector<CTCDecoder::Output> expected_output = {
      {{0, 1, 2}},
    };

    Eigen::Map<const Eigen::ArrayXi> seq_len(&sequence_lengths[0], batch_size);
    std::vector<Eigen::Map<const Eigen::MatrixXf>> inputs;
    inputs.reserve(timesteps);
    for (int t = 0; t < timesteps; ++t) {
      inputs.emplace_back(&input_data_mat[t][0][0], batch_size, num_classes);
    }

    // Prepare containers for output and scores.
    std::vector<CTCDecoder::Output> outputs(top_paths);
    for (CTCDecoder::Output& output : outputs) {
      output.resize(batch_size);
    }
    float score[batch_size][top_paths] = {{0.0}};
    Eigen::Map<Eigen::MatrixXf> scores(&score[0][0], batch_size, top_paths);

    EXPECT_TRUE(decoder.Decode(seq_len, inputs, &outputs, &scores).ok());
    for (int path = 0; path < top_paths; ++path) {
      EXPECT_EQ(outputs[path][0], expected_output[0][path]);
    }
}


TEST(CtcBeamSearch, DecodingWithAndWithoutDictionary) {
  const int batch_size = 1;
  const int timesteps = 5;
  const int top_paths = 3;
  const int num_classes = 6;

  // Plain decoder using hibernating beam search algorithm.
  CTCBeamSearchDecoder<>::DefaultBeamScorer default_scorer;
  CTCBeamSearchDecoder<> decoder(num_classes, 10 * top_paths, &default_scorer);

  std::vector<std::vector<int>> dictionary {
      {1}, {1, 3}, {1, 3, 1}, {1, 3, 1, 3}, {1, 3, 1, 3, 1},
      {3}, {3, 1}, {3, 1, 3}, {3, 1, 3, 1}, {3, 1, 3, 1, 3}};


  TrieBeamScorer dictionary_scorer(dictionary, num_classes, false);
  CTCBeamSearchDecoder<TrieBeamState> dictionary_decoder(
      num_classes, 10 * top_paths, &dictionary_scorer);

  // Raw data containers (arrays of floats, ints, etc.).
  int sequence_lengths[batch_size] = {timesteps};
  float input_data_mat[timesteps][batch_size][num_classes] = {
      {{0, 0.6, 0, 0.4, 0, 0}},
      {{0, 0.5, 0, 0.5, 0, 0}},
      {{0, 0.4, 0, 0.6, 0, 0}},
      {{0, 0.4, 0, 0.6, 0, 0}},
      {{0, 0.4, 0, 0.6, 0, 0}}};

  // The CTCDecoder works with log-probs.
  for (int t = 0; t < timesteps; ++t) {
    for (int b = 0; b < batch_size; ++b) {
      for (int c = 0; c < num_classes; ++c) {
        input_data_mat[t][b][c] = std::log(input_data_mat[t][b][c]);
      }
    }
  }

  // Plain output, without any additional scoring.
  std::vector<CTCDecoder::Output> expected_output = {
      {{1, 3}, {1, 3, 1}, {3, 1, 3}},
  };

  // Dictionary outputs: preference for dictionary candidates. The
  // second-candidate is there, despite it not being a dictionary word, due to
  // stronger probability in the input to the decoder.
  std::vector<CTCDecoder::Output> expected_dict_output = {
      {{1, 3}, {1, 3, 1}, {3, 1, 3}},
  };

  // Convert data containers to the format accepted by the decoder, simply
  // mapping the memory from the container to an Eigen::ArrayXi,::MatrixXf,
  // using Eigen::Map.
  Eigen::Map<const Eigen::ArrayXi> seq_len(&sequence_lengths[0], batch_size);
  std::vector<Eigen::Map<const Eigen::MatrixXf>> inputs;
  inputs.reserve(timesteps);
  for (int t = 0; t < timesteps; ++t) {
    inputs.emplace_back(&input_data_mat[t][0][0], batch_size, num_classes);
  }

  // Prepare containers for output and scores.
  std::vector<CTCDecoder::Output> outputs(top_paths);
  for (CTCDecoder::Output& output : outputs) {
    output.resize(batch_size);
  }
  float score[batch_size][top_paths] = {{0.0}};
  Eigen::Map<Eigen::MatrixXf> scores(&score[0][0], batch_size, top_paths);

  EXPECT_TRUE(decoder.Decode(seq_len, inputs, &outputs, &scores).ok());
  for (int path = 0; path < top_paths; ++path) {
    EXPECT_EQ(outputs[path][0], expected_output[0][path]);
  }
  
  // Prepare dictionary outputs.
  std::vector<CTCDecoder::Output> dict_outputs(top_paths);
  for (CTCDecoder::Output& output : dict_outputs) {
    output.resize(batch_size);
  }
  float dict_score[batch_size][top_paths] = {{0.0}};
  Eigen::Map<Eigen::MatrixXf> dict_scores(&dict_score[0][0], batch_size, top_paths);
  
  EXPECT_TRUE(
      dictionary_decoder.Decode(seq_len, inputs, &dict_outputs, &dict_scores).ok());
  for (int path = 0; path < top_paths; ++path) {
    EXPECT_EQ(dict_outputs[path][0], expected_dict_output[0][path]);
  }

  // Ensure that dictionary scores are same as standard scores
  for (int path = 0; path < top_paths; ++path) {
    EXPECT_EQ(scores(path), dict_scores(path));
  }
}

TEST(CtcBeamSearch, DecodingWithRestrictDict) {
  const int batch_size = 1;
  const int timesteps = 5;
  const int top_paths = 4;
  const int num_classes = 6;

  // Dictionary decoder, allowing only three dictionary words : {1}, {3}, {3, 1}.
  std::vector<std::vector<int>> dictionary {{3}, {3, 1}, {1}};

  TrieBeamScorer dictionary_scorer(dictionary, num_classes, false);
  CTCBeamSearchDecoder<TrieBeamState> dictionary_decoder(
      num_classes, 10 * top_paths, &dictionary_scorer);

  // Raw data containers (arrays of floats, ints, etc.).
  int sequence_lengths[batch_size] = {timesteps};
  float input_data_mat[timesteps][batch_size][num_classes] = {
      {{0, 0.6, 0, 0.4, 0, 0}},
      {{0, 0.5, 0, 0.5, 0, 0}},
      {{0, 0.4, 0, 0.6, 0, 0}},
      {{0, 0.4, 0, 0.6, 0, 0}},
      {{0, 0.4, 0, 0.6, 0, 0}}};

  // The CTCDecoder works with log-probs.
  for (int t = 0; t < timesteps; ++t) {
    for (int b = 0; b < batch_size; ++b) {
      for (int c = 0; c < num_classes; ++c) {
        input_data_mat[t][b][c] = std::log(input_data_mat[t][b][c]);
      }
    }
  }

  // Dictionary outputs: preference for dictionary candidates. The
  // second-candidate is there, despite it not being a dictionary word, due to
  // stronger probability in the input to the decoder.
  std::vector<CTCDecoder::Output> expected_dict_output = {
    {{3, 1}, {3}, {1}, {}},
  };

  // Convert data containers to the format accepted by the decoder, simply
  // mapping the memory from the container to an Eigen::ArrayXi,::MatrixXf,
  // using Eigen::Map.
  Eigen::Map<const Eigen::ArrayXi> seq_len(&sequence_lengths[0], batch_size);
  std::vector<Eigen::Map<const Eigen::MatrixXf>> inputs;
  inputs.reserve(timesteps);
  for (int t = 0; t < timesteps; ++t) {
    inputs.emplace_back(&input_data_mat[t][0][0], batch_size, num_classes);
  }

  // Prepare containers scores.
  float score[batch_size][top_paths] = {{0.0}};
  Eigen::Map<Eigen::MatrixXf> scores(&score[0][0], batch_size, top_paths);

  // Prepare dictionary outputs.
  std::vector<CTCDecoder::Output> dict_outputs(top_paths);
  for (CTCDecoder::Output& output : dict_outputs) {
    output.resize(batch_size);
  }
  EXPECT_TRUE(
      dictionary_decoder.Decode(seq_len, inputs, &dict_outputs, &scores).ok());
  for (int path = 0; path < top_paths; ++path) {
    EXPECT_EQ(dict_outputs[path][0], expected_dict_output[0][path]);
  }
}

TEST(CtcBeamSearch, DecodingWithEmptyDict) {
  const int batch_size = 1;
  const int timesteps = 5;
  const int top_paths = 1;
  const int num_classes = 6;

  // Dictionary decoder, allowing only the empty word {}
  std::vector<std::vector<int>> dictionary {{}};

  TrieBeamScorer dictionary_scorer(dictionary, num_classes, false);
  CTCBeamSearchDecoder<TrieBeamState> dictionary_decoder(
      num_classes, 10 * top_paths, &dictionary_scorer);

  // Raw data containers (arrays of floats, ints, etc.).
  int sequence_lengths[batch_size] = {timesteps};
  float input_data_mat[timesteps][batch_size][num_classes] = {
      {{0, 0.6, 0, 0.4, 0, 0}},
      {{0, 0.5, 0, 0.5, 0, 0}},
      {{0, 0.4, 0, 0.6, 0, 0}},
      {{0, 0.4, 0, 0.6, 0, 0}},
      {{0, 0.4, 0, 0.6, 0, 0}}};

  // The CTCDecoder works with log-probs.
  for (int t = 0; t < timesteps; ++t) {
    for (int b = 0; b < batch_size; ++b) {
      for (int c = 0; c < num_classes; ++c) {
        input_data_mat[t][b][c] = std::log(input_data_mat[t][b][c]);
      }
    }
  }

  // Dictionary outputs: preference for dictionary candidates. The
  // second-candidate is there, despite it not being a dictionary word, due to
  // stronger probability in the input to the decoder.
  std::vector<CTCDecoder::Output> expected_dict_output = {
      {{}},
  };

  // Convert data containers to the format accepted by the decoder, simply
  // mapping the memory from the container to an Eigen::ArrayXi,::MatrixXf,
  // using Eigen::Map.
  Eigen::Map<const Eigen::ArrayXi> seq_len(&sequence_lengths[0], batch_size);
  std::vector<Eigen::Map<const Eigen::MatrixXf>> inputs;
  inputs.reserve(timesteps);
  for (int t = 0; t < timesteps; ++t) {
    inputs.emplace_back(&input_data_mat[t][0][0], batch_size, num_classes);
  }

  // Prepare containers scores.
  float score[batch_size][top_paths] = {{0.0}};
  Eigen::Map<Eigen::MatrixXf> scores(&score[0][0], batch_size, top_paths);

  // Prepare dictionary outputs.
  std::vector<CTCDecoder::Output> dict_outputs(top_paths);
  for (CTCDecoder::Output& output : dict_outputs) {
    output.resize(batch_size);
  }
  EXPECT_TRUE(
      dictionary_decoder.Decode(seq_len, inputs, &dict_outputs, &scores).ok());
  for (int path = 0; path < top_paths; ++path) {
    EXPECT_EQ(dict_outputs[path][0], expected_dict_output[0][path]);
  }
}

TEST(CtcBeamSearch, DecodingWithDisjointDict) {
  const int batch_size = 1;
  const int timesteps = 5;
  const int top_paths = 8;
  const int num_classes = 6;

  // Dictionary decoder, allowing only empty word
  std::vector<std::vector<int>> dictionary {{2, 2}, {2, 4}, {4, 2}, {4, 4},
                                            {3, 2}, {1, 3, 3, 2}, {2, 3, 1}};

  TrieBeamScorer dictionary_scorer(dictionary, num_classes, false);
  CTCBeamSearchDecoder<TrieBeamState> dictionary_decoder(
      num_classes, 10 * top_paths, &dictionary_scorer);

  // Raw data containers (arrays of floats, ints, etc.).
  int sequence_lengths[batch_size] = {timesteps};
  float input_data_mat[timesteps][batch_size][num_classes] = {
      {{0, 0.6, 0, 0.4, 0, 0}},
      {{0, 0.5, 0, 0.5, 0, 0}},
      {{0, 0.4, 0, 0.6, 0, 0}},
      {{0, 0.4, 0, 0.6, 0, 0}},
      {{0, 0.4, 0, 0.6, 0, 0}}};

  // The CTCDecoder works with log-probs.
  for (int t = 0; t < timesteps; ++t) {
    for (int b = 0; b < batch_size; ++b) {
      for (int c = 0; c < num_classes; ++c) {
        input_data_mat[t][b][c] = std::log(input_data_mat[t][b][c]);
      }
    }
  }
  
  // Convert data containers to the format accepted by the decoder, simply
  // mapping the memory from the container to an Eigen::ArrayXi,::MatrixXf,
  // using Eigen::Map.
  Eigen::Map<const Eigen::ArrayXi> seq_len(&sequence_lengths[0], batch_size);
  std::vector<Eigen::Map<const Eigen::MatrixXf>> inputs;
  inputs.reserve(timesteps);
  for (int t = 0; t < timesteps; ++t) {
    inputs.emplace_back(&input_data_mat[t][0][0], batch_size, num_classes);
  }

  // Prepare containers scores.
  float score[batch_size][top_paths] = {{0.0}};
  Eigen::Map<Eigen::MatrixXf> scores(&score[0][0], batch_size, top_paths);

  // Prepare dictionary outputs.
  std::vector<CTCDecoder::Output> dict_outputs(top_paths);
  for (CTCDecoder::Output& output : dict_outputs) {
    output.resize(batch_size);
  }

  EXPECT_TRUE(
      dictionary_decoder.Decode(seq_len, inputs, &dict_outputs, &scores).ok());
  for (int path = 0; path < top_paths; ++path) {
    EXPECT_EQ((dict_outputs[path][0]).size(), 0);
  }
}
}  // namespace
