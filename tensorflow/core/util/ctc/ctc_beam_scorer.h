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

// Collection of scoring classes that can be extended and provided to the
// CTCBeamSearchDecoder to incorporate additional scoring logic (such as a
// language model).
//
// To build a custom scorer extend and implement the pure virtual methods from
// BeamScorerInterface. The default CTC decoding behavior is implemented
// through BaseBeamScorer.

#ifndef TENSORFLOW_CORE_UTIL_CTC_CTC_BEAM_SCORER_H_
#define TENSORFLOW_CORE_UTIL_CTC_CTC_BEAM_SCORER_H_

#include "tensorflow/core/util/ctc/ctc_beam_entry.h"
#include "tensorflow/core/util/ctc/ctc_vocabulary.h"

#include <limits>

namespace tensorflow {
namespace ctc {

using namespace ctc_beam_search;

// Base implementation of a beam scorer used by default by the decoder that can
// be subclassed and provided as an argument to CTCBeamSearchDecoder, if complex
// scoring is required. Its main purpose is to provide a thin layer for
// integrating language model scoring easily.
template <typename CTCBeamState>
class BaseBeamScorer {
 public:
  virtual ~BaseBeamScorer() {}
  // State initialization.
  virtual void InitializeState(CTCBeamState* root) const {}
  // ExpandState is called when expanding a beam to one of its children.
  // Called at most once per child beam. In the simplest case, no state
  // expansion is done.
  virtual void ExpandState(const CTCBeamState& from_state, int from_label,
                           CTCBeamState* to_state, int to_label) const {}
  // ExpandStateEnd is called after decoding has finished. Its purpose is to
  // allow a final scoring of the beam in its current state, before resorting
  // and retrieving the TopN requested candidates. Called at most once per beam.
  virtual void ExpandStateEnd(CTCBeamState* state) const {}
  // GetStateExpansionScore should be an inexpensive method to retrieve the
  // (cached) expansion score computed within ExpandState. The score is
  // multiplied (log-addition) with the input score at the current step from
  // the network.
  //
  // The score returned should be a log-probability. In the simplest case, as
  // there's no state expansion logic, the expansion score is zero.
  virtual float GetStateExpansionScore(const CTCBeamState& state,
                                       float previous_score) const {
    return previous_score;
  }
  // GetStateEndExpansionScore should be an inexpensive method to retrieve the
  // (cached) expansion score computed within ExpandStateEnd. The score is
  // multiplied (log-addition) with the final probability of the beam.
  //
  // The score returned should be a log-probability.
  virtual float GetStateEndExpansionScore(const CTCBeamState& state) const {
    return 0;
  }
}; // BaseBeamScorer

class TrieBeamScorer : public BaseBeamScorer<TrieBeamState> {
 public:
  virtual ~TrieBeamScorer() {
    delete vocabulary;
    delete trieRoot;
  }

  TrieBeamScorer(std::vector<std::vector<int32>> vocab_list,
    int alpha_size, bool multiWord) :
    multiWord(multiWord) {
      vocabulary = new Vocabulary(vocab_list, alpha_size);

      // root has blank label and alphabet size
      trieRoot = new TrieNode(alpha_size-1, alpha_size);
      for (std::vector<int32> word : vocab_list) {
          trieRoot->Insert(word);
      }
    }

  TrieBeamScorer(const char *dictionary_path, int alpha_size, bool multiWord) :
    multiWord(multiWord) {
      vocabulary = new Vocabulary(dictionary_path, alpha_size);
      std::vector<std::vector<int32>> vocab_list = vocabulary->GetVocabList();

      trieRoot = new TrieNode(alpha_size);
      for (std::vector<int32> word : vocab_list) {
        trieRoot->Insert(word);
      }
    }

  // State initialization
  void InitializeState(TrieBeamState* root) const override {
    root->incomplete_word_trie_node = trieRoot;
    root->score = 0;
  }
  // ExpandState is called when expanding a beam to one of its children.
  // Called at most once per child beam. In the simplest case, no state
  // expansion is done.
  void ExpandState(const TrieBeamState& from_state, int from_label,
                   TrieBeamState* to_state, int to_label) const override {

    // Ensure that from state has a trie node
    // If from state does not have a trie node, then we return without a state expansion
    TrieNode *node;
    if ((node = from_state.incomplete_word_trie_node) == nullptr) {
      to_state->incomplete_word_trie_node = nullptr;
      to_state->score = kLogZero;
      return;
    }

    // the from state is not null, copy it to the to state and expand appropriately
    CopyState(to_state, from_state);

    if (from_label == to_label)
      return;

    // If the the from state is at a leaf of the trie, then set the to state to
    // the root of the trie to expand from a new word
    if (node->IsEnd() && multiWord) {
      ResetIncompleteWord(to_state);
      node = to_state->incomplete_word_trie_node;
    }

    // set the node to be the child of the current node
    if (!vocabulary->IsBlankLabel(to_label)) {
      node = node->GetChildAt(to_label);
      to_state->incomplete_word_trie_node = node;
      to_state->incomplete_word.push_back(to_label);
      to_state->score = node != nullptr ? std::log(1.0) : kLogZero;
    }
  }
  // ExpandStateEnd is called after decoding has finished. Its purpose is to
  // allow a final scoring of the beam in its current state, before resorting
  // and retrieving the TopN requested candidates. Called at most once per beam.
  void ExpandStateEnd(TrieBeamState* state) const override {
    std::cout << "ExpandStateEnd" << std::endl;
    if (state->incomplete_word_trie_node == nullptr ||
        !state->incomplete_word_trie_node->IsEnd()) {
      state->incomplete_word.clear();
      state->incomplete_word_trie_node = nullptr;
      state->score = kLogZero;
    } else {
      for (int i : state->incomplete_word) std::cout << i << " ";
      std::cout << std::endl;
      state->score = std::log(1.0);
    }
  }
  // GetStateExpansionScore should be an inexpensive method to retrieve the
  // (cached) expansion score computed within ExpandState. The score is
  // multiplied (log-addition) with the input score at the current step from
  // the network.
  //
  // The score returned should be a log-probability. In the simplest case, as
  // there's no state expansion logic, the expansion score is zero.
  float GetStateExpansionScore(const TrieBeamState& state,
                               float previous_score) const override {
    return previous_score + state.score;
  }
  // GetStateEndExpansionScore should be an inexpensive method to retrieve the
  // (cached) expansion score computed within ExpandStateEnd. The score is
  // multiplied (log-addition) with the final probability of the beam.
  //
  // The score returned should be a log-probability.
  float GetStateEndExpansionScore(const TrieBeamState& state) const override {
    return state.score;
  }

  TrieNode *GetTrieRoot() {
    return trieRoot;
  }

 private:
  Vocabulary *vocabulary;
  TrieNode *trieRoot;
  bool multiWord;

  void CopyState(TrieBeamState* to_state, const TrieBeamState& from_state) const {
    to_state->incomplete_word_trie_node = from_state.incomplete_word_trie_node;
    to_state->incomplete_word = from_state.incomplete_word;
    to_state->score = from_state.score;
  }

  void ResetIncompleteWord(TrieBeamState* state) const {
    state->incomplete_word.clear();
    state->incomplete_word_trie_node = trieRoot;
  }
}; // TrieBeamState

}  // namespace ctc
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_UTIL_CTC_CTC_BEAM_SCORER_H_
