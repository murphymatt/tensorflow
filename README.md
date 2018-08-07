# Lexicon-Restricted CTC Decoder -- Design Document
## Matt Murphy
## CSC499: Deep Learning OCR

### __General Problem, Approach__
The goal of this project was to augment the standard CTC decoder, implemented in
TensorFlow, to produce output, restricted by a pre-defined lexicon. Given such a
restriction, nonsensical, or "almost-correct" decodes should be pruned from the
network output.

Our approach to efficiently implement such a dictionary restriction relies on
storing the lexicon within a trie, which is then referenced on each step within
the CTC decoder's Beam Search. Beam Search is a path-based algorithm, which
considers only the _n_ most likely moves on each ply. In this case, each move
represents performing a state expansion to a specific label, based on the 
label's logit scores.

In the case of our lexicon-augmented decoder, during each state expansion, upon
considering moving to the label in question, we check to see if that label is a
child of our current location within the trie. If the label is a child, then we
know that performing a state expansion to that label would be expanding our
frontier within a prefix of some dictionary word. This, such an expansion would
be valid, as restricted by our dictionary. If the label in question does not 
have a corresponding child of our current location within the trie, then we do
set a score for that expansion to be the 0 log probability (-inf) and that state
is not added to our beam, thus ending our beam search in that path through the
input logits.

Upon reaching the end of the beam search, we determine if our current location 
within the dictionary trie signifies the end of a lexicon word. If it is not, we
set the score for the decode at that state to be the 0 log probability, so that
decode path is not outputted by the CTC beam search decoder.

#### __Concrete examples of interaction with trie__

The beam search repeatedly interacts with the trie in three locations within the
`TrieBeamScorer`: `InitializeState`, `ExpandState`, and `ExpandStateEnd`. In
`InitializeState`, this scorer function is called by the beam search decoder 
with a beam search root state as a parameter. Accordingly, we set the trie node
root to be equal to root of the dictionary trie. On each state expansion, if the
`to_label` is different from the `from-label` and the `to_label` is not blank,
then we search for the `to_label` within the children of our current trie node
of our `from_state` to see if that new label continues a prefix from our 
dictionary. Subsequently, at the end of the beam search, the `ExpandStateEnd` 
function references the `endWord` flag of the current state's trie node to set 
that decode path's score appropriately. Each of these functions are called for
any non-trivial decode, so the trie must be referenced within this scorer 
implementation.

So, if we have an input dictionary of
```
root
|
1
|\
1 3
| |
3 1
```
and an input pre-log-scored logit sequence of

```
{{{0, 0.4, 0, 0.6, 0}},   // timestep 0
 {{0, 0.4, 0, 0.4, 0.2}}, // timestep 1
 {{0, 1,   0, 0,   0}},   // timestep 2
 {{0, 0.6, 0, 0.4, 0}},   // timestep 3
 {{0, 0.5, 0, 0.5, 0}}}   // timestep 4
```
On timestep 0, while label 1 has a lower log-score than label 3, we must select
label 1, since it is the only child of our root within the trie. Within 
`ExapndState`, given a `to_label` of 3, we would set the score to be log 0, and
that state expansion would not be added to `leaves_`: the data structure that
maintains the current frontiers of the beam search.

Suppose that at timestep 4, we have a current path of [1, 4, 1, 1], where 4
signifies a blank label. Our beam state must maintain a reference to the trie
node with label 1 on the second level below the root of our trie. A `to_label` 
of 0, 1, or 2, would not be valid, as none of those labels are children of that
node within the trie. Appropriately the scores of each of those 'potential
state expansions' should be log 0, and none would be considered candidate
path expansions to be added to `leaves_`. Only a state expansion to label 3 
would be considered valid for timestep 4 and our new frontier state would 
maintain a reference to this node. Subsequently, in the call to 
`ExpandStateEnd` at after this last timestep, we check to see if this frontier
node denotes the end of a word and see that it does and thus the decode
[1, 4, 1, 1, 3] is considered as a potential output path. More cases similar to
this are listed within `ctc_trie_beam_search_test.h`

### __Files Developed__

This MAP involved the creating the additional files `ctc_trie_node.h` and
`ctc_vocabulary.h`, as well as creating large additions to the files 
`ctc_beam_entry.h` and `ctc_beam_scorer.h`. For some small-scale testing 
purposes, a slight modification to `ctc_beam_search.h` was made. Tests were
constructed within the files `ctc_trie_node_test.h` and 
`ctc_trie_beam_seach_test.h`.

Additionally, an operation implementing these CTC additions is defined by an 
operation, which serves as a header for the implementation, which is constructed
within the kernel.

Moreover, to call the back-end C++ op from python code, I made an additional
definition for this call within the python operation definition.

Lastly, I made a change to Professor Weinman's 
[__cnn_lstm_ctc_ocr__](https://github.com/weinman/cnn_lstm_ctc_ocr/tree/update-to-tf-1.8) 
repository, such that the lexicon-restricted decoder may be called with the flag 
`--lexicon /path/to/lexicon`. At the time of writing this report, this change
lies within the branch `update-to-tf-1.8`.

#### __Where they live__
The core utilities for CTC live within the directory 
`//tensorflow/core/util/ctc/`. 

The op definition lives within the file
`//tensorflow/core/ops/ctc_ops.cc` and the kernel lives within the file
`//tensorflow/core/kernels/ctc_decoder_ops.cc`.

Additionally, the python definition, which calls the back-end C++ op lives within
the file `//tensorflow/python/ops/ctc_ops.py`.

#### __Why they're there__
The CTC logic for decoding and all the supporting classes and functions exist as
utilities, since they simply used operation kernels, as sorts of 'helper 
functions'. Following the standard already in place from the base beam search
decoder tests, I decided to include the tests for the additional logic within 
the same directory. Since these are simply additions that rely on the same
dependencies as the built-in base beam-search decoder, I decided it was
appropriate to include these additions within the `core` directory, rather than
some `contrib` repository.

Originally, I wanted to place definition and kernel for `ctc_beam_search_trie`
within a separate `contrib` repository, in order to adhere to the general 
TensorFlow contribution guidelines. However, after much headache trying to get
everything to compile, I found in some research that two of the dependencies of
the kernel implementation of the op, namely `//tensorflow/core:lib` and
`//tensorflow/core:framework` are not able to be included within the build
path of files within `contrib`.

So, I placed the new op definition and kernel within the `core` op files, as
simple additions appended to the end of these libraries. Note that in doing this
workaround, these additions are not currently following the standard 
contribution guidelines. If a pull request to the TensorFlow master branch is to
be made, then this explanation should be referenced.

#### __What they do__
To implement the trie as a storage and navigation mechanism for the lexicon, I
created the additional files `ctc_trie_node.h` and `ctc_vocabulary.h`. 
`ctc_vocabulary.h` provides helper functions 

The core augmentation utilizing the trie is an additional beam scorer class 
within `ctc_beam_scorer.h` called `TrieBeamScorer`, which references the
trie at each time-step within the path state expansion. If the character is not
valid, as restricted by the trie, then the scorer assigns a log 0 score to that
beam state. Therefore, the contributing score of that state expansion sets the
score of the whole path to be log 0. 

To maintain information of the trie at each state within the beam search, a new
state structure, `TrieBeamState`, maintains a reference to a node within the 
trie, as well as a vector containing the labels which compose the trie 
navigation path up to that point. This 'prefix' vector is primarily for 
debugging purposes, as it is not used in beam search navigation decisions,
but is useful for tracking intermediate progress within the search. Note, that
the 'word' contained within this vector forms the post-merge-repeated sequence
of labels, as the trie only contains complete words, without label repeats as
would be generally present among adjacent timesteps within the logit input.
All said, this new beam state within `ctc_beam_entry.h` serves as an interface
between the beam search and a backing trie structure, generated from a lexicon
before entering the beam search.

TensorFlow works by building relatively high level operations, or `ops` in the
`core` back-end, and then invoking these operations through front-end APIs (in
our case, in Python). So, if we want to create a way for someone building their
model in python to use CTC decoding, we must create such an operation.  Every
operation, requires a definition within an 'op' file, as well as a 'kernel' 
which servers as the implementation of the function definition. An op
definition, `CTCBeamSearchDecoderTrie`, is defined within `ctc_ops.cc` and
serves as a function definition for this operation, and also does some input
checking to ensure correct input tensor rank and dimension equality, as well as
set up the output tensors. The kernel implementation,
`CTCBeamSearchDecoderTrieOp`, within `ctc_decoder_ops.cc`, provides the 
necessary compute function, which sets up the dictionary trie and uses the CTC
utilities to decode an input logit sequence. This is the `core` operation that 
is called by the Python API function in order to perform CTC decoding.

Lastly, from within the Python directory within TensorFlow, we implement a
function that calls our backend C++ operation, and export this function as a
module under `nn` so that the user may call this function in python code.


### __How to make things work__
Currently, all this code lies within my personal 
[__TensorFlow fork__](https://github.com/murphymatt/tensorflow/), within the
master branch. First, to get each of these files, simply clone a copy of this 
repository using the command:
```sh
git clone https://github.com/murphymatt/tensorflow.git
```
to your machine and build from there. Additionally, to run Professor Weinman's
model which uses the augmented trie, clone the repository using the following 
command:
```sh
git clone https://github.com/weinman/cnn_lstm_ctc_ocr.git
```

TensorFlow also requires some third-party dependencies to run. Note, that on the
mathlan, or some other machines, you may or may not have sudo access, so I like
to use __virtualenv__ to create a local, isolated environment which permits 
installing pip packages to without sudo access. Installing the dependencies 
should then proceed as follows:

```sh
virtualenv ~/tf_custom_env
source ~/tf_custom_env/bin/activate
pip install six numpy wheel 
```

After that, simply build the custom TF package from within the repository clone
and then proceed. Once the build process is completed, outside of the TensorFlow
repository, install the custom package

#### __Compiling__
First, to build TensorFlow from source, you must install Bazel. System-specific Bazel installation instructions can be found
[__here__](https://bazel.build/versions/master/docs/install.html). 

Subsequently, you must simply configure and then build with Bazel. From within
the repository, call `./configure` from the root of the repository clone. Some
installation preferences may pop up, but all that we have really used for 
testing purposes this semester has been CUDA support. So in configuration, you
can simply say 'yes' to that option, and 'no' to everything else. All default
file paths may be used when prompted.

Subsequently, after configuration, installation of the entire package is simply
```sh
bazel build --config=opt --config=cuda //tensorflow/tools/pip_package:build_pip_package
```
Depending on the machine you are using, this build process may take up to an
hour or more, especially when building initially. Patience is a virtue.

Once the compilation is done, create the pip package with the following:
```sh
bazel-bin/tensorflow/tools/pip_package/build_pip_package ~/tmp/tensorflow_pkg
```
This will create a pip package within a user-level directory that you can install
with pip. After that is completed, outside of the directory, and with a custom
virtualenv activated, simply call
```sh
pip install ~/tmp/tensorflow_pkg/tensorflow-X.X.X-cpPYTHONVERSION-SYSTEMSPEC.whl
```
substituting the appropriate variables with what exists inside of the directory.
Now, the custom TensorFlow installation is usable any time you are inside this
virtualenv. To make incremental upgrades, simply repeat the compilation process.
The `build_pip_package` usually does not take too long, following small changes.

#### __Updating tests__
Before using the custom TF build on the mjsynth data, the user should first run
the tests of the dictionary-restricted components. Running these tests is simply

```sh
bazel test --dbg //tensorflow/core/util/ctc:ctc_trie_node_test
```
```sh
bazel test --dbg //tensorflow/core/util/ctc:ctc_trie_beam_search_test
```

Any additional test cases regarding the trie structure should go within the 
first test file, but changes to the beam search scorer, or entry, which utilize
the trie logic should go within `ctc_trie_beam_search_test.cc`.

#### __Anything else to know__
Currently, there is a small bug in which sometimes incomplete prefixes are
emitted by the beam search decoder. This bug was observed when using the custom
build over mjsynth data, on select words. For example, running the end-to-end
model over `2210/3/72_FENCES_28542.jpg` will output multiple 'FENCES' outputs, 
but will also output 'ENCE' as one of the lower-ranked top paths output. While
'ENCE' is a strict-prefix of a dictionary word, it is not a complete word and
should not be emitted by the decoder. This is likely an issue within
`ExapandStateEnd` within the `TrieBeamScorer`. This function should check to
see if the input `TrieBeamState`'s current node has the `endWord` flag set. If
this flag is `false`, then the scorer should set the state expansion score to be
log 0. Over my last week and a half of my work on this project, I tried various
changes and debugging techniques, but 

It is also worth noting that beam search extensions should be made within
new beam scorers and beam entries, as opposed to adding to or modifying 
`ctc_beam_search.h`. However, I made two small changes to `ctc_beam_search.h`, 
in order to get everything running. Originally, in the `TopPaths` function,
if the number of top paths stored within the `leaves_` structure (the structure 
that holds candidate paths, updated on each timestep of the beam search) is less
than the number _n_ of paths to return, then the function returns an error and 
no top paths are returned. In my modified version, in order to get the 
small-scale test cases to pass, I set _n_ equal to the minimum of _n_ and 
the number of leaves stored. This way, the beam search then outputs at most _n_
top paths.

Additionally, something I found that may be a bug within the beam search had to
do with checking candidate values after performing the `ExpandStateEnd` function
at the end of the beam search, before determining the top paths. In the base
beam search, this function does not apply any changes to the beam state, but, in
this modified search, this function should set the scores of incomplete words to
log 0. Originally, the function does not check to see if the state is still a
candidate after this function, but I put in a quick fix to check to see if it is
still a candidate. I will be submitting a written up pull request to TensorFlow
master soon.

Lastly, you can debug developments to the core framework using gdb. For
example, if after the development of additional test cases in
``ctc_trie_beam_search_test.cc``, some tests are failing, you can compile the
test library with debugging flags as follows:
```sh
bazel build --config=dbg //tensorflow/core/util/ctc:ctc_trie_beam_search_test
```
and then run the test using
```sh
gdb bazel-bin/tensorflow/core/util/ctc/ctc_trie_beam_search_test
```

It is worth noting that I was only ever able to make this work when compiling on
a machine without CUDA support. Additionally, if there are some issues from 
running the mjsynth Python code, after compiling the pip package with debugging
flags, you can debug the python code using IPython as follows:

```sh
ipython
!ps | grep -i ipython
```

to get the process id of the IPython process, then attach this process with a 
debugger and then continue the process within the debugger, then run the python
code by hopping back to IPython and calling

```sh
run example_script.py
```

This was very helpful for debugging issues that resulted when calling 
`validate.py` over the mjsynth dataset.

#### __How might this tool change as TF changes?__
If you are working on a branch of my fork or your own fork, you should regularly
keep up to date with the master TensorFlow repository. To do this, add the master
repo as an upstream repository as follows:
```sh
git remote add upstream https://github.com/tensorflow/tensorflow.git
```

You should fetch and merge regularly as follows:
```sh
git fetch upstream
git merge upstream/master
```

It is unlikely that there would be merge conflicts, but if they are, they may
be resolved appropriately. If any CTC utility file would be most likely to 
obtain merge conflicts, it would likely be `ctc_beam_search.h` since this is
the utility that should generally not change. It may also be possible that the
kernel or op specifications may change, but if so, the format of the op or 
kernel should just stay consistent with the others within their respective 
files.

---

<div align="center">
  <img src="https://www.tensorflow.org/images/tf_logo_transp.png"><br><br>
</div>

-----------------


| **`Documentation`** |
|-----------------|
| [![Documentation](https://img.shields.io/badge/api-reference-blue.svg)](https://www.tensorflow.org/api_docs/) |

**TensorFlow** is an open source software library for numerical computation using
data flow graphs.  The graph nodes represent mathematical operations, while
the graph edges represent the multidimensional data arrays (tensors) that flow
between them.  This flexible architecture enables you to deploy computation to one
or more CPUs or GPUs in a desktop, server, or mobile device without rewriting
code.  TensorFlow also includes [TensorBoard](https://www.tensorflow.org/programmers_guide/summaries_and_tensorboard), a data visualization toolkit.

TensorFlow was originally developed by researchers and engineers
working on the Google Brain team within Google's Machine Intelligence Research
organization for the purposes of conducting machine learning and deep neural
networks research.  The system is general enough to be applicable in a wide
variety of other domains, as well.

Keep up to date with release announcements and security updates by
subscribing to
[announce@tensorflow.org](https://groups.google.com/a/tensorflow.org/forum/#!forum/announce).

## Installation
*See [Installing TensorFlow](https://www.tensorflow.org/get_started/os_setup.html) for instructions on how to install our release binaries or how to build from source.*

People who are a little more adventurous can also try our nightly binaries:

**Nightly pip packages**
* We are pleased to announce that TensorFlow now offers nightly pip packages
under the [tf-nightly](https://pypi.python.org/pypi/tf-nightly) and
[tf-nightly-gpu](https://pypi.python.org/pypi/tf-nightly-gpu) project on pypi.
Simply run `pip install tf-nightly` or `pip install tf-nightly-gpu` in a clean
environment to install the nightly TensorFlow build. We support CPU and GPU
packages on Linux, Mac, and Windows.


#### *Try your first TensorFlow program*
```shell
$ python
```
```python
>>> import tensorflow as tf
>>> hello = tf.constant('Hello, TensorFlow!')
>>> sess = tf.Session()
>>> sess.run(hello)
'Hello, TensorFlow!'
>>> a = tf.constant(10)
>>> b = tf.constant(32)
>>> sess.run(a + b)
42
>>> sess.close()
```
Learn more examples about how to do specific tasks in TensorFlow at the [tutorials page of tensorflow.org](https://www.tensorflow.org/tutorials/).

## Contribution guidelines

**If you want to contribute to TensorFlow, be sure to review the [contribution
guidelines](CONTRIBUTING.md). This project adheres to TensorFlow's
[code of conduct](CODE_OF_CONDUCT.md). By participating, you are expected to
uphold this code.**

**We use [GitHub issues](https://github.com/tensorflow/tensorflow/issues) for
tracking requests and bugs. So please see
[TensorFlow Discuss](https://groups.google.com/a/tensorflow.org/forum/#!forum/discuss) for general questions
and discussion, and please direct specific questions to [Stack Overflow](https://stackoverflow.com/questions/tagged/tensorflow).**

The TensorFlow project strives to abide by generally accepted best practices in open-source software development:

[![CII Best Practices](https://bestpractices.coreinfrastructure.org/projects/1486/badge)](https://bestpractices.coreinfrastructure.org/projects/1486)


## Continuous build status

### Official Builds

| Build Type      | Status | Artifacts |
| ---             | ---    | ---       |
| **Linux CPU**   | ![Status](https://storage.googleapis.com/tensorflow-kokoro-build-badges/ubuntu-cc.png) | [pypi](https://pypi.org/project/tf-nightly/) |
| **Linux GPU**   | ![Status](https://storage.googleapis.com/tensorflow-kokoro-build-badges/ubuntu-gpu-cc.png) | [pypi](https://pypi.org/project/tf-nightly-gpu/) |
| **Linux XLA**   | TBA | TBA |
| **MacOS**       | ![Status](https://storage.googleapis.com/tensorflow-kokoro-build-badges/macos-py2-cc.png) | [pypi](https://pypi.org/project/tf-nightly/) |
| **Windows CPU** | [![Status](https://ci.tensorflow.org/buildStatus/icon?job=tensorflow-master-win-cmake-py)](https://ci.tensorflow.org/job/tensorflow-master-win-cmake-py) | [pypi](https://pypi.org/project/tf-nightly/) |
| **Windows GPU** | [![Status](http://ci.tensorflow.org/job/tf-master-win-gpu-cmake/badge/icon)](http://ci.tensorflow.org/job/tf-master-win-gpu-cmake/) | [pypi](https://pypi.org/project/tf-nightly-gpu/) |
| **Android**     | [![Status](https://ci.tensorflow.org/buildStatus/icon?job=tensorflow-master-android)](https://ci.tensorflow.org/job/tensorflow-master-android) | [![Download](https://api.bintray.com/packages/google/tensorflow/tensorflow/images/download.svg)](https://bintray.com/google/tensorflow/tensorflow/_latestVersion) [demo APK](https://ci.tensorflow.org/view/Nightly/job/nightly-android/lastSuccessfulBuild/artifact/out/tensorflow_demo.apk), [native libs](https://ci.tensorflow.org/view/Nightly/job/nightly-android/lastSuccessfulBuild/artifact/out/native/) [build history](https://ci.tensorflow.org/view/Nightly/job/nightly-android/) |


### Community Supported Builds

| Build Type      | Status | Artifacts |
| ---             | ---    | ---       |
| **IBM s390x**       | [![Build Status](http://ibmz-ci.osuosl.org/job/TensorFlow_IBMZ_CI/badge/icon)](http://ibmz-ci.osuosl.org/job/TensorFlow_IBMZ_CI/) | TBA |
| **IBM ppc64le CPU** | [![Build Status](http://powerci.osuosl.org/job/TensorFlow_Ubuntu_16.04_CPU/badge/icon)](http://powerci.osuosl.org/job/TensorFlow_Ubuntu_16.04_CPU/) | TBA |


## For more information

* [TensorFlow Website](https://www.tensorflow.org)
* [TensorFlow White Papers](https://www.tensorflow.org/about/bib)
* [TensorFlow YouTube Channel](https://www.youtube.com/channel/UC0rqucBdTuFTjJiefW5t-IQ)
* [TensorFlow Model Zoo](https://github.com/tensorflow/models)
* [TensorFlow MOOC on Udacity](https://www.udacity.com/course/deep-learning--ud730)
* [TensorFlow Course at Stanford](https://web.stanford.edu/class/cs20si)

Learn more about the TensorFlow community at the [community page of tensorflow.org](https://www.tensorflow.org/community) for a few ways to participate.

## License

[Apache License 2.0](LICENSE)
