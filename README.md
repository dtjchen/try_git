```python
"""
Digital Signal Processing Lab
Project: 12.18.2015
"""
team  = ['Derek Chen', 'Miguel Amigot']
title = 'Speaker Identification: Linear Model'
```

### Background
#### Objective
The objective of this project was to develop a machine learning-based model to identify speakers from a known set. That is, by training the model on a speaker's voice, it would subsequently be able to identify.

*Note that this is a stepping-stone in our project. Ultimately, the objective for the second half of our design project will be to train the model in an unsupervised manner (i.e. no labeled data). This will make it possible to feed a speech signal with multiple speakers into our system and, through a Recurrent Neural Network (RNN), extract and cluster chunks belonging to each speaker.*

#### Challenge
> Simply put, the identity information of the speaker is embedded in how speech is spoken, not necessarily in what is being said.

> –<cite>[Hansen and Hasan, Ref. 1]</cite>

The challenge in speaker identification lies in developing a system that is capable of dealing with the way in which people change their voices, either willingly or unwillingly.

Heuristics based on e.g. a speaker's average frequencies do not work well if he deliberately heightens the pitch or volume of his voice, whispers or simply falls ill with a cold. Moreover, the reasoning behind the heuristic techniques we came across (e.g. calculating derivatives of the time-domain waveform of a speaker) seemed rather arbitrary and weak when dealing with complex datasets.

#### Ideal Feature Parameters
By utilizing a linear model as our system, we are making the assumption that each speaker's speech traits can be clustered into some section of an *n-dimensional* space (the number of dimensions is fixed, as will be discussed later).

These traits, developed through a series of feature vectors, must meet the following criteria in order to be successfully clustered:

>1. Show high between-speaker variability and low within-
speaker variability

>2. Be resistant to attempted disguise or mimicry

>3. Have a high frequency of occurrence in relevant materials

>4. Be robust in transmission

>5. Be relatively easy to extract and measure.

> –<cite>[Hansen and Hasan, Ref. 1]</cite>

### Dataset

The segments of audio in the dataset were obtained from animated TV shows, where actors lend their voices to characters on the show. Voice actors often record for a multitude of characters, both from the same program and across different shows. One notable example of the numerous amount of characters one voice actor can perform as can be seen with actor Hank Azaria. On "*The Simpsons*", he voices over 30 characters on the show, from regulars Moe Szyslak, Chief Wiggum and Apu Nahasapeemapetilon, to more minor characters like Disco Stu or Drederick Tatum. To the casual listener, it may not be apparent that one person voices such a diverse range of characters.

The audio in the dataset are from the shows "*The Simpsons*" and "*Futurama*." These shows were chosen because the actors use their natural voices on the show, as opposed to shows such as "*South Park*" in which actors may pitch shift their voice. The speech clips were extracted from random episodes of both shows and labeled according to their corresponding speaker with the aid of transcripts. Over 4 GB of audio was retrieved, and split into 80% for the training set and 20% for the test set. Each sample was around a couple seconds in length.

| Speaker ID  | Name | Wavfile Count |
| :-------------: | :------------: | :-------------: |
| 0  | Julie Kavner | 2274 |
| 1  | Harry Shearer| 2803 |
| 2  | Maurice LaMarche| 390 |
| 3  | David Herman | 198 |
| 4  | Maggie Roswell | 91 |
| 5  | John DiMaggio | 987 |
| 6  | Pamela Hayden | 281 |
| **7**  | **Dan Castellaneta** | **5231** |
| 8  | Hank Azaria | 215 |
| 9  | Phil LaMarr | 1836 |
| 10 | Tress MacNeille | 443 |
| 11 | Billy West | 1458 |
| 12 | Nancy Cartwright | 1793  |

In addition to the disguised voices, the dataset was also particularly difficult to train with due to the uneven distribution of audio files available for each actor. Dan Castellaneta, who famously voices Homer Simpson, among other voices, has over 5000 samples in the training set. Maggie Roswell, who voices minor characters like Helen Lovejoy on "The Simpsons", has fewer than 100 samples. This discrepancy was somewhat problematic in having the program identify these voices.

### Approach

#### Input Data
In order to develop a model that can accurately determine who the speaker of a particular `.wav` file is, input data must be converted to feature vectors which meet the criteria outlined in *"Ideal Feature Parameters"* above as closely as possible.

Initially, we planned to develop these feature vectors by flattening `.wav` files' spectrogram matrices, generated using `sox` or `matplotlib` (explained below). Moreover, each matrix would correspond to 20 ms. of a recording; thus a long recording would lead to multiple matrices of a normalized dimension as opposed to a single one, covering a large portion of data.

![spect img](http://i.imgur.com/n8Q78Th.png)

>Example of a spectrogram from a `.wav` file in the dataset, generated using `sox` as well as the "high contrast" flag to exacerbate the differences between frequency intensities.

##### Mel-Frequency Cepstral Coefficients (MFCCs)

Ultimately, we created feature vectors from our data by applying the MFCC procedure to each input `.wav` file. This process is well-established in the industry of speech recognition because of its relative accuracy as far as extracting revealing patterns in human speech is concerned.

20 MFCC coefficients were chosen for this task (there did not appear to be a direct correlation between a higher number of coefficients and the accuracy of the program). These were calculated using the `librosa` library, namely `librosa.feature.mfcc(y, sr=22050, n_mfcc=20)`, where `y` denotes the input vector and `sr` specifies its sampling rate.

MFCC coefficients were evidently computed in order to train and test the data (see `linearclassifier/train.py`, line 51 and `linearclassifier/test.py`, line 38). The function accepted a vector of an arbitrary length (varying depending on the size of the `.wav` file) and returned a fixed-sized `numpy.ndarray` object.

### Linear Model
Provided an input feature vector `x`, calculated by applying the MFCC procedure to a signal, the system outputs a vector `y` denoting the probabilities that the input vector belongs to each of the known speakers.

Evidently, if the input vector `x` belongs to *speaker 1*, the probability that corresponds to *speaker 1* in `y` should be higher than those of other speakers.

As outlined in `linearclassifier/train.py` (lines 22 and 23), the dimensions of `x` are `[NUMBER_OF_MFCC_COEFFICIENTS, 1]` and those of `y` are `[NUMBER_OF_CLASSES_IN_DATASET, 1]`, where `NUMBER_OF_MFCC_COEFFICIENTS` is fixed and `NUMBER_OF_CLASSES_IN_DATASET` denotes the number of speakers (one class per speaker).

The process is summarized by the following code (`linearclassifier/train.py`, line 35) and diagram, from [tensorflow.org](tensorflow.org). Our system develops `W` and `b`, and applies the Softmax procedure to turn `Wx + b` into a vector of speaker probabilities.

```python
# y = softmax(Wx + b)
y = tf.nn.softmax(tf.matmul(x,W) + b)
```

![alt](https://www.tensorflow.org/versions/master/images/softmax-regression-scalargraph.png)

At a high level, the program is divided into `linearclassifier/` and `processor/`. It uses a series of utilities and functions that use the dataset to train the model and save `W` and `b` as `.npy` files on the root directory of the project. These matrices are consequently read by `linearclassifier/test.py` to perform the required calculations and output a prediction from a user-provided recording (provided as a command-line argument).

* `linearclassifier/`
  * `__init__.py`: configuration variables and utility functions to write and read the relevant `.npy` files.
  * `train.py`: generate `W` and `b`, and store them separately.
  * `test.py`: accept an input `.wav` and make a prediction utilizing `W` and `b`.
  * `utils.py`: utility functions applicable only to the linear model, such as `generate_one_hot_vector()`.
* `processor/`
  * `__init__.py`: make relevant modules accessible at the package level.
  * `normalize.py`: functions to process `.wav` files while training and testing, such as to calculate MFCC coefficients using the `librosa` library.
  * `wavreader.py`: contains functions to load `.wav` files as `scipy` vectors (time-domain waveforms). Accepts either a path to a `.wav` file or utilizes environment variables to generate vectors for all `.wav` files in a given directory, matched to their labels in the dataset. Utilizes Python generators (see the `yield` keyword), which loads a single `.wav` file's vector to memory at-a-time (as opposed to loading gigabytes' worth of vectors to RAM at once).
  * `spectrogram.py`: contains classes to create spectrograms for `.wav` files. Accepts either a path to a `.wav` file or utilizes environment variables to generate spectrograms for all `.wav` files in a given directory, matched to their labels in the dataset. Spectrograms can be returned as images (using the `sox` utility, which writes to the filesystem) or as matrices using `matplotlib`. In the case of the latter, matrices are provided to the program using Python generators, which load chunks of data to memory one-at-a-time (as opposed to loading gigabytes' worth of spectrograms to RAM at once) *(unused in final version)*.
  * `chunkify.py`: take a `.wav` file and split into smaller `.wav` files using silence as a delimiter *(unused in final version)*.
* `fetchdata.sh`: download the dataset using `curl` and save it into directories at the root level of the program (decompressed using `tar`).
* `driver.py`: entry point into the program through the command-line. Provides commands which train and test the system.
* `requirements.txt`: contains names and versions of relevant Python packages (install through `pip install -r requirements.txt`).

#### TensorFlow
The linear model and softmax regression was developed using TensorFlow, a machine learning library developed by Google's Brain Team and released in November 2015. Its main selling point, aside from the intuitive and performant array of machine learning algorithms it provides, lies in its Python-based interface.

Though typically machine learning scientists and engineers debug their models using simpler languages such as Python, porting their algorithms to a GPU cluster where full datasets can be explored and analyzed requires them to use a faster language such as C++. This approach does not only tend to be time-consuming, but also prone to tedious bugs. With TensorFlow, however, the code that one writes to debug on the CPU can be directly ported to a GPU cluster (it is built to use CUDA, etc.).

#### Procedure
```bash
# Install dependencies
# (Follow TensorFlow's guidelines depending on your system: https://www.tensorflow.org/versions/master/get_started/os_setup.html#download-and-setup)
pip install -r requirements.txt

# Download the dataset (~4GB of training data and ~1GB of test data)
. fetchdata.sh

# Set env. variables to make data accessible to the model
# AUDIO_DIR: high-level directory containing wavfiles
# LABELS_FILE: called raw_list.txt in the provided dataset; maps wavfiles to labels/speakers
export AUDIO_DIR=~/dataset/train/raw/
export LABELS_FILE=~/dataset/train/annotations/raw_list.txt

# Train the model
python driver.py --linearclassifier --train

# Test the model using a .wav file
# (Prints prediction to stdout)
python driver.py --linearclassifier --test --wavfile /path/to/some/random/wavfile.wav
```

##### Additional Libraries
* [`numpy`](http://www.numpy.org/) and [`scipy`](www.scipy.org)
  * NumPy and SciPy are popular, widely used open-source scientific computing packages for the Python language. NumPy has useful multi-dimensional array constructs and functions for quick array manipulation. SciPy provides a myriad of scientific tools, including a number of mathematics and signal processing functions. These two fundamental packages are often integral to using other libraries.
* [`librosa`](https://bmcfee.github.io/librosa/generated/librosa.feature.mfcc.html#librosa.feature.mfcc)
  * Librosa is a package designed for audio analysis and signal processing. It facilitates work with utilities for feature extraction, audio effects and input/output, among others.
* [`matplotlib`](http://matplotlib.org/)
  * Matplotlib is a plotting library for Python with similar utilities to those MATLAB offers. It may be used to obtain spectrogram matrices through the `specgram` function.
* [`sox`](http://sox.sourceforge.net/)
  * Sound eXchange, or SoX, is an audio processing tool that can be accessed through the command-line. Branded as the "Swiss Army knife of sound processing programs," it may be used for such things as digital filtering or audio visualization.

### Results and Future Work
Some successful tests (corresponding to `.wav` files from speakers 0 and 7 in our environment, therefore the path on yours will differ) are shown below.

|# | Speaker  | Wavfile |
|:--:| :-------------: | :------------- |
|0|0|`~/dev/zebra/full_dataset/test_raw/2_07x06.Treehouse__Of__Horror__VI.x264.ac3/217.wav`|
|1|0|`~/dev/zebra/full_dataset/test_raw/2_07x20.Bart__On__The__Road.x264.ac3/39.wav`|
|**2**|**7**|`~/dev/zebra/full_dataset/test_raw/2_07x13.Two__Bad__Neighbors.x264.ac3/228.wav`|
|**3**|**7**|`~/dev/zebra/full_dataset/test_raw/2_07x15.Bart__The__Fink.x264.ac3/340.wav`|

Despite the difficulty of the dataset, the accuracy could be further improved. A stronger focus on feature extraction would likely improve the results of training and testing with the data. Perhaps identifying different patterns or incorporating other features may allow for better distinction and differentiation of voices. In addition, using more robust models may also improve the way the data is processed and the voices are learned. Using more sophisticated techniques such as k-means clustering, neural networks, or deep learning approaches would improve the accuracy of the results. Our next step will experiment with recurrent neural networks. Further training and testing can also be done with other datasets.

Once a satisfactory program for speaker recognition is developed, other ideas may be explored. One route may be to look at open-set recognition, where the number of speakers may expand after initial training, or unknown speakers may be tested. Audio segmentation, or creating a automated transcript of multiple speakers may be examined. Using speaker recognition for encryption purposes could be done. A more ambitious project may be speech generation using the features extracted from speech as parameters.

### References

1. Hansen, John HL, and Taufiq Hasan. "Speaker Recognition by Machines and Humans: A tutorial review." Signal Processing Magazine, IEEE 32.6 (2015): 74-99.
2. Uzan, Lior, and Lior Wolf. "I know that voice: Identifying the voice actor behind the voice." Biometrics (ICB), 2015 International Conference on. IEEE, 2015.
3. Hannun, Awni, et al. "DeepSpeech: Scaling up end-to-end speech recognition." arXiv preprint arXiv:1412.5567 (2014).
4. Neural Networks and Deep Learning, Softmax: http://neuralnetworksanddeeplearning.com/chap3.html#softmax
5. Speaker Recognition Using MFCC, http://www.slideshare.net/HiraShaukat/speaker-recognition-using-mffc
