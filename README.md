# EmoNoBa: A Dataset for Analyzing Fine-Grained Emotions on Noisy Bangla Texts

This is the implementation of our paper "EmoNoBa: A Dataset for Analyzing Fine-Grained Emotions on
Noisy Bangla Texts". This work has been accepted at the AACL-IJCNLP 2022.

## Abstract
For low-resourced Bangla language, works on detecting emotions on textual data suffer from size and cross-domain adaptability. In our paper, we propose a manually annotated dataset of 22,698 Bangla public comments from social media sites covering 12 different domains such as Personal, Politics, and Health, labeled for 6 fine-grained emotion categories of the Junto Emotion Wheel. We invest efforts in the data preparation to 1) preserve the linguistic richness and 2) challenge any classification model. Our experiments to develop a benchmark classification system show that random baselines perform better than neural networks and pre-trained language models as hand-crafted features provide superior performance.

## Authors

* Khondoker Ittehadul Islam <sup>1</sup>
* Tanvir Hossain Yuvraz <sup>1</sup>
* Md Saiful Islam <sup>1,2</sup>
* Enamul Hassan <sup>1</sup>

<sup>1</sup> Shahjalal University of Science and Technology, Bangladesh
<br>
<br>
<sup>2</sup> University of Alberta, Canada

## EmoNoBa Dataset is available [here](https://www.kaggle.com/datasets/saifsust/emonoba) 

#### List of files

* Train.csv
* Val.csv
* Test.csv

#### Files Format
Column Title | Description
------------ | -------------
Data | Social media comment
Love | 0, 1. '1' for Love, '0' for Not Love
Joy | 0, 1. '1' for Joy, '0' for Not Joy
Surprise | 0, 1. '1' for Surprise, '0' for Not Surprise
Anger | 0, 1. '1' for Anger, '0' for Not Anger
Sadness | 0, 1. '1' for Sadness, '0' for Not Sadness
Fear | 0, 1. '1' for Sadness, '0' for Not Fear
Topic | Topic of the comment
Domain | Source of the comment from {Youtube, Facebook and Twitter}

## INSTALLATION

Requires the following packages:
* Python 3.10.7 or higher

It is recommended to use virtual environment packages such as **virtualenv**. Follow the steps below to setup the project:
* Clone this repository via `git clone https://github.com/KhondokerIslam/EmoNoBa.git`
* Use this command to install required packages `pip install -r requirements.txt`
* Run the setup.sh file to download additional data and setup pre-processing

## Usage

1. Download the EmoNoBa dataset from [here](https://www.kaggle.com/datasets/saifsust/emonoba)
2. Unzip the folder
3. Ensure the folder name is "EmoNoBa Dataset"
4. Go to data_processing folder and run `python preprocess.py` to obtain the preprocessed data.

#### Feature-Based Experiments
* Go to Models folder
* Use `python feature_based.py`
* Type in the model name when you will be asked to specify the model name in the console
* Model Names (Please follow the paper to read the details about experiments):
  * W1
  * W2
  * W3
  * W4
  * W1+W2
  * W1+W2+W3
  * W1+W2+W3+W4
  * C2
  * C3
  * C4
  * C5
  * C1+C2+C3
  * C1+C2+C3+C4
  * C1+C2+C3+C4+C5
  * W1+C1+C2+C3+C4+C5
  * W1+W2+W3+C1+C2+C3
  * W1+W2+W3+W4+C1+C2+C3
 
 #### Neural Network Experiments
 
##### Random Initialize

* Go to Models folder
* Use "python neural_network_(random).py" to run an experiment.

##### FastText

* Go to Models folder
* Use "python neural_network_(embedding).py" to run an experiment.

#### Bangla-BERT

* Go to Models folder
* Use "python bangla-bert.py" to run an experiment.
