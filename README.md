# NLP-NER-Classification

Implementation of a GRU recurrent neural network in Pytorch for Named-Entity Recognition (NER). 
The input sentences have been encoded using FastText pre-trained word embedding.
Project developed for Natural Language Processing course held by professor R. Navigli (Sapienza University of Rome).

## Requirements
- Install anaconda
- Create the environment: `$ conda create --name nlp-ner python==3.7`
- Activate the environment: `$ conda activate nlp-ner`
- Install requirements: `pip install requirements.txt`

## Usage
Before starting, download the FastText pretrained word-embedding [wiki-news-300d-1M.vec.zip](https://dl.fbaipublicfiles.com/fasttext/vectors-english/wiki-news-300d-1M.vec.zip)
unzipping it inside the `model` folder. Any other pretrained word-embedding can be used, just remember to change 
accordingly the embedding size.
To train and test the network just run the Jupyter Notebook `main.pynb`. It's possible to change hyperparameters of
the network modifying values inside `model.py`.
## Results
This configuration reach 91% F1-score on test set.
