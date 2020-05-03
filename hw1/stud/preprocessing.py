from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np

# Transform a list of sentences in a list of windows
# Each window is list of tokens/labels of fixed size where
# their size is "window_size"
# Append "None" for padding
def create_windows (sentences, window_size, window_shift):
  data = []
  for sentence in sentences:
    for i in range(0, len(sentence), window_shift):
      window = sentence[i:i+window_size]
      if len(window) < window_size:
        window = window + [None]*(window_size - len(window))
      assert len(window) == window_size
      data.append(window)
  return data

#Create a vocabulary (token -> key) from a list of sentences
def create_vocabulary(sentences):
    vocabulary = {}
    for sentence in sentences:
        for i,c in enumerate(sentence):
            vocabulary [c] = 1 #assign 1 temporarly

    for i,c in enumerate(vocabulary):
        vocabulary[c] = i

    #Append UNK and PAD as last elements of the vocabulary
    vocabulary.update({'<UNK>':len(vocabulary.keys())})
    vocabulary.update({'<PAD>': len(vocabulary.keys())})

    return vocabulary

#Create a vocabulary (label -> key) from a list of labels
def create_labels_vocabulary():
  labels = ['PER', 'LOC', 'ORG', 'O']
  vocabulary = {}
  for i,c in enumerate(labels):
    i = float(i) #Need a float number to use as output of NN
    vocabulary[c] = i
  vocabulary.update({None: float(len(vocabulary.keys()))}) #None = <PAD>

  return vocabulary

# Create the embedding matrix from a pretrained-embedding
def create_embedding (path, dictionary, embedding_size):
    embedding_vocab = {}
    with open (path, "r", newline='\n') as file:
      for line in file:
        temp = np.str.split(line)
        temp_float = []
        for t in temp[1:]:
          temp_float.append(float(t))
        #temp_float is a a numpy array of ['string', float, float, ... x100]
        temp_float = np.asarray(temp_float)
        #temp[0] is the "token"
        token = temp[0]

        #vocabulary of pretrained embeddings
        embedding_vocab [token] = np.asarray(temp_float)


    embedding_list = []
    for i,c in enumerate(dictionary):
        if (c=='<PAD>'):
            #Append array of zeros if <PAD>
            embedding_list.append(np.zeros(embedding_size))
        elif (c in embedding_vocab):
            #Append the corresponding embedding if the word is in sentences dictionary
            embedding_list.append(embedding_vocab[c])
        else:
            #Append a random embedding
            embedding_list.append(np.random.rand(embedding_size))

    embedding_matrix = torch.FloatTensor(np.asarray(embedding_list))
    return embedding_matrix

# Transform each token in the windows using the vocabulary previously
def encode_sentences_in_windows(windows, dictionary):
  encoding = []
  for window in windows:
    e = []
    for c in window:
      e.append(dictionary.get(c, dictionary['<UNK>'])) #If c is not found in dictionary, return value for 'UNK'
    encoding.append(e)
  return encoding

# Transform each albel in the windows using the vocabulary
def encode_labels_in_windows(windows, dictionary):
  encoding = []
  for window in windows:
    e = []
    for c in window:
      e.append(dictionary.get(c))
    encoding.append(e)
  return encoding

# Create a dataset for train and evaluation set
class NERDataset(Dataset):
    """An automatically generated dataset for our task."""

    def __init__(self, sentences_enc, labels_encoding, sentences_dictionary, labels_dictionary):
        self.data = []
        self.num_samples = len(sentences_enc)

        for i in range(len(sentences_enc)):
            window_sentence = sentences_enc[i]
            window_label = labels_encoding[i]
            item = {'inputs': torch.LongTensor(window_sentence), 'outputs': torch.Tensor(window_label)}
            self.data.append(item)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.data[idx]

# Transform each token in the sentences using the vocabulary
def encode_sentences(sentences, dictionary):
  encoding = []
  for sentence in sentences:
    e = []
    for token in sentence:
      e.append(dictionary.get(token, dictionary['<UNK>']))
    encoding.append(e)
  return encoding

# Transform each label in the sentences using the vocabulary
def encode_labels(sentences, dictionary):
  encoding = []
  for sentence in sentences:
    e = []
    for label in sentence:
      e.append(dictionary.get(label))
    encoding.append(e)
  return encoding
