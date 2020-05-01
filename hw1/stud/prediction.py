import numpy as np
import torch

# Return a vocabulary from number to label
def create_labels_reverse_vocabulary():
  labels = ['PER', 'LOC', 'ORG', 'O']
  vocabulary = {}
  for i,c in enumerate(labels):
    vocabulary[i] = c
  vocabulary.update({len(vocabulary.keys()): None})

  return vocabulary

# Predict the labels of each sentence of the list
# input/output: List[List[str]]
def predict_labels (sentences_enc, model, labels_reverse_vocabulary):
    sentences_pred = []
    model.eval() # NO Dropout
    for sentence in sentences_enc:
        pred = model(torch.LongTensor(sentence).unsqueeze(0)).tolist() #predict
        sentence_pred = []
        for label in pred:
            if ( np.argmax(label) != None):
                label = labels_reverse_vocabulary[np.argmax(label)]
            else:
                # Check if it predicts a "PAD", if so return a "O" as a label.
                # It never happens when the model is well trained: just for safety.
                label = 'O'
            sentence_pred.append(label)
        sentences_pred.append(sentence_pred)
    return sentences_pred
