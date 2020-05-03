import numpy as np
from typing import List, Tuple

from model import Model
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


from stud.my_model import MyModel
from stud.preprocessing import *
from stud.prediction import *

import numpy as np

def build_model(device: str) -> Model:
    # STUDENT: return StudentModel()
    # STUDENT: your model MUST be loaded on the device "device" indicates
    return StudentModel(device)


class StudentModel(Model):

    # STUDENT: construct here your model
    # this class should be loading your weights and vocabulary

    def __init__(self, device):
        #Initialize the instance of MyModel
        embedding_layer = torch.load("model/best_embedding_layer.pt")
        self.model = MyModel(
            input_size=100843,
            output_size=5,
            embedding_layer=embedding_layer)

        self.model.to(device)


        #Load the weights so we can use in prediction mode
        self.model.load_state_dict(torch.load("model/best_model.pt"))
        self.sentences_vocabulary = torch.load("model/sentences_vocabulary.pt")


    def predict(self, tokens: List[List[str]]) -> List[List[str]]:
        # STUDENT: implement here your predict function
        # remember to respect the same order of tokens!

        #Preprocessing
        test_sentences_enc = encode_sentences(tokens, self.sentences_vocabulary)
        #Prediction
        labels_reverse_vocabulary = create_labels_reverse_vocabulary()
        predictions = predict_labels(test_sentences_enc, self.model, labels_reverse_vocabulary)

        return predictions
