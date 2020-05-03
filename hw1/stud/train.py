import torch
import torch.optim as optim

#TQDM is a A Fast, Extensible Progress Bar for Python and CLI https://tqdm.github.io
from tqdm import tqdm
import time
import matplotlib.pyplot as plt
import numpy as np

#Class to train and evaluate the model
class Trainer():

    def __init__(
        self,
        model,
        loss_function,
        optimizer):

        self.model = model
        self.loss_function = loss_function
        self.optimizer = optimizer

    #Plot train and validation loss
    def plotLearning (self, N, train_loss_list, val_loss_list):
        x = [i for i in range(N)]

        plt.ylabel('Loss')
        plt.xlabel('Epochs')


        plt.plot(x, train_loss_list, 'b-', label='train_loss')
        plt.plot(x, val_loss_list, 'c-', label='val_loss')
        plt.legend(loc="upper right")
        plt.plot()
        #plt.savefig(filename)

    def train(self, train_dataset, valid_dataset, epochs=1, print_step=False):
        """
        Args:
            train_dataset: a Dataset or DatasetLoader instance containing
                the training instances.
            valid_dataset: a Dataset or DatasetLoader instance used to evaluate
                learning progress.
            epochs: the number of times to iterate over train_dataset.
            print_step: True will print all the steps into each epoch
        Returns:
            avg_train_loss: the average training loss on train_dataset over
                epochs.
        """
        train_loss_list = []
        val_loss_list = []


        for epoch in tqdm(range(epochs)):  # loop over the dataset multiple times
            time.sleep(0.1)
            epoch_loss = 0
            self.model.train() #Train mode: with dropout
            for step, sample in enumerate(train_dataset):
                inputs = sample['inputs']
                labels = sample['outputs']

                # we need to set the gradients to zero before starting to do backpropragation
                # because PyTorch accumulates the gradients on subsequent backward passes
                self.optimizer.zero_grad()

                predictions = self.model(inputs)

                '''
                print ("Input shape: ",inputs.shape)
                print ("Predictions shape: ", predictions.shape)
                print ("Predictions[0]: ",predictions[0])
                print ("Labels shape: ", labels.shape)
                print ("Labels[0]: ",labels[0])
                '''
                loss = self.loss_function(predictions, labels.view(-1).long())

                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()
                #Print at each step the epoch loss
                if (print_step):
                    print('[%d, %5d] loss: %.3f' %
                        (epoch + 1, step + 1, epoch_loss/(step+1) ))


            avg_epoch_loss = epoch_loss / len(train_dataset)
            val_loss= self.evaluate(valid_dataset)

            #Instantiated for plotLearning
            train_loss_list.append(avg_epoch_loss)
            val_loss_list.append(val_loss)

            print ("\nepoch: ", epoch+1, "avg_epoch_loss: ", avg_epoch_loss,"val_loss: ", val_loss)
        print('Finished Training')

        self.plotLearning(epochs, train_loss_list, val_loss_list)

    def evaluate(self, valid_dataset):
        """
        Args:
            valid_dataset: the dataset to use to evaluate the model.
        Returns:
            avg_valid_loss: the average validation loss over valid_dataset.
        """
        valid_loss = 0.0

        self.model.eval() #Evaluation mode: no dropout

        # no gradient updates here
        with torch.no_grad():
            for sample in valid_dataset:
                inputs = sample['inputs']
                labels = sample['outputs']

                predictions = self.model.eval()(inputs)

                loss = self.loss_function(predictions, labels.view(-1).long())
                valid_loss += loss.item()

        return valid_loss / len(valid_dataset)
