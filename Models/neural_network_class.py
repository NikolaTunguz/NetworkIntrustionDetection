import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

class MyNN(nn.Module):
    def __init__(self, X, y):
        super(MyNN, self).__init__()

        #model variables
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        #variables
        self.X = X
        self.y = y

        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None

        self.train_loader = None
        self.test_loader = None

        #dimensionality
        input_dim = X.shape[1]
        output_dim = len(np.unique(y))
        hidden1 = 256
        hidden2 = 128
        hidden3 = 64

        #layer definitions
        self.fc1 = nn.Linear(input_dim, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, hidden3)
        self.fc4 = nn.Linear(hidden3, output_dim)

        self.relu = nn.ReLU()

        self.prepare_data()

    #forward pass through model
    def forward(self, x):
        output = self.fc1(x)
        output = self.relu(output)

        output = self.fc2(output)
        output = self.relu(output)

        output = self.fc3(output)
        output = self.relu(output)

        output = self.fc4(output)

        return output
        
    def prepare_data(self):
        #split 
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42) 

        #convert to tensors
        self.X_train = torch.tensor(self.X_train.to_numpy(), dtype=torch.float32)
        self.y_train = torch.tensor(self.y_train.to_numpy(), dtype=torch.long)
        self.X_test = torch.tensor(self.X_test.to_numpy(), dtype=torch.float32)
        self.y_test = torch.tensor(self.y_test.to_numpy(), dtype=torch.long)

        #place datasets in dataloader batches
        train_dataset = TensorDataset(self.X_train, self.y_train)
        test_dataset = TensorDataset(self.X_test, self.y_test)

        batch_size = 64
        self.train_loader = DataLoader(train_dataset, batch_size = batch_size)
        self.test_loader = DataLoader(test_dataset, batch_size = batch_size)

    def train_model(self):
        #hyperparameters
        num_epochs = 5
        learning_rate = 0.001
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.parameters(), lr = learning_rate)

        epoch_losses = []
        #training loop
        self.to(self.device)
        for epoch in range(num_epochs):
            #print(epoch)
            curr_loss = 0.0

            self.train()
            for features, labels in self.train_loader:
                features, labels = features.to(self.device), labels.to(self.device)

                optimizer.zero_grad()
                predictions = self(features)

                loss = criterion(predictions, labels)
                loss.backward()
                optimizer.step()

                curr_loss += loss.item()

            epoch_losses.append(curr_loss / len(self.train_loader))
        
        #plotting loss
        plt.plot(range(1, num_epochs + 1), epoch_losses, marker='o')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss Over Epochs')
        plt.show()

    def predict(self):
        self.eval()
        predictions = []

        self.to(self.device)
        with torch.no_grad():
            for features, labels in self.test_loader:
                features, labels = features.to(self.device), labels.to(self.device)

                #forward pass
                output = self(features)

                #prediction
                predicted_class = torch.argmax(output, dim = 1)
                predictions.append(predicted_class.cpu())

        #combining results from batches
        predictions = torch.cat(predictions).numpy()
        return predictions

    def evaluate(self):
        prediction = self.predict()

        accuracy = accuracy_score(self.y_test, prediction)
        cm = confusion_matrix(self.y_test, prediction)
        report = classification_report(self.y_test, prediction)

        return accuracy, cm, report


