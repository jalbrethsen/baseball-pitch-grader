import torch.optim as optim
import torch
import torch.nn as nn
import seaborn as sn
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm


class BaseballClassifierTrainer:
    def __init__(
        self,
        model,
        train_loader,
        test_loader,
        learning_rate=0.001,
        device="cuda",
        verbose=True,
    ):
        self.device = device
        self.verbose = verbose
        self.model = model.to(device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.CrossEntropyLoss()
        self.epochs = 0
        self.epoch = 0

    def train_step(self):
        self.model.train()
        ep_loss = 0
        num_batches = 1
        for batch_x, batch_y in tqdm(self.train_loader):
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.type(torch.LongTensor).to(self.device)
            self.optimizer.zero_grad()
            # forward pass
            outputs = self.model(batch_x)
            # calculate loss
            loss = self.criterion(outputs, batch_y)
            # backward pass
            loss.backward()
            # optimize
            self.optimizer.step()
            ep_loss += loss.item()
            num_batches += 1
        if self.verbose:
            print(
                f"Epoch [{self.epoch + 1}/{self.epochs}], loss: {ep_loss/num_batches}"
            )

    def evaluate(self):
        # Evaluate the model
        self.model.eval()
        with torch.no_grad():
            total = 0
            correct = 0
            pred = []
            y_test = []
            for batch_x, batch_y in tqdm(self.test_loader):
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.type(torch.LongTensor).to(self.device)
                outputs = nn.functional.softmax(self.model(batch_x), dim=1)
                predicted = torch.argmax(outputs.data, dim=1)
                pred.extend(predicted.cpu().detach().numpy())
                y_test.extend(batch_y.cpu().detach().numpy())
                total += len(batch_y)
                correct += (predicted == batch_y).sum().item()
            if self.verbose:
                print(f"Test Accuracy of the model is: {100*correct/total}")
                cm = confusion_matrix(y_test, pred)
                sn.heatmap(cm, cmap="viridis", annot=True, fmt="d")
                print(classification_report(y_test, pred))

    def train(self, epochs):
        self.epochs = epochs
        # Train the model
        for epoch in range(epochs):
            self.epoch = epoch
            self.train_step()
            if epoch % 10 == 0:
                self.evaluate()
