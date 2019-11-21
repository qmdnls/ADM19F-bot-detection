import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optim
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sklearn.metrics as metrics
from torch.autograd import Variable

np.random.seed(0)

df = pd.read_csv('data/train.csv', encoding='utf8', engine='python', chunksize=None)

# Use 75% of entries as training data
df['is_train'] = np.random.uniform(0, 1, len(df)) <= 0.75
train, test = df[df['is_train']==True], df[df['is_train']==False]

# Define features and target
features = list(df.columns)
features.remove('label')
features.remove('is_train')
y_train = train['label']

x = torch.from_numpy(train[features].values).float()
y = torch.from_numpy(train['label'].values).long()

x_test = torch.from_numpy(test[features].values).float()
y_test = torch.from_numpy(test['label'].values).long()

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.dout = nn.Dropout(0.05)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(9, 500)
        self.fc2 = nn.Linear(500, 200)
        self.fc3 = nn.Linear(200, 200)
        self.out = nn.Linear(200, 2)

    def forward(self, input_):
        out = self.dout(self.relu(self.fc1(input_)))
        out = self.dout(self.relu(self.fc2(out)))
        out = self.relu(self.fc3(out))
        out = self.out(out)
        return out

net = Net()
opt = optim.Adam(net.parameters(), lr=0.0001, betas=(0.9, 0.999))
criterion = nn.CrossEntropyLoss()

def train_epoch(model, x, y, opt, criterion, batch_size=50):
    model.train()
    losses = []
    scores = []
    for beg_i in range(0, x.size(0), batch_size):
        x_batch = x[beg_i:beg_i + batch_size, :]
        y_batch = y[beg_i:beg_i + batch_size]
        x_batch = Variable(x_batch)
        y_batch = Variable(y_batch)

        opt.zero_grad()
        # (1) Forward
        y_hat = model(x_batch)
        # (2) Compute diff
        loss = criterion(y_hat, y_batch)
        # (3) Compute gradients
        loss.backward()
        # (4) update weights
        opt.step()
        losses.append(loss.data.numpy())

    # Validation set
    model.eval()
    pred_prob = f.softmax(model(x_test), dim=1)
    nn_pred = torch.argmax(pred_prob, dim=1)
    nn_pred = nn_pred.detach().numpy()
    scores.append(metrics.f1_score(y_test, nn_pred))
    net.train()

    return losses, scores

e_losses = []
e_scores = []
num_epochs = 1300

for e in range(num_epochs):
    # Progress
    print("Epoch", e, "of", num_epochs, end="\r", flush=True)
    # Get losses, scores
    l, s = train_epoch(net, x, y, opt, criterion)
    e_losses += l
    e_scores += s

#plt.plot(e_losses)
plt.plot(e_scores)
plt.show()

# Evaluation
net.eval()
pred_prob = f.softmax(net(x_test), dim=1)
nn_pred = torch.argmax(pred_prob, dim=1)
nn_pred = nn_pred.detach().numpy()
print(nn_pred)
nn_fpr, nn_tpr, _ = metrics.roc_curve(y_test, nn_pred)
nn_auc = metrics.auc(nn_fpr, nn_tpr)
print("NN:", "TPR:", round(nn_tpr[1], 4), "FPR:", round(nn_fpr[1], 4), "F1 score:", round(metrics.f1_score(y_test, nn_pred), 4), "AUC:", round(nn_auc, 4))
