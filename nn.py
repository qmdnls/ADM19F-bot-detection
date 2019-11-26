import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optim
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sklearn.model_selection as model_selection
import sklearn.metrics as metrics
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR

np.random.seed(0)

df = pd.read_csv('data/train_baseline.csv', encoding='utf8', engine='python', chunksize=None)

# Define features and target
features = list(df.columns)
features.remove('label')

# Standardize the feature data (mean 0, std 1)
standardize = lambda x: (x-x.mean()) / x.std()
for feature in features:
    df[feature] = df[feature].pipe(standardize)

x_train, x_test, y_train, y_test = model_selection.train_test_split(df[features], df['label'], test_size=0.25, shuffle=True, stratify=df['label'])

print(df[features].head())

x = torch.from_numpy(x_train.values).float()
y = torch.from_numpy(y_train.values).long()

x_test = torch.from_numpy(x_test.values).float()
y_test = torch.from_numpy(y_test.values).float()

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        #self.relu = nn.ReLU()
        self.dout = nn.Dropout(p=0.05)
        self.fc1 = nn.Linear(9, 500)
        self.pr1 = nn.PReLU()
        self.fc2 = nn.Linear(500, 200)
        self.pr2 = nn.PReLU()
        self.fc3 = nn.Linear(200, 200)
        self.pr3 = nn.PReLU()
        self.out = nn.Linear(200, 1)

    def forward(self, input_):
        out = self.dout(self.pr1(self.fc1(input_)))
        out = self.dout(self.pr2(self.fc2(out)))
        out = self.pr3(self.fc3(out))
        out = self.out(out)
        return out

net = Net()
opt = optim.Adam(net.parameters(), lr=1e-3, betas=(0.9, 0.999))
#opt = torch.optim.SGD(net.parameters(), lr = 0.01, momentum=0.9, nesterov=True)
criterion = nn.BCEWithLogitsLoss()

y_t = y_test.unsqueeze(dim=1)
def train_epoch(model, x, y, opt, criterion, batch_size=32):
    model.train()
    losses = []
    scores = []
    y = torch.unsqueeze(y,dim=1).float()
    for beg_i in range(0, x.size(0), batch_size):
        x_batch = x[beg_i:beg_i + batch_size, :]
        y_batch = y[beg_i:beg_i + batch_size, :]
        x_batch = Variable(x_batch)
        y_batch = Variable(y_batch)
        # (0) Zero gradients
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
        with torch.no_grad():
            model.eval()
            valid_loss = criterion(model(x_test), y_t)
            scores.append(valid_loss)
            model.train()
    
    return losses, scores

e_losses = []
e_scores = []
num_epochs = 400

for e in range(num_epochs):
    # Progress
    print("Epoch", e, "of", num_epochs, end="\r", flush=True)
    # Train one poch and get losses, scores
    l, s = train_epoch(net, x, y, opt, criterion)
    e_losses += l
    e_scores += s

#fig, axs = plt.subplots(2)
#axs[0].plot(e_losses)
#axs[1].plot(e_scores)
plt.plot(e_losses, label="train")
plt.plot(e_scores, label="valid")
plt.legend()
plt.show()

# Evaluation
net.eval()
nn_pred = torch.sigmoid(net(x_test))
#nn_pred = torch.argmax(pred_prob, dim=1)
nn_pred = nn_pred.round()
nn_pred = nn_pred.squeeze(dim=1).detach().numpy()
print(nn_pred)
nn_fpr, nn_tpr, _ = metrics.roc_curve(y_test, nn_pred)
nn_auc = metrics.auc(nn_fpr, nn_tpr)
print("NN:", "Acc:", round(metrics.accuracy_score(y_test, nn_pred), 4), "TPR:", round(nn_tpr[1], 4), "FPR:", round(nn_fpr[1], 4), "F1 score:", round(metrics.f1_score(y_test, nn_pred), 4), "AUC:", round(nn_auc, 4))
