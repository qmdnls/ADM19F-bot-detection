import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optim
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sklearn.model_selection as model_selection
import sklearn.metrics as metrics
import pickle
from scipy import stats
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR

np.random.seed(0)

df = pd.read_csv('data/train_graph.csv', encoding='utf8', engine='python', chunksize=None)

# Define features and target
features = list(df.columns)
features.remove('label')
features = ['favourites_count', 'statuses_count', 'following', 'followers', 'favorites_predecessors', 'outdegree_predecessors', 'account_age', 'ego_reciprocity']
num_features = len(features)

# Standardize the feature data (mean 0, std 1)
standardize = lambda x: (x-x.mean()) / x.std()
for feature in features:
    df[feature] = df[feature].pipe(standardize)

x_train, x_test, y_train, y_test = model_selection.train_test_split(df[features], df['label'], test_size=0.2, shuffle=True, stratify=df['label'])

z = np.abs(stats.zscore(x_train))
x_train = x_train[(z < 3).all(axis=1)]
y_train = y_train[(z < 3).all(axis=1)]

print("Training set shape:", x_train.shape)

print(df[features].head())

x = torch.from_numpy(x_train.values).float()
y = torch.from_numpy(y_train.values).long()

x_test = torch.from_numpy(x_test.values).float()
y_test = torch.from_numpy(y_test.values).float()

class Net(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.dout = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(num_features, 500)
        self.pr1 = nn.PReLU()
        self.bn1 = nn.BatchNorm1d(500)
        self.fc2 = nn.Linear(500, 200)
        self.pr2 = nn.PReLU()
        self.bn2 = nn.BatchNorm1d(200)
        self.fc3 = nn.Linear(200, 200)
        self.pr3 = nn.PReLU()
        self.bn3 = nn.BatchNorm1d(200)
        self.out = nn.Linear(200, 1)

    def forward(self, input_):
        out = self.dout(self.bn1(self.pr1(self.fc1(input_))))
        out = self.dout(self.bn2(self.pr2(self.fc2(out))))
        out = self.dout(self.bn3(self.pr3(self.fc3(out))))
        out = self.out(out)
        return out

net = Net(num_features)
opt = optim.Adam(net.parameters(), lr=1e-3, betas=(0.9, 0.999), weight_decay=1e-4)
criterion = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([0.81]))

# gamma = decaying factor
#scheduler = StepLR(opt, step_size=400, gamma=0.975)

y_t = y_test.unsqueeze(dim=1)
def train_epoch(model, x, y, opt, criterion, batch_size=6000):
    model.train()
    losses = []
    valid_losses = []
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
            pred = model(x_test)
            valid_loss = criterion(pred, y_t)
            valid_losses.append(valid_loss)
            pred = torch.sigmoid(pred).round().squeeze(dim=1).detach().numpy()
            scores.append(metrics.f1_score(y_test, pred))
            model.train()
    
    #scheduler.step()
    return losses, valid_losses, scores

e_losses = []
v_losses = []
e_scores = []
num_epochs = 500

for e in range(num_epochs):
    # Progress
    print("Epoch", e, "of", num_epochs, end="\r", flush=True)
    # Train one poch and get losses, scores
    l, v_l, s = train_epoch(net, x, y, opt, criterion)
    e_losses += l
    v_losses += v_l
    e_scores += s

fig, (ax1, ax2) = plt.subplots(2)
ax1.plot(e_losses, label="train")
ax1.plot(v_losses, label="valid")
ax2.plot(e_scores, label="f1score")
ax1.legend()
ax2.legend()
plt.show()

# Evaluation
net.eval()
nn_prob = torch.sigmoid(net(x_test))
nn_pred = nn_prob.round()
nn_prob = nn_prob.squeeze(dim=1).detach().numpy()
nn_pred = nn_pred.squeeze(dim=1).detach().numpy()
print(nn_pred)
nn_fpr, nn_tpr, _ = metrics.roc_curve(y_test, nn_pred)
nn_fpr_array, nn_tpr_array, _ = metrics.roc_curve(y_test, nn_prob[:,1])
nn_auc = metrics.roc_auc_score(y_test, nn_prob)
print("NN:", "Acc:", round(metrics.accuracy_score(y_test, nn_pred), 4), "TPR:", round(nn_tpr[1], 4), "FPR:", round(nn_fpr[1], 4), "F1 score:", round(metrics.f1_score(y_test, nn_pred), 4), "AUC:", round(nn_auc, 4))

data = [nn_fpr_array, nn_tpr_array, nn_auc]

with open('data/roc_nn.data', 'wb') as filehandle:
    pickle.dump(data, filehandle)
