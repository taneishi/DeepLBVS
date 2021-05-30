import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn import preprocessing
import argparse
import timeit
import os

#import intel_pytorch_extension as ipex
#ipex.enable_auto_mixed_precision(mixed_dtype = torch.bfloat16)

class MLP(nn.Module):
    def __init__(self, input_dim=1974, dropout=0.1):
        # 2000, 100, 2, relu, adam, lr=0.0001, dropout=0
        # 1000, 1000, 2, sigmoid, sgd, lr=0.01, momentum=0.9, mse
        super(MLP, self).__init__()
        self.input_layer = nn.Linear(input_dim, 3000)
        self.dropout1 = nn.Dropout(dropout)
        self.fc1 = nn.Linear(3000, 50)
        self.dropout2 = nn.Dropout(dropout)
        self.output_layer = nn.Linear(50, 2)
        
    def forward(self, x):
        x = F.relu(self.input_layer(x))
        x = self.dropout1(x)
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = torch.sigmoid(self.output_layer(x))
        return x

def load_dataset(args, device):
    start_time = timeit.default_timer()

    data = np.load(args.datafile)['data']
    data = preprocessing.minmax_scale(data)

    np.random.seed(args.random_seed)
    np.random.shuffle(data)

    train, test = train_test_split(data, test_size=args.test_size, random_state=args.random_seed)

    print('%d training, %d test samples.' % (train.shape[0], test.shape[0]))

    train_x, train_y = train[:, :-1], train[:, -1]
    test_x, test_y = test[:, :-1], test[:, -1]

    # create torch tensor from numpy array
    train_x = torch.FloatTensor(train_x).to(device)
    train_y = torch.LongTensor(train_y).to(device)
    test_x = torch.FloatTensor(test_x).to(device)
    test_y = torch.LongTensor(test_y).to(device)

    train = torch.utils.data.TensorDataset(train_x, train_y)
    test = torch.utils.data.TensorDataset(test_x, test_y)

    print('%5.2f sec for preprocessing.' % (timeit.default_timer() - start_time))

    train_dataloader = torch.utils.data.DataLoader(train, batch_size=args.batch_size, shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(test, batch_size=args.batch_size, shuffle=True)

    return train_dataloader, test_dataloader

def train(dataloader, net, optimizer, loss_func, epoch):
    net.train()
    train_loss = 0

    for index, (data, label) in enumerate(dataloader, 1):
        optimizer.zero_grad()
        output = net(data)
        loss = loss_func(output, label, reduction='mean')
        train_loss += loss.item()
        loss.backward()
        optimizer.step()

    print('epoch %4d batch %4d/%4d train_loss %6.3f' % (epoch, index, len(dataloader), train_loss / index), end='')

    return train_loss / index

def test(dataloader, net, loss_func):
    net.eval()
    test_loss = 0
    y_score, y_true = [], []

    for index, (data, label) in enumerate(dataloader, 1):
        with torch.no_grad():
            output = net(data)
        loss = loss_func(output, label, reduction='mean')
        test_loss += loss.item()
        y_score.append(output.cpu())
        y_true.append(label.cpu())

    y_score = np.concatenate(y_score)
    y_pred = [np.argmax(x) for x in y_score]
    y_true = np.concatenate(y_true)

    if np.sum(y_pred) != 0:
        acc = metrics.accuracy_score(y_true, y_pred)
        auc = metrics.roc_auc_score(y_true, y_score[:,1])
        prec = metrics.precision_score(y_true, y_pred)
        recall = metrics.recall_score(y_true, y_pred)

        confusion_matrix = metrics.confusion_matrix(y_true, y_pred).flatten()

        print(' %s test_loss %5.3f test_auc %5.3F test_prec %5.3f test_recall %5.3f' % (confusion_matrix, test_loss / index, auc, prec, recall), end='')

    else:
        print(' %s test_loss %5.3f' % (confusion_matrix, test_loss / index), end='')

    return test_loss / index

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    #device = ipex.DEVICE
    print('Using %s device.' % device)

    train_dataloader, test_dataloader = load_dataset(args, device)

    net = MLP(input_dim=1974, dropout=args.dropout).to(device)

    if args.modelfile:
        net.load_state_dict(torch.load(args.modelfile))

    # define our optimizer and loss function
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_func = F.cross_entropy

    test_losses = []

    for epoch in range(args.epochs):
        epoch_start = timeit.default_timer()

        train(train_dataloader, net, optimizer, loss_func, epoch)
        test_loss = test(test_dataloader, net, loss_func)

        print(' %5.2f sec' % (timeit.default_timer() - epoch_start))

        test_losses.append(test_loss)

        if test_loss <= min(test_losses):
            torch.save(net.state_dict(), os.path.join(args.root_dir, args.model_dir, '%5.3f.pth' % min(test_losses)))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--datafile', default=os.path.join(os.path.dirname(__file__), 'data', 'cpi.npz'), type=str)
    parser.add_argument('--root_dir', default=os.path.dirname(__file__), type=str)
    parser.add_argument('--random_seed', default=123, type=int)
    parser.add_argument('--test_size', default=0.2, type=float)
    parser.add_argument('--model_dir', default='model', type=str)
    parser.add_argument('--modelfile', default=None, type=str)
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--batch_size', default=100, type=int)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--weight_decay', default=0., type=float)
    parser.add_argument('--dropout', default=0.1, type=float)
    parser.add_argument('--cpu', action='store_true')

    args = parser.parse_args()
    print(vars(args))
    os.makedirs(os.path.join(args.root_dir, args.model_dir), exist_ok=True)

    main(args)
