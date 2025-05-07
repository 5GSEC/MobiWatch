import numpy as np
import torch.nn as nn
import torch
import torch.optim as optim
import torch.utils.data as Data
import more_itertools

def se2rmse(a):
    return torch.sqrt(sum(a.t())/a.shape[1])


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# hyper params
feature_size = None # will be set based on the input data
hidden_len = 64
batch_size = 256
num_layer = 1
lr = 1e-3
weight_decay = 1e-5
epoches = 400
seq_len = 5

class LSTM_multivariate(nn.Module):
    def __init__(self):
        super(LSTM_multivariate, self).__init__()

        self.rnn = nn.LSTM(         
            input_size=feature_size,
            hidden_size=hidden_len,         # rnn hidden unit
            num_layers=num_layer,           # number of rnn layer
            batch_first=True,       # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
        )

        self.out = nn.Linear(hidden_len, feature_size)

    def forward(self, x):
        # x shape (batch, time_step, input_size)
        # r_out shape (batch, time_step, output_size)
        # h_n shape (n_layers, batch, hidden_size)
        # h_c shape (n_layers, batch, hidden_size)
        r_out, (h_n, h_c) = self.rnn(x, None)   # None represents zero initial hidden state

        # choose r_out at the last time step
        out = self.out(r_out[:, -1, :])
        return out

def get_loss_function():
    # MSE
    return nn.MSELoss()
    # CE
    # return nn.CrossEntropyLoss()


def train(X_train, y_train):
    global feature_size
    feature_size = y_train.shape[1]

    model = LSTM_multivariate().to(device)
    optimizier = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = get_loss_function()
    getMSEvec = nn.MSELoss(reduction='none')

    model.train()

    # X_train = more_itertools.windowed(train_data,n=seq_len,step=1)
    # X_train = np.asarray(list(X_train))
    # y_train = np.asarray(train_data[seq_len-1:])

    print("SHAPE",X_train.shape,y_train.shape)
    X_train = torch.from_numpy(X_train).type(torch.float).to(device)
    y_train = torch.from_numpy(y_train).type(torch.float).to(device)

    
    torch_dataset = Data.TensorDataset(X_train, y_train)
    loader = Data.DataLoader(
        dataset=torch_dataset,
        batch_size=batch_size,
        shuffle=True,
    )

    for epoch in range(epoches):
        for step, (batch_x, batch_y) in enumerate(loader):
            output = model(batch_x)
            loss = criterion(output, batch_y)
            optimizier.zero_grad()
            loss.backward()
            optimizier.step()
            # if epoch % 10 == 0 :
            #     print('epoch:{}/{}'.format(epoch,step), '|Loss:', loss.item())
        if epoch % 10 == 0 :
            print('epoch:{}'.format(epoch), '|Loss:', loss.item())
    
    model.eval()
    output = model(X_train)

    with torch.no_grad():
        mse_vec = torch.mean((output - y_train) ** 2, dim=1)
        mse_vec_unsort = mse_vec.numpy()
    print("max AD score",max(mse_vec))
    thres = max(mse_vec)
    mse_vec.sort()
    pctg = 0.99
    thres = mse_vec[int(len(mse_vec)*pctg)] + 0.001 # hack
    print("thres:",thres)
    return model, thres, mse_vec_unsort

    # mse_vec = getMSEvec(output,y_train)
    # rmse_vec = se2rmse(mse_vec).cpu().data.numpy()
    # rmse_vec_unsort = rmse_vec.copy()

    # print("max AD score",max(rmse_vec))
    # thres = max(rmse_vec)
    # rmse_vec.sort()
    # pctg = 0.99
    # thres = rmse_vec[int(len(rmse_vec)*pctg)]
    # print("thres:",thres)
    # return model, thres, rmse_vec_unsort
    

# @torch.no_grad()
def test(model, thres, X_test, y_test):
    global feature_size
    feature_size = y_test.shape[1]

    getMSEvec = nn.MSELoss(reduction='none')

    model.eval()
    # X_test = more_itertools.windowed(test_data,n=seq_len,step=1)
    # X_test = np.asarray(list(X_test))
    # y_test = np.asarray(test_data[seq_len-1:])
    # X_test = more_itertools.windowed(test_data,n=seq_len,step=1)
    # X_test = np.asarray(list(X_test)[:-1])
    # y_test = np.asarray(test_data[seq_len:])

    X_test = torch.from_numpy(X_test).type(torch.float).to(device)
    y_test = torch.from_numpy(y_test).type(torch.float).to(device)

    with torch.no_grad():
        output = model(X_test)
        # mse_vec = getMSEvec(output,y_test)
        # rmse_vec = se2rmse(mse_vec).cpu().data.numpy()
        mse_vec = torch.mean((output - y_test) ** 2, dim=1)

    # rmse_vec = np.concatenate((np.asarray([0.]*(seq_len-1)),rmse_vec))
    # idx_mal = np.where(rmse_vec>thres)
    # idx_ben = np.where(rmse_vec<=thres)
    # print(len(rmse_vec[idx_ben]),len(rmse_vec[idx_mal]))
    return mse_vec # rmse_vec


@torch.no_grad()
def test_from_iter(model, thres, X_test):
    model.eval()
    y_test = X_test[:,-1,:]
    # print("X_test",X_test.shape,"y_test",y_test.shape)
    X_test = torch.from_numpy(X_test).type(torch.float).to(device)
    y_test = torch.from_numpy(y_test).type(torch.float).to(device)

    output = model(X_test)
    # print("output",output.size(),"y_test",y_test.size())
    mse_vec = getMSEvec(output,y_test)
    rmse_vec = se2rmse(mse_vec).cpu().data.numpy()
    rmse_vec = np.concatenate((np.asarray([0.]*(seq_len-1)),rmse_vec))
    idx_mal = np.where(rmse_vec>thres)
    idx_ben = np.where(rmse_vec<=thres)
    # print(len(rmse_vec[idx_ben]),len(rmse_vec[idx_mal]))
    return rmse_vec


