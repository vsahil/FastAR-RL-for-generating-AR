import sys, os
import numpy as np
import torch as T
device = T.device("cpu")
from sklearn.metrics import accuracy_score
sys.path.append("../../")
output_dim = int(os.environ["OUTPUT_DIM"])    # 1 for Dice-Gradient and 2 for Dice-VAE
print(output_dim, "see output dim")


class BanknoteDataset(T.utils.data.Dataset):

  def __init__(self, src_file, num_rows=None):
    # import ipdb; ipdb.set_trace()
    all_data = np.loadtxt(src_file, max_rows=num_rows,
      usecols=range(1,6), delimiter="\t", skiprows=0,
      dtype=np.float32)  # strip IDs off

    self.x_data = T.tensor(all_data[:,0:4],
      dtype=T.float32).to(device)
    self.y_data = T.tensor(all_data[:,4],
      dtype=T.float32).to(device)

    # n_vals = len(self.y_data)
    # self.y_data = self.y_data.reshape(n_vals,1)
    self.y_data = self.y_data.reshape(-1, 1)

  def __len__(self):
    return len(self.x_data)

  def __getitem__(self, idx):
    if T.is_tensor(idx):
      idx = idx.tolist()
    preds = self.x_data[idx,:]  # idx rows, all 4 cols
    lbl = self.y_data[idx,:]    # idx rows, the 1 col
    sample = { 'predictors' : preds, 'target' : lbl }
    # sample = dict()   # or sample = {}
    # sample["predictors"] = preds
    # sample["target"] = lbl

    return sample

  def shape(self):
    print(self.x_data.shape)
    print(self.y_data.shape)


class GermanDataset(T.utils.data.Dataset):
  
  def __init__(self, dataset, labels):
    # all_data = np.loadtxt(src_file, max_rows=num_rows,
    #   usecols=range(1,6), delimiter="\t", skiprows=0,
    #   dtype=np.float32)  # strip IDs off

    # self.x_data = T.tensor(all_data[:,0:4],
    #   dtype=T.float32).to(device)
    # self.y_data = T.tensor(all_data[:,4],
    #   dtype=T.float32).to(device)

    # n_vals = len(self.y_data)
    # self.y_data = self.y_data.reshape(n_vals,1)
    self.x_data = T.tensor(dataset, dtype=T.float32).to(device)
    self.y_data = T.tensor(labels, dtype=T.float32).to(device)
    target = self.y_data.long()
    if output_dim == 1:
      self.y_data = self.y_data.reshape(-1, 1)
    elif output_dim == 2:
      self.y_data = target.reshape(-1, 1)
      # self.y_data = T.nn.functional.one_hot(target)

  def __len__(self):
    return len(self.x_data)

  def __getitem__(self, idx):
    if T.is_tensor(idx):
      idx = idx.tolist()
    # import ipdb; ipdb.set_trace()
    preds = self.x_data[idx,:]  # idx rows, all 4 cols
    lbl = self.y_data[idx,:]    # idx rows, the 1 col
    sample = { 'predictors' : preds, 'target' : lbl }
    # sample = dict()   # or sample = {}
    # sample["predictors"] = preds
    # sample["target"] = lbl
    return sample
  
  def shape(self):
    print(self.x_data.shape)
    print(self.y_data.shape)
  

# ---------------------------------------------------------

def accuracy(model, ds):
  # ds is a iterable Dataset of Tensors
  n_correct = 0; n_wrong = 0

  # alt: create DataLoader and then enumerate it
  for i in range(len(ds)):
    inpts = ds[i]['predictors']
    target = ds[i]['target']    # float32  [0.0] or [1.0]
    with T.no_grad():
      oupt = model(inpts)

    # avoid 'target == 1.0'
    if target < 0.5 and oupt < 0.5:  # .item() not needed
      n_correct += 1
    elif target >= 0.5 and oupt >= 0.5:
      n_correct += 1
    else:
      n_wrong += 1

  return (n_correct * 1.0) / (n_correct + n_wrong)

# ---------------------------------------------------------

def acc_coarse(model, ds):
  inpts = ds[:]['predictors']  # all rows
  targets = ds[:]['target']    # all target 0s and 1s
  with T.no_grad():
    oupts = model(inpts)         # all computed ouputs
  pred_y = oupts >= 0.5        # tensor of 0s and 1s
  num_correct = T.sum(targets==pred_y)
  acc = (num_correct.item() * 1.0 / len(ds))  # scalar
  return acc

# ----------------------------------------------------------

def my_bce(model, batch):
  # mean binary cross entropy error. somewhat slow
  sum = 0.0
  inpts = batch['predictors']
  targets = batch['target']
  with T.no_grad():
    oupts = model(inpts)
  for i in range(len(inpts)):
    oupt = oupts[i]
    # should prevent log(0) which is -infinity
    if targets[i] >= 0.5:  # avoiding == 1.0
      sum += T.log(oupt)
    else:
      sum += T.log(1 - oupt)

  return -sum / len(inpts)

# ----------------------------------------------------------

class Net(T.nn.Module):
  def __init__(self, input_dim, hidden_dim1, hidden_dim2, output_dim):
    super(Net, self).__init__()
    # self.hid1 = T.nn.Linear(4, 8)  # 4-(8-8)-1
    # self.hid2 = T.nn.Linear(8, 8)
    # self.oupt = T.nn.Linear(8, 1)
    self.output_dim = output_dim
    self.hid1 = T.nn.Linear(input_dim, hidden_dim1)  # 4-(8-8)-1
    self.hid2 = T.nn.Linear(hidden_dim1, hidden_dim2)
    self.oupt = T.nn.Linear(hidden_dim2, output_dim)

    T.nn.init.xavier_uniform_(self.hid1.weight) 
    T.nn.init.zeros_(self.hid1.bias)
    T.nn.init.xavier_uniform_(self.hid2.weight) 
    T.nn.init.zeros_(self.hid2.bias)
    T.nn.init.xavier_uniform_(self.oupt.weight) 
    T.nn.init.zeros_(self.oupt.bias)

  def forward(self, x):
    # z = T.tanh(self.hid1(x))
    # z = T.tanh(self.hid2(z))
    z = T.relu(self.hid1(x))
    z = T.relu(self.hid2(z))
    if self.output_dim == 1:
      z = T.sigmoid(self.oupt(z))
    elif self.output_dim == 2:
      z = T.nn.functional.log_softmax(self.oupt(z), dim=1)    # need to use NLL loss with softmax. 
    return z

  def orig_predict(self, dt):
    if isinstance(dt, np.ndarray):
      dt = T.tensor(dt).float()
    return self.forward(dt).detach().numpy()

# ----------------------------------------------------------

def train(batch_size, train_ldr, net, train_ds, test_ds):
    # 3. train network
    print("\nPreparing training")
    net = net.train()
    # set training mode
    lrn_rate = 0.001 
    if output_dim == 1:
      loss_obj = T.nn.BCELoss()
    # binary cross entropy
    elif output_dim == 2:
      # loss_obj = T.nn.CrossEntropyLoss()
      loss_obj = T.nn.NLLLoss()
    # cross entropy
    optimizer = T.optim.Adam(net.parameters(), lr=lrn_rate)
    max_epochs = 200
    ep_log_interval = 10
    print("Loss function: " + str(loss_obj))
    print("Optimizer: SGD")
    print(f"Learn rate: {lrn_rate}")
    print(f"Batch size: {batch_size}")
    print("Max epochs: " + str(max_epochs))
    print("\nStarting training")
    # import ipdb; ipdb.set_trace()
    for epoch in range(0, max_epochs):
      epoch_loss = 0.0
      # for one full epoch
      num_lines_read = 0
      for (batch_idx, batch) in enumerate(train_ldr):
        X = batch['predictors']
        # [10,4]  inputs
        Y = batch['target']
        # [10,1]  targets
        oupt = net(X)
        # [10,1]  computeds 
        if output_dim == 2:
          Y = Y.squeeze()
        loss_val = loss_obj(oupt, Y)
        # a tensor
        epoch_loss += loss_val.item()
        loss_val.backward()
        # compute all gradients
        optimizer.step()
      # update all weights
      
      if epoch % ep_log_interval == 0:
        print("epoch = %4d   loss = %0.4f" % \
        (epoch, epoch_loss))
        acc_test = evaluate(train_ds, test_ds, net)
        if acc_test > 75:
          if output_dim == 1:
            path = f"saved_models/{dataset_name}_model_{int(acc_test)}.pth"
          elif output_dim == 2:
            path = f"saved_models/{dataset_name}_model_softmax_{int(acc_test)}.pth"
          T.save(net.state_dict(), path)
        net = net.train()
    return net


def evaluate(train_ds, test_ds, net):
    # 4. evaluate model
    net = net.eval()
    if output_dim == 1:
      predicted_train = net(train_ds.x_data).detach().numpy().squeeze()
      predicted_test = net(test_ds.x_data).detach().numpy().squeeze()
      actual_train = train_ds.y_data.numpy().squeeze()
      acc_train = accuracy_score(actual_train, (predicted_train > 0.5).astype(float)) * 100
      actual_test = test_ds.y_data.numpy().squeeze()
      acc_test = accuracy_score(actual_test, (predicted_test > 0.5).astype(float)) * 100
      # acc_train = accuracy(net, train_ds) * 100,    
      # acc_test = accuracy(net, test_ds) * 100
    elif output_dim == 2:
      predicted_train = T.argmax(net(train_ds.x_data), axis=1)
      actual_train = train_ds.y_data.squeeze()
      acc_train = accuracy_score(actual_train, predicted_train) * 100
      predicted_test = T.argmax(net(test_ds.x_data), axis=1)
      actual_test = test_ds.y_data.squeeze()
      acc_test = accuracy_score(actual_test, predicted_test) * 100
      # print(predicted_test)
    # print(predicted_train.shape, predicted_test.shape)
    print(f"Accuracy on train data = {round(acc_train, 3)}")
    print(f"Accuracy on test data = {round(acc_test, 3)}")
    
    net.train()
    return acc_test


def main(dataset_name):
  # 0. get started
  T.manual_seed(1)
  np.random.seed(1)

  # 1. create Dataset and DataLoader objects
  # print("Creating Banknote train and test DataLoader ")
  if dataset_name == "bank":
    train_file = "banknote_k20_train.txt"
    test_file = "banknote_k20_test.txt"

    train_ds = BanknoteDataset(train_file)  # all rows
    test_ds = BanknoteDataset(test_file)

  elif dataset_name == "german":
    import classifier_dataset
    file1 = f"{os.path.dirname(os.path.realpath(__file__))}/../../datasets/german_redone.csv"
    _, dataset, scaler1, X_test, X_train, y_train, y_test = classifier_dataset.train_model_german(file=file1, parameter=2, drop=False)
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler(feature_range=(0, 1))   # our scaler is between 0 and 1 as required by DiCE. 
    scaler.fit(X_train)
    assert X_train.shape[0] == y_train.shape[0]
    assert X_test.shape[0] == y_test.shape[0]
    assert dataset.shape[0] == X_train.shape[0] + X_test.shape[0]
    train_ds = GermanDataset(scaler.transform(X_train), y_train.to_numpy())  # all rows
    test_ds = GermanDataset(scaler.transform(X_test), y_test.to_numpy())

  elif dataset_name == "adult":
    import classifier_dataset
    file1 = f"{os.path.dirname(os.path.realpath(__file__))}/../../datasets/adult_redone.csv"
    dataset, scaler1, X_test, X_train, y_train, y_test = classifier_dataset.train_model_adult(file=file1, parameter=3, drop=False)
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(X_train)
    assert X_train.shape[0] == y_train.shape[0]
    assert X_test.shape[0] == y_test.shape[0]
    assert dataset.shape[0] == X_train.shape[0] + X_test.shape[0]
    train_ds = GermanDataset(scaler.transform(X_train), y_train.to_numpy())  # all rows
    test_ds = GermanDataset(scaler.transform(X_test), y_test.to_numpy())

  elif dataset_name == "default":
    import classifier_dataset
    file1 = f"{os.path.dirname(os.path.realpath(__file__))}/../../datasets/default_redone.csv"
    dataset, scaler1, X_test, X_train, y_train, y_test = classifier_dataset.train_model_default(file=file1, parameter=3, drop=False)
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(X_train)
    assert X_train.shape[0] == y_train.shape[0]
    assert X_test.shape[0] == y_test.shape[0]
    assert dataset.shape[0] == X_train.shape[0] + X_test.shape[0]
    train_ds = GermanDataset(scaler.transform(X_train), y_train.to_numpy())  # all rows
    test_ds = GermanDataset(scaler.transform(X_test), y_test.to_numpy())
  
  print(train_ds.shape(), test_ds.shape())

  batch_size = 64
  train_ldr = T.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)

  input_dim = train_ds.x_data.shape[1]
  hidden_dim1 = 5
  hidden_dim2 = 3

  net = Net(input_dim, hidden_dim1, hidden_dim2, output_dim).to(device)
  train_ = True
  
  if train_ == True:
    net = train(batch_size, train_ldr, net, train_ds, test_ds)
  else:
    if dataset_name == "german":
      if output_dim == 1:
          saved_model_path = "saved_models/german_model_76.pth"
      elif output_dim == 2:
          saved_model_path = "saved_models/german_model_softmax_76.pth"
    elif dataset_name == "adult":
        saved_model_path = "saved_models/adult_model_75.pth"
    elif dataset_name == "default":
        saved_model_path = "saved_models/default_model_78.pth"
    net.load_state_dict(T.load(saved_model_path))
    
  # if dataset_name == "default":
  acc_test = evaluate(train_ds, test_ds, net)

  # 5. save model
  if dataset_name == "adult" and acc_test < 80:
    print("Not saving model")
    return
  
  print("\nSaving trained model state_dict \n")
  if output_dim == 1:
    path = f"saved_models/{dataset_name}_model_{int(acc_test)}.pth"
  elif output_dim == 2:
    path = f"saved_models/{dataset_name}_model_softmax_{int(acc_test)}.pth"
  T.save(net.state_dict(), path)


if __name__== "__main__":
  dataset_name = sys.argv[1]
  assert dataset_name in ["german", "adult", "default"]
  main(dataset_name)
