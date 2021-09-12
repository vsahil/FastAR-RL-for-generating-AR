import numpy as np
import pandas as pd
import sys, os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import accuracy_score
from sklearn.utils import resample

sigmoid = True
smaller = True
balanced = False
output_dim = int(os.environ["OUTPUT_DIM"])      # 1 for Dice-Gradient and 2 for Dice-VAE
print(output_dim, "see output dim")


class trainData(Dataset):
    
    def __init__(self, X_data, y_data):
        self.X_data = X_data
        self.y_data = y_data
        
    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]
        
    def __len__ (self):
        return len(self.X_data)


class testData(Dataset):
    
    def __init__(self, X_data):
        self.X_data = X_data
        
    def __getitem__(self, index):
        return self.X_data[index]
        
    def __len__ (self):
        return len(self.X_data)


class binaryClassification(nn.Module):
    def __init__(self, input_dim):
        super(binaryClassification, self).__init__()
        if not smaller:
            self.layer_1 = nn.Linear(input_dim, 64) 
            self.layer_2 = nn.Linear(64, 64)
            self.layer_out = nn.Linear(64, output_dim) 
            self.batchnorm1 = nn.BatchNorm1d(64)
            self.batchnorm2 = nn.BatchNorm1d(64)
        else:
            self.layer_1 = nn.Linear(input_dim, 5) 
            self.layer_2 = nn.Linear(5, 3)
            self.layer_out = nn.Linear(3, output_dim) 
            self.batchnorm1 = nn.BatchNorm1d(5)
            self.batchnorm2 = nn.BatchNorm1d(3)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, inputs):
        # if not from_here:   # someone else calling it.
        #     pred_value = self.orig_predict(inputs)
        #     # print("someone else", inputs, pred_value)
        #     return pred_value
        if inputs.ndim == 1:
            inputs = inputs.reshape(1, -1)
        # print(inputs.shape, "See")
        x = self.relu(self.layer_1(inputs))
        x = self.batchnorm1(x)
        x = self.relu(self.layer_2(x))
        x = self.batchnorm2(x)
        x = self.dropout(x)
        x = self.layer_out(x)
        if output_dim == 1:
            if sigmoid:
                x = self.sigmoid(x)
        elif output_dim == 2:
            x = nn.functional.log_softmax(x, dim=1)    # need to use NLL loss with softmax. 
        return x


    def orig_predict(self, dt):
        if isinstance(dt, np.ndarray):
            dt = torch.FloatTensor(dt)
        if dt.ndim == 1:
            dt = dt.reshape(1, -1)
        y_test_pred = self.forward(dt)
        if not sigmoid:
            y_test_pred = torch.sigmoid(y_test_pred)
        if output_dim == 1:
            y_pred_tag = torch.round(y_test_pred)
        elif output_dim == 2:
            y_pred_tag = y_test_pred
        return y_pred_tag.detach()


def binary_acc(y_pred, y_test):
    if not sigmoid:
        y_pred = torch.sigmoid(y_pred)

    if output_dim == 1:
        y_pred_tag = torch.round(y_pred)
        correct_results_sum = (y_pred_tag == y_test).sum().float()
    elif output_dim == 2:
        y_pred_tag = torch.argmax(y_pred, dim=1)
        correct_results_sum = (y_pred_tag == y_test).sum().float()
    acc = correct_results_sum/y_test.shape[0]
    acc = torch.round(acc * 100)
    
    return acc


EPOCHS = 20
BATCH_SIZE = 64
LEARNING_RATE = 0.001

def training(model, train_loader, optimizer, criterion, test_loader, device, y_test, dataset_name):
    model.train()
    best_acc_test = 0
    for e in range(1, EPOCHS+1):
        epoch_loss = 0
        epoch_acc = 0
        # import ipdb; ipdb.set_trace()
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            
            y_pred = model(X_batch)
            if output_dim == 1:
                loss = criterion(y_pred, y_batch.unsqueeze(1))
                acc = binary_acc(y_pred, y_batch.unsqueeze(1))
            if output_dim == 2:
                loss = criterion(y_pred, y_batch)
                acc = binary_acc(y_pred, y_batch) 
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            epoch_acc += acc.item()

        acc_test, _ = evaluate(model, test_loader, device, y_test)
        print(f'Epoch {e+0:03}: | Loss: {epoch_loss/len(train_loader):.5f} | Train Acc: {epoch_acc/len(train_loader):.3f} | Test Acc: {acc_test:.3f}')
        if acc_test > best_acc_test:
            best_acc_test = acc_test
            print("Saving trained model state_dict")
            path = f"saved_models/{dataset_name}"
            if sigmoid:
                path += "_sigmoid"
            if smaller:
                path += "_smallest"
            if balanced:
                path += "_balanced"
            else:
                path += "_unbalanced"
            if output_dim == 2:
                path += "_softmax"
            path += str(int(acc_test)) + ".pth"
            # path = dataset_name + "_sigmoid" if sigmoid else '' +  if balanced else '_unbalanced' + "_model_" + str(int(acc_test)) + ".pth"
            print(path)
            torch.save(model.state_dict(), path)


def evaluate(model, test_loader, device, y_test):
    y_pred_list = []
    model.eval()
    with torch.no_grad():
        for X_batch in test_loader:
            X_batch = X_batch.to(device)
            y_test_pred = model(X_batch)    
            if not sigmoid:
                y_test_pred = torch.sigmoid(y_test_pred)
            if output_dim == 1:
                y_pred_tag = torch.round(y_test_pred).cpu().numpy()
            elif output_dim == 2:
                y_pred_tag = torch.argmax(y_test_pred, dim=1).cpu().numpy()
            y_check = model.orig_predict(X_batch).cpu().numpy()
            assert y_pred_tag == y_check, f"{y_check, y_pred_tag}"
            y_pred_list.append(y_pred_tag)
    y_pred_list = [a.squeeze().tolist() for a in y_pred_list]
    # import ipdb; ipdb.set_trace()
    acc_test = accuracy_score(y_test, y_pred_list) * 100
    model.train()
    return acc_test, y_pred_list


def main(dataset_name, scaling):

    sys.path.append("../../")
    import classifier_dataset
    
    if dataset_name == "adult":
        file1 = f"{os.path.dirname(os.path.realpath(__file__))}/../../datasets/adult_redone.csv"
        dataset_old, _, X_test_old, X_train_old, y_train_old, y_test_old = classifier_dataset.train_model_adult(file=file1, parameter=3, drop=False)

    elif dataset_name == "default":
        file1 = f"{os.path.dirname(os.path.realpath(__file__))}/../../datasets/default_redone.csv"
        dataset_old, _, X_test_old, X_train_old, y_train_old, y_test_old = classifier_dataset.train_model_default(file=file1, parameter=3, drop=False)

    if balanced:
        dataset = pd.read_csv(file1)
        df_majority = dataset[dataset.target==0]
        df_minority = dataset[dataset.target==1]
        
        # Downsample majority class
        df_majority_downsampled = resample(df_majority, replace=False,    # sample without replacement
                                        n_samples=df_minority.shape[0],     # to match minority class
                                        random_state=123) # reproducible results
        
        # Combine minority class with downsampled majority class
        df_downsampled = pd.concat([df_majority_downsampled, df_minority])
        drop_ = ['target']
        X = df_downsampled.drop(columns=[*drop_])
        X_train, X_test, y_train, y_test = train_test_split(X, df_downsampled['target'], test_size=0.20, random_state=123)
        y_train = y_train.to_numpy()
        y_test = y_test.to_numpy()

    else: 
        X_test = X_test_old
        X_train = X_train_old
        y_train = y_train_old.to_numpy()
        y_test = y_test_old.to_numpy()

    if scaling == "min":
        scaler = MinMaxScaler(feature_range=(0, 1))
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
    elif scaling == "mean":
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    assert X_train.shape[0] == y_train.shape[0]
    assert X_test.shape[0] == y_test.shape[0]
    # print(X_train.shape, X_test.shape, "sEE the shapes")
    
    train_data = trainData(torch.FloatTensor(X_train), torch.LongTensor(y_train))
    train_data_pred = testData(torch.FloatTensor(X_train))
    test_data = testData(torch.FloatTensor(X_test))

    train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
    train_loader_pred = DataLoader(dataset=train_data_pred, batch_size=1)
    test_loader = DataLoader(dataset=test_data, batch_size=1)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    model = binaryClassification(input_dim = X_train.shape[1])
    model.to(device)

    if output_dim == 1:
        if not sigmoid:
            criterion = nn.BCEWithLogitsLoss()
        else:
            criterion = nn.BCELoss()
    elif output_dim == 2:
        criterion = nn.NLLLoss()

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train = True
    if train:
        training(model, train_loader, optimizer, criterion, test_loader, device, y_test, dataset_name)
    else:
        if dataset_name == "adult":
            saved_model_path = "saved_models/adult_sigmoid_smallest_unbalanced83.pth"
        elif dataset_name == "default":
            saved_model_path = "saved_models/default_sigmoid_smallest_unbalanced83.pth"
        print("Loading saved model...")
        model.load_state_dict(torch.load(saved_model_path))
        acc_train, trainy_pred_list = evaluate(model, train_loader_pred, device, y_train)
        acc_test, testy_pred_list = evaluate(model, test_loader, device, y_test)
        print(acc_train, sum(trainy_pred_list), len(trainy_pred_list))
        print(acc_test, sum(testy_pred_list), len(testy_pred_list))


if __name__ == "__main__":
    dataset_name = sys.argv[1]
    assert dataset_name in ["german", "adult", "default"]
    scaling = "min"
    main(dataset_name, scaling)
