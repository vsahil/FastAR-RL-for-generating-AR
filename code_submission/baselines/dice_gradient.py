import sys, os, time
import dice_ml
from dice_ml.utils import helpers # helper functions
import pandas as pd
import numpy as np
import cal_metrics
import pickle
sys.path.append("../")
sys.path.append("train_pytorch_model/")
import classifier_dataset as classifier
import torch

backend = 'PYT'
os.environ['OUTPUT_DIM'] = '1'     # This is used in the model which is trained on the respective dataset. 

# In this file the correlated features will be normalized, just like in evaluation.py. 
def run_dice(dataset_name, balanced):
    print(f"Gen CFES for {dataset_name}, balancing: {balanced}")
    if dataset_name == "german":
        file1 = f"{os.path.dirname(os.path.realpath(__file__))}/../datasets/german_redone.csv"
        dataset, _, X_test, X_train, _, _ = classifier.train_model_german(file=file1, parameter=3, drop=False)
        continuous_features = ['Months', 'Credit-amount', 'Insatllment-rate', 'Present-residence-since', 'age', 'Number-of-existing-credits', 'Number-of-people-being-lible']
        immutable_features = ['Personal-status', 'Number-of-people-being-lible', 'Foreign-worker', 'Purpose']
        non_decreasing_features = ['age', 'Job']
        correlated_features = []
    
    elif dataset_name == "adult":
        file1 = f"{os.path.dirname(os.path.realpath(__file__))}/../datasets/adult_redone.csv"
        dataset, _, X_test, X_train, _, _ = classifier.train_model_adult(file=file1, parameter=3, drop=False)
        continuous_features = ['age', 'fnlwgt', 'capitalgain', 'capitalloss', 'hoursperweek']
        immutable_features = ['marital-status', 'race', 'native-country', 'sex']
        non_decreasing_features = ['age', 'education']
        correlated_features = [('education', 'age', 0.054)]     # With each increase in level of education, we increase the age by 2. 
    
    elif dataset_name == "default":
        file1 = f"{os.path.dirname(os.path.realpath(__file__))}/../datasets/default_redone.csv"
        dataset, _, X_test, X_train, _, _ = classifier.train_model_default(file=file1, parameter=3, drop=False)
        continuous_features = ['LIMIT_BAL', 'AGE', 'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6', 'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']
        immutable_features = ['sex', 'MARRIAGE']
        non_decreasing_features = ['AGE', 'EDUCATION']
        correlated_features = [('EDUCATION', 'AGE', 0.027)]   # the increase in unnormalized form is much higher. 
 
    d = dice_ml.Data(dataframe=dataset, continuous_features = [i for i in X_train.columns], 
            outcome_name='target')

    if dataset_name == "german":
        saved_model_path = "train_pytorch_model/saved_models/german_model_76.pth"
    elif dataset_name == "adult":
        # balanced is used for having equal number of 0 and 1 in the training set.
        if balanced:
            saved_model_path = "train_pytorch_model/saved_models/adult_sigmoid_smallest_balanced81.pth"
        else:
            saved_model_path = "train_pytorch_model/saved_models/adult_sigmoid_smallest_unbalanced83.pth"
    elif dataset_name == "default":
        if balanced:
            saved_model_path = "train_pytorch_model/saved_models/default_sigmoid_smallest_balanced69.pth"
        else:
            saved_model_path = "train_pytorch_model/saved_models/default_sigmoid_smallest_unbalanced83.pth"
    
    sys.path.append("train_pytorch_model/saved_models")
    if dataset_name == "german":
        import german_pytorch
        input_dim = X_train.shape[1]
        hidden_dim1 = 5
        hidden_dim2 = 3
        output_dim = 1
        model = german_pytorch.Net(input_dim=input_dim, hidden_dim1=hidden_dim1, 
                            hidden_dim2=hidden_dim2, output_dim=output_dim)
    
    elif dataset_name in ["adult", "default"]:
        import main
        input_dim = X_train.shape[1]
        model = main.binaryClassification(input_dim=input_dim)

    model.load_state_dict(torch.load(saved_model_path))
    model.eval()

    m = dice_ml.Model(model=model, backend=backend)
    undesirable_x = np.load(f"undesirable_x_{dataset_name}.npy")
    print(len(undesirable_x), "Total points to run the approach on")
    # Several of these will have prediction 1 and the CFE will have prediction 0. 
    find_cfs_points = pd.DataFrame(undesirable_x, columns=X_test.columns.tolist())

    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(X_train)
    
    exp = dice_ml.Dice(d, m)
    start = time.time()
    features_to_vary = [f for f in X_test.columns.tolist() if f not in immutable_features]
    assert len(features_to_vary) == len(X_test.columns.tolist()) - len(immutable_features)
    
    method = "dice-gradient" 
    method += ("_balanced-" if balanced else "_unbalanced-")
    
    if dataset_name == "german":
        num_datapoints = find_cfs_points.shape[0]
        num_datapoints = 10
    elif dataset_name == "adult":
        num_datapoints = 10     # 500
    elif dataset_name == "default":
        num_datapoints = 100

    save_file = f'saved_files/{dataset_name}_{method}_explanations_{num_datapoints}.pkl'

    if os.path.exists(save_file):
        print("loading saved results")
        with open(save_file, 'rb') as input:
            explainer_objects = pickle.load(input)
        cfs_found = np.load(f'saved_files/{dataset_name}_{method}_cfs_found_{num_datapoints}.npy')
        save = False
    else:
        # I will have to iterate over the list of CF points, in other dice methods this was internal, not here. 
        cfs_found = []
        explainer_objects = []
        save = False
        # import ipdb; ipdb.set_trace()
        for num in range(num_datapoints):
            e1, this_cf_found = exp.generate_counterfactuals(find_cfs_points[num: num+1], total_CFs=1, 
                desired_class="opposite", features_to_vary=features_to_vary) #, proximity_weight=0.0, diversity_weight=0.0)
            explainer_objects.append(e1)
            cfs_found.append(this_cf_found)
        
        # if sum(cfs_found) > 0:
        #     with open(save_file, 'wb') as output:
        #         pickle.dump(explainer_objects, output, pickle.HIGHEST_PROTOCOL)
        #     np.save(f'saved_files/{dataset_name}_{method}_cfs_found_{num_datapoints}.npy', np.array(cfs_found))

    print(f"{method} : No. of found cfes: {sum(cfs_found)} Numdatapoints: {num_datapoints} Time taken: {time.time() - start}")
    time_taken = time.time() - start
    
    if save:
        with open("results.txt", "a") as f:
            print(f"{method + dataset_name}: {sum(cfs_found)} : {num_datapoints} : {time_taken}", file=f)

    cal_metrics.calculate_metrics(method + dataset_name, explainer_objects, cfs_found, find_cfs_points[:num_datapoints], model, 
            dataset.drop(columns=['target']), continuous_features, d.get_mads(), 
            immutable_features, non_decreasing_features, correlated_features, scaler, time_taken, save=save)


if __name__ == '__main__':
    dataset_name = sys.argv[1]
    balanced = 0
    run_dice(dataset_name, balanced)

