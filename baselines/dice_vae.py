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
import argparse
torch.manual_seed(0)

os.environ['OUTPUT_DIM'] = '2'      # This is used in the model which is trained on the respective dataset. 
# backend = 'PYT'

# In this file the correlated features will be normalized, just like in evaluation.py. 
def run_dice(args):
    # print(args)
    print(f"Gen CFES for {args.dataset_name}, balancing: {args.balanced}")
    if args.dataset_name == "german":
        file1 = f"{os.path.dirname(os.path.realpath(__file__))}/../datasets/german_redone.csv"
        dataset, _, X_test, X_train, _, _ = classifier.train_model_german(file=file1, parameter=3, drop=False)
        continuous_features = ['Months', 'Credit-amount', 'Insatllment-rate', 'Present-residence-since', 'age', 'Number-of-existing-credits', 'Number-of-people-being-lible']
        immutable_features = ['Personal-status', 'Number-of-people-being-lible', 'Foreign-worker', 'Purpose']
        non_decreasing_features = ['age', 'Job']
        correlated_features = []
    
    elif args.dataset_name == "adult":
        file1 = f"{os.path.dirname(os.path.realpath(__file__))}/../datasets/adult_redone.csv"
        dataset, _, X_test, X_train, _, _ = classifier.train_model_adult(file=file1, parameter=3, drop=False)
        continuous_features = ['age', 'fnlwgt', 'capitalgain', 'capitalloss', 'hoursperweek']
        immutable_features = ['marital-status', 'race', 'native-country', 'sex']
        non_decreasing_features = ['age', 'education']
        correlated_features = [('education', 'age', 0.054)]     # With each increase in level of education, we increase the age by 2. 
    
    elif args.dataset_name == "default":
        file1 = f"{os.path.dirname(os.path.realpath(__file__))}/../datasets/default_redone.csv"
        dataset, _, X_test, X_train, _, _ = classifier.train_model_default(file=file1, parameter=3, drop=False)
        continuous_features = ['LIMIT_BAL', 'AGE', 'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6', 'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']
        immutable_features = ['sex', 'MARRIAGE']
        non_decreasing_features = ['AGE', 'EDUCATION']
        correlated_features = [('EDUCATION', 'AGE', 0.027)]   # the increase in unnormalized form is much higher. 

    # import ipdb; ipdb.set_trace()
    # all continuous otherwise it converts to one-hot, headache. 
    d = dice_ml.Data(dataframe=dataset, continuous_features = [i for i in X_train.columns], 
            outcome_name='target', data_name=args.dataset_name)

    if args.dataset_name == "german":
        saved_model_path = "train_pytorch_model/saved_models/german_model_softmax_78.pth"
    elif args.dataset_name == "adult":
        # balanced is used for having equal number of 0 and 1 in the training set.
        if args.balanced:
            saved_model_path = "train_pytorch_model/saved_models/adult_sigmoid_smallest_balanced81.pth"
        else:
            saved_model_path = "train_pytorch_model/saved_models/adult_sigmoid_smallest_unbalanced_softmax83.pth"
    elif args.dataset_name == "default":
        if args.balanced:
            saved_model_path = "train_pytorch_model/saved_models/default_sigmoid_smallest_balanced69.pth"
        else:
            saved_model_path = "train_pytorch_model/saved_models/default_sigmoid_smallest_unbalanced_softmax83.pth"
    
    sys.path.append("train_pytorch_model/saved_models")
    if args.dataset_name == "german":
        import german_pytorch
        input_dim = X_train.shape[1]
        hidden_dim1 = 5
        hidden_dim2 = 3
        output_dim = 2
        model = german_pytorch.Net(input_dim=input_dim, hidden_dim1=hidden_dim1, 
                            hidden_dim2=hidden_dim2, output_dim=output_dim)
    
    elif args.dataset_name in ["adult", "default"]:
        import main
        input_dim = X_train.shape[1]
        model = main.binaryClassification(input_dim=input_dim)

    model.load_state_dict(torch.load(saved_model_path))
    model.eval()

    backend = {'model': 'pytorch_model.PyTorchModel',
           'explainer': 'feasible_base_vae.FeasibleBaseVAE'}
    m = dice_ml.Model(model=model, backend=backend)
    undesirable_x = np.load(f"undesirable_x_{args.dataset_name}.npy")
    print(len(undesirable_x), "Total points to run the approach on")
    # Several of these will have prediction 1 and the CFE will have prediction 0. 
    find_cfs_points = pd.DataFrame(undesirable_x, columns=X_test.columns.tolist())

    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler(feature_range=(0, 1))     # this is between 0 and 1 as DiCE requires
    scaler.fit(X_train)
    
    # dataset = helpers.load_adult_income_dataset()
    # d = dice_ml.Data(dataframe=dataset, continuous_features=['age', 'hours_per_week'],
    #              outcome_name='income', data_name='adult', test_size=0.1)
    # ML_modelpath = helpers.get_adult_income_modelpath(backend='PYT')
    # ML_modelpath = ML_modelpath[:-4] + '_2nodes.pth'
    # m = dice_ml.Model(model_path=ML_modelpath, backend=backend)
    # m.load_model()
    # print('ML Model', m.model)
    validity_reg = args.validity_reg
    margin = args.margin
    epochs = args.epochs
    lr = args.lr
    batch_size = args.batch_size
    encoded_size = args.encoded_size

    exp = dice_ml.Dice(d, m, encoded_size=encoded_size, lr=lr,
                   batch_size=batch_size, validity_reg=validity_reg, margin=margin, epochs=epochs,
                   wm1=1e-2, wm2=1e-2, wm3=1e-2)
    # import ipdb; ipdb.set_trace()
    pre_trained = False
    if os.path.exists(f"dice_ml/utils/sample_trained_models/{args.dataset_name}-margin-{margin}-validity_reg-{validity_reg}-epoch-{epochs}-batchsize-{batch_size}-lr-{lr}-encoded-{encoded_size}-base-gen.pth"):
        pre_trained = True
    exp.train(pre_trained=pre_trained)
    
    start = time.time()
    features_to_vary = [f for f in X_test.columns.tolist() if f not in immutable_features]
    assert len(features_to_vary) == len(X_test.columns.tolist()) - len(immutable_features)

    method = "dice-vae"
    method += ("_balanced-" if args.balanced else "_unbalanced-")
    
    if args.dataset_name == "german":
        num_datapoints = find_cfs_points.shape[0]
        # num_datapoints = 3
    elif args.dataset_name == "adult":
        num_datapoints = find_cfs_points.shape[0]
        # num_datapoints = 5
    elif args.dataset_name == "default":
        num_datapoints = find_cfs_points.shape[0]
        # num_datapoints = 5

    save_file = f'saved_files/{args.dataset_name}_{method}_explanations_{num_datapoints}.pkl'

    if os.path.exists(save_file):
        print("loading saved results")
        with open(save_file, 'rb') as input:
            explainer_objects = pickle.load(input)
        cfs_found = np.load(f'saved_files/{args.dataset_name}_{method}_cfs_found_{num_datapoints}.npy')
        save = True
    else:
        # I will have to iterate over the list of CF points, in other dice methods this was internal, not here. 
        cfs_found = []
        explainer_objects = []
        save = True
        # import ipdb; ipdb.set_trace()
        for num in range(num_datapoints):
            # print("Datapoint: ", num+1)
            e1, this_cf_found = exp.generate_counterfactuals(find_cfs_points[num: num+1], total_CFs=1, 
                desired_class="opposite")       #, features_to_vary=features_to_vary)    #, proximity_weight=0.0, diversity_weight=0.0)
            explainer_objects.append(e1)
            cfs_found.append(this_cf_found)
        
        # if sum(cfs_found) > 0:
        #     with open(save_file, 'wb') as output:
        #         pickle.dump(explainer_objects, output, pickle.HIGHEST_PROTOCOL)
        #     np.save(f'saved_files/{args.dataset_name}_{method}_cfs_found_{num_datapoints}.npy', np.array(cfs_found))

    print(f"{method} : No. of found cfes: {sum(cfs_found)} Numdatapoints: {num_datapoints} Time taken: {time.time() - start}")
    time_taken = time.time() - start
    
    if save:
        with open("results_vae.txt", "a") as f:
            print(f"{method + str(args)}: {sum(cfs_found)} : {num_datapoints} : {time_taken}", file=f)

    save = False
    cal_metrics.calculate_metrics(method + args.dataset_name, explainer_objects, cfs_found, find_cfs_points[:num_datapoints], model, 
            dataset.drop(columns=['target']), continuous_features, d.get_mads(), 
            immutable_features, non_decreasing_features, correlated_features, scaler, time_taken, save=save)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', default='german', type=str)
    parser.add_argument('--validity_reg', default=84, type=int)
    parser.add_argument('--margin', default=0.165, type=float)
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--lr', default=3e-2, type=float)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--encoded_size', default=10, type=int)
    parser.add_argument('--balanced', default=False, type=lambda x: (str(x).lower() == 'true'), help='If true, uses same number of 0 and 1 labels for training')
    
    args = parser.parse_args()
    run_dice(args)

