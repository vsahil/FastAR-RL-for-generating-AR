import sys, os, time
sys.path.append("/scratch/vsahil/RL-for-Counterfactuals/paper_expts/")
import dice_ml
from dice_ml.utils import helpers # helper functions
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors


def find_proximity_cont(cfe, original_datapoint, continuous_features, mads):
    # import ipdb; ipdb.set_trace()
    diff = original_datapoint[continuous_features].to_numpy() - cfe[continuous_features].to_numpy()
    dist_cont = np.mean(np.divide(np.abs(diff), mads))
    sparsity_cont = diff[0].nonzero()[0].shape[0]
    # print(diff, sparsity_cont, "see")
    return dist_cont, sparsity_cont


def find_proximity_cat(cfe, original_datapoint, categorical_features):
    # import ipdb; ipdb.set_trace()
    cfe_cats = cfe[categorical_features].to_numpy().astype(int)
    diff = original_datapoint[categorical_features].to_numpy() - cfe_cats
    sparsity_cat = diff[0].nonzero()[0].shape[0]
    # print(diff, sparsity_cat, "see")
    dist_cat = sparsity_cat * 1.0 / len(categorical_features)
    return dist_cat, sparsity_cat


def follows_causality(cfe, original_datapoint, immutable_features, non_decreasing_features, correlated_features):
    # import ipdb; ipdb.set_trace()
    follows = True
    diff = cfe.to_numpy().astype(int) - original_datapoint.to_numpy()
    m2 = (diff != 0)[0].nonzero()
    changed_columns = cfe.columns[m2].tolist()      # m2.index[m2].tolist()
    assert len(set(changed_columns).intersection(set(immutable_features))) == 0
    print(diff, changed_columns)
    
    diff_nondecrease = cfe[non_decreasing_features].to_numpy().astype(int) - original_datapoint[non_decreasing_features].to_numpy()
    # m2 = (diff_nondecrease < 0)
    m2 = (diff_nondecrease < 0)[0].nonzero()
    # decreased_columns = m2[0].shape[0]        # m2.index[m2].tolist()
    if m2[0].shape[0] > 0:
        follows = False
        return follows
    
    for f1, f2, linear_add in correlated_features:
        seq_f1 = cfe.columns.tolist().index(f1)
        seq_f2 = cfe.columns.tolist().index(f2)
        if diff[0][seq_f1] > 0:      # If there is an increase in f1
            if not diff[0][seq_f2] >= linear_add:      # then there must be an increase in f2 of value 'linear_add'
                follows = False
                return follows

    return follows


def find_manifold_dist(cfe, knn):
    # import ipdb; ipdb.set_trace()
    cfe = scaler.transform(cfe.to_numpy())
    nearest_dist, nearest_points = knn.kneighbors(cfe, 1, return_distance=True)
    quantity = np.mean(nearest_dist)		# Now we have that lambda, so no need to divide by self.no_points 
    # print(quantity, nearest_dist)
    return quantity


def calculate_metrics(method, e1, cfs_found, find_cfs_points, model, dataset, continuous_features, 
                mads, immutable_features, non_decreasing_features, correlated_features, scaler):
    
    # import ipdb; ipdb.set_trace()
    knn = NearestNeighbors(n_neighbors=5, p=1)
    knn.fit(scaler.transform(dataset))
    mads = [mads[key] if mads[key]!= 0.0 else 1.0 for key in mads]

    avg_proximity_cont = []
    avg_proximity_cat = []
    avg_sparsity = []
    avg_causality = []
    avg_manifold_dist = []
    computed_cfs = []
    for seq, dt in enumerate(cfs_found):
        if dt:  # find the metrics only if a cfe was found for a datapoint it was requested for. 
            cfe = e1.cf_examples_list[seq].final_cfs_df
            cfe = cfe.drop(columns=['target'])
            computed_cfs.append(cfe.to_numpy())
            cfe_prediction = model.predict(cfe)[0]
            original_datapoint = find_cfs_points[seq: seq+1]
            original_prediction = model.predict(original_datapoint)[0]
            # print(cfe.to_numpy(), cfe_prediction, original_prediction)
            assert cfe_prediction != original_prediction
            proximity_cont, sparsity_cont = find_proximity_cont(cfe, original_datapoint, continuous_features, mads)
            categorical_features = [f for f in dataset.columns.tolist() if f not in continuous_features]
            assert len(categorical_features) + len(continuous_features) == len(dataset.columns.tolist())
            proximity_cat, sparsity_cat = find_proximity_cat(cfe, original_datapoint, categorical_features)
            sparsity = sparsity_cont + sparsity_cat
            causality = follows_causality(cfe, original_datapoint, immutable_features, non_decreasing_features, correlated_features)
            manifold_dist = find_manifold_dist(cfe, knn)
            # print(proximity_cont, proximity_cat, sparsity, causality, manifold_dist)
            
            avg_proximity_cont.append(proximity_cont)
            avg_proximity_cat.append(proximity_cat)
            avg_sparsity.append(sparsity)
            avg_causality.append(causality)
            avg_manifold_dist.append(manifold_dist)
    
    np.save(f"saved_files/{method}_avg_proximity_cont.npy", avg_proximity_cont)
    np.save(f"saved_files/{method}_avg_proximity_cat.npy", avg_proximity_cat)
    np.save(f"saved_files/{method}_avg_sparsity.npy", avg_sparsity)
    np.save(f"saved_files/{method}_avg_causality.npy", avg_causality)
    np.save(f"saved_files/{method}_avg_manifold_dist.npy", avg_manifold_dist)
    np.save(f"saved_files/{method}_computed_cfes.npy", np.array(computed_cfs))
    print(avg_proximity_cont, avg_proximity_cat, avg_sparsity, avg_causality, avg_manifold_dist)



# import ipdb; ipdb.set_trace()
# dataset_dice = helpers.load_adult_income_dataset()
# d = dice_ml.Data(dataframe=dataset_dice, continuous_features=['age', 'hours_per_week'], outcome_name='income')
import classifier_german as classifier
file1 = "/scratch/vsahil/RL-for-Counterfactuals/paper_expts/adult_redone.csv"
model, dataset, scaler, X_test, X_train = classifier.train_model_adult(file=file1, parameter=1, drop=False)
# X_train_ = scaler.transform(X_train)
# X_test_ = scaler.transform(X_test)
continuous_features = ['age', 'fnlwgt', 'capitalgain', 'capitalloss', 'hoursperweek']
d = dice_ml.Data(dataframe=dataset, continuous_features=continuous_features, outcome_name='target')

def predict_proba_dice(input_instance):
    if isinstance(input_instance, pd.DataFrame):
        scaled_input = scaler.transform(input_instance)
        pr = model.predict_probability(scaled_input)
        # print(input_instance, scaled_input, pr)
        return pr
    else:
        print(input_instance, type(input_instance), "see")
        raise NotImplementedError


def new_predict(input_instance):
    pr = model.predict_proba(input_instance)
    classes = np.argmax(pr, axis=1)
    assert classes.shape[0] == input_instance.shape[0]
    return classes


try:
    undesirable_x = np.load("undesirable_x.npy")
except:
    assert not os.path.exists("undesirable_x.npy")
    undesirable_x = []
    for no, i in enumerate(X_test.to_numpy()):
        if classifier.predict_single(i, scaler, model) == 0:
            undesirable_x.append(tuple(i))
    undesirable_x = np.array(undesirable_x)
    np.save("undesirable_x.npy", undesirable_x)

find_cfs_points = pd.DataFrame(undesirable_x, columns=X_test.columns.tolist())

model.predict_probability = model.predict_proba
model.predict_proba = predict_proba_dice

model.orig_predict = model.predict
model.predict = new_predict

# Using sklearn backend
m = dice_ml.Model(model=model, backend="sklearn")
# Using method=random for generating CFs
method = str(sys.argv[1])
# method = "kdtree"
# method = "genetic"
# method = "random"

exp = dice_ml.Dice(d, m, method=method)

start = time.time()
# immutable_features = []
immutable_features = ['Marital-status', 'Race', 'Native-country', 'Sex']
immutable_features = [i.lower() for i in immutable_features]
features_to_vary = [f for f in X_test.columns.tolist() if f not in immutable_features]
assert len(features_to_vary) == len(X_test.columns.tolist()) - len(immutable_features)
num_datapoints = find_cfs_points.shape[0]
e1, cfs_found = exp.generate_counterfactuals(find_cfs_points[:num_datapoints], total_CFs=1, 
        desired_class="opposite", 
        features_to_vary=features_to_vary)

with open("results.txt", "a") as f:
    print(f"{method} : No. of found cfes: {sum(cfs_found)} Numdatapoints: {num_datapoints} Time taken: {time.time() - start}")
    print(f"{method}: {sum(cfs_found)} : {num_datapoints} : {time.time() - start}", file=f)

non_decreasing_features = ['age', 'education']
correlated_features = [('education', 'age', 2)]     # With each increase in level of education, we increase the age by 2. 
# calculate_metrics(method, e1, cfs_found, find_cfs_points, model, dataset.drop(columns=['target']), continuous_features, 
#             d.get_mads(), immutable_features, non_decreasing_features, correlated_features, scaler)

