import numpy as np
from sklearn.neighbors import NearestNeighbors
import pandas as pd
import os


def find_proximity_cont(cfe, original_datapoint, continuous_features, mads):
    diff = original_datapoint[continuous_features].to_numpy() - cfe[continuous_features].to_numpy()
    dist_cont = np.mean(np.divide(np.abs(diff), mads))
    sparsity_cont = diff[0].nonzero()[0].shape[0]
    return dist_cont, sparsity_cont


def find_proximity_cat(cfe, original_datapoint, categorical_features):
    cfe_cats = cfe[categorical_features].to_numpy().astype(float)
    diff = original_datapoint[categorical_features].to_numpy() - cfe_cats
    sparsity_cat = diff[0].nonzero()[0].shape[0]
    dist_cat = sparsity_cat * 1.0 / len(categorical_features)
    return dist_cat, sparsity_cat


def follows_causality(cfe, original_datapoint, immutable_features, 
        non_decreasing_features, correlated_features, method):
    follows = True
    diff = cfe.to_numpy().astype(float) - original_datapoint.to_numpy()
    m2 = (diff != 0)[0].nonzero()
    changed_columns = cfe.columns[m2].tolist()      # m2.index[m2].tolist()
    if any(x in method for x in ["dice-vae", "MACE", "CFproto"]):
        if len(set(changed_columns).intersection(set(immutable_features))) > 0:
            follows = False
            return follows
    else:
            assert len(set(changed_columns).intersection(set(immutable_features))) == 0
    
    diff_nondecrease = cfe[non_decreasing_features].to_numpy().astype(float) - original_datapoint[non_decreasing_features].to_numpy()
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
    nearest_dist, nearest_points = knn.kneighbors(cfe.to_numpy(), 1, return_distance=True)
    quantity = np.mean(nearest_dist)		# Now we have that lambda, so no need to divide by self.no_points 
    return quantity


def calculate_metrics(method, e1, cfs_found, find_cfs_points, model, dataset, continuous_features, 
                mads, immutable_features, non_decreasing_features, correlated_features, scaler, time_taken, save=False):
    
    # import ipdb; ipdb.set_trace()
    if not any(x in method for x in ["random", "greedy", "MACE", "CFproto"]):
        find_cfs_points = pd.DataFrame(scaler.transform(find_cfs_points), columns=dataset.columns.tolist())     # this is same normalized as DiCE does it. 
    knn = NearestNeighbors(n_neighbors=5, p=1)
    transformed_dataset = scaler.transform(dataset)
    knn.fit(transformed_dataset)
    transformed_dataset = pd.DataFrame(transformed_dataset, columns=dataset.columns.tolist())
    normalized_mads = {}
    for feature in continuous_features:
        normalized_mads[feature] = np.median(abs(transformed_dataset[feature].values - np.median(transformed_dataset[feature].values)))
    del mads
    mads = [normalized_mads[key] if normalized_mads[key]!= 0.0 else 1.0 for key in normalized_mads]

    avg_proximity_cont = []
    avg_proximity_cat = []
    avg_sparsity = []
    avg_causality = []
    avg_manifold_dist = []
    computed_cfs = []
    # import ipdb; ipdb.set_trace()
    for seq, dt in enumerate(cfs_found):
        if dt:  # find the metrics only if a cfe was found for a datapoint it was requested for. 
            if "dice-gradient" in method or "dice-vae" in method:
                cfe = e1[seq].final_cfs_df
            if any(x in method for x in ["random", "greedy", "MACE", "CFproto"]):
                cfe = e1[seq:seq+1]
            else:
                cfe = e1.cf_examples_list[seq].final_cfs_df
            
            if not any(x in method for x in ["random", "greedy", "MACE", "CFproto"]):
                cfe = cfe.drop(columns=['target'])
                cfe = scaler.transform(cfe)
            computed_cfs.append(cfe)
            
            if any(x in method for x in ["random", "greedy", "MACE", "CFproto"]):
                cfe_prediction = model.predict(cfe)[0]
            else:
                cfe_prediction = model.orig_predict(cfe)[0]
            
            if "dice-vae" in method or "CFproto" in method:
                cfe_prediction = np.argmax(cfe_prediction)
            original_datapoint = find_cfs_points[seq: seq+1]
            
            if "dice-gradient" in method:
                original_prediction = model.orig_predict(original_datapoint.to_numpy())[0]
            elif "dice-vae" in method:
                original_prediction = model.orig_predict(original_datapoint.to_numpy())[0]
                original_prediction = np.argmax(original_prediction)
            elif any(x in method for x in ["random", "greedy", "MACE", "CFproto"]):
                original_prediction = model.predict(original_datapoint)[0]
                if "CFproto" in method:
                    original_prediction = np.argmax(original_prediction)
            else:
                original_prediction = model.orig_predict(original_datapoint)[0]
            # print(cfe, cfe_prediction, original_prediction)
            try:
                assert cfe_prediction != original_prediction
            except:
                if "MACE" in method:
                    pass
                else:
                    print(seq, "failing as CFE")
            if not any(x in method for x in ["random", "greedy", "MACE", "CFproto"]):
                cfe = pd.DataFrame(cfe, columns=dataset.columns.tolist())
            proximity_cont, sparsity_cont = find_proximity_cont(cfe, original_datapoint, continuous_features, mads)
            categorical_features = [f for f in dataset.columns.tolist() if f not in continuous_features]
            assert len(categorical_features) + len(continuous_features) == len(dataset.columns.tolist())
            proximity_cat, sparsity_cat = find_proximity_cat(cfe, original_datapoint, categorical_features)
            sparsity = sparsity_cont + sparsity_cat
            causality = follows_causality(cfe, original_datapoint, immutable_features, 
                    non_decreasing_features, correlated_features, method)
            manifold_dist = find_manifold_dist(cfe, knn)
            
            avg_proximity_cont.append(proximity_cont)
            avg_proximity_cat.append(proximity_cat)
            avg_sparsity.append(sparsity)
            avg_causality.append(causality)
            avg_manifold_dist.append(manifold_dist)
    
    validity = sum(cfs_found) * 100.0 / find_cfs_points.shape[0]
    # Header: setting,validity,proximity_cont,proximity_cat,sparsity,manifold_dist,causality,time
    file = f"{os.path.dirname(os.path.realpath(__file__))}/results/all_metrics_baselines.csv"
    if not os.path.exists(file):
        with open(file, "a") as f:
            print("setting,validity,proximity_cont,proximity_cat,sparsity,manifold,causality,time", file=f)
    with open(file, "a") as f:
        print(method, round(validity, 3), round(np.mean(avg_proximity_cont), 3), 
                round(np.mean(avg_proximity_cat), 3), round(np.mean(avg_sparsity), 3), 
                round(np.mean(avg_manifold_dist), 3), round(np.mean(avg_causality) * 100.0, 3), 
                round(time_taken, 2), sep=',', file=f)
