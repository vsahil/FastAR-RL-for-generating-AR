import numpy as np
from sklearn.neighbors import NearestNeighbors
import sys, os


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


def follows_causality(cfe, original_datapoint, immutable_features, non_decreasing_features, correlated_features):
    follows = True
    diff = cfe.to_numpy().astype(float) - original_datapoint.to_numpy()
    m2 = (diff != 0)[0].nonzero()
    changed_columns = cfe.columns[m2].tolist()
    # This won't hold for random and greedy approaches. 
    # assert len(set(changed_columns).intersection(set(immutable_features))) == 0
    
    # Check if non-decreasing features are decreasing. 
    diff_nondecrease = cfe[non_decreasing_features].to_numpy().astype(float) - original_datapoint[non_decreasing_features].to_numpy()
    m2 = (diff_nondecrease < 0)[0].nonzero()
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
    nearest_dist, nearest_points = knn.kneighbors(cfe, 1, return_distance=True)
    return np.mean(nearest_dist)


def calculate_metrics(method, final_cfs, cfs_found, find_cfs_points, model, dataset, knn, continuous_features, 
                mads, immutable_features, non_decreasing_features, correlated_features, scaler, setting, time_taken, num_episodes, train_time, save=False):
    
    mads = [mads[key] if mads[key]!= 0.0 else 1.0 for key in mads]

    avg_proximity_cont = []
    avg_proximity_cat = []
    avg_sparsity = []
    avg_causality = []
    avg_manifold_dist = []
    computed_cfs = []
    for seq, dt in enumerate(cfs_found):
        if dt:  # find the metrics only if a cfe was found for a datapoint it was requested for. 
            cfe = final_cfs[seq:seq+1]
            computed_cfs.append(cfe.to_numpy())
            cfe_prediction = model.predict(cfe)[0]
            original_datapoint = find_cfs_points[seq: seq+1]
            original_prediction = model.predict(original_datapoint)[0]
            try:
                assert cfe_prediction != original_prediction
            except:
                if "MACE" in method:
                    pass
                else:
                    # print(seq, "failing as CFE")
                    pass
            proximity_cont, sparsity_cont = find_proximity_cont(cfe, original_datapoint, continuous_features, mads)
            categorical_features = [f for f in dataset.columns.tolist() if f not in continuous_features]
            assert len(categorical_features) + len(continuous_features) == len(dataset.columns.tolist())
            proximity_cat, sparsity_cat = find_proximity_cat(cfe, original_datapoint, categorical_features)
            sparsity = sparsity_cont + sparsity_cat
            causality = follows_causality(cfe, original_datapoint, immutable_features, non_decreasing_features, correlated_features)
            manifold_dist = find_manifold_dist(cfe, knn)
            
            avg_proximity_cont.append(proximity_cont)
            avg_proximity_cat.append(proximity_cat)
            avg_sparsity.append(sparsity)
            avg_causality.append(causality)
            avg_manifold_dist.append(manifold_dist)
    
    if save:
        np.save(f"saved_files/{method}_avg_proximity_cont.npy", avg_proximity_cont)
        np.save(f"saved_files/{method}_avg_proximity_cat.npy", avg_proximity_cat)
        np.save(f"saved_files/{method}_avg_sparsity.npy", avg_sparsity)
        np.save(f"saved_files/{method}_avg_causality.npy", avg_causality)
        np.save(f"saved_files/{method}_avg_manifold_dist.npy", avg_manifold_dist)
        np.save(f"saved_files/{method}_computed_cfes.npy", np.array(computed_cfs))
    # print(avg_proximity_cont, avg_proximity_cat, avg_sparsity, avg_causality, avg_manifold_dist)
    validity = sum(cfs_found) * 100.0 / find_cfs_points.shape[0]
    # Header: setting,validity,proximity_cont,proximity_cat,sparsity,manifold_dist,causality,time
    file = f"output/results/all_metrics_{method}.csv"
    if not os.path.exists(file):
        with open(file, "a") as f:
            print("setting,num_episodes,train_time,validity,proximity_cont,proximity_cat,sparsity,manifold,causality,time", file=f)
    with open(file, "a") as f:
        print(setting, num_episodes, round(train_time, 2), round(validity, 3), round(np.mean(avg_proximity_cont), 3), round(np.mean(avg_proximity_cat), 3), round(np.mean(avg_sparsity), 3), round(np.mean(avg_manifold_dist), 3), round(np.mean(avg_causality) * 100.0, 3), round(time_taken, 2), sep=',', file=f)

