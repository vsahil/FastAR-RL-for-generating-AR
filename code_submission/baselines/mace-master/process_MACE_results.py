import numpy as np
import pandas as pd
import sys, os
# sys.path.append("../../fastar/")
sys.path.append("../")
import cal_metrics


def create_dict(line):
    Dict = dict((x.strip(), y.strip())
        for x, y in (element.split(':') 
            for element in line.split(', ')))
    assert isinstance(Dict, dict)
    del Dict['y']
    if "{x0" in Dict.keys():
        Dict['x0'] = Dict['{x0']
        del Dict['{x0']
    # print(Dict)
    for old_key in Dict.keys():
        if isinstance(old_key, str):
            assert "x" == old_key[0]
            new_key = int(old_key[1:])
            Dict[new_key] = float(Dict.pop(old_key))      # both the keys and values are to be made into integers
    return Dict


def append_(main_dict, datapoint):
    assert isinstance(main_dict, dict)
    assert isinstance(datapoint, dict)
    num_keys = len(datapoint.keys())
    if len(main_dict.keys()) == 0:
        for i in range(num_keys):
            main_dict[i] = []
    
    assert len(main_dict.keys()) == num_keys
    for k, v in datapoint.items():
        main_dict[k].append(v)
    
    assert len(main_dict.keys()) == num_keys
    return main_dict


def extract_points_from_files(result_direc):
    final_cfs = {}
    find_cfs_points = {}
    num_files = len(os.listdir(result_direc))
    for files in os.listdir(result_direc):
        with open(result_direc + files, "r") as f:
            content = f.readlines()
        for lines in content:
            if "Factual sample:" in lines:
                # print(files, lines, "See")
                lines = lines.replace("Factual sample:", "").strip()
                dict_factual = create_dict(lines)
            if "Nearest counterfactual sample" in lines:
                # print(files, lines, "See2")
                lines = lines.replace("Nearest counterfactual sample:", "")
                if "(verified)" in lines:
                    lines = lines.replace("(verified)", "").strip()
                    dict_cf = create_dict(lines)
                
                find_cfs_points = append_(find_cfs_points, dict_factual)
                final_cfs = append_(final_cfs, dict_cf)
        
    num_features = 20
    assert all(len(value) == num_files for value in final_cfs.values())
    assert all(len(value) == num_files for value in find_cfs_points.values())
    assert len(final_cfs) == len(find_cfs_points) == num_features
    normal = pd.DataFrame.from_dict(find_cfs_points)
    cfes = pd.DataFrame.from_dict(final_cfs)
    # print(normal)
    # print(cfes)
    assert normal.shape == cfes.shape
    return normal, cfes


def compute_time(result_direc):
    assert "minimum_distances.txt" in os.listdir(result_direc)
    total_time = []
    with open(result_direc + "minimum_distances.txt", "r") as f:
        content = f.readlines()
    for lines in content:
        if 'cfe_time' in lines:
            lines = lines.replace("'cfe_time':", "").strip()
            lines = lines.replace(",", "")
            total_time.append(float(lines))
    return total_time


def prepare_for_cal_metrics(find_cfs_points, final_cfs, setting, time_taken):
    sys.path.append("../../fastar/gym-midline/gym_midline/envs/")
    import german_credit
    env_ = german_credit.GermanCredit01()
    # print(final_cfs)
    find_cfs_points = env_.scaler.transform(find_cfs_points)
    final_cfs = env_.scaler.transform(final_cfs)
    assert find_cfs_points.shape[1] == len(env_.dataset.columns)
    find_cfs_points = pd.DataFrame(find_cfs_points, columns=env_.dataset.columns)
    final_cfs = pd.DataFrame(final_cfs, columns=env_.dataset.columns)
    assert len(time_taken) == final_cfs.shape[0]
    time_taken = sum(time_taken)
    # print(final_cfs)
    # import ipdb; ipdb.set_trace()
    continuous_features = ['Months', 'Credit-amount', 'Insatllment-rate', 'Present-residence-since', 'age', 'Number-of-existing-credits', 'Number-of-people-being-lible']
    immutable_features = ['Personal-status', 'Number-of-people-being-lible', 'Foreign-worker', 'Purpose']
    non_decreasing_features = ['age', 'Job']
    correlated_features = []
    name_dataset = "german"
    normalized_mads = {'Months': 0.17647058823529416, 'Credit-amount': 0.12077693408165513, 'Insatllment-rate': 0.6666666666666665, 'Present-residence-since': 0.6666666666666665, 'age': 0.2500000000000001, 'Number-of-existing-credits': 0.0, 'Number-of-people-being-lible': 0.0}
    method = f"MACE-{name_dataset}-{sys.argv[1]}"
    cfs_found = [True for i in range(final_cfs.shape[0])]

    # cal_metrics.calculate_metrics(method + name_dataset, final_cfs, cfs_found, find_cfs_points, env_.classifier, env_.dataset,
    #         env_.knn, continuous_features, normalized_mads, 
    #         immutable_features, non_decreasing_features, correlated_features, env_.scaler, setting, time_taken, save=False)
    
    cal_metrics.calculate_metrics(method, final_cfs, cfs_found, find_cfs_points, env_.classifier, env_.dataset,
            continuous_features, normalized_mads, 
            immutable_features, non_decreasing_features, correlated_features, env_.scaler, time_taken, save=False)


if __name__ == '__main__':

    # Dummy runs with 10 CFEs each 
    # result_direc = "_experiments/2021.05.25_12.00.15__german_our__lr__one_norm__MACE_eps_1e-3__batch0__samples10__pid0/"
    # result_direc = "_experiments/2021.05.25_11.51.26__german_our__forest__one_norm__MACE_eps_1e-3__batch0__samples10__pid0/"
    
    # Full runs with CFEs generated for all datapoints
    if sys.argv[1] == "LR":
        result_direc = "_experiments/2021.05.25_13.44.19__german_our__lr__one_norm__MACE_eps_1e-3__batch0__samples500__pid0/"
    elif sys.argv[1] == "RF":
        result_direc = "_experiments/2021.05.25_13.45.10__german_our__forest__one_norm__MACE_eps_1e-3__batch0__samples500__pid0/"
    else:
        raise ValueError("Wrong argument")
    assert "german" in result_direc     # currently preparing only for german credit dataset. 
    
    normal, cfes = extract_points_from_files(result_direc + "__explanation_log/")
    time_taken = compute_time(result_direc)

    if "_lr_" in result_direc:
        method = "LR"
    if "_forest_" in result_direc:
        method = "forest"
    if "_mlp_" in result_direc:
        method = "mlp"
    prepare_for_cal_metrics(normal, cfes, method, time_taken)
