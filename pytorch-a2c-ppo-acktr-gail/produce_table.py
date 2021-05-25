import numpy as np
import os 

numbers = {"our":{}}    #, "genetic":{}, "kdtree":{}}
directory = "saved_files/"

metrics = ["validity", "prox-cont", "prox-cat", "manifolddist", "sparsity", "causality", "time"]

with open("results.txt", "r") as f:
    res = f.readlines()

total_points = 0

for r in res:
    x = r.split(":")
    method = x[0]
    cfe_found = int(x[1].strip())
    total_cfes = int(x[2].strip())
    if total_points > 0:
        assert total_cfes == total_points
    else:
        total_points = total_cfes
    time = float(x[3].strip())
    validity = round(cfe_found * 100.0 / total_points, 3)
    time_ = round(time, 2)
    # print(method, cfe_found, total_points, time, validity, time_)
    numbers[method]["validity"] = validity
    numbers[method]["time"] = time_

# print(numbers)

for file in os.listdir(directory):
    if "avg" in file:
        assert ".npy" == file[-4:]
        data = np.load(directory + file)
        x = file[:-4].split("_")
        method = x[0]
        metric = x[2] if len(x) == 3 else x[2] + "-" + x[3]
        if data.shape[0] > 0:
            value = np.mean(data)
        else:
            value = 0
        # print(file[:-4], data.shape, x, method, metric, value)
        if metric == "causality":
            value = 100.0 * value
            # if data.shape[0] > 0:
            #     print(method, sum(data)/data.shape[0])
        numbers[method][metric] = value

for key in numbers:
    print(key, numbers[key], end="\n")
