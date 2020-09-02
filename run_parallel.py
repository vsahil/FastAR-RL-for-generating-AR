import sys, os
import multiprocessing, subprocess
from itertools import product


def experiment_command(closest_points, dist_lambda):
    os.system(f"python policy_gradient1.py {closest_points} {dist_lambda}")


closest_points = [1, 2, 5, 10]
dist_lambda = [0.01, 0.1, 1, 10, 100, 1000]
pool = multiprocessing.Pool(len(closest_points)*len(dist_lambda))
mr = pool.starmap_async(experiment_command, product(closest_points, dist_lambda))
while not mr.ready():
    sys.stdout.flush()
    mr.wait(0.1)
print("DONE!")
