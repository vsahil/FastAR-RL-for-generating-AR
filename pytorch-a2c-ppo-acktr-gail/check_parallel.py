import sys, os
import multiprocessing, subprocess
from itertools import product


def experiment_command(setting, removal_percent):
    os.system(f"python -W ignore train_all_permutations.py {setting} {removal_percent}")

# pool = multiprocessing.Pool(120)
# l = [i for i in range(26*5, 51*5)]      # upto 25% removal in steps of 0.2%, from 27 to 40.6 minimum discrimination 
# # mr = pool.map_async(run_command, l)
# settings = [i for i in range(240)]       # for first 120 settings in total
# mr = pool.starmap_async(experiment_command, product(settings, l))
# while not mr.ready():
#     sys.stdout.flush()
#     mr.wait(0.1)
# print("DONE!")


def run_command(problem, lambda_):
    # print(f"python -W ignore main.py --algo ppo --use-gae --lr 2.5e-4 --clip-param 0.1 --value-loss-coef 0.5 --num-processes 1 --num-steps 128 --num-mini-batch 4 --log-interval 10 --use-linear-lr-decay --entropy-coef 0.01 --save-dir './trained_models/ppo/trapezium_slant2_actions' --env-name 'gym_midline:{problem}-v{lambda_}' --num-env-steps 1000000")
    os.system(f"python -W ignore main.py --algo ppo --use-gae --lr 2.5e-4 --clip-param 0.1 --value-loss-coef 0.5 --num-processes 1 --num-steps 128 --num-mini-batch 4 --log-interval 10 --use-linear-lr-decay --entropy-coef 0.01 --save-dir './trained_models/ppo/followstep_conti_actions' --env-name 'gym_midline:{problem}-v{lambda_}' --num-env-steps 1000000")


# problem = ["trapezium"]
problem = ["step"]
lambdas = ["01", "1", "10", "100", "1000"]
pool = multiprocessing.Pool(len(lambdas))
# mr = pool.map_async(run_command, lambdas)
mr = pool.starmap_async(run_command, product(problem, lambdas))
while not mr.ready():
    sys.stdout.flush()
    mr.wait(0.1)
print("DONE!")
