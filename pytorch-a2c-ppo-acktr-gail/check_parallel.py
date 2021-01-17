import sys, os
import multiprocessing, subprocess
from itertools import product


def run_command(problem, lambda_, gamma, num_steps, lr, clip):
    var = lambda_ + "_" + str(gamma) + "_" + str(num_steps) + "_" + str(lr) + "_" + str(clip)
    # os.system(f"python -W ignore main.py --algo ppo --use-gae --lr 2.5e-4 --clip-param 0.1 --value-loss-coef 0.5 --num-processes 1 --num-steps 128 --num-mini-batch 4 --log-interval 10 --use-linear-lr-decay --entropy-coef 0.01 --save-dir './trained_models/ppo/trapezium_slant3_actions' --env-name 'gym_midline:{problem}-v{lambda_}' --num-env-steps 1000000")
    # os.system(f"python -W ignore main.py --algo ppo --use-gae --lr 2.5e-4 --clip-param 0.1 --value-loss-coef 0.5 --num-processes 1 --num-steps 128 --num-mini-batch 4 --log-interval 10 --use-linear-lr-decay --entropy-coef 0.01 --save-dir './trained_models/ppo/followstep_conti3_actions' --env-name 'gym_midline:{problem}-v{lambda_}' --num-env-steps 1000000")
    # os.system(f"python -W ignore main.py --algo ppo --use-gae --lr {lr} --clip-param 0.1 --value-loss-coef 0.5 --num-processes 1 --num-steps {num_steps} --num-mini-batch 4 --log-interval 10 --use-linear-lr-decay --entropy-coef 0.01 --save-dir './trained_models/ppo/followsine_perpendicular_cont_search_{var}' --env-name 'gym_midline:{problem}-v{lambda_}' --num-env-steps 10000000 --save-interval 2000 --gamma {gamma} --eval")

    # German 4 is the setting with constraints on immutable features, age, and job. No lambda role yet. 
    # var = lambda_ + "_" + str(gamma) + "_" + str(num_steps) + "_" + str(lr) + "_" + str(clip) 
    # os.system(f"python -W ignore main.py --algo ppo --use-gae --lr {lr} --clip-param {clip} --value-loss-coef 0.5 --num-processes 1 --num-steps {num_steps} --num-mini-batch 4 --log-interval 50 --use-linear-lr-decay --entropy-coef 0.01 --save-dir './trained_models/ppo/german4_search_{var}' --env-name 'gym_midline:{problem}-v{lambda_}' --num-env-steps 5000000 --save-interval 5000 --gamma {gamma} --eval")

    # German 5 is the setting with constraints on immutable features, age, and job, with lambda. 
    # os.system(f"python -W ignore main.py --algo ppo --use-gae --lr {lr} --clip-param {clip} --value-loss-coef 0.5 --num-processes 1 --num-steps {num_steps} --num-mini-batch 4 --log-interval 50 --use-linear-lr-decay --entropy-coef 0.01 --save-dir './trained_models/ppo/german5_sampletrain_search_{var}' --env-name 'gym_midline:{problem}-v{lambda_}' --num-env-steps 5000000 --save-interval 5000 --gamma {gamma}")
    # os.system(f"python -W ignore main.py --algo ppo --use-gae --lr {lr} --clip-param {clip} --value-loss-coef 0.5 --num-processes 1 --num-steps {num_steps} --num-mini-batch 4 --log-interval 50 --use-linear-lr-decay --entropy-coef 0.01 --save-dir './trained_models/ppo/german5_onehot_sampletrain_search_{var}' --env-name 'gym_midline:{problem}-v{lambda_}' --num-env-steps 5000000 --save-interval 5000 --gamma {gamma}")
    os.system(f"python -W ignore main.py --algo ppo --use-gae --lr {lr} --clip-param {clip} --value-loss-coef 0.5 --num-processes 1 --num-steps {num_steps} --num-mini-batch 4 --log-interval 50 --use-linear-lr-decay --entropy-coef 0.01 --save-dir './trained_models/ppo/german5_onehot_contiaction_sampletrain_search_{var}' --env-name 'gym_midline:{problem}-v{lambda_}' --num-env-steps 10000000 --save-interval 5000 --gamma {gamma} --eval")


# problem = ["trapezium"]
# problem = ["step"]
# problem = ["sine"]
problem = ["german"]
# lambdas = ["01", "1", "10", "100", "1000", "10000"]
lambdas = ["0", "01", "1", "10", "100"]
gamma = [0.99]
clip = [0.05, 0.1, 0.2]
num_steps = [128, 256]
lr = [1e-3, 1e-4, 1e-5]

# lambdas = ["01"]
# gamma = [0.99]
# num_steps = [128, 256]
# lr = [1e-3, 1e-4]

# pool = multiprocessing.Pool(len(lambdas) * len(gamma) * len(num_steps) * len(lr))
pool = multiprocessing.Pool(len(clip) * len(gamma) * len(num_steps) * len(lr) * len(lambdas))
# mr = pool.map_async(run_command, lambdas)
mr = pool.starmap_async(run_command, product(problem, lambdas, gamma, num_steps, lr, clip))
while not mr.ready():
    sys.stdout.flush()
    mr.wait(0.1)
print("DONE!")
