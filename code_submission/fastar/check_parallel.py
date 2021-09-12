import sys, os
import multiprocessing, subprocess
from itertools import product

# problem_set = ["adult"]
problem_set = ["german"]
lambda_set = ["0", "01", "1"]   #, "10", "100"]
gamma_set = [0.99]
clip_set = [0.05, 0.1, 0.2]
num_steps_set = [128, 256]
lr_set = [1e-3, 1e-4]


def run_command(problem, lambda_, gamma, num_steps, lr, clip):
    var = lambda_ + "_" + str(gamma) + "_" + str(num_steps) + "_" + str(lr) + "_" + str(clip)
    total = [i for i in product(problem_set, lambda_set, gamma_set, num_steps_set, lr_set, clip_set)]
    this_instance = (problem, lambda_, gamma, num_steps, lr, clip)
    index = total.index(this_instance) + 10
    # print(this_instance in total, total.index(this_instance))
    # print(this_instance, total)
    # os.system(f"python -W ignore main.py --algo ppo --use-gae --lr 2.5e-4 --clip-param 0.1 --value-loss-coef 0.5 --num-processes 1 --num-steps 128 --num-mini-batch 4 --log-interval 10 --use-linear-lr-decay --entropy-coef 0.01 --save-dir './trained_models/ppo/trapezium_slant3_actions' --env-name 'gym_midline:{problem}-v{lambda_}' --num-env-steps 1000000")
    # os.system(f"python -W ignore main.py --algo ppo --use-gae --lr 2.5e-4 --clip-param 0.1 --value-loss-coef 0.5 --num-processes 1 --num-steps 128 --num-mini-batch 4 --log-interval 10 --use-linear-lr-decay --entropy-coef 0.01 --save-dir './trained_models/ppo/followstep_conti3_actions' --env-name 'gym_midline:{problem}-v{lambda_}' --num-env-steps 1000000")
    # os.system(f"python -W ignore main.py --algo ppo --use-gae --lr {lr} --clip-param 0.1 --value-loss-coef 0.5 --num-processes 1 --num-steps {num_steps} --num-mini-batch 4 --log-interval 10 --use-linear-lr-decay --entropy-coef 0.01 --save-dir './trained_models/ppo/followsine_perpendicular_cont_search_{var}' --env-name 'gym_midline:{problem}-v{lambda_}' --num-env-steps 10000000 --save-interval 2000 --gamma {gamma} --eval")

    # German 4 is the setting with constraints on immutable features, age, and job. No lambda role yet. 
    # var = lambda_ + "_" + str(gamma) + "_" + str(num_steps) + "_" + str(lr) + "_" + str(clip) 
    # os.system(f"python -W ignore main.py --algo ppo --use-gae --lr {lr} --clip-param {clip} --value-loss-coef 0.5 --num-processes 1 --num-steps {num_steps} --num-mini-batch 4 --log-interval 50 --use-linear-lr-decay --entropy-coef 0.01 --save-dir './trained_models/ppo/german4_search_{var}' --env-name 'gym_midline:{problem}-v{lambda_}' --num-env-steps 5000000 --save-interval 5000 --gamma {gamma} --eval")

    # German 5 is the setting with constraints on immutable features, age, and job, with lambda. 
    os.system(f"taskset -c {index} python -W ignore main.py --algo ppo --use-gae --lr {lr} --clip-param {clip} --value-loss-coef 0.5 --num-processes 1 --num-steps {num_steps} --num-mini-batch 4 --log-interval 50 --use-linear-lr-decay --entropy-coef 0.01 --save-dir './temp_trained/ppo/german5_sampletrain_search_{var}' --env-name 'gym_midline:{problem}-v{lambda_}' --num-env-steps 5000000 --save-interval 500 --gamma {gamma} --eval-interval 500")
    # os.system(f"python -W ignore main.py --algo ppo --use-gae --lr {lr} --clip-param {clip} --value-loss-coef 0.5 --num-processes 1 --num-steps {num_steps} --num-mini-batch 4 --log-interval 50 --use-linear-lr-decay --entropy-coef 0.01 --save-dir './trained_models/ppo/german5_onehot_sampletrain_search_{var}' --env-name 'gym_midline:{problem}-v{lambda_}' --num-env-steps 5000000 --save-interval 5000 --gamma {gamma}")
    # os.system(f"python -W ignore main.py --algo ppo --use-gae --lr {lr} --clip-param {clip} --value-loss-coef 0.5 --num-processes 1 --num-steps {num_steps} --num-mini-batch 4 --log-interval 50 --use-linear-lr-decay --entropy-coef 0.01 --save-dir './trained_models/ppo/german5_onehot_contiaction_sampletrain_search_{var}' --env-name 'gym_midline:{problem}-v{lambda_}' --num-env-steps 10000000 --save-interval 5000 --gamma {gamma} --eval")

    # Adult
    # print(f"taskset -c {index} python -W ignore main.py --algo ppo --use-gae --lr {lr} --clip-param {clip} --value-loss-coef 0.5 --num-processes 1 --num-steps {num_steps} --num-mini-batch 4 --log-interval 50 --use-linear-lr-decay --entropy-coef 0.01 --save-dir './trained_models/ppo/adult_sampletrain_search_{var}' --env-name 'gym_midline:{problem}-v{lambda_}' --num-env-steps 10000000 --save-interval 5000 --gamma {gamma} --eval")
    # Default
    # print(f"taskset -c {index} python -W ignore main.py --algo ppo --use-gae --lr {lr} --clip-param {clip} --value-loss-coef 0.5 --num-processes 1 --num-steps {num_steps} --num-mini-batch 4 --log-interval 50 --use-linear-lr-decay --entropy-coef 0.01 --save-dir './trained_models/ppo/default_sampletrain_search_{var}' --env-name 'gym_midline:{problem}-v{lambda_}' --num-env-steps 10000000 --save-interval 5000 --gamma {gamma} --eval")


# problem = ["trapezium"]
# problem = ["step"]
# problem = ["sine"]
# problem = ["german"]
# lambdas = ["01", "1", "10", "100", "1000", "10000"]

# lambdas = ["01"]
# gamma = [0.99]
# num_steps = [128, 256]
# lr = [1e-3, 1e-4]

# pool = multiprocessing.Pool(len(lambdas) * len(gamma) * len(num_steps) * len(lr))
pool = multiprocessing.Pool(len(clip_set) * len(gamma_set) * len(num_steps_set) * len(lr_set) * len(lambda_set))
# mr = pool.map_async(run_command, lambdas)
mr = pool.starmap_async(run_command, product(problem_set, lambda_set, gamma_set, num_steps_set, lr_set, clip_set))
while not mr.ready():
    sys.stdout.flush()
    mr.wait(0.1)
print("DONE!")
