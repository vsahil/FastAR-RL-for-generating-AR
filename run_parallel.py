import sys, os
import multiprocessing, subprocess
from itertools import product

def experiment_command2(lr, gamma, episodes, max_time, fig_directory):
    # fig_directory = 'plots/try_follow_straightline2'
    os.system(f"python follow_straightline_snake.py --lr {lr} --gamma {gamma} --episodes {episodes} --max_time {max_time} --fig_directory {fig_directory}")


# lr = [0.001, 0.005, 0.01, 0.05]
lr = [0.01, 0.05]
# gamma = [1.0, 0.999, 0.99]
gamma = [0.99]
# episodes = [500, 1000, 2000, 3000]
episodes = [2000, 3000]
max_time = [1000, 2000, 4000]
fig_directory = ['plots/try_follow_straightline_linear6']
pool = multiprocessing.Pool(96)
mr = pool.starmap_async(experiment_command2, product(lr, gamma, episodes, max_time, fig_directory))
# print("check fig directory")
# exit(0)
while not mr.ready():
    sys.stdout.flush()
    mr.wait(0.1)
print("DONE!")
