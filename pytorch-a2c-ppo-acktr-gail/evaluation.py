import numpy as np
import torch
import matplotlib.pyplot as plt
from a2c_ppo_acktr import utils
from a2c_ppo_acktr.envs import make_vec_envs
import sys, time, os
sys.path.append("/scratch/vsahil/RL-for-Counterfactuals/paper_expts/")
# import classifier_german as classifier
debug = True


def plot_trajectories(env_name, env, args, trajectories):
    plt.figure()
    if "step" in env_name:
        plt.axvline(x=5)
        # This plots the step function which is the data manifold
        xs = [env.x0_boundary, env.x1_boundary, env.x1_boundary, env.x2_boundary, env.x2_boundary]
        ys = [0, 0, env.y_height, env.y_height, 0]
        plt.plot(xs, ys)
    elif "sine" in env_name:
        plt.axvline(x=4.7)
        # This plots the sine function which is the data manifold
        xs = np.linspace(-0.5, 10, 201)
        ys = np.sin(xs / 3)
        plt.plot(xs, ys)
    elif "trapezium" in env_name:
        plt.axvline(x=5)
        # This plots the sine function which is the data manifold
        xs = [0, 2, 8, 10]
        ys = [0, 2, 2, 0]
        plt.plot(xs, ys)
    elif "midline" in env_name:
        plt.axvline(x=5)
        plt.scatter([4, 6], [0, 0], color="w")      # this is just for the limits on the x-axis which is not working with xlim
        # plt.plot([4, 0], [6, 0])

    for path in trajectories:
        xs = [i[0] for i in path]
        ys = [i[1] for i in path]
        markers_on = [0]        # the first and last point ais marked with a shape
        plt.plot(xs, ys, '-D', markevery=markers_on)
        markers_on = [-1]
        plt.plot(xs, ys, '-o', markevery=markers_on)  # the end might be circular
        # print(path[0], path[-1], f"see length of trajectory:{len(path)}")
        # print(env.distance_from_manifold(path[0]), env.distance_from_manifold(path[-1]), "see distances")

    plt.xlabel('x1')
    plt.ylabel('y')
    plt.axis('tight')
    lambda_ = env_name.split("v")[-1]
    print(lambda_, args.gamma, args.num_steps, args.lr)
    var = lambda_ + "_" + str(args.gamma) + "_" + str(args.num_steps) + "_" + str(args.lr)
    # plt.savefig(f'plots/{env_name}.png')
    plt.savefig(f'plots/followsine_perpendicular_cont_search_{var}_eval.png')
    # plt.title('Reward type = -(absolute_distance)**2')
    # if args.deter:
        # plt.savefig(f'{args.fig_directory}/trajectories_l2_deter_{args.lr}_{args.gamma}_{args.dist_lambda}_{args.episodes}_{args.max_time}.png')
    # else:
        # plt.savefig(f'{args.fig_directory}/trajectories_l2_nondeter_{args.lr}_{args.gamma}_{args.dist_lambda}_{args.episodes}_{args.max_time}.png')
    print("PLOT DONE")


# remember that currently we are adding up knn distances all along the path, not just final KNN dist.
def return_counterfactual(obs_original, obs, eval_recurrent_hidden_states, eval_masks, actor_critic, eval_envs, device, episode, env_name, args):
    done = False
    steps = 0
    path = [obs_original[0]]
    max_steps = 200
    if args.eval:
        max_steps = 50
    knn_distances = 0
    env_ = eval_envs.venv.venv.envs[0].env
    # print("Starting: ", obs_original)
    # import ipdb; ipdb.set_trace()
    while steps < max_steps:
        with torch.no_grad():
            _, action, _, eval_recurrent_hidden_states = actor_critic.act(
                obs,
                eval_recurrent_hidden_states,
                eval_masks,
                deterministic=True)

        # Obser reward and next obs
        obs, reward, done, infos, obs_original = eval_envs.step(action)
        if ("german" in env_name) or ("adult" in env_name) or ("default" in env_name):
            knn_distances += env_.distance_to_closest_k_points(obs_original)
        # print(obs)
        # original = obs * np.sqrt(ob_rms.var + args.eps) + ob_rms.mean

        eval_masks = torch.tensor(
            [[0.0] if done_ else [1.0] for done_ in done],
            dtype=torch.float32,
            device=device)

        # print(obs_original, "hello", steps, action, reward)
        steps += 1

        if "step" in env_name:
            if obs_original[0][0] >= 5.0:
                done = True
            path.append(obs_original[0])

        if done or steps == max_steps:
            if debug:
                print(f"Episode: {episode}, Steps taken:{steps}")
            # eval_episode_rewards.append(reward)     # append only the last reward. 
            # if not done:
            #     reward = 0
            break
        path.append(obs_original[0])

    # this returns only the last reward, and return the avg knn_distance not sum
    return path, reward, done, knn_distances / len(path)


def german_evaluate(eval_envs, vec_norm, actor_critic):
    # import ipdb; ipdb.set_trace()
    undesirable_x = []
    env_ = eval_envs.venv.venv.envs[0].env
    for no, i in enumerate(env_.dataset.to_numpy()):
        if classifier.predict_single(i, env_.scaler, env_.classifier) == 0: # and i.tolist() == [1, 3, 0, 3, 4, 1]:    # [0, 3, 0, 2, 4, 1]: # [1, 3, 0, 3, 4, 1]:
            undesirable_x.append(tuple(i))
    print(len(undesirable_x), "Total points to run the approach on")
    if len(undesirable_x) == 0:
        return
    # print(undesirable_x)
    successful_transitions = 0
    total_path_len = 0
    knn_dist = 0
    path_lengths = []
    st = time.time()
    for no_, individual in enumerate(undesirable_x):
        transit, path_length, single_knn_dist = return_counterfactual(individual, successful_transitions)
        path_lengths.append(path_length)
        if transit > successful_transitions:
            successful_transitions = transit
            total_path_len += path_length
            knn_dist += (single_knn_dist / path_length)

    try:
        avg_path_len = total_path_len / successful_transitions
        avg_knn_dist = knn_dist / successful_transitions
        print(successful_transitions, len(undesirable_x), avg_path_len, avg_knn_dist)
    except:		# due to zero division error 
        pass

    success_rate = successful_transitions / len(undesirable_x)
    print(success_rate)
    exit(0)


def evaluate(actor_critic, ob_rms, env_name, seed, num_processes, eval_log_dir,
             device, args, env=None):
    # print(env)
    # import ipdb; ipdb.set_trace()
    global debug
    eval_envs = make_vec_envs(env_name, seed + num_processes, num_processes,
                              None, eval_log_dir, device, True)

    vec_norm = utils.get_vec_normalize(eval_envs)
    if vec_norm is not None:
        vec_norm.eval()
        vec_norm.ob_rms = ob_rms

    eval_episode_rewards = []
    episodes = 3
    st = time.time()
    if args.eval and ("german" in env_name) or ("adult" in env_name) or ("default" in env_name):
        episodes = len(eval_envs.venv.venv.envs[0].env.undesirable_x)
    # episodes = 3
    trajectories = []
    knn_dist = 0
    correct = 0
    # import ipdb; ipdb.set_trace()
    for episode in range(episodes):
        if args.eval:
            os.environ['SEQ'] = f"{episode}"
            # debug = False
        obs, obs_original = eval_envs.reset()
        # import ipdb; ipdb.set_trace()
        eval_recurrent_hidden_states = torch.zeros(
            num_processes, actor_critic.recurrent_hidden_state_size, device=device)
        eval_masks = torch.zeros(num_processes, 1, device=device)
        path, reward, done, knn_distances = return_counterfactual(obs_original, obs, eval_recurrent_hidden_states, eval_masks, actor_critic, eval_envs, device, episode, env_name, args)
        if args.eval and ("german" in env_name) or ("adult" in env_name) or ("default" in env_name):
            if done:
                correct += 1
                trajectories.append(len(path))
                knn_dist += knn_distances
        else:
            trajectories.append(path)
        if debug:
            print(reward, knn_distances)
        eval_episode_rewards.append(reward)
    eval_envs.close()
    # import ipdb; ipdb.set_trace()

    if args.eval and ("german" in env_name) or ("adult" in env_name) or ("default" in env_name):
        print(f"Time: {time.time() - st}")
        lambda_ = env_name.split("v")[-1]
        # print(lambda_, args.gamma, args.num_steps, args.lr, args.clip_param)
        # var = lambda_ + "_" + str(args.gamma) + "_" + str(args.num_steps) + "_" + str(args.lr) + "_" + str(args.clip_param)
        var = str(lambda_) + "_" + str(args.gamma) + "_" + str(args.num_steps) + "_" + str(args.lr) + "_" + str(args.clip_param) 

        # with open("correct_german_4.txt", "a") as f:
        # with open("correct_german_sampletrain.txt", "a") as f:
        # with open("correct_german_onehot_sampletrain.txt", "a") as f:
        # with open("correct_german_onehot_contiaction_sampletrain.txt", "a") as f:
        # with open("correct_adult_sampletrain.txt", "a") as f:
        # with open("correct_default_sampletrain.txt", "a") as f:
        #     if correct > 0:
        #         print(f"Setting:{var}, Correct: {correct}, KNN: {knn_dist/correct:.2f}, Path: {np.mean(np.array(trajectories)):.2f}", file=f)
        #     else:
        #         print(f"Setting:{var}, Correct: {correct}, KNN: 0, Path: 0", file=f)

        print(f"Setting:{var}, Correct: {correct}")
        print(f"Avg. KNN Distance: {knn_dist/correct:.2f}")
        if len(trajectories) > 0:
            print(f"Avg. path length: {np.mean(np.array(trajectories)):.2f}")

    # plot_trajectories(env_name, env, args, trajectories)
    # print(eval_episode_rewards, len(eval_episode_rewards))

    print("Evaluation using {} episodes: mean reward {:.5f}\n".format(
            episodes, np.mean(eval_episode_rewards)))
