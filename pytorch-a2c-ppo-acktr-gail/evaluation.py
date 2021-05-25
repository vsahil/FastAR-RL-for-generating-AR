import numpy as np
import torch
import matplotlib.pyplot as plt
from a2c_ppo_acktr import utils
from a2c_ppo_acktr.envs import make_vec_envs
import sys, time, os
import pandas as pd
sys.path.append("/scratch/vsahil/RL-for-Counterfactuals/paper_expts/")
# import classifier_german as classifier
debug = False
debug2 = False
import cal_metrics


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
    path = [obs_original[0].copy()]
    max_steps = 200
    if args.eval:
        max_steps = 50
    knn_distances = 0
    env_ = eval_envs.venv.venv.envs[0].env
    if debug2:
        print("Starting: ", obs_original.shape, list(obs_original))

    while (steps < max_steps) and (not done):
        with torch.no_grad():
            _, action, _, eval_recurrent_hidden_states = actor_critic.act(
                obs,
                eval_recurrent_hidden_states,
                eval_masks,
                deterministic=True)

        # Obser reward and next obs
        obs, reward, done, infos, obs_original = eval_envs.step(action)
        if ("german" in env_name) or ("adult" in env_name) or ("default" in env_name):
            this_knn_dist = env_.distance_to_closest_k_points(obs_original)
            knn_distances += this_knn_dist
            # print(this_knn_dist, len(path), "KNN dist")
        # print(obs)
        # original = obs * np.sqrt(ob_rms.var + args.eps) + ob_rms.mean

        eval_masks = torch.tensor(
            [[0.0] if done_ else [1.0] for done_ in done],
            dtype=torch.float32,
            device=device)

        if debug2:
            print(obs_original.shape, list(obs_original), "hello", steps, action, reward)
        steps += 1

        if "step" in env_name:
            raise NotImplementedError
            if obs_original[0][0] >= 5.0:
                done = True
            path.append(obs_original[0])


        if done or steps == max_steps:
            if debug:
                lambda_ = env_name.split("v")[-1]
                # with open(f"german_num_steps{lambda_}.txt", "a") as f:
                #     print(f"{steps}", file=f)
                print(f"Episode: {episode}, Steps taken:{steps}")
            # eval_episode_rewards.append(reward)     # append only the last reward. 
            # if not done:
            #     reward = 0
            # break

        path.append(obs_original[0].copy())

    # this returns only the last reward, and return the avg knn_distance not sum
    return path, reward, done, knn_distances / len(path)


def evaluate(actor_critic, ob_rms, env_name, seed, num_processes, eval_log_dir,
             device, args, env=None):
    # print(env)

    global debug
    eval_envs = make_vec_envs(env_name, seed + num_processes, num_processes,
                              None, eval_log_dir, device, True)

    vec_norm = utils.get_vec_normalize(eval_envs)
    if vec_norm is not None:
        vec_norm.eval()
        vec_norm.ob_rms = ob_rms

    env_ = eval_envs.venv.venv.envs[0].env
    eval_episode_rewards = []
    episodes = 3
    
    if args.eval and ("german" in env_name) or ("adult" in env_name) or ("default" in env_name):
        episodes = len(env_.undesirable_x)
    find_cfs_points = env_.scaler.transform(env_.undesirable_x[:episodes])
    
    trajectories = []
    knn_dist = 0
    correct = 0
    cfs_found = []
    final_cfs = []
    # import ipdb; ipdb.set_trace()
    # normalized_mads = {}
    # dataset = env_.scaler.transform(env_.dataset)
    # dataset = pd.DataFrame(dataset, columns=env_.dataset.columns.tolist())
    # continuous_features = ['age', 'fnlwgt', 'capitalgain', 'capitalloss', 'hoursperweek']
    # for feature in continuous_features:
    #     normalized_mads[feature] = np.median(abs(dataset[feature].values - np.median(dataset[feature].values)))

    num_datapoints = episodes
    # num_datapoints = 3
    st = time.time()
    # This loop can be parallelized using joblib
    for episode in range(num_datapoints):
        if args.eval:
            os.environ['SEQ'] = f"{episode}"
        obs, obs_original = eval_envs.reset()

        eval_recurrent_hidden_states = torch.zeros(
            num_processes, actor_critic.recurrent_hidden_state_size, device=device)
        eval_masks = torch.zeros(num_processes, 1, device=device)
        path, reward, done, knn_distances = return_counterfactual(obs_original, obs, eval_recurrent_hidden_states, eval_masks, actor_critic, eval_envs, device, episode, env_name, args)
        if args.eval and ("german" in env_name) or ("adult" in env_name) or ("default" in env_name):
            if done:
                try:
                    # assert env_.classifier.predict(path[0].reshape(1, -1))[0] == 0    # This is not true for some datapoints as classifier has changed. 
                    assert env_.classifier.predict(path[-1].reshape(1, -1))[0] == 1     # a valid CFE
                except:
                    import ipdb; ipdb.set_trace()
                correct += 1
                trajectories.append(len(path))
                knn_dist += knn_distances
                cfs_found.append(True)
            else:
                cfs_found.append(False)
            final_cfs.append(path[-1])      # there will be problem if we append boolean here
        else:
            trajectories.append(path)
        
        if episode % 1000 == 0:
            print(episode)

        if debug:
            print(reward, knn_distances, episode)
        
        eval_episode_rewards.append(reward)
    
    eval_envs.close()

    if args.eval and ("german" in env_name) or ("adult" in env_name) or ("default" in env_name):
        method = "our"
        # with open(f"results.txt", "a") as f:
        #     print(f"Setting {var}, {method}: {sum(cfs_found)} : {num_datapoints} : {time.time() - st}", file=f)
        time_taken = time.time() - st
        print(f"Time: {time.time() - st}")
        lambda_ = env_name.split("v")[-1]
        # print(lambda_, args.gamma, args.num_steps, args.lr, args.clip_param)
        # var = lambda_ + "_" + str(args.gamma) + "_" + str(args.num_steps) + "_" + str(args.lr) + "_" + str(args.clip_param)
        var = str(lambda_) + "_" + str(args.gamma) + "_" + str(args.num_steps) + "_" + str(args.lr) + "_" + str(args.clip_param) 

        # with open("correct_german_4.txt", "a") as f:
        # with open("correct_german_sampletrain.txt", "a") as f:
        # with open("correct_german_onehot_sampletrain.txt", "a") as f:
        # with open("correct_german_onehot_contiaction_sampletrain.txt", "a") as f:
        # with open("correct_adult_sampletrain_rerun.txt", "a") as f:
        #     if correct > 0:
        #         print(f"Setting:{var}, Correct: {correct}, KNN: {knn_dist/correct:.2f}, Path: {np.mean(np.array(trajectories)):.2f}", file=f)
        #     else:
        #         print(f"Setting:{var}, Correct: {correct}, KNN: 0, Path: 0", file=f)


        print(f"Setting:{var}, Correct: {correct}")
        print(f"Avg. KNN Distance: {knn_dist/correct:.2f}")
        if len(trajectories) > 0:
            print(f"Avg. path length: {np.mean(np.array(trajectories)):.2f}")

        # import ipdb; ipdb.set_trace()

        dataset = env_.scaler.transform(env_.dataset)
        dataset = pd.DataFrame(dataset, columns=env_.dataset.columns.tolist())
        
        if "adult" in env_name:
            continuous_features = ['age', 'fnlwgt', 'capitalgain', 'capitalloss', 'hoursperweek']
            immutable_features = ['Marital-status', 'Race', 'Native-country', 'Sex']
            non_decreasing_features = ['age', 'education']
            correlated_features = [('education', 'age', 0.054)]     # in normalized data the increase is 0.05
            name_dataset = "adult"
        elif "german" in env_name:
            numerical_features = env_.numerical_features
            continuous_features = env_.dataset.columns[numerical_features].tolist()
            immutable_features = ['Personal-status', 'Number-of-people-being-lible', 'Foreign-worker', 'Purpose']
            non_decreasing_features = ['age', 'Job']
            correlated_features = []
            name_dataset = "german"

        normalized_mads = {}
        for feature in continuous_features:
            normalized_mads[feature] = np.median(abs(dataset[feature].values - np.median(dataset[feature].values)))
        # Adult: {'age': 0.2739726027397258, 'fnlwgt': 0.08197802435899859, 'capitalgain': 0.0, 'capitalloss': 0.0, 'hoursperweek': 0.061224489795918324}
        # German: {'Months': 0.17647058823529416, 'Credit-amount': 0.12077693408165513, 'Insatllment-rate': 0.6666666666666665, 'Present-residence-since': 0.6666666666666665, 'age': 0.2500000000000001, 'Number-of-existing-credits': 0.0, 'Number-of-people-being-lible': 0.0}
        final_cfs = pd.DataFrame(final_cfs, columns=env_.dataset.columns.tolist())
        find_cfs_points = pd.DataFrame(find_cfs_points, columns=env_.dataset.columns.tolist())
        # print(normalized_mads, "see")
        cal_metrics.calculate_metrics(method + name_dataset, final_cfs, cfs_found, find_cfs_points, env_.classifier, env_.dataset,
            env_.knn, continuous_features, normalized_mads, 
            immutable_features, non_decreasing_features, correlated_features, env_.scaler, var, time_taken, save=False)
            

    # plot_trajectories(env_name, env, args, trajectories)
    # print(eval_episode_rewards, len(eval_episode_rewards))

    print("Evaluation using {} episodes: mean reward {:.5f}\n".format(
            episodes, np.mean(eval_episode_rewards)))


# DONE: Change the avg KNN dist to just the final KNN distance as we are doing for the baselines
# DONe: Instead of measuring avg number of steps, measure the proximity-cont and proximity-cat
# DONE: Instead of saying "yes" or "no" to causality - find the percentage of cfes what follow the causal constraints - ours is always 100%
# DONE: Measure sparsity
