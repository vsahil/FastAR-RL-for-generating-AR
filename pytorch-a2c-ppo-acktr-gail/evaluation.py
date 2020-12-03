import numpy as np
import torch
import matplotlib.pyplot as plt
from a2c_ppo_acktr import utils
from a2c_ppo_acktr.envs import make_vec_envs


def plot_trajectories(env_name, env, trajectories):
    plt.figure()
    plt.axvline(x=5)
    if "step" in env_name:
        # This plots the step function which is the data manifold
        xs = [env.x0_boundary, env.x1_boundary, env.x1_boundary, env.x2_boundary, env.x2_boundary]
        ys = [0, 0, env.y_height, env.y_height, 0]
        plt.plot(xs, ys)
    elif "sine" in env_name:
        # This plots the sine function which is the data manifold
        xs = np.linspace(-0.5, 10, 201)
        ys = np.sin(xs / 3)
        plt.plot(xs, ys)
    elif "trapezium" in env_name:
        # This plots the sine function which is the data manifold
        xs = [0, 2, 8, 10]
        ys = [0, 2, 2, 0]
        plt.plot(xs, ys)
    elif "midline" in env_name:
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
    plt.savefig(f'plots/{env_name}.png')
    # plt.title('Reward type = -(absolute_distance)**2')
    # if args.deter:
        # plt.savefig(f'{args.fig_directory}/trajectories_l2_deter_{args.lr}_{args.gamma}_{args.dist_lambda}_{args.episodes}_{args.max_time}.png')
    # else:
        # plt.savefig(f'{args.fig_directory}/trajectories_l2_nondeter_{args.lr}_{args.gamma}_{args.dist_lambda}_{args.episodes}_{args.max_time}.png')
    print("PLOT DONE")


def evaluate(actor_critic, ob_rms, env_name, seed, num_processes, eval_log_dir,
             device, args, env=None):
    eval_envs = make_vec_envs(env_name, seed + num_processes, num_processes,
                              None, eval_log_dir, device, True)

    vec_norm = utils.get_vec_normalize(eval_envs)
    if vec_norm is not None:
        vec_norm.eval()
        vec_norm.ob_rms = ob_rms

    eval_episode_rewards = []
    episodes = 10
    trajectories = []
    for episode in range(episodes):
        obs, obs_original = eval_envs.reset()
        eval_recurrent_hidden_states = torch.zeros(
            num_processes, actor_critic.recurrent_hidden_state_size, device=device)
        eval_masks = torch.zeros(num_processes, 1, device=device)
        done = False
        steps = 0
        path = [obs_original[0]]
        max_steps = 100
        print("Starting: ", obs_original)
        while steps < max_steps:
            with torch.no_grad():
                _, action, _, eval_recurrent_hidden_states = actor_critic.act(
                    obs,
                    eval_recurrent_hidden_states,
                    eval_masks,
                    deterministic=True)

            # Obser reward and next obs
            obs, reward, done, infos, obs_original = eval_envs.step(action)
            # print(obs)
            # original = obs * np.sqrt(ob_rms.var + args.eps) + ob_rms.mean

            eval_masks = torch.tensor(
                [[0.0] if done_ else [1.0] for done_ in done],
                dtype=torch.float32,
                device=device)

            print(obs_original, "hello", steps, action, reward)
            steps += 1

            if done or steps == max_steps:
                print(steps, episode, "Complete")
                eval_episode_rewards.append(reward)     # append only the last reward. 
                break
            path.append(obs_original[0])
            # for info in infos:
            #     if 'episode' in info.keys():
            #         eval_episode_rewards.append(info['episode']['r'])
            # print(steps, episode, "Complete")
        trajectories.append(path)
    eval_envs.close()
    # import ipdb; ipdb.set_trace()
    plot_trajectories(env_name, env, trajectories)
    print(eval_episode_rewards)
    # with open(f"{env_name}_results.txt", "a") as f:
    print("Evaluation using {} episodes: mean reward {:.5f}\n".format(
            episodes, np.mean(eval_episode_rewards)))
