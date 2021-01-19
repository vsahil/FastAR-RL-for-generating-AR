import copy
import glob
import os
import time
from collections import deque

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from a2c_ppo_acktr import algo, utils
from a2c_ppo_acktr.algo import gail
from a2c_ppo_acktr.arguments import get_args
from a2c_ppo_acktr.envs import make_vec_envs
from a2c_ppo_acktr.model import Policy
from a2c_ppo_acktr.storage import RolloutStorage
from evaluation import evaluate


def main():

    args = get_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    # import ipdb; ipdb.set_trace()
    if args.cuda and torch.cuda.is_available() and args.cuda_deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    log_dir = os.path.expanduser(args.log_dir)
    eval_log_dir = log_dir + "_eval"
    utils.cleanup_log_dir(log_dir)
    utils.cleanup_log_dir(eval_log_dir)

    torch.set_num_threads(1)
    device = torch.device("cuda:0" if args.cuda else "cpu")
    st = time.time()
    envs = make_vec_envs(args.env_name, args.seed, args.num_processes,
                         args.gamma, args.log_dir, device, False)

    actor_critic = Policy(
        envs.observation_space.shape,
        envs.action_space,
        base_kwargs={'recurrent': args.recurrent_policy})
    actor_critic.to(device)

    if args.algo == 'a2c':
        agent = algo.A2C_ACKTR(
            actor_critic,
            args.value_loss_coef,
            args.entropy_coef,
            lr=args.lr,
            eps=args.eps,
            alpha=args.alpha,
            max_grad_norm=args.max_grad_norm)
    elif args.algo == 'ppo':
        agent = algo.PPO(
            actor_critic,
            args.clip_param,
            args.ppo_epoch,
            args.num_mini_batch,
            args.value_loss_coef,
            args.entropy_coef,
            lr=args.lr,
            eps=args.eps,
            max_grad_norm=args.max_grad_norm)
    elif args.algo == 'acktr':
        agent = algo.A2C_ACKTR(
            actor_critic, args.value_loss_coef, args.entropy_coef, acktr=True)

    if args.gail:
        assert len(envs.observation_space.shape) == 1
        discr = gail.Discriminator(
            envs.observation_space.shape[0] + envs.action_space.shape[0], 100,
            device)
        file_name = os.path.join(
            args.gail_experts_dir, "trajs_{}.pt".format(
                args.env_name.split('-')[0].lower()))

        expert_dataset = gail.ExpertDataset(
            file_name, num_trajectories=4, subsample_frequency=20)
        drop_last = len(expert_dataset) > args.gail_batch_size
        gail_train_loader = torch.utils.data.DataLoader(
            dataset=expert_dataset,
            batch_size=args.gail_batch_size,
            shuffle=True,
            drop_last=drop_last)

    rollouts = RolloutStorage(args.num_steps, args.num_processes,
                              envs.observation_space.shape, envs.action_space,
                              actor_critic.recurrent_hidden_state_size)

    if args.eval:
        # import ipdb; ipdb.set_trace()
        if args.env_name == "gym_midline:midline-v0":
            # num_episodes = 300        # discrete actions
            num_episodes = 800          # continuous actions
        elif args.env_name in ["gym_midline:step-v01", "gym_midline:step-v1", "gym_midline:step-v10"]:
            num_episodes = 12500    # discrete actions
            num_episodes = 7500     # continuous actions
        elif args.env_name == "gym_midline:step-v100":
            num_episodes = 10500    # discrete actions
            num_episodes = 7500     # continuous actions
        elif args.env_name == "gym_midline:step-v1000":
            num_episodes = 10500    # discrete actions
            num_episodes = 7500     # continuous actions

        elif args.env_name in ["gym_midline:sine-v01", "gym_midline:sine-v1", "gym_midline:sine-v10"]:
            # num_episodes = 28500
            num_episodes = 23000        # perpendicular distance, discrete actions
            num_episodes = 78000        # perpendicular distance, continuous actions
            # num_episodes = 38000        # perpendicular distance, continuous actions
        elif args.env_name in ["gym_midline:sine-v100", "gym_midline:sine-v1000", "gym_midline:sine-v10000"]:
            # num_episodes = 28500
            num_episodes = 23000        # perpendicular distance, discrete actions
            num_episodes = 78000        # perpendicular distance, continuous actions
            # num_episodes = 38000        # perpendicular distance, continuous actions

        elif args.env_name in ["gym_midline:trapezium-v01", "gym_midline:trapezium-v1"]:
            # num_episodes = 18500  # version 1
            # num_episodes = 27500    # version 2
            num_episodes = 7500    # version 3
        elif args.env_name in ["gym_midline:trapezium-v10", "gym_midline:trapezium-v100", "gym_midline:trapezium-v1000"]:
            # num_episodes = 16500  # version 1
            # num_episodes = 27500    # version 2
            num_episodes = 7500    # version 3

        elif args.env_name in ['gym_midline:germanreduced-v01', 'gym_midline:germanreduced-v1', 'gym_midline:germanreduced-v10', 'gym_midline:germanreduced-v100', 'gym_midline:germanreduced-v1000']:
            num_episodes = 7700

        # German 4
        elif 'german4' in args.save_dir and args.env_name in ['gym_midline:german-v01', 'gym_midline:german-v1', 'gym_midline:german-v10', 'gym_midline:german-v100', 'gym_midline:german-v1000']:
            if args.num_steps == 128:
                num_episodes = 156000
            elif args.num_steps == 256:
                num_episodes = 78000
            else:
                raise NotImplementedError

        # German 5 with sampling from training dataset
        elif 'german5' in args.save_dir and 'sampletrain' in args.save_dir and args.env_name in ['gym_midline:german-v0', 'gym_midline:german-v01', 'gym_midline:german-v1', 'gym_midline:german-v10', 'gym_midline:german-v100']:
            if "contiaction" in args.save_dir:  # onehot with continuous actions
                if args.num_steps == 128:
                    num_episodes = 60000     # 234374
                elif args.num_steps == 256:
                    num_episodes = 30000     # 117186
                else:
                    raise NotImplementedError
            else:       # here for vanilla sampletrain and onehot with sampletrain
                if args.num_steps == 128:
                    num_episodes = 39061     # 234374
                elif args.num_steps == 256:
                    num_episodes = 19530     # 117186
                else:
                    raise NotImplementedError

        # German 5
        elif 'german5' in args.save_dir and args.env_name in ['gym_midline:german-v01', 'gym_midline:german-v1', 'gym_midline:german-v10', 'gym_midline:german-v100']:
            if args.num_steps == 128:
                num_episodes = 230000     # 234374
            elif args.num_steps == 256:
                num_episodes = 115000     # 117186
            else:
                raise NotImplementedError

        elif 'try' in args.save_dir and args.env_name in ['gym_midline:german-v01', 'gym_midline:german-v1', 'gym_midline:german-v10', 'gym_midline:german-v100', 'gym_midline:german-v1000']:
            if args.num_steps == 128:
                num_episodes = 0     # 234374
            elif args.num_steps == 256:
                num_episodes = 0     # 117186
            else:
                raise NotImplementedError

        else:
            raise NotImplementedError
        # print(num_episodes, "SEE")
        save_path = os.path.join(args.save_dir, args.algo, args.env_name + f"_{num_episodes}.pt")
        actor_critic, ob_rms = torch.load(save_path)
        actor_critic.eval()
        evaluate(actor_critic, ob_rms, args.env_name, args.seed,
                     args.num_processes, eval_log_dir, device, args, envs.venv.venv.envs[0].env)
        exit(0)
    obs, _ = envs.reset()
    rollouts.obs[0].copy_(obs)
    rollouts.to(device)

    episode_rewards = deque(maxlen=10)

    start = time.time()
    num_updates = int(args.num_env_steps) // args.num_steps // args.num_processes
    t1 = time.time()
    print(t1 - st, "First time")
    for j in range(num_updates):

        if args.use_linear_lr_decay:
            # decrease learning rate linearly
            utils.update_linear_schedule(
                agent.optimizer, j, num_updates,
                agent.optimizer.lr if args.algo == "acktr" else args.lr)
        t2 = time.time()
        for step in range(args.num_steps):
            # Sample actions
            with torch.no_grad():
                value, action, action_log_prob, recurrent_hidden_states = actor_critic.act(
                    rollouts.obs[step], rollouts.recurrent_hidden_states[step],
                    rollouts.masks[step])

            # Obser reward and next obs
            obs, reward, done, infos, _ = envs.step(action)

            for info in infos:
                if 'episode' in info.keys():
                    episode_rewards.append(info['episode']['r'])

            # If done then clean the history of observations.
            masks = torch.FloatTensor(
                [[0.0] if done_ else [1.0] for done_ in done])
            bad_masks = torch.FloatTensor(
                [[0.0] if 'bad_transition' in info.keys() else [1.0]
                 for info in infos])
            rollouts.insert(obs, recurrent_hidden_states, action,
                            action_log_prob, value, reward, masks, bad_masks)

        print(time.time() - t2, "IN LOOP", j)

        with torch.no_grad():
            next_value = actor_critic.get_value(
                rollouts.obs[-1], rollouts.recurrent_hidden_states[-1],
                rollouts.masks[-1]).detach()

        if args.gail:
            if j >= 10:
                envs.venv.eval()

            gail_epoch = args.gail_epoch
            if j < 10:
                gail_epoch = 100  # Warm up
            for _ in range(gail_epoch):
                discr.update(gail_train_loader, rollouts,
                             utils.get_vec_normalize(envs)._obfilt)

            for step in range(args.num_steps):
                rollouts.rewards[step] = discr.predict_reward(
                    rollouts.obs[step], rollouts.actions[step], args.gamma,
                    rollouts.masks[step])

        rollouts.compute_returns(next_value, args.use_gae, args.gamma,
                                 args.gae_lambda, args.use_proper_time_limits)

        value_loss, action_loss, dist_entropy = agent.update(rollouts)

        rollouts.after_update()

        # save for every interval-th episode or for the last epoch
        if (j % args.save_interval == 0
                or j == num_updates - 1) and args.save_dir != "":
            save_path = os.path.join(args.save_dir, args.algo)
            try:
                os.makedirs(save_path)
            except OSError:
                pass

            torch.save([
                actor_critic,
                getattr(utils.get_vec_normalize(envs), 'ob_rms', None)
            ], os.path.join(save_path, args.env_name + f"_{j}.pt"))

        if j % args.log_interval == 0 and len(episode_rewards) > 1:
            total_num_steps = (j + 1) * args.num_processes * args.num_steps
            end = time.time()
            print(
                "Updates {}, num timesteps {}, FPS {} \n Last {} training episodes: mean/median reward {:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f}\n"
                .format(j, total_num_steps,
                        int(total_num_steps / (end - start)),
                        len(episode_rewards), np.mean(episode_rewards),
                        np.median(episode_rewards), np.min(episode_rewards),
                        np.max(episode_rewards), dist_entropy, value_loss,
                        action_loss))

        if (args.eval_interval is not None and len(episode_rewards) > 1
                and j % args.eval_interval == 0):
            ob_rms = utils.get_vec_normalize(envs).ob_rms
            evaluate(actor_critic, ob_rms, args.env_name, args.seed,
                     args.num_processes, eval_log_dir, device, args)


if __name__ == "__main__":
    main()
