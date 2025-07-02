import numpy as np
import gym
import tensorflow as tf
import utils, td3
from tqdm import tqdm
import matplotlib.pyplot as plt
import pickle

def generate_trajectory(args, env, actor, critic, buffer_list, special_segment_buffer, count, target_reward=4800, noisy_actions=0,  save_image_path=None):

    eval_rew = 0
    eval_len = 0
    early_terminations = 0
    eval_policy_avg = []
    current_state_list = []
    action_list = []
    reward_list = []
    terminal_list = []
    next_frame_list = []
    action_true_list = []

    current_state = env.reset()
    for i in range(5):
        next_frame, reward, terminal, info = env.step(np.clip(np.random.uniform(-1.2, 1.2), -1, 1))
        current_state = next_frame

    for i in range(args.max_eps_len):
        raw_action = actor.get_current_action(np.array([current_state]), verbose=True, noisy=False)
        if noisy_actions > 0:
            action = np.clip(raw_action + np.clip(np.random.normal(loc=0, scale=1, size=raw_action.shape) * noisy_actions, -noisy_actions * 2.5, noisy_actions * 2.5), -1, 1)
        else:
            action = raw_action
        next_frame, reward, terminal, info = env.step(action)

        for i in range(len(buffer_list)):
            buffer_list[i].add(current_state, action, reward, terminal, next_frame)


        current_state_list.append(current_state)
        action_list.append(action)
        reward_list.append(reward)
        terminal_list.append(terminal)
        next_frame_list.append(next_frame)
        action_true_list.append(raw_action)
        # special_segment_buffer.add(current_state, action, reward, terminal, 1, next_frame, -1)

        current_state = next_frame
        eval_len += 1
        eval_rew += reward
        if terminal:
            break
    if eval_rew > target_reward and terminal == False:
        print("Adding ... ")
        for i in range(eval_len):
            special_segment_buffer.add(current_state_list[i], action_list[i], reward_list[i], terminal_list[i], 1, next_frame_list[i], -1, action_true_list[i], 0)
        special_segment_buffer.trajectory_end()
        success = 1
    else:
        success = 0
    return eval_rew, eval_len, np.mean(eval_policy_avg), early_terminations, success



def main():
    args = utils.get_args()
    tf.random.set_seed(args.seed)
    np.random.seed(args.seed)

    env = gym.make(args.env_id)
    env = env.env

    ob_space = env.observation_space
    ac_space = env.action_space
    # print(ac_space, ob_space)
    # quit()

    critic = td3.critic(args, ob_space.shape, ac_space.shape[-1])
    actor = td3.actor(args, ob_space.shape, ac_space.shape[-1])
    tflogger = utils.tensorflowboard_logger(args.log_dir + args.custom_id, args)

    buffer_list = []
    for i in range(args.ensemble_size):
        buffer = utils.ReplayBufferPriority(args, size=1000000)
        buffer_list.append(buffer)

    special_segment_buffer = utils.TrajectorySegmentBuffer_v5(args, size=10000)



    eps_number = 0
    frame_number = 0
    eval_frame = 0
    save_frame = 0
    buffer_save_frame = 0

    if args.env_id == "Humanoid-v2":
        actor.current_model.load_weights(actor.args.checkpoint_dir + "/" + "humanoid_expert_policy" + "/actor_current_network_" + str(0))
        target_reward = 4000
        num_sampling = 5
    elif args.env_id == "Walker2d-v2":
        actor.current_model.load_weights(actor.args.checkpoint_dir + "/" + "walker_expert_policy" + "/actor_current_network_" + str(0))
        num_sampling = 5
        target_reward = 4000
    elif args.env_id == "HalfCheetah-v2":
        actor.current_model.load_weights(actor.args.checkpoint_dir + "/" + "halfcheetah_expert_policy" + "/actor_current_network_" + str(0))
        num_sampling = 8
        target_reward = 8000
    elif args.env_id == "Hopper-v2":
        actor.current_model.load_weights(actor.args.checkpoint_dir + "/" + "hopper_expert_policy" + "/actor_current_network_" + str(0))
        num_sampling = 5
        target_reward = 3000 #
    elif args.env_id == "Ant-v2":
        actor.current_model.load_weights(actor.args.checkpoint_dir + "/" + "ant_expert_policy" + "/actor_current_network_" + str(0))
        num_sampling = 3
        target_reward = 4000
    else:
        print("Error env")
        quit()


    eval_frame = 0
    eval_rews_list = []
    eval_len_list = []
    eval_policy_avg = []
    eval_early_terminations_list = []
    num_success = 0
    while num_success < num_sampling:
        eval_rew, eval_len, avg_policy, early_terminations, success = generate_trajectory(args, env, actor, critic, buffer_list, special_segment_buffer, i,
                                                                                          noisy_actions=0.0, save_image_path=args.log_dir + args.custom_id,
                                                                                          target_reward=target_reward)
        num_success += success
        print("Episode: ", i, eval_rew, "Eval len: ", eval_len)
        eval_rews_list.append(eval_rew)
        eval_len_list.append(eval_len)
        eval_policy_avg.append(avg_policy)
        eval_early_terminations_list.append(early_terminations)
        print(num_success/num_sampling)


    mean_val = np.mean(eval_rews_list)
    std_val = np.std(eval_rews_list)
    low_ci = mean_val - 2.086 * std_val / np.sqrt(20)
    up_ci = mean_val + 2.086 * std_val / np.sqrt(20)
    rew_per_frame = mean_val / np.mean(eval_len_list)
    print("Standard Eval rew: ", mean_val, "Eval len: ", np.mean(eval_len_list))
    tflogger.log_scalar("Evaluation_standard/eps_rew", mean_val, frame_number)
    tflogger.log_scalar("Evaluation_standard/low_ci", low_ci, frame_number)
    tflogger.log_scalar("Evaluation_standard/up_ci", up_ci, frame_number)
    tflogger.log_scalar("Evaluation_standard/rew_per_frame", rew_per_frame, frame_number)
    tflogger.log_scalar("Evaluation_standard/policy_avg", np.mean(eval_policy_avg), frame_number)
    tflogger.log_scalar("Evaluation_standard/early_termination",
                        np.mean(eval_early_terminations_list) / np.mean(eval_len_list), frame_number)


    if args.env_id == "Humanoid-v2":
        pickle.dump(special_segment_buffer, open("../data/original_humanoid_no_noise_initial_cost_data_delay_1_v5_0.pkl", "wb"))
    elif args.env_id == "Walker2d-v2":
        pickle.dump(special_segment_buffer, open("../data/original_walker_no_noise_initial_cost_data_delay_1_v5_0.pkl", "wb"))
    elif args.env_id == "HalfCheetah-v2":
        pickle.dump(special_segment_buffer, open("../data/original_halfcheetah_no_noise_initial_cost_data_delay_1_v5_0.pkl", "wb"))
    elif args.env_id == "Hopper-v2":
        pickle.dump(special_segment_buffer, open("../data/original_hopper_no_noise_initial_cost_data_delay_1_v5_0.pkl", "wb"))
    elif args.env_id == "Ant-v2":
        pickle.dump(special_segment_buffer, open("../data/original_ant_no_noise_initial_cost_data_delay_1_v5_0.pkl", "wb"))
    else:
        print("Error env")
        quit()


if __name__ == "__main__":
    main()