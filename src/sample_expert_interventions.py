import numpy as np
import gym
import tensorflow as tf
import utils, td3, core_network
import pickle
import json
from utils import error_function
import evaluation
import core_network


def sample_episode(args, env, cost_buffer, actor, expert, count, intervention_threshold, policy_noise, error_function,
                   supervision_window=30, preintervention_interval=30, expert_intervention_length=30,
                   expert_interventions=True):
    """
    Samples delayed intervention trajectory
    Trajectory delay depends on intervention threshold
    """
    eps_rew = 0
    eps_len = 0
    intervention_steps = 0
    intervention_freq = 0
    ensemble_uncertainty_list = []

    state_list = []
    action_list = []
    next_state_list = []
    termination_list = []
    reward_list = []
    preintervention_status_list = []
    expert_intervention = []

    expert_steps = 0
    expert_steps_rewards = 0
    current_state = env.reset()

    cost_list = []
    for i in range(preintervention_interval):
        cost_list.append(0)

    threshold_cost_list = []
    segment_threshold_list = []
    expert_action_list = []
    preint_threshold_list = []
    intervention_delay_list = []
    early_intervention_flag = False
    segment_label_list = []

    for step in range(args.max_eps_len):
        action, action_std = actor.get_current_action(np.array([current_state]), noisy=False)
        expert_action = expert.get_current_action(np.array([current_state]), noisy=False)
        noise = np.clip(np.random.normal(size=action.shape) * policy_noise, -policy_noise * 2.5,
                        policy_noise * 2.5)
        noisy_action = np.clip(noise + action, -1, 1)

        if expert_steps > 0:
            noisy_executed_action = expert_action
        else:
            noisy_executed_action = noisy_action
        ensemble_uncertainty_list.append(action_std)

        expert_cost, _ = error_function(expert_action, noisy_executed_action)
        cost_list.append(expert_cost)
        threshold_cost_list.append(np.sum(cost_list[-supervision_window:]))

        if expert_steps > 0:
            preintervention_status_list.append(-1)
            expert_intervention.append(1)
            intervention_steps += 1
        else:
            preintervention_status_list.append(0)
            expert_intervention.append(0)

        next_frame, reward, terminal, _ = env.step(noisy_executed_action)
        eps_len += 1
        eps_rew += reward

        reward_list.append(reward)
        expert_action_list.append(expert_action)
        state_list.append(current_state)
        action_list.append(noisy_executed_action)
        termination_list.append(terminal)
        next_state_list.append(next_frame)

        if expert_steps > 0:
            expert_steps -= 1
            expert_steps_rewards += reward

        current_state = next_frame

        segment_threshold_list.append(np.sum(cost_list[-supervision_window:]))
        if ((np.sum(cost_list[
                    -supervision_window:]) >= intervention_threshold)) and expert_steps <= 0 and expert_interventions:

            if step < args.segment_length - 1:
                early_intervention_flag = True
            preint_threshold_list.append(np.sum(cost_list[-supervision_window:]))
            expert_steps = expert_intervention_length
            expert_steps_rewards = 0
            intervention_freq += 1

            for i in range(preintervention_interval):
                if step - i >= 0:
                    preintervention_status_list[step - i] = 1
                else:
                    break;

        if step >= args.segment_length - 1:
            if np.mean(expert_intervention[-supervision_window:]) == 1:
                segment_label = 0
            elif ((step == args.segment_length - 1) and early_intervention_flag) or \
                    (expert_steps == expert_intervention_length):
                segment_label = 2
                early_intervention_flag = False
            elif expert_intervention[-1] == 0 and expert_steps != expert_intervention_length:
                segment_label = 1
            else:
                segment_label = -1
            segment_label_list.append(segment_label)

        if terminal or (eps_len > 100 and np.mean(reward_list[-100:]) < 0):
            break

    if eps_len >= args.segment_length and eps_rew > 100:
        for i in range(supervision_window):
            reward_list.append(reward_list[-1])

        for i in range(eps_len):
            cost_buffer.add(state_list[i], action_list[i], reward_list[i], termination_list[i], expert_intervention[i],
                            next_state_list[i], preintervention_status_list[i], expert_action_list[i],
                            intervention_threshold)

        for i in range(len(segment_label_list)):
            segment_estimated_cost = 0
            selected_indices = []
            for k in range(args.segment_length):
                target_index = i + k
                segment_estimated_cost += cost_list[target_index + preintervention_interval]

                expert_cost, _ = error_function(action_list[target_index], expert_action_list[
                    target_index])
                if np.abs(expert_cost - cost_list[target_index + preintervention_interval]) > 0.01:
                    print(target_index, "Error, Cost Error", expert_cost,
                          cost_list[target_index + preintervention_interval])
                    quit()

                selected_indices.append(target_index)

            if segment_label_list[i] == 2:
                if segment_estimated_cost < intervention_threshold:
                    print(i + args.segment_length - 1, "Preint Segment Error ... !", segment_estimated_cost,
                          intervention_threshold)
                    quit()

            elif segment_label_list[i] == 0:
                if segment_estimated_cost > 0:
                    print(i + args.segment_length - 1, "Expert Segment Error ... !", segment_estimated_cost,
                          intervention_threshold)
                    quit()
            elif segment_label_list[i] == 1:
                if segment_estimated_cost >= intervention_threshold:
                    print(i + args.segment_length - 1, "Inbetween Segment Error ... !", segment_estimated_cost,
                          intervention_threshold)
                    quit()

            cost_buffer.add_segment(np.array(selected_indices), 0, intervention_threshold)
            if segment_label_list[i] != -1:
                cost_buffer.add_segment(np.array(selected_indices), segment_label_list[i] + 1, intervention_threshold)
        cost_buffer.trajectory_end()
        print(eps_len, eps_rew, intervention_freq, intervention_steps, np.amax(threshold_cost_list),
              intervention_threshold)
        return eps_rew, eps_len, intervention_steps, intervention_freq, ensemble_uncertainty_list, intervention_delay_list,
    return eps_rew, eps_len, 0, 0, ensemble_uncertainty_list, intervention_delay_list


def main():
    args = utils.get_args()
    tf.random.set_seed(args.seed)
    np.random.seed(args.seed)

    env = gym.make(args.env_id)
    env = env.env

    ob_space = env.observation_space
    ac_space = env.action_space

    global global_error_scale_var
    global_error_scale_var = args.error_function_cap
    expert_policy = td3.actor(args, ob_space.shape, ac_space.shape[-1])
    intervention_thresholds_list = json.loads(args.intervention_thresholds)

    eval_policy_noise = 0.2
    target_sampled_expert_steps = 800
    if args.env_id == "Humanoid-v2":
        env_name = "humanoid"
    elif args.env_id == "Walker2d-v2":
        env_name = "walker"
    elif args.env_id == "HalfCheetah-v2":
        env_name = "halfcheetah"
    elif args.env_id == "Hopper-v2":
        env_name = "hopper"
        eval_policy_noise = 0.1
    elif args.env_id == "Ant-v2":
        env_name = "ant"
    else:
        print("Error env")
        quit()
    for i in range(len(intervention_thresholds_list)):
        intervention_thresholds_list[i] = int(intervention_thresholds_list[i] * args.segment_length / 30)

    expert_policy.current_model.load_weights(expert_policy.args.checkpoint_dir + "/" + env_name + "_expert_policy" + "/actor_current_network_" + str(0))
    old_cost_buffer = pickle.load(open("../data/original_" + env_name + "_no_noise_initial_cost_data_delay_1_v5_0.pkl", "rb"))
    cost_buffer = utils.TrajectorySegmentBuffer_v5(args, size=10000)
    cost_buffer.reinit(old_cost_buffer, error_function)

    cost_buffer_validation = utils.TrajectorySegmentBuffer_v5(args, size=10000)
    cost_buffer_validation.reinit(old_cost_buffer, error_function)


    print(cost_buffer.raw_data_count)
    print(cost_buffer.expert_single_state_count)

    imitation_policy = core_network.actor(args, ob_space.shape, ac_space.shape[-1])

    tflogger = utils.tensorflowboard_logger(args.log_dir + args.custom_id, args)
    frame_number = 0
    pretrain_steps = int(100000 * (1 + args.cost_loss_bias))
    total_expert_intervention = 0

    bc_loss_list = []
    intervention_freq_list = []
    sampled_length_list = []
    average_delay_list = []
    total_expert_intervention = 0
    total_sampled_steps = 0

    initial_imitation_policy = core_network.actor(args, ob_space.shape, ac_space.shape[-1])

    if args.train_initial_model:
        for i in range(100000):
            bc_loss = initial_imitation_policy.pretrain_step_actor(cost_buffer)
            bc_loss_list.append(bc_loss)
            if i % 10000 == 0 and i > 0:
                print(i / 100000, np.mean(bc_loss_list[-10000]))
                simple_eval_rew_list = []
                simple_eval_len_list = []
                for j in range(10):
                    eval_rew, eval_len = evaluation.simple_evaluation(args, env, initial_imitation_policy, noisy_actions=0)
                    simple_eval_rew_list.append(eval_rew)
                    simple_eval_len_list.append(eval_len)
                print(i / 100000, np.mean(simple_eval_rew_list), np.mean(simple_eval_len_list))

        tflogger.log_scalar("BC_loss", np.mean(bc_loss), frame_number)
        # Load a pretrained imitation policy ...
        initial_imitation_policy.current_model_list[0].save_weights(
            "../data/" + env_name + "_no_noise_current_actor_network_0_v2_" + str(0))

    initial_imitation_policy.current_model_list[0].load_weights(
        "../data/" + env_name + "_no_noise_current_actor_network_0_v2_" + str(0))
    # simple_eval_rew_list = []
    # simple_eval_len_list = []
    for j in range(len(intervention_thresholds_list)):
        collected_data = 0
        count = 0
        while collected_data < target_sampled_expert_steps:
            sampled_rew, sampled_len, expert_intervention_steps, intervention_freq, ensemble_uncertainty, average_delay = sample_episode(
                args, env,
                cost_buffer_validation,
                initial_imitation_policy,
                expert_policy, count,
                intervention_thresholds_list[j],
                policy_noise=eval_policy_noise, preintervention_interval=args.segment_length)
            intervention_freq_list.append(intervention_freq)
            sampled_length_list.append(sampled_len)
            collected_data += expert_intervention_steps
            total_sampled_steps += sampled_len
            count += 1
            average_delay_list.extend(average_delay)
            print("Progress: ", intervention_thresholds_list[j], collected_data / target_sampled_expert_steps,
                  np.sum(intervention_freq_list))

        collected_data = 0
        count = 0

        while collected_data < target_sampled_expert_steps:
            sampled_rew, sampled_len, expert_intervention_steps, intervention_freq, ensemble_uncertainty, average_delay = sample_episode(
                args, env,
                cost_buffer,
                initial_imitation_policy,
                expert_policy, count,
                intervention_thresholds_list[j],
                policy_noise=eval_policy_noise,
                preintervention_interval=args.segment_length)
            intervention_freq_list.append(intervention_freq)
            sampled_length_list.append(sampled_len)
            collected_data += expert_intervention_steps
            total_sampled_steps += sampled_len
            count += 1
            average_delay_list.extend(average_delay)
            print("Progress Expert: ", intervention_thresholds_list[j],
                  collected_data / target_sampled_expert_steps, np.sum(intervention_freq_list))

            # if expert_intervention_steps % args.segment_length != 0 and sampled_len < args.max_eps_len:
            #     raise ("Error here!")

            if expert_intervention_steps > 0:
                for i in range(3000 + int(15000 * expert_intervention_steps / target_sampled_expert_steps)):
                    bc_loss = initial_imitation_policy.pretrain_step_actor(cost_buffer)
                # if i % 1000 == 0 and i > 0:
                #     print(i/20000)
            # simple_eval_rew_list = []
            # simple_eval_len_list = []
            # for j in range(10):
            #     eval_rew, eval_len = simple_evaluation(args, env, initial_imitation_policy, noisy_actions=0)
            #     simple_eval_rew_list.append(eval_rew)
            #     simple_eval_len_list.append(eval_len)
            # print(i/2000, "Rew:", np.mean(simple_eval_rew_list), "Length: ", np.mean(simple_eval_len_list))

        print("Retraining ... ", j / len(intervention_thresholds_list))

        print(cost_buffer_validation.preintervention_segment_count)
        print(cost_buffer_validation.raw_data_count)
        print(cost_buffer_validation.expert_single_state_count)

        print(cost_buffer.preintervention_segment_count)
        print(cost_buffer.raw_data_count)
        print(cost_buffer.expert_single_state_count)

        # new_cost_buffer_validation = utils.TrajectorySegmentBuffer_v5(args, size=10000)
        # new_cost_buffer_validation.reinit(cost_buffer, error_function)
        # cost_buffer = new_cost_buffer_validation

        cost_buffer.verify_preint(error_function)
        cost_buffer.verify_inbetween(error_function)
        cost_buffer.verify_expert(error_function)

        if j == len(intervention_thresholds_list) - 1:
            # collected_data = 0
            # count = 0
            # while collected_data < target_sampled_no_intervention_steps * 5:
            #     sampled_rew, sampled_len = sample_no_expert(args, env, cost_buffer,
            #                                                              initial_imitation_policy, expert_policy, policy_noise=0.2, max_threshold=30)

            #     collected_data += sampled_len
            #     count += 1
            #     if count % 10 == 0:
            #         print("Progress No Expert: ", intervention_thresholds_list[j], collected_data/target_sampled_no_intervention_steps)

            # collected_data = 0
            # count = 0
            # while collected_data < target_sampled_no_intervention_steps * 5:
            #     sampled_rew, sampled_len = sample_no_expert(args, env, cost_buffer_validation,
            #                                                              initial_imitation_policy, expert_policy, policy_noise=0.2, max_threshold=30)

            #     collected_data += sampled_len
            #     count += 1
            #     if count % 10 == 0:
            #         print("Progress No Expert: ", intervention_thresholds_list[j], collected_data/target_sampled_no_intervention_steps)

            pickle.dump(cost_buffer,
                        open("../data/" + env_name + "_training_cost_no_expert_noise_delay_v3_" + str(
                            args.intervention_delay) + "_length_" + str(args.segment_length) + "_v50_0.pkl",
                             "wb"))
            pickle.dump(cost_buffer_validation, open(
                "../data/" + env_name + "_validation_cost_no_expert_noise_delay_v3_" + str(
                    args.intervention_delay) + "_length_" + str(args.segment_length) + "_v50_0.pkl", "wb"))

        # else:

