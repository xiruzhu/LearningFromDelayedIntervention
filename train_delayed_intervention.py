import numpy as np
import gym
import tensorflow as tf
import utils, td3, network_models
from tqdm import tqdm
import matplotlib.pyplot as plt
import pickle
from sklearn import metrics


global_error_scale_var = 0.3
def error_function(a, b):
    if len(a.shape) == 1:
        raw_error = np.mean((a - b) ** 2)
        output = np.clip(raw_error, 0, global_error_scale_var)/global_error_scale_var, raw_error
    elif len(a.shape) == 2:
        raw_error = np.mean((a - b) ** 2, axis=1)
        output = np.clip(raw_error, 0, global_error_scale_var)/global_error_scale_var, raw_error
    elif len(a.shape) == 3:
        error = np.mean(np.clip(np.mean((a - b) ** 2, axis=2), 0, global_error_scale_var)/global_error_scale_var, axis=1)
        raw_error = np.clip(np.mean((a - b) ** 2, axis=2), 0, global_error_scale_var)/global_error_scale_var
        output = error, raw_error
    elif len(a.shape) == 4:
        error = np.mean(np.mean(np.clip(np.mean((a - b) ** 2, axis=3), 0, global_error_scale_var)/global_error_scale_var, axis=2), axis=1)
        raw_error = np.clip(np.mean((a - b) ** 2, axis=2), 0, global_error_scale_var)/global_error_scale_var
        output = error, raw_error
    #return np.clip(raw_error, 0, 0.5)/0.5, raw_error
    return output

def buffer_action_eval(args, actor, buffer, tflogger, frame_number, name="validation", iterations=200, batch_size=128):
    preint_error_list = []
    inbetween_error_list = []
    expert_error_list = []
    total_error_list = []

    preint_error_raw_l1_list = []
    inbetween_error_raw_l1_list = []
    expert_error_raw_l1_list = []
    total_error_raw_l1_list = []

    preint_error_raw_l2_list = []
    inbetween_error_raw_l2_list = []
    expert_error_raw_l2_list = []
    total_error_raw_l2_list = []

    for _ in range(iterations):
        #s_t_no_label, a_t_no_label, _, _, _, _, label_t, expert_a_t, _ = cost_buffer.simple_sample(batch_size, mode=0)
        s_t, a_t, _, _, intervention_status_t, _, label_t, expert_a_t, _ = buffer.simple_sample(args.batch_size, mode=0)
        current_action = actor.current_model_list[0](s_t)
        expert_error, raw_expert_error = error_function(current_action, expert_a_t) #np.mean(np.abs(current_action - expert_a_t), axis=1)

        for j in range(batch_size):
            if intervention_status_t[j] == 1:
                expert_error_list.append(expert_error[j])
                expert_error_raw_l1_list.append(raw_expert_error[j])
                expert_error_raw_l2_list.append(raw_expert_error[j] ** 2)

            elif label_t[j] == 0:
                inbetween_error_list.append(expert_error[j])
                inbetween_error_raw_l1_list.append(raw_expert_error[j])
                inbetween_error_raw_l2_list.append(raw_expert_error[j] ** 2)
            else:
                preint_error_list.append(expert_error[j])
                preint_error_raw_l1_list.append(raw_expert_error[j])
                preint_error_raw_l2_list.append(raw_expert_error[j] ** 2)

            total_error_list.append(expert_error[j])
            total_error_raw_l1_list.append(raw_expert_error[j])
            total_error_raw_l2_list.append(raw_expert_error[j] ** 2)

    tflogger.log_scalar("Evaluation_standard/" + name + "_preint_error_list", np.mean(preint_error_list), frame_number)
    tflogger.log_scalar("Evaluation_standard/" + name + "_inbetween_error_list", np.mean(inbetween_error_list), frame_number)
    tflogger.log_scalar("Evaluation_standard/" + name + "_expert_error_list", np.mean(expert_error_list), frame_number)
    tflogger.log_scalar("Evaluation_standard/" + name + "_total_error_list", np.mean(total_error_list), frame_number)

    tflogger.log_scalar("Evaluation_standard/" + name + "_preint_error_raw_l1_list", np.mean(preint_error_raw_l1_list), frame_number)
    tflogger.log_scalar("Evaluation_standard/" + name + "_inbetween_error_raw_l1_list", np.mean(inbetween_error_raw_l1_list), frame_number)
    tflogger.log_scalar("Evaluation_standard/" + name + "_expert_error_raw_l1_list", np.mean(expert_error_raw_l1_list), frame_number)
    tflogger.log_scalar("Evaluation_standard/" + name + "_total_error_raw_l1_list", np.mean(total_error_raw_l1_list), frame_number)

    tflogger.log_scalar("Evaluation_standard/" + name + "_preint_error_raw_l2_list", np.mean(preint_error_raw_l2_list), frame_number)
    tflogger.log_scalar("Evaluation_standard/" + name + "_inbetween_error_raw_l2_list", np.mean(inbetween_error_raw_l2_list), frame_number)
    tflogger.log_scalar("Evaluation_standard/" + name + "_expert_error_raw_l2_list", np.mean(expert_error_raw_l2_list), frame_number)
    tflogger.log_scalar("Evaluation_standard/" + name + "_total_error_raw_l2_list", np.mean(total_error_raw_l2_list), frame_number)

def sampling_evaluation(args, env, initial_policy, actor, expert, noisy_actions=0, figure=False, preintervention_interval=30, intervention_length=30):
    eps_rew = 0
    eps_len = 0
    intervention_steps = 0
    intervention_freq = 0
    current_state = env.reset()

    cost_list = []
    for i in range(preintervention_interval):
        cost_list.append(0)

    intervention_raw_l1_error_list = []
    intervention_raw_l2_error_list = []

    for step in range(args.max_eps_len):
        corrective_policy_action, action_std = actor.get_current_action(np.array([current_state]), noisy=False)
        expert_action = expert.get_current_action(np.array([current_state]), noisy=False)
        initial_policy_action, _ = initial_policy.get_current_action(np.array([current_state]), noisy=False)


        noise = np.clip(np.random.normal(size=initial_policy_action.shape) * 0.1, -0.25, 0.25)
        noisy_action = np.clip(noise + initial_policy_action, -1, 1)
        noisy_executed_action = noisy_action


        expert_cost, expert_cost_raw = error_function(expert_action, corrective_policy_action) #np.mean(np.abs(current_action - expert_a_t), axis=1)
        cost_list.append(expert_cost)
        intervention_raw_l1_error_list.append(expert_cost_raw)
        intervention_raw_l2_error_list.append(expert_cost_raw ** 2)


        next_frame, reward, terminal, _ = env.step(noisy_executed_action)
        eps_len += 1
        eps_rew += reward

        current_state = next_frame
        if terminal:
            break
    #print(eps_len, eps_rew, intervention_freq, intervention_steps, intervention_threshold)
    return intervention_raw_l1_error_list, intervention_raw_l2_error_list



def evaluate(args, env, actor, expert, cost_distribution, threshold_list, noisy_actions=0, figure=False):
    eval_rew = 0
    eval_len = 0
    action_std_list = []
    cost_list = []
    current_state = env.reset()
    # for i in range(5):
    #     next_frame, reward, terminal, info = env.step(np.clip(np.random.uniform(-1.2, 1.2), -1, 1))
    #     current_state = next_frame

    expert_diff_list = []
    expert_diff_raw_l2_list = []
    expert_diff_raw_list = []

    future_reward_list_short = []
    future_reward_list_long = []

    reward_list = []
    terminal_list = []

    expert_diff_list_5 = []
    action_std_list_5 = []
    cost_list_5 = []

    cost_sensitivity_list = []

    #Future rewards ....

    true_error_list = []
    for i in range(args.segment_length):
        true_error_list.append(0)

    intervention_threshold_list = threshold_list
    intervention_steps_list = np.zeros_like(intervention_threshold_list)

    cost_sum = 0
    for i in range(args.max_eps_len):
        action, action_std = actor.get_current_action(np.array([current_state]), verbose=True, noisy=False)
        #expert, _, _ = expert.get_current_action(np.array([current_state]), verbose=False, noisy=False)

        action_expert = expert.get_current_action(np.array([current_state]), verbose=True, noisy=False)
        expert_difference, raw_expert_difference = error_function(action_expert, action) #np.mean(np.abs(current_action - expert_a_t), axis=1)

        expert_diff_list.append(expert_difference)
        expert_diff_raw_list.append(raw_expert_difference)
        expert_diff_raw_l2_list.append(raw_expert_difference ** 2)

        if noisy_actions > 0:
            noise = np.clip(np.random.normal(size=action.shape) * noisy_actions, -noisy_actions * 2.5, noisy_actions * 2.5)
            noisy_action = np.clip(action + noise, -1, 1)
        else:
            noisy_action = action
        cost_sensitivity_list.append(0)

        estimated_cost, _, _ = actor.current_cost_model([np.expand_dims(current_state, axis=0), np.expand_dims(noisy_action, axis=0)])
        cost_list.append(estimated_cost)
        cost_distribution.append(np.squeeze(estimated_cost))


        noisy_expert_difference, _ = error_function(action_expert, noisy_action) #np.mean(np.abs(current_action - expert_a_t), axis=1)
        true_error_list.append(noisy_expert_difference)

        for j in range(len(intervention_threshold_list)):
            if np.sum(true_error_list[-args.segment_length:]) >= intervention_threshold_list[j]:
                intervention_steps_list[j] += 1



        expert_diff_list_5.append(np.mean(expert_diff_list[-5:]))
        action_std_list_5.append(np.mean(action_std_list[-5:]))
        cost_list_5.append(np.mean(cost_list[-5:]))

        #print(i, action, action_std)
        next_frame, reward, terminal, info = env.step(noisy_action)
        current_state = next_frame
        eval_len += 1
        eval_rew += reward
        reward_list.append(reward)
        # if i > 30:
        #     if np.mean(reward_list[-30:]) < -2:
        #         # print("Was here ... ", np.mean(reward_list[-preintervention_interval:]))
        #         # for j in range(eps_len):
        #         #     print(j, reward_list[j])
        #         terminal = True
        terminal_list.append(terminal)
        if terminal:
            break

    # cost_list = np.squeeze(cost_list)
    # moving_average_cost = []
    # for i in range(cost_list.shape[0]):
    #     average_cost = 0
    #     for j in range(10):
    #         if i - j >= 0:
    #             average_cost += cost_list[i - j]
    #     moving_average_cost.append(average_cost/10)
    # moving_average_cost = np.squeeze(moving_average_cost)

    future_reward_list = []
    for i in range(eval_len):
        future_reward = 0
        for j in range(9, -1, -1):
            if eval_len - 1 < i + j:
                if terminal_list[-1]:
                    future_reward = -10 + future_reward
                else:
                    future_reward = reward_list[-1] + future_reward
            else:
                future_reward = reward_list[i + j] + future_reward
        future_reward_list.append(future_reward/10)

    weighted_expert_difference_list = np.array(expert_difference) * (1 - np.clip(np.array(reward_list)/8, 0, 8))
    weighted_future_expert_difference_list = np.array(expert_difference) * (1 - np.clip(np.array(future_reward_list)/8, 0, 8))
    return eval_rew, eval_len, action_std_list_5, cost_list_5, expert_diff_list_5, future_reward_list_long, \
           future_reward_list_short, cost_sensitivity_list, expert_diff_list, expert_diff_raw_list, expert_diff_raw_l2_list, \
           weighted_expert_difference_list, weighted_future_expert_difference_list, terminal, intervention_threshold_list, intervention_steps_list


def special_evaluation(args, actor, cost_buffer, tflogger, name, frame_num, mode):
    iterations = 30
    bin_count = 25
    noise_range_up_list = []
    noise_range_down_list = []

    trained_action_error_dict = []

    trained_action_error_up_dict = []
    trained_action_error_down_dict = []


    trained_action_direction_error_dict = []
    trained_action_count_dict = []

    expert_error_list = []
    expert_up_error_list = []
    expert_down_error_list = []

    inbetween_error_list = []
    inbetween_up_error_list = []
    inbetween_down_error_list = []

    preint_error_list = []
    preint_up_error_list = []
    preint_down_error_list = []

    current_expert_error_list = []
    current_preint_error_list = []
    current_inbetween_error_list = []


    for i in range(bin_count):
        trained_action_error_dict.append(0)
        trained_action_direction_error_dict.append(0)
        trained_action_count_dict.append(0)

        trained_action_error_up_dict.append(0)
        trained_action_error_down_dict.append(0)

        # bin_dict.append(0)
        # bin_dict_direction.append(0)
        # bin_num.append(0)
        noise_range_up = -(i + 1) * 0.05
        noise_range_down = (i + 1) * 0.05

        noise_range_up_list.append(noise_range_up)
        noise_range_down_list.append(noise_range_down)

    noise_range_up_list = np.array(noise_range_up_list)
    noise_range_down_list = np.array(noise_range_down_list)

    batch_size = 240

    #batch_size = 256
    # 25,
    noise_range_up_list = np.repeat(np.expand_dims(noise_range_up_list, axis=1), actor.action_dim, axis=1)
    noise_range_up_list = np.repeat(np.expand_dims(noise_range_up_list, axis=0), batch_size, axis=0)

    noise_range_up_list = np.reshape(noise_range_up_list, [-1, actor.action_dim])

    noise_range_down_list = np.repeat(np.expand_dims(noise_range_down_list, axis=1), actor.action_dim, axis=1)
    noise_range_down_list = np.repeat(np.expand_dims(noise_range_down_list, axis=0), batch_size, axis=0)
    noise_range_down_list = np.reshape(noise_range_down_list, [-1, actor.action_dim])

    estimated_data_error_list = []
    estimated_data_error_direction_list = []

    estimated_data_error_lower_list = []
    estimated_data_error_upper_list = []

    relative_error = 0
    relative_error_no_noise = 0
    for i in range(iterations):
        s_t_raw_preintervention_2, a_t_raw_preintervention_2, r_t_raw_preintervention_2, terminal_t_raw_preintervention_2, expert_only_s_t_raw_preintervention_2, \
        s_t_1_raw_preintervention_2, mean_label_t_raw_preintervention_2, raw_label_t_list_raw_preintervention_2, \
        expert_a_t_raw_preintervention_2, preintervention_raw_segment_intervention_level_2, _ = cost_buffer.sample(
            8, mode=mode)

        s_t_no_label = np.reshape(s_t_raw_preintervention_2, [-1, actor.state_dim[0]])
        a_t_no_label = np.reshape(a_t_raw_preintervention_2, [-1, actor.action_dim])
        expert_a_t = np.reshape(expert_a_t_raw_preintervention_2, [-1, actor.action_dim])
        label_t = np.reshape(raw_label_t_list_raw_preintervention_2, [-1, 1])
        intervention_t = np.reshape(expert_only_s_t_raw_preintervention_2, [-1, 1])

        s_t_no_label_orig = np.copy(s_t_no_label)
        a_t_no_label_orig = np.copy(a_t_no_label)
        expert_a_t_orig = np.copy(expert_a_t)

        #
        # pre_true_cost_no_noise_2 = np.clip(np.mean((expert_a_t_orig - a_t_no_label_orig) ** 2, axis=1), 0, 0.5) / 0.5
        # estimated_current_cost = np.squeeze(estimated_current_cost)
        # true_per_step_error_mode_3 = np.mean(np.abs(pre_true_cost_no_noise_2 - estimated_current_cost))
        # print(np.mean(true_per_step_error_mode_3))

        s_t_no_label = np.repeat(np.expand_dims(s_t_no_label, axis=1), bin_count, axis=1)
        s_t_no_label = np.reshape(s_t_no_label, [-1, actor.state_dim[0]])

        expert_a_t = np.repeat(np.expand_dims(expert_a_t, axis=1), bin_count, axis=1)
        expert_a_t = np.reshape(expert_a_t, [-1, actor.action_dim])

        sampled_noise = np.random.uniform(noise_range_up_list, noise_range_down_list)
        noisy_expert_a_t = np.clip(expert_a_t + sampled_noise, -1, 1)

        #true_cost = np.clip(np.mean(np.abs(noisy_expert_a_t - expert_a_t), axis=1), 0, 0.75)/0.75

        true_cost, _ = error_function(noisy_expert_a_t, expert_a_t) #np.mean(np.abs(current_action - expert_a_t), axis=1)


        estimated_noisy_cost, _, _ = actor.current_cost_model([s_t_no_label, noisy_expert_a_t])
        estimated_noisy_cost = np.squeeze(estimated_noisy_cost)



        estimated_current_cost, _, _ = actor.current_cost_model([s_t_no_label_orig, a_t_no_label_orig])
        estimated_current_cost = np.squeeze(estimated_current_cost)
        #true_current_cost = np.clip(np.mean(np.abs(expert_a_t_orig - a_t_no_label_orig), axis=1), 0, 0.75)/0.75
        true_current_cost, _ = error_function(expert_a_t_orig, a_t_no_label_orig) #np.mean(np.abs(current_action - expert_a_t), axis=1)


        current_error = np.abs(estimated_current_cost - true_current_cost)
        estimated_data_error_list.append(np.mean(current_error))
        estimated_data_error_direction_list.append(np.mean(estimated_current_cost - true_current_cost))

        for j in range(estimated_noisy_cost.shape[0] * 4):
            index_1 = np.random.randint(0, estimated_noisy_cost.shape[0])
            index_2 = np.random.randint(0, estimated_noisy_cost.shape[0])
            while index_2 == index_1:
                index_2 = np.random.randint(0, estimated_noisy_cost.shape[0])
            if (true_cost[index_1] <= true_cost[index_2]) != (estimated_noisy_cost[index_1] <= estimated_noisy_cost[index_2]):
                relative_error += 1

        for j in range(estimated_current_cost.shape[0] * 4):
            index_1 = np.random.randint(0, estimated_current_cost.shape[0])
            index_2 = np.random.randint(0, estimated_current_cost.shape[0])
            while index_2 == index_1:
                index_2 = np.random.randint(0, estimated_current_cost.shape[0])
            if (true_current_cost[index_1] <= true_current_cost[index_2]) != (estimated_current_cost[index_1] <= estimated_current_cost[index_2]):
                relative_error_no_noise += 1


        #quit()
        for j in range(batch_size):
            # estimated_cost_of_state = estimated_current_cost[j]
            # estimated_bin_index = int(np.clip(true_current_cost[j] * bin_count, 0, bin_count - 1))
            estimated_bin_index = int(np.clip(estimated_current_cost[j] * bin_count, 0, bin_count - 1))

            if intervention_t[j] == 1:
                current_expert_error_list.append(current_error[j])
            elif label_t[j] == 0:
                current_inbetween_error_list.append(current_error[j])
            else:
                current_preint_error_list.append(current_error[j])

            for k in range(bin_count):
                list_index = k + j * bin_count
                error_raw = (true_cost[list_index] - estimated_noisy_cost[list_index])
                error = np.abs(error_raw)

                trained_action_error_dict[estimated_bin_index] += error
                if estimated_noisy_cost[list_index] > estimated_current_cost[j]:
                    trained_action_error_up_dict[estimated_bin_index] += error
                else:
                    trained_action_error_down_dict[estimated_bin_index] += error


                trained_action_direction_error_dict[estimated_bin_index] += error_raw
                trained_action_count_dict[estimated_bin_index] += 1
                if true_cost[list_index] < true_current_cost[j]:
                    estimated_data_error_lower_list.append(error)
                else:
                    estimated_data_error_upper_list.append(error)

                if intervention_t[j] == 1:
                    expert_error_list.append(error)
                    if true_cost[list_index] < true_current_cost[j]:
                        expert_down_error_list.append(error)
                    else:
                        expert_up_error_list.append(error)
                elif label_t[j] == 0:
                    inbetween_error_list.append(error)
                    if true_cost[list_index] < true_current_cost[j]:
                        inbetween_down_error_list.append(error)
                    else:
                        inbetween_up_error_list.append(error)
                else:
                    preint_error_list.append(error)
                    if true_cost[list_index] < true_current_cost[j]:
                        preint_down_error_list.append(error)
                    else:
                        preint_up_error_list.append(error)

    # print("-----------------------")
    # for i in range(bin_count * 2):
    #     print(i, class_difference_bin[i]/(max(1, class_difference_count[i])),
    #           class_bin_error_direction[i] / max(1, class_difference_count[i]),
    #           class_difference_count[i]/np.sum(class_difference_count))
    if len(expert_error_list) == 0:
        expert_error_list.append(0)
        expert_up_error_list.append(0)
        expert_down_error_list.append(0)
        current_expert_error_list.append(0)
    if len(inbetween_error_list) == 0:
        inbetween_error_list.append(0)
        inbetween_up_error_list.append(0)
        inbetween_down_error_list.append(0)
        current_inbetween_error_list.append(0)
    if len(preint_error_list) == 0:
        preint_error_list.append(0)
        preint_up_error_list.append(0)
        preint_down_error_list.append(0)
        current_preint_error_list.append(0)


    # for i in range(bin_count):
    #     print(i, trained_action_error_dict[i] / (max(1, trained_action_count_dict[i])),
    #           trained_action_error_down_dict[i] / max(1, trained_action_count_dict[i]),
    #           trained_action_error_up_dict[i] / max(1, trained_action_count_dict[i]),
    #           trained_action_direction_error_dict[i] / (max(1, trained_action_count_dict[i])),
    #           trained_action_count_dict[i] / np.sum(trained_action_count_dict))

    print("Expert Error, down, up: ", np.mean(expert_error_list), np.mean(expert_down_error_list), np.mean(expert_up_error_list))
    print("Inbetween Error, down, up: ", np.mean(inbetween_error_list), np.mean(inbetween_down_error_list), np.mean(inbetween_up_error_list))
    print("Preint Error, down, up: ", np.mean(preint_error_list), np.mean(preint_down_error_list), np.mean(preint_up_error_list))
    print("Total Random Error: ", np.mean(preint_error_list + inbetween_error_list + expert_error_list))
    print("Current expert, inbetween, preint, total: ", np.mean(current_expert_error_list), np.mean(current_inbetween_error_list), np.mean(current_preint_error_list),
          np.mean(current_expert_error_list + current_inbetween_error_list + current_preint_error_list))
    print("Lower estimated error, upper estimated error: ", np.mean(estimated_data_error_lower_list), np.mean(estimated_data_error_upper_list))
    print("Relative error: ", relative_error/(4 * (estimated_noisy_cost.shape[0] - 1) * iterations))
    print("Relative error no noise: ", relative_error_no_noise/(4 * (estimated_noisy_cost.shape[0] - 1) * iterations))


    tflogger.log_scalar(name + "/expert_error_list", np.mean(expert_error_list), frame_num)
    tflogger.log_scalar(name + "/inbetween_error_list", np.mean(inbetween_error_list), frame_num)
    tflogger.log_scalar(name + "/preint_error_list", np.mean(preint_error_list), frame_num)
    tflogger.log_scalar(name + "/total_error_list", np.mean(expert_error_list + inbetween_error_list + preint_error_list), frame_num)

    tflogger.log_scalar(name + "/current_expert_error_list", np.mean(current_expert_error_list), frame_num)
    tflogger.log_scalar(name + "/current_inbetween_error_list", np.mean(current_inbetween_error_list), frame_num)
    tflogger.log_scalar(name + "/current_preint_error_list", np.mean(current_preint_error_list), frame_num)
    tflogger.log_scalar(name + "/current_total_error_list", np.mean(current_expert_error_list + current_inbetween_error_list + current_preint_error_list), frame_num)

    tflogger.log_scalar(name + "/estimated_data_error_lower_list", np.mean(estimated_data_error_lower_list), frame_num)
    tflogger.log_scalar(name + "/estimated_data_error_upper_list", np.mean(estimated_data_error_upper_list), frame_num)

    tflogger.log_scalar(name + "/expert_down_error_list", np.mean(expert_down_error_list), frame_num)
    tflogger.log_scalar(name + "/expert_up_error_list", np.mean(expert_up_error_list), frame_num)

    tflogger.log_scalar(name + "/inbetween_down_error_list", np.mean(inbetween_down_error_list), frame_num)
    tflogger.log_scalar(name + "/inbetween_up_error_list", np.mean(inbetween_up_error_list), frame_num)

    tflogger.log_scalar(name + "/preint_down_error_list", np.mean(preint_down_error_list), frame_num)
    tflogger.log_scalar(name + "/preint_up_error_list", np.mean(preint_up_error_list), frame_num)
    tflogger.log_scalar(name + "/relative_error", relative_error/(4 * iterations * (estimated_noisy_cost.shape[0] - 1)), frame_num)
    tflogger.log_scalar(name + "/relative_error_no_noise", relative_error_no_noise/(4 * iterations * (estimated_noisy_cost.shape[0] - 1)), frame_num)

    print("-----------------------------------------")



def sample_episode_without_expert(args, env, actor, expert, preintervention_interval=30, supervision_window=30):
    cost_list = []
    for i in range(preintervention_interval):
        cost_list.append(0)

    segment_supervision_cost_list = []
    current_state = env.reset()
    for step in range(args.max_eps_len):
        action, action_std = actor.get_current_action(np.array([current_state]), noisy=False)
        expert_action  = expert.get_current_action(np.array([current_state]), noisy=False)
        noise = np.clip(np.random.normal(size=action.shape) * actor.policy_noise, -actor.noise_clip,
                        actor.noise_clip)
        noisy_action = np.clip(noise + action, -1, 1)

        expert_cost, _ = error_function(expert_action, noisy_action) #np.mean(np.abs(current_action - expert_a_t), axis=1)
        cost_list.append(expert_cost)
        segment_supervision_cost_list.append(np.sum(cost_list[-supervision_window:]))
        next_frame, reward, terminal, _ = env.step(noisy_action)
        current_state = next_frame
        if terminal:
            break
    return segment_supervision_cost_list



def sample_episode(args, env, cost_buffer, actor, expert, count, intervention_threshold, policy_noise=0,
                   supervision_window=30, preintervention_interval=30, expert_intervention_length=30, expert_interventions=True):
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

    # threshold_list = [8, 10, 12, 12, 14, 14, 16, 16, 16]
    # intervention_threshold = threshold_list[np.random.randint(len(threshold_list))] #If the accumulated action std for the past 25 steps exceeds this ...
    threshold_cost_list = []
    segment_threshold_list = []
    expert_action_list = []

    preint_threshold_list = []

    intervention_delay_list = []

    early_intervention_flag = False

    #expert = 0
    #inbetween = 1
    #preint = 2

    segment_label_list = []


    for step in range(args.max_eps_len):
        action, action_std = actor.get_current_action(np.array([current_state]), noisy=False)
        expert_action  = expert.get_current_action(np.array([current_state]), noisy=False)
        noise = np.clip(np.random.normal(size=action.shape) * policy_noise, -policy_noise * 2.5, policy_noise * 2.5)
        noisy_action = np.clip(noise + action, -1, 1)


        # noise = np.clip(np.random.normal(size=action.shape) * actor.policy_noise, -actor.noise_clip,
        #                 actor.noise_clip)
        # noisy_expert_action = np.clip(noise + expert_action, -1, 1)
        if expert_steps > 0:
            noisy_executed_action = expert_action
        else:
            noisy_executed_action = noisy_action

        ensemble_uncertainty_list.append(action_std)

        expert_cost, _ = error_function(expert_action, noisy_executed_action) #np.mean(np.abs(current_action - expert_a_t), axis=1)
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
        # if step > 30:
        #     if np.mean(reward_list[-30:]) < -2:
        #         # print("Was here ... ", np.mean(reward_list[-preintervention_interval:]))
        #         # for j in range(eps_len):
        #         #     print(j, reward_list[j])
        #         terminal = True
        expert_action_list.append(expert_action)
        state_list.append(current_state)
        action_list.append(noisy_executed_action)
        termination_list.append(terminal)
        next_state_list.append(next_frame)


        # print(step, expert_steps > 0, np.mean(expert_intervention[-supervision_window:]))
        if expert_steps > 0:
            expert_steps -= 1
            expert_steps_rewards += reward

        current_state = next_frame


        segment_threshold_list.append(np.sum(cost_list[-supervision_window:]))
        intervention_threshold_randomization = intervention_threshold# + np.random.uniform(-1, 1)



        if ((np.sum(cost_list[-supervision_window:]) >= intervention_threshold_randomization)) and expert_steps <= 0 and expert_interventions:

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
            elif expert_intervention[-1] == 0 and expert_steps != expert_intervention_length:
                segment_label = 1
            else:
                segment_label = -1
            segment_label_list.append(segment_label)



        if terminal:
            break

        # if eps_len == args.segment_length + 1:
        #     break
    # if terminal:
    if eps_len >= args.segment_length - 1:


        for i in range(supervision_window):
            reward_list.append(reward_list[-1])


        for i in range(eps_len):
            cost_buffer.add(state_list[i], action_list[i], reward_list[i], termination_list[i], expert_intervention[i],
                            next_state_list[i], preintervention_status_list[i], expert_action_list[i], intervention_threshold)

        for i in range(len(segment_label_list)):
            segment_estimated_cost = 0
            selected_indices = []
            for k in range(args.segment_length):
                target_index = i + k
                segment_estimated_cost += cost_list[target_index + preintervention_interval]

                expert_cost, _ = error_function(action_list[target_index], expert_action_list[target_index])  # np.mean(np.abs(current_action - expert_a_t), axis=1)
                if np.abs(expert_cost - cost_list[target_index + preintervention_interval]) > 0.01:
                    print(target_index, "Error, Cost Error", expert_cost, cost_list[target_index + preintervention_interval])
                    quit()

                selected_indices.append(target_index)
                #print(i, k, target_index, expert_intervention[target_index],  preintervention_status_list[target_index], segment_estimated_cost)

            if segment_label_list[i] == 2:
                if segment_estimated_cost < intervention_threshold:
                    print(i + args.segment_length - 1, "Preint Segment Error ... !", segment_estimated_cost, intervention_threshold)
                    quit()

            elif segment_label_list[i] == 0:
                if segment_estimated_cost > 0:
                    print(i + args.segment_length - 1, "Expert Segment Error ... !", segment_estimated_cost, intervention_threshold)
                    quit()
            elif segment_label_list[i] == 1:
                if segment_estimated_cost >= intervention_threshold:
                    print(i + args.segment_length - 1, "Inbetween Segment Error ... !", segment_estimated_cost, intervention_threshold)
                    quit()

            cost_buffer.add_segment(np.array(selected_indices), 0, intervention_threshold)
            if segment_label_list[i] != -1:
                cost_buffer.add_segment(np.array(selected_indices), segment_label_list[i] + 1, intervention_threshold)
        cost_buffer.trajectory_end()


        print(eps_len, eps_rew, intervention_freq, intervention_steps, np.amax(threshold_cost_list), intervention_threshold)
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
    #load expert policy
    eval_policy_noise = 0.2
    if args.env_id == "Humanoid-v2":
        env_name = "humanoid"
        intervention_thresholds_list = [12, 16, 20, 24]
        target_sampled_expert_steps = 800
        target_sampled_no_intervention_steps = 10000

    elif args.env_id == "Walker2d-v2":
        env_name = "walker"
        intervention_thresholds_list = [12, 16, 20, 24]
        target_sampled_expert_steps = 800
        target_sampled_no_intervention_steps = 10000

    elif args.env_id == "HalfCheetah-v2":
        env_name = "halfcheetah"
        intervention_thresholds_list = [12, 16, 20, 24]
        target_sampled_expert_steps = 800
        target_sampled_no_intervention_steps = 10000

    elif args.env_id == "Hopper-v2":
        env_name = "hopper"
        intervention_thresholds_list = [12, 14, 16, 18]
        target_sampled_expert_steps = 800
        target_sampled_no_intervention_steps = 10000
        eval_policy_noise = 0.1
    elif args.env_id == "Ant-v2":
        env_name = "ant"
        intervention_thresholds_list = [12, 16, 20, 24]
        target_sampled_expert_steps = 800
        target_sampled_no_intervention_steps = 10000
    else:
        print("Error env")
        quit()
    for i in range(len(intervention_thresholds_list)):
        intervention_thresholds_list[i] = int(intervention_thresholds_list[i] * args.segment_length / 30)


    expert_policy.current_model.load_weights(expert_policy.args.checkpoint_dir + "/" + env_name + "_expert_policy" + "/actor_current_network_" + str(0))
    old_cost_buffer = pickle.load(open("../data/original_" + env_name + "_no_noise_initial_cost_data_delay_1_v4_0.pkl", "rb"))
    cost_buffer = old_cost_buffer
    cost_buffer = utils.TrajectorySegmentBuffer_v5(args, size=10000)
    cost_buffer.reinit(old_cost_buffer, error_function)

    cost_buffer_validation = utils.TrajectorySegmentBuffer_v5(args, size=10000)
    cost_buffer_validation.reinit(old_cost_buffer, error_function)


    print(cost_buffer.raw_data_count)
    print(cost_buffer.expert_single_state_count)


    imitation_policy = network_models.actor(args, ob_space.shape, ac_space.shape[-1])
    initial_imitation_policy = network_models.actor(args, ob_space.shape, ac_space.shape[-1])

    tflogger = utils.tensorflowboard_logger(args.log_dir + args.custom_id, args)
    eps_number = 0
    frame_number = 0
    eval_frame = 0
    save_frame = 0
    buffer_save_frame = 0
    pretrain_steps = int(100000 * (1 + args.cost_loss_bias))

    expert_intervention_steps_list = []
    intervention_freq_list = []
    sampled_length_list = []
    average_delay_list = []
    total_expert_intervention = 0
    total_sampled_steps = 0

    bc_loss_list = []
    policy_loss_list = []
    #intervention_thresholds_list = [10, 15, 20]
    #intervention_thresholds_list = [12, 16, 20]

    print(cost_buffer.preintervention_segment_count)
    print(cost_buffer.raw_data_count)
    print(cost_buffer.expert_single_state_count)


    cost_buffer = pickle.load(open("../data/" + env_name + "_training_cost_no_expert_noise_delay_v6_" + str(args.intervention_delay) + "_length_" + str(args.segment_length) + "_v50_0.pkl", "rb"))
    cost_buffer.generate_segment_weights(intervention_thresholds_list[-1])


    cost_loss_list = []

    validation_cost_buffer = pickle.load(open("../data/" + env_name + "_validation_cost_no_expert_noise_delay_v6_" + str(args.intervention_delay) + "_length_" + str(args.segment_length) + "_v50_0.pkl", "rb"))
    validation_cost_buffer.generate_segment_weights(intervention_thresholds_list[-1])


    print(cost_buffer.raw_data_count)
    print(cost_buffer.expert_single_state_count)
    print(validation_cost_buffer.raw_data_count)
    print(validation_cost_buffer.expert_single_state_count)    # special_evaluation(args, imitation_policy, validation_cost_buffer, tflogger, "eval_mode_3", 0, 3)
    # quit()
    #cost_buffer.verify_preint(error_function)
    cost_buffer.verify_inbetween(error_function)
    cost_buffer.verify_expert(error_function)
    #quit(())

    if args.intervention_loss:
        if args.cost_only == 0:
            print("Attempting to load data ...")
            imitation_policy.current_cost_model.load_weights("../data/" + env_name + "_cost_model/cost_model_delay_" + str(args.intervention_delay) + "_70_5_1_v5_mod_" + str(int(args.coefficient)) +  "_bias_" + str(0.0) + "_l2_" + str(args.l2) + "_seed_" + str(args.seed))
            imitation_policy.special_evaluation_state(cost_buffer, error_function, mode=2)
            imitation_policy.special_evaluation_state(cost_buffer, error_function, mode=3)
            print("Data loaded ... ")
        else:
            additional_loss_list = []
            print("Training cost model for ", pretrain_steps * 1, "steps")
            for i in range(pretrain_steps * 1):
                if i % 1000 == 0 and i > 0:
                    verbose=True
                else:
                    verbose=False
                cost_loss, additional_loss = imitation_policy.train_step_cost_true_v5(cost_buffer, error_function, expert_policy=expert_policy, verbose=verbose)

                cost_loss_list.append(cost_loss)
                additional_loss_list.append(additional_loss)
                if i % 1000 == 0 and i > 0:
                    print(i/(pretrain_steps * 1), np.mean(cost_loss_list))
                    tflogger.log_scalar("Cost Training/loss", np.mean(cost_loss_list), i)
                    tflogger.log_scalar("Cost Training/additional_loss", np.mean(additional_loss_list), i)
                    # cost_loss = imitation_policy.train_step_cost(cost_buffer, verbose=True, expert_policy=expert_policy)

                    additional_loss_list = []
                    cost_loss_list = []
                if i % 5000 == 0 and i > 0:
                    special_evaluation(args, imitation_policy, validation_cost_buffer, tflogger, "eval_mode_2", i, 2)
                    special_evaluation(args, imitation_policy, validation_cost_buffer, tflogger, "eval_mode_3", i, 3)

                    preference_loss_capped, preference_loss, preference_noisy_loss_capped, preference_noisy_loss = imitation_policy.evaluate_preference_loss(cost_buffer, error_function)
                    tflogger.log_scalar("Cost Training/raw_preference_loss", preference_loss, i)
                    tflogger.log_scalar("Cost Training/capped_preference_loss", preference_loss_capped, i)
                    tflogger.log_scalar("Cost Training/raw_preference_random_loss", preference_noisy_loss, i)
                    tflogger.log_scalar("Cost Training/capped_preference_random_loss", preference_noisy_loss_capped, i)
                    print("Sampled Error: ", preference_loss, "Sampled Error Capped: ", preference_loss_capped)
                    print("Sampled Random Segment Error: ", preference_noisy_loss, "Sampled Error Random Segment Capped: ", preference_noisy_loss_capped)
                    imitation_policy.special_evaluation_state(cost_buffer, error_function, mode=2)
                    imitation_policy.special_evaluation_state(cost_buffer, error_function, mode=3)
                    print("---------------------------------")
                    #cost_loss = imitation_policy.train_step_cost(cost_buffer, expert_policy=expert_policy, verbose=True, validation_buffer=cost_buffer)
                    #evaluate_cost_data(args, env, imitation_policy, expert_policy, i, cost_buffer,tflogger, name="Cost Training")
                    #evaluate_cost_data(args, env, imitation_policy, expert_policy, i, validation_cost_buffer,tflogger, name="Cost Validation")
                    imitation_policy.current_cost_model.save_weights("../data/" + env_name + "_cost_model/cost_model_delay_" + str(args.intervention_delay) + "_70_5_1_v5_mod_" + str(int(args.coefficient)) +  "_bias_" + str(0.0) + "_l2_" + str(args.l2) + "_seed_" + str(args.seed))
            imitation_policy.current_cost_model.save_weights("../data/" + env_name + "_cost_model/cost_model_delay_" + str(args.intervention_delay) + "_70_5_1_v5_mod_" + str(int(args.coefficient)) +  "_bias_" + str(0.0) + "_l2_" + str(args.l2) + "_seed_" + str(args.seed))
            special_evaluation(args, imitation_policy, validation_cost_buffer, tflogger, "eval_mode_2", i, 2)
            special_evaluation(args, imitation_policy, validation_cost_buffer, tflogger, "eval_mode_3", i, 3)

            preference_loss_capped, preference_loss, preference_noisy_loss_capped, preference_noisy_loss = imitation_policy.evaluate_preference_loss(cost_buffer, error_function)
            tflogger.log_scalar("Cost Training/raw_preference_loss", preference_loss, i)
            tflogger.log_scalar("Cost Training/capped_preference_loss", preference_loss_capped, i)
            tflogger.log_scalar("Cost Training/raw_preference_random_loss", preference_noisy_loss, i)
            tflogger.log_scalar("Cost Training/capped_preference_random_loss", preference_noisy_loss_capped, i)
            print("Sampled Error: ", preference_loss, "Sampled Error Capped: ", preference_loss_capped)
            print("Sampled Random Segment Error: ", preference_noisy_loss, "Sampled Error Random Segment Capped: ",preference_noisy_loss_capped)
            print("Done pretraining cost model...")

    # print("was here ... ")
    # trajectory_count = 0
    # total_reward = 0
    # if args.intervention_loss:
    #     for _ in range(1):
    #         sampled_rew, sampled_len, expert_intervention_steps, intervention_freq, ensemble_uncertainty, average_delay = sample_episode(args, env, buffer_list,
    #                                                                                                 cost_buffer,
    #                                                                                                 imitation_policy,
    #                                                                                                 expert_policy, trajectory_count, preintervention_interval=args.segment_length, expert_interventions=False)
    #         trajectory_count += 1
    #         total_reward += sampled_rew
    # print(total_reward/30)
    # quit()
    #

    #special_evaluation(args, imitation_policy, validation_cost_buffer, tflogger, "eval_mode_3", 0, 3)
    #special_evaluation(args, imitation_policy, validation_cost_buffer, tflogger, "eval_mode_10", 0, 10)
    # quit()

        # self.cost_model.load_weights("../data/walker_delay_1_steps_100000_cost_model")
    # _ = imitation_policy.train_step_cost(cost_buffer, verbose=True, expert_policy=expert_policy)
    # quit()
    # cost_loss = imitation_policy.train_step_cost(cost_buffer, verbose=True, expert_policy=expert_policy)
    # quit()
    #evaluate_cost_data(args, env, imitation_policy, expert_policy, i, validation_cost_buffer, tflogger, name="Cost Validation")
    #quit()
    # cost_loss = imitation_policy.train_step_cost(cost_buffer, expert_policy=expert_policy, verbose=True)
    #value current cost policy for fun ...

    cost_loss_list= []
    for _ in range(args.policy_training_steps):
        # if frame_number % 10000 == 0 and frame_number > 0:
        #     policy_error_l1_list = []
        #     policy_error_l2_list = []
        #
        #     for k in range(20):
        #         l1_error, l2_error = sampling_evaluation(args, env, initial_imitation_policy, imitation_policy, expert_policy)
        #         policy_error_l1_list.extend(l1_error)
        #         policy_error_l2_list.extend(l2_error)
        #
        #     tflogger.log_scalar("Evaluation_sampling/l1_error", np.mean(policy_error_l1_list), frame_number)
        #     tflogger.log_scalar("Evaluation_sampling/l2_error", np.mean(policy_error_l2_list), frame_number)
        #     print("Evaluation Sampling, l1 error: ", np.mean(policy_error_l1_list), "l2 error: ", np.mean(policy_error_l2_list))
        #     print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")


        if frame_number % 10000 == 0 and frame_number > 0:
            print(frame_number, total_expert_intervention)
            eval_rews_list = []
            eval_len_list = []

            expert_action_diff_list = []
            cost_list = []
            long_pred_reward_list = []
            short_pred_reward_list = []
            evaluation_action_std_list = []
            cost_sensitivity_list = []
            cost_distribution = []
            expert_deviation_list = []

            expert_deviation_raw_l1_list = []
            expert_deviation_raw_l2_list = []

            expert_deviation_weighted_list = []
            expert_deviation_weighted_future_list = []

            target_num_sampled_states = 10000
            num_sampled_states = 0
            num_sampled_reward = 0

            num_terminal = 0
            num_sampled_runs = 0
            total_intervention_steps = 0
            while num_sampled_states < target_num_sampled_states:
                eval_rew, eval_len, action_std_list, cost, expert_action_diff, long_pred_reward, \
                short_pred_reward, cost_sensitivity, expert_deviation, expert_deviation_raw_l1, expert_deviation_raw_l2, expert_deviation_weighted, expert_deviation_future_weighted, terminal, \
                intervention_thresholds, intervention_steps = evaluate(args, env, imitation_policy, expert_policy, cost_distribution, intervention_thresholds_list,
                                                                       noisy_actions=0, figure=num_sampled_states == 0)

                eval_rews_list.append(eval_rew)
                eval_len_list.append(eval_len)
                expert_deviation_list.extend(expert_deviation)

                expert_deviation_raw_l1_list.extend(expert_deviation_raw_l1)
                expert_deviation_raw_l2_list.extend(expert_deviation_raw_l2)

                expert_deviation_weighted_list.extend(expert_deviation_weighted)
                expert_deviation_weighted_future_list.extend(expert_deviation_future_weighted)

                num_sampled_states += eval_len
                num_sampled_reward += eval_rew
                num_terminal += terminal
                num_sampled_runs += 1

                total_intervention_steps += intervention_steps

                long_pred_reward_list.extend((long_pred_reward))
                short_pred_reward_list.extend((short_pred_reward))
                cost_sensitivity_list.extend(cost_sensitivity)
                cost_list.extend(cost)
                expert_action_diff_list.extend(expert_action_diff)
                evaluation_action_std_list.extend(action_std_list)
                print("Episode:", num_sampled_states, ", Reward: ", eval_rew, "Eval len: ", eval_len)

            total_intervention_steps = total_intervention_steps/num_sampled_states

            mean_val = np.mean(eval_rews_list)
            std_val = np.std(eval_rews_list)
            low_ci = mean_val - 2.086 * std_val / np.sqrt(20)
            up_ci = mean_val + 2.086 * std_val / np.sqrt(20)
            rew_per_frame = num_sampled_reward / num_sampled_states
            print("Standard Eval rew: ", mean_val, "Eval len: ", np.mean(eval_len_list), "Terminal Rate: ", num_terminal / num_sampled_states * 10000)
            for i in range(len(intervention_thresholds)):
                print("intervention_threshold_" + str(int(intervention_thresholds[i])) + "_steps", total_intervention_steps[i])
                tflogger.log_scalar("Evaluation_standard/intervention_threshold_" + str(int(intervention_thresholds[i])) + "_steps", total_intervention_steps[i], frame_number)

            tflogger.log_scalar("Evaluation_standard/eps_rew", mean_val, frame_number)
            tflogger.log_scalar("Evaluation_standard/avg_expert_deviation", np.mean(expert_deviation_list), frame_number)
            tflogger.log_scalar("Evaluation_standard/avg_expert_deviation_raw_l1", np.mean(expert_deviation_raw_l1_list), frame_number)
            tflogger.log_scalar("Evaluation_standard/avg_expert_deviation_raw_l2", np.mean(expert_deviation_raw_l2_list), frame_number)

            tflogger.log_scalar("Evaluation_standard/avg_expert_weighted_deviation", np.mean(expert_deviation_weighted_list), frame_number)
            tflogger.log_scalar("Evaluation_standard/avg_expert_weighted_future_deviation", np.mean(expert_deviation_weighted_future_list), frame_number)

            tflogger.log_scalar("Evaluation_standard/low_ci", low_ci, frame_number)
            tflogger.log_scalar("Evaluation_standard/up_ci", up_ci, frame_number)
            tflogger.log_scalar("Evaluation_standard/rew_per_frame", rew_per_frame, frame_number)
            tflogger.log_scalar("Evaluation_standard/terminal_rate", num_terminal / num_sampled_states * 10000, frame_number)






            eval_rews_list = []
            eval_len_list = []

            expert_action_diff_list = []
            cost_list = []
            long_pred_reward_list = []
            short_pred_reward_list = []
            evaluation_action_std_list = []
            cost_sensitivity_list = []
            cost_distribution = []
            expert_deviation_list = []

            expert_deviation_raw_l1_list = []
            expert_deviation_raw_l2_list = []

            expert_deviation_weighted_list = []
            expert_deviation_weighted_future_list = []

            target_num_sampled_states = 10000
            num_sampled_states = 0
            num_sampled_reward = 0

            num_terminal = 0
            num_sampled_runs = 0
            total_intervention_steps = 0
            while num_sampled_states < target_num_sampled_states:
                eval_rew, eval_len, action_std_list, cost, expert_action_diff, long_pred_reward, \
                short_pred_reward, cost_sensitivity, expert_deviation, expert_deviation_raw_l1, expert_deviation_raw_l2, expert_deviation_weighted, expert_deviation_future_weighted, terminal, \
                intervention_thresholds, intervention_steps = evaluate(args, env, imitation_policy, expert_policy,
                                                                       cost_distribution, intervention_thresholds_list,
                                                                       noisy_actions=eval_policy_noise, figure=num_sampled_states == 0)

                eval_rews_list.append(eval_rew)
                eval_len_list.append(eval_len)
                expert_deviation_list.extend(expert_deviation)

                expert_deviation_raw_l1_list.extend(expert_deviation_raw_l1)
                expert_deviation_raw_l2_list.extend(expert_deviation_raw_l2)

                expert_deviation_weighted_list.extend(expert_deviation_weighted)
                expert_deviation_weighted_future_list.extend(expert_deviation_future_weighted)

                num_sampled_states += eval_len
                num_sampled_reward += eval_rew
                num_terminal += terminal
                num_sampled_runs += 1

                total_intervention_steps += intervention_steps

                long_pred_reward_list.extend((long_pred_reward))
                short_pred_reward_list.extend((short_pred_reward))
                cost_sensitivity_list.extend(cost_sensitivity)
                cost_list.extend(cost)
                expert_action_diff_list.extend(expert_action_diff)
                evaluation_action_std_list.extend(action_std_list)
                print("Episode:", num_sampled_states, ", Reward: ", eval_rew, "Eval len: ", eval_len)

            total_intervention_steps = total_intervention_steps / num_sampled_states

            mean_val = np.mean(eval_rews_list)
            std_val = np.std(eval_rews_list)
            low_ci = mean_val - 2.086 * std_val / np.sqrt(20)
            up_ci = mean_val + 2.086 * std_val / np.sqrt(20)
            rew_per_frame = num_sampled_reward / num_sampled_states
            print("Noisy Standard Eval rew: ", mean_val, "Eval len: ", np.mean(eval_len_list), "Terminal Rate: ",
                  num_terminal / num_sampled_states * 10000)
            for i in range(len(intervention_thresholds)):
                print("intervention_threshold_" + str(int(intervention_thresholds[i])) + "_steps",
                      total_intervention_steps[i])
                tflogger.log_scalar(
                    "Evaluation_standard_noisy/intervention_threshold_" + str(int(intervention_thresholds[i])) + "_steps",
                    total_intervention_steps[i], frame_number)

            tflogger.log_scalar("Evaluation_standard_noisy/eps_rew", mean_val, frame_number)
            tflogger.log_scalar("Evaluation_standard_noisy/avg_expert_deviation", np.mean(expert_deviation_list), frame_number)
            tflogger.log_scalar("Evaluation_standard_noisy/avg_expert_deviation_raw_l1", np.mean(expert_deviation_raw_l1_list), frame_number)
            tflogger.log_scalar("Evaluation_standard_noisy/avg_expert_deviation_raw_l2", np.mean(expert_deviation_raw_l2_list), frame_number)

            tflogger.log_scalar("Evaluation_standard_noisy/avg_expert_weighted_deviation", np.mean(expert_deviation_weighted_list), frame_number)
            tflogger.log_scalar("Evaluation_standard_noisy/avg_expert_weighted_future_deviation", np.mean(expert_deviation_weighted_future_list), frame_number)

            tflogger.log_scalar("Evaluation_standard_noisy/low_ci", low_ci, frame_number)
            tflogger.log_scalar("Evaluation_standard_noisy/up_ci", up_ci, frame_number)
            tflogger.log_scalar("Evaluation_standard_noisy/rew_per_frame", rew_per_frame, frame_number)
            tflogger.log_scalar("Evaluation_standard_noisy/terminal_rate", num_terminal / num_sampled_states * 10000, frame_number)

        frame_number += 1

        #trans_loss = imitation_policy.train_step_transition_model(transition_buffer)
        policy_loss, bc_loss, cost_loss = imitation_policy.train_step_actor_v2(None, cost_buffer, expert_policy)
        policy_loss_list.append(policy_loss)
        bc_loss_list.append(bc_loss)


        cost_loss_list.append(cost_loss)

        #tflogger.log_scalar("RAW/Trans_loss", trans_loss, frame_number)
        if frame_number % 1000 == 0:
            tflogger.log_scalar("RAW/Policy_loss", np.mean(policy_loss_list), frame_number)
            tflogger.log_scalar("RAW/BC_loss", np.mean(bc_loss_list), frame_number)
            tflogger.log_scalar("RAW/Policy_Cost_loss", np.mean(cost_loss_list), frame_number)

            policy_loss_list = []
            bc_loss_list = []
            cost_loss_list = []
        # if args.intervention_loss:
        #     raw_cost_loss = imitation_policy.train_step_cost(cost_buffer)
        #     #raw_cost_list.append(raw_cost_loss)
        #     tflogger.log_scalar("RAW/Cost_loss", raw_cost_loss, frame_number)
        #


if __name__ == "__main__":
    main()