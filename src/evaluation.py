import numpy as np
import tensorflow as tf

def evaluate(args, env, actor, expert, cost_distribution, threshold_list, error_function, noisy_actions=0, figure=False):
    """
    Evaluates the policy model by sampling
    Can be noisy or not noisy environment based on the input arguments
    """
    eval_rew = 0
    eval_len = 0
    action_std_list = []
    cost_list = []
    current_state = env.reset()
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

    for i in range(args.max_eps_len):
        action, action_std = actor.get_current_action(np.array([current_state]), verbose=True, noisy=False)

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

        next_frame, reward, terminal, info = env.step(noisy_action)
        current_state = next_frame
        eval_len += 1
        eval_rew += reward
        reward_list.append(reward)
        terminal_list.append(terminal)
        if terminal:
            break

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

def special_evaluation(args, actor, cost_buffer, tflogger, name, frame_num, mode, error_function):
    """
    Evaluates the cost model's performance without training a policy from the cost model.
    When performance is poor, trained cost model tend to be bad
    """
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

        noise_range_up = -(i + 1) * 0.05
        noise_range_down = (i + 1) * 0.05

        noise_range_up_list.append(noise_range_up)
        noise_range_down_list.append(noise_range_down)

    noise_range_up_list = np.array(noise_range_up_list)
    noise_range_down_list = np.array(noise_range_down_list)

    batch_size = 240
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

        s_t_no_label = np.repeat(np.expand_dims(s_t_no_label, axis=1), bin_count, axis=1)
        s_t_no_label = np.reshape(s_t_no_label, [-1, actor.state_dim[0]])

        expert_a_t = np.repeat(np.expand_dims(expert_a_t, axis=1), bin_count, axis=1)
        expert_a_t = np.reshape(expert_a_t, [-1, actor.action_dim])

        sampled_noise = np.random.uniform(noise_range_up_list, noise_range_down_list)
        noisy_expert_a_t = np.clip(expert_a_t + sampled_noise, -1, 1)

        true_cost, _ = error_function(noisy_expert_a_t, expert_a_t) #np.mean(np.abs(current_action - expert_a_t), axis=1)

        estimated_noisy_cost, _, _ = actor.current_cost_model([s_t_no_label, noisy_expert_a_t])
        estimated_noisy_cost = np.squeeze(estimated_noisy_cost)

        estimated_current_cost, _, _ = actor.current_cost_model([s_t_no_label_orig, a_t_no_label_orig])
        estimated_current_cost = np.squeeze(estimated_current_cost)
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


        for j in range(batch_size):
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
