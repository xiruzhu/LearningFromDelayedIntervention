import numpy as np

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
