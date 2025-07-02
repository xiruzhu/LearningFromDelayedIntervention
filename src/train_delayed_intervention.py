import numpy as np
import gym
import tensorflow as tf
import utils, td3, core_network
import pickle
import json
import evaluation

global_error_scale_var = 0.3
def error_function(a, b):
    """
    Oracle Error Cost Function to calculate error per action. Error is capped to [0, 1]
    """
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
    return output

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
    policy_loss_list = []


    print(cost_buffer.preintervention_segment_count)
    print(cost_buffer.raw_data_count)
    print(cost_buffer.expert_single_state_count)


    cost_buffer = pickle.load(open("../data/" + env_name + "_training_cost_no_expert_noise_delay_v" + str(args.buffer_id) + "_" + str(args.intervention_delay) + "_length_" + str(args.segment_length) + "_v50_0.pkl", "rb"))
    cost_buffer.generate_segment_weights(intervention_thresholds_list[-1])


    cost_loss_list = []
    validation_cost_buffer = pickle.load(open("../data/" + env_name + "_validation_cost_no_expert_noise_delay_v" + str(args.buffer_id) + "_" + str(args.intervention_delay) + "_length_" + str(args.segment_length) + "_v50_0.pkl", "rb"))
    validation_cost_buffer.generate_segment_weights(intervention_thresholds_list[-1])


    print(cost_buffer.raw_data_count)
    print(cost_buffer.expert_single_state_count)
    print(validation_cost_buffer.raw_data_count)
    print(validation_cost_buffer.expert_single_state_count)
    cost_buffer.verify_inbetween(error_function)
    cost_buffer.verify_expert(error_function)

    if args.intervention_loss and args.cost_version != 2: #2 is BC and does not require cost function training
        if args.cost_only == 0:
            print("Attempting to load data ...")
            imitation_policy.current_cost_model.load_weights("../data/" + env_name + "_cost_model/cost_model_delay_" + str(args.intervention_delay) + "_70_" + str(args.cost_version) + "_1_v" + str(args.buffer_id) + "_mod_" + str(int(args.coefficient)) +  "_bias_" + str(0.0) + "_l2_" + str(args.l2) + "_seed_" + str(args.seed))
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
                cost_loss, additional_loss = imitation_policy.train_step_cost_core(cost_buffer, error_function, expert_policy=expert_policy, verbose=verbose)

                cost_loss_list.append(cost_loss)
                additional_loss_list.append(additional_loss)
                if i % 1000 == 0 and i > 0:
                    print(i/(pretrain_steps * 1), np.mean(cost_loss_list))
                    tflogger.log_scalar("Cost Training/loss", np.mean(cost_loss_list), i)
                    tflogger.log_scalar("Cost Training/additional_loss", np.mean(additional_loss_list), i)
                    additional_loss_list = []
                    cost_loss_list = []
                if i % 5000 == 0 and i > 0:
                    evaluation.special_evaluation(args, imitation_policy, validation_cost_buffer, tflogger, "eval_mode_2", i, 2, error_function)
                    evaluation.special_evaluation(args, imitation_policy, validation_cost_buffer, tflogger, "eval_mode_3", i, 3, error_function)

                    preference_loss_capped, preference_loss, preference_noisy_loss_capped, preference_noisy_loss = evaluation.evaluate_preference_loss(imitation_policy, cost_buffer, error_function)
                    tflogger.log_scalar("Cost Training/raw_preference_loss", preference_loss, i)
                    tflogger.log_scalar("Cost Training/capped_preference_loss", preference_loss_capped, i)
                    tflogger.log_scalar("Cost Training/raw_preference_random_loss", preference_noisy_loss, i)
                    tflogger.log_scalar("Cost Training/capped_preference_random_loss", preference_noisy_loss_capped, i)
                    print("Sampled Error: ", preference_loss, "Sampled Error Capped: ", preference_loss_capped)
                    print("Sampled Random Segment Error: ", preference_noisy_loss, "Sampled Error Random Segment Capped: ", preference_noisy_loss_capped)
                    imitation_policy.special_evaluation_state(cost_buffer, error_function, mode=2)
                    imitation_policy.special_evaluation_state(cost_buffer, error_function, mode=3)
                    print("---------------------------------")
                    imitation_policy.current_cost_model.save_weights("../data/" + env_name + "_cost_model/cost_model_delay_" + str(args.intervention_delay) + "_70_" + str(args.cost_version) + "_1_v" + str(args.buffer_id) + "_mod_" + str(int(args.coefficient)) +  "_bias_" + str(0.0) + "_l2_" + str(args.l2) + "_seed_" + str(args.seed))
            imitation_policy.current_cost_model.save_weights("../data/" + env_name + "_cost_model/cost_model_delay_" + str(args.intervention_delay) + + "_70_" + str(args.cost_version) + "_1_v" + str(args.buffer_id) + "_mod_" + str(int(args.coefficient)) +  "_bias_" + str(0.0) + "_l2_" + str(args.l2) + "_seed_" + str(args.seed))
            evaluation.special_evaluation(args, imitation_policy, validation_cost_buffer, tflogger, "eval_mode_2", i, 2, error_function)
            evaluation.special_evaluation(args, imitation_policy, validation_cost_buffer, tflogger, "eval_mode_3", i, 3, error_function)

            preference_loss_capped, preference_loss, preference_noisy_loss_capped, preference_noisy_loss = evaluation.evaluate_preference_loss(imitation_policy, cost_buffer, error_function)
            tflogger.log_scalar("Cost Training/raw_preference_loss", preference_loss, i)
            tflogger.log_scalar("Cost Training/capped_preference_loss", preference_loss_capped, i)
            tflogger.log_scalar("Cost Training/raw_preference_random_loss", preference_noisy_loss, i)
            tflogger.log_scalar("Cost Training/capped_preference_random_loss", preference_noisy_loss_capped, i)
            print("Sampled Error: ", preference_loss, "Sampled Error Capped: ", preference_loss_capped)
            print("Sampled Random Segment Error: ", preference_noisy_loss, "Sampled Error Random Segment Capped: ",preference_noisy_loss_capped)
            print("Done pretraining cost model...")
    quit()

    #value current cost policy for fun ...
    cost_loss_list= []
    for _ in range(args.policy_training_steps):
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
                intervention_thresholds, intervention_steps = evaluation.evaluate(args, env, imitation_policy, expert_policy, cost_distribution, intervention_thresholds_list, error_function,
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
                intervention_thresholds, intervention_steps = evaluation.evaluate(args, env, imitation_policy, expert_policy,
                                                                       cost_distribution, intervention_thresholds_list, error_function,
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

        if args.cost_version == 2:
            bc_loss = imitation_policy.pretrain_step_actor(cost_buffer) #BC ONLY
            policy_loss_list.append(0)
            bc_loss_list.append(bc_loss)
        else:
            policy_loss, bc_loss, cost_loss = imitation_policy.train_step_actor(None, cost_buffer, expert_policy) #With Cost
            policy_loss_list.append(policy_loss)
            bc_loss_list.append(bc_loss)


        cost_loss_list.append(cost_loss)
        if frame_number % 1000 == 0:
            tflogger.log_scalar("RAW/Policy_loss", np.mean(policy_loss_list), frame_number)
            tflogger.log_scalar("RAW/BC_loss", np.mean(bc_loss_list), frame_number)
            tflogger.log_scalar("RAW/Policy_Cost_loss", np.mean(cost_loss_list), frame_number)

            policy_loss_list = []
            bc_loss_list = []
            cost_loss_list = []



if __name__ == "__main__":
    main()