import numpy as np
import tensorflow as tf

def train_step_cost(actor, cost_buffer, error_function, activate_loss=False, expert_policy=None,
                            verbose=False):
    with tf.GradientTape() as tape:
        s_t, a_t, _, _, intervention_status_t, _, label_t, _, _ = cost_buffer.simple_sample(
            actor.args.batch_size,
            mode=1)

        a_t = a_t.astype(np.float32)
        double_a_t = tf.concat([a_t, a_t], axis=0)
        double_s_t = tf.concat([s_t, s_t], axis=0)

        uniform_noise_scale = np.clip(np.random.uniform(-0.1, 1.2, size=(a_t.shape[0], 1)), 0, 2)
        action_low_noise = tf.clip_by_value(tf.random.normal(a_t.shape) * uniform_noise_scale,
                                            -uniform_noise_scale * 2.5, uniform_noise_scale * 2.5)
        action_high_noise = tf.clip_by_value(tf.random.normal(a_t.shape) * uniform_noise_scale,
                                             -uniform_noise_scale * 2.5, uniform_noise_scale * 2.5)
        action_noise = tf.concat([action_low_noise, action_high_noise], axis=0)

        noisy_a_t = tf.clip_by_value(double_a_t + action_noise, -1, 1)
        cost_label, _ = error_function(noisy_a_t, double_a_t)
        cost_label = np.expand_dims(cost_label, axis=1)

        noisy_cost_s_t, _, _ = actor.current_cost_model([double_s_t, noisy_a_t])
        buffer_expert_loss_raw = (noisy_cost_s_t - cost_label) ** 2
        buffer_expert_loss = tf.reduce_mean(
            buffer_expert_loss_raw)
        segment_batch_size = 16
        s_t_raw_preintervention, a_t_raw_preintervention, r_t_raw_preintervention, terminal_t_raw_preintervention, expert_only_s_t_raw_preintervention, \
        s_t_1_raw_preintervention, mean_label_t_raw_preintervention, raw_label_t_list_raw_preintervention, \
        expert_a_t_raw_preintervention, preintervention_raw_segment_intervention_level, segment_weight_1 = cost_buffer.sample_combined(
            [3], [segment_batch_size])

        a_t_preintervention = np.reshape(a_t_raw_preintervention, [-1, actor.action_dim]).astype(np.float32)
        is_preint_1 = np.concatenate([np.zeros((segment_batch_size,))], axis=0)
        is_expert_1 = np.concatenate([np.zeros((segment_batch_size,))], axis=0)
        supervision_thresholds_1 = np.max(preintervention_raw_segment_intervention_level, axis=1)

        s_t_raw_preintervention_2, a_t_raw_preintervention_2, r_t_raw_preintervention_2, terminal_t_raw_preintervention_2, expert_only_s_t_raw_preintervention_2, \
        s_t_1_raw_preintervention_2, mean_label_t_raw_preintervention_2, raw_label_t_list_raw_preintervention_2, \
        expert_a_t_raw_preintervention_2, preintervention_raw_segment_intervention_level_2, segment_weight_2, is_random_2 = cost_buffer.sample_preint(
            segment_batch_size, supervision_thresholds_1, is_preint_1, is_expert_1, preint_include_eq=False,
            all_legal=False)

        supervision_thresholds_2 = np.max(preintervention_raw_segment_intervention_level_2, axis=1)
        a_t_preintervention_2 = np.reshape(a_t_raw_preintervention_2, [-1, actor.action_dim]).astype(np.float32)

        noise = np.random.uniform(0.0, 0.0, size=a_t_raw_preintervention.shape)
        random_actions_raw_1 = np.clip(noise + a_t_raw_preintervention, -1, 1)
        random_actions_1 = np.reshape(random_actions_raw_1, [-1, actor.action_dim]).astype(np.float32)
        offset_error_1, _ = error_function(a_t_raw_preintervention, random_actions_raw_1)

        noise_2 = np.random.uniform(0.0, 0.0, size=a_t_raw_preintervention_2.shape)
        random_actions_raw_2 = np.clip(noise_2 + a_t_raw_preintervention_2, -1, 1)
        random_actions_2 = np.reshape(random_actions_raw_2, [-1, actor.action_dim]).astype(np.float32)
        offset_error_2, _ = error_function(a_t_raw_preintervention_2, random_actions_raw_2)

        noise_3 = np.random.normal(loc=0, scale=0.2, size=a_t_raw_preintervention.shape)
        random_actions_raw_3 = np.clip(noise_3 + a_t_raw_preintervention, -1, 1)
        random_actions_3 = np.reshape(random_actions_raw_3, [-1, actor.action_dim]).astype(np.float32)
        offset_error_3, offset_single_error_3 = error_function(a_t_raw_preintervention, random_actions_raw_3)

        noise_4 = np.random.normal(loc=0, scale=0.2, size=a_t_raw_preintervention_2.shape)
        random_actions_raw_4 = np.clip(noise_4 + a_t_raw_preintervention_2, -1, 1)
        random_actions_4 = np.reshape(random_actions_raw_4, [-1, actor.action_dim]).astype(np.float32)
        offset_error_4, offset_single_error_4 = error_function(a_t_raw_preintervention_2, random_actions_raw_4)
        s_t_raw = np.concatenate([s_t_raw_preintervention, s_t_raw_preintervention_2, s_t_raw_preintervention,
                                  s_t_raw_preintervention_2], axis=0)
        s_t = np.reshape(s_t_raw, [-1, actor.state_dim[0]])
        a_t_noisy = np.concatenate([random_actions_1, random_actions_2, random_actions_3, random_actions_4], axis=0)

        expert_intervention_t_raw = np.concatenate(
            [expert_only_s_t_raw_preintervention, expert_only_s_t_raw_preintervention_2], axis=0)

        noisy_cost_s_t, cost_distribution, max_ent_loss = actor.current_cost_model([s_t, a_t_noisy])

        noisy_cost_s_t_expert_no_grad = noisy_cost_s_t  # + tf.stop_gradient(noisy_cost_s_t) * expert_intervention_t
        noisy_cost_s_t_expert_no_grad_reshaped = tf.reshape(noisy_cost_s_t_expert_no_grad,
                                                            [s_t_raw.shape[0], actor.args.segment_length])
        single_cost_1, single_cost_2, single_cost_3, single_cost_4 = tf.split(
            noisy_cost_s_t_expert_no_grad_reshaped, 4, axis=0)

        noisy_estimated_cost_reshaped = tf.reshape(noisy_cost_s_t_expert_no_grad,
                                                   [s_t_raw.shape[0], actor.args.segment_length])
        noisy_estimated_cost_segment = tf.reduce_sum(noisy_estimated_cost_reshaped,
                                                     axis=1)  # prevent instability issues ...
        noisy_estimated_intervention_probability = noisy_estimated_cost_segment / actor.args.segment_length
        noisy_estimated_cost_1, noisy_estimated_cost_2, noisy_estimated_cost_3, noisy_estimated_cost_4 = tf.split(
            noisy_estimated_intervention_probability, 4, axis=0)

        coef = actor.args.coefficient
        true_cost_no_noise_1, true_per_state_error_1 = error_function(expert_a_t_raw_preintervention,
                                                                      random_actions_raw_1)
        true_cost_no_noise_2, true_per_state_error_2 = error_function(expert_a_t_raw_preintervention_2,
                                                                      random_actions_raw_2)
        preference_loss_1, l2_error, _ = actor.get_preference_loss_true_v6(None, None, noisy_estimated_cost_1,
                                                                          noisy_estimated_cost_2,
                                                                          true_cost_no_noise_1 * coef,
                                                                          true_cost_no_noise_2 * coef,
                                                                          supervision_thresholds_1,
                                                                          supervision_thresholds_2,
                                                                          np.zeros_like(is_preint_1),
                                                                          None, offset_error_1,
                                                                          offset_error_2,
                                                                          is_random=None, coef=coef,
                                                                          sanity_check=True)

        preference_loss_2, l2_error, _ = actor.get_preference_loss_true_v6(None, None, noisy_estimated_cost_1,
                                                                          noisy_estimated_cost_4,
                                                                          true_cost_no_noise_1 * coef,
                                                                          true_cost_no_noise_2 * coef,
                                                                          supervision_thresholds_1,
                                                                          supervision_thresholds_2,
                                                                          np.zeros_like(is_preint_1),
                                                                          None, offset_error_1,
                                                                          offset_error_2,
                                                                          is_random=None, coef=coef,
                                                                          sanity_check=True)

        preference_loss_3, l2_error_3, _ = actor.get_preference_loss_true_v6(None, None, noisy_estimated_cost_2,
                                                                            np.ones_like(noisy_estimated_cost_2),
                                                                            true_cost_no_noise_2 * coef,
                                                                            np.ones_like(
                                                                                true_cost_no_noise_2) * coef,
                                                                            single_cost_1, single_cost_2,
                                                                            np.zeros_like(is_preint_1),
                                                                            None, offset_error_1,
                                                                            offset_error_2,
                                                                            is_random=None, coef=coef,
                                                                            sanity_check=False)
        new_offset_single_error_3 = 1 - np.clip(
            np.mean(np.abs(a_t_raw_preintervention - random_actions_raw_3), axis=2), 0, 0.5) / 0.5
        new_offset_single_error_3 = new_offset_single_error_3.astype(np.float32) ** 2
        new_offset_single_error_4 = 1 - np.clip(
            np.mean(np.abs(a_t_raw_preintervention_2 - random_actions_raw_4), axis=2), 0, 0.5) / 0.5
        new_offset_single_error_4 = new_offset_single_error_4.astype(np.float32) ** 2
        simple_l2_loss = (new_offset_single_error_3 * (single_cost_1 - single_cost_3) ** 2
                          + new_offset_single_error_4 * (single_cost_2 - single_cost_4) ** 2)
        simple_l2_loss = tf.reduce_mean(simple_l2_loss) * 0.0001
        preference_loss = preference_loss_1 + preference_loss_3 + preference_loss_2
        classification_loss = tf.reduce_mean(
            preference_loss) * 0.01
        classification_loss = classification_loss * actor.args.cost_loss_modifier + simple_l2_loss

        if verbose:
            cost_gt_0_5 = np.mean(noisy_estimated_cost_reshaped > 0.5, axis=1)

            error_single_state_1_l2 = tf.reduce_mean(tf.abs(single_cost_1 - true_per_state_error_1) ** 2)
            error_single_state_2_l2 = tf.reduce_mean(tf.abs(single_cost_2 - true_per_state_error_2) ** 2)

            error_single_state_1_l1 = tf. \
                reduce_mean(tf.abs(single_cost_1 - true_per_state_error_1))
            error_single_state_2_l1 = tf.reduce_mean(tf.abs(single_cost_2 - true_per_state_error_2))

            print("\n")
            print(np.mean(preference_loss), np.mean(simple_l2_loss), np.mean(classification_loss))
            print("State Err L1, L2:", np.mean(error_single_state_1_l1), np.mean(error_single_state_2_l1),
                  np.mean(error_single_state_1_l2), np.mean(error_single_state_2_l2), )
            print("Current noise", np.mean(noisy_estimated_cost_1) * actor.args.segment_length,
                  np.mean(noisy_estimated_cost_2) * actor.args.segment_length)
            print("True no noise", np.mean(true_cost_no_noise_1) * actor.args.segment_length,
                  np.mean(true_cost_no_noise_2) * actor.args.segment_length)
            print("Segment Cost Diversity: ", np.mean(np.std(noisy_estimated_cost_reshaped, axis=1)))
            for i in range(8):
                idx = np.random.randint(0, segment_batch_size)
                print(idx, noisy_estimated_cost_1[idx].numpy(), noisy_estimated_cost_2[idx].numpy(),
                      ", gt 0.5:", cost_gt_0_5[idx], cost_gt_0_5[idx + segment_batch_size],
                      ", True: ", true_cost_no_noise_1[idx], true_cost_no_noise_2[idx],
                      preference_loss[idx].numpy(), supervision_thresholds_1[idx],
                      supervision_thresholds_2[idx])
            print("-----------------------------")

        l2_loss = 0
        layers = 0
        for v in actor.current_cost_model.trainable_weights:
            if 'bias' not in v.name and "cost_network" in v.name:
                l2_loss += tf.reduce_mean(tf.nn.l2_loss(v)) * actor.args.l2
                layers += 1
        l2_loss = l2_loss / layers
        cost_loss = classification_loss + buffer_expert_loss + l2_loss

        grads = tape.gradient(cost_loss, actor.current_cost_model.trainable_weights)
        actor.current_cost_optimizer.apply_gradients(zip(grads, actor.current_cost_model.trainable_weights))
    return cost_loss, np.mean(preference_loss)
