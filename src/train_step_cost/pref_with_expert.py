import numpy as np
import tensorflow as tf

def train_step_cost(actor, cost_buffer, error_function, preference_dataset,
                                           activate_loss=False, expert_policy=None, verbose=False,
                                           validation_buffer=None):
    """
    Trains one step of the cost function sampled preferences
    Assumes preference dataset is provided and uses noisy expert loss in addition to the preference loss
    """
    with tf.GradientTape() as tape:
        s_t, a_t, _, _, intervention_status_t, _, label_t, _, _ = cost_buffer.simple_sample(actor.args.batch_size,
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

        s_t_raw_1 = []
        a_t_raw_1 = []
        expert_only_t_raw_1 = []
        expert_a_t_raw_1 = []
        raw_segment_intervention_level_1 = []

        s_t_raw_2 = []
        a_t_raw_2 = []
        expert_only_t_raw_2 = []
        expert_a_t_raw_2 = []
        raw_segment_intervention_level_2 = []

        sampled_indices = np.random.permutation(len(preference_dataset))
        for i in range(segment_batch_size):
            segment_i, segment_j = preference_dataset[sampled_indices[i]]

            segment_s_t, segment_s_t_1, segment_a_t, segment_terminal_t, segment_r_t, segment_label_t, \
            segment_intervention_status_t, segment_expert_a_t, segment_intervention_threshold = segment_i

            s_t_raw_1.append(segment_s_t)
            a_t_raw_1.append(segment_a_t)
            expert_only_t_raw_1.append(segment_intervention_status_t)
            expert_a_t_raw_1.append(segment_expert_a_t)
            raw_segment_intervention_level_1.append(segment_intervention_threshold)

            segment_s_t, segment_s_t_1, segment_a_t, segment_terminal_t, segment_r_t, segment_label_t, \
            segment_intervention_status_t, segment_expert_a_t, segment_intervention_threshold = segment_j

            s_t_raw_2.append(segment_s_t)
            a_t_raw_2.append(segment_a_t)
            expert_only_t_raw_2.append(segment_intervention_status_t)
            expert_a_t_raw_2.append(segment_expert_a_t)
            raw_segment_intervention_level_2.append(segment_intervention_threshold)

        s_t_raw_1 = np.squeeze(np.array(s_t_raw_1))
        a_t_raw_1 = np.squeeze(np.array(a_t_raw_1))
        expert_only_t_raw_1 = np.squeeze(np.array(expert_only_t_raw_1))
        expert_a_t_raw_1 = np.squeeze(np.array(expert_a_t_raw_1))
        raw_segment_intervention_level_1 = np.squeeze(np.array(raw_segment_intervention_level_1))
        supervision_thresholds_1 = np.max(np.squeeze(raw_segment_intervention_level_1), axis=1)

        s_t_raw_2 = np.squeeze(np.array(s_t_raw_2))
        a_t_raw_2 = np.squeeze(np.array(a_t_raw_2))
        expert_only_t_raw_2 = np.squeeze(np.array(expert_only_t_raw_2))
        expert_a_t_raw_2 = np.squeeze(np.array(expert_a_t_raw_2))
        raw_segment_intervention_level_2 = np.squeeze(np.array(raw_segment_intervention_level_2))
        supervision_thresholds_2 = np.max(np.squeeze(raw_segment_intervention_level_2), axis=1)

        noise = np.random.uniform(0.0, 0.0, size=a_t_raw_1.shape)
        random_actions_raw_1 = np.clip(noise + a_t_raw_1, -1, 1)
        random_actions_1 = np.reshape(random_actions_raw_1, [-1, actor.action_dim]).astype(np.float32)
        offset_error_1, _ = error_function(a_t_raw_1, random_actions_raw_1)

        noise_2 = np.random.uniform(0.0, 0.0, size=a_t_raw_2.shape)
        random_actions_raw_2 = np.clip(noise_2 + a_t_raw_2, -1, 1)
        random_actions_2 = np.reshape(random_actions_raw_2, [-1, actor.action_dim]).astype(np.float32)
        offset_error_2, _ = error_function(a_t_raw_2, random_actions_raw_2)

        s_t_raw = np.concatenate([s_t_raw_1, s_t_raw_2], axis=0)
        s_t = np.reshape(s_t_raw, [-1, actor.state_dim[0]])
        a_t_noisy = np.concatenate([random_actions_1, random_actions_2], axis=0)

        expert_intervention_t_raw = np.concatenate([expert_only_t_raw_1, expert_only_t_raw_2], axis=0)
        expert_intervention_t = np.reshape(expert_intervention_t_raw, [-1, 1])

        noisy_cost_s_t, cost_distribution, max_ent_loss = actor.current_cost_model([s_t, a_t_noisy])
        noisy_cost_s_t_expert_no_grad = noisy_cost_s_t * (
                1 - expert_intervention_t)  # + tf.stop_gradient(noisy_cost_s_t) * expert_intervention_t
        noisy_cost_s_t_expert_no_grad_reshaped = tf.reshape(noisy_cost_s_t_expert_no_grad,
                                                            [s_t_raw.shape[0], actor.args.segment_length])
        single_cost_1, single_cost_2 = tf.split(noisy_cost_s_t_expert_no_grad_reshaped, 2, axis=0)
        noisy_estimated_cost_reshaped = tf.reshape(noisy_cost_s_t_expert_no_grad,
                                                   [s_t_raw.shape[0], actor.args.segment_length])
        noisy_estimated_cost_segment = tf.reduce_sum(noisy_estimated_cost_reshaped,
                                                     axis=1)  # prevent instability issues ...
        noisy_estimated_intervention_probability = noisy_estimated_cost_segment / actor.args.segment_length
        noisy_estimated_cost_1, noisy_estimated_cost_2 = tf.split(noisy_estimated_intervention_probability, 2,
                                                                  axis=0)

        coef = actor.args.coefficient

        true_cost_no_noise_1, true_per_state_error_1 = error_function(expert_a_t_raw_1, random_actions_raw_1)
        true_cost_no_noise_2, true_per_state_error_2 = error_function(expert_a_t_raw_2, random_actions_raw_2)

        preference_loss_1, _, _ = actor.get_preference_loss_true_v8(None, None, noisy_estimated_cost_1,
                                                                   noisy_estimated_cost_2,
                                                                   true_cost_no_noise_1 * coef,
                                                                   true_cost_no_noise_2 * coef,
                                                                   single_cost_1, single_cost_2,
                                                                   None,
                                                                   None, offset_error_1,
                                                                   offset_error_2,
                                                                   is_random=None, coef=coef, sanity_check=False)

        preference_loss_3, l2_error_3, _ = actor.get_preference_loss_true_v6(None, None, noisy_estimated_cost_2,
                                                                            np.ones_like(noisy_estimated_cost_2),
                                                                            true_cost_no_noise_2 * coef,
                                                                            np.ones_like(
                                                                                true_cost_no_noise_2) * coef,
                                                                            single_cost_1, single_cost_2,
                                                                            None, None, offset_error_1,
                                                                            offset_error_2,
                                                                            is_random=None, coef=coef,
                                                                            sanity_check=False)

        preference_loss = preference_loss_1 + preference_loss_3  # + preference_loss_4
        loss_weight = 0.01
        classification_loss = tf.reduce_mean(
            preference_loss) * loss_weight
        if verbose:
            cost_gt_0_5 = np.mean(noisy_estimated_cost_reshaped > 0.5, axis=1)

            error_single_state_1_l2 = tf.reduce_mean(tf.abs(single_cost_1 - true_per_state_error_1) ** 2)
            error_single_state_2_l2 = tf.reduce_mean(tf.abs(single_cost_2 - true_per_state_error_2) ** 2)

            error_single_state_1_l1 = tf.reduce_mean(tf.abs(single_cost_1 - true_per_state_error_1))
            error_single_state_2_l1 = tf.reduce_mean(tf.abs(single_cost_2 - true_per_state_error_2))

            print(error_single_state_1_l1.shape)
            print("\n")
            print(np.mean(preference_loss), np.mean(classification_loss))
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
                      preference_loss[idx].numpy(), supervision_thresholds_1[idx], supervision_thresholds_2[idx])
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
