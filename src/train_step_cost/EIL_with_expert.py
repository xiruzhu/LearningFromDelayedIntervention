import numpy as np
import tensorflow as tf

def train_step_cost(actor, cost_buffer, error_function, activate_loss=False, expert_policy=None,
                               verbose=False, validation_buffer=None, group_size=4):

    """
    Trains one step of the cost function with EIL with expert loss
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

        segment_batch_size = 4
        s_t_raw_preintervention, a_t_raw_preintervention, r_t_raw_preintervention, terminal_t_raw_preintervention, expert_only_s_t_raw_preintervention, \
        s_t_1_raw_preintervention, mean_label_t_raw_preintervention, raw_label_t_list_raw_preintervention, \
        expert_a_t_raw_preintervention, preintervention_raw_segment_intervention_level, segment_weight_1 = cost_buffer.sample_combined(
            [3], [segment_batch_size])

        s_t_raw_preintervention_2, a_t_raw_preintervention_2, r_t_raw_preintervention_2, terminal_t_raw_preintervention_2, expert_only_s_t_raw_preintervention_2, \
        s_t_1_raw_preintervention_2, mean_label_t_raw_preintervention_2, raw_label_t_list_raw_preintervention_2, \
        expert_a_t_raw_preintervention_2, preintervention_raw_segment_intervention_level_2, segment_weight_2, = cost_buffer.sample_combined(
            [2], [segment_batch_size * 6])

        s_t_reshape_1 = np.reshape(s_t_raw_preintervention, [-1, actor.state_dim[0]])
        s_t_reshape_2 = np.reshape(s_t_raw_preintervention_2[:, -5:], [-1, actor.state_dim[0]])

        a_t_reshape_1 = np.reshape(a_t_raw_preintervention, [-1, actor.action_dim])
        a_t_reshape_2 = np.reshape(a_t_raw_preintervention_2[:, -5:], [-1, actor.action_dim])

        inbetween_cost_t, _, _ = actor.current_cost_model([s_t_reshape_1, a_t_reshape_1])
        preint_cost_t, _, _ = actor.current_cost_model([s_t_reshape_2, a_t_reshape_2])
        EID_raw_error = -(tf.math.log(1 - inbetween_cost_t + 1e-8) + tf.math.log(preint_cost_t + 1e-8))
        EID_error = tf.reduce_mean(EID_raw_error)

        if verbose:
            print("Mean Inbetween Error: ", np.mean(inbetween_cost_t), "Mean Preint Error: ",
                  np.mean(preint_cost_t))

        l2_loss = 0
        layers = 0
        for v in actor.current_cost_model.trainable_weights:
            if 'bias' not in v.name and "cost_network" in v.name:
                l2_loss += tf.reduce_mean(tf.nn.l2_loss(v)) * actor.args.l2
                layers += 1
        l2_loss = l2_loss / layers
        cost_loss = EID_error + buffer_expert_loss + l2_loss

        grads = tape.gradient(cost_loss, actor.current_cost_model.trainable_weights)
        actor.current_cost_optimizer.apply_gradients(zip(grads, actor.current_cost_model.trainable_weights))
    return cost_loss, np.mean(EID_raw_error)
