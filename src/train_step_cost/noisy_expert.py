import numpy as np
import tensorflow as tf


def train_step_cost(actor, cost_buffer, error_function, expert_policy=None, verbose=False, validation_buffer=None):
    """
    Trains one step of the cost function with only noisy loss
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
        buffer_expert_loss = tf.reduce_mean(buffer_expert_loss_raw)

        l2_loss = 0
        layers = 0
        for v in actor.current_cost_model.trainable_weights:
            if 'bias' not in v.name and "cost_network" in v.name:
                l2_loss += tf.reduce_mean(tf.nn.l2_loss(v)) * actor.args.l2
                layers += 1
        l2_loss = l2_loss / layers
        cost_loss = buffer_expert_loss + l2_loss

        grads = tape.gradient(cost_loss, actor.current_cost_model.trainable_weights)
        actor.current_cost_optimizer.apply_gradients(zip(grads, actor.current_cost_model.trainable_weights))
    return buffer_expert_loss, 0