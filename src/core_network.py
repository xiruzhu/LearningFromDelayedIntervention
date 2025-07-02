import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras.optimizers import Adam
import utils
from tensorflow.keras import regularizers
import math

LOG_SIG_CAP_MAX = 2
LOG_SIG_CAP_MIN = -20


class actor:
    def __init__(self, args, state_dim, action_dim, model_name="ensemble_"):
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.args = args
        self.model_name = model_name
        self.state_input = tf.keras.layers.Input(shape=self.state_dim)
        self.policy_noise = args.policy_noise
        self.noise_clip = args.noise_clip

        self.current_model_list = []
        self.current_vars_list = []
        self.current_optimizer_list = []

        self.bce = tf.keras.losses.BinaryCrossentropy()

        for i in range(self.args.ensemble_size):
            current_model, current_vars, current_optimizer = self.build_network(
                model_name=self.model_name + "_" + str(i) + "_current")
            self.current_model_list.append(current_model)
            self.current_vars_list.append(current_vars)
            self.current_optimizer_list.append(current_optimizer)

        self.huber_loss = tf.keras.losses.Huber(
            delta=1.0,
            reduction='sum_over_batch_size',
            name='huber_loss')
        self.current_cost_model, self.current_cost_vars, self.current_cost_optimizer = self.build_cost(
            model_name="current")
        self.target_cost_model, self.target_cost_vars, self.target_cost_optimizer = self.build_cost(model_name="target")


    def build_cost(self, model_name="current", epsilon=1e-6):
        init = tf.keras.initializers.GlorotNormal()
        action_input = tf.keras.layers.Input(shape=self.action_dim)
        input_value = tf.concat((self.state_input, action_input), axis=1)

        l2 = tf.keras.layers.Dense(self.args.hidden, kernel_initializer=init, activation=tf.nn.relu,
                                   name=model_name + "_cost_network_1")(input_value)

        l3 = tf.keras.layers.Dense(self.args.hidden, kernel_initializer=init, activation=tf.nn.relu,
                                   name=model_name + "_cost_network_2")(l2)

        l4 = tf.keras.layers.Dense(self.args.hidden, kernel_initializer=init, activation=tf.nn.relu,
                                   name=model_name + "_cost_network_3")(l3)
        l5 = tf.keras.layers.Dense(self.args.hidden // 2, kernel_initializer=init, activation=tf.nn.relu,
                                   name=model_name + "_cost_network_4")(l4)

        l6 = tf.keras.layers.Dense(1, kernel_initializer=init, activation=None, name=model_name + "_cost_network_5")(l5)
        mean_cost = (tf.nn.tanh(l6) + 1) / 2
        current_model = tf.keras.Model(inputs=[self.state_input, action_input],
                                       outputs=[mean_cost, mean_cost * 0, mean_cost * 0])

        current_vars = current_model.trainable_weights

        if self.args.cosine_lr:
            decay_steps = 1000
            initial_learning_rate = 0.1
            lr_decayed_fn = tf.keras.optimizers.schedules.CosineDecay(
                initial_learning_rate, decay_steps)
            optimizer = Adam(learning_rate=lr_decayed_fn)
        else:
            optimizer = Adam(learning_rate=self.args.vf_lr)

        return current_model, current_vars, optimizer

    def build_network(self, model_name="current", epsilon=1e-6):
        init = tf.keras.initializers.GlorotNormal()

        l2 = tf.keras.layers.Dense(self.args.hidden, kernel_initializer=init, activation=tf.nn.relu,
                                   name=model_name + "_actor_network_1")(self.state_input)
        l3 = tf.keras.layers.Dense(self.args.hidden, kernel_initializer=init, activation=tf.nn.relu,
                                   name=model_name + "_actor_network_2")(l2)
        l4 = tf.keras.layers.Dense(self.args.hidden, kernel_initializer=init, activation=tf.nn.relu,
                                   name=model_name + "_actor_network_3")(l3)
        l5 = tf.keras.layers.Dense(self.action_dim, kernel_initializer=init, activation=None,
                                   name=model_name + "_actor_network_4")(l4)
        current_action = tf.tanh(l5)

        current_model = tf.keras.Model(inputs=[self.state_input], outputs=[current_action])
        current_vars = current_model.trainable_weights

        optimizer = Adam(learning_rate=self.args.ac_lr)
        return current_model, current_vars, optimizer

    def get_random_action(self):
        return np.random.uniform(-1, 1, size=(self.action_dim))

    def get_current_action(self, state, verbose=False, noisy=False):
        # state = np.expand_dims(state, axis=0)
        # randomly sample a model
        action_list = []
        action = self.current_model_list[0](state)[0]
        action_std = 0
        if noisy:
            noise = np.clip(np.random.normal(size=action.shape) * self.policy_noise, -self.noise_clip, self.noise_clip)
            noisy_actions = np.clip(noise + action, -1, 1)
            return noisy_actions, action_std
        else:
            return action.numpy(), action_std

    def evaluate_preference_sampling(self, cost_buffer):
        segment_batch_size = 1024
        s_t_raw_preintervention, a_t_raw_preintervention, r_t_raw_preintervention, terminal_t_raw_preintervention, expert_only_s_t_raw_preintervention, \
        s_t_1_raw_preintervention, mean_label_t_raw_preintervention, raw_label_t_list_raw_preintervention, \
        expert_a_t_raw_preintervention, preintervention_raw_segment_intervention_level = cost_buffer.sample_combined(
            [3], [segment_batch_size])

        a_t_preintervention = np.reshape(a_t_raw_preintervention, [-1, self.action_dim]).astype(np.float32)
        is_preint_1 = np.zeros(
            (segment_batch_size,))  # (np.mean(raw_label_t_list_raw_preintervention, axis=1) == 1).astype(np.float32)

        supervision_thresholds_1 = np.max(preintervention_raw_segment_intervention_level, axis=1)

        s_t_raw_preintervention_2, a_t_raw_preintervention_2, r_t_raw_preintervention_2, terminal_t_raw_preintervention_2, expert_only_s_t_raw_preintervention_2, \
        s_t_1_raw_preintervention_2, mean_label_t_raw_preintervention_2, raw_label_t_list_raw_preintervention_2, \
        expert_a_t_raw_preintervention_2, preintervention_raw_segment_intervention_level_2 = cost_buffer.sample_preint(
            segment_batch_size, supervision_thresholds_1, is_preint_1, preint_include_eq=False, all_legal=True)
        supervision_thresholds_2 = np.max(preintervention_raw_segment_intervention_level_2, axis=1)
        a_t_preintervention_2 = np.reshape(a_t_raw_preintervention_2, [-1, self.action_dim]).astype(np.float32)

        s_t_raw_preintervention_3, a_t_raw_preintervention_3, r_t_raw_preintervention_3, terminal_t_raw_preintervention_3, expert_only_s_t_raw_preintervention_3, \
        s_t_1_raw_preintervention_3, mean_label_t_raw_preintervention_3, raw_label_t_list_raw_preintervention_3, \
        expert_a_t_raw_preintervention_3, preintervention_raw_segment_intervention_level_3 = cost_buffer.sample_preint(
            segment_batch_size, supervision_thresholds_2, np.ones_like(is_preint_1), preint_include_eq=False,
            all_legal=True)
        supervision_thresholds_3 = np.max(preintervention_raw_segment_intervention_level_3, axis=1)
        a_t_preintervention_3 = np.reshape(a_t_raw_preintervention_3, [-1, self.action_dim]).astype(np.float32)

        pre_true_cost_no_noise_1 = np.clip(
            np.mean(np.abs(expert_a_t_raw_preintervention - a_t_raw_preintervention), axis=2), 0, 1) / 1
        true_cost_no_noise_1 = np.mean(pre_true_cost_no_noise_1, axis=1)

        pre_true_cost_no_noise_2 = np.clip(
            np.mean(np.abs(expert_a_t_raw_preintervention_2 - a_t_raw_preintervention_2), axis=2), 0, 1) / 1
        true_cost_no_noise_2 = np.mean(pre_true_cost_no_noise_2, axis=1)

        pre_true_cost_no_noise_3 = np.clip(
            np.mean(np.abs(expert_a_t_raw_preintervention_3 - a_t_raw_preintervention_3), axis=2), 0, 1) / 1
        true_cost_no_noise_3 = np.mean(pre_true_cost_no_noise_3, axis=1)

        random_sampled_data = np.squeeze(supervision_thresholds_3 == supervision_thresholds_2).astype(np.float32)

        mean_difference_1 = np.mean(true_cost_no_noise_2 - true_cost_no_noise_1) * self.args.coefficient
        mean_difference_2 = np.mean(true_cost_no_noise_3 - true_cost_no_noise_2) * self.args.coefficient
        mean_difference_3 = np.sum((true_cost_no_noise_3 - true_cost_no_noise_2) * (1 - random_sampled_data)) / (
            np.sum(1 - random_sampled_data)) * self.args.coefficient
        print(true_cost_no_noise_1[0])
        print("Avg difference between inbetween/preint: ", mean_difference_1)
        print("Avg difference between preint/preint: ", mean_difference_2)
        print("Avg difference between preint/preint ignore random: ", mean_difference_3)
        quit()


    def train_step_EIL_without_expert(self, cost_buffer, error_function, activate_loss=False, expert_policy=None,
                                      verbose=False, validation_buffer=None, group_size=4):
        with tf.GradientTape() as tape:
            segment_batch_size = 4
            s_t_raw_preintervention, a_t_raw_preintervention, r_t_raw_preintervention, terminal_t_raw_preintervention, expert_only_s_t_raw_preintervention, \
            s_t_1_raw_preintervention, mean_label_t_raw_preintervention, raw_label_t_list_raw_preintervention, \
            expert_a_t_raw_preintervention, preintervention_raw_segment_intervention_level, segment_weight_1 = cost_buffer.sample_combined(
                [3], [segment_batch_size])

            s_t_raw_preintervention_2, a_t_raw_preintervention_2, r_t_raw_preintervention_2, terminal_t_raw_preintervention_2, expert_only_s_t_raw_preintervention_2, \
            s_t_1_raw_preintervention_2, mean_label_t_raw_preintervention_2, raw_label_t_list_raw_preintervention_2, \
            expert_a_t_raw_preintervention_2, preintervention_raw_segment_intervention_level_2, segment_weight_2, = cost_buffer.sample_combined(
                [2], [segment_batch_size * 6])

            s_t_reshape_1 = np.reshape(s_t_raw_preintervention, [-1, self.state_dim[0]])
            s_t_reshape_2 = np.reshape(s_t_raw_preintervention_2[:, -5:], [-1, self.state_dim[0]])

            a_t_reshape_1 = np.reshape(a_t_raw_preintervention, [-1, self.action_dim])
            a_t_reshape_2 = np.reshape(a_t_raw_preintervention_2[:, -5:], [-1, self.action_dim])

            inbetween_cost_t, _, _ = self.current_cost_model([s_t_reshape_1, a_t_reshape_1])
            preint_cost_t, _, _ = self.current_cost_model([s_t_reshape_2, a_t_reshape_2])
            EID_raw_error = -(tf.math.log(1 - inbetween_cost_t + 1e-8) + tf.math.log(preint_cost_t + 1e-8))
            EID_error = tf.reduce_mean(EID_raw_error)

            if verbose:
                print("Mean Inbetween Error: ", np.mean(inbetween_cost_t), "Mean Preint Error: ",
                      np.mean(preint_cost_t))

            l2_loss = 0
            layers = 0
            for v in self.current_cost_model.trainable_weights:
                if 'bias' not in v.name and "cost_network" in v.name:
                    l2_loss += tf.reduce_mean(tf.nn.l2_loss(v)) * self.args.l2
                    layers += 1
            l2_loss = l2_loss / layers
            cost_loss = EID_error + l2_loss

            grads = tape.gradient(cost_loss, self.current_cost_model.trainable_weights)
            self.current_cost_optimizer.apply_gradients(zip(grads, self.current_cost_model.trainable_weights))
        return cost_loss, np.mean(EID_raw_error)

    def train_step_EIL_with_expert(self, cost_buffer, error_function, activate_loss=False, expert_policy=None,
                                   verbose=False, validation_buffer=None, group_size=4):
        with tf.GradientTape() as tape:
            s_t, a_t, _, _, intervention_status_t, _, label_t, _, _ = cost_buffer.simple_sample(self.args.batch_size,
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

            noisy_cost_s_t, _, _ = self.current_cost_model([double_s_t, noisy_a_t])
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

            s_t_reshape_1 = np.reshape(s_t_raw_preintervention, [-1, self.state_dim[0]])
            s_t_reshape_2 = np.reshape(s_t_raw_preintervention_2[:, -5:], [-1, self.state_dim[0]])

            a_t_reshape_1 = np.reshape(a_t_raw_preintervention, [-1, self.action_dim])
            a_t_reshape_2 = np.reshape(a_t_raw_preintervention_2[:, -5:], [-1, self.action_dim])

            inbetween_cost_t, _, _ = self.current_cost_model([s_t_reshape_1, a_t_reshape_1])
            preint_cost_t, _, _ = self.current_cost_model([s_t_reshape_2, a_t_reshape_2])
            EID_raw_error = -(tf.math.log(1 - inbetween_cost_t + 1e-8) + tf.math.log(preint_cost_t + 1e-8))
            EID_error = tf.reduce_mean(EID_raw_error)

            if verbose:
                print("Mean Inbetween Error: ", np.mean(inbetween_cost_t), "Mean Preint Error: ",
                      np.mean(preint_cost_t))

            l2_loss = 0
            layers = 0
            for v in self.current_cost_model.trainable_weights:
                if 'bias' not in v.name and "cost_network" in v.name:
                    l2_loss += tf.reduce_mean(tf.nn.l2_loss(v)) * self.args.l2
                    layers += 1
            l2_loss = l2_loss / layers
            cost_loss = EID_error + buffer_expert_loss + l2_loss

            grads = tape.gradient(cost_loss, self.current_cost_model.trainable_weights)
            self.current_cost_optimizer.apply_gradients(zip(grads, self.current_cost_model.trainable_weights))
        return cost_loss, np.mean(EID_raw_error)

    def train_step_cost_true_v5(self, cost_buffer, error_function, activate_loss=False, expert_policy=None,
                                verbose=False, validation_buffer=None, group_size=4):
        with tf.GradientTape() as tape:
            s_t, a_t, _, _, intervention_status_t, _, label_t, _, _ = cost_buffer.simple_sample(
                self.args.batch_size,
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

            noisy_cost_s_t, _, _ = self.current_cost_model([double_s_t, noisy_a_t])
            buffer_expert_loss_raw = (noisy_cost_s_t - cost_label) ** 2
            # noisy_cost_l2 = (noisy_cost_s_t - 0.5) ** 2 * 0.003
            buffer_expert_loss = tf.reduce_mean(
                buffer_expert_loss_raw)  # + tf.reduce_mean(noisy_cost_l2)  # + tf.reduce_mean(segment_preference_loss_raw) + tf.reduce_mean(intervention_loss_raw)

            segment_batch_size = 16
            s_t_raw_preintervention, a_t_raw_preintervention, r_t_raw_preintervention, terminal_t_raw_preintervention, expert_only_s_t_raw_preintervention, \
            s_t_1_raw_preintervention, mean_label_t_raw_preintervention, raw_label_t_list_raw_preintervention, \
            expert_a_t_raw_preintervention, preintervention_raw_segment_intervention_level, segment_weight_1 = cost_buffer.sample_combined(
                [3], [segment_batch_size])

            a_t_preintervention = np.reshape(a_t_raw_preintervention, [-1, self.action_dim]).astype(np.float32)
            is_preint_1 = np.concatenate([np.zeros((segment_batch_size,))], axis=0)
            is_expert_1 = np.concatenate([np.zeros((segment_batch_size,))], axis=0)
            supervision_thresholds_1 = np.max(preintervention_raw_segment_intervention_level, axis=1)

            s_t_raw_preintervention_2, a_t_raw_preintervention_2, r_t_raw_preintervention_2, terminal_t_raw_preintervention_2, expert_only_s_t_raw_preintervention_2, \
            s_t_1_raw_preintervention_2, mean_label_t_raw_preintervention_2, raw_label_t_list_raw_preintervention_2, \
            expert_a_t_raw_preintervention_2, preintervention_raw_segment_intervention_level_2, segment_weight_2, is_random_2 = cost_buffer.sample_preint(
                segment_batch_size, supervision_thresholds_1, is_preint_1, is_expert_1, preint_include_eq=False,
                all_legal=False)

            supervision_thresholds_2 = np.max(preintervention_raw_segment_intervention_level_2, axis=1)
            a_t_preintervention_2 = np.reshape(a_t_raw_preintervention_2, [-1, self.action_dim]).astype(np.float32)

            noise = np.random.uniform(0.0, 0.0, size=a_t_raw_preintervention.shape)
            random_actions_raw_1 = np.clip(noise + a_t_raw_preintervention, -1, 1)
            random_actions_1 = np.reshape(random_actions_raw_1, [-1, self.action_dim]).astype(np.float32)
            offset_error_1, _ = error_function(a_t_raw_preintervention, random_actions_raw_1)

            noise_2 = np.random.uniform(0.0, 0.0, size=a_t_raw_preintervention_2.shape)
            random_actions_raw_2 = np.clip(noise_2 + a_t_raw_preintervention_2, -1, 1)
            random_actions_2 = np.reshape(random_actions_raw_2, [-1, self.action_dim]).astype(np.float32)
            offset_error_2, _ = error_function(a_t_raw_preintervention_2, random_actions_raw_2)

            noise_3 = np.random.normal(loc=0, scale=0.1, size=a_t_raw_preintervention.shape)
            random_actions_raw_3 = np.clip(noise_3 + a_t_raw_preintervention, -1, 1)
            random_actions_3 = np.reshape(random_actions_raw_3, [-1, self.action_dim]).astype(np.float32)
            offset_error_3, offset_single_error_3 = error_function(a_t_raw_preintervention, random_actions_raw_3)

            noise_4 = np.random.normal(loc=0, scale=0.1, size=a_t_raw_preintervention_2.shape)
            random_actions_raw_4 = np.clip(noise_4 + a_t_raw_preintervention_2, -1, 1)
            random_actions_4 = np.reshape(random_actions_raw_4, [-1, self.action_dim]).astype(np.float32)
            offset_error_4, offset_single_error_4 = error_function(a_t_raw_preintervention_2, random_actions_raw_4)

            s_t_raw = np.concatenate([s_t_raw_preintervention, s_t_raw_preintervention_2, s_t_raw_preintervention,
                                      s_t_raw_preintervention_2], axis=0)
            s_t = np.reshape(s_t_raw, [-1, self.state_dim[0]])
            a_t_noisy = np.concatenate([random_actions_1, random_actions_2, random_actions_3, random_actions_4], axis=0)

            expert_intervention_t_raw = np.concatenate(
                [expert_only_s_t_raw_preintervention, expert_only_s_t_raw_preintervention_2], axis=0)
            # expert_intervention_t = np.reshape(expert_intervention_t_raw, [-1, 1])

            noisy_cost_s_t, cost_distribution, max_ent_loss = self.current_cost_model([s_t, a_t_noisy])

            noisy_cost_s_t_expert_no_grad = noisy_cost_s_t  # + tf.stop_gradient(noisy_cost_s_t) * expert_intervention_t
            noisy_cost_s_t_expert_no_grad_reshaped = tf.reshape(noisy_cost_s_t_expert_no_grad,
                                                                [s_t_raw.shape[0], self.args.segment_length])
            single_cost_1, single_cost_2, single_cost_3, single_cost_4 = tf.split(
                noisy_cost_s_t_expert_no_grad_reshaped, 4, axis=0)

            noisy_estimated_cost_reshaped = tf.reshape(noisy_cost_s_t_expert_no_grad,
                                                       [s_t_raw.shape[0], self.args.segment_length])
            noisy_estimated_cost_segment = tf.reduce_sum(noisy_estimated_cost_reshaped,
                                                         axis=1)  # prevent instability issues ...
            noisy_estimated_intervention_probability = noisy_estimated_cost_segment / self.args.segment_length
            noisy_estimated_cost_1, noisy_estimated_cost_2, noisy_estimated_cost_3, noisy_estimated_cost_4 = tf.split(
                noisy_estimated_intervention_probability, 4, axis=0)

            coef = self.args.coefficient
            true_cost_no_noise_1, true_per_state_error_1 = error_function(expert_a_t_raw_preintervention,
                                                                          random_actions_raw_1)
            true_cost_no_noise_2, true_per_state_error_2 = error_function(expert_a_t_raw_preintervention_2,
                                                                          random_actions_raw_2)
            true_cost_no_noise_3, true_per_state_error_3 = error_function(expert_a_t_raw_preintervention,
                                                                          random_actions_raw_3)
            true_cost_no_noise_4, true_per_state_error_4 = error_function(expert_a_t_raw_preintervention_2,
                                                                          random_actions_raw_4)

            preference_loss_1, l2_error, _ = self.get_preference_loss_true_v6(None, None, noisy_estimated_cost_1,
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

            preference_loss_2, l2_error, _ = self.get_preference_loss_true_v6(None, None, noisy_estimated_cost_1,
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

            preference_loss_3, l2_error_3, _ = self.get_preference_loss_true_v6(None, None, noisy_estimated_cost_2,
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

            preference_loss_4, l2_error_4, _ = self.get_preference_loss_true_v6(None, None, noisy_estimated_cost_3,
                                                                                np.ones_like(noisy_estimated_cost_3),
                                                                                true_cost_no_noise_3 * coef,
                                                                                np.ones_like(
                                                                                    true_cost_no_noise_3) * coef,
                                                                                single_cost_1, single_cost_2,
                                                                                np.zeros_like(is_preint_1),
                                                                                None, offset_error_1,
                                                                                offset_error_2,
                                                                                is_random=None, coef=coef,
                                                                                sanity_check=False)


            new_offset_single_error_3 = 1 - np.clip(
                np.mean(np.abs(a_t_raw_preintervention - random_actions_raw_3), axis=2), 0, 0.25) / 0.25
            new_offset_single_error_3 = new_offset_single_error_3.astype(np.float32) ** 2
            new_offset_single_error_4 = 1 - np.clip(
                np.mean(np.abs(a_t_raw_preintervention_2 - random_actions_raw_4), axis=2), 0, 0.25) / 0.25
            new_offset_single_error_4 = new_offset_single_error_4.astype(np.float32) ** 2
            # print(new_offset_single_error_3.shape, new_offset_single_error_4.shape)
            # quit()

            simple_l2_loss = (new_offset_single_error_3 * (single_cost_1 - single_cost_3) ** 2
                              + new_offset_single_error_4 * (single_cost_2 - single_cost_4) ** 2)
            # print(simple_l2_loss.shape)
            # print(offset_single_error_3[0, 0], new_offset_single_error_3[0, 0])
            # quit()
            simple_l2_loss = tf.reduce_mean(simple_l2_loss) * 0.0001
            preference_loss = preference_loss_1 + preference_loss_3 + preference_loss_2 + preference_loss_4  # + preference_loss_5
            classification_loss = tf.reduce_mean(
                preference_loss) * 0.01  # + tf.reduce_mean(l2_error) * 0.05  # + tf.reduce_mean(l2_loss_2) * 0.003
            classification_loss = classification_loss * self.args.cost_loss_modifier + simple_l2_loss

            # classification_loss = tf.reduce_mean(special_l2)

            if verbose:
                # element = 4
                # for i in range(30):
                #     print(i, estimated_cost_reshaped[element][i].numpy(), pre_true_cost_no_noise_1[element][i])
                # print(true_cost_no_noise_1[element], estimated_cost_1[element].numpy() * 30)
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
                # print("Error Mode 10:", true_total_error_mode_10, true_per_step_error_mode_10)
                # print("Error Mode 3:", true_total_error_mode_3, true_per_step_error_mode_3)
                print("Current noise", np.mean(noisy_estimated_cost_1) * self.args.segment_length,
                      np.mean(noisy_estimated_cost_2) * self.args.segment_length)
                print("True no noise", np.mean(true_cost_no_noise_1) * self.args.segment_length,
                      np.mean(true_cost_no_noise_2) * self.args.segment_length)
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
            for v in self.current_cost_model.trainable_weights:
                if 'bias' not in v.name and "cost_network" in v.name:
                    l2_loss += tf.reduce_mean(tf.nn.l2_loss(v)) * self.args.l2
                    layers += 1
            l2_loss = l2_loss / layers
            cost_loss = classification_loss + buffer_expert_loss + l2_loss

            grads = tape.gradient(cost_loss, self.current_cost_model.trainable_weights)
            self.current_cost_optimizer.apply_gradients(zip(grads, self.current_cost_model.trainable_weights))
        return cost_loss, np.mean(preference_loss)

    def train_step_cost_true(self, cost_buffer, error_function, activate_loss=False, expert_policy=None, verbose=False,
                             validation_buffer=None, group_size=4):

        with tf.GradientTape() as tape:
            s_t, a_t, _, _, intervention_status_t, _, label_t, _, _ = cost_buffer.simple_sample(
                self.args.batch_size,
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

            noisy_cost_s_t, _, _ = self.current_cost_model([double_s_t, noisy_a_t])
            buffer_expert_loss_raw = (noisy_cost_s_t - cost_label) ** 2
            buffer_expert_loss = tf.reduce_mean(
                buffer_expert_loss_raw)

            segment_batch_size = 16
            s_t_raw_preintervention, a_t_raw_preintervention, r_t_raw_preintervention, terminal_t_raw_preintervention, expert_only_s_t_raw_preintervention, \
            s_t_1_raw_preintervention, mean_label_t_raw_preintervention, raw_label_t_list_raw_preintervention, \
            expert_a_t_raw_preintervention, preintervention_raw_segment_intervention_level, segment_weight_1 = cost_buffer.sample_combined(
                [3], [segment_batch_size])

            a_t_preintervention = np.reshape(a_t_raw_preintervention, [-1, self.action_dim]).astype(np.float32)
            is_preint_1 = np.concatenate([np.zeros((segment_batch_size,))], axis=0)
            is_expert_1 = np.concatenate([np.zeros((segment_batch_size,))], axis=0)
            supervision_thresholds_1 = np.max(preintervention_raw_segment_intervention_level, axis=1)

            s_t_raw_preintervention_2, a_t_raw_preintervention_2, r_t_raw_preintervention_2, terminal_t_raw_preintervention_2, expert_only_s_t_raw_preintervention_2, \
            s_t_1_raw_preintervention_2, mean_label_t_raw_preintervention_2, raw_label_t_list_raw_preintervention_2, \
            expert_a_t_raw_preintervention_2, preintervention_raw_segment_intervention_level_2, segment_weight_2, is_random_2 = cost_buffer.sample_preint(
                segment_batch_size, supervision_thresholds_1, is_preint_1, is_expert_1, preint_include_eq=False,
                all_legal=False)

            supervision_thresholds_2 = np.max(preintervention_raw_segment_intervention_level_2, axis=1)
            a_t_preintervention_2 = np.reshape(a_t_raw_preintervention_2, [-1, self.action_dim]).astype(np.float32)

            noise = np.random.uniform(0.0, 0.0, size=a_t_raw_preintervention.shape)
            random_actions_raw_1 = np.clip(noise + a_t_raw_preintervention, -1, 1)
            random_actions_1 = np.reshape(random_actions_raw_1, [-1, self.action_dim]).astype(np.float32)
            offset_error_1, _ = error_function(a_t_raw_preintervention, random_actions_raw_1)

            noise_2 = np.random.uniform(0.0, 0.0, size=a_t_raw_preintervention_2.shape)
            random_actions_raw_2 = np.clip(noise_2 + a_t_raw_preintervention_2, -1, 1)
            random_actions_2 = np.reshape(random_actions_raw_2, [-1, self.action_dim]).astype(np.float32)
            offset_error_2, _ = error_function(a_t_raw_preintervention_2, random_actions_raw_2)

            noise_3 = np.random.normal(loc=0, scale=0.1, size=a_t_raw_preintervention.shape)
            random_actions_raw_3 = np.clip(noise_3 + a_t_raw_preintervention, -1, 1)
            random_actions_3 = np.reshape(random_actions_raw_3, [-1, self.action_dim]).astype(np.float32)
            offset_error_3, offset_single_error_3 = error_function(a_t_raw_preintervention, random_actions_raw_3)

            noise_4 = np.random.normal(loc=0, scale=0.1, size=a_t_raw_preintervention_2.shape)
            random_actions_raw_4 = np.clip(noise_4 + a_t_raw_preintervention_2, -1, 1)
            random_actions_4 = np.reshape(random_actions_raw_4, [-1, self.action_dim]).astype(np.float32)
            offset_error_4, offset_single_error_4 = error_function(a_t_raw_preintervention_2, random_actions_raw_4)
            s_t_raw = np.concatenate([s_t_raw_preintervention, s_t_raw_preintervention_2, s_t_raw_preintervention,
                                      s_t_raw_preintervention_2], axis=0)
            s_t = np.reshape(s_t_raw, [-1, self.state_dim[0]])
            a_t_noisy = np.concatenate([random_actions_1, random_actions_2, random_actions_3, random_actions_4], axis=0)

            expert_intervention_t_raw = np.concatenate(
                [expert_only_s_t_raw_preintervention, expert_only_s_t_raw_preintervention_2], axis=0)

            noisy_cost_s_t, cost_distribution, max_ent_loss = self.current_cost_model([s_t, a_t_noisy])

            noisy_cost_s_t_expert_no_grad = noisy_cost_s_t  # + tf.stop_gradient(noisy_cost_s_t) * expert_intervention_t
            noisy_cost_s_t_expert_no_grad_reshaped = tf.reshape(noisy_cost_s_t_expert_no_grad,
                                                                [s_t_raw.shape[0], self.args.segment_length])
            single_cost_1, single_cost_2, single_cost_3, single_cost_4 = tf.split(
                noisy_cost_s_t_expert_no_grad_reshaped, 4, axis=0)

            noisy_estimated_cost_reshaped = tf.reshape(noisy_cost_s_t_expert_no_grad,
                                                       [s_t_raw.shape[0], self.args.segment_length])
            noisy_estimated_cost_segment = tf.reduce_sum(noisy_estimated_cost_reshaped,
                                                         axis=1)  # prevent instability issues ...
            noisy_estimated_intervention_probability = noisy_estimated_cost_segment / self.args.segment_length
            noisy_estimated_cost_1, noisy_estimated_cost_2, noisy_estimated_cost_3, noisy_estimated_cost_4 = tf.split(
                noisy_estimated_intervention_probability, 4, axis=0)

            coef = self.args.coefficient
            true_cost_no_noise_1, true_per_state_error_1 = error_function(expert_a_t_raw_preintervention,
                                                                          random_actions_raw_1)
            true_cost_no_noise_2, true_per_state_error_2 = error_function(expert_a_t_raw_preintervention_2,
                                                                          random_actions_raw_2)
            true_cost_no_noise_3, true_per_state_error_3 = error_function(expert_a_t_raw_preintervention,
                                                                          random_actions_raw_3)

            preference_loss_1, l2_error, _ = self.get_preference_loss_true_v6(None, None, noisy_estimated_cost_1,
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

            preference_loss_2, l2_error, _ = self.get_preference_loss_true_v6(None, None, noisy_estimated_cost_1,
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

            preference_loss_3, l2_error_3, _ = self.get_preference_loss_true_v6(None, None, noisy_estimated_cost_2,
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

            preference_loss_4, l2_error_4, _ = self.get_preference_loss_true_v6(None, None, noisy_estimated_cost_3,
                                                                                np.ones_like(noisy_estimated_cost_3),
                                                                                true_cost_no_noise_3 * coef,
                                                                                np.ones_like(
                                                                                    true_cost_no_noise_3) * coef,
                                                                                single_cost_1, single_cost_2,
                                                                                np.zeros_like(is_preint_1),
                                                                                None, offset_error_1,
                                                                                offset_error_2,
                                                                                is_random=None, coef=coef,
                                                                                sanity_check=False)

            preference_loss = preference_loss_1 + preference_loss_3 + preference_loss_2 + preference_loss_4  # + preference_loss_5
            classification_loss = tf.reduce_mean(
                preference_loss) * 0.01
            classification_loss = classification_loss * self.args.cost_loss_modifier

            if verbose:
                cost_gt_0_5 = np.mean(noisy_estimated_cost_reshaped > 0.5, axis=1)

                error_single_state_1_l2 = tf.reduce_mean(tf.abs(single_cost_1 - true_per_state_error_1) ** 2)
                error_single_state_2_l2 = tf.reduce_mean(tf.abs(single_cost_2 - true_per_state_error_2) ** 2)

                error_single_state_1_l1 = tf. \
                    reduce_mean(tf.abs(single_cost_1 - true_per_state_error_1))
                error_single_state_2_l1 = tf.reduce_mean(tf.abs(single_cost_2 - true_per_state_error_2))

                print("\n")
                print(np.mean(preference_loss), np.mean(classification_loss))
                print("State Err L1, L2:", np.mean(error_single_state_1_l1), np.mean(error_single_state_2_l1),
                      np.mean(error_single_state_1_l2), np.mean(error_single_state_2_l2), )
                # print("Error Mode 10:", true_total_error_mode_10, true_per_step_error_mode_10)
                # print("Error Mode 3:", true_total_error_mode_3, true_per_step_error_mode_3)
                print("Current noise", np.mean(noisy_estimated_cost_1) * self.args.segment_length,
                      np.mean(noisy_estimated_cost_2) * self.args.segment_length)
                print("True no noise", np.mean(true_cost_no_noise_1) * self.args.segment_length,
                      np.mean(true_cost_no_noise_2) * self.args.segment_length)
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
            for v in self.current_cost_model.trainable_weights:
                if 'bias' not in v.name and "cost_network" in v.name:
                    l2_loss += tf.reduce_mean(tf.nn.l2_loss(v)) * self.args.l2
                    layers += 1
            l2_loss = l2_loss / layers
            cost_loss = classification_loss + buffer_expert_loss + l2_loss

            grads = tape.gradient(cost_loss, self.current_cost_model.trainable_weights)
            self.current_cost_optimizer.apply_gradients(zip(grads, self.current_cost_model.trainable_weights))
        return cost_loss, np.mean(preference_loss)

    def train_step_cost_true_v2(self, cost_buffer, error_function, activate_loss=False, expert_policy=None,
                                verbose=False, validation_buffer=None, preference_error=1, group_size=4):

        with tf.GradientTape() as tape:
            s_t, a_t, _, _, intervention_status_t, _, label_t, _, _ = cost_buffer.simple_sample(
                self.args.batch_size,
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

            noisy_cost_s_t, _, _ = self.current_cost_model([double_s_t, noisy_a_t])
            buffer_expert_loss_raw = (noisy_cost_s_t - cost_label) ** 2
            buffer_expert_loss = tf.reduce_mean(
                buffer_expert_loss_raw)
            segment_batch_size = 16
            s_t_raw_preintervention, a_t_raw_preintervention, r_t_raw_preintervention, terminal_t_raw_preintervention, expert_only_s_t_raw_preintervention, \
            s_t_1_raw_preintervention, mean_label_t_raw_preintervention, raw_label_t_list_raw_preintervention, \
            expert_a_t_raw_preintervention, preintervention_raw_segment_intervention_level, segment_weight_1 = cost_buffer.sample_combined(
                [3], [segment_batch_size])

            a_t_preintervention = np.reshape(a_t_raw_preintervention, [-1, self.action_dim]).astype(np.float32)
            is_preint_1 = np.concatenate([np.zeros((segment_batch_size,))], axis=0)
            is_expert_1 = np.concatenate([np.zeros((segment_batch_size,))], axis=0)
            supervision_thresholds_1 = np.max(preintervention_raw_segment_intervention_level, axis=1)

            s_t_raw_preintervention_2, a_t_raw_preintervention_2, r_t_raw_preintervention_2, terminal_t_raw_preintervention_2, expert_only_s_t_raw_preintervention_2, \
            s_t_1_raw_preintervention_2, mean_label_t_raw_preintervention_2, raw_label_t_list_raw_preintervention_2, \
            expert_a_t_raw_preintervention_2, preintervention_raw_segment_intervention_level_2, segment_weight_2, is_random_2 = cost_buffer.sample_preint(
                segment_batch_size, supervision_thresholds_1, is_preint_1, is_expert_1, preint_include_eq=False,
                all_legal=False)

            supervision_thresholds_2 = np.max(preintervention_raw_segment_intervention_level_2, axis=1)
            a_t_preintervention_2 = np.reshape(a_t_raw_preintervention_2, [-1, self.action_dim]).astype(np.float32)

            noise = np.random.uniform(0.0, 0.0, size=a_t_raw_preintervention.shape)
            random_actions_raw_1 = np.clip(noise + a_t_raw_preintervention, -1, 1)
            random_actions_1 = np.reshape(random_actions_raw_1, [-1, self.action_dim]).astype(np.float32)
            offset_error_1, _ = error_function(a_t_raw_preintervention, random_actions_raw_1)

            noise_2 = np.random.uniform(0.0, 0.0, size=a_t_raw_preintervention_2.shape)
            random_actions_raw_2 = np.clip(noise_2 + a_t_raw_preintervention_2, -1, 1)
            random_actions_2 = np.reshape(random_actions_raw_2, [-1, self.action_dim]).astype(np.float32)
            offset_error_2, _ = error_function(a_t_raw_preintervention_2, random_actions_raw_2)

            noise_3 = np.random.normal(loc=0, scale=0.1, size=a_t_raw_preintervention.shape)
            random_actions_raw_3 = np.clip(noise_3 + a_t_raw_preintervention, -1, 1)
            random_actions_3 = np.reshape(random_actions_raw_3, [-1, self.action_dim]).astype(np.float32)
            offset_error_3, offset_single_error_3 = error_function(a_t_raw_preintervention, random_actions_raw_3)

            noise_4 = np.random.normal(loc=0, scale=0.1, size=a_t_raw_preintervention_2.shape)
            random_actions_raw_4 = np.clip(noise_4 + a_t_raw_preintervention_2, -1, 1)
            random_actions_4 = np.reshape(random_actions_raw_4, [-1, self.action_dim]).astype(np.float32)
            offset_error_4, offset_single_error_4 = error_function(a_t_raw_preintervention_2, random_actions_raw_4)

            s_t_raw = np.concatenate([s_t_raw_preintervention, s_t_raw_preintervention_2, s_t_raw_preintervention,
                                      s_t_raw_preintervention_2], axis=0)
            s_t = np.reshape(s_t_raw, [-1, self.state_dim[0]])
            a_t_noisy = np.concatenate([random_actions_1, random_actions_2, random_actions_3, random_actions_4], axis=0)

            expert_intervention_t_raw = np.concatenate(
                [expert_only_s_t_raw_preintervention, expert_only_s_t_raw_preintervention_2], axis=0)
            # expert_intervention_t = np.reshape(expert_intervention_t_raw, [-1, 1])

            noisy_cost_s_t, cost_distribution, max_ent_loss = self.current_cost_model([s_t, a_t_noisy])

            noisy_cost_s_t_expert_no_grad = noisy_cost_s_t  # + tf.stop_gradient(noisy_cost_s_t) * expert_intervention_t
            noisy_cost_s_t_expert_no_grad_reshaped = tf.reshape(noisy_cost_s_t_expert_no_grad,
                                                                [s_t_raw.shape[0], self.args.segment_length])
            single_cost_1, single_cost_2, single_cost_3, single_cost_4 = tf.split(
                noisy_cost_s_t_expert_no_grad_reshaped, 4, axis=0)

            noisy_estimated_cost_reshaped = tf.reshape(noisy_cost_s_t_expert_no_grad,
                                                       [s_t_raw.shape[0], self.args.segment_length])
            noisy_estimated_cost_segment = tf.reduce_sum(noisy_estimated_cost_reshaped,
                                                         axis=1)  # prevent instability issues ...
            noisy_estimated_intervention_probability = noisy_estimated_cost_segment / self.args.segment_length
            noisy_estimated_cost_1, noisy_estimated_cost_2, noisy_estimated_cost_3, noisy_estimated_cost_4 = tf.split(
                noisy_estimated_intervention_probability, 4, axis=0)

            coef = self.args.coefficient
            true_cost_no_noise_1, true_per_state_error_1 = error_function(expert_a_t_raw_preintervention,
                                                                          random_actions_raw_1)
            true_cost_no_noise_2, true_per_state_error_2 = error_function(expert_a_t_raw_preintervention_2,
                                                                          random_actions_raw_2)
            true_cost_no_noise_3, true_per_state_error_3 = error_function(expert_a_t_raw_preintervention,
                                                                          random_actions_raw_3)

            preference_loss_1, l2_error, _ = self.get_preference_loss_true_v6(None, None, noisy_estimated_cost_1,
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

            preference_loss_2, l2_error, _ = self.get_preference_loss_true_v6(None, None, noisy_estimated_cost_1,
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

            preference_loss_3, l2_error_3, _ = self.get_preference_loss_true_v6(None, None, noisy_estimated_cost_2,
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

            preference_loss_4, l2_error_4, _ = self.get_preference_loss_true_v6(None, None, noisy_estimated_cost_3,
                                                                                np.ones_like(noisy_estimated_cost_3),
                                                                                true_cost_no_noise_3 * coef,
                                                                                np.ones_like(
                                                                                    true_cost_no_noise_3) * coef,
                                                                                single_cost_1, single_cost_2,
                                                                                np.zeros_like(is_preint_1),
                                                                                None, offset_error_1,
                                                                                offset_error_2,
                                                                                is_random=None, coef=coef,
                                                                                sanity_check=False)

            new_offset_single_error_3 = 1 - np.clip(
                np.mean(np.abs(a_t_raw_preintervention - random_actions_raw_3), axis=2), 0, 0.25) / 0.25
            new_offset_single_error_3 = new_offset_single_error_3.astype(np.float32) ** 2
            new_offset_single_error_4 = 1 - np.clip(
                np.mean(np.abs(a_t_raw_preintervention_2 - random_actions_raw_4), axis=2), 0, 0.25) / 0.25
            new_offset_single_error_4 = new_offset_single_error_4.astype(np.float32) ** 2

            simple_l2_loss = (new_offset_single_error_3 * (single_cost_1 - single_cost_3) ** 2
                              + new_offset_single_error_4 * (single_cost_2 - single_cost_4) ** 2)

            simple_l2_loss = tf.reduce_mean(simple_l2_loss) * 0.0003
            preference_loss = preference_loss_1 + preference_loss_3 + preference_loss_2 + preference_loss_4  # + preference_loss_5
            classification_loss = tf.reduce_mean(
                preference_loss) * 0.01
            classification_loss = classification_loss * self.args.cost_loss_modifier + simple_l2_loss

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
                print("Current noise", np.mean(noisy_estimated_cost_1) * self.args.segment_length,
                      np.mean(noisy_estimated_cost_2) * self.args.segment_length)
                print("True no noise", np.mean(true_cost_no_noise_1) * self.args.segment_length,
                      np.mean(true_cost_no_noise_2) * self.args.segment_length)
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
            for v in self.current_cost_model.trainable_weights:
                if 'bias' not in v.name and "cost_network" in v.name:
                    l2_loss += tf.reduce_mean(tf.nn.l2_loss(v)) * self.args.l2
                    layers += 1
            l2_loss = l2_loss / layers
            cost_loss = classification_loss + buffer_expert_loss + l2_loss

            grads = tape.gradient(cost_loss, self.current_cost_model.trainable_weights)
            self.current_cost_optimizer.apply_gradients(zip(grads, self.current_cost_model.trainable_weights))
        return cost_loss, np.mean(preference_loss)

    def update_target_network(self):
        current_weights = self.current_cost_model.get_weights()
        target_weights = self.target_cost_model.get_weights()

        new_target_weights = []
        for i in range(len(current_weights)):
            new_weight = current_weights[i] * self.args.target_update_rate + target_weights[i] * (
                        1 - self.args.target_update_rate)
            new_target_weights.append(new_weight)
        self.target_cost_model.set_weights(new_target_weights)

    def train_step_cost_true_v3(self, cost_buffer, error_function, activate_loss=False, expert_policy=None,
                                verbose=False, validation_buffer=None, group_size=4):
        with tf.GradientTape() as tape:
            s_t, a_t, _, _, intervention_status_t, _, label_t, _, _ = cost_buffer.simple_sample(self.args.batch_size,
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

            noisy_cost_s_t, _, _ = self.current_cost_model([double_s_t, noisy_a_t])
            buffer_expert_loss_raw = (noisy_cost_s_t - cost_label) ** 2
            buffer_expert_loss = tf.reduce_mean(buffer_expert_loss_raw)
            segment_batch_size = 16
            s_t_raw_preintervention, a_t_raw_preintervention, r_t_raw_preintervention, terminal_t_raw_preintervention, expert_only_s_t_raw_preintervention, \
            s_t_1_raw_preintervention, mean_label_t_raw_preintervention, raw_label_t_list_raw_preintervention, \
            expert_a_t_raw_preintervention, preintervention_raw_segment_intervention_level, segment_weight_1 = cost_buffer.sample_combined(
                [3], [segment_batch_size])

            a_t_preintervention = np.reshape(a_t_raw_preintervention, [-1, self.action_dim]).astype(np.float32)
            is_preint_1 = np.concatenate([np.zeros((segment_batch_size,))], axis=0)

            is_expert_1 = np.concatenate([np.zeros((segment_batch_size,))], axis=0)
            supervision_thresholds_1 = np.max(preintervention_raw_segment_intervention_level, axis=1)

            s_t_raw_preintervention_2, a_t_raw_preintervention_2, r_t_raw_preintervention_2, terminal_t_raw_preintervention_2, expert_only_s_t_raw_preintervention_2, \
            s_t_1_raw_preintervention_2, mean_label_t_raw_preintervention_2, raw_label_t_list_raw_preintervention_2, \
            expert_a_t_raw_preintervention_2, preintervention_raw_segment_intervention_level_2, segment_weight_2, is_random_2 = cost_buffer.sample_preint(
                segment_batch_size, supervision_thresholds_1, is_preint_1, is_expert_1, preint_include_eq=False,
                all_legal=True)

            supervision_thresholds_2 = np.max(preintervention_raw_segment_intervention_level_2, axis=1)
            a_t_preintervention_2 = np.reshape(a_t_raw_preintervention_2, [-1, self.action_dim]).astype(np.float32)

            noise = np.random.uniform(0.0, 0.0, size=a_t_raw_preintervention.shape)
            random_actions_raw_1 = np.clip(noise + a_t_raw_preintervention, -1, 1)
            random_actions_1 = np.reshape(random_actions_raw_1, [-1, self.action_dim]).astype(np.float32)
            offset_error_1, _ = error_function(a_t_raw_preintervention, random_actions_raw_1)

            noise_2 = np.random.uniform(0.0, 0.0, size=a_t_raw_preintervention_2.shape)
            random_actions_raw_2 = np.clip(noise_2 + a_t_raw_preintervention_2, -1, 1)
            random_actions_2 = np.reshape(random_actions_raw_2, [-1, self.action_dim]).astype(np.float32)
            offset_error_2, _ = error_function(a_t_raw_preintervention_2, random_actions_raw_2)

            s_t_raw = np.concatenate([s_t_raw_preintervention, s_t_raw_preintervention_2], axis=0)
            s_t = np.reshape(s_t_raw, [-1, self.state_dim[0]])
            a_t_noisy = np.concatenate([random_actions_1, random_actions_2], axis=0)

            expert_intervention_t_raw = np.concatenate(
                [expert_only_s_t_raw_preintervention, expert_only_s_t_raw_preintervention_2], axis=0)
            expert_intervention_t = np.reshape(expert_intervention_t_raw, [-1, 1])

            noisy_cost_s_t, cost_distribution, max_ent_loss = self.current_cost_model([s_t, a_t_noisy])

            noisy_cost_s_t_expert_no_grad = noisy_cost_s_t  # + tf.stop_gradient(noisy_cost_s_t) * expert_intervention_t
            noisy_cost_s_t_expert_no_grad_reshaped = tf.reshape(noisy_cost_s_t_expert_no_grad,
                                                                [s_t_raw.shape[0], self.args.segment_length])
            single_cost_1, single_cost_2 = tf.split(
                noisy_cost_s_t_expert_no_grad_reshaped, 2, axis=0)

            noisy_estimated_cost_reshaped = tf.reshape(noisy_cost_s_t_expert_no_grad,
                                                       [s_t_raw.shape[0], self.args.segment_length])
            noisy_estimated_cost_segment = tf.reduce_sum(noisy_estimated_cost_reshaped,
                                                         axis=1)  # prevent instability issues ...
            noisy_estimated_intervention_probability = noisy_estimated_cost_segment / self.args.segment_length
            noisy_estimated_cost_1, noisy_estimated_cost_2 = tf.split(
                noisy_estimated_intervention_probability, 2, axis=0)

            coef = self.args.coefficient
            true_cost_no_noise_1, true_per_state_error_1 = error_function(expert_a_t_raw_preintervention,
                                                                          random_actions_raw_1)
            true_cost_no_noise_2, true_per_state_error_2 = error_function(expert_a_t_raw_preintervention_2,
                                                                          random_actions_raw_2)

            preference_loss_1, _, _ = self.get_preference_loss_true_v6(None, None, noisy_estimated_cost_1,
                                                                       noisy_estimated_cost_2,
                                                                       true_cost_no_noise_1 * coef,
                                                                       true_cost_no_noise_2 * coef,
                                                                       single_cost_1, single_cost_2,
                                                                       np.zeros_like(is_preint_1),
                                                                       None, offset_error_1,
                                                                       offset_error_2,
                                                                       is_random=None, coef=coef)

            preference_loss = preference_loss_1
            special_l2 = noisy_estimated_cost_2 ** 2
            classification_loss = (tf.reduce_mean(preference_loss) * 0.01 + tf.reduce_mean(
                special_l2) * 0.005)
            classification_loss = classification_loss

            if verbose:
                cost_gt_0_5 = np.mean(noisy_estimated_cost_reshaped > 0.5, axis=1)

                error_single_state_1_l2 = tf.reduce_mean(tf.abs(single_cost_1 - true_per_state_error_1) ** 2)
                error_single_state_2_l2 = tf.reduce_mean(tf.abs(single_cost_2 - true_per_state_error_2) ** 2)

                error_single_state_1_l1 = tf. \
                    reduce_mean(tf.abs(single_cost_1 - true_per_state_error_1))
                error_single_state_2_l1 = tf.reduce_mean(tf.abs(single_cost_2 - true_per_state_error_2))

                print(error_single_state_1_l1.shape)

                print("\n")
                print(np.mean(preference_loss), np.mean(special_l2), np.mean(classification_loss))
                print("State Err L1, L2:", np.mean(error_single_state_1_l1), np.mean(error_single_state_2_l1),
                      np.mean(error_single_state_1_l2), np.mean(error_single_state_2_l2), )
                print("Current noise", np.mean(noisy_estimated_cost_1) * self.args.segment_length,
                      np.mean(noisy_estimated_cost_2) * self.args.segment_length)
                print("True no noise", np.mean(true_cost_no_noise_1) * self.args.segment_length,
                      np.mean(true_cost_no_noise_2) * self.args.segment_length)
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
            for v in self.current_cost_model.trainable_weights:
                if 'bias' not in v.name and "cost_network" in v.name:
                    l2_loss += tf.reduce_mean(tf.nn.l2_loss(v)) * self.args.l2
                    layers += 1
            l2_loss = l2_loss / layers
            cost_loss = classification_loss + buffer_expert_loss + l2_loss

            grads = tape.gradient(cost_loss, self.current_cost_model.trainable_weights)
            self.current_cost_optimizer.apply_gradients(zip(grads, self.current_cost_model.trainable_weights))
        return cost_loss, np.mean(preference_loss)

    def train_step_cost_true_v4(self, cost_buffer, error_function, activate_loss=False, expert_policy=None,
                                verbose=False, validation_buffer=None, group_size=4):
        with tf.GradientTape() as tape:
            s_t, a_t, _, _, intervention_status_t, _, label_t, _, _ = cost_buffer.simple_sample(
                self.args.batch_size,
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

            noisy_cost_s_t, _, _ = self.current_cost_model([double_s_t, noisy_a_t])
            buffer_expert_loss_raw = (noisy_cost_s_t - cost_label) ** 2
            # noisy_cost_l2 = (noisy_cost_s_t - 0.5) ** 2 * 0.003
            buffer_expert_loss = tf.reduce_mean(
                buffer_expert_loss_raw)  # + tf.reduce_mean(noisy_cost_l2)  # + tf.reduce_mean(segment_preference_loss_raw) + tf.reduce_mean(intervention_loss_raw)

            segment_batch_size = 16
            s_t_raw_preintervention, a_t_raw_preintervention, r_t_raw_preintervention, terminal_t_raw_preintervention, expert_only_s_t_raw_preintervention, \
            s_t_1_raw_preintervention, mean_label_t_raw_preintervention, raw_label_t_list_raw_preintervention, \
            expert_a_t_raw_preintervention, preintervention_raw_segment_intervention_level, segment_weight_1 = cost_buffer.sample_combined(
                [3], [segment_batch_size])

            a_t_preintervention = np.reshape(a_t_raw_preintervention, [-1, self.action_dim]).astype(np.float32)
            is_preint_1 = np.concatenate([np.zeros((segment_batch_size,))], axis=0)
            is_expert_1 = np.concatenate([np.zeros((segment_batch_size,))], axis=0)
            supervision_thresholds_1 = np.max(preintervention_raw_segment_intervention_level, axis=1)

            s_t_raw_preintervention_2, a_t_raw_preintervention_2, r_t_raw_preintervention_2, terminal_t_raw_preintervention_2, expert_only_s_t_raw_preintervention_2, \
            s_t_1_raw_preintervention_2, mean_label_t_raw_preintervention_2, raw_label_t_list_raw_preintervention_2, \
            expert_a_t_raw_preintervention_2, preintervention_raw_segment_intervention_level_2, segment_weight_2, is_random_2 = cost_buffer.sample_preint(
                segment_batch_size, supervision_thresholds_1, is_preint_1, is_expert_1, preint_include_eq=False,
                all_legal=False)

            supervision_thresholds_2 = np.max(preintervention_raw_segment_intervention_level_2, axis=1)
            a_t_preintervention_2 = np.reshape(a_t_raw_preintervention_2, [-1, self.action_dim]).astype(np.float32)

            noise = np.random.uniform(0.0, 0.0, size=a_t_raw_preintervention.shape)
            random_actions_raw_1 = np.clip(noise + a_t_raw_preintervention, -1, 1)
            random_actions_1 = np.reshape(random_actions_raw_1, [-1, self.action_dim]).astype(np.float32)
            offset_error_1, _ = error_function(a_t_raw_preintervention, random_actions_raw_1)

            noise_2 = np.random.uniform(0.0, 0.0, size=a_t_raw_preintervention_2.shape)
            random_actions_raw_2 = np.clip(noise_2 + a_t_raw_preintervention_2, -1, 1)
            random_actions_2 = np.reshape(random_actions_raw_2, [-1, self.action_dim]).astype(np.float32)
            offset_error_2, _ = error_function(a_t_raw_preintervention_2, random_actions_raw_2)

            noise_3 = np.random.normal(loc=0, scale=0.2, size=a_t_raw_preintervention.shape)
            random_actions_raw_3 = np.clip(noise_3 + a_t_raw_preintervention, -1, 1)
            random_actions_3 = np.reshape(random_actions_raw_3, [-1, self.action_dim]).astype(np.float32)
            offset_error_3, offset_single_error_3 = error_function(a_t_raw_preintervention, random_actions_raw_3)

            noise_4 = np.random.normal(loc=0, scale=0.2, size=a_t_raw_preintervention_2.shape)
            random_actions_raw_4 = np.clip(noise_4 + a_t_raw_preintervention_2, -1, 1)
            random_actions_4 = np.reshape(random_actions_raw_4, [-1, self.action_dim]).astype(np.float32)
            offset_error_4, offset_single_error_4 = error_function(a_t_raw_preintervention_2, random_actions_raw_4)

            s_t_raw = np.concatenate([s_t_raw_preintervention, s_t_raw_preintervention_2, s_t_raw_preintervention,
                                      s_t_raw_preintervention_2], axis=0)
            s_t = np.reshape(s_t_raw, [-1, self.state_dim[0]])
            a_t_noisy = np.concatenate([random_actions_1, random_actions_2, random_actions_3, random_actions_4], axis=0)

            expert_intervention_t_raw = np.concatenate(
                [expert_only_s_t_raw_preintervention, expert_only_s_t_raw_preintervention_2], axis=0)

            noisy_cost_s_t, cost_distribution, max_ent_loss = self.current_cost_model([s_t, a_t_noisy])

            noisy_cost_s_t_expert_no_grad = noisy_cost_s_t  # + tf.stop_gradient(noisy_cost_s_t) * expert_intervention_t
            noisy_cost_s_t_expert_no_grad_reshaped = tf.reshape(noisy_cost_s_t_expert_no_grad,
                                                                [s_t_raw.shape[0], self.args.segment_length])
            single_cost_1, single_cost_2, single_cost_3, single_cost_4 = tf.split(
                noisy_cost_s_t_expert_no_grad_reshaped, 4, axis=0)

            noisy_estimated_cost_reshaped = tf.reshape(noisy_cost_s_t_expert_no_grad,
                                                       [s_t_raw.shape[0], self.args.segment_length])
            noisy_estimated_cost_segment = tf.reduce_sum(noisy_estimated_cost_reshaped,
                                                         axis=1)  # prevent instability issues ...
            noisy_estimated_intervention_probability = noisy_estimated_cost_segment / self.args.segment_length
            noisy_estimated_cost_1, noisy_estimated_cost_2, noisy_estimated_cost_3, noisy_estimated_cost_4 = tf.split(
                noisy_estimated_intervention_probability, 4, axis=0)

            coef = self.args.coefficient
            true_cost_no_noise_1, true_per_state_error_1 = error_function(expert_a_t_raw_preintervention,
                                                                          random_actions_raw_1)
            true_cost_no_noise_2, true_per_state_error_2 = error_function(expert_a_t_raw_preintervention_2,
                                                                          random_actions_raw_2)
            true_cost_no_noise_3, true_per_state_error_3 = error_function(expert_a_t_raw_preintervention,
                                                                          random_actions_raw_3)

            preference_loss_1, l2_error, _ = self.get_preference_loss_true_v6(None, None, noisy_estimated_cost_1,
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

            preference_loss_2, l2_error, _ = self.get_preference_loss_true_v6(None, None, noisy_estimated_cost_1,
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

            preference_loss_3, l2_error_3, _ = self.get_preference_loss_true_v6(None, None, noisy_estimated_cost_2,
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

            preference_loss_4, l2_error_4, _ = self.get_preference_loss_true_v6(None, None, noisy_estimated_cost_3,
                                                                                np.ones_like(noisy_estimated_cost_3),
                                                                                true_cost_no_noise_3 * coef,
                                                                                np.ones_like(
                                                                                    true_cost_no_noise_3) * coef,
                                                                                single_cost_1, single_cost_2,
                                                                                np.zeros_like(is_preint_1),
                                                                                None, offset_error_1,
                                                                                offset_error_2,
                                                                                is_random=None, coef=coef,
                                                                                sanity_check=False)


            preference_loss = preference_loss_1 + preference_loss_3 + preference_loss_2 + preference_loss_4  # + preference_loss_5
            classification_loss = tf.reduce_mean(
                preference_loss) * 0.01
            classification_loss = classification_loss * self.args.cost_loss_modifier
            if verbose:
                cost_gt_0_5 = np.mean(noisy_estimated_cost_reshaped > 0.5, axis=1)

                error_single_state_1_l2 = tf.reduce_mean(tf.abs(single_cost_1 - true_per_state_error_1) ** 2)
                error_single_state_2_l2 = tf.reduce_mean(tf.abs(single_cost_2 - true_per_state_error_2) ** 2)

                error_single_state_1_l1 = tf. \
                    reduce_mean(tf.abs(single_cost_1 - true_per_state_error_1))
                error_single_state_2_l1 = tf.reduce_mean(tf.abs(single_cost_2 - true_per_state_error_2))

                print("\n")
                print(np.mean(preference_loss), np.mean(classification_loss))
                print("State Err L1, L2:", np.mean(error_single_state_1_l1), np.mean(error_single_state_2_l1),
                      np.mean(error_single_state_1_l2), np.mean(error_single_state_2_l2), )
                print("Current noise", np.mean(noisy_estimated_cost_1) * self.args.segment_length,
                      np.mean(noisy_estimated_cost_2) * self.args.segment_length)
                print("True no noise", np.mean(true_cost_no_noise_1) * self.args.segment_length,
                      np.mean(true_cost_no_noise_2) * self.args.segment_length)
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
            for v in self.current_cost_model.trainable_weights:
                if 'bias' not in v.name and "cost_network" in v.name:
                    l2_loss += tf.reduce_mean(tf.nn.l2_loss(v)) * self.args.l2
                    layers += 1
            l2_loss = l2_loss / layers
            cost_loss = classification_loss + buffer_expert_loss + l2_loss

            grads = tape.gradient(cost_loss, self.current_cost_model.trainable_weights)
            self.current_cost_optimizer.apply_gradients(zip(grads, self.current_cost_model.trainable_weights))
        return cost_loss, np.mean(preference_loss)

    def train_step_cost_true_v7(self, cost_buffer, error_function, activate_loss=False, expert_policy=None,
                                verbose=False, validation_buffer=None, group_size=4):
        with tf.GradientTape() as tape:
            s_t, a_t, _, _, intervention_status_t, _, label_t, _, _ = cost_buffer.simple_sample(
                self.args.batch_size,
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

            noisy_cost_s_t, _, _ = self.current_cost_model([double_s_t, noisy_a_t])
            buffer_expert_loss_raw = (noisy_cost_s_t - cost_label) ** 2
            buffer_expert_loss = tf.reduce_mean(
                buffer_expert_loss_raw)
            segment_batch_size = 16
            s_t_raw_preintervention, a_t_raw_preintervention, r_t_raw_preintervention, terminal_t_raw_preintervention, expert_only_s_t_raw_preintervention, \
            s_t_1_raw_preintervention, mean_label_t_raw_preintervention, raw_label_t_list_raw_preintervention, \
            expert_a_t_raw_preintervention, preintervention_raw_segment_intervention_level, segment_weight_1 = cost_buffer.sample_combined(
                [3], [segment_batch_size])

            a_t_preintervention = np.reshape(a_t_raw_preintervention, [-1, self.action_dim]).astype(np.float32)
            is_preint_1 = np.concatenate([np.zeros((segment_batch_size,))], axis=0)
            is_expert_1 = np.concatenate([np.zeros((segment_batch_size,))], axis=0)
            supervision_thresholds_1 = np.max(preintervention_raw_segment_intervention_level, axis=1)

            s_t_raw_preintervention_2, a_t_raw_preintervention_2, r_t_raw_preintervention_2, terminal_t_raw_preintervention_2, expert_only_s_t_raw_preintervention_2, \
            s_t_1_raw_preintervention_2, mean_label_t_raw_preintervention_2, raw_label_t_list_raw_preintervention_2, \
            expert_a_t_raw_preintervention_2, preintervention_raw_segment_intervention_level_2, segment_weight_2, is_random_2 = cost_buffer.sample_preint(
                segment_batch_size, supervision_thresholds_1, is_preint_1, is_expert_1, preint_include_eq=False,
                all_legal=False)

            supervision_thresholds_2 = np.max(preintervention_raw_segment_intervention_level_2, axis=1)
            a_t_preintervention_2 = np.reshape(a_t_raw_preintervention_2, [-1, self.action_dim]).astype(np.float32)

            noise = np.random.uniform(0.0, 0.0, size=a_t_raw_preintervention.shape)
            random_actions_raw_1 = np.clip(noise + a_t_raw_preintervention, -1, 1)
            random_actions_1 = np.reshape(random_actions_raw_1, [-1, self.action_dim]).astype(np.float32)
            offset_error_1, _ = error_function(a_t_raw_preintervention, random_actions_raw_1)

            noise_2 = np.random.uniform(0.0, 0.0, size=a_t_raw_preintervention_2.shape)
            random_actions_raw_2 = np.clip(noise_2 + a_t_raw_preintervention_2, -1, 1)
            random_actions_2 = np.reshape(random_actions_raw_2, [-1, self.action_dim]).astype(np.float32)
            offset_error_2, _ = error_function(a_t_raw_preintervention_2, random_actions_raw_2)

            noise_3 = np.random.normal(loc=0, scale=0.1, size=a_t_raw_preintervention.shape)
            random_actions_raw_3 = np.clip(noise_3 + a_t_raw_preintervention, -1, 1)
            random_actions_3 = np.reshape(random_actions_raw_3, [-1, self.action_dim]).astype(np.float32)
            offset_error_3, offset_single_error_3 = error_function(a_t_raw_preintervention, random_actions_raw_3)

            noise_4 = np.random.normal(loc=0, scale=0.1, size=a_t_raw_preintervention_2.shape)
            random_actions_raw_4 = np.clip(noise_4 + a_t_raw_preintervention_2, -1, 1)
            random_actions_4 = np.reshape(random_actions_raw_4, [-1, self.action_dim]).astype(np.float32)
            offset_error_4, offset_single_error_4 = error_function(a_t_raw_preintervention_2, random_actions_raw_4)

            s_t_raw = np.concatenate([s_t_raw_preintervention, s_t_raw_preintervention_2, s_t_raw_preintervention,
                                      s_t_raw_preintervention_2], axis=0)
            s_t = np.reshape(s_t_raw, [-1, self.state_dim[0]])
            a_t_noisy = np.concatenate([random_actions_1, random_actions_2, random_actions_3, random_actions_4], axis=0)

            expert_intervention_t_raw = np.concatenate(
                [expert_only_s_t_raw_preintervention, expert_only_s_t_raw_preintervention_2], axis=0)
            noisy_cost_s_t, cost_distribution, max_ent_loss = self.current_cost_model([s_t, a_t_noisy])

            noisy_cost_s_t_expert_no_grad = noisy_cost_s_t  # + tf.stop_gradient(noisy_cost_s_t) * expert_intervention_t
            noisy_cost_s_t_expert_no_grad_reshaped = tf.reshape(noisy_cost_s_t_expert_no_grad,
                                                                [s_t_raw.shape[0], self.args.segment_length])
            single_cost_1, single_cost_2, single_cost_3, single_cost_4 = tf.split(
                noisy_cost_s_t_expert_no_grad_reshaped, 4, axis=0)

            noisy_estimated_cost_reshaped = tf.reshape(noisy_cost_s_t_expert_no_grad,
                                                       [s_t_raw.shape[0], self.args.segment_length])
            noisy_estimated_cost_segment = tf.reduce_sum(noisy_estimated_cost_reshaped,
                                                         axis=1)  # prevent instability issues ...
            noisy_estimated_intervention_probability = noisy_estimated_cost_segment / self.args.segment_length
            noisy_estimated_cost_1, noisy_estimated_cost_2, noisy_estimated_cost_3, noisy_estimated_cost_4 = tf.split(
                noisy_estimated_intervention_probability, 4, axis=0)

            coef = self.args.coefficient
            true_cost_no_noise_1, true_per_state_error_1 = error_function(expert_a_t_raw_preintervention,
                                                                          random_actions_raw_1)
            true_cost_no_noise_2, true_per_state_error_2 = error_function(expert_a_t_raw_preintervention_2,
                                                                          random_actions_raw_2)
            preference_loss_1, l2_error, _ = self.get_preference_loss_true_v6(None, None, noisy_estimated_cost_1,
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

            preference_loss_2, l2_error, _ = self.get_preference_loss_true_v6(None, None, noisy_estimated_cost_1,
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

            preference_loss_3, l2_error_3, _ = self.get_preference_loss_true_v6(None, None, noisy_estimated_cost_2,
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
                np.mean(np.abs(a_t_raw_preintervention - random_actions_raw_3), axis=2), 0, 0.25) / 0.25
            new_offset_single_error_3 = new_offset_single_error_3.astype(np.float32) ** 2
            new_offset_single_error_4 = 1 - np.clip(
                np.mean(np.abs(a_t_raw_preintervention_2 - random_actions_raw_4), axis=2), 0, 0.25) / 0.25
            new_offset_single_error_4 = new_offset_single_error_4.astype(np.float32) ** 2
            simple_l2_loss = (new_offset_single_error_3 * (single_cost_1 - single_cost_3) ** 2
                              + new_offset_single_error_4 * (single_cost_2 - single_cost_4) ** 2)

            simple_l2_loss = tf.reduce_mean(simple_l2_loss) * 0.0001
            preference_loss = preference_loss_1 + preference_loss_3 + preference_loss_2
            classification_loss = tf.reduce_mean(
                preference_loss) * 0.01
            classification_loss = classification_loss * self.args.cost_loss_modifier + simple_l2_loss
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
                print("Current noise", np.mean(noisy_estimated_cost_1) * self.args.segment_length,
                      np.mean(noisy_estimated_cost_2) * self.args.segment_length)
                print("True no noise", np.mean(true_cost_no_noise_1) * self.args.segment_length,
                      np.mean(true_cost_no_noise_2) * self.args.segment_length)
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
            for v in self.current_cost_model.trainable_weights:
                if 'bias' not in v.name and "cost_network" in v.name:
                    l2_loss += tf.reduce_mean(tf.nn.l2_loss(v)) * self.args.l2
                    layers += 1
            l2_loss = l2_loss / layers
            cost_loss = classification_loss + buffer_expert_loss + l2_loss

            grads = tape.gradient(cost_loss, self.current_cost_model.trainable_weights)
            self.current_cost_optimizer.apply_gradients(zip(grads, self.current_cost_model.trainable_weights))
        return cost_loss, np.mean(preference_loss)

    def train_step_cost_true_v8(self, cost_buffer, error_function, activate_loss=False, expert_policy=None,
                                verbose=False, validation_buffer=None, group_size=4):
        with tf.GradientTape() as tape:
            s_t, a_t, _, _, intervention_status_t, _, label_t, _, _ = cost_buffer.simple_sample(
                self.args.batch_size,
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

            noisy_cost_s_t, _, _ = self.current_cost_model([double_s_t, noisy_a_t])
            buffer_expert_loss_raw = (noisy_cost_s_t - cost_label) ** 2
            buffer_expert_loss = tf.reduce_mean(
                buffer_expert_loss_raw)
            segment_batch_size = 16
            s_t_raw_preintervention, a_t_raw_preintervention, r_t_raw_preintervention, terminal_t_raw_preintervention, expert_only_s_t_raw_preintervention, \
            s_t_1_raw_preintervention, mean_label_t_raw_preintervention, raw_label_t_list_raw_preintervention, \
            expert_a_t_raw_preintervention, preintervention_raw_segment_intervention_level, segment_weight_1 = cost_buffer.sample_combined(
                [3], [segment_batch_size])

            a_t_preintervention = np.reshape(a_t_raw_preintervention, [-1, self.action_dim]).astype(np.float32)
            is_preint_1 = np.concatenate([np.zeros((segment_batch_size,))], axis=0)
            is_expert_1 = np.concatenate([np.zeros((segment_batch_size,))], axis=0)
            supervision_thresholds_1 = np.max(preintervention_raw_segment_intervention_level, axis=1)

            s_t_raw_preintervention_2, a_t_raw_preintervention_2, r_t_raw_preintervention_2, terminal_t_raw_preintervention_2, expert_only_s_t_raw_preintervention_2, \
            s_t_1_raw_preintervention_2, mean_label_t_raw_preintervention_2, raw_label_t_list_raw_preintervention_2, \
            expert_a_t_raw_preintervention_2, preintervention_raw_segment_intervention_level_2, segment_weight_2, is_random_2 = cost_buffer.sample_preint(
                segment_batch_size, supervision_thresholds_1, is_preint_1, is_expert_1, preint_include_eq=False,
                all_legal=False)

            supervision_thresholds_2 = np.max(preintervention_raw_segment_intervention_level_2, axis=1)
            a_t_preintervention_2 = np.reshape(a_t_raw_preintervention_2, [-1, self.action_dim]).astype(np.float32)

            noise = np.random.uniform(0.0, 0.0, size=a_t_raw_preintervention.shape)
            random_actions_raw_1 = np.clip(noise + a_t_raw_preintervention, -1, 1)
            random_actions_1 = np.reshape(random_actions_raw_1, [-1, self.action_dim]).astype(np.float32)
            offset_error_1, _ = error_function(a_t_raw_preintervention, random_actions_raw_1)

            noise_2 = np.random.uniform(0.0, 0.0, size=a_t_raw_preintervention_2.shape)
            random_actions_raw_2 = np.clip(noise_2 + a_t_raw_preintervention_2, -1, 1)
            random_actions_2 = np.reshape(random_actions_raw_2, [-1, self.action_dim]).astype(np.float32)
            offset_error_2, _ = error_function(a_t_raw_preintervention_2, random_actions_raw_2)

            noise_3 = np.random.normal(loc=0, scale=0.2, size=a_t_raw_preintervention.shape)
            random_actions_raw_3 = np.clip(noise_3 + a_t_raw_preintervention, -1, 1)
            random_actions_3 = np.reshape(random_actions_raw_3, [-1, self.action_dim]).astype(np.float32)
            offset_error_3, offset_single_error_3 = error_function(a_t_raw_preintervention, random_actions_raw_3)

            noise_4 = np.random.normal(loc=0, scale=0.2, size=a_t_raw_preintervention_2.shape)
            random_actions_raw_4 = np.clip(noise_4 + a_t_raw_preintervention_2, -1, 1)
            random_actions_4 = np.reshape(random_actions_raw_4, [-1, self.action_dim]).astype(np.float32)
            offset_error_4, offset_single_error_4 = error_function(a_t_raw_preintervention_2, random_actions_raw_4)
            s_t_raw = np.concatenate([s_t_raw_preintervention, s_t_raw_preintervention_2, s_t_raw_preintervention,
                                      s_t_raw_preintervention_2], axis=0)
            s_t = np.reshape(s_t_raw, [-1, self.state_dim[0]])
            a_t_noisy = np.concatenate([random_actions_1, random_actions_2, random_actions_3, random_actions_4], axis=0)

            expert_intervention_t_raw = np.concatenate(
                [expert_only_s_t_raw_preintervention, expert_only_s_t_raw_preintervention_2], axis=0)

            noisy_cost_s_t, cost_distribution, max_ent_loss = self.current_cost_model([s_t, a_t_noisy])

            noisy_cost_s_t_expert_no_grad = noisy_cost_s_t  # + tf.stop_gradient(noisy_cost_s_t) * expert_intervention_t
            noisy_cost_s_t_expert_no_grad_reshaped = tf.reshape(noisy_cost_s_t_expert_no_grad,
                                                                [s_t_raw.shape[0], self.args.segment_length])
            single_cost_1, single_cost_2, single_cost_3, single_cost_4 = tf.split(
                noisy_cost_s_t_expert_no_grad_reshaped, 4, axis=0)

            noisy_estimated_cost_reshaped = tf.reshape(noisy_cost_s_t_expert_no_grad,
                                                       [s_t_raw.shape[0], self.args.segment_length])
            noisy_estimated_cost_segment = tf.reduce_sum(noisy_estimated_cost_reshaped,
                                                         axis=1)  # prevent instability issues ...
            noisy_estimated_intervention_probability = noisy_estimated_cost_segment / self.args.segment_length
            noisy_estimated_cost_1, noisy_estimated_cost_2, noisy_estimated_cost_3, noisy_estimated_cost_4 = tf.split(
                noisy_estimated_intervention_probability, 4, axis=0)

            coef = self.args.coefficient
            true_cost_no_noise_1, true_per_state_error_1 = error_function(expert_a_t_raw_preintervention,
                                                                          random_actions_raw_1)
            true_cost_no_noise_2, true_per_state_error_2 = error_function(expert_a_t_raw_preintervention_2,
                                                                          random_actions_raw_2)
            preference_loss_1, l2_error, _ = self.get_preference_loss_true_v6(None, None, noisy_estimated_cost_1,
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

            preference_loss_2, l2_error, _ = self.get_preference_loss_true_v6(None, None, noisy_estimated_cost_1,
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

            preference_loss_3, l2_error_3, _ = self.get_preference_loss_true_v6(None, None, noisy_estimated_cost_2,
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
            classification_loss = classification_loss * self.args.cost_loss_modifier + simple_l2_loss

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
                print("Current noise", np.mean(noisy_estimated_cost_1) * self.args.segment_length,
                      np.mean(noisy_estimated_cost_2) * self.args.segment_length)
                print("True no noise", np.mean(true_cost_no_noise_1) * self.args.segment_length,
                      np.mean(true_cost_no_noise_2) * self.args.segment_length)
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
            for v in self.current_cost_model.trainable_weights:
                if 'bias' not in v.name and "cost_network" in v.name:
                    l2_loss += tf.reduce_mean(tf.nn.l2_loss(v)) * self.args.l2
                    layers += 1
            l2_loss = l2_loss / layers
            cost_loss = classification_loss + buffer_expert_loss + l2_loss

            grads = tape.gradient(cost_loss, self.current_cost_model.trainable_weights)
            self.current_cost_optimizer.apply_gradients(zip(grads, self.current_cost_model.trainable_weights))
        return cost_loss, np.mean(preference_loss)


    def train_step_sampled_preferences_with_l2(self, cost_buffer, error_function, preference_dataset,
                                               activate_loss=False, expert_policy=None, verbose=False,
                                               validation_buffer=None):
        with tf.GradientTape() as tape:
            s_t, a_t, _, _, intervention_status_t, _, label_t, _, _ = cost_buffer.simple_sample(self.args.batch_size,
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

            noisy_cost_s_t, _, _ = self.current_cost_model([double_s_t, noisy_a_t])
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
            random_actions_1 = np.reshape(random_actions_raw_1, [-1, self.action_dim]).astype(np.float32)
            offset_error_1, _ = error_function(a_t_raw_1, random_actions_raw_1)

            noise_2 = np.random.uniform(0.0, 0.0, size=a_t_raw_2.shape)
            random_actions_raw_2 = np.clip(noise_2 + a_t_raw_2, -1, 1)
            random_actions_2 = np.reshape(random_actions_raw_2, [-1, self.action_dim]).astype(np.float32)
            offset_error_2, _ = error_function(a_t_raw_2, random_actions_raw_2)

            s_t_raw = np.concatenate([s_t_raw_1, s_t_raw_2], axis=0)
            s_t = np.reshape(s_t_raw, [-1, self.state_dim[0]])
            a_t_noisy = np.concatenate([random_actions_1, random_actions_2], axis=0)

            expert_intervention_t_raw = np.concatenate([expert_only_t_raw_1, expert_only_t_raw_2], axis=0)
            expert_intervention_t = np.reshape(expert_intervention_t_raw, [-1, 1])

            noisy_cost_s_t, cost_distribution, max_ent_loss = self.current_cost_model([s_t, a_t_noisy])
            noisy_cost_s_t_expert_no_grad = noisy_cost_s_t * (
                        1 - expert_intervention_t)  # + tf.stop_gradient(noisy_cost_s_t) * expert_intervention_t
            noisy_cost_s_t_expert_no_grad_reshaped = tf.reshape(noisy_cost_s_t_expert_no_grad,
                                                                [s_t_raw.shape[0], self.args.segment_length])
            single_cost_1, single_cost_2 = tf.split(noisy_cost_s_t_expert_no_grad_reshaped, 2, axis=0)
            noisy_estimated_cost_reshaped = tf.reshape(noisy_cost_s_t_expert_no_grad,
                                                       [s_t_raw.shape[0], self.args.segment_length])
            noisy_estimated_cost_segment = tf.reduce_sum(noisy_estimated_cost_reshaped,
                                                         axis=1)  # prevent instability issues ...
            noisy_estimated_intervention_probability = noisy_estimated_cost_segment / self.args.segment_length
            noisy_estimated_cost_1, noisy_estimated_cost_2 = tf.split(noisy_estimated_intervention_probability, 2,
                                                                      axis=0)

            coef = self.args.coefficient

            true_cost_no_noise_1, true_per_state_error_1 = error_function(expert_a_t_raw_1, random_actions_raw_1)
            true_cost_no_noise_2, true_per_state_error_2 = error_function(expert_a_t_raw_2, random_actions_raw_2)

            preference_loss_1, _, _ = self.get_preference_loss_true_v8(None, None, noisy_estimated_cost_1,
                                                                       noisy_estimated_cost_2,
                                                                       true_cost_no_noise_1 * coef,
                                                                       true_cost_no_noise_2 * coef,
                                                                       single_cost_1, single_cost_2,
                                                                       None,
                                                                       None, offset_error_1,
                                                                       offset_error_2,
                                                                       is_random=None, coef=coef, sanity_check=False)

            preference_loss_3, l2_error_3, _ = self.get_preference_loss_true_v6(None, None, noisy_estimated_cost_2,
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

            if self.args.env_id == "Walker2d-v2" or self.args.env_id == "HalfCheetah-v2":
                loss_weight = 0.01
            else:
                loss_weight = 0.001
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
                print("Current noise", np.mean(noisy_estimated_cost_1) * self.args.segment_length,
                      np.mean(noisy_estimated_cost_2) * self.args.segment_length)
                print("True no noise", np.mean(true_cost_no_noise_1) * self.args.segment_length,
                      np.mean(true_cost_no_noise_2) * self.args.segment_length)
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
            for v in self.current_cost_model.trainable_weights:
                if 'bias' not in v.name and "cost_network" in v.name:
                    l2_loss += tf.reduce_mean(tf.nn.l2_loss(v)) * self.args.l2
                    layers += 1
            l2_loss = l2_loss / layers
            cost_loss = classification_loss + buffer_expert_loss + l2_loss

            grads = tape.gradient(cost_loss, self.current_cost_model.trainable_weights)
            self.current_cost_optimizer.apply_gradients(zip(grads, self.current_cost_model.trainable_weights))
        return cost_loss, np.mean(preference_loss)

    def train_step_sampled_preferences(self, cost_buffer, error_function, preference_dataset, activate_loss=False,
                                       expert_policy=None, verbose=False, validation_buffer=None):
        with tf.GradientTape() as tape:
            s_t, a_t, _, _, intervention_status_t, _, label_t, _, _ = cost_buffer.simple_sample(self.args.batch_size,
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

            noisy_cost_s_t, _, _ = self.current_cost_model([double_s_t, noisy_a_t])
            buffer_expert_loss_raw = (noisy_cost_s_t - cost_label) ** 2
            buffer_expert_loss = tf.reduce_mean(buffer_expert_loss_raw)
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
            random_actions_1 = np.reshape(random_actions_raw_1, [-1, self.action_dim]).astype(np.float32)
            offset_error_1, _ = error_function(a_t_raw_1, random_actions_raw_1)

            noise_2 = np.random.uniform(0.0, 0.0, size=a_t_raw_2.shape)
            random_actions_raw_2 = np.clip(noise_2 + a_t_raw_2, -1, 1)
            random_actions_2 = np.reshape(random_actions_raw_2, [-1, self.action_dim]).astype(np.float32)
            offset_error_2, _ = error_function(a_t_raw_2, random_actions_raw_2)

            s_t_raw = np.concatenate([s_t_raw_1, s_t_raw_2], axis=0)
            s_t = np.reshape(s_t_raw, [-1, self.state_dim[0]])
            a_t_noisy = np.concatenate([random_actions_1, random_actions_2], axis=0)

            expert_intervention_t_raw = np.concatenate([expert_only_t_raw_1, expert_only_t_raw_2], axis=0)
            expert_intervention_t = np.reshape(expert_intervention_t_raw, [-1, 1])

            noisy_cost_s_t, cost_distribution, max_ent_loss = self.current_cost_model([s_t, a_t_noisy])
            noisy_cost_s_t_expert_no_grad = noisy_cost_s_t * (
                        1 - expert_intervention_t)  # + tf.stop_gradient(noisy_cost_s_t) * expert_intervention_t
            noisy_cost_s_t_expert_no_grad_reshaped = tf.reshape(noisy_cost_s_t_expert_no_grad,
                                                                [s_t_raw.shape[0], self.args.segment_length])
            single_cost_1, single_cost_2 = tf.split(noisy_cost_s_t_expert_no_grad_reshaped, 2, axis=0)
            noisy_estimated_cost_reshaped = tf.reshape(noisy_cost_s_t_expert_no_grad,
                                                       [s_t_raw.shape[0], self.args.segment_length])
            noisy_estimated_cost_segment = tf.reduce_sum(noisy_estimated_cost_reshaped,
                                                         axis=1)  # prevent instability issues ...
            noisy_estimated_intervention_probability = noisy_estimated_cost_segment / self.args.segment_length
            noisy_estimated_cost_1, noisy_estimated_cost_2 = tf.split(noisy_estimated_intervention_probability, 2,
                                                                      axis=0)

            coef = self.args.coefficient

            true_cost_no_noise_1, true_per_state_error_1 = error_function(expert_a_t_raw_1, random_actions_raw_1)
            true_cost_no_noise_2, true_per_state_error_2 = error_function(expert_a_t_raw_2, random_actions_raw_2)

            preference_loss_1, _, _ = self.get_preference_loss_true_v8(None, None, noisy_estimated_cost_1,
                                                                       noisy_estimated_cost_2,
                                                                       true_cost_no_noise_1 * coef,
                                                                       true_cost_no_noise_2 * coef,
                                                                       single_cost_1, single_cost_2,
                                                                       None,
                                                                       None, offset_error_1,
                                                                       offset_error_2,
                                                                       is_random=None, coef=coef)

            preference_loss_3, l2_error_3, _ = self.get_preference_loss_true_v6(None, None, noisy_estimated_cost_2,
                                                                                np.ones_like(noisy_estimated_cost_2),
                                                                                true_cost_no_noise_2 * coef,
                                                                                np.ones_like(
                                                                                    true_cost_no_noise_2) * coef,
                                                                                single_cost_1, single_cost_2,
                                                                                None, None, offset_error_1,
                                                                                offset_error_2,
                                                                                is_random=None, coef=coef,
                                                                                sanity_check=False)
            preference_loss = preference_loss_1 + preference_loss_3
            special_l2 = noisy_estimated_cost_2 ** 2

            classification_loss = (tf.reduce_mean(
                preference_loss) * 0.001) * self.args.preference_loss_weight
            classification_loss = classification_loss * self.args.cost_loss_modifier


            if verbose:
                cost_gt_0_5 = np.mean(noisy_estimated_cost_reshaped > 0.5, axis=1)

                error_single_state_1_l2 = tf.reduce_mean(tf.abs(single_cost_1 - true_per_state_error_1) ** 2)
                error_single_state_2_l2 = tf.reduce_mean(tf.abs(single_cost_2 - true_per_state_error_2) ** 2)

                error_single_state_1_l1 = tf. \
                    reduce_mean(tf.abs(single_cost_1 - true_per_state_error_1))
                error_single_state_2_l1 = tf.reduce_mean(tf.abs(single_cost_2 - true_per_state_error_2))

                print(error_single_state_1_l1.shape)

                print("\n")
                print(np.mean(preference_loss), np.mean(special_l2), np.mean(classification_loss))
                print("State Err L1, L2:", np.mean(error_single_state_1_l1), np.mean(error_single_state_2_l1),
                      np.mean(error_single_state_1_l2), np.mean(error_single_state_2_l2), )
                print("Current noise", np.mean(noisy_estimated_cost_1) * self.args.segment_length,
                      np.mean(noisy_estimated_cost_2) * self.args.segment_length)
                print("True no noise", np.mean(true_cost_no_noise_1) * self.args.segment_length,
                      np.mean(true_cost_no_noise_2) * self.args.segment_length)
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
            for v in self.current_cost_model.trainable_weights:
                if 'bias' not in v.name and "cost_network" in v.name:
                    l2_loss += tf.reduce_mean(tf.nn.l2_loss(v)) * self.args.l2
                    layers += 1
            l2_loss = l2_loss / layers
            cost_loss = classification_loss + buffer_expert_loss + l2_loss

            grads = tape.gradient(cost_loss, self.current_cost_model.trainable_weights)
            self.current_cost_optimizer.apply_gradients(zip(grads, self.current_cost_model.trainable_weights))
        return cost_loss, np.mean(preference_loss)

    def train_step_cost_true_v6(self, cost_buffer, error_function, activate_loss=False, expert_policy=None,
                                verbose=False, validation_buffer=None, group_size=8):

        segment_batch_size = 1
        s_t_raw_preintervention, a_t_raw_preintervention, r_t_raw_preintervention, terminal_t_raw_preintervention, expert_only_s_t_raw_preintervention, \
        s_t_1_raw_preintervention, mean_label_t_raw_preintervention, raw_label_t_list_raw_preintervention, \
        expert_a_t_raw_preintervention, preintervention_raw_segment_intervention_level, segment_weight_1 = cost_buffer.sample_combined(
            [7], [segment_batch_size])
        a_t_preintervention = np.reshape(a_t_raw_preintervention, [-1, self.action_dim]).astype(np.float32)
        is_preint_1 = np.concatenate([np.zeros((segment_batch_size,))], axis=0)

        is_expert_1 = np.concatenate([np.zeros((segment_batch_size,))], axis=0)
        supervision_thresholds_1 = np.max(preintervention_raw_segment_intervention_level, axis=1)

        s_t_raw_preintervention_2, a_t_raw_preintervention_2, r_t_raw_preintervention_2, terminal_t_raw_preintervention_2, expert_only_s_t_raw_preintervention_2, \
        s_t_1_raw_preintervention_2, mean_label_t_raw_preintervention_2, raw_label_t_list_raw_preintervention_2, \
        expert_a_t_raw_preintervention_2, preintervention_raw_segment_intervention_level_2, segment_weight_2, = cost_buffer.sample_combined(
            [7], [segment_batch_size])

        supervision_thresholds_2 = np.max(preintervention_raw_segment_intervention_level_2, axis=1)
        a_t_preintervention_2 = np.reshape(a_t_raw_preintervention_2, [-1, self.action_dim]).astype(np.float32)

        noise = np.random.uniform(0.0, 0.0, size=a_t_raw_preintervention.shape)
        random_actions_raw_1 = np.clip(noise + a_t_raw_preintervention, -1, 1)
        random_actions_1 = np.reshape(random_actions_raw_1, [-1, self.action_dim]).astype(np.float32)
        offset_error_1, _ = error_function(a_t_raw_preintervention, random_actions_raw_1)

        noise_2 = np.random.uniform(0.0, 0.0, size=a_t_raw_preintervention_2.shape)
        random_actions_raw_2 = np.clip(noise_2 + a_t_raw_preintervention_2, -1, 1)
        random_actions_2 = np.reshape(random_actions_raw_2, [-1, self.action_dim]).astype(np.float32)
        offset_error_2, _ = error_function(a_t_raw_preintervention_2, random_actions_raw_2)

        noise_3 = np.random.uniform(0.5, 1.0, size=a_t_raw_preintervention.shape)
        random_actions_raw_3 = np.clip(noise_3 + a_t_raw_preintervention, -1, 1)
        random_actions_3 = np.reshape(random_actions_raw_3, [-1, self.action_dim]).astype(np.float32)
        offset_error_3, offset_single_error_3 = error_function(a_t_raw_preintervention, random_actions_raw_3)

        noise_4 = np.random.uniform(0.5, 1.0, size=a_t_raw_preintervention_2.shape)
        random_actions_raw_4 = np.clip(noise_4 + a_t_raw_preintervention_2, -1, 1)
        random_actions_4 = np.reshape(random_actions_raw_4, [-1, self.action_dim]).astype(np.float32)
        offset_error_4, offset_single_error_4 = error_function(a_t_raw_preintervention_2, random_actions_raw_4)

        s_t_raw = np.concatenate(
            [s_t_raw_preintervention, s_t_raw_preintervention_2, s_t_raw_preintervention, s_t_raw_preintervention_2],
            axis=0)
        s_t = np.reshape(s_t_raw, [-1, self.state_dim[0]])
        a_t_noisy = np.concatenate([random_actions_1, random_actions_2, random_actions_3, random_actions_4], axis=0)
        for i in range(1000):
            with tf.GradientTape() as tape:

                noisy_cost_s_t, cost_distribution, max_ent_loss = self.current_cost_model([s_t, a_t_noisy])

                noisy_cost_s_t_expert_no_grad = noisy_cost_s_t
                noisy_cost_s_t_expert_no_grad_reshaped = tf.reshape(noisy_cost_s_t_expert_no_grad,
                                                                    [s_t_raw.shape[0], self.args.segment_length])
                single_cost_1, single_cost_2, single_cost_3, single_cost_4 = tf.split(
                    noisy_cost_s_t_expert_no_grad_reshaped, 4, axis=0)

                noisy_estimated_cost_reshaped = tf.reshape(noisy_cost_s_t_expert_no_grad,
                                                           [s_t_raw.shape[0], self.args.segment_length])
                noisy_estimated_cost_segment = tf.reduce_sum(noisy_estimated_cost_reshaped,axis=1)  # prevent instability issues ...
                noisy_estimated_intervention_probability = noisy_estimated_cost_segment / self.args.segment_length
                noisy_estimated_cost_1, noisy_estimated_cost_2, noisy_estimated_cost_3, noisy_estimated_cost_4 = tf.split(
                    noisy_estimated_intervention_probability, 4, axis=0)

                coef = self.args.coefficient
                true_cost_no_noise_1, true_per_state_error_1 = error_function(expert_a_t_raw_preintervention,
                                                                              random_actions_raw_1)
                true_cost_no_noise_2, true_per_state_error_2 = error_function(expert_a_t_raw_preintervention_2,
                                                                              random_actions_raw_2)
                true_cost_no_noise_3, true_per_state_error_3 = error_function(expert_a_t_raw_preintervention,
                                                                              random_actions_raw_3)
                true_cost_no_noise_4, true_per_state_error_4 = error_function(expert_a_t_raw_preintervention_2,
                                                                              random_actions_raw_4)

                preference_loss_1, l2_error, _ = self.get_preference_loss_true_v9(None, None, noisy_estimated_cost_1,
                                                                                  noisy_estimated_cost_2,
                                                                                  true_cost_no_noise_1 * coef,
                                                                                  true_cost_no_noise_2 * coef,
                                                                                  single_cost_1, single_cost_2,
                                                                                  np.zeros_like(is_preint_1),
                                                                                  None, offset_error_1,
                                                                                  offset_error_2,
                                                                                  is_random=None, coef=coef)

                preference_loss_2, l2_error_2, _ = self.get_preference_loss_true_v10(None, None, noisy_estimated_cost_1,
                                                                                     noisy_estimated_cost_3,
                                                                                     true_cost_no_noise_1 * coef,
                                                                                     true_cost_no_noise_3 * coef,
                                                                                     single_cost_1, single_cost_3,
                                                                                     np.zeros_like(is_preint_1),
                                                                                     None, offset_error_1,
                                                                                     offset_error_2,
                                                                                     is_random=None, coef=coef)

                preference_loss_3, l2_error_3, _ = self.get_preference_loss_true_v10(None, None, noisy_estimated_cost_2,
                                                                                     np.ones_like(
                                                                                         noisy_estimated_cost_4),
                                                                                     true_cost_no_noise_2 * coef,
                                                                                     true_cost_no_noise_4 * coef,
                                                                                     single_cost_2, single_cost_4,
                                                                                     np.zeros_like(is_preint_1),
                                                                                     None, offset_error_1,
                                                                                     offset_error_2,
                                                                                     is_random=None, coef=coef)

                preference_loss = preference_loss_1  # + preference_loss_2 + preference_loss_3
                classification_loss = tf.reduce_mean(
                    preference_loss) * 0.05  # + tf.reduce_mean(special_l2) * 0.0001# + tf.reduce_mean(l2_loss_2) * 0.005  # + tf.reduce_mean(l2_loss_2) * 0.003
                classification_loss = classification_loss  # * self.args.cost_loss_modifier
                l2_loss = 0
                layers = 0
                for v in self.current_cost_model.trainable_weights:
                    if 'bias' not in v.name and "cost_network" in v.name:
                        l2_loss += tf.reduce_mean(tf.nn.l2_loss(v)) * self.args.l2
                        layers += 1
                l2_loss = l2_loss / layers
                cost_loss = classification_loss + l2_loss

                grads = tape.gradient(cost_loss, self.current_cost_model.trainable_weights)
                self.current_cost_optimizer.apply_gradients(zip(grads, self.current_cost_model.trainable_weights))

                print(i, classification_loss.numpy(), preference_loss.numpy(), l2_error.numpy(), "||",
                      noisy_estimated_cost_1.numpy(),
                      noisy_estimated_cost_2.numpy(), true_cost_no_noise_1, true_cost_no_noise_2)

        return cost_loss, np.mean(preference_loss)


    def get_preference_loss_true_v6(self,
                                    supervision_threshold_1, supervision_threshold_2,
                                    noisy_estimated_cost_1, noisy_estimated_cost_2,
                                    true_cost_1, true_cost_2,
                                    single_cost_1, single_cost_2,
                                    is_preint_1, is_equal,
                                    segment_1_upper_cap_error, segment_2_upper_cap_error, is_random=None, coef=30,
                                    sanity_check=True):

        true_2_gt_1 = (true_cost_2 >= true_cost_1).astype(np.float32)
        true_1_gt_2 = (true_cost_1 >= true_cost_2).astype(np.float32)
        if np.mean(true_1_gt_2) > 0 and sanity_check:
            for i in range(true_cost_1.shape[0]):
                if true_cost_2[i] < true_cost_1[i]:
                    print(i, "Error 1, ", true_cost_1[i], true_cost_2[i], single_cost_1[i], single_cost_2[i])
                    quit()



        modified_cost_1 = tf.clip_by_value(noisy_estimated_cost_1 * coef, 0, 50)
        modified_cost_2 = tf.clip_by_value(noisy_estimated_cost_2 * coef, 0, 50)
        prob_2_gt_1 = tf.exp(modified_cost_2) / (tf.exp(modified_cost_2) + tf.exp(modified_cost_1))

        preference_loss = -tf.math.log(prob_2_gt_1 + 1e-8)

        l2_loss = 0
        equal_preference_loss = 0

        return preference_loss, equal_preference_loss, l2_loss


    def get_preference_loss_true_v8(self,
                                    supervision_threshold_1, supervision_threshold_2,
                                    noisy_estimated_cost_1, noisy_estimated_cost_2,
                                    true_cost_1, true_cost_2,
                                    single_cost_1, single_cost_2,
                                    is_preint_1, is_equal,
                                    segment_1_upper_cap_error, segment_2_upper_cap_error, is_random=None, coef=30,
                                    sanity_check=True):

        true_2_gt_1 = (true_cost_2 >= true_cost_1).astype(np.float32)
        true_1_gt_2 = (true_cost_1 >= true_cost_2).astype(np.float32)
        modified_cost_1 = tf.clip_by_value(noisy_estimated_cost_1 * coef, 0, 50)
        modified_cost_2 = tf.clip_by_value(noisy_estimated_cost_2 * coef, 0, 50)
        prob_2_gt_1 = tf.exp(modified_cost_2) / (tf.exp(modified_cost_2) + tf.exp(modified_cost_1))
        prob_1_gt_2 = 1 - prob_2_gt_1

        preference_loss = -(
                    true_2_gt_1 * tf.math.log(prob_2_gt_1 + 1e-8) + true_1_gt_2 * tf.math.log(prob_1_gt_2 + 1e-8))

        l2_loss = true_2_gt_1 * noisy_estimated_cost_2 + true_1_gt_2 * noisy_estimated_cost_1

        return preference_loss, l2_loss, 0


    def train_step_cost(self, cost_buffer, error_function, expert_policy=None, verbose=False, validation_buffer=None):
        with tf.GradientTape() as tape:
            s_t, a_t, _, _, intervention_status_t, _, label_t, _, _ = cost_buffer.simple_sample(self.args.batch_size,
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

            noisy_cost_s_t, _, _ = self.current_cost_model([double_s_t, noisy_a_t])
            buffer_expert_loss_raw = (noisy_cost_s_t - cost_label) ** 2
            buffer_expert_loss = tf.reduce_mean(buffer_expert_loss_raw)

            l2_loss = 0
            layers = 0
            for v in self.current_cost_model.trainable_weights:
                if 'bias' not in v.name and "cost_network" in v.name:
                    l2_loss += tf.reduce_mean(tf.nn.l2_loss(v)) * self.args.l2
                    layers += 1
            l2_loss = l2_loss / layers
            cost_loss = buffer_expert_loss + l2_loss

            grads = tape.gradient(cost_loss, self.current_cost_model.trainable_weights)
            self.current_cost_optimizer.apply_gradients(zip(grads, self.current_cost_model.trainable_weights))
        return buffer_expert_loss, 0


    def pretrain_step_actor(self, cost_buffer):
        i = 0

        with tf.GradientTape() as tape:
            # actor update step
            s_t, a_t, _, _, intervention_status_t, _, label_t, expert_a_t, _ = cost_buffer.simple_sample(
                self.args.batch_size, mode=1)
            # a_t = expert_a_t

            bc_a_t = self.current_model_list[i](s_t)
            bc_loss_raw = (bc_a_t - a_t) ** 2  # only learn the expert states
            bc_loss = tf.reduce_mean(bc_loss_raw)

            l2_loss = 0
            layers = 0
            for v in self.current_model_list[i].trainable_weights:
                if 'bias' not in v.name and "current_actor" in v.name:
                    l2_loss += tf.reduce_mean(tf.nn.l2_loss(v)) * self.args.l2
                    layers += 1
            l2_loss = l2_loss / layers
            policy_loss = bc_loss + l2_loss
            grads = tape.gradient(policy_loss, self.current_model_list[i].trainable_weights)
        self.current_optimizer_list[i].apply_gradients(zip(grads, self.current_model_list[i].trainable_weights))
        return bc_loss

    def pretrain_step_actor_partial(self, cost_buffer):
        i = 0

        with tf.GradientTape() as tape:
            # actor update step
            s_t, a_t, _, _, intervention_status_t, _, label_t, expert_a_t, _ = cost_buffer.simple_sample(
                self.args.batch_size, mode=7)
            # a_t = expert_a_t

            bc_a_t = self.current_model_list[i](s_t)
            bc_loss_raw = (bc_a_t - a_t) ** 2  # only learn the expert states
            bc_loss = tf.reduce_mean(bc_loss_raw)

            l2_loss = 0
            layers = 0
            for v in self.current_model_list[i].trainable_weights:
                if 'bias' not in v.name and "current_actor" in v.name:
                    l2_loss += tf.reduce_mean(tf.nn.l2_loss(v)) * self.args.l2
                    layers += 1
            l2_loss = l2_loss / layers
            policy_loss = bc_loss + l2_loss
            grads = tape.gradient(policy_loss, self.current_model_list[i].trainable_weights)
        self.current_optimizer_list[i].apply_gradients(zip(grads, self.current_model_list[i].trainable_weights))
        return bc_loss

    def pretrain_step_actor_all(self, cost_buffer):
        i = 0

        with tf.GradientTape() as tape:
            # actor update step
            s_t, a_t, _, _, intervention_status_t, _, label_t, expert_a_t, _ = cost_buffer.simple_sample(
                self.args.batch_size, mode=0)
            bc_a_t = self.current_model_list[i](s_t)
            bc_loss_raw = (bc_a_t - expert_a_t) ** 2  # only learn the expert states
            bc_loss = tf.reduce_mean(bc_loss_raw)

            l2_loss = 0
            layers = 0
            for v in self.current_model_list[i].trainable_weights:
                if 'bias' not in v.name and "current_actor" in v.name:
                    l2_loss += tf.reduce_mean(tf.nn.l2_loss(v)) * self.args.l2
                    layers += 1
            l2_loss = l2_loss / layers
            policy_loss = bc_loss + l2_loss
            grads = tape.gradient(policy_loss, self.current_model_list[i].trainable_weights)
        self.current_optimizer_list[i].apply_gradients(zip(grads, self.current_model_list[i].trainable_weights))
        return bc_loss

    def train_step_actor(self, buffer_list, cost_buffer, expert_policy):
        i = 0
        with tf.GradientTape() as tape:
            # actor update step
            s_t, a_t, _, _, intervention_status_t, _, label_t, _, _ = cost_buffer.simple_sample(self.args.batch_size,
                                                                                                mode=1)
            bc_a_t = self.current_model_list[i](s_t)
            bc_loss_raw = tf.reduce_mean((bc_a_t - a_t) ** 2, axis=1)  # only learn the expert states
            bc_loss = tf.reduce_mean(bc_loss_raw)

            intervention_loss = 0
            if self.args.intervention_loss:
                s_t_no_expert, a_t_no_expert, _, _, intervention_status_t_no_expert, _, label_t, _, _ = cost_buffer.simple_sample(
                    self.args.batch_size, mode=6)

                predicted_a_t_no_expert = self.current_model_list[i](s_t_no_expert)
                cost_non_expert_action, _, _ = self.current_cost_model(
                    [s_t_no_expert, predicted_a_t_no_expert])  # Good state = 0, bad state = 1
                cost_non_expert_action = tf.clip_by_value(cost_non_expert_action - self.args.cost_cap, 0, 1) / (
                            1 - self.args.cost_cap)
                intervention_loss = tf.reduce_mean(cost_non_expert_action)

            l2_loss = 0
            layers = 0
            for v in self.current_model_list[i].trainable_weights:
                if 'bias' not in v.name and "current_actor" in v.name:
                    l2_loss += tf.reduce_mean(tf.nn.l2_loss(v)) * self.args.l2
                    layers += 1
            l2_loss = l2_loss / layers
            policy_loss = bc_loss + intervention_loss + l2_loss

            grads = tape.gradient(policy_loss, self.current_model_list[i].trainable_weights)
        self.current_optimizer_list[i].apply_gradients(zip(grads, self.current_model_list[i].trainable_weights))
        return policy_loss, bc_loss, intervention_loss

    def train_step_actor_v2(self, buffer_list, cost_buffer, expert_policy):
        i = 0
        with tf.GradientTape() as tape:
            # actor update step
            s_t, a_t, _, _, intervention_status_t, _, label_t, _, _ = cost_buffer.simple_sample(self.args.batch_size,
                                                                                                mode=1)
            bc_a_t = self.current_model_list[i](s_t)
            bc_loss_raw = tf.reduce_mean((bc_a_t - a_t) ** 2, axis=1)  # only learn the expert states
            bc_loss = tf.reduce_mean(bc_loss_raw)

            intervention_loss = 0
            if self.args.intervention_loss:
                s_t_no_expert, a_t_no_expert, _, _, intervention_status_t_no_expert, _, label_t, _, _ = cost_buffer.simple_sample(
                    self.args.batch_size, mode=6)

                predicted_a_t_no_expert = self.current_model_list[i](s_t_no_expert)

                if self.args.env_id == "Walker2d-v2":
                    noisy_predicted_a_t_no_expert = predicted_a_t_no_expert
                elif self.args.env_id == "Hopper-v2" or self.args.env_id == "Ant-v2":
                    noise = tf.clip_by_value(tf.random.normal(a_t.shape) * 0.1, -0.25, 0.25)
                    noisy_predicted_a_t_no_expert = tf.clip_by_value(predicted_a_t_no_expert + noise, -1, 1)
                else:
                    noise = tf.clip_by_value(tf.random.normal(a_t.shape) * 0.2, -0.5, 0.5)
                    noisy_predicted_a_t_no_expert = tf.clip_by_value(predicted_a_t_no_expert + noise, -1, 1)

                cost_non_expert_action, _, _ = self.current_cost_model(
                    [s_t_no_expert, noisy_predicted_a_t_no_expert])  # Good state = 0, bad state = 1
                cost_non_expert_action = tf.clip_by_value(cost_non_expert_action - self.args.cost_cap, 0, 1) / (
                            1 - self.args.cost_cap)
                intervention_loss = tf.reduce_mean(cost_non_expert_action)

            l2_loss = 0
            layers = 0
            for v in self.current_model_list[i].trainable_weights:
                if 'bias' not in v.name and "current_actor" in v.name:
                    l2_loss += tf.reduce_mean(tf.nn.l2_loss(v)) * self.args.l2
                    layers += 1
            l2_loss = l2_loss / layers
            policy_loss = bc_loss + intervention_loss + l2_loss

            grads = tape.gradient(policy_loss, self.current_model_list[i].trainable_weights)
        self.current_optimizer_list[i].apply_gradients(zip(grads, self.current_model_list[i].trainable_weights))
        return policy_loss, bc_loss, intervention_loss


    def load_imitation_model(self, frame_num):
        for i in range(self.args.ensemble_size):
            self.current_model_list[i].load_weights(
                self.args.checkpoint_dir + "/" + self.args.custom_id + "/current_actor_network_" + str(
                    frame_num) + "_" + str(i))

    def save_imitation_model(self, frame_num):
        # save current + target
        for i in range(self.args.ensemble_size):
            self.current_model_list[i].save_weights(
                self.args.checkpoint_dir + "/" + self.args.custom_id + "/current_actor_network_" + str(
                    frame_num) + "_" + str(i),
                overwrite=True)

    def load_cost_model(self, name):
        self.current_cost_model.load_weights(
            self.args.checkpoint_dir + "/" + self.args.custom_id + "/current_cost_network_" + name)

    def save_cost_model(self, name):
        # save current + target
        self.current_cost_model.save_weights(
            self.args.checkpoint_dir + "/" + self.args.custom_id + "/current_cost_network_" + name,
            overwrite=True)
