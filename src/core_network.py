import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras.optimizers import Adam
import utils
from tensorflow.keras import regularizers
import math

from train_step_cost import noisy_expert
from train_step_cost import EIL_without_expert
from train_step_cost import EIL_with_expert
from train_step_cost import pref_without_expert
from train_step_cost import pref_with_expert
from train_step_cost import delayed_intervention_debugging_v1


from train_step_cost import delayed_intervention_v5
from train_step_cost import delayed_intervention_v6
from train_step_cost import delayed_intervention_v7
from train_step_cost import delayed_intervention_v9


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



    def train_step_cost_core(self, cost_buffer, error_function, activate_loss=False, expert_policy=None,
                                verbose=False):

        if self.args.cost_version == 1:
            #Noisy Expert
            return noisy_expert.train_step_cost(self, cost_buffer, error_function, expert_policy=expert_policy, verbose=verbose)

        elif self.args.cost_version == 5:
            #debuggiong cost version
            return delayed_intervention_v5.train_step_cost(self, cost_buffer, error_function, expert_policy=expert_policy, verbose=verbose)
        elif self.args.cost_version == 6:
            #debuggiong cost version
            return delayed_intervention_v6.train_step_cost(self, cost_buffer, error_function, expert_policy=expert_policy, verbose=verbose)
        elif self.args.cost_version == 7:
            #debuggiong cost version
            return delayed_intervention_v7.train_step_cost(self, cost_buffer, error_function, expert_policy=expert_policy, verbose=verbose)
        elif self.args.cost_version == 9:
            #debuggiong cost version
            return delayed_intervention_v9.train_step_cost(self, cost_buffer, error_function, expert_policy=expert_policy, verbose=verbose)


        elif self.args.cost_version == 10:
            #EIL without expert
            return EIL_without_expert.train_step_cost(self, cost_buffer, error_function, expert_policy=expert_policy, verbose=verbose)
        elif self.args.cost_version == 11:
            # EIL expert
            return EIL_with_expert.train_step_cost(self, cost_buffer, error_function, expert_policy=expert_policy, verbose=verbose)
        elif self.args.cost_version == 12:
            #pref with expert
            return pref_without_expert.train_step_cost(self, cost_buffer, error_function, expert_policy=expert_policy, verbose=verbose)
        elif self.args.cost_version == 13:
            #pref without expert
            return pref_with_expert.train_step_cost(self, cost_buffer, error_function, expert_policy=expert_policy, verbose=verbose)
        elif self.args.cost_version == 20:
            #debuggiong cost version
            return delayed_intervention_debugging_v1.train_step_cost(self, cost_buffer, error_function, expert_policy=expert_policy, verbose=verbose)



    def update_target_network(self):
        current_weights = self.current_cost_model.get_weights()
        target_weights = self.target_cost_model.get_weights()

        new_target_weights = []
        for i in range(len(current_weights)):
            new_weight = current_weights[i] * self.args.target_update_rate + target_weights[i] * (
                        1 - self.args.target_update_rate)
            new_target_weights.append(new_weight)
        self.target_cost_model.set_weights(new_target_weights)




    def get_preference_loss_true_v6(self,
                                    supervision_threshold_1, supervision_threshold_2,
                                    noisy_estimated_cost_1, noisy_estimated_cost_2,
                                    true_cost_1, true_cost_2,
                                    single_cost_1, single_cost_2,
                                    is_preint_1, is_equal,
                                    segment_1_upper_cap_error, segment_2_upper_cap_error, is_random=None, coef=30,
                                    sanity_check=True):
        """
        Only uses true cost for sanity checking, ensuring that the preference hold where segment 2 should always have higher cost than segment 1.
        Outputs the preference loss between the two
        """
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
        """
        True Preference -> Assumes full knowledge of the true cost of both edges
        Outputs the preference loss between the two
        """
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
