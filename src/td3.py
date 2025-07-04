import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras.optimizers import Adam
import utils
from tensorflow.keras import regularizers

LOG_SIG_CAP_MAX = 2
LOG_SIG_CAP_MIN = -20

class actor:
    def __init__(self, args, state_dim, action_dim, model_name="simple_gaussian_actor"):
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.args = args
        self.model_name = model_name
        self.state_input = tf.keras.layers.Input(shape=self.state_dim)
        self.policy_noise=args.policy_noise
        self.noise_clip=args.noise_clip
        self.current_model, self.current_vars, self.current_optimizer = self.build_network(model_name="current")
        self.target_model, self.target_vars, self.target_optimizer = self.build_network(model_name="current")

    def build_network(self, model_name="current", epsilon=1e-6):
        """
        Constructs the policy network.
        """
        init = tf.keras.initializers.GlorotNormal()

        l2 = tf.keras.layers.Dense(self.args.hidden,kernel_initializer=init,activation=tf.nn.relu, name=model_name + "_actor_network_1")(self.state_input)
        l3 = tf.keras.layers.Dense(self.args.hidden,kernel_initializer=init,activation=tf.nn.relu, name=model_name + "_actor_network_2")(l2)
        l4 = tf.keras.layers.Dense(self.args.hidden,kernel_initializer=init,activation=tf.nn.relu, name=model_name + "_actor_network_3")(l3)
        l5 = tf.keras.layers.Dense(self.action_dim, kernel_initializer=init,activation=None, name=model_name + "_actor_network_4")(l4)
        current_action = tf.tanh(l5)

        current_model = tf.keras.Model(inputs=self.state_input, outputs=[current_action])
        current_vars = current_model.trainable_weights
        optimizer = Adam(learning_rate=self.args.ac_lr)
        return current_model, current_vars, optimizer

    def update_actor_network(self):
        """
        Soft updates target network parameters.
        """
        current_weights = self.current_model.get_weights()
        target_weights = self.target_model.get_weights()

        new_target_weights = []
        for i in range(len(current_weights)):
            new_weight = current_weights[i] * self.args.target_update_rate + target_weights[i] * (1 - self.args.target_update_rate)
            new_target_weights.append(new_weight)
        self.target_model.set_weights(new_target_weights)

    def get_random_action(self):
        return np.random.uniform(-1, 1, size=(self.action_dim))

    def get_current_action(self, state, verbose=False, noisy=False):
        action = self.current_model(state)[0]
        if noisy:
            noise = np.clip(np.random.normal(size=action.shape) * self.policy_noise, -self.noise_clip, self.noise_clip)
            noisy_actions = np.clip(noise + action, -1, 1)
            return noisy_actions
        else:
            return action.numpy()

    def train_step_actor(self, buffer, critic):
        """
        Performs one actor update and returns (policy_loss, value_loss).
        """
        s_t, s_t_1, a_t, r_t, terminal_t = buffer.sample()
        with tf.GradientTape() as tape:

            actions = self.current_model(s_t)
            noise = tf.clip_by_value(tf.random.normal(actions.shape) * self.policy_noise, -self.noise_clip, self.noise_clip)
            noisy_actions = tf.clip_by_value(noise + actions, -1, 1)
            qvf1 = critic.target_qvf_model_1([s_t, noisy_actions])
            qvf2 = critic.target_qvf_model_2([s_t, noisy_actions])

            min_target = tf.minimum(qvf1, qvf2)
            l2_loss = 0
            layers = 0
            for v in self.current_model.trainable_weights:
                if 'bias' not in v.name and "current_actor" in v.name:
                    l2_loss += tf.reduce_mean(tf.nn.l2_loss(v)) * self.args.l2
                    layers += 1
            l2_loss = l2_loss/layers
            policy_loss = tf.reduce_mean(- qvf1) + l2_loss
            grads = tape.gradient(policy_loss, self.current_model.trainable_weights)
        self.current_optimizer.apply_gradients(zip(grads, self.current_model.trainable_weights))
        vf_loss = critic.train_step_vf(s_t, min_target)
        return policy_loss, vf_loss

    def load_model(self, frame_num):
        """
        Saves current model weights to disk.
        """
        self.current_model.load_weights(self.args.checkpoint_dir + "/" + self.args.custom_id +  "/actor_current_network_" + str(frame_num))

    def save_model(self, frame_num):
        """
        Loads model weights from disk.
        """
        self.current_model.save_weights(self.args.checkpoint_dir + "/" + self.args.custom_id +  "/actor_current_network_" + str(frame_num), overwrite=True)


class critic:
    def __init__(self, args, state_dim, action_dim, model_name="simple_gaussian_critic"):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.args = args
        self.model_name = model_name

        state_input = tf.keras.layers.Input(shape=self.state_dim, name="state_input")
        action_input = tf.keras.layers.Input(shape=self.action_dim, name="action_input")

        self.current_qvf_1 = self.build_q_network("current_qvf1", state_input, action_input)
        self.current_qvf_2 = self.build_q_network("current_qvf2", state_input, action_input)
        self.target_qvf_1 = self.build_q_network("target_qvf1", state_input, action_input)
        self.target_qvf_2 = self.build_q_network("target_qvf2", state_input, action_input)


        self.current_vf = self.build_value_network("current_vf", state_input)
        self.target_vf = self.build_value_network("target_vf", state_input)

        self.current_qvf_model_1 = tf.keras.Model(inputs=[state_input, action_input], outputs=[self.current_qvf_1])
        self.current_qvf_model_2 = tf.keras.Model(inputs=[state_input, action_input], outputs=[self.current_qvf_2])
        self.target_qvf_model_1 = tf.keras.Model(inputs=[state_input, action_input], outputs=[self.target_qvf_1])
        self.target_qvf_model_2 = tf.keras.Model(inputs=[state_input, action_input], outputs=[self.target_qvf_2])


        self.current_vf_model = tf.keras.Model(inputs=state_input, outputs=self.current_vf)
        self.target_vf_model = tf.keras.Model(inputs=state_input, outputs=self.target_vf)

        self.optimizer_qvf_1 = Adam(learning_rate=self.args.qvf_lr)
        self.optimizer_qvf_2 = Adam(learning_rate=self.args.qvf_lr)

        self.optimizer_vf = Adam(learning_rate=self.args.vf_lr)
        self.policy_prior = tfp.distributions.Normal(
            loc=tf.zeros((action_dim, )), scale=tf.ones((self.action_dim, )))

    def train_step_vf(self, state, min_log_target, epsilon=1e-6):
        """
        Updates value network toward min(Q1, Q2).
        """
        target_value = tf.stop_gradient(min_log_target)
        with tf.GradientTape() as tape:
            vf = self.current_vf_model(state)
            vf_loss = 0.5 * tf.reduce_mean((vf - target_value) ** 2)
            grads = tape.gradient(vf_loss, self.current_vf_model.trainable_weights)
        self.optimizer_vf.apply_gradients(zip(grads, self.current_vf_model.trainable_weights))
        return vf_loss

    def train_step_qvf(self, buffer):
        """
        Performs one training step for both Q-networks.
        """
        s_t, s_t_1, a_t, r_t, terminal_t, indices, weight = buffer.sample_priority()

        weight = np.expand_dims(weight, axis=1)
        reward = np.expand_dims(r_t, axis=1)
        terminal = np.expand_dims(terminal_t, axis=1)
        vf = tf.stop_gradient(self.target_vf_model(s_t_1))
        target_q = self.args.reward_scale * reward + self.args.gamma * (1 - terminal) * tf.stop_gradient(vf).numpy()

        with tf.GradientTape() as tape:
            qvf1 = self.current_qvf_model_1([s_t, a_t])
            raw_td_loss_1 = 0.5 * (qvf1 - target_q) ** 2
            td_loss_1 =  tf.reduce_mean(raw_td_loss_1 * weight)
            grads = tape.gradient(td_loss_1, self.current_qvf_model_1.trainable_weights)
        self.optimizer_qvf_1.apply_gradients(zip(grads, self.current_qvf_model_1.trainable_weights))

        with tf.GradientTape() as tape:
            qvf2 = self.current_qvf_model_2([s_t, a_t])
            raw_td_loss_2 = 0.5 * (qvf2 - target_q) ** 2
            td_loss_2 =  tf.reduce_mean(raw_td_loss_2 * weight)
            grads = tape.gradient(td_loss_2, self.current_qvf_model_2.trainable_weights)
        self.optimizer_qvf_2.apply_gradients(zip(grads, self.current_qvf_model_2.trainable_weights))

        updated_weights = (raw_td_loss_1 + raw_td_loss_2)/2
        buffer.update_priorities(indices, updated_weights)
        return td_loss_1 + td_loss_2

    def update_vf_network(self):
        """
        Soft update for value network
        """
        current_weights = self.current_vf_model.get_weights()
        target_weights = self.target_vf_model.get_weights()

        new_target_weights = []
        for i in range(len(current_weights)):
            new_weight = current_weights[i] * self.args.target_update_rate + target_weights[i] * (1 - self.args.target_update_rate)
            new_target_weights.append(new_weight)
        self.target_vf_model.set_weights(new_target_weights)

    def update_qvf_target_network(self):
        """
        Soft update for qvalue network
        """
        current_weights = self.current_qvf_model_1.get_weights()
        target_weights = self.target_qvf_model_1.get_weights()

        new_target_weights = []
        for i in range(len(current_weights)):
            new_weight = current_weights[i] * self.args.target_update_rate + target_weights[i] * (1 - self.args.target_update_rate)
            new_target_weights.append(new_weight)
        self.target_qvf_model_1.set_weights(new_target_weights)

        current_weights = self.current_qvf_model_2.get_weights()
        target_weights = self.target_qvf_model_2.get_weights()

        new_target_weights = []
        for i in range(len(current_weights)):
            new_weight = current_weights[i] * self.args.target_update_rate + target_weights[i] * (1 - self.args.target_update_rate)
            new_target_weights.append(new_weight)
        self.target_qvf_model_2.set_weights(new_target_weights)


    def build_value_network(self, name, state):
        init = tf.keras.initializers.GlorotNormal()
        l2 = tf.keras.layers.Dense(self.args.hidden,kernel_initializer=init,activation=tf.nn.relu, name=name + "_l2")(state)
        l3 = tf.keras.layers.Dense(self.args.hidden,kernel_initializer=init,activation=tf.nn.relu, name=name + "_l3")(l2)
        l4 = tf.keras.layers.Dense(self.args.hidden,kernel_initializer=init,activation=tf.nn.relu, name=name + "_l4")(l3)

        raw_output = tf.keras.layers.Dense(1, kernel_initializer=init,activation=None, name=name + "_out")(l4)
        return raw_output

    def build_q_network(self, name, state, action):
        init = tf.keras.initializers.GlorotNormal()
        l1 = tf.concat((state, action), axis=1)
        l2 = tf.keras.layers.Dense(self.args.hidden,kernel_initializer=init,activation=tf.nn.relu, name=name + "_l2")(l1)
        l3 = tf.keras.layers.Dense(self.args.hidden,kernel_initializer=init,activation=tf.nn.relu, name=name + "_l3")(l2)
        l4 = tf.keras.layers.Dense(self.args.hidden,kernel_initializer=init,activation=tf.nn.relu, name=name + "_l4")(l3)
        raw_output = tf.keras.layers.Dense(1, kernel_initializer=init,activation=None, name=name + "_out")(l4)
        return raw_output


    def save_model(self, frame_num):
        """
        Save all qvalue and value network weights on the disk specified by args.checkpoint directory
        """
        self.current_qvf_model_1.save_weights(self.args.checkpoint_dir + "/" + self.args.custom_id + "/critic_qvf_1_current_network_" + str(frame_num), overwrite=True)
        self.current_qvf_model_2.save_weights(self.args.checkpoint_dir + "/" + self.args.custom_id + "/critic_qvf_2_current_network_" + str(frame_num), overwrite=True)

        self.target_qvf_model_1.save_weights(self.args.checkpoint_dir + "/" + self.args.custom_id + "/critic_qvf_1_target_network_" + str(frame_num), overwrite=True)
        self.target_qvf_model_2.save_weights(self.args.checkpoint_dir + "/" + self.args.custom_id + "/critic_qvf_2_target_network_" + str(frame_num), overwrite=True)

        self.current_vf_model.save_weights(self.args.checkpoint_dir + "/" + self.args.custom_id + "/critic_vf_current_network_" + str(frame_num), overwrite=True)
        self.target_vf_model.save_weights(self.args.checkpoint_dir + "/" + self.args.custom_id + "/critic_vf_target_network_" + str(frame_num), overwrite=True)

    def load_model_qvf1(self, frame_num):
        self.current_qvf_model_1.load_weights(self.args.checkpoint_dir + "/" + self.args.custom_id + "/critic_qvf_1_current_network_" + str(frame_num))
