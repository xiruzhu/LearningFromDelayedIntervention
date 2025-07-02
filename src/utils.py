import random, time, json
from collections import deque
import numpy as np
import argparse
import tensorflow as tf
import imageio
# from skimage.transform import resize
from operator import itemgetter
# import d4rl
import numpy as np
import random
import pickle
import operator
from PIL import Image


class TrajectorySegmentBuffer:
    def __init__(self, args, size=5000):
        self.buffer = {}
        self.simple_buffer = {}

        self.expert_buffer = {}
        self.non_expert_buffer = {}

        self.args = args
        self.size = size
        self.count = 0
        self.expert_data_count = 0
        self.non_expert_data_count = 0
        self.add_index = 0

        self.weight_count = 0
        self.expert_count = 0
        self.non_label_count = 0
        self.intervention_count = 0

    def reinit(self, old_obj):
        self.buffer = old_obj.buffer
        self.args = old_obj.args
        self.size = old_obj.size
        # self.count = old_obj.count
        self.add_index = old_obj.add_index

        for key in self.buffer:
            trajectory = self.buffer[key]
            for i in range(len(trajectory)):
                s_t, a_t, r_t, terminal_t, intervention_status_t, s_t_1, label_t = trajectory[i]
                if label_t == -1:
                    self.expert_count += 1
                elif label_t == 0:
                    self.non_label_count += 1
                elif label_t == 1:
                    self.intervention_count += 1
                else:
                    print("Error, invalid label", label_t)
                    quit()

                self.simple_buffer[self.count] = (s_t, a_t, r_t, terminal_t, intervention_status_t, s_t_1, label_t)
                self.count += 1

                if label_t == -1:
                    # expert actions ...
                    self.expert_buffer[self.expert_data_count] = (
                    s_t, a_t, r_t, terminal_t, intervention_status_t, s_t_1, label_t)
                    self.expert_data_count += 1
                else:
                    self.non_expert_buffer[self.non_expert_data_count] = (
                    s_t, a_t, r_t, terminal_t, intervention_status_t, s_t_1, label_t)
                    self.non_expert_data_count += 1

    def add(self, s_t, a_t, r_t, terminal_t, intervention_status_t, s_t_1, label_t, trajectory_num):
        if not trajectory_num in self.buffer:
            self.buffer[trajectory_num] = []

        # print(s_t_1.shape, s_t_1)
        self.buffer[trajectory_num].append([s_t, a_t, r_t, terminal_t, intervention_status_t, s_t_1, label_t])
        self.simple_buffer[self.count] = (s_t, a_t, r_t, terminal_t, intervention_status_t, s_t_1, label_t)
        if label_t == -1:
            self.expert_buffer[self.expert_data_count] = [s_t, a_t, r_t, terminal_t, intervention_status_t, s_t_1,
                                                          label_t]
            self.expert_data_count += 1
        else:
            self.non_expert_buffer[self.non_expert_data_count] = [s_t, a_t, r_t, terminal_t, intervention_status_t,
                                                                  s_t_1, label_t]
            self.non_expert_data_count += 1

        self.count += 1

        if label_t == -1:
            self.expert_count += 1
        elif label_t == 0:
            self.non_label_count += 1
        elif label_t == 1:
            self.intervention_count += 1
        else:
            print("Error, invalid label", label_t)
            quit()

    def simple_sample(self, batch_size, mode=0):
        s_t_list = []
        s_t_1_list = []
        r_t_list = []
        terminal_t_list = []
        a_t_list = []
        intervention_status_t_list = []
        label_t_list = []

        if mode == 0:
            buffer = self.simple_buffer
            count = self.count
        elif mode == 1:
            buffer = self.expert_buffer
            count = self.expert_data_count
        elif mode == 2:
            buffer = self.non_expert_buffer
            count = self.non_expert_data_count

        sampled_indices = np.random.randint(0, count, size=(batch_size))
        for i in range(batch_size):
            s_t, a_t, r_t, terminal_t, intervention_status_t, s_t_1, label_t = buffer[sampled_indices[i]]
            s_t_list.append(s_t)
            s_t_1_list.append(s_t_1)

            a_t_list.append(a_t)
            terminal_t_list.append(terminal_t)
            intervention_status_t_list.append(intervention_status_t)
            r_t_list.append(r_t)

            label_t_list.append(label_t)

        s_t_list = np.array(s_t_list)
        s_t_1_list = np.array(s_t_1_list)
        a_t_list = np.array(a_t_list)
        terminal_t_list = np.array(terminal_t_list)
        r_t_list = np.array(r_t_list)
        label_t_list = np.array(label_t_list)
        intervention_status_t_list = np.array(intervention_status_t_list)
        return s_t_list, a_t_list, r_t_list, terminal_t_list, intervention_status_t_list, s_t_1_list, label_t_list

    def sample(self, batch_size, segment_length):
        sampled_trajectories = np.random.randint(0, len(self.buffer), size=batch_size)
        s_t_list = []
        s_t_1_list = []
        r_t_list = []
        terminal_t_list = []
        a_t_list = []
        label_t_list = []
        label_t_raw_list = []
        weight_t_list = []
        intervention_status_t_list = []

        for i in range(batch_size):
            trajectory = self.buffer[sampled_trajectories[i]]
            if len(trajectory) - segment_length < 0:
                trajectory = self.buffer[np.random.randint(0, len(self.buffer))]
                while len(trajectory) - segment_length < 0:
                    trajectory.append(trajectory[-1])  # Append the last state

                # for k in range(segment_length):
                #     print(k, trajectory[k][-1])
                # quit()

            if len(trajectory) - segment_length == 0:
                trajectory_index = 0
            else:
                trajectory_index = np.random.randint(0, len(trajectory) - segment_length)
            segment = trajectory[trajectory_index:min(trajectory_index + segment_length, len(trajectory))]

            weight = 0
            label_sum = 0
            for j in range(len(segment)):
                s_t, a_t, r_t, terminal_t, intervention_status_t, s_t_1, label_t = segment[j]
                s_t_list.append(s_t)
                s_t_1_list.append(s_t_1)
                a_t_list.append(a_t)
                label_t_raw_list.append(label_t)
                terminal_t_list.append(terminal_t)
                intervention_status_t_list.append(intervention_status_t)
                r_t_list.append(r_t)

                if label_t == -1:
                    weight += 1 - self.expert_count / self.count
                    label_sum += 0

                elif label_t == 0:
                    weight += 1 - self.non_label_count / self.count
                    label_sum += 0
                elif label_t == 1:
                    # print("was here")
                    weight += (1 - self.intervention_count / self.count)
                    label_sum += 1
                else:
                    print("Error, invalid label", label_t)
                    quit()

            # for j in range(segment_length - len(segment)):
            #     s_t, a_t, r_t, terminal_t, intervention_status_t, s_t_1, label_t = segment[-1]
            #     s_t_list.append(s_t)
            #     s_t_1_list.append(s_t_1)
            #     a_t_list.append(a_t)
            #     label_t_raw_list.append(label_t)
            #     terminal_t_list.append(terminal_t)
            #     intervention_status_t_list.append(intervention_status_t)
            #     r_t_list.append(r_t)
            #
            #     if label_t == -1:
            #         weight += 1 - self.expert_count/self.count
            #         label_sum += 0
            #
            #     elif label_t == 0:
            #         weight += 1 - self.non_label_count/self.count
            #         label_sum += 0
            #
            #     elif label_t == 1:
            #         # print("was here")
            #         weight += (1 - self.intervention_count/self.count)
            #         label_sum += 1
            #     else:
            #         print("Error, invalid label", label_t)
            #         quit()
            # if label_sum/segment_length > 0:
            #     print(weight/segment_length, label_sum/segment_length, self.intervention_count, self.non_label_count, self.expert_count, self.count)
            label_sum = np.clip(label_sum, 0, segment_length)
            label_t_list.append(label_sum / (segment_length))
            weight_t_list.append(weight / segment_length)

        s_t_list = np.reshape(s_t_list, [batch_size, segment_length, -1])
        s_t_1_list = np.reshape(s_t_1_list, [batch_size, segment_length, -1])
        a_t_list = np.reshape(a_t_list, [batch_size, segment_length, -1])
        terminal_t_list = np.reshape(terminal_t_list, [batch_size, segment_length, -1])
        r_t_list = np.reshape(r_t_list, [batch_size, segment_length, -1])
        label_t_raw_list = np.reshape(label_t_raw_list, [batch_size, segment_length, -1])

        label_t_list = np.array(label_t_list)

        intervention_status_t_list = np.reshape(intervention_status_t_list, [batch_size, segment_length, 1])
        weight_t_list = np.expand_dims(weight_t_list, axis=1)
        return s_t_list, a_t_list, r_t_list, terminal_t_list, intervention_status_t_list, s_t_1_list, label_t_list, weight_t_list, label_t_raw_list


class TrajectorySegmentBuffer_v2:
    def __init__(self, args, size=5000):
        self.buffer = {}
        self.simple_buffer = {}

        self.expert_buffer = {}
        self.non_expert_buffer = {}
        self.preintervention_buffer = {}

        self.args = args
        self.size = size
        self.count = 0
        self.expert_data_count = 0
        self.non_expert_data_count = 0
        self.preintervention_data_count = 0
        self.add_index = 0

        self.weight_count = 0
        self.expert_count = 0
        self.non_label_count = 0
        self.intervention_count = 0

    def get_existing_buffer_size(self):
        return len(self.buffer)

    def reinit(self, old_obj):
        self.buffer = old_obj.buffer
        self.intervention_trajectory_buffer = []
        self.expert_trajectory_buffer = []

        self.args = old_obj.args
        self.size = old_obj.size
        # self.count = old_obj.count
        self.add_index = old_obj.add_index

        for key in self.buffer:
            trajectory = self.buffer[key]
            s_t_list = []
            s_t_1_list = []
            r_t_list = []
            terminal_t_list = []
            a_t_list = []
            label_t_raw_list = []
            intervention_status_t_list = []

            no_preintervention_t_list = []

            for i in range(len(trajectory)):
                s_t, a_t, r_t, terminal_t, intervention_status_t, s_t_1, label_t = trajectory[i]

                s_t_list.append(s_t)
                s_t_1_list.append(s_t_1)
                a_t_list.append(a_t)
                label_t_raw_list.append(label_t)
                terminal_t_list.append(terminal_t)
                intervention_status_t_list.append(intervention_status_t)
                r_t_list.append(r_t)

                if label_t == -1:
                    self.expert_count += 1
                    no_preintervention_t_list.append(1)
                elif label_t == 0:
                    self.non_label_count += 1
                    no_preintervention_t_list.append(1)
                elif label_t == 1:
                    self.intervention_count += 1
                    no_preintervention_t_list.append(0)
                else:
                    print("Error, invalid label", label_t)
                    quit()

                self.simple_buffer[self.count] = (s_t, a_t, r_t, terminal_t, intervention_status_t, s_t_1, label_t)
                self.count += 1

                if label_t == -1:
                    # expert actions ...
                    self.expert_buffer[self.expert_data_count] = (
                        s_t, a_t, r_t, terminal_t, intervention_status_t, s_t_1, label_t)
                    self.expert_data_count += 1
                else:
                    self.non_expert_buffer[self.non_expert_data_count] = (
                        s_t, a_t, r_t, terminal_t, intervention_status_t, s_t_1, label_t)
                    self.non_expert_data_count += 1

                    if label_t == 1:
                        self.preintervention_buffer[self.preintervention_data_count] = (
                            s_t, a_t, r_t, terminal_t, intervention_status_t, s_t_1, label_t)
                        self.preintervention_data_count += 1

                if np.sum(label_t_raw_list[-self.args.segment_length:]) == self.args.segment_length:

                    data = (s_t_list[-self.args.segment_length:],
                            s_t_1_list[-self.args.segment_length:],
                            a_t_list[-self.args.segment_length:],
                            label_t_raw_list[-self.args.segment_length:],
                            terminal_t_list[-self.args.segment_length:],
                            intervention_status_t_list[-self.args.segment_length:],
                            r_t_list[-self.args.segment_length:])

                    self.intervention_trajectory_buffer.append(data)
                elif np.sum(no_preintervention_t_list[-self.args.segment_length:]) == self.args.segment_length:
                    data = (s_t_list[-self.args.segment_length:],
                            s_t_1_list[-self.args.segment_length:],
                            a_t_list[-self.args.segment_length:],
                            label_t_raw_list[-self.args.segment_length:],
                            terminal_t_list[-self.args.segment_length:],
                            intervention_status_t_list[-self.args.segment_length:],
                            r_t_list[-self.args.segment_length:])
                    self.expert_trajectory_buffer.append(data)
        print("Total interventions: ", len(self.intervention_trajectory_buffer), len(self.expert_trajectory_buffer))
        print("Dataset Size: ", self.count, self.expert_data_count, self.non_expert_data_count)

    def add(self, s_t, a_t, r_t, terminal_t, intervention_status_t, s_t_1, label_t, trajectory_num):
        if not trajectory_num in self.buffer:
            self.buffer[trajectory_num] = []

        # print(s_t_1.shape, s_t_1)
        self.buffer[trajectory_num].append([s_t, a_t, r_t, terminal_t, intervention_status_t, s_t_1, label_t])
        self.simple_buffer[self.count] = (s_t, a_t, r_t, terminal_t, intervention_status_t, s_t_1, label_t)
        if label_t == -1:
            self.expert_buffer[self.expert_data_count] = [s_t, a_t, r_t, terminal_t, intervention_status_t, s_t_1,
                                                          label_t]
            self.expert_data_count += 1
        else:
            self.non_expert_buffer[self.non_expert_data_count] = [s_t, a_t, r_t, terminal_t, intervention_status_t,
                                                                  s_t_1, label_t]
            self.non_expert_data_count += 1

            if label_t == 1:
                self.preintervention_buffer[self.preintervention_data_count] = (
                    s_t, a_t, r_t, terminal_t, intervention_status_t, s_t_1, label_t)
                self.preintervention_data_count += 1

        self.count += 1

        if label_t == -1:
            self.expert_count += 1
        elif label_t == 0:
            self.non_label_count += 1
        elif label_t == 1:
            self.intervention_count += 1
        else:
            print("Error, invalid label", label_t)
            quit()

    def simple_sample(self, batch_size, mode=0):
        s_t_list = []
        s_t_1_list = []
        r_t_list = []
        terminal_t_list = []
        a_t_list = []
        intervention_status_t_list = []
        label_t_list = []

        if mode == 0:
            buffer = self.simple_buffer
            count = self.count
        elif mode == 1:
            buffer = self.expert_buffer
            count = self.expert_data_count
        elif mode == 2:
            buffer = self.non_expert_buffer
            count = self.non_expert_data_count
        elif mode == 3:
            buffer = self.preintervention_buffer
            count = self.preintervention_data_count

        sampled_indices = np.random.randint(0, count, size=(batch_size))

        for i in range(batch_size):
            print("i", sampled_indices[i])
            s_t, a_t, r_t, terminal_t, intervention_status_t, s_t_1, label_t = buffer[sampled_indices[i]]
            s_t_list.append(s_t)
            s_t_1_list.append(s_t_1)

            a_t_list.append(a_t)
            terminal_t_list.append(terminal_t)
            intervention_status_t_list.append(intervention_status_t)
            r_t_list.append(r_t)
            label_t_list.append(label_t)

        s_t_list = np.array(s_t_list)
        s_t_1_list = np.array(s_t_1_list)
        a_t_list = np.array(a_t_list)
        terminal_t_list = np.array(terminal_t_list)
        r_t_list = np.array(r_t_list)
        label_t_list = np.array(label_t_list)
        intervention_status_t_list = np.array(intervention_status_t_list)
        return s_t_list, a_t_list, r_t_list, terminal_t_list, intervention_status_t_list, s_t_1_list, label_t_list

    def sample(self, batch_size, segment_length, mode=0):
        sampled_trajectories = np.random.randint(0, len(self.buffer), size=batch_size)
        s_t_list = []
        s_t_1_list = []
        r_t_list = []
        terminal_t_list = []
        a_t_list = []
        label_t_list = []
        label_t_raw_list = []
        weight_t_list = []
        intervention_status_t_list = []

        # sampled_trajectories = np.random.randint(0, len(self.buffer), size=batch_size * segment_length)
        # for i in range(batch_size * segment_length):
        #     trajectory = self.buffer[sampled_trajectories[i]]
        #     index = np.random.randint(0, len(trajectory))
        #
        #     # print(i, sampled_trajectories[i], index)
        #     s_t, a_t, r_t, terminal_t, intervention_status_t, s_t_1, label_t = trajectory[index]
        #     s_t_list.append(s_t)
        #     s_t_1_list.append(s_t_1)
        #     a_t_list.append(a_t)
        #     label_t_raw_list.append(label_t)
        #     terminal_t_list.append(terminal_t)
        #     intervention_status_t_list.append(intervention_status_t)
        #     r_t_list.append(r_t)
        #     label_t_list.append(0.5)
        #     weight_t_list.append(1)

        # sampled_indices = np.random.randint(0, self.count, size=(batch_size * segment_length))
        # for i in range(batch_size * segment_length):
        #     s_t, a_t, r_t, terminal_t, intervention_status_t, s_t_1, label_t = self.simple_buffer[sampled_indices[i]]
        #     s_t_list.append(s_t)
        #     s_t_1_list.append(s_t_1)
        #
        #     a_t_list.append(a_t)
        #     terminal_t_list.append(terminal_t)
        #     intervention_status_t_list.append(intervention_status_t)
        #     r_t_list.append(r_t)
        #
        #     label_t_raw_list.append(label_t)
        #
        #     label_t_list.append(0.5)
        #     weight_t_list.append(1)

        for i in range(batch_size):
            if mode == 0:
                trajectory = self.buffer[sampled_trajectories[i]]
                while len(trajectory) - segment_length < 0:
                    trajectory = self.buffer[np.random.randint(0, len(self.buffer))]
                    continue
                    # trajectory = self.buffer[np.random.randint(0, len(self.buffer))]
                    # while len(trajectory) - segment_length < 0:
                    #     trajectory.append(trajectory[-1])  # Append the last state

                    # for k in range(segment_length):
                    #     print(k, trajectory[k][-1])
                    # quit()

                if len(trajectory) - segment_length == 0:
                    trajectory_index = 0
                else:
                    trajectory_index = np.random.randint(0, len(trajectory) - segment_length)
                segment = trajectory[trajectory_index:(trajectory_index + segment_length)]

                weight = 0
                label_sum = 0
                for j in range(segment_length):
                    s_t, a_t, r_t, terminal_t, intervention_status_t, s_t_1, label_t = segment[j]
                    s_t_list.append(s_t)
                    s_t_1_list.append(s_t_1)
                    a_t_list.append(a_t)
                    label_t_raw_list.append(label_t)
                    terminal_t_list.append(terminal_t)
                    intervention_status_t_list.append(intervention_status_t)
                    r_t_list.append(r_t)

                    if label_t == -1:
                        weight += 1 - self.expert_count / self.count
                        label_sum += 0

                    elif label_t == 0:
                        weight += 1 - self.non_label_count / self.count
                        label_sum += 0
                    elif label_t == 1:
                        # print("was here")
                        weight += (1 - self.intervention_count / self.count)
                        label_sum += 1
                    else:
                        print("Error, invalid label", label_t)
                        quit()
                label_sum = np.clip(label_sum, 0, segment_length)
                label_t_list.append(label_sum / (segment_length))
                weight_t_list.append(weight / segment_length)
            elif mode == 1:
                segment_index = np.random.randint(0, len(self.intervention_trajectory_buffer))

                segment_s_t, segment_s_t_1, segment_a_t, segment_label_t_raw, segment_terminal_t, segment_intervention_status_t, segment_r_t = \
                self.intervention_trajectory_buffer[segment_index]
                s_t_list.extend(segment_s_t)
                s_t_1_list.extend(segment_s_t_1)
                a_t_list.extend(segment_a_t)
                label_t_raw_list.extend(segment_label_t_raw)
                terminal_t_list.extend(segment_terminal_t)
                intervention_status_t_list.extend(segment_intervention_status_t)
                r_t_list.extend(segment_r_t)

                label_t_list.append(1)
                weight_t_list.append(1 - self.intervention_count / self.count)
            elif mode == 2:
                segment_index = np.random.randint(0, len(self.expert_trajectory_buffer))

                segment_s_t, segment_s_t_1, segment_a_t, segment_label_t_raw, segment_terminal_t, segment_intervention_status_t, segment_r_t = \
                self.expert_trajectory_buffer[segment_index]
                s_t_list.extend(segment_s_t)
                s_t_1_list.extend(segment_s_t_1)
                a_t_list.extend(segment_a_t)
                label_t_raw_list.extend(segment_label_t_raw)
                terminal_t_list.extend(segment_terminal_t)
                intervention_status_t_list.extend(segment_intervention_status_t)
                r_t_list.extend(segment_r_t)

                label_t_list.append(1)
                weight_t_list.append(1 - self.intervention_count / self.count)

        s_t_list = np.reshape(s_t_list, [batch_size, segment_length, -1])
        s_t_1_list = np.reshape(s_t_1_list, [batch_size, segment_length, -1])
        a_t_list = np.reshape(a_t_list, [batch_size, segment_length, -1])
        terminal_t_list = np.reshape(terminal_t_list, [batch_size, segment_length, -1])
        r_t_list = np.reshape(r_t_list, [batch_size, segment_length, -1])
        label_t_raw_list = np.reshape(label_t_raw_list, [batch_size, segment_length, -1])
        label_t_list = np.array(label_t_list)
        intervention_status_t_list = np.reshape(intervention_status_t_list, [batch_size, segment_length, 1])
        weight_t_list = np.expand_dims(weight_t_list, axis=1)
        return s_t_list, a_t_list, r_t_list, terminal_t_list, intervention_status_t_list, s_t_1_list, label_t_list, weight_t_list, label_t_raw_list


class TrajectorySegmentBuffer_v3:
    def __init__(self, args, size=5000):
        self.args = args
        self.raw_data_buffer = {}
        self.raw_segment_count = 0
        self.raw_data_count = 0

        self.combined_single_state_buffer = {}
        self.combined_single_state_count = 0

        self.expert_single_state_buffer = {}
        self.expert_single_state_count = 0

        self.preintervention_single_state_buffer = {}
        self.preintervention_single_state_count = 0

        self.non_expert_single_state_buffer = {}
        self.non_expert_single_state_count = 0

        self.combined_segment_buffer = {}
        self.combined_segment_count = 0

        self.expert_segment_buffer = {}
        self.expert_segment_count = 0

        self.non_expert_segment_buffer = {}
        self.non_expert_segment_count = 0

        self.non_preintervention_segment_buffer = {}
        self.non_preintervention_segment_count = 0

        self.non_expert_plus_segment_buffer = {}
        self.non_expert_plus_segment_count = 0

    def reinit(self, old_buffer):
        self.raw_data_buffer = old_buffer.raw_data_buffer
        self.raw_data_count = old_buffer.raw_data_count

        self.combined_single_state_buffer = old_buffer.combined_single_state_buffer
        self.combined_single_state_count = old_buffer.combined_single_state_count

        self.expert_single_state_buffer = old_buffer.expert_single_state_buffer
        self.expert_single_state_count = old_buffer.expert_single_state_count

        self.preintervention_single_state_buffer = old_buffer.preintervention_single_state_buffer
        self.preintervention_single_state_count = old_buffer.preintervention_single_state_count

        self.non_expert_single_state_buffer = old_buffer.non_expert_single_state_buffer
        self.non_expert_single_state_count = old_buffer.non_expert_single_state_count

        self.raw_segment_count = old_buffer.raw_segment_count
        self.combined_segment_buffer = old_buffer.combined_segment_buffer
        self.combined_segment_count = old_buffer.combined_segment_count

        self.expert_segment_buffer = old_buffer.expert_segment_buffer
        self.expert_segment_count = old_buffer.expert_segment_count

        self.non_expert_segment_buffer = old_buffer.non_expert_segment_buffer
        self.non_expert_segment_count = old_buffer.non_expert_segment_count

        self.preintervention_segment_buffer = old_buffer.preintervention_segment_buffer
        self.preintervention_segment_count = old_buffer.preintervention_segment_count

        self.non_preintervention_segment_buffer = {}
        self.non_preintervention_segment_count = 0

        self.non_expert_plus_segment_buffer = {}
        self.non_expert_plus_segment_count = 0

        # print(old_buffer.combined_segment_count, len(old_buffer.combined_segment_buffer))
        # print(old_buffer.expert_segment_count, len(old_buffer.expert_segment_buffer))
        # print(old_buffer.non_expert_segment_count, len(old_buffer.non_expert_segment_buffer))
        # print(old_buffer.preintervention_segment_count, len(old_buffer.preintervention_segment_buffer))
        # print(old_buffer.non_preintervention_segment_count, len(old_buffer.non_preintervention_segment_buffer))
        for i in range(old_buffer.combined_segment_count):

            segment_label_expert = 0
            segment_label_preintervention = 0

            for j in range(self.args.segment_length):
                trajectory_index, current_index = old_buffer.combined_segment_buffer[i][j]

                s_t, a_t, r_t, terminal_t, intervention_status_t, s_t_1, label_t = \
                self.raw_data_buffer[trajectory_index][current_index]
                if label_t == -1:
                    segment_label_expert += 1
                if label_t == 1:
                    segment_label_preintervention += 1

            if segment_label_preintervention == 0:
                self.non_preintervention_segment_buffer[self.non_preintervention_segment_count] = \
                old_buffer.combined_segment_buffer[i]
                self.non_preintervention_segment_count += 1

            if segment_label_preintervention == self.args.segment_length:
                self.non_expert_plus_segment_buffer[self.non_expert_plus_segment_count] = \
                old_buffer.combined_segment_buffer[i]
                self.non_expert_plus_segment_count += 1
        #
        # print(self.combined_segment_count, len(self.combined_segment_buffer))
        # print(self.expert_segment_count, len(self.expert_segment_buffer))
        # print(self.non_expert_segment_count, len(self.non_expert_segment_buffer))
        # print(self.preintervention_segment_count, len(self.preintervention_segment_buffer))
        # print(self.non_preintervention_segment_count, len(self.non_preintervention_segment_buffer))
        # print(self.non_expert_plus_segment_count, len(self.non_expert_plus_segment_buffer))

    def add(self, s_t, a_t, r_t, terminal_t, intervention_status_t, s_t_1, label_t):
        if not self.raw_segment_count in self.raw_data_buffer:
            self.raw_data_buffer[self.raw_segment_count] = {}

        data_entry = [s_t, a_t, r_t, terminal_t, intervention_status_t, s_t_1, label_t]
        # print(s_t_1.shape, s_t_1)
        current_trajectory_count = len(self.raw_data_buffer[self.raw_segment_count])
        self.raw_data_buffer[self.raw_segment_count][current_trajectory_count] = data_entry

        self.combined_single_state_buffer[self.combined_single_state_count] = [self.raw_segment_count,
                                                                               current_trajectory_count]
        self.combined_single_state_count += 1
        if label_t == -1 or intervention_status_t == 1:
            self.expert_single_state_buffer[self.expert_single_state_count] = [self.raw_segment_count,
                                                                               current_trajectory_count]
            self.expert_single_state_count += 1
        if label_t == 1:
            self.preintervention_single_state_buffer[self.preintervention_single_state_count] = [self.raw_segment_count,
                                                                                                 current_trajectory_count]
            self.preintervention_single_state_count += 1
        if label_t == 1 or label_t == 0:
            self.non_expert_single_state_buffer[self.non_expert_single_state_count] = [self.raw_segment_count,
                                                                                       current_trajectory_count]
            self.non_expert_single_state_count += 1
        self.raw_data_count += 1

    def add_segment(self, current_trajectory, current_trajectory_index, current_index):
        segment_label_expert = 0
        segment_label_preintervention = 0
        segment_indices = []

        for j in range(self.args.segment_length):
            if current_index + j >= len(current_trajectory):
                # check if last state is terminal ...
                added_index = len(current_trajectory) - 1
            else:
                added_index = current_index + j

            s_t, a_t, r_t, terminal_t, intervention_status_t, s_t_1, label_t = current_trajectory[added_index]

            if label_t == -1 or intervention_status_t == 1:
                segment_label_expert += 1
            if label_t == 1:
                segment_label_preintervention += 1

            segment_indices.append([current_trajectory_index, added_index])

        self.combined_segment_buffer[self.combined_segment_count] = segment_indices
        self.combined_segment_count += 1

        if segment_label_expert == self.args.segment_length:
            self.expert_segment_buffer[self.expert_segment_count] = segment_indices
            self.expert_segment_count += 1

        if segment_label_preintervention == self.args.segment_length:
            self.preintervention_segment_buffer[self.preintervention_segment_count] = segment_indices
            self.preintervention_segment_count += 1

        if segment_label_expert == 0:
            self.non_expert_segment_buffer[self.non_expert_segment_count] = segment_indices
            self.non_expert_segment_count += 1

        if segment_label_preintervention == 0:
            self.non_preintervention_segment_buffer[self.non_preintervention_segment_count] = segment_indices
            self.non_preintervention_segment_count += 1

        if segment_label_expert == 0 and segment_label_preintervention == self.args.segment_length:
            self.non_expert_plus_segment_buffer[self.non_expert_plus_segment_count] = segment_indices
            self.non_expert_plus_segment_count += 1

        # if segment_label_preintervention == 0:
        #     self.non_preintervention_segment_buffer[self.non_preintervention_segment_count] = segment_indices
        #     self.non_preintervention_segment_count += 1
        #
        # if segment_label_expert == 0 and segment_label_preintervention < self.args.segment_length:
        #     self.non_full_expert_full_preintervention_buffer[self.non_full_expert_full_preintervention_count] = segment_indices
        #     self.non_full_expert_full_preintervention_count += 1
        #
        #
        # if segment_label_expert == 0 and segment_label_preintervention == 0:
        #     self.no_label_buffer[self.no_label_count] = segment_indices
        #     self.no_label_count += 1

    def trajectory_end(self):
        current_trajectory = self.raw_data_buffer[self.raw_segment_count]
        for i in range(len(current_trajectory)):
            if i + self.args.segment_length >= len(current_trajectory):
                # check if it's a terminal, otherwise stop
                s_t, a_t, r_t, terminal_t, intervention_status_t, s_t_1, label_t = current_trajectory[
                    len(current_trajectory) - 1]
                if terminal_t == False:
                    break
            self.add_segment(current_trajectory, self.raw_segment_count, i)
        self.raw_segment_count += 1

        # print("-------------------------------------------")
        # print(self.combined_single_state_count, len(self.combined_single_state_buffer))
        # print(self.expert_single_state_count, len(self.expert_single_state_buffer))
        # print(self.non_expert_single_state_count, len(self.non_expert_single_state_buffer))
        # print(self.preintervention_single_state_count, len(self.preintervention_single_state_buffer))
        #
        # print("+++++++++++++++++++++++++++++++++++++++++++")
        #
        # print(self.combined_segment_count, len(self.combined_segment_buffer))
        # print(self.expert_segment_count, len(self.expert_segment_buffer))
        # print(self.non_expert_segment_count, len(self.non_expert_segment_buffer))
        # print(self.preintervention_segment_count, len(self.preintervention_segment_buffer))
        # print(self.non_expert_plus_segment_count, len(self.non_expert_plus_segment_buffer))
        # print("-------------------------------------------")

    def simple_sample(self, batch_size, mode=0):
        s_t_list = []
        s_t_1_list = []
        r_t_list = []
        terminal_t_list = []
        a_t_list = []
        intervention_status_t_list = []
        label_t_list = []

        if mode == 0:
            buffer = self.combined_single_state_buffer
            count = self.combined_single_state_count
        elif mode == 1:
            buffer = self.expert_single_state_buffer
            count = self.expert_single_state_count
        elif mode == 2:
            buffer = self.non_expert_single_state_buffer
            count = self.non_expert_single_state_count
        elif mode == 3:
            buffer = self.preintervention_single_state_buffer
            count = self.preintervention_single_state_count

        sampled_indices = np.random.randint(0, count, size=(batch_size))
        for i in range(batch_size):
            raw_buffer_indices = buffer[sampled_indices[i]]

            s_t, a_t, r_t, terminal_t, intervention_status_t, s_t_1, label_t = \
            self.raw_data_buffer[raw_buffer_indices[0]][raw_buffer_indices[1]]
            s_t_list.append(s_t)
            s_t_1_list.append(s_t_1)

            a_t_list.append(a_t)
            terminal_t_list.append(terminal_t)
            intervention_status_t_list.append(intervention_status_t)
            r_t_list.append(r_t)
            label_t_list.append(label_t)

        s_t_list = np.array(s_t_list)
        s_t_1_list = np.array(s_t_1_list)
        a_t_list = np.array(a_t_list)
        terminal_t_list = np.array(terminal_t_list)
        r_t_list = np.array(r_t_list)
        label_t_list = np.array(label_t_list)
        intervention_status_t_list = np.array(intervention_status_t_list)
        return s_t_list, a_t_list, r_t_list, terminal_t_list, intervention_status_t_list, s_t_1_list, label_t_list

    def sample(self, batch_size, mode=0):
        segment_s_t_list = []
        segment_s_t_1_list = []
        segment_r_t_list = []
        segment_terminal_t_list = []
        segment_a_t_list = []
        segment_intervention_status_t_list = []
        segment_label_t_list = []
        segment_mean_label_list = []

        if mode == 0:
            buffer = self.combined_segment_buffer
            count = self.combined_segment_count
        elif mode == 1:
            buffer = self.expert_segment_buffer
            count = self.expert_segment_count
        elif mode == 2:
            buffer = self.non_expert_segment_buffer
            count = self.non_expert_segment_count
        elif mode == 3:
            buffer = self.preintervention_segment_buffer
            count = self.preintervention_segment_count
        elif mode == 4:
            buffer = self.non_preintervention_segment_buffer
            count = self.non_preintervention_segment_count
        elif mode == 5:
            buffer = self.non_expert_plus_segment_buffer
            count = self.non_expert_plus_segment_count

        sampled_indices = np.random.randint(0, count, size=(batch_size))

        for i in range(batch_size):
            raw_buffer_segment_indices = buffer[sampled_indices[i]]
            mean_label = 0
            for j in range(self.args.segment_length):
                # if j == 0:
                #     print("start", raw_buffer_segment_indices[j])
                # else:
                #     print(raw_buffer_segment_indices[j])

                # print("Sampling ... ", i, raw_buffer_segment_indices[j])

                s_t, a_t, r_t, terminal_t, intervention_status_t, s_t_1, label_t = \
                self.raw_data_buffer[raw_buffer_segment_indices[j][0]][raw_buffer_segment_indices[j][1]]

                segment_s_t_list.append(s_t)
                segment_s_t_1_list.append(s_t_1)
                segment_a_t_list.append(a_t)
                segment_terminal_t_list.append(terminal_t)
                segment_r_t_list.append(r_t)
                segment_label_t_list.append(label_t)
                segment_intervention_status_t_list.append(intervention_status_t)
                mean_label += (label_t + 1) / 2
            mean_label = mean_label / self.args.segment_length
            segment_mean_label_list.append(mean_label)

        s_t_list = np.reshape(segment_s_t_list, [batch_size, self.args.segment_length, -1])
        s_t_1_list = np.reshape(segment_s_t_1_list, [batch_size, self.args.segment_length, -1])
        a_t_list = np.reshape(segment_a_t_list, [batch_size, self.args.segment_length, -1])
        terminal_t_list = np.reshape(segment_terminal_t_list, [batch_size, self.args.segment_length, -1])
        r_t_list = np.reshape(segment_r_t_list, [batch_size, self.args.segment_length, -1])
        label_t_raw_list = np.reshape(segment_label_t_list, [batch_size, self.args.segment_length, -1])
        label_t_list = np.array(segment_mean_label_list)
        intervention_status_t_list = np.reshape(segment_intervention_status_t_list,
                                                [batch_size, self.args.segment_length, 1])
        return s_t_list, a_t_list, r_t_list, terminal_t_list, intervention_status_t_list, s_t_1_list, label_t_list, label_t_raw_list



class TrajectorySegmentBuffer_v5:
    def __init__(self, args, size=5000):
        self.args = args
        self.raw_data_buffer = {}
        self.raw_segment_count = 0
        self.raw_data_count = 0

        self.combined_single_state_buffer = {}
        self.combined_single_state_count = 0

        self.expert_single_state_buffer = {}
        self.expert_single_state_count = 0

        self.preintervention_single_state_buffer = {}
        self.preintervention_single_state_count = 0

        self.non_expert_non_preintervention_single_state_buffer = {}
        self.non_expert_non_preintervention_single_state_count = 0

        self.non_expert_single_state_buffer = {}
        self.non_expert_single_state_count = 0

        self.combined_segment_buffer = {}
        self.combined_segment_count = 0

        self.expert_segment_buffer = {}
        self.expert_segment_count = 0

        self.preintervention_segment_buffer = {}
        self.preintervention_segment_count = 0

        self.inbetween_segment_buffer = {}
        self.inbetween_segment_count = 0

        self.special_preint_buffer = {}
        self.special_preint_count = {}

        self.special_inbetween_buffer = {}
        self.special_inbetween_count = {}

    def reinit(self, old_buffer, error_function, threshold_list=None, debug_mode=False):
        self.raw_data_buffer = old_buffer.raw_data_buffer.copy()
        self.raw_data_count = old_buffer.raw_data_count
        self.raw_segment_count = old_buffer.raw_segment_count

        # print(len(self.raw_data_buffer))
        # for i in range(self.raw_segment_count):
        #     print(i, len(self.raw_data_buffer[i]))

        self.combined_single_state_buffer = old_buffer.combined_single_state_buffer.copy()
        self.combined_single_state_count = old_buffer.combined_single_state_count

        self.expert_single_state_buffer = old_buffer.expert_single_state_buffer.copy()
        self.expert_single_state_count = old_buffer.expert_single_state_count

        # self.preintervention_single_state_buffer = old_buffer.preintervention_single_state_buffer
        # self.preintervention_single_state_count = old_buffer.preintervention_single_state_count
        #
        # self.non_expert_non_preintervention_single_state_buffer = old_buffer.non_expert_non_preintervention_single_state_buffer
        # self.non_expert_non_preintervention_single_state_count = old_buffer.non_expert_non_preintervention_single_state_count
        #
        # self.non_expert_single_state_buffer = old_buffer.non_expert_single_state_buffer
        # self.non_expert_single_state_count = old_buffer.non_expert_single_state_count
        #
        #
        self.combined_segment_buffer = old_buffer.combined_segment_buffer.copy()
        self.combined_segment_count = old_buffer.combined_segment_count

        self.expert_segment_buffer = old_buffer.expert_segment_buffer.copy()
        self.expert_segment_count = old_buffer.expert_segment_count

        # self.no_intervention_segment_buffer = old_buffer.no_intervention_segment_buffer
        # self.no_intervention_segment_count = old_buffer.no_intervention_segment_count

        # self.preintervention_segment_buffer = old_buffer.preintervention_segment_buffer
        # self.preintervention_segment_count = old_buffer.preintervention_segment_countz
        #
        # self.inbetween_segment_buffer = old_buffer.inbetween_segment_buffer
        # self.inbetween_segment_count = old_buffer.inbetween_segment_count
        #
        # self.special_preint_buffer = old_buffer.special_preint_buffer
        # self.special_preint_count = old_buffer.special_preint_count

    def generate_segment_weights(self, max_threshold):
        self.raw_data_weight_no_expert = {}
        self.raw_data_weight_all = {}
        # initialize weight buffer
        for traj_idx in self.raw_data_buffer:
            self.raw_data_weight_no_expert[traj_idx] = {}
            self.raw_data_weight_all[traj_idx] = {}
            for state_idx in self.raw_data_buffer[traj_idx]:
                self.raw_data_weight_no_expert[traj_idx][state_idx] = 0
                self.raw_data_weight_all[traj_idx][state_idx] = 0

        for idx in self.expert_segment_buffer:
            segment = self.expert_segment_buffer[idx]
            # print(len(segment))
            # quit()

            for idx_pair in segment:
                self.raw_data_weight_all[idx_pair[0]][idx_pair[1]] += 1

        for idx in self.preintervention_segment_buffer:
            segment = self.preintervention_segment_buffer[idx]
            for idx_pair in segment:
                # print(idx_pair, len(self.raw_data_weight_no_expert[idx_pair[0]]))
                self.raw_data_weight_no_expert[idx_pair[0]][idx_pair[1]] += 1
                self.raw_data_weight_all[idx_pair[0]][idx_pair[1]] += 1

        for idx in self.inbetween_segment_buffer:
            segment = self.inbetween_segment_buffer[idx]
            for idx_pair in segment:
                self.raw_data_weight_no_expert[idx_pair[0]][idx_pair[1]] += 1
                self.raw_data_weight_all[idx_pair[0]][idx_pair[1]] += 1

        for traj_idx in self.raw_data_weight_all:
            for segment_idx in self.raw_data_weight_all[traj_idx]:
                if self.raw_data_weight_all[traj_idx][segment_idx] > self.args.segment_length:
                    print("Error! Oversampled Trajectory with Expert!", traj_idx, segment_idx,
                          self.raw_data_weight_all[traj_idx][segment_idx])
                    quit()
                self.raw_data_weight_all[traj_idx][segment_idx] = 1 - self.raw_data_weight_all[traj_idx][
                    segment_idx] / self.args.segment_length

        for traj_idx in self.raw_data_weight_no_expert:
            for segment_idx in self.raw_data_weight_no_expert[traj_idx]:
                if self.raw_data_weight_no_expert[traj_idx][segment_idx] > self.args.segment_length:
                    print("Error! Oversampled Trajectory no Expert!", traj_idx, segment_idx,
                          self.raw_data_weight_no_expert[traj_idx][segment_idx])
                    quit()
                self.raw_data_weight_no_expert[traj_idx][segment_idx] = 1 - self.raw_data_weight_no_expert[traj_idx][
                    segment_idx] / self.args.segment_length

        self.preintervention_segment_special_buffer = {}
        self.preintervention_segment_special_count = 0

        for i in range(self.preintervention_segment_count):
            supervision_threshold_list = []
            expert_steps = 0
            for j in range(self.args.segment_length):
                trajectory_index, current_index = self.preintervention_segment_buffer[i][j]
                s_t, a_t, r_t, terminal_t, intervention_status_t, s_t_1, label_t, _, supervision_threshold_t = \
                self.raw_data_buffer[trajectory_index][current_index]
                supervision_threshold_list.append(supervision_threshold_t)
                if intervention_status_t == 1:
                    expert_steps += 1

                data = (s_t, a_t, r_t, terminal_t, intervention_status_t, s_t_1, 1, _, supervision_threshold_t)
                self.raw_data_buffer[trajectory_index][current_index] = data

            if np.max(supervision_threshold_list) < max_threshold:
                self.preintervention_segment_special_buffer[self.preintervention_segment_special_count] = \
                self.preintervention_segment_buffer[i]
                self.preintervention_segment_special_count += 1

        self.non_expert_segment_special_buffer = {}
        self.non_expert_segment_special_count = 0
        for i in range(self.preintervention_segment_special_count):
            self.non_expert_segment_special_buffer[self.non_expert_segment_special_count] = \
            self.preintervention_segment_special_buffer[i]
            self.non_expert_segment_special_count += 1

        for i in range(self.inbetween_segment_count):
            self.non_expert_segment_special_buffer[self.non_expert_segment_special_count] = \
            self.inbetween_segment_buffer[i]
            self.non_expert_segment_special_count += 1
        # print(self.non_expert_segment_special_count, self.preintervention_segment_special_count, self.inbetween_segment_count)
        # quit()
        #

        self.inbetween_special_buffer = {}
        self.inbetween_special_count = 0
        for i in range(self.inbetween_segment_count):

            preint_count = 0
            for j in range(self.args.segment_length):
                trajectory_index, current_index = self.inbetween_segment_buffer[i][j]
                s_t, a_t, r_t, terminal_t, intervention_status_t, s_t_1, label_t, _, supervision_threshold_t = \
                    self.raw_data_buffer[trajectory_index][current_index]

                if label_t == 1:
                    preint_count += 1

            if preint_count == 0:
                self.inbetween_special_buffer[self.inbetween_special_count] = self.inbetween_segment_buffer[i]
                self.inbetween_special_count += 1

        self.inbetween_special_segment_buffer = {}
        self.inbetween_special_segment_count = {}
        for i in range(self.inbetween_segment_count):
            trajectory = self.inbetween_segment_buffer[i]
            supervision_threshold_list = []
            for j in range(self.args.segment_length):
                raw_buffer_indices = trajectory[j]
                s_t, a_t, r_t, terminal_t, intervention_status_t, s_t_1, label_t, expert_a_t, supervision_level_t = \
                self.raw_data_buffer[raw_buffer_indices[0]][raw_buffer_indices[1]]
                supervision_threshold_list.append(supervision_level_t)
            supervision_threshold = np.max(supervision_threshold_list)

            if not supervision_threshold in self.inbetween_special_segment_count:
                self.inbetween_special_segment_buffer[supervision_threshold] = {}
                self.inbetween_special_segment_count[supervision_threshold] = 0
            self.inbetween_special_segment_buffer[supervision_threshold][
                self.inbetween_special_segment_count[supervision_threshold]] = trajectory
            self.inbetween_special_segment_count[supervision_threshold] += 1

        # for key in self.inbetween_special_segment_count:
        #     print(key, self.inbetween_special_segment_count[key])
        # for key in self.special_preint_buffer:
        #     print(key, self.special_preint_count[key])
        # quit()

    def get_segment(self, raw_buffer_segment_indices):
        segment_s_t = []
        segment_s_t_1 = []
        segment_a_t = []
        segment_terminal_t = []
        segment_r_t = []
        segment_intervention_status_t = []
        segment_label_t = []
        segment_expert_a_t = []
        intervention_threshold_list = []

        mean_label = 0
        segment_weight = 0
        for j in range(self.args.segment_length):
            s_t, a_t, r_t, terminal_t, intervention_status_t, s_t_1, label_t, expert_a_t, intervention_threshold = \
                self.raw_data_buffer[raw_buffer_segment_indices[j][0]][raw_buffer_segment_indices[j][1]]

            segment_s_t.append(s_t)
            segment_s_t_1.append(s_t_1)
            segment_a_t.append(a_t)
            segment_terminal_t.append(terminal_t)
            segment_r_t.append(r_t)
            segment_label_t.append(label_t)
            segment_intervention_status_t.append(intervention_status_t)
            mean_label += (label_t + 1) / 2
            segment_expert_a_t.append(expert_a_t)
            intervention_threshold_list.append(intervention_threshold)
            segment_weight += self.raw_data_weight_no_expert[raw_buffer_segment_indices[j][0]][
                raw_buffer_segment_indices[j][1]]
        return np.expand_dims(np.array(segment_s_t), axis=0), np.expand_dims(np.array(segment_s_t_1),
                                                                             axis=0), np.expand_dims(
            np.array(segment_a_t), axis=0), \
               np.expand_dims(np.array(segment_terminal_t), axis=0), \
               np.expand_dims(np.array(segment_r_t), axis=0), np.expand_dims(np.array(segment_label_t), axis=0), \
               np.expand_dims(np.array(segment_intervention_status_t), axis=0), np.expand_dims(
            np.array(segment_expert_a_t), axis=0), np.expand_dims(np.array(intervention_threshold_list), axis=0)

    def sample_group_preint(self, initial_supervision_threshold_list, group_is_preint_list, sample_per_group=4,
                            all_legal=False):
        batch_segment_s_t = []
        batch_segment_s_t_1 = []
        batch_segment_a_t = []
        batch_segment_terminal_t = []
        batch_segment_r_t = []
        batch_segment_intervention_status_t = []
        batch_segment_label_t = []
        batch_segment_expert_a_t = []
        batch_intervention_threshold_list = []
        group_supervision_threshold_list = []

        data_buffer = self.special_preint_buffer
        data_count = self.special_preint_count

        data_index_list = list(self.inbetween_special_segment_buffer.keys())

        for group_num in range(initial_supervision_threshold_list.shape[0]):
            group_supervision_threshold = initial_supervision_threshold_list[group_num]
            potential_threshold_list = []
            if group_is_preint_list[group_num]:
                for threshold in data_index_list:
                    if group_supervision_threshold < threshold:
                        potential_threshold_list.append(threshold)
            else:
                for threshold in data_index_list:
                    if group_supervision_threshold <= threshold:
                        potential_threshold_list.append(threshold)

            if all_legal:
                sampled_threshold_index = potential_threshold_list[np.random.randint(0, len(potential_threshold_list))]
            else:
                sampled_threshold_index = min(potential_threshold_list)

            group_supervision_threshold_list.append(sampled_threshold_index)

            # idx = data_index_list[sampled_threshold_index]
            sampled_data_buffer = data_buffer[sampled_threshold_index]
            sampled_data_count = data_count[sampled_threshold_index]

            group_segment_s_t = []
            group_segment_s_t_1 = []
            group_segment_a_t = []
            group_segment_terminal_t = []
            group_segment_r_t = []
            group_segment_intervention_status_t = []
            group_segment_label_t = []
            group_segment_expert_a_t = []
            group_intervention_threshold_list = []

            sampled_group_index = np.random.choice(sampled_data_count, size=(sample_per_group))
            for index in range(sample_per_group):
                raw_buffer_segment_indices = sampled_data_buffer[sampled_group_index[index]]

                segment_s_t, segment_s_t_1, segment_a_t, segment_terminal_t, segment_r_t, segment_label_t, \
                segment_intervention_status_t, segment_expert_a_t, segment_intervention_threshold = self.get_segment(
                    raw_buffer_segment_indices)

                group_segment_s_t.append(segment_s_t)
                group_segment_s_t_1.append(segment_s_t_1)
                group_segment_a_t.append(segment_a_t)
                group_segment_terminal_t.append(segment_terminal_t)
                group_segment_r_t.append(segment_r_t)
                group_segment_label_t.append(segment_label_t)
                group_segment_intervention_status_t.append(segment_intervention_status_t)
                group_segment_expert_a_t.append(segment_expert_a_t)
                group_intervention_threshold_list.append(segment_intervention_threshold)

            group_segment_s_t = np.expand_dims(np.concatenate(group_segment_s_t, axis=0), axis=0)
            group_segment_s_t_1 = np.expand_dims(np.concatenate(group_segment_s_t_1, axis=0), axis=0)
            group_segment_a_t = np.expand_dims(np.concatenate(group_segment_a_t, axis=0), axis=0)
            group_segment_terminal_t = np.expand_dims(np.concatenate(group_segment_terminal_t, axis=0), axis=0)
            group_segment_r_t = np.expand_dims(np.concatenate(group_segment_r_t, axis=0), axis=0)
            group_segment_label_t = np.expand_dims(np.concatenate(group_segment_label_t, axis=0), axis=0)
            group_segment_intervention_status_t = np.expand_dims(
                np.concatenate(group_segment_intervention_status_t, axis=0), axis=0)
            group_segment_expert_a_t = np.expand_dims(np.concatenate(group_segment_expert_a_t, axis=0), axis=0)
            group_intervention_threshold_list = np.expand_dims(
                np.concatenate(group_intervention_threshold_list, axis=0), axis=0)

            batch_segment_s_t.append(group_segment_s_t)
            batch_segment_s_t_1.append(group_segment_s_t_1)
            batch_segment_a_t.append(group_segment_a_t)
            batch_segment_terminal_t.append(group_segment_terminal_t)
            batch_segment_r_t.append(group_segment_r_t)
            batch_segment_intervention_status_t.append(group_segment_intervention_status_t)
            batch_segment_label_t.append(group_segment_label_t)
            batch_segment_expert_a_t.append(group_segment_expert_a_t)
            batch_intervention_threshold_list.append(group_intervention_threshold_list)

        batch_segment_s_t = np.concatenate(batch_segment_s_t, axis=0)
        batch_segment_s_t_1 = np.concatenate(batch_segment_s_t_1, axis=0)
        batch_segment_a_t = np.concatenate(batch_segment_a_t, axis=0)
        batch_segment_terminal_t = np.concatenate(batch_segment_terminal_t, axis=0)
        batch_segment_r_t = np.concatenate(batch_segment_r_t, axis=0)
        batch_segment_intervention_status_t = np.concatenate(batch_segment_intervention_status_t, axis=0)
        batch_segment_label_t = np.concatenate(batch_segment_label_t, axis=0)
        batch_segment_expert_a_t = np.concatenate(batch_segment_expert_a_t, axis=0)
        batch_intervention_threshold_list = np.concatenate(batch_intervention_threshold_list, axis=0)
        group_supervision_threshold_list = np.array(group_supervision_threshold_list)

        return batch_segment_s_t, batch_segment_s_t_1, batch_segment_a_t, batch_segment_terminal_t, batch_segment_r_t, batch_segment_intervention_status_t, \
               batch_segment_label_t, batch_segment_expert_a_t, batch_intervention_threshold_list, group_supervision_threshold_list

    def sample_group_combined(self, mode_list, mode_batch_size_list, sample_per_group=4):

        batch_segment_s_t = []
        batch_segment_s_t_1 = []
        batch_segment_a_t = []
        batch_segment_terminal_t = []
        batch_segment_r_t = []
        batch_segment_intervention_status_t = []
        batch_segment_label_t = []
        batch_segment_expert_a_t = []
        batch_intervention_threshold_list = []
        group_supervision_threshold_list = []
        group_is_preint_list = []

        for mode_idx in range(len(mode_list)):
            mode = mode_list[mode_idx]
            if mode == 3:
                data_buffer = self.inbetween_special_segment_buffer
                data_count = self.inbetween_special_segment_count
                is_preint = 0
            elif mode == 4:
                data_buffer = self.special_preint_buffer
                data_count = self.special_preint_count
                is_preint = 1

            data_index_list = list(data_buffer.keys())
            for group_num in range(mode_batch_size_list[mode_idx]):
                if is_preint:
                    sampled_threshold_index = np.random.randint(0, len(self.special_preint_buffer) - 1)
                else:
                    sampled_threshold_index = np.random.randint(0, len(self.inbetween_special_segment_buffer))
                group_is_preint_list.append(is_preint)

                idx = data_index_list[sampled_threshold_index]
                group_supervision_threshold_list.append(idx)
                sampled_data_buffer = data_buffer[idx]
                sampled_data_count = data_count[idx]

                group_segment_s_t = []
                group_segment_s_t_1 = []
                group_segment_a_t = []
                group_segment_terminal_t = []
                group_segment_r_t = []
                group_segment_intervention_status_t = []
                group_segment_label_t = []
                group_segment_expert_a_t = []
                group_intervention_threshold_list = []

                sampled_group_index = np.random.choice(sampled_data_count, size=(sample_per_group))
                for index in range(sample_per_group):
                    raw_buffer_segment_indices = sampled_data_buffer[sampled_group_index[index]]

                    segment_s_t, segment_s_t_1, segment_a_t, segment_terminal_t, segment_r_t, segment_label_t, \
                    segment_intervention_status_t, segment_expert_a_t, segment_intervention_threshold = self.get_segment(
                        raw_buffer_segment_indices)

                    group_segment_s_t.append(segment_s_t)
                    group_segment_s_t_1.append(segment_s_t_1)
                    group_segment_a_t.append(segment_a_t)
                    group_segment_terminal_t.append(segment_terminal_t)
                    group_segment_r_t.append(segment_r_t)
                    group_segment_label_t.append(segment_label_t)
                    group_segment_intervention_status_t.append(segment_intervention_status_t)
                    group_segment_expert_a_t.append(segment_expert_a_t)
                    group_intervention_threshold_list.append(segment_intervention_threshold)

                group_segment_s_t = np.expand_dims(np.concatenate(group_segment_s_t, axis=0), axis=0)
                group_segment_s_t_1 = np.expand_dims(np.concatenate(group_segment_s_t_1, axis=0), axis=0)
                group_segment_a_t = np.expand_dims(np.concatenate(group_segment_a_t, axis=0), axis=0)
                group_segment_terminal_t = np.expand_dims(np.concatenate(group_segment_terminal_t, axis=0), axis=0)
                group_segment_r_t = np.expand_dims(np.concatenate(group_segment_r_t, axis=0), axis=0)
                group_segment_label_t = np.expand_dims(np.concatenate(group_segment_label_t, axis=0), axis=0)
                group_segment_intervention_status_t = np.expand_dims(
                    np.concatenate(group_segment_intervention_status_t, axis=0), axis=0)
                group_segment_expert_a_t = np.expand_dims(np.concatenate(group_segment_expert_a_t, axis=0), axis=0)
                group_intervention_threshold_list = np.expand_dims(
                    np.concatenate(group_intervention_threshold_list, axis=0), axis=0)

                batch_segment_s_t.append(group_segment_s_t)
                batch_segment_s_t_1.append(group_segment_s_t_1)
                batch_segment_a_t.append(group_segment_a_t)
                batch_segment_terminal_t.append(group_segment_terminal_t)
                batch_segment_r_t.append(group_segment_r_t)
                batch_segment_intervention_status_t.append(group_segment_intervention_status_t)
                batch_segment_label_t.append(group_segment_label_t)
                batch_segment_expert_a_t.append(group_segment_expert_a_t)
                batch_intervention_threshold_list.append(group_intervention_threshold_list)

        batch_segment_s_t = np.concatenate(batch_segment_s_t, axis=0)
        batch_segment_s_t_1 = np.concatenate(batch_segment_s_t_1, axis=0)
        batch_segment_a_t = np.concatenate(batch_segment_a_t, axis=0)
        batch_segment_terminal_t = np.concatenate(batch_segment_terminal_t, axis=0)
        batch_segment_r_t = np.concatenate(batch_segment_r_t, axis=0)
        batch_segment_intervention_status_t = np.concatenate(batch_segment_intervention_status_t, axis=0)
        batch_segment_label_t = np.concatenate(batch_segment_label_t, axis=0)
        batch_segment_expert_a_t = np.concatenate(batch_segment_expert_a_t, axis=0)
        batch_intervention_threshold_list = np.concatenate(batch_intervention_threshold_list, axis=0)
        group_supervision_threshold_list = np.array(group_supervision_threshold_list)
        group_is_preint_list = np.array(group_is_preint_list)

        return batch_segment_s_t, batch_segment_s_t_1, batch_segment_a_t, batch_segment_terminal_t, batch_segment_r_t, batch_segment_intervention_status_t, \
               batch_segment_label_t, batch_segment_expert_a_t, batch_intervention_threshold_list, group_supervision_threshold_list, group_is_preint_list

    def add_no_intervention_data(self, s_t, a_t, r_t, terminal_t, intervention_status_t, s_t_1, label_t, expert_a_t,
                                 intervention_threshold):
        if not self.raw_segment_count in self.raw_data_buffer:
            self.raw_data_buffer[self.raw_segment_count] = {}
        data_entry = [s_t, a_t, r_t, terminal_t, intervention_status_t, s_t_1, label_t, expert_a_t,
                      intervention_threshold]
        current_trajectory_index = len(self.raw_data_buffer[self.raw_segment_count])
        self.raw_data_buffer[self.raw_segment_count][current_trajectory_index] = data_entry
        index_entry = [self.raw_segment_count, current_trajectory_index]

        self.combined_single_state_buffer[self.combined_single_state_count] = index_entry
        self.combined_single_state_count += 1

        self.non_expert_non_preintervention_single_state_buffer[
            self.non_expert_non_preintervention_single_state_count] = index_entry
        self.non_expert_non_preintervention_single_state_count += 1

        self.non_expert_single_state_buffer[self.non_expert_single_state_count] = index_entry
        self.non_expert_single_state_count += 1
        self.raw_data_count += 1

    def add(self, s_t, a_t, r_t, terminal_t, intervention_status_t, s_t_1, label_t, expert_a_t, intervention_threshold,
            raw_only=False):
        if not self.raw_segment_count in self.raw_data_buffer:
            self.raw_data_buffer[self.raw_segment_count] = {}

        data_entry = [s_t, a_t, r_t, terminal_t, intervention_status_t, s_t_1, label_t, expert_a_t,
                      intervention_threshold]
        # print(s_t_1.shape, s_t_1)

        current_trajectory_index = len(self.raw_data_buffer[self.raw_segment_count])
        self.raw_data_buffer[self.raw_segment_count][current_trajectory_index] = data_entry
        self.raw_data_count += 1

        if not raw_only:
            index_entry = [self.raw_segment_count, current_trajectory_index]
            self.combined_single_state_buffer[self.combined_single_state_count] = index_entry
            self.combined_single_state_count += 1
            if intervention_status_t == 1:
                self.expert_single_state_buffer[self.expert_single_state_count] = index_entry
                self.expert_single_state_count += 1
            if label_t == 1:
                self.preintervention_single_state_buffer[self.preintervention_single_state_count] = index_entry
                self.preintervention_single_state_count += 1

            if label_t == 0:
                self.non_expert_non_preintervention_single_state_buffer[
                    self.non_expert_non_preintervention_single_state_count] = index_entry
                self.non_expert_non_preintervention_single_state_count += 1

            if (label_t == 1 or label_t == 0) and intervention_status_t == 0:
                self.non_expert_single_state_buffer[self.non_expert_single_state_count] = index_entry
                self.non_expert_single_state_count += 1

    def simple_sample(self, batch_size, mode=0):
        s_t_list = []
        s_t_1_list = []
        r_t_list = []
        terminal_t_list = []
        a_t_list = []
        intervention_status_t_list = []
        label_t_list = []

        expert_a_t_list = []
        supervision_level_t_list = []

        if mode == 0:
            buffer = self.combined_single_state_buffer
            count = self.combined_single_state_count
        elif mode == 1:
            buffer = self.expert_single_state_buffer
            count = self.expert_single_state_count
        elif mode == 2:
            buffer = self.non_expert_non_preintervention_single_state_buffer
            count = self.non_expert_non_preintervention_single_state_count
        elif mode == 3:
            buffer = self.preintervention_single_state_buffer
            count = self.preintervention_single_state_count
        elif mode == 6:
            buffer = self.non_expert_single_state_buffer
            count = self.non_expert_single_state_count
        elif mode == 7:
            buffer = self.partial_expert_buffer
            count = self.partial_expert_count

        sampled_indices = np.random.randint(0, count, size=(batch_size))
        for i in range(batch_size):
            raw_buffer_indices = buffer[sampled_indices[i]]

            s_t, a_t, r_t, terminal_t, intervention_status_t, s_t_1, label_t, expert_a_t, supervision_level_t = \
            self.raw_data_buffer[raw_buffer_indices[0]][raw_buffer_indices[1]]
            s_t_list.append(s_t)
            s_t_1_list.append(s_t_1)

            a_t_list.append(a_t)
            terminal_t_list.append(terminal_t)
            intervention_status_t_list.append(intervention_status_t)
            r_t_list.append(r_t)
            label_t_list.append(label_t)

            expert_a_t_list.append(expert_a_t)
            supervision_level_t_list.append(supervision_level_t)

        s_t_list = np.array(s_t_list)
        s_t_1_list = np.array(s_t_1_list)
        a_t_list = np.array(a_t_list)
        terminal_t_list = np.array(terminal_t_list)
        r_t_list = np.array(r_t_list)
        label_t_list = np.array(label_t_list)
        intervention_status_t_list = np.array(intervention_status_t_list)

        expert_a_t_list = np.array(expert_a_t_list)
        supervision_level_t_list = np.array(supervision_level_t_list)

        return s_t_list, a_t_list, r_t_list, terminal_t_list, intervention_status_t_list, s_t_1_list, label_t_list, expert_a_t_list, supervision_level_t_list

    def add_segment(self, trajectory_index_list, label, supervision_threshold):
        segment_indices = []

        current_trajectory_index = self.raw_segment_count

        for i in range(trajectory_index_list.shape[0]):
            segment_indices.append([current_trajectory_index, trajectory_index_list[i]])

        if label == 0:
            self.combined_segment_buffer[self.combined_segment_count] = segment_indices
            self.combined_segment_count += 1
        elif label == 1:
            self.expert_segment_buffer[self.expert_segment_count] = segment_indices
            self.expert_segment_count += 1
        elif label == 2:
            self.inbetween_segment_buffer[self.inbetween_segment_count] = segment_indices
            self.inbetween_segment_count += 1
        elif label == 3:
            self.preintervention_segment_buffer[self.preintervention_segment_count] = segment_indices
            self.preintervention_segment_count += 1

            if not supervision_threshold in self.special_preint_buffer:
                self.special_preint_buffer[supervision_threshold] = {}
                self.special_preint_count[supervision_threshold] = 0
            self.special_preint_buffer[supervision_threshold][
                self.special_preint_count[supervision_threshold]] = segment_indices
            self.special_preint_count[supervision_threshold] += 1

            # print(supervision_threshold, self.special_preint_count[supervision_threshold])

    def verify_preint(self, error_function):
        error_count = 0
        for i in range(self.preintervention_segment_count):
            outcome = self.verify_segment(self.preintervention_segment_buffer[i], error_function, is_preint=True)
            if outcome is False:
                error_count += 1
        print(self.preintervention_segment_count)
        print("Preint Error: ", error_count / self.preintervention_segment_count)

    def verify_inbetween(self, error_function):
        error_count = 0
        for i in range(self.inbetween_segment_count):
            outcome = self.verify_segment(self.inbetween_segment_buffer[i], error_function, is_preint=False)
            if outcome is False:
                error_count += 1
        print("Inbetween Error: ", error_count / self.inbetween_segment_count)

    def verify_expert(self, error_function):
        for i in range(self.expert_segment_count):
            intervention_threshold_list = []
            expert_threshold_list = []
            true_error_list = []
            preint_list = []
            for j in range(self.args.segment_length):
                s_t, a_t, r_t, terminal_t, intervention_status_t, s_t_1, label_t, expert_a_t, intervention_threshold = \
                self.raw_data_buffer[self.expert_segment_buffer[i][j][0]][self.expert_segment_buffer[i][j][1]]
                intervention_threshold_list.append(intervention_threshold)

                true_error, _ = error_function(a_t,
                                               expert_a_t)  # np.clip(np.mean(np.abs(a_t - expert_a_t)), 0, 0.5) / 0.5
                true_error_list.append(true_error)

                expert_threshold_list.append(intervention_status_t)
                preint_list.append(label_t)

    def verify_segment(self, segment_indices, error_function, is_preint):
        intervention_threshold_list = []
        expert_threshold_list = []
        true_error_list = []
        preint_list = []
        intervention_status_list = []
        for j in range(self.args.segment_length):
            s_t, a_t, r_t, terminal_t, intervention_status_t, s_t_1, label_t, expert_a_t, intervention_threshold = \
                self.raw_data_buffer[segment_indices[j][0]][segment_indices[j][1]]
            intervention_threshold_list.append(intervention_threshold)
            intervention_status_list.append(intervention_status_t)

            true_error, _ = error_function(a_t, expert_a_t)
            # true_error = np.clip(np.mean(np.abs(a_t - expert_a_t)), 0, 0.5) / 0.5
            true_error_list.append(true_error)

            expert_threshold_list.append(intervention_status_t)
            preint_list.append(label_t)

        supervision_threshold = np.max(intervention_threshold_list)
        if is_preint:
            if np.sum(true_error_list) >= supervision_threshold and np.sum(true_error_list) < supervision_threshold + 2:
                return True
            else:
                # if supervision_threshold > np.sum(true_error_list) + 1:

                trajectory = self.raw_data_buffer[segment_indices[j][0]]
                for step in trajectory:
                    s_t, a_t, r_t, terminal_t, intervention_status_t, s_t_1, label_t, expert_a_t, intervention_threshold = \
                    trajectory[step]
                    step_error, _ = error_function(a_t, expert_a_t)

                    print(step, step_error, intervention_threshold, intervention_status_t, label_t)
                print(np.sum(true_error_list), segment_indices[0][0], segment_indices[0][1], supervision_threshold,
                      np.mean(intervention_threshold_list))
                # for i in range(self.raw_segment_count):
                #     print(i, len(self.raw_data_buffer[i]))
                print("Preint Error!")
                raise ("Error!")
                quit()
                return False
        else:
            if np.sum(true_error_list) <= supervision_threshold:
                return True
            else:
                trajectory = self.raw_data_buffer[segment_indices[j][0]]
                for step in trajectory:
                    s_t, a_t, r_t, terminal_t, intervention_status_t, s_t_1, label_t, expert_a_t, intervention_threshold = \
                    trajectory[step]

                    step_error, _ = error_function(a_t, expert_a_t)

                    print(step, step_error, intervention_threshold, intervention_status_t, label_t)
                print(np.sum(true_error_list), segment_indices[0][0], segment_indices[0][1], supervision_threshold,
                      np.mean(intervention_threshold_list))

                # for i in range(self.raw_segment_count):
                #     print(i, len(self.raw_data_buffer[i]))
                print("Inbetween Error!")
                quit()

                return False

    def get_all_supervision_threshold(self, supervision_threshold, is_inbetween):
        greater_than_list = []
        for key in self.special_preint_buffer:
            if key > supervision_threshold:
                greater_than_list.append(key)

        if len(greater_than_list) == 0:
            return [supervision_threshold]
        else:
            if is_inbetween:

                if supervision_threshold == 0:
                    return greater_than_list
                else:
                    return greater_than_list + [supervision_threshold]
            else:
                return greater_than_list

    def get_next_supervision_threshold(self, supervision_threshold):
        greater_than_list = []
        for key in self.special_preint_buffer:
            if key > supervision_threshold:
                greater_than_list.append(key)

        if len(greater_than_list) == 0:
            return [supervision_threshold]
        else:
            return [min(greater_than_list), supervision_threshold]

    def sample_preint(self, batch_size, supervision_thresholds, is_preint, is_expert, preint_include_eq=False,
                      all_legal=False):
        segment_s_t_list = []
        segment_s_t_1_list = []
        segment_r_t_list = []
        segment_terminal_t_list = []
        segment_a_t_list = []
        segment_intervention_status_t_list = []
        segment_label_t_list = []
        segment_mean_label_list = []
        segment_weight_list = []

        segment_expert_a_t_list = []
        intervention_threshold_list = []
        random_segment_list = []
        for i in range(batch_size):
            random_actions = False
            if all_legal:
                current_supervision_threshold_list = self.get_all_supervision_threshold(int(supervision_thresholds[i]),
                                                                                        (1 - is_preint[
                                                                                            i]) or preint_include_eq)
                current_supervision_threshold = int(
                    current_supervision_threshold_list[np.random.randint(0, len(current_supervision_threshold_list))])
            else:
                if is_preint[i]:
                    current_supervision_threshold_list = self.get_next_supervision_threshold(
                        int(supervision_thresholds[i]))
                    if preint_include_eq:
                        current_supervision_threshold = int(current_supervision_threshold_list[np.random.randint(0, len(
                            current_supervision_threshold_list))])
                    else:
                        current_supervision_threshold = int(current_supervision_threshold_list[0])
                else:
                    if supervision_thresholds[i] == 0:
                        current_supervision_threshold_list = self.get_next_supervision_threshold(
                            int(supervision_thresholds[i]))
                        current_supervision_threshold = int(current_supervision_threshold_list[0])
                    else:
                        current_supervision_threshold = int(supervision_thresholds[i])

            if not preint_include_eq:
                # print(i, is_preint[i], supervision_thresholds[i] == current_supervision_threshold)
                if is_preint[i] and supervision_thresholds[i] == current_supervision_threshold:
                    random_actions = True
                    # raise("Error! Random Action Raised!")
            buffer = self.special_preint_buffer[current_supervision_threshold]
            count = self.special_preint_count[current_supervision_threshold]

            # print(current_supervision_threshold, count, sampled_index)
            # if all_legal and np.random.uniform(0, 1) < 0.2:
            #     random_actions = True

            sampled_index = np.random.randint(0, count)
            raw_buffer_segment_indices = buffer[sampled_index]
            mean_label = 0
            segment_weight = 0
            for j in range(self.args.segment_length):
                s_t, a_t, r_t, terminal_t, intervention_status_t, s_t_1, label_t, expert_a_t, intervention_threshold = \
                self.raw_data_buffer[raw_buffer_segment_indices[j][0]][raw_buffer_segment_indices[j][1]]

                if random_actions:
                    a_t = np.clip(np.random.uniform(-1.2, 1.2, size=a_t.shape), -1, 1)

                segment_s_t_list.append(s_t)
                segment_s_t_1_list.append(s_t_1)
                segment_a_t_list.append(a_t)
                segment_terminal_t_list.append(terminal_t)
                segment_r_t_list.append(r_t)
                segment_label_t_list.append(label_t)
                segment_intervention_status_t_list.append(intervention_status_t)
                mean_label += (label_t + 1) / 2
                segment_expert_a_t_list.append(expert_a_t)
                intervention_threshold_list.append(intervention_threshold)
                segment_weight += self.raw_data_weight_no_expert[raw_buffer_segment_indices[j][0]][
                    raw_buffer_segment_indices[j][1]]

            segment_weight_list.append(segment_weight / self.args.segment_length)

            mean_label = mean_label / self.args.segment_length
            segment_mean_label_list.append(mean_label)
            random_segment_list.append(random_actions)

        s_t_list = np.reshape(segment_s_t_list, [batch_size, self.args.segment_length, -1])
        # quit()

        s_t_1_list = np.reshape(segment_s_t_1_list, [batch_size, self.args.segment_length, -1])
        a_t_list = np.reshape(segment_a_t_list, [batch_size, self.args.segment_length, -1])
        terminal_t_list = np.reshape(segment_terminal_t_list, [batch_size, self.args.segment_length, -1])
        r_t_list = np.reshape(segment_r_t_list, [batch_size, self.args.segment_length, -1])

        expert_a_t_list = np.reshape(segment_expert_a_t_list, [batch_size, self.args.segment_length, -1])
        intervention_threshold_list = np.reshape(intervention_threshold_list,
                                                 [batch_size, self.args.segment_length, -1])

        label_t_raw_list = np.reshape(segment_label_t_list, [batch_size, self.args.segment_length, -1])
        label_t_list = np.array(segment_mean_label_list)
        intervention_status_t_list = np.reshape(segment_intervention_status_t_list,
                                                [batch_size, self.args.segment_length, 1])
        segment_weight_list = np.array(segment_weight_list)
        random_segment_list = np.array(random_segment_list)
        return s_t_list, a_t_list, r_t_list, terminal_t_list, intervention_status_t_list, s_t_1_list, \
               label_t_list, label_t_raw_list, expert_a_t_list, intervention_threshold_list, segment_weight_list, random_segment_list

    def sample(self, batch_size, mode=0, fixed=False):
        segment_s_t_list = []
        segment_s_t_1_list = []
        segment_r_t_list = []
        segment_terminal_t_list = []
        segment_a_t_list = []
        segment_intervention_status_t_list = []
        segment_label_t_list = []
        segment_mean_label_list = []

        segment_expert_a_t_list = []
        intervention_threshold_list = []
        segment_weight_list = []

        if mode == 0:
            buffer = self.combined_segment_buffer
            count = self.combined_segment_count
        elif mode == 1:
            buffer = self.expert_segment_buffer
            count = self.expert_segment_count
        elif mode == 2:
            buffer = self.preintervention_segment_buffer
            count = self.preintervention_segment_count
        elif mode == 3:
            buffer = self.inbetween_segment_buffer
            count = self.inbetween_segment_count
        elif mode == 4:
            buffer = self.preintervention_segment_special_buffer
            count = self.preintervention_segment_special_count
        elif mode == 5:
            buffer = self.inbetween_special_buffer
            count = self.inbetween_special_count
        elif mode == 6:
            buffer = self.preintervention_segment_special_no_expert_buffer
            count = self.preintervention_segment_special_no_expert_count
        elif mode == 7:
            buffer = self.non_expert_segment_special_buffer
            count = self.non_expert_segment_special_count

        # print(mode, count)
        if fixed:
            sampled_indices = np.zeros((batch_size,))
        else:
            # print("Was here", count, mode, self.preintervention_segment_special_count)
            sampled_indices = np.random.randint(0, count, size=(batch_size))
        for i in range(batch_size):
            raw_buffer_segment_indices = buffer[sampled_indices[i]]

            mean_label = 0
            segment_weight = 0
            for j in range(self.args.segment_length):
                s_t, a_t, r_t, terminal_t, intervention_status_t, s_t_1, label_t, expert_a_t, intervention_threshold = \
                self.raw_data_buffer[raw_buffer_segment_indices[j][0]][raw_buffer_segment_indices[j][1]]

                segment_weight += self.raw_data_weight_no_expert[raw_buffer_segment_indices[j][0]][
                    raw_buffer_segment_indices[j][1]]

                segment_s_t_list.append(s_t)
                segment_s_t_1_list.append(s_t_1)
                segment_a_t_list.append(a_t)
                segment_terminal_t_list.append(terminal_t)
                segment_r_t_list.append(r_t)
                segment_label_t_list.append(label_t)
                segment_intervention_status_t_list.append(intervention_status_t)
                mean_label += (label_t + 1) / 2
                segment_expert_a_t_list.append(expert_a_t)
                intervention_threshold_list.append(intervention_threshold)

            mean_label = mean_label / self.args.segment_length
            segment_mean_label_list.append(mean_label)

            segment_weight_list.append(segment_weight / self.args.segment_length)

        s_t_list = np.reshape(segment_s_t_list, [batch_size, self.args.segment_length, -1])

        s_t_1_list = np.reshape(segment_s_t_1_list, [batch_size, self.args.segment_length, -1])
        a_t_list = np.reshape(segment_a_t_list, [batch_size, self.args.segment_length, -1])
        terminal_t_list = np.reshape(segment_terminal_t_list, [batch_size, self.args.segment_length, -1])
        r_t_list = np.reshape(segment_r_t_list, [batch_size, self.args.segment_length, -1])

        expert_a_t_list = np.reshape(segment_expert_a_t_list, [batch_size, self.args.segment_length, -1])
        intervention_threshold_list = np.reshape(intervention_threshold_list,
                                                 [batch_size, self.args.segment_length, -1])

        label_t_raw_list = np.reshape(segment_label_t_list, [batch_size, self.args.segment_length, -1])
        label_t_list = np.array(segment_mean_label_list)
        intervention_status_t_list = np.reshape(segment_intervention_status_t_list,
                                                [batch_size, self.args.segment_length, 1])
        trajectory_weight_list = np.array(segment_weight_list)
        return s_t_list, a_t_list, r_t_list, terminal_t_list, intervention_status_t_list, \
               s_t_1_list, label_t_list, label_t_raw_list, expert_a_t_list, intervention_threshold_list, trajectory_weight_list

    def generate_part_expert_data(self, partial_percent=0.8):

        self.partial_expert_buffer = {}
        self.partial_expert_count = 0

        for i in range(self.expert_single_state_count):
            if np.random.uniform(0, 1) < partial_percent:
                self.partial_expert_buffer[self.partial_expert_count] = self.expert_single_state_buffer[i]
                self.partial_expert_count += 1

    def query_additional_preint_data(self, err_func, fixed_intervention_thresholds_list, sorted_segment_indices_list,
                                     segment_list_action_std, target_sampled_count):
        min_threshold = fixed_intervention_thresholds_list[0]
        fixed_intervention_thresholds_list = np.array(fixed_intervention_thresholds_list)[
                                             :-1]  # last threshold not necessary

        new_inbetween_buffer = {}
        new_inbetween_count = 0
        # scan thru all data without preint
        queried_segments = 0

        new_count = 0
        for i in range(self.inbetween_segment_count):
            index = sorted_segment_indices_list[-1 - i]
            segment_indices = self.inbetween_segment_buffer[index]

            segment_error_list = []
            s_t_list = []
            a_t_list = []
            r_t_list = []
            terminal_t_list = []
            intervention_status_t_list = []
            s_t_1_list = []
            label_t_list = []
            expert_a_t_list = []
            intervention_threshold_list = []

            for j in range(self.args.segment_length):
                s_t, a_t, r_t, terminal_t, intervention_status_t, s_t_1, label_t, expert_a_t, intervention_threshold = \
                self.raw_data_buffer[segment_indices[j][0]][segment_indices[j][1]]

                true_error, _ = err_func(a_t, expert_a_t)
                segment_error_list.append(true_error)

                s_t_list.append(s_t)
                a_t_list.append(a_t)
                r_t_list.append(r_t)
                terminal_t_list.append(terminal_t)
                intervention_status_t_list.append(intervention_status_t)
                s_t_1_list.append(s_t_1)
                label_t_list.append(label_t)
                expert_a_t_list.append(expert_a_t)
                intervention_threshold_list.append(intervention_threshold)

            if new_count < target_sampled_count:
                print(i, index, np.max(intervention_threshold_list), np.sum(segment_error_list),
                      segment_list_action_std[i], queried_segments)
            if np.max(intervention_threshold_list) != min_threshold and (not new_count >= target_sampled_count):

                new_count += 1
                segment_error = np.sum(segment_error_list)
                selected_threshold = None
                for k in range(fixed_intervention_thresholds_list.shape[0] - 1, -1, -1):
                    if segment_error > fixed_intervention_thresholds_list[k]:
                        selected_threshold = fixed_intervention_thresholds_list[k]
                        break

                if not selected_threshold is None:
                    # Update the new data
                    for k in range(self.args.segment_length):
                        # intervention_threshold_list[k] = selected_threshold
                        label_t_list[k] = 1

                    new_segment_indices = []
                    for k in range(self.args.segment_length):
                        self.add(s_t_list[k], a_t_list[k], r_t_list[k], terminal_t_list[k],
                                 intervention_status_t_list[k],
                                 s_t_1_list[k], label_t_list[k], expert_a_t_list[k], selected_threshold, raw_only=True)
                        new_segment_indices.append(k)

                    self.add_segment(np.array(new_segment_indices), 3, selected_threshold)  # Add preint
                    self.trajectory_end()
                    queried_segments += 1
                else:
                    # for k in range(self.args.segment_length):
                    #     intervention_threshold_list[k] = min_threshold

                    new_segment_indices = []
                    for k in range(self.args.segment_length):
                        self.add(s_t_list[k], a_t_list[k], r_t_list[k], terminal_t_list[k],
                                 intervention_status_t_list[k],
                                 s_t_1_list[k], label_t_list[k], expert_a_t_list[k], min_threshold, raw_only=True)
                        new_segment_indices.append(k)
                    self.add_segment(np.array(new_segment_indices), 2, min_threshold)  # Add preint
                    self.trajectory_end()
                    queried_segments += 1

                    # for k in range(self.args.segment_length):
                    #     s_t, a_t, r_t, terminal_t, intervention_status_t, s_t_1, label_t, expert_a_t, intervention_threshold = self.raw_data_buffer[segment_indices[k][0]][segment_indices[k][1]]
                    #     self.raw_data_buffer[segment_indices[k][0]][segment_indices[k][1]] = (s_t, a_t, r_t, terminal_t, intervention_status_t, s_t_1, label_t, expert_a_t, min_threshold)
                    #
                    # # for k in range(self.args.segment_length):
                    # #     intervention_threshold_list[k] = min_threshold
                    # new_inbetween_buffer[new_inbetween_count] = self.inbetween_segment_buffer[index]
                    # new_inbetween_count += 1
            else:
                new_inbetween_buffer[new_inbetween_count] = self.inbetween_segment_buffer[index]
                new_inbetween_count += 1

        self.inbetween_segment_buffer = new_inbetween_buffer
        self.inbetween_segment_count = new_inbetween_count
        print("Queried Expert: ", new_count, queried_segments)

    def sample_combined(self, mode_list, mode_batch_size_list):
        s_t_list = []
        a_t_list = []
        r_t_list = []
        terminal_t_list = []
        intervention_status_t_list = []
        s_t_1_list = []
        label_t_list = []
        label_t_raw_list = []
        expert_a_t_list = []
        intervention_threshold_list = []
        trajectory_weight_list = []
        for i in range(len(mode_list)):
            s_t, a_t, r_t, terminal_t, \
            intervention_status_t, s_t_1, label_t, \
            label_t_raw, expert_a_t, intervention_threshold, trajectory_weight = self.sample(mode_batch_size_list[i],
                                                                                             mode_list[i])

            s_t_list.append(s_t)
            a_t_list.append(a_t)
            r_t_list.append(r_t)
            terminal_t_list.append(terminal_t)
            intervention_status_t_list.append(intervention_status_t)
            s_t_1_list.append(s_t_1)
            label_t_list.append(label_t)
            label_t_raw_list.append(label_t_raw)
            expert_a_t_list.append(expert_a_t)
            intervention_threshold_list.append(intervention_threshold)
            trajectory_weight_list.append(trajectory_weight)

        s_t_list = np.concatenate(s_t_list, axis=0)
        a_t_list = np.concatenate(a_t_list, axis=0)
        r_t_list = np.concatenate(r_t_list, axis=0)
        terminal_t_list = np.concatenate(terminal_t_list, axis=0)
        intervention_status_t_list = np.concatenate(intervention_status_t_list, axis=0)
        s_t_1_list = np.concatenate(s_t_1_list, axis=0)
        label_t_list = np.concatenate(label_t_list, axis=0)
        label_t_raw_list = np.concatenate(label_t_raw_list, axis=0)
        expert_a_t_list = np.concatenate(expert_a_t_list, axis=0)
        intervention_threshold_list = np.concatenate(intervention_threshold_list, axis=0)
        trajectory_weight_list = np.concatenate(trajectory_weight_list, axis=0)

        return s_t_list, a_t_list, r_t_list, terminal_t_list, intervention_status_t_list, s_t_1_list, \
               label_t_list, label_t_raw_list, expert_a_t_list, intervention_threshold_list, trajectory_weight_list

    def trajectory_end(self):
        # print("Bug Testing ... ", self.raw_segment_count, len(self.raw_data_buffer),len(self.raw_data_buffer[self.raw_segment_count]))
        # current_trajectory = self.raw_data_buffer[self.raw_segment_count]
        # for i in range(len(current_trajectory)):
        #     if i + self.args.segment_length <= len(current_trajectory):
        #         #print("Was here ... ")
        #         self.add_segment(current_trajectory, self.raw_segment_count, i)
        #     else:
        #         break
        self.raw_segment_count += 1


class tensorflowboard_logger:
    def __init__(self, figure_dir, args):
        print("Writing logs to ", figure_dir + "/" + str(int(time.time())))
        self.directory = figure_dir + "/" + str(int(time.time()))

        self.writer = tf.summary.create_file_writer(self.directory)
        self.logged_scalar_dict = {}

        # dump the initial settings in the directory as well ...
        jsonStr = json.dumps(args.__dict__)
        file = open(self.directory + "/args.json", "w")
        file.write(jsonStr)
        file.close()

    def log_scalar(self, scalar_name, scalar_value, step):
        with self.writer.as_default():
            tf.summary.scalar(scalar_name, scalar_value, step=step)
            self.writer.flush()


class rl_dataset:
    def __init__(self, args, env, buffer, n_step_next_action=False):
        self.dataset = env.get_dataset()

        self.state = self.dataset["observations"]
        self.actions = self.dataset["actions"]
        self.rewards = self.dataset["rewards"]
        self.terminals = self.dataset["terminals"]
        self.timeouts = self.dataset["timeouts"]
        self.args = args

        state_list = []
        action_list = []
        reward_list = []

        max_reward = np.max(self.rewards)
        min_reward = np.min(self.rewards)
        # median_reward = (np.median(self.rewards) - min_reward)/(max_reward - min_reward)
        # median_reward = median_reward * 2 - 1

        terminal_list = []

        self.traj_len_list = []
        self.traj_reward_list = []
        traj_len = 0
        traj_reward = 0
        added_count = 0
        for i in range(self.state.shape[0]):

            # np.sign(list_rew[i]) * np.log(1 + list_rew[i])

            state_list.append(self.state[i])
            action_list.append(self.actions[i])
            # reward_list.append(np.sign(self.rewards[i]) * np.log(1 + np.abs(self.rewards[i])))

            rescaled_reward = self.rewards[i] / (max_reward - min_reward)
            reward_list.append(rescaled_reward)
            # reward_list.append(self.rewards[i])
            terminal_list.append(self.terminals[i])

            traj_len += 1
            traj_reward += self.rewards[i]

            if self.terminals[i] or self.timeouts[i]:
                # add dummy values for next state, next action
                action_list.append(np.copy(self.actions[i]))  # Dummy actions ...

                n_step_next_action_list = []
                n_step_action_list = []
                n_step_state_list = []
                n_step_terminal = []

                for j in range(len(reward_list) - 1):
                    n_step_action = 0
                    n_step_state_index = min(len(reward_list) - 1, j + args.n_step)
                    value_power = 1
                    for k in range(j, n_step_state_index):
                        n_step_action += value_power * action_list[k + 1]
                        value_power *= args.gamma_action
                    n_step_state_list.append(state_list[n_step_state_index])
                    n_step_action_list.append(n_step_action)
                    n_step_next_action_list.append(action_list[n_step_state_index + 1])
                    n_step_terminal.append(terminal_list[n_step_state_index - 1])

                # state, next_state, action, next_action, rewards, terminal, n_step_state, n_step_terminal, n_step_action_mean, abstract_action

                for j in range(len(reward_list) - 1):
                    # if terminal_list[j]:
                    # print(i, self.timeouts[j], len(state_list), len(action_list))
                    # quit()
                    #
                    # print(self.state[i - 2])
                    # print(self.state[i - 1])
                    # print(self.state[i])
                    # print(self.state[i + 1])
                    # print(self.state[i + 2])

                    if self.timeouts[j + 1] and j > len(
                            reward_list) - args.n_step:  # do not add last n_steps states because it's a bit messed up
                        break
                    added_count += 1
                    if n_step_next_action:
                        # if terminal_list[j + 1]:
                        #     for _ in range(10):
                        #         buffer.add(np.copy(state_list[j]), np.copy(state_list[j + 1]), np.copy(action_list[j]), np.copy(action_list[j + 1]),
                        #                    np.copy(reward_list[j]), np.copy(terminal_list[j + 1]), np.copy(n_step_state_list[j]),
                        #                    np.copy(n_step_terminal[j]), np.copy(n_step_action_list[j]), n_step_next_action_list[j])
                        # else:
                        buffer.add(np.copy(state_list[j]), np.copy(state_list[j + 1]), np.copy(action_list[j]),
                                   np.copy(action_list[j + 1]),
                                   np.copy(reward_list[j]), np.copy(terminal_list[j + 1]),
                                   np.copy(n_step_state_list[j]),
                                   np.copy(n_step_terminal[j]), np.copy(n_step_action_list[j]),
                                   n_step_next_action_list[j])
                    else:
                        # if terminal_list[j + 1]:
                        #     for _ in range(10):
                        #         buffer.add(np.copy(state_list[j]), np.copy(state_list[j + 1]), np.copy(action_list[j]), np.copy(action_list[j + 1]),
                        #                    np.copy(reward_list[j]), np.copy(terminal_list[j]), np.copy(n_step_state_list[j]), np.copy(n_step_terminal[j]), np.copy(n_step_action_list[j]))
                        # else:
                        buffer.add(np.copy(state_list[j]), np.copy(state_list[j + 1]), np.copy(action_list[j]),
                                   np.copy(action_list[j + 1]),
                                   np.copy(reward_list[j]), np.copy(terminal_list[j + 1]),
                                   np.copy(n_step_state_list[j]), np.copy(n_step_terminal[j]),
                                   np.copy(n_step_action_list[j]))

                self.traj_len_list.append(traj_len)
                self.traj_reward_list.append(traj_reward)
                # reset for next trajectory
                traj_len = 0
                traj_reward = 0
                state_list = []
                action_list = []
                reward_list = []
                terminal_list = []

        print(len(self.traj_len_list), np.max(self.traj_len_list), np.min(self.traj_len_list),
              np.max(self.traj_reward_list), np.min(self.traj_reward_list))
        print("Average Reward: ", np.mean(self.rewards), "Max Normalized Reward: ",
              env.get_normalized_score(np.max(self.traj_reward_list)), "Max Reward: ", np.max(self.traj_reward_list))
        print("Buffer Size: ", added_count)

    def sample(self):
        idxes = np.random.randint(0, self.size, size=(self.args.batch_size))

        state = self.dataset["observations"][idxes]
        actions = self.dataset["actions"][idxes]
        next_state = self.dataset["next_observations"][idxes]
        rewards = self.dataset["rewards"][idxes]
        terminals = self.dataset["terminals"][idxes]
        return state, next_state, actions, rewards, terminals

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_dir', type=str, default="../models/")
    parser.add_argument('--log_dir', type=str, default="../logs/")
    parser.add_argument('--gif_dir', type=str, default="../gif/")

    parser.add_argument('--custom_id', type=str, default="sac")
    parser.add_argument('--l2', type=float, default=0.0001)
    parser.add_argument('--special_l2', type=float, default=0.0005)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--gamma_action', type=float, default=0.99)

    parser.add_argument('--atoms', type=int, default=32)
    parser.add_argument('--task_id', type=int, default=-1)

    parser.add_argument('--gamma_action_1', type=float, default=0.99)
    parser.add_argument('--gamma_action_2', type=float, default=0.9)
    parser.add_argument('--gamma_action_3', type=float, default=0.75)
    parser.add_argument('--gamma_action_4', type=float, default=0.75)
    parser.add_argument('--policy_loss_coef', type=float, default=0.2)

    parser.add_argument('--qvf_lr', type=float, default=0.0003)
    parser.add_argument('--vf_lr', type=float, default=0.0003)
    parser.add_argument('--ac_lr', type=float, default=0.0003)
    parser.add_argument('--transition_lr', type=float, default=0.0005)

    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--temperature', type=float, default=0.2)
    parser.add_argument('--planning_loss_coef', type=float, default=1.0)
    parser.add_argument('--planning_loss_qval', type=float, default=200)

    parser.add_argument('--action_scale', type=float, default=1)
    parser.add_argument('--reward_scale', type=float, default=1)
    parser.add_argument('--history_length', type=int, default=4)

    parser.add_argument('--hidden', type=int, default=256)
    parser.add_argument('--seed', type=int, default=0)

    parser.add_argument('--max_steps', type=int, default=10000000)
    parser.add_argument('--max_eps_len', type=int, default=1000)

    parser.add_argument('--eval_freq', type=int, default=29999)

    parser.add_argument('--load_model_index', type=int, default=-1)
    parser.add_argument('--num_envs', type=int, default=10)

    parser.add_argument('--save_freq', type=int, default=200000)
    parser.add_argument('--eval_num', type=int, default=20)
    parser.add_argument('--n_step', type=int, default=10)

    parser.add_argument('--ensemble_size', type=int, default=3)

    parser.add_argument('--policy_train_freq', type=int, default=1)
    parser.add_argument('--critic_train_freq', type=int, default=1)
    parser.add_argument('--replay_start_size', type=int, default=20000)
    parser.add_argument('--action_history_length', type=int, default=4)

    parser.add_argument('--distribution_divergence_loss', choices=["kl", "js", "mean"], default="kl")
    parser.add_argument('--critic_target_dist', choices=["raw", "est"], default="est")
    parser.add_argument('--load_buffer', type=str, default=None)

    parser.add_argument('--alpha', type=float, help='Max Episode Length', default=0.6)
    parser.add_argument('--beta', type=float, help='Max Episode Length', default=0.4)
    parser.add_argument('--env_id', type=str, default="Hopper-v2")
    parser.add_argument('--difficulty_id', type=str, default="easy")
    parser.add_argument('--input_resize', type=list, default=[64, 64, 1])
    parser.add_argument('--min_sigma', type=float, default=0.0001)
    parser.add_argument('--coefficient', type=float, default=30)

    parser.add_argument('--policy_noise', type=float, default=0.1)
    parser.add_argument('--noise_clip', type=float, default=0.2)
    parser.add_argument('--annealing_frame', type=float, default=1500000)
    parser.add_argument('--env_annealing_frame', type=float, default=1500000)

    parser.add_argument('--goal_range', type=float, default=0.025)
    parser.add_argument('--target_update_rate', type=float, default=0.005)
    parser.add_argument('--toy_grid_size', type=int, default=40)

    parser.add_argument('--hard_target_update', type=int, default=0)
    parser.add_argument('--minimum_action_weight', type=float, default=0.05)
    parser.add_argument('--gif_mode', type=int, default=0)
    parser.add_argument('--segment_length', type=int, default=30)
    parser.add_argument('--intervention_delay', type=int, default=1)
    parser.add_argument('--cost_cap', type=float, default=0.2)

    parser.add_argument('--intervention_loss', type=int, default=0)
    parser.add_argument('--cost_loss_modifier', type=float, default=1)
    parser.add_argument('--cost_loss_bias', type=float, default=0)
    parser.add_argument('--cost_only', type=int, default=0)
    parser.add_argument('--train_initial_model', type=int, default=0)
    parser.add_argument('--cosine_lr', type=int, default=0)

    parser.add_argument('--policy_training_steps', type=int, default=500001)
    parser.add_argument('--preference_loss_weight', type=float, default=1)
    parser.add_argument('--error_function_cap', type=float, default=0.3)

    args = parser.parse_args()
    return args

