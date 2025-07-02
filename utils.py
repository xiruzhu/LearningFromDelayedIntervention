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


class SegmentTree(object):
    def __init__(self, capacity, operation, neutral_element):
        """Build a Segment Tree data structure.
        https://en.wikipedia.org/wiki/Segment_tree
        Can be used as regular array, but with two
        important differences:
            a) setting item's value is slightly slower.
               It is O(lg capacity) instead of O(1).
            b) user has access to an efficient ( O(log segment size) )
               `reduce` operation which reduces `operation` over
               a contiguous subsequence of items in the array.
        Paramters
        ---------
        capacity: int
            Total size of the array - must be a power of two.
        operation: lambda obj, obj -> obj
            and operation for combining elements (eg. sum, max)
            must form a mathematical group together with the set of
            possible values for array elements (i.e. be associative)
        neutral_element: obj
            neutral element for the operation above. eg. float('-inf')
            for max and 0 for sum.
        """
        assert capacity > 0 and capacity & (capacity - 1) == 0, "capacity must be positive and a power of 2."
        self._capacity = capacity
        self._value = [neutral_element for _ in range(2 * capacity)]
        self._operation = operation

    def _reduce_helper(self, start, end, node, node_start, node_end):
        if start == node_start and end == node_end:
            return self._value[node]
        mid = (node_start + node_end) // 2
        if end <= mid:
            return self._reduce_helper(start, end, 2 * node, node_start, mid)
        else:
            if mid + 1 <= start:
                return self._reduce_helper(start, end, 2 * node + 1, mid + 1, node_end)
            else:
                return self._operation(
                    self._reduce_helper(start, mid, 2 * node, node_start, mid),
                    self._reduce_helper(mid + 1, end, 2 * node + 1, mid + 1, node_end)
                )

    def reduce(self, start=0, end=None):
        """Returns result of applying `self.operation`
        to a contiguous subsequence of the array.
            self.operation(arr[start], operation(arr[start+1], operation(... arr[end])))
        Parameters
        ----------
        start: int
            beginning of the subsequence
        end: int
            end of the subsequences
        Returns
        -------
        reduced: obj
            result of reducing self.operation over the specified range of array elements.
        """
        if end is None:
            end = self._capacity
        if end < 0:
            end += self._capacity
        end -= 1
        return self._reduce_helper(start, end, 1, 0, self._capacity - 1)

    def __setitem__(self, idx, val):
        # index of the leaf
        idx += self._capacity
        self._value[idx] = val
        idx //= 2
        while idx >= 1:
            self._value[idx] = self._operation(
                self._value[2 * idx],
                self._value[2 * idx + 1]
            )
            idx //= 2

    def __getitem__(self, idx):
        assert 0 <= idx < self._capacity
        return self._value[self._capacity + idx]


class SumSegmentTree(SegmentTree):
    def __init__(self, capacity):
        super(SumSegmentTree, self).__init__(
            capacity=capacity,
            operation=operator.add,
            neutral_element=0.0
        )

    def sum(self, start=0, end=None):
        """Returns arr[start] + ... + arr[end]"""
        return super(SumSegmentTree, self).reduce(start, end)

    def find_prefixsum_idx(self, prefixsum):
        """Find the highest index `i` in the array such that
            sum(arr[0] + arr[1] + ... + arr[i - i]) <= prefixsum
        if array values are probabilities, this function
        allows to sample indexes according to the discrete
        probability efficiently.
        Parameters
        ----------
        perfixsum: float
            upperbound on the sum of array prefix
        Returns
        -------
        idx: int
            highest index satisfying the prefixsum constraint
        """
        assert 0 <= prefixsum <= self.sum() + 1e-5
        idx = 1
        while idx < self._capacity:  # while non-leaf
            if self._value[2 * idx] > prefixsum:
                idx = 2 * idx
            else:
                prefixsum -= self._value[2 * idx]
                idx = 2 * idx + 1
        return idx - self._capacity


class MinSegmentTree(SegmentTree):
    def __init__(self, capacity):
        super(MinSegmentTree, self).__init__(
            capacity=capacity,
            operation=min,
            neutral_element=float('inf')
        )

    def min(self, start=0, end=None):
        """Returns min(arr[start], ...,  arr[end])"""

        return super(MinSegmentTree, self).reduce(start, end)


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


class TrajectoryPairsBuffer:
    def __init__(self, args, size=5000):
        self.args = args
        self.buffer = []

    def add_pairs(self, trajectory_1, trajectory_2):
        self.buffer.append(trajectory_1, trajectory_2)

    def sample_pairs(self, batch_size):
        print()


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


class TrajectorySegmentBuffer_v4:
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

        self.combined_segment_buffer = {}
        self.combined_segment_count = 0

        self.expert_segment_buffer = {}
        self.expert_segment_count = 0

        self.preintervention_segment_buffer = {}
        self.preintervention_segment_count = 0

    def reinit(self, old_buffer, error_function, threshold_list=None, debug_mode=False):
        self.raw_data_buffer = old_buffer.raw_data_buffer
        self.raw_data_count = old_buffer.raw_data_count

        self.combined_single_state_buffer = old_buffer.combined_single_state_buffer
        self.combined_single_state_count = old_buffer.combined_single_state_count

        self.expert_single_state_buffer = old_buffer.expert_single_state_buffer
        self.expert_single_state_count = old_buffer.expert_single_state_count

        self.preintervention_single_state_buffer = old_buffer.preintervention_single_state_buffer
        self.preintervention_single_state_count = old_buffer.preintervention_single_state_count

        self.non_expert_non_preintervention_single_state_buffer = {}
        self.non_expert_non_preintervention_single_state_count = 0

        self.non_expert_include_termination_single_state_buffer = {}
        self.non_expert_include_termination_single_state_count = 0

        self.expert_exclude_termination_single_state_buffer = {}
        self.expert_exclude_termination_single_state_count = 0

        self.raw_segment_count = old_buffer.raw_segment_count
        self.combined_segment_buffer = old_buffer.combined_segment_buffer
        self.combined_segment_count = old_buffer.combined_segment_count

        # print("Initial: ",self.combined_segment_count)
        if not threshold_list is None:
            self.add_additional_intervention_labels(threshold_list, error_function)

        self.expert_segment_buffer = {}  # old_buffer.expert_segment_buffer
        self.expert_segment_count = 0  # old_buffer.expert_segment_count

        # self.expert_segment_buffer = old_buffer.expert_segment_buffer
        # self.expert_segment_count = old_buffer.expert_segment_count

        self.preintervention_segment_buffer = {}  # old_buffer.preintervention_segment_buffer#old_buffer.preintervention_segment_buffer
        self.preintervention_segment_count = 0  # old_buffer.preintervention_segment_count#old_buffer.preintervention_segment_count

        self.preintervention_segment_special_buffer = {}  # old_buffer.preintervention_segment_buffer#old_buffer.preintervention_segment_buffer
        self.preintervention_segment_special_count = 0  # old_buffer.preintervention_segment_count#old_buffer.preintervention_segment_count

        self.non_expert_segment_buffer = {}
        self.non_expert_segment_count = 0

        self.non_preintervention_segment_buffer = {}
        self.non_preintervention_segment_count = 0

        self.inbetween_segment_buffer = {}
        self.inbetween_segment_count = 0

        self.special_buffer = {}
        self.special_count = 0

        self.special_buffer_2 = {}
        self.special_count_2 = 0

        self.special_buffer_3 = {}
        self.special_count_3 = 0

        self.expert_exclude_termination_buffer = {}
        self.expert_exclude_termination_count = 0

        self.non_expert_exclude_termination_buffer = {}
        self.non_expert_exclude_termination_count = 0

        self.preintervention_exclude_expert_buffer = {}
        self.preintervention_exclude_expert_count = 0

        self.non_expert_include_termination_single_state_buffer = {}
        self.non_expert_include_termination_single_state_count = 0

        self.expert_exclude_termination_single_state_buffer = {}
        self.expert_exclude_termination_single_state_count = 0

        self.non_expert_single_state_buffer = {}
        self.non_expert_single_state_count = 0

        supervision_threshold_list = []
        for i in range(old_buffer.combined_single_state_count):
            trajectory_index, current_index = old_buffer.combined_single_state_buffer[i]
            s_t, a_t, r_t, terminal_t, intervention_status_t, s_t_1, label_t, _, supervision_threshold_t = \
            self.raw_data_buffer[trajectory_index][current_index]
            if label_t == 0:
                self.non_expert_non_preintervention_single_state_buffer[
                    self.non_expert_non_preintervention_single_state_count] = old_buffer.combined_single_state_buffer[i]
                self.non_expert_non_preintervention_single_state_count += 1

            if intervention_status_t == 0 or label_t == 1:
                self.non_expert_include_termination_single_state_buffer[
                    self.non_expert_include_termination_single_state_count] = old_buffer.combined_single_state_buffer[i]
                self.non_expert_include_termination_single_state_count += 1

            if intervention_status_t == 1 and label_t == -1:
                self.expert_exclude_termination_single_state_buffer[
                    self.expert_exclude_termination_single_state_count] = old_buffer.combined_single_state_buffer[i]
                self.expert_exclude_termination_single_state_count += 1

            if intervention_status_t == 0:
                self.non_expert_single_state_buffer[self.non_expert_single_state_count] = \
                old_buffer.combined_single_state_buffer[i]
                self.non_expert_single_state_count += 1

            supervision_threshold_list.append(supervision_threshold_t)
        max_supervision_level = np.max(supervision_threshold_list)

        self.special_buffer_2_priority = ReplayBufferPriorityIndex(self.args, size=50000)

        self.special_preint_buffer = {}
        self.special_preint_count = {}

        for i in range(self.combined_segment_count):
            segment_label_expert = 0
            segment_label_preintervention = 0
            segment_label_expert_exclude_termination = 0
            segment_label_preintervention_exclude_termination = 0
            supervision_threshold_list = []
            for j in range(self.args.segment_length):
                trajectory_index, current_index = self.combined_segment_buffer[i][j]
                s_t, a_t, r_t, terminal_t, intervention_status_t, s_t_1, label_t, _, supervision_threshold_t = \
                self.raw_data_buffer[trajectory_index][current_index]
                if intervention_status_t == 1:
                    segment_label_expert += 1
                if label_t == 1:
                    segment_label_preintervention += 1

                if label_t == -1:
                    segment_label_expert_exclude_termination += 1

                if label_t == 1 and intervention_status_t == 0:
                    segment_label_preintervention_exclude_termination += 1
                supervision_threshold_list.append(supervision_threshold_t)
            supervision_threshold = int(np.max(supervision_threshold_list))
            # if segment_label_expert != segment_label_expert_exclude_termination:
            #     print(trajectory_index, old_buffer.combined_segment_buffer[i][0][1], old_buffer.combined_segment_buffer[i][-1][1], segment_label_expert_exclude_termination, segment_label_expert, segment_label_preintervention)

            # if segment_label_expert == 0 and segment_label_preintervention < self.args.segment_length:
            #     self.special_buffer[self.special_count] = old_buffer.combined_segment_buffer[i]
            #     self.special_count += 1
            # if segment_label_expert == 0 and segment_label_preintervention < self.args.segment_length:
            #     self.special_buffer[self.special_count] = self.combined_segment_buffer[i]
            #     self.special_count += 1
            # elif segment_label_expert == 0 and segment_label_preintervention == self.args.segment_length:
            #     self.special_buffer[self.special_count] = self.combined_segment_buffer[i]
            #     self.special_count += 1

            if segment_label_expert == self.args.segment_length:
                self.expert_segment_buffer[self.expert_segment_count] = self.combined_segment_buffer[i]
                self.expert_segment_count += 1

            if (segment_label_expert == 0 and (segment_label_preintervention < self.args.segment_length)):
                self.special_buffer[self.special_count] = self.combined_segment_buffer[i]
                self.special_count += 1

            # if segment_label_expert == 0 and segment_label_preintervention == 0 or segment_label_preintervention == self.args.segment_length:
            if (not intervention_status_t == 1 and (segment_label_preintervention < self.args.segment_length)):
                self.special_buffer_2[self.special_count_2] = self.combined_segment_buffer[i]
                self.special_buffer_2_priority.add(self.special_count_2)
                self.special_count_2 += 1

            if (segment_label_expert == 0) and (segment_label_preintervention == 0):
                self.special_buffer_3[self.special_count_3] = self.combined_segment_buffer[i]
                self.special_count_3 += 1

            if segment_label_preintervention == self.args.segment_length:
                self.preintervention_segment_buffer[self.preintervention_segment_count] = self.combined_segment_buffer[
                    i]
                self.preintervention_segment_count += 1

                if not supervision_threshold in self.special_preint_buffer:
                    self.special_preint_buffer[supervision_threshold] = {}
                    self.special_preint_count[supervision_threshold] = 0
                self.special_preint_buffer[supervision_threshold][self.special_preint_count[supervision_threshold]] = \
                self.combined_segment_buffer[i]
                self.special_preint_count[supervision_threshold] += 1

                if not supervision_threshold == max_supervision_level:
                    self.preintervention_segment_special_buffer[self.preintervention_segment_special_count] = \
                    self.combined_segment_buffer[i]
                    self.preintervention_segment_special_count += 1

            if segment_label_expert < self.args.segment_length:
                self.non_expert_segment_buffer[self.non_expert_segment_count] = self.combined_segment_buffer[i]
                self.non_expert_segment_count += 1

            if segment_label_expert_exclude_termination == self.args.segment_length:
                self.expert_exclude_termination_buffer[self.expert_exclude_termination_count] = \
                self.combined_segment_buffer[i]
                self.expert_exclude_termination_count += 1

            if segment_label_expert_exclude_termination == 0:
                self.non_expert_exclude_termination_buffer[self.non_expert_exclude_termination_count] = \
                self.combined_segment_buffer[i]
                self.non_expert_exclude_termination_count += 1

            if segment_label_preintervention < self.args.segment_length:
                self.non_preintervention_segment_buffer[self.non_preintervention_segment_count] = \
                self.combined_segment_buffer[i]
                self.non_preintervention_segment_count += 1

            if segment_label_preintervention == self.args.segment_length and segment_label_expert == 0:
                self.preintervention_exclude_expert_buffer[self.preintervention_exclude_expert_count] = \
                self.combined_segment_buffer[i]
                self.preintervention_exclude_expert_count += 1

            if segment_label_preintervention == 0 and segment_label_expert == 0:
                self.inbetween_segment_buffer[self.inbetween_segment_count] = self.combined_segment_buffer[i]
                self.inbetween_segment_count += 1

        # print("add special ... " ,self.special_count, self.combined_segment_count, self.preintervention_segment_count)

    def get_state_distribution(self, mode=0, atoms=30):
        s_t, a_t, _, _, _, _, _, _, supervision_threshold_t = self.raw_data_buffer[0][0]
        state_distribution = np.zeros((s_t.shape[0], atoms))
        for i in range(self.combined_single_state_count):
            trajectory_index, current_index = self.combined_single_state_buffer[i]
            s_t, a_t, r_t, terminal_t, intervention_status_t, s_t_1, label_t, _, supervision_threshold_t = \
            self.raw_data_buffer[trajectory_index][current_index]

            s_t_one_hot = np.round((np.clip(s_t, -1, 1) + 1) / 2 * (atoms - 1)).astype(np.int32)
            if intervention_status_t == mode:  # 0 non expert, 1 expert
                for j in range(state_distribution.shape[0]):
                    state_distribution[j, s_t_one_hot[j]] += 1
        action_distribution = state_distribution / np.repeat(np.expand_dims(np.sum(state_distribution, axis=1), axis=1),
                                                             atoms, axis=1)
        return action_distribution

    def get_action_distribution(self, mode=0, atoms=30):
        s_t, a_t, _, _, _, _, _, _, supervision_threshold_t = self.raw_data_buffer[0][0]
        action_distribution = np.zeros((a_t.shape[0], atoms))
        for i in range(self.combined_single_state_count):
            trajectory_index, current_index = self.combined_single_state_buffer[i]
            s_t, a_t, r_t, terminal_t, intervention_status_t, s_t_1, label_t, _, supervision_threshold_t = \
            self.raw_data_buffer[trajectory_index][current_index]

            a_t_one_hot = np.round((a_t + 1) / 2 * (atoms - 1)).astype(np.int32)
            if intervention_status_t == mode:  # 0 non expert, 1 expert
                for j in range(action_distribution.shape[0]):
                    action_distribution[j, a_t_one_hot[j]] += 1
        action_distribution = action_distribution / np.repeat(
            np.expand_dims(np.sum(action_distribution, axis=1), axis=1), atoms, axis=1)
        return action_distribution

    def get_uniform_action_distribution(self, data=100000, atoms=30):
        _, a_t, _, _, _, _, _, _, supervision_threshold_t = self.raw_data_buffer[0][0]
        action_distribution = np.zeros((a_t.shape[0], atoms))
        for i in range(100000):
            a_t = np.random.uniform(-1, 1, size=a_t.shape)
            a_t_one_hot = np.round((a_t + 1) / 2 * (atoms - 1)).astype(np.int32)
            for j in range(action_distribution.shape[0]):
                action_distribution[j, a_t_one_hot[j]] += 1
        action_distribution = action_distribution / np.repeat(
            np.expand_dims(np.sum(action_distribution, axis=1), axis=1), atoms, axis=1)
        return action_distribution

    def generate_part_expert_data(self, partial_percent=0.8):

        self.partial_expert_buffer = {}
        self.partial_expert_count = 0

        for i in self.expert_single_state_count:
            if np.random.uniform(0, 1) < partial_percent:
                self.partial_expert_buffer[self.partial_expert_count] = self.expert_single_state_buffer[i]
                self.partial_expert_count += 1

    def add_additional_intervention_labels(self, supervision_thresholds, error_function, additional_labels=500):

        # print("Was here ... ")

        add_count = 0
        n_perm = np.random.permutation(self.combined_segment_count)
        counter = 0
        while add_count < additional_labels:
            if counter == self.combined_segment_count:
                print(("ERROR, not enough data ... cannot add more labels"))
                break

            # randomly sample trajectory segments
            sampled_indx = n_perm[counter]
            counter += 1

            trajectory_segment_indices = self.combined_segment_buffer[sampled_indx]

            # print(len(trajectory_segment_indices), trajectory_segment_indices)

            segment_error = 0
            segment_label_preintervention = 0
            segment_label_expert = 0
            intervention_thresold_list = []
            for j in range(self.args.segment_length):
                s_t, a_t, r_t, terminal_t, intervention_status_t, s_t_1, label_t, expert_a_t, supervision_threshold_t = \
                    self.raw_data_buffer[trajectory_segment_indices[j][0]][trajectory_segment_indices[j][1]]

                if label_t == 1:
                    segment_label_preintervention += 1

                if label_t == -1:
                    segment_label_expert += 1

                expert_error, raw_error = error_function(a_t, expert_a_t)
                segment_error += expert_error
                intervention_thresold_list.append(supervision_threshold_t)

            if segment_label_preintervention == self.args.segment_length:
                continue  # skip
            # if segment_label_expert == self.args.segment_length:
            #     continue #skip

            for threshold in supervision_thresholds:
                if segment_error >= threshold and segment_error < threshold + 2:
                    add_count += 1
                    # print(add_count, segment_error, threshold)

                    for j in range(self.args.segment_length):
                        s_t, a_t, r_t, terminal_t, intervention_status_t, s_t_1, label_t, expert_a_t, supervision_threshold_t = \
                            self.raw_data_buffer[trajectory_segment_indices[j][0]][trajectory_segment_indices[j][1]]

                        self.add(np.copy(s_t), np.copy(a_t), r_t, terminal_t, intervention_status_t, np.copy(s_t_1), 1,
                                 expert_a_t, threshold)
                    self.trajectory_end()
                    break

            # print("expert error: ", segment_error, threshold, add_count)
            # quit()

        # quit()

        # if terminal_t == 1:
        #     total_cost = 0
        #     for i in range(self.args.segment_length):
        #         s_t, a_t, r_t, terminal_t, intervention_status_t, s_t_1, label_t, expert_a_t, intervention_threshold = \
        #             self.raw_data_buffer[trajectory_segment_indices[i][0]][trajectory_segment_indices[i][1]]
        #         action_cost = np.mean(np.clip(np.abs(a_t - expert_a_t), 0, 0.75)/0.75)
        #         total_cost += action_cost
        #     new_intervention_threshold = None
        #
        #     supervision_thresholds.reverse()
        #     for threshold in supervision_thresholds:
        #         if threshold < total_cost:
        #             new_intervention_threshold = threshold
        #             break
        #
        #     if not new_intervention_threshold is None:
        #         add_count += 1
        #         print("Added", add_count, new_intervention_threshold)
        #         for i in range(self.args.segment_length):
        #             self.raw_data_buffer[trajectory_segment_indices[i][0]][trajectory_segment_indices[i][1]] = \
        #             (s_t, a_t, r_t, terminal_t, intervention_status_t, s_t_1, -1, expert_a_t, new_intervention_threshold)
        #

    # def add_terminal_preference_labels(self, supervision_thresholds):
    #     terminal_count = 0
    #     loop_count = 0
    #     add_count = 0
    #     for indx in self.combined_segment_buffer:
    #         trajectory_segment_indices = self.combined_segment_buffer[indx]
    #
    #         #print(len(trajectory_segment_indices), trajectory_segment_indices)
    #
    #         s_t, a_t, r_t, terminal_t, intervention_status_t, s_t_1, label_t, expert_a_t, intervention_threshold = \
    #             self.raw_data_buffer[trajectory_segment_indices[-1][0]][trajectory_segment_indices[-1][1]]
    #
    #         if terminal_t == 1:
    #             total_cost = 0
    #             for i in range(self.args.segment_length):
    #                 s_t, a_t, r_t, terminal_t, intervention_status_t, s_t_1, label_t, expert_a_t, intervention_threshold = \
    #                     self.raw_data_buffer[trajectory_segment_indices[i][0]][trajectory_segment_indices[i][1]]
    #                 action_cost = np.mean(np.clip(np.abs(a_t - expert_a_t), 0, 0.75)/0.75)
    #                 total_cost += action_cost
    #
    #             print(total_cost)
    #             new_intervention_threshold = None
    #
    #             supervision_thresholds.reverse()
    #             for threshold in supervision_thresholds:
    #                 if threshold < total_cost:
    #                     new_intervention_threshold = threshold
    #                     break
    #
    #             if not new_intervention_threshold is None:
    #                 add_count += 1
    #                 print("Added", add_count, new_intervention_threshold)
    #                 for i in range(self.args.segment_length):
    #                     self.raw_data_buffer[trajectory_segment_indices[i][0]][trajectory_segment_indices[i][1]] = \
    #                     (s_t, a_t, r_t, terminal_t, intervention_status_t, s_t_1, -1, expert_a_t, new_intervention_threshold)

    def verify_preint(self, error_function):
        error_count = 0
        for i in range(self.preintervention_segment_count):
            outcome = self.verify_segment(self.preintervention_segment_buffer[i], error_function)
            if outcome is False:
                error_count += 1
        print("Preint Error: ", error_count / self.preintervention_segment_count)

    def verify_inbetween(self, error_function):
        error_count = 0
        for i in range(self.special_count_2):
            outcome = self.verify_segment(self.special_buffer_2[i], error_function)
            if outcome is False:
                error_count += 1
        print("Inbetween Error: ", error_count / self.special_count_2)

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

            if np.mean(true_error) > 0.4:
                print(i, np.mean(true_error), np.mean(expert_threshold_list), preint_list)

    def verify_segment(self, segment_indices, error_function):
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
        if np.mean(preint_list) == 1:
            if np.sum(true_error_list) >= supervision_threshold and np.sum(true_error_list) < supervision_threshold + 2:
                return True
            else:
                # if supervision_threshold > np.sum(true_error_list) + 1:
                #     print(np.sum(true_error_list), supervision_threshold, np.mean(intervention_threshold_list))
                return False
        else:
            if np.sum(true_error_list) <= supervision_threshold:
                return True
            else:
                print(np.sum(true_error_list), supervision_threshold, np.mean(expert_threshold_list),
                      np.mean(preint_list))
                for i in range(self.args.segment_length):
                    print(i, true_error_list[i], intervention_status_list[i], intervention_threshold_list[i],
                          preint_list[i])
                quit()

                return False

        # print(self.non_preintervention_non_intervention_count, self.special_count, self.preintervention_segment_count, self.expert_segment_count, self.partial_preintervention_segment_count, self.combined_segment_count)

    def add(self, s_t, a_t, r_t, terminal_t, intervention_status_t, s_t_1, label_t, expert_a_t, intervention_threshold):
        if not self.raw_segment_count in self.raw_data_buffer:
            self.raw_data_buffer[self.raw_segment_count] = {}

        data_entry = [s_t, a_t, r_t, terminal_t, intervention_status_t, s_t_1, label_t, expert_a_t,
                      intervention_threshold]
        # print(s_t_1.shape, s_t_1)
        current_trajectory_index = len(self.raw_data_buffer[self.raw_segment_count])
        self.raw_data_buffer[self.raw_segment_count][current_trajectory_index] = data_entry
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
        self.raw_data_count += 1

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
        elif mode == 4:
            buffer = self.non_expert_include_termination_single_state_buffer
            count = self.non_expert_include_termination_single_state_count
        elif mode == 5:
            buffer = self.expert_exclude_termination_single_state_buffer
            count = self.expert_exclude_termination_single_state_count
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

    def add_segment(self, current_trajectory, current_trajectory_index, current_index):
        segment_label_expert = 0
        segment_label_preintervention = 0
        segment_label_expert_exclude_termination = 0
        segment_indices = []

        for j in range(self.args.segment_length):
            added_index = current_index + j
            s_t, a_t, r_t, terminal_t, intervention_status_t, s_t_1, label_t, _, _ = current_trajectory[added_index]
            if label_t == -1 or intervention_status_t == 1:
                segment_label_expert += 1
            if label_t == 1:
                segment_label_preintervention += 1

            if label_t == -1:
                segment_label_expert_exclude_termination += 1

            segment_indices.append([current_trajectory_index, added_index])

        self.combined_segment_buffer[self.combined_segment_count] = segment_indices
        self.combined_segment_count += 1

        if segment_label_expert == self.args.segment_length:
            self.expert_segment_buffer[self.expert_segment_count] = segment_indices
            self.expert_segment_count += 1

        if segment_label_preintervention == self.args.segment_length:
            self.preintervention_segment_buffer[self.preintervention_segment_count] = segment_indices
            self.preintervention_segment_count += 1

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

    def sample_preint(self, batch_size, supervision_thresholds, is_preint, preint_include_eq=False, all_legal=False):
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
                if is_preint[i] and supervision_thresholds[i] == current_supervision_threshold:
                    random_actions = True

            buffer = self.special_preint_buffer[current_supervision_threshold]
            count = self.special_preint_count[current_supervision_threshold]
            sampled_index = np.random.randint(0, count)

            # print(current_supervision_threshold, count, sampled_index)
            # if all_legal and np.random.uniform(0, 1) < 0.2:
            #     random_actions = True

            raw_buffer_segment_indices = buffer[sampled_index]
            mean_label = 0
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
            mean_label = mean_label / self.args.segment_length
            segment_mean_label_list.append(mean_label)

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

        return s_t_list, a_t_list, r_t_list, terminal_t_list, intervention_status_t_list, \
               s_t_1_list, label_t_list, label_t_raw_list, expert_a_t_list, intervention_threshold_list

    def update_priority(self, update_indices, loss):
        self.special_buffer_2_priority.update_priorities(update_indices, loss)

    def sample_priority(self, batch_size):
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

        buffer = self.special_buffer_2
        sampled_indices, update_indices, weight = self.special_buffer_2_priority.sample_priority(batch_size)
        sampled_indices = np.squeeze(sampled_indices)
        for i in range(batch_size):
            raw_buffer_segment_indices = buffer[sampled_indices[i]]

            mean_label = 0
            for j in range(self.args.segment_length):
                s_t, a_t, r_t, terminal_t, intervention_status_t, s_t_1, label_t, expert_a_t, intervention_threshold = \
                self.raw_data_buffer[raw_buffer_segment_indices[j][0]][raw_buffer_segment_indices[j][1]]

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
        return s_t_list, a_t_list, r_t_list, terminal_t_list, intervention_status_t_list, \
               s_t_1_list, label_t_list, label_t_raw_list, expert_a_t_list, intervention_threshold_list, \
               update_indices, weight

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

        for i in range(len(mode_list)):
            s_t, a_t, r_t, terminal_t, \
            intervention_status_t, s_t_1, label_t, \
            label_t_raw, expert_a_t, intervention_threshold = self.sample(mode_batch_size_list[i], mode_list[i])

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

        return s_t_list, a_t_list, r_t_list, terminal_t_list, intervention_status_t_list, s_t_1_list, label_t_list, label_t_raw_list, expert_a_t_list, intervention_threshold_list

    def sample(self, batch_size, mode=0):
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
            buffer = self.special_buffer
            count = self.special_count
        elif mode == 6:
            buffer = self.expert_exclude_termination_buffer
            count = self.expert_exclude_termination_count
        elif mode == 7:
            buffer = self.non_expert_exclude_termination_buffer
            count = self.non_expert_exclude_termination_count
        elif mode == 8:
            buffer = self.preintervention_exclude_expert_buffer
            count = self.preintervention_exclude_expert_count
        elif mode == 9:
            buffer = self.special_buffer_2
            count = self.special_count_2
        elif mode == 10:
            buffer = self.special_buffer_3
            count = self.special_count_3
        elif mode == 11:
            buffer = self.inbetween_segment_buffer
            count = self.inbetween_segment_count
        elif mode == 12:
            buffer = self.preintervention_segment_special_buffer
            count = self.preintervention_segment_special_count

        sampled_indices = np.random.randint(0, count, size=(batch_size))
        for i in range(batch_size):
            raw_buffer_segment_indices = buffer[sampled_indices[i]]

            mean_label = 0
            for j in range(self.args.segment_length):
                s_t, a_t, r_t, terminal_t, intervention_status_t, s_t_1, label_t, expert_a_t, intervention_threshold = \
                self.raw_data_buffer[raw_buffer_segment_indices[j][0]][raw_buffer_segment_indices[j][1]]

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
        return s_t_list, a_t_list, r_t_list, terminal_t_list, intervention_status_t_list, s_t_1_list, label_t_list, label_t_raw_list, expert_a_t_list, intervention_threshold_list

    def trajectory_end(self):
        current_trajectory = self.raw_data_buffer[self.raw_segment_count]
        for i in range(len(current_trajectory)):
            if i + self.args.segment_length <= len(current_trajectory):
                # print("Was here ... ")
                self.add_segment(current_trajectory, self.raw_segment_count, i)
            else:
                break
        self.raw_segment_count += 1


class ReplayBufferPriorityIndex:
    def __init__(self, args, size=1000000):
        self.buffer = {}
        self.args = args
        self.size = size
        self.count = 0
        self.add_index = 0

        alpha = args.alpha
        self._beta = args.beta

        assert alpha >= 0
        self._alpha = alpha

        it_capacity = 1
        while it_capacity < size:
            it_capacity *= 2

        self._it_sum = SumSegmentTree(it_capacity)
        self._it_min = MinSegmentTree(it_capacity)
        self._max_priority = 5

    def add(self, index):
        # state = state.astype(np.uint8)
        # next_state = next_state.astype(np.uint8)
        self.count = min(self.count + 1, self.size)
        self.buffer[self.add_index] = [index]
        self._it_sum[self.add_index] = self._max_priority ** self._alpha
        self._it_min[self.add_index] = self._max_priority ** self._alpha
        self.add_index = (self.add_index + 1) % self.size

    def sample(self, batch_size=None, no_reshape=True):
        if batch_size is None:
            batch_size = self.args.batch_size

        indices = self._get_random_indices(batch_size)
        samples = []
        for i in range(batch_size):
            samples.append(self.buffer[indices[i]])
        sampled_index = samples
        return sampled_index

    def sample_priority(self, batch_size):
        indices = self._sample_proportional(batch_size)
        samples = []
        for i in range(batch_size):
            samples.append(self.buffer[indices[i]])
        sampled_index = samples
        total_weight = self._it_sum.sum()
        weights = []
        p_min = self._it_min.min() / total_weight
        max_weight = (p_min * self.count) ** (-self._beta)

        for idx in indices:
            p_sample = self._it_sum[idx] / total_weight
            weight = (p_sample * self.count) ** (-self._beta)
            weights.append(weight / max_weight)
        weights = np.array(weights)

        return sampled_index, indices, weights

    def size(self):
        return len(self.buffer)

    def _get_random_indices(self, batch_size):
        idxes = []
        while len(idxes) < batch_size:
            index = np.random.randint(0, self.count - 1)
            idxes.append(index)
        return idxes

    def _sample_proportional(self, batch_size):
        res = []
        p_total = self._it_sum.sum(0, self.count - 1)
        every_range_len = p_total / batch_size
        i = 0
        count = 0
        while len(res) < batch_size:
            count += 1
            if count > 10000:
                idx = np.random.randint(0, self.count)
            else:
                mass = random.random() * every_range_len + i * every_range_len
                idx = self._it_sum.find_prefixsum_idx(mass)
            # if idx < self.agent_history_length + 1:
            #     continue
            # if idx >= self.expert_idx and idx - self.agent_history_length < self.expert_idx:
            #     continue
            # if idx >= self._next_idx and idx - self.agent_history_length < self._next_idx:
            #     continue
            # if np.sum(self.terminal_flags[idx - self.agent_history_length:idx]) > 0:
            #     # if count > 9900:
            #     #     print(self.terminal_flags[idx - 25:idx + 25])
            #     continue
            res.append(idx)
            i += 1
        return res

    def update_priorities(self, idxes, priorities, min_priority=0.001):
        """Update priorities of sampled transitions.
        sets priority of transition at index idxes[i] in buffer
        to priorities[i].
        Parameters
        ----------
        idxes: [int]
            List of idxes of sampled transitions
        priorities: [float]
            List of updated priorities corresponding to
            transitions at the sampled idxes denoted by
            variable `idxes`.
        """
        # Boost expert priority as time goes on ....
        priorities = np.squeeze(priorities)
        assert len(idxes) == priorities.shape[0]
        count = 0
        # print(expert_priority_modifier)
        for idx, priority in zip(idxes, priorities):
            priority = max(priority, 0)
            assert 0 <= idx < self.count

            new_priority = priority * (1 - min_priority) + min_priority
            self._max_priority = max(self._max_priority, new_priority)
            # print(idx, new_priority, expert_idxes[count], count)
            self._it_sum[idx] = (new_priority) ** self._alpha
            self._it_min[idx] = (new_priority) ** self._alpha
            count += 1


class ReplayBufferPriority:
    def __init__(self, args, size=1000000):
        self.buffer = {}
        self.args = args
        self.size = size
        self.count = 0
        self.add_index = 0

        alpha = args.alpha
        self._beta = args.beta

        assert alpha >= 0
        self._alpha = alpha

        it_capacity = 1
        while it_capacity < size:
            it_capacity *= 2

        self._it_sum = SumSegmentTree(it_capacity)
        self._it_min = MinSegmentTree(it_capacity)
        self._max_priority = 5

    def add(self, states, actions, rewards, done, next_states):
        # state = state.astype(np.uint8)
        # next_state = next_state.astype(np.uint8)
        self.count = min(self.count + 1, self.size)
        self.buffer[self.add_index] = [states, actions, rewards, next_states, done]
        self._it_sum[self.add_index] = self._max_priority ** self._alpha
        self._it_min[self.add_index] = self._max_priority ** self._alpha
        self.add_index = (self.add_index + 1) % self.size

    def sample(self, batch_size=None, no_reshape=True):
        if batch_size is None:
            batch_size = self.args.batch_size

        indices = self._get_random_indices(batch_size)
        samples = []
        for i in range(batch_size):
            samples.append(self.buffer[indices[i]])
        states, actions, rewards, next_states, done = map(np.asarray, zip(*samples))
        if no_reshape:
            states = np.array(states).reshape(batch_size, -1)
            next_states = np.array(next_states).reshape(batch_size, -1)
        return states, next_states, actions, rewards, done

    def sample_priority(self, no_reshape=True):
        indices = self._sample_proportional(self.args.batch_size)
        samples = []
        for i in range(self.args.batch_size):
            samples.append(self.buffer[indices[i]])
        states, actions, rewards, next_states, done = map(np.asarray, zip(*samples))
        total_weight = self._it_sum.sum()
        weights = []
        p_min = self._it_min.min() / total_weight
        max_weight = (p_min * self.count) ** (-self._beta)

        for idx in indices:
            p_sample = self._it_sum[idx] / total_weight
            weight = (p_sample * self.count) ** (-self._beta)
            weights.append(weight / max_weight)
        weights = np.array(weights)

        if no_reshape:
            states = np.array(states).reshape(self.args.batch_size, -1)
            next_states = np.array(next_states).reshape(self.args.batch_size, -1)

        return states, next_states, actions, rewards, done, indices, weights

    def size(self):
        return len(self.buffer)

    def _get_random_indices(self, batch_size):
        idxes = []
        while len(idxes) < batch_size:
            index = np.random.randint(0, self.count - 1)
            idxes.append(index)
        return idxes

    def _sample_proportional(self, batch_size):
        res = []
        p_total = self._it_sum.sum(0, self.count - 1)
        every_range_len = p_total / batch_size
        i = 0
        count = 0
        while len(res) < batch_size:
            count += 1
            if count > 10000:
                idx = np.random.randint(0, self.count)
            else:
                mass = random.random() * every_range_len + i * every_range_len
                idx = self._it_sum.find_prefixsum_idx(mass)
            # if idx < self.agent_history_length + 1:
            #     continue
            # if idx >= self.expert_idx and idx - self.agent_history_length < self.expert_idx:
            #     continue
            # if idx >= self._next_idx and idx - self.agent_history_length < self._next_idx:
            #     continue
            # if np.sum(self.terminal_flags[idx - self.agent_history_length:idx]) > 0:
            #     # if count > 9900:
            #     #     print(self.terminal_flags[idx - 25:idx + 25])
            #     continue
            res.append(idx)
            i += 1
        return res

    def update_priorities(self, idxes, priorities, min_priority=0.001):
        """Update priorities of sampled transitions.
        sets priority of transition at index idxes[i] in buffer
        to priorities[i].
        Parameters
        ----------
        idxes: [int]
            List of idxes of sampled transitions
        priorities: [float]
            List of updated priorities corresponding to
            transitions at the sampled idxes denoted by
            variable `idxes`.
        """
        # Boost expert priority as time goes on ....
        priorities = np.squeeze(priorities)
        assert len(idxes) == priorities.shape[0]
        count = 0
        # print(expert_priority_modifier)
        for idx, priority in zip(idxes, priorities):
            priority = max(priority, 0)
            assert 0 <= idx < self.count

            new_priority = priority * (1 - min_priority) + min_priority
            self._max_priority = max(self._max_priority, new_priority)
            # print(idx, new_priority, expert_idxes[count], count)
            self._it_sum[idx] = (new_priority) ** self._alpha
            self._it_min[idx] = (new_priority) ** self._alpha
            count += 1


class ReplayBufferPriorityAccCost:
    def __init__(self, args, size=1000000):
        self.buffer = {}
        self.args = args
        self.size = size
        self.count = 0
        self.add_index = 0

        alpha = args.alpha
        self._beta = args.beta

        assert alpha >= 0
        self._alpha = alpha

        it_capacity = 1
        while it_capacity < size:
            it_capacity *= 2

        self._it_sum = SumSegmentTree(it_capacity)
        self._it_min = MinSegmentTree(it_capacity)
        self._max_priority = 5

    def add(self, states_segment, actions_segment, rewards_segment, terminal_segment, next_states_segment, mask,
            segment_status):
        # state = state.astype(np.uint8)
        # next_state = next_state.astype(np.uint8)
        self.count = min(self.count + 1, self.size)
        self.buffer[self.add_index] = [states_segment, actions_segment, rewards_segment, next_states_segment,
                                       terminal_segment, mask, segment_status]
        self._it_sum[self.add_index] = self._max_priority ** self._alpha
        self._it_min[self.add_index] = self._max_priority ** self._alpha

        self.add_index = (self.add_index + 1) % self.size

    def sample(self, batch_size, segment_length=None):
        indices = self._get_random_indices(batch_size)
        samples = []

        if segment_length is None:
            segment_length = self.args.segment_length

        for i in range(batch_size):
            samples.append(self.buffer[indices[i]])
        states_segment, actions_segment, rewards_segment, next_states_segment, terminal_segment, mask, segment_status = map(
            np.asarray, zip(*samples))

        states_segment = np.array(states_segment).reshape(batch_size, segment_length, -1)
        next_states_segment = np.array(next_states_segment).reshape(batch_size, segment_length, -1)

        actions_segment = np.array(actions_segment).reshape(batch_size, segment_length, -1)
        rewards_segment = np.array(rewards_segment).reshape(batch_size, segment_length, -1)
        terminal_segment = np.array(terminal_segment).reshape(batch_size, segment_length, -1)

        return states_segment, next_states_segment, actions_segment, rewards_segment, terminal_segment, mask, segment_status

    def sample_priority(self, batch_size):
        indices = self._sample_proportional(batch_size)
        samples = []
        for i in range(batch_size):
            samples.append(self.buffer[indices[i]])
        states_segment, actions_segment, rewards_segment, next_states_segment, terminal_segment, mask, segment_status = map(
            np.asarray, zip(*samples))
        total_weight = self._it_sum.sum()
        weights = []
        p_min = self._it_min.min() / total_weight
        max_weight = (p_min * self.count) ** (-self._beta)

        for idx in indices:
            p_sample = self._it_sum[idx] / total_weight
            weight = (p_sample * self.count) ** (-self._beta)
            weights.append(weight / max_weight)
        weights = np.array(weights)

        states_segment = np.array(states_segment).reshape(batch_size, self.args.segment_length, -1)
        next_states_segment = np.array(next_states_segment).reshape(batch_size, self.args.segment_length, -1)

        actions_segment = np.array(actions_segment).reshape(batch_size, self.args.segment_length, -1)
        rewards_segment = np.array(rewards_segment).reshape(batch_size, self.args.segment_length, -1)
        terminal_segment = np.array(terminal_segment).reshape(batch_size, self.args.segment_length, -1)

        return states_segment, next_states_segment, actions_segment, rewards_segment, terminal_segment, mask, segment_status, indices, weights

    def size(self):
        return len(self.buffer)

    def _get_random_indices(self, batch_size):
        idxes = []
        while len(idxes) < batch_size:
            index = np.random.randint(0, self.count - 1)
            idxes.append(index)
        return idxes

    def _sample_proportional(self, batch_size):
        res = []
        p_total = self._it_sum.sum(0, self.count - 1)
        every_range_len = p_total / batch_size
        i = 0
        count = 0
        while len(res) < batch_size:
            count += 1
            if count > 10000:
                idx = np.random.randint(0, self.count)
            else:
                mass = random.random() * every_range_len + i * every_range_len
                idx = self._it_sum.find_prefixsum_idx(mass)
            # if idx < self.agent_history_length + 1:
            #     continue
            # if idx >= self.expert_idx and idx - self.agent_history_length < self.expert_idx:
            #     continue
            # if idx >= self._next_idx and idx - self.agent_history_length < self._next_idx:
            #     continue
            # if np.sum(self.terminal_flags[idx - self.agent_history_length:idx]) > 0:
            #     # if count > 9900:
            #     #     print(self.terminal_flags[idx - 25:idx + 25])
            #     continue
            res.append(idx)
            i += 1
        return res

    def update_priorities(self, idxes, priorities, min_priority=0.001):
        """Update priorities of sampled transitions.
        sets priority of transition at index idxes[i] in buffer
        to priorities[i].
        Parameters
        ----------
        idxes: [int]
            List of idxes of sampled transitions
        priorities: [float]
            List of updated priorities corresponding to
            transitions at the sampled idxes denoted by
            variable `idxes`.
        """
        # Boost expert priority as time goes on ....
        priorities = np.squeeze(priorities)
        assert len(idxes) == priorities.shape[0]
        count = 0
        # print(expert_priority_modifier)
        for idx, priority in zip(idxes, priorities):
            priority = max(priority, 0)
            assert 0 <= idx < self.count

            new_priority = priority * (1 - min_priority) + min_priority
            self._max_priority = max(self._max_priority, new_priority)
            # print(idx, new_priority, expert_idxes[count], count)
            self._it_sum[idx] = (new_priority) ** self._alpha
            self._it_min[idx] = (new_priority) ** self._alpha
            count += 1


class ReplayBufferPriorityEIL:
    def __init__(self, args, size=1000000):
        self.buffer = {}
        self.args = args
        self.size = size
        self.count = 0
        self.add_index = 0

        alpha = args.alpha
        self._beta = args.beta

        assert alpha >= 0
        self._alpha = alpha

        it_capacity = 1
        while it_capacity < size:
            it_capacity *= 2

        self._it_sum = SumSegmentTree(it_capacity)
        self._it_min = MinSegmentTree(it_capacity)
        self._max_priority = 5

    def add(self, states, actions, rewards, done, next_states, state_status):
        # state = state.astype(np.uint8)
        # next_state = next_state.astype(np.uint8)
        self.count = min(self.count + 1, self.size)
        self.buffer[self.add_index] = [states, actions, rewards, next_states, done, state_status]
        self._it_sum[self.add_index] = self._max_priority ** self._alpha
        self._it_min[self.add_index] = self._max_priority ** self._alpha

        self.add_index = (self.add_index + 1) % self.size

    def sample(self):
        indices = self._get_random_indices(self.args.batch_size)
        samples = []
        for i in range(self.args.batch_size):
            samples.append(self.buffer[indices[i]])
        states, actions, rewards, next_states, done, state_status = map(np.asarray, zip(*samples))
        states = np.array(states).reshape(self.args.batch_size, -1)
        next_states = np.array(next_states).reshape(self.args.batch_size, -1)
        return states, next_states, actions, rewards, done, state_status

    def sample_priority(self):
        indices = self._sample_proportional(self.args.batch_size)
        samples = []
        for i in range(self.args.batch_size):
            samples.append(self.buffer[indices[i]])
        states, actions, rewards, next_states, done, state_status = map(np.asarray, zip(*samples))
        total_weight = self._it_sum.sum()
        weights = []
        p_min = self._it_min.min() / total_weight
        max_weight = (p_min * self.count) ** (-self._beta)

        for idx in indices:
            p_sample = self._it_sum[idx] / total_weight
            weight = (p_sample * self.count) ** (-self._beta)
            weights.append(weight / max_weight)
        weights = np.array(weights)

        states = np.array(states).reshape(self.args.batch_size, -1)
        next_states = np.array(next_states).reshape(self.args.batch_size, -1)

        return states, next_states, actions, rewards, done, state_status, indices, weights

    def size(self):
        return len(self.buffer)

    def _get_random_indices(self, batch_size):
        idxes = []
        while len(idxes) < batch_size:
            index = np.random.randint(0, self.count - 1)
            idxes.append(index)
        return idxes

    def _sample_proportional(self, batch_size):
        res = []
        p_total = self._it_sum.sum(0, self.count - 1)
        every_range_len = p_total / batch_size
        i = 0
        count = 0
        while len(res) < batch_size:
            count += 1
            if count > 10000:
                idx = np.random.randint(0, self.count)
            else:
                mass = random.random() * every_range_len + i * every_range_len
                idx = self._it_sum.find_prefixsum_idx(mass)
            # if idx < self.agent_history_length + 1:
            #     continue
            # if idx >= self.expert_idx and idx - self.agent_history_length < self.expert_idx:
            #     continue
            # if idx >= self._next_idx and idx - self.agent_history_length < self._next_idx:
            #     continue
            # if np.sum(self.terminal_flags[idx - self.agent_history_length:idx]) > 0:
            #     # if count > 9900:
            #     #     print(self.terminal_flags[idx - 25:idx + 25])
            #     continue
            res.append(idx)
            i += 1
        return res

    def update_priorities(self, idxes, priorities, min_priority=0.001):
        """Update priorities of sampled transitions.
        sets priority of transition at index idxes[i] in buffer
        to priorities[i].
        Parameters
        ----------
        idxes: [int]
            List of idxes of sampled transitions
        priorities: [float]
            List of updated priorities corresponding to
            transitions at the sampled idxes denoted by
            variable `idxes`.
        """
        # Boost expert priority as time goes on ....
        priorities = np.squeeze(priorities)
        assert len(idxes) == priorities.shape[0]
        count = 0
        # print(expert_priority_modifier)
        for idx, priority in zip(idxes, priorities):
            priority = max(priority, 0)
            assert 0 <= idx < self.count

            new_priority = priority * (1 - min_priority) + min_priority
            self._max_priority = max(self._max_priority, new_priority)
            # print(idx, new_priority, expert_idxes[count], count)
            self._it_sum[idx] = (new_priority) ** self._alpha
            self._it_min[idx] = (new_priority) ** self._alpha
            count += 1


class ReplayBufferPixelPriority:
    def __init__(self, args, size=1000000):
        self.buffer = {}
        self.args = args
        self.size = size
        self.count = 0
        self.add_index = 0

        alpha = args.alpha
        self._beta = args.beta

        assert alpha >= 0
        self._alpha = alpha

        it_capacity = 1
        while it_capacity < size:
            it_capacity *= 2

        self._it_sum = SumSegmentTree(it_capacity)
        self._it_min = MinSegmentTree(it_capacity)
        self._max_priority = 5

    def add(self, state, action, action_mean, action_std, prev_action, reward, done, next_state):
        # state = state.astype(np.uint8)
        # next_state = next_state.astype(np.uint8)
        self.count = min(self.count + 1, self.size)
        self.buffer[self.add_index] = [state[0], state[1], action, action_mean, action_std, prev_action, reward,
                                       next_state[0], next_state[1], done]
        self._it_sum[self.add_index] = self._max_priority ** self._alpha
        self._it_min[self.add_index] = self._max_priority ** self._alpha

        self.add_index = (self.add_index + 1) % self.size

    def sample(self):
        indices = self._get_random_indices(self.args.batch_size)
        samples = []
        for i in range(self.args.batch_size):
            samples.append(self.buffer[indices[i]])
        states_image, states_dense, actions, action_mean, action_std, prev_action, rewards, next_states_image, next_states_dense, done = map(
            np.asarray, zip(*samples))

        states_image = states_image / 127.5 - 1
        next_states_image = next_states_image / 127.5 - 1
        return [states_image, states_dense], [next_states_image,
                                              next_states_dense], actions, action_mean, action_std, prev_action, rewards, done

    def sample_priority(self):

        indices = self._sample_proportional(self.args.batch_size)
        samples = []
        for i in range(self.args.batch_size):
            samples.append(self.buffer[indices[i]])
        states_image, states_dense, actions, action_mean, action_std, prev_action, rewards, next_states_image, next_states_dense, done = map(
            np.asarray, zip(*samples))

        total_weight = self._it_sum.sum()
        weights = []
        p_min = self._it_min.min() / total_weight
        max_weight = (p_min * self.count) ** (-self._beta)

        for idx in indices:
            p_sample = self._it_sum[idx] / total_weight
            weight = (p_sample * self.count) ** (-self._beta)
            weights.append(weight / max_weight)
        weights = np.array(weights)

        states_image = states_image / 127.5 - 1
        next_states_image = next_states_image / 127.5 - 1
        return [states_image, states_dense], [next_states_image,
                                              next_states_dense], actions, action_mean, action_std, prev_action, rewards, done, indices, weights

    def size(self):
        return len(self.buffer)

    def _get_random_indices(self, batch_size):
        idxes = []
        while len(idxes) < batch_size:
            index = np.random.randint(0, self.count - 1)
            idxes.append(index)
        return idxes

    def _sample_proportional(self, batch_size):
        res = []
        p_total = self._it_sum.sum(0, self.count - 1)
        every_range_len = p_total / batch_size
        i = 0
        count = 0
        while len(res) < batch_size:
            count += 1
            if count > 10000:
                idx = np.random.randint(0, self.count)
            else:
                mass = random.random() * every_range_len + i * every_range_len
                idx = self._it_sum.find_prefixsum_idx(mass)
            # if idx < self.agent_history_length + 1:
            #     continue
            # if idx >= self.expert_idx and idx - self.agent_history_length < self.expert_idx:
            #     continue
            # if idx >= self._next_idx and idx - self.agent_history_length < self._next_idx:
            #     continue
            # if np.sum(self.terminal_flags[idx - self.agent_history_length:idx]) > 0:
            #     # if count > 9900:
            #     #     print(self.terminal_flags[idx - 25:idx + 25])
            #     continue
            res.append(idx)
            i += 1
        return res

    def update_priorities(self, idxes, priorities, min_priority=0.001):
        """Update priorities of sampled transitions.
        sets priority of transition at index idxes[i] in buffer
        to priorities[i].
        Parameters
        ----------
        idxes: [int]
            List of idxes of sampled transitions
        priorities: [float]
            List of updated priorities corresponding to
            transitions at the sampled idxes denoted by
            variable `idxes`.
        """
        # Boost expert priority as time goes on ....
        priorities = np.squeeze(priorities)
        assert len(idxes) == priorities.shape[0]
        count = 0
        # print(expert_priority_modifier)
        for idx, priority in zip(idxes, priorities):
            priority = max(priority, 0)
            assert 0 <= idx < self.count

            new_priority = priority * (1 - min_priority) + min_priority
            self._max_priority = max(self._max_priority, new_priority)
            # print(idx, new_priority, expert_idxes[count], count)
            self._it_sum[idx] = (new_priority) ** self._alpha
            self._it_min[idx] = (new_priority) ** self._alpha
            count += 1


class ReplayBufferPriorityNStepReducedActions:
    def __init__(self, args, size=1000000, n_step_std=False):
        self.buffer = {}
        self.args = args
        self.size = size
        self.count = 0
        self.add_index = 0
        self.n_step_std = n_step_std

        alpha = args.alpha
        self._beta = args.beta

        assert alpha >= 0
        self._alpha = alpha

        it_capacity = 1
        while it_capacity < size:
            it_capacity *= 2

        self._it_sum = SumSegmentTree(it_capacity)
        self._it_min = MinSegmentTree(it_capacity)
        self._max_priority = 5

    def add(self, state, next_state, action, next_action, rewards, terminal, n_step_state, n_step_terminal,
            n_step_action, n_step_next_action, abstract_action=None):
        # state = state.astype(np.uint8)
        # next_state = next_state.astype(np.uint8)
        self.count = min(self.count + 1, self.size)
        self.buffer[self.add_index] = [state, next_state, action, next_action, rewards, terminal, n_step_state,
                                       n_step_terminal, n_step_action, n_step_next_action, abstract_action]
        self._it_sum[self.add_index] = self._max_priority ** self._alpha
        self._it_min[self.add_index] = self._max_priority ** self._alpha

        self.add_index = (self.add_index + 1) % self.size

    def sample(self, all=False):
        indices = self._get_random_indices(self.args.batch_size)
        samples = []
        for i in range(self.args.batch_size):
            samples.append(self.buffer[indices[i]])
        state, next_state, action, next_action, rewards, terminal, n_step_state, n_step_terminal, n_step_action, n_step_next_action, abstract_action = map(
            np.asarray, zip(*samples))

        if all:
            if not abstract_action[0] is None:
                return state, next_state, action, next_action, rewards, terminal, n_step_state, n_step_terminal, n_step_action, n_step_next_action, abstract_action

            return state, next_state, action, next_action, rewards, terminal, n_step_state, n_step_terminal, n_step_next_action, n_step_action
        else:
            return state, next_state, action, next_action, rewards, terminal

    def sample_priority(self, partial_uniform_sampling=0):
        total_weight = self._it_sum.sum()
        weights = []
        p_min = self._it_min.min() / total_weight
        max_weight = (p_min * self.count) ** (-self._beta)

        if partial_uniform_sampling > 0:
            partial_batch = self.args.batch_size - partial_uniform_sampling
            indices_1 = self._get_random_indices(partial_uniform_sampling)
            indices_2 = self._sample_proportional(partial_batch)
            indices = indices_1 + indices_2
            for _ in indices_1:
                p_sample = 1 / self.count
                weight = (p_sample * self.count) ** (-self._beta)
                weights.append(weight / max_weight)

            for idx in indices_2:
                p_sample = self._it_sum[idx] / total_weight
                weight = (p_sample * self.count) ** (-self._beta)
                weights.append(weight / max_weight)
            weights = np.array(weights)
        else:
            indices = self._sample_proportional(self.args.batch_size)
            for idx in indices:
                p_sample = self._it_sum[idx] / total_weight
                weight = (p_sample * self.count) ** (-self._beta)
                weights.append(weight / max_weight)
            weights = np.array(weights)

        samples = []
        for i in range(self.args.batch_size):
            samples.append(self.buffer[indices[i]])
        state, next_state, action, next_action, rewards, terminal, n_step_state, n_step_terminal, n_step_action, n_step_next_action, abstract_action = map(
            np.asarray, zip(*samples))

        if not abstract_action[0] is None:
            return state, next_state, action, next_action, rewards, terminal, n_step_state, indices, weights, n_step_terminal, n_step_action, n_step_next_action, abstract_action
        return state, next_state, action, next_action, rewards, terminal, indices, weights, n_step_state, n_step_terminal, n_step_action, n_step_next_action

    def size(self):
        return len(self.buffer)

    def _get_random_indices(self, batch_size):
        idxes = []
        while len(idxes) < batch_size:
            index = np.random.randint(0, self.count - 1)
            idxes.append(index)
        return idxes

    def _sample_proportional(self, batch_size):
        res = []
        p_total = self._it_sum.sum(0, self.count - 1)
        every_range_len = p_total / batch_size
        i = 0
        count = 0
        while len(res) < batch_size:
            count += 1
            if count > 10000:
                idx = np.random.randint(0, self.count)
            else:
                mass = random.random() * every_range_len + i * every_range_len
                idx = self._it_sum.find_prefixsum_idx(mass)
            # if idx < self.agent_history_length + 1:
            #     continue
            # if idx >= self.expert_idx and idx - self.agent_history_length < self.expert_idx:
            #     continue
            # if idx >= self._next_idx and idx - self.agent_history_length < self._next_idx:
            #     continue
            # if np.sum(self.terminal_flags[idx - self.agent_history_length:idx]) > 0:
            #     # if count > 9900:
            #     #     print(self.terminal_flags[idx - 25:idx + 25])
            #     continue
            res.append(idx)
            i += 1
        return res

    def update_priorities(self, idxes, priorities, min_priority=0.001):
        """Update priorities of sampled transitions.
        sets priority of transition at index idxes[i] in buffer
        to priorities[i].
        Parameters
        ----------
        idxes: [int]
            List of idxes of sampled transitions
        priorities: [float]
            List of updated priorities corresponding to
            transitions at the sampled idxes denoted by
            variable `idxes`.
        """
        # Boost expert priority as time goes on ....
        priorities = np.squeeze(priorities)
        assert len(idxes) == priorities.shape[0]
        count = 0
        # print(expert_priority_modifier)
        for idx, priority in zip(idxes, priorities):
            priority = max(priority, 0)
            assert 0 <= idx < self.count

            new_priority = priority * (1 - min_priority) + min_priority
            self._max_priority = max(self._max_priority, new_priority)
            # print(idx, new_priority, expert_idxes[count], count)
            self._it_sum[idx] = (new_priority) ** self._alpha
            self._it_min[idx] = (new_priority) ** self._alpha
            count += 1


class ReplayBufferPriorityNStepReduced:
    def __init__(self, args, size=1000000, n_step_std=False):
        self.buffer = {}
        self.args = args
        self.size = size
        self.count = 0
        self.add_index = 0
        self.n_step_std = n_step_std

        alpha = args.alpha
        self._beta = args.beta

        assert alpha >= 0
        self._alpha = alpha

        it_capacity = 1
        while it_capacity < size:
            it_capacity *= 2

        self._it_sum = SumSegmentTree(it_capacity)
        self._it_min = MinSegmentTree(it_capacity)
        self._max_priority = 5

    def add(self, state, next_state, action, next_action, rewards, terminal, n_step_state, n_step_terminal,
            n_step_action_mean, abstract_action=None):
        # state = state.astype(np.uint8)
        # next_state = next_state.astype(np.uint8)
        self.count = min(self.count + 1, self.size)
        self.buffer[self.add_index] = [state, next_state, action, next_action, rewards, terminal, n_step_state,
                                       n_step_terminal, n_step_action_mean, abstract_action]
        self._it_sum[self.add_index] = self._max_priority ** self._alpha
        self._it_min[self.add_index] = self._max_priority ** self._alpha

        self.add_index = (self.add_index + 1) % self.size

    def sample(self, all=False):
        indices = self._get_random_indices(self.args.batch_size)
        samples = []
        for i in range(self.args.batch_size):
            samples.append(self.buffer[indices[i]])
        state, next_state, action, next_action, rewards, terminal, n_step_state, n_step_terminal, n_step_action_mean, abstract_action = map(
            np.asarray, zip(*samples))

        if all:
            if not abstract_action[0] is None:
                return state, next_state, action, next_action, rewards, terminal, n_step_state, n_step_terminal, n_step_action_mean, abstract_action

            return state, next_state, action, next_action, rewards, terminal, n_step_state, n_step_terminal, n_step_action_mean
        else:
            return state, next_state, action, next_action, rewards, terminal

    def sample_priority(self, partial_uniform_sampling=0, all=True):
        total_weight = self._it_sum.sum()
        weights = []
        p_min = self._it_min.min() / total_weight
        max_weight = (p_min * self.count) ** (-self._beta)

        if partial_uniform_sampling > 0:
            partial_batch = self.args.batch_size - partial_uniform_sampling
            indices_1 = self._get_random_indices(partial_uniform_sampling)
            indices_2 = self._sample_proportional(partial_batch)
            indices = indices_1 + indices_2
            for _ in indices_1:
                p_sample = 1 / self.count
                weight = (p_sample * self.count) ** (-self._beta)
                weights.append(weight / max_weight)

            for idx in indices_2:
                p_sample = self._it_sum[idx] / total_weight
                weight = (p_sample * self.count) ** (-self._beta)
                weights.append(weight / max_weight)
            weights = np.array(weights)
        else:
            indices = self._sample_proportional(self.args.batch_size)
            for idx in indices:
                p_sample = self._it_sum[idx] / total_weight
                weight = (p_sample * self.count) ** (-self._beta)
                weights.append(weight / max_weight)
            weights = np.array(weights)

        samples = []
        for i in range(self.args.batch_size):
            samples.append(self.buffer[indices[i]])
        state, next_state, action, next_action, rewards, terminal, n_step_state, n_step_terminal, n_step_action_mean, abstract_action = map(
            np.asarray, zip(*samples))

        if all:
            if not abstract_action[0] is None:
                return state, next_state, action, next_action, rewards, terminal, n_step_state, indices, weights, n_step_terminal, n_step_action_mean, abstract_action
            return state, next_state, action, next_action, rewards, terminal, indices, weights, n_step_state, n_step_terminal, n_step_action_mean
        else:
            return state, next_state, action, next_action, rewards, terminal, indices, weights

    def size(self):
        return len(self.buffer)

    def _get_random_indices(self, batch_size):
        idxes = []
        while len(idxes) < batch_size:
            index = np.random.randint(0, self.count - 1)
            idxes.append(index)
        return idxes

    def _sample_proportional(self, batch_size):
        res = []
        p_total = self._it_sum.sum(0, self.count - 1)
        every_range_len = p_total / batch_size
        i = 0
        count = 0
        while len(res) < batch_size:
            count += 1
            if count > 10000:
                idx = np.random.randint(0, self.count)
            else:
                mass = random.random() * every_range_len + i * every_range_len
                idx = self._it_sum.find_prefixsum_idx(mass)
            # if idx < self.agent_history_length + 1:
            #     continue
            # if idx >= self.expert_idx and idx - self.agent_history_length < self.expert_idx:
            #     continue
            # if idx >= self._next_idx and idx - self.agent_history_length < self._next_idx:
            #     continue
            # if np.sum(self.terminal_flags[idx - self.agent_history_length:idx]) > 0:
            #     # if count > 9900:
            #     #     print(self.terminal_flags[idx - 25:idx + 25])
            #     continue
            res.append(idx)
            i += 1
        return res

    def update_priorities(self, idxes, priorities, min_priority=0.001):
        """Update priorities of sampled transitions.
        sets priority of transition at index idxes[i] in buffer
        to priorities[i].
        Parameters
        ----------
        idxes: [int]
            List of idxes of sampled transitions
        priorities: [float]
            List of updated priorities corresponding to
            transitions at the sampled idxes denoted by
            variable `idxes`.
        """
        # Boost expert priority as time goes on ....
        priorities = np.squeeze(priorities)
        assert len(idxes) == priorities.shape[0]
        count = 0
        # print(expert_priority_modifier)
        for idx, priority in zip(idxes, priorities):
            priority = max(priority, 0)
            assert 0 <= idx < self.count

            new_priority = priority * (1 - min_priority) + min_priority
            self._max_priority = max(self._max_priority, new_priority)
            # print(idx, new_priority, expert_idxes[count], count)
            self._it_sum[idx] = (new_priority) ** self._alpha
            self._it_min[idx] = (new_priority) ** self._alpha
            count += 1


class ReplayBufferPriorityNStep:
    def __init__(self, args, size=1000000, n_step_std=False):
        self.buffer = {}
        self.args = args
        self.size = size
        self.count = 0
        self.add_index = 0
        self.n_step_std = n_step_std

        alpha = args.alpha
        self._beta = args.beta

        assert alpha >= 0
        self._alpha = alpha

        it_capacity = 1
        while it_capacity < size:
            it_capacity *= 2

        self._it_sum = SumSegmentTree(it_capacity)
        self._it_min = MinSegmentTree(it_capacity)
        self._max_priority = 5

        self.visited_positions = []

    def add_position(self, position):
        self.visited_positions.append(position)

    def draw_path_distribution(self, name, image_size=[256, 256], range_x=None, range_y=None):
        visited_position = np.array(self.visited_positions)
        position_list_x = visited_position[:, 0]
        position_list_y = visited_position[:, 1]
        if range_x:
            min_x = range_x[0]
            max_x = range_x[1]
            min_y = range_y[0]
            max_y = range_y[1]
        else:
            max_x = np.max(position_list_x)
            min_x = np.min(position_list_x)
            max_y = np.max(position_list_y)
            min_y = np.min(position_list_y)

        position_map = np.zeros((image_size[0], image_size[1]))
        for i in range(len(position_list_y)):
            position_x = position_list_x[i]
            position_y = position_list_y[i]
            x = np.clip(position_x, min_x, max_x)
            y = np.clip(position_y, min_y, max_y)
            x = int((x - min_x) / (max_x - min_x) * (range_x[1] - range_x[0]))
            y = int((y - min_y) / (max_y - min_y) * (range_y[1] - range_y[0]))

            position_map[x, y] += 1

        position_map = np.expand_dims(np.log(1 + position_map), axis=2)
        position_map = (position_map / (np.max(np.max(position_map))) * 200 + 55).astype(np.uint8)
        position_map = np.concatenate([position_map, position_map, position_map], axis=2)
        im = Image.fromarray(position_map)
        im.save(name + ".png")

    def add(self, state, action, action_mean, action_std, prev_action, reward, done, next_state, n_step_state,
            n_step_terminal, n_step_action_mean, n_step_std=None, abstract_action=None):
        # state = state.astype(np.uint8)
        # next_state = next_state.astype(np.uint8)
        self.count = min(self.count + 1, self.size)
        self.buffer[self.add_index] = [state, action, action_mean, action_std, prev_action, reward, next_state, done,
                                       n_step_action_mean, n_step_std, n_step_state, n_step_terminal, abstract_action]
        self._it_sum[self.add_index] = self._max_priority ** self._alpha
        self._it_min[self.add_index] = self._max_priority ** self._alpha

        self.add_index = (self.add_index + 1) % self.size

    def sample(self, all=False):
        indices = self._get_random_indices(self.args.batch_size)
        samples = []
        for i in range(self.args.batch_size):
            samples.append(self.buffer[indices[i]])
        states_dense, actions, action_mean, action_std, prev_action, rewards, next_states_dense, done, n_step_action_mean, n_step_action_std, n_step_state_dense, n_step_terminal, abstract_action = map(
            np.asarray, zip(*samples))

        if all:
            if not abstract_action[0] is None:
                return states_dense, next_states_dense, actions, action_mean, \
                       action_std, prev_action, rewards, done, n_step_action_mean, n_step_state_dense, n_step_terminal, abstract_action

            return states_dense, next_states_dense, actions, action_mean, \
                   action_std, prev_action, rewards, done, n_step_action_mean, n_step_state_dense, n_step_terminal
        else:
            return states_dense, next_states_dense, actions, action_mean, action_std, prev_action, rewards, done

    def sample_priority(self, partial_uniform_sampling=0):
        total_weight = self._it_sum.sum()
        weights = []
        p_min = self._it_min.min() / total_weight
        max_weight = (p_min * self.count) ** (-self._beta)

        if partial_uniform_sampling > 0:
            partial_batch = self.args.batch_size - partial_uniform_sampling
            indices_1 = self._get_random_indices(partial_uniform_sampling)
            indices_2 = self._sample_proportional(partial_batch)
            indices = indices_1 + indices_2
            for _ in indices_1:
                p_sample = 1 / self.count
                weight = (p_sample * self.count) ** (-self._beta)
                weights.append(weight / max_weight)

            for idx in indices_2:
                p_sample = self._it_sum[idx] / total_weight
                weight = (p_sample * self.count) ** (-self._beta)
                weights.append(weight / max_weight)
            weights = np.array(weights)
        else:
            indices = self._sample_proportional(self.args.batch_size)
            for idx in indices:
                p_sample = self._it_sum[idx] / total_weight
                weight = (p_sample * self.count) ** (-self._beta)
                weights.append(weight / max_weight)
            weights = np.array(weights)

        samples = []
        for i in range(self.args.batch_size):
            samples.append(self.buffer[indices[i]])
        states_dense, actions, action_mean, action_std, prev_action, rewards, next_states_dense, done, n_step_action_mean, n_step_action_std, n_step_state_dense, n_step_terminal, abstract_action = map(
            np.asarray, zip(*samples))
        if self.n_step_std:
            return states_dense, next_states_dense, actions, action_mean, \
                   action_std, prev_action, rewards, done, indices, weights, n_step_action_mean, n_step_action_std, n_step_state_dense, n_step_terminal
        else:
            if not abstract_action[0] is None:
                return states_dense, next_states_dense, actions, action_mean, \
                       action_std, prev_action, rewards, done, indices, weights, n_step_action_mean, n_step_state_dense, n_step_terminal, abstract_action

            return states_dense, next_states_dense, actions, action_mean, \
                   action_std, prev_action, rewards, done, indices, weights, n_step_action_mean, n_step_state_dense, n_step_terminal

    def size(self):
        return len(self.buffer)

    def _get_random_indices(self, batch_size):
        idxes = []
        while len(idxes) < batch_size:
            index = np.random.randint(0, self.count - 1)
            idxes.append(index)
        return idxes

    def _sample_proportional(self, batch_size):
        res = []
        p_total = self._it_sum.sum(0, self.count - 1)
        every_range_len = p_total / batch_size
        i = 0
        count = 0
        while len(res) < batch_size:
            count += 1
            if count > 10000:
                idx = np.random.randint(0, self.count)
            else:
                mass = random.random() * every_range_len + i * every_range_len
                idx = self._it_sum.find_prefixsum_idx(mass)
            # if idx < self.agent_history_length + 1:
            #     continue
            # if idx >= self.expert_idx and idx - self.agent_history_length < self.expert_idx:
            #     continue
            # if idx >= self._next_idx and idx - self.agent_history_length < self._next_idx:
            #     continue
            # if np.sum(self.terminal_flags[idx - self.agent_history_length:idx]) > 0:
            #     # if count > 9900:
            #     #     print(self.terminal_flags[idx - 25:idx + 25])
            #     continue
            res.append(idx)
            i += 1
        return res

    def update_priorities(self, idxes, priorities, min_priority=0.001):
        """Update priorities of sampled transitions.
        sets priority of transition at index idxes[i] in buffer
        to priorities[i].
        Parameters
        ----------
        idxes: [int]
            List of idxes of sampled transitions
        priorities: [float]
            List of updated priorities corresponding to
            transitions at the sampled idxes denoted by
            variable `idxes`.
        """
        # Boost expert priority as time goes on ....
        priorities = np.squeeze(priorities)
        assert len(idxes) == priorities.shape[0]
        count = 0
        # print(expert_priority_modifier)
        for idx, priority in zip(idxes, priorities):
            priority = max(priority, 0)
            assert 0 <= idx < self.count

            new_priority = priority * (1 - min_priority) + min_priority
            self._max_priority = max(self._max_priority, new_priority)
            # print(idx, new_priority, expert_idxes[count], count)
            self._it_sum[idx] = (new_priority) ** self._alpha
            self._it_min[idx] = (new_priority) ** self._alpha
            count += 1


class ReplayBufferPriorityNStepHistory(ReplayBufferPriorityNStep):
    def __init__(self, args, size=1000000, n_step_std=False):
        super().__init__(args, size=size, n_step_std=n_step_std)

        self.abstract_normalization_mean = None
        self.abstract_normalization_std = None

    # state, next_state, next_action, next_action_mean, next_action_std, action, reward, terminal, n_step_abstract_action_mean, n_step_state, n_step_terminal, next_abstract_action, past_actions, past_next_actions = buffer.sample(all=True)

    def add(self, state, next_state, action, next_action, reward, terminal, n_step_state, n_step_terminal,
            n_step_abstract_action, n_step_std=None, next_abstract_action=None, past_actions=None,
            past_next_actions=None):
        self.count = min(self.count + 1, self.size)
        self.buffer[self.add_index] = [state, next_state, action, next_action, reward, terminal, n_step_state,
                                       n_step_terminal, n_step_abstract_action, n_step_std, next_abstract_action,
                                       past_actions, past_next_actions]
        self._it_sum[self.add_index] = self._max_priority ** self._alpha
        self._it_min[self.add_index] = self._max_priority ** self._alpha

        self.add_index = (self.add_index + 1) % self.size

    def sample(self, all=False):
        indices = self._get_random_indices(self.args.batch_size)
        samples = []
        for i in range(self.args.batch_size):
            samples.append(self.buffer[indices[i]])
        state, next_state, action, next_action, reward, terminal, n_step_state, n_step_terminal, n_step_abstract_action, n_step_std, next_abstract_action, past_actions, past_next_actions = map(
            np.asarray, zip(*samples))

        if not self.abstract_normalization_mean is None:
            n_step_abstract_action = np.copy(next_abstract_action) / (self.abstract_normalization_std)
        if all:
            return state, next_state, action, next_action, \
                   reward, terminal, n_step_state, n_step_abstract_action, n_step_terminal, next_abstract_action, past_actions, past_next_actions
        else:
            return state, next_state, action, next_action, reward, terminal

    def sample_priority(self, partial_uniform_sampling=0):
        total_weight = self._it_sum.sum()
        weights = []
        p_min = self._it_min.min() / total_weight
        max_weight = (p_min * self.count) ** (-self._beta)

        if partial_uniform_sampling > 0:
            partial_batch = self.args.batch_size - partial_uniform_sampling
            indices_1 = self._get_random_indices(partial_uniform_sampling)
            indices_2 = self._sample_proportional(partial_batch)
            indices = indices_1 + indices_2
            for _ in indices_1:
                p_sample = 1 / self.count
                weight = (p_sample * self.count) ** (-self._beta)
                weights.append(weight / max_weight)

            for idx in indices_2:
                p_sample = self._it_sum[idx] / total_weight
                weight = (p_sample * self.count) ** (-self._beta)
                weights.append(weight / max_weight)
            weights = np.array(weights)
        else:
            indices = self._sample_proportional(self.args.batch_size)
            for idx in indices:
                p_sample = self._it_sum[idx] / total_weight
                weight = (p_sample * self.count) ** (-self._beta)
                weights.append(weight / max_weight)
            weights = np.array(weights)

        samples = []
        for i in range(self.args.batch_size):
            samples.append(self.buffer[indices[i]])
        state, next_state, action, next_action, reward, terminal, n_step_state, n_step_terminal, n_step_abstract_action, n_step_std, next_abstract_action, past_actions, past_next_actions = map(
            np.asarray, zip(*samples))

        if not self.abstract_normalization_mean is None:
            n_step_abstract_action = np.copy(next_abstract_action) / (self.abstract_normalization_std)

        return state, next_state, action, next_action, \
               reward, terminal, indices, weights, n_step_state, n_step_abstract_action, n_step_terminal, next_abstract_action, past_actions, past_next_actions

        # return state, next_state, action, next_action, reward, terminal, indices, weights, n_step_action_mean, n_step_state, n_step_terminal, abstract_action, past_actions, past_next_actions

    def compute_normalized_abstract_actions(self):
        abstract_action_list = []
        for i in range(self.count):
            _, _, _, _, _, _, _, _, _, _, abstract_action, _, _ = self.buffer[i]
            abstract_action_list.append(abstract_action)

        abstract_action_list = np.array(abstract_action_list)
        self.abstract_normalization_mean = np.zeros(abstract_action.shape)
        self.abstract_normalization_std = np.zeros(abstract_action.shape)
        for i in range(abstract_action.shape[0]):
            self.abstract_normalization_mean[i] = np.mean(abstract_action_list[:, i])
            self.abstract_normalization_std[i] = np.std(abstract_action_list[:, i])
            if self.abstract_normalization_std[i] < 0.0001:
                self.abstract_normalization_std[i] = 1
        self.abstract_normalization_std *= 2
        print("Normalization: ", self.abstract_normalization_mean, self.abstract_normalization_std)

    def size(self):
        return len(self.buffer)


class ReplayBufferPriorityNStepHistoryV2(ReplayBufferPriorityNStepHistory):
    def __init__(self, args, size=1000000, n_step_std=False):
        super().__init__(args, size=size, n_step_std=n_step_std)

    # s_t, s_t_1, a_t, a_t_1, r_t, terminal_t, terminal_t_1, \
    # s_t_n, a_t_n, abstract_a_t_n, terminal_t_n, abstract_a_t, abstract_a_t_1, past_actions_t, past_actions_t_1 = buffer.sample(all=True)
    def add(self, state, next_state, action, next_action, reward, terminal, n_step_state, a_t_n, n_step_terminal,
            n_step_abstract_action, n_step_std=None, abstract_action_t=None, next_abstract_action=None,
            past_actions=None, past_next_actions=None, past_action_n=None):
        self.count = min(self.count + 1, self.size)
        self.buffer[self.add_index] = [state, next_state, action, next_action, reward, terminal, n_step_state, a_t_n,
                                       n_step_terminal, n_step_abstract_action, n_step_std, abstract_action_t,
                                       next_abstract_action, past_actions, past_next_actions, past_action_n]
        self._it_sum[self.add_index] = self._max_priority ** self._alpha
        self._it_min[self.add_index] = self._max_priority ** self._alpha

        self.add_index = (self.add_index + 1) % self.size

    def sample(self, all=False):
        indices = self._get_random_indices(self.args.batch_size)
        samples = []
        for i in range(self.args.batch_size):
            samples.append(self.buffer[indices[i]])
        state, next_state, action, next_action, reward, terminal, n_step_state, a_t_n, n_step_terminal, \
        n_step_abstract_action, n_step_std, abstract_action_t, next_abstract_action, past_actions, past_next_actions, past_action_n = map(
            np.asarray, zip(*samples))

        if not self.abstract_normalization_mean is None:
            n_step_abstract_action = np.copy(next_abstract_action) / (self.abstract_normalization_std)
        if all:
            return state, next_state, action, next_action, \
                   reward, terminal, n_step_state, a_t_n, n_step_abstract_action, n_step_terminal, abstract_action_t, next_abstract_action, past_actions, past_next_actions, past_action_n
        else:
            return state, next_state, action, next_action, reward, terminal

    def sample_priority(self, partial_uniform_sampling=0):
        total_weight = self._it_sum.sum()
        weights = []
        p_min = self._it_min.min() / total_weight
        max_weight = (p_min * self.count) ** (-self._beta)

        if partial_uniform_sampling > 0:
            partial_batch = self.args.batch_size - partial_uniform_sampling
            indices_1 = self._get_random_indices(partial_uniform_sampling)
            indices_2 = self._sample_proportional(partial_batch)
            indices = indices_1 + indices_2
            for _ in indices_1:
                p_sample = 1 / self.count
                weight = (p_sample * self.count) ** (-self._beta)
                weights.append(weight / max_weight)

            for idx in indices_2:
                p_sample = self._it_sum[idx] / total_weight
                weight = (p_sample * self.count) ** (-self._beta)
                weights.append(weight / max_weight)
            weights = np.array(weights)
        else:
            indices = self._sample_proportional(self.args.batch_size)
            for idx in indices:
                p_sample = self._it_sum[idx] / total_weight
                weight = (p_sample * self.count) ** (-self._beta)
                weights.append(weight / max_weight)
            weights = np.array(weights)

        samples = []
        for i in range(self.args.batch_size):
            samples.append(self.buffer[indices[i]])
        state, next_state, action, next_action, reward, terminal, n_step_state, a_t_n, n_step_terminal, \
        n_step_abstract_action, n_step_std, abstract_action_t, next_abstract_action, past_actions, past_next_actions, past_action_n = map(
            np.asarray, zip(*samples))

        if not self.abstract_normalization_mean is None:
            n_step_abstract_action = np.copy(next_abstract_action) / (self.abstract_normalization_std)

        return state, next_state, action, next_action, reward, terminal, indices, weights, n_step_state, a_t_n, n_step_abstract_action, \
               n_step_terminal, abstract_action_t, next_abstract_action, past_actions, past_next_actions, past_action_n

        # return state, next_state, action, next_action, \
        # reward, terminal, indices, weights, n_step_state, n_step_abstract_action, n_step_terminal, next_abstract_action, past_actions, past_next_actions

    def compute_normalized_abstract_actions(self):
        abstract_action_list = []
        for i in range(self.count):
            _, _, _, _, _, _, _, _, _, _, _, abstract_action, _, _, _, _ = self.buffer[i]
            abstract_action_list.append(abstract_action)

        abstract_action_list = np.array(abstract_action_list)
        self.abstract_normalization_mean = np.zeros(abstract_action.shape)
        self.abstract_normalization_std = np.zeros(abstract_action.shape)
        for i in range(abstract_action.shape[0]):
            self.abstract_normalization_mean[i] = np.mean(abstract_action_list[:, i])
            self.abstract_normalization_std[i] = np.std(abstract_action_list[:, i])
            if self.abstract_normalization_std[i] < 0.0001:
                self.abstract_normalization_std[i] = 1
        self.abstract_normalization_std *= 2
        print("Normalization: ", self.abstract_normalization_mean, self.abstract_normalization_std)


class ReplayBufferPriorityNStepHistoryV4(ReplayBufferPriorityNStepHistory):
    def __init__(self, args, size=1000000, n_step_std=False):
        super().__init__(args, size=size, n_step_std=n_step_std)
        # s_t, s_t_1, s_t_n__1, s_t_n, \
        # a_t, a_t_1, a_t_n, \
        # pa_t, pa_t_n__1, \
        # aa_t, aa_t_n__1, \
        # terminal_t, terminal_t_n, reward_t = buffer.sample(all=True)
        self.abstract_normalization_n_offset = None

    def add(self, s_t, s_t_1, s_t_n__1, s_t_n,
            a_t, a_t_1, a_t_n,
            pa_t, pa_t_n__1,
            aa_t, aa_t_n__1,
            terminal_t, terminal_t_n, reward_t):
        self.count = min(self.count + 1, self.size)
        self.buffer[self.add_index] = [s_t, s_t_1, s_t_n__1, s_t_n,
                                       a_t, a_t_1, a_t_n,
                                       pa_t, pa_t_n__1,
                                       aa_t, aa_t_n__1,
                                       terminal_t, terminal_t_n, reward_t]

        self._it_sum[self.add_index] = self._max_priority ** self._alpha
        self._it_min[self.add_index] = self._max_priority ** self._alpha

        self.add_index = (self.add_index + 1) % self.size

    def sample(self, all=False):
        indices = self._get_random_indices(self.args.batch_size)
        samples = []
        for i in range(self.args.batch_size):
            samples.append(self.buffer[indices[i]])
        s_t, s_t_1, s_t_n__1, s_t_n, \
        a_t, a_t_1, a_t_n, \
        pa_t, pa_t_n__1, \
        aa_t, aa_t_n__1, \
        terminal_t, terminal_t_n, reward_t = map(np.asarray, zip(*samples))

        if not self.abstract_normalization_mean is None:
            aa_t_n__1 = np.copy(aa_t_n__1 - self.abstract_normalization_n_offset) / (self.abstract_normalization_std)
            aa_t = np.copy(aa_t - self.abstract_normalization_mean) / (self.abstract_normalization_std)
        if all:
            return s_t, s_t_1, s_t_n__1, s_t_n, \
                   a_t, a_t_1, a_t_n, \
                   pa_t, pa_t_n__1, \
                   aa_t, aa_t_n__1, \
                   terminal_t, terminal_t_n, reward_t
        else:
            return s_t, s_t_1, a_t, a_t_1, reward_t, terminal_t

    def sample_priority(self, partial_uniform_sampling=0):
        total_weight = self._it_sum.sum()
        weights = []
        p_min = self._it_min.min() / total_weight
        max_weight = (p_min * self.count) ** (-self._beta)

        if partial_uniform_sampling > 0:
            partial_batch = self.args.batch_size - partial_uniform_sampling
            indices_1 = self._get_random_indices(partial_uniform_sampling)
            indices_2 = self._sample_proportional(partial_batch)
            indices = indices_1 + indices_2
            for _ in indices_1:
                p_sample = 1 / self.count
                weight = (p_sample * self.count) ** (-self._beta)
                weights.append(weight / max_weight)

            for idx in indices_2:
                p_sample = self._it_sum[idx] / total_weight
                weight = (p_sample * self.count) ** (-self._beta)
                weights.append(weight / max_weight)
            weights = np.array(weights)
        else:
            indices = self._sample_proportional(self.args.batch_size)
            for idx in indices:
                p_sample = self._it_sum[idx] / total_weight
                weight = (p_sample * self.count) ** (-self._beta)
                weights.append(weight / max_weight)
            weights = np.array(weights)

        samples = []
        for i in range(self.args.batch_size):
            samples.append(self.buffer[indices[i]])
        s_t, s_t_1, s_t_n__1, s_t_n, \
        a_t, a_t_1, a_t_n, \
        pa_t, pa_t_n__1, \
        aa_t, aa_t_n__1, \
        terminal_t, terminal_t_n, reward_t = map(np.asarray, zip(*samples))

        if not self.abstract_normalization_mean is None:
            aa_t_n__1 = np.copy(aa_t_n__1 - self.abstract_normalization_n_offset) / (self.abstract_normalization_std)
            aa_t = np.copy(aa_t - self.abstract_normalization_mean) / (self.abstract_normalization_std)

        return s_t, s_t_1, s_t_n__1, s_t_n, \
               a_t, a_t_1, a_t_n, \
               pa_t, pa_t_n__1, \
               aa_t, aa_t_n__1, \
               terminal_t, terminal_t_n, reward_t, \
               indices, weights

        # return state, next_state, action, next_action, \
        # reward, terminal, indices, weights, n_step_state, n_step_abstract_action, n_step_terminal, next_abstract_action, past_actions, past_next_actions

    def compute_normalized_abstract_actions(self):
        abstract_action_list = []
        for i in range(self.count):
            s_t, s_t_1, s_t_n__1, s_t_n, \
            a_t, a_t_1, a_t_n, \
            pa_t, pa_t_n__1, \
            aa_t, aa_t_n__1, \
            terminal_t, terminal_t_n, reward_t = self.buffer[i]
            abstract_action_list.append(aa_t)

        abstract_action_list = np.array(abstract_action_list)
        self.abstract_normalization_mean = np.zeros(aa_t.shape)
        self.abstract_normalization_std = np.zeros(aa_t.shape)
        for i in range(aa_t.shape[0]):
            self.abstract_normalization_mean[i] = np.mean(abstract_action_list[:, i])
            self.abstract_normalization_std[i] = np.std(abstract_action_list[:, i])
            if self.abstract_normalization_std[i] < 0.0001:
                self.abstract_normalization_std[i] = 1

        # sum gamma^i * (x_i - mean)/std
        # = sum gamma^i * x_i/std - sum gamma^i * mean/std
        # = (sum gamma^i * x_i)/std - (sum gamma^i * mean)/std
        self.abstract_normalization_n_offset = 0
        gamma = 1
        for i in range(self.args.n_step):
            self.abstract_normalization_n_offset += gamma * self.abstract_normalization_mean
            gamma *= self.args.gamma_action

        self.abstract_normalization_std *= 8

        print("Normalization: ", self.abstract_normalization_mean, self.abstract_normalization_std)


class ReplayBufferPriorityNStepHistoryV5(ReplayBufferPriorityNStepHistory):
    def __init__(self, args, size=1000000, n_step_std=False):
        super().__init__(args, size=size, n_step_std=n_step_std)
        # s_t, s_t_1, s_t_n__1, s_t_n, \
        # a_t, a_t_1, a_t_n, \
        # pa_t, pa_t_n__1, \
        # aa_t, aa_t_n__1, \
        # terminal_t, terminal_t_n, reward_t = buffer.sample(all=True)
        self.abstract_normalization_n_offset = None

    def add(self, s_t, s_t_1, s_t_n__1, s_t_n,
            a_t, a_t_1, a_t_n,
            pa_t, pa_t_n__1,
            aa_t, aa_t_n__1,
            terminal_t, terminal_t_n, reward_t):
        self.count = min(self.count + 1, self.size)
        self.buffer[self.add_index] = [s_t, s_t_1, s_t_n__1, s_t_n,
                                       a_t, a_t_1, a_t_n,
                                       pa_t, pa_t_n__1,
                                       aa_t, aa_t_n__1,
                                       terminal_t, terminal_t_n, reward_t]

        self._it_sum[self.add_index] = self._max_priority ** self._alpha
        self._it_min[self.add_index] = self._max_priority ** self._alpha

        self.add_index = (self.add_index + 1) % self.size

    def sample(self, all=False):
        indices = self._get_random_indices(self.args.batch_size)
        samples = []
        for i in range(self.args.batch_size):
            samples.append(self.buffer[indices[i]])
        s_t, s_t_1, s_t_n__1, s_t_n, \
        a_t, a_t_1, a_t_n, \
        pa_t, pa_t_n__1, \
        aa_t, aa_t_n__1, \
        terminal_t, terminal_t_n, reward_t = map(np.asarray, zip(*samples))
        if not self.abstract_normalization_mean is None:
            aa_t_n__1 = np.copy(aa_t_n__1) / (
                        self.abstract_normalization_max - self.abstract_normalization_min) * 2 - self.abstract_normalization_n_offset
            aa_t = np.copy(aa_t - self.abstract_normalization_min) / (
                        self.abstract_normalization_max - self.abstract_normalization_min) * 2 - 1
        if all:
            return s_t, s_t_1, s_t_n__1, s_t_n, \
                   a_t, a_t_1, a_t_n, \
                   pa_t, pa_t_n__1, \
                   aa_t, aa_t_n__1, \
                   terminal_t, terminal_t_n, reward_t
        else:
            return s_t, s_t_1, a_t, a_t_1, reward_t, terminal_t

    def sample_priority(self, partial_uniform_sampling=0):
        total_weight = self._it_sum.sum()
        weights = []
        p_min = self._it_min.min() / total_weight
        max_weight = (p_min * self.count) ** (-self._beta)

        if partial_uniform_sampling > 0:
            partial_batch = self.args.batch_size - partial_uniform_sampling
            indices_1 = self._get_random_indices(partial_uniform_sampling)
            indices_2 = self._sample_proportional(partial_batch)
            indices = indices_1 + indices_2
            for _ in indices_1:
                p_sample = 1 / self.count
                weight = (p_sample * self.count) ** (-self._beta)
                weights.append(weight / max_weight)

            for idx in indices_2:
                p_sample = self._it_sum[idx] / total_weight
                weight = (p_sample * self.count) ** (-self._beta)
                weights.append(weight / max_weight)
            weights = np.array(weights)
        else:
            indices = self._sample_proportional(self.args.batch_size)
            for idx in indices:
                p_sample = self._it_sum[idx] / total_weight
                weight = (p_sample * self.count) ** (-self._beta)
                weights.append(weight / max_weight)
            weights = np.array(weights)

        samples = []
        for i in range(self.args.batch_size):
            samples.append(self.buffer[indices[i]])
        s_t, s_t_1, s_t_n__1, s_t_n, \
        a_t, a_t_1, a_t_n, \
        pa_t, pa_t_n__1, \
        aa_t, aa_t_n__1, \
        terminal_t, terminal_t_n, reward_t = map(np.asarray, zip(*samples))

        if not self.abstract_normalization_mean is None:
            aa_t_n__1 = np.copy(aa_t_n__1) / (
                        self.abstract_normalization_max - self.abstract_normalization_min) * 2 - self.abstract_normalization_n_offset
            aa_t = np.copy(aa_t - self.abstract_normalization_min) / (
                        self.abstract_normalization_max - self.abstract_normalization_min) * 2 - 1

        return s_t, s_t_1, s_t_n__1, s_t_n, \
               a_t, a_t_1, a_t_n, \
               pa_t, pa_t_n__1, \
               aa_t, aa_t_n__1, \
               terminal_t, terminal_t_n, reward_t, \
               indices, weights

        # return state, next_state, action, next_action, \
        # reward, terminal, indices, weights, n_step_state, n_step_abstract_action, n_step_terminal, next_abstract_action, past_actions, past_next_actions

    def compute_normalized_abstract_actions(self):
        abstract_action_list = []
        for i in range(self.count):
            s_t, s_t_1, s_t_n__1, s_t_n, \
            a_t, a_t_1, a_t_n, \
            pa_t, pa_t_n__1, \
            aa_t, aa_t_n__1, \
            terminal_t, terminal_t_n, reward_t = self.buffer[i]
            abstract_action_list.append(aa_t)

        abstract_action_list = np.array(abstract_action_list)
        self.abstract_normalization_mean = np.zeros(aa_t.shape)
        self.abstract_normalization_std = np.zeros(aa_t.shape)
        self.abstract_normalization_max = np.zeros(aa_t.shape)
        self.abstract_normalization_min = np.zeros(aa_t.shape)
        for i in range(aa_t.shape[0]):
            self.abstract_normalization_mean[i] = np.mean(abstract_action_list[:, i])
            self.abstract_normalization_std[i] = np.std(abstract_action_list[:, i])
            if self.abstract_normalization_std[i] < 0.0001:
                self.abstract_normalization_std[i] = 1

            self.abstract_normalization_max[i] = np.percentile(abstract_action_list[:, i], 98)
            self.abstract_normalization_min[i] = np.percentile(abstract_action_list[:, i], 2)
        # sum gamma^i * ((x_i - min)/(max - min) * 2 - 1)
        # = sum gamma^i * (x_i - min)/(max - min) * 2 - sum gamma^i
        # = (sum gamma^i * x_i)/(max - min) * 2 - sum gamma^i * (min/(max - min) * 2 - 1)
        # aa_t_n__1/(max - min) * 2 - self.abstract_normalization_n_offset
        self.abstract_normalization_n_offset = 0
        gamma = 1
        for i in range(self.args.n_step):
            self.abstract_normalization_n_offset += gamma * (self.abstract_normalization_min / (
                        self.abstract_normalization_max - self.abstract_normalization_min) * 2 + 1)
            gamma *= self.args.gamma_action
        print("Normalization: ", self.abstract_normalization_max, self.abstract_normalization_min)


class ReplayBufferPriorityNStepHistoryV6(ReplayBufferPriorityNStepHistory):
    def __init__(self, args, size=1000000, n_step_std=False):
        super().__init__(args, size=size, n_step_std=n_step_std)
        # s_t, s_t_1, s_t_n, \
        # a_t, a_t_1, a_t_n, \
        # pa_t, \
        # aa_t, aa_t_n, \
        # terminal_t, terminal_t_n, reward_t
        self.abstract_normalization_n_offset = None

    def add(self, s_t, s_t_1, s_t_n,
            a_t, a_t_1, a_t_n,
            pa_t,
            aa_t, aa_t_n,
            terminal_t, terminal_t_n, reward_t):
        self.count = min(self.count + 1, self.size)
        self.buffer[self.add_index] = [s_t, s_t_1, s_t_n,
                                       a_t, a_t_1, a_t_n,
                                       pa_t,
                                       aa_t, aa_t_n,
                                       terminal_t, terminal_t_n, reward_t]

        self._it_sum[self.add_index] = self._max_priority ** self._alpha
        self._it_min[self.add_index] = self._max_priority ** self._alpha
        self.add_index = (self.add_index + 1) % self.size

    def sample(self, all=False):
        indices = self._get_random_indices(self.args.batch_size)
        samples = []
        for i in range(self.args.batch_size):
            samples.append(self.buffer[indices[i]])
        s_t, s_t_1, s_t_n, \
        a_t, a_t_1, a_t_n, \
        pa_t, \
        aa_t, aa_t_n, \
        terminal_t, terminal_t_n, reward_t = map(np.asarray, zip(*samples))
        if not self.abstract_normalization_mean is None:
            aa_t_n = np.copy(aa_t_n) / (
                        self.abstract_normalization_max - self.abstract_normalization_min) * 2 - self.abstract_normalization_n_offset
            aa_t = np.copy(aa_t - self.abstract_normalization_min) / (
                        self.abstract_normalization_max - self.abstract_normalization_min) * 2 - 1
        # else:
        #     print("TBD")

        if all:
            return s_t, s_t_1, s_t_n, \
                   a_t, a_t_1, a_t_n, \
                   pa_t, \
                   aa_t, aa_t_n, \
                   terminal_t, terminal_t_n, reward_t
        else:
            return s_t, s_t_1, a_t, a_t_1, reward_t, terminal_t

    # def sample_priority(self, partial_uniform_sampling=0):
    #     total_weight = self._it_sum.sum()
    #     weights = []
    #     p_min = self._it_min.min() / total_weight
    #     max_weight = (p_min * self.count) ** (-self._beta)
    #
    #     if partial_uniform_sampling > 0:
    #         partial_batch = self.args.batch_size - partial_uniform_sampling
    #         indices_1 = self._get_random_indices(partial_uniform_sampling)
    #         indices_2 = self._sample_proportional(partial_batch)
    #         indices = indices_1 + indices_2
    #         for _ in indices_1:
    #             p_sample = 1 / self.count
    #             weight = (p_sample * self.count) ** (-self._beta)
    #             weights.append(weight / max_weight)
    #
    #         for idx in indices_2:
    #             p_sample = self._it_sum[idx] / total_weight
    #             weight = (p_sample * self.count) ** (-self._beta)
    #             weights.append(weight / max_weight)
    #         weights = np.array(weights)
    #     else:
    #         indices = self._sample_proportional(self.args.batch_size)
    #         for idx in indices:
    #             p_sample = self._it_sum[idx] / total_weight
    #             weight = (p_sample * self.count) ** (-self._beta)
    #             weights.append(weight / max_weight)
    #         weights = np.array(weights)
    #
    #     samples = []
    #     for i in range(self.args.batch_size):
    #         samples.append(self.buffer[indices[i]])
    #     s_t, s_t_1, s_t_n__1, s_t_n, \
    #     a_t, a_t_1, a_t_n, \
    #     pa_t, pa_t_n__1, \
    #     aa_t, aa_t_n__1, \
    #     terminal_t, terminal_t_n, reward_t = map(np.asarray, zip(*samples))
    #
    #     if not self.abstract_normalization_mean is None:
    #         aa_t_n__1 = np.copy(aa_t_n__1) / (self.abstract_normalization_max - self.abstract_normalization_min) * 2 - self.abstract_normalization_n_offset
    #         aa_t = np.copy(aa_t - self.abstract_normalization_min) / (self.abstract_normalization_max - self.abstract_normalization_min) * 2 - 1
    #
    #     return s_t, s_t_1, s_t_n__1, s_t_n, \
    #                a_t, a_t_1, a_t_n, \
    #                pa_t, pa_t_n__1, \
    #                aa_t, aa_t_n__1, \
    #                terminal_t, terminal_t_n, reward_t, \
    #                indices, weights

    # return state, next_state, action, next_action, \
    # reward, terminal, indices, weights, n_step_state, n_step_abstract_action, n_step_terminal, next_abstract_action, past_actions, past_next_actions

    def compute_normalized_abstract_actions(self):
        abstract_action_list = []
        for i in range(self.count):
            s_t, s_t_1, s_t_n, \
            a_t, a_t_1, a_t_n, \
            pa_t, \
            aa_t, aa_t_n, \
            terminal_t, terminal_t_n, reward_t = self.buffer[i]
            abstract_action_list.append(aa_t)

        abstract_action_list = np.array(abstract_action_list)
        self.abstract_normalization_mean = np.zeros(aa_t.shape)
        self.abstract_normalization_std = np.zeros(aa_t.shape)
        self.abstract_normalization_max = np.zeros(aa_t.shape)
        self.abstract_normalization_min = np.zeros(aa_t.shape)
        for i in range(aa_t.shape[0]):
            self.abstract_normalization_mean[i] = np.mean(abstract_action_list[:, i])
            self.abstract_normalization_std[i] = np.std(abstract_action_list[:, i])
            if self.abstract_normalization_std[i] < 0.0001:
                self.abstract_normalization_std[i] = 1

            self.abstract_normalization_max[i] = np.percentile(abstract_action_list[:, i], 98)
            self.abstract_normalization_min[i] = np.percentile(abstract_action_list[:, i], 2)
        # sum gamma^i * ((x_i - min)/(max - min) * 2 - 1)
        # = sum gamma^i * (x_i - min)/(max - min) * 2 - sum gamma^i
        # = (sum gamma^i * x_i)/(max - min) * 2 - sum gamma^i * (min/(max - min) * 2 - 1)
        # aa_t_n__1/(max - min) * 2 - self.abstract_normalization_n_offset
        self.abstract_normalization_n_offset = 0
        gamma = 1
        for i in range(self.args.n_step):
            self.abstract_normalization_n_offset += gamma * (self.abstract_normalization_min / (
                        self.abstract_normalization_max - self.abstract_normalization_min) * 2 + 1)
            gamma *= self.args.gamma_action
        print("Normalization: ", self.abstract_normalization_max, self.abstract_normalization_min)


class ReplayBufferPrioritySimple(ReplayBufferPriorityNStepHistory):
    def __init__(self, args, size=1000000, n_step_std=False):
        super().__init__(args, size=size, n_step_std=n_step_std)
        # s_t, s_t_1, s_t_n__1, s_t_n, \
        # a_t, a_t_1, a_t_n, \
        # pa_t, pa_t_n__1, \
        # aa_t, aa_t_n__1, \
        # terminal_t, terminal_t_n, reward_t = buffer.sample(all=True)
        self.abstract_normalization_n_offset = None

    def add(self, s_t, s_t_1, s_t_n,
            a_t, aa_t, aa_t_n,
            terminal_t, terminal_t_n, reward_t):

        self.count = min(self.count + 1, self.size)
        self.buffer[self.add_index] = [s_t, s_t_1, s_t_n,
                                       a_t, aa_t, aa_t_n,
                                       terminal_t, terminal_t_n, reward_t]

        self._it_sum[self.add_index] = self._max_priority ** self._alpha
        self._it_min[self.add_index] = self._max_priority ** self._alpha

        self.add_index = (self.add_index + 1) % self.size

    def sample(self, all=False):
        indices = self._get_random_indices(self.args.batch_size)
        samples = []
        for i in range(self.args.batch_size):
            samples.append(self.buffer[indices[i]])

        s_t, s_t_1, s_t_n, \
        a_t, aa_t, aa_t_n, \
        terminal_t, terminal_t_n, reward_t = map(np.asarray, zip(*samples))
        if not self.abstract_normalization_mean is None:
            aa_t_n = np.copy(aa_t_n) / (
                        self.abstract_normalization_max - self.abstract_normalization_min) * 2 - self.abstract_normalization_n_offset
            aa_t = np.copy(aa_t - self.abstract_normalization_min) / (
                        self.abstract_normalization_max - self.abstract_normalization_min) * 2 - 1
        if all:
            return s_t, s_t_1, s_t_n, \
                   a_t, aa_t, aa_t_n, \
                   terminal_t, terminal_t_n, reward_t
        else:
            return s_t, s_t_1, a_t, reward_t, terminal_t

    # def sample_priority(self, partial_uniform_sampling=0):
    #     total_weight = self._it_sum.sum()
    #     weights = []
    #     p_min = self._it_min.min() / total_weight
    #     max_weight = (p_min * self.count) ** (-self._beta)
    #
    #     if partial_uniform_sampling > 0:
    #         partial_batch = self.args.batch_size - partial_uniform_sampling
    #         indices_1 = self._get_random_indices(partial_uniform_sampling)
    #         indices_2 = self._sample_proportional(partial_batch)
    #         indices = indices_1 + indices_2
    #         for _ in indices_1:
    #             p_sample = 1 / self.count
    #             weight = (p_sample * self.count) ** (-self._beta)
    #             weights.append(weight / max_weight)
    #
    #         for idx in indices_2:
    #             p_sample = self._it_sum[idx] / total_weight
    #             weight = (p_sample * self.count) ** (-self._beta)
    #             weights.append(weight / max_weight)
    #         weights = np.array(weights)
    #     else:
    #         indices = self._sample_proportional(self.args.batch_size)
    #         for idx in indices:
    #             p_sample = self._it_sum[idx] / total_weight
    #             weight = (p_sample * self.count) ** (-self._beta)
    #             weights.append(weight / max_weight)
    #         weights = np.array(weights)
    #
    #     samples = []
    #     for i in range(self.args.batch_size):
    #         samples.append(self.buffer[indices[i]])
    #     s_t, s_t_1, s_t_n__1, s_t_n, \
    #     a_t, a_t_1, a_t_n, \
    #     pa_t, pa_t_n__1, \
    #     aa_t, aa_t_n__1, \
    #     terminal_t, terminal_t_n, reward_t = map(np.asarray, zip(*samples))
    #
    #     if not self.abstract_normalization_mean is None:
    #         aa_t_n__1 = np.copy(aa_t_n__1) / (self.abstract_normalization_max - self.abstract_normalization_min) * 2 - self.abstract_normalization_n_offset
    #         aa_t = np.copy(aa_t - self.abstract_normalization_min) / (self.abstract_normalization_max - self.abstract_normalization_min) * 2 - 1
    #
    #     return s_t, s_t_1, s_t_n__1, s_t_n, \
    #                a_t, a_t_1, a_t_n, \
    #                pa_t, pa_t_n__1, \
    #                aa_t, aa_t_n__1, \
    #                terminal_t, terminal_t_n, reward_t, \
    #                indices, weights

    # return state, next_state, action, next_action, \
    # reward, terminal, indices, weights, n_step_state, n_step_abstract_action, n_step_terminal, next_abstract_action, past_actions, past_next_actions

    def compute_normalized_abstract_actions(self):
        abstract_action_list = []
        for i in range(self.count):
            s_t, s_t_1, s_t_n, \
            a_t, aa_t, aa_t_n, \
            terminal_t, terminal_t_n, reward_t = self.buffer[i]
            abstract_action_list.append(aa_t)

        abstract_action_list = np.array(abstract_action_list)
        self.abstract_normalization_mean = np.zeros(aa_t.shape)
        self.abstract_normalization_std = np.zeros(aa_t.shape)
        self.abstract_normalization_max = np.zeros(aa_t.shape)
        self.abstract_normalization_min = np.zeros(aa_t.shape)
        for i in range(aa_t.shape[0]):
            self.abstract_normalization_mean[i] = np.mean(abstract_action_list[:, i])
            self.abstract_normalization_std[i] = np.std(abstract_action_list[:, i])
            if self.abstract_normalization_std[i] < 0.0001:
                self.abstract_normalization_std[i] = 1

            self.abstract_normalization_max[i] = np.percentile(abstract_action_list[:, i], 98)
            self.abstract_normalization_min[i] = np.percentile(abstract_action_list[:, i], 2)
        # sum gamma^i * ((x_i - min)/(max - min) * 2 - 1)
        # = sum gamma^i * (x_i - min)/(max - min) * 2 - sum gamma^i
        # = (sum gamma^i * x_i)/(max - min) * 2 - sum gamma^i * (min/(max - min) * 2 - 1)
        # aa_t_n__1/(max - min) * 2 - self.abstract_normalization_n_offset
        self.abstract_normalization_n_offset = 0
        gamma = 1
        for i in range(self.args.n_step):
            self.abstract_normalization_n_offset += gamma * (self.abstract_normalization_min / (
                        self.abstract_normalization_max - self.abstract_normalization_min) * 2 + 1)
            gamma *= self.args.gamma_action
        print("Normalization: ", self.abstract_normalization_max, self.abstract_normalization_min)


# for neural network ...
def get_dm_state(dm_timestep):
    data = []
    for key in dm_timestep.observation:
        data.append(np.reshape(dm_timestep.observation[key], [-1]))
    reward = dm_timestep.reward
    if reward is None:
        reward = 0
    return np.concatenate(data, axis=0), reward


def get_rotation_change(new_angle, old_angle):
    # new_angle = new_angle + 180
    # old_angle = old_angle + 180
    # Now ranges between 0, 360
    # There are two path for the distance, we use the smaller one ...
    # dist1 = new_angle - old_angle

    # e.g new = 2, old = 355, distance=  -353 + 360 = 7
    # e.g new = 355, old = 2, distance=  353 - 360 = -7

    distance = new_angle - old_angle
    if distance > 180:
        distance -= 360
    elif distance < -180:
        distance += 360
    return distance


class ReplayBufferPixelPriorityNStep:
    def __init__(self, args, size=1000000):
        self.buffer = {}
        self.args = args
        self.size = size
        self.count = 0
        self.add_index = 0

        alpha = args.alpha
        self._beta = args.beta

        assert alpha >= 0
        self._alpha = alpha

        it_capacity = 1
        while it_capacity < size:
            it_capacity *= 2

        self._it_sum = SumSegmentTree(it_capacity)
        self._it_min = MinSegmentTree(it_capacity)
        self._max_priority = 5

    def add(self, state, action, action_mean, action_std, prev_action, reward, done, next_state, n_step_state,
            n_step_action_mean, n_step_terminal):
        # state = state.astype(np.uint8)
        # next_state = next_state.astype(np.uint8)
        self.count = min(self.count + 1, self.size)
        self.buffer[self.add_index] = [state[0], state[1], action, action_mean, action_std, prev_action, reward,
                                       next_state[0], next_state[1], done, n_step_action_mean, n_step_state[0],
                                       n_step_state[1], n_step_terminal]
        self._it_sum[self.add_index] = self._max_priority ** self._alpha
        self._it_min[self.add_index] = self._max_priority ** self._alpha

        self.add_index = (self.add_index + 1) % self.size

    def sample(self):
        indices = self._get_random_indices(self.args.batch_size)
        samples = []
        for i in range(self.args.batch_size):
            samples.append(self.buffer[indices[i]])
        states_image, states_dense, actions, action_mean, action_std, prev_action, rewards, next_states_image, next_states_dense, done, n_step_action_mean, n_step_state_image, n_step_state_dense, n_step_terminal = map(
            np.asarray, zip(*samples))

        states_image = states_image / 127.5 - 1
        next_states_image = next_states_image / 127.5 - 1
        return [states_image, states_dense], [next_states_image,
                                              next_states_dense], actions, action_mean, action_std, prev_action, rewards, done

    def sample_priority(self):
        indices = self._sample_proportional(self.args.batch_size)
        samples = []
        for i in range(self.args.batch_size):
            samples.append(self.buffer[indices[i]])
        states_image, states_dense, actions, action_mean, action_std, prev_action, rewards, next_states_image, next_states_dense, done, n_step_action_mean, n_step_state_image, n_step_state_dense, n_step_terminal = map(
            np.asarray, zip(*samples))
        total_weight = self._it_sum.sum()
        weights = []
        p_min = self._it_min.min() / total_weight
        max_weight = (p_min * self.count) ** (-self._beta)

        for idx in indices:
            p_sample = self._it_sum[idx] / total_weight
            weight = (p_sample * self.count) ** (-self._beta)
            weights.append(weight / max_weight)
        weights = np.array(weights)

        states_image = states_image / 127.5 - 1
        n_step_state_image = n_step_state_image / 127.5 - 1
        next_states_image = next_states_image / 127.5 - 1
        return [states_image, states_dense], [next_states_image, next_states_dense], actions, action_mean, \
               action_std, prev_action, rewards, done, indices, weights, n_step_action_mean, [n_step_state_image,
                                                                                              n_step_state_dense], n_step_terminal

    def size(self):
        return len(self.buffer)

    def _get_random_indices(self, batch_size):
        idxes = []
        while len(idxes) < batch_size:
            index = np.random.randint(0, self.count - 1)
            idxes.append(index)
        return idxes

    def _sample_proportional(self, batch_size):
        res = []
        p_total = self._it_sum.sum(0, self.count - 1)
        every_range_len = p_total / batch_size
        i = 0
        count = 0
        while len(res) < batch_size:
            count += 1
            if count > 10000:
                idx = np.random.randint(0, self.count)
            else:
                mass = random.random() * every_range_len + i * every_range_len
                idx = self._it_sum.find_prefixsum_idx(mass)
            # if idx < self.agent_history_length + 1:
            #     continue
            # if idx >= self.expert_idx and idx - self.agent_history_length < self.expert_idx:
            #     continue
            # if idx >= self._next_idx and idx - self.agent_history_length < self._next_idx:
            #     continue
            # if np.sum(self.terminal_flags[idx - self.agent_history_length:idx]) > 0:
            #     # if count > 9900:
            #     #     print(self.terminal_flags[idx - 25:idx + 25])
            #     continue
            res.append(idx)
            i += 1
        return res

    def update_priorities(self, idxes, priorities, min_priority=0.001):
        """Update priorities of sampled transitions.
        sets priority of transition at index idxes[i] in buffer
        to priorities[i].
        Parameters
        ----------
        idxes: [int]
            List of idxes of sampled transitions
        priorities: [float]
            List of updated priorities corresponding to
            transitions at the sampled idxes denoted by
            variable `idxes`.
        """
        # Boost expert priority as time goes on ....
        priorities = np.squeeze(priorities)
        assert len(idxes) == priorities.shape[0]
        count = 0
        # print(expert_priority_modifier)
        for idx, priority in zip(idxes, priorities):
            priority = max(priority, 0)
            assert 0 <= idx < self.count

            new_priority = priority * (1 - min_priority) + min_priority
            self._max_priority = max(self._max_priority, new_priority)
            # print(idx, new_priority, expert_idxes[count], count)
            self._it_sum[idx] = (new_priority) ** self._alpha
            self._it_min[idx] = (new_priority) ** self._alpha
            count += 1


class ReplayBufferActionDistHistory:
    def __init__(self, args, size=1000000):
        self.buffer = deque(maxlen=size)
        self.args = args
        self.count = 0
        self.size = size

    def add(self, state, action, reward, done, next_state, estimated_discounted_mean, estimated_discounted_std,
            action_history):
        if self.count < self.size:
            self.count += 1
        self.buffer.append(
            [state, action, reward, next_state, done, estimated_discounted_mean, estimated_discounted_std,
             action_history])

    def sample(self):
        sample = random.sample(self.buffer, self.args.batch_size)
        states, actions, rewards, next_states, done, estimated_discounted_mean, estimated_discounted_std, action_history = map(
            np.asarray, zip(*sample))
        states = np.array(states).reshape(self.args.batch_size, -1)
        next_states = np.array(next_states).reshape(self.args.batch_size, -1)

        return states, next_states, actions, rewards, done, estimated_discounted_mean, estimated_discounted_std, action_history

    def size(self):
        return len(self.buffer)


class ReplayBufferPredictions_baseline:
    def __init__(self, args, size=1000000):
        self.buffer = deque(maxlen=size)
        self.args = args
        self.size = size
        self.count = 0

    def add(self, reward, done, current_state, step_1_state, step_5_state, step_20_state, step_50_state,
            step_1_action, step_5_action, step_20_action, step_50_action,
            action_dist_mean_99, action_dist_std_99, action_dist_mean_90, action_dist_std_90,
            action_dist_mean_75, action_dist_std_75, action_dist_mean_50, action_dist_std_50):
        if self.count < self.size:
            self.count += 1
        self.buffer.append([reward, done, current_state,
                            step_1_state, step_5_state, step_20_state, step_50_state,
                            step_1_action, step_5_action, step_20_action, step_50_action,
                            action_dist_mean_99, action_dist_std_99, action_dist_mean_90, action_dist_std_90,
                            action_dist_mean_75, action_dist_std_75, action_dist_mean_50, action_dist_std_50
                            ])

    def sample(self):
        sample = random.sample(self.buffer, self.args.batch_size)
        rewards, done, current_state, step_1_state, step_5_state, step_20_state, step_50_state, \
        step_1_action, step_5_action, step_20_action, step_50_action, \
        action_dist_mean_99, action_dist_std_99, action_dist_mean_90, action_dist_std_90, \
        action_dist_mean_75, action_dist_std_75, action_dist_mean_50, action_dist_std_50 = map(np.asarray, zip(*sample))
        current_state = np.array(current_state).reshape(self.args.batch_size, -1)

        return rewards, done, current_state, \
               step_1_state, step_5_state, step_20_state, step_50_state, \
               step_1_action, step_5_action, step_20_action, step_50_action, \
               action_dist_mean_99, action_dist_std_99, action_dist_mean_90, action_dist_std_90, \
               action_dist_mean_75, action_dist_std_75, action_dist_mean_50, action_dist_std_50

    def size(self):
        return len(self.buffer)


class ReplayBufferPredictions_v3_0:
    def __init__(self, args, size=1000000):
        self.buffer = deque(maxlen=size)
        self.args = args
        self.size = size
        self.count = 0

    def add(self, reward, done, action_mean, action_std, future_mean, future_std, current_state, step_1_state,
            step_5_state, step_20_state, step_50_state,
            step_1_action, step_5_action, step_20_action, step_50_action):
        if self.count < self.size:
            self.count += 1
        self.buffer.append([reward, done, action_mean, action_std, future_mean, future_std, current_state,
                            step_1_state, step_5_state, step_20_state, step_50_state,
                            step_1_action, step_5_action, step_20_action, step_50_action
                            ])

    def sample(self):
        sample = random.sample(self.buffer, self.args.batch_size)
        rewards, done, action_mean, action_std, future_mean, future_std, current_state, step_1_state, step_5_state, step_20_state, step_50_state, \
        step_1_action, step_5_action, step_20_action, step_50_action = map(np.asarray, zip(*sample))
        current_state = np.array(current_state).reshape(self.args.batch_size, -1)

        return rewards, done, action_mean, action_std, future_mean, future_std, current_state, \
               step_1_state, step_5_state, step_20_state, step_50_state, \
               step_1_action, step_5_action, step_20_action, step_50_action

    def size(self):
        return len(self.buffer)


class ReplayBufferActionDistv10:
    def __init__(self, args, size=1000000):
        self.buffer = deque(maxlen=size)
        self.args = args
        self.size = size

    def add(self, state, action, reward, done, next_state, prev_action_mean,
            prev_action_std, current_action_mean, current_action_std, current_action_hist, next_action_hist, frame_num):
        self.buffer.append([state, action, reward, next_state, done,
                            prev_action_mean, prev_action_std, current_action_mean, current_action_std,
                            current_action_hist, next_action_hist, frame_num])

    def sample(self):
        sample = random.sample(self.buffer, self.args.batch_size)
        states, actions, rewards, next_states, done, \
        prev_action_mean, prev_action_std, current_action_mean, current_action_std, \
        current_action_hist, next_action_hist, frame_num = map(np.asarray, zip(*sample))

        states = np.array(states).reshape(self.args.batch_size, -1)
        next_states = np.array(next_states).reshape(self.args.batch_size, -1)
        frame_num = np.expand_dims(frame_num, axis=1)

        current_action_hist = np.reshape(current_action_hist, [self.args.batch_size, -1])
        next_action_hist = np.reshape(next_action_hist, [self.args.batch_size, -1])

        prev_action_mean = np.array(prev_action_mean)
        prev_action_std = np.array(prev_action_std)
        return states, next_states, actions, rewards, done, \
               prev_action_mean, prev_action_std, current_action_mean, current_action_std, current_action_hist, next_action_hist, frame_num


class ReplayBufferActionDistv9:
    def __init__(self, args, size=1000000):
        self.buffer = deque(maxlen=size)
        self.args = args
        self.size = size

    def add(self, state, action, reward, done, next_state, prev_action_mean,
            prev_action_std, current_action_mean, current_action_std, frame_num):
        self.buffer.append([state, action, reward, next_state, done,
                            prev_action_mean, prev_action_std, current_action_mean, current_action_std, frame_num])

    def sample(self):
        sample = random.sample(self.buffer, self.args.batch_size)
        states, actions, rewards, next_states, done, \
        prev_action_mean, prev_action_std, current_action_mean, current_action_std, frame_num = map(np.asarray,
                                                                                                    zip(*sample))

        states = np.array(states).reshape(self.args.batch_size, -1)
        next_states = np.array(next_states).reshape(self.args.batch_size, -1)
        frame_num = np.expand_dims(frame_num, axis=1)

        prev_action_mean = np.array(prev_action_mean)
        prev_action_std = np.array(prev_action_std)
        return states, next_states, actions, rewards, done, \
               prev_action_mean, prev_action_std, current_action_mean, current_action_std, frame_num

    def size(self):
        return len(self.buffer)


class ReplayBufferActionDistv8:
    def __init__(self, args, size=1000000):
        self.buffer = deque(maxlen=size)
        self.args = args
        self.size = size

    def add(self, state, action, reward, done, next_state, prev_action_mean,
            prev_action_std, current_action_mean, current_action_std,
            prev_action_history, current_action_history,
            target_99, target_90, target_75, target_50, frame_num):
        self.buffer.append([state, action, reward, next_state, done,
                            prev_action_mean, prev_action_std, current_action_mean, current_action_std,
                            prev_action_history, current_action_history,
                            target_99, target_90, target_75, target_50, frame_num])

    def sample(self):
        sample = random.sample(self.buffer, self.args.batch_size)
        states, actions, rewards, next_states, done, \
        prev_action_mean, prev_action_std, current_action_mean, current_action_std, prev_action_history, \
        current_action_history, target_99, target_90, target_75, target_50, frame_num = map(np.asarray, zip(*sample))

        states = np.array(states).reshape(self.args.batch_size, -1)
        next_states = np.array(next_states).reshape(self.args.batch_size, -1)
        frame_num = np.expand_dims(frame_num, axis=1)

        prev_action_mean = np.array(prev_action_mean)
        prev_action_std = np.array(prev_action_std)
        prev_action_history = np.reshape(prev_action_history, [self.args.batch_size, -1])
        current_action_history = np.reshape(current_action_history, [self.args.batch_size, -1])
        return states, next_states, actions, rewards, done, \
               prev_action_mean, prev_action_std, current_action_mean, current_action_std, prev_action_history, current_action_history, \
               target_99, target_90, target_75, target_50, frame_num

    def size(self):
        return len(self.buffer)


class ReplayBufferActionDistv7:
    def __init__(self, args, size=1000000):
        self.buffer = deque(maxlen=size)
        self.args = args
        self.size = size

    def add(self, state, action, reward, done, next_state, prev_action_mean,
            prev_action_std, current_action_mean, current_action_std,
            prev_action_history, current_action_history, discounted_action, frame_num):
        self.buffer.append([state, action, reward, next_state, done,
                            prev_action_mean, prev_action_std, current_action_mean, current_action_std,
                            prev_action_history, current_action_history, discounted_action, frame_num])

    def sample(self):
        sample = random.sample(self.buffer, self.args.batch_size)
        states, actions, rewards, next_states, done, \
        prev_action_mean, prev_action_std, current_action_mean, current_action_std, prev_action_history, current_action_history, discounted_action, frame_num = map(
            np.asarray, zip(*sample))

        states = np.array(states).reshape(self.args.batch_size, -1)
        next_states = np.array(next_states).reshape(self.args.batch_size, -1)
        frame_num = np.expand_dims(frame_num, axis=1)

        prev_action_mean = np.array(prev_action_mean)
        prev_action_std = np.array(prev_action_std)
        prev_action_history = np.reshape(prev_action_history, [self.args.batch_size, -1])
        current_action_history = np.reshape(current_action_history, [self.args.batch_size, -1])
        return states, next_states, actions, rewards, done, \
               prev_action_mean, prev_action_std, current_action_mean, current_action_std, prev_action_history, current_action_history, discounted_action, frame_num

    def size(self):
        return len(self.buffer)


class ReplayBufferActionDistv6:
    def __init__(self, args, size=1000000):
        self.buffer = deque(maxlen=size)
        self.args = args
        self.size = size

    def add(self, state, action, reward, done, next_state, prev_action_mean,
            prev_action_std, current_action_mean, current_action_std,
            prev_action_history, current_action_history, frame_num):
        self.buffer.append([state, action, reward, next_state, done,
                            prev_action_mean, prev_action_std, current_action_mean, current_action_std,
                            prev_action_history, current_action_history, frame_num])

    def sample(self):
        sample = random.sample(self.buffer, self.args.batch_size)
        states, actions, rewards, next_states, done, \
        prev_action_mean, prev_action_std, current_action_mean, current_action_std, prev_action_history, current_action_history, frame_num = map(
            np.asarray, zip(*sample))

        states = np.array(states).reshape(self.args.batch_size, -1)
        next_states = np.array(next_states).reshape(self.args.batch_size, -1)
        frame_num = np.expand_dims(frame_num, axis=1)

        prev_action_mean = np.array(prev_action_mean)
        prev_action_std = np.array(prev_action_std)
        prev_action_history = np.reshape(prev_action_history, [self.args.batch_size, -1])
        current_action_history = np.reshape(current_action_history, [self.args.batch_size, -1])
        return states, next_states, actions, rewards, done, \
               prev_action_mean, prev_action_std, current_action_mean, current_action_std, prev_action_history, current_action_history, frame_num

    def size(self):
        return len(self.buffer)


class ReplayBufferActionDistv5:
    def __init__(self, args, size=1000000):
        self.buffer = deque(maxlen=size)
        self.args = args
        self.size = size

    def add(self, state, action, reward, done, next_state, prev_action_mean,
            prev_action_std, current_action_mean, current_action_std, frame_num):
        self.buffer.append([state, action, reward, next_state, done,
                            prev_action_mean, prev_action_std, current_action_mean, current_action_std, frame_num])

    def sample(self):
        sample = random.sample(self.buffer, self.args.batch_size)
        states, actions, rewards, next_states, done, \
        prev_action_mean, prev_action_std, current_action_mean, current_action_std, frame_num = map(np.asarray,
                                                                                                    zip(*sample))

        states = np.array(states).reshape(self.args.batch_size, -1)
        next_states = np.array(next_states).reshape(self.args.batch_size, -1)
        frame_num = np.expand_dims(frame_num, axis=1)

        prev_action_mean = np.array(prev_action_mean)
        prev_action_std = np.array(prev_action_std)
        return states, next_states, actions, rewards, done, \
               prev_action_mean, prev_action_std, current_action_mean, current_action_std, frame_num

    def size(self):
        return len(self.buffer)


class ReplayBufferActionDistv4:
    def __init__(self, args, size=1000000):
        self.buffer = deque(maxlen=size)
        self.args = args
        self.size = size

    def add(self, state, action, reward, done, next_state, trajectory_mean, trajectory_std,
            prev_action_mean, prev_action_std, current_action_mean, current_action_std, frame_num):
        self.buffer.append([state, action, reward, next_state, done, trajectory_mean, trajectory_std,
                            prev_action_mean, prev_action_std, current_action_mean, current_action_std, frame_num])

    def sample(self):
        sample = random.sample(self.buffer, self.args.batch_size)
        states, actions, rewards, next_states, done, trajectory_mean, trajectory_std, \
        prev_action_mean, prev_action_std, current_action_mean, current_action_std, frame_num = map(np.asarray,
                                                                                                    zip(*sample))

        states = np.array(states).reshape(self.args.batch_size, -1)
        next_states = np.array(next_states).reshape(self.args.batch_size, -1)
        frame_num = np.expand_dims(frame_num, axis=1)

        prev_action_mean = np.array(prev_action_mean)
        prev_action_std = np.array(prev_action_std)
        return states, next_states, actions, rewards, done, trajectory_mean, trajectory_std, \
               prev_action_mean, prev_action_std, current_action_mean, current_action_std, frame_num

    def size(self):
        return len(self.buffer)


class ReplayBufferActionDistv3:
    def __init__(self, args, size=1000000):
        self.buffer = deque(maxlen=size)
        self.args = args
        self.size = size

    def add(self, state, action, reward, done, next_state, estimated_discounted_mean, estimated_discounted_std,
            action_mean, action_std, frame_num):
        self.buffer.append(
            [state, action, reward, next_state, done, estimated_discounted_mean, estimated_discounted_std, action_mean,
             action_std, frame_num])

    def sample(self):
        sample = random.sample(self.buffer, self.args.batch_size)
        states, actions, rewards, next_states, done, estimated_discounted_mean, estimated_discounted_std, action_mean, action_std, frame_num = map(
            np.asarray, zip(*sample))
        states = np.array(states).reshape(self.args.batch_size, -1)
        next_states = np.array(next_states).reshape(self.args.batch_size, -1)
        frame_num = np.expand_dims(frame_num, axis=1)
        return states, next_states, actions, rewards, done, estimated_discounted_mean, estimated_discounted_std, action_mean, action_std, frame_num

    def size(self):
        return len(self.buffer)


class ReplayBufferActionDistv2:
    def __init__(self, args, size=1000000):
        self.buffer = deque(maxlen=size)
        self.args = args

    def add(self, state, action, reward, done, next_state, estimated_discounted_mean, estimated_discounted_std,
            action_mean, action_std):
        self.buffer.append(
            [state, action, reward, next_state, done, estimated_discounted_mean, estimated_discounted_std, action_mean,
             action_std])

    def sample(self):
        sample = random.sample(self.buffer, self.args.batch_size)
        states, actions, rewards, next_states, done, estimated_discounted_mean, estimated_discounted_std, action_mean, action_std = map(
            np.asarray, zip(*sample))
        states = np.array(states).reshape(self.args.batch_size, -1)
        next_states = np.array(next_states).reshape(self.args.batch_size, -1)
        return states, next_states, actions, rewards, done, estimated_discounted_mean, estimated_discounted_std, action_mean, action_std

    def size(self):
        return len(self.buffer)


class ReplayBufferActionDist:
    def __init__(self, args, size=1000000):
        self.buffer = deque(maxlen=size)
        self.args = args

    def add(self, state, action, reward, done, next_state, estimated_discounted_mean, estimated_discounted_std):
        self.buffer.append(
            [state, action, reward, next_state, done, estimated_discounted_mean, estimated_discounted_std])

    def sample(self):
        sample = random.sample(self.buffer, self.args.batch_size)
        states, actions, rewards, next_states, done, estimated_discounted_mean, estimated_discounted_std = map(
            np.asarray, zip(*sample))
        states = np.array(states).reshape(self.args.batch_size, -1)
        next_states = np.array(next_states).reshape(self.args.batch_size, -1)
        return states, next_states, actions, rewards, done, estimated_discounted_mean, estimated_discounted_std

    def size(self):
        return len(self.buffer)


class ReplayBufferPixelHistory:
    def __init__(self, args, obs_shape, action_dim, size=1000000):
        self.num_trajectories = size // args.max_eps_len
        self.trajectory_dict = {}
        self.trajectory_length_storage = np.zeros((self.num_trajectories,))

        self.current_trajectory_index = -1
        self.trajectory_size = 0
        self.obs_space = obs_shape
        self.action_dim = action_dim
        self.args = args

        self.state_buffer = np.zeros((self.args.batch_size, obs_shape[0], obs_shape[1], self.args.history_length + 1),
                                     dtype=np.float32)
        self.action_buffer = np.zeros((self.args.batch_size, action_dim), dtype=np.float32)
        self.reward_buffer = np.zeros((self.args.batch_size, 1), dtype=np.float32)
        self.done_buffer = np.zeros((self.args.batch_size, 1), dtype=np.float32)

    def add_new_trajectory(self):
        self.current_trajectory_index = (self.current_trajectory_index + 1) % self.num_trajectories
        self.trajectory_dict[self.current_trajectory_index] = {}
        self.trajectory_size = min(1 + self.trajectory_size, self.num_trajectories)
        self.trajectory_length_storage[self.current_trajectory_index] = 0

    def add(self, state, action, reward, done):
        state = state.astype(np.uint8)
        self.trajectory_dict[self.current_trajectory_index][
            self.trajectory_length_storage[self.current_trajectory_index]] = [state, action, reward, done]
        self.trajectory_length_storage[self.current_trajectory_index] += 1

    def sample(self):
        trajectory_indices = np.random.randint(0, self.trajectory_size, size=self.args.batch_size)
        inner_indices = np.random.randint(0, self.trajectory_length_storage[trajectory_indices],
                                          size=self.args.batch_size)
        for i in range(self.args.batch_size):
            for j in range(self.args.history_length + 1):
                augmented_inner_index = min(inner_indices[i] + j,
                                            self.trajectory_length_storage[trajectory_indices[i]] - 1)
                state, action, reward, done = self.trajectory_dict[trajectory_indices[i]][augmented_inner_index]
                self.state_buffer[i, :, :, j] = state[:, :, 0]
                if j == (self.args.history_length - 2):
                    self.action_buffer[i] = action
                    self.reward_buffer[i] = reward
                    self.done_buffer[i] = done

        return self.state_buffer[:, :, :, 0:self.args.history_length], self.state_buffer[:, :, :,
                                                                       1:], self.action_buffer, self.reward_buffer, self.done_buffer


class ReplayBufferPixelToy:
    def __init__(self, args, size=1000000):
        self.buffer = deque(maxlen=size)
        self.args = args

    def add(self, state, action, reward, done, next_state):
        # state = state.astype(np.uint8)
        # next_state = next_state.astype(np.uint8)
        self.buffer.append([state[0], state[1], action, reward, next_state[0], next_state[1], done])

    def sample(self):
        sample = random.sample(self.buffer, self.args.batch_size)
        states_image, states_dense, actions, rewards, next_states_image, next_states_dense, done = map(np.asarray,
                                                                                                       zip(*sample))

        states_image = states_image / 127.5 - 1
        next_states_image = next_states_image / 127.5 - 1
        return [states_image, states_dense], [next_states_image, next_states_dense], actions, rewards, done

    def size(self):
        return len(self.buffer)


class ReplayBufferPixelToyNoStd:
    def __init__(self, args, size=1000000):
        self.buffer = deque(maxlen=size)
        self.args = args

    def add(self, state, action, prev_action, reward, done, next_state):
        # state = state.astype(np.uint8)
        # next_state = next_state.astype(np.uint8)
        self.buffer.append([state[0], state[1], action, prev_action, reward, next_state[0], next_state[1], done])

    def sample(self):
        sample = random.sample(self.buffer, self.args.batch_size)
        states_image, states_dense, actions, prev_action, rewards, next_states_image, next_states_dense, done = map(
            np.asarray, zip(*sample))

        states_image = states_image / 127.5 - 1
        next_states_image = next_states_image / 127.5 - 1
        return [states_image, states_dense], [next_states_image, next_states_dense], actions, prev_action, rewards, done

    def size(self):
        return len(self.buffer)


class ReplayBufferPixelToyFuturePlan:
    def __init__(self, args, size=1000000):
        self.buffer = deque(maxlen=size)
        self.args = args

    def add(self, state, action, action_mean, action_std, prev_action, reward, done, next_state):
        # state = state.astype(np.uint8)
        # next_state = next_state.astype(np.uint8)
        self.buffer.append(
            [state[0], state[1], action, action_mean, action_std, prev_action, reward, next_state[0], next_state[1],
             done])

    def sample(self):
        sample = random.sample(self.buffer, self.args.batch_size)
        states_image, states_dense, actions, action_mean, action_std, prev_action, rewards, next_states_image, next_states_dense, done = map(
            np.asarray, zip(*sample))

        states_image = states_image / 127.5 - 1
        next_states_image = next_states_image / 127.5 - 1
        return [states_image, states_dense], [next_states_image,
                                              next_states_dense], actions, action_mean, action_std, prev_action, rewards, done

    def size(self):
        return len(self.buffer)


class ReplayBufferPixelToyFuturePlanv2:
    def __init__(self, args, size=1000000):
        self.buffer = deque(maxlen=size)
        self.args = args

    def add(self, state, action, action_mean, action_std, prev_action, prev_actions, reward, done, next_state):
        # state = state.astype(np.uint8)
        # next_state = next_state.astype(np.uint8)
        self.buffer.append(
            [state[0], state[1], action, action_mean, action_std, prev_action, prev_actions, reward, next_state[0],
             next_state[1], done])

    def sample(self):
        sample = random.sample(self.buffer, self.args.batch_size)
        states_image, states_dense, actions, action_mean, action_std, prev_action, prev_actions, rewards, next_states_image, next_states_dense, done = map(
            np.asarray, zip(*sample))

        states_image = states_image / 127.5 - 1
        next_states_image = next_states_image / 127.5 - 1
        return [states_image, states_dense], [next_states_image,
                                              next_states_dense], actions, action_mean, action_std, prev_action, np.copy(
            prev_actions), rewards, done

    def size(self):
        return len(self.buffer)


class ReplayBufferPixel:
    def __init__(self, args, size=1000000):
        self.buffer = deque(maxlen=size)
        self.args = args

    def add(self, state, action, reward, done, next_state):
        state = state.astype(np.uint8)
        next_state = next_state.astype(np.uint8)
        self.buffer.append([state, action, reward, next_state, done])

    def sample(self):
        sample = random.sample(self.buffer, self.args.batch_size)
        states, actions, rewards, next_states, done = map(np.asarray, zip(*sample))
        states = states / 127.5 - 1
        next_states = next_states / 127.5 - 1
        return states, next_states, actions, rewards, done

    def size(self):
        return len(self.buffer)


class ReplayBufferDistPixel:
    def __init__(self, args, size=1000000):
        self.buffer = deque(maxlen=size)
        self.args = args

    def add(self, state, action, action_mean, action_std, reward, done, next_state):
        self.buffer.append([state, action, action_mean, action_std, reward, next_state, done])

    def sample(self):
        sample = random.sample(self.buffer, self.args.batch_size)
        states, actions, action_mean, action_std, rewards, next_states, done = map(np.asarray, zip(*sample))
        states = states / 255 * 2 - 1
        next_states = next_states / 255 * 2 - 1
        return states, next_states, actions, action_mean, action_std, rewards, done

    def size(self):
        return len(self.buffer)


class ReplayBufferDist:
    def __init__(self, args, size=1000000):
        self.buffer = deque(maxlen=size)
        self.args = args

    def add(self, state, action, action_mean, action_std, reward, done, next_state):
        self.buffer.append([state, action, action_mean, action_std, reward, next_state, done])

    def sample(self):
        sample = random.sample(self.buffer, self.args.batch_size)
        states, actions, action_mean, action_std, rewards, next_states, done = map(np.asarray, zip(*sample))
        states = np.array(states).reshape(self.args.batch_size, -1)
        next_states = np.array(next_states).reshape(self.args.batch_size, -1)
        return states, next_states, actions, action_mean, action_std, rewards, done

    def size(self):
        return len(self.buffer)


class ReplayBuffer:
    def __init__(self, args, size=1000000):
        self.buffer = deque(maxlen=size)
        self.args = args

    def add(self, state, action, reward, done, next_state):
        self.buffer.append([state, action, reward, next_state, done])

    def sample(self):
        sample = random.sample(self.buffer, self.args.batch_size)
        states, actions, rewards, next_states, done = map(np.asarray, zip(*sample))
        states = np.array(states).reshape(self.args.batch_size, -1)
        next_states = np.array(next_states).reshape(self.args.batch_size, -1)
        return states, next_states, actions, rewards, done

    def size(self):
        return len(self.buffer)


class ReplayBufferTransition:
    def __init__(self, args, size=1000000):
        self.buffer = deque(maxlen=size)
        self.args = args

    def add(self, state, action, timestep, target_state):
        self.buffer.append([state, action, np.array([timestep]), target_state])

    def sample(self):
        sample = random.sample(self.buffer, self.args.batch_size)
        states, actions, timestep, target_states = map(np.asarray, zip(*sample))
        states = np.array(states).reshape(self.args.batch_size, -1)
        target_states = np.array(target_states).reshape(self.args.batch_size, -1)
        return states, actions, timestep, target_states

    def size(self):
        return len(self.buffer)


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


# def generate_gif(frame_number, frames_for_gif, reward, path):
#     """
#         Args:
#             frame_number: Integer, determining the number of the current frame
#             frames_for_gif: A sequence of (210, 160, 3) frames of an Atari game in RGB
#             reward: Integer, Total reward of the episode that es ouputted as a gif
#             path: String, path where gif is saved
#     """
#     for idx, frame_idx in enumerate(frames_for_gif):
#         frames_for_gif[idx] = resize(frame_idx, (420, 320, 3),
#
#     imageio.mimsave(f'{path}{"ATARI_frame_{0}_reward_{1}.gif".format(frame_number, reward)}',#                                      preserve_range=True, order=0).astype(np.uint8)
#                     frames_for_gif, duration=1 / 30)


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

