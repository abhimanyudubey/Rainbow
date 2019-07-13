from collections import namedtuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


Transition = namedtuple(
    'Transition', ('timestep', 'state', 'action', 'reward', 'nonterminal'))
blank_trans = Transition(
    0, torch.zeros(84, 84, dtype=torch.uint8), None, 0, False)


class SegmentTree():
    def __init__(self, size):
        self.index = 0
        self.size = size
        self.full = False  # Used to track actual capacity
        self.sum_tree = np.zeros((2 * size - 1, ), dtype=np.float32)
        self.data = np.array([None] * size)  # Wrap-around cyclic buffer
        self.max = 1  # Initial max value to return (1 = 1^ω)

    def _propagate(self, index, value):
        parent = (index - 1) // 2
        left, right = 2 * parent + 1, 2 * parent + 2
        self.sum_tree[parent] = self.sum_tree[left] + self.sum_tree[right]
        if parent != 0:
            self._propagate(parent, value)

    def update(self, index, value):
        self.sum_tree[index] = value  # Set new value
        self._propagate(index, value)  # Propagate value
        self.max = max(value, self.max)

    def append(self, data, value):
        self.data[self.index] = data  # Store data in underlying data structure
        self.update(self.index + self.size - 1, value)  # Update tree
        self.index = (self.index + 1) % self.size  # Update index
        self.full = self.full or self.index == 0  # Save when capacity reached
        self.max = max(value, self.max)

    # Searches for the location of a value in sum tree
    def _retrieve(self, index, value):
        left, right = 2 * index + 1, 2 * index + 2
        if left >= len(self.sum_tree):
            return index
        elif value <= self.sum_tree[left]:
            return self._retrieve(left, value)
        else:
            return self._retrieve(right, value - self.sum_tree[left])

    def find(self, value):
        index = self._retrieve(0, value)  # Search for index of item from root
        data_index = index - self.size + 1
        return (self.sum_tree[index], data_index, index)

    # Returns data given a data index
    def get(self, data_index):
        return self.data[data_index % self.size]

    def total(self):
        return self.sum_tree[0]


def normal_init(m, mean, std):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()


class Generator(nn.Module):
    # initializers
    def __init__(self, d=128):
        super(Generator, self).__init__()
        self.deconv1_1 = nn.ConvTranspose2d(100, d*4, 4, 1, 0)
        self.deconv1_1_bn = nn.BatchNorm2d(d*4)
        self.deconv1_2 = nn.ConvTranspose2d(2, d*4, 4, 1, 0)
        self.deconv1_2_bn = nn.BatchNorm2d(d*4)
        self.deconv2 = nn.ConvTranspose2d(d*8, d*4, 4, 2, 1)
        self.deconv2_bn = nn.BatchNorm2d(d*4)
        self.deconv3 = nn.ConvTranspose2d(d*4, d*2, 4, 2, 1)
        self.deconv3_bn = nn.BatchNorm2d(d*2)
        # self.deconv4 = nn.ConvTranspose2d(d, 3, 4, 2, 1)
        self.deconv4 = nn.ConvTranspose2d(d*2, d, 4, 2, 1)
        self.deconv4_bn = nn.BatchNorm2d(d)
        self.deconv5 = nn.ConvTranspose2d(d, 3, 4, 2, 1)

        # self.conv1 = nn.Conv2d()

    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    # forward method
    # def forward(self, input):
    def forward(self, input, label, base):
        x = F.leaky_relu(self.deconv1_1_bn(self.deconv1_1(input)), 0.2)
        y = F.leaky_relu(self.deconv1_2_bn(self.deconv1_2(label)), 0.2)
        x = torch.cat([x, y], 1)
        x = F.leaky_relu(self.deconv2_bn(self.deconv2(x)), 0.2)
        x = F.leaky_relu(self.deconv3_bn(self.deconv3(x)), 0.2)
        # x = F.tanh(self.deconv4(x))
        x = F.leaky_relu(self.deconv4_bn(self.deconv4(x)), 0.2)
        x = F.tanh(self.deconv5(x))

        return x + base


class Discriminator(nn.Module):
    # initializers
    def __init__(self, d=128):
        super(Discriminator, self).__init__()
        self.conv1_1 = nn.Conv2d(3, d/2, 4, 2, 1)
        self.conv1_2 = nn.Conv2d(2, d/2, 4, 2, 1)
        self.conv2 = nn.Conv2d(d, d*2, 4, 2, 1)
        self.conv2_bn = nn.BatchNorm2d(d*2)
        self.conv3 = nn.Conv2d(d*2, d*4, 4, 2, 1)
        self.conv3_bn = nn.BatchNorm2d(d*4)
        # self.conv4 = nn.Conv2d(d*4, 1, 4, 1, 0)
        self.conv4 = nn.Conv2d(d*4, d*8, 4, 2, 1)
        self.conv4_bn = nn.BatchNorm2d(d*8)
        self.conv5 = nn.Conv2d(d*8, 1, 4, 1, 0)

    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    # forward method
    # def forward(self, input):
    def forward(self, input, label):
        x = F.leaky_relu(self.conv1_1(input), 0.2)
        y = F.leaky_relu(self.conv1_2(label), 0.2)
        x = torch.cat([x, y], 1)
        x = F.leaky_relu(self.conv2_bn(self.conv2(x)), 0.2)
        x = F.leaky_relu(self.conv3_bn(self.conv3(x)), 0.2)
        # x = F.sigmoid(self.conv4(x))
        x = F.leaky_relu(self.conv4_bn(self.conv4(x)), 0.2)
        x = F.sigmoid(self.conv5(x))

        return x


class ReplayMemory():
    def __init__(self, args, capacity):
        self.device = args.device
        self.capacity = capacity
        self.history = args.history_length
        self.discount = args.discount
        self.n = args.multi_step
        self.priority_weight = args.priority_weight
        self.priority_exponent = args.priority_exponent
        self.t = 0  # Internal episode timestep counter
        self.transitions = SegmentTree(capacity)

    # Adds state and action at time t, reward and terminal at time t + 1
    def append(self, state, action, reward, terminal):
        state = state[-1].mul(255).to(
            dtype=torch.uint8, device=torch.device('cpu'))
        self.transitions.append(
            Transition(self.t, state, action, reward, not terminal),
            self.transitions.max)
        self.t = 0 if terminal else self.t + 1  # Start new episodes with t = 0

    # Returns a transition with blank states w here appropriate
    def _get_transition(self, idx):
        transition = np.array([None] * (self.history + self.n))
        transition[self.history - 1] = self.transitions.get(idx)
        for t in range(self.history - 2, -1, -1):  # e.g. 2 1 0
            if transition[t + 1].timestep == 0:
                transition[t] = blank_trans  # If future frame has timestep 0
            else:
                transition[t] = self.transitions.get(
                    idx - self.history + 1 + t)
                for t in range(self.history, self.history + self.n):
                    if transition[t - 1].nonterminal:
                        transition[t] = self.transitions.get(
                            idx - self.history + 1 + t)
                    else:
                        transition[t] = blank_trans
        return transition

    # Returns a valid sample from a segment
    def _get_sample_from_segment(self, segment, i):
        valid = False
        while not valid:
            sample = np.random.uniform(i * segment, (i + 1) * segment)
            prob, idx, tree_idx = self.transitions.find(sample)
            if (self.transitions.index - idx) % self.capacity > self.n and \
                    (idx - self.transitions.index) % self.capacity >=\
                    self.history and prob != 0:
                valid = True

        # Retrieve all required transition data (from t - h to t + n)
        transition = self._get_transition(idx)
        # Create un-discretised state and nth next state
        state = torch.stack(
            [trans.state for trans in transition[:self.history]]).to(
            dtype=torch.float32, device=self.device).div_(255)
        next_state = torch.stack(
            [trans.state for trans in transition[self.n:self.n +
            self.history]]).to(dtype=torch.float32,
            device=self.device).div_(255)
        # Discrete action to be used as index
        action = torch.tensor(
            [transition[self.history - 1].action],
            dtype=torch.int64, device=self.device)
        # Calculate truncated n-step discounted return R^n = Σ_k=0->n-1
        # (γ^k)R_t+k+1 (note that invalid nth next states have reward 0)
        R = torch.tensor(
            [sum(self.discount ** n * transition[self.history + n - 1].reward
            for n in range(self.n))], dtype=torch.float32, device=self.device)
    # Mask for non-terminal nth next states
        nonterminal = torch.tensor(
            [transition[self.history + self.n - 1].nonterminal],
            dtype=torch.float32, device=self.device)

        return prob, idx, tree_idx, state, action, R, next_state, nonterminal

    def sample(self, batch_size):
        p_total = self.transitions.total()
        segment = p_total / batch_size
        batch = [
            self._get_sample_from_segment(segment, i)
            for i in range(batch_size)]  # Get batch of valid samples
        probs, idxs, tree_idxs, states, actions, returns, next_states, nonterminals = zip(*batch)
        states, next_states, = torch.stack(states), torch.stack(next_states)
        actions, returns, nonterminals = torch.cat(actions), torch.cat(returns), torch.stack(nonterminals)
        probs = np.array(probs, dtype=np.float32) / p_total  # Calculate normalised probabilities
        capacity = self.capacity if self.transitions.full else self.transitions.index
        weights = (capacity * probs) ** -self.priority_weight  # Compute importance-sampling weights w
        weights = torch.tensor(weights / weights.max(), dtype=torch.float32, device=self.device)  # Normalise by max importance-sampling weight from batch
        return tree_idxs, states, actions, returns, next_states, nonterminals, weights


    def update_priorities(self, idxs, priorities):
        priorities = np.power(priorities, self.priority_exponent)
        [self.transitions.update(idx, priority) for idx, priority in zip(idxs, priorities)]

    # Set up internal state for iterator
    def __iter__(self):
        self.current_idx = 0
        return self

    # Return valid states for validation
    def __next__(self):
        if self.current_idx == self.capacity:
            raise StopIteration
        # Create stack of states
        state_stack = [None] * self.history
        state_stack[-1] = self.transitions.data[self.current_idx].state
        prev_timestep = self.transitions.data[self.current_idx].timestep
        for t in reversed(range(self.history - 1)):
            if prev_timestep == 0:
                state_stack[t] = blank_trans.state
            else:
                state_stack[t] = self.transitions.data[self.current_idx + t - self.history + 1].state
                prev_timestep -= 1
        state = torch.stack(state_stack, 0).to(dtype=torch.float32, device=self.device).div_(255)  # Agent will turn into batch
        self.current_idx += 1
        return state
