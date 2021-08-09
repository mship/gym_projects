import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.optimizers import Adam
import numpy as np

# set up a class for our DQN
class DDQN(keras.Model):
    def __init__(self, n_actions):
        super(DDQN, self).__init__()
        self.dense1 = keras.layers.Dense(128, activation='relu')
        self.dense2 = keras.layers.Dense(128, activation='relu')
        self.V = keras.layers.Dense(1, activation=None)
        self.A = keras.layers.Dense(n_actions, activation=None)

    def call(self, state):
        x = self.dense1(state)
        x = self.dense2(x)
        V = self.V(x)
        A = self.A(x)
        
        Q = (V + (A - tf.math.reduce_mean(A, axis=1, keepdims=True)))

        return Q

    def advantage(self, state):
        x = self.dense1(state)
        x = self.dense2(x)
        a = self.A(x)
        return a

# class for storing memory of Agent and creating samples for training
class ReplayBuffer():
    def __init__(self, max_size, input_shape):
        self.mem_size = max_size
        self.mem_cntr = 0
        # here we initialize numpy arrays as 0 
        self.state_memory = np.zeros((self.mem_size, *input_shape), dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, *input_shape), dtype = np.float32)
        self.action_memory = np.zeros(self.mem_size, dtype=np.int32)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool)

    # method to count store info in and count through various arrays
    def store_transition(self, state, action, reward, new_state, done):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = new_state
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done
        self.mem_cntr += 1

    # method to check memory and make sure we aren't sampling those initial 0s
    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, batch_size, replace=False)
        states = self.state_memory[batch]
        new_states = self.new_state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        dones = self.terminal_memory[batch]

        return states, actions, rewards, new_states, dones

# our Agent! 
# class to use the DDQN in order to learn how to win the gym environment
class Agent():
    #initialize, typical q learning variables here
    def __init__(self, alpha, gamma, n_actions, batch_size, input_dims, 
        mem_size, replace):
        self.action_space = [i for i in range(n_actions)] # 4 actions for lunarlander
        self.gamma = gamma
        self.epsilon = 1
        self.eps_dec = .001
        self.eps_end = .01
        self.replace = replace
        self.batch_size = batch_size

        self.learn_step_counter = 0
        self.memory = ReplayBuffer(mem_size, input_dims)
        self.q_next = DDQN(n_actions, 128, 128)
        self.q_eval = DDQN(n_actions, 128, 128)
        # optimizers are occasionally failing here - not sure why
        # it does not appear to affect the learning agent
        self.q_next.compile(optimizer=Adam(learning_rate=alpha), loss='mean_squared_error')
        self.q_eval.compile(optimizer=Adam(learning_rate=alpha), loss='mean_squared_error')

    def store_transition(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)

    #method to choose action to take
    def choose_action(self, observation):
        # employ epsilon to create a series of random actions - overrides the initialized 0s
        if np.random.random() < self.epsilon:
            action = np.random.choice(self.action_space)
        else:
            state = np.array([observation])
            # I tried calling the 'call' method here with no apparent change to learning
            # advantage does seem to be faster, though
            actions = self.q_next.advantage(state)
            action = tf.math.argmax(actions, axis=1).numpy()[0]

        return action

    # method to update q values in agent as we learn
    # this method seems to have a lot of performance issues as I've written it
    def learn(self):
        # we learn in batches. if batch is not complete, skip
        if self.memory.mem_cntr < self.batch_size:
            return
        # batch is complete, so we set the weights for the batch
        if self.learn_step_counter % self.replace == 0:
            self.q_next.set_weights(self.q_eval.get_weights())

        states, actions, rewards, states_, dones = self.memory.sample_buffer(self.batch_size)

        q_pred = self.q_eval(states)
        q_next = tf.math.reduce_max(self.q_next(states_), axis=1, keepdims=True).numpy()

        q_target = np.copy(q_pred)
        # we are looking for terminal states here and updating predicted values
        for i, j in enumerate(dones):
            if j:
                q_next[i] = 0.0
            q_target[i, actions[i]] = rewards[i] + self.gamma*q_next[i]

        self.q_eval.train_on_batch(states, q_target)

        if self.epsilon > self.eps_end:
            self.epsilon -= self.eps_dec
        else:
            self.epsilon = self.eps_end

        self.learn_step_counter += 1
    