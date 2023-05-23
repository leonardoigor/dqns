import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np


class DQNAgent:
    def __init__(self, state_size, action_size, discount_rate=0.95, learning_rate=0.001, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01):
        self.state_size = state_size
        self.action_size = action_size
        self.discount_rate = discount_rate
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        # Create the main model and the target model
        self.main_model = self.build_model()
        self.target_model = self.build_model()
        self.update_target_network()

    def build_model(self):
        model = models.Sequential()
        model.add(layers.Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(layers.Dense(24, activation='relu'))
        model.add(layers.Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(
            lr=self.learning_rate))
        return model

    def get_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.action_size)
        q_values = self.main_model.predict(
            state.reshape(-1, self.state_size), verbose=0)
        return np.argmax(q_values[0])

    def update_policy(self, replay_buffer, batch_size):
        # Sample a batch of experiences from the replay buffer
        batch = replay_buffer.sample(batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        states = np.array(states)
        actions = np.array(actions)
        rewards = np.array(rewards)
        next_states = np.array(next_states)
        dones = np.array(dones)

        # Compute the TD targets
        q_values = self.main_model.predict(states, verbose=0)
        next_q_values = self.target_model.predict(next_states, verbose=0)
        max_next_q_values = np.max(np.max(next_q_values, axis=1), axis=1)
        td_targets = rewards + self.discount_rate * \
            max_next_q_values * (1 - dones)
        # td_targets = td_targets.reshape(-1, 1)
        target_indices = np.array([i for i in range(batch_size)])

        # Update the main model
        q_updates = td_targets - q_values[target_indices, 0, actions]
        q_values[target_indices, 0, actions] += q_updates
        self.main_model.fit(states, q_values, verbose=0)

        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def update_target_network(self):
        self.target_model.set_weights(self.main_model.get_weights())

    def save(self, name):
        self.main_model.save_weights(name+"_main")
        self.target_model.save_weights(name+"_target")

    def load(self, name):
        self.main_model.load_weights(name+"_main")
        self.target_model.load_weights(name+"_target")
