import gym
from game import TicTacToeEnv
from replay_buffer import ReplayBuffer
from dqn_agent import DQNAgent

from tqdm import tqdm


# Create the environment
env = TicTacToeEnv()

# Set the hyperparameters
state_size = env.observation_space.shape[0]
action_size = env.action_space.n
buffer_size = 10000
batch_size = 32
discount_rate = 0.95
learning_rate = 0.001
epsilon = 1.0
epsilon_decay = 0.995
epsilon_min = 0.01
num_episodes = 1000
max_steps = 500
target_update_frequency = 10

# Create the replay buffer and the DQN agent
replay_buffer = ReplayBuffer(buffer_size)
dqn_agent = DQNAgent(state_size, action_size, discount_rate,
                     learning_rate, epsilon, epsilon_decay, epsilon_min)

# Train the DQN agent
for episode in tqdm(range(num_episodes), desc='Episodes'):
    state = env.reset()
    score = 0
    for step in tqdm(range(max_steps), desc='Steps'):
        # Get an action from the DQN agent
        action = dqn_agent.get_action(state)

        # Take a step in the environment
        next_state, reward, done, info = env.step(action)
        score += reward
        # Add the experience to the replay buffer
        replay_buffer.add(state, action, reward, next_state, done)

        # Update the DQN agent's policy
        if len(replay_buffer) > batch_size:
            dqn_agent.update_policy(replay_buffer, batch_size)

        # Update the target network
        if step % target_update_frequency == 0:
            dqn_agent.update_target_network()

        # Update the current state and decrement epsilon
        state = next_state
        if epsilon > epsilon_min:
            epsilon *= epsilon_decay

        # If the episode is over, break out of the loop
        if done:
            break

    # Print the episode information
    print('Episode: {}/{}, Score: {}, Epsilon: {:.2f}'.format(episode +
          1, num_episodes, score+1, epsilon))

# Close the environment
env.close()
