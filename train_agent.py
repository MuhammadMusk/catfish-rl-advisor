import numpy as np
import matplotlib.pyplot as plt
from env_simulation import PondEnvironment

# Step 1: Discretizer for state
class Discretizer:
    def __init__(self, bins):
        self.bins = bins

    def discretize(self, obs):
        return tuple(np.digitize(o, b) for o, b in zip(obs, self.bins))

# Step 2: Binning (adjust as needed for accuracy)
bins = [
    np.linspace(20, 35, 6),   # Temperature: 6 bins
    np.linspace(5.5, 9.5, 5), # pH: 5 bins
    np.linspace(0, 10, 6)     # DO: 6 bins
]
discretizer = Discretizer(bins)

# Step 3: Q-table and learning params
n_actions = 4 
q_table = {}
 
alpha = 0.1   # Learning rate
gamma = 0.9   # Discount factor
epsilon = 0.1 # Exploration rate 
episodes = 500

def ensure_q_values(state):
    if state not in q_table:
        q_table[state] = np.zeros(n_actions)

# Step 4: Training loop
env = PondEnvironment()
reward_history = []

for ep in range(episodes):
    state = env.reset()
    state_disc = discretizer.discretize(state)
    total_reward = 0

    for _ in range(50):
        ensure_q_values(state_disc)

        # Choose action
        if np.random.rand() < epsilon:
            action = np.random.randint(n_actions)
        else:
            action = np.argmax(q_table[state_disc])

        # Environment step
        next_state, reward, done = env.step(action)
        next_state_disc = discretizer.discretize(next_state)
        ensure_q_values(next_state_disc)

        # Q-learning update
        q_table[state_disc][action] += alpha * (
            reward + gamma * np.max(q_table[next_state_disc]) - q_table[state_disc][action]
        )

        state_disc = next_state_disc
        total_reward += reward

        if done:
            break

    reward_history.append(total_reward)

# Step 5: Save Q-table
np.save("q_table.npy", q_table)
print("✅ Training complete. Q-table saved to 'q_table.npy'.")

# Step 6: Plot reward curve
plt.plot(reward_history)
plt.title("Total Reward per Episode")
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.grid(True)
plt.tight_layout()
plt.show() 
   