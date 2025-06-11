import numpy as np

class PondEnvironment:
    def __init__(self):
        self.reset()

    def reset(self):
        self.temp = np.random.uniform(25, 32)  # Catfish safe range
        self.ph = np.random.uniform(6.0, 9.0)
        self.do = np.random.uniform(3.0, 7.0)
        self.time = 0
        return self._get_state()

    def _get_state(self):
        return np.array([self.temp, self.ph, self.do])

    def step(self, action):
        # Natural degradation
        self.temp += np.random.uniform(-0.2, 0.2)
        self.ph += np.random.uniform(-0.1, 0.1)
        self.do -= np.random.uniform(0.1, 0.3)

        # Apply control actions
        if action == 1:  # Aerate
            self.do += 1.0
            self.temp += 0.2
        elif action == 2:  # Add buffer
            self.ph += 0.5 if self.ph < 7.5 else -0.5
        elif action == 3:  # Replace water
            self.temp = np.random.uniform(26, 28)
            self.ph = np.random.uniform(7.0, 7.5)
            self.do = np.random.uniform(5.5, 6.5)

        # Boundaries
        self.temp = np.clip(self.temp, 20, 35)
        self.ph = np.clip(self.ph, 5.5, 9.5)
        self.do = np.clip(self.do, 0, 10)

        self.time += 1
        reward = self._calculate_reward()
        done = self.time >= 50
        return self._get_state(), reward, done

    def _calculate_reward(self):
        reward = 0
        if 26 <= self.temp <= 30:
            reward += 1
        else:
            reward -= abs(self.temp - 28) / 5

        if 6.5 <= self.ph <= 8.0:
            reward += 1
        else:
            reward -= abs(self.ph - 7.2) / 1.5

        if self.do >= 5.0:
            reward += 1
        else:
            reward -= (5.0 - self.do) / 3

        return reward
 