import os

# Make sure we're in the src directory
os.makedirs('src', exist_ok=True)

# File 1: environment.py
environment_code = '''import os
import sys
import numpy as np
import traci
import sumolib
from gym import Env, spaces

class SumoTrafficEnv(Env):
    def __init__(self, net_file, route_file, add_file, use_gui=False, max_steps=1800):
        super(SumoTrafficEnv, self).__init__()
        self.net_file = net_file
        self.route_file = route_file
        self.add_file = add_file
        self.use_gui = use_gui
        self.max_steps = max_steps
        self.current_step = 0
        self.tl_id = "center"
        self.min_green_time = 7
        self.max_green_time = 60
        self.extension_time = 3
        self.yellow_time = 4
        self.current_phase = 0
        self.current_phase_duration = 0
        self.phases = [0, 2]
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(low=0, high=1, shape=(10,), dtype=np.float32)
        self.total_waiting_time = 0
        self.collision_count = 0

    def _start_simulation(self):
        sumo_binary = sumolib.checkBinary('sumo-gui' if self.use_gui else 'sumo')
        sumo_cmd = [sumo_binary, "-n", self.net_file, "-r", self.route_file, "-a", self.add_file,
                    "--step-length", "1", "--collision.action", "warn", "--time-to-teleport", "-1",
                    "--no-warnings", "true", "--no-step-log", "true"]
        if self.use_gui:
            sumo_cmd.extend(["--start", "--quit-on-end"])
        traci.start(sumo_cmd)

    def reset(self):
        if traci.isLoaded():
            traci.close()
        self._start_simulation()
        self.current_step = 0
        self.current_phase = 0
        self.current_phase_duration = 0
        self.total_waiting_time = 0
        self.collision_count = 0
        traci.trafficlight.setPhase(self.tl_id, self.phases[self.current_phase])
        return self._get_observation()

    def _get_observation(self):
        lanes = ["north_in_0", "north_in_1", "south_in_0", "south_in_1",
                 "east_in_0", "east_in_1", "west_in_0", "west_in_1"]
        queue_lengths = []
        for lane in lanes:
            try:
                halting = traci.lane.getLastStepHaltingNumber(lane)
                queue_lengths.append(min(halting / 10.0, 1.0))
            except:
                queue_lengths.append(0.0)
        phase_norm = self.current_phase / len(self.phases)
        time_norm = min(self.current_phase_duration / self.max_green_time, 1.0)
        return np.array(queue_lengths + [phase_norm, time_norm], dtype=np.float32)

    def _apply_vac_logic(self, action):
        detectors = self._get_active_detectors()
        vehicles_detected = any(detectors)
        if self.current_phase_duration < self.min_green_time:
            return False
        if self.current_phase_duration >= self.max_green_time:
            return True
        if action == 1:
            if vehicles_detected and self.current_phase_duration < self.max_green_time:
                return False
            else:
                return True
        return False

    def _get_active_detectors(self):
        if self.current_phase == 0:
            detector_ids = ["det_n", "det_s"]
        else:
            detector_ids = ["det_e", "det_w"]
        detections = []
        for det_id in detector_ids:
            try:
                vehs = traci.inductionloop.getLastStepVehicleNumber(det_id)
                detections.append(vehs > 0)
            except:
                detections.append(False)
        return detections

    def _switch_phase(self):
        traci.trafficlight.setPhase(self.tl_id, self.phases[self.current_phase] + 1)
        for _ in range(self.yellow_time):
            traci.simulationStep()
            self.current_step += 1
        self.current_phase = (self.current_phase + 1) % len(self.phases)
        traci.trafficlight.setPhase(self.tl_id, self.phases[self.current_phase])
        self.current_phase_duration = 0

    def _calculate_reward(self):
        waiting_time = 0
        queue_length = 0
        for veh_id in traci.vehicle.getIDList():
            waiting_time += traci.vehicle.getWaitingTime(veh_id)
            speed = traci.vehicle.getSpeed(veh_id)
            if speed < 0.1:
                queue_length += 1
        collision_penalty = 0
        collisions = traci.simulation.getCollidingVehiclesNumber()
        if collisions > 0:
            collision_penalty = collisions * 100
            self.collision_count += collisions
        reward = -(waiting_time * 0.1 + queue_length * 1.0 + collision_penalty)
        self.total_waiting_time += waiting_time
        return reward

    def step(self, action):
        should_switch = self._apply_vac_logic(action)
        if should_switch:
            self._switch_phase()
        else:
            traci.simulationStep()
            self.current_step += 1
            self.current_phase_duration += 1
        observation = self._get_observation()
        reward = self._calculate_reward()
        done = self.current_step >= self.max_steps
        info = {'waiting_time': self.total_waiting_time, 'vehicles': len(traci.vehicle.getIDList()),
                'collisions': self.collision_count, 'current_phase': self.current_phase,
                'phase_duration': self.current_phase_duration}
        return observation, reward, done, info

    def close(self):
        if traci.isLoaded():
            traci.close()

    def render(self, mode='human'):
        pass
'''

# File 2: dqn_agent.py
dqn_code = '''import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque

class DQN(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(DQN, self).__init__()
        self.network = nn.Sequential(nn.Linear(state_dim, hidden_dim), nn.ReLU(),
                                      nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                                      nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                                      nn.Linear(hidden_dim, action_dim))
    def forward(self, x):
        return self.network(x)

class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (np.array(states), np.array(actions), np.array(rewards), 
                np.array(next_states), np.array(dones))
    def __len__(self):
        return len(self.buffer)

class DQNAgent:
    def __init__(self, state_dim, action_dim, learning_rate=0.001, gamma=0.95):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = 1.0
        self.epsilon_end = 0.01
        self.epsilon_decay = 0.995
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net = DQN(state_dim, action_dim).to(self.device)
        self.target_net = DQN(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.replay_buffer = ReplayBuffer(capacity=10000)
        self.batch_size = 64
        self.target_update_freq = 10
        self.update_counter = 0

    def select_action(self, state, training=True):
        if training and random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state_tensor)
            return q_values.argmax().item()

    def store_transition(self, state, action, reward, next_state, done):
        self.replay_buffer.push(state, action, reward, next_state, done)

    def train_step(self):
        if len(self.replay_buffer) < self.batch_size:
            return 0
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        current_q_values = self.policy_net(states).gather(1, actions.unsqueeze(1))
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()
        self.update_counter += 1
        if self.update_counter % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        return loss.item()

    def save(self, path):
        torch.save({'policy_net': self.policy_net.state_dict(), 'target_net': self.target_net.state_dict(),
                    'optimizer': self.optimizer.state_dict(), 'epsilon': self.epsilon}, path)

    def load(self, path):
        checkpoint = torch.load(path)
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.target_net.load_state_dict(checkpoint['target_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint['epsilon']
'''

# File 3: train.py
train_code = '''import numpy as np
import matplotlib.pyplot as plt
from environment import SumoTrafficEnv
from dqn_agent import DQNAgent
import os

def train_agent(episodes=30, max_steps=1800):
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    net_file = os.path.join(base_dir, 'network', 'intersection.net.xml')
    route_file = os.path.join(base_dir, 'network', 'routes.rou.xml')
    add_file = os.path.join(base_dir, 'network', 'additional.add.xml')
    model_dir = os.path.join(base_dir, 'models')
    save_path = os.path.join(model_dir, 'trained_model.pth')
    
    env = SumoTrafficEnv(net_file=net_file, route_file=route_file, add_file=add_file, 
                         use_gui=False, max_steps=max_steps)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent = DQNAgent(state_dim, action_dim)
    episode_rewards = []
    episode_waiting_times = []

    print("Starting training...")
    for episode in range(episodes):
        state = env.reset()
        episode_reward = 0
        steps = 0
        while steps < max_steps:
            action = agent.select_action(state, training=True)
            next_state, reward, done, info = env.step(action)
            agent.store_transition(state, action, reward, next_state, done)
            agent.train_step()
            episode_reward += reward
            steps += 1
            state = next_state
            if done:
                break
        episode_rewards.append(episode_reward)
        episode_waiting_times.append(info['waiting_time'])
        print(f"Episode {episode+1}/{episodes} | Reward: {episode_reward:.1f} | "
              f"Waiting: {info['waiting_time']:.1f}s | Epsilon: {agent.epsilon:.3f}")
        if (episode + 1) % 5 == 0:
            os.makedirs(model_dir, exist_ok=True)
            agent.save(save_path)
            print(f"[OK] Model saved")
    env.close()
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(episode_rewards)
    plt.title('Episode Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.grid(True)
    plt.subplot(1, 2, 2)
    plt.plot(episode_waiting_times)
    plt.title('Total Waiting Time')
    plt.xlabel('Episode')
    plt.ylabel('Waiting Time (s)')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(base_dir, 'training_results.png'))
    print("Training completed!")
    return agent

if __name__ == '__main__':
    agent = train_agent(episodes=30)
'''

# File 4: test.py
test_code = '''from environment import SumoTrafficEnv
from dqn_agent import DQNAgent
import os

def test_agent(episodes=2):
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    net_file = os.path.join(base_dir, 'network', 'intersection.net.xml')
    route_file = os.path.join(base_dir, 'network', 'routes.rou.xml')
    add_file = os.path.join(base_dir, 'network', 'additional.add.xml')
    model_path = os.path.join(base_dir, 'models', 'trained_model.pth')
    
    env = SumoTrafficEnv(net_file=net_file, route_file=route_file, add_file=add_file,
                         use_gui=True, max_steps=1800)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent = DQNAgent(state_dim, action_dim)
    agent.load(model_path)
    agent.epsilon = 0
    
    print("Testing trained agent with GUI...")
    for episode in range(episodes):
        state = env.reset()
        episode_reward = 0
        steps = 0
        print(f"\\nEpisode {episode + 1}/{episodes}")
        while steps < env.max_steps:
            action = agent.select_action(state, training=False)
            next_state, reward, done, info = env.step(action)
            episode_reward += reward
            steps += 1
            state = next_state
            if steps % 100 == 0:
                print(f"Step {steps}: Vehicles={info['vehicles']}, Waiting={info['waiting_time']:.1f}s")
            if done:
                break
        print(f"Episode done | Reward: {episode_reward:.1f} | Waiting: {info['waiting_time']:.1f}s")
    env.close()

if __name__ == '__main__':
    test_agent(episodes=2)
'''

# Write all files
with open('src/environment.py', 'w', encoding='utf-8') as f:
    f.write(environment_code)
    
with open('src/dqn_agent.py', 'w', encoding='utf-8') as f:
    f.write(dqn_code)
    
with open('src/train.py', 'w', encoding='utf-8') as f:
    f.write(train_code)
    
with open('src/test.py', 'w', encoding='utf-8') as f:
    f.write(test_code)

print("[OK] All Python files created in src/ folder!")
print("Files created:")
print("  - src/environment.py")
print("  - src/dqn_agent.py")
print("  - src/train.py")
print("  - src/test.py")
print("\nYou can now run: python src/train.py")
