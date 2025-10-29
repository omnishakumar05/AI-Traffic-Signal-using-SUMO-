import numpy as np
import matplotlib.pyplot as plt
from environment import SumoTrafficEnv
from dqn_agent import DQNAgent
import os
import sys

def train_agent(episodes=30, max_steps=1800):
    # Get absolute paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    net_file = os.path.join(base_dir, 'network', 'intersection.net.xml')
    route_file = os.path.join(base_dir, 'network', 'routes.rou.xml')
    add_file = os.path.join(base_dir, 'network', 'additional.add.xml')
    model_dir = os.path.join(base_dir, 'models')
    save_path = os.path.join(model_dir, 'trained_model.pth')
    
    print(f"Network file: {net_file}")
    print(f"Route file: {route_file}")
    print(f"Additional file: {add_file}")
    
    env = SumoTrafficEnv(
        net_file=net_file,
        route_file=route_file,
        add_file=add_file,
        use_gui=False,
        max_steps=max_steps
    )
    
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
        print(f"Episode {episode+1}/{episodes} | Reward: {episode_reward:.1f} | Waiting: {info['waiting_time']:.1f}s | Epsilon: {agent.epsilon:.3f}")
        
        if (episode + 1) % 5 == 0:
            os.makedirs(model_dir, exist_ok=True)
            agent.save(save_path)
            print(f"[OK] Model saved to {save_path}")
    
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
    print("Training plot saved!")
    return agent

if __name__ == '__main__':
    agent = train_agent(episodes=30)
    print("Training completed!")
