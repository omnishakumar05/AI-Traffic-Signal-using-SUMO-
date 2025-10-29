from environment import SumoTrafficEnv
from dqn_agent import DQNAgent
import os

def test_agent(episodes=2):
    # Get absolute paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    net_file = os.path.join(base_dir, 'network', 'intersection.net.xml')
    route_file = os.path.join(base_dir, 'network', 'routes.rou.xml')
    add_file = os.path.join(base_dir, 'network', 'additional.add.xml')
    model_path = os.path.join(base_dir, 'models', 'trained_model.pth')
    
    env = SumoTrafficEnv(
        net_file=net_file,
        route_file=route_file,
        add_file=add_file,
        use_gui=True,
        max_steps=5800
    )
    
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
        print(f"\nEpisode {episode + 1}/{episodes}")
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
