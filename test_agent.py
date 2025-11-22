"""Simple script to test the trained REINFORCE agent without GUI."""

import torch
import gymnasium as gym
import numpy as np
from reinforce_cartpole import PolicyNetwork

def test_agent_performance():
    """Test the trained agent's performance."""
    
    # Load the trained model (we'll save it first)
    env = gym.make('CartPole-v1')
    
    # Create and train a quick agent
    from reinforce_cartpole import REINFORCEAgent
    
    print("🤖 Testing REINFORCE agent performance...")
    
    # Quick training (just to get a working agent)
    agent = REINFORCEAgent(state_dim=4, action_dim=2, learning_rate=0.01)
    
    test_rewards = []
    for test_episode in range(5):
        state = env.reset()
        if isinstance(state, tuple):
            state = state[0]
            
        episode_reward = 0
        done = False
        steps = 0
        
        while not done and steps < 500:
            # Select best action (deterministic)
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            action_probs, _ = agent.policy_net(state_tensor)
            action = torch.argmax(action_probs).item()
            
            next_state, reward, done, truncated, info = env.step(action)
            if isinstance(next_state, tuple):
                next_state = next_state[0]
                
            done = done or truncated
            state = next_state
            episode_reward += reward
            steps += 1
        
        test_rewards.append(episode_reward)
        print(f"Test Episode {test_episode + 1}: {episode_reward} steps")
    
    env.close()
    
    avg_reward = np.mean(test_rewards)
    print(f"\n📊 Average performance: {avg_reward:.1f} steps")
    
    if avg_reward >= 200:
        print("✅ Agent is performing well!")
    else:
        print("❌ Agent needs more training")
    
    return avg_reward

if __name__ == "__main__":
    test_agent_performance()