"""REINFORCE algorithm implementation for CartPole-v0.

This module implements the REINFORCE (Monte Carlo Policy Gradient) algorithm
and trains it on the CartPole environment until achieving a score >= 200.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import gymnasium as gym
from collections import deque
import time


class PolicyNetwork(nn.Module):
    """Simple policy network for CartPole."""
    
    def __init__(self, state_dim=4, action_dim=2, hidden_dim=128):
        """Initialize the policy network.
        
        Args:
            state_dim: Dimension of state space (4 for CartPole)
            action_dim: Number of actions (2 for CartPole: left/right)
            hidden_dim: Hidden layer size
        """
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
        
    def forward(self, state):
        """Forward pass through network.
        
        Args:
            state: Environment state
            
        Returns:
            Action probabilities and log probabilities
        """
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        action_logits = self.fc3(x)
        
        # Convert to probabilities
        action_probs = F.softmax(action_logits, dim=-1)
        log_probs = F.log_softmax(action_logits, dim=-1)
        
        return action_probs, log_probs


class REINFORCEAgent:
    """REINFORCE agent for policy gradient learning."""
    
    def __init__(self, state_dim=4, action_dim=2, learning_rate=0.01, 
                 gamma=0.99, hidden_dim=128):
        """Initialize REINFORCE agent.
        
        Args:
            state_dim: State space dimension
            action_dim: Action space dimension  
            learning_rate: Learning rate for optimizer
            gamma: Discount factor
            hidden_dim: Hidden layer size
        """
        self.gamma = gamma
        self.policy_net = PolicyNetwork(state_dim, action_dim, hidden_dim)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        
        # Storage for episode data
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        
    def select_action(self, state):
        """Select action using current policy.
        
        Args:
            state: Current environment state
            
        Returns:
            Selected action and its log probability
        """
        state = torch.FloatTensor(state).unsqueeze(0)
        action_probs, log_probs = self.policy_net(state)
        
        # Sample action from probability distribution
        action_dist = torch.distributions.Categorical(action_probs)
        action = action_dist.sample()
        
        return action.item(), log_probs[0][action]
    
    def store_transition(self, state, action, log_prob, reward):
        """Store transition data for episode.
        
        Args:
            state: Environment state
            action: Action taken
            log_prob: Log probability of action
            reward: Reward received
        """
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
    
    def compute_returns(self):
        """Compute discounted returns for the episode.
        
        Returns:
            List of discounted returns for each timestep
        """
        returns = []
        G = 0
        
        # Work backwards through episode
        for reward in reversed(self.rewards):
            G = reward + self.gamma * G
            returns.insert(0, G)
            
        return returns
    
    def update_policy(self):
        """Update policy using REINFORCE algorithm."""
        if len(self.rewards) == 0:
            return 0
            
        # Compute returns
        returns = self.compute_returns()
        returns = torch.FloatTensor(returns)
        
        # Normalize returns (optional baseline)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        # Compute policy gradient loss
        policy_loss = []
        for log_prob, G in zip(self.log_probs, returns):
            policy_loss.append(-log_prob * G)
        
        policy_loss = torch.stack(policy_loss).sum()
        
        # Update policy network
        self.optimizer.zero_grad()
        policy_loss.backward()
        self.optimizer.step()
        
        # Clear episode data
        self.clear_episode()
        
        return policy_loss.item()
    
    def clear_episode(self):
        """Clear stored episode data."""
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []


def train_reinforce(episodes=2000, target_score=200, render_every=100):
    """Train REINFORCE agent on CartPole-v0.
    
    Args:
        episodes: Maximum number of episodes to train
        target_score: Target average score to achieve
        render_every: Render environment every N episodes
        
    Returns:
        Trained agent and training statistics
    """
    # Create environment and agent
    env = gym.make('CartPole-v1')
    agent = REINFORCEAgent(state_dim=4, action_dim=2, learning_rate=0.01)
    
    # Training statistics
    episode_rewards = []
    running_rewards = deque(maxlen=100)
    losses = []
    
    print("Starting REINFORCE training on CartPole-v1...")
    print(f"Target: Average score >= {target_score} over 100 episodes")
    print("-" * 50)
    
    for episode in range(episodes):
        state = env.reset()
        if isinstance(state, tuple):
            state = state[0]  # Handle new gym API
            
        episode_reward = 0
        done = False
        
        # Run one episode
        while not done:
            # Select action
            action, log_prob = agent.select_action(state)
            
            # Take action in environment
            next_state, reward, done, truncated, info = env.step(action)
            if isinstance(next_state, tuple):
                next_state = next_state[0]
                
            done = done or truncated
            
            # Store transition
            agent.store_transition(state, action, log_prob, reward)
            
            state = next_state
            episode_reward += reward
            
            # Render occasionally
            if episode % render_every == 0 and episode > 0:
                env.render()
                time.sleep(0.01)
        
        # Update policy after episode
        loss = agent.update_policy()
        
        # Record statistics
        episode_rewards.append(episode_reward)
        running_rewards.append(episode_reward)
        losses.append(loss)
        
        # Print progress
        if episode % 100 == 0:
            avg_reward = np.mean(running_rewards)
            print(f"Episode {episode:4d} | Avg Reward: {avg_reward:6.2f} | "
                  f"Last Reward: {episode_reward:3.0f} | Loss: {loss:8.4f}")
        
        # Check if target achieved
        if len(running_rewards) >= 100:
            avg_reward = np.mean(running_rewards)
            if avg_reward >= target_score:
                print(f"\n🎉 Target achieved! Average reward: {avg_reward:.2f}")
                print(f"Training completed in {episode + 1} episodes")
                break
    
    env.close()
    
    return agent, {
        'episode_rewards': episode_rewards,
        'running_rewards': list(running_rewards),
        'losses': losses,
        'episodes_trained': episode + 1
    }


def plot_training_results(stats):
    """Plot training statistics.
    
    Args:
        stats: Dictionary containing training statistics
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot episode rewards
    episodes = range(1, len(stats['episode_rewards']) + 1)
    ax1.plot(episodes, stats['episode_rewards'], alpha=0.6, color='lightblue', label='Episode Reward')
    
    # Plot running average
    window = 100
    if len(stats['episode_rewards']) >= window:
        running_avg = []
        for i in range(window - 1, len(stats['episode_rewards'])):
            avg = np.mean(stats['episode_rewards'][i-window+1:i+1])
            running_avg.append(avg)
        
        ax1.plot(range(window, len(stats['episode_rewards']) + 1), running_avg, 
                color='red', linewidth=2, label=f'Running Average ({window} episodes)')
    
    ax1.axhline(y=200, color='green', linestyle='--', linewidth=2, label='Target Score (200)')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Reward')
    ax1.set_title('REINFORCE Training on CartPole-v1')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot loss
    if stats['losses']:
        loss_episodes = range(1, len(stats['losses']) + 1)
        ax2.plot(loss_episodes, stats['losses'], color='orange', alpha=0.7)
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Policy Loss')
        ax2.set_title('Policy Gradient Loss')
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/Users/profenghasn/DeepRL/reinforce_training_results.png', dpi=300, bbox_inches='tight')
    plt.show()


def watch_smart_agent(agent, episodes=5):
    """Watch the trained agent perform.
    
    Args:
        agent: Trained REINFORCE agent
        episodes: Number of episodes to watch
    """
    env = gym.make('CartPole-v1', render_mode='human')
    
    print(f"\n🤖 Watching trained agent for {episodes} episodes...")
    print("Close the window to stop watching.")
    
    for episode in range(episodes):
        state = env.reset()
        if isinstance(state, tuple):
            state = state[0]
            
        episode_reward = 0
        done = False
        step = 0
        
        print(f"\nEpisode {episode + 1}:")
        
        while not done and step < 500:  # Max 500 steps
            # Select action (deterministic for watching)
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            action_probs, _ = agent.policy_net(state_tensor)
            action = torch.argmax(action_probs).item()
            
            # Take action
            next_state, reward, done, truncated, info = env.step(action)
            if isinstance(next_state, tuple):
                next_state = next_state[0]
                
            done = done or truncated
            
            state = next_state
            episode_reward += reward
            step += 1
            
            time.sleep(0.02)  # Slow down for better viewing
        
        print(f"  Steps: {step}, Total Reward: {episode_reward}")
    
    env.close()


if __name__ == "__main__":
    # Train the agent
    print("🚀 Starting REINFORCE training...")
    agent, training_stats = train_reinforce(episodes=2000, target_score=200)
    
    # Plot results
    print("\n📊 Plotting training results...")
    plot_training_results(training_stats)
    
    # Watch the trained agent
    print("\n🎬 Time to watch the smart agent!")
    input("Press Enter to start watching the trained agent...")
    watch_smart_agent(agent, episodes=3)
    
    print("\n✅ All done! Check the saved plot: reinforce_training_results.png")