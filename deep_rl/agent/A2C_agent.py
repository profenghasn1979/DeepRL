"""A2C (Advantage Actor-Critic) agent implementation.

This module implements the A2C algorithm for reinforcement learning.
"""
#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

import numpy as np
import torch
import torch.nn as nn
from ..component import Storage
from ..utils.torch_utils import tensor, to_np  
from .BaseAgent import BaseAgent


class A2CAgent(BaseAgent):
    """Advantage Actor-Critic agent implementation.
    
    This agent uses the A2C algorithm to train both an actor (policy) and
    critic (value function) network using advantage estimation for policy updates.
    """
    def __init__(self, config):
        """Initialize the A2C agent.
        
        Args:
            config: Configuration object containing hyperparameters and setup functions.
        """
        BaseAgent.__init__(self, config)
        self.config = config
        self.task = config.task_fn()
        self.network = config.network_fn()
        self.optimizer = config.optimizer_fn(self.network.parameters())
        self.total_steps = 0
        self.states = self.task.reset()

    def step(self):
        """Execute one training step of the A2C algorithm.
        
        Collects rollout data, computes advantages, and performs network updates.
        """
        config = self.config
        storage = Storage(config.rollout_length)
        states = self.states
        
        # Rollout collection phase
        for _ in range(config.rollout_length):
            prediction = self.network(config.state_normalizer(states))
            action = to_np(prediction['action'])
            next_states, rewards, terminals, info = self.task.step(action)
            self.record_online_return(info)
            rewards = config.reward_normalizer(rewards)
            storage.feed(prediction)
            storage.feed({'reward': tensor(rewards).unsqueeze(-1),
                         'mask': tensor(1 - terminals).unsqueeze(-1)})

            states = next_states
            self.total_steps += config.num_workers

        self.states = states
        prediction = self.network(config.state_normalizer(states))
        storage.feed(prediction)
        storage.placeholder()

        # Compute advantages and returns
        advantages = tensor(np.zeros((config.num_workers, 1)))
        returns = prediction['v'].detach()
        for i in reversed(range(config.rollout_length)):
            returns = (storage.reward[i] + 
                      config.discount * storage.mask[i] * returns)
            if not config.use_gae:
                advantages = returns - storage.v[i].detach()
            else:
                next_value = storage.v[i + 1] if i < config.rollout_length - 1 else returns
                td_error = (storage.reward[i] + 
                           config.discount * storage.mask[i] * next_value - 
                           storage.v[i])
                advantages = (advantages * config.gae_tau * config.discount * 
                             storage.mask[i] + td_error)
            storage.advantage[i] = advantages.detach()
            storage.ret[i] = returns.detach()

        # Compute losses
        entries = storage.extract(['log_pi_a', 'v', 'ret', 'advantage', 'entropy'])
        policy_loss = -(entries.log_pi_a * entries.advantage).mean()
        value_loss = 0.5 * (entries.ret - entries.v).pow(2).mean()
        entropy_loss = entries.entropy.mean()

        # Update network
        self.optimizer.zero_grad()
        total_loss = (policy_loss - config.entropy_weight * entropy_loss +
                     config.value_loss_weight * value_loss)
        total_loss.backward()
        nn.utils.clip_grad_norm_(self.network.parameters(), config.gradient_clip)
        self.optimizer.step()

    def eval_step(self, state):
        """Evaluate policy for a single state during testing.
        
        Args:
            state: Environment state to evaluate.
            
        Returns:
            Action selected by the policy.
        """
        with torch.no_grad():
            state = self.config.state_normalizer(state)
            prediction = self.network(state)
            return to_np(prediction['action'])

    def record_step(self, state):
        """Record step for episode recording.
        
        Args:
            state: Current environment state.
            
        Returns:
            Action to take for recording purposes.
        """
        return self.eval_step(state)
