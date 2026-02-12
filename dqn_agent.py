import os
import random
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from datetime import datetime
import pygame
import time  # Import time for adding delays
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

import game  
from game import GameManager, load_map, interpolate_paths

# Neural Network for DQN
class DQN(nn.Module):
    def __init__(self, action_size):
        super(DQN, self).__init__()
        
        # CNN for processing grid state (4 input channels, not 3)
        self.conv1 = nn.Conv2d(4, 16, kernel_size=3, stride=1, padding=1)  
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        
        # Calculate output size 
        conv_output_size = 32 * GRID_HEIGHT * GRID_WIDTH
        
        # Scalar features processing
        self.scalar_fc = nn.Linear(4, 64)
        
        # Combined processing
        self.fc1 = nn.Linear(conv_output_size + 64, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, action_size)
    
    def forward(self, grid_state, scalar_state):
        # Process grid state with CNN
        x1 = F.relu(self.conv1(grid_state))
        x1 = F.relu(self.conv2(x1))
        x1 = F.relu(self.conv3(x1))
        x1 = x1.view(x1.size(0), -1)  # Flattening
        
        # Process scalar state
        x2 = F.relu(self.scalar_fc(scalar_state))
        
        # Combine both states
        x = torch.cat((x1, x2), dim=1)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        
        return x

# Replay memory
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        
        states_grid = torch.FloatTensor(np.array([s['grid_state'] for s, _, _, _, _ in batch]))
        states_scalar = torch.FloatTensor(np.array([s['scalar_state'] for s, _, _, _, _ in batch]))
        
        next_states_grid = torch.FloatTensor(np.array([s['grid_state'] for _, _, _, s, _ in batch]))
        next_states_scalar = torch.FloatTensor(np.array([s['scalar_state'] for _, _, _, s, _ in batch]))
        
        actions = torch.LongTensor(np.array([a for _, a, _, _, _ in batch]))
        rewards = torch.FloatTensor(np.array([r for _, _, r, _, _ in batch]))
        dones = torch.FloatTensor(np.array([d for _, _, _, _, d in batch]))
        
        return (states_grid, states_scalar), actions, rewards, (next_states_grid, next_states_scalar), dones
    
    def __len__(self):
        return len(self.buffer)

# Agent with Deep Q-Network
class DQNAgent:
    def __init__(self, action_size, device='cpu'):
        self.action_size = action_size
        self.device = device
        
        # Q-Networks (policy and target)
        self.policy_net = DQN(action_size).to(device)
        self.target_net = DQN(action_size).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        # Optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=0.0001)
        
        # Replay memory
        self.memory = ReplayBuffer(10000)
        self.batch_size = 64
        
        # Learning parameters
        self.gamma = 0.99  # discount factor
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.99995
        self.target_update = 10  # how often to update target network
        self.q_values_history = []
        self.avg_q_values_per_episode = []
        self.max_q_values_per_episode = []
    
    def select_action(self, state):
        # Epsilon-greedy action selection
        if random.random() < self.epsilon:
            return random.randrange(self.action_size)
        else:
            grid_state = torch.FloatTensor(state['grid_state']).unsqueeze(0).to(self.device)
            scalar_state = torch.FloatTensor(state['scalar_state']).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                q_values = self.policy_net(grid_state, scalar_state)
                
                # Track Q-values
                max_q = q_values.max().item()
                avg_q = q_values.mean().item()
                all_q = q_values.cpu().numpy().flatten()
                
                # Track action-specific Q-values
                action_values = {}
                for i in range(min(5, len(all_q))):  # Track top 5 actions
                    action_idx = np.argsort(all_q)[-i-1]  # Get index of i-th highest Q-value
                    action_values[int(action_idx)] = all_q[action_idx]
                    
                self.q_values_history.append({
                    'max_q': max_q, 
                    'avg_q': avg_q,
                    'action_values': action_values
                })
                
                return q_values.argmax().item()
    
    def compute_episode_q_stats(self):
        if not self.q_values_history:
            return 0, 0, {}
        
        # Compute max and average Q-values for this episode
        max_q_values = [entry['max_q'] for entry in self.q_values_history]
        avg_q_values = [entry['avg_q'] for entry in self.q_values_history]
        
        # Track action-specific Q-values
        action_values = {}
        for entry in self.q_values_history:
            for action_idx, value in entry['action_values'].items():
                if action_idx not in action_values:
                    action_values[action_idx] = []
                action_values[action_idx].append(value)
        
        # Compute average Q-value for each action
        avg_action_values = {}
        for action_idx, values in action_values.items():
            avg_action_values[action_idx] = np.mean(values)
        
        # Store episode stats
        self.max_q_values_per_episode.append(np.mean(max_q_values))
        self.avg_q_values_per_episode.append(np.mean(avg_q_values))
        
        # Update action-specific Q-value history
        if not hasattr(self, 'action_q_values_per_episode'):
            self.action_q_values_per_episode = {}
        
        for action_idx, avg_value in avg_action_values.items():
            if action_idx not in self.action_q_values_per_episode:
                self.action_q_values_per_episode[action_idx] = []
            self.action_q_values_per_episode[action_idx].append(avg_value)
        
        # Clear history for next episode
        self.q_values_history = []
        
        return np.mean(max_q_values), np.mean(avg_q_values), self.action_q_values_per_episode
    
    def learn(self):
        if len(self.memory) < self.batch_size:
            return
   
        # Sample batch from memory
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        states_grid, states_scalar = states
        next_states_grid, next_states_scalar = next_states
        
        # Move to device
        states_grid = states_grid.to(self.device)
        states_scalar = states_scalar.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states_grid = next_states_grid.to(self.device)
        next_states_scalar = next_states_scalar.to(self.device)
        dones = dones.to(self.device)
        
        # Get Q-values for current states and actions
        q_values = self.policy_net(states_grid, states_scalar).gather(1, actions.unsqueeze(1))
        
        # Get max Q-values for next states
        with torch.no_grad():
            next_q_values = self.target_net(next_states_grid, next_states_scalar).max(1)[0]
        
        # Calculate target Q-values
        target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        # Compute loss
        loss = F.smooth_l1_loss(q_values.squeeze(), target_q_values)
        
        # Optimize the policy network
        self.optimizer.zero_grad()
        loss.backward()
        # Clip gradients to stabilize training
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
        
        old_epsilon = self.epsilon
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        
        return loss.item()
    
    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())
    
    def save(self, filename):
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon
        }, filename)
    
    def load(self, filename):
        checkpoint = torch.load(filename)
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.target_net.load_state_dict(checkpoint['target_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint['epsilon']



# Training function
def train_agent(num_episodes=1000, render_interval=100):
    

    game.DISPLAY_GAME = False
    


        # Create game environment
    env = GameManager()

    
    # Get grid dimensions from environment
    global GRID_HEIGHT, GRID_WIDTH
    GRID_HEIGHT = len(env.game_map)
    GRID_WIDTH = len(env.game_map[0])
    
    
    # Set up device (use CUDA if available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create agent
    action_size = 3 * 10 + 3 * 10 + 1 # Places, upgrades, and do nothing
    try:
        agent = DQNAgent(action_size, device)
        print("Successfully created DQNAgent")
    except Exception as e:
        print(f"Error creating DQNAgent: {e}")
        return None
    
    # Training metrics
    all_rewards = []
    all_losses = []
    episode_rewards = []
    episode_lengths = []
    wave_progress = []  # Track highest wave reached per episode
    epsilon_history = []  # Track epsilon value per episode
    lives_remaining = []  # Track lives remaining at end of episode
    tower_positions = []  # Track all tower placements
    training_maps = ["map1_0", "map1_1", "map1_2", "map1_3", "map1_4"]
    tower_upgrades = []
    tower_positions_by_map = {map_name: [] for map_name in training_maps}
    tower_upgrades_by_map = {map_name: [] for map_name in training_maps}
    q_values_by_map = {map_name: {'max': [], 'avg': []} for map_name in training_maps}
    all_q_values = []  # Track all Q-values 
    
    best_reward = -float('inf')
    
    # Create directories for models and plots
    os.makedirs("models", exist_ok=True)
    os.makedirs("training_plots", exist_ok=True)
    
    # Training loop
    for episode in range(1, num_episodes + 1):
        # print(f"Starting episode {episode}...")
        env.current_map = training_maps[episode % len(training_maps)]
        env.game_map, env.all_paths, env.paths = load_map(env.current_map)
        env.reset_game()  # Reset the environment
        state = env.get_state()  # Get initial state
        episode_reward = 0
        episode_loss = 0
        steps = 0
        done = False
        max_wave = 1  # Track the highest wave reached in this episode
        episode_tower_positions = set()  # Track tower positions for this episode
        
        
        # Set render flag
        game.DISPLAY_GAME = (episode % render_interval == 0)
        
        # Debug print to confirm rendering status
        if game.DISPLAY_GAME:
            print(f"Episode {episode} - RENDERING ENABLED")

        while not done:
            # Handle Pygame events to prevent freezing
            # print("current wave",env.wave_number)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return
            
            # Track highest wave reached
            max_wave = max(max_wave, env.wave_number)
            
            # Record current tower positions
            tower_count_before = len(env.towers)

            # Track towers and their levels before the action
            tower_levels_before = {tower.grid_position: tower.level for tower in env.towers}
            
            # Select action
            action = agent.select_action(state)
            
            # Take action
            next_state, reward, done, _ = env.step(action)
            
            # Check if a new tower was placed
            if len(env.towers) > tower_count_before:
                # A new tower was placed, record its grid position
                new_tower = env.towers[-1]
                grid_pos = new_tower.grid_position
                episode_tower_positions.add(grid_pos)
                tower_positions.append(grid_pos)
                tower_positions_by_map[env.current_map].append(grid_pos)
            
            #Check for tower upgrades
            for tower in env.towers:
                grid_pos = tower.grid_position
                if grid_pos in tower_levels_before and tower.level > tower_levels_before[grid_pos]:
                    # This tower was upgraded
                    tower_upgrades.append(grid_pos)
                    tower_upgrades_by_map[env.current_map].append(grid_pos)
            
            # Store transition in memory
            agent.memory.add(state, action, reward, next_state, done)
            
            # Learn
            if steps % 25==0 and len(agent.memory) >= agent.batch_size: 
                loss = agent.learn()
                if loss is not None:
                    episode_loss += loss
                    all_losses.append(loss)

            # Update state
            state = next_state
            episode_reward += reward
            steps += 1
            

        
        # Update target network
        if episode % agent.target_update == 0:
            agent.update_target_network()
        
        max_q, avg_q, episode_q_dist = agent.compute_episode_q_stats()
        q_values_by_map[env.current_map]['max'].append(max_q)
        q_values_by_map[env.current_map]['avg'].append(avg_q)

        # Record episode statistics
        episode_rewards.append(episode_reward)
        episode_lengths.append(steps)
        all_rewards.extend([episode_reward] * steps)
        wave_progress.append(max_wave)  # Record highest wave reached
        epsilon_history.append(agent.epsilon)  # Record current epsilon
        lives_remaining.append(env.player_lives)  # Record lives remaining at end
        
        # Save best model
        if episode_reward > best_reward:
            best_reward = episode_reward
            agent.save(f"models/best_model.pth")
        
        # Save checkpoint
        if episode % 100 == 0:
            agent.save(f"models/model_episode_{episode}.pth")
        
        # Printing progress
        print(f"Episode {episode}/{num_episodes} - Reward: {episode_reward:.2f}, "
              f"Steps: {steps}, Epsilon: {agent.epsilon:.4f}, "
              f"Lives: {env.player_lives}, Max Wave: {max_wave}, "
              f"Towers: {len(episode_tower_positions)}")
        print(f"Memory buffer size: {len(agent.memory)}, Batch size: {agent.batch_size}")
        
        # Plot progress every 10 episodes to see more frequent updates
        if episode % 5 == 0:
            plot_training_progress(all_rewards, episode_rewards, episode_lengths, 
                                 all_losses, wave_progress, epsilon_history, 
                                 lives_remaining, tower_positions, 
                                 tower_upgrades=tower_upgrades,
                                 tower_positions_by_map=tower_positions_by_map,
                                 tower_upgrades_by_map=tower_upgrades_by_map,
                                 q_values={'max': agent.max_q_values_per_episode, 
                                        'avg': agent.avg_q_values_per_episode,
                                        'by_map': q_values_by_map,
                                        'distribution': all_q_values})
    
    # Save final model
    agent.save("models/best_model.pth")
    

    
    return agent

# Visualization function
def plot_training_progress(all_rewards, episode_rewards, episode_lengths, all_losses, 
                           wave_progress, epsilon_history, lives_remaining, tower_positions, 
                           tower_upgrades=None, tower_positions_by_map=None, tower_upgrades_by_map=None,
                           training_maps=None, **kwargs):
    # Create a directory for plots if it doesn't exist
    plots_dir = "training_plots"
    os.makedirs(plots_dir, exist_ok=True)
    
    # Create a timestamp for unique filenames
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Figure for rewards and learning metrics
    plt.figure(figsize=(15, 15))
    
    # Plot accumulated reward per episode
    plt.subplot(3, 2, 1)
    plt.plot(episode_rewards)
    plt.title('Accumulated Reward per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # # Plot number of waves passed per episode
    # plt.subplot(3, 2, 2)
    # plt.plot(wave_progress, 'g-')
    # plt.title('Waves Completed per Episode')
    # plt.xlabel('Episode')
    # plt.ylabel('Wave Number')
    # plt.grid(True, linestyle='--', alpha=0.7)
    
    # # Plot epsilon decay
    # plt.subplot(3, 2, 3)
    # plt.plot(epsilon_history, 'r-')
    # plt.title('Exploration Rate (Epsilon) Decay')
    # plt.xlabel('Episode')
    # plt.ylabel('Epsilon')
    # plt.grid(True, linestyle='--', alpha=0.7)
    
    # Plot lives remaining
    plt.subplot(3, 2, 4)
    plt.plot(lives_remaining, 'orange')
    plt.title('Lives Remaining at Episode End')
    plt.xlabel('Episode')
    plt.ylabel('Lives')
    plt.grid(True, linestyle='--', alpha=0.7)
    


    
    plt.tight_layout()
    plt.savefig(f"{plots_dir}/training_metrics_{timestamp}.png", dpi=150)
    plt.close()
    
    # Create a summary plot with just the key metrics
    plt.figure(figsize=(10, 8))
    
    plt.subplot(2, 2, 1)
    plt.plot(episode_rewards, label='Episode Reward', color='blue')
    plt.title('Reward Progress')
    plt.ylabel('Reward')
    plt.grid(True, linestyle='--', alpha=0.7)
    

    
    plt.subplot(2, 2, 3)
    plt.plot(lives_remaining, label='Lives Remaining', color='orange')
    plt.title('Survival Rate')
    plt.xlabel('Episode')
    plt.ylabel('Lives')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # plt.subplot(2, 2, 4)
    # plt.plot(epsilon_history, label='Epsilon', color='red')
    # plt.title('Exploration Rate')
    # plt.xlabel('Episode')
    # plt.ylabel('Epsilon')
    # plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(f"{plots_dir}/summary_{timestamp}.png", dpi=150)
    plt.close()
    
    # Plot map-specific metrics 
    if training_maps and episode_rewards and len(training_maps) > 1:
        # Calculate average rewards per map
        map_rewards = {map_name: [] for map_name in training_maps}
        map_waves = {map_name: [] for map_name in training_maps}
        map_lives = {map_name: [] for map_name in training_maps}
        
        for i, reward in enumerate(episode_rewards):
            map_name = training_maps[i % len(training_maps)]
            map_rewards[map_name].append(reward)
            
            if i < len(wave_progress):
                map_waves[map_name].append(wave_progress[i])
            
            if i < len(lives_remaining):
                map_lives[map_name].append(lives_remaining[i])
        
        # Plot average rewards per map
        plt.figure(figsize=(15, 10))
        
        plt.subplot(2, 2, 1)
        for map_name, rewards in map_rewards.items():
            # Calculate moving average for smoothing
            window_size = min(10, max(1, len(rewards) // 5))
            if len(rewards) > window_size:
                smoothed = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
                plt.plot(range(window_size-1, len(rewards)), smoothed, label=f'Map: {map_name}')
            else:
                plt.plot(rewards, label=f'Map: {map_name}')
        
        plt.title('Average Reward by Map')
        plt.xlabel('Episodes')
        plt.ylabel('Reward')
        plt.legend()
        plt.grid(True)
        
        # Plot average waves completed per map
        plt.subplot(2, 2, 2)
        for map_name, waves in map_waves.items():
            if waves:
                # Calculate moving average for smoothing
                window_size = min(10, max(1, len(waves) // 5))
                if len(waves) > window_size:
                    smoothed = np.convolve(waves, np.ones(window_size)/window_size, mode='valid')
                    plt.plot(range(window_size-1, len(waves)), smoothed, label=f'Map: {map_name}')
                else:
                    plt.plot(waves, label=f'Map: {map_name}')
        
        plt.title('Average Waves Completed by Map')
        plt.xlabel('Episodes')
        plt.ylabel('Waves')
        plt.legend()
        plt.grid(True)
        
        # Plot average lives remaining per map
        plt.subplot(2, 2, 3)
        for map_name, lives in map_lives.items():
            if lives:
                # Calculate moving average for smoothing
                window_size = min(10, max(1, len(lives) // 5))
                if len(lives) > window_size:
                    smoothed = np.convolve(lives, np.ones(window_size)/window_size, mode='valid')
                    plt.plot(range(window_size-1, len(lives)), smoothed, label=f'Map: {map_name}')
                else:
                    plt.plot(lives, label=f'Map: {map_name}')
        
        plt.title('Average Lives Remaining by Map')
        plt.xlabel('Episodes')
        plt.ylabel('Lives')
        plt.legend()
        plt.grid(True)
        
        # Box plot of rewards by map
        plt.subplot(2, 2, 4)
        plt.boxplot([rewards for map_name, rewards in map_rewards.items() if rewards], 
                    labels=[map_name for map_name, rewards in map_rewards.items() if rewards])
        plt.title('Reward Distribution by Map')
        plt.xlabel('Map')
        plt.ylabel('Reward')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(f"{plots_dir}/map_comparison_{timestamp}.png", dpi=150)
        plt.close()
    

    # Plot Q-values
    if 'q_values' in kwargs and kwargs['q_values']:
        q_values = kwargs['q_values']
        
        plt.figure(figsize=(15, 10))
        
        # Plot max Q-values (main metric from the DQN paper)
        plt.subplot(2, 2, 1)
        if 'max' in q_values and q_values['max']:
            plt.plot(q_values['max'], label='Max Q-value', color='blue')
            plt.title('Average Maximum Q-value per Episode')
            plt.xlabel('Episode')
            plt.ylabel('Q-value')
            plt.grid(True)
        
        # Plot average Q-values
        plt.subplot(2, 2, 2)
        if 'avg' in q_values and q_values['avg']:
            plt.plot(q_values['avg'], label='Average Q-value', color='green')
            plt.title('Average Q-value per Episode')
            plt.xlabel('Episode')
            plt.ylabel('Q-value')
            plt.grid(True)
        
        # Plot Q-values by map
        plt.subplot(2, 2, 3)
        if 'by_map' in q_values:
            for map_name, map_q in q_values['by_map'].items():
                if 'max' in map_q and map_q['max']:
                    # Calculate moving average for smoothing
                    window_size = min(10, max(1, len(map_q['max']) // 5))
                    if len(map_q['max']) > window_size:
                        smoothed = np.convolve(map_q['max'], np.ones(window_size)/window_size, mode='valid')
                        plt.plot(range(window_size-1, len(map_q['max'])), smoothed, label=f'Map: {map_name}')
                    else:
                        plt.plot(map_q['max'], label=f'Map: {map_name}')
            
            plt.title('Max Q-value by Map')
            plt.xlabel('Episodes')
            plt.ylabel('Q-value')
            plt.legend()
            plt.grid(True)
        
        # Plot action-specific Q-values
        plt.subplot(2, 2, 4)
        if 'action_values' in q_values and q_values['action_values']:
            # Get the data for top 5 actions
            action_data = q_values['action_values']
            # Only plot up to 5 lines to avoid clutter
            for action_idx in list(action_data.keys())[:5]:
                action_values = action_data[action_idx]
                # Smoothing the values
                window_size = min(10, max(1, len(action_values) // 5))
                if len(action_values) > window_size:
                    smoothed = np.convolve(action_values, np.ones(window_size)/window_size, mode='valid')
                    plt.plot(range(window_size-1, len(action_values)), smoothed, label=f'Action {action_idx}')
                else:
                    plt.plot(action_values, label=f'Action {action_idx}')
                
            plt.title('Q-values for Top Actions')
            plt.xlabel('Episode')
            plt.ylabel('Q-value')
            plt.legend()
            plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(f"{plots_dir}/q_values_{timestamp}.png", dpi=150)
        plt.close()
    
    # Create map-specific tower placement heatmaps
    # if tower_positions_by_map:
    #     from maps import maps  # Import maps data
        
    #     for map_name, positions in tower_positions_by_map.items():
    #         if not positions:  # Skip maps with no tower placements
    #             continue
                
    #         # Get map dimensions from maps data
    #         map_grid = maps[map_name]["grid"]
    #         map_height = len(map_grid)
    #         map_width = len(map_grid[0])
            
    #         # Create heatmap for this map
    #         heatmap = np.zeros((map_height, map_width))
            

    #         for pos in positions:
    #             x, y = pos
    #             if 0 <= x < map_width and 0 <= y < map_height:
    #                 heatmap[y][x] += 1
            
    #         # Get the path cells for overlay
    #         path_grid = np.zeros((map_height, map_width))
    #         for y in range(map_height):
    #             for x in range(map_width):
    #                 if map_grid[y][x] in [1, 3, 5]:  # Path tiles
    #                     path_grid[y][x] = 0.5  # Use a visible but not dominant value
            
    #         # Create the plot
    #         plt.figure(figsize=(12, 10))
            
    #         # Show the path grid with a different colormap
    #         plt.imshow(path_grid, cmap='Greys', alpha=0.7, interpolation='nearest')
            
    #         # Overlay tower placement heatmap with transparency
    #         heatmap_plot = plt.imshow(heatmap, cmap='hot', alpha=0.7, interpolation='nearest')
    #         plt.colorbar(heatmap_plot, label='Tower Placement Frequency')
    #         plt.title(f'Tower Placement Heatmap - {map_name}')
    #         plt.xlabel('Grid X')
    #         plt.ylabel('Grid Y')
            
    #         # Add grid lines
    #         plt.grid(True, linestyle='-', color='black', alpha=0.3)
    #         ax = plt.gca()
    #         ax.set_xticks(np.arange(-.5, map_width, 1), minor=True)
    #         ax.set_yticks(np.arange(-.5, map_height, 1), minor=True)
    #         ax.tick_params(which='minor', size=0)
            
    #         # Save map-specific heatmap
    #         plt.savefig(f"{plots_dir}/tower_heatmap_{map_name}_{timestamp}.png", dpi=150)
    #         plt.close()
    
    # # Create map-specific tower upgrade heatmaps
    # if tower_upgrades_by_map:
    #     from maps import maps
        
    #     for map_name, upgrades in tower_upgrades_by_map.items():
    #         if not upgrades:  # Skip maps with no upgrades
    #             continue
                
    #         # Get map dimensions
    #         map_grid = maps[map_name]["grid"]
    #         map_height = len(map_grid)
    #         map_width = len(map_grid[0])
            
    #         # Create upgrade heatmap
    #         upgrade_heatmap = np.zeros((map_height, map_width))
            
    #         # Fill in the upgrade heatmap
    #         for pos in upgrades:
    #             x, y = pos
    #             if 0 <= x < map_width and 0 <= y < map_height:
    #                 upgrade_heatmap[y][x] += 1
            
    #         # Create path overlay
    #         path_grid = np.zeros((map_height, map_width))
    #         for y in range(map_height):
    #             for x in range(map_width):
    #                 if map_grid[y][x] in [1, 3, 5]:
    #                     path_grid[y][x] = 0.5
            
    #         # Create plot
    #         plt.figure(figsize=(12, 10))
    #         plt.imshow(path_grid, cmap='Greys', alpha=0.7, interpolation='nearest')
            
    #         # Overlay upgrade heatmap
    #         upgrade_plot = plt.imshow(upgrade_heatmap, cmap='cool', alpha=0.7, interpolation='nearest')
    #         plt.colorbar(upgrade_plot, label='Tower Upgrade Frequency')
    #         plt.title(f'Tower Upgrade Heatmap - {map_name}')
    #         plt.xlabel('Grid X')
    #         plt.ylabel('Grid Y')
            
    #         # Add grid lines
    #         plt.grid(True, linestyle='-', color='black', alpha=0.3)
    #         ax = plt.gca()
    #         ax.set_xticks(np.arange(-.5, map_width, 1), minor=True)
    #         ax.set_yticks(np.arange(-.5, map_height, 1), minor=True)
    #         ax.tick_params(which='minor', size=0)
            
    #         # Save map-specific upgrade heatmap
    #         plt.savefig(f"{plots_dir}/tower_upgrade_heatmap_{map_name}_{timestamp}.png", dpi=150)
    #         plt.close()

    # if tower_positions:
    #     # Create a grid of zeros with the same dimensions as the game map
    #     heatmap = np.zeros((GRID_HEIGHT, GRID_WIDTH))
        
    #     # Fill in the heatmap with tower placement counts
    #     for pos in tower_positions:
    #         x, y = pos
    #         if 0 <= x < GRID_WIDTH and 0 <= y < GRID_HEIGHT:
    #             heatmap[y][x] += 1
                
    #     # Create plot
    #     plt.figure(figsize=(10, 8))
        
    #     # Just show the heatmap without trying to overlay the path
    #     heatmap_plot = plt.imshow(heatmap, cmap='hot', interpolation='nearest')
    #     plt.colorbar(heatmap_plot, label='Tower Placement Frequency')
    #     plt.title('Tower Placement Heatmap (All Maps)')
    #     plt.xlabel('Grid X')
    #     plt.ylabel('Grid Y')
        
    #     # Add grid lines
    #     plt.grid(True, linestyle='-', color='black', alpha=0.3)
    #     ax = plt.gca()
    #     ax.set_xticks(np.arange(-.5, GRID_WIDTH, 1), minor=True)
    #     ax.set_yticks(np.arange(-.5, GRID_HEIGHT, 1), minor=True)
    #     ax.tick_params(which='minor', size=0)
        
    #     # Save the heatmap
    #     plt.savefig(f"{plots_dir}/tower_heatmap_combined_{timestamp}.png", dpi=150)
    #     plt.close()
    
    # For backward compatibility, keep the original combined upgrade heatmap
    # if tower_upgrades and len(tower_upgrades) > 0:
    #     # Create a grid for tracking upgrade frequency
    #     upgrade_heatmap = np.zeros((GRID_HEIGHT, GRID_WIDTH))
        
    #     # Fill in the upgrade heatmap
    #     for pos in tower_upgrades:
    #         x, y = pos
    #         if 0 <= x < GRID_WIDTH and 0 <= y < GRID_HEIGHT:
    #             upgrade_heatmap[y][x] += 1
        
    #     # Create plot
    #     plt.figure(figsize=(10, 8))
        
    #     # Just show the upgrade heatmap without path overlay
    #     upgrade_plot = plt.imshow(upgrade_heatmap, cmap='cool', interpolation='nearest')
    #     plt.colorbar(upgrade_plot, label='Tower Upgrade Frequency')
    #     plt.title('Tower Upgrade Heatmap (All Maps)')
    #     plt.xlabel('Grid X')
    #     plt.ylabel('Grid Y')
        
    #     # Add grid lines
    #     plt.grid(True, linestyle='-', color='black', alpha=0.3)
    #     ax = plt.gca()
    #     ax.set_xticks(np.arange(-.5, GRID_WIDTH, 1), minor=True)
    #     ax.set_yticks(np.arange(-.5, GRID_HEIGHT, 1), minor=True)
    #     ax.tick_params(which='minor', size=0)
        
    #     # Save the upgrade heatmap
    #     plt.savefig(f"{plots_dir}/tower_upgrade_heatmap_combined_{timestamp}.png", dpi=150)
    #     plt.close()

# Testing function
def test_agent(model_path, num_episodes=5):
    # Create game environment
    env = GameManager()
    
    # Get grid dimensions
    global GRID_HEIGHT, GRID_WIDTH
    GRID_HEIGHT = len(env.game_map)
    GRID_WIDTH = len(env.game_map[0])
    
    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create agent
    action_size = 3 * 10 + 3 * 10 + 1 # Places, upgrades, and do nothing
    agent = DQNAgent(action_size, device)
    
    # Load trained model
    agent.load(model_path)
    agent.epsilon = 0.0  # No exploration during testing
    
    # Enable rendering - set it directly in game module
    game.DISPLAY_GAME = True
    print("Test Mode - RENDERING ENABLED")
    
    # Testing loop
    for episode in range(1, num_episodes + 1):
        env.reset_game()  # Reset the environment
        state = env.get_state()  # Get initial state
        episode_reward = 0
        done = False
        
        print(f"Testing Episode {episode}/{num_episodes}")
        
        while not done:
            # Handle Pygame events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return
            
            # Select best action
            action = agent.select_action(state)
            
            # Take action
            next_state, reward, done, _ = env.step(action)
            
            # Update state
            state = next_state
            episode_reward += reward
            
        
        print(f"Episode {episode} finished with score {env.score}, lives: {env.player_lives}, wave: {env.wave_number}")