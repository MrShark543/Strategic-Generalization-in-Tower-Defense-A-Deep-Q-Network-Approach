import pygame
import math
import sys
import random
from maps import *
import numpy as np

# Initialize Pygame
pygame.init()

# Game constants
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
GRID_SIZE = 40
ENEMY_SPEED = 2
TOWER_RANGE = 150
TOWER_DAMAGE = 10
TOWER_FIRE_RATE = 1  # shots per second
ENEMY_HEALTH = 100
PLAYER_LIVES = 10
PLAYER_MONEY = 400
TOWER_COST = 50
TOWER_UPGRADE_COST = 70

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = [0, 0, 255]
GRAY = (128, 128, 128)
BROWN = (165, 42, 42)
YELLOW = (255, 255, 0)
PURPLE = (128, 0, 128)

# Global display flag
DISPLAY_GAME = False


screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Tower Defense ")


def interpolate_paths(path_dict, step_size = 10):
    """Creating max of 10 pixel points between two consecutive coordinate positions 
    to achieve smooth motion of the enemies
    """
    interpolated_paths = {}

    for path_name, waypoints in path_dict.items():
        smooth_path = []

        for i in range(len(waypoints)-1):
            start_x, start_y = waypoints[i]
            end_x, end_y = waypoints[i+1]
            distance = math.sqrt((end_x - start_x)**2 + (end_y - start_y)**2) 
            steps = max(1, int(distance/step_size))
            smooth_path.append((start_x, start_y))
            for step in range(1, steps):
                x = start_x + (end_x - start_x) * step / steps
                y = start_y + (end_y - start_y) * step / steps
                smooth_path.append((x,y))
        smooth_path.append(waypoints[-1])
        interpolated_paths[path_name] = smooth_path
    return interpolated_paths

def load_map(map_name):
    map_data = maps[map_name]
    game_map = map_data["grid"]

    all_paths = {}
    for path_name, points in map_data["waypoints"].items():
        pixel_points = []
        for grid_x, grid_y in points:
            pixel_x = grid_x * GRID_SIZE + GRID_SIZE // 2
            pixel_y = grid_y * GRID_SIZE + GRID_SIZE // 2
            pixel_points.append((pixel_x, pixel_y))
        all_paths[path_name] = pixel_points
    all_paths = interpolate_paths(all_paths)
    return game_map, all_paths, map_data["paths"]
            


GRID_WIDTH = 20
GRID_HEIGHT = 10


font = pygame.font.SysFont(None, 30)

def place_text(text, x, y, color):
    if not DISPLAY_GAME:
        return
    text = font.render(text, True, color)
    screen.blit(text, (x,y))

# Enemy class
class Enemy:
    def __init__(self, path_name, all_paths):
        self.path_name = path_name
        self.path = all_paths[path_name]
        self.path_index = 0
        self.position = list(self.path[0])
        self.health = ENEMY_HEALTH
        self.max_health = ENEMY_HEALTH
        self.radius = 15
        self.dead = False
        self.reached_end = False
        self.color = 'cyan'
     
    def move(self):
        if self.path_index < len(self.path) - 1:
            target = self.path[self.path_index + 1]
            direction_x = target[0] - self.position[0]
            direction_y = target[1] - self.position[1]
            distance = math.sqrt(direction_x ** 2 + direction_y ** 2)
            
            if distance < ENEMY_SPEED:
                self.path_index += 1
            else:
                self.position[0] += direction_x / distance * ENEMY_SPEED
                self.position[1] += direction_y / distance * ENEMY_SPEED
        else:
            # Enemy reached the end of the path
            self.reached_end = True
            self.dead = True
    
    def take_damage(self, damage):
        self.health -= damage
        if self.health <= 0:
            self.dead = True
    
    def draw(self):
        if not DISPLAY_GAME:
            return
            
      
        # Drawing enemy w
        pygame.draw.circle(screen, self.color, (int(self.position[0]), int(self.position[1])), self.radius)
        
        # Draw health bar
        health_bar_length = 30
        health_bar_height = 5
        health_ratio = self.health / self.max_health
        health_bar_fill = health_ratio * health_bar_length
        
        pygame.draw.rect(screen, BLACK, (self.position[0] - health_bar_length // 2, 
                                    self.position[1] - self.radius - 10, 
                                    health_bar_length, health_bar_height))
        pygame.draw.rect(screen, GREEN, (self.position[0] - health_bar_length // 2, 
                                    self.position[1] - self.radius - 10, 
                                    health_bar_fill, health_bar_height))


# Tower class
class Tower:
    def __init__(self, x, y):
        self.position = (x, y)
        self.grid_position = (x // GRID_SIZE, y // GRID_SIZE)
        self.range = TOWER_RANGE
        self.damage = TOWER_DAMAGE
        self.fire_rate = TOWER_FIRE_RATE
        self.last_shot_time = 0 #conrols fire rate
        self.target = None
        self.upgrade_multiplier = 1
        self.color = [0, 0, 255]
        self.level = 1
        self.tower_rect = pygame.Rect(
            self.position[0] - GRID_SIZE // 2,
            self.position[1] - GRID_SIZE // 2,
            GRID_SIZE,
            GRID_SIZE
        )
    
    def find_target(self, enemies):
        self.target = None
        shortest_distance = float('inf')
        
        for enemy in enemies:
            if enemy.dead:
                continue
            
            distance = math.sqrt((enemy.position[0] - self.position[0]) ** 2 + 
                                 (enemy.position[1] - self.position[1]) ** 2)
            if distance < self.range and distance < shortest_distance:
                shortest_distance = distance
                self.target = enemy
    
    def shoot(self, frame_counter):
        frames_per_shot = int(30 / self.fire_rate)
        if self.target and (frame_counter - self.last_shot_time) >= frames_per_shot:
            self.target.take_damage(self.damage)
            self.last_shot_time = frame_counter
            return True
        return False
    
    def draw(self):
        if not DISPLAY_GAME:
            return
            
    
        # Draw tower
        self.tower_rect = pygame.draw.rect(screen, self.color, 
                                        (self.position[0] - GRID_SIZE // 2, 
                                        self.position[1] - GRID_SIZE // 2, 
                                        GRID_SIZE, GRID_SIZE))
        place_text(f"{self.level}", (self.position[0] - GRID_SIZE // 2 ) + 15, (self.position[1] - GRID_SIZE // 2) + 10, 'white' )
        
        # Draw line to target
        if self.target and not self.target.dead:
            pygame.draw.line(screen, self.color, self.position, 
                        (self.target.position[0], self.target.position[1]), 2)
    
   
    def upgrade(self):
        if self.level <= 4:
            self.damage += 5 # Increase damage more significantly
            self.fire_rate += 0.2  # Increase fire rate
            self.upgrade_multiplier += 2
            self.color[2] -= 25 if self.color[2] > 0 else 0
            self.color[0] += 25 if self.color[0] < 255 else 0
            self.level += 1

# Projectile class
class Projectile:
    def __init__(self, start_pos, target):
        self.position = list(start_pos)
        self.target = target
        self.speed = 10
        self.radius = 5
        self.reached_target = False
    
    def move(self):
        if self.target.dead:
            self.reached_target = True
            return
        
        direction_x = self.target.position[0] - self.position[0]
        direction_y = self.target.position[1] - self.position[1]
        distance = math.sqrt(direction_x ** 2 + direction_y ** 2)
        
        if distance < self.speed:
            self.reached_target = True
        else:
            self.position[0] += direction_x / distance * self.speed
            self.position[1] += direction_y / distance * self.speed
    
    def draw(self):
        if not DISPLAY_GAME:
            return
            
   
        pygame.draw.circle(screen, BLUE, (int(self.position[0]), int(self.position[1])), self.radius)


class GameManager:
    def __init__(self):
        # Game states
        self.enemies = []
        self.towers = []
        self.projectiles = []
        self.spawn_timer = 0
        self.player_lives = PLAYER_LIVES
        self.player_money = PLAYER_MONEY
        self.game_over = False
        self.score = 0
        self.wave_number = 1
        self.enemies_in_wave = 10
        self.enemies_spawned = 0
        self.wave_cleared = False
        self.steps_taken = 0
        self.max_waves = 3 
        self.current_map = "map1_2"
        self.game_map, self.all_paths, self.paths = load_map(self.current_map)
        self.frame_counter = 0
        self.clock = pygame.time.Clock()
        self.GRID_WIDTH = len(self.game_map[0])
        self.GRID_HEIGHT = len(self.game_map)

        global GRID_HEIGHT, GRID_WIDTH
        GRID_WIDTH = len(self.game_map[0])
        GRID_HEIGHT = len(self.game_map)

    # Helper functions
    def draw_grid(self):
        if not DISPLAY_GAME:
            return
            
        try:
            for x in range(0, SCREEN_WIDTH, GRID_SIZE):
                pygame.draw.line(screen, GRAY, (x, 0), (x, SCREEN_HEIGHT), 1)
            for y in range(0, SCREEN_HEIGHT, GRID_SIZE):
                pygame.draw.line(screen, GRAY, (0, y), (SCREEN_WIDTH, y), 1)
        except pygame.error as e:
            print(f"Warning: Failed to draw grid: {e}")

    def draw_map(self):
        if not DISPLAY_GAME:
            return
            
        try:
            for y, row in enumerate(self.game_map):
                for x, cell in enumerate(row):
                    rect = pygame.Rect(x * GRID_SIZE, y * GRID_SIZE, GRID_SIZE, GRID_SIZE)
                    if cell == 1:  # Path 1
                        pygame.draw.rect(screen, BROWN, rect)
                    elif cell == 5:  # Converged path
                        pygame.draw.rect(screen, BROWN, rect)
                    elif cell == 2:  # Tower placable tile
                        pygame.draw.rect(screen, GRAY, rect)
                    else:  #grass
                        pygame.draw.rect(screen, GREEN, rect)
        except pygame.error as e:
            print(f"Warning: Failed to draw map: {e}")

    def is_buildable(self, x, y):
        grid_x = x // GRID_SIZE
        grid_y = y // GRID_SIZE
        if grid_x < 0 or grid_x >= len(self.game_map[0]) or grid_y < 0 or grid_y >= len(self.game_map):
            return False
        return self.game_map[grid_y][grid_x] == 2 and not any(
            tower.position[0] // GRID_SIZE == grid_x and tower.position[1] // GRID_SIZE == grid_y
            for tower in self.towers
        )

    def is_upgradable(self, x, y):
        """checking to see if the tower is upgradable"""
        for tower in self.towers:
            if tower.tower_rect.collidepoint((x, y)):
                if self.can_upgrade_tower(tower):
                    self.player_money -= TOWER_UPGRADE_COST * tower.upgrade_multiplier
                    tower.upgrade()
                    return True
        return False

    def can_afford_tower(self):
        return self.player_money >= TOWER_COST

    def can_upgrade_tower(self, tower):
        return self.player_money >= TOWER_UPGRADE_COST * tower.upgrade_multiplier
    
    def can_be_upgraded(self, tower_position):
        """check if a tower can be upgraded without performing the upgrade"""
        for tower in self.towers:
            if tower.tower_rect.collidepoint(tower_position):
                return self.can_upgrade_tower(tower)
        return False

    def spawn_enemy(self):
        path_choice = self.paths[0]
        
        self.enemies.append(Enemy(path_choice, self.all_paths))
        self.enemies_spawned += 1
        self.spawn_timer = pygame.time.get_ticks()

    def draw_ui(self):
        if not DISPLAY_GAME:
            return
            

        # Draw lives
        place_text(f"Lives: {self.player_lives}", 10, 10, 'black')
        
        # Draw money
        place_text(f"Money: ${self.player_money}", 10, 40, 'black')
        
        # Draw score
        place_text(f"Score: {self.score}", 10, 70, 'black')
        
        # Draw wave number
        place_text(f"Wave: {self.wave_number}", 10, 100, 'black')

        if self.game_over:
            place_text("GAME OVER!!", SCREEN_WIDTH // 2 - 100, SCREEN_HEIGHT // 2, 'red')
        
        # Draw wave cleared message 
        if self.wave_cleared and not self.enemies:
            place_text("Wave Cleared!!", SCREEN_WIDTH // 2 - 100, SCREEN_HEIGHT // 2, 'purple')
 
            
    def draw_game_entities(self):
        if not DISPLAY_GAME:
            return
            
        try:
            for tower in self.towers:
                tower.draw()

            for projectile in self.projectiles:
                projectile.draw()

            for enemy in self.enemies:
                enemy.draw()
            
            self.draw_ui()
        except (TypeError, pygame.error) as e:
            print(f"Warning: Failed to draw game entities: {e}")
        
    def reset_game(self):
        self.enemies = []
        self.towers = []
        self.projectiles = []
        self.player_lives = PLAYER_LIVES
        self.player_money = PLAYER_MONEY
        self.game_over = False
        self.score = 0
        self.wave_number = 1
        self.enemies_in_wave = 10
        self.enemies_spawned = 0
        self.wave_cleared = False
        self.reward = 0
        self.steps_taken = 0
    
    def update(self):

        step_reward = 0
        # Initialize enemies_killed outside the conditional block
        enemies_killed = 0
        
        # Smaller per-step penalty to avoid large negative rewards over time
        step_reward -= 0.0001  
        
        self.frame_counter+=1
        if not self.game_over and not self.wave_cleared:
            # Spawn enemies
            if self.enemies_spawned < self.enemies_in_wave and self.frame_counter % 30 == 0:
                self.spawn_enemy()
            
            if self.enemies_spawned >= self.enemies_in_wave and not self.enemies:
                self.wave_cleared = True
                step_reward += 150  
                self.player_money += 100  
            
            # Update towers (targeting and shooting)
            for tower in self.towers:
                tower.find_target(self.enemies)
                if tower.shoot(self.frame_counter):
                    self.projectiles.append(Projectile(tower.position, tower.target))
                    # Small reward for shooting
                    step_reward += 0.2
            
            # Update projectiles
            for projectile in self.projectiles[:]:
                projectile.move()
                if projectile.reached_target:
                    self.projectiles.remove(projectile)
            
            # Update enemies
            for enemy in self.enemies[:]:
                enemy.move()
                
                if enemy.reached_end:
                    self.player_lives -= 1
                    step_reward -= 5 
                    if self.player_lives <= 0:
                        self.game_over = True
                        step_reward -= 50 
                
                if enemy.dead:
                    if not enemy.reached_end:
                        self.player_money += 20
                        self.score += 10
                        enemies_killed += 1
                    self.enemies.remove(enemy)
                    
        # Increase wave clear bonus based on wave number
        if self.wave_cleared and not self.enemies:
            step_reward += 150 + (self.wave_number * 30)  
        
        # Reward for killing enemies
        step_reward += enemies_killed * 15  
        
        # Reward for having towers placed 
        step_reward += len(self.towers) * 0.1
        
        # Additional reward for tower upgrades
        upgraded_tower_count = sum(1 for tower in self.towers if tower.level > 1)
        step_reward += upgraded_tower_count * 0.2
        
        self.steps_taken += 1
        self.reward = step_reward
        return step_reward, self.game_over or self.wave_cleared

    def place_tower(self, grid_x, grid_y):
        pixel_x = grid_x * GRID_SIZE + GRID_SIZE // 2
        pixel_y = grid_y * GRID_SIZE + GRID_SIZE // 2
        
        if self.can_afford_tower() and not any(tower.grid_position == (grid_x, grid_y) for tower in self.towers):
            self.towers.append(Tower(pixel_x, pixel_y))
            self.player_money -= TOWER_COST
            return True
        return False

    def display(self):
        if not DISPLAY_GAME:
            return
            

        screen.fill(WHITE)
        
        
        self.draw_map()
        self.draw_grid()
        self.draw_game_entities()
        
        # Update the display
        pygame.display.update()
   
        
    def get_state(self):
        """
        Create a state representation using relative distances from paths.
        """
        # Distance maps 
        path_distance_map = np.ones((GRID_HEIGHT, GRID_WIDTH)) * 999
        
        # Find all path cells
        path_cells = []
        for y in range(GRID_HEIGHT):
            for x in range(GRID_WIDTH):
                if self.game_map[y][x] in [1, 5]:  # Path tiles
                    path_cells.append((x, y))
        
        # Calculate distance from each cell to nearest path
        for y in range(GRID_HEIGHT):
            for x in range(GRID_WIDTH):
                for px, py in path_cells:
                    dist = np.sqrt((x - px)**2 + (y - py)**2)
                    path_distance_map[y][x] = min(path_distance_map[y][x], dist)
        
        # Normalize distances
        max_dist = np.max(path_distance_map)
        path_distance_map = path_distance_map / max_dist
        
        # Existing tower placement grid
        tower_grid = np.zeros((GRID_HEIGHT, GRID_WIDTH))
        for tower in self.towers:
            x, y = tower.grid_position
            if 0 <= x < GRID_WIDTH and 0 <= y < GRID_HEIGHT:
                tower_grid[y][x] = 1
        
        # Enemy density map
        enemy_grid = np.zeros((GRID_HEIGHT, GRID_WIDTH))
        for enemy in self.enemies:
            grid_x = int(enemy.position[0] // GRID_SIZE)
            grid_y = int(enemy.position[1] // GRID_SIZE)
            if 0 <= grid_x < GRID_WIDTH and 0 <= grid_y < GRID_HEIGHT:
                enemy_grid[grid_y][grid_x] += enemy.health / enemy.max_health
        
        # Path coverage 
        coverage_map = np.zeros((GRID_HEIGHT, GRID_WIDTH))
        for y in range(GRID_HEIGHT):
            for x in range(GRID_WIDTH):
                if self.game_map[y][x] == 2:  
                    tower_pos = (x * GRID_SIZE + GRID_SIZE // 2, y * GRID_SIZE + GRID_SIZE // 2)
                    coverage = 0
                    for px, py in path_cells:
                        path_pos = (px * GRID_SIZE + GRID_SIZE // 2, py * GRID_SIZE + GRID_SIZE // 2)
                        dist = math.sqrt((tower_pos[0] - path_pos[0])**2 + (tower_pos[1] - path_pos[1])**2)
                        if dist <= TOWER_RANGE:
                            coverage += 1
                    coverage_map[y][x] = coverage / len(path_cells) if path_cells else 0
        
        state = np.stack([
            tower_grid,
            path_distance_map,
            coverage_map,
            enemy_grid
        ], axis=0)
    
        # Same scalar state
        scalar_state = np.array([
            self.player_money / 500,
            self.wave_number / 10,
            len(self.enemies) / 20,
            self.player_lives / PLAYER_LIVES
        ])
        
        return {
            'grid_state': state,
            'scalar_state': scalar_state
        }
    
    def get_strategic_positions(self):
        """Get positions based on strategic value rather than fixed coordinates"""
        positions = []
        
        # Find all buildable positions
        buildable = []
        for y in range(GRID_HEIGHT):
            for x in range(GRID_WIDTH):
                if self.game_map[y][x] == 2:
                    buildable.append((x, y))
        
        # Group positions by strategic value
        close_to_path = []
        medium_distance = []
        far_from_path = []
        
        # Calculate path coverage for each position
        path_cells = [(x, y) for y in range(GRID_HEIGHT) for x in range(GRID_WIDTH) 
                    if self.game_map[y][x] in [1, 3, 5]]
        
        for x, y in buildable:
            tower_pos = (x * GRID_SIZE + GRID_SIZE // 2, y * GRID_SIZE + GRID_SIZE // 2)
            
            # Find nearest path cell
            min_dist = float('inf')
            for px, py in path_cells:
                path_pos = (px * GRID_SIZE + GRID_SIZE // 2, py * GRID_SIZE + GRID_SIZE // 2)
                dist = math.sqrt((tower_pos[0] - path_pos[0])**2 + (tower_pos[1] - path_pos[1])**2)
                min_dist = min(min_dist, dist)
            
            # Categorize by distance
            if min_dist < GRID_SIZE * 1.5:
                close_to_path.append((x, y))
            elif min_dist < GRID_SIZE * 3:
                medium_distance.append((x, y))
            else:
                far_from_path.append((x, y))
        
        # Return all categories
        return {
            'close': close_to_path,
            'medium': medium_distance,
            'far': far_from_path
        }

    def step(self, action):
        """
        Take a step using strategic positions for better generalization across maps.
        
        Actions:
        - 0-9: Place tower at 'close' position (0-9)
        - 10-19: Place tower at 'medium' position (0-9)
        - 20-29: Place tower at 'far' position (0-9)
        - 30-39: Upgrade tower at 'close' position (0-9)
        - 40-49: Upgrade tower at 'medium' position (0-9)
        - 50-59: Upgrade tower at 'far' position (0-9)
        - 60: Do nothing (save money)
        """
        strategic_positions = self.get_strategic_positions()
        action_reward = 0
        max_positions_per_category = 10  # Maximum positions we consider per category
        
        # Decode the action
        if action < 3 * max_positions_per_category:  # Tower placement actions
            # Determine category
            category_index = action // max_positions_per_category
            position_index = action % max_positions_per_category
            
            # Map category index to category name
            categories = ['close', 'medium', 'far']
            category = categories[category_index]
            
            # Get positions for this category
            positions = strategic_positions[category]
            
            # Check if position exists
            if position_index < len(positions):
                grid_x, grid_y = positions[position_index]
                placed = self.place_tower(grid_x, grid_y)
                if placed:
                    # Extra reward for placing in more strategic locations
                    base_reward = 5
                    if category == 'close':
                        action_reward += base_reward * 1.5  # More reward for close positions
                    elif category == 'medium':
                        action_reward += base_reward * 1.0
                    else:  # 'far'
                        action_reward += base_reward * 0.7  # Less reward for far positions
        
        elif action < 6 * max_positions_per_category:  # Tower upgrade actions
            # Adjust index for upgrade actions
            adjusted_action = action - (3 * max_positions_per_category)
            
            # Determine category for upgrade
            category_index = adjusted_action // max_positions_per_category
            position_index = adjusted_action % max_positions_per_category
            
            # Map category index to category name
            categories = ['close', 'medium', 'far']
            category = categories[category_index]
            
            # Get positions for this category
            positions = strategic_positions[category]
            
            # Check if position exists
            if position_index < len(positions):
                grid_x, grid_y = positions[position_index]
                pos_x = grid_x * GRID_SIZE + GRID_SIZE // 2
                pos_y = grid_y * GRID_SIZE + GRID_SIZE // 2
                upgraded = self.is_upgradable(pos_x, pos_y)
                if upgraded:
                    # Extra reward for upgrading more strategic locations
                    base_reward = 10
                    if category == 'close':
                        action_reward += base_reward * 1.5
                    elif category == 'medium':
                        action_reward += base_reward * 1.0
                    else:  # 'far'
                        action_reward += base_reward * 0.7
        
        # else: do nothing (save money)
        
        # Update game state
        game_reward, done = self.update()
        
        # Combine action reward with game reward
        reward = action_reward + game_reward
        
        if self.wave_cleared and not self.enemies:
            if self.wave_number >= self.max_waves:
                reward += 500  # Big bonus for completing all waves
                done = True
            else:
                # Start next wave automatically
                self.wave_number += 1
                self.enemies_in_wave = 20 + 5 * (self.wave_number - 1)
                self.enemies_spawned = 0
                self.wave_cleared = False
                self.player_money += 100
                reward += 200
                done = False  # Continue the episode
        
        # Only return done=True if game is over or all waves completed
        done = self.game_over or (self.wave_number >= self.max_waves and self.wave_cleared and not self.enemies)
            
        # Display the game
        self.display()

        if DISPLAY_GAME:
            self.clock.tick(30)
        
        # Return step information
        return self.get_state(), reward, done, {}
