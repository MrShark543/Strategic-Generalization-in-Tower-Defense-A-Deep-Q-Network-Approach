import pygame
import torch
import os
import time
import sys
import game
from game import GameManager
from dqn_agent import train_agent, test_agent

# Configure error handling
def handle_error(error_msg, fatal=False):
    print(f"\nERROR: {error_msg}")
    if fatal:
        print("Exiting program due to fatal error.")
        sys.exit(1)
    else:
        print("Continuing with fallback...")



    
pygame.init()
pygame.display.init()


# Now import the modules with proper error handling



if __name__ == "__main__":
  
    
    # Choose mode
    mode = input("Enter 'train' to train a new agent or 'test' to test an existing model: ").strip().lower()
    
    if mode == 'train':
        # Training parameters
        try:
            num_episodes = int(input("Enter number of episodes to train (default 1000): ") or 1000)
            render_interval = int(input("Render every N episodes (default 100): ") or 100)
        except ValueError:
            print("Invalid input. Using default values.")
            num_episodes = 1000
            render_interval = 100
        
        # Train the agent
        print(f"\nStarting training for {num_episodes} episodes...")
        print(f"Rendering will happen every {render_interval} episodes.")
        
        # Test the rendering directly
        print("\nTesting rendering capability...")
        try:
            game.DISPLAY_GAME = True
            test_env = game.GameManager()
            test_env.display()  # Try to render once
            pygame.display.flip()
            time.sleep(0.5)  # Keep the window open briefly
            
            # Check if the display is working
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    print("Display window closed by user.")
                    pygame.quit()
                    sys.exit(0)
                    
            print("Rendering test completed successfully.")
        except Exception as e:
            handle_error(f"Rendering test failed: {e}")
            response = input("\nDo you want to continue without rendering? (y/n): ").strip().lower()
            if response != 'y':
                print("Exiting...")
                sys.exit(0)
            render_interval = float('inf')  # Disable rendering
        
        print("\nBeginning training...\n")
        
        try:
            agent = train_agent(num_episodes=num_episodes, render_interval=render_interval)
            print("\nTraining completed successfully!")
            
            # Ask if user wants to test the trained agent
            test_trained = input("\nDo you want to test the trained agent? (y/n): ").strip().lower()
            if test_trained == 'y':
                test_agent("models/best_model.pth", num_episodes=5)
        except Exception as e:
            handle_error(f"Training failed: {e}")
            # Try to save model if possible
            try:
                if 'agent' in locals() and hasattr(agent, 'save'):
                    agent.save("models/emergency_save.pth")
                    print("Emergency model save created at models/emergency_save.pth")
            except:
                print("Could not create emergency save.")
    
    elif mode == 'test':
        # Get model path
        model_path = input("Enter the path to the model file (default 'models/best_model.pth'): ") or "models/best_model.pth"
        
        # Verify model file exists
        if not os.path.exists(model_path):
            handle_error(f"Model file not found at {model_path}", fatal=True)
            
        try:
            num_test_episodes = int(input("Enter number of test episodes (default 5): ") or 5)
        except ValueError:
            print("Invalid input. Using default value of 5 episodes.")
            num_test_episodes = 5
        
        # Test the agent
        print(f"\nTesting agent from {model_path} for {num_test_episodes} episodes...")
        try:
            test_agent(model_path, num_episodes=num_test_episodes)
        except Exception as e:
            handle_error(f"Testing failed: {e}")
    
    else:
        print("Invalid mode. Please enter 'train' or 'test'.")
    
    # Clean up
    try:
        leave = input()
        pygame.quit()
    except:
        pass
    print("\nDone!")