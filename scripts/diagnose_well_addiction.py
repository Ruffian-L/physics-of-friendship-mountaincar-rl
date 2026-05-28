import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from src.agent import QSMA_Agent
from src.tda import TopologicalBrain
from src.steering import SteeringController

def diagnose_well_addiction():
    env = gym.make('MountainCar-v0')
    agent = QSMA_Agent()
    ctrl = SteeringController()
    brain = TopologicalBrain(ctrl)
    
    # Data collection: [Position, Velocity, Energy, Flux]
    history = []
    
    print("🔬 Running 50 episodes to diagnose Well Addiction...")
    
    for episode in range(50):
        state, _ = env.reset()
        done = False
        while not done:
            action = agent.act(state)
            next_state, reward, done, _, _ = env.step(action)
            
            # Use the exact energy formula from the agent
            pos, vel = next_state
            energy = (pos**2) + (vel**2)
            
            # Track flux accumulation
            disc_state = agent.discretize(state)
            flux = agent.flux[disc_state, action]
            
            history.append([pos, vel, energy, flux])
            
            # Standard learning step
            agent.learn(state, action, reward, next_state)
            state = next_state
            
    env.close()
    data = np.array(history)
    
    # Visualization: The Anatomy of Addiction
    plt.figure(figsize=(12, 5))
    
    # 1. Position vs Energy (The Addiction Curve)
    plt.subplot(1, 2, 1)
    plt.scatter(data[:, 0], data[:, 2], c=data[:, 3], cmap='inferno', s=1, alpha=0.5)
    plt.title("The Trap: Energy vs Position")
    plt.xlabel("Position (Bottom is -0.5)")
    plt.ylabel("Energy (pos² + vel²)")
    plt.axvline(-0.5, color='cyan', linestyle='--', label='Well Bottom')
    plt.colorbar(label='Flux (Habit Strength)')
    plt.grid(True, alpha=0.3)
    
    # 2. Flux Distribution by Position
    plt.subplot(1, 2, 2)
    plt.hexbin(data[:, 0], data[:, 3], gridsize=50, cmap='Blues', bins='log')
    plt.title("The Habit: Flux vs Position")
    plt.xlabel("Position")
    plt.ylabel("Flux (Memory Depth)")
    plt.axvline(-0.5, color='red', linestyle='--', label='Well Bottom')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('well_addiction_diagnosis.png')
    print("✅ Diagnosis complete. Saved 'well_addiction_diagnosis.png'")
    print(f"Stats: Max Flux = {np.max(data[:,3]):.2f} at Mean Pos = {np.mean(data[data[:,3]>5, 0]):.2f}")

if __name__ == "__main__":
    diagnose_well_addiction()
