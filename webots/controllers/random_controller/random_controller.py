import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
from webots_env import WebotsCarEnv
import random
import numpy as np
        
env = WebotsCarEnv()
obs = env.reset()

for step in range(1000):
    steering_angle = random.uniform(-0.9, 0.9) 
    speed = random.uniform(10, 20)

    action = np.array([steering_angle, speed], dtype=np.float32)

    obs, reward, done, _ = env.step(action)

    print(f"Step {step + 1}: Steering = {action[0]:.2f}, Speed = {action[1]:.2f}, Reward = {reward:.2f}")
    
    if done:
        break

print("Episode finished. Resetting environment.")
obs = env.reset()


