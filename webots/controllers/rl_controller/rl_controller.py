from webots_env import WebotsCarEnv
import random

class SimpleRLController:
    def __init__(self, env):
        self.env = env
        self.steering_angle = 0.0  # no steering
        self.speed = 2.0  # slow speed

    def get_action(self, state):
        """
        Very basic RL: Adjust the steering and speed randomly.
        """
        # small random adjustments to steering and speed
        self.steering_angle += random.uniform(-0.1, 0.1)
        self.steering_angle = max(-1, min(self.steering_angle, 1))  

        self.speed += random.uniform(-0.2, 0.2)
        self.speed = max(0, min(self.speed, 5))  

        return [self.steering_angle, self.speed]

    def run(self, steps=50):
        """Runs the controller for a given number of steps in Webots."""
        obs = self.env.reset()
        
        for _ in range(steps):
            action = self.get_action(obs) 
            obs, reward, done, info = self.env.step(action)  
            print(f"Steering: {action[0]:.2f}, Speed: {action[1]:.2f}, Reward: {reward:.2f}")
            
            if done:
                print("Episode finished. Resetting environment.")
                obs = self.env.reset()

env = WebotsCarEnv()
controller = SimpleRLController(env)

controller.run(steps=50)
