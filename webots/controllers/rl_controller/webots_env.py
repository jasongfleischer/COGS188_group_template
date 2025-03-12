import gymnasium as gym
from gymnasium import spaces
import numpy as np
from vehicle import Car

class WebotsCarEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 20}
    
    max_steer_angle = 1.0
    min_steer_angle = -1.0
    
    def __init__(self):
        super(WebotsCarEnv, self).__init__()
        
        self.robot = Car()
        self.time_step = int(self.robot.getBasicTimeStep())
        
        # action space: [steering, speed]
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        
        # TODO: Update observation space to include camera output & other stuff
        # obs space: [speed, position[0], position[1], np.mean(lidar_data)]
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32)
        
        self.camera = self.robot.getDevice("camera")
        if self.camera:
            self.camera.enable(self.time_step)

        self.gps = self.robot.getDevice("gps")
        if self.gps:
            self.gps.enable(self.time_step)

        self.lidar = self.robot.getDevice("lidar")
        if self.lidar:
            self.lidar.enable(self.time_step)
                
    def step(self, action):
        steering_angle = np.clip(action[0], self.min_steer_angle, self.max_steer_angle)
        speed = action[1]
        
        self.robot.setSteeringAngle(steering_angle)
        self.robot.setCruisingSpeed(speed)

        self.robot.step()

        observation = self._get_observation()
        reward = self._compute_reward()
        done = self._is_done()
        
        if np.isnan(observation).any():
            raise ValueError("Observation contains NaN values")
        if np.isnan(reward):
            raise ValueError("Reward contains NaN values")

        return observation, reward, done, {}
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.robot.setSteeringAngle(0)
        self.robot.setCruisingSpeed(0)
        self.robot.step() 

        obs = self._get_observation()
        if np.isnan(obs).any():
            raise ValueError(f"NaN detected in reset observation: {obs}")

        return obs, {}
    
    def render(self, mode="human"):
        pass 
    
    def _get_observation(self):
        speed = self.robot.getCurrentSpeed()
        position = self.gps.getValues() if self.gps else [0, 0, 0]
        lidar_data = self.lidar.getRangeImage() if self.lidar else [0]
        lidar_data = np.nan_to_num(lidar_data, nan=0.0, posinf=100.0, neginf=0.0)
        
        if lidar_data is None or len(lidar_data) == 0:
            lidar_data = [0]
            
        if speed is None or np.isnan(speed) or np.isinf(speed):
            speed = 0

        if any(np.isnan(position)) or np.isnan(speed) or np.isnan(np.mean(lidar_data)):
            raise ValueError(f"Invalid observation values: speed={speed}, position={position}, lidar={lidar_data}")

        return np.array([speed, position[0], position[1], np.mean(lidar_data)], dtype=np.float32)
        
    # TODO: Implement actual rewards function
    def _compute_reward(self):
        speed = self.robot.getCurrentSpeed()
        if speed is None or np.isnan(speed) or np.isinf(speed):
            return 0 
        return np.clip(speed / 250, -10, 10)
    
    def _is_done(self):
        return False
    
    # TODO: Interpret LIDAR
    # TODO: Interpret Camera
    # TODO: Interpret GPS
