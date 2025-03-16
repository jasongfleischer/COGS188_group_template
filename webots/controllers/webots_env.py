import gymnasium as gym
from gymnasium import spaces
import numpy as np
from vehicle import Driver

MAX_STEER_ANGLE = 0.5
MIN_STEER_ANGLE = -0.5
MAX_SPEED = 250.0   # ~ 155 mph
MIN_SPEED = 0.0

MAX_SAFE_SPEED = 112.65 # ~ 70 mph
CITY_SPEED_LIMIT = 72.42  # ~ 45 mph

MAX_SIM_TIME = 120 # 2 min max sim time

GOAL_COORDS = [-36.75, 59.5]
GOAL_THRESHOLD = [1.75, 0.5] # how close to goal to count as reached (in meters)


class WebotsCarEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 20}
    
    def __init__(self):
        super(WebotsCarEnv, self).__init__()
        
        self.agent = Driver()
        self.time_step = int(self.agent.getBasicTimeStep())
        
        # action space: [steering, speed]
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        
        # TODO: Update observation space to include camera output & other stuff
        # obs space: [speed, position[0], position[1], np.mean(lidar_data)]
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32)
        
        self.camera = self.agent.getDevice("camera")
        self.camera.enable(self.time_step)

        self.gps = self.agent.getDevice("gps")
        self.gps.enable(self.time_step)

        self.lidar = self.agent.getDevice("lidar")
        self.lidar.enable(self.time_step)
        
        self.prev_gps_speed = 0.0
        self.gps_speed = 0.0
        self.gps_coords = [0.0, 0.0, 0.0]
        
        self.reset_flag = self.agent.getFromDef("RESET_FLAG")
                        
                
    def step(self, action):
        steering_angle = action[0]
        speed = np.clip(action[1], MIN_SPEED, MAX_SPEED)
        
        self._set_steering_angle(steering_angle)
        self.agent.setCruisingSpeed(speed)
        self._calc_gps_speed()

        self.agent.step()

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
        
        self.agent.setSteeringAngle(0)
        self.agent.setCruisingSpeed(0)
        self.reset_flag.getField("translation").setSFVec3f([1, 0, 0]) # send out flag to dummy node
        
        self.agent.step() 

        obs = self._get_observation()
        if np.isnan(obs).any():
            raise ValueError(f"NaN detected in reset observation: {obs}")

        return obs, {}
    
    
    def render(self, mode="human"):
        pass 
    
    
    def _get_observation(self):
        speed = self.agent.getCurrentSpeed()
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
        reward = 0.0
        
        # collision penalty
        if self._has_collided():
            reward -= 100
            
        # TODO: add lane deviation penalty
        
        if self.gps_speed > MAX_SAFE_SPEED:
            reward -=50
        else:
            speed_penalty = max(0, abs(self.gps_speed - CITY_SPEED_LIMIT) - 8) # penalize going more than +/- 8kph (5mph)
            reward -= speed_penalty
            
        if self.gps_speed > 0:
            reward += 1  # reward for making progress
            
        if self._has_reached_goal():
            reward += 100
            
        return reward
    
    
    def _is_done(self):
        if self._has_collided():
            print("Collision detected.")
            return True

        if self._has_reached_goal():
            print("Goal reached.")
            return True

        current_time = self.agent.getTime()
        if current_time >= MAX_SIM_TIME:
            print("Time limit reached.")
            return True
    
    
    def _has_collided(self):
        # BUG: sometimes lidar detects collision when there are no obstacles present
        # LIDAR collision detection
        lidar_distances = np.array(self.lidar.getRangeImage())
        min_distance = np.min(lidar_distances)
        if min_distance < 1.0:
            print(f"lidar dist: {min_distance}")
            return True
                
        # rapid speed change collision detection
        speed_change = abs(self.prev_gps_speed - self.gps_speed)
        if self.prev_gps_speed > 2.0 and self.gps_speed < 0.5 and speed_change > 2.0:
            print(f"speed_change: {speed_change}")
            return True 
        
        return False
    
    
    def _has_reached_goal(self):
        current_pos = self.gps_coords
        x_dist = abs(current_pos[0] - GOAL_COORDS[0]) < GOAL_THRESHOLD[0]
        y_dist = abs(current_pos[1] - GOAL_COORDS[1]) < GOAL_THRESHOLD[1]
    
        return x_dist and y_dist
        
        
    def _set_steering_angle(self, wheel_angle):
        steering_angle = self.agent.getSteeringAngle()
        if (wheel_angle - steering_angle > 0.1):
            wheel_angle = steering_angle + 0.1
        if (wheel_angle - steering_angle < -0.1):
            wheel_angle = steering_angle - 0.1
        steering_angle = wheel_angle
        
        if (wheel_angle > MAX_STEER_ANGLE):
            wheel_angle = MAX_STEER_ANGLE
        elif (wheel_angle < MIN_STEER_ANGLE):
            wheel_angle = MIN_STEER_ANGLE
            
        self.agent.setSteeringAngle(wheel_angle)
        
        return
        
        
    def _calc_gps_speed(self):
        coords = self.gps.getValues()
        speed_ms = self.gps.getSpeed() * 3.6
        
        if coords is not None:
            self.prev_gps_speed = self.gps_speed 
            self.gps_speed = speed_ms 
            self.gps_coords = list(coords)
            
            
    # TODO: Interpret LIDAR
    # TODO: Interpret Camera
    # TODO: Interpret GPS
