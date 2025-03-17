import gymnasium as gym
from gymnasium import spaces
import numpy as np
import cv2
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
        self.action_space = spaces.Box(
            low=np.array([MIN_STEER_ANGLE, MIN_SPEED]), 
            high=np.array([MAX_STEER_ANGLE, MAX_SPEED]),
            dtype=np.float32
        )
        
        # state space 
        self.state_space = spaces.Dict({
            "speed": spaces.Box(low=0, high=MAX_SPEED, shape=(1,), dtype=np.float32), # gps speed
            "gps": spaces.Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32), # (x, y) gps coordinates
            "lidar_dist": spaces.Box(low=0, high=100, shape=(1,), dtype=np.float32), # distance to nearest obstacle
            "lidar_angle": spaces.Box(low=-180, high=180, shape=(1,), dtype=np.float32),  # angle to nearest obstacle
            "lane_deviation": spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.float32), # the pixels away from center of the lane
            "lane_mask": spaces.Box(low=0, high=1, shape=(64, 128, 1), dtype=np.uint8) # binaray mask for lane line (yellow line only)
        })
        
        # initialize the camera
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

        state = self._get_state()
        reward = self._compute_reward()
        done = self._is_done()

        return state, reward, done
    
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.agent.setSteeringAngle(0)
        self.agent.setCruisingSpeed(0)
        self.reset_flag.getField("translation").setSFVec3f([1, 0, 0]) # send out flag to dummy node
        
        self.agent.step() 

        state = self._get_state()

        return state
    
    
    def render(self, mode="human"):
        pass 
    
    
    def _get_state(self):
        speed = self.gps_speed
        position = self.gps.getValues() if self.gps else [0, 0, 0]
        lidar_data = self.lidar.getRangeImage() if self.lidar else [0]
        lidar_data = np.nan_to_num(lidar_data, nan=0.0, posinf=100.0, neginf=0.0)
        
        if lidar_data is None or len(lidar_data) == 0:
            lidar_data = [0]
            
        frame = self._process_image()
        # debugging
        # cv2.imsthow("Frame", frame)
        if frame is not None:
            edges = self._create_lane_mask(frame)
            # cv2.imshow("Edges", edges)
        else:
            edges = np.zeros((64, 128), dtype=np.float32)  # give a default blank image if no impage available

        # debugging
        # cv2.waitKey(0)  # press 0 to close windows
        # cv2.destroyAllWindows()
        
        # prevent NaN values
        if any(np.isnan(position)) or np.isnan(speed) or np.isnan(np.mean(lidar_data)):
            raise ValueError(f"Invalid observation values: speed={speed}, position={position}, lidar={lidar_data}")

        lane_deviation = self._calc_lane_penalty(k=1)
        
        lidar_angle = 0 # TODO: ADD LIDAR ANGLE FUNCTION
        
        # return the updataed values
        return {
            "speed": np.array([speed], dtype=np.float32),
            "gps": np.array([position[0], position[1]], dtype=np.float32),
            "lidar_dist": np.array([np.min(lidar_data)], dtype=np.float32), 
            "lidar_angle": np.array([lidar_angle], dtype=np.float32),
            "lane_deviation": np.array([lane_deviation], dtype=np.float32),
            "lane_mask": np.expand_dims(edges, axis=-1).astype(np.uint8)
        }
    
    
    def _compute_reward(self):
        reward = 0.0
        
        # collision penalty
        if self._has_collided():
            reward -= 100
            
        lane_penalty = self._calc_lane_penalty(k=0.5)
        reward -= lane_penalty
        
        if self.gps_speed > MAX_SAFE_SPEED:
            reward -=50
        elif self.agent.getTime() > 10 or self.gps_speed >= CITY_SPEED_LIMIT: # give car 10 seconds to get up to city speed limits before penalizing
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
            
    
    def _calc_lane_penalty(self, k=1.0):
        frame = self._process_image()
        edges = self._create_lane_mask(frame)
        left_lane = self._sliding_window_detect_lanes(edges)
        
        # avg of x pos of lane points closest to the car
        # x_right = max(right_lane, key=lambda pt: pt[1])[0]
        x_left = max(left_lane, key=lambda pt: pt[1])[0]
            
        if x_left == 0:
            return -80
        
    
        x_vehicle = self.camera.getWidth() // 2
        lane_center = (x_left + 48 )
        deviation = abs(x_vehicle - lane_center)
        
        penalty = k * (deviation)
        
        return penalty
        
      
    def _process_image(self):
        width = self.camera.getWidth()
        height = self.camera.getHeight()
        raw_image = self.camera.getImage()
        
        img = np.frombuffer(raw_image, np.uint8).reshape((height, width, 4))
        img_bgr = img[:, :, :3] # remove alpha channel and only have BGR channels
        
        return img_bgr
    
    
    def _create_lane_mask(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) # hue, saturation, value color filtering

        # yellow line detection
        lower_bound_yellow = np.array([18, 94, 140], dtype=np.uint8)
        upper_bound_yellow = np.array([48, 255, 255], dtype=np.uint8)
        yellow_mask = cv2.inRange(hsv, lower_bound_yellow, upper_bound_yellow)
        
        # white line detection
        # lower_bound_white = np.array([0, 0, 150], dtype=np.uint8)
        # upper_bound_white = np.array([180, 35, 255], dtype=np.uint8)
        # white_mask = cv2.inRange(hsv, lower_bound_white, upper_bound_white)
        
        # combined_mask = cv2.bitwise_or(yellow_mask, white_mask)

        # strengthen dashed lines
        # kernel = np.ones((3, 3), np.uint8)
        # combined_mask = cv2.dilate(combined_mask, kernel, iterations=2)
        
        height, width = frame.shape[:2]
        mask_roi = np.zeros_like(yellow_mask)
        
        # focus on bottom half of the frame
        roi_vertices = np.array([[
            (0, height), (width, height), (width, height//2), (0, height//2)
        ]], dtype=np.int32)

        cv2.fillPoly(mask_roi, roi_vertices, 255)
        combined_mask = cv2.bitwise_and(yellow_mask, mask_roi)  # apply region mask

        edges = cv2.Canny(combined_mask, 80, 150) 
        return edges
        
    
    # https://ieeexplore.ieee.org/document/9208278
    # https://www.youtube.com/watch?v=ApYo6tXcjjQ
    def _sliding_window_detect_lanes(self, edges):
        height, width = edges.shape
        
        # only get histogram for lower half of image where the lanes are
        histogram = np.sum(edges[height//2:, :], axis=0)
        
        midpoint = width // 2
        left_x_base = np.argmax(histogram[:midpoint])  # yellow lane line
        # right_x_base = np.argmax(histogram[midpoint:]) + midpoint # white dotted lane line
        
        # sliding window parameters
        num_windows = 10
        window_height = height // num_windows
        margin = 60
        min_pixels = 30
        
        left_x_current = left_x_base
        # right_x_current = right_x_base
        left_lane_pts = []
        # right_lane_pts = []
        
        for window in range(num_windows):
            # sliding window boundaries
            y_low = height - (window + 1) * window_height
            y_high = height - window * window_height

            x_left_low = int(left_x_current - margin)
            x_left_high = int(left_x_current + margin)

            # x_right_low = int(right_x_current - margin)
            # x_right_high = int(right_x_current + margin)
            
            # get lane pixels in each widnow
            left_lane_indices = np.where((edges[y_low:y_high, x_left_low:x_left_high] > 0))
            # right_lane_indices = np.where((edges[y_low:y_high, x_right_low:x_right_high] > 0))
            
            # yellow solid line
            if len(left_lane_indices[0]) > min_pixels:
                left_x_current = np.mean(left_lane_indices[1]) + x_left_low
                
            # white dotted line: check for lane pixels, if currently in gap use the previous window
            # if len(right_lane_indices[0]) > min_pixels:
            #     right_x_current = np.mean(right_lane_indices[1]) + x_right_low
            # elif len(right_lane_pts) > 0:
            #     right_x_current = right_lane_pts[-1][0]
                
            left_lane_pts.append((left_x_current, (y_low + y_high) // 2)) # yellow lane line
            # right_lane_pts.append((right_x_current, (y_low + y_high) // 2)) # white dotted lane line
            
        return left_lane_pts#, right_lane_pts
             
    # TODO: Interpret LIDAR