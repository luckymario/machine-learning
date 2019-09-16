import numpy as np
from physics_sim import PhysicsSim

class MyTask():
    """Task (environment) that defines the goal and provides feedback to the agent."""
    def __init__(self, init_pose=None, init_velocities=None, 
        init_angle_velocities=None, runtime=5., target_pos=None):
        """Initialize a Task object.
        Params
        ======
            init_pose: initial position of the quadcopter in (x,y,z) dimensions and the Euler angles
            init_velocities: initial velocity of the quadcopter in (x,y,z) dimensions
            init_angle_velocities: initial radians/second for each of the three Euler angles
            runtime: time limit for each episode
            target_pos: target/goal (x,y,z) position for the agent
        """
        # Simulation
        self.sim = PhysicsSim(init_pose, init_velocities, init_angle_velocities, runtime) 
        self.action_repeat = 3

        self.state_size = self.action_repeat * 6
        self.action_low = 0
        self.action_high = 900
        self.action_size = 4

        # Goal
        self.target_pos = target_pos if target_pos is not None else np.array([0., 0., 10.]) 

    def get_reward(self):
        """Uses current pose of sim to return reward."""
        #reward = 1.-.3*(abs(self.sim.pose[:3] - self.target_pos)).sum()

        reward = 0
        current_position = self.sim.pose[:3]
        penalty_factor = 0.0003
        reward_factor = 100
        
        # penalty for euler angles
        reward -= abs(self.sim.pose[3:6]).sum() * penalty_factor

        # additional reward for close distance
        distance = abs(current_position[0] - self.target_pos[0]) + abs(current_position[1] - self.target_pos[1]) + abs(current_position[2] - self.target_pos[2])
        if (distance < 20):
            reward += (20 - distance) * reward_factor
            
        x_distance = abs(current_position[0] - self.target_pos[0])
        if (x_distance < 3):
            reward += (3 - x_distance) * reward_factor
            
        y_distance = abs(current_position[1] - self.target_pos[1])
        if (y_distance < 3):
            reward += (3 - y_distance) * reward_factor
            
        z_distance = abs(current_position[2] - self.target_pos[2])
        if (z_distance < 10):
            reward += (10 - z_distance) * reward_factor
        
        # additional reward for velocity close to [0,0,20]
        x_velocity = abs(self.sim.v[2])
        if (x_velocity < 2):
            reward += (2 - x_velocity) * reward_factor
            
        y_velocity = abs(self.sim.v[2])
        if (y_velocity < 2):
            reward += (2 - y_velocity) * reward_factor    
            
        z_velocity = abs(self.sim.v[2] - 20)
        if (z_velocity < 15):
            reward += (15 - z_velocity) * reward_factor
         
        # review: reward for reaching the target height
        done = False
        if (self.sim.pose[2] >= self.target_pos[2]):
            reward += 50.0
            done = True
        
        # review: penalize the agent when the agent crashes
        if (done and self.sim.time < self.sim.runtime): 
            reward = -10

        return reward

    def step(self, rotor_speeds):
        """Uses action to obtain next state, reward, done."""
        reward = 0
        pose_all = []
        for _ in range(self.action_repeat):
            done = self.sim.next_timestep(rotor_speeds) # update the sim pose and velocities
            reward += self.get_reward() 
            pose_all.append(self.sim.pose)
        next_state = np.concatenate(pose_all)
        return next_state, reward, done

    def reset(self):
        """Reset the sim to start a new episode."""
        self.sim.reset()
        state = np.concatenate([self.sim.pose] * self.action_repeat) 
        return state