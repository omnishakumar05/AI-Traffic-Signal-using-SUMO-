import os
import sys
import numpy as np
import traci
import sumolib
from gym import Env, spaces

class SumoTrafficEnv(Env):
    def __init__(self, net_file, route_file, add_file, use_gui=False, max_steps=1800):
        super(SumoTrafficEnv, self).__init__()
        self.net_file = net_file
        self.route_file = route_file
        self.add_file = add_file
        self.use_gui = use_gui
        self.max_steps = max_steps
        self.current_step = 0
        self.tl_id = None  # Will be detected automatically
        self.min_green_time = 7
        self.max_green_time = 60
        self.extension_time = 3
        self.yellow_time = 4
        self.current_phase = 0
        self.current_phase_duration = 0
        self.phases = [0, 2]
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(low=0, high=1, shape=(10,), dtype=np.float32)
        self.total_waiting_time = 0
        self.collision_count = 0
        self.lanes = None  # Will be detected

    def _start_simulation(self):
        sumo_binary = sumolib.checkBinary('sumo-gui' if self.use_gui else 'sumo')
        sumo_cmd = [sumo_binary, "-n", self.net_file, "-r", self.route_file, "-a", self.add_file,
                    "--step-length", "1", "--collision.action", "warn", "--time-to-teleport", "-1",
                    "--no-warnings", "true", "--no-step-log", "true"]
        if self.use_gui:
         sumo_cmd.extend(["--start", "--delay", "100"])  # Remove quit-on-end, add delay
        traci.start(sumo_cmd)
        
        # Auto-detect traffic light ID
        tl_ids = traci.trafficlight.getIDList()
        if len(tl_ids) > 0:
            self.tl_id = tl_ids[0]  # Use first traffic light
            print(f"[INFO] Using traffic light: {self.tl_id}")
        else:
            raise Exception("No traffic lights found in network!")
        
        # Auto-detect lanes
        self.lanes = traci.lane.getIDList()
        # Filter out internal lanes (those starting with ':')
        self.lanes = [l for l in self.lanes if not l.startswith(':')]
        print(f"[INFO] Found {len(self.lanes)} lanes")

    def reset(self):
        if traci.isLoaded():
            traci.close()
        self._start_simulation()
        self.current_step = 0
        self.current_phase = 0
        self.current_phase_duration = 0
        self.total_waiting_time = 0
        self.collision_count = 0
        if self.tl_id:
            traci.trafficlight.setPhase(self.tl_id, self.phases[self.current_phase])
        return self._get_observation()

    def _get_observation(self):
        # Use first 8 lanes for observation
        queue_lengths = []
        for i, lane in enumerate(self.lanes[:8]):
            try:
                halting = traci.lane.getLastStepHaltingNumber(lane)
                queue_lengths.append(min(halting / 10.0, 1.0))
            except:
                queue_lengths.append(0.0)
        
        # Pad if we have fewer than 8 lanes
        while len(queue_lengths) < 8:
            queue_lengths.append(0.0)
        
        phase_norm = self.current_phase / len(self.phases)
        time_norm = min(self.current_phase_duration / self.max_green_time, 1.0)
        return np.array(queue_lengths + [phase_norm, time_norm], dtype=np.float32)

    def _apply_vac_logic(self, action):
        if self.current_phase_duration < self.min_green_time:
            return False
        if self.current_phase_duration >= self.max_green_time:
            return True
        if action == 1:
            return self.current_phase_duration >= self.min_green_time
        return False

    def _get_active_detectors(self):
        return [True]  # Simplified

    def _switch_phase(self):
        if self.tl_id:
            traci.trafficlight.setPhase(self.tl_id, self.phases[self.current_phase] + 1)
        for _ in range(self.yellow_time):
            traci.simulationStep()
            self.current_step += 1
        self.current_phase = (self.current_phase + 1) % len(self.phases)
        if self.tl_id:
            traci.trafficlight.setPhase(self.tl_id, self.phases[self.current_phase])
        self.current_phase_duration = 0

    def _calculate_reward(self):
        waiting_time = 0
        queue_length = 0
        for veh_id in traci.vehicle.getIDList():
            waiting_time += traci.vehicle.getWaitingTime(veh_id)
            speed = traci.vehicle.getSpeed(veh_id)
            if speed < 0.1:
                queue_length += 1
        collision_penalty = 0
        collisions = traci.simulation.getCollidingVehiclesNumber()
        if collisions > 0:
            collision_penalty = collisions * 100
            self.collision_count += collisions
        reward = -(waiting_time * 0.1 + queue_length * 1.0 + collision_penalty)
        self.total_waiting_time += waiting_time
        return reward

    def step(self, action):
        should_switch = self._apply_vac_logic(action)
        if should_switch:
            self._switch_phase()
        else:
            traci.simulationStep()
            self.current_step += 1
            self.current_phase_duration += 1
        observation = self._get_observation()
        reward = self._calculate_reward()
        done = self.current_step >= self.max_steps
        info = {
            'waiting_time': self.total_waiting_time,
            'vehicles': len(traci.vehicle.getIDList()),
            'collisions': self.collision_count,
            'current_phase': self.current_phase,
            'phase_duration': self.current_phase_duration
        }
        return observation, reward, done, info

    def close(self):
        if traci.isLoaded():
            traci.close()

    def render(self, mode='human'):
        pass
