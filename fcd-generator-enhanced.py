#!/usr/bin/env python3
"""
Comprehensive FCD Data Generator with Flexible Configuration and Node Positioning
=================================================================================

Generates realistic FCD files with:
- Fixed vehicle count OR variable density modes
- Custom speed ranges
- Flexible vehicle type distribution
- Lane distribution control
- Multiple traffic scenarios
- Realistic acceleration/deceleration
- Lane changing behavior
- **NEW: Specific Defined Node Positioning (Static & Dynamic)**

Perfect for reinforcement learning training with customizable scenarios.
"""

import numpy as np
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from typing import Dict, List, Tuple, Optional
import random
import math
import json
from datetime import datetime

class FlexibleTrafficConfig:
    """Flexible configuration for traffic scenarios"""
    
    @staticmethod
    def create_fixed_vehicle_config(
        vehicle_count: int = 45,
        speed_range_kmh: Tuple[float, float] = (18, 36),
        vehicle_types: Dict[str, float] = None,
        lane_distribution: str = 'balanced',  # 'balanced', 'random', 'heavy_right', 'heavy_left'
        spawn_pattern: str = 'continuous',    # 'continuous', 'burst', 'gradual'
        behavior_profile: str = 'normal',     # 'aggressive', 'normal', 'conservative'
        **kwargs
    ) -> Dict:
        """Create configuration for fixed vehicle count mode"""
        
        if vehicle_types is None:
            vehicle_types = {'car': 0.75, 'truck': 0.15, 'bus': 0.05, 'motorcycle': 0.05}
        
        return {
            'mode': 'fixed_count',
            'vehicle_count': vehicle_count,
            'speed_range_kmh': speed_range_kmh,
            'speed_range_ms': (speed_range_kmh[0] / 3.6, speed_range_kmh[1] / 3.6),
            'vehicle_types': vehicle_types,
            'lane_distribution': lane_distribution,
            'spawn_pattern': spawn_pattern,
            'behavior_profile': behavior_profile,
            'maintain_count': kwargs.get('maintain_count', True),
            'spawn_rate': kwargs.get('spawn_rate', 3),  # vehicles per timestep when spawning
            'speed_variation': kwargs.get('speed_variation', 0.2),  # ±20% speed variation
            **kwargs
        }
    
    @staticmethod
    def create_density_config(
        scenario: str = 'learning_optimized',
        density_factor: float = None,
        speed_adaptation: bool = True,
        **kwargs
    ) -> Dict:
        """Create configuration for density-based mode (original behavior)"""
        return {
            'mode': 'density_based',
            'scenario': scenario,
            'density_factor': density_factor,
            'speed_adaptation': speed_adaptation,
            **kwargs
        }
    
    @staticmethod
    def create_specific_nodes_config(
        node_mode: str = 'static',  # 'static' or 'dynamic'
        node_positions: Tuple[float, float] = (100.0, 300.0),  # positions for static mode
        node_speed_kmh: float = 30.0,  # speed for dynamic mode
        lane_number: int = 1,  # which lane to use (1-based)
        duration: float = 300.0,  # simulation duration
        node_types: Tuple[str, str] = ('car', 'car'),  # types of the two nodes
        node_ids: Tuple[str, str] = ('node1', 'node2'),  # custom IDs for nodes
        **kwargs
    ) -> Dict:
        """
        Create configuration for specific defined node positioning
        
        Args:
            node_mode: 'static' (stationary nodes) or 'dynamic' (moving nodes)
            node_positions: (pos1, pos2) positions in meters for static mode
            node_speed_kmh: Speed in km/h for dynamic mode
            lane_number: Lane number (1-based) for node placement
            duration: Simulation duration in seconds
            node_types: Vehicle types for the two nodes
            node_ids: Custom IDs for the nodes
        """
        return {
            'mode': 'specific_defined_nodes',
            'node_mode': node_mode,
            'node_positions': node_positions,
            'node_speed_kmh': node_speed_kmh,
            'node_speed_ms': node_speed_kmh / 3.6,
            'lane_number': lane_number,
            'duration': duration,
            'node_types': node_types,
            'node_ids': node_ids,
            'road_length': kwargs.get('road_length', 500.0),
            'enable_background_traffic': kwargs.get('enable_background_traffic', False),
            'background_vehicle_count': kwargs.get('background_vehicle_count', 10),
            **kwargs
        }

class TrafficScenario:
    """Defines different traffic scenarios with density patterns"""
    
    SCENARIOS = {
        'rush_hour': {
            'description': 'Morning/evening rush hour with high density periods',
            'base_density': 0.6,
            'peak_density': 1.0,
            'peak_periods': [(1800, 2700), (7200, 8100)],  # 30-45min and 2h-2h15min
            'speed_reduction': 0.4
        },
        'highway_congestion': {
            'description': 'Highway with periodic congestion waves',
            'base_density': 0.3,
            'peak_density': 0.9,
            'wave_frequency': 1800,  # Congestion every 30 minutes
            'wave_duration': 600,    # 10 minute congestion
            'speed_reduction': 0.6
        },
        'city_traffic': {
            'description': 'Urban traffic with traffic light cycles',
            'base_density': 0.5,
            'peak_density': 0.8,
            'cycle_frequency': 120,  # Traffic light cycle every 2 minutes
            'cycle_duration': 30,    # 30 second red light
            'speed_reduction': 0.3
        },
        'random_varying': {
            'description': 'Randomly varying density for diverse learning',
            'base_density': 0.4,
            'variation_amplitude': 0.4,
            'variation_frequency': 300,  # Change every 5 minutes
            'speed_reduction': 0.5
        },
        'learning_optimized': {
            'description': 'Optimized for RL with gradual density changes',
            'phases': [
                {'duration': 2000, 'density': 0.2, 'description': 'Light traffic'},
                {'duration': 2000, 'density': 0.5, 'description': 'Moderate traffic'},
                {'duration': 2000, 'density': 0.8, 'description': 'Heavy traffic'},
                {'duration': 2000, 'density': 0.95, 'description': 'Congestion'},
                {'duration': 2000, 'density': 0.6, 'description': 'Recovery'},
            ]
        }
    }

class SpecificNode:
    """Represents a specific node with defined positioning and behavior"""
    
    def __init__(self, node_id: str, node_type: str, lane: int, config: Dict):
        self.id = node_id
        self.vehicle_type = node_type
        self.lane = lane
        self.y = (lane - 1) * 3.7  # Lane width = 3.7m
        self.config = config
        self.active = True
        self.length = self._get_vehicle_length()
        
        # Initialize based on mode
        if config['node_mode'] == 'static':
            self._init_static_mode()
        elif config['node_mode'] == 'dynamic':
            self._init_dynamic_mode()
    
    def _get_vehicle_length(self) -> float:
        """Get vehicle length based on type"""
        lengths = {
            'car': 4.5, 
            'truck': 16.0,
            'bus': 12.0, 
            'motorcycle': 2.2
        }
        return lengths.get(self.vehicle_type, 4.5)
    
    def _init_static_mode(self):
        """Initialize for static mode"""
        positions = self.config['node_positions']
        if self.id == self.config['node_ids'][0]:
            self.x = positions[0]
        else:
            self.x = positions[1]
        
        self.speed = 0.0
        self.target_speed = 0.0
        self.direction = 'static'
        self.phase = 'stationary'
    
    def _init_dynamic_mode(self):
        """Initialize for dynamic mode"""
        road_length = self.config['road_length']
        self.speed = self.config['node_speed_ms']
        self.target_speed = self.speed
        
        if self.id == self.config['node_ids'][0]:
            # Node 1 starts from left, moves right
            self.x = 0.0
            self.direction = 'right'
            self.start_position = 0.0
            self.end_position = road_length
        else:
            # Node 2 starts from right, moves left
            self.x = road_length
            self.direction = 'left'
            self.start_position = road_length
            self.end_position = 0.0
        
        self.phase = 'moving_to_center'
        self.meeting_point = road_length / 2.0
        self.has_met = False
        self.return_started = False
    
    def update_position(self, dt: float, current_time: float, other_node: Optional['SpecificNode'] = None):
        """Update node position based on mode and phase"""
        if not self.active:
            return
        
        if self.config['node_mode'] == 'static':
            self._update_static_position(dt, current_time)
        elif self.config['node_mode'] == 'dynamic':
            self._update_dynamic_position(dt, current_time, other_node)
    
    def _update_static_position(self, dt: float, current_time: float):
        """Update static node (no movement)"""
        # Static nodes don't move, but we can add some effects here if needed
        self.speed = 0.0
        
        # Check if simulation duration is exceeded
        if current_time >= self.config['duration']:
            self.active = False
    
    def _update_dynamic_position(self, dt: float, current_time: float, other_node: Optional['SpecificNode']):
        """Update dynamic node with complex movement pattern"""
        if self.phase == 'moving_to_center':
            # Move towards center
            if self.direction == 'right':
                self.x += self.speed * dt
                if self.x >= self.meeting_point:
                    self.x = self.meeting_point
                    self.phase = 'at_center'
            else:  # direction == 'left'
                self.x -= self.speed * dt
                if self.x <= self.meeting_point:
                    self.x = self.meeting_point
                    self.phase = 'at_center'
        
        elif self.phase == 'at_center':
            # Wait briefly at center, then start return journey
            if other_node and other_node.phase == 'at_center':
                self.has_met = True
                self.phase = 'returning'
                # Reverse direction
                if self.direction == 'right':
                    self.direction = 'left'
                    self.target_position = self.start_position
                else:
                    self.direction = 'right'
                    self.target_position = self.start_position
        
        elif self.phase == 'returning':
            # Return to starting position
            if self.direction == 'right':
                self.x += self.speed * dt
                if self.x >= self.target_position:
                    self.x = self.target_position
                    self.phase = 'completed'
            else:  # direction == 'left'
                self.x -= self.speed * dt
                if self.x <= self.target_position:
                    self.x = self.target_position
                    self.phase = 'completed'
        
        elif self.phase == 'completed':
            # Journey completed
            self.speed = 0.0
            if other_node and other_node.phase == 'completed':
                # Both nodes completed, can end simulation
                pass
    
    def get_angle(self) -> float:
        """Get vehicle angle based on direction"""
        if self.config['node_mode'] == 'static':
            return 90.0
        else:
            if self.direction == 'right':
                return 90.0
            elif self.direction == 'left':
                return 270.0
            else:
                return 90.0
    
    def get_lane_id(self) -> str:
        """Get SUMO-style lane ID"""
        lane_index = (self.lane - 1) % 3
        return f"E0_{lane_index}"
    
    def is_journey_complete(self) -> bool:
        """Check if the node has completed its journey"""
        if self.config['node_mode'] == 'static':
            return False  # Static nodes don't complete journeys
        else:
            return self.phase == 'completed'

class Vehicle:
    """Represents a single vehicle with realistic behavior"""
    
    def __init__(self, vehicle_id: str, lane: int, initial_x: float, initial_speed: float, 
                 direction: str = 'forward', vehicle_type: str = 'car', behavior_profile: str = 'normal'):
        self.id = vehicle_id
        self.lane = lane
        self.x = initial_x
        self.y = lane * 3.7  # Lane width = 3.7m
        self.speed = initial_speed
        self.target_speed = initial_speed
        self.direction = direction  # 'forward' or 'backward'
        self.vehicle_type = vehicle_type
        self.behavior_profile = behavior_profile
        self.length = self._get_vehicle_length()
        self.max_acceleration, self.max_deceleration = self._get_acceleration_limits()
        self.active = True
        self.lane_change_cooldown = 0
        self.following_distance = self._get_following_distance()
        
    def _get_vehicle_length(self) -> float:
        """Get REALISTIC vehicle length based on type"""
        lengths = {
            'car': 4.5, 
            'truck': 16.0,      # Increased for realism (truck + trailer)
            'bus': 12.0, 
            'motorcycle': 2.2
        }
        return lengths.get(self.vehicle_type, 4.5)
    
    def _get_acceleration_limits(self) -> Tuple[float, float]:
        """Get acceleration limits based on vehicle type and behavior profile"""
        base_limits = {
            'car': (2.5, 4.5),
            'truck': (1.5, 3.5),
            'bus': (1.8, 4.0),
            'motorcycle': (3.0, 5.0)
        }
        
        base_accel, base_decel = base_limits.get(self.vehicle_type, (2.5, 4.5))
        
        # Adjust based on behavior profile
        if self.behavior_profile == 'aggressive':
            return base_accel * 1.3, base_decel * 1.2
        elif self.behavior_profile == 'conservative':
            return base_accel * 0.8, base_decel * 0.9
        else:  # normal
            return base_accel, base_decel
    
    def _get_following_distance(self) -> float:
        """Get following distance based on behavior profile"""
        base_distance = 20.0
        if self.behavior_profile == 'aggressive':
            return base_distance * 0.7
        elif self.behavior_profile == 'conservative':
            return base_distance * 1.4
        else:
            return base_distance
    
    def update_position(self, dt: float, traffic_density: float, nearby_vehicles: List, 
                       custom_speed_range: Tuple[float, float] = None):
        """Update vehicle position with FLEXIBLE speed control"""
        if not self.active:
            return
        
        # Use custom speed range if provided, otherwise use density-based speeds
        if custom_speed_range:
            min_speed, max_speed = custom_speed_range
            # Apply some variation based on traffic density
            density_factor = min(1.0, traffic_density)
            speed_reduction = 0.3 * density_factor
            self.target_speed = max_speed * (1.0 - speed_reduction)
            # Ensure speed stays within range
            self.target_speed = max(min_speed, min(max_speed, self.target_speed))
        else:
            # Original density-based speed calculation
            self._calculate_density_based_speed(traffic_density)
        
        # Vehicle type speed characteristics
        if self.vehicle_type == 'truck':
            self.target_speed *= 0.9    # Trucks 10% slower
        elif self.vehicle_type == 'bus':
            self.target_speed *= 0.85   # Buses 15% slower  
        elif self.vehicle_type == 'motorcycle':
            self.target_speed *= 1.1    # Motorcycles 10% faster
        
        # REALISTIC following distance based on speed
        self.following_distance = max(self._get_following_distance(), self.speed * 1.5)
        
        # Car following behavior (filter out SpecificNode objects)
        regular_vehicles = [v for v in nearby_vehicles if isinstance(v, Vehicle)]
        leader = self._find_leader(regular_vehicles)
        if leader:
            gap = self._calculate_gap(leader)
            safe_speed = self._calculate_safe_speed(gap, leader.speed)
            self.target_speed = min(self.target_speed, safe_speed)
        
        # Smooth acceleration/deceleration with realistic limits
        speed_diff = self.target_speed - self.speed
        if speed_diff > 0:
            acceleration = min(self.max_acceleration, speed_diff / dt)
        else:
            acceleration = max(-self.max_deceleration, speed_diff / dt)
        
        self.speed = max(0, self.speed + acceleration * dt)
        
        # Update position
        if self.direction == 'forward':
            self.x += self.speed * dt
        else:
            self.x -= self.speed * dt
            
        # Handle lane change cooldown
        if self.lane_change_cooldown > 0:
            self.lane_change_cooldown -= dt
    
    def _calculate_density_based_speed(self, traffic_density: float):
        """Original density-based speed calculation"""
        if traffic_density <= 0.2:
            base_speed = 3.20 if self.direction == 'forward' else 3.0
        elif traffic_density <= 0.4:
            base_speed = 2.80 if self.direction == 'forward' else 2.60
        elif traffic_density <= 0.6:
            base_speed = 2.20 if self.direction == 'forward' else 2.0
        elif traffic_density <= 0.8:
            base_speed = 1.40 if self.direction == 'forward' else 1.20
        else:
            base_speed = 0.6 if self.direction == 'forward' else 0.5
        
        density_factor = min(1.0, traffic_density)
        speed_reduction = 0.3 * density_factor
        self.target_speed = base_speed * (1.0 - speed_reduction)
        
        if traffic_density > 0.9:
            self.target_speed = max(0.4, self.target_speed)  # Minimum crawling speed
    
    def _find_leader(self, nearby_vehicles: List['Vehicle']) -> Optional['Vehicle']:
        """Find the leading vehicle in the same lane"""
        same_lane_vehicles = [v for v in nearby_vehicles 
                             if v.lane == self.lane and v.id != self.id and v.active]
        
        if self.direction == 'forward':
            ahead_vehicles = [v for v in same_lane_vehicles if v.x > self.x]
            return min(ahead_vehicles, key=lambda v: v.x) if ahead_vehicles else None
        else:
            ahead_vehicles = [v for v in same_lane_vehicles if v.x < self.x]
            return max(ahead_vehicles, key=lambda v: v.x) if ahead_vehicles else None
    
    def _calculate_gap(self, leader: 'Vehicle') -> float:
        """Calculate gap to leading vehicle"""
        if self.direction == 'forward':
            return leader.x - self.x - leader.length
        else:
            return self.x - leader.x - self.length
    
    def _calculate_safe_speed(self, gap: float, leader_speed: float) -> float:
        """Calculate safe speed based on gap and leader speed"""
        min_gap = 5.0  # Minimum safe gap
        desired_gap = self.following_distance
        
        if gap < min_gap:
            return 0.0
        elif gap < desired_gap:
            gap_factor = gap / desired_gap
            return leader_speed * gap_factor * 0.8
        else:
            return leader_speed * 1.1
    
    def attempt_lane_change(self, nearby_vehicles: List['Vehicle'], num_lanes: int) -> bool:
        """Attempt to change lanes if beneficial and safe"""
        if self.lane_change_cooldown > 0:
            return False
            
        current_lane_vehicles = [v for v in nearby_vehicles 
                               if v.lane == self.lane and v.id != self.id and v.active]
        
        # Check if lane change is beneficial
        leader = self._find_leader(current_lane_vehicles)
        if not leader or self._calculate_gap(leader) > 30.0:
            return False  # No need to change lanes
        
        # Adjust lane change frequency based on behavior profile
        aggressiveness = {
            'aggressive': 1.5,
            'normal': 1.0,
            'conservative': 0.6
        }.get(self.behavior_profile, 1.0)
        
        if random.random() > aggressiveness * 0.3:  # Base 30% chance
            return False
        
        # Try adjacent lanes
        for target_lane in [self.lane - 1, self.lane + 1]:
            if self._is_valid_lane(target_lane, num_lanes) and self._is_lane_change_safe(target_lane, nearby_vehicles):
                self.lane = target_lane
                self.y = target_lane * 3.7
                self.lane_change_cooldown = 5.0 / aggressiveness  # Aggressive drivers change more often
                return True
        
        return False
    
    def _is_valid_lane(self, lane: int, num_lanes: int) -> bool:
        """Check if lane number is valid"""
        if self.direction == 'forward':
            return 1 <= lane <= num_lanes // 2
        else:
            return (num_lanes // 2 + 1) <= lane <= num_lanes
    
    def _is_lane_change_safe(self, target_lane: int, nearby_vehicles: List['Vehicle']) -> bool:
        """Check if lane change is safe"""
        target_lane_vehicles = [v for v in nearby_vehicles 
                              if v.lane == target_lane and v.active]
        
        # Adjust safety margins based on behavior profile
        safety_margin = {
            'aggressive': 12.0,
            'normal': 15.0,
            'conservative': 20.0
        }.get(self.behavior_profile, 15.0)
        
        for vehicle in target_lane_vehicles:
            gap = abs(vehicle.x - self.x)
            if gap < safety_margin:
                return False
        
        return True
    
    def get_angle(self) -> float:
        """Get vehicle angle based on direction"""
        return 90.0 if self.direction == 'forward' else 270.0
    
    def get_lane_id(self) -> str:
        """Get SUMO-style lane ID"""
        lane_index = (self.lane - 1) % 3
        return f"E0_{lane_index}"

class ComprehensiveFCDGenerator:
    """Main class for generating comprehensive FCD data with flexible configuration"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.vehicles: Dict[str, Vehicle] = {}
        self.specific_nodes: Dict[str, SpecificNode] = {}
        self.vehicle_counter = 0
        self.current_time = 0.0
        # Override road length from config first, before setting up num_lanes
        if config.get('traffic_config', {}).get('mode') == 'specific_defined_nodes':
            traffic_config = config.get('traffic_config', {})
            self.road_length = traffic_config.get('road_length', 1000)
            self.simulation_duration = traffic_config.get('duration', 10000)
        else:
            self.road_length = config.get('road_length', 1000)
            self.simulation_duration = config.get('simulation_duration', 10000)
            
        self.num_lanes = config.get('num_lanes_per_direction', 3) * 2
        self.time_step = config.get('time_step', 1.0)
        
        # Handle flexible configuration
        self.traffic_config = config.get('traffic_config', {})
        self.operation_mode = self.traffic_config.get('mode', 'density_based')
        
        # Set up scenario for density-based mode
        if self.operation_mode == 'density_based':
            scenario_name = self.traffic_config.get('scenario', 'learning_optimized')
            self.scenario = TrafficScenario.SCENARIOS.get(scenario_name)
        else:
            self.scenario = None
        
        # Initialize specific nodes if in specific_defined_nodes mode
        if self.operation_mode == 'specific_defined_nodes':
            self._initialize_specific_nodes()
        
        # Visualization
        self.enable_visualization = config.get('enable_visualization', False)
        self.fig = None
        self.ax = None
        self.vehicle_plots = {}
        
        # Statistics
        self.density_history = []
        self.speed_history = []
        self.vehicle_count_history = []
        self.node_distance_history = []  # Track distance between nodes
        self.node_phase_history = []     # Track node phases
    
    def _initialize_specific_nodes(self):
        """Initialize specific nodes based on configuration"""
        node_config = self.traffic_config
        node_ids = node_config.get('node_ids', ('node1', 'node2'))
        node_types = node_config.get('node_types', ('car', 'car'))
        lane_number = node_config.get('lane_number', 1)
        
        # Create the two specific nodes
        for i, (node_id, node_type) in enumerate(zip(node_ids, node_types)):
            node = SpecificNode(node_id, node_type, lane_number, node_config)
            self.specific_nodes[node_id] = node
        
        print(f"Initialized {len(self.specific_nodes)} specific nodes in {node_config['node_mode']} mode")
        
        if node_config['node_mode'] == 'static':
            positions = node_config['node_positions']
            print(f"Static nodes at positions: {positions[0]}m and {positions[1]}m on lane {lane_number}")
        elif node_config['node_mode'] == 'dynamic':
            speed = node_config['node_speed_kmh']
            print(f"Dynamic nodes with speed {speed} km/h on lane {lane_number}")
            print(f"Road length: {self.road_length}m, Meeting point: {self.road_length/2}m")
    
    def generate_fcd(self, output_file: str):
        """Generate the complete FCD file with flexible configuration"""
        mode_desc = f"{self.operation_mode} mode"
        if self.operation_mode == 'fixed_count':
            target_count = self.traffic_config.get('vehicle_count', 45)
            speed_range = self.traffic_config.get('speed_range_kmh', (30, 120))
            mode_desc += f" ({target_count} vehicles, {speed_range[0]}-{speed_range[1]} km/h)"
        elif self.operation_mode == 'specific_defined_nodes':
            node_mode = self.traffic_config.get('node_mode', 'static')
            lane_num = self.traffic_config.get('lane_number', 1)
            if node_mode == 'static':
                positions = self.traffic_config.get('node_positions', (100, 300))
                mode_desc += f" (Static nodes at {positions[0]}m & {positions[1]}m, Lane {lane_num})"
            else:
                speed = self.traffic_config.get('node_speed_kmh', 30)
                mode_desc += f" (Dynamic nodes at {speed} km/h, Lane {lane_num})"
        elif self.scenario:
            mode_desc += f" ({self.scenario['description']})"
            
        print(f"Starting FCD generation...")
        print(f"Configuration: {mode_desc}")
        print(f"Duration: {self.simulation_duration} seconds")
        print(f"Road: {self.road_length}m with {self.num_lanes} lanes")
        
        # Setup visualization if enabled
        if self.enable_visualization:
            self._setup_visualization()
        
        # Create XML structure
        root = ET.Element('fcd-export')
        root.set('xmlns:xsi', 'http://www.w3.org/2001/XMLSchema-instance')
        root.set('xsi:noNamespaceSchemaLocation', 'http://sumo.dlr.de/xsd/fcd_file.xsd')
        
        # Simulation loop
        time_steps = np.arange(0, self.simulation_duration, self.time_step)
        
        for i, time in enumerate(time_steps):
            self.current_time = time
            
            # Handle specific nodes mode
            if self.operation_mode == 'specific_defined_nodes':
                simulation_complete = self._update_specific_nodes()
                
                # Optionally manage background traffic
                if self.traffic_config.get('enable_background_traffic', False):
                    self._manage_background_traffic()
                
                current_density = 0.1  # Low density for specific nodes mode
                
                # Check if simulation should end early (for dynamic mode)
                if simulation_complete and self.traffic_config['node_mode'] == 'dynamic':
                    print(f"Dynamic nodes completed journey at time {time:.1f}s")
                    # Add few more timesteps to capture completion
                    if i > len(time_steps) - 10:
                        break
            else:
                # Original modes
                if self.operation_mode == 'fixed_count':
                    self._manage_fixed_vehicle_population()
                    current_density = self._calculate_effective_density()
                else:
                    current_density = self._calculate_traffic_density(time)
                    self._manage_vehicle_population(current_density)
            
            # Update all regular vehicles
            active_vehicles = [v for v in self.vehicles.values() if v.active]
            speed_range_ms = self.traffic_config.get('speed_range_ms') if self.operation_mode == 'fixed_count' else None
            
            # Create combined list of all moving objects for interaction
            all_moving_objects = active_vehicles + [node for node in self.specific_nodes.values() if node.active]
            
            for vehicle in active_vehicles:
                vehicle.update_position(self.time_step, current_density, all_moving_objects, speed_range_ms)
                
                # Attempt lane changes occasionally (but not in same lane as specific nodes)
                if random.random() < 0.01:  # 1% chance per timestep
                    vehicle.attempt_lane_change(active_vehicles, self.num_lanes)
                
                # Handle boundary conditions
                self._handle_boundary_conditions(vehicle)
            
            # Create timestep element
            timestep = ET.SubElement(root, 'timestep')
            timestep.set('time', f'{time:.2f}')
            
            # Add specific nodes data first
            for node in self.specific_nodes.values():
                if node.active:
                    veh_elem = ET.SubElement(timestep, 'vehicle')
                    veh_elem.set('id', node.id)
                    veh_elem.set('x', f'{node.x:.2f}')
                    veh_elem.set('y', f'{node.y:.2f}')
                    veh_elem.set('angle', f'{node.get_angle():.2f}')
                    veh_elem.set('type', 'SPECIFIC_NODE')
                    veh_elem.set('speed', f'{node.speed:.2f}')
                    veh_elem.set('pos', f'{node.x:.2f}')
                    veh_elem.set('lane', node.get_lane_id())
                    veh_elem.set('slope', '0.00')
            
            # Add regular vehicle data
            for vehicle in active_vehicles:
                if vehicle.active:
                    veh_elem = ET.SubElement(timestep, 'vehicle')
                    veh_elem.set('id', vehicle.id)
                    veh_elem.set('x', f'{vehicle.x:.2f}')
                    veh_elem.set('y', f'{vehicle.y:.2f}')
                    veh_elem.set('angle', f'{vehicle.get_angle():.2f}')
                    veh_elem.set('type', 'DEFAULT_VEHTYPE')
                    veh_elem.set('speed', f'{vehicle.speed:.2f}')
                    veh_elem.set('pos', f'{vehicle.x:.2f}')
                    veh_elem.set('lane', vehicle.get_lane_id())
                    veh_elem.set('slope', '0.00')
            
            # Update statistics
            self._update_statistics(current_density, active_vehicles)
            
            # Update visualization
            update_freq = getattr(self, 'animation_config', {}).get('update_frequency', 5)
            if self.enable_visualization and i % update_freq == 0:
                self._update_visualization(active_vehicles)
            
            # Progress reporting
            if i % 500 == 0 or (self.operation_mode == 'specific_defined_nodes' and i % 50 == 0):
                self._report_progress(i, len(time_steps), time, active_vehicles, current_density)
        
        # Write XML file
        tree = ET.ElementTree(root)
        ET.indent(tree, space="  ", level=0)
        tree.write(output_file, encoding='utf-8', xml_declaration=True)
        
        print(f"FCD data written to {output_file}")
        self._generate_statistics_report(output_file)
    
    def _update_specific_nodes(self) -> bool:
        """Update specific nodes and return True if simulation should complete"""
        if not self.specific_nodes:
            return False
        
        nodes = list(self.specific_nodes.values())
        
        if len(nodes) == 2:
            # Update both nodes
            nodes[0].update_position(self.time_step, self.current_time, nodes[1])
            nodes[1].update_position(self.time_step, self.current_time, nodes[0])
            
            # Track statistics for dynamic mode
            if self.traffic_config['node_mode'] == 'dynamic':
                distance = abs(nodes[0].x - nodes[1].x)
                self.node_distance_history.append(distance)
                self.node_phase_history.append(f"{nodes[0].phase}-{nodes[1].phase}")
                
                # Check if both nodes completed their journey
                if all(node.is_journey_complete() for node in nodes):
                    return True
        else:
            # Update single node
            nodes[0].update_position(self.time_step, self.current_time)
        
        return False
    
    def _manage_background_traffic(self):
        """Manage background traffic in specific nodes mode"""
        target_count = self.traffic_config.get('background_vehicle_count', 10)
        active_vehicles = [v for v in self.vehicles.values() if v.active]
        current_count = len(active_vehicles)
        
        # Avoid spawning in the same lane as specific nodes
        specific_node_lane = self.traffic_config.get('lane_number', 1)
        
        if current_count < target_count:
            vehicles_to_spawn = min(2, target_count - current_count)
            for _ in range(vehicles_to_spawn):
                self._spawn_background_vehicle(avoid_lane=specific_node_lane)
        elif current_count > target_count * 1.2:
            vehicles_to_remove = min(2, current_count - target_count)
            despawnable = [v for v in active_vehicles if self._can_despawn(v)]
            for vehicle in random.sample(despawnable, min(vehicles_to_remove, len(despawnable))):
                vehicle.active = False
    
    def _spawn_background_vehicle(self, avoid_lane: int = None):
        """Spawn a background vehicle avoiding specific lanes"""
        direction = random.choice(['forward', 'backward'])
        
        # Choose lane avoiding the specific node lane
        if direction == 'forward':
            available_lanes = [l for l in range(1, self.num_lanes // 2 + 1) if l != avoid_lane]
        else:
            available_lanes = [l for l in range(self.num_lanes // 2 + 1, self.num_lanes + 1)]
        
        if not available_lanes:
            return  # No available lanes
            
        lane = random.choice(available_lanes)
        
        if direction == 'forward':
            start_x = -50
        else:
            start_x = self.road_length + 50
        
        vehicle_type = random.choices(
            ['car', 'truck', 'bus', 'motorcycle'], 
            weights=[0.8, 0.1, 0.05, 0.05]
        )[0]
        
        initial_speed = random.uniform(1.0, 2.5)  # Moderate speed for background
        
        vehicle_id = f'bg_veh{self.vehicle_counter}'
        self.vehicle_counter += 1
        
        vehicle = Vehicle(vehicle_id, lane, start_x, initial_speed, direction, vehicle_type)
        self.vehicles[vehicle_id] = vehicle
    
    def _manage_fixed_vehicle_population(self):
        """Manage vehicle population for fixed count mode"""
        target_count = self.traffic_config.get('vehicle_count', 45)
        maintain_count = self.traffic_config.get('maintain_count', True)
        spawn_rate = self.traffic_config.get('spawn_rate', 3)
        
        active_vehicles = [v for v in self.vehicles.values() if v.active]
        current_count = len(active_vehicles)
        
        if current_count < target_count:
            # Spawn new vehicles
            vehicles_to_spawn = min(spawn_rate, target_count - current_count)
            for _ in range(vehicles_to_spawn):
                self._spawn_vehicle_flexible()
        
        elif maintain_count and current_count > target_count * 1.15:  # Allow 15% buffer
            # Remove excess vehicles
            vehicles_to_remove = min(2, current_count - target_count)
            despawnable = [v for v in active_vehicles if self._can_despawn(v)]
            for vehicle in random.sample(despawnable, min(vehicles_to_remove, len(despawnable))):
                vehicle.active = False
    
    def _spawn_vehicle_flexible(self):
        """Spawn a vehicle with flexible configuration"""
        # Get configuration parameters
        vehicle_types = self.traffic_config.get('vehicle_types', 
                                               {'car': 0.75, 'truck': 0.15, 'bus': 0.05, 'motorcycle': 0.05})
        lane_distribution = self.traffic_config.get('lane_distribution', 'balanced')
        behavior_profile = self.traffic_config.get('behavior_profile', 'normal')
        speed_range_ms = self.traffic_config.get('speed_range_ms', (1.0, 3.0))
        speed_variation = self.traffic_config.get('speed_variation', 0.2)
        
        # Choose direction and lane based on distribution
        direction = random.choice(['forward', 'backward'])
        lane = self._choose_lane(direction, lane_distribution)
        
        # Set starting position
        if direction == 'forward':
            start_x = -50
        else:
            start_x = self.road_length + 50
        
        # Choose vehicle type based on distribution
        vehicle_type = random.choices(
            list(vehicle_types.keys()),
            weights=list(vehicle_types.values())
        )[0]
        
        # Set initial speed with variation
        base_speed = random.uniform(speed_range_ms[0], speed_range_ms[1])
        speed_var = random.uniform(1 - speed_variation, 1 + speed_variation)
        initial_speed = base_speed * speed_var
        
        # Vehicle type speed adjustment
        type_factors = {
            'truck': 0.9,
            'bus': 0.85,
            'motorcycle': 1.1,
            'car': 1.0
        }
        initial_speed *= type_factors.get(vehicle_type, 1.0)
        
        # Create vehicle
        vehicle_id = f'veh{self.vehicle_counter}'
        self.vehicle_counter += 1
        
        vehicle = Vehicle(vehicle_id, lane, start_x, initial_speed, direction, vehicle_type, behavior_profile)
        self.vehicles[vehicle_id] = vehicle
    
    def _choose_lane(self, direction: str, distribution: str) -> int:
        """Choose lane based on distribution pattern"""
        if direction == 'forward':
            available_lanes = list(range(1, self.num_lanes // 2 + 1))
        else:
            available_lanes = list(range(self.num_lanes // 2 + 1, self.num_lanes + 1))
        
        if distribution == 'balanced':
            return random.choice(available_lanes)
        elif distribution == 'heavy_right':
            # Prefer rightmost lanes
            weights = [1.0 + i * 0.5 for i in range(len(available_lanes))]
            return random.choices(available_lanes, weights=weights)[0]
        elif distribution == 'heavy_left':
            # Prefer leftmost lanes
            weights = [1.0 + (len(available_lanes) - 1 - i) * 0.5 for i in range(len(available_lanes))]
            return random.choices(available_lanes, weights=weights)[0]
        else:  # random
            return random.choice(available_lanes)
    
    def _calculate_effective_density(self) -> float:
        """Calculate effective density for fixed count mode"""
        active_vehicles = [v for v in self.vehicles.values() if v.active]
        vehicle_count = len(active_vehicles)
        
        # Calculate density based on vehicles per km per direction
        road_length_km = self.road_length / 1000.0
        vehicles_per_direction = vehicle_count / 2
        vehicles_per_km_per_direction = vehicles_per_direction / road_length_km
        
        # Map to density factor (reverse of the original calculation)
        if vehicles_per_km_per_direction <= 30:
            return 0.1
        elif vehicles_per_km_per_direction <= 60:
            return 0.3
        elif vehicles_per_km_per_direction <= 120:
            return 0.5
        elif vehicles_per_km_per_direction <= 180:
            return 0.7
        else:
            return 0.9
    
    def _calculate_traffic_density(self, time: float) -> float:
        """Calculate traffic density based on scenario and time (original method)"""
        if not self.scenario:
            return 0.5
            
        scenario = self.scenario
        
        if 'phases' in scenario:
            # Learning optimized scenario with phases
            cumulative_time = 0
            for phase in scenario['phases']:
                if time < cumulative_time + phase['duration']:
                    return phase['density']
                cumulative_time += phase['duration']
            return scenario['phases'][-1]['density']
        
        elif scenario.get('peak_periods'):
            # Rush hour scenario
            density = scenario['base_density']
            for start, end in scenario['peak_periods']:
                if start <= time <= end:
                    peak_factor = 0.5 * (1 + math.cos(2 * math.pi * (time - start) / (end - start)))
                    density = scenario['base_density'] + (scenario['peak_density'] - scenario['base_density']) * peak_factor
            return density
        
        elif scenario.get('wave_frequency'):
            # Highway congestion with waves
            wave_time = time % scenario['wave_frequency']
            if wave_time < scenario['wave_duration']:
                wave_factor = 0.5 * (1 - math.cos(2 * math.pi * wave_time / scenario['wave_duration']))
                return scenario['base_density'] + (scenario['peak_density'] - scenario['base_density']) * wave_factor
            return scenario['base_density']
        
        elif scenario.get('cycle_frequency'):
            # City traffic with cycles
            cycle_time = time % scenario['cycle_frequency']
            if cycle_time < scenario['cycle_duration']:
                return scenario['peak_density']
            return scenario['base_density']
        
        elif scenario.get('variation_frequency'):
            # Random varying scenario
            base = scenario['base_density']
            amplitude = scenario['variation_amplitude']
            frequency = scenario['variation_frequency']
            noise = 0.1 * (random.random() - 0.5)
            sine_component = amplitude * math.sin(2 * math.pi * time / frequency)
            return max(0.1, min(1.0, base + sine_component + noise))
        
        return 0.5  # Default density
    
    def _manage_vehicle_population(self, target_density: float):
        """Manage vehicle population for density-based mode (original method)"""
        active_vehicles = [v for v in self.vehicles.values() if v.active]
        current_count = len(active_vehicles)
        
        # Calculate target number of vehicles based on REALISTIC STANDARDS
        road_length_km = self.road_length / 1000.0
        
        # Map density factor to realistic vehicles/km/direction
        if target_density <= 0.2:
            vehicles_per_km_per_direction = 30
        elif target_density <= 0.4:
            vehicles_per_km_per_direction = 60
        elif target_density <= 0.6:
            vehicles_per_km_per_direction = 120
        elif target_density <= 0.8:
            vehicles_per_km_per_direction = 180
        else:
            vehicles_per_km_per_direction = 240
        
        target_count = int(vehicles_per_km_per_direction * road_length_km * 2)
        variation = random.uniform(0.8, 1.2)
        target_count = int(target_count * variation)
        
        if current_count < target_count:
            spawn_rate = min(8, max(1, target_count - current_count))
            vehicles_to_spawn = min(spawn_rate, target_count - current_count)
            for _ in range(vehicles_to_spawn):
                self._spawn_vehicle()
        
        elif current_count > target_count * 1.3:
            vehicles_to_remove = min(5, current_count - target_count)
            inactive_candidates = [v for v in active_vehicles if self._can_despawn(v)]
            for vehicle in random.sample(inactive_candidates, min(vehicles_to_remove, len(inactive_candidates))):
                vehicle.active = False
    
    def _spawn_vehicle(self):
        """Original spawn vehicle method for density-based mode"""
        direction = random.choice(['forward', 'backward'])
        if direction == 'forward':
            lane = random.randint(1, self.num_lanes // 2)
            start_x = -50
        else:
            lane = random.randint(self.num_lanes // 2 + 1, self.num_lanes)
            start_x = self.road_length + 50
        
        vehicle_type = random.choices(
            ['car', 'truck', 'bus', 'motorcycle'], 
            weights=[0.75, 0.15, 0.05, 0.05]
        )[0]
        
        current_density = self._calculate_traffic_density(self.current_time)
        
        if current_density <= 0.2:
            speed_range = (2.5, 3.5)
        elif current_density <= 0.4:
            speed_range = (2.0, 3.0)
        elif current_density <= 0.6:
            speed_range = (1.5, 2.5)
        elif current_density <= 0.8:
            speed_range = (0.8, 1.8)
        else:
            speed_range = (0.2, 0.8)
        
        if vehicle_type == 'truck':
            initial_speed = random.uniform(speed_range[0] * 0.85, speed_range[1] * 0.9)
        elif vehicle_type == 'bus':
            initial_speed = random.uniform(speed_range[0] * 0.8, speed_range[1] * 0.85)
        elif vehicle_type == 'motorcycle':
            initial_speed = random.uniform(speed_range[0] * 1.1, speed_range[1] * 1.15)
        else:
            initial_speed = random.uniform(speed_range[0], speed_range[1])
        
        vehicle_id = f'veh{self.vehicle_counter}'
        self.vehicle_counter += 1
        
        vehicle = Vehicle(vehicle_id, lane, start_x, initial_speed, direction, vehicle_type)
        self.vehicles[vehicle_id] = vehicle
    
    def _can_despawn(self, vehicle: Vehicle) -> bool:
        """Check if vehicle can be despawned (outside road boundaries)"""
        if vehicle.direction == 'forward':
            return vehicle.x > self.road_length + 100
        else:
            return vehicle.x < -100
    
    def _handle_boundary_conditions(self, vehicle: Vehicle):
        """Handle vehicles reaching road boundaries"""
        if self._can_despawn(vehicle):
            vehicle.active = False
    
    def _report_progress(self, i: int, total_steps: int, time: float, active_vehicles: List[Vehicle], 
                        current_density: float):
        """Report simulation progress with detailed statistics"""
        progress = (i / total_steps) * 100
        
        if self.operation_mode == 'specific_defined_nodes':
            # Special reporting for node positioning mode
            node_info = []
            for node_id, node in self.specific_nodes.items():
                if node.active:
                    if hasattr(node, 'phase'):
                        node_info.append(f"{node_id}: {node.phase} @ {node.x:.1f}m")
                    else:
                        node_info.append(f"{node_id} @ {node.x:.1f}m")
            
            print(f"Progress: {progress:.1f}% (Time: {time:.0f}s)")
            print(f"   Nodes: {' | '.join(node_info)}")
            
            if self.traffic_config['node_mode'] == 'dynamic' and len(self.specific_nodes) == 2:
                nodes = list(self.specific_nodes.values())
                distance = abs(nodes[0].x - nodes[1].x)
                print(f"   Distance between nodes: {distance:.1f}m")
            
            if active_vehicles:
                print(f"   Background vehicles: {len(active_vehicles)}")
        else:
            # Original reporting for other modes
            road_length_km = self.road_length / 1000.0
            vehicles_per_direction = len(active_vehicles) / 2
            vehicles_per_km_per_direction = vehicles_per_direction / road_length_km
            
            if active_vehicles:
                current_avg_speed_ms = np.mean([v.speed for v in active_vehicles])
                current_avg_speed_kmh = current_avg_speed_ms * 3.6
            else:
                current_avg_speed_kmh = 0
            
            # Determine channel model
            if vehicles_per_km_per_direction <= 30:
                channel_model = "AWGN"
                expected_speed = "100-120 km/h"
            elif vehicles_per_km_per_direction <= 60:
                channel_model = "R-LOS"
                expected_speed = "80-100 km/h"
            elif vehicles_per_km_per_direction <= 90:
                channel_model = "H-LOS"
                expected_speed = "60-80 km/h"
            elif vehicles_per_km_per_direction <= 120:
                channel_model = "H-NLOS"
                expected_speed = "50-70 km/h"
            elif vehicles_per_km_per_direction <= 150:
                channel_model = "UA-LOS-ENH"
                expected_speed = "30-50 km/h"
            else:
                channel_model = "C-NLOS-ENH"
                expected_speed = "5-30 km/h"
            
            mode_info = ""
            if self.operation_mode == 'fixed_count':
                target_count = self.traffic_config.get('vehicle_count', 45)
                mode_info = f" | Target: {target_count}"
            
            print(f"Progress: {progress:.1f}% (Time: {time:.0f}s)")
            print(f"   Vehicles: {len(active_vehicles)} total{mode_info} | {vehicles_per_km_per_direction:.0f} vehicles/km/direction")
            print(f"   Channel: {channel_model} | Density Factor: {current_density:.2f}")
            print(f"   Current Speed: {current_avg_speed_kmh:.1f} km/h | Expected: {expected_speed}")
        
        print(f"   " + "="*60)
    
    def _update_statistics(self, density: float, vehicles: List[Vehicle]):
        """Update simulation statistics"""
        self.density_history.append(float(density))
        self.vehicle_count_history.append(len(vehicles))
        
        if vehicles:
            avg_speed = np.mean([v.speed for v in vehicles])
            self.speed_history.append(float(avg_speed * 3.6))
        else:
            self.speed_history.append(0.0)
    
    def _setup_visualization(self):
        """Setup enhanced matplotlib visualization with beautiful road animation"""
        # Create figure with subplots for main view and statistics
        self.fig = plt.figure(figsize=(16, 10))
        
        # Main road view (takes up most of the space)
        self.ax = plt.subplot2grid((4, 4), (0, 0), colspan=4, rowspan=3)
        
        # Statistics panels
        self.stats_ax = plt.subplot2grid((4, 4), (3, 0), colspan=2)
        self.speed_ax = plt.subplot2grid((4, 4), (3, 2), colspan=2)
        
        # Setup main road view
        self.ax.set_xlim(-50, self.road_length + 50)
        self.ax.set_ylim(-2, self.num_lanes * 3.7 + 2)
        self.ax.set_xlabel('Road Length (m)', fontsize=12)
        self.ax.set_ylabel('Lane Position (m)', fontsize=12)
        
        # Set title based on mode
        if self.operation_mode == 'specific_defined_nodes':
            node_mode = self.traffic_config.get('node_mode', 'static')
            self.ax.set_title(f'Specific Node Positioning - {node_mode.title()} Mode', fontsize=14, fontweight='bold')
        else:
            self.ax.set_title('Real-time Vehicle Movement Simulation', fontsize=14, fontweight='bold')
        
        # Draw beautiful road infrastructure
        self._draw_road_infrastructure()
        
        # Setup statistics displays
        self._setup_statistics_display()
        
        # Initialize vehicle tracking
        self.vehicle_plots = {}
        self.vehicle_labels = {}
        self.speed_indicators = {}
        
        # Animation configuration
        self.animation_config = {
            'show_vehicle_ids': True,
            'show_speed_bars': True,
            'show_statistics': True,
            'update_frequency': 5,  # Update every N timesteps
            'max_labels': 20,  # Max vehicle labels to show
        }
        
        # Animation settings
        self.frame_count = 0
        self.last_update_time = 0
        
        plt.tight_layout()
        plt.ion()
        plt.show()
    
    def _draw_road_infrastructure(self):
        """Draw detailed road infrastructure"""
        lane_width = 3.7
        
        # Draw road base (asphalt color)
        road_background = plt.Rectangle((0, 0), self.road_length, self.num_lanes * lane_width, 
                                       facecolor='#404040', alpha=0.8, zorder=0)
        self.ax.add_patch(road_background)
        
        # Draw lane dividers
        for lane in range(self.num_lanes + 1):
            y = lane * lane_width
            if lane == 0 or lane == self.num_lanes:
                # Road edges - solid white lines
                self.ax.axhline(y=y, color='white', linewidth=3, alpha=0.9, zorder=1)
            elif lane == self.num_lanes // 2:
                # Center divider - double yellow lines
                self.ax.axhline(y=y-0.1, color='yellow', linewidth=3, alpha=0.9, zorder=1)
                self.ax.axhline(y=y+0.1, color='yellow', linewidth=3, alpha=0.9, zorder=1)
            else:
                # Lane dividers - dashed white lines
                self.ax.axhline(y=y, color='white', linestyle='--', linewidth=2, alpha=0.7, zorder=1)
        
        # Add direction arrows (skip if only one lane for specific nodes)
        if self.operation_mode != 'specific_defined_nodes' or self.num_lanes > 2:
            for direction in ['forward', 'backward']:
                if direction == 'forward':
                    lanes = range(1, self.num_lanes // 2 + 1)
                    arrow_direction = '=>'
                else:
                    lanes = range(self.num_lanes // 2 + 1, self.num_lanes + 1)
                    arrow_direction = '<='
                
                for lane in lanes:
                    y = (lane - 0.5) * lane_width
                    for x in range(100, int(self.road_length), 200):  # Every 200m
                        self.ax.text(x, y, arrow_direction, fontsize=16, ha='center', va='center', 
                                   color='white', alpha=0.6, fontweight='bold', zorder=1)
        
        # Highlight specific node lane if in specific nodes mode
        if self.operation_mode == 'specific_defined_nodes':
            node_lane = self.traffic_config.get('lane_number', 1)
            lane_y = (node_lane - 1) * lane_width
            # Add a subtle highlight to the specific node lane
            highlight = plt.Rectangle((0, lane_y), self.road_length, lane_width, 
                                    facecolor='lightblue', alpha=0.1, zorder=0.5)
            self.ax.add_patch(highlight)
            
            # Add label for specific node lane
            self.ax.text(-30, lane_y + lane_width/2, f'Node Lane {node_lane}', 
                        fontsize=12, ha='center', va='center', fontweight='bold',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgreen', alpha=0.8))
            
            # Add position markers for static mode
            if self.traffic_config['node_mode'] == 'static':
                positions = self.traffic_config['node_positions']
                for i, pos in enumerate(positions):
                    self.ax.axvline(x=pos, color='red', linestyle=':', linewidth=2, alpha=0.7)
                    self.ax.text(pos, -1, f'Node {i+1}\nPosition', ha='center', va='top',
                               bbox=dict(boxstyle="round,pad=0.2", facecolor='yellow', alpha=0.7))
        
        # Add lane numbers for other lanes
        for lane in range(1, self.num_lanes + 1):
            if self.operation_mode == 'specific_defined_nodes' and lane == self.traffic_config.get('lane_number', 1):
                continue  # Skip the specific node lane as it's already labeled
                
            y = (lane - 0.5) * lane_width
            direction = 'Forward' if lane <= self.num_lanes // 2 else 'Backward'
            lane_num = lane if lane <= self.num_lanes // 2 else lane - self.num_lanes // 2
            self.ax.text(-30, y, f'Lane {lane_num}\n({direction})', fontsize=10, ha='center', va='center',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor='lightblue', alpha=0.7))
    
    def _setup_statistics_display(self):
        """Setup real-time statistics displays"""
        # Vehicle count chart
        self.stats_ax.set_title('Vehicle Statistics', fontsize=12, fontweight='bold')
        self.stats_ax.set_xlim(0, 100)
        self.stats_ax.set_ylim(0, 100)
        self.stats_ax.set_xlabel('Time Progress (%)')
        self.stats_ax.set_ylabel('Vehicles')
        
        # Speed distribution chart  
        self.speed_ax.set_title('Speed Distribution', fontsize=12, fontweight='bold')
        self.speed_ax.set_xlim(0, 150)
        self.speed_ax.set_ylim(0, 20)
        self.speed_ax.set_xlabel('Speed (km/h)')
        self.speed_ax.set_ylabel('Vehicle Count')
        
        # Initialize empty plots
        self.vehicle_count_line, = self.stats_ax.plot([], [], 'b-', linewidth=2, label='Vehicle Count')
        
        if self.operation_mode == 'specific_defined_nodes' and self.traffic_config['node_mode'] == 'dynamic':
            self.node_distance_line, = self.stats_ax.plot([], [], 'r-', linewidth=2, label='Node Distance')
            self.stats_ax.legend()
        else:
            self.target_count_line, = self.stats_ax.plot([], [], 'r--', linewidth=2, label='Target Count')
            self.stats_ax.legend()
        
        # Speed histogram bars
        self.speed_bins = np.arange(0, 151, 10)
        self.speed_bars = self.speed_ax.bar(self.speed_bins[:-1], np.zeros(len(self.speed_bins)-1), 
                                           width=8, alpha=0.7, color='green')
    
    def _update_visualization(self, vehicles: List[Vehicle]):
        """Enhanced visualization with beautiful animations and real-time stats"""
        if not self.enable_visualization:
            return
        
        # Only update every few frames for smooth animation
        self.frame_count += 1
        if self.frame_count % 5 != 0:  # Update every 5 timesteps
            return
            
        # Clear previous vehicle plots
        for plot in self.vehicle_plots.values():
            if isinstance(plot, list):
                for p in plot:
                    p.remove()
            else:
                plot.remove()
        for label in self.vehicle_labels.values():
            label.remove()
        for indicator in self.speed_indicators.values():
            indicator.remove()
            
        self.vehicle_plots.clear()
        self.vehicle_labels.clear()
        self.speed_indicators.clear()
        
        # Plot specific nodes first (with special styling)
        for node in self.specific_nodes.values():
            if node.active:
                self._draw_specific_node(node)
        
        # Plot regular vehicles
        visible_vehicles = [v for v in vehicles if -50 <= v.x <= self.road_length + 50 and v.active]
        
        for vehicle in visible_vehicles:
            self._draw_enhanced_vehicle(vehicle)
        
        # Update statistics
        self._update_statistics_display(vehicles)
        
        # Update main title with comprehensive info
        if self.operation_mode == 'specific_defined_nodes':
            node_mode = self.traffic_config['node_mode']
            title = f'Specific Node Positioning - {node_mode.title()} Mode | Time: {self.current_time:.0f}s'
            
            if node_mode == 'dynamic' and len(self.specific_nodes) == 2:
                nodes = list(self.specific_nodes.values())
                distance = abs(nodes[0].x - nodes[1].x)
                title += f' | Distance: {distance:.1f}m'
                
                # Add phase information
                phases = [node.phase for node in nodes if hasattr(node, 'phase')]
                if phases:
                    title += f' | Phases: {"-".join(phases)}'
            
            title += f' | Background: {len(vehicles)} vehicles'
        else:
            # Original title format for other modes
            if vehicles:
                avg_speed = np.mean([v.speed for v in vehicles]) * 3.6
                max_speed = np.max([v.speed for v in vehicles]) * 3.6
                min_speed = np.min([v.speed for v in vehicles]) * 3.6
            else:
                avg_speed = max_speed = min_speed = 0
                
            title = f'Live Traffic Simulation | Time: {self.current_time:.0f}s | Vehicles: {len(vehicles)}\n'
            title += f'Speed: Avg {avg_speed:.1f} km/h | Range {min_speed:.1f}-{max_speed:.1f} km/h'
            
            if self.operation_mode == 'fixed_count':
                target = self.traffic_config.get('vehicle_count', 45)
                title += f' | Target: {target}'
            
        self.ax.set_title(title, fontsize=12, fontweight='bold')
        
        # Force redraw
        plt.pause(0.001)
    
    def _draw_specific_node(self, node: SpecificNode):
        """Draw specific node with special styling"""
        # Node type configurations
        node_config = {
            'car': {'color': '#FF1744', 'marker': 'D', 'size': 150, 'symbol': 'NODE'},
            'truck': {'color': '#D32F2F', 'marker': 'D', 'size': 200, 'symbol': 'N-TRK'},
            'bus': {'color': '#F57C00', 'marker': 'D', 'size': 180, 'symbol': 'N-BUS'},
            'motorcycle': {'color': '#8E24AA', 'marker': 'D', 'size': 120, 'symbol': 'N-MCY'}
        }
        
        config = node_config.get(node.vehicle_type, node_config['car'])
        
        # Main node body with special diamond shape and larger size
        node_plot = self.ax.scatter(node.x, node.y, c=config['color'], 
                                   marker=config['marker'], s=config['size'], 
                                   alpha=0.9, edgecolors='white', linewidth=3, zorder=5)
        
        # Node ID and status label (always show for nodes)
        speed_kmh = node.speed * 3.6
        if hasattr(node, 'phase') and node.config['node_mode'] == 'dynamic':
            label_text = f'{node.id}\n{speed_kmh:.0f} km/h\n{node.phase}'
        else:
            label_text = f'{node.id}\n{speed_kmh:.0f} km/h\nStatic'
            
        label = self.ax.text(node.x, node.y - 1.5, label_text, 
                           fontsize=10, ha='center', va='top', fontweight='bold',
                           bbox=dict(boxstyle="round,pad=0.3", facecolor='yellow', alpha=0.9),
                           zorder=5)
        self.vehicle_labels[node.id] = label
        
        # Direction indicator for dynamic nodes
        if node.config['node_mode'] == 'dynamic' and hasattr(node, 'direction'):
            if node.direction == 'right':
                arrow = '→'
                offset_x = node.length / 2
            elif node.direction == 'left':
                arrow = '←'
                offset_x = -node.length / 2
            else:
                arrow = '●'  # Static or at center
                offset_x = 0
                
            direction_plot = self.ax.text(node.x + offset_x, node.y + 0.5, arrow, 
                                        fontsize=14, ha='center', va='center', 
                                        color='white', fontweight='bold', zorder=6)
            
            self.vehicle_plots[node.id] = [node_plot, direction_plot]
        else:
            self.vehicle_plots[node.id] = node_plot
    
    def _draw_enhanced_vehicle(self, vehicle: Vehicle):
        """Draw individual vehicle with enhanced graphics"""
        # Vehicle type configurations
        vehicle_config = {
            'car': {'color': '#4CAF50', 'marker': 's', 'size': 80, 'symbol': 'CAR'},
            'truck': {'color': '#F44336', 'marker': 's', 'size': 120, 'symbol': 'TRK'},
            'bus': {'color': '#FF9800', 'marker': 's', 'size': 100, 'symbol': 'BUS'},
            'motorcycle': {'color': '#9C27B0', 'marker': 'o', 'size': 60, 'symbol': 'MCY'}
        }
        
        config = vehicle_config.get(vehicle.vehicle_type, vehicle_config['car'])
        
        # Main vehicle body
        vehicle_plot = self.ax.scatter(vehicle.x, vehicle.y, c=config['color'], 
                                     marker=config['marker'], s=config['size'], 
                                     alpha=0.8, edgecolors='black', linewidth=1, zorder=3)
        
        # Speed-based color intensity (faster = brighter)
        speed_kmh = vehicle.speed * 3.6
        if self.operation_mode == 'fixed_count':
            max_speed = self.traffic_config.get('speed_range_kmh', (30, 120))[1]
        else:
            max_speed = 120
        
        intensity = min(1.0, speed_kmh / max_speed)
        
        # Vehicle direction indicator (small arrow)
        if vehicle.direction == 'forward':
            arrow = '>'
            offset_x = vehicle.length / 2
        else:
            arrow = '<'
            offset_x = -vehicle.length / 2
            
        direction_plot = self.ax.text(vehicle.x + offset_x, vehicle.y + 0.3, arrow, 
                                    fontsize=8, ha='center', va='center', 
                                    color='white', fontweight='bold', zorder=4)
        
        # Vehicle ID and speed label (controlled by animation config)
        show_labels = getattr(self, 'animation_config', {}).get('show_vehicle_ids', True)
        max_labels = getattr(self, 'animation_config', {}).get('max_labels', 20)
        
        if show_labels and (len(self.vehicle_plots) < max_labels or vehicle.id.endswith(('0', '5'))):
            label_text = f'{vehicle.id}\n{speed_kmh:.0f} km/h'
            label = self.ax.text(vehicle.x, vehicle.y - 0.8, label_text, 
                               fontsize=8, ha='center', va='top',
                               bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.7),
                               zorder=4)
            self.vehicle_labels[vehicle.id] = label
        
        # Speed indicator bar (controlled by animation config)
        show_speed_bars = getattr(self, 'animation_config', {}).get('show_speed_bars', True)
        if show_speed_bars and speed_kmh > 0:
            bar_width = vehicle.length * (speed_kmh / max_speed)
            speed_bar = plt.Rectangle((vehicle.x - bar_width/2, vehicle.y + 1.0), 
                                    bar_width, 0.3, 
                                    facecolor='lime' if speed_kmh > max_speed*0.7 else 'yellow' if speed_kmh > max_speed*0.4 else 'red',
                                    alpha=0.7, zorder=3)
            self.ax.add_patch(speed_bar)
            self.speed_indicators[vehicle.id] = speed_bar
        
        self.vehicle_plots[vehicle.id] = [vehicle_plot, direction_plot]
    
    def _update_statistics_display(self, vehicles: List[Vehicle]):
        """Update real-time statistics charts"""
        # Update vehicle count chart
        progress = (self.current_time / self.simulation_duration) * 100
        
        # Get historical data for plotting
        if hasattr(self, 'vehicle_count_history') and self.vehicle_count_history:
            times = np.linspace(0, progress, len(self.vehicle_count_history))
            self.vehicle_count_line.set_data(times, self.vehicle_count_history)
            
            # Special handling for specific nodes mode
            if self.operation_mode == 'specific_defined_nodes' and self.traffic_config['node_mode'] == 'dynamic':
                if hasattr(self, 'node_distance_history') and self.node_distance_history:
                    distance_times = np.linspace(0, progress, len(self.node_distance_history))
                    # Scale distance to fit in the same chart
                    scaled_distances = [d / 10 for d in self.node_distance_history]  # Scale by 10
                    self.node_distance_line.set_data(distance_times, scaled_distances)
                    self.stats_ax.set_ylabel('Vehicles / Distance(÷10)')
            else:
                # Add target line for fixed count mode
                if self.operation_mode == 'fixed_count':
                    target = self.traffic_config.get('vehicle_count', 45)
                    target_line = [target] * len(self.vehicle_count_history)
                    self.target_count_line.set_data(times, target_line)
            
            # Adjust y-axis limits
            if self.vehicle_count_history:
                max_count = max(self.vehicle_count_history)
                self.stats_ax.set_ylim(0, max_count * 1.1)
        
        # Update speed distribution
        if vehicles:
            speeds = [v.speed * 3.6 for v in vehicles]  # Convert to km/h
            hist, _ = np.histogram(speeds, bins=self.speed_bins)
            
            for bar, height in zip(self.speed_bars, hist):
                bar.set_height(height)
            
            self.speed_ax.set_ylim(0, max(hist) * 1.1 if max(hist) > 0 else 1)
        
        # Add text statistics
        if self.operation_mode == 'specific_defined_nodes':
            stats_text = f'Specific Nodes: {len([n for n in self.specific_nodes.values() if n.active])}\n'
            stats_text += f'Background Vehicles: {len(vehicles)}\n'
            
            if self.traffic_config['node_mode'] == 'dynamic' and len(self.specific_nodes) == 2:
                nodes = list(self.specific_nodes.values())
                distance = abs(nodes[0].x - nodes[1].x)
                stats_text += f'Node Distance: {distance:.1f}m\n'
                
                phases = [node.phase for node in nodes if hasattr(node, 'phase')]
                if phases:
                    stats_text += f'Phases: {", ".join(phases)}\n'
        else:
            stats_text = f'Active Vehicles: {len(vehicles)}\n'
            if vehicles:
                avg_speed = np.mean([v.speed * 3.6 for v in vehicles])
                stats_text += f'Avg Speed: {avg_speed:.1f} km/h\n'
                
                # Vehicle type breakdown
                type_counts = {}
                for v in vehicles:
                    type_counts[v.vehicle_type] = type_counts.get(v.vehicle_type, 0) + 1
                
                stats_text += 'Types: '
                for vtype, count in type_counts.items():
                    stats_text += f'{vtype}({count}) '
        
        # Clear previous text and add new
        for txt in self.stats_ax.texts:
            if any(keyword in txt.get_text() for keyword in ['Active Vehicles', 'Specific Nodes']):
                txt.remove()
                
        self.stats_ax.text(0.02, 0.98, stats_text, transform=self.stats_ax.transAxes,
                          fontsize=10, verticalalignment='top',
                          bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgray', alpha=0.8))
    
    def _generate_statistics_report(self, output_file: str):
        """Generate comprehensive statistics report"""
        stats_file = output_file.replace('.xml', '_statistics.json')
        
        def convert_numpy(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj
        
        stats = {
            'simulation_info': {
                'duration': int(self.simulation_duration),
                'road_length': int(self.road_length),
                'num_lanes': int(self.num_lanes),
                'operation_mode': self.operation_mode,
                'traffic_config': self.traffic_config,
                'total_vehicles_spawned': int(self.vehicle_counter)
            },
            'traffic_statistics': {
                'avg_density': float(np.mean(self.density_history)) if self.density_history else 0.0,
                'max_density': float(np.max(self.density_history)) if self.density_history else 0.0,
                'min_density': float(np.min(self.density_history)) if self.density_history else 0.0,
                'avg_vehicle_count': float(np.mean(self.vehicle_count_history)) if self.vehicle_count_history else 0.0,
                'max_vehicle_count': int(np.max(self.vehicle_count_history)) if self.vehicle_count_history else 0,
                'min_vehicle_count': int(np.min(self.vehicle_count_history)) if self.vehicle_count_history else 0,
                'avg_speed_kmh': float(np.mean(self.speed_history)) if self.speed_history else 0.0,
                'min_speed_kmh': float(np.min(self.speed_history)) if self.speed_history else 0.0,
                'max_speed_kmh': float(np.max(self.speed_history)) if self.speed_history else 0.0
            }
        }
        
        # Add specific node statistics if in that mode
        if self.operation_mode == 'specific_defined_nodes':
            node_stats = {
                'node_mode': self.traffic_config['node_mode'],
                'node_count': len(self.specific_nodes),
                'node_lane': self.traffic_config.get('lane_number', 1)
            }
            
            if self.traffic_config['node_mode'] == 'static':
                node_stats['node_positions'] = self.traffic_config['node_positions']
            elif self.traffic_config['node_mode'] == 'dynamic':
                node_stats['node_speed_kmh'] = self.traffic_config['node_speed_kmh']
                if self.node_distance_history:
                    node_stats['min_distance'] = float(np.min(self.node_distance_history))
                    node_stats['max_distance'] = float(np.max(self.node_distance_history))
                    node_stats['avg_distance'] = float(np.mean(self.node_distance_history))
            
            stats['node_statistics'] = node_stats
        
        # Add density phases analysis
        stats['density_phases'] = self._analyze_density_phases()
        
        def clean_for_json(obj):
            if isinstance(obj, dict):
                return {k: clean_for_json(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [clean_for_json(v) for v in obj]
            else:
                return convert_numpy(obj)
        
        stats = clean_for_json(stats)
        
        try:
            with open(stats_file, 'w') as f:
                json.dump(stats, f, indent=2)
            
            print(f"Statistics report saved to {stats_file}")
            self._print_summary_report(stats)
            
        except Exception as e:
            print(f"Warning: Could not save statistics report: {e}")
            self._print_basic_statistics()
    
    def _print_summary_report(self, stats: Dict):
        """Print comprehensive summary report"""
        print("\n  Simulation Summary:")
        print(f"   Operation Mode: {stats['simulation_info']['operation_mode']}")
        
        if self.operation_mode == 'specific_defined_nodes':
            node_stats = stats.get('node_statistics', {})
            print(f"   Node Mode: {node_stats.get('node_mode', 'unknown')}")
            print(f"   Node Count: {node_stats.get('node_count', 0)}")
            print(f"   Node Lane: {node_stats.get('node_lane', 1)}")
            
            if node_stats.get('node_mode') == 'static':
                positions = node_stats.get('node_positions', [])
                print(f"   Node Positions: {positions[0]}m and {positions[1]}m")
            elif node_stats.get('node_mode') == 'dynamic':
                speed = node_stats.get('node_speed_kmh', 0)
                print(f"   Node Speed: {speed} km/h")
                if 'min_distance' in node_stats:
                    print(f"   Distance Range: {node_stats['min_distance']:.1f}m - {node_stats['max_distance']:.1f}m")
                    print(f"   Average Distance: {node_stats['avg_distance']:.1f}m")
        else:
            print(f"   Average Density Factor: {stats['traffic_statistics']['avg_density']:.2f}")
            
        print(f"   Vehicle Count Range: {stats['traffic_statistics']['min_vehicle_count']} - {stats['traffic_statistics']['max_vehicle_count']}")
        
        # Calculate vehicles per km per direction for validation
        road_length_km = self.road_length / 1000.0
        avg_vehicles_per_direction = stats['traffic_statistics']['avg_vehicle_count'] / 2
        max_vehicles_per_direction = stats['traffic_statistics']['max_vehicle_count'] / 2
        avg_vehicles_per_km_per_direction = avg_vehicles_per_direction / road_length_km
        max_vehicles_per_km_per_direction = max_vehicles_per_direction / road_length_km
        
        print(f"   Vehicles/km/direction: Avg={avg_vehicles_per_km_per_direction:.0f}, Max={max_vehicles_per_km_per_direction:.0f}")
        print(f"   Speed Range: {stats['traffic_statistics']['min_speed_kmh']:.1f} - {stats['traffic_statistics']['max_speed_kmh']:.1f} km/h")
        
        # Additional info for specific modes
        if self.operation_mode == 'fixed_count':
            target_count = self.traffic_config.get('vehicle_count', 45)
            speed_range = self.traffic_config.get('speed_range_kmh', (30, 120))
            print(f"   Target Vehicle Count: {target_count}")
            print(f"   Target Speed Range: {speed_range[0]}-{speed_range[1]} km/h")
            print(f"   Achievement Rate: {(avg_vehicles_per_direction * 2 / target_count * 100):.1f}%")
        
        print(f"\n  Generation completed successfully!")
    
    def _print_basic_statistics(self):
        """Print basic statistics in case of JSON error"""
        print("  Basic Statistics:")
        if self.density_history:
            print(f"   Average Density Factor: {np.mean(self.density_history):.2f}")
            road_length_km = self.road_length / 1000.0
            avg_vehicles = np.mean(self.vehicle_count_history) if self.vehicle_count_history else 0
            max_vehicles = np.max(self.vehicle_count_history) if self.vehicle_count_history else 0
            avg_per_direction = avg_vehicles / 2
            max_per_direction = max_vehicles / 2
            avg_per_km_per_direction = avg_per_direction / road_length_km
            max_per_km_per_direction = max_per_direction / road_length_km
            print(f"   Vehicles/km/direction: Avg={avg_per_km_per_direction:.0f}, Max={max_per_km_per_direction:.0f}")
        print(f"   Total Vehicles Spawned: {self.vehicle_counter}")
        if self.speed_history:
            print(f"   Average Speed: {np.mean(self.speed_history):.1f} km/h")
    
    def _analyze_density_phases(self) -> List[Dict]:
        """Analyze density changes over time (original method)"""
        phases = []
        if not self.density_history:
            return phases
            
        current_phase = {'start_time': 0.0, 'density': float(self.density_history[0])}
        threshold = 0.1
        
        for i, density in enumerate(self.density_history):
            time = float(i * self.time_step)
            if abs(density - current_phase['density']) > threshold:
                current_phase['end_time'] = time
                current_phase['duration'] = time - current_phase['start_time']
                phases.append(current_phase.copy())
                current_phase = {'start_time': time, 'density': float(density)}
        
        final_time = float(len(self.density_history) * self.time_step)
        current_phase['end_time'] = final_time
        current_phase['duration'] = final_time - current_phase['start_time']
        phases.append(current_phase)
        
        return phases

# =============================================================================
# EXAMPLE CONFIGURATIONS FOR SPECIFIC NODE POSITIONING
# =============================================================================

def create_static_nodes_demo():
    """Create demo for static nodes"""
    print("Creating Static Nodes Demo")
    print("=" * 40)
    
    # Static nodes configuration
    traffic_config = FlexibleTrafficConfig.create_specific_nodes_config(
        node_mode='static',
        node_positions=(100.0, 50.0),  # Two nodes at 100m and 300m
        lane_number=1,                  # Lane 1
        duration=500.0,                 # 5 minutes
        node_types=('car', 'truck'),    # Different types
        node_ids=('static_node_1', 'static_node_2'),
        road_length=500.0,              # 400m road
        enable_background_traffic=False,  # Add some background vehicles
        background_vehicle_count=8
    )
    
    config = {
        'simulation_duration': 500,     # Will be overridden by traffic_config duration
        'road_length': 500,             # Will be overridden by traffic_config road_length
        'num_lanes_per_direction': 2,   # 2 lanes each direction
        'time_step': 1.0,
        'enable_visualization': True,   # ENABLE ANIMATION
        'traffic_config': traffic_config
    }
    
    print("Static Nodes Configuration:")
    print(f"   - Node 1 (car) at {traffic_config['node_positions'][0]}m")
    print(f"   - Node 2 (truck) at {traffic_config['node_positions'][1]}m")
    print(f"   - Lane: {traffic_config['lane_number']}")
    print(f"   - Duration: {traffic_config['duration']} seconds")
    print(f"   - Background traffic: {traffic_config['background_vehicle_count']} vehicles")
    print(f"   - Road length: {traffic_config['road_length']}m")
    
    generator = ComprehensiveFCDGenerator(config)
    generator.generate_fcd('static_nodes 50m distance.xml')

def create_dynamic_nodes_demo():
    """Create demo for dynamic nodes"""
    print("Creating Dynamic Nodes Demo")
    print("=" * 40)
    
    # Dynamic nodes configuration
    traffic_config = FlexibleTrafficConfig.create_specific_nodes_config(
        node_mode='dynamic',
        node_speed_kmh=50.0,            # 50 km/h speed
        lane_number=2,                  # Lane 2
        duration=400.0,                 # 6.67 minutes
        node_types=('car', 'car'),      # Both cars
        node_ids=('dynamic_node_A', 'dynamic_node_B'),
        road_length=500.0,              # 500m road
        enable_background_traffic=True,  # Add some background vehicles
        background_vehicle_count=12
    )
    
    config = {
        'simulation_duration': 400,     # Will be overridden by traffic_config duration
        'road_length': 500,             # Will be overridden by traffic_config road_length
        'num_lanes_per_direction': 2,   # 2 lanes each direction
        'time_step': 0.5,               # Smaller timestep for smoother animation
        'enable_visualization': True,   # ENABLE ANIMATION
        'traffic_config': traffic_config
    }
    
    print("Dynamic Nodes Configuration:")
    print(f"   - Node A starts at 0m, moves right")
    print(f"   - Node B starts at {traffic_config['road_length']}m, moves left")
    print(f"   - Speed: {traffic_config['node_speed_kmh']} km/h ({traffic_config['node_speed_ms']:.2f} m/s)")
    print(f"   - Meeting point: {traffic_config['road_length']/2}m")
    print(f"   - Lane: {traffic_config['lane_number']}")
    print(f"   - Duration: {traffic_config['duration']} seconds")
    print(f"   - Background traffic: {traffic_config['background_vehicle_count']} vehicles")
    print(f"   - Road length: {traffic_config['road_length']}m")
    
    # Calculate expected meeting time
    meeting_distance = traffic_config['road_length'] / 2
    speed_ms = traffic_config['node_speed_ms']
    meeting_time = meeting_distance / speed_ms
    total_journey_time = (traffic_config['road_length'] / speed_ms) * 2  # There and back
    
    print(f"   - Expected meeting time: {meeting_time:.1f} seconds")
    print(f"   - Expected total journey time: {total_journey_time:.1f} seconds")
    
    generator = ComprehensiveFCDGenerator(config)
    generator.generate_fcd('dynamic_nodes_demo.xml')

def create_animation_demo():
    """Create a demo configuration with enhanced animation"""
    
    print("Creating Enhanced Animation Demo")
    print("=" * 50)
    
    # Enhanced animation configuration
    traffic_config = FlexibleTrafficConfig.create_fixed_vehicle_config(
        vehicle_count=35,  # Slightly fewer for clearer animation
        speed_range_kmh=(40, 100),
        vehicle_types={'car': 0.7, 'truck': 0.15, 'bus': 0.1, 'motorcycle': 0.05},
        lane_distribution='balanced',
        behavior_profile='normal',
        maintain_count=True,
        speed_variation=0.2
    )
    
    config = {
        'simulation_duration': 300,  # Shorter for demo
        'road_length': 400,
        'num_lanes_per_direction': 3,
        'time_step': 0.5,  # Smoother animation with smaller timesteps
        'enable_visualization': True,  # ANIMATION ENABLED
        'traffic_config': traffic_config
    }
    
    print("Animation Features:")
    print("   - Real-time vehicle movement with colors and symbols")
    print("   - Speed indicators and vehicle IDs")
    print("   - Lane markings and direction arrows")
    print("   - Live statistics dashboard")
    print("   - Speed distribution chart")
    print("   - Vehicle type breakdown")
    print("\nStarting animation demo...")
    
    generator = ComprehensiveFCDGenerator(config)
    
    # Optional: Enable frame saving for video creation
    # generator.enable_frame_saving(True, 'demo_frames')
    
    generator.generate_fcd('demo_animated_traffic.xml')

def main():
    """Main function with flexible configuration examples including new node positioning modes"""
    
    print(" Comprehensive FCD Generator with Specific Node Positioning")
    print("=" * 70)
    print("Available modes:")
    print("1. Fixed Vehicle Count")
    print("2. Density-based Traffic")
    print("3.  Specific Defined Node Positioning (NEW)")
    print("   - Static Nodes: Two stationary nodes at defined positions")
    print("   - Dynamic Nodes: Two nodes moving towards each other, meeting, and returning")
    print("=" * 70)
    
    # =========================================================================
    # CHOOSE YOUR CONFIGURATION MODE
    # =========================================================================
    
    # OPTION 1: NEW - STATIC NODES DEMO (Two stationary nodes)
    create_static_nodes_demo()
    return
    
    # OPTION 2: NEW - DYNAMIC NODES DEMO (Two nodes meeting in the middle)
    # create_dynamic_nodes_demo()
    # return
    
    # OPTION 3: ORIGINAL ANIMATION DEMO (Regular traffic with visualization)
    # create_animation_demo()
    # return
    
    # =========================================================================
    # OPTION 4: YOUR ORIGINAL REQUEST - 45 cars, 6 lanes, 10000 seconds
    # =========================================================================
    # Uncomment the lines below if you want to run the original configuration
    
    traffic_config = FlexibleTrafficConfig.create_fixed_vehicle_config(
        vehicle_count=45,
        speed_range_kmh=(30, 120),  # 30-120 km/h speed range
        vehicle_types={'car': 0.8, 'truck': 0.1, 'bus': 0.05, 'motorcycle': 0.05},
        lane_distribution='balanced',  # 'balanced', 'heavy_right', 'heavy_left', 'random'
        spawn_pattern='continuous',    # 'continuous', 'burst', 'gradual'
        behavior_profile='normal',     # 'aggressive', 'normal', 'conservative'
        maintain_count=True,          # Keep vehicle count stable
        speed_variation=0.15          # ±15% speed variation
    )
    
    # Base simulation settings
    DURATION = 10000               # 10000 seconds as requested
    ROAD_LENGTH = 500            # 500m road length  
    OUTPUT_FILE = 'fcd_45cars_500m_10000seconds.xml'
    ENABLE_VISUALIZATION = False   # SET TO TRUE TO SEE AMAZING ANIMATION!
    NUM_LANES_PER_DIRECTION = 3  # 3 lanes each direction = 6 total lanes
    
    # =========================================================================
    # MORE EXAMPLE CONFIGURATIONS FOR SPECIFIC NODE POSITIONING
    # =========================================================================
    
    # Example: Custom Static Nodes Configuration
    # traffic_config = FlexibleTrafficConfig.create_specific_nodes_config(
    #     node_mode='static',
    #     node_positions=(150.0, 350.0),  # Nodes at 150m and 350m
    #     lane_number=1,                  # Use lane 1
    #     duration=600.0,                 # 10 minutes
    #     node_types=('car', 'bus'),      # Car and bus
    #     node_ids=('RSU_1', 'RSU_2'),    # Custom IDs (like Road Side Units)
    #     road_length=500.0,
    #     enable_background_traffic=True,
    #     background_vehicle_count=15
    # )
    # DURATION = 600
    # ROAD_LENGTH = 500
    # OUTPUT_FILE = 'static_rsu_nodes.xml'
    # ENABLE_VISUALIZATION = True
    
    # Example: Custom Dynamic Nodes Configuration
    # traffic_config = FlexibleTrafficConfig.create_specific_nodes_config(
    #     node_mode='dynamic',
    #     node_speed_kmh=40.0,            # 40 km/h
    #     lane_number=2,                  # Use lane 2
    #     duration=500.0,                 # Duration
    #     node_types=('motorcycle', 'motorcycle'),  # Two motorcycles
    #     node_ids=('moto_alpha', 'moto_beta'),
    #     road_length=600.0,
    #     enable_background_traffic=True,
    #     background_vehicle_count=20
    # )
    # DURATION = 500
    # ROAD_LENGTH = 600
    # OUTPUT_FILE = 'dynamic_motorcycle_nodes.xml'
    # ENABLE_VISUALIZATION = True
    
    # Example: High-speed Dynamic Nodes for Highway Scenario
    # traffic_config = FlexibleTrafficConfig.create_specific_nodes_config(
    #     node_mode='dynamic',
    #     node_speed_kmh=80.0,            # High speed - 80 km/h
    #     lane_number=1,                  # Fast lane
    #     duration=300.0,                 # Shorter duration due to high speed
    #     node_types=('car', 'car'),
    #     node_ids=('highway_node_1', 'highway_node_2'),
    #     road_length=800.0,              # Longer road for highway
    #     enable_background_traffic=True,
    #     background_vehicle_count=25
    # )
    # DURATION = 300
    # ROAD_LENGTH = 800
    # OUTPUT_FILE = 'highway_dynamic_nodes.xml'
    # ENABLE_VISUALIZATION = True
    
    # =========================================================================
    # OTHER EXAMPLE CONFIGURATIONS (original modes)
    # =========================================================================
    
    # Example: HIGH-SPEED HIGHWAY ANIMATION
    # traffic_config = FlexibleTrafficConfig.create_fixed_vehicle_config(
    #     vehicle_count=30,
    #     speed_range_kmh=(80, 140),
    #     vehicle_types={'car': 0.85, 'truck': 0.15},
    #     behavior_profile='aggressive',
    #     lane_distribution='heavy_right'
    # )
    # ENABLE_VISUALIZATION = True
    # OUTPUT_FILE = 'highway_animation.xml'
    
    # Example: CITY TRAFFIC ANIMATION
    # traffic_config = FlexibleTrafficConfig.create_fixed_vehicle_config(
    #     vehicle_count=60,
    #     speed_range_kmh=(15, 50),
    #     vehicle_types={'car': 0.6, 'truck': 0.2, 'bus': 0.15, 'motorcycle': 0.05},
    #     behavior_profile='conservative',
    #     lane_distribution='balanced'
    # )
    # ENABLE_VISUALIZATION = True
    # OUTPUT_FILE = 'city_traffic_animation.xml'
    
    # Example: DENSITY-BASED LEARNING SCENARIO
    # traffic_config = FlexibleTrafficConfig.create_density_config(
    #     scenario='learning_optimized',  # Gradual density changes for RL
    #     speed_adaptation=True
    # )
    # DURATION = 10000
    # OUTPUT_FILE = 'learning_scenario.xml'
    
    # =========================================================================
    # End of configuration selection
    # =========================================================================
    
    # Adjust timestep for smoother animation
    timestep = 0.5 if ENABLE_VISUALIZATION else 1.0
    
    config = {
        'simulation_duration': DURATION,
        'road_length': ROAD_LENGTH,
        'num_lanes_per_direction': NUM_LANES_PER_DIRECTION,
        'time_step': timestep,
        'enable_visualization': ENABLE_VISUALIZATION,
        'traffic_config': traffic_config
    }
    
    print(f"Selected Configuration:")
    print(f"Mode: {traffic_config['mode']}")
    
    if traffic_config['mode'] == 'fixed_count':
        print(f"Vehicles: {traffic_config['vehicle_count']}")
        print(f"Speed Range: {traffic_config['speed_range_kmh'][0]}-{traffic_config['speed_range_kmh'][1]} km/h")
        print(f"Behavior: {traffic_config['behavior_profile']}")
    elif traffic_config['mode'] == 'specific_defined_nodes':
        node_mode = traffic_config['node_mode']
        print(f"Node Mode: {node_mode}")
        print(f"Lane: {traffic_config['lane_number']}")
        if node_mode == 'static':
            positions = traffic_config['node_positions']
            print(f"Positions: {positions[0]}m & {positions[1]}m")
        else:
            speed = traffic_config['node_speed_kmh']
            print(f"Speed: {speed} km/h")
        print(f"Node Types: {traffic_config['node_types']}")
        print(f"Node IDs: {traffic_config['node_ids']}")
        print(f"Background Traffic: {traffic_config.get('enable_background_traffic', False)}")
    elif traffic_config['mode'] == 'density_based':
        scenario = traffic_config.get('scenario', 'learning_optimized')
        print(f"Scenario: {scenario}")
    
    print(f"Duration: {DURATION} seconds")
    print(f"Road: {ROAD_LENGTH}m with {NUM_LANES_PER_DIRECTION*2} lanes")
    print(f"Output: {OUTPUT_FILE}")
    
    if ENABLE_VISUALIZATION:
        print(f"\n🎬 ANIMATION FEATURES ENABLED:")
        print(f"   - Real-time vehicle movement visualization")
        print(f"   - Beautiful road with lane markings and arrows")
        print(f"   - Vehicle types: Cars, Trucks, Buses, Motorcycles")
        print(f"   - Color-coded vehicles with speed indicators")
        print(f"   - Live statistics and speed distribution charts")
        print(f"   - Vehicle IDs and real-time speed display")
        
        if traffic_config['mode'] == 'specific_defined_nodes':
            print(f"   -  Special node highlighting and tracking")
            print(f"   - Phase information for dynamic nodes")
            print(f"   - Distance tracking between nodes")
            
        print(f"\n TIP: Close the animation window to continue when done watching!")
    
    # Generate FCD with animation
    generator = ComprehensiveFCDGenerator(config)
    generator.generate_fcd(OUTPUT_FILE)

if __name__ == "__main__":
    print(" Welcome to the Enhanced FCD Generator!")
    print("\n" + "="*70)
    print("QUICK START EXAMPLES:")
    print("="*70)
    print(" Currently Running: Static Nodes Demo")
    print("    Two stationary nodes at defined positions")
    print("\n To Switch Modes:")
    print(" For Dynamic Nodes Demo:")
    print("   Comment: create_static_nodes_demo() and return")
    print("   Uncomment: create_dynamic_nodes_demo() and return")
    print("\n For Regular Traffic Animation:")
    print("   Comment: create_static_nodes_demo() and return") 
    print("   Uncomment: create_animation_demo() and return")
    print("\n For Original 45-Car Configuration:")
    print("   Comment: create_static_nodes_demo() and return")
    print("   Comment: # OPTION 4 section (remove # from all lines)")
    print("="*70)
    print("\n Starting Static Nodes Demo...")
    
    main()
