#!/usr/bin/env python3
"""
Comprehensive FCD Data Generator for Variable Density VANET Learning
====================================================================

Generates realistic FCD files with:
- Variable vehicle density over time
- Multiple traffic scenarios (rush hour, congestion, free flow)
- Lane changing behavior
- Speed variations based on traffic conditions
- Realistic acceleration/deceleration
- Different vehicle types and behaviors

Perfect for reinforcement learning training with diverse scenarios.
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

class Vehicle:
    """Represents a single vehicle with realistic behavior"""
    
    def __init__(self, vehicle_id: str, lane: int, initial_x: float, initial_speed: float, 
                 direction: str = 'forward', vehicle_type: str = 'car'):
        self.id = vehicle_id
        self.lane = lane
        self.x = initial_x
        self.y = lane * 3.7  # Lane width = 3.7m
        self.speed = initial_speed
        self.target_speed = initial_speed
        self.direction = direction  # 'forward' or 'backward'
        self.vehicle_type = vehicle_type
        self.length = self._get_vehicle_length()
        self.max_acceleration = 2.5  # m/s²
        self.max_deceleration = 4.5  # m/s²
        self.active = True
        self.lane_change_cooldown = 0
        self.following_distance = 20.0  # Desired following distance
        
    def _get_vehicle_length(self) -> float:
        """Get vehicle length based on type"""
        lengths = {'car': 4.5, 'truck': 12.0, 'bus': 15.0, 'motorcycle': 2.2}
        return lengths.get(self.vehicle_type, 4.5)
    
    def update_position(self, dt: float, traffic_density: float, nearby_vehicles: List['Vehicle']):
        """Update vehicle position with realistic car-following behavior"""
        if not self.active:
            return
            
        # Adjust target speed based on traffic density
        base_speed = 30.0 if self.direction == 'forward' else 28.0  # m/s (108/100 km/h)
        self.target_speed = base_speed * (1.0 - traffic_density * 0.6)
        
        # Car following behavior
        leader = self._find_leader(nearby_vehicles)
        if leader:
            gap = self._calculate_gap(leader)
            safe_speed = self._calculate_safe_speed(gap, leader.speed)
            self.target_speed = min(self.target_speed, safe_speed)
        
        # Smooth acceleration/deceleration
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
            # Reduce speed proportionally
            gap_factor = gap / desired_gap
            return leader_speed * gap_factor * 0.8
        else:
            return leader_speed * 1.1  # Can go slightly faster when gap is large
    
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
        
        # Try adjacent lanes
        for target_lane in [self.lane - 1, self.lane + 1]:
            if self._is_valid_lane(target_lane, num_lanes) and self._is_lane_change_safe(target_lane, nearby_vehicles):
                self.lane = target_lane
                self.y = target_lane * 3.7
                self.lane_change_cooldown = 5.0  # 5 second cooldown
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
        
        # Check gaps in target lane
        for vehicle in target_lane_vehicles:
            gap = abs(vehicle.x - self.x)
            if gap < 15.0:  # Minimum safe gap for lane change
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
    """Main class for generating comprehensive FCD data"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.vehicles: Dict[str, Vehicle] = {}
        self.vehicle_counter = 0
        self.current_time = 0.0
        self.road_length = config.get('road_length', 1000)  # Increased for realism
        self.num_lanes = config.get('num_lanes_per_direction', 3) * 2
        self.simulation_duration = config.get('simulation_duration', 10000)
        self.time_step = config.get('time_step', 1.0)
        self.scenario = TrafficScenario.SCENARIOS.get(config.get('scenario', 'learning_optimized'))
        
        # Visualization
        self.enable_visualization = config.get('enable_visualization', False)
        self.fig = None
        self.ax = None
        self.vehicle_plots = {}
        
        # Statistics
        self.density_history = []
        self.speed_history = []
        self.vehicle_count_history = []
    
    def generate_fcd(self, output_file: str):
        """Generate the complete FCD file"""
        print(f"🚗 Starting FCD generation...")
        print(f"📊 Scenario: {self.scenario['description']}")
        print(f"⏱️  Duration: {self.simulation_duration} seconds")
        print(f"🛣️  Road: {self.road_length}m with {self.num_lanes} lanes")
        
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
            
            # Calculate current traffic density
            current_density = self._calculate_traffic_density(time)
            
            # Manage vehicle spawning and despawning
            self._manage_vehicle_population(current_density)
            
            # Update all vehicles
            active_vehicles = [v for v in self.vehicles.values() if v.active]
            for vehicle in active_vehicles:
                vehicle.update_position(self.time_step, current_density, active_vehicles)
                
                # Attempt lane changes occasionally
                if random.random() < 0.01:  # 1% chance per timestep
                    vehicle.attempt_lane_change(active_vehicles, self.num_lanes)
                
                # Handle boundary conditions
                self._handle_boundary_conditions(vehicle)
            
            # Create timestep element
            timestep = ET.SubElement(root, 'timestep')
            timestep.set('time', f'{time:.2f}')
            
            # Add vehicle data
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
            if self.enable_visualization and i % 10 == 0:  # Update every 10 timesteps
                self._update_visualization(active_vehicles)
            
            # Progress reporting
            if i % 1000 == 0:
                progress = (i / len(time_steps)) * 100
                print(f"⏳ Progress: {progress:.1f}% (Time: {time:.0f}s, Vehicles: {len(active_vehicles)}, Density: {current_density:.2f})")
        
        # Write XML file
        tree = ET.ElementTree(root)
        ET.indent(tree, space="  ", level=0)  # Pretty formatting
        tree.write(output_file, encoding='utf-8', xml_declaration=True)
        
        print(f"✅ FCD data written to {output_file}")
        self._generate_statistics_report(output_file)
    
    def _calculate_traffic_density(self, time: float) -> float:
        """Calculate traffic density based on scenario and time"""
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
                    # Smooth transition to peak density
                    peak_factor = 0.5 * (1 + math.cos(2 * math.pi * (time - start) / (end - start)))
                    density = scenario['base_density'] + (scenario['peak_density'] - scenario['base_density']) * peak_factor
            return density
        
        elif scenario.get('wave_frequency'):
            # Highway congestion with waves
            wave_time = time % scenario['wave_frequency']
            if wave_time < scenario['wave_duration']:
                # Smooth congestion wave
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
        """Manage vehicle spawning and despawning to achieve target density"""
        active_vehicles = [v for v in self.vehicles.values() if v.active]
        current_count = len(active_vehicles)
        
        # Calculate target number of vehicles
        max_vehicles_per_direction = int(self.road_length / 25)  # Rough estimate
        target_count = int(target_density * max_vehicles_per_direction * 2)  # Both directions
        
        if current_count < target_count:
            # Spawn new vehicles
            vehicles_to_spawn = min(5, target_count - current_count)  # Limit spawning rate
            for _ in range(vehicles_to_spawn):
                self._spawn_vehicle()
        
        elif current_count > target_count * 1.2:  # Add hysteresis
            # Remove some vehicles (simulate vehicles leaving)
            vehicles_to_remove = min(3, current_count - target_count)
            inactive_candidates = [v for v in active_vehicles if self._can_despawn(v)]
            for vehicle in random.sample(inactive_candidates, min(vehicles_to_remove, len(inactive_candidates))):
                vehicle.active = False
    
    def _spawn_vehicle(self):
        """Spawn a new vehicle"""
        # Choose direction and lane
        direction = random.choice(['forward', 'backward'])
        if direction == 'forward':
            lane = random.randint(1, self.num_lanes // 2)
            start_x = -50  # Start before road begins
        else:
            lane = random.randint(self.num_lanes // 2 + 1, self.num_lanes)
            start_x = self.road_length + 50  # Start after road ends
        
        # Choose vehicle type and initial speed
        vehicle_type = random.choices(['car', 'truck', 'bus'], weights=[0.8, 0.15, 0.05])[0]
        initial_speed = random.uniform(20, 35)  # m/s
        
        # Create vehicle
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
    
    def _update_statistics(self, density: float, vehicles: List[Vehicle]):
        """Update simulation statistics"""
        self.density_history.append(density)
        self.vehicle_count_history.append(len(vehicles))
        
        if vehicles:
            avg_speed = np.mean([v.speed for v in vehicles])
            self.speed_history.append(avg_speed * 3.6)  # Convert to km/h
        else:
            self.speed_history.append(0)
    
    def _setup_visualization(self):
        """Setup matplotlib visualization"""
        self.fig, self.ax = plt.subplots(figsize=(15, 8))
        self.ax.set_xlim(0, self.road_length)
        self.ax.set_ylim(0, self.num_lanes * 3.7)
        self.ax.set_xlabel('Road Length (m)')
        self.ax.set_ylabel('Lane Position (m)')
        self.ax.set_title('Real-time Vehicle Movement Simulation')
        
        # Draw lane lines
        for lane in range(self.num_lanes + 1):
            y = lane * 3.7
            self.ax.axhline(y=y, color='k', linestyle='--', alpha=0.5)
        
        plt.ion()
        plt.show()
    
    def _update_visualization(self, vehicles: List[Vehicle]):
        """Update visualization with current vehicle positions"""
        if not self.enable_visualization:
            return
            
        # Clear previous vehicle plots
        for plot in self.vehicle_plots.values():
            plot.remove()
        self.vehicle_plots.clear()
        
        # Plot current vehicles
        colors = {'car': 'blue', 'truck': 'red', 'bus': 'green'}
        for vehicle in vehicles:
            if 0 <= vehicle.x <= self.road_length:  # Only plot vehicles on road
                color = colors.get(vehicle.vehicle_type, 'blue')
                plot = self.ax.plot(vehicle.x, vehicle.y, 'o', color=color, markersize=8)[0]
                self.vehicle_plots[vehicle.id] = plot
        
        # Update title with current statistics
        self.ax.set_title(f'Time: {self.current_time:.0f}s | Vehicles: {len(vehicles)} | '
                         f'Avg Speed: {np.mean([v.speed for v in vehicles]) * 3.6:.1f} km/h')
        
        plt.pause(0.01)
    
    def _generate_statistics_report(self, output_file: str):
        """Generate statistics report"""
        stats_file = output_file.replace('.xml', '_statistics.json')
        
        stats = {
            'simulation_info': {
                'duration': self.simulation_duration,
                'road_length': self.road_length,
                'num_lanes': self.num_lanes,
                'scenario': self.config.get('scenario', 'learning_optimized'),
                'total_vehicles_spawned': self.vehicle_counter
            },
            'traffic_statistics': {
                'avg_density': np.mean(self.density_history),
                'max_density': np.max(self.density_history),
                'min_density': np.min(self.density_history),
                'avg_vehicle_count': np.mean(self.vehicle_count_history),
                'max_vehicle_count': np.max(self.vehicle_count_history),
                'avg_speed_kmh': np.mean(self.speed_history),
                'min_speed_kmh': np.min(self.speed_history),
                'max_speed_kmh': np.max(self.speed_history)
            },
            'density_phases': self._analyze_density_phases()
        }
        
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
        
        print(f"📊 Statistics report saved to {stats_file}")
        print("\n🎯 Simulation Summary:")
        print(f"   Average Density: {stats['traffic_statistics']['avg_density']:.2f}")
        print(f"   Vehicle Count Range: {stats['traffic_statistics']['max_vehicle_count']} - {np.min(self.vehicle_count_history)}")
        print(f"   Speed Range: {stats['traffic_statistics']['min_speed_kmh']:.1f} - {stats['traffic_statistics']['max_speed_kmh']:.1f} km/h")
    
    def _analyze_density_phases(self) -> List[Dict]:
        """Analyze density changes over time"""
        phases = []
        if not self.density_history:
            return phases
            
        current_phase = {'start_time': 0, 'density': self.density_history[0]}
        threshold = 0.1  # Density change threshold
        
        for i, density in enumerate(self.density_history):
            time = i * self.time_step
            if abs(density - current_phase['density']) > threshold:
                # End current phase
                current_phase['end_time'] = time
                current_phase['duration'] = time - current_phase['start_time']
                phases.append(current_phase.copy())
                
                # Start new phase
                current_phase = {'start_time': time, 'density': density}
        
        # Close final phase
        current_phase['end_time'] = len(self.density_history) * self.time_step
        current_phase['duration'] = current_phase['end_time'] - current_phase['start_time']
        phases.append(current_phase)
        
        return phases

def main():
    """Main function with direct configuration - EDIT THESE SETTINGS"""
    
    # =========================================================================
    # 🔧 EDIT THESE SETTINGS DIRECTLY (No command line needed!)
    # =========================================================================
    
    # Choose scenario (change this to what you want):
    # 'learning_optimized' - Best for RL training (gradual difficulty)
    # 'rush_hour' - Morning/evening peaks  
    # 'highway_congestion' - Periodic congestion waves
    # 'city_traffic' - Traffic light cycles
    # 'random_varying' - Unpredictable changes
    SCENARIO = 'learning_optimized'
    
    # Simulation settings
    DURATION = 10000                # Simulation time in seconds
    ROAD_LENGTH = 1000             # Road length in meters
    OUTPUT_FILE = 'training_fcd_data.xml'  # Output filename
    ENABLE_VISUALIZATION = False    # Set to True to see real-time animation
    TIME_STEP = 1.0                # Time step in seconds
    NUM_LANES_PER_DIRECTION = 3    # Lanes per direction (3 = 6 total lanes)
    
    # =========================================================================
    # End of settings - script runs automatically
    # =========================================================================
    
    # Configuration (automatically uses your settings above)
    config = {
        'scenario': SCENARIO,
        'simulation_duration': DURATION,
        'road_length': ROAD_LENGTH,
        'num_lanes_per_direction': NUM_LANES_PER_DIRECTION,
        'time_step': TIME_STEP,
        'enable_visualization': ENABLE_VISUALIZATION
    }
    
    print("🚗 Comprehensive FCD Generator for Variable Density Learning")
    print("=" * 60)
    print(f"📊 Scenario: {SCENARIO}")
    print(f"⏱️  Duration: {DURATION} seconds")
    print(f"🛣️  Road: {ROAD_LENGTH}m with {NUM_LANES_PER_DIRECTION*2} lanes")
    print(f"📁 Output: {OUTPUT_FILE}")
    print(f"👁️  Visualization: {'Enabled' if ENABLE_VISUALIZATION else 'Disabled'}")
    
    # Generate FCD
    generator = ComprehensiveFCDGenerator(config)
    generator.generate_fcd(OUTPUT_FILE)
    
    print("\n✅ FCD generation completed successfully!")
    print(f"📁 Output files:")
    print(f"   - FCD data: {OUTPUT_FILE}")
    print(f"   - Statistics: {OUTPUT_FILE.replace('.xml', '_statistics.json')}")

if __name__ == "__main__":
    main()