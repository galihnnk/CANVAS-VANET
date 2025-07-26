#!/usr/bin/env python3
"""
IEEE 802.11bd VANET Simulation with Layer 3 Stack and SDN Capabilities
Enhanced with complete networking stack and Software Defined Networking, Enable VANET DDOS Simulation Attack
Various Traffic, and Visualization.  by Galih Nugraha Nurkahfi, galih.nugraha.nurkahfi@brin.go.id
"""

import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET
import math
import random
import socket
import json
import time
from typing import Dict, List, Tuple, Any, Optional, Set
from dataclasses import dataclass, field
import os
from datetime import datetime
import threading
import queue
import heapq
from collections import defaultdict, deque
import networkx as nx
from enum import Enum
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import ListedColormap, Normalize
from matplotlib.animation import FuncAnimation
import matplotlib.gridspec as gridspec

import matplotlib
matplotlib.use('Qt5Agg')  # Force Qt backend for better interactivity
from dataclasses import dataclass, field



# ===============================================================================
# SIMULATION CONFIGURATION - MODIFY THESE PARAMETERS
# ===============================================================================

# FILE AND OUTPUT CONFIGURATION
FCD_FILE = "fcd_training_data.xml"  # Path to your FCD XML file
OUTPUT_FILENAME = "Baseline-trainingdata-omnidirectional-L3.xlsx"  # Set to None for automatic naming, or specify custom name

# FCD DATA RELOADING CONFIGURATION
FCD_RELOAD_COUNT = 1  # Number of times to reload FCD data
FCD_RELOAD_VEHICLE_ID_STRATEGY = "suffix"  # "suffix" or "reuse"

# REAL-TIME OUTPUT CONFIGURATION
ENABLE_REALTIME_CSV = False  # Enable CSV output per timestamp
CSV_UPDATE_FREQUENCY = 1  # Write CSV every N timestamps
EXCEL_UPDATE_FREQUENCY = 1  # Update Excel file every N timestamps

# RL INTEGRATION CONFIGURATION
ENABLE_RL = False  # Set to True to enable RL optimization
RL_HOST = '127.0.0.1'  # RL server host address
RL_PORT = 5005  # RL server port

# NEW: LAYER 3 NETWORKING CONFIGURATION
ENABLE_LAYER3 = True  # Enable Layer 3 networking stack
ROUTING_PROTOCOL = "HYBRID"  # Options: "AODV", "OLSR", "GEOGRAPHIC", "HYBRID"
ENABLE_MULTI_HOP = True  # Enable multi-hop communication
MAX_HOP_COUNT = 5  # Maximum number of hops for route discovery
ROUTE_DISCOVERY_TIMEOUT = 3.0  # Route discovery timeout in seconds
HELLO_INTERVAL = 1.0  # Hello message interval for topology discovery
TOPOLOGY_UPDATE_INTERVAL = 2.0  # Network topology update interval

# NEW: SDN CONFIGURATION
ENABLE_SDN = False  # Enable Software Defined Networking
SDN_CONTROLLER_TYPE = "CENTRALIZED"  # Options: "CENTRALIZED", "DISTRIBUTED", "HYBRID"
SDN_CONTROL_PROTOCOL = "OpenFlow-VANET"  # Control protocol for VANET SDN
FLOW_TABLE_SIZE = 1000  # Maximum flow table entries per vehicle
FLOW_RULE_TIMEOUT = 30.0  # Flow rule timeout in seconds
SDN_UPDATE_INTERVAL = 1.0  # SDN controller update interval
ENABLE_QOS_MANAGEMENT = True  # Enable QoS flow management
ENABLE_TRAFFIC_ENGINEERING = True  # Enable traffic engineering optimization

# PACKET SIMULATION CONFIGURATION
ENABLE_PACKET_SIMULATION = True  # Enable detailed packet-level simulation
PACKET_GENERATION_RATE = 5.0  # Packets per second per vehicle
PACKET_SIZE_BYTES = 512  # Average packet size in bytes
APPLICATION_TYPES = ["SAFETY", "INFOTAINMENT", "SENSING"]  # Supported applications
QOS_CLASSES = ["EMERGENCY", "SAFETY", "SERVICE", "BACKGROUND"]  # QoS priority classes
TRAFFIC_PATTERNS = ["CBR", "POISSON", "BURSTY"]  # Traffic generation patterns

# SIMULATION PARAMETERS
RANDOM_SEED = 42  # For reproducible results
TIME_STEP = 1.0  # Simulation time step in seconds

# PHY LAYER CONFIGURATION (IEEE 802.11bd compliant)
TRANSMISSION_POWER_DBM = 20.0
BANDWIDTH = 10e6
NOISE_FIGURE = 9.0
CHANNEL_MODEL = "highway_los"
MCS = 1
BEACON_RATE = 10.0
APPLICATION_TYPE = "safety"
FREQUENCY = 5.9e9

# ANTENNA CONFIGURATION
TRANSMITTER_GAIN = 2.15
RECEIVER_GAIN = 2.15

# IEEE 802.11bd PHY ENHANCEMENTS CONFIGURATION
ENABLE_LDPC = True
ENABLE_MIDAMBLES = True
ENABLE_DCM = True
ENABLE_EXTENDED_RANGE = True
ENABLE_MIMO_STBC = False

# MAC LAYER CONFIGURATION
SLOT_TIME = 9e-6
SIFS = 16e-6
DIFS = 34e-6
CONTENTION_WINDOW_MIN = 15
CONTENTION_WINDOW_MAX = 1023
RETRY_LIMIT = 7
MAC_HEADER_BYTES = 36

# PROPAGATION MODEL CONFIGURATION
PATH_LOSS_EXPONENT = 2.0
WAVELENGTH = 0.0508
RECEIVER_SENSITIVITY_DBM = -89

# BACKGROUND TRAFFIC CONFIGURATION
BACKGROUND_TRAFFIC_LOAD = 0.1
HIDDEN_NODE_FACTOR = 0.15
INTER_SYSTEM_INTERFERENCE = 0.05

# ENHANCED INTERFERENCE MODELING PARAMETERS
THERMAL_NOISE_DENSITY = -174
INTERFERENCE_THRESHOLD_DB = -95
FADING_MARGIN_DB = 10
SHADOWING_STD_DB = 4

#RL Debug
RL_DEBUG_LOGGING = True  # Enable detailed RL communication logging
RL_LOG_FREQUENCY = 1    # Log every N RL communications

# ===============================================================================
# ANTENNA CONFIGURATION
# ===============================================================================

# ANTENNA TYPE CONFIGURATION
ANTENNA_TYPE = "OMNIDIRECTIONAL"  # Options: "OMNIDIRECTIONAL", "SECTORAL"

# RL-controlled vs static sectors for sectoral antennas
RL_CONTROLLED_SECTORS = ['front', 'rear']  # Only these will be adjusted by RL
RL_STATIC_SECTORS = ['left', 'right']     # These remain static
SIDE_ANTENNA_STATIC_POWER = 5.0          # Static power for side antennas (dBm) - INCREASED from 3.0

# FAIR STATIC BASELINE
SECTORAL_ANTENNA_CONFIG = {
    "front": {"power_dbm": 15.0, "gain_db": 8.0, "beamwidth_deg": 60, "enabled": True},  # 23.0 dBm EIRP
    "rear": {"power_dbm": 15.0, "gain_db": 8.0, "beamwidth_deg": 60, "enabled": True},   # 23.0 dBm EIRP  
    "left": {"power_dbm": 5.0, "gain_db": 5.0, "beamwidth_deg": 90, "enabled": True},    # 10.0 dBm EIRP
    "right": {"power_dbm": 5.0, "gain_db": 5.0, "beamwidth_deg": 90, "enabled": True}    # 10.0 dBm EIRP
}

OMNIDIRECTIONAL_ANTENNA_CONFIG = {
    "power_dbm": 20.0,    # 23 dBm EIRP uniform
    "gain_db": 3
}

# ENHANCED VISUALIZATION CONFIGURATION
LIVE_VISUALIZATION = False
SEPARATE_PLOT_WINDOWS = True
VISUALIZATION_OUTPUT_DIR = "visualization_output"
L3_PATH_VISUALIZATION = True
PLOT_UPDATE_INTERVAL = 500  # Update plots every N timesteps
SAVE_PLOT_FRAMES = True

# In configuration section
VISUALIZATION_CONFIG = {
    'enabled': False,           # Master switch
    'live_plots': False,        # Live plotting
    'save_frames': False,       # Frame saving
    'separate_windows': False,  # Multiple windows
    'update_interval': 500      # Update frequency
}

# ===============================================================================
# DoS/DDoS ATTACK CONFIGURATION
# ===============================================================================

# ATTACK SIMULATION CONFIGURATION
ENABLE_ATTACK_SIMULATION = False  # Set to True to enable attack simulation
ATTACK_TYPE = "COMBINED"  # Options: "BEACON_FLOODING", "HIGH_POWER_JAMMING", "ASYNC_BEACON", "COMBINED"
NUMBER_OF_ATTACKERS = 50  # Number of attacker vehicles (max = total vehicles)
ATTACKER_SELECTION_STRATEGY = "RANDOM"  # Options: "RANDOM", "CENTRAL", "DISTRIBUTED", "MANUAL"
ATTACKER_IDS = []  # Manual attacker IDs if using "MANUAL" strategy

# BEACON FLOODING ATTACK PARAMETERS
FLOODING_BEACON_RATE = 100.0  # Flooding beacon rate (Hz) - normal is 10 Hz
FLOODING_PACKET_SIZE = 1000   # Flooding packet size (bytes) - normal is 100-300 bytes
FLOODING_DURATION_RATIO = 0.8  # Fraction of simulation time to attack (0.0-1.0)

# HIGH POWER JAMMING ATTACK PARAMETERS
JAMMING_POWER_DBM = 40.0      # Jamming transmission power (dBm) - normal is 20 dBm
JAMMING_BANDWIDTH_RATIO = 1.5  # Bandwidth occupation ratio
JAMMING_CONTINUOUS_MODE = True  # True for continuous, False for intermittent

# ASYNC BEACON ATTACK PARAMETERS
ASYNC_MIN_INTERVAL = 0.01     # Minimum beacon interval (seconds)
ASYNC_MAX_INTERVAL = 1.0      # Maximum beacon interval (seconds)
ASYNC_RANDOMNESS_FACTOR = 0.9  # Randomness factor (0.0-1.0)

# COMBINED ATTACK PARAMETERS
COMBINED_ATTACK_WEIGHTS = {
    "flooding": 0.4,
    "jamming": 0.3,
    "async": 0.3
}

# ATTACK DETECTION DATASET CONFIGURATION
GENERATE_ATTACK_DATASET = False  # Generate ML dataset for attack detection
DETECTION_WINDOW_SIZE = 10      # Time window for feature calculation (seconds)
DETECTION_FEATURES_UPDATE_INTERVAL = 1  # Feature calculation interval

# VISUALIZATION CONFIGURATION
ENABLE_VISUALIZATION = False      # Enable/disable real-time visualization
VISUALIZATION_UPDATE_FREQ = 500    # Update every N timesteps (higher = faster simulation)
SAVE_VISUALIZATION_FRAMES = False # Save frames periodically
SHOW_COMMUNICATION_RANGES = False # Show communication range circles
SHOW_NEIGHBOR_CONNECTIONS = False # Show lines between neighbors


# ===============================================================================
# ENHANCED BACKGROUND TRAFFIC CONFIGURATION
# ===============================================================================

# BACKGROUND TRAFFIC COMPOSITION
ENABLE_BACKGROUND_TRAFFIC_MODELING = True
BACKGROUND_TRAFFIC_TOTAL_LOAD = 0.2  # Total background traffic load (0.0 to 1.0)

# MANAGEMENT TRAFFIC CONFIGURATION
# ENHANCED BACKGROUND TRAFFIC CONFIGURATION - HIGHER RATES
MANAGEMENT_TRAFFIC_CONFIG = {
    "enabled": True,
    "load_ratio": 0.4,
    "packet_types": {
        "routing_updates": {
            "rate_hz": 2.0,  #
            "packet_size_bytes": 64,  # Increased from 64
            "qos_priority": "high",
            "periodic": True
        },
        "topology_discovery": {
            "rate_hz": 1.0,  # Increased from 1.0
            "packet_size_bytes": 128,  # Increased from 128
            "qos_priority": "high", 
            "periodic": True
        },
        "channel_coordination": {
            "rate_hz": 5.0,  # Increased from 5.0
            "packet_size_bytes": 32,  # Increased from 32
            "qos_priority": "medium",
            "periodic": True
        },
        "security_messages": {
            "rate_hz": 5.0,  # Increased from 0.5
            "packet_size_bytes": 256,  # Increased from 256
            "qos_priority": "high",
            "periodic": True
        },
        "cooperative_awareness": {
            "rate_hz": 4.0,  # Increased from 4.0
            "packet_size_bytes": 200,  # Increased from 200
            "qos_priority": "medium",
            "periodic": True
        }
    }
}

INFOTAINMENT_TRAFFIC_CONFIG = {
    "enabled": True,
    "load_ratio": 0.6,
    "packet_types": {
        "internet_browsing": {
            "rate_hz": 10.0,  # Increased from 10.0
            "packet_size_bytes": 512,  # Increased from 512
            "qos_priority": "low",
            "periodic": False,
            "burst_probability": 0.3  # Increased from 0.3
        },
        "video_streaming": {
            "rate_hz": 30.0,  # Increased from 30.0
            "packet_size_bytes": 1200,  # Increased from 1200
            "qos_priority": "medium",
            "periodic": True,
            "enable_probability": 0.2  # Increased from 0.2
        },
        "audio_streaming": {
            "rate_hz": 50.0,  # Increased from 50.0
            "packet_size_bytes": 160,  # Increased from 160
            "qos_priority": "medium", 
            "periodic": True,
            "enable_probability": 0.15  # Increased from 0.15
        },
        "file_transfer": {
            "rate_hz": 20.0,  # Increased from 20.0
            "packet_size_bytes": 1400,  # Same but with higher rate
            "qos_priority": "low",
            "periodic": False,
            "burst_probability": 0.1,  # Increased from 0.1
            "burst_duration": 5.0  # Increased from 5.0
        },
        "navigation_updates": {
            "rate_hz": 1.0,  # Increased from 0.2
            "packet_size_bytes": 300,  # Increased from 300
            "qos_priority": "medium",
            "periodic": True
        },
        "social_media": {
            "rate_hz": 0.5,  # Increased from 0.1
            "packet_size_bytes": 800,  # Increased from 800
            "qos_priority": "low",
            "periodic": False,
            "burst_probability": 0.05  # Increased from 0.05
        }
    }
}

# BACKGROUND TRAFFIC INTERFERENCE MODELING
BACKGROUND_INTERFERENCE_CONFIG = {
    "adjacent_channel_interference": 0.15,  # From adjacent frequency channels
    "co_channel_interference": 0.25,       # From same frequency different cells
    "non_vanet_interference": 0.1,         # From WiFi, cellular, etc.
    "hidden_terminal_factor": 0.2,         # Enhanced hidden terminal effect
    "capture_effect_threshold": 6.0        # dB threshold for capture effect
}

# ===============================================================================
# END OF CONFIGURATION - DO NOT MODIFY BELOW THIS LINE
# ===============================================================================

# Set random seed for reproducibility
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# NEW: Layer 3 and SDN Enumerations
class RoutingProtocol(Enum):
    AODV = "AODV"
    OLSR = "OLSR"
    GEOGRAPHIC = "GEOGRAPHIC"
    HYBRID = "HYBRID"

class PacketType(Enum):
    DATA = "DATA"
    RREQ = "RREQ"  # Route Request
    RREP = "RREP"  # Route Reply
    RERR = "RERR"  # Route Error
    HELLO = "HELLO"
    TC = "TC"  # Topology Control (OLSR)
    BEACON = "BEACON"
    SDN_CONTROL = "SDN_CONTROL"

class QoSClass(Enum):
    EMERGENCY = 0  # Highest priority
    SAFETY = 1
    SERVICE = 2
    BACKGROUND = 3  # Lowest priority

class FlowState(Enum):
    ACTIVE = "ACTIVE"
    PENDING = "PENDING"
    EXPIRED = "EXPIRED"
    DELETED = "DELETED"

# NEW: Network Packet Class
@dataclass
class NetworkPacket:
    """Network layer packet with full L3 information"""
    packet_id: str
    packet_type: PacketType
    source_id: str
    destination_id: str
    source_ip: str
    destination_ip: str
    payload_size: int
    qos_class: QoSClass
    application_type: str
    ttl: int = 64
    hop_count: int = 0
    creation_time: float = 0.0
    route: List[str] = field(default_factory=list)
    flow_id: str = ""
    sequence_number: int = 0
    
    # Routing protocol specific fields
    rreq_id: int = 0  # For AODV
    destination_sequence: int = 0  # For AODV
    originator_addr: str = ""  # For OLSR
    ansn: int = 0  # For OLSR
    
    # Performance tracking
    delay: float = 0.0
    delivery_status: str = "PENDING"
    
    def __post_init__(self):
        if not self.flow_id:
            self.flow_id = f"{self.source_id}_{self.destination_id}_{self.application_type}"

# NEW: Flow Table Entry
@dataclass
class FlowEntry:
    """SDN Flow table entry"""
    flow_id: str
    match_fields: Dict[str, Any]  # Match criteria
    actions: List[Dict[str, Any]]  # Actions to perform
    priority: int
    timeout: float
    creation_time: float
    packet_count: int = 0
    byte_count: int = 0
    last_used: float = 0.0
    state: FlowState = FlowState.ACTIVE
    qos_requirements: Dict[str, float] = field(default_factory=dict)

# NEW: Routing Table Entry
@dataclass
class RouteEntry:
    """Layer 3 routing table entry"""
    destination: str
    next_hop: str
    hop_count: int
    metric: float
    sequence_number: int
    lifetime: float
    route_type: str = "DYNAMIC"  # STATIC, DYNAMIC, SDN
    interface: str = "wlan0"
    timestamp: float = 0.0

# NEW: Network Topology Node
@dataclass
class TopologyNode:
    """Network topology node information"""
    node_id: str
    ip_address: str
    position: Tuple[float, float]
    last_seen: float
    neighbors: Set[str] = field(default_factory=set)
    link_quality: Dict[str, float] = field(default_factory=dict)
    routing_table: Dict[str, RouteEntry] = field(default_factory=dict)
    flow_table: Dict[str, FlowEntry] = field(default_factory=dict)

def safe_field_access(obj, field_name, default_value):
    """Safely access object field with default value"""
    if hasattr(obj, field_name):
        value = getattr(obj, field_name)
        if value is not None and not (isinstance(value, float) and math.isnan(value)):
            return value
    return default_value

def bound_value(value, min_val, max_val):
    """Bound value within min/max range"""
    return max(min_val, min(max_val, value))

def enforce_vehicle_format(vehicle_id):
    """Ensure vehicle ID is in proper format"""
    if isinstance(vehicle_id, str):
        if not vehicle_id.startswith('veh'):
            return f"veh{vehicle_id}"
        return vehicle_id
    else:
        return f"veh{vehicle_id}"

@dataclass
class SimulationConfig:
    """ENHANCED IEEE 802.11bd configuration with improved PHY/MAC parameters"""
    # Basic simulation parameters
    time_step: float = TIME_STEP
    
    # PHY parameters (IEEE 802.11bd specific - CORRECTED from script 2)
    transmission_power_dbm: float = TRANSMISSION_POWER_DBM
    bandwidth: float = BANDWIDTH
    noise_figure: float = NOISE_FIGURE
    channel_model: str = CHANNEL_MODEL
    mcs: int = MCS
    beacon_rate: float = BEACON_RATE
    application_type: str = APPLICATION_TYPE
    frequency: float = FREQUENCY
    
    # Dynamic payload length based on application type
    @property
    def payload_length(self) -> int:
        if self.application_type == "safety":
            return 100
        elif self.application_type == "high_throughput":
            return 300
        else:
            return 100
    
    # IEEE 802.11bd PHY enhancements
    enable_ldpc: bool = ENABLE_LDPC
    enable_midambles: bool = ENABLE_MIDAMBLES
    enable_dcm: bool = ENABLE_DCM
    enable_extended_range: bool = ENABLE_EXTENDED_RANGE
    enable_mimo_stbc: bool = ENABLE_MIMO_STBC
    
    # Antenna gains
    g_t: float = TRANSMITTER_GAIN
    g_r: float = RECEIVER_GAIN
    
    # MAC parameters (IEEE 802.11bd correct values from script 2)
    slot_time: float = SLOT_TIME
    sifs: float = SIFS
    difs: float = DIFS
    cw_min: int = CONTENTION_WINDOW_MIN
    cw_max: int = CONTENTION_WINDOW_MAX
    retry_limit: int = RETRY_LIMIT
    mac_header_bytes: int = MAC_HEADER_BYTES
    
    # Path loss and communication range parameters
    path_loss_exponent: float = PATH_LOSS_EXPONENT
    wavelength: float = WAVELENGTH
    receiver_sensitivity_dbm: float = RECEIVER_SENSITIVITY_DBM
    
    # BALANCED: More realistic background factors (from script 2)
    background_traffic_load: float = 0.25  # BALANCED
    hidden_node_factor: float = 0.12       # BALANCED  
    inter_system_interference: float = 0.035  # BALANCED
    
    # Enhanced interference modeling parameters
    thermal_noise_density: float = THERMAL_NOISE_DENSITY
    interference_threshold_db: float = INTERFERENCE_THRESHOLD_DB
    fading_margin_db: float = FADING_MARGIN_DB
    shadowing_std_db: float = SHADOWING_STD_DB
    
    # Layer 3 networking parameters (keep from script 1)
    enable_layer3: bool = ENABLE_LAYER3
    routing_protocol: str = ROUTING_PROTOCOL
    enable_multi_hop: bool = ENABLE_MULTI_HOP
    max_hop_count: int = MAX_HOP_COUNT
    route_discovery_timeout: float = ROUTE_DISCOVERY_TIMEOUT
    hello_interval: float = HELLO_INTERVAL
    topology_update_interval: float = TOPOLOGY_UPDATE_INTERVAL
    
    # SDN parameters (keep from script 1)
    enable_sdn: bool = ENABLE_SDN
    sdn_controller_type: str = SDN_CONTROLLER_TYPE
    sdn_control_protocol: str = SDN_CONTROL_PROTOCOL
    flow_table_size: int = FLOW_TABLE_SIZE
    flow_rule_timeout: float = FLOW_RULE_TIMEOUT
    sdn_update_interval: float = SDN_UPDATE_INTERVAL
    enable_qos_management: bool = ENABLE_QOS_MANAGEMENT
    enable_traffic_engineering: bool = ENABLE_TRAFFIC_ENGINEERING
    
    # Packet simulation parameters (keep from script 1)
    enable_packet_simulation: bool = ENABLE_PACKET_SIMULATION
    packet_generation_rate: float = PACKET_GENERATION_RATE
    packet_size_bytes: int = PACKET_SIZE_BYTES
    application_types: List[str] = field(default_factory=lambda: APPLICATION_TYPES)
    qos_classes: List[str] = field(default_factory=lambda: QOS_CLASSES)
    traffic_patterns: List[str] = field(default_factory=lambda: TRAFFIC_PATTERNS)
    
    


# ===============================================================================
# BACKGROUND TRAFFIC CLASS
# ===============================================================================

class BackgroundTrafficType(Enum):
    MANAGEMENT = "MANAGEMENT"
    INFOTAINMENT = "INFOTAINMENT"

class TrafficPriority(Enum):
    HIGH = "high"
    MEDIUM = "medium" 
    LOW = "low"

@dataclass
class BackgroundPacket:
    """Background traffic packet representation"""
    packet_id: str
    traffic_type: BackgroundTrafficType
    packet_subtype: str  # e.g., "routing_updates", "video_streaming"
    size_bytes: int
    priority: TrafficPriority
    generation_time: float
    source_vehicle: str
    is_periodic: bool
    burst_id: str = None
    
class BackgroundTrafficGenerator:
    """Generates realistic VANET background traffic"""
    
    def __init__(self, vehicle_id: str, config: SimulationConfig):
        self.vehicle_id = vehicle_id
        self.config = config
        
        # Traffic state tracking
        self.management_last_generated = {}
        self.infotainment_last_generated = {}
        self.active_bursts = {}  # Track ongoing burst transmissions
        self.packet_sequence = 0
        
        # Vehicle-specific traffic characteristics
        self._initialize_vehicle_traffic_profile()
        
        # Statistics
        self.generated_packets = {
            BackgroundTrafficType.MANAGEMENT: 0,
            BackgroundTrafficType.INFOTAINMENT: 0
        }
        self.total_bytes_generated = 0
        
    def _initialize_vehicle_traffic_profile(self):
        """Initialize vehicle-specific traffic characteristics"""
        # Some vehicles are more active in certain traffic types
        self.management_activity_factor = random.uniform(0.8, 1.2)
        self.infotainment_activity_factor = random.uniform(0.5, 1.5)
        
        # Determine which infotainment services this vehicle uses
        self.active_infotainment_services = set()
        
        for service, config in INFOTAINMENT_TRAFFIC_CONFIG["packet_types"].items():
            enable_prob = config.get("enable_probability", 1.0)
            if random.random() < enable_prob:
                self.active_infotainment_services.add(service)
        
        print(f"[BACKGROUND TRAFFIC] Vehicle {self.vehicle_id} active services: {self.active_infotainment_services}")
    
    def generate_background_packets(self, current_time: float, neighbors_count: int) -> List[BackgroundPacket]:
        """Generate background traffic packets for current timestep"""
        if not ENABLE_BACKGROUND_TRAFFIC_MODELING:
            return []
        
        generated_packets = []
        
        # Generate management traffic
        if MANAGEMENT_TRAFFIC_CONFIG["enabled"]:
            management_packets = self._generate_management_traffic(current_time, neighbors_count)
            generated_packets.extend(management_packets)
        
        # Generate infotainment traffic  
        if INFOTAINMENT_TRAFFIC_CONFIG["enabled"]:
            infotainment_packets = self._generate_infotainment_traffic(current_time, neighbors_count)
            generated_packets.extend(infotainment_packets)
        
        # Update statistics
        for packet in generated_packets:
            self.generated_packets[packet.traffic_type] += 1
            self.total_bytes_generated += packet.size_bytes
        
        return generated_packets
    
    def _generate_management_traffic(self, current_time: float, neighbors_count: int) -> List[BackgroundPacket]:
        """Generate VANET management traffic"""
        packets = []
        
        for packet_type, type_config in MANAGEMENT_TRAFFIC_CONFIG["packet_types"].items():
            
            # Adjust rate based on network conditions
            base_rate = type_config["rate_hz"] * self.management_activity_factor
            
            # Some management traffic scales with neighbor count
            if packet_type in ["topology_discovery", "channel_coordination"]:
                neighbor_factor = 1.0 + (neighbors_count * 0.1)  # More neighbors = more management traffic
                effective_rate = base_rate * neighbor_factor
            else:
                effective_rate = base_rate
            
            # Check if it's time to generate this packet type
            last_time = self.management_last_generated.get(packet_type, 0)
            interval = 1.0 / effective_rate if effective_rate > 0 else float('inf')
            
            if current_time - last_time >= interval:
                # Generate packet
                packet = BackgroundPacket(
                    packet_id=f"mgmt_{self.vehicle_id}_{packet_type}_{self.packet_sequence}",
                    traffic_type=BackgroundTrafficType.MANAGEMENT,
                    packet_subtype=packet_type,
                    size_bytes=type_config["packet_size_bytes"],
                    priority=TrafficPriority(type_config["qos_priority"]),
                    generation_time=current_time,
                    source_vehicle=self.vehicle_id,
                    is_periodic=type_config["periodic"]
                )
                
                packets.append(packet)
                self.management_last_generated[packet_type] = current_time
                self.packet_sequence += 1
        
        return packets
    
    def _generate_infotainment_traffic(self, current_time: float, neighbors_count: int) -> List[BackgroundPacket]:
        """Generate infotainment/multimedia traffic"""
        packets = []
        
        for packet_type, type_config in INFOTAINMENT_TRAFFIC_CONFIG["packet_types"].items():
            
            # Skip if vehicle doesn't use this service
            if packet_type not in self.active_infotainment_services:
                continue
            
            # Base rate with vehicle activity factor
            base_rate = type_config["rate_hz"] * self.infotainment_activity_factor
            
            if type_config["periodic"]:
                # Periodic traffic (streaming, navigation updates)
                last_time = self.infotainment_last_generated.get(packet_type, 0)
                interval = 1.0 / base_rate if base_rate > 0 else float('inf')
                
                if current_time - last_time >= interval:
                    packet = self._create_infotainment_packet(packet_type, type_config, current_time)
                    packets.append(packet)
                    self.infotainment_last_generated[packet_type] = current_time
            
            else:
                # Bursty/random traffic (web browsing, file transfer, social media)
                burst_prob = type_config.get("burst_probability", 0.1)
                
                # Check for new burst initiation
                if packet_type not in self.active_bursts and random.random() < burst_prob * self.config.time_step:
                    # Start new burst
                    burst_duration = type_config.get("burst_duration", 2.0)
                    self.active_bursts[packet_type] = {
                        "start_time": current_time,
                        "end_time": current_time + burst_duration,
                        "last_packet": current_time,
                        "burst_id": f"burst_{self.vehicle_id}_{packet_type}_{int(current_time)}"
                    }
                
                # Generate packets for active bursts
                if packet_type in self.active_bursts:
                    burst_info = self.active_bursts[packet_type]
                    
                    if current_time <= burst_info["end_time"]:
                        # Still in burst period
                        interval = 1.0 / base_rate if base_rate > 0 else float('inf')
                        
                        if current_time - burst_info["last_packet"] >= interval:
                            packet = self._create_infotainment_packet(packet_type, type_config, current_time)
                            packet.burst_id = burst_info["burst_id"]
                            packets.append(packet)
                            burst_info["last_packet"] = current_time
                    else:
                        # Burst ended
                        del self.active_bursts[packet_type]
        
        return packets
    
    def _create_infotainment_packet(self, packet_type: str, type_config: Dict, current_time: float) -> BackgroundPacket:
        """Create an infotainment packet with realistic size variation"""
        base_size = type_config["packet_size_bytes"]
        
        # Add realistic size variation for different traffic types
        if packet_type == "video_streaming":
            # Video packets have high size variation (I-frames vs P-frames)
            size_variation = random.uniform(0.5, 2.0)
        elif packet_type == "audio_streaming":
            # Audio packets have low size variation
            size_variation = random.uniform(0.9, 1.1)
        elif packet_type in ["file_transfer", "internet_browsing"]:
            # Data packets have medium variation
            size_variation = random.uniform(0.7, 1.3)
        else:
            size_variation = random.uniform(0.8, 1.2)
        
        actual_size = int(base_size * size_variation)
        
        return BackgroundPacket(
            packet_id=f"info_{self.vehicle_id}_{packet_type}_{self.packet_sequence}",
            traffic_type=BackgroundTrafficType.INFOTAINMENT,
            packet_subtype=packet_type,
            size_bytes=actual_size,
            priority=TrafficPriority(type_config["qos_priority"]),
            generation_time=current_time,
            source_vehicle=self.vehicle_id,
            is_periodic=type_config["periodic"]
        )
    
    def get_current_load_mbps(self) -> Dict[str, float]:
        """Calculate current traffic load in Mbps for last second"""
        current_time = time.time()
        recent_packets = []
        
        # This would need to track recent packets in a real implementation
        # For now, estimate based on configuration
        
        mgmt_load = 0.0
        for packet_type, config in MANAGEMENT_TRAFFIC_CONFIG["packet_types"].items():
            rate = config["rate_hz"] * self.management_activity_factor
            size = config["packet_size_bytes"]
            mgmt_load += rate * size * 8 / 1e6  # Convert to Mbps
        
        info_load = 0.0
        for packet_type, config in INFOTAINMENT_TRAFFIC_CONFIG["packet_types"].items():
            if packet_type in self.active_infotainment_services:
                rate = config["rate_hz"] * self.infotainment_activity_factor
                size = config["packet_size_bytes"]
                enable_factor = 1.0 if config["periodic"] else config.get("burst_probability", 0.1)
                info_load += rate * size * 8 * enable_factor / 1e6
        
        return {
            "management_mbps": mgmt_load,
            "infotainment_mbps": info_load,
            "total_mbps": mgmt_load + info_load
        }

class BackgroundTrafficManager:
    """Manages background traffic for entire simulation"""
    
    def __init__(self, config: SimulationConfig):
        self.config = config
        self.vehicle_generators = {}
        self.network_load_history = []
        self.current_network_load = 0.0
        
        # Global traffic statistics
        self.total_management_packets = 0
        self.total_infotainment_packets = 0
        self.total_background_bytes = 0
        
    def initialize_vehicle_generator(self, vehicle_id: str):
        """Initialize background traffic generator for vehicle"""
        if vehicle_id not in self.vehicle_generators:
            self.vehicle_generators[vehicle_id] = BackgroundTrafficGenerator(vehicle_id, self.config)
    
    def generate_network_background_traffic(self, current_time: float, vehicles: Dict) -> Dict[str, List[BackgroundPacket]]:
        """Generate background traffic for all vehicles"""
        network_packets = {}
        total_load = 0.0
        
        for vehicle_id, vehicle in vehicles.items():
            # Initialize generator if needed
            self.initialize_vehicle_generator(vehicle_id)
            
            generator = self.vehicle_generators[vehicle_id]
            neighbors_count = len(getattr(vehicle, 'neighbors', []))
            
            # Generate background packets
            bg_packets = generator.generate_background_packets(current_time, neighbors_count)
            network_packets[vehicle_id] = bg_packets
            
            # Calculate load contribution
            vehicle_load = generator.get_current_load_mbps()
            total_load += vehicle_load["total_mbps"]
            
            # Update global statistics
            for packet in bg_packets:
                if packet.traffic_type == BackgroundTrafficType.MANAGEMENT:
                    self.total_management_packets += 1
                else:
                    self.total_infotainment_packets += 1
                self.total_background_bytes += packet.size_bytes
        
        # Update network load tracking
        self.current_network_load = total_load
        self.network_load_history.append({
            'time': current_time,
            'total_load_mbps': total_load,
            'vehicle_count': len(vehicles)
        })
        
        # Keep only recent history
        if len(self.network_load_history) > 1000:
            self.network_load_history = self.network_load_history[-1000:]
        
        return network_packets
    
    def calculate_background_interference_contribution(self, vehicle_id: str, neighbors: List[Dict]) -> float:
        """Calculate interference contribution from background traffic"""
        if not ENABLE_BACKGROUND_TRAFFIC_MODELING:
            return 0.0
        
        total_interference_mw = 0.0
        
        # Get background traffic load for this vehicle and neighbors  
        if vehicle_id in self.vehicle_generators:
            vehicle_load = self.vehicle_generators[vehicle_id].get_current_load_mbps()
            
            # Convert load to interference power
            # Higher background load = more channel occupancy = more interference
            load_factor = vehicle_load["total_mbps"] / 10.0  # Normalize to 10 Mbps max
            base_interference_mw = 1e-9  # Base interference level (nW)
            
            # Management traffic causes different interference than infotainment
            mgmt_interference = vehicle_load["management_mbps"] * base_interference_mw * 2.0  # More frequent, higher priority
            info_interference = vehicle_load["infotainment_mbps"] * base_interference_mw * 1.5  # Bursty, lower priority
            
            vehicle_interference = mgmt_interference + info_interference
            total_interference_mw += vehicle_interference
        
        # Add interference from neighbors' background traffic
        for neighbor in neighbors:
            neighbor_id = neighbor['id']
            if neighbor_id in self.vehicle_generators:
                neighbor_load = self.vehicle_generators[neighbor_id].get_current_load_mbps()
                distance = neighbor['distance']
                
                # Distance-based interference reduction
                distance_factor = 1.0 / (1.0 + distance / 100.0)
                
                neighbor_interference = (neighbor_load["total_mbps"] * base_interference_mw * 
                                       distance_factor * 0.5)  # Reduced effect from neighbors
                total_interference_mw += neighbor_interference
        
        # Add configured background interference sources
        config = BACKGROUND_INTERFERENCE_CONFIG
        
        # Adjacent channel interference
        adjacent_interference = base_interference_mw * config["adjacent_channel_interference"] * len(neighbors)
        
        # Co-channel interference from distant vehicles
        co_channel_interference = base_interference_mw * config["co_channel_interference"] * 3.0
        
        # Non-VANET interference (WiFi, cellular, etc.)
        non_vanet_interference = base_interference_mw * config["non_vanet_interference"] * 2.0
        
        total_interference_mw += adjacent_interference + co_channel_interference + non_vanet_interference
        
        return total_interference_mw
    
    def get_effective_cbr_contribution(self, vehicle_id: str, base_cbr: float) -> float:
        """FIXED: Calculate effective CBR including background traffic with realistic scaling"""
        if not ENABLE_BACKGROUND_TRAFFIC_MODELING or vehicle_id not in self.vehicle_generators:
            return base_cbr
        
        generator = self.vehicle_generators[vehicle_id]
        vehicle_load = generator.get_current_load_mbps()
        
        # FIXED: More realistic scaling with conservative channel capacity
        # IEEE 802.11bd 10 MHz channel theoretical capacity ≈ 27 Mbps
        channel_capacity_mbps = 27.0
        
        # FIXED: More conservative CBR contribution from background traffic
        mgmt_cbr_contribution = (vehicle_load["management_mbps"] / channel_capacity_mbps) * 1.0      # Reduced from 1.5
        info_cbr_contribution = (vehicle_load["infotainment_mbps"] / channel_capacity_mbps) * 1.2    # Reduced from 2.0
        
        # FIXED: More conservative additional CBR from packet overhead and contention
        total_bg_load = vehicle_load["total_mbps"]
        if total_bg_load > 0:
            # FIXED: Reduced background traffic overhead
            contention_overhead = (total_bg_load / channel_capacity_mbps) * 0.5      # Reduced from 0.8
            collision_overhead = (total_bg_load / channel_capacity_mbps) * 0.3       # Reduced from 0.6
            
            mgmt_cbr_contribution += contention_overhead
            info_cbr_contribution += collision_overhead
        
        background_cbr_contribution = mgmt_cbr_contribution + info_cbr_contribution
        
        # FIXED: More conservative background traffic load ratio
        background_cbr_contribution *= BACKGROUND_TRAFFIC_TOTAL_LOAD * 0.8  # NEW: 20% reduction
        
        # FIXED: More conservative scaling for high background loads
        if background_cbr_contribution > 0.08:  # Raised threshold from 0.1
            background_cbr_contribution *= 1.2  # Reduced from 1.5 (50% → 20% amplification)
        
        # Total offered load (can exceed 1.0)
        total_offered_load = base_cbr + background_cbr_contribution
        
        # Return offered load (will be bounded to CBR ≤ 1.0 in calling function)
        return total_offered_load
    
    def get_traffic_statistics(self) -> Dict[str, Any]:
        """Get comprehensive background traffic statistics"""
        stats = {
            'total_management_packets': self.total_management_packets,
            'total_infotainment_packets': self.total_infotainment_packets,
            'total_background_bytes': self.total_background_bytes,
            'current_network_load_mbps': self.current_network_load,
            'active_vehicles': len(self.vehicle_generators),
            'avg_management_load': 0.0,
            'avg_infotainment_load': 0.0
        }
        
        if self.vehicle_generators:
            total_mgmt = sum(gen.get_current_load_mbps()["management_mbps"] 
                           for gen in self.vehicle_generators.values())
            total_info = sum(gen.get_current_load_mbps()["infotainment_mbps"] 
                           for gen in self.vehicle_generators.values())
            
            stats['avg_management_load'] = total_mgmt / len(self.vehicle_generators)
            stats['avg_infotainment_load'] = total_info / len(self.vehicle_generators)
        
        return stats
    
# ===============================================================================
# ANTENNA CLASS
# ===============================================================================


class AntennaType(Enum):
    OMNIDIRECTIONAL = "OMNIDIRECTIONAL"
    SECTORAL = "SECTORAL"

class AntennaDirection(Enum):
    FRONT = "front"
    REAR = "rear"  
    LEFT = "left"
    RIGHT = "right"

@dataclass
class SectorConfig:
    """Configuration for one sector of sectoral antenna"""
    power_dbm: float
    gain_db: float
    beamwidth_deg: float
    enabled: bool = True
    
    def __post_init__(self):
        # Validate beamwidth
        if self.beamwidth_deg <= 0 or self.beamwidth_deg > 180:
            raise ValueError(f"Invalid beamwidth: {self.beamwidth_deg}. Must be between 0 and 180 degrees")

@dataclass
class AntennaConfiguration:
    """Complete antenna configuration for a vehicle"""
    antenna_type: AntennaType
    omnidirectional_config: Dict[str, float] = None
    sectoral_config: Dict[str, SectorConfig] = None
    
    def __post_init__(self):
        if self.antenna_type == AntennaType.OMNIDIRECTIONAL:
            if not self.omnidirectional_config:
                self.omnidirectional_config = {
                    "power_dbm": OMNIDIRECTIONAL_ANTENNA_CONFIG["power_dbm"],
                    "gain_db": OMNIDIRECTIONAL_ANTENNA_CONFIG["gain_db"]
                }
        elif self.antenna_type == AntennaType.SECTORAL:
            if not self.sectoral_config:
                self.sectoral_config = {}
                for direction, config in SECTORAL_ANTENNA_CONFIG.items():
                    self.sectoral_config[direction] = SectorConfig(**config)

class SectoralAntennaSystem:
    """RL-enhanced sectoral antenna system with front/rear-only RL control"""
    
    def __init__(self, config: AntennaConfiguration):
        self.config = config
        self.antenna_type = config.antenna_type
        
        # Store base power for distribution
        self.base_power_dbm = 20.0  # Default base power
        
        # Sector relative angles (corrected)
        self.sector_relative_angles = {
            AntennaDirection.FRONT: 0,      # 0° relative to vehicle heading
            AntennaDirection.RIGHT: 90,     # 90° relative to vehicle heading 
            AntennaDirection.REAR: 180,     # 180° relative to vehicle heading
            AntennaDirection.LEFT: 270      # 270° relative to vehicle heading
        }
        
        # Current neighbor distribution per sector (for weighting)
        self.neighbor_distribution = {
            'front': 0, 'rear': 0, 'left': 0, 'right': 0
        }
        
        # Separate tracking for RL-controlled vs static sectors
        self.rl_controlled_neighbor_distribution = {
            'front': 0, 'rear': 0
        }
        
        # Current power allocation per sector
        self.sector_powers = {
            'front': SECTORAL_ANTENNA_CONFIG['front']['power_dbm'],
            'rear': SECTORAL_ANTENNA_CONFIG['rear']['power_dbm'],
            'left': SIDE_ANTENNA_STATIC_POWER,  # Static side power
            'right': SIDE_ANTENNA_STATIC_POWER  # Static side power
        }
        

    
    def update_neighbor_distribution(self, neighbors: List[Dict]):
        """Update neighbor distribution per sector with separate RL-controlled tracking"""
        # Reset all distribution
        self.neighbor_distribution = {'front': 0, 'rear': 0, 'left': 0, 'right': 0}
        self.rl_controlled_neighbor_distribution = {'front': 0, 'rear': 0}
        
        if not neighbors:
            return
        
        # Count neighbors per sector
        for neighbor in neighbors:
            sector = neighbor.get('neighbor_sector', 'front')
            if sector in self.neighbor_distribution:
                self.neighbor_distribution[sector] += 1
                
                # Track RL-controlled sectors separately
                if sector in RL_CONTROLLED_SECTORS:
                    self.rl_controlled_neighbor_distribution[sector] += 1
    
    def get_weighted_average_power(self) -> float:
        """Calculate weighted average power for RL (ONLY front/rear sectors)"""
        if self.antenna_type == AntennaType.OMNIDIRECTIONAL:
            # FIXED: For omnidirectional, this should return the same as vehicle.transmission_power
            return self.config.omnidirectional_config["power_dbm"]
        
        # Only consider RL-controlled sectors (front/rear)
        total_rl_neighbors = sum(self.rl_controlled_neighbor_distribution.values())
        
        if total_rl_neighbors == 0:
            # No neighbors in RL-controlled sectors - return simple average of front/rear
            front_power = self.sector_powers['front']
            rear_power = self.sector_powers['rear']
            return (front_power + rear_power) / 2
        
        # Calculate weighted average based on neighbor distribution in RL-controlled sectors only
        weighted_power = 0.0
        for sector, count in self.rl_controlled_neighbor_distribution.items():
            weight = count / total_rl_neighbors
            weighted_power += self.sector_powers[sector] * weight
        
        return weighted_power
    
    def get_rl_controlled_power(self) -> Dict[str, float]:
        """Get power levels for RL-controlled sectors only"""
        return {
            sector: self.sector_powers[sector] 
            for sector in RL_CONTROLLED_SECTORS
        }
    
    def distribute_power_from_rl(self, rl_power: float):
        """Distribute RL power ONLY across front/rear sectors, keep sides static"""
        if self.antenna_type == AntennaType.OMNIDIRECTIONAL:
            self.config.omnidirectional_config["power_dbm"] = rl_power
            return
        
        self.base_power_dbm = rl_power
        total_rl_neighbors = sum(self.rl_controlled_neighbor_distribution.values())
        
        if total_rl_neighbors == 0:
            # No neighbors in RL-controlled sectors - distribute equally between front/rear
            for sector in RL_CONTROLLED_SECTORS:
                self.sector_powers[sector] = rl_power
        else:
            # Distribute based on neighbor density in RL-controlled sectors only
            min_power = max(1.0, rl_power - 5.0)  # Minimum power per sector
            max_power = min(30.0, rl_power + 5.0)  # Maximum power per sector
            
            for sector in RL_CONTROLLED_SECTORS:
                count = self.rl_controlled_neighbor_distribution.get(sector, 0)
                if total_rl_neighbors > 0:
                    # Higher neighbor count = higher power allocation
                    neighbor_ratio = count / total_rl_neighbors
                    
                    # Power boost for sectors with more neighbors
                    power_boost = (neighbor_ratio - 0.5) * 6.0  # Scale factor for 2 sectors
                    allocated_power = rl_power + power_boost
                    
                    # Apply bounds
                    self.sector_powers[sector] = max(min_power, min(max_power, allocated_power))
                else:
                    self.sector_powers[sector] = rl_power
        
        # Keep side antennas at static power (DO NOT CHANGE)
        for sector in RL_STATIC_SECTORS:
            self.sector_powers[sector] = SIDE_ANTENNA_STATIC_POWER
        
        # Update the original config for RL-controlled sectors only
        for sector_name, power in self.sector_powers.items():
            if sector_name in self.config.sectoral_config and sector_name in RL_CONTROLLED_SECTORS:
                self.config.sectoral_config[sector_name].power_dbm = power
    
    def get_effective_transmission_power(self, vehicle_heading: float, 
                               target_angle: float) -> Tuple[float, AntennaDirection]:
        """FIXED: Get effective transmission power with better sector selection"""
        if self.antenna_type == AntennaType.OMNIDIRECTIONAL:
            total_power = self.config.omnidirectional_config["power_dbm"] + self.config.omnidirectional_config["gain_db"]
            return (total_power, None)
        
        # FIXED: First determine the geometrically correct sector
        geometrical_sector = self.get_sector_for_angle(vehicle_heading, target_angle)
        
        # Check if the geometrical sector gives good enough performance
        geo_sector_config = self.config.sectoral_config[geometrical_sector.value]
        if geo_sector_config.enabled:
            geo_gain = self.calculate_antenna_gain(vehicle_heading, target_angle, geometrical_sector)
            geo_power = self.sector_powers[geometrical_sector.value]
            geo_effective_power = geo_power + geo_gain
            
            # If geometrical sector gives reasonable performance (within 6 dB of max), use it
            max_possible_power = max(self.sector_powers.values()) + max(
                config.gain_db for config in self.config.sectoral_config.values()
            )
            
            if geo_effective_power >= (max_possible_power - 6.0):  # Within 6 dB is acceptable
                return geo_effective_power, geometrical_sector
        
        # Otherwise, find the best performing sector
        best_power = -100.0
        best_sector = geometrical_sector  # Default to geometrical
        
        for sector in AntennaDirection:
            sector_config = self.config.sectoral_config[sector.value]
            if not sector_config.enabled:
                continue
            
            # Calculate antenna gain for this sector
            gain = self.calculate_antenna_gain(vehicle_heading, target_angle, sector)
            sector_power = self.sector_powers[sector.value]
            effective_power = sector_power + gain
            
            if effective_power > best_power:
                best_power = effective_power
                best_sector = sector
        
        return best_power, best_sector
    
    def calculate_antenna_gain(self, vehicle_heading: float, target_angle: float, 
                     sector: AntennaDirection) -> float:
        """FIXED: Calculate antenna gain with realistic sectoral pattern"""
        if self.antenna_type == AntennaType.OMNIDIRECTIONAL:
            return self.config.omnidirectional_config["gain_db"]
        
        sector_config = self.config.sectoral_config[sector.value]
        if not sector_config.enabled:
            return -50.0
        
        # Calculate sector center based on vehicle heading
        sector_relative_angle = self.sector_relative_angles[sector]
        sector_absolute_angle = (vehicle_heading + sector_relative_angle) % 360
        
        # Calculate angle difference
        angle_diff = abs(((target_angle - sector_absolute_angle + 180) % 360) - 180)
        
        # FIXED: More realistic beamwidth-based gain calculation
        half_beamwidth = sector_config.beamwidth_deg / 2  # 45° for 90° beamwidth
        
        if angle_diff <= half_beamwidth:
            # Within main beam - full gain
            gain_factor = 1.0
        elif angle_diff <= half_beamwidth * 1.5:  # Up to 67.5°
            # First sidelobe - gradual reduction
            excess_angle = angle_diff - half_beamwidth
            gain_factor = 1.0 - (excess_angle / (half_beamwidth * 0.5)) * 0.15  # Only 15% reduction
        elif angle_diff <= half_beamwidth * 2.5:  # Up to 112.5°
            # Extended coverage - still useful gain
            gain_factor = 0.85 - ((angle_diff - half_beamwidth * 1.5) / (half_beamwidth)) * 0.35  # Reduce to 50%
        elif angle_diff <= half_beamwidth * 3.0:  # Up to 135°
            # Far sidelobe - minimal but non-zero gain
            gain_factor = 0.5 - ((angle_diff - half_beamwidth * 2.5) / (half_beamwidth * 0.5)) * 0.3  # Reduce to 20%
        else:
            # Back lobe - minimal gain but not zero
            gain_factor = 0.2  # 20% of full gain (not 5%)
        
        # Apply gain factor to maximum sector gain
        final_gain = sector_config.gain_db * gain_factor
        
        return final_gain

    
    def get_sector_for_angle(self, vehicle_heading: float, target_angle: float) -> AntennaDirection:
        """FIXED: Determine which sector a target angle falls into with better logic"""
        if self.antenna_type == AntennaType.OMNIDIRECTIONAL:
            return AntennaDirection.FRONT
        
        # Calculate relative angle from vehicle heading to target
        relative_angle = (target_angle - vehicle_heading + 360) % 360
        
        # FIXED: More intuitive sector assignment based on vehicle orientation
        # Front: -45° to +45° relative to vehicle heading
        # Right: +45° to +135° 
        # Rear: +135° to +225°
        # Left: +225° to +315°
        
        if relative_angle <= 45 or relative_angle > 315:
            return AntennaDirection.FRONT
        elif 45 < relative_angle <= 135:
            return AntennaDirection.RIGHT
        elif 135 < relative_angle <= 225:
            return AntennaDirection.REAR
        else:  # 225 < relative_angle <= 315
            return AntennaDirection.LEFT
    
    def get_sector_power_summary(self) -> str:
        """Get summary of current sector power allocation with RL status"""
        if self.antenna_type == AntennaType.OMNIDIRECTIONAL:
            return f"Omni: {self.config.omnidirectional_config['power_dbm']:.1f}dBm"
        
        return (f"F:{self.sector_powers['front']:.1f}(RL), "
                f"R:{self.sector_powers['rear']:.1f}(RL), "
                f"L:{self.sector_powers['left']:.1f}(Static), "
                f"R:{self.sector_powers['right']:.1f}(Static) dBm")
    
    def get_communication_sectors(self) -> List[AntennaDirection]:
        """Get list of enabled communication sectors (unchanged)"""
        if self.antenna_type == AntennaType.OMNIDIRECTIONAL:
            return [AntennaDirection.FRONT]  # Omnidirectional uses front as default
        
        enabled_sectors = []
        for sector in AntennaDirection:
            sector_config = self.config.sectoral_config.get(sector.value)
            if sector_config and sector_config.enabled:
                enabled_sectors.append(sector)
        
        return enabled_sectors
    
# ===============================================================================
# DoS/DDoS ATTACK Classes and Functions
# ===============================================================================

class AttackType(Enum):
    BEACON_FLOODING = "BEACON_FLOODING"
    HIGH_POWER_JAMMING = "HIGH_POWER_JAMMING"
    ASYNC_BEACON = "ASYNC_BEACON"
    COMBINED = "COMBINED"
    NONE = "NONE"

class AttackerSelectionStrategy(Enum):
    RANDOM = "RANDOM"
    CENTRAL = "CENTRAL"
    DISTRIBUTED = "DISTRIBUTED"
    MANUAL = "MANUAL"

@dataclass
class AttackMetrics:
    """Metrics for tracking attack behavior and detection features"""
    # Basic attack info
    is_attacker: bool = False
    attack_type: AttackType = AttackType.NONE
    attack_start_time: float = 0.0
    attack_duration: float = 0.0
    attack_intensity: float = 0.0
    
    # Beacon-related attack metrics
    beacon_rate_variance: float = 0.0
    beacon_interval_irregularity: float = 0.0
    beacon_size_anomaly: float = 0.0
    beacon_flooding_score: float = 0.0
    
    # Power/Interference attack metrics
    tx_power_anomaly: float = 0.0
    interference_level: float = 0.0
    jamming_score: float = 0.0
    
    # Network impact metrics
    neighbor_disruption_ratio: float = 0.0
    packet_delivery_impact: float = 0.0
    throughput_degradation: float = 0.0
    latency_increase_ratio: float = 0.0
    
    # Temporal behavior metrics
    communication_pattern_entropy: float = 0.0
    burst_communication_ratio: float = 0.0
    silent_period_ratio: float = 0.0
    
    # Detection features (for ML dataset)
    detection_features: Dict[str, float] = field(default_factory=dict)

@dataclass
class AttackConfiguration:
    """Configuration for different attack types"""
    attack_type: AttackType
    intensity_level: float = 1.0  # 0.0 to 1.0
    duration_ratio: float = 1.0   # Fraction of simulation time
    target_vehicles: List[str] = field(default_factory=list)
    parameters: Dict[str, Any] = field(default_factory=dict)

class VANETAttackManager:
    """Manager for coordinating DoS/DDoS attacks in VANET simulation"""
    
    def __init__(self, config: SimulationConfig):
        self.config = config
        self.attackers: Dict[str, AttackConfiguration] = {}
        self.attack_metrics: Dict[str, AttackMetrics] = {}
        self.normal_vehicles: Set[str] = set()
        self.attack_start_time: float = 0.0
        self.attack_active: bool = False
        
        # Detection features calculation
        self.detection_window_data: Dict[str, List[Dict]] = defaultdict(list)
        self.detection_features_history: List[Dict] = []
        
    def initialize_attackers(self, vehicle_ids: List[str], mobility_data: List[Dict]):
        """Initialize attacker vehicles based on configuration"""
        if not ENABLE_ATTACK_SIMULATION:
            return
        
        total_vehicles = len(vehicle_ids)
        num_attackers = min(NUMBER_OF_ATTACKERS, total_vehicles)
        
        if num_attackers == 0:
            return
        
        # Select attacker vehicles
        if ATTACKER_SELECTION_STRATEGY == "RANDOM":
            selected_attackers = random.sample(vehicle_ids, num_attackers)
        elif ATTACKER_SELECTION_STRATEGY == "CENTRAL":
            selected_attackers = self._select_central_vehicles(vehicle_ids, mobility_data, num_attackers)
        elif ATTACKER_SELECTION_STRATEGY == "DISTRIBUTED":
            selected_attackers = self._select_distributed_vehicles(vehicle_ids, mobility_data, num_attackers)
        elif ATTACKER_SELECTION_STRATEGY == "MANUAL":
            selected_attackers = ATTACKER_IDS[:num_attackers]
        else:
            selected_attackers = vehicle_ids[:num_attackers]
        
        # FIXED: Configure attackers based on attack type
        if ATTACK_TYPE == "COMBINED":
            # Distribute attack types across attackers according to weights
            weights = COMBINED_ATTACK_WEIGHTS
            flooding_count = int(num_attackers * weights["flooding"])
            jamming_count = int(num_attackers * weights["jamming"])
            async_count = num_attackers - flooding_count - jamming_count  # Remaining get async
            
            attack_assignments = (
                ["BEACON_FLOODING"] * flooding_count +
                ["HIGH_POWER_JAMMING"] * jamming_count +
                ["ASYNC_BEACON"] * async_count
            )
            
            # Shuffle to randomize assignment
            random.shuffle(attack_assignments)
            
            for i, attacker_id in enumerate(selected_attackers):
                assigned_attack = attack_assignments[i] if i < len(attack_assignments) else "BEACON_FLOODING"
                attack_config = self._create_attack_configuration(assigned_attack)
                self.attackers[attacker_id] = attack_config
                self.attack_metrics[attacker_id] = AttackMetrics(
                    is_attacker=True,
                    attack_type=AttackType(assigned_attack)
                )
            
            print(f"[ATTACK MANAGER] COMBINED attack distributed: {flooding_count} flooding, {jamming_count} jamming, {async_count} async")
        else:
            # Single attack type for all attackers
            for attacker_id in selected_attackers:
                attack_config = self._create_attack_configuration(ATTACK_TYPE)
                self.attackers[attacker_id] = attack_config
                self.attack_metrics[attacker_id] = AttackMetrics(
                    is_attacker=True,
                    attack_type=AttackType(ATTACK_TYPE)
                )
        
        # Mark normal vehicles
        self.normal_vehicles = set(vehicle_ids) - set(selected_attackers)
        
        # FIXED: Initialize normal vehicle metrics (for ML dataset)
        for vehicle_id in self.normal_vehicles:
            self.attack_metrics[vehicle_id] = AttackMetrics(is_attacker=False)
        
        print(f"[ATTACK MANAGER] Initialized {len(selected_attackers)} attackers and {len(self.normal_vehicles)} normal vehicles using {ATTACK_TYPE} attack")
        print(f"[ATTACK MANAGER] Attacker vehicles: {selected_attackers}")
    
    def _select_central_vehicles(self, vehicle_ids: List[str], mobility_data: List[Dict], num_attackers: int) -> List[str]:
        """Select vehicles closest to network center"""
        # Calculate center of all vehicles
        all_positions = [(data['x'], data['y']) for data in mobility_data if data['id'] in vehicle_ids]
        if not all_positions:
            return random.sample(vehicle_ids, num_attackers)
        
        center_x = sum(pos[0] for pos in all_positions) / len(all_positions)
        center_y = sum(pos[1] for pos in all_positions) / len(all_positions)
        
        # Calculate distances to center for each vehicle
        vehicle_distances = []
        for vehicle_id in vehicle_ids:
            vehicle_data = next((data for data in mobility_data if data['id'] == vehicle_id), None)
            if vehicle_data:
                distance = math.sqrt((vehicle_data['x'] - center_x)**2 + (vehicle_data['y'] - center_y)**2)
                vehicle_distances.append((vehicle_id, distance))
        
        # Select closest vehicles
        vehicle_distances.sort(key=lambda x: x[1])
        return [veh_id for veh_id, _ in vehicle_distances[:num_attackers]]
    
    def _select_distributed_vehicles(self, vehicle_ids: List[str], mobility_data: List[Dict], num_attackers: int) -> List[str]:
        """Select vehicles distributed across the network"""
        if num_attackers >= len(vehicle_ids):
            return vehicle_ids.copy()
        
        selected = []
        remaining = vehicle_ids.copy()
        
        # Select first vehicle randomly
        selected.append(random.choice(remaining))
        remaining.remove(selected[0])
        
        # Select remaining vehicles to maximize minimum distance
        for _ in range(num_attackers - 1):
            max_min_distance = 0
            best_candidate = None
            
            for candidate in remaining:
                candidate_data = next((data for data in mobility_data if data['id'] == candidate), None)
                if not candidate_data:
                    continue
                
                min_distance = float('inf')
                for selected_id in selected:
                    selected_data = next((data for data in mobility_data if data['id'] == selected_id), None)
                    if selected_data:
                        distance = math.sqrt((candidate_data['x'] - selected_data['x'])**2 + 
                                           (candidate_data['y'] - selected_data['y'])**2)
                        min_distance = min(min_distance, distance)
                
                if min_distance > max_min_distance:
                    max_min_distance = min_distance
                    best_candidate = candidate
            
            if best_candidate:
                selected.append(best_candidate)
                remaining.remove(best_candidate)
        
        return selected
    
    def _create_attack_configuration(self, attack_type: str) -> AttackConfiguration:
        """Create attack configuration based on type"""
        if attack_type == "BEACON_FLOODING":
            return AttackConfiguration(
                attack_type=AttackType.BEACON_FLOODING,
                intensity_level=1.0,
                duration_ratio=FLOODING_DURATION_RATIO,
                parameters={
                    'beacon_rate': FLOODING_BEACON_RATE,
                    'packet_size': FLOODING_PACKET_SIZE
                }
            )
        elif attack_type == "HIGH_POWER_JAMMING":
            return AttackConfiguration(
                attack_type=AttackType.HIGH_POWER_JAMMING,
                intensity_level=1.0,
                duration_ratio=1.0,
                parameters={
                    'jamming_power': JAMMING_POWER_DBM,
                    'bandwidth_ratio': JAMMING_BANDWIDTH_RATIO,
                    'continuous_mode': JAMMING_CONTINUOUS_MODE
                }
            )
        elif attack_type == "ASYNC_BEACON":
            return AttackConfiguration(
                attack_type=AttackType.ASYNC_BEACON,
                intensity_level=ASYNC_RANDOMNESS_FACTOR,
                duration_ratio=1.0,
                parameters={
                    'min_interval': ASYNC_MIN_INTERVAL,
                    'max_interval': ASYNC_MAX_INTERVAL,
                    'randomness_factor': ASYNC_RANDOMNESS_FACTOR
                }
            )
        else:
            # Default to beacon flooding for unknown types
            return AttackConfiguration(
                attack_type=AttackType.BEACON_FLOODING,
                intensity_level=1.0,
                duration_ratio=FLOODING_DURATION_RATIO,
                parameters={
                    'beacon_rate': FLOODING_BEACON_RATE,
                    'packet_size': FLOODING_PACKET_SIZE
                }
            )
    
    def update_attacker_behavior(self, vehicle_id: str, vehicle_state: 'VehicleState', current_time: float):
        """Update attacker behavior during simulation - FIXED for new COMBINED logic"""
        # Always calculate detection features for all vehicles (normal and attackers)
        self._calculate_detection_features(vehicle_id, vehicle_state, current_time)
        
        # Only apply attack behavior if this is an attacker
        if vehicle_id not in self.attackers:
            return
        
        attack_config = self.attackers[vehicle_id]
        attack_metrics = self.attack_metrics[vehicle_id]
        
        # Check if attack should be active
        simulation_duration = current_time
        attack_end_time = simulation_duration * attack_config.duration_ratio
        
        if current_time <= attack_end_time:
            if not self.attack_active:
                self.attack_active = True
                self.attack_start_time = current_time
                attack_metrics.attack_start_time = current_time
            
            # Apply attack-specific behavior based on the assigned attack type
            if attack_config.attack_type == AttackType.BEACON_FLOODING:
                self._apply_beacon_flooding(vehicle_state, attack_config, current_time)
            elif attack_config.attack_type == AttackType.HIGH_POWER_JAMMING:
                self._apply_high_power_jamming(vehicle_state, attack_config, current_time)
            elif attack_config.attack_type == AttackType.ASYNC_BEACON:
                self._apply_async_beacon(vehicle_state, attack_config, current_time)
            elif attack_config.attack_type == AttackType.COMBINED:
                # This should not happen with new logic, but keep for safety
                self._apply_combined_attack(vehicle_state, attack_config, current_time)
        
        # Update attack metrics
        self._update_attack_metrics(vehicle_id, vehicle_state, current_time)
    
    def _apply_beacon_flooding(self, vehicle_state: 'VehicleState', attack_config: AttackConfiguration, current_time: float):
        """Apply beacon flooding attack"""
        params = attack_config.parameters
        
        # Increase beacon rate dramatically
        vehicle_state.beacon_rate = params['beacon_rate']
        
        # Increase packet size to consume more bandwidth
        # This would affect the payload_length in the config or vehicle state
        
        # Add variance to make detection harder
        rate_variance = random.uniform(0.8, 1.2)
        vehicle_state.beacon_rate *= rate_variance
        
        # Update attack metrics
        attack_metrics = self.attack_metrics[vehicle_state.vehicle_id]
        attack_metrics.beacon_flooding_score = params['beacon_rate'] / 10.0  # Normalized to normal rate
        attack_metrics.beacon_rate_variance = abs(rate_variance - 1.0)
    
    def _apply_high_power_jamming(self, vehicle_state: 'VehicleState', attack_config: AttackConfiguration, current_time: float):
        """Apply high power jamming attack"""
        params = attack_config.parameters
        
        # Increase transmission power
        vehicle_state.transmission_power = params['jamming_power']
        
        # Add random noise to make jamming more effective
        if params['continuous_mode']:
            # Continuous jamming
            noise_factor = 1.0
        else:
            # Intermittent jamming
            noise_factor = 1.0 if random.random() < 0.7 else 0.1
        
        vehicle_state.transmission_power *= noise_factor
        
        # Update attack metrics
        attack_metrics = self.attack_metrics[vehicle_state.vehicle_id]
        attack_metrics.jamming_score = (params['jamming_power'] - 20.0) / 15.0  # Normalized
        attack_metrics.tx_power_anomaly = (vehicle_state.transmission_power - 20.0) / 20.0
    
    def _apply_async_beacon(self, vehicle_state: 'VehicleState', attack_config: AttackConfiguration, current_time: float):
        """Apply asynchronous beacon attack"""
        params = attack_config.parameters
        
        # Randomize beacon timing
        min_interval = params['min_interval']
        max_interval = params['max_interval']
        randomness = params['randomness_factor']
        
        # Generate random beacon rate within range
        random_rate = random.uniform(1.0 / max_interval, 1.0 / min_interval)
        vehicle_state.beacon_rate = random_rate
        
        # Add additional randomness
        if random.random() < randomness:
            vehicle_state.beacon_rate *= random.uniform(0.1, 3.0)
        
        # Update attack metrics
        attack_metrics = self.attack_metrics[vehicle_state.vehicle_id]
        expected_rate = 10.0  # Normal beacon rate
        attack_metrics.beacon_interval_irregularity = abs(vehicle_state.beacon_rate - expected_rate) / expected_rate
        attack_metrics.communication_pattern_entropy = self._calculate_entropy(vehicle_state.beacon_rate)
    
    def _apply_combined_attack(self, vehicle_state: 'VehicleState', attack_config: AttackConfiguration, current_time: float):
        """Apply combined attack - NOTE: This should not be called anymore with the new COMBINED logic"""
        # This method is kept for backward compatibility but should not be used
        # with the new COMBINED attack distribution logic
        weights = attack_config.parameters
        
        # Apply weighted combination of attacks (legacy behavior)
        if random.random() < weights['flooding']:
            flooding_config = AttackConfiguration(
                attack_type=AttackType.BEACON_FLOODING,
                parameters={'beacon_rate': FLOODING_BEACON_RATE * 0.5, 'packet_size': FLOODING_PACKET_SIZE}
            )
            self._apply_beacon_flooding(vehicle_state, flooding_config, current_time)
        
        if random.random() < weights['jamming']:
            jamming_config = AttackConfiguration(
                attack_type=AttackType.HIGH_POWER_JAMMING,
                parameters={'jamming_power': JAMMING_POWER_DBM * 0.8, 'bandwidth_ratio': 1.2, 'continuous_mode': False}
            )
            self._apply_high_power_jamming(vehicle_state, jamming_config, current_time)
        
        if random.random() < weights['async']:
            async_config = AttackConfiguration(
                attack_type=AttackType.ASYNC_BEACON,
                parameters={'min_interval': 0.05, 'max_interval': 0.5, 'randomness_factor': 0.7}
            )
            self._apply_async_beacon(vehicle_state, async_config, current_time)
    
    def _calculate_entropy(self, beacon_rate: float) -> float:
        """Calculate communication pattern entropy (simplified)"""
        # Simplified entropy calculation based on beacon rate deviation
        normal_rate = 10.0
        deviation = abs(beacon_rate - normal_rate) / normal_rate
        return min(1.0, deviation)
    
    def _update_attack_metrics(self, vehicle_id: str, vehicle_state: 'VehicleState', current_time: float):
        """Update comprehensive attack metrics for detection"""
        attack_metrics = self.attack_metrics[vehicle_id]
        
        if attack_metrics.is_attacker and self.attack_active:
            attack_metrics.attack_duration = current_time - attack_metrics.attack_start_time
            
            # Calculate attack intensity based on behavior deviation
            normal_beacon_rate = 10.0
            normal_tx_power = 20.0
            
            beacon_deviation = abs(vehicle_state.beacon_rate - normal_beacon_rate) / normal_beacon_rate
            power_deviation = abs(vehicle_state.transmission_power - normal_tx_power) / normal_tx_power
            
            attack_metrics.attack_intensity = min(1.0, (beacon_deviation + power_deviation) / 2.0)
        
        # Update detection features
        self._calculate_detection_features(vehicle_id, vehicle_state, current_time)
    
    def _calculate_detection_features(self, vehicle_id: str, vehicle_state: 'VehicleState', current_time: float):
        """Calculate features for ML-based attack detection - FIXED to include all vehicles"""
        # Ensure we have attack metrics for this vehicle (normal or attacker)
        if vehicle_id not in self.attack_metrics:
            self.attack_metrics[vehicle_id] = AttackMetrics(is_attacker=False)
        
        attack_metrics = self.attack_metrics[vehicle_id]
        
        # Store current vehicle state in detection window
        current_data = {
            'timestamp': current_time,
            'beacon_rate': vehicle_state.beacon_rate,
            'tx_power': vehicle_state.transmission_power,
            'neighbor_count': len(vehicle_state.neighbors),
            'throughput': vehicle_state.current_throughput,
            'cbr': vehicle_state.current_cbr,
            'sinr': vehicle_state.current_snr,
            'per': vehicle_state.current_per
        }
        
        # Maintain sliding window
        window_data = self.detection_window_data[vehicle_id]
        window_data.append(current_data)
        
        # Remove old data outside window
        window_start_time = current_time - DETECTION_WINDOW_SIZE
        window_data[:] = [data for data in window_data if data['timestamp'] >= window_start_time]
        
        if len(window_data) < 5:  # Need minimum data points
            return
        
        # Calculate detection features
        features = {}
        
        # Beacon rate features
        beacon_rates = [data['beacon_rate'] for data in window_data]
        features['beacon_rate_mean'] = np.mean(beacon_rates)
        features['beacon_rate_std'] = np.std(beacon_rates)
        features['beacon_rate_variance'] = np.var(beacon_rates)
        features['beacon_rate_max'] = np.max(beacon_rates)
        features['beacon_rate_min'] = np.min(beacon_rates)
        features['beacon_rate_range'] = features['beacon_rate_max'] - features['beacon_rate_min']
        
        # Power features
        tx_powers = [data['tx_power'] for data in window_data]
        features['tx_power_mean'] = np.mean(tx_powers)
        features['tx_power_std'] = np.std(tx_powers)
        features['tx_power_max'] = np.max(tx_powers)
        features['tx_power_anomaly_score'] = (features['tx_power_mean'] - 20.0) / 20.0
        
        # Network impact features
        throughputs = [data['throughput'] for data in window_data]
        cbrs = [data['cbr'] for data in window_data]
        sinrs = [data['sinr'] for data in window_data]
        pers = [data['per'] for data in window_data]
        
        features['throughput_mean'] = np.mean(throughputs)
        features['throughput_degradation'] = max(0, (np.max(throughputs) - np.mean(throughputs)) / np.max(throughputs)) if np.max(throughputs) > 0 else 0
        features['cbr_mean'] = np.mean(cbrs)
        features['cbr_increase_ratio'] = (np.mean(cbrs) - 0.2) / 0.2 if np.mean(cbrs) > 0.2 else 0
        features['sinr_mean'] = np.mean(sinrs)
        features['sinr_degradation'] = max(0, 15.0 - np.mean(sinrs)) / 15.0
        features['per_mean'] = np.mean(pers)
        features['per_increase_ratio'] = np.mean(pers) / 0.01 if np.mean(pers) > 0.01 else 0
        
        # Temporal pattern features
        neighbor_counts = [data['neighbor_count'] for data in window_data]
        features['neighbor_disruption_ratio'] = np.std(neighbor_counts) / (np.mean(neighbor_counts) + 1)
        
        # Communication pattern features
        beacon_intervals = []
        for i in range(1, len(beacon_rates)):
            if beacon_rates[i] > 0:
                interval = 1.0 / beacon_rates[i]
                beacon_intervals.append(interval)
        
        if beacon_intervals:
            features['beacon_interval_irregularity'] = np.std(beacon_intervals) / np.mean(beacon_intervals)
            features['communication_pattern_entropy'] = self._calculate_pattern_entropy(beacon_intervals)
        else:
            features['beacon_interval_irregularity'] = 0
            features['communication_pattern_entropy'] = 0
        
        # Attack-specific features
        features['flooding_indicator'] = 1.0 if features['beacon_rate_mean'] > 50 else 0.0
        features['jamming_indicator'] = 1.0 if features['tx_power_mean'] > 30 else 0.0
        features['async_indicator'] = 1.0 if features['beacon_interval_irregularity'] > 0.5 else 0.0
        
        # Overall anomaly score
        features['anomaly_score'] = (
            min(1.0, features['beacon_rate_range'] / 100) * 0.3 +
            min(1.0, abs(features['tx_power_anomaly_score'])) * 0.3 +
            min(1.0, features['beacon_interval_irregularity']) * 0.4
        )
        
        # Store features
        attack_metrics.detection_features = features
    
    def _calculate_pattern_entropy(self, intervals: List[float]) -> float:
        """Calculate entropy of communication pattern"""
        if not intervals:
            return 0.0
        
        # Discretize intervals into bins
        bins = np.histogram(intervals, bins=10)[0]
        probs = bins / np.sum(bins)
        probs = probs[probs > 0]  # Remove zero probabilities
        
        if len(probs) <= 1:
            return 0.0
        
        entropy = -np.sum(probs * np.log2(probs))
        return entropy / np.log2(len(probs))  # Normalize
    
    def get_detection_dataset(self) -> List[Dict]:
        """Generate dataset for ML-based attack detection - FIXED to include both normal and attacker vehicles"""
        dataset = []
        
        for vehicle_id, metrics in self.attack_metrics.items():
            if not metrics.detection_features:
                continue
            
            record = {
                'vehicle_id': vehicle_id,
                'is_attacker': metrics.is_attacker,
                'attack_type': metrics.attack_type.value if metrics.is_attacker else 'NORMAL',
                'timestamp': time.time(),
                **metrics.detection_features
            }
            dataset.append(record)
        
        # Debug output to verify dataset composition
        attacker_count = sum(1 for record in dataset if record['is_attacker'])
        normal_count = sum(1 for record in dataset if not record['is_attacker'])
        
        print(f"[DATASET DEBUG] Generated {len(dataset)} records: {attacker_count} attackers, {normal_count} normal vehicles")
        
        return dataset
    
    def is_attacker(self, vehicle_id: str) -> bool:
        """Check if vehicle is an attacker"""
        return vehicle_id in self.attackers
    
    def get_attack_metrics(self, vehicle_id: str) -> AttackMetrics:
        """Get attack metrics for vehicle"""
        return self.attack_metrics.get(vehicle_id, AttackMetrics())
    
    def get_attack_summary(self) -> Dict[str, Any]:
        """Get summary of attack simulation"""
        if not ENABLE_ATTACK_SIMULATION:
            return {}
        
        attacker_count = len(self.attackers)
        normal_count = len(self.normal_vehicles)
        
        attack_types = [config.attack_type.value for config in self.attackers.values()]
        attack_type_counts = {attack_type: attack_types.count(attack_type) for attack_type in set(attack_types)}
        
        return {
            'attack_enabled': ENABLE_ATTACK_SIMULATION,
            'total_attackers': attacker_count,
            'total_normal_vehicles': normal_count,
            'attack_type_distribution': attack_type_counts,
            'attacker_selection_strategy': ATTACKER_SELECTION_STRATEGY,
            'attack_start_time': self.attack_start_time,
            'attack_active': self.attack_active
        }

class IEEE80211bdMapper:
    """IEEE 802.11bd performance mapping with Layer 3 integration"""
    
    def __init__(self):
        # CORRECTED: IEEE 802.11bd MCS table (MCS 0-9, not 0-10) from script 2
        self.mcs_table = {
            0: {'modulation': 'BPSK-DCM', 'order': 2, 'code_rate': 0.5, 'data_rate': 1.5, 'snr_threshold': 1.0, 'dcm': True},
            1: {'modulation': 'BPSK', 'order': 2, 'code_rate': 0.5, 'data_rate': 3.0, 'snr_threshold': 3.0, 'dcm': False},
            2: {'modulation': 'QPSK', 'order': 4, 'code_rate': 0.5, 'data_rate': 6.0, 'snr_threshold': 6.0, 'dcm': False},
            3: {'modulation': 'QPSK', 'order': 4, 'code_rate': 0.75, 'data_rate': 9.0, 'snr_threshold': 9.0, 'dcm': False},
            4: {'modulation': '16-QAM', 'order': 16, 'code_rate': 0.5, 'data_rate': 12.0, 'snr_threshold': 12.0, 'dcm': False},
            5: {'modulation': '16-QAM', 'order': 16, 'code_rate': 0.75, 'data_rate': 18.0, 'snr_threshold': 15.0, 'dcm': False},
            6: {'modulation': '64-QAM', 'order': 64, 'code_rate': 0.5, 'data_rate': 18.0, 'snr_threshold': 18.0, 'dcm': False},
            7: {'modulation': '64-QAM', 'order': 64, 'code_rate': 0.667, 'data_rate': 24.0, 'snr_threshold': 21.0, 'dcm': False},
            8: {'modulation': '64-QAM', 'order': 64, 'code_rate': 0.75, 'data_rate': 27.0, 'snr_threshold': 24.0, 'dcm': False},
            9: {'modulation': '256-QAM', 'order': 256, 'code_rate': 0.75, 'data_rate': 36.0, 'snr_threshold': 30.0, 'dcm': False}
        }
        
        self.data_rates = {mcs: config['data_rate'] for mcs, config in self.mcs_table.items()}
        
        self.snr_thresholds = {
            mcs: {
                'success': config['snr_threshold'],
                'marginal': config['snr_threshold'] - 2.0,
                'failure': config['snr_threshold'] - 5.0
            }
            for mcs, config in self.mcs_table.items()
        }
        
        self.max_frame_efficiency = {
            0: 0.92, 1: 0.90, 2: 0.88, 3: 0.86, 4: 0.84, 
            5: 0.82, 6: 0.80, 7: 0.78, 8: 0.76, 9: 0.74
        }
        
        self.application_configs = {
            'safety': {
                'packet_size_bytes': 350,
                'target_per': 0.01,
                'target_pdr': 0.99,
                'latency_requirement_ms': 10,
                'reliability': 'high'
            },
            'high_throughput': {
                'packet_size_bytes': 300,
                'target_per': 0.05,
                'target_pdr': 0.95,
                'latency_requirement_ms': 100,
                'reliability': 'medium'
            }
        }
    
    def get_ber_from_sinr(self, sinr_db: float, mcs: int, enable_ldpc: bool = True) -> float:
        """Calculate BER from SINR using proper IEEE 802.11bd modulation schemes"""
        if mcs not in self.mcs_table:
            mcs = 1
        
        mcs_config = self.mcs_table[mcs]
        modulation = mcs_config['modulation']
        is_dcm = mcs_config.get('dcm', False)
        
        required_sinr = self.snr_thresholds[mcs]['success']
        if sinr_db < required_sinr - 8.0:
            return 0.5
        
        sinr_linear = 10**(sinr_db / 10.0)
        
        # LDPC coding gain
        if enable_ldpc:
            code_rate = mcs_config['code_rate']
            ldpc_gain_db = 2.0 + (1.0 - code_rate) * 1.5
            ldpc_gain_linear = 10**(ldpc_gain_db / 10.0)
            sinr_linear *= ldpc_gain_linear
        
        # DCM diversity gain
        if is_dcm:
            dcm_gain_linear = 10**(3.0 / 10.0)  # 3 dB gain
            sinr_linear *= dcm_gain_linear
        
        try:
            if modulation in ['BPSK-DCM', 'BPSK']:
                ber = 0.5 * math.erfc(math.sqrt(sinr_linear))
            elif modulation == 'QPSK':
                ber = 0.5 * math.erfc(math.sqrt(sinr_linear / 2.0))
            elif modulation == '16-QAM':
                sqrt_sinr = math.sqrt(sinr_linear / 10.0)
                ber = (3.0/8.0) * math.erfc(sqrt_sinr) + (1.0/8.0) * math.erfc(3.0 * sqrt_sinr)
            elif modulation == '64-QAM':
                sqrt_sinr = math.sqrt(sinr_linear / 42.0)
                ber = (7.0/24.0) * math.erfc(sqrt_sinr) + (1.0/24.0) * math.erfc(5.0 * sqrt_sinr)
            elif modulation == '256-QAM':
                sqrt_sinr = math.sqrt(sinr_linear / 170.0)
                ber = (15.0/64.0) * math.erfc(sqrt_sinr) + (1.0/64.0) * math.erfc(7.0 * sqrt_sinr)
            else:
                ber = 0.5 * math.erfc(math.sqrt(sinr_linear / 2.0))
            
            ber = max(1e-12, min(0.5, ber))
        except:
            ber = 0.1 if sinr_db > required_sinr else 0.4
        
        return ber
    
    def get_ser_from_ber(self, ber: float, mcs: int) -> float:
        """Calculate SER from BER with proper Gray coding theory"""
        if mcs not in self.mcs_table:
            mcs = 1
        
        mcs_config = self.mcs_table[mcs]
        modulation_order = mcs_config['order']
        
        if modulation_order == 2:
            ser = ber
        else:
            bits_per_symbol = math.log2(modulation_order)
            if modulation_order == 4:
                ser = 2 * ber * (1 - ber)
            elif modulation_order == 16:
                ser = 1.0 - (1.0 - ber)**(bits_per_symbol * 0.75)
            elif modulation_order == 64:
                ser = 1.0 - (1.0 - ber)**(bits_per_symbol * 0.8)
            elif modulation_order >= 256:
                ser = 1.0 - (1.0 - ber)**(bits_per_symbol * 0.85)
            else:
                ser = 1.0 - (1.0 - ber)**bits_per_symbol
        
        return max(1e-12, min(0.999, ser))
    
    def get_per_from_ser(self, ser: float, packet_length_bits: int, mcs: int, enable_ldpc: bool = True) -> float:
        """Calculate PER from SER with proper OFDM and LDPC structure"""
        if ser <= 1e-12:
            return 1e-12
        
        data_subcarriers = 48  # 10 MHz bandwidth
        mcs_config = self.mcs_table[mcs]
        modulation_order = mcs_config['order']
        code_rate = mcs_config['code_rate']
        
        bits_per_subcarrier = math.log2(modulation_order)
        coded_bits_per_ofdm_symbol = data_subcarriers * bits_per_subcarrier
        info_bits_per_ofdm_symbol = coded_bits_per_ofdm_symbol * code_rate
        
        symbols_per_packet = math.ceil(packet_length_bits / info_bits_per_ofdm_symbol)
        
        if enable_ldpc:
            max_correctable_errors = max(1, int(symbols_per_packet * 0.15))
            per = 0.0
            for k in range(max_correctable_errors + 1, symbols_per_packet + 1):
                try:
                    prob_k_errors = math.comb(symbols_per_packet, k) * (ser**k) * ((1-ser)**(symbols_per_packet-k))
                    per += prob_k_errors
                except OverflowError:
                    per = 1.0 - (1.0 - ser)**symbols_per_packet
                    break
        else:
            per = 1.0 - (1.0 - ser)**symbols_per_packet
        
        return max(1e-12, min(0.999, per))
    
    def get_per_from_snr(self, snr_db: float, mcs: int, packet_length_bits: int = None) -> float:
        """Get PER from SINR using correct BER->SER->PER calculation flow"""
        if packet_length_bits is None:
            packet_length_bits = (100 + 36) * 8
        
        ber = self.get_ber_from_sinr(snr_db, mcs)
        ser = self.get_ser_from_ber(ber, mcs)
        per = self.get_per_from_ser(ser, packet_length_bits, mcs)
        
        return per
    
    def get_cbr_collision_probability(self, cbr: float, neighbor_count: int, beacon_rate: float = 10.0) -> float:
        """ENHANCED collision probability with slightly more aggressive neighbor impact"""
        if neighbor_count == 0:
            return 0.001
        
        slot_time = 9e-6
        beacon_interval = 1.0 / beacon_rate
        slots_per_beacon = beacon_interval / slot_time
        base_tx_prob = 1.0 / slots_per_beacon
        
        # ENHANCED: More aggressive CBR impact
        if cbr <= 0.2:
            cbr_factor = 1.0
        elif cbr <= 0.4:
            cbr_factor = 1.3   # Increased from 1.25
        elif cbr <= 0.6:
            cbr_factor = 1.8   # Increased from 1.6
        elif cbr <= 0.8:
            cbr_factor = 2.5   # Increased from 2.2
        else:
            excess_cbr = cbr - 0.8
            cbr_factor = 2.5 + (excess_cbr * 4.5)  # Increased from 4.0
            cbr_factor = min(cbr_factor, 5.0)      # Increased cap from 4.0
        
        # ENHANCED: More aggressive neighbor impact with non-linear scaling
        if neighbor_count <= 5:
            neighbor_multiplier = 1.0 + (neighbor_count * 0.15)  # Increased from 0.12
        elif neighbor_count <= 10:
            neighbor_multiplier = 1.75 + ((neighbor_count - 5) * 0.2)  # More aggressive scaling
        else:
            neighbor_multiplier = 2.75 + ((neighbor_count - 10) * 0.25)  # Even more aggressive for high density
        
        effective_tx_prob = min(0.15, base_tx_prob * cbr_factor * neighbor_multiplier)  # Increased cap
        
        # ENHANCED: More realistic collision probability with neighbor density bonus
        if neighbor_count == 1:
            collision_prob = effective_tx_prob * 0.8
        else:
            collision_prob = 1.0 - (1.0 - effective_tx_prob)**(neighbor_count)
        
        # ENHANCED: More aggressive hidden terminal and density effects
        hidden_terminal_prob = min(0.03, neighbor_count * 0.001)  # Increased
        
        # NEW: Additional neighbor density collision penalty
        if neighbor_count > 8:
            density_bonus = (neighbor_count - 8) * 0.002  # Additional 0.2% per neighbor above 8
            density_collision_prob = min(0.03, density_bonus)
        else:
            density_collision_prob = 0
        
        total_collision_prob = collision_prob + hidden_terminal_prob + density_collision_prob
        
        return min(0.5, total_collision_prob)  # Increased cap from 0.45
    
    def get_mac_efficiency(self, cbr: float, per: float, neighbor_count: int) -> float:
        """ENHANCED MAC efficiency with stronger neighbor impact modeling"""
        
        # IEEE 802.11bd base efficiency is higher than 802.11p but still affected by congestion
        base_efficiency = 0.82  # Slightly reduced from 0.85 for more realistic modeling
        
        # ENHANCED: More aggressive CBR impact with non-linear scaling
        if cbr <= 0.2:
            cbr_efficiency = base_efficiency
        elif cbr <= 0.4:
            cbr_efficiency = base_efficiency * (1.0 - (cbr - 0.2) * 1.5)  # Increased from 1.2
        elif cbr <= 0.6:
            cbr_efficiency = base_efficiency * (1.0 - (cbr - 0.2) * 1.8)  # Increased from 1.5
        elif cbr <= 0.8:
            cbr_efficiency = base_efficiency * (1.0 - (cbr - 0.2) * 2.2)  # Increased from 1.8
        else:
            # ENHANCED: Very aggressive efficiency loss at high CBR
            excess_cbr = cbr - 0.8
            efficiency_loss = (cbr - 0.2) * 2.2 + (excess_cbr * 3.0)  # Additional penalty for very high CBR
            cbr_efficiency = base_efficiency * (1.0 - efficiency_loss)
        
        cbr_efficiency = max(0.15, cbr_efficiency)  # Reduced minimum from 0.25
        
        # ENHANCED: More aggressive neighbor penalty with non-linear scaling
        if neighbor_count == 0:
            neighbor_penalty = 1.0
        elif neighbor_count <= 5:
            neighbor_penalty = 1.0 - 0.015 * neighbor_count  # Increased from 0.01
        elif neighbor_count <= 15:
            neighbor_penalty = 0.925 - 0.02 * (neighbor_count - 5)  # Increased from 0.015
        elif neighbor_count <= 25:
            neighbor_penalty = 0.725 - 0.025 * (neighbor_count - 15)  # Increased penalty
        else:
            neighbor_penalty = 0.475 - 0.008 * min(20, neighbor_count - 25)  # More aggressive for very dense
        
        neighbor_penalty = max(0.15, neighbor_penalty)  # Reduced minimum from 0.3
        
        # ENHANCED: More realistic overhead calculations with stronger impact
        contention_overhead = 1.0 + (neighbor_count * 0.004)**1.2  # Increased from 0.003
        retry_overhead = 1.0 + (per * neighbor_count * 0.03)  # Increased from 0.02
        
        # ENHANCED: Additional overhead for high neighbor density
        if neighbor_count > 15:
            density_overhead = 1.0 + ((neighbor_count - 15) * 0.02)  # Additional overhead
        else:
            density_overhead = 1.0
        
        # ENHANCED: PER impact on efficiency (higher PER = lower efficiency)
        per_efficiency_impact = 1.0 - (per * 0.3)  # PER directly reduces efficiency
        per_efficiency_impact = max(0.5, per_efficiency_impact)
        
        final_efficiency = (cbr_efficiency * neighbor_penalty * per_efficiency_impact) / (contention_overhead * retry_overhead * density_overhead)
        
        return max(0.15, min(0.82, final_efficiency))  # Adjusted bounds

# AODV Routing Protocol Implementation
class AODVRoutingProtocol:
    """Enhanced AODV (Ad-hoc On-Demand Distance Vector) routing protocol implementation"""
    
    def __init__(self, node_id: str, config: SimulationConfig):
        self.node_id = node_id
        self.config = config
        self.routing_table: Dict[str, RouteEntry] = {}
        self.sequence_number = random.randint(1, 65535)  # Own sequence number
        self.rreq_id = 0
        self.pending_routes: Dict[str, Dict] = {}  # destination -> {timeout, rreq_id, retry_count}
        self.route_cache: Dict[str, List[str]] = {}
        self.rreq_buffer: Dict[str, float] = {}  # Track seen RREQs: "source_rreq_id" -> timestamp
        self.packet_buffer: Dict[str, List[NetworkPacket]] = {}  # Buffer packets during route discovery
        
        # AODV specific parameters
        self.active_route_timeout = 30.0  # seconds
        self.hello_interval = 5.0  # seconds
        self.allowed_hello_loss = 3
        self.rreq_retries = 3
        self.rreq_retry_timeout = 2.0  # seconds
        self.net_diameter = 35  # network diameter in hops
        self.node_traversal_time = 0.04  # 40ms
        
        # Neighbor management
        self.neighbors: Dict[str, Dict] = {}  # neighbor_id -> {last_seen, hello_count, valid}
        self.last_hello_time = 0.0
        
        # Statistics
        self.stats = {
            'rreq_sent': 0,
            'rreq_received': 0,
            'rrep_sent': 0,
            'rrep_received': 0,
            'rerr_sent': 0,
            'rerr_received': 0,
            'routes_discovered': 0,
            'route_errors': 0
        }
    
    def initiate_route_discovery(self, destination: str) -> Optional[NetworkPacket]:
        """Enhanced route discovery with retry mechanism and packet buffering"""
        current_time = time.time()
        
        # Check if route discovery already in progress
        if destination in self.pending_routes:
            pending_info = self.pending_routes[destination]
            if current_time < pending_info['timeout']:
                return None  # Discovery already in progress
            else:
                # Previous discovery timed out, increment retry
                if pending_info['retry_count'] >= self.rreq_retries:
                    # Max retries reached, cleanup
                    del self.pending_routes[destination]
                    self._drop_buffered_packets(destination)
                    return None
                else:
                    pending_info['retry_count'] += 1
                    pending_info['timeout'] = current_time + self.rreq_retry_timeout
        else:
            # New route discovery
            self.pending_routes[destination] = {
                'timeout': current_time + self.rreq_retry_timeout,
                'rreq_id': self.rreq_id + 1,
                'retry_count': 1
            }
        
        self.rreq_id += 1
        self.sequence_number += 1
        
        # Get destination sequence number if available
        dest_seq = 0
        if destination in self.routing_table:
            dest_seq = self.routing_table[destination].sequence_number
        
        # Create RREQ packet
        rreq_packet = NetworkPacket(
            packet_id=f"RREQ_{self.node_id}_{self.rreq_id}",
            packet_type=PacketType.RREQ,
            source_id=self.node_id,
            destination_id="BROADCAST",
            source_ip=f"192.168.1.{hash(self.node_id) % 254 + 1}",
            destination_ip="255.255.255.255",
            payload_size=64,
            qos_class=QoSClass.SERVICE,
            application_type="ROUTING",
            ttl=self.net_diameter,
            rreq_id=self.rreq_id,
            destination_sequence=dest_seq,
            creation_time=current_time,
            route=[self.node_id]  # Initialize route with source
        )
        
        self.stats['rreq_sent'] += 1
        return rreq_packet
    
    def process_rreq(self, packet: NetworkPacket, sender: str, current_time: float) -> Optional[NetworkPacket]:
        """Enhanced RREQ processing with loop prevention and proper route establishment"""
        self.stats['rreq_received'] += 1
        
        # Check for duplicate RREQ
        rreq_key = f"{packet.source_id}_{packet.rreq_id}"
        if rreq_key in self.rreq_buffer:
            if current_time - self.rreq_buffer[rreq_key] < 5.0:  # Seen within 5 seconds
                return None  # Drop duplicate
        self.rreq_buffer[rreq_key] = current_time
        
        # Check if we are in the route (loop prevention)
        if self.node_id in packet.route:
            return None  # Drop to prevent loops
        
        # Update/create route to source
        source_hop_count = len(packet.route)
        source_route_entry = self.routing_table.get(packet.source_id)
        
        create_reverse_route = False
        if not source_route_entry:
            create_reverse_route = True
        elif (source_hop_count < source_route_entry.hop_count or 
              packet.destination_sequence > source_route_entry.sequence_number):
            create_reverse_route = True
        
        if create_reverse_route:
            self.routing_table[packet.source_id] = RouteEntry(
                destination=packet.source_id,
                next_hop=sender,
                hop_count=source_hop_count,
                metric=source_hop_count,
                sequence_number=packet.destination_sequence,
                lifetime=current_time + self.active_route_timeout,
                route_type="DYNAMIC",
                timestamp=current_time
            )
        
        # Check if we are the destination
        destination_node = packet.destination_id if hasattr(packet, 'destination_id') else None
        if not destination_node:
            # Extract destination from RREQ payload (simplified)
            for node_id in packet.route:
                if node_id != packet.source_id and node_id != self.node_id:
                    destination_node = node_id
                    break
        
        if destination_node and destination_node == self.node_id:
            # We are the destination, send RREP
            self.sequence_number += 1
            
            # Create route back through the path
            reverse_route = packet.route.copy()
            reverse_route.reverse()
            
            rrep_packet = NetworkPacket(
                packet_id=f"RREP_{self.node_id}_{packet.rreq_id}",
                packet_type=PacketType.RREP,
                source_id=self.node_id,
                destination_id=packet.source_id,
                source_ip=f"192.168.1.{hash(self.node_id) % 254 + 1}",
                destination_ip=packet.source_ip,
                payload_size=64,
                qos_class=QoSClass.SERVICE,
                application_type="ROUTING",
                ttl=64,
                rreq_id=packet.rreq_id,
                destination_sequence=self.sequence_number,
                hop_count=source_hop_count,
                route=reverse_route,
                creation_time=current_time
            )
            
            self.stats['rrep_sent'] += 1
            return rrep_packet
        
        # Check if we have a fresh route to destination
        dest_route_entry = self.routing_table.get(destination_node) if destination_node else None
        if (dest_route_entry and 
            dest_route_entry.sequence_number >= packet.destination_sequence and
            current_time < dest_route_entry.lifetime):
            
            # Send intermediate RREP
            rrep_packet = NetworkPacket(
                packet_id=f"RREP_{self.node_id}_{packet.rreq_id}",
                packet_type=PacketType.RREP,
                source_id=self.node_id,
                destination_id=packet.source_id,
                source_ip=f"192.168.1.{hash(self.node_id) % 254 + 1}",
                destination_ip=packet.source_ip,
                payload_size=64,
                qos_class=QoSClass.SERVICE,
                application_type="ROUTING",
                ttl=64,
                rreq_id=packet.rreq_id,
                destination_sequence=dest_route_entry.sequence_number,
                hop_count=source_hop_count + dest_route_entry.hop_count,
                creation_time=current_time
            )
            
            self.stats['rrep_sent'] += 1
            return rrep_packet
        
        # Forward RREQ if TTL allows
        if packet.ttl > 1:
            # Add ourselves to the route
            packet.route.append(self.node_id)
            packet.ttl -= 1
            packet.hop_count = len(packet.route) - 1
            return packet
        
        return None
    
    def process_rrep(self, packet: NetworkPacket, sender: str, current_time: float) -> bool:
        """Enhanced RREP processing with route establishment and packet forwarding"""
        self.stats['rrep_received'] += 1
        
        # If this RREP is for us, install the route
        if packet.destination_id == self.node_id:
            route_destination = packet.source_id
            
            # Install route to destination
            self.routing_table[route_destination] = RouteEntry(
                destination=route_destination,
                next_hop=sender,
                hop_count=packet.hop_count,
                metric=packet.hop_count,
                sequence_number=packet.destination_sequence,
                lifetime=current_time + self.active_route_timeout,
                route_type="DYNAMIC",
                timestamp=current_time
            )
            
            # Remove from pending routes
            if route_destination in self.pending_routes:
                del self.pending_routes[route_destination]
            
            # Forward buffered packets
            self._forward_buffered_packets(route_destination)
            
            self.stats['routes_discovered'] += 1
            return True
        else:
            # Forward RREP to next hop toward destination
            dest_route = self.routing_table.get(packet.destination_id)
            if dest_route and current_time < dest_route.lifetime:
                return True  # Indicate should be forwarded
        
        return False
    
    def process_rerr(self, packet: NetworkPacket, sender: str, current_time: float):
        """Process Route Error messages"""
        self.stats['rerr_received'] += 1
        
        # Remove broken routes
        unreachable_destinations = getattr(packet, 'unreachable_destinations', [])
        for dest in unreachable_destinations:
            if dest in self.routing_table:
                route = self.routing_table[dest]
                if route.next_hop == sender:
                    del self.routing_table[dest]
                    self.stats['route_errors'] += 1
    
    def generate_hello_packet(self, current_time: float) -> Optional[NetworkPacket]:
        """Generate HELLO packet for neighbor discovery with current_time parameter"""
        if current_time - self.last_hello_time < self.hello_interval:
            return None
        
        self.last_hello_time = current_time
        
        hello_packet = NetworkPacket(
            packet_id=f"HELLO_{self.node_id}_{current_time}",
            packet_type=PacketType.HELLO,
            source_id=self.node_id,
            destination_id="BROADCAST",
            source_ip=f"192.168.1.{hash(self.node_id) % 254 + 1}",
            destination_ip="255.255.255.255",
            payload_size=32,
            qos_class=QoSClass.SERVICE,
            application_type="ROUTING",
            ttl=1,  # HELLO packets are not forwarded
            sequence_number=self.sequence_number,
            creation_time=current_time
        )
        
        return hello_packet
    
    def process_hello_packet(self, packet: NetworkPacket, sender: str, current_time: float):
        """Process received HELLO packet for neighbor management"""
        if sender not in self.neighbors:
            self.neighbors[sender] = {
                'last_seen': current_time,
                'hello_count': 1,
                'valid': True,
                'sequence_number': packet.sequence_number
            }
        else:
            neighbor_info = self.neighbors[sender]
            neighbor_info['last_seen'] = current_time
            neighbor_info['hello_count'] += 1
            neighbor_info['sequence_number'] = packet.sequence_number
    
    def get_next_hop(self, destination: str, current_time: float = None) -> Optional[str]:
        """Get next hop for destination with route validation"""
        if current_time is None:
            current_time = time.time()
        
        if destination in self.routing_table:
            route = self.routing_table[destination]
            if current_time < route.lifetime:
                # Validate that next hop is still a neighbor
                if route.next_hop in self.neighbors:
                    neighbor_info = self.neighbors[route.next_hop]
                    if (current_time - neighbor_info['last_seen'] < 
                        self.hello_interval * self.allowed_hello_loss):
                        return route.next_hop
                
                # Next hop no longer valid, remove route
                del self.routing_table[destination]
                self.stats['route_errors'] += 1
        
        return None
    
    def buffer_packet(self, packet: NetworkPacket):
        """Buffer packet during route discovery"""
        destination = packet.destination_id
        if destination not in self.packet_buffer:
            self.packet_buffer[destination] = []
        
        # Limit buffer size
        if len(self.packet_buffer[destination]) < 10:
            self.packet_buffer[destination].append(packet)
    
    def _forward_buffered_packets(self, destination: str):
        """Forward buffered packets after route discovery"""
        if destination in self.packet_buffer:
            packets = self.packet_buffer[destination]
            del self.packet_buffer[destination]
            # In a real implementation, these would be forwarded
            # For simulation, we just mark them as delivered
            return len(packets)
        return 0
    
    def _drop_buffered_packets(self, destination: str):
        """Drop buffered packets when route discovery fails"""
        if destination in self.packet_buffer:
            dropped_count = len(self.packet_buffer[destination])
            del self.packet_buffer[destination]
            return dropped_count
        return 0
    
    def cleanup_expired_routes(self, current_time: float = None):
        """Enhanced route cleanup with neighbor validation"""
        if current_time is None:
            current_time = time.time()
        
        # Cleanup expired routes
        expired_routes = []
        for dest, route in self.routing_table.items():
            if current_time >= route.lifetime:
                expired_routes.append(dest)
        
        for dest in expired_routes:
            del self.routing_table[dest]
        
        # Cleanup expired neighbors
        expired_neighbors = []
        for neighbor_id, neighbor_info in self.neighbors.items():
            if (current_time - neighbor_info['last_seen'] > 
                self.hello_interval * self.allowed_hello_loss):
                expired_neighbors.append(neighbor_id)
        
        for neighbor_id in expired_neighbors:
            del self.neighbors[neighbor_id]
        
        # Cleanup expired pending route discoveries
        expired_pending = []
        for dest, pending_info in self.pending_routes.items():
            if current_time > pending_info['timeout']:
                expired_pending.append(dest)
        
        for dest in expired_pending:
            del self.pending_routes[dest]
            self._drop_buffered_packets(dest)
        
        # Cleanup old RREQ buffer entries
        old_rreqs = []
        for rreq_key, timestamp in self.rreq_buffer.items():
            if current_time - timestamp > 10.0:  # 10 second cleanup
                old_rreqs.append(rreq_key)
        
        for rreq_key in old_rreqs:
            del self.rreq_buffer[rreq_key]
    
    def get_routing_statistics(self) -> Dict:
        """Get comprehensive routing statistics"""
        return {
            **self.stats,
            'routing_table_size': len(self.routing_table),
            'neighbor_count': len(self.neighbors),
            'pending_discoveries': len(self.pending_routes),
            'buffered_packets': sum(len(packets) for packets in self.packet_buffer.values())
        }
    
    def cleanup_expired_entries(self, current_time: float = None):
        """Unified cleanup method for consistency"""
        self.cleanup_expired_routes(current_time)

# OLSR Routing Protocol Implementation
class OLSRRoutingProtocol:
    """Enhanced OLSR (Optimized Link State Routing) protocol implementation"""
    
    def __init__(self, node_id: str, config: SimulationConfig):
        self.node_id = node_id
        self.config = config
        
        # Core OLSR data structures
        self.neighbor_set: Dict[str, Dict] = {}  # One-hop neighbors
        self.two_hop_neighbors: Dict[str, Set[str]] = {}  # Two-hop neighbors via each neighbor
        self.topology_set: Dict[str, Dict[str, Dict]] = {}  # Network topology
        self.routing_table: Dict[str, RouteEntry] = {}
        self.mpr_set: Set[str] = set()  # Our selected MPRs
        self.mpr_selector_set: Set[str] = set()  # Nodes that selected us as MPR
        
        # Sequence numbers
        self.ansn = 0  # Advertised Neighbor Sequence Number
        self.packet_sequence = 0
        
        # Timing parameters
        self.hello_interval = 2.0  # seconds
        self.tc_interval = 5.0  # seconds
        self.neighbor_hold_time = 6.0  # 3 * hello_interval
        self.topology_hold_time = 15.0  # 3 * tc_interval
        self.duplicate_hold_time = 30.0
        
        # Last message times
        self.last_hello_time = 0.0
        self.last_tc_time = 0.0
        
        # Message sequence tracking
        self.duplicate_set: Dict[str, float] = {}  # "originator_seq" -> timestamp
        
        # Link quality tracking
        self.link_quality: Dict[str, Dict] = {}  # neighbor -> {quality, symmetric}
        
        # Statistics
        self.stats = {
            'hello_sent': 0,
            'hello_received': 0,
            'tc_sent': 0,
            'tc_received': 0,
            'mpr_selections': 0,
            'routing_calculations': 0
        }
    
    def generate_hello_packet(self, current_time: float) -> Optional[NetworkPacket]:
        """Generate enhanced HELLO packet with neighbor information"""
        if current_time - self.last_hello_time < self.hello_interval:
            return None
        
        self.last_hello_time = current_time
        self.packet_sequence += 1
        
        # Include neighbor information in HELLO
        neighbor_info = {}
        for neighbor_id, neighbor_data in self.neighbor_set.items():
            if current_time - neighbor_data['last_seen'] < self.neighbor_hold_time:
                neighbor_info[neighbor_id] = {
                    'link_type': neighbor_data.get('link_type', 'ASYM'),
                    'neighbor_type': neighbor_data.get('neighbor_type', 'NOT_MPR')
                }
        
        hello_packet = NetworkPacket(
            packet_id=f"HELLO_{self.node_id}_{self.packet_sequence}",
            packet_type=PacketType.HELLO,
            source_id=self.node_id,
            destination_id="BROADCAST",
            source_ip=f"192.168.1.{hash(self.node_id) % 254 + 1}",
            destination_ip="255.255.255.255",
            payload_size=64 + len(neighbor_info) * 8,  # Base + neighbor info
            qos_class=QoSClass.SERVICE,
            application_type="ROUTING",
            ttl=1,  # HELLO packets are not forwarded
            originator_addr=self.node_id,
            sequence_number=self.packet_sequence,
            creation_time=current_time
        )
        
        # Store neighbor info in packet (simulation)
        hello_packet.neighbor_info = neighbor_info
        
        self.stats['hello_sent'] += 1
        return hello_packet
    
    def process_hello_packet(self, packet: NetworkPacket, sender: str, current_time: float):
        """Enhanced HELLO packet processing with link sensing and MPR calculation"""
        # Add null checks
        if not packet or not sender:
            return
        
        self.stats['hello_received'] += 1
        
        # Update neighbor information
        if sender not in self.neighbor_set:
            self.neighbor_set[sender] = {
                'last_seen': current_time,
                'first_seen': current_time,
                'willingness': 3,  # Default willingness
                'link_type': 'ASYM',  # Start as asymmetric
                'neighbor_type': 'NOT_MPR'
            }
        
        neighbor_data = self.neighbor_set[sender]
        neighbor_data['last_seen'] = current_time
        
        # Process neighbor information from HELLO (with safe access)
        sender_neighbors = getattr(packet, 'neighbor_info', {})
        if not isinstance(sender_neighbors, dict):
            sender_neighbors = {}
        
        # Check if sender can hear us (symmetric link)
        if self.node_id in sender_neighbors:
            neighbor_data['link_type'] = 'SYM'
            
            # Update link quality
            if sender not in self.link_quality:
                self.link_quality[sender] = {'quality': 1.0, 'symmetric': True}
            self.link_quality[sender]['symmetric'] = True
        else:
            neighbor_data['link_type'] = 'ASYM'
            if sender in self.link_quality:
                self.link_quality[sender]['symmetric'] = False
        
        # Update two-hop neighbor information
        self._update_two_hop_neighbors(sender, sender_neighbors, current_time)
        
        # Check if we are selected as MPR by this neighbor
        our_info = sender_neighbors.get(self.node_id, {})
        if our_info.get('neighbor_type') == 'MPR':
            self.mpr_selector_set.add(sender)
        else:
            self.mpr_selector_set.discard(sender)
        
        # Trigger MPR recalculation
        self._select_mprs(current_time)
    
    def _update_two_hop_neighbors(self, neighbor_id: str, neighbor_info: Dict, current_time: float):
        """Update two-hop neighbor information"""
        # Clear old two-hop neighbors via this neighbor
        if neighbor_id in self.two_hop_neighbors:
            self.two_hop_neighbors[neighbor_id].clear()
        else:
            self.two_hop_neighbors[neighbor_id] = set()
        
        # Add current two-hop neighbors
        for reported_neighbor in neighbor_info.keys():
            if (reported_neighbor != self.node_id and 
                reported_neighbor not in self.neighbor_set):
                self.two_hop_neighbors[neighbor_id].add(reported_neighbor)
    
    def generate_tc_packet(self, current_time: float) -> Optional[NetworkPacket]:
        """Generate enhanced TC packet if we are an MPR"""
        if current_time - self.last_tc_time < self.tc_interval:
            return None
        
        # Only generate TC if we have MPR selectors
        if not self.mpr_selector_set:
            return None
        
        self.last_tc_time = current_time
        self.ansn += 1
        self.packet_sequence += 1
        
        # Include our MPR selector set in TC
        mpr_selector_info = list(self.mpr_selector_set)
        
        tc_packet = NetworkPacket(
            packet_id=f"TC_{self.node_id}_{self.ansn}",
            packet_type=PacketType.TC,
            source_id=self.node_id,
            destination_id="BROADCAST",
            source_ip=f"192.168.1.{hash(self.node_id) % 254 + 1}",
            destination_ip="255.255.255.255",
            payload_size=64 + len(mpr_selector_info) * 4,
            qos_class=QoSClass.SERVICE,
            application_type="ROUTING",
            ttl=255,  # TC packets are flooded
            originator_addr=self.node_id,
            ansn=self.ansn,
            sequence_number=self.packet_sequence,
            creation_time=current_time
        )
        
        # Store MPR selector info in packet
        tc_packet.mpr_selector_info = mpr_selector_info
        
        self.stats['tc_sent'] += 1
        return tc_packet
    
    def process_tc_packet(self, packet: NetworkPacket, sender: str, current_time: float) -> bool:
        """Enhanced TC packet processing with duplicate detection and topology update"""
        # Check for duplicate
        dup_key = f"{packet.originator_addr}_{packet.ansn}"
        if dup_key in self.duplicate_set:
            return False  # Duplicate, don't forward
        
        self.duplicate_set[dup_key] = current_time
        self.stats['tc_received'] += 1
        
        # Process topology information
        originator = packet.originator_addr
        
        if originator not in self.topology_set:
            self.topology_set[originator] = {}
        
        # Clear old topology entries for this originator
        self.topology_set[originator].clear()
        
        # Add new topology information
        mpr_selectors = getattr(packet, 'mpr_selector_info', [])
        for selector in mpr_selectors:
            self.topology_set[originator][selector] = {
                'last_update': current_time,
                'ansn': packet.ansn
            }
        
        # Mark topology as updated
        self.topology_set[originator]['_last_update'] = current_time
        
        # Trigger routing table calculation
        self._calculate_routing_table(current_time)
        
        # Forward if TTL allows and we are selected as MPR by sender
        if packet.ttl > 1 and sender in self.mpr_selector_set:
            packet.ttl -= 1
            return True
        
        return False
    
    def _select_mprs(self, current_time: float):
        """Enhanced MPR selection algorithm"""
        self.mpr_set.clear()
        
        # Get all symmetric neighbors
        symmetric_neighbors = []
        for neighbor_id, neighbor_data in self.neighbor_set.items():
            if (neighbor_data.get('link_type') == 'SYM' and 
                current_time - neighbor_data['last_seen'] < self.neighbor_hold_time):
                symmetric_neighbors.append(neighbor_id)
        
        if not symmetric_neighbors:
            return
        
        # Get all two-hop neighbors that need to be covered
        two_hop_to_cover = set()
        neighbor_coverage = {}  # neighbor -> set of two-hop neighbors it covers
        
        for neighbor_id in symmetric_neighbors:
            if neighbor_id in self.two_hop_neighbors:
                coverage = self.two_hop_neighbors[neighbor_id].copy()
                # Remove nodes that are already one-hop neighbors
                coverage = coverage - set(symmetric_neighbors)
                coverage.discard(self.node_id)
                
                neighbor_coverage[neighbor_id] = coverage
                two_hop_to_cover.update(coverage)
        
        if not two_hop_to_cover:
            # No two-hop neighbors to cover
            return
        
        # MPR selection algorithm
        uncovered = two_hop_to_cover.copy()
        
        # Step 1: Select neighbors that are the only way to reach some two-hop neighbors
        for two_hop in list(uncovered):
            covering_neighbors = [n for n, coverage in neighbor_coverage.items() 
                                if two_hop in coverage]
            if len(covering_neighbors) == 1:
                mpr = covering_neighbors[0]
                self.mpr_set.add(mpr)
                uncovered -= neighbor_coverage[mpr]
        
        # Step 2: Greedy selection - choose neighbors covering most uncovered two-hop neighbors
        while uncovered:
            best_neighbor = None
            best_coverage = 0
            
            for neighbor_id, coverage in neighbor_coverage.items():
                if neighbor_id not in self.mpr_set:
                    new_coverage = len(coverage & uncovered)
                    if new_coverage > best_coverage:
                        best_coverage = new_coverage
                        best_neighbor = neighbor_id
            
            if best_neighbor:
                self.mpr_set.add(best_neighbor)
                uncovered -= neighbor_coverage[best_neighbor]
            else:
                break  # No more progress possible
        
        # Update neighbor types
        for neighbor_id in symmetric_neighbors:
            if neighbor_id in self.mpr_set:
                self.neighbor_set[neighbor_id]['neighbor_type'] = 'MPR'
            else:
                self.neighbor_set[neighbor_id]['neighbor_type'] = 'SYM'
        
        self.stats['mpr_selections'] += 1
    
    def _calculate_routing_table(self, current_time: float):
        """Enhanced routing table calculation using Dijkstra's algorithm"""
        self.routing_table.clear()
        self.stats['routing_calculations'] += 1
        
        # Build network graph from topology information
        graph = {}  # node -> {neighbor: distance}
        nodes = {self.node_id}
        
        # Add one-hop neighbors
        for neighbor_id, neighbor_data in self.neighbor_set.items():
            if (neighbor_data.get('link_type') == 'SYM' and 
                current_time - neighbor_data['last_seen'] < self.neighbor_hold_time):
                
                if self.node_id not in graph:
                    graph[self.node_id] = {}
                graph[self.node_id][neighbor_id] = 1.0
                nodes.add(neighbor_id)
                
                # Add route to neighbor
                self.routing_table[neighbor_id] = RouteEntry(
                    destination=neighbor_id,
                    next_hop=neighbor_id,
                    hop_count=1,
                    metric=1.0,
                    sequence_number=0,
                    lifetime=current_time + self.neighbor_hold_time,
                    route_type="DYNAMIC",
                    timestamp=current_time
                )
        
        # Add topology information
        for originator, topology_info in self.topology_set.items():
            if '_last_update' in topology_info:
                if current_time - topology_info['_last_update'] < self.topology_hold_time:
                    nodes.add(originator)
                    if originator not in graph:
                        graph[originator] = {}
                    
                    for destination, link_info in topology_info.items():
                        if destination.startswith('_'):  # Skip metadata
                            continue
                        
                        if current_time - link_info['last_update'] < self.topology_hold_time:
                            graph[originator][destination] = 1.0
                            nodes.add(destination)
        
        # Run Dijkstra's algorithm
        distances = {node: float('inf') for node in nodes}
        distances[self.node_id] = 0
        previous = {}
        unvisited = nodes.copy()
        
        while unvisited:
            current = min(unvisited, key=lambda node: distances[node])
            if distances[current] == float('inf'):
                break
            
            unvisited.remove(current)
            
            if current in graph:
                for neighbor, weight in graph[current].items():
                    if neighbor in unvisited:
                        new_distance = distances[current] + weight
                        if new_distance < distances[neighbor]:
                            distances[neighbor] = new_distance
                            previous[neighbor] = current
        
        # Build routing table from shortest paths
        for destination in nodes:
            if destination != self.node_id and destination in previous:
                # Find next hop
                path = []
                current = destination
                while current in previous:
                    path.append(current)
                    current = previous[current]
                
                if path:
                    next_hop = path[-1]  # First hop from source
                    hop_count = len(path)
                    
                    self.routing_table[destination] = RouteEntry(
                        destination=destination,
                        next_hop=next_hop,
                        hop_count=hop_count,
                        metric=distances[destination],
                        sequence_number=0,
                        lifetime=current_time + self.topology_hold_time,
                        route_type="DYNAMIC",
                        timestamp=current_time
                    )
    
    def get_next_hop(self, destination: str, current_time: float = None) -> Optional[str]:
        """Get next hop for destination with route validation"""
        if current_time is None:
            current_time = time.time()
        
        if destination in self.routing_table:
            route = self.routing_table[destination]
            if current_time < route.lifetime:
                return route.next_hop
            else:
                del self.routing_table[destination]
        
        return None
    
    def cleanup_expired_entries(self, current_time: float = None):
        """Enhanced cleanup with proper timing validation"""
        if current_time is None:
            current_time = time.time()
        
        # Cleanup expired neighbors
        expired_neighbors = []
        for neighbor_id, neighbor_data in self.neighbor_set.items():
            if current_time - neighbor_data['last_seen'] > self.neighbor_hold_time:
                expired_neighbors.append(neighbor_id)
        
        for neighbor_id in expired_neighbors:
            del self.neighbor_set[neighbor_id]
            if neighbor_id in self.two_hop_neighbors:
                del self.two_hop_neighbors[neighbor_id]
            if neighbor_id in self.link_quality:
                del self.link_quality[neighbor_id]
            self.mpr_set.discard(neighbor_id)
            self.mpr_selector_set.discard(neighbor_id)
        
        # Cleanup expired topology entries
        for originator in list(self.topology_set.keys()):
            topology_info = self.topology_set[originator]
            if '_last_update' in topology_info:
                if current_time - topology_info['_last_update'] > self.topology_hold_time:
                    del self.topology_set[originator]
                else:
                    # Cleanup individual entries
                    expired_entries = []
                    for dest, link_info in topology_info.items():
                        if not dest.startswith('_'):
                            if current_time - link_info['last_update'] > self.topology_hold_time:
                                expired_entries.append(dest)
                    
                    for dest in expired_entries:
                        del topology_info[dest]
        
        # Cleanup expired routes
        expired_routes = []
        for dest, route in self.routing_table.items():
            if current_time >= route.lifetime:
                expired_routes.append(dest)
        
        for dest in expired_routes:
            del self.routing_table[dest]
        
        # Cleanup old duplicate entries
        old_duplicates = []
        for dup_key, timestamp in self.duplicate_set.items():
            if current_time - timestamp > self.duplicate_hold_time:
                old_duplicates.append(dup_key)
        
        for dup_key in old_duplicates:
            del self.duplicate_set[dup_key]
        
        # Recalculate MPRs and routing table if neighbors changed
        if expired_neighbors:
            self._select_mprs(current_time)
            self._calculate_routing_table(current_time)
    
    def get_routing_statistics(self) -> Dict:
        """Get comprehensive routing statistics"""
        return {
            **self.stats,
            'routing_table_size': len(self.routing_table),
            'neighbor_count': len(self.neighbor_set),
            'two_hop_neighbor_count': sum(len(neighbors) for neighbors in self.two_hop_neighbors.values()),
            'mpr_count': len(self.mpr_set),
            'mpr_selector_count': len(self.mpr_selector_set),
            'topology_entries': sum(len(info) - 1 for info in self.topology_set.values() if '_last_update' in info)
        }
    
    def cleanup_expired_routes(self, current_time: float = None):
        """Alias for consistency with other protocols"""
        self.cleanup_expired_entries(current_time)

# NEW: Geographic Routing Implementation
class GeographicRoutingProtocol:
    """Enhanced Geographic routing with greedy forwarding and perimeter routing"""
    
    def __init__(self, node_id: str, config: SimulationConfig):
        self.node_id = node_id
        self.config = config
        self.position_table: Dict[str, Dict] = {}  # node_id -> {x, y, timestamp, speed, heading}
        self.routing_table: Dict[str, RouteEntry] = {}
        
        # Geographic routing parameters
        self.position_timeout = 10.0  # seconds
        self.beacon_interval = 1.0  # Position beacon interval
        self.last_beacon_time = 0.0
        self.communication_range = 200.0  # meters (updated dynamically)
        
        # Perimeter routing state
        self.perimeter_mode: Dict[str, Dict] = {}  # packet_id -> perimeter state
        self.void_recovery_enabled = True
        
        # Statistics
        self.stats = {
            'position_beacons_sent': 0,
            'position_beacons_received': 0,
            'greedy_forwards': 0,
            'perimeter_forwards': 0,
            'forwarding_failures': 0,
            'void_recoveries': 0
        }
    
    def update_position(self, node_id: str, x: float, y: float, speed: float = 0.0, heading: float = 0.0):
        """Enhanced position update with mobility information"""
        current_time = time.time()
        
        self.position_table[node_id] = {
            'x': x,
            'y': y,
            'timestamp': current_time,
            'speed': speed,
            'heading': heading,
            'reliable': True
        }
        
        # Update communication range based on current conditions
        self._update_communication_range()
    
    def generate_position_beacon(self, current_pos: Tuple[float, float], 
                               current_time: float, speed: float = 0.0, 
                               heading: float = 0.0) -> Optional[NetworkPacket]:
        """Generate position beacon for neighbor discovery"""
        if current_time - self.last_beacon_time < self.beacon_interval:
            return None
        
        self.last_beacon_time = current_time
        
        # Update our own position
        self.update_position(self.node_id, current_pos[0], current_pos[1], speed, heading)
        
        beacon_packet = NetworkPacket(
            packet_id=f"POS_BEACON_{self.node_id}_{current_time}",
            packet_type=PacketType.HELLO,  # Use HELLO type for position beacons
            source_id=self.node_id,
            destination_id="BROADCAST",
            source_ip=f"192.168.1.{hash(self.node_id) % 254 + 1}",
            destination_ip="255.255.255.255",
            payload_size=48,  # Position + speed + heading + timestamp
            qos_class=QoSClass.SERVICE,
            application_type="ROUTING",
            ttl=1,  # Position beacons are not forwarded
            creation_time=current_time
        )
        
        # Store position information in packet
        beacon_packet.position_info = {
            'x': current_pos[0],
            'y': current_pos[1],
            'speed': speed,
            'heading': heading,
            'timestamp': current_time
        }
        
        self.stats['position_beacons_sent'] += 1
        return beacon_packet
    
    def process_position_beacon(self, packet: NetworkPacket, sender: str, current_time: float):
            """Process received position beacon with null checks"""
            # Add null checks
            if not packet or not sender:
                return
            
            self.stats['position_beacons_received'] += 1
            
            position_info = getattr(packet, 'position_info', {})
            if isinstance(position_info, dict) and position_info:
                self.update_position(
                    sender,
                    position_info.get('x', 0.0),
                    position_info.get('y', 0.0),
                    position_info.get('speed', 0.0),
                    position_info.get('heading', 0.0)
                )
    
    def get_next_hop_geographic(self, destination: str, current_pos: Tuple[float, float], 
                              packet_id: str = None, current_time: float = None) -> Optional[str]:
        """Enhanced geographic forwarding with perimeter routing fallback"""
        if current_time is None:
            current_time = time.time()
        
        if destination not in self.position_table:
            return None
        
        dest_info = self.position_table[destination]
        if current_time - dest_info['timestamp'] > self.position_timeout:
            # Destination position too old
            del self.position_table[destination]
            return None
        
        dest_pos = (dest_info['x'], dest_info['y'])
        current_distance = self._calculate_distance(current_pos, dest_pos)
        
        # Check if we're in perimeter mode for this packet
        in_perimeter_mode = packet_id and packet_id in self.perimeter_mode
        
        if not in_perimeter_mode:
            # Try greedy forwarding first
            next_hop = self._greedy_forwarding(destination, current_pos, dest_pos, current_time)
            if next_hop:
                self.stats['greedy_forwards'] += 1
                return next_hop
            
            # Greedy forwarding failed, try perimeter routing if enabled
            if self.void_recovery_enabled and packet_id:
                return self._start_perimeter_routing(packet_id, destination, current_pos, 
                                                   dest_pos, current_time)
        else:
            # Continue perimeter routing
            return self._continue_perimeter_routing(packet_id, destination, current_pos, 
                                                  dest_pos, current_time)
        
        self.stats['forwarding_failures'] += 1
        return None
    
    def _greedy_forwarding(self, destination: str, current_pos: Tuple[float, float], 
                          dest_pos: Tuple[float, float], current_time: float) -> Optional[str]:
        """Greedy forwarding: select neighbor closest to destination"""
        current_distance = self._calculate_distance(current_pos, dest_pos)
        best_neighbor = None
        best_distance = current_distance
        
        # Consider all neighbors with valid positions
        for neighbor_id, neighbor_info in self.position_table.items():
            if (neighbor_id == self.node_id or neighbor_id == destination or
                current_time - neighbor_info['timestamp'] > self.position_timeout):
                continue
            
            neighbor_pos = (neighbor_info['x'], neighbor_info['y'])
            
            # Check if neighbor is within communication range
            neighbor_distance_to_us = self._calculate_distance(current_pos, neighbor_pos)
            if neighbor_distance_to_us > self.communication_range:
                continue
            
            # Calculate distance from neighbor to destination
            neighbor_to_dest_distance = self._calculate_distance(neighbor_pos, dest_pos)
            
            # Select if closer to destination than current node
            if neighbor_to_dest_distance < best_distance:
                # Additional checks for mobility prediction
                if self._is_neighbor_suitable(neighbor_info, dest_pos, current_time):
                    best_neighbor = neighbor_id
                    best_distance = neighbor_to_dest_distance
        
        return best_neighbor
    
    def _is_neighbor_suitable(self, neighbor_info: Dict, dest_pos: Tuple[float, float], 
                            current_time: float) -> bool:
        """Check if neighbor is suitable considering mobility"""
        # Basic suitability check
        if not neighbor_info.get('reliable', True):
            return False
        
        # Predict neighbor position based on mobility
        if neighbor_info.get('speed', 0) > 0 and neighbor_info.get('heading') is not None:
            # Simple linear prediction
            time_diff = 1.0  # Predict 1 second ahead
            speed = neighbor_info['speed']
            heading_rad = math.radians(neighbor_info['heading'])
            
            predicted_x = neighbor_info['x'] + speed * time_diff * math.cos(heading_rad)
            predicted_y = neighbor_info['y'] + speed * time_diff * math.sin(heading_rad)
            
            # Check if predicted position is still useful
            current_dist = self._calculate_distance((neighbor_info['x'], neighbor_info['y']), dest_pos)
            predicted_dist = self._calculate_distance((predicted_x, predicted_y), dest_pos)
            
            # Prefer neighbors moving toward destination
            return predicted_dist <= current_dist * 1.1  # Allow 10% tolerance
        
        return True
    
    def _start_perimeter_routing(self, packet_id: str, destination: str, 
                               current_pos: Tuple[float, float], dest_pos: Tuple[float, float],
                               current_time: float) -> Optional[str]:
        """Start perimeter routing around void area"""
        self.stats['void_recoveries'] += 1
        
        # Find the face (planar graph edge) to follow
        next_hop = self._find_perimeter_next_hop(current_pos, dest_pos, current_time)
        
        if next_hop:
            # Initialize perimeter state
            self.perimeter_mode[packet_id] = {
                'start_pos': current_pos,
                'start_time': current_time,
                'face_edges': [],
                'direction': 'right_hand',  # Right-hand rule
                'last_hop': self.node_id
            }
            
            self.stats['perimeter_forwards'] += 1
            return next_hop
        
        return None
    
    def _continue_perimeter_routing(self, packet_id: str, destination: str,
                                  current_pos: Tuple[float, float], dest_pos: Tuple[float, float],
                                  current_time: float) -> Optional[str]:
        """Continue perimeter routing"""
        if packet_id not in self.perimeter_mode:
            return None
        
        perimeter_state = self.perimeter_mode[packet_id]
        
        # Check if we can switch back to greedy forwarding
        greedy_next = self._greedy_forwarding(destination, current_pos, dest_pos, current_time)
        if greedy_next:
            # Clean up perimeter state and switch to greedy
            del self.perimeter_mode[packet_id]
            self.stats['greedy_forwards'] += 1
            return greedy_next
        
        # Check for perimeter routing termination conditions
        start_pos = perimeter_state['start_pos']
        if (self._calculate_distance(current_pos, start_pos) < 10.0 and  # Back near start
            current_time - perimeter_state['start_time'] > 5.0):  # Enough time passed
            # Failed to find route, terminate
            del self.perimeter_mode[packet_id]
            return None
        
        # Continue perimeter routing
        next_hop = self._find_perimeter_next_hop(current_pos, dest_pos, current_time,
                                               perimeter_state['last_hop'])
        
        if next_hop:
            perimeter_state['last_hop'] = self.node_id
            self.stats['perimeter_forwards'] += 1
            return next_hop
        
        # Perimeter routing failed
        del self.perimeter_mode[packet_id]
        return None
    
    def _find_perimeter_next_hop(self, current_pos: Tuple[float, float], 
                                dest_pos: Tuple[float, float], current_time: float,
                                avoid_node: str = None) -> Optional[str]:
        """Find next hop for perimeter routing using right-hand rule"""
        # Get all valid neighbors
        neighbors = []
        for neighbor_id, neighbor_info in self.position_table.items():
            if (neighbor_id == self.node_id or neighbor_id == avoid_node or
                current_time - neighbor_info['timestamp'] > self.position_timeout):
                continue
            
            neighbor_pos = (neighbor_info['x'], neighbor_info['y'])
            distance = self._calculate_distance(current_pos, neighbor_pos)
            
            if distance <= self.communication_range:
                # Calculate angle from current position to neighbor
                angle = math.atan2(neighbor_pos[1] - current_pos[1], 
                                 neighbor_pos[0] - current_pos[0])
                neighbors.append((neighbor_id, neighbor_pos, angle, distance))
        
        if not neighbors:
            return None
        
        # Sort neighbors by angle (for right-hand rule)
        neighbors.sort(key=lambda x: x[2])  # Sort by angle
        
        # Apply right-hand rule: select the neighbor that maintains the perimeter
        # For simplicity, select the neighbor with the smallest angle change
        if neighbors:
            return neighbors[0][0]  # Return the first neighbor
        
        return None
    
    def _update_communication_range(self):
        """Update communication range based on current network conditions"""
        # Simple adaptive range based on neighbor density
        neighbor_count = len([n for n, info in self.position_table.items() 
                            if time.time() - info['timestamp'] < self.position_timeout])
        
        if neighbor_count < 3:
            self.communication_range = min(300.0, self.communication_range * 1.1)
        elif neighbor_count > 8:
            self.communication_range = max(150.0, self.communication_range * 0.9)
    
    def _calculate_distance(self, pos1: Tuple[float, float], pos2: Tuple[float, float]) -> float:
        """Calculate Euclidean distance between two positions"""
        return math.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
    
    def get_next_hop(self, destination: str, current_time: float = None) -> Optional[str]:
        """Get next hop for destination (compatibility method)"""
        if current_time is None:
            current_time = time.time()
        
        # This method is for compatibility with other routing protocols
        # Geographic routing needs position information which isn't available here
        # Return cached route if available
        if destination in self.routing_table:
            route = self.routing_table[destination]
            if current_time < route.lifetime:
                return route.next_hop
            else:
                del self.routing_table[destination]
        
        return None
    
    def cleanup_expired_positions(self, current_time: float = None):
        """Remove expired position information and perimeter states"""
        if current_time is None:
            current_time = time.time()
        
        # Remove expired positions
        expired_nodes = []
        for node_id, position_info in self.position_table.items():
            if current_time - position_info['timestamp'] > self.position_timeout:
                expired_nodes.append(node_id)
        
        for node_id in expired_nodes:
            del self.position_table[node_id]
        
        # Remove expired routes
        expired_routes = []
        for dest, route in self.routing_table.items():
            if current_time >= route.lifetime:
                expired_routes.append(dest)
        
        for dest in expired_routes:
            del self.routing_table[dest]
        
        # Cleanup old perimeter states
        expired_perimeter = []
        for packet_id, state in self.perimeter_mode.items():
            if current_time - state['start_time'] > 30.0:  # 30 second timeout
                expired_perimeter.append(packet_id)
        
        for packet_id in expired_perimeter:
            del self.perimeter_mode[packet_id]
    
    def get_routing_statistics(self) -> Dict:
        """Get comprehensive routing statistics"""
        current_time = time.time()
        valid_positions = sum(1 for info in self.position_table.values() 
                            if current_time - info['timestamp'] < self.position_timeout)
        
        return {
            **self.stats,
            'valid_positions': valid_positions,
            'total_positions': len(self.position_table),
            'routing_table_size': len(self.routing_table),
            'active_perimeter_routes': len(self.perimeter_mode),
            'communication_range': self.communication_range
        }
    def cleanup_expired_entries(self, current_time: float = None):
        """Unified cleanup method for consistency"""
        self.cleanup_expired_positions(current_time)
    
    def cleanup_expired_routes(self, current_time: float = None):
        """Alias for consistency with other protocols"""
        self.cleanup_expired_positions(current_time)
        
# SDN Controller Implementation
class VANETSDNController:
    """Enhanced Centralized SDN Controller for VANET with proper differentiation"""
    
    def __init__(self, config: SimulationConfig):
        self.config = config
        self.network_topology = nx.Graph()
        self.node_positions: Dict[str, Tuple[float, float]] = {}
        self.flow_tables: Dict[str, Dict[str, FlowEntry]] = {}
        self.global_flows: Dict[str, FlowEntry] = {}
        self.traffic_matrix: Dict[Tuple[str, str], float] = {}
        self.qos_requirements: Dict[str, Dict[str, float]] = {}
        self.last_topology_update = 0.0
        self.packet_counters: Dict[str, int] = defaultdict(int)
        
        # NEW: SDN-specific performance tracking
        self.controller_processing_delays: List[float] = []
        self.flow_installation_times: List[float] = []
        self.path_computation_cache: Dict[Tuple[str, str], List[str]] = {}
        self.load_balancing_enabled = True
        self.centralized_optimization_factor = 1.15  # 15% improvement from centralized view
        
    def update_node_position(self, node_id: str, x: float, y: float):
        """Update node position in network topology"""
        self.node_positions[node_id] = (x, y)
        
        if node_id not in self.network_topology:
            self.network_topology.add_node(node_id, x=x, y=y)
        else:
            self.network_topology.nodes[node_id]['x'] = x
            self.network_topology.nodes[node_id]['y'] = y
    
    def update_network_topology(self, node_id: str, neighbors: List[Dict]):
        """Update network topology based on neighbor information"""
        current_time = time.time()
        
        # Update node in topology
        if node_id not in self.network_topology:
            pos = self.node_positions.get(node_id, (0, 0))
            self.network_topology.add_node(node_id, x=pos[0], y=pos[1], last_seen=current_time)
        else:
            self.network_topology.nodes[node_id]['last_seen'] = current_time
        
        # Update edges based on neighbors
        current_neighbors = set()
        for neighbor in neighbors:
            neighbor_id = neighbor['id']
            current_neighbors.add(neighbor_id)
            
            # Add neighbor node if not exists
            if neighbor_id not in self.network_topology:
                self.network_topology.add_node(neighbor_id, last_seen=current_time)
            
            # Add or update edge with SDN-enhanced metrics
            distance = neighbor['distance']
            link_quality = 1.0 / (1.0 + distance / 100.0)
            
            # SDN uses more sophisticated link metrics
            sdn_weight = self._calculate_sdn_link_weight(distance, link_quality, neighbor_id)
            
            if self.network_topology.has_edge(node_id, neighbor_id):
                # Update edge attributes
                self.network_topology[node_id][neighbor_id]['distance'] = distance
                self.network_topology[node_id][neighbor_id]['quality'] = link_quality
                self.network_topology[node_id][neighbor_id]['sdn_weight'] = sdn_weight
                self.network_topology[node_id][neighbor_id]['last_seen'] = current_time
            else:
                # Add new edge
                self.network_topology.add_edge(node_id, neighbor_id, 
                                             distance=distance, 
                                             quality=link_quality,
                                             sdn_weight=sdn_weight,
                                             last_seen=current_time)
        
        # Remove edges to nodes no longer neighbors
        edges_to_remove = []
        for neighbor_id in list(self.network_topology.neighbors(node_id)):
            if neighbor_id not in current_neighbors:
                edges_to_remove.append((node_id, neighbor_id))
        
        for edge in edges_to_remove:
            self.network_topology.remove_edge(*edge)
        
        self.last_topology_update = current_time
    
    def _calculate_sdn_link_weight(self, distance: float, quality: float, neighbor_id: str) -> float:
        """Calculate SDN-specific link weight considering global network state"""
        base_weight = distance / (quality + 0.1)
        
        # SDN considers global traffic patterns
        traffic_load = self.traffic_matrix.get((neighbor_id, neighbor_id), 0.0)
        load_factor = 1.0 + (traffic_load / 1e6)  # Normalize to Mbps
        
        # SDN can optimize based on centralized view
        centralized_factor = 0.9  # 10% improvement from global optimization
        
        return base_weight * load_factor * centralized_factor
    
    def compute_optimal_path(self, source: str, destination: str, qos_requirements: Dict[str, float] = None) -> Optional[List[str]]:
        """Compute optimal path with SDN-specific optimizations"""
        if source not in self.network_topology or destination not in self.network_topology:
            return None
        
        # Check cache first (SDN advantage)
        cache_key = (source, destination)
        if cache_key in self.path_computation_cache:
            cached_path = self.path_computation_cache[cache_key]
            # Validate cached path is still valid
            if self._validate_path(cached_path):
                return cached_path
            else:
                del self.path_computation_cache[cache_key]
        
        start_time = time.time()
        
        try:
            if self.load_balancing_enabled and len(list(self.network_topology.nodes())) > 10:
                # SDN uses advanced path computation with load balancing
                primary_path = nx.shortest_path(self.network_topology, source, destination, 
                                              weight='sdn_weight')
                
                # Try to find alternative path for load balancing
                temp_graph = self.network_topology.copy()
                if len(primary_path) > 2:
                    # Remove middle node to force alternative path
                    middle_node = primary_path[len(primary_path)//2]
                    temp_graph.remove_node(middle_node)
                    
                    try:
                        alternative_path = nx.shortest_path(temp_graph, source, destination, 
                                                          weight='sdn_weight')
                        
                        # Choose path based on current load
                        primary_load = self._calculate_path_load(primary_path)
                        alt_load = self._calculate_path_load(alternative_path)
                        
                        if alt_load < primary_load * 0.8:  # 20% load difference threshold
                            path = alternative_path
                        else:
                            path = primary_path
                    except nx.NetworkXNoPath:
                        path = primary_path
                else:
                    path = primary_path
            else:
                # Standard shortest path for smaller networks
                if qos_requirements and 'min_bandwidth' in qos_requirements:
                    path = nx.shortest_path(self.network_topology, source, destination, 
                                          weight='sdn_weight')
                else:
                    path = nx.shortest_path(self.network_topology, source, destination, 
                                          weight='distance')
            
            # Cache the computed path
            self.path_computation_cache[cache_key] = path
            
            # Track SDN-specific metrics
            computation_time = time.time() - start_time
            self.controller_processing_delays.append(computation_time)
            
            return path
            
        except nx.NetworkXNoPath:
            return None
    
    def _validate_path(self, path: List[str]) -> bool:
        """Validate that a cached path is still valid"""
        for i in range(len(path) - 1):
            if not self.network_topology.has_edge(path[i], path[i+1]):
                return False
        return True
    
    def _calculate_path_load(self, path: List[str]) -> float:
        """Calculate current load on a path"""
        total_load = 0.0
        for i in range(len(path) - 1):
            node_pair = (path[i], path[i+1])
            load = self.traffic_matrix.get(node_pair, 0.0)
            total_load += load
        return total_load
    
    def install_flow_rule(self, node_id: str, flow_entry: FlowEntry) -> bool:
        """Install flow rule on specific node with SDN overhead"""
        start_time = time.time()
        
        if node_id not in self.flow_tables:
            self.flow_tables[node_id] = {}
        
        # Check flow table capacity
        if len(self.flow_tables[node_id]) >= self.config.flow_table_size:
            # Remove oldest flow entry
            oldest_flow = min(self.flow_tables[node_id].values(), 
                            key=lambda x: x.last_used)
            del self.flow_tables[node_id][oldest_flow.flow_id]
        
        self.flow_tables[node_id][flow_entry.flow_id] = flow_entry
        
        # Track installation time
        installation_time = time.time() - start_time
        self.flow_installation_times.append(installation_time)
        
        return True
    
    def handle_packet_in(self, packet: NetworkPacket, from_node: str) -> List[Tuple[str, FlowEntry]]:
        """Handle packet-in message with SDN-specific processing overhead"""
        start_time = time.time()
        
        source = packet.source_id
        destination = packet.destination_id
        
        # SDN-specific delay for packet-in processing
        processing_delay = random.uniform(0.001, 0.005)  # 1-5ms controller processing
        time.sleep(processing_delay)  # Simulate processing delay
        
        # Compute optimal path with SDN enhancements
        qos_req = None
        if packet.qos_class in [QoSClass.EMERGENCY, QoSClass.SAFETY]:
            qos_req = {'min_bandwidth': 2.0, 'max_delay': 0.01}
        elif packet.qos_class == QoSClass.SERVICE:
            qos_req = {'min_bandwidth': 1.0, 'max_delay': 0.05}
        
        path = self.compute_optimal_path(source, destination, qos_req)
        
        if not path:
            processing_time = time.time() - start_time
            self.controller_processing_delays.append(processing_time)
            return []
        
        # Create flow rules for the path with SDN-specific optimizations
        flow_rules = self._create_optimized_flow_rules(path, packet)
        
        # Install flow rules on all nodes in path
        for node_id, flow_entry in flow_rules:
            self.install_flow_rule(node_id, flow_entry)
        
        processing_time = time.time() - start_time
        self.controller_processing_delays.append(processing_time)
        
        return flow_rules
    
    def _create_optimized_flow_rules(self, path: List[str], packet: NetworkPacket) -> List[Tuple[str, FlowEntry]]:
        """Create optimized flow rules with SDN-specific enhancements"""
        flow_rules = []
        
        if len(path) < 2:
            return flow_rules
        
        for i in range(len(path) - 1):
            current_node = path[i]
            next_node = path[i + 1]
            
            # Create match fields with more specific SDN criteria
            match_fields = {
                'flow_id': packet.flow_id,
                'destination': path[-1],
                'qos_class': packet.qos_class.name,
                'application_type': packet.application_type
            }
            
            # Create actions with SDN-specific optimizations
            actions = [
                {'type': 'forward', 'next_hop': next_node},
                {'type': 'update_ttl', 'decrement': 1}
            ]
            
            # Add SDN-specific QoS actions
            if packet.qos_class in [QoSClass.EMERGENCY, QoSClass.SAFETY]:
                actions.append({'type': 'set_priority', 'priority': 'high'})
                actions.append({'type': 'reserve_bandwidth', 'bandwidth': '2Mbps'})
                timeout = self.config.flow_rule_timeout * 2  # Longer timeout for critical traffic
            elif packet.qos_class == QoSClass.SERVICE:
                actions.append({'type': 'set_priority', 'priority': 'medium'})
                timeout = self.config.flow_rule_timeout
            else:
                actions.append({'type': 'set_priority', 'priority': 'low'})
                timeout = self.config.flow_rule_timeout * 0.5  # Shorter timeout for background
            
            # Add SDN-specific traffic engineering actions
            if self.config.enable_traffic_engineering:
                actions.append({'type': 'monitor_traffic', 'interval': '1s'})
                actions.append({'type': 'adaptive_routing', 'enabled': True})
            
            # Create flow entry with SDN-specific parameters
            flow_entry = FlowEntry(
                flow_id=f"{packet.flow_id}_{current_node}",
                match_fields=match_fields,
                actions=actions,
                priority=100 - packet.qos_class.value,
                timeout=timeout,
                creation_time=time.time(),
                qos_requirements={
                    'qos_class': packet.qos_class.name,
                    'max_delay': 0.01 if packet.qos_class in [QoSClass.EMERGENCY, QoSClass.SAFETY] else 0.1,
                    'min_bandwidth': 2.0 if packet.qos_class in [QoSClass.EMERGENCY, QoSClass.SAFETY] else 1.0
                }
            )
            
            flow_rules.append((current_node, flow_entry))
        
        return flow_rules
    
    def get_average_controller_latency(self) -> float:
        """Get average SDN controller processing latency"""
        if not self.controller_processing_delays:
            return 0.0
        return sum(self.controller_processing_delays) / len(self.controller_processing_delays)
    
    def get_flow_installation_overhead(self) -> float:
        """Get average flow installation overhead"""
        if not self.flow_installation_times:
            return 0.0
        return sum(self.flow_installation_times) / len(self.flow_installation_times)
    
    def perform_traffic_engineering(self) -> Dict[str, List[Tuple[str, FlowEntry]]]:
        """Perform network-wide traffic engineering optimization with real SDN benefits"""
        if not self.config.enable_traffic_engineering:
            return {}
    
        optimization_results = {}
        current_time = time.time()
        
        # Only perform traffic engineering if we have sufficient network data
        if (len(self.network_topology.nodes()) < 5 or 
            current_time - self.last_topology_update > 10.0):
            return {}
        
        # Analyze current traffic patterns
        self._analyze_traffic_patterns()
        
        # Identify congested links
        congested_links = self._identify_congested_links()
        
        # SDN-specific traffic engineering actions
        if congested_links:
            print(f"[SDN TRAFFIC ENGINEERING] Found {len(congested_links)} congested links")
            
            # Reroute flows around congested links
            for link in congested_links:
                alternative_flows = self._find_alternative_routes(link)
                optimization_results.update(alternative_flows)
            
            # Update traffic matrix based on new routes
            self._update_traffic_matrix_after_optimization()
        
        # Proactive load balancing for SDN
        if len(self.network_topology.nodes()) > 10:
            load_balancing_flows = self._perform_proactive_load_balancing()
            optimization_results.update(load_balancing_flows)
        
        # QoS-based flow optimization
        if self.config.enable_qos_management:
            qos_flows = self._optimize_qos_flows()
            optimization_results.update(qos_flows)
        
        return optimization_results
    
    def _analyze_traffic_patterns(self):
        """Analyze current traffic patterns in the network"""
        # Update traffic matrix based on active flows
        current_time = time.time()
        
        for node_id, flows in self.flow_tables.items():
            for flow_id, flow_entry in flows.items():
                if (flow_entry.state == FlowState.ACTIVE and 
                    current_time - flow_entry.last_used < 5.0):  # Recent activity
                    
                    # Extract source and destination from flow
                    match_fields = flow_entry.match_fields
                    if 'destination' in match_fields:
                        dest = match_fields['destination']
                        traffic_key = (node_id, dest)
                        
                        # Calculate current load based on packet count and time
                        time_active = max(1.0, current_time - flow_entry.creation_time)
                        current_load = flow_entry.byte_count / time_active  # Bytes per second
                        
                        # Update traffic matrix
                        self.traffic_matrix[traffic_key] = current_load
    
    def _identify_congested_links(self) -> List[Tuple[str, str]]:
        """Identify congested links in the network based on SDN flow statistics"""
        congested_links = []
        
        # Analyze edge utilization based on flow table data
        for edge in self.network_topology.edges():
            node1, node2 = edge
            
            # Calculate link utilization based on flows passing through this edge
            total_traffic = 0.0
            
            # Check flows in both directions
            for direction in [(node1, node2), (node2, node1)]:
                src, dst = direction
                
                # Look for flows that traverse this link
                if src in self.flow_tables:
                    for flow_id, flow_entry in self.flow_tables[src].items():
                        if flow_entry.state == FlowState.ACTIVE:
                            # Check if flow's next hop is dst
                            for action in flow_entry.actions:
                                if (action.get('type') == 'forward' and 
                                    action.get('next_hop') == dst):
                                    
                                    # Add this flow's traffic to link utilization
                                    time_active = max(1.0, time.time() - flow_entry.creation_time)
                                    flow_rate = flow_entry.byte_count / time_active
                                    total_traffic += flow_rate
            
            # Assume link capacity of 10 Mbps per direction
            link_capacity = 10e6  # 10 Mbps in bytes per second
            utilization = total_traffic / link_capacity
            
            # Mark as congested if utilization > 70% (more aggressive than the original 80%)
            if utilization > 0.7:
                congested_links.append(edge)
                
                # Update edge weight to reflect congestion
                if self.network_topology.has_edge(*edge):
                    current_weight = self.network_topology[edge[0]][edge[1]].get('sdn_weight', 1.0)
                    congestion_penalty = 1.0 + utilization  # Increase weight based on congestion
                    self.network_topology[edge[0]][edge[1]]['sdn_weight'] = current_weight * congestion_penalty
        
        return congested_links
    
    def _find_alternative_routes(self, congested_link: Tuple[str, str]) -> Dict[str, List[Tuple[str, FlowEntry]]]:
        """Find alternative routes around congested link using SDN global view"""
        alternative_flows = {}
        
        # Temporarily increase weight of congested link (rather than removing it)
        if self.network_topology.has_edge(*congested_link):
            original_weight = self.network_topology[congested_link[0]][congested_link[1]].get('sdn_weight', 1.0)
            # Increase weight significantly to discourage usage
            self.network_topology[congested_link[0]][congested_link[1]]['sdn_weight'] = original_weight * 10.0
            
            # Find flows that currently use this link and reroute them
            affected_flows = self._find_flows_using_link(congested_link)
            
            for flow_id, (source, destination, qos_class) in affected_flows.items():
                # Compute alternative path with updated weights
                alt_path = self.compute_optimal_path(source, destination)
                
                if alt_path and len(alt_path) > 1:
                    # Create new flow rules for alternative path
                    # Create a temporary packet for flow rule generation
                    temp_packet = NetworkPacket(
                        packet_id=f"reroute_{flow_id}",
                        packet_type=PacketType.DATA,
                        source_id=source,
                        destination_id=destination,
                        source_ip=f"192.168.1.{hash(source) % 254 + 1}",
                        destination_ip=f"192.168.1.{hash(destination) % 254 + 1}",
                        payload_size=512,
                        qos_class=qos_class,
                        application_type="REROUTED",
                        flow_id=flow_id
                    )
                    
                    new_flow_rules = self._create_optimized_flow_rules(alt_path, temp_packet)
                    alternative_flows[flow_id] = new_flow_rules
            
            # Restore original weight
            self.network_topology[congested_link[0]][congested_link[1]]['sdn_weight'] = original_weight
        
        return alternative_flows
    
    def _find_flows_using_link(self, link: Tuple[str, str]) -> Dict[str, Tuple[str, str, QoSClass]]:
        """Find flows that use a specific link"""
        flows_using_link = {}
        
        # Check flow tables of both nodes in the link
        for node_id in [link[0], link[1]]:
            if node_id in self.flow_tables:
                for flow_id, flow_entry in self.flow_tables[node_id].items():
                    if flow_entry.state == FlowState.ACTIVE:
                        # Check if flow actions involve the congested link
                        for action in flow_entry.actions:
                            if (action.get('type') == 'forward' and 
                                ((node_id == link[0] and action.get('next_hop') == link[1]) or
                                 (node_id == link[1] and action.get('next_hop') == link[0]))):
                                
                                # Extract flow information
                                destination = flow_entry.match_fields.get('destination', '')
                                qos_class_name = flow_entry.match_fields.get('qos_class', 'SERVICE')
                                
                                # Convert QoS class name back to enum
                                try:
                                    qos_class = QoSClass[qos_class_name]
                                except KeyError:
                                    qos_class = QoSClass.SERVICE
                                
                                flows_using_link[flow_id] = (node_id, destination, qos_class)
                                break  # Found the flow, no need to check more actions
        
        return flows_using_link
    
    def _perform_proactive_load_balancing(self) -> Dict[str, List[Tuple[str, FlowEntry]]]:
        """Perform proactive load balancing across multiple paths"""
        load_balancing_flows = {}
        
        # Find node pairs with multiple possible paths
        nodes = list(self.network_topology.nodes())
        
        for i, source in enumerate(nodes):
            for j, destination in enumerate(nodes[i+1:], i+1):
                if source != destination:
                    try:
                        # Find primary path
                        primary_path = nx.shortest_path(self.network_topology, source, destination, weight='sdn_weight')
                        
                        # Find alternative path by temporarily removing a middle node
                        if len(primary_path) > 2:
                            temp_graph = self.network_topology.copy()
                            middle_node = primary_path[len(primary_path)//2]
                            temp_graph.remove_node(middle_node)
                            
                            try:
                                alt_path = nx.shortest_path(temp_graph, source, destination, weight='sdn_weight')
                                
                                # If alternative path exists and is reasonable, create load balancing rules
                                if len(alt_path) <= len(primary_path) + 2:  # Not too much longer
                                    # Create flow rule for load balancing
                                    temp_packet = NetworkPacket(
                                        packet_id=f"loadbalance_{source}_{destination}",
                                        packet_type=PacketType.DATA,
                                        source_id=source,
                                        destination_id=destination,
                                        source_ip=f"192.168.1.{hash(source) % 254 + 1}",
                                        destination_ip=f"192.168.1.{hash(destination) % 254 + 1}",
                                        payload_size=512,
                                        qos_class=QoSClass.SERVICE,
                                        application_type="LOAD_BALANCED",
                                        flow_id=f"lb_{source}_{destination}"
                                    )
                                    
                                    lb_flow_rules = self._create_optimized_flow_rules(alt_path, temp_packet)
                                    load_balancing_flows[f"lb_{source}_{destination}"] = lb_flow_rules
                                    
                            except nx.NetworkXNoPath:
                                continue
                                
                    except nx.NetworkXNoPath:
                        continue
        
        return load_balancing_flows
    
    def _optimize_qos_flows(self) -> Dict[str, List[Tuple[str, FlowEntry]]]:
        """Optimize flows based on QoS requirements"""
        qos_optimized_flows = {}
        
        # Find high-priority flows that might benefit from better paths
        for node_id, flows in self.flow_tables.items():
            for flow_id, flow_entry in flows.items():
                if flow_entry.state == FlowState.ACTIVE:
                    qos_class_name = flow_entry.match_fields.get('qos_class', 'SERVICE')
                    
                    # Optimize paths for high-priority traffic
                    if qos_class_name in ['EMERGENCY', 'SAFETY']:
                        destination = flow_entry.match_fields.get('destination')
                        
                        if destination:
                            # Find the best possible path for high-priority traffic
                            best_path = self.compute_optimal_path(
                                node_id, 
                                destination, 
                                {'min_bandwidth': 2.0, 'max_delay': 0.01}
                            )
                            
                            if best_path:
                                # Create optimized flow rules
                                temp_packet = NetworkPacket(
                                    packet_id=f"qos_opt_{flow_id}",
                                    packet_type=PacketType.DATA,
                                    source_id=node_id,
                                    destination_id=destination,
                                    source_ip=f"192.168.1.{hash(node_id) % 254 + 1}",
                                    destination_ip=f"192.168.1.{hash(destination) % 254 + 1}",
                                    payload_size=512,
                                    qos_class=QoSClass.EMERGENCY if qos_class_name == 'EMERGENCY' else QoSClass.SAFETY,
                                    application_type="QOS_OPTIMIZED",
                                    flow_id=f"qos_{flow_id}"
                                )
                                
                                qos_flow_rules = self._create_optimized_flow_rules(best_path, temp_packet)
                                qos_optimized_flows[f"qos_{flow_id}"] = qos_flow_rules
        
        return qos_optimized_flows
    
    def _update_traffic_matrix_after_optimization(self):
        """Update traffic matrix after traffic engineering optimization"""
        # Reset traffic matrix to reflect new routing decisions
        old_matrix = self.traffic_matrix.copy()
        self.traffic_matrix.clear()
        
        # Recalculate based on current active flows
        current_time = time.time()
        
        for node_id, flows in self.flow_tables.items():
            for flow_id, flow_entry in flows.items():
                if (flow_entry.state == FlowState.ACTIVE and 
                    current_time - flow_entry.last_used < 2.0):  # Very recent activity
                    
                    match_fields = flow_entry.match_fields
                    if 'destination' in match_fields:
                        dest = match_fields['destination']
                        traffic_key = (node_id, dest)
                        
                        # Use previous load as estimate
                        old_load = old_matrix.get(traffic_key, 0.0)
                        # Apply a small random variation to simulate traffic engineering effect
                        new_load = old_load * random.uniform(0.85, 1.0)  # Slight improvement from optimization
                        self.traffic_matrix[traffic_key] = new_load
                        
    def cleanup_expired_flows(self):
        """Remove expired flow entries from all node flow tables"""
        current_time = time.time()
        total_expired = 0
        
        # Clean up flow tables for all nodes
        for node_id in list(self.flow_tables.keys()):
            if node_id not in self.flow_tables:
                continue
                
            expired_flows = []
            node_flows = self.flow_tables[node_id]
            
            for flow_id, flow_entry in node_flows.items():
                # Check if flow has expired
                flow_age = current_time - flow_entry.creation_time
                is_expired = False
                
                # Check timeout expiration
                if flow_age > flow_entry.timeout:
                    is_expired = True
                    flow_entry.state = FlowState.EXPIRED
                
                # Check if flow is inactive for too long
                elif hasattr(flow_entry, 'last_used') and flow_entry.last_used > 0:
                    inactive_time = current_time - flow_entry.last_used
                    if inactive_time > flow_entry.timeout * 0.5:  # 50% of timeout for inactivity
                        is_expired = True
                        flow_entry.state = FlowState.EXPIRED
                
                # Check if flow state is already marked for deletion
                elif flow_entry.state in [FlowState.EXPIRED, FlowState.DELETED]:
                    is_expired = True
                
                if is_expired:
                    expired_flows.append(flow_id)
            
            # Remove expired flows
            for flow_id in expired_flows:
                del self.flow_tables[node_id][flow_id]
                total_expired += 1
            
            # Remove empty flow tables
            if not self.flow_tables[node_id]:
                del self.flow_tables[node_id]
        
        # Clean up global flows
        expired_global_flows = []
        for flow_id, flow_entry in self.global_flows.items():
            flow_age = current_time - flow_entry.creation_time
            if (flow_age > flow_entry.timeout or 
                flow_entry.state in [FlowState.EXPIRED, FlowState.DELETED]):
                expired_global_flows.append(flow_id)
        
        for flow_id in expired_global_flows:
            del self.global_flows[flow_id]
            total_expired += 1
        
        # Clean up old traffic matrix entries
        expired_traffic_entries = []
        for traffic_key, load in self.traffic_matrix.items():
            # Remove very old or zero-load entries
            if load <= 0.001:  # Very low traffic
                expired_traffic_entries.append(traffic_key)
        
        for traffic_key in expired_traffic_entries:
            del self.traffic_matrix[traffic_key]
        
        # Clean up old path computation cache
        cache_cleanup_threshold = 30.0  # 30 seconds
        expired_cache_entries = []
        
        for cache_key, path in self.path_computation_cache.items():
            # Remove cached paths older than threshold or invalid paths
            if not self._validate_path(path):
                expired_cache_entries.append(cache_key)
        
        for cache_key in expired_cache_entries:
            del self.path_computation_cache[cache_key]
        
        # Limit cache size to prevent memory bloat
        if len(self.path_computation_cache) > 1000:  # Maximum 1000 cached paths
            # Remove oldest entries (simple FIFO cleanup)
            cache_items = list(self.path_computation_cache.items())
            entries_to_remove = len(cache_items) - 800  # Keep 800, remove excess
            
            for i in range(entries_to_remove):
                cache_key = cache_items[i][0]
                del self.path_computation_cache[cache_key]
        
        # Clean up old performance tracking data
        max_tracking_entries = 1000
        
        if len(self.controller_processing_delays) > max_tracking_entries:
            # Keep only recent entries
            self.controller_processing_delays = self.controller_processing_delays[-max_tracking_entries:]
        
        if len(self.flow_installation_times) > max_tracking_entries:
            # Keep only recent entries
            self.flow_installation_times = self.flow_installation_times[-max_tracking_entries:]
        
        # Periodic topology cleanup - remove very old nodes
        topology_cleanup_threshold = 60.0  # 60 seconds
        old_nodes = []
        
        for node_id in self.network_topology.nodes():
            node_data = self.network_topology.nodes[node_id]
            if 'last_seen' in node_data:
                if current_time - node_data['last_seen'] > topology_cleanup_threshold:
                    old_nodes.append(node_id)
        
        for node_id in old_nodes:
            self.network_topology.remove_node(node_id)
            # Also remove from position tracking
            if node_id in self.node_positions:
                del self.node_positions[node_id]
        
        # Debug output for cleanup (only occasionally)
        if total_expired > 0 and random.random() < 0.1:  # 10% chance to print
            print(f"[SDN CLEANUP] Removed {total_expired} expired flows, "
                  f"Cache size: {len(self.path_computation_cache)}, "
                  f"Active nodes: {len(self.network_topology.nodes())}, "
                  f"Traffic entries: {len(self.traffic_matrix)}")
    
    def get_flow_statistics(self) -> Dict[str, Any]:
        """Get comprehensive flow table statistics"""
        current_time = time.time()
        
        total_flows = 0
        active_flows = 0
        expired_flows = 0
        total_nodes_with_flows = len(self.flow_tables)
        
        flow_age_distribution = []
        flow_usage_distribution = []
        qos_distribution = {'EMERGENCY': 0, 'SAFETY': 0, 'SERVICE': 0, 'BACKGROUND': 0}
        
        for node_id, flows in self.flow_tables.items():
            for flow_id, flow_entry in flows.items():
                total_flows += 1
                
                # Check flow state
                if flow_entry.state == FlowState.ACTIVE:
                    active_flows += 1
                elif flow_entry.state in [FlowState.EXPIRED, FlowState.DELETED]:
                    expired_flows += 1
                
                # Age distribution
                flow_age = current_time - flow_entry.creation_time
                flow_age_distribution.append(flow_age)
                
                # Usage distribution
                if hasattr(flow_entry, 'packet_count'):
                    flow_usage_distribution.append(flow_entry.packet_count)
                
                # QoS distribution
                qos_reqs = flow_entry.qos_requirements
                if 'qos_class' in qos_reqs:
                    qos_class = qos_reqs['qos_class']
                    if qos_class in qos_distribution:
                        qos_distribution[qos_class] += 1
        
        # Calculate statistics
        avg_flow_age = sum(flow_age_distribution) / max(1, len(flow_age_distribution))
        avg_flow_usage = sum(flow_usage_distribution) / max(1, len(flow_usage_distribution))
        
        return {
            'total_flows': total_flows,
            'active_flows': active_flows,
            'expired_flows': expired_flows,
            'nodes_with_flows': total_nodes_with_flows,
            'avg_flow_age_seconds': avg_flow_age,
            'avg_packets_per_flow': avg_flow_usage,
            'qos_distribution': qos_distribution,
            'cache_size': len(self.path_computation_cache),
            'traffic_matrix_size': len(self.traffic_matrix),
            'topology_nodes': len(self.network_topology.nodes()),
            'topology_edges': len(self.network_topology.edges()),
            'avg_controller_latency': self.get_average_controller_latency(),
            'avg_flow_installation_time': self.get_flow_installation_overhead()
        }
    
    def force_flow_cleanup(self, node_id: str = None):
        """Force immediate cleanup of flows for specific node or all nodes"""
        if node_id:
            # Clean specific node
            if node_id in self.flow_tables:
                flows_removed = len(self.flow_tables[node_id])
                del self.flow_tables[node_id]
                print(f"[SDN FORCE CLEANUP] Removed {flows_removed} flows from node {node_id}")
        else:
            # Clean all nodes
            total_flows = sum(len(flows) for flows in self.flow_tables.values())
            self.flow_tables.clear()
            self.global_flows.clear()
            self.path_computation_cache.clear()
            self.traffic_matrix.clear()
            print(f"[SDN FORCE CLEANUP] Removed all {total_flows} flows from all nodes")
    
    def optimize_flow_tables(self):
        """Optimize flow tables by consolidating similar flows"""
        optimization_count = 0
        
        for node_id, flows in self.flow_tables.items():
            flow_list = list(flows.items())
            flows_to_merge = []
            
            # Look for flows with similar match fields that can be consolidated
            for i, (flow_id1, flow1) in enumerate(flow_list):
                for j, (flow_id2, flow2) in enumerate(flow_list[i+1:], i+1):
                    if self._flows_can_be_merged(flow1, flow2):
                        flows_to_merge.append((flow_id1, flow_id2, flow1, flow2))
            
            # Merge compatible flows
            for flow_id1, flow_id2, flow1, flow2 in flows_to_merge:
                if flow_id1 in flows and flow_id2 in flows:
                    # Merge flow2 into flow1
                    merged_flow = self._merge_flows(flow1, flow2)
                    flows[flow_id1] = merged_flow
                    del flows[flow_id2]
                    optimization_count += 1
        
        if optimization_count > 0:
            print(f"[SDN OPTIMIZATION] Merged {optimization_count} similar flows")
    
    def _flows_can_be_merged(self, flow1: FlowEntry, flow2: FlowEntry) -> bool:
        """Check if two flows can be merged based on similar match criteria"""
        # Flows can be merged if they have the same destination and similar QoS requirements
        match1 = flow1.match_fields
        match2 = flow2.match_fields
        
        # Must have same destination
        if match1.get('destination') != match2.get('destination'):
            return False
        
        # Must have same QoS class
        if match1.get('qos_class') != match2.get('qos_class'):
            return False
        
        # Must have compatible actions
        if len(flow1.actions) != len(flow2.actions):
            return False
        
        # Check if actions are similar enough
        for action1, action2 in zip(flow1.actions, flow2.actions):
            if action1.get('type') != action2.get('type'):
                return False
            if action1.get('next_hop') != action2.get('next_hop'):
                return False
        
        return True
    
    def _merge_flows(self, flow1: FlowEntry, flow2: FlowEntry) -> FlowEntry:
        """Merge two compatible flows into one"""
        # Create merged flow based on flow1
        merged_flow = FlowEntry(
            flow_id=f"{flow1.flow_id}_merged",
            match_fields=flow1.match_fields.copy(),
            actions=flow1.actions.copy(),
            priority=max(flow1.priority, flow2.priority),  # Use higher priority
            timeout=max(flow1.timeout, flow2.timeout),     # Use longer timeout
            creation_time=min(flow1.creation_time, flow2.creation_time),  # Earlier creation time
            packet_count=flow1.packet_count + flow2.packet_count,
            byte_count=flow1.byte_count + flow2.byte_count,
            last_used=max(flow1.last_used, flow2.last_used),
            state=FlowState.ACTIVE,
            qos_requirements=flow1.qos_requirements.copy()
        )
        
        return merged_flow
    
    def get_network_statistics(self) -> Dict[str, Any]:
        """Get comprehensive network statistics from SDN controller perspective"""
        current_time = time.time()
        
        # Basic network topology statistics
        total_nodes = self.network_topology.number_of_nodes()
        total_edges = self.network_topology.number_of_edges()
        
        # Calculate network diameter and average path length
        network_diameter = 0
        average_path_length = 0.0
        
        if total_nodes > 1:
            try:
                if nx.is_connected(self.network_topology):
                    network_diameter = nx.diameter(self.network_topology)
                    average_path_length = nx.average_shortest_path_length(self.network_topology)
                else:
                    # For disconnected graphs, calculate for largest component
                    largest_component = max(nx.connected_components(self.network_topology), key=len)
                    subgraph = self.network_topology.subgraph(largest_component)
                    if len(largest_component) > 1:
                        network_diameter = nx.diameter(subgraph)
                        average_path_length = nx.average_shortest_path_length(subgraph)
            except:
                # Handle any graph computation errors
                network_diameter = 0
                average_path_length = 0.0
        
        # Flow table statistics
        total_flows = 0
        active_flows = 0
        expired_flows = 0
        nodes_with_flows = len(self.flow_tables)
        
        flow_distribution_by_qos = {'EMERGENCY': 0, 'SAFETY': 0, 'SERVICE': 0, 'BACKGROUND': 0}
        flow_ages = []
        flow_packet_counts = []
        
        for node_id, flows in self.flow_tables.items():
            for flow_id, flow_entry in flows.items():
                total_flows += 1
                
                if flow_entry.state == FlowState.ACTIVE:
                    active_flows += 1
                elif flow_entry.state in [FlowState.EXPIRED, FlowState.DELETED]:
                    expired_flows += 1
                
                # Flow age
                flow_age = current_time - flow_entry.creation_time
                flow_ages.append(flow_age)
                
                # Packet count
                flow_packet_counts.append(flow_entry.packet_count)
                
                # QoS distribution
                qos_class = flow_entry.qos_requirements.get('qos_class', 'SERVICE')
                if qos_class in flow_distribution_by_qos:
                    flow_distribution_by_qos[qos_class] += 1
        
        # Calculate averages
        avg_flow_age = sum(flow_ages) / max(1, len(flow_ages))
        avg_packets_per_flow = sum(flow_packet_counts) / max(1, len(flow_packet_counts))
        
        # Link utilization statistics
        link_utilizations = {}
        total_network_load = 0.0
        congested_links = 0
        
        for edge in self.network_topology.edges():
            node1, node2 = edge
            edge_key = f"{node1}-{node2}"
            
            # Calculate traffic on this link
            edge_traffic = 0.0
            for (src, dst), load in self.traffic_matrix.items():
                # Check if traffic might use this edge (simplified)
                if src == node1 or dst == node2 or src == node2 or dst == node1:
                    edge_traffic += load
            
            # Assume 10 Mbps link capacity
            link_capacity = 10e6  # bytes per second
            utilization = min(1.0, edge_traffic / link_capacity)
            link_utilizations[edge_key] = utilization
            
            total_network_load += edge_traffic
            
            if utilization > 0.7:  # 70% utilization threshold
                congested_links += 1
        
        # Calculate average link utilization
        avg_link_utilization = sum(link_utilizations.values()) / max(1, len(link_utilizations))
        
        # Controller performance statistics
        avg_controller_latency = 0.0
        if self.controller_processing_delays:
            avg_controller_latency = sum(self.controller_processing_delays) / len(self.controller_processing_delays)
        
        avg_flow_installation_time = 0.0
        if self.flow_installation_times:
            avg_flow_installation_time = sum(self.flow_installation_times) / len(self.flow_installation_times)
        
        # Traffic engineering statistics
        total_traffic_entries = len(self.traffic_matrix)
        cached_paths = len(self.path_computation_cache)
        
        # Network efficiency metrics
        network_density = 0.0
        if total_nodes > 1:
            max_possible_edges = total_nodes * (total_nodes - 1) / 2
            network_density = total_edges / max_possible_edges
        
        # QoS performance estimation
        qos_performance = {}
        for qos_class, count in flow_distribution_by_qos.items():
            if count > 0:
                # Estimate QoS performance based on network conditions
                if qos_class in ['EMERGENCY', 'SAFETY']:
                    # High priority traffic should have better performance
                    estimated_delay = avg_controller_latency * 0.5  # Priority handling
                    estimated_throughput = total_network_load * 1.2  # Better allocation
                else:
                    estimated_delay = avg_controller_latency
                    estimated_throughput = total_network_load * 0.8
                
                qos_performance[qos_class] = {
                    'estimated_delay_ms': estimated_delay * 1000,
                    'estimated_throughput_mbps': estimated_throughput / 1e6,
                    'flow_count': count
                }
        
        # Topology stability metrics
        topology_age = current_time - self.last_topology_update
        
        # Network reachability
        reachable_node_pairs = 0
        total_node_pairs = 0
        
        nodes_list = list(self.network_topology.nodes())
        for i, source in enumerate(nodes_list):
            for j, target in enumerate(nodes_list[i+1:], i+1):
                total_node_pairs += 1
                try:
                    if nx.has_path(self.network_topology, source, target):
                        reachable_node_pairs += 1
                except:
                    pass
        
        network_reachability = reachable_node_pairs / max(1, total_node_pairs)
        
        # Load balancing effectiveness
        if link_utilizations:
            utilization_values = list(link_utilizations.values())
            utilization_std = np.std(utilization_values) if len(utilization_values) > 1 else 0.0
            load_balancing_effectiveness = 1.0 - min(1.0, utilization_std)  # Lower std = better balancing
        else:
            load_balancing_effectiveness = 0.0
        
        # Compile comprehensive statistics
        stats = {
            # Basic topology
            'nodes_count': total_nodes,
            'edges_count': total_edges,
            'network_diameter': network_diameter,
            'average_path_length': average_path_length,
            'network_density': network_density,
            'network_reachability': network_reachability,
            
            # Flow management
            'total_flows': total_flows,
            'active_flows': active_flows,
            'expired_flows': expired_flows,
            'nodes_with_flows': nodes_with_flows,
            'avg_flow_age_seconds': avg_flow_age,
            'avg_packets_per_flow': avg_packets_per_flow,
            'flow_distribution_by_qos': flow_distribution_by_qos,
            
            # Network load and performance
            'total_network_load_mbps': total_network_load / 1e6,
            'avg_link_utilization': avg_link_utilization,
            'congested_links_count': congested_links,
            'congestion_ratio': congested_links / max(1, total_edges),
            'load_balancing_effectiveness': load_balancing_effectiveness,
            
            # Controller performance
            'avg_controller_latency_ms': avg_controller_latency * 1000,
            'avg_flow_installation_time_ms': avg_flow_installation_time * 1000,
            'controller_processing_samples': len(self.controller_processing_delays),
            'flow_installation_samples': len(self.flow_installation_times),
            
            # Traffic engineering
            'traffic_matrix_entries': total_traffic_entries,
            'cached_paths': cached_paths,
            'cache_hit_ratio': min(1.0, cached_paths / max(1, total_traffic_entries)),
            
            # QoS management
            'qos_performance': qos_performance,
            'high_priority_flows': flow_distribution_by_qos['EMERGENCY'] + flow_distribution_by_qos['SAFETY'],
            'low_priority_flows': flow_distribution_by_qos['SERVICE'] + flow_distribution_by_qos['BACKGROUND'],
            
            # Operational metrics
            'topology_last_update_age_seconds': topology_age,
            'topology_update_frequency': 1.0 / max(1.0, topology_age),  # Updates per second
            'control_plane_efficiency': active_flows / max(1, total_flows),
            
            # Advanced metrics
            'link_utilization_distribution': link_utilizations,
            'network_convergence_quality': min(1.0, network_reachability * load_balancing_effectiveness),
            'sdn_overhead_ratio': (total_flows * avg_flow_installation_time) / max(1.0, current_time),
            'traffic_engineering_benefit': max(0.0, load_balancing_effectiveness - 0.5) * 2.0,  # 0-1 scale
            
            # Resource utilization
            'memory_usage_estimate': {
                'flow_tables_kb': (total_flows * 200) / 1024,  # Estimate 200 bytes per flow
                'topology_kb': (total_nodes * 100 + total_edges * 50) / 1024,  # Estimate
                'cache_kb': (cached_paths * 150) / 1024,  # Estimate 150 bytes per cached path
                'traffic_matrix_kb': (total_traffic_entries * 50) / 1024  # Estimate
            }
        }
        
        return stats
    
    def get_detailed_link_statistics(self) -> Dict[str, Dict[str, Any]]:
        """Get detailed statistics for each network link"""
        link_stats = {}
        
        for edge in self.network_topology.edges(data=True):
            node1, node2, edge_data = edge
            link_key = f"{node1}-{node2}"
            
            # Calculate traffic on this link
            link_traffic = 0.0
            flow_count = 0
            
            # Count flows using this link
            for node_id in [node1, node2]:
                if node_id in self.flow_tables:
                    for flow_id, flow_entry in self.flow_tables[node_id].items():
                        for action in flow_entry.actions:
                            if (action.get('type') == 'forward' and 
                                ((node_id == node1 and action.get('next_hop') == node2) or
                                 (node_id == node2 and action.get('next_hop') == node1))):
                                
                                # Calculate traffic from this flow
                                time_active = max(1.0, time.time() - flow_entry.creation_time)
                                flow_rate = flow_entry.byte_count / time_active
                                link_traffic += flow_rate
                                flow_count += 1
            
            # Link capacity and utilization
            link_capacity = 10e6  # 10 Mbps
            utilization = min(1.0, link_traffic / link_capacity)
            
            # Link quality metrics
            distance = edge_data.get('distance', 0)
            quality = edge_data.get('quality', 0)
            sdn_weight = edge_data.get('sdn_weight', distance)
            
            link_stats[link_key] = {
                'distance_m': distance,
                'link_quality': quality,
                'sdn_weight': sdn_weight,
                'traffic_mbps': link_traffic / 1e6,
                'utilization': utilization,
                'capacity_mbps': link_capacity / 1e6,
                'flow_count': flow_count,
                'congestion_level': 'high' if utilization > 0.8 else 'medium' if utilization > 0.5 else 'low',
                'last_seen': edge_data.get('last_seen', 0)
            }
        
        return link_stats
    
    def get_node_statistics(self) -> Dict[str, Dict[str, Any]]:
        """Get detailed statistics for each network node"""
        node_stats = {}
        
        for node_id in self.network_topology.nodes():
            node_data = self.network_topology.nodes[node_id]
            
            # Flow table statistics for this node
            node_flows = self.flow_tables.get(node_id, {})
            active_flows = sum(1 for flow in node_flows.values() if flow.state == FlowState.ACTIVE)
            total_packets = sum(flow.packet_count for flow in node_flows.values())
            total_bytes = sum(flow.byte_count for flow in node_flows.values())
            
            # Neighbor count
            neighbor_count = len(list(self.network_topology.neighbors(node_id)))
            
            # Position information
            position = self.node_positions.get(node_id, (0, 0))
            
            # Node centrality metrics
            try:
                degree_centrality = nx.degree_centrality(self.network_topology)[node_id]
                betweenness_centrality = nx.betweenness_centrality(self.network_topology)[node_id]
            except:
                degree_centrality = 0.0
                betweenness_centrality = 0.0
            
            node_stats[node_id] = {
                'position_x': position[0],
                'position_y': position[1],
                'neighbor_count': neighbor_count,
                'flow_table_size': len(node_flows),
                'active_flows': active_flows,
                'total_packets_processed': total_packets,
                'total_bytes_processed': total_bytes,
                'degree_centrality': degree_centrality,
                'betweenness_centrality': betweenness_centrality,
                'last_seen': node_data.get('last_seen', 0),
                'node_importance': (degree_centrality + betweenness_centrality) / 2.0
            }
        
        return node_stats

class RealisticInterferenceCalculator:
    """CORRECTED IEEE 802.11bd compliant interference calculator - COMPLETE VERSION"""
    
    def __init__(self, config):
        self.config = config
        self.thermal_noise_power = self._calculate_thermal_noise_power()
        self.thermal_noise_power_dbm = -174 + 10 * math.log10(config.bandwidth) + config.noise_figure
        
        # CORRECTED: IEEE 802.11bd specific parameters
        self.ofdm_subcarriers = 48 if config.bandwidth == 10e6 else 108  # 10/20 MHz
        self.guard_interval = 0.8e-6  # 0.8 μs for 802.11bd
        self.symbol_duration = 3.2e-6 + self.guard_interval  # 4 μs total
    
    def _calculate_thermal_noise_power(self) -> float:
        """Calculate thermal noise power in Watts with proper IEEE 802.11bd parameters"""
        k_boltzmann = 1.38e-23  # J/K
        temperature = 290  # K (room temperature)
        bandwidth = self.config.bandwidth  # Hz
        noise_figure_linear = 10**(self.config.noise_figure / 10.0)
        
        thermal_noise_watts = k_boltzmann * temperature * bandwidth * noise_figure_linear
        return thermal_noise_watts
    
    def calculate_path_loss_db(self, distance_m: float, frequency_hz: float, channel_model: str = 'highway_los') -> float:
        """CORRECTED: Calculate path loss using IEEE 802.11bd vehicular propagation model"""
        if distance_m <= 1.0:
            distance_m = 1.0
        
        # CORRECTED: Free space path loss for 5.9 GHz
        wavelength = 3e8 / frequency_hz  # wavelength in meters
        fspl_db = 20 * math.log10(4 * math.pi * distance_m / wavelength)
        
        # CORRECTED: Channel-specific path loss model
        if 'highway_los' in channel_model:
            path_loss_exponent = 2.1
            shadowing_std = 3.5
            env_loss = 2.0 + (distance_m / 100) * 1.5
        elif 'highway_nlos' in channel_model:
            path_loss_exponent = 2.6
            shadowing_std = 5.0
            env_loss = 5.0 + (distance_m / 100) * 2.0
        elif 'urban_approaching_los' in channel_model:
            path_loss_exponent = 2.2
            shadowing_std = 4.0
            env_loss = 3.0 + (distance_m / 100) * 1.8
        elif 'urban_crossing_nlos' in channel_model:
            path_loss_exponent = 2.8
            shadowing_std = 6.0
            env_loss = 8.0 + (distance_m / 100) * 2.5
        elif 'rural_los' in channel_model:
            path_loss_exponent = 2.0
            shadowing_std = 3.0
            env_loss = 1.5 + (distance_m / 100) * 1.0
        else:
            path_loss_exponent = 2.0
            shadowing_std = 4.0
            env_loss = 2.0
        
        # Additional loss for longer distances
        if distance_m > 100:
            additional_loss = 10 * math.log10((distance_m / 100) ** (path_loss_exponent - 2.0))
        else:
            additional_loss = 0
        
        # CORRECTED: Realistic shadowing for vehicular environment
        shadowing = abs(random.gauss(0, shadowing_std + distance_m / 200))  # Always positive loss
        
        total_path_loss = fspl_db + env_loss + additional_loss + shadowing
        
        # Minimum theoretical FSPL at 1m for 5.9GHz is 32.44 dB
        return max(32.44, total_path_loss)
    
    def calculate_received_power_dbm(self, tx_power_dbm: float, distance_m: float, 
                                   tx_gain_db: float = 0, rx_gain_db: float = 0,
                                   channel_model: str = 'highway_los') -> float:
        """CORRECTED: Calculate received power with IEEE 802.11bd propagation"""
        path_loss_db = self.calculate_path_loss_db(distance_m, self.config.frequency, channel_model)
        received_power_dbm = (tx_power_dbm + tx_gain_db + rx_gain_db - path_loss_db)
        return received_power_dbm
    
    def calculate_dynamic_communication_range(self, tx_power_dbm: float, channel_model: str = 'highway_los') -> float:
        """CORRECTED: IEEE 802.11bd realistic communication range calculation"""
        sensitivity_dbm = self.config.receiver_sensitivity_dbm
        antenna_gains = self.config.g_t + self.config.g_r
        implementation_margin = 3.0
        fading_margin = self.config.fading_margin_db
        
        # CORRECTED: Link budget calculation
        total_budget = tx_power_dbm + antenna_gains - sensitivity_dbm - implementation_margin - fading_margin
        
        # CORRECTED: Path loss at reference distance (1m) for 5.9 GHz
        frequency_mhz = self.config.frequency / 1e6
        reference_loss = 32.44 + 20 * math.log10(frequency_mhz)  # 1m path loss
        
        if total_budget > reference_loss:
            # Calculate range using path loss model
            if 'highway_los' in channel_model:
                path_loss_exponent = 2.1
            elif 'highway_nlos' in channel_model:
                path_loss_exponent = 2.6
            elif 'urban_approaching_los' in channel_model:
                path_loss_exponent = 2.2
            elif 'urban_crossing_nlos' in channel_model:
                path_loss_exponent = 2.8
            elif 'rural_los' in channel_model:
                path_loss_exponent = 2.0
            else:
                path_loss_exponent = 2.0
            
            # Solve for distance: PL(d) = PL(1m) + 10*n*log10(d)
            distance_factor = (total_budget - reference_loss) / (10 * path_loss_exponent)
            range_km = 10**distance_factor
            range_m = range_km * 1000
        else:
            range_m = 10  # Minimum range
        
        # CORRECTED: Channel-specific range bounds
        channel_range_bounds = {
            'highway_los': (100, 400),
            'highway_nlos': (60, 250),
            'urban_approaching_los': (80, 300),
            'urban_crossing_nlos': (40, 150),
            'rural_los': (150, 500)
        }
        
        min_range, max_range = channel_range_bounds.get(channel_model, (80, 300))
        
        # Apply bounds and variation
        final_range = max(min_range, min(range_m, max_range))
        variation = random.uniform(0.85, 1.15)  # ±15% variation
        final_range *= variation
        
        return final_range
    
    def calculate_sinr_with_interference(self, vehicle_id: str, neighbors: list, 
                           vehicle_tx_power: float, channel_model: str,
                           background_traffic_manager=None) -> float:
        """ENHANCED SINR calculation with stronger neighbor interference modeling"""
        
        if not neighbors:
            # No neighbors - use noise-limited scenario
            reference_distance = 100  # meters
            signal_power_dbm = self.calculate_received_power_dbm(
                vehicle_tx_power, reference_distance, self.config.g_t, self.config.g_r, channel_model
            )
            
            thermal_noise_power_mw = 10**((self.thermal_noise_power_dbm - 30) / 10.0)
            
            if background_traffic_manager:
                background_interference_mw = background_traffic_manager.calculate_background_interference_contribution(
                    vehicle_id, [])
            else:
                background_interference_mw = thermal_noise_power_mw * self.config.background_traffic_load * 2.0
            
            total_noise_mw = thermal_noise_power_mw + background_interference_mw
            signal_power_mw = 10**((signal_power_dbm - 30) / 10.0)
            
            if total_noise_mw > 0 and signal_power_mw > 0:
                snr_linear = signal_power_mw / total_noise_mw
                snr_db = 10 * math.log10(snr_linear)
            else:
                snr_db = -10.0
            
            return max(-25, snr_db)
        
        # Use strongest signal as desired signal
        strongest_neighbor = max(neighbors, key=lambda n: n.get('rx_power_dbm', -150))
        signal_power_dbm = strongest_neighbor.get('rx_power_dbm', -100)
        
        # ENHANCED: Calculate interference from all other neighbors with stronger modeling
        total_interference_power_mw = 0
        thermal_noise_power_mw = 10**((self.thermal_noise_power_dbm - 30) / 10.0)
        
        for neighbor in neighbors:
            if neighbor['id'] != strongest_neighbor['id']:
                distance = neighbor['distance']
                
                if distance > 3 and distance < 2500:  # Expanded interference range
                    intf_power_dbm = self.calculate_received_power_dbm(
                        neighbor['tx_power'], 
                        distance,
                        self.config.g_t, 
                        self.config.g_r,
                        channel_model
                    )
                    
                    # ENHANCED: Lower threshold for interference consideration
                    if intf_power_dbm > self.thermal_noise_power_dbm + 6:  # Reduced from 8
                        intf_power_mw = 10**((intf_power_dbm - 30) / 10.0)
                        
                        # ENHANCED: Stronger activity factor based on realistic VANET behavior
                        beacon_rate = neighbor.get('beacon_rate', 10.0)
                        packet_duration = 300e-6  # 300 μs average packet duration
                        duty_cycle = beacon_rate * packet_duration
                        activity_factor = min(0.6, duty_cycle * 1.2)  # Increased from 0.4
                        
                        # ENHANCED: More aggressive distance-based interference
                        if distance <= 50:
                            distance_factor = 1.0    # Full interference at close range
                        elif distance <= 100:
                            distance_factor = 0.85   # Increased from 0.8
                        elif distance <= 200:
                            distance_factor = 0.65   # Increased from 0.5
                        elif distance <= 400:
                            distance_factor = 0.35   # Increased from 0.25
                        else:
                            distance_factor = 0.15   # Increased from 0.1
                        
                        # ENHANCED: More realistic capture effect (less forgiving)
                        signal_power_mw = 10**((signal_power_dbm - 30) / 10.0)
                        if signal_power_mw > intf_power_mw * 20:      # Increased threshold from 15
                            capture_factor = 0.08   # Increased from 0.05
                        elif signal_power_mw > intf_power_mw * 8:    # Increased from 5
                            capture_factor = 0.4    # Increased from 0.3
                        else:
                            capture_factor = 0.9    # Increased from 0.8
                        
                        weighted_interference = (intf_power_mw * activity_factor * 
                                               distance_factor * capture_factor)
                        total_interference_power_mw += weighted_interference
        
        # ENHANCED: Background traffic interference with stronger impact
        if background_traffic_manager:
            background_interference_mw = background_traffic_manager.calculate_background_interference_contribution(
                vehicle_id, neighbors)
            background_interference_mw *= 1.0  # No reduction - full impact
        else:
            # ENHANCED: More aggressive fallback background interference
            background_load = self.config.background_traffic_load
            mgmt_interference = thermal_noise_power_mw * (background_load * 0.4) * 2.5  # Increased
            info_interference = thermal_noise_power_mw * (background_load * 0.5) * 3.0  # Increased
            background_interference_mw = mgmt_interference + info_interference
        
        # ENHANCED: Stronger other interference sources
        hidden_node_factor = self.config.hidden_node_factor * 1.0  # No reduction
        num_neighbors = len(neighbors)
        hidden_node_interference_mw = thermal_noise_power_mw * hidden_node_factor * num_neighbors * 0.12  # Increased from 0.05
        
        # ENHANCED: More aggressive inter-system interference
        config = BACKGROUND_INTERFERENCE_CONFIG if 'BACKGROUND_INTERFERENCE_CONFIG' in globals() else {
            "adjacent_channel_interference": 0.15,  # Restored original values
            "co_channel_interference": 0.25,       
            "non_vanet_interference": 0.1         
        }
        
        adjacent_channel_interference_mw = thermal_noise_power_mw * config["adjacent_channel_interference"] * 2.0
        co_channel_interference_mw = thermal_noise_power_mw * config["co_channel_interference"] * 1.5
        non_vanet_interference_mw = thermal_noise_power_mw * config["non_vanet_interference"] * 3.0
        
        # ENHANCED: Additional neighbor density interference
        if num_neighbors > 5:
            density_interference_factor = 1.0 + ((num_neighbors - 5) * 0.02)  # 2% per neighbor above 5
            density_interference_mw = thermal_noise_power_mw * density_interference_factor * 0.5
        else:
            density_interference_mw = 0
        
        # Total noise + interference with enhanced modeling
        total_noise_interference_mw = (thermal_noise_power_mw + 
                                      background_interference_mw + 
                                      total_interference_power_mw + 
                                      hidden_node_interference_mw +
                                      adjacent_channel_interference_mw +
                                      co_channel_interference_mw +
                                      non_vanet_interference_mw +
                                      density_interference_mw)
        
        # Calculate SINR
        signal_power_mw = 10**((signal_power_dbm - 30) / 10.0)
        
        if total_noise_interference_mw > 0 and signal_power_mw > 0:
            sinr_linear = signal_power_mw / total_noise_interference_mw
            sinr_db = 10 * math.log10(sinr_linear)
        else:
            sinr_db = -20.0
        
        # ENHANCED: Channel-specific SINR adjustments (more aggressive)
        if 'nlos' in channel_model.lower():
            sinr_db -= 3.0  # Restored original 3.0 dB NLOS penalty
        
        # ENHANCED: More aggressive neighbor density penalty
        if num_neighbors > 6:  # Lower threshold from 8
            density_penalty = (num_neighbors - 6) * 0.4  # Increased from 0.3 dB per neighbor
            sinr_db -= min(density_penalty, 12.0)  # Increased cap to 12 dB
        
        # Physical bounds
        min_sinr = -25.0
        sinr_db = max(min_sinr, sinr_db)
        
        # Small random variation
        sinr_db += random.gauss(0, 0.4)  # Increased from 0.3
        
        return sinr_db
    
    def _analyze_fcd_scenario(self, mobility_data: List[Dict]) -> Dict:
        """DYNAMIC analysis of FCD data to determine scenario characteristics"""
        if not mobility_data:
            return {}
        
        # Extract basic scenario information
        all_times = sorted(set(data['time'] for data in mobility_data))
        all_vehicles = set(data['id'] for data in mobility_data)
        
        # Analyze spatial characteristics
        all_x = [data['x'] for data in mobility_data]
        all_y = [data['y'] for data in mobility_data]
        
        x_min, x_max = min(all_x), max(all_x)
        y_min, y_max = min(all_y), max(all_y)
        
        road_length = x_max - x_min if (x_max - x_min) > (y_max - y_min) else y_max - y_min
        road_width = y_max - y_min if (x_max - x_min) > (y_max - y_min) else x_max - x_min
        
        # Estimate lanes (assuming each lane is ~3-4 meters wide)
        estimated_lanes = max(1, int(road_width / 3.5))
        
        # Calculate vehicle density at peak time
        vehicle_counts_per_time = {}
        for time_point in all_times:
            count = len([data for data in mobility_data if data['time'] == time_point])
            vehicle_counts_per_time[time_point] = count
        
        max_vehicles = max(vehicle_counts_per_time.values())
        avg_vehicles = sum(vehicle_counts_per_time.values()) / len(vehicle_counts_per_time)
        
        # Calculate road area and vehicle density
        road_area = road_length * road_width  # m²
        peak_density = max_vehicles / road_area if road_area > 0 else 0
        avg_density = avg_vehicles / road_area if road_area > 0 else 0
        
        # Analyze vehicle speeds to determine scenario type
        all_speeds = [data['speed'] for data in mobility_data if 'speed' in data]
        avg_speed = sum(all_speeds) / len(all_speeds) if all_speeds else 0
        max_speed = max(all_speeds) if all_speeds else 0
        
        # Determine scenario type based on characteristics
        if avg_speed > 15:  # >54 km/h
            scenario_type = "highway"
        elif avg_speed > 8:  # >29 km/h
            scenario_type = "urban_arterial"
        else:
            scenario_type = "urban_intersection"
        
        scenario_info = {
            'total_vehicles': len(all_vehicles),
            'max_concurrent_vehicles': max_vehicles,
            'avg_concurrent_vehicles': avg_vehicles,
            'road_length_m': road_length,
            'road_width_m': road_width,
            'estimated_lanes': estimated_lanes,
            'road_area_m2': road_area,
            'peak_density_veh_per_m2': peak_density,
            'avg_density_veh_per_m2': avg_density,
            'avg_speed_ms': avg_speed,
            'max_speed_ms': max_speed,
            'scenario_type': scenario_type,
            'simulation_duration_s': max(all_times) - min(all_times),
            'time_points': len(all_times),
            'density_category': self._categorize_density(peak_density, max_vehicles),
            'x_range': (x_min, x_max),
            'y_range': (y_min, y_max)
        }
        
        return scenario_info
    
    def _categorize_density(self, density: float, vehicle_count: int) -> str:
        """Categorize vehicle density for adaptive range calculation"""
        if vehicle_count < 10:
            return "low_density"
        elif vehicle_count < 25:
            return "medium_density"
        elif vehicle_count < 50:
            return "high_density"
        else:
            return "very_high_density"

# NEW: Enhanced Vehicle State with Layer 3 and SDN capabilities
class VehicleState:
    """Enhanced vehicle state with Layer 3 networking and SDN capabilities"""
    def __init__(self, vehicle_id: str, mac_address: str, ip_address: str):
        self.vehicle_id = vehicle_id
        self.mac_address = mac_address
        self.ip_address = ip_address
        
        # Position and mobility (from FCD)
        self.x = 0.0
        self.y = 0.0
        self.speed = 0.0
        self.angle = 0.0
        
        # Communication parameters (IEEE 802.11bd)
        self.transmission_power = TRANSMISSION_POWER_DBM
        self.mcs = MCS
        self.beacon_rate = BEACON_RATE
        
        # Derived parameters (dynamic)
        self.neighbors = []
        self.neighbors_number = 0
        self.comm_range = 0.0
        
        # Performance metrics
        self.current_cbr = 0.0
        self.current_snr = 0.0
        self.current_per = 0.0
        self.current_ber = 0.0
        self.current_ser = 0.0
        self.current_throughput = 0.0
        self.current_latency = 0.0
        self.current_pdr = 1.0
        
        # MAC metrics
        self.mac_success = 0
        self.mac_retries = 0
        self.mac_drops = 0
        self.mac_total_attempts = 0
        self.mac_delays = []
        self.mac_latencies = []
        self.mac_throughputs = []
        
        # Counters
        self.total_tx = 0
        self.successful_tx = 0
        
        # NEW: Layer 3 networking components
        self.routing_protocol = None
        self.routing_table: Dict[str, RouteEntry] = {}
        self.neighbor_discovery_time = 0.0
        self.last_hello_time = 0.0
        
        # NEW: SDN agent components
        self.flow_table: Dict[str, FlowEntry] = {}
        self.sdn_agent_active = False
        self.controller_connection = None
        self.pending_packet_ins: List[NetworkPacket] = []
        
        # NEW: Packet processing components
        self.packet_queue: queue.Queue = queue.Queue()
        self.packet_counters = {
            'generated': 0,
            'received': 0,
            'forwarded': 0,
            'dropped': 0,
            'delivered': 0
        }
        self.application_traffic: Dict[str, List[NetworkPacket]] = {
            'SAFETY': [],
            'INFOTAINMENT': [],
            'SENSING': []
        }
        
        # NEW: QoS and traffic management
        self.qos_queues: Dict[QoSClass, queue.Queue] = {
            QoSClass.EMERGENCY: queue.Queue(),
            QoSClass.SAFETY: queue.Queue(),
            QoSClass.SERVICE: queue.Queue(),
            QoSClass.BACKGROUND: queue.Queue()
        }
        
        # NEW: Performance tracking for Layer 3
        self.l3_metrics = {
            'route_discovery_attempts': 0,
            'route_discovery_success': 0,
            'route_discovery_latency': [],
            'packet_forwarding_ratio': 0.0,
            'end_to_end_delay': [],
            'hop_count_distribution': [],
            'routing_overhead_ratio': 0.0
        }
        
        # NEW: SDN-specific metrics
        self.sdn_metrics = {
            'flow_installation_time': [],
            'flow_rule_count': 0,
            'packet_in_count': 0,
            'flow_mod_count': 0,
            'controller_latency': [],
            'flow_utilization': {}
        }
        
        self.is_attacker = False
        self.attack_type = AttackType.NONE
        self.attack_metrics = AttackMetrics()
        self.normal_beacon_rate = BEACON_RATE  # Store original beacon rate
        self.normal_tx_power = TRANSMISSION_POWER_DBM  # Store original power
        
        # Attack behavior tracking
        self.attack_behavior_history = []
        self.last_attack_action_time = 0.0
        
        # Detection features
        self.detection_window_data = []
        self.baseline_performance = {
            'throughput': 0.0,
            'latency': 0.0,
            'pdr': 1.0,
            'cbr': 0.0
        }
        
    def sync_antenna_power_with_transmission_power(self):
        """Ensure antenna system power is synchronized with transmission_power"""
        if hasattr(self, 'antenna_system'):
            if ANTENNA_TYPE == "OMNIDIRECTIONAL":
                # Sync omnidirectional config with transmission_power
                self.antenna_system.config.omnidirectional_config["power_dbm"] = self.transmission_power
            # For sectoral antennas, the distribute_power_from_rl method handles the sync

# NEW: Enhanced VANET Simulator with Layer 3 and SDN
class VANET_IEEE80211bd_L3_SDN_Simulator:
    """Enhanced IEEE 802.11bd VANET simulator with Layer 3 stack and SDN capabilities"""
    
    def __init__(self, config: SimulationConfig, fcd_file: str, enable_rl: bool = False, 
             rl_host: str = '127.0.0.1', rl_port: int = 5000):
        self.config = config
        self.fcd_file = fcd_file
        self.enable_rl = enable_rl
        self.rl_host = rl_host
        self.rl_port = rl_port
        self.vehicles = {}
        self.simulation_results = []
        self.ieee_mapper = IEEE80211bdMapper()
        self.interference_calculator = RealisticInterferenceCalculator(config)
        
        # FCD reloading parameters
        self.fcd_reload_count = FCD_RELOAD_COUNT
        self.fcd_reload_strategy = FCD_RELOAD_VEHICLE_ID_STRATEGY
        self.original_simulation_duration = 0
        self.total_simulation_duration = 0
        
        # RL client connection
        self.rl_client = None
        if self.enable_rl:
            self._initialize_rl_connection()
        
        # Load FCD data with reloading capability
        self.mobility_data = self._load_fcd_data_with_reloading()
        
        # Scenario analysis
        self.scenario_info = self.interference_calculator._analyze_fcd_scenario(self.mobility_data)
        
        # NEW: Layer 3 networking components
        self.routing_protocols: Dict[str, Any] = {}
        self.network_topology = nx.Graph()
        self.global_routing_table: Dict[str, Dict[str, RouteEntry]] = {}
        
        # NEW: SDN Controller
        self.sdn_controller = None
        if self.config.enable_sdn:
            self.sdn_controller = VANETSDNController(config)
        
        # NEW: Packet simulation infrastructure
        self.active_flows: Dict[str, NetworkPacket] = {}
        self.packet_trace: List[Dict] = []
        self.traffic_generators: Dict[str, Any] = {}
        
        # NEW: Performance monitoring
        self.l3_performance_monitor = {
            'total_packets_generated': 0,
            'total_packets_delivered': 0,
            'total_routing_overhead': 0,
            'average_route_length': 0.0,
            'route_discovery_success_rate': 0.0,
            'end_to_end_delays': [],
            'packet_delivery_ratios': []
        }
        
        self.sdn_performance_monitor = {
            'total_flow_installations': 0,
            'successful_flow_installations': 0,
            'controller_processing_times': [],
            'network_convergence_times': [],
            'traffic_engineering_events': 0,
            'qos_violations': 0
        }
        
        # FIXED: Initialize attack manager BEFORE initializing vehicles
        self.attack_manager = VANETAttackManager(config) if ENABLE_ATTACK_SIMULATION else None
        self.attack_dataset = []
        
        # Initialize vehicles (now attack_manager exists)
        self._initialize_vehicles()
        
        # Initialize Layer 3 protocols
        if self.config.enable_layer3:
            self._initialize_layer3_protocols()
        
        # Initialize output files
        self._initialize_output_files()
        
        # Initialize visualization
        if VISUALIZATION_CONFIG['enabled']:
            self._initialize_visualization()
        
        # Display scenario information
        if self.scenario_info:
            self._display_scenario_info()
        if ENABLE_BACKGROUND_TRAFFIC_MODELING:
            self.background_traffic_manager = BackgroundTrafficManager(config)
            print(f"[BACKGROUND TRAFFIC] Enhanced background traffic modeling enabled")
            print(f"  Management traffic: {MANAGEMENT_TRAFFIC_CONFIG['enabled']}")
            print(f"  Infotainment traffic: {INFOTAINMENT_TRAFFIC_CONFIG['enabled']}")
            print(f"  Total background load: {BACKGROUND_TRAFFIC_TOTAL_LOAD}")
        else:
            self.background_traffic_manager = None
        
    def get_antenna_configuration_summary(self) -> str:
        """Get summary of antenna configuration for logging"""
        if ANTENNA_TYPE == "OMNIDIRECTIONAL":
            return f"Omnidirectional antenna at {OMNIDIRECTIONAL_ANTENNA_CONFIG['power_dbm']}dBm"
        else:
            rl_controlled = ", ".join(RL_CONTROLLED_SECTORS)
            static_sectors = ", ".join(RL_STATIC_SECTORS)
            return (f"Sectoral antenna: RL-controlled({rl_controlled}), "
                    f"Static({static_sectors} at {SIDE_ANTENNA_STATIC_POWER}dBm)")
    
    def _load_fcd_data_with_reloading(self) -> List[Dict]:
        """Load FCD data with reloading capability for extended simulation duration (from script 2)"""
        print(f"[FCD RELOADING] Loading FCD data with {self.fcd_reload_count} reload(s)")
        
        # Load original data first
        original_data = self._load_original_fcd_data()
        
        if not original_data:
            raise ValueError("Failed to load original FCD data")
        
        # Calculate original simulation parameters
        original_times = [data['time'] for data in original_data]
        self.original_simulation_duration = max(original_times) - min(original_times)
        original_vehicle_ids = set(data['id'] for data in original_data)
        
        print(f"[FCD RELOADING] Original simulation: {self.original_simulation_duration:.1f}s, {len(original_vehicle_ids)} vehicles")
        
        # If reload count is 1, just return original data
        if self.fcd_reload_count <= 1:
            self.total_simulation_duration = self.original_simulation_duration
            print(f"[FCD RELOADING] Using original data only (total duration: {self.total_simulation_duration:.1f}s)")
            return original_data
        
        # Create extended data by reloading
        extended_data = []
        
        for reload_episode in range(self.fcd_reload_count):
            print(f"[FCD RELOADING] Processing episode {reload_episode + 1}/{self.fcd_reload_count}")
            
            # Calculate time offset for this episode
            time_offset = reload_episode * (self.original_simulation_duration + 1.0)  # +1s gap between episodes
            
            # Process each data point in the original data
            for original_point in original_data:
                # Create new data point with time offset
                new_point = original_point.copy()
                new_point['time'] = original_point['time'] + time_offset
                
                # Handle vehicle ID strategy
                if self.fcd_reload_strategy == "suffix" and reload_episode > 0:
                    # Add episode suffix to vehicle ID
                    original_id = original_point['id']
                    new_point['id'] = f"{original_id}_ep{reload_episode + 1}"
                elif self.fcd_reload_strategy == "reuse":
                    # Keep original vehicle IDs (vehicles "continue" across episodes)
                    new_point['id'] = original_point['id']
                else:
                    # Default: use suffix for all episodes except first
                    if reload_episode > 0:
                        original_id = original_point['id']
                        new_point['id'] = f"{original_id}_ep{reload_episode + 1}"
                
                extended_data.append(new_point)
        
        # Calculate total simulation duration
        extended_times = [data['time'] for data in extended_data]
        self.total_simulation_duration = max(extended_times) - min(extended_times)
        total_vehicle_ids = set(data['id'] for data in extended_data)
        
        print(f"[FCD RELOADING] Extended simulation created:")
        print(f"  - Total duration: {self.total_simulation_duration:.1f}s ({self.fcd_reload_count}x reloaded)")
        print(f"  - Total unique vehicles: {len(total_vehicle_ids)}")
        print(f"  - Total data points: {len(extended_data)}")
        print(f"  - Vehicle ID strategy: {self.fcd_reload_strategy}")
        
        return extended_data
    
    def _load_original_fcd_data(self) -> List[Dict]:
        """Load original FCD data from XML file"""
        mobility_data = []
        
        try:
            tree = ET.parse(self.fcd_file)
            root = tree.getroot()
            
            for timestep in root.findall('timestep'):
                time = float(timestep.get('time'))
                
                for vehicle in timestep.findall('vehicle'):
                    vehicle_data = {
                        'time': time,
                        'id': vehicle.get('id'),
                        'x': float(vehicle.get('x')),
                        'y': float(vehicle.get('y')),
                        'speed': float(vehicle.get('speed', 0.0)),
                        'angle': float(vehicle.get('angle', 0.0)),
                        'lane': vehicle.get('lane', ''),
                        'pos': float(vehicle.get('pos', 0.0))
                    }
                    mobility_data.append(vehicle_data)
            
            print(f"[INFO] Loaded {len(mobility_data)} original mobility data points from {self.fcd_file}")
            
        except Exception as e:
            raise ValueError(f"Failed to load FCD file: {e}")
        
        return mobility_data
    
    def _display_scenario_info(self):
        """Display scenario analysis information including reloading info"""
        print(f"\n[ENHANCED SCENARIO ANALYSIS]")
        print(f"=" * 80)
        print(f"Scenario Type: {self.scenario_info['scenario_type']}")
        print(f"Vehicle Count: {self.scenario_info['total_vehicles']} total, {self.scenario_info['max_concurrent_vehicles']} max concurrent")
        print(f"Road Dimensions: {self.scenario_info['road_length_m']:.0f}m x {self.scenario_info['road_width_m']:.0f}m")
        print(f"Density Category: {self.scenario_info['density_category']}")
        print(f"Average Speed: {self.scenario_info['avg_speed_ms']:.1f} m/s ({self.scenario_info['avg_speed_ms']*3.6:.1f} km/h)")
        
        # Layer 3 and SDN capabilities
        print(f"=" * 80)
        print(f"[LAYER 3 NETWORKING CONFIGURATION]")
        print(f"Layer 3 Enabled: {self.config.enable_layer3}")
        print(f"Routing Protocol: {self.config.routing_protocol}")
        print(f"Multi-hop Communication: {self.config.enable_multi_hop}")
        print(f"Max Hop Count: {self.config.max_hop_count}")
        print(f"Route Discovery Timeout: {self.config.route_discovery_timeout}s")
        
        print(f"[SDN CONFIGURATION]")
        print(f"SDN Enabled: {self.config.enable_sdn}")
        print(f"Controller Type: {self.config.sdn_controller_type}")
        print(f"Control Protocol: {self.config.sdn_control_protocol}")
        print(f"Flow Table Size: {self.config.flow_table_size}")
        print(f"QoS Management: {self.config.enable_qos_management}")
        print(f"Traffic Engineering: {self.config.enable_traffic_engineering}")
        
        print(f"[PACKET SIMULATION]")
        print(f"Packet Simulation: {self.config.enable_packet_simulation}")
        print(f"Packet Generation Rate: {self.config.packet_generation_rate} pkt/s")
        print(f"Application Types: {', '.join(self.config.application_types)}")
        print(f"QoS Classes: {', '.join(self.config.qos_classes)}")
        
        # FCD reloading information
        if self.fcd_reload_count > 1:
            print(f"=" * 80)
            print(f"[FCD RELOADING CONFIGURATION]")
            print(f"Original Simulation Duration: {self.original_simulation_duration:.0f} seconds")
            print(f"Reload Count: {self.fcd_reload_count}x")
            print(f"Total Extended Duration: {self.total_simulation_duration:.0f} seconds")
            print(f"Vehicle ID Strategy: {self.fcd_reload_strategy}")
            print(f"Effective Training Duration: {self.fcd_reload_count}x longer")
        else:
            print(f"Simulation Duration: {self.scenario_info['simulation_duration_s']:.0f} seconds")
        
        print(f"=" * 80)
    
    def _initialize_output_files(self):
        """Initialize output file names with Layer 3 and SDN info"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Build mode string
        mode_parts = ["ieee80211bd"]
        if self.config.enable_layer3:
            mode_parts.append("L3")
            mode_parts.append(self.config.routing_protocol)
        if self.config.enable_sdn:
            mode_parts.append("SDN")
        if self.enable_rl:
            mode_parts.append("RL")
        
        mode = "_".join(mode_parts)
        
        # Add reloading info to filenames
        if self.fcd_reload_count > 1:
            reload_suffix = f"_reload{self.fcd_reload_count}x"
        else:
            reload_suffix = ""
        
        # Real-time CSV file
        self.realtime_csv_file = f"{mode}_realtime{reload_suffix}_{timestamp}.csv"
        
        # Final Excel file
        if OUTPUT_FILENAME:
            base_name, ext = os.path.splitext(OUTPUT_FILENAME)
            self.final_excel_file = f"{base_name}{reload_suffix}{ext}"
        else:
            self.final_excel_file = f"{mode}_results{reload_suffix}_{timestamp}.xlsx"
        
        # Single progressive Excel file
        if EXCEL_UPDATE_FREQUENCY > 0:
            self.progressive_excel_file = f"{mode}_progressive{reload_suffix}_{timestamp}.xlsx"
        
        # Initialize CSV file with enhanced headers
        if ENABLE_REALTIME_CSV:
            self._initialize_csv_file()
    
    def _initialize_csv_file(self):
        """FIXED: Initialize CSV file with ENHANCED headers including offered load metrics"""
        headers = [
            # Basic information
            'Timestamp', 'VehicleID', 'MACAddress', 'IPAddress', 'ChannelModel', 'ApplicationType',
            'PayloadLength', 'Neighbors', 'NeighborNumbers', 'PowerTx', 'MCS', 'MCS_Source',
            'BeaconRate', 'CommRange',
            
            # PHY/MAC performance
            'PHYDataRate', 'PHYThroughput_Legacy', 'PHYThroughput_80211bd',
            'PHYThroughput', 'ThroughputImprovement', 'MACThroughput', 'MACEfficiency', 'Throughput',
            
            # ENHANCED: Detailed Latency Breakdown
            'Total_Latency_ms', 'PHY_Latency_ms', 'MAC_Latency_ms',
            'Preamble_Latency_ms', 'DataTX_Latency_ms', 'DIFS_Latency_ms', 
            'Backoff_Latency_ms', 'Retry_Latency_ms', 'Queue_Latency_ms',
            
            # Layer 3 and SDN latency components
            'L3_Routing_Delay_ms', 'SDN_Processing_Delay_ms',
            
            # Performance metrics
            'BER', 'SER', 'PER_PHY_Base', 'PER_PHY_Enhanced', 'PER_Total', 'PER',
            'CollisionProb', 
            
            # FIXED: CBR and offered load information
            'CBR', 'OfferedLoad', 'CongestionRatio', 'ChannelOverload',
            
            'SINR', 'SignalPower_dBm', 'InterferencePower_dBm', 
            'ThermalNoise_dBm', 'PDR', 'TargetPER_Met', 'TargetPDR_Met',
            
            # IEEE 802.11bd features
            'LDPC_Enabled', 'Midambles_Enabled', 'DCM_Enabled', 'ExtendedRange_Enabled',
            'MIMO_STBC_Enabled',
            
            # MAC statistics
            'MACSuccess', 'MACRetries', 'MACDrops', 'MACAttempts',
            'AvgMACDelay', 'AvgMACLatency', 'AvgMACThroughput',
            
            # Environment
            'BackgroundTraffic', 'HiddenNodeFactor', 'InterSystemInterference',
            
            # Episode tracking (for FCD reloading)
            'Episode', 'TotalEpisodes', 'OriginalTimestamp', 'ReloadStrategy',
            
            # Layer 3 metrics
            'L3_Enabled', 'RoutingProtocol', 'RoutingTableSize', 'RouteDiscoveryAttempts',
            'RouteDiscoverySuccess', 'AvgRouteLength', 'RoutingOverheadRatio',
            'PacketsGenerated', 'PacketsReceived', 'PacketsForwarded', 'PacketsDropped',
            'EndToEndDelay_ms', 'HopCount', 'L3_PDR',
            
            # SDN metrics
            'SDN_Enabled', 'FlowTableSize', 'ActiveFlows', 'FlowInstallationTime_ms',
            'PacketInCount', 'FlowModCount', 'ControllerLatency_ms', 'QoSViolations',
            'TrafficEngineeringEvents', 'SDN_Throughput_Improvement',
            
            # Application-specific metrics
            'SafetyPackets', 'InfotainmentPackets', 'SensingPackets',
            'EmergencyQoS', 'SafetyQoS', 'ServiceQoS', 'BackgroundQoS',
            'QoS_DelayViolations', 'QoS_ThroughputViolations'
        ]
        
        try:
            with open(self.realtime_csv_file, 'w', newline='') as csvfile:
                import csv
                writer = csv.writer(csvfile)
                writer.writerow(headers)
            print(f"[INFO] Enhanced real-time CSV with offered load metrics initialized: {self.realtime_csv_file}")
        except Exception as e:
            print(f"[WARNING] Failed to initialize enhanced CSV file: {e}")
    
    def _initialize_rl_connection(self):
        """Initialize connection to RL server"""
        try:
            print(f"[RL] Attempting to connect to RL server at {self.rl_host}:{self.rl_port}")
            self.rl_client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.rl_client.settimeout(30)
            self.rl_client.connect((self.rl_host, self.rl_port))
            print(f"[RL]  Successfully connected to RL server at {self.rl_host}:{self.rl_port}")
        except Exception as e:
            print(f"[RL ERROR] ✗ Failed to connect to RL server: {e}")
            print(f"[RL ERROR] Make sure your RL server is running on {self.rl_host}:{self.rl_port}")
            self.enable_rl = False
            self.rl_client = None
    
    def _initialize_vehicles(self):
        """Initialize vehicles with antenna systems and L3 components"""
        vehicle_ids = set(data['id'] for data in self.mobility_data)
        
        for i, vehicle_id in enumerate(sorted(vehicle_ids), 1):
            mac_address = f"00:16:3E:{(i >> 8) & 0xFF:02X}:{i & 0xFF:02X}:{random.randint(0, 255):02X}"
            ip_address = f"192.168.{(i-1)//255}.{((i-1)%255)+1}"
            
            vehicle = VehicleState(vehicle_id, mac_address, ip_address)
            
            # Initialize antenna system
            if ANTENNA_TYPE == "OMNIDIRECTIONAL":
                antenna_config = AntennaConfiguration(AntennaType.OMNIDIRECTIONAL)
            else:
                antenna_config = AntennaConfiguration(AntennaType.SECTORAL)
            
            vehicle.antenna_system = SectoralAntennaSystem(antenna_config)
            
            # FIXED: Ensure proper L3 initialization
            self._ensure_vehicle_l3_initialization(vehicle)
            
            self.vehicles[vehicle_id] = vehicle
        
        print(f"[INFO] Initialized {len(self.vehicles)} vehicles with {ANTENNA_TYPE} antenna systems")
        
        # Initialize attack manager
        if self.attack_manager:
            vehicle_ids = list(self.vehicles.keys())
            self.attack_manager.initialize_attackers(vehicle_ids, self.mobility_data)
            
            for vehicle_id, vehicle in self.vehicles.items():
                if self.attack_manager.is_attacker(vehicle_id):
                    vehicle.is_attacker = True
                    vehicle.attack_metrics = self.attack_manager.get_attack_metrics(vehicle_id)
        
    def _update_attacker_behavior(self, current_time: float):
        """Update attacker behavior during simulation - FIXED to process all vehicles for ML dataset"""
        if not self.attack_manager:
            return
        
        # FIXED: Process ALL vehicles (both normal and attackers) for detection feature calculation
        for vehicle_id, vehicle in self.vehicles.items():
            # Update behavior for all vehicles - this calculates detection features for everyone
            self.attack_manager.update_attacker_behavior(vehicle_id, vehicle, current_time)
    
    def _collect_attack_detection_data(self, current_time: float):
        """Collect data for attack detection dataset"""
        if not self.attack_manager or not GENERATE_ATTACK_DATASET:
            return
        
        if int(current_time) % DETECTION_FEATURES_UPDATE_INTERVAL == 0:
            detection_data = self.attack_manager.get_detection_dataset()
            self.attack_dataset.extend(detection_data)
    
    def _initialize_layer3_protocols(self):
        """Initialize Layer 3 routing protocols for all vehicles with proper timing"""
        if not self.config.enable_layer3:
            return
        
        protocol_name = self.config.routing_protocol.upper()
        
        for vehicle_id, vehicle in self.vehicles.items():
            try:
                if protocol_name == "AODV":
                    vehicle.routing_protocol = AODVRoutingProtocol(vehicle_id, self.config)
                elif protocol_name == "OLSR":
                    vehicle.routing_protocol = OLSRRoutingProtocol(vehicle_id, self.config)
                elif protocol_name == "GEOGRAPHIC":
                    vehicle.routing_protocol = GeographicRoutingProtocol(vehicle_id, self.config)
                elif protocol_name == "HYBRID":
                    # Use AODV as primary with geographic fallback
                    vehicle.routing_protocol = AODVRoutingProtocol(vehicle_id, self.config)
                
                # Initialize routing protocol with current time
                if vehicle.routing_protocol:
                    self.routing_protocols[vehicle_id] = vehicle.routing_protocol
                    
                    # Set initial timing to avoid immediate packet generation
                    current_time = time.time()
                    if hasattr(vehicle.routing_protocol, 'last_hello_time'):
                        vehicle.routing_protocol.last_hello_time = current_time
                    if hasattr(vehicle.routing_protocol, 'last_tc_time'):
                        vehicle.routing_protocol.last_tc_time = current_time
                    if hasattr(vehicle.routing_protocol, 'last_beacon_time'):
                        vehicle.routing_protocol.last_beacon_time = current_time
                        
            except Exception as e:
                print(f"[ERROR] Failed to initialize routing protocol for {vehicle_id}: {e}")
                vehicle.routing_protocol = None
        
        print(f"[L3] Initialized {protocol_name} routing protocol for {len(self.vehicles)} vehicles")
    
    def _calculate_distance(self, pos1: Tuple[float, float], pos2: Tuple[float, float]) -> float:
        """Calculate Euclidean distance"""
        return math.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
    
    def _find_neighbors(self, vehicle_id: str, current_time: float) -> List[Dict]:
        """Enhanced neighbor finding with COMPLETELY FIXED sectoral antenna modeling"""
        neighbors = []
        
        vehicle_data = next((data for data in self.mobility_data 
                           if data['time'] == current_time and data['id'] == vehicle_id), None)
        if not vehicle_data:
            return neighbors
        
        vehicle = self.vehicles[vehicle_id]
        vehicle_pos = (vehicle_data['x'], vehicle_data['y'])
        
        # Get vehicle heading from FCD data (0°=North, 90°=East, 180°=South, 270°=West)
        vehicle_heading = vehicle_data.get('angle', 0)
        
        # Use the vehicle's existing antenna system
        if hasattr(vehicle, 'antenna_system') and vehicle.antenna_system:
            antenna_system = vehicle.antenna_system
            antenna_config = antenna_system.config
        else:
            # Fallback: create antenna system
            if ANTENNA_TYPE == "OMNIDIRECTIONAL":
                antenna_config = AntennaConfiguration(AntennaType.OMNIDIRECTIONAL)
            else:
                antenna_config = AntennaConfiguration(AntennaType.SECTORAL)
            antenna_system = SectoralAntennaSystem(antenna_config)
            vehicle.antenna_system = antenna_system
        
        # Base communication range calculation
        base_comm_range = self.interference_calculator.calculate_dynamic_communication_range(
            vehicle.transmission_power)
        
        # Debug setup
        debug_neighbor_count = 0
        debug_enabled = (vehicle_id in ["veh0", "veh1"]) and (int(current_time) % 100 == 0)
        
        if debug_enabled:
            print(f"\n[ANTENNA DEBUG] Vehicle {vehicle_id} at t={current_time:.1f}s:")
            print(f"  Antenna type: {ANTENNA_TYPE}")
            print(f"  Vehicle heading: {vehicle_heading:.1f}° (0°=North, 90°=East)")
            print(f"  Vehicle transmission_power: {vehicle.transmission_power:.1f} dBm")
            print(f"  Base comm range: {base_comm_range:.1f}m")
            
            if ANTENNA_TYPE == "SECTORAL":
                if hasattr(antenna_system, 'sector_powers'):
                    print(f"  ACTUAL Sector powers: F={antenna_system.sector_powers.get('front', 'N/A'):.1f}, "
                          f"R={antenna_system.sector_powers.get('rear', 'N/A'):.1f}, "
                          f"L={antenna_system.sector_powers.get('left', 'N/A'):.1f}, "
                          f"R={antenna_system.sector_powers.get('right', 'N/A'):.1f} dBm")
                
                # Force correct powers if wrong
                if (antenna_system.sector_powers.get('front', 0) != SECTORAL_ANTENNA_CONFIG['front']['power_dbm'] or
                    antenna_system.sector_powers.get('rear', 0) != SECTORAL_ANTENNA_CONFIG['rear']['power_dbm']):
                    
                    print(f"  [FORCE FIX] Correcting sector powers from config!")
                    antenna_system.sector_powers['front'] = SECTORAL_ANTENNA_CONFIG['front']['power_dbm']
                    antenna_system.sector_powers['rear'] = SECTORAL_ANTENNA_CONFIG['rear']['power_dbm']
                    antenna_system.sector_powers['left'] = SECTORAL_ANTENNA_CONFIG['left']['power_dbm']
                    antenna_system.sector_powers['right'] = SECTORAL_ANTENNA_CONFIG['right']['power_dbm']
                    
                    # Also update the config
                    antenna_config.sectoral_config['front'].power_dbm = SECTORAL_ANTENNA_CONFIG['front']['power_dbm']
                    antenna_config.sectoral_config['rear'].power_dbm = SECTORAL_ANTENNA_CONFIG['rear']['power_dbm']
                    antenna_config.sectoral_config['left'].power_dbm = SECTORAL_ANTENNA_CONFIG['left']['power_dbm']
                    antenna_config.sectoral_config['right'].power_dbm = SECTORAL_ANTENNA_CONFIG['right']['power_dbm']
                
                print(f"  Sector orientations relative to vehicle:")
                print(f"    FRONT: {vehicle_heading:.1f}° (vehicle heading direction)")
                print(f"    RIGHT: {(vehicle_heading + 90) % 360:.1f}°")
                print(f"    REAR:  {(vehicle_heading + 180) % 360:.1f}°") 
                print(f"    LEFT:  {(vehicle_heading + 270) % 360:.1f}°")
        
        potential_neighbors = [data for data in self.mobility_data 
                              if data['time'] == current_time and data['id'] != vehicle_id and data['id'] in self.vehicles]
        
        for neighbor_data in potential_neighbors:
            other_pos = (neighbor_data['x'], neighbor_data['y'])
            distance = self._calculate_distance(vehicle_pos, other_pos)
            
            # Calculate angle to neighbor using atan2 (0°=East, 90°=North, 180°=West, 270°=South)
            angle_to_neighbor_atan2 = math.degrees(math.atan2(
                neighbor_data['y'] - vehicle_data['y'],
                neighbor_data['x'] - vehicle_data['x']
            )) % 360
            
            # Convert from atan2 coordinate system to FCD coordinate system
            # atan2: 0°=East → FCD: 0°=North
            angle_to_neighbor_fcd = (90 - angle_to_neighbor_atan2) % 360
            
            # Calculate effective transmission power
            if ANTENNA_TYPE == "SECTORAL":
                # Calculate relative angle from vehicle heading to neighbor
                relative_angle = (angle_to_neighbor_fcd - vehicle_heading + 360) % 360
                
                # Determine geometrically correct sector based on relative angle
                if relative_angle <= 45 or relative_angle > 315:
                    geometrical_sector = AntennaDirection.FRONT
                elif 45 < relative_angle <= 135:
                    geometrical_sector = AntennaDirection.RIGHT
                elif 135 < relative_angle <= 225:
                    geometrical_sector = AntennaDirection.REAR
                else:  # 225 < relative_angle <= 315
                    geometrical_sector = AntennaDirection.LEFT
                
                # Calculate gain for geometrical sector
                sector_config = antenna_config.sectoral_config[geometrical_sector.value]
                if sector_config.enabled:
                    # Calculate sector center angle in FCD coordinate system
                    sector_relative_angles = {
                        AntennaDirection.FRONT: 0,     # Same as vehicle heading
                        AntennaDirection.RIGHT: 90,    # +90° from vehicle heading
                        AntennaDirection.REAR: 180,    # +180° from vehicle heading
                        AntennaDirection.LEFT: 270     # +270° from vehicle heading
                    }
                    sector_relative_angle = sector_relative_angles[geometrical_sector]
                    sector_absolute_angle = (vehicle_heading + sector_relative_angle) % 360
                    
                    # Calculate angle difference between neighbor and sector center
                    angle_diff = abs(((angle_to_neighbor_fcd - sector_absolute_angle + 180) % 360) - 180)
                    
                    # FIXED: Realistic beamwidth-based gain calculation
                    half_beamwidth = sector_config.beamwidth_deg / 2  # 30° for 60° beamwidth
                    
                    if angle_diff <= half_beamwidth:
                        # Within main beam - full gain
                        gain_factor = 1.0
                    elif angle_diff <= half_beamwidth * 1.5:  # Up to 45°
                        # First sidelobe - gradual reduction
                        excess_angle = angle_diff - half_beamwidth
                        gain_factor = 1.0 - (excess_angle / (half_beamwidth * 0.5)) * 0.15  # Only 15% reduction
                    elif angle_diff <= half_beamwidth * 2.5:  # Up to 75°
                        # Extended coverage - still useful gain
                        gain_factor = 0.85 - ((angle_diff - half_beamwidth * 1.5) / (half_beamwidth)) * 0.35  # Reduce to 50%
                    elif angle_diff <= half_beamwidth * 3.0:  # Up to 90°
                        # Far sidelobe - minimal but non-zero gain
                        gain_factor = 0.5 - ((angle_diff - half_beamwidth * 2.5) / (half_beamwidth * 0.5)) * 0.3  # Reduce to 20%
                    else:
                        # Back lobe - minimal gain but not zero
                        gain_factor = 0.2  # 20% of full gain
                    
                    antenna_gain = sector_config.gain_db * gain_factor
                    sector_power = antenna_system.sector_powers[geometrical_sector.value]
                    effective_power = sector_power + antenna_gain
                    best_sector = geometrical_sector
                    
                    # Debug output for first few neighbors
                    if debug_enabled and debug_neighbor_count < 5:
                        print(f"\n  Neighbor {neighbor_data['id']} (distance: {distance:.1f}m):")
                        print(f"    Angle to neighbor (atan2): {angle_to_neighbor_atan2:.1f}°")
                        print(f"    Angle to neighbor (FCD): {angle_to_neighbor_fcd:.1f}°")
                        print(f"    Relative angle: {relative_angle:.1f}°")
                        print(f"    Geometrical sector: {geometrical_sector.value}")
                        print(f"    Sector center angle: {sector_absolute_angle:.1f}°")
                        print(f"    Angle difference: {angle_diff:.1f}°")
                        print(f"    Half beamwidth: {half_beamwidth:.1f}°")
                        print(f"    Gain factor: {gain_factor:.2f}")
                        print(f"    Sector power: {sector_power:.1f} dBm")
                        print(f"    Antenna gain: {antenna_gain:.1f} dB")
                        print(f"    Effective power: {effective_power:.1f} dBm")
                else:
                    effective_power = -100
                    best_sector = geometrical_sector
                    if debug_enabled and debug_neighbor_count < 5:
                        print(f"\n  Neighbor {neighbor_data['id']}: Sector {geometrical_sector.value} disabled")
            else:
                # Omnidirectional antenna
                effective_power = (OMNIDIRECTIONAL_ANTENNA_CONFIG["power_dbm"] + 
                                 OMNIDIRECTIONAL_ANTENNA_CONFIG["gain_db"])
                best_sector = None
                
                if debug_enabled and debug_neighbor_count < 5:
                    print(f"\n  Neighbor {neighbor_data['id']} (distance: {distance:.1f}m):")
                    print(f"    Angle to neighbor: {angle_to_neighbor_fcd:.1f}°")
                    print(f"    Best sector: None")
                    print(f"    Effective power: {effective_power:.1f} dBm")
            
            # Debug comparison for sectoral antennas
            if debug_enabled and debug_neighbor_count < 5:
                omni_power = OMNIDIRECTIONAL_ANTENNA_CONFIG["power_dbm"]
                omni_gain = OMNIDIRECTIONAL_ANTENNA_CONFIG["gain_db"]
                omni_total = omni_power + omni_gain
                print(f"    Omni equivalent: {omni_total:.1f} dBm ({omni_power:.1f} + {omni_gain:.1f})")
                if ANTENNA_TYPE == "SECTORAL":
                    power_advantage = effective_power - omni_total
                    print(f"    Power advantage: {power_advantage:+.1f} dB")
            
            # Calculate communication range with FIXED bounds checking
            if ANTENNA_TYPE == "SECTORAL":
                if best_sector and sector_config.enabled:
                    # Calculate range with FIXED bounds
                    omni_effective_power = (OMNIDIRECTIONAL_ANTENNA_CONFIG["power_dbm"] + 
                                          OMNIDIRECTIONAL_ANTENNA_CONFIG["gain_db"])
                    power_difference_db = effective_power - omni_effective_power
                    
                    # FIXED: Limit power difference to reasonable bounds
                    power_difference_db = max(-6.0, min(power_difference_db, 8.0))  # Between -6 dB and +8 dB
                    
                    # Convert power difference to range multiplier
                    range_multiplier = 10**(power_difference_db / 20.0)
                    
                    # FIXED: Ensure minimum reasonable range
                    min_range_multiplier = 0.7  # Never less than 70% of omnidirectional range
                    max_range_multiplier = 2.0  # Never more than 200% of omnidirectional range
                    range_multiplier = max(min_range_multiplier, min(max_range_multiplier, range_multiplier))
                    
                    directional_range = base_comm_range * range_multiplier
                    
                    if debug_enabled and debug_neighbor_count < 5:
                        print(f"    Power difference: {effective_power - omni_effective_power:.1f} dB (bounded to {power_difference_db:.1f} dB)")
                        print(f"    Range multiplier: {range_multiplier:.2f}")
                        print(f"    Directional range: {directional_range:.1f}m")
                else:
                    continue
            else:
                # Omnidirectional antenna
                directional_range = base_comm_range
                if debug_enabled and debug_neighbor_count < 5:
                    print(f"    Omnidirectional range: {directional_range:.1f}m")
            
            # Check if neighbor is within range
            if distance <= directional_range:
                other_vehicle = self.vehicles[neighbor_data['id']]
                
                # Calculate received power
                rx_power_dbm = self.interference_calculator.calculate_received_power_dbm(
                    effective_power,
                    distance,
                    0,  # Don't add gain again
                    2.15,  # Receiver gain
                    self.config.channel_model
                )
                
                if debug_enabled and debug_neighbor_count < 5:
                    print(f"    RX power before penalties: {rx_power_dbm:.1f} dBm")
                
                link_quality_threshold_dbm = -85
                link_quality_ok = True
                
                # Distance-based availability
                if distance > directional_range * 0.8:
                    link_availability = max(0.3, 1.0 - (distance - directional_range * 0.8) / (directional_range * 0.2))
                    if random.random() > link_availability:
                        link_quality_ok = False
                        if debug_enabled and debug_neighbor_count < 5:
                            print(f"    FAILED: Distance availability check")
                
                # Channel model penalties
                nlos_penalty = 0
                if self.config.channel_model in ['highway_nlos', 'urban_crossing_nlos']:
                    nlos_penalty = abs(random.gauss(0, 3.0))
                    rx_power_dbm -= nlos_penalty
                    if debug_enabled and debug_neighbor_count < 5:
                        print(f"    NLOS penalty: -{nlos_penalty:.1f} dB")
                
                # FIXED: Minimal off-axis penalty for sectoral antennas
                off_axis_penalty = 0
                if ANTENNA_TYPE == "SECTORAL" and best_sector:
                    # Only apply penalty if significantly outside beamwidth
                    beam_edge = sector_config.beamwidth_deg / 2
                    if angle_diff > beam_edge * 1.5:  # 50% beyond beam edge
                        excess_angle = angle_diff - (beam_edge * 1.5)
                        off_axis_penalty = excess_angle * 0.01  # Very small penalty: 0.01 dB per degree
                        rx_power_dbm -= off_axis_penalty
                        if debug_enabled and debug_neighbor_count < 5:
                            print(f"    Minimal off-axis penalty: -{off_axis_penalty:.1f} dB (angle diff: {angle_diff:.1f}°)")
                    elif debug_enabled and debug_neighbor_count < 5:
                        print(f"    No off-axis penalty (angle diff: {angle_diff:.1f}° within extended beam)")
                
                if rx_power_dbm < link_quality_threshold_dbm:
                    link_quality_ok = False
                    if debug_enabled and debug_neighbor_count < 5:
                        print(f"    FAILED: RX power {rx_power_dbm:.1f} < threshold {link_quality_threshold_dbm} dBm")
                
                if link_quality_ok:
                    # Determine which sector this neighbor is in relative to vehicle
                    neighbor_sector = best_sector.value if best_sector else 'omnidirectional'
                    
                    neighbor_info = {
                        'id': neighbor_data['id'],
                        'distance': distance,
                        'angle_to_neighbor': angle_to_neighbor_fcd,  # Use FCD coordinate system
                        'relative_angle': relative_angle if ANTENNA_TYPE == "SECTORAL" else 0,
                        'vehicle_heading': vehicle_heading,
                        'neighbor_sector': neighbor_sector,
                        'tx_power': effective_power,
                        'antenna_sector': best_sector.value if best_sector else 'omnidirectional',
                        'beacon_rate': other_vehicle.beacon_rate,
                        'mcs': other_vehicle.mcs,
                        'rx_power_dbm': rx_power_dbm,
                        'link_quality': 'good' if rx_power_dbm > -75 else 'marginal',
                        'position': other_pos,
                        'directional_range': directional_range
                    }
                    
                    neighbors.append(neighbor_info)
                    
                    if debug_enabled and debug_neighbor_count < 5:
                        print(f"    ACCEPTED: Added to neighbors list")
                        print(f"    Final RX power: {rx_power_dbm:.1f} dBm")
                
                debug_neighbor_count += 1
                if debug_neighbor_count >= 5:
                    debug_enabled = False
        
        # Sort by distance
        neighbors.sort(key=lambda x: x['distance'])
        
        # FIXED: Final debug summary with CORRECT gain values
        if (vehicle_id in ["veh0", "veh1"]) and (int(current_time) % 100 == 0):
            print(f"\n  SUMMARY: Found {len(neighbors)} neighbors")
            if neighbors:
                avg_rx_power = sum(n['rx_power_dbm'] for n in neighbors) / len(neighbors)
                print(f"  Average RX power: {avg_rx_power:.1f} dBm")
                
                if ANTENNA_TYPE == "SECTORAL":
                    # Count neighbors per sector
                    sector_counts = {}
                    for neighbor in neighbors:
                        sector = neighbor['neighbor_sector']
                        sector_counts[sector] = sector_counts.get(sector, 0) + 1
                    
                    print(f"  Neighbors per sector: {sector_counts}")
                    
                    # FIXED: Use actual gain values from configuration
                    front_power = antenna_system.sector_powers.get('front', 0)
                    rear_power = antenna_system.sector_powers.get('rear', 0)
                    front_gain = antenna_config.sectoral_config['front'].gain_db  # Use actual config gain
                    rear_gain = antenna_config.sectoral_config['rear'].gain_db    # Use actual config gain
                    
                    expected_front_eirp = front_power + front_gain
                    expected_rear_eirp = rear_power + rear_gain
                    omni_eirp = OMNIDIRECTIONAL_ANTENNA_CONFIG["power_dbm"] + OMNIDIRECTIONAL_ANTENNA_CONFIG["gain_db"]
                    
                    print(f"  Expected Front EIRP: {expected_front_eirp:.1f} dBm (advantage: {expected_front_eirp - omni_eirp:+.1f} dB)")
                    print(f"  Expected Rear EIRP: {expected_rear_eirp:.1f} dBm (advantage: {expected_rear_eirp - omni_eirp:+.1f} dB)")
            print()
        
        return neighbors
    
    def _generate_application_packets(self, vehicle_id: str, current_time: float) -> List[NetworkPacket]:
        """Generate application packets for vehicle"""
        if not self.config.enable_packet_simulation:
            return []
        
        packets = []
        vehicle = self.vehicles[vehicle_id]
        
        # Determine packet generation based on rate and time
        packets_to_generate = np.random.poisson(self.config.packet_generation_rate * self.config.time_step)
        
        for _ in range(packets_to_generate):
            # Select application type
            app_type = random.choice(self.config.application_types)
            
            # Select QoS class based on application
            if app_type == "SAFETY":
                qos_class = random.choice([QoSClass.EMERGENCY, QoSClass.SAFETY])
            elif app_type == "INFOTAINMENT":
                qos_class = random.choice([QoSClass.SERVICE, QoSClass.BACKGROUND])
            else:  # SENSING
                qos_class = QoSClass.SERVICE
            
            # Select destination (random neighbor or broadcast)
            if vehicle.neighbors and random.random() > 0.1:  # 90% unicast
                destination = random.choice(vehicle.neighbors)['id']
                dest_ip = self.vehicles[destination].ip_address
            else:  # 10% broadcast
                destination = "BROADCAST"
                dest_ip = "255.255.255.255"
            
            # Create packet
            packet = NetworkPacket(
                packet_id=f"{vehicle_id}_{vehicle.packet_counters['generated']}_{current_time}",
                packet_type=PacketType.DATA,
                source_id=vehicle_id,
                destination_id=destination,
                source_ip=vehicle.ip_address,
                destination_ip=dest_ip,
                payload_size=self.config.packet_size_bytes,
                qos_class=qos_class,
                application_type=app_type,
                creation_time=current_time,
                sequence_number=vehicle.packet_counters['generated']
            )
            
            packets.append(packet)
            vehicle.packet_counters['generated'] += 1
            vehicle.application_traffic[app_type].append(packet)
        
        return packets
    
    def _process_layer3_routing(self, vehicle_id: str, packet: NetworkPacket, current_time: float) -> Optional[str]:
        """Enhanced Layer 3 routing with comprehensive error handling"""
        if not self.config.enable_layer3 or packet.destination_id == "BROADCAST":
            return None
        
        # Add null checks
        if not vehicle_id or vehicle_id not in self.vehicles or not packet:
            return None
        
        try:
            vehicle = self.vehicles[vehicle_id]
            routing_protocol = vehicle.routing_protocol
            
            if not routing_protocol:
                return None
            
            # Update routing protocol state with error handling
            try:
                self._update_routing_protocol_state(vehicle, current_time)
            except Exception as e:
                print(f"[ERROR] Failed to update routing protocol state for {vehicle_id}: {e}")
                # Continue with routing even if state update fails
            
            # Check for direct neighbor delivery
            if packet.destination_id in [n['id'] for n in vehicle.neighbors]:
                # Direct delivery to neighbor
                vehicle.l3_metrics['direct_deliveries'] = vehicle.l3_metrics.get('direct_deliveries', 0) + 1
                return packet.destination_id
            
            # Route lookup based on protocol type
            next_hop = None
            
            try:
                if isinstance(routing_protocol, AODVRoutingProtocol):
                    next_hop = routing_protocol.get_next_hop(packet.destination_id, current_time)
                    
                    if not next_hop:
                        # Initiate route discovery
                        rreq_packet = routing_protocol.initiate_route_discovery(packet.destination_id)
                        if rreq_packet:
                            vehicle.l3_metrics['route_discovery_attempts'] += 1
                            # Buffer the original packet
                            routing_protocol.buffer_packet(packet)
                            # Broadcast RREQ (simulation)
                            self._broadcast_routing_packet(vehicle_id, rreq_packet, current_time)
                            # Return None to indicate packet is buffered
                            return "BUFFERED"
                
                elif isinstance(routing_protocol, OLSRRoutingProtocol):
                    next_hop = routing_protocol.get_next_hop(packet.destination_id, current_time)
                    
                    # No route discovery needed in OLSR - routes are proactively maintained
                    if not next_hop:
                        vehicle.l3_metrics['route_failures'] = vehicle.l3_metrics.get('route_failures', 0) + 1
                
                elif isinstance(routing_protocol, GeographicRoutingProtocol):
                    # Update our position in the routing protocol
                    vehicle_data = next((data for data in self.mobility_data 
                                       if data['time'] == current_time and data['id'] == vehicle_id), None)
                    if vehicle_data:
                        routing_protocol.update_position(
                            vehicle_id, vehicle_data['x'], vehicle_data['y'],
                            vehicle_data.get('speed', 0), vehicle_data.get('angle', 0)
                        )
                        
                        # Update neighbor positions
                        for neighbor in vehicle.neighbors:
                            neighbor_data = next((data for data in self.mobility_data 
                                                if data['time'] == current_time and data['id'] == neighbor['id']), None)
                            if neighbor_data:
                                routing_protocol.update_position(
                                    neighbor['id'], neighbor_data['x'], neighbor_data['y'],
                                    neighbor_data.get('speed', 0), neighbor_data.get('angle', 0)
                                )
                        
                        # Get next hop using geographic forwarding
                        current_pos = (vehicle_data['x'], vehicle_data['y'])
                        next_hop = routing_protocol.get_next_hop_geographic(
                            packet.destination_id, current_pos, packet.packet_id, current_time
                        )
                        
            except Exception as e:
                print(f"[ERROR] Failed to find next hop for {vehicle_id} to {packet.destination_id}: {e}")
                return None
            
            # Update routing metrics
            if next_hop and next_hop != "BUFFERED":
                vehicle.l3_metrics['successful_routes'] = vehicle.l3_metrics.get('successful_routes', 0) + 1
                
                # Update packet route information
                if not hasattr(packet, 'route') or not packet.route:
                    packet.route = [vehicle_id]
                if vehicle_id not in packet.route:
                    packet.route.append(vehicle_id)
                
                packet.hop_count = len(packet.route) - 1
                
                # Update hop count distribution
                if 'hop_count_distribution' not in vehicle.l3_metrics:
                    vehicle.l3_metrics['hop_count_distribution'] = []
                vehicle.l3_metrics['hop_count_distribution'].append(packet.hop_count)
            
            return next_hop
            
        except Exception as e:
            print(f"[ERROR] Critical error in L3 routing for {vehicle_id}: {e}")
            return None
    
    def _update_routing_protocol_state(self, vehicle: 'VehicleState', current_time: float):
        """Fixed routing protocol state update with unified cleanup calls"""
        if not vehicle.routing_protocol:
            return
        
        # Generate and process routing protocol messages
        if isinstance(vehicle.routing_protocol, AODVRoutingProtocol):
            # Generate our own HELLO packet if needed
            hello_packet = vehicle.routing_protocol.generate_hello_packet(current_time)
            if hello_packet:
                self._broadcast_routing_packet(vehicle.vehicle_id, hello_packet, current_time)
            
            # Process simulated HELLO packets FROM neighbors (not our own HELLO TO neighbors)
            for neighbor in vehicle.neighbors:
                # Create simulated HELLO packet from each neighbor
                neighbor_hello = self._create_simulated_hello(neighbor['id'], current_time)
                if neighbor_hello:  # Add null check
                    vehicle.routing_protocol.process_hello_packet(neighbor_hello, neighbor['id'], current_time)
            
            # FIXED: Use correct cleanup method
            vehicle.routing_protocol.cleanup_expired_routes(current_time)
        
        elif isinstance(vehicle.routing_protocol, OLSRRoutingProtocol):
            # Generate our own HELLO packet
            hello_packet = vehicle.routing_protocol.generate_hello_packet(current_time)
            if hello_packet:
                self._broadcast_routing_packet(vehicle.vehicle_id, hello_packet, current_time)
            
            # Process simulated HELLO packets from neighbors
            for neighbor in vehicle.neighbors:
                neighbor_hello = self._create_simulated_olsr_hello(neighbor['id'], current_time)
                if neighbor_hello:  # Add null check
                    vehicle.routing_protocol.process_hello_packet(neighbor_hello, neighbor['id'], current_time)
            
            # Generate TC packet if we are an MPR
            tc_packet = vehicle.routing_protocol.generate_tc_packet(current_time)
            if tc_packet:
                self._broadcast_routing_packet(vehicle.vehicle_id, tc_packet, current_time)
            
            # FIXED: Use correct cleanup method
            vehicle.routing_protocol.cleanup_expired_entries(current_time)
        
        elif isinstance(vehicle.routing_protocol, GeographicRoutingProtocol):
            # Generate position beacon
            vehicle_data = next((data for data in self.mobility_data 
                               if data['time'] == current_time and data['id'] == vehicle.vehicle_id), None)
            if vehicle_data:
                current_pos = (vehicle_data['x'], vehicle_data['y'])
                beacon = vehicle.routing_protocol.generate_position_beacon(
                    current_pos, current_time, 
                    vehicle_data.get('speed', 0), vehicle_data.get('angle', 0)
                )
                if beacon:
                    self._broadcast_routing_packet(vehicle.vehicle_id, beacon, current_time)
                
                # Process position beacons from neighbors
                for neighbor in vehicle.neighbors:
                    neighbor_data = next((data for data in self.mobility_data 
                                        if data['time'] == current_time and data['id'] == neighbor['id']), None)
                    if neighbor_data:
                        neighbor_beacon = self._create_simulated_position_beacon(
                            neighbor['id'], neighbor_data, current_time
                        )
                        if neighbor_beacon:  # Add null check
                            vehicle.routing_protocol.process_position_beacon(neighbor_beacon, neighbor['id'], current_time)
            
            # FIXED: Use correct cleanup method
            vehicle.routing_protocol.cleanup_expired_positions(current_time)
    
    def _broadcast_routing_packet(self, vehicle_id: str, packet: NetworkPacket, current_time: float):
        """Simulate broadcasting routing protocol packet to neighbors with error handling"""
        # Add null checks
        if not vehicle_id or not packet or vehicle_id not in self.vehicles:
            return
        
        try:
            vehicle = self.vehicles[vehicle_id]
            
            # Simulate packet transmission to all neighbors
            for neighbor in vehicle.neighbors:
                neighbor_id = neighbor.get('id')
                if neighbor_id and neighbor_id in self.vehicles:
                    neighbor_vehicle = self.vehicles[neighbor_id]
                    
                    # Simulate packet reception and processing
                    if hasattr(neighbor_vehicle, 'routing_protocol') and neighbor_vehicle.routing_protocol:
                        self._process_received_routing_packet(neighbor_vehicle, packet, vehicle_id, current_time)
                        
        except Exception as e:
            print(f"[ERROR] Failed to broadcast routing packet from {vehicle_id}: {e}")
    
    def _process_received_routing_packet(self, vehicle: 'VehicleState', packet: NetworkPacket, 
                                   sender: str, current_time: float):
        """Process received routing protocol packet with null checks"""
        # Add null checks
        if not vehicle or not packet or not sender:
            return
        
        routing_protocol = vehicle.routing_protocol
        if not routing_protocol:
            return
        
        try:
            if packet.packet_type == PacketType.HELLO:
                if isinstance(routing_protocol, AODVRoutingProtocol):
                    routing_protocol.process_hello_packet(packet, sender, current_time)
                elif isinstance(routing_protocol, OLSRRoutingProtocol):
                    routing_protocol.process_hello_packet(packet, sender, current_time)
                elif isinstance(routing_protocol, GeographicRoutingProtocol):
                    routing_protocol.process_position_beacon(packet, sender, current_time)
            
            elif packet.packet_type == PacketType.RREQ:
                if isinstance(routing_protocol, AODVRoutingProtocol):
                    rrep = routing_protocol.process_rreq(packet, sender, current_time)
                    if rrep:
                        # Send RREP back (simulation)
                        self._unicast_routing_packet(vehicle.vehicle_id, sender, rrep, current_time)
            
            elif packet.packet_type == PacketType.RREP:
                if isinstance(routing_protocol, AODVRoutingProtocol):
                    should_forward = routing_protocol.process_rrep(packet, sender, current_time)
                    if should_forward and packet.destination_id != vehicle.vehicle_id:
                        # Forward RREP toward destination
                        next_hop = routing_protocol.get_next_hop(packet.destination_id, current_time)
                        if next_hop:
                            self._unicast_routing_packet(vehicle.vehicle_id, next_hop, packet, current_time)
            
            elif packet.packet_type == PacketType.TC:
                if isinstance(routing_protocol, OLSRRoutingProtocol):
                    should_forward = routing_protocol.process_tc_packet(packet, sender, current_time)
                    if should_forward:
                        # Forward TC to other neighbors
                        self._broadcast_routing_packet(vehicle.vehicle_id, packet, current_time)
                        
        except Exception as e:
            print(f"[ERROR] Failed to process routing packet from {sender} to {vehicle.vehicle_id}: {e}")
    
    def _unicast_routing_packet(self, sender_id: str, receiver_id: str, 
                          packet: NetworkPacket, current_time: float):
        """Simulate unicast routing packet transmission with error handling"""
        # Add null checks
        if not sender_id or not receiver_id or not packet:
            return
        
        try:
            if receiver_id in self.vehicles:
                receiver_vehicle = self.vehicles[receiver_id]
                if hasattr(receiver_vehicle, 'routing_protocol') and receiver_vehicle.routing_protocol:
                    self._process_received_routing_packet(receiver_vehicle, packet, sender_id, current_time)
                    
        except Exception as e:
            print(f"[ERROR] Failed to unicast routing packet from {sender_id} to {receiver_id}: {e}")
    
    def _create_simulated_hello(self, node_id: str, current_time: float) -> Optional[NetworkPacket]:
        """Create simulated HELLO packet for AODV protocol testing with null safety"""
        try:
            return NetworkPacket(
                packet_id=f"SIM_HELLO_{node_id}_{current_time}",
                packet_type=PacketType.HELLO,
                source_id=node_id,
                destination_id="BROADCAST",
                source_ip=f"192.168.1.{hash(node_id) % 254 + 1}",
                destination_ip="255.255.255.255",
                payload_size=32,
                qos_class=QoSClass.SERVICE,
                application_type="ROUTING",
                ttl=1,
                sequence_number=random.randint(1, 65535),
                creation_time=current_time
            )
        except Exception as e:
            print(f"[ERROR] Failed to create simulated HELLO for {node_id}: {e}")
            return None
    
    def _create_simulated_olsr_hello(self, node_id: str, current_time: float) -> Optional[NetworkPacket]:
        """Create simulated OLSR HELLO packet with neighbor information"""
        try:
            # Create basic HELLO packet
            hello_packet = NetworkPacket(
                packet_id=f"SIM_OLSR_HELLO_{node_id}_{current_time}",
                packet_type=PacketType.HELLO,
                source_id=node_id,
                destination_id="BROADCAST",
                source_ip=f"192.168.1.{hash(node_id) % 254 + 1}",
                destination_ip="255.255.255.255",
                payload_size=64,
                qos_class=QoSClass.SERVICE,
                application_type="ROUTING",
                ttl=1,
                originator_addr=node_id,
                sequence_number=random.randint(1, 65535),
                creation_time=current_time
            )
            
            # Add simulated neighbor information
            neighbor_info = {}
            if node_id in self.vehicles:
                vehicle = self.vehicles[node_id]
                for neighbor in vehicle.neighbors[:3]:  # Limit to 3 neighbors for simulation
                    neighbor_info[neighbor['id']] = {
                        'link_type': 'SYM',
                        'neighbor_type': 'NOT_MPR'
                    }
            
            hello_packet.neighbor_info = neighbor_info
            return hello_packet
            
        except Exception as e:
            print(f"[ERROR] Failed to create simulated OLSR HELLO for {node_id}: {e}")
            return None

    def _create_simulated_position_beacon(self, node_id: str, node_data: Dict, current_time: float) -> Optional[NetworkPacket]:
        """Create simulated position beacon for geographic routing"""
        try:
            beacon_packet = NetworkPacket(
                packet_id=f"SIM_POS_BEACON_{node_id}_{current_time}",
                packet_type=PacketType.HELLO,
                source_id=node_id,
                destination_id="BROADCAST",
                source_ip=f"192.168.1.{hash(node_id) % 254 + 1}",
                destination_ip="255.255.255.255",
                payload_size=48,
                qos_class=QoSClass.SERVICE,
                application_type="ROUTING",
                ttl=1,
                creation_time=current_time
            )
            
            # Add position information
            beacon_packet.position_info = {
                'x': node_data.get('x', 0.0),
                'y': node_data.get('y', 0.0),
                'speed': node_data.get('speed', 0.0),
                'heading': node_data.get('angle', 0.0),
                'timestamp': current_time
            }
            
            return beacon_packet
            
        except Exception as e:
            print(f"[ERROR] Failed to create simulated position beacon for {node_id}: {e}")
            return None
    
    def _process_sdn_flow_matching(self, vehicle_id: str, packet: NetworkPacket) -> Optional[List[Dict]]:
        """Process SDN flow matching with proper differentiation from L3 routing"""
        if not self.config.enable_sdn:
            return None
        
        vehicle = self.vehicles[vehicle_id]
        
        # SDN-specific processing delay
        sdn_processing_start = time.time()
        
        # Check flow table for matching entry
        matching_flow = None
        for flow_id, flow_entry in vehicle.flow_table.items():
            match_fields = flow_entry.match_fields
            
            # Enhanced matching with SDN-specific criteria
            match = True
            if 'destination' in match_fields and match_fields['destination'] != packet.destination_id:
                match = False
            if 'flow_id' in match_fields and match_fields['flow_id'] != packet.flow_id:
                match = False
            if 'qos_class' in match_fields and match_fields['qos_class'] != packet.qos_class.name:
                match = False
            if 'application_type' in match_fields and match_fields['application_type'] != packet.application_type:
                match = False
            
            if match and flow_entry.state == FlowState.ACTIVE:
                matching_flow = flow_entry
                break
        
        if matching_flow:
            # Flow hit - update statistics and return actions
            matching_flow.packet_count += 1
            matching_flow.byte_count += packet.payload_size
            matching_flow.last_used = time.time()
            
            # Add SDN-specific processing delay even for flow hits
            sdn_processing_delay = time.time() - sdn_processing_start + 0.0001  # 0.1ms minimum
            vehicle.sdn_metrics['flow_processing_times'] = vehicle.sdn_metrics.get('flow_processing_times', [])
            vehicle.sdn_metrics['flow_processing_times'].append(sdn_processing_delay)
            
            return matching_flow.actions
        
        # Flow miss - send packet-in to controller
        if self.sdn_controller:
            vehicle.pending_packet_ins.append(packet)
            vehicle.sdn_metrics['packet_in_count'] += 1
            
            # Controller communication overhead
            controller_comm_start = time.time()
            
            # Request flow rules from controller
            flow_rules = self.sdn_controller.handle_packet_in(packet, vehicle_id)
            
            controller_comm_delay = time.time() - controller_comm_start
            vehicle.sdn_metrics['controller_latency'] = vehicle.sdn_metrics.get('controller_latency', [])
            vehicle.sdn_metrics['controller_latency'].append(controller_comm_delay)
            
            # Install new flow rules
            for node_id, flow_entry in flow_rules:
                if node_id == vehicle_id:
                    vehicle.flow_table[flow_entry.flow_id] = flow_entry
                    vehicle.sdn_metrics['flow_mod_count'] += 1
                    
                    # Return actions from newly installed flow
                    return flow_entry.actions
            
            # If no flow rules were installed for this vehicle, use default SDN behavior
            # This is different from L3 routing - SDN should drop or use default rules
            return [{'type': 'default_forward', 'method': 'sdn_flooding'}]
        
        return None
    
    def _calculate_packet_transmission_time(self, packet_size_bytes: int, mcs: int) -> float:
        """Calculate actual packet transmission time including PHY overhead"""
        # Total packet size including MAC header
        total_bits = (packet_size_bytes + self.config.mac_header_bytes) * 8
        
        # Get data rate for current MCS (in Mbps, convert to bps)
        data_rate_mbps = self.ieee_mapper.data_rates.get(mcs, 3.0)
        data_rate_bps = data_rate_mbps * 1e6
        
        # IEEE 802.11bd PHY overhead
        preamble_duration = 40e-6  # 40μs preamble
        phy_header_duration = 8e-6  # 8μs PHY header
        
        # Data transmission time
        data_duration = total_bits / data_rate_bps
        
        # Total transmission time
        total_duration = preamble_duration + phy_header_duration + data_duration
        
        return total_duration
    
    def _calculate_cbr_realistic_dynamic(self, vehicle_id: str, neighbors: List[Dict]) -> float:
        """CBR calculation with enhanced neighbor sensitivity"""
        vehicle = self.vehicles[vehicle_id]
        num_neighbors = len(neighbors)
        
        # Base CBR from safety beacon traffic
        base_background = 0.02
        
        if num_neighbors == 0:
            actual_packet_duration = self._calculate_packet_transmission_time(
                self.config.payload_length, vehicle.mcs)
            own_occupancy = vehicle.beacon_rate * actual_packet_duration
            base_cbr = base_background + own_occupancy
        else:
            # Calculate channel occupancy from safety beacons
            total_channel_time = 0.0
            
            # Own beacon transmission time
            own_packet_duration = self._calculate_packet_transmission_time(
                self.config.payload_length, vehicle.mcs)
            own_occupancy = vehicle.beacon_rate * own_packet_duration
            total_channel_time += own_occupancy
            
            # Neighbor beacon contributions
            for neighbor in neighbors:
                neighbor_beacon_rate = neighbor.get('beacon_rate', 10.0)
                neighbor_mcs = neighbor.get('mcs', 1)
                distance = neighbor['distance']
                
                neighbor_packet_duration = self._calculate_packet_transmission_time(
                    self.config.payload_length, neighbor_mcs)
                
                # ENHANCED: More aggressive sensing probability
                if distance <= 100:
                    sensing_prob = 1.0
                elif distance <= 200:
                    sensing_prob = 0.95  # Increased from 0.9
                elif distance <= 300:
                    sensing_prob = 0.8   # Increased from 0.7
                else:
                    sensing_prob = 0.5   # Increased from 0.4
                
                neighbor_occupancy = (neighbor_beacon_rate * neighbor_packet_duration * 
                                    sensing_prob)
                total_channel_time += neighbor_occupancy
            
            # ENHANCED: More aggressive MAC overhead calculation
            if num_neighbors <= 5:
                mac_overhead = 1.1
            elif num_neighbors <= 10:
                mac_overhead = 1.15 + (num_neighbors - 5) * 0.03  # Increased from 0.02
            elif num_neighbors <= 15:
                mac_overhead = 1.3 + (num_neighbors - 10) * 0.04  # Increased
            else:
                mac_overhead = 1.5 + (num_neighbors - 15) * 0.05  # Increased
            
            # ENHANCED: Slightly more aggressive factors
            hidden_terminal_factor = 1.0 + self.config.hidden_node_factor * (num_neighbors ** 0.75)  # Increased from 0.7
            inter_system_factor = 1.0 + self.config.inter_system_interference * 0.6  # Increased from 0.5
            
            raw_offered_load = (total_channel_time * mac_overhead * 
                               hidden_terminal_factor * inter_system_factor)
            
            # ENHANCED: More aggressive neighbor density scaling
            if num_neighbors <= 8:   # Reduced from 10
                min_neighbor_cbr = 0.05
            elif num_neighbors <= 15: # Reduced from 20
                min_neighbor_cbr = 0.18  # Increased from 0.15
            elif num_neighbors <= 25: # Reduced from 30
                min_neighbor_cbr = 0.4   # Increased from 0.35
            else:
                min_neighbor_cbr = 0.7   # Increased from 0.60
            
            base_cbr = base_background + max(min_neighbor_cbr, raw_offered_load)
        
        # Add background traffic contribution (unchanged)
        if hasattr(self, 'background_traffic_manager') and self.background_traffic_manager:
            offered_load_with_bg = self.background_traffic_manager.get_effective_cbr_contribution(vehicle_id, base_cbr)
        else:
            background_factor = self.config.background_traffic_load * 0.2
            offered_load_with_bg = base_cbr + background_factor
        
        # Apply scaling factors
        packet_size_factor = (self.config.payload_length + self.config.mac_header_bytes) / 400.0
        packet_size_factor = max(0.7, min(2.0, packet_size_factor))
        offered_load_with_bg *= packet_size_factor
        
        if num_neighbors > 0:
            density_factor = 1.0 + (num_neighbors * 0.06)  # Increased from 0.05
            offered_load_with_bg *= density_factor
        
        vehicle.offered_load = max(base_background, offered_load_with_bg)
        actual_cbr = offered_load_with_bg
        
        return actual_cbr
    
    def _calculate_sinr(self, vehicle_id: str, neighbors: List[Dict]) -> float:
        """Calculate SINR using enhanced interference modeling"""
        vehicle = self.vehicles[vehicle_id]
        
        sinr_db = self.interference_calculator.calculate_sinr_with_interference(
            vehicle_id, 
            neighbors, 
            vehicle.transmission_power, 
            self.config.channel_model
        )
        
        return sinr_db
    
    def _get_failure_metrics(self) -> Dict:
        """Return failure metrics when SINR is too low"""
        return {
            'ber': 0.5, 'ser': 0.5, 'per': 0.99, 'per_phy': 0.99, 'pdr': 0.01,
            'phy_throughput': 0.0, 'mac_throughput': 0.0, 'throughput': 0.0,
            'latency': 10.0, 'collision_prob': 0.8, 'mac_efficiency': 0.01,
            'selected_mcs': 1,
            'phy_latency_ms': 5.0, 'mac_latency_ms': 5.0,
            'l3_routing_delay': 0.0, 'sdn_processing_delay': 0.0,
            'preamble_latency_ms': 0.04, 'data_tx_latency_ms': 1.0,
            'difs_latency_ms': 0.034, 'backoff_latency_ms': 2.0,
            'retry_latency_ms': 2.0, 'queue_latency_ms': 0.5
        }
    
    
    def _calculate_performance_metrics(self, vehicle_id: str, sinr_db: float, cbr: float, 
                         neighbors: List[Dict], channel_model: str = 'highway_los') -> Dict:
        """FIXED: Enhanced performance metrics with smoother transitions and offered load tracking"""
        
        vehicle = self.vehicles[vehicle_id]
        num_neighbors = len(neighbors)
        
        # MCS selection for range 0-9 with smoother transitions
        selected_mcs = 0
        for test_mcs in range(9, -1, -1):
            threshold = self.ieee_mapper.snr_thresholds[test_mcs]['success']
            # FIXED: More forgiving MCS selection (reduced margin from 1.0 to 0.5)
            if sinr_db >= threshold - 0.5:
                selected_mcs = test_mcs
                break
        
        # FIXED: Less harsh failure conditions - use marginal performance instead of complete failure
        if sinr_db < self.ieee_mapper.snr_thresholds[0]['success'] - 8.0:  # Changed from 5.0 to 8.0
            return self._get_failure_metrics()
        
        # Update vehicle MCS
        vehicle.mcs = selected_mcs
        
        # Calculate performance metrics using corrected methods
        packet_bits = (self.config.payload_length + self.config.mac_header_bytes) * 8
        
        ber = self.ieee_mapper.get_ber_from_sinr(sinr_db, selected_mcs, self.config.enable_ldpc)
        ser = self.ieee_mapper.get_ser_from_ber(ber, selected_mcs)
        per_phy = self.ieee_mapper.get_per_from_ser(ser, packet_bits, selected_mcs, self.config.enable_ldpc)
        
        collision_prob = self.ieee_mapper.get_cbr_collision_probability(cbr, num_neighbors)
        per_total = per_phy + collision_prob - (per_phy * collision_prob)
        per_total = min(0.999, max(1e-10, per_total))
        
        mac_efficiency = self.ieee_mapper.get_mac_efficiency(cbr, per_total, num_neighbors)
        
        # PHY throughput calculation (enhanced)
        data_rate_mbps = self.ieee_mapper.data_rates.get(selected_mcs, 3.0)
        data_rate_bps = data_rate_mbps * 1e6
        frame_efficiency = self.ieee_mapper.max_frame_efficiency.get(selected_mcs, 0.8)
        
        enhancement_factor = 1.0
        if self.config.enable_ldpc:
            enhancement_factor *= 1.05
        if self.config.enable_midambles:
            enhancement_factor *= 1.02
        
        phy_throughput = data_rate_bps * frame_efficiency * (1 - per_phy) * enhancement_factor
        final_throughput = phy_throughput * mac_efficiency
        
        # Calculate detailed latency components
        phy_components = self._calculate_phy_latency_components(selected_mcs, packet_bits)
        mac_components = self._calculate_mac_latency_components(cbr, per_total, num_neighbors, 
                                                              phy_components['total_phy_latency_ms']/1000)
        
        # Protocol-specific adjustments for L3/SDN
        l3_routing_delay = 0.0
        if self.config.enable_layer3 and vehicle.routing_protocol:
            if isinstance(vehicle.routing_protocol, AODVRoutingProtocol):
                route_exists = vehicle.routing_protocol.get_next_hop(vehicle_id)
                if route_exists:
                    l3_routing_delay = 0.000005  # 5 microseconds for route table lookup
                else:
                    l3_routing_delay = 0.002  # 2 ms for route discovery
            elif isinstance(vehicle.routing_protocol, OLSRRoutingProtocol):
                l3_routing_delay = 0.000002  # 2 microseconds for route table lookup
            elif isinstance(vehicle.routing_protocol, GeographicRoutingProtocol):
                l3_routing_delay = 0.000008  # 8 microseconds for geographic calculation
        
        sdn_delay = 0.0
        if self.config.enable_sdn:
            flow_lookup_delay = 0.000003  # 3 microseconds for flow table lookup
            sdn_delay += flow_lookup_delay
            
            if hasattr(vehicle, 'sdn_metrics') and 'controller_latency' in vehicle.sdn_metrics:
                recent_controller_latencies = vehicle.sdn_metrics['controller_latency'][-10:]
                if recent_controller_latencies:
                    avg_controller_latency = sum(recent_controller_latencies) / len(recent_controller_latencies)
                    sdn_delay += avg_controller_latency
            else:
                sdn_delay += 0.003  # 3 ms for controller communication
            
            # SDN traffic engineering benefit
            if self.sdn_controller and hasattr(self.sdn_controller, 'centralized_optimization_factor'):
                mac_efficiency *= self.sdn_controller.centralized_optimization_factor
        
        # Total latency including L3/SDN components
        total_latency_ms = (phy_components['total_phy_latency_ms'] + 
                           mac_components['total_mac_latency_ms'] +
                           l3_routing_delay * 1000 + sdn_delay * 1000)
        
        # FIXED: Store offered load information for analysis
        offered_load = getattr(vehicle, 'offered_load', cbr)
        congestion_ratio = offered_load / 1.0 if offered_load > 1.0 else 1.0
        
        return {
            'ber': ber,
            'ser': ser,
            'per': per_total,
            'per_phy': per_phy,
            'pdr': 1 - per_total,
            'phy_throughput': phy_throughput,
            'mac_throughput': final_throughput,
            'throughput': final_throughput,
            'latency': total_latency_ms,
            'collision_prob': collision_prob,
            'mac_efficiency': mac_efficiency,
            'selected_mcs': selected_mcs,
            
            # Detailed latency breakdown
            'phy_latency_ms': phy_components['total_phy_latency_ms'],
            'mac_latency_ms': mac_components['total_mac_latency_ms'],
            'l3_routing_delay': l3_routing_delay * 1000,
            'sdn_processing_delay': sdn_delay * 1000,
            'preamble_latency_ms': phy_components['preamble_latency_ms'],
            'data_tx_latency_ms': phy_components.get('data_transmission_latency_ms', 0),
            'difs_latency_ms': mac_components['difs_latency_ms'],
            'backoff_latency_ms': mac_components['backoff_latency_ms'],
            'retry_latency_ms': mac_components['retry_latency_ms'],
            'queue_latency_ms': mac_components['queue_latency_ms'],
            
            # NEW: Offered load and congestion tracking
            'offered_load': offered_load,
            'congestion_ratio': congestion_ratio,
            'channel_overload': 'Yes' if offered_load > 1.0 else 'No'
        }
    
    def _calculate_performance_metrics_rl_aware(self, vehicle_id: str, sinr_db: float, cbr: float, 
                 neighbors: List[Dict], channel_model: str = 'highway_los') -> Dict:
        """ENHANCED RL-aware performance metrics with stronger neighbor impact modeling"""
        
        vehicle = self.vehicles[vehicle_id]
        num_neighbors = len(neighbors)
        
        # MCS selection (only if RL is disabled)
        if not self.enable_rl:
            selected_mcs = 0
            for test_mcs in range(9, -1, -1):
                threshold = self.ieee_mapper.snr_thresholds[test_mcs]['success']
                if sinr_db >= threshold - 0.5:
                    selected_mcs = test_mcs
                    break
            vehicle.mcs = selected_mcs
        else:
            selected_mcs = vehicle.mcs
        
        # ENHANCED: More realistic failure conditions
        if sinr_db < self.ieee_mapper.snr_thresholds[0]['success'] - 6.0:  # Reduced from 8.0
            return self._get_failure_metrics()
        
        # Calculate performance metrics
        packet_bits = (self.config.payload_length + self.config.mac_header_bytes) * 8
        
        ber = self.ieee_mapper.get_ber_from_sinr(sinr_db, selected_mcs, self.config.enable_ldpc)
        ser = self.ieee_mapper.get_ser_from_ber(ber, selected_mcs)
        per_phy = self.ieee_mapper.get_per_from_ser(ser, packet_bits, selected_mcs, self.config.enable_ldpc)
    
        # ENHANCED: More aggressive PER combination with stronger neighbor impact
        collision_prob = self.ieee_mapper.get_cbr_collision_probability(cbr, num_neighbors)
        
        # Base PER combination
        per_total = per_phy + collision_prob - (per_phy * collision_prob)
        
        # ENHANCED: Stronger neighbor density penalty on PER (increased impact)
        if num_neighbors > 0:
            if num_neighbors <= 5:
                neighbor_per_penalty = num_neighbors * 0.002  # 0.2% per neighbor
            elif num_neighbors <= 10:
                neighbor_per_penalty = 0.01 + ((num_neighbors - 5) * 0.003)  # 0.3% per neighbor above 5
            elif num_neighbors <= 15:
                neighbor_per_penalty = 0.025 + ((num_neighbors - 10) * 0.004)  # 0.4% per neighbor above 10
            else:
                neighbor_per_penalty = 0.045 + ((num_neighbors - 15) * 0.005)  # 0.5% per neighbor above 15
            
            neighbor_per_penalty = min(0.08, neighbor_per_penalty)  # Cap at 8%
            per_total = per_total + neighbor_per_penalty - (per_total * neighbor_per_penalty)
        
        # ENHANCED: Stronger CBR impact on PER
        if cbr > 0.5:  # Lower threshold from 0.6
            cbr_per_penalty = (cbr - 0.5) * 0.12  # Increased from 0.1
            per_total = per_total + cbr_per_penalty - (per_total * cbr_per_penalty)
        
        # NEW: Additional interference penalty for very dense networks
        if num_neighbors > 12:
            interference_penalty = (num_neighbors - 12) * 0.003  # 0.3% per neighbor above 12
            interference_penalty = min(0.05, interference_penalty)  # Cap at 5%
            per_total = per_total + interference_penalty - (per_total * interference_penalty)
        
        per_total = min(0.999, max(1e-10, per_total))
        
        # ENHANCED: MAC efficiency with stronger neighbor impact
        mac_efficiency = self.ieee_mapper.get_mac_efficiency(cbr, per_total, num_neighbors)
        
        # PHY throughput calculation
        data_rate_mbps = self.ieee_mapper.data_rates.get(selected_mcs, 3.0)
        data_rate_bps = data_rate_mbps * 1e6
        frame_efficiency = self.ieee_mapper.max_frame_efficiency.get(selected_mcs, 0.8)
        
        enhancement_factor = 1.0
        if self.config.enable_ldpc:
            enhancement_factor *= 1.05
        if self.config.enable_midambles:
            enhancement_factor *= 1.02
        
        phy_throughput = data_rate_bps * frame_efficiency * (1 - per_phy) * enhancement_factor
        final_throughput = phy_throughput * mac_efficiency
        
        # ENHANCED: Throughput reduction due to neighbor contention
        if num_neighbors > 5:
            contention_factor = 1.0 - min(0.3, (num_neighbors - 5) * 0.02)  # Up to 30% reduction
            final_throughput *= contention_factor
        
        # Calculate detailed latency components
        phy_components = self._calculate_phy_latency_components(selected_mcs, packet_bits)
        mac_components = self._calculate_mac_latency_components(cbr, per_total, num_neighbors, 
                                                              phy_components['total_phy_latency_ms']/1000)
        
        # Protocol-specific adjustments
        l3_routing_delay = 0.0
        if self.config.enable_layer3 and vehicle.routing_protocol:
            if isinstance(vehicle.routing_protocol, AODVRoutingProtocol):
                route_exists = vehicle.routing_protocol.get_next_hop(vehicle_id)
                if route_exists:
                    l3_routing_delay = 0.000005
                else:
                    l3_routing_delay = 0.002
            elif isinstance(vehicle.routing_protocol, OLSRRoutingProtocol):
                l3_routing_delay = 0.000002
            elif isinstance(vehicle.routing_protocol, GeographicRoutingProtocol):
                l3_routing_delay = 0.000008
        
        sdn_delay = 0.0
        if self.config.enable_sdn:
            flow_lookup_delay = 0.000003
            sdn_delay += flow_lookup_delay
            
            if hasattr(vehicle, 'sdn_metrics') and 'controller_latency' in vehicle.sdn_metrics:
                recent_controller_latencies = vehicle.sdn_metrics['controller_latency'][-10:]
                if recent_controller_latencies:
                    avg_controller_latency = sum(recent_controller_latencies) / len(recent_controller_latencies)
                    sdn_delay += avg_controller_latency
            else:
                sdn_delay += 0.003
            
            if self.sdn_controller and hasattr(self.sdn_controller, 'centralized_optimization_factor'):
                mac_efficiency *= self.sdn_controller.centralized_optimization_factor
        
        # ENHANCED: Total latency with neighbor impact
        base_latency_ms = (phy_components['total_phy_latency_ms'] + 
                          mac_components['total_mac_latency_ms'] +
                          l3_routing_delay * 1000 + sdn_delay * 1000)
        
        # Additional latency due to neighbor contention
        if num_neighbors > 3:
            contention_latency_ms = (num_neighbors - 3) * 0.2  # 0.2ms per neighbor above 3
            base_latency_ms += contention_latency_ms
        
        # Store offered load information
        offered_load = getattr(vehicle, 'offered_load', cbr)
        congestion_ratio = offered_load / 1.0 if offered_load > 1.0 else 1.0
        
        return {
            'ber': ber,
            'ser': ser,
            'per': per_total,
            'per_phy': per_phy,
            'pdr': 1 - per_total,
            'phy_throughput': phy_throughput,
            'mac_throughput': final_throughput,
            'throughput': final_throughput,
            'latency': base_latency_ms,
            'collision_prob': collision_prob,
            'mac_efficiency': mac_efficiency,
            'selected_mcs': selected_mcs,
            
            # Detailed latency breakdown
            'phy_latency_ms': phy_components['total_phy_latency_ms'],
            'mac_latency_ms': mac_components['total_mac_latency_ms'],
            'l3_routing_delay': l3_routing_delay * 1000,
            'sdn_processing_delay': sdn_delay * 1000,
            'preamble_latency_ms': phy_components['preamble_latency_ms'],
            'data_tx_latency_ms': phy_components.get('data_transmission_latency_ms', 0),
            'difs_latency_ms': mac_components['difs_latency_ms'],
            'backoff_latency_ms': mac_components['backoff_latency_ms'],
            'retry_latency_ms': mac_components['retry_latency_ms'],
            'queue_latency_ms': mac_components['queue_latency_ms'],
            
            # Offered load and congestion tracking
            'offered_load': offered_load,
            'congestion_ratio': congestion_ratio,
            'channel_overload': 'Yes' if offered_load > 1.0 else 'No'
        }
    
    
    def _calculate_phy_latency_components(self, mcs: int, packet_bits: int) -> Dict[str, float]:
        """Calculate detailed PHY layer latency components in milliseconds (from script 2)"""
        
        # IEEE 802.11bd PHY timing parameters
        preamble_duration = 40e-6  # 40 μs preamble
        symbol_duration = 4e-6     # 4 μs OFDM symbol
        processing_delay = 10e-6   # 10 μs PHY processing
        
        # Calculate OFDM symbols needed
        mcs_config = self.ieee_mapper.mcs_table.get(mcs, self.ieee_mapper.mcs_table[1])
        modulation_order = mcs_config['order']
        code_rate = mcs_config['code_rate']
        
        bits_per_subcarrier = math.log2(modulation_order)
        data_subcarriers = 48  # IEEE 802.11bd 10 MHz
        coded_bits_per_symbol = data_subcarriers * bits_per_subcarrier
        info_bits_per_symbol = coded_bits_per_symbol * code_rate
        
        ofdm_symbols_needed = math.ceil(packet_bits / info_bits_per_symbol)
        data_transmission_time = ofdm_symbols_needed * symbol_duration
        
        # Total PHY transmission time
        total_phy_time = preamble_duration + data_transmission_time + processing_delay
        
        return {
            'preamble_latency_ms': preamble_duration * 1000,
            'data_transmission_latency_ms': data_transmission_time * 1000,
            'phy_processing_latency_ms': processing_delay * 1000,
            'total_phy_latency_ms': total_phy_time * 1000
        }
        
    def _calculate_enhanced_contention_delay(self, cbr: float, num_neighbors: int) -> float:
        """Calculate enhanced contention delay with neighbor impact"""
        difs = self.config.difs
        slot_time = self.config.slot_time
            
        base_cw = self.config.cw_min
            
        if cbr <= 0.3:
            cw_multiplier = 1.0
        elif cbr <= 0.5:
            cw_multiplier = 2.0
        elif cbr <= 0.7:
            cw_multiplier = 4.0
        else:
            cw_multiplier = 8.0
            
        neighbor_multiplier = 1.0 + (num_neighbors * 0.1)
            
        final_cw = min(self.config.cw_max, base_cw * cw_multiplier * neighbor_multiplier)
        avg_backoff = (final_cw / 2) * slot_time
            
        if cbr > 0.6 or num_neighbors > 20:
            congestion_factor = 1 + (cbr - 0.6) * 2.0 + max(0, num_neighbors - 20) * 0.1
            queuing_delay = congestion_factor * (difs + avg_backoff)
        else:
            queuing_delay = 0
            
        return difs + avg_backoff + queuing_delay

    
    def _calculate_mac_latency_components(self, cbr: float, per: float, neighbor_count: int, 
                                        phy_tx_time: float) -> Dict[str, float]:
        """Calculate detailed MAC layer latency components in milliseconds (from script 2)"""
        
        # IEEE 802.11bd MAC timing parameters
        sifs = 16e-6  # 16 μs
        difs = 34e-6  # 34 μs
        slot_time = 9e-6  # 9 μs
        
        # Calculate contention window based on CBR and neighbors
        base_cw = 15  # IEEE 802.11bd CWmin
        
        if cbr <= 0.3:
            cw_multiplier = 1.0
        elif cbr <= 0.5:
            cw_multiplier = 1.4
        elif cbr <= 0.7:
            cw_multiplier = 2.2
        else:
            cw_multiplier = 3.5
        
        neighbor_multiplier = 1.0 + (neighbor_count * 0.05)
        effective_cw = min(1023, base_cw * cw_multiplier * neighbor_multiplier)
        
        # Average backoff time
        avg_backoff_time = (effective_cw / 2) * slot_time
        
        # DIFS waiting time
        difs_time = difs
        
        # Retransmission delays
        if per > 0.001:
            expected_retries = min(7, per / (1 - per + 1e-10))
            neighbor_factor = 1.0 + (neighbor_count * 0.008)
            total_retries = expected_retries * neighbor_factor * 0.5
            
            retry_delay = total_retries * (phy_tx_time + avg_backoff_time + difs)
        else:
            retry_delay = 0
            total_retries = 0
        
        # Queue waiting time
        if cbr > 0.6 or neighbor_count > 30:
            congestion_factor = 1 + (cbr - 0.6) * 1.0 + max(0, neighbor_count - 30) * 0.05
            queue_delay = congestion_factor * (difs + avg_backoff_time) * 0.3
        else:
            queue_delay = 0
        
        # MAC processing overhead
        mac_processing = 5e-6  # 5 μs MAC processing
        
        total_mac_latency = difs_time + avg_backoff_time + retry_delay + queue_delay + mac_processing
        
        return {
            'difs_latency_ms': difs_time * 1000,
            'backoff_latency_ms': avg_backoff_time * 1000,
            'retry_latency_ms': retry_delay * 1000,
            'queue_latency_ms': queue_delay * 1000,
            'mac_processing_latency_ms': mac_processing * 1000,
            'total_mac_latency_ms': total_mac_latency * 1000,
            'retry_count': total_retries
        }
    
    def _calculate_enhanced_retransmission_delay(self, per: float, tx_time: float, num_neighbors: int) -> float:
        """Calculate enhanced retransmission delay with neighbor impact"""
        if per <= 0.001:
            return 0
        
        max_retries = self.config.retry_limit
        
        base_expected_retries = per / (1 - per + 1e-10)
        neighbor_factor = 1.0 + (num_neighbors * 0.02)
        expected_retries = min(max_retries, base_expected_retries * neighbor_factor)
        
        if expected_retries <= 0:
            return 0
        
        total_backoff_delay = 0
        for retry in range(int(expected_retries) + 1):
            backoff_window = min(self.config.cw_max, self.config.cw_min * (2 ** retry))
            
            if num_neighbors > 15:
                backoff_window = min(self.config.cw_max, backoff_window * 1.5)
            
            avg_retry_delay = (backoff_window / 2) * self.config.slot_time
            total_backoff_delay += avg_retry_delay
        
        return expected_retries * (tx_time + total_backoff_delay + self.config.difs)
    
    def _update_layer3_metrics(self, vehicle_id: str, packets_processed: int, current_time: float):
        """Enhanced Layer 3 performance metrics collection with proper initialization"""
        if not self.config.enable_layer3:
            return
        
        vehicle = self.vehicles[vehicle_id]
        
        # Initialize L3 metrics if not present (MUST BE FIRST)
        if not hasattr(vehicle, 'l3_metrics') or vehicle.l3_metrics is None:
            vehicle.l3_metrics = {}
        
        # Initialize all required keys with default values
        default_metrics = {
            'route_discovery_attempts': 0,
            'route_discovery_success': 0,
            'route_discovery_latency': [],
            'packet_forwarding_ratio': 0.0,
            'end_to_end_delay': [],
            'hop_count_distribution': [],
            'routing_overhead_ratio': 0.0,
            'successful_routes': 0,
            'route_failures': 0,
            'direct_deliveries': 0,
            'routing_table_updates': 0,
            'neighbor_changes': 0,
            'protocol_messages_sent': 0,
            'protocol_messages_received': 0,
            'packets_processed': 0,
            'routing_table_size': 0,
            'average_hop_count': 0.0,
            'route_discovery_success_rate': 0.0
        }
        
        # Initialize missing keys only
        for key, default_value in default_metrics.items():
            if key not in vehicle.l3_metrics:
                if isinstance(default_value, list):
                    vehicle.l3_metrics[key] = []
                else:
                    vehicle.l3_metrics[key] = default_value
        
        # Update basic metrics (now safe to access)
        vehicle.l3_metrics['packets_processed'] = packets_processed
        
        # Update neighbor change tracking (now safe)
        current_neighbors = set(n['id'] for n in vehicle.neighbors) if vehicle.neighbors else set()
        previous_neighbors = getattr(vehicle, '_previous_neighbors', set())
        
        if current_neighbors != previous_neighbors:
            vehicle.l3_metrics['neighbor_changes'] += 1
            vehicle._previous_neighbors = current_neighbors
        
        # Update routing table size and changes
        if vehicle.routing_protocol:
            if hasattr(vehicle.routing_protocol, 'routing_table'):
                new_table_size = len(vehicle.routing_protocol.routing_table)
                old_table_size = vehicle.l3_metrics.get('routing_table_size', 0)
                
                if new_table_size != old_table_size:
                    vehicle.l3_metrics['routing_table_updates'] += 1
                    vehicle.l3_metrics['routing_table_size'] = new_table_size
            
            # Get protocol-specific statistics safely
            if hasattr(vehicle.routing_protocol, 'get_routing_statistics'):
                try:
                    protocol_stats = vehicle.routing_protocol.get_routing_statistics()
                    
                    # Update protocol-specific metrics safely
                    for key, value in protocol_stats.items():
                        if isinstance(value, (int, float)):
                            vehicle.l3_metrics[f'protocol_{key}'] = value
                except Exception as e:
                    print(f"[WARNING] Failed to get routing statistics for {vehicle_id}: {e}")
        
        # Calculate routing overhead ratio safely
        routing_packets = sum([
            vehicle.l3_metrics.get('protocol_hello_sent', 0),
            vehicle.l3_metrics.get('protocol_rreq_sent', 0),
            vehicle.l3_metrics.get('protocol_rrep_sent', 0),
            vehicle.l3_metrics.get('protocol_tc_sent', 0),
            vehicle.l3_metrics.get('protocol_position_beacons_sent', 0)
        ])
        
        # Safe access to packet counters
        if hasattr(vehicle, 'packet_counters') and vehicle.packet_counters:
            data_packets = vehicle.packet_counters.get('generated', 1)  # Avoid division by zero
            vehicle.l3_metrics['routing_overhead_ratio'] = routing_packets / data_packets if data_packets > 0 else 0
            
            # Calculate packet forwarding ratio safely
            forwarded = vehicle.packet_counters.get('forwarded', 0)
            generated = vehicle.packet_counters.get('generated', 1)
            vehicle.l3_metrics['packet_forwarding_ratio'] = forwarded / generated if generated > 0 else 0
        else:
            vehicle.l3_metrics['routing_overhead_ratio'] = 0
            vehicle.l3_metrics['packet_forwarding_ratio'] = 0
        
        # Update end-to-end delay tracking safely
        if hasattr(vehicle, 'packet_delays') and vehicle.packet_delays:
            avg_delay = sum(vehicle.packet_delays) / len(vehicle.packet_delays)
            vehicle.l3_metrics['end_to_end_delay'].append(avg_delay)
            
            # Keep only recent measurements
            if len(vehicle.l3_metrics['end_to_end_delay']) > 100:
                vehicle.l3_metrics['end_to_end_delay'] = vehicle.l3_metrics['end_to_end_delay'][-100:]
        
        # Calculate route discovery success rate safely
        attempts = vehicle.l3_metrics.get('route_discovery_attempts', 0)
        successes = vehicle.l3_metrics.get('route_discovery_success', 0)
        
        if attempts > 0:
            vehicle.l3_metrics['route_discovery_success_rate'] = successes / attempts
        else:
            vehicle.l3_metrics['route_discovery_success_rate'] = 0.0
        
        # Calculate average hop count safely
        hop_counts = vehicle.l3_metrics.get('hop_count_distribution', [])
        if hop_counts:
            vehicle.l3_metrics['average_hop_count'] = sum(hop_counts) / len(hop_counts)
        else:
            vehicle.l3_metrics['average_hop_count'] = 0.0
        
        # Protocol-specific metric updates with error handling
        try:
            if isinstance(vehicle.routing_protocol, AODVRoutingProtocol):
                # AODV-specific metrics
                vehicle.l3_metrics['aodv_active_routes'] = len(vehicle.routing_protocol.routing_table)
                vehicle.l3_metrics['aodv_pending_discoveries'] = len(vehicle.routing_protocol.pending_routes)
                vehicle.l3_metrics['aodv_buffered_packets'] = sum(
                    len(packets) for packets in vehicle.routing_protocol.packet_buffer.values()
                )
            
            elif isinstance(vehicle.routing_protocol, OLSRRoutingProtocol):
                # OLSR-specific metrics
                vehicle.l3_metrics['olsr_neighbor_count'] = len(vehicle.routing_protocol.neighbor_set)
                vehicle.l3_metrics['olsr_mpr_count'] = len(vehicle.routing_protocol.mpr_set)
                vehicle.l3_metrics['olsr_topology_entries'] = sum(
                    len(info) - 1 for info in vehicle.routing_protocol.topology_set.values() 
                    if '_last_update' in info
                )
            
            elif isinstance(vehicle.routing_protocol, GeographicRoutingProtocol):
                # Geographic routing specific metrics
                vehicle.l3_metrics['geo_known_positions'] = len(vehicle.routing_protocol.position_table)
                vehicle.l3_metrics['geo_perimeter_routes'] = len(vehicle.routing_protocol.perimeter_mode)
                vehicle.l3_metrics['geo_communication_range'] = vehicle.routing_protocol.communication_range
                
        except Exception as e:
            print(f"[WARNING] Failed to update protocol-specific metrics for {vehicle_id}: {e}")
    
    def _cleanup_routing_protocol(self, vehicle: 'VehicleState', current_time: float):
        """Universal routing protocol cleanup method"""
        if not vehicle.routing_protocol:
            return
        
        try:
            # Call appropriate cleanup method based on protocol type
            if isinstance(vehicle.routing_protocol, AODVRoutingProtocol):
                vehicle.routing_protocol.cleanup_expired_routes(current_time)
            elif isinstance(vehicle.routing_protocol, OLSRRoutingProtocol):
                vehicle.routing_protocol.cleanup_expired_entries(current_time)
            elif isinstance(vehicle.routing_protocol, GeographicRoutingProtocol):
                vehicle.routing_protocol.cleanup_expired_positions(current_time)
            else:
                # Try both methods for unknown protocols
                if hasattr(vehicle.routing_protocol, 'cleanup_expired_routes'):
                    vehicle.routing_protocol.cleanup_expired_routes(current_time)
                elif hasattr(vehicle.routing_protocol, 'cleanup_expired_entries'):
                    vehicle.routing_protocol.cleanup_expired_entries(current_time)
                elif hasattr(vehicle.routing_protocol, 'cleanup_expired_positions'):
                    vehicle.routing_protocol.cleanup_expired_positions(current_time)
                    
        except Exception as e:
            print(f"[ERROR] Failed to cleanup routing protocol for {vehicle.vehicle_id}: {e}")
        
        
    def _ensure_vehicle_l3_initialization(self, vehicle: 'VehicleState'):
        """Ensure vehicle has proper L3 initialization"""
        # Initialize packet counters if missing
        if not hasattr(vehicle, 'packet_counters') or vehicle.packet_counters is None:
            vehicle.packet_counters = {
                'generated': 0,
                'received': 0,
                'forwarded': 0,
                'dropped': 0,
                'delivered': 0
            }
        
        # Initialize L3 metrics if missing
        if not hasattr(vehicle, 'l3_metrics') or vehicle.l3_metrics is None:
            vehicle.l3_metrics = {}
        
        # Initialize packet delays if missing
        if not hasattr(vehicle, 'packet_delays') or vehicle.packet_delays is None:
            vehicle.packet_delays = []
        
        # Initialize previous neighbors tracking if missing
        if not hasattr(vehicle, '_previous_neighbors'):
            vehicle._previous_neighbors = set()
   
    def _get_l3_performance_summary(self) -> Dict[str, float]:
        """Get comprehensive Layer 3 performance summary"""
        if not self.config.enable_layer3:
            return {}
        
        total_metrics = {
            'total_route_discoveries': 0,
            'successful_route_discoveries': 0,
            'total_packets_generated': 0,
            'total_packets_delivered': 0,
            'total_routing_overhead': 0,
            'average_hop_count': 0.0,
            'average_end_to_end_delay': 0.0,
            'active_routes_count': 0,
            'neighbor_changes_count': 0
        }
        
        vehicle_count = 0
        
        for vehicle in self.vehicles.values():
            if hasattr(vehicle, 'l3_metrics'):
                metrics = vehicle.l3_metrics
                vehicle_count += 1
                
                total_metrics['total_route_discoveries'] += metrics.get('route_discovery_attempts', 0)
                total_metrics['successful_route_discoveries'] += metrics.get('route_discovery_success', 0)
                total_metrics['total_packets_generated'] += vehicle.packet_counters.get('generated', 0)
                total_metrics['total_packets_delivered'] += vehicle.packet_counters.get('delivered', 0)
                total_metrics['total_routing_overhead'] += metrics.get('protocol_messages_sent', 0)
                total_metrics['active_routes_count'] += metrics.get('routing_table_size', 0)
                total_metrics['neighbor_changes_count'] += metrics.get('neighbor_changes', 0)
                
                # Average hop count
                hop_counts = metrics.get('hop_count_distribution', [])
                if hop_counts:
                    total_metrics['average_hop_count'] += sum(hop_counts) / len(hop_counts)
                
                # Average end-to-end delay
                delays = metrics.get('end_to_end_delay', [])
                if delays:
                    total_metrics['average_end_to_end_delay'] += sum(delays) / len(delays)
        
        # Calculate averages
        if vehicle_count > 0:
            total_metrics['average_hop_count'] /= vehicle_count
            total_metrics['average_end_to_end_delay'] /= vehicle_count
            
            # Calculate overall PDR
            if total_metrics['total_packets_generated'] > 0:
                total_metrics['overall_pdr'] = (total_metrics['total_packets_delivered'] / 
                                              total_metrics['total_packets_generated'])
            else:
                total_metrics['overall_pdr'] = 0.0
            
            # Calculate route discovery success rate
            if total_metrics['total_route_discoveries'] > 0:
                total_metrics['route_discovery_success_rate'] = (total_metrics['successful_route_discoveries'] / 
                                                               total_metrics['total_route_discoveries'])
            else:
                total_metrics['route_discovery_success_rate'] = 0.0
        
        return total_metrics

    def _validate_l3_implementation(self) -> Dict[str, bool]:
        """Validate Layer 3 implementation correctness"""
        validation_results = {
            'routing_protocol_initialized': False,
            'routing_tables_populated': False,
            'packet_generation_working': False,
            'route_discovery_functional': False,
            'neighbor_discovery_working': False,
            'metrics_collection_active': False
        }
        
        if not self.config.enable_layer3:
            return validation_results
        
        # Check if routing protocols are properly initialized
        initialized_count = 0
        for vehicle in self.vehicles.values():
            if hasattr(vehicle, 'routing_protocol') and vehicle.routing_protocol:
                initialized_count += 1
        
        validation_results['routing_protocol_initialized'] = initialized_count > 0
        
        # Check if routing tables have entries
        total_routes = 0
        for vehicle in self.vehicles.values():
            if (hasattr(vehicle, 'routing_protocol') and vehicle.routing_protocol and
                hasattr(vehicle.routing_protocol, 'routing_table')):
                total_routes += len(vehicle.routing_protocol.routing_table)
        
        validation_results['routing_tables_populated'] = total_routes > 0
        
        # Check packet generation
        total_packets = sum(vehicle.packet_counters.get('generated', 0) for vehicle in self.vehicles.values())
        validation_results['packet_generation_working'] = total_packets > 0
        
        # Check route discovery activity (for AODV)
        route_discoveries = 0
        for vehicle in self.vehicles.values():
            if hasattr(vehicle, 'l3_metrics'):
                route_discoveries += vehicle.l3_metrics.get('route_discovery_attempts', 0)
        
        validation_results['route_discovery_functional'] = route_discoveries > 0
        
        # Check neighbor discovery
        total_neighbors = sum(len(vehicle.neighbors) for vehicle in self.vehicles.values())
        validation_results['neighbor_discovery_working'] = total_neighbors > 0
        
        # Check metrics collection
        metrics_active = 0
        for vehicle in self.vehicles.values():
            if hasattr(vehicle, 'l3_metrics') and vehicle.l3_metrics:
                metrics_active += 1
        
        validation_results['metrics_collection_active'] = metrics_active > 0
        
        return validation_results
    
    def _update_sdn_metrics(self, vehicle_id: str, current_time: float):
        """Update SDN performance metrics"""
        if not self.config.enable_sdn:
            return
        
        vehicle = self.vehicles[vehicle_id]
        
        # Update flow table metrics
        vehicle.sdn_metrics['flow_rule_count'] = len(vehicle.flow_table)
        
        # Calculate flow utilization
        total_flows = len(vehicle.flow_table)
        active_flows = sum(1 for flow in vehicle.flow_table.values() 
                          if flow.state == FlowState.ACTIVE)
        
        if total_flows > 0:
            vehicle.sdn_metrics['flow_utilization']['active_ratio'] = active_flows / total_flows
        
        # Update controller communication metrics
        if self.sdn_controller:
            controller_stats = self.sdn_controller.get_network_statistics()
            vehicle.sdn_metrics['network_topology_size'] = controller_stats.get('nodes_count', 0)
    
    def _write_timestamp_results(self, timestamp_results: List[Dict], current_time: float):
        """FIXED: Write enhanced results with offered load information to CSV file"""
        if not ENABLE_REALTIME_CSV or not timestamp_results:
            return
        
        try:
            import csv
            with open(self.realtime_csv_file, 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                
                for result in timestamp_results:
                    # Calculate offered load and congestion metrics
                    offered_load = result.get('offered_load', result.get('CBR', 0))
                    actual_cbr = offered_load
                    congestion_ratio = offered_load
                    
                    row = [
                        # Basic information
                        result.get('Timestamp', ''),
                        result.get('VehicleID', ''),
                        result.get('MACAddress', ''),
                        result.get('IPAddress', ''),
                        result.get('ChannelModel', ''),
                        result.get('ApplicationType', ''),
                        result.get('PayloadLength', ''),
                        result.get('Neighbors', ''),
                        result.get('NeighborNumbers', ''),
                        result.get('PowerTx', ''),
                        result.get('MCS', ''),
                        result.get('MCS_Source', ''),
                        result.get('BeaconRate', ''),
                        result.get('CommRange', ''),
                        
                        # PHY/MAC performance
                        result.get('PHYDataRate', ''),
                        result.get('PHYThroughput_Legacy', ''),
                        result.get('PHYThroughput_80211bd', ''),
                        result.get('PHYThroughput', ''),
                        result.get('ThroughputImprovement', ''),
                        result.get('MACThroughput', ''),
                        result.get('MACEfficiency', ''),
                        result.get('Throughput', ''),
                        result.get('Latency', ''),
                        result.get('BER', ''),
                        result.get('SER', ''),
                        result.get('PER_PHY_Base', ''),
                        result.get('PER_PHY_Enhanced', ''),
                        result.get('PER_Total', ''),
                        result.get('PER', ''),
                        result.get('CollisionProb', ''),
                        
                        # FIXED: CBR and offered load information
                        actual_cbr,  # Bounded CBR
                        offered_load,  # Can exceed 1.0
                        congestion_ratio,  # Congestion indicator
                        'Yes' if offered_load > 1.0 else 'No',  # Channel overload flag
                        
                        result.get('SINR', ''),
                        result.get('SignalPower_dBm', ''),
                        result.get('InterferencePower_dBm', ''),
                        result.get('ThermalNoise_dBm', ''),
                        result.get('PDR', ''),
                        result.get('TargetPER_Met', ''),
                        result.get('TargetPDR_Met', ''),
                        
                        # IEEE 802.11bd features
                        result.get('LDPC_Enabled', ''),
                        result.get('Midambles_Enabled', ''),
                        result.get('DCM_Enabled', ''),
                        result.get('ExtendedRange_Enabled', ''),
                        result.get('MIMO_STBC_Enabled', ''),
                        
                        # MAC statistics
                        result.get('MACSuccess', ''),
                        result.get('MACRetries', ''),
                        result.get('MACDrops', ''),
                        result.get('MACAttempts', ''),
                        result.get('AvgMACDelay', ''),
                        result.get('AvgMACLatency', ''),
                        result.get('AvgMACThroughput', ''),
                        
                        # Environment
                        result.get('BackgroundTraffic', ''),
                        result.get('HiddenNodeFactor', ''),
                        result.get('InterSystemInterference', ''),
                        
                        # Episode tracking
                        result.get('Episode', ''),
                        result.get('TotalEpisodes', ''),
                        result.get('OriginalTimestamp', ''),
                        result.get('ReloadStrategy', ''),
                        
                        # Layer 3 metrics
                        result.get('L3_Enabled', ''),
                        result.get('RoutingProtocol', ''),
                        result.get('RoutingTableSize', ''),
                        result.get('RouteDiscoveryAttempts', ''),
                        result.get('RouteDiscoverySuccess', ''),
                        result.get('AvgRouteLength', ''),
                        result.get('RoutingOverheadRatio', ''),
                        result.get('PacketsGenerated', ''),
                        result.get('PacketsReceived', ''),
                        result.get('PacketsForwarded', ''),
                        result.get('PacketsDropped', ''),
                        result.get('EndToEndDelay_ms', ''),
                        result.get('HopCount', ''),
                        result.get('L3_PDR', ''),
                        
                        # SDN metrics
                        result.get('SDN_Enabled', ''),
                        result.get('FlowTableSize', ''),
                        result.get('ActiveFlows', ''),
                        result.get('FlowInstallationTime_ms', ''),
                        result.get('PacketInCount', ''),
                        result.get('FlowModCount', ''),
                        result.get('ControllerLatency_ms', ''),
                        result.get('QoSViolations', ''),
                        result.get('TrafficEngineeringEvents', ''),
                        result.get('SDN_Throughput_Improvement', ''),
                        
                        # Application-specific metrics
                        result.get('SafetyPackets', ''),
                        result.get('InfotainmentPackets', ''),
                        result.get('SensingPackets', ''),
                        result.get('EmergencyQoS', ''),
                        result.get('SafetyQoS', ''),
                        result.get('ServiceQoS', ''),
                        result.get('BackgroundQoS', ''),
                        result.get('QoS_DelayViolations', ''),
                        result.get('QoS_ThroughputViolations', '')
                    ]
                    writer.writerow(row)
            
            print(f"[CSV UPDATE] t={current_time:.1f}s: {len(timestamp_results)} enhanced records with offered load metrics written")
            
        except Exception as e:
            print(f"[WARNING] Failed to write enhanced CSV data for t={current_time:.1f}s: {e}")
    
    def _initialize_visualization(self):
        """Initialize enhanced live visualization system"""
        try:
            if LIVE_VISUALIZATION and SEPARATE_PLOT_WINDOWS:
                self.live_plot_manager = LivePlotManager(self.config, self.scenario_info)
                print("[VISUALIZER] Enhanced live visualization with separate windows enabled")
                return True
            elif ENABLE_VISUALIZATION:  # Only if explicitly enabled
                self.visualizer = VANETVisualizer(self.config, self.scenario_info, enable_animation=True)
                print("[VISUALIZER] Standard visualization enabled")
                return True
            else:
                print("[VISUALIZER] All visualization disabled")
                return False
        except Exception as e:
            print(f"[VISUALIZER WARNING] Failed to initialize visualization: {e}")
            self.live_plot_manager = None
            self.visualizer = None
            return False
    
    def run_simulation(self) -> List[Dict]:
        """Run enhanced simulation with Layer 3 and SDN capabilities - FIXED to preserve RL parameters"""
        print(f"[INFO] Starting Enhanced IEEE 802.11bd VANET simulation with Layer 3 and SDN")
        print(f"[INFO] Layer 3 Protocol: {self.config.routing_protocol if self.config.enable_layer3 else 'Disabled'}")
        print(f"[INFO] SDN Controller: {self.config.sdn_controller_type if self.config.enable_sdn else 'Disabled'}")
        print(f"[INFO] Packet Simulation: {'Enabled' if self.config.enable_packet_simulation else 'Disabled'}")
        print(f"[INFO] Total Simulation Duration: {self.total_simulation_duration:.0f} seconds")
        print(f"[INFO] Channel Model: {self.config.channel_model}")
        print(f"[INFO] Application Type: {self.config.application_type}")
        print(f"[INFO] RL optimization: {'Enabled' if self.enable_rl else 'Disabled'}")
        
        # Get time points
        time_points = sorted(list(set(data['time'] for data in self.mobility_data)))
        
        results = []
        validation_warnings = 0
        
        # Application-specific performance targets
        app_config = self.ieee_mapper.application_configs.get(
            self.config.application_type, 
            self.ieee_mapper.application_configs['safety']
        )
        target_per = app_config['target_per']
        target_pdr = app_config['target_pdr']
        
        print(f"[INFO] Target PER: {target_per*100:.1f}% | Target PDR: {target_pdr*100:.1f}%")
        print(f"[INFO] Processing {len(time_points)} time points...")
        
        # Episode tracking for reloaded data
        episode_boundaries = []
        if self.fcd_reload_count > 1:
            for episode in range(self.fcd_reload_count):
                episode_start_time = episode * (self.original_simulation_duration + 1.0)
                episode_end_time = episode_start_time + self.original_simulation_duration
                episode_boundaries.append((episode + 1, episode_start_time, episode_end_time))
        
        current_episode = 1
        episode_start_idx = 0
        
        # NEW: Initialize RL pending updates storage
        pending_rl_updates = {}
        self.rl_log_counter = 0  # Initialize RL logging counter
        
        for time_idx, current_time in enumerate(time_points):
            
            # Episode progress reporting
            if self.fcd_reload_count > 1:
                for episode_num, episode_start, episode_end in episode_boundaries:
                    if episode_start <= current_time <= episode_end and episode_num != current_episode:
                        current_episode = episode_num
                        episode_start_idx = time_idx
                        print(f"[EPISODE TRANSITION] Starting Episode {current_episode}/{self.fcd_reload_count} at t={current_time:.1f}s")
                        break
                
                if time_idx % 50 == 0:
                    episode_progress = (time_idx - episode_start_idx) / (len(time_points) / self.fcd_reload_count) * 100
                    overall_progress = time_idx / len(time_points) * 100
                    print(f"[PROGRESS] Episode {current_episode}/{self.fcd_reload_count}: {episode_progress:.1f}% | "
                          f"Overall: {overall_progress:.1f}% | t={current_time:.1f}s ({time_idx+1}/{len(time_points)})")
            else:
                if time_idx % 50 == 0:
                    print(f"[PROGRESS] Processing t={current_time:.1f}s ({time_idx+1}/{len(time_points)})")
            
            # NEW PHASE 0: Apply RL updates from previous timestamp
            if pending_rl_updates:
                self._apply_pending_rl_updates(pending_rl_updates)
                
                # FIXED: Ensure antenna systems are synchronized with transmission_power
                for vehicle in self.vehicles.values():
                    if hasattr(vehicle, 'sync_antenna_power_with_transmission_power'):
                        vehicle.sync_antenna_power_with_transmission_power()
                
                pending_rl_updates.clear()
            
            # Get vehicles at current time
            current_vehicles = set(data['id'] for data in self.mobility_data 
                                 if data['time'] == current_time)
            
            # FIXED: Store RL parameters before any processing that might override them
            rl_preserved_params = {}
            if self.enable_rl:
                for vehicle_id in current_vehicles:
                    if vehicle_id in self.vehicles:
                        vehicle = self.vehicles[vehicle_id]
                        rl_preserved_params[vehicle_id] = {
                            'power': getattr(vehicle, 'transmission_power', 20.0),
                            'mcs': getattr(vehicle, 'mcs', 1),
                            'beacon_rate': getattr(vehicle, 'beacon_rate', 10.0)
                        }
            
            # FIXED Phase - Update attacker behavior BEFORE other calculations (and make it RL-aware)
            if self.attack_manager:
                # Pass RL status to attack manager
                if hasattr(self.attack_manager, 'enable_rl'):
                    self.attack_manager.enable_rl = self.enable_rl
                self._update_attacker_behavior(current_time)
            
            # Phase 1: Update positions and find neighbors
            for vehicle_id in current_vehicles:
                if vehicle_id not in self.vehicles:
                    continue
                
                vehicle = self.vehicles[vehicle_id]
                
                # Update position from FCD
                vehicle_data = next((data for data in self.mobility_data 
                                   if data['time'] == current_time and data['id'] == vehicle_id), None)
                if vehicle_data:
                    vehicle.x = vehicle_data['x']
                    vehicle.y = vehicle_data['y']
                    vehicle.speed = vehicle_data['speed']
                    vehicle.angle = vehicle_data['angle']
                
                # Find neighbors
                neighbors = self._find_neighbors(vehicle_id, current_time)
                vehicle.neighbors = neighbors
                vehicle.neighbors_number = len(neighbors)
                
                # Update SDN controller topology
                if self.sdn_controller:
                    self.sdn_controller.update_node_position(vehicle_id, vehicle.x, vehicle.y)
                    self.sdn_controller.update_network_topology(vehicle_id, neighbors)
            
            # Phase 1.5: Generate background traffic
            if self.background_traffic_manager:
                network_bg_packets = self.background_traffic_manager.generate_network_background_traffic(
                    current_time, {vid: v for vid, v in self.vehicles.items() if vid in current_vehicles})
                
                # Update vehicle background traffic statistics
                for vehicle_id, bg_packets in network_bg_packets.items():
                    if vehicle_id in self.vehicles:
                        vehicle = self.vehicles[vehicle_id]
                        if not hasattr(vehicle, 'background_packets_generated'):
                            vehicle.background_packets_generated = 0
                            vehicle.background_bytes_generated = 0
                        
                        vehicle.background_packets_generated += len(bg_packets)
                        vehicle.background_bytes_generated += sum(p.size_bytes for p in bg_packets)
            
        
            # Phase 2: Enhanced Layer 3 Protocol Updates
            if self.config.enable_layer3:
                self._update_all_routing_protocols(current_vehicles, current_time)
            # Phase 3: Packet Generation and Processing
            if self.config.enable_packet_simulation:
                for vehicle_id in current_vehicles:
                    if vehicle_id not in self.vehicles:
                        continue
                    
                    vehicle = self.vehicles[vehicle_id]
                    
                    # Generate application packets
                    new_packets = self._generate_application_packets(vehicle_id, current_time)
                    
                    # Process packets (routing and forwarding)
                    packets_processed = 0
                    for packet in new_packets:
                        # Layer 3 routing
                        next_hop = None
                        if self.config.enable_layer3:
                            next_hop = self._process_layer3_routing(vehicle_id, packet, current_time)
                        
                        # SDN flow processing
                        if self.config.enable_sdn:
                            sdn_actions = self._process_sdn_flow_matching(vehicle_id, packet)
                            if sdn_actions:
                                for action in sdn_actions:
                                    if action.get('type') == 'forward':
                                        next_hop = action.get('next_hop')
                        
                        # Update packet delivery statistics
                        if next_hop or packet.destination_id == "BROADCAST":
                            vehicle.packet_counters['forwarded'] += 1
                            packets_processed += 1
                            
                            # Simulate packet delivery
                            if packet.destination_id != "BROADCAST":
                                delivery_success = random.random() > 0.1  # 90% delivery success
                                if delivery_success:
                                    packet.delivery_status = "DELIVERED"
                                    packet.delay = current_time - packet.creation_time
                                    vehicle.packet_counters['delivered'] += 1
                        else:
                            vehicle.packet_counters['dropped'] += 1
                    
                    # Update Layer 3 metrics
                    self._update_layer3_metrics(vehicle_id, packets_processed, current_time)
            
            # Phase 4: SDN Traffic Engineering
            if self.config.enable_sdn and self.sdn_controller:
                if time_idx % int(self.config.sdn_update_interval / self.config.time_step) == 0:
                    # Perform traffic engineering optimization
                    optimization_results = self.sdn_controller.perform_traffic_engineering()
                    
                    if optimization_results:
                        self.sdn_performance_monitor['traffic_engineering_events'] += 1
                        
                        # Apply optimization results
                        for flow_id, flow_rules in optimization_results.items():
                            for node_id, flow_entry in flow_rules:
                                if node_id in self.vehicles:
                                    self.vehicles[node_id].flow_table[flow_entry.flow_id] = flow_entry
                    
                    # Cleanup expired flows
                    self.sdn_controller.cleanup_expired_flows()
            
            # MODIFIED Phase 5: RL Optimization (store response, don't apply immediately)
            if self.enable_rl and self.rl_client:
                    rl_data = {}
                    for vehicle_id in current_vehicles:
                        if vehicle_id not in self.vehicles:
                            continue
                            
                        vehicle = self.vehicles[vehicle_id]
                        neighbor_count = len(vehicle.neighbors) if hasattr(vehicle, 'neighbors') and vehicle.neighbors else 0
                        
                        # Update antenna system with current neighbor distribution
                        if hasattr(vehicle, 'antenna_system'):
                            vehicle.antenna_system.update_neighbor_distribution(vehicle.neighbors)
                        
                        # Standard RL state (same as before)
                        rl_vehicle_data = {
                            'CBR': getattr(vehicle, 'current_cbr', 0) if hasattr(vehicle, 'current_cbr') and not math.isnan(getattr(vehicle, 'current_cbr', 0)) else 0,
                            'SINR': getattr(vehicle, 'current_snr', 0) if hasattr(vehicle, 'current_snr') and not math.isnan(getattr(vehicle, 'current_snr', 0)) else 0,
                            'neighbors': neighbor_count,
                            'MCS': getattr(vehicle, 'mcs', 1),
                            'beaconRate': getattr(vehicle, 'beacon_rate', 10),
                            'timestamp': current_time,
                            'channelModel': self.config.channel_model,
                            'applicationType': self.config.application_type,
                            'antenna_type': ANTENNA_TYPE,
                            'throughput': getattr(vehicle, 'current_throughput', 0) / 1e6,
                            'PDR': getattr(vehicle, 'current_pdr', 0),
                            'PER': getattr(vehicle, 'current_per', 0),
                            
                            # Enhanced features
                            'layer3_enabled': self.config.enable_layer3,
                            'routing_protocol': self.config.routing_protocol,
                            'sdn_enabled': self.config.enable_sdn,
                            'flow_table_size': len(vehicle.flow_table) if hasattr(vehicle, 'flow_table') else 0,
                            'packet_delivery_ratio': vehicle.packet_counters['delivered'] / max(1, vehicle.packet_counters['generated']) if hasattr(vehicle, 'packet_counters') else 0,
                            'routing_overhead': vehicle.l3_metrics.get('routing_overhead_ratio', 0) if hasattr(vehicle, 'l3_metrics') else 0
                        }
                        
                        # Episode info for FCD reloading
                        if self.fcd_reload_count > 1:
                            current_episode = 1
                            for episode_num in range(1, self.fcd_reload_count + 1):
                                episode_start = (episode_num - 1) * (self.original_simulation_duration + 1.0)
                                episode_end = episode_start + self.original_simulation_duration
                                if episode_start <= current_time <= episode_end:
                                    current_episode = episode_num
                                    break
                            
                            rl_vehicle_data['episode'] = current_episode
                            rl_vehicle_data['total_episodes'] = self.fcd_reload_count
                            rl_vehicle_data['episode_time'] = current_time - (episode_start if 'episode_start' in locals() else 0)
                        
                        rl_data[vehicle_id] = rl_vehicle_data
                    
                    try:
                        # NEW: Store RL response for next timestamp instead of applying immediately
                        rl_response = self._communicate_with_rl(rl_data)
                        if rl_response and 'vehicles' in rl_response:
                            pending_rl_updates = rl_response  # Store for next timestamp
                            
                            if time_idx % 100 == 0:
                                updated_vehicles = len(rl_response['vehicles'])
                                antenna_info = f"({ANTENNA_TYPE})"
                                if ANTENNA_TYPE == "SECTORAL":
                                    antenna_info = f"(SECTORAL-FR: Front/Rear RL controlled)"
                                print(f"[RL] Received response for {updated_vehicles} vehicles {antenna_info} - will apply next timestamp")
                    
                    except Exception as e:
                        if time_idx % 100 == 0:
                            print(f"[RL ERROR] Communication failed at t={current_time:.2f}: {e}")
            
            # Phase 6: Calculate CBR and SINR
            for vehicle_id in current_vehicles:
                if vehicle_id not in self.vehicles:
                    continue
                vehicle = self.vehicles[vehicle_id]
                vehicle.current_cbr = self._calculate_cbr_realistic_dynamic(vehicle_id, vehicle.neighbors)
                
                # FIXED: Use the correct transmission power for SINR calculation
                if hasattr(vehicle, 'antenna_system'):
                    effective_tx_power = vehicle.transmission_power  # Use vehicle's actual power
                else:
                    effective_tx_power = vehicle.transmission_power
                
                vehicle.current_snr = self.interference_calculator.calculate_sinr_with_interference(
                    vehicle_id, 
                    vehicle.neighbors, 
                    effective_tx_power,  # Use effective power
                    self.config.channel_model,
                    self.background_traffic_manager
                )
                
                # FIXED: Only validate MCS in non-RL mode
                if not self.enable_rl:
                    mcs_thresholds = {0: 5.0, 1: 8.0, 2: 11.0, 3: 14.0, 4: 17.0, 5: 20.0, 6: 23.0, 7: 26.0, 8: 29.0, 9: 32.0, 10: 35.0}
                    validated_mcs = 0
                    for test_mcs in range(10, -1, -1):
                        if vehicle.current_snr >= mcs_thresholds[test_mcs] + 1.0:
                            validated_mcs = test_mcs
                            break
                    if validated_mcs != vehicle.mcs:
                        vehicle.mcs = validated_mcs
                        validation_warnings += 1
            
            # Phase 7: Update SDN metrics
            if self.config.enable_sdn:
                for vehicle_id in current_vehicles:
                    if vehicle_id in self.vehicles:
                        self._update_sdn_metrics(vehicle_id, current_time)
            
            # Phase 8: Calculate performance metrics and create results
            timestamp_results = []
            
            for vehicle_id in current_vehicles:
                if vehicle_id not in self.vehicles:
                    continue
                
                vehicle = self.vehicles[vehicle_id]
                
                # FIXED: Calculate enhanced performance metrics (with RL-aware MCS handling)
                metrics = self._calculate_performance_metrics_rl_aware(
                    vehicle_id, 
                    vehicle.current_snr, 
                    vehicle.current_cbr, 
                    vehicle.neighbors,
                    self.config.channel_model
                )
                
                vehicle.current_ber = metrics['ber']
                vehicle.current_ser = metrics['ser']
                vehicle.current_per = metrics['per']
                vehicle.current_pdr = metrics['pdr']
                vehicle.current_throughput = metrics['throughput']
                vehicle.current_latency = metrics['latency']
                
                # FIXED: Restore RL parameters after any calculations that might have overridden them
                if self.enable_rl and vehicle_id in rl_preserved_params:
                    preserved = rl_preserved_params[vehicle_id]
                    vehicle.transmission_power = preserved['power']
                    vehicle.mcs = preserved['mcs']
                    vehicle.beacon_rate = preserved['beacon_rate']
                    
                    # Sync antenna system
                    if hasattr(vehicle, 'antenna_system'):
                        if ANTENNA_TYPE == "SECTORAL":
                            vehicle.antenna_system.distribute_power_from_rl(preserved['power'])
                        else:
                            vehicle.antenna_system.config.omnidirectional_config["power_dbm"] = preserved['power']
                
                # Enhanced debug output
                if random.random() < 0.002:
                    episode_info = f"Ep{current_episode} " if self.fcd_reload_count > 1 else ""
                    l3_info = f"L3:{self.config.routing_protocol}" if self.config.enable_layer3 else "L3:Off"
                    sdn_info = f"SDN:{len(vehicle.flow_table)}" if self.config.enable_sdn else "SDN:Off"
                    
                    print(f"[ENHANCED DEBUG] {episode_info}Vehicle {vehicle_id} | {l3_info} | {sdn_info}")
                    print(f"  Neighbors: {len(vehicle.neighbors)} | SINR: {vehicle.current_snr:.1f} dB | CBR: {vehicle.current_cbr:.3f}")
                    print(f"  PER: {metrics['per']:.4f} | PDR: {metrics['pdr']:.4f} | Throughput: {metrics['throughput']/1e6:.2f} Mbps")
                    print(f"  L3 Delay: {metrics.get('l3_routing_delay', 0)*1000:.2f} ms | SDN Delay: {metrics.get('sdn_processing_delay', 0)*1000:.2f} ms")
                    
                    if self.enable_rl:
                        print(f"  RL Params: Power={vehicle.transmission_power:.1f}dBm, MCS={vehicle.mcs}, Beacon={vehicle.beacon_rate:.1f}Hz")
                    
                    if self.config.enable_packet_simulation:
                        pkt_gen = vehicle.packet_counters.get('generated', 0)
                        pkt_del = vehicle.packet_counters.get('delivered', 0)
                        pdr_l3 = pkt_del / max(1, pkt_gen)
                        print(f"  Packets: Gen={pkt_gen}, Del={pkt_del}, L3_PDR={pdr_l3:.3f}")
                
                # Create enhanced result record
                result = {
                    # Basic information
                    'Timestamp': current_time,
                    'VehicleID': vehicle_id,
                    'MACAddress': vehicle.mac_address,
                    'IPAddress': vehicle.ip_address,
                    'ChannelModel': self.config.channel_model,
                    'ApplicationType': self.config.application_type,
                    'PayloadLength': self.config.payload_length,
                    'Neighbors': ', '.join([n['id'] for n in vehicle.neighbors]) if vehicle.neighbors else 'None',
                    'NeighborNumbers': len(vehicle.neighbors),
                    'PowerTx': vehicle.transmission_power,
                    'MCS': vehicle.mcs,
                    'MCS_Source': 'RL' if self.enable_rl else 'SINR_Adaptive',
                    'BeaconRate': vehicle.beacon_rate,
                    'CommRange': vehicle.comm_range,
                    
                    # PHY/MAC performance
                    'PHYDataRate': self.ieee_mapper.data_rates.get(vehicle.mcs, 0),
                    'PHYThroughput_Legacy': metrics.get('phy_throughput_legacy', 0) / 1e6,
                    'PHYThroughput_80211bd': metrics.get('phy_throughput', 0) / 1e6,
                    'PHYThroughput': metrics.get('phy_throughput', 0) / 1e6,
                    'ThroughputImprovement': metrics.get('throughput_improvement_factor', 1.0),
                    'MACThroughput': metrics['throughput'] / 1e6,
                    'MACEfficiency': metrics.get('mac_efficiency', 0),
                    'Throughput': metrics['throughput'] / 1e6,
                    'Latency': metrics['latency'] * 1000,  # Convert to ms
                    'BER': metrics['ber'],
                    'SER': metrics['ser'],
                    'PER_PHY_Base': metrics.get('per_phy', metrics['per']),
                    'PER_PHY_Enhanced': metrics.get('per_phy', metrics['per']),
                    'PER_Total': metrics['per'],
                    'PER': metrics['per'],
                    'CollisionProb': metrics.get('collision_prob', 0),
                    'CBR': vehicle.current_cbr,
                    'SINR': vehicle.current_snr,
                    'SignalPower_dBm': metrics.get('signal_power_dbm', 0),
                    'InterferencePower_dBm': metrics.get('interference_power_dbm', 0),
                    'ThermalNoise_dBm': metrics.get('thermal_noise_dbm', 0),
                    'PDR': metrics['pdr'],
                    'TargetPER_Met': 'Yes' if metrics['per'] <= target_per else 'No',
                    'TargetPDR_Met': 'Yes' if metrics['pdr'] >= target_pdr else 'No',
                    
                    # IEEE 802.11bd features
                    'LDPC_Enabled': self.config.enable_ldpc,
                    'Midambles_Enabled': self.config.enable_midambles,
                    'DCM_Enabled': self.config.enable_dcm,
                    'ExtendedRange_Enabled': self.config.enable_extended_range,
                    'MIMO_STBC_Enabled': self.config.enable_mimo_stbc,
                    
                    # MAC statistics
                    'MACSuccess': vehicle.mac_success,
                    'MACRetries': vehicle.mac_retries,
                    'MACDrops': vehicle.mac_drops,
                    'MACAttempts': 1,
                    'AvgMACDelay': 0,
                    'AvgMACLatency': 0,
                    'AvgMACThroughput': 0,
                    
                    # Environment
                    'BackgroundTraffic': self.config.background_traffic_load,
                    'HiddenNodeFactor': self.config.hidden_node_factor,
                    'InterSystemInterference': self.config.inter_system_interference,
                    
                    # Episode tracking
                    'Episode': current_episode if self.fcd_reload_count > 1 else 1,
                    'TotalEpisodes': self.fcd_reload_count,
                    'OriginalTimestamp': current_time - (episode_boundaries[current_episode-1][1] if self.fcd_reload_count > 1 else 0),
                    'ReloadStrategy': self.fcd_reload_strategy,
                    
                    # NEW: Layer 3 metrics
                    'L3_Enabled': self.config.enable_layer3,
                    'RoutingProtocol': self.config.routing_protocol if self.config.enable_layer3 else 'None',
                    'RoutingTableSize': len(vehicle.routing_table) if hasattr(vehicle, 'routing_table') else 0,
                    'RouteDiscoveryAttempts': vehicle.l3_metrics.get('route_discovery_attempts', 0) if hasattr(vehicle, 'l3_metrics') else 0,
                    'RouteDiscoverySuccess': vehicle.l3_metrics.get('route_discovery_success', 0) if hasattr(vehicle, 'l3_metrics') else 0,
                    'AvgRouteLength': sum(vehicle.l3_metrics.get('hop_count_distribution', [0])) / max(1, len(vehicle.l3_metrics.get('hop_count_distribution', [1]))) if hasattr(vehicle, 'l3_metrics') else 0,
                    'RoutingOverheadRatio': vehicle.l3_metrics.get('routing_overhead_ratio', 0) if hasattr(vehicle, 'l3_metrics') else 0,
                    'PacketsGenerated': vehicle.packet_counters.get('generated', 0) if hasattr(vehicle, 'packet_counters') else 0,
                    'PacketsReceived': vehicle.packet_counters.get('received', 0) if hasattr(vehicle, 'packet_counters') else 0,
                    'PacketsForwarded': vehicle.packet_counters.get('forwarded', 0) if hasattr(vehicle, 'packet_counters') else 0,
                    'PacketsDropped': vehicle.packet_counters.get('dropped', 0) if hasattr(vehicle, 'packet_counters') else 0,
                    'EndToEndDelay_ms': (sum(vehicle.l3_metrics.get('end_to_end_delay', [0])) / max(1, len(vehicle.l3_metrics.get('end_to_end_delay', [1])))) * 1000 if hasattr(vehicle, 'l3_metrics') else 0,
                    'HopCount': sum(vehicle.l3_metrics.get('hop_count_distribution', [0])) / max(1, len(vehicle.l3_metrics.get('hop_count_distribution', [1]))) if hasattr(vehicle, 'l3_metrics') else 0,
                    'L3_PDR': vehicle.packet_counters.get('delivered', 0) / max(1, vehicle.packet_counters.get('generated', 1)) if hasattr(vehicle, 'packet_counters') else 0,
                    
                    # NEW: SDN metrics
                    'SDN_Enabled': self.config.enable_sdn,
                    'FlowTableSize': len(vehicle.flow_table) if hasattr(vehicle, 'flow_table') else 0,
                    'ActiveFlows': sum(1 for flow in vehicle.flow_table.values() if flow.state == FlowState.ACTIVE) if hasattr(vehicle, 'flow_table') else 0,
                    'FlowInstallationTime_ms': (sum(vehicle.sdn_metrics.get('flow_installation_time', [0])) / max(1, len(vehicle.sdn_metrics.get('flow_installation_time', [1])))) * 1000 if hasattr(vehicle, 'sdn_metrics') else 0,
                    'PacketInCount': vehicle.sdn_metrics.get('packet_in_count', 0) if hasattr(vehicle, 'sdn_metrics') else 0,
                    'FlowModCount': vehicle.sdn_metrics.get('flow_mod_count', 0) if hasattr(vehicle, 'sdn_metrics') else 0,
                    'ControllerLatency_ms': (sum(vehicle.sdn_metrics.get('controller_latency', [0])) / max(1, len(vehicle.sdn_metrics.get('controller_latency', [1])))) * 1000 if hasattr(vehicle, 'sdn_metrics') else 0,
                    'QoSViolations': 0,  # Placeholder for QoS violation count
                    'TrafficEngineeringEvents': self.sdn_performance_monitor.get('traffic_engineering_events', 0),
                    'SDN_Throughput_Improvement': 1.0,  # Placeholder for SDN throughput improvement
                    
                    # NEW: Application-specific metrics
                    'SafetyPackets': len(vehicle.application_traffic.get('SAFETY', [])) if hasattr(vehicle, 'application_traffic') else 0,
                    'InfotainmentPackets': len(vehicle.application_traffic.get('INFOTAINMENT', [])) if hasattr(vehicle, 'application_traffic') else 0,
                    'SensingPackets': len(vehicle.application_traffic.get('SENSING', [])) if hasattr(vehicle, 'application_traffic') else 0,
                    'EmergencyQoS': vehicle.qos_queues[QoSClass.EMERGENCY].qsize() if hasattr(vehicle, 'qos_queues') else 0,
                    'SafetyQoS': vehicle.qos_queues[QoSClass.SAFETY].qsize() if hasattr(vehicle, 'qos_queues') else 0,
                    'ServiceQoS': vehicle.qos_queues[QoSClass.SERVICE].qsize() if hasattr(vehicle, 'qos_queues') else 0,
                    'BackgroundQoS': vehicle.qos_queues[QoSClass.BACKGROUND].qsize() if hasattr(vehicle, 'qos_queues') else 0,
                    'QoS_DelayViolations': 0,  # Placeholder
                    'QoS_ThroughputViolations': 0  # Placeholder
                }
                
                timestamp_results.append(result)
                results.append(result)
            
            # Phase 9: Update visualization (replace existing Phase 9)
            if (hasattr(self, 'live_plot_manager') and self.live_plot_manager and 
                time_idx % PLOT_UPDATE_INTERVAL == 0):
                try:
                    print(f"[LIVE PLOTS] Updating plots at t={current_time:.1f}s with {len(self.vehicles)} vehicles")
                    
                    # Filter vehicles that have position data
                    active_vehicles = {}
                    for vid, vehicle in self.vehicles.items():
                        if vid in current_vehicles and hasattr(vehicle, 'x') and vehicle.x is not None:
                            active_vehicles[vid] = vehicle
                    
                    if active_vehicles:
                        self.live_plot_manager.update_all_plots(current_time, active_vehicles, self.attack_manager)
                    else:
                        print(f"[LIVE PLOTS WARNING] No vehicles with position data at t={current_time:.1f}s")
                        
                except Exception as e:
                    print(f"[LIVE PLOTS ERROR] Update failed at t={current_time:.1f}s: {e}")
                    import traceback
                    traceback.print_exc()
            
            elif (hasattr(self, 'visualizer') and self.visualizer and time_idx % 5 == 0):
                # Fallback to original visualizer
                try:
                    active_vehicles = {vid: v for vid, v in self.vehicles.items() 
                                     if vid in current_vehicles and hasattr(v, 'x') and v.x is not None}
                    if active_vehicles:
                        self.visualizer.update_visualization(current_time, active_vehicles, self.attack_manager)
                    if time_idx % 100 == 0:
                        self.visualizer.save_current_frame()
                except Exception as e:
                    print(f"[VISUALIZER WARNING] Update failed: {e}")
            
            # Phase 10: Write timestamp results to CSV
            if time_idx % CSV_UPDATE_FREQUENCY == 0:
                self._write_timestamp_results(timestamp_results, current_time)
            
            # Phase 11: Update progressive Excel file
            if EXCEL_UPDATE_FREQUENCY > 0 and time_idx % EXCEL_UPDATE_FREQUENCY == 0:
                self._write_progressive_excel(time_idx, current_time)
            
            # Phase 12: Enhanced validation output
            if time_idx % 100 == 0 and timestamp_results:
                neighbor_counts = [r['NeighborNumbers'] for r in timestamp_results]
                cbr_values = [r['CBR'] for r in timestamp_results]
                per_values = [r['PER'] for r in timestamp_results]
                throughput_values = [r['Throughput'] for r in timestamp_results]
                l3_pdr_values = [r['L3_PDR'] for r in timestamp_results if r['L3_Enabled']]
                flow_table_sizes = [r['FlowTableSize'] for r in timestamp_results if r['SDN_Enabled']]
                
                episode_info = f"Episode {current_episode}/{self.fcd_reload_count} | " if self.fcd_reload_count > 1 else ""
                print(f"[ENHANCED VALIDATION] {episode_info}t={current_time:.1f}s:")
                print(f"  Neighbors: min={min(neighbor_counts)}, max={max(neighbor_counts)}, avg={sum(neighbor_counts)/len(neighbor_counts):.1f}")
                print(f"  CBR: min={min(cbr_values):.3f}, max={max(cbr_values):.3f}, avg={sum(cbr_values)/len(cbr_values):.3f}")
                print(f"  PER: min={min(per_values):.4f}, max={max(per_values):.4f}, avg={sum(per_values)/len(per_values):.4f}")
                print(f"  Throughput: min={min(throughput_values):.2f}, max={max(throughput_values):.2f}, avg={sum(throughput_values)/len(throughput_values):.2f} Mbps")
                
                if self.config.enable_layer3 and l3_pdr_values:
                    print(f"  L3 PDR: min={min(l3_pdr_values):.3f}, max={max(l3_pdr_values):.3f}, avg={sum(l3_pdr_values)/len(l3_pdr_values):.3f}")
                
                if self.config.enable_sdn and flow_table_sizes:
                    print(f"  Flow Tables: min={min(flow_table_sizes)}, max={max(flow_table_sizes)}, avg={sum(flow_table_sizes)/len(flow_table_sizes):.1f}")
            
            # NEW: Phase - Collect attack detection data
            if self.attack_manager:
                self._collect_attack_detection_data(current_time)
        
        self.simulation_results = results
        print(f"[INFO] Enhanced IEEE 802.11bd simulation with Layer 3 and SDN completed. Generated {len(results)} data points.")
        print(f"[INFO] Effective simulation duration: {self.total_simulation_duration:.0f} seconds ({self.fcd_reload_count}x original)")
        
        # Calculate final performance statistics
        if results and self.config.enable_layer3:
            total_packets_generated = sum(r['PacketsGenerated'] for r in results)
            total_packets_delivered = sum(r['PacketsGenerated'] * r['L3_PDR'] for r in results)
            overall_l3_pdr = total_packets_delivered / max(1, total_packets_generated)
            print(f"[L3 SUMMARY] Overall L3 PDR: {overall_l3_pdr:.3f}, Total packets: {total_packets_generated}")
        
        if results and self.config.enable_sdn:
            total_flows = sum(r['FlowTableSize'] for r in results)
            avg_flows_per_vehicle = total_flows / len(results) if results else 0
            print(f"[SDN SUMMARY] Average flows per vehicle: {avg_flows_per_vehicle:.1f}, Traffic engineering events: {self.sdn_performance_monitor['traffic_engineering_events']}")
        
        return results
    
    def _apply_pending_rl_updates(self, rl_response: Dict):
        """Apply RL parameter updates from previous timestamp"""
        if not rl_response or 'vehicles' not in rl_response:
            return
        
        successful_updates = 0
        
        for vehicle_id, vehicle_response in rl_response['vehicles'].items():
            if vehicle_id not in self.vehicles:
                continue
            
            if 'status' in vehicle_response and vehicle_response['status'] == 'error':
                continue
            
            vehicle = self.vehicles[vehicle_id]
            
            # Apply power, MCS, and beacon rate updates
            if ('transmissionPower' in vehicle_response and 
                'MCS' in vehicle_response and 
                'beaconRate' in vehicle_response):
                
                # Bounds checking
                new_power = max(1, min(30, vehicle_response['transmissionPower']))
                new_mcs = max(0, min(9, round(vehicle_response['MCS'])))
                new_beacon = max(1, min(20, vehicle_response['beaconRate']))
                
                # FIXED: Update both antenna system AND vehicle.transmission_power for both antenna types
                if hasattr(vehicle, 'antenna_system') and ANTENNA_TYPE == "SECTORAL":
                    vehicle.antenna_system.distribute_power_from_rl(new_power)
                    vehicle.transmission_power = new_power  # CRITICAL: Also update vehicle attribute
                else:
                    # Omnidirectional antenna
                    vehicle.transmission_power = new_power
                    # CRITICAL FIX: Also update the antenna system config
                    if hasattr(vehicle, 'antenna_system'):
                        vehicle.antenna_system.config.omnidirectional_config["power_dbm"] = new_power
                
                # Apply MCS and beacon rate (same for both antenna types)
                vehicle.mcs = new_mcs
                vehicle.beacon_rate = new_beacon
                
                successful_updates += 1
            
            # Apply other RL optimizations
            if 'routing_aggressiveness' in vehicle_response and self.config.enable_layer3:
                aggressiveness = max(0.1, min(2.0, vehicle_response['routing_aggressiveness']))
                if hasattr(vehicle.routing_protocol, 'route_discovery_timeout'):
                    vehicle.routing_protocol.route_discovery_timeout = self.config.route_discovery_timeout * aggressiveness
            
            if 'flow_timeout' in vehicle_response and self.config.enable_sdn:
                flow_timeout = max(5.0, min(60.0, vehicle_response['flow_timeout']))
                for flow_entry in vehicle.flow_table.values():
                    flow_entry.timeout = flow_timeout
        
        if successful_updates > 0:
            antenna_info = f"({ANTENNA_TYPE})"
            if ANTENNA_TYPE == "SECTORAL":
                antenna_info = f"(SECTORAL-FR: Front/Rear RL controlled)"
            print(f"[RL APPLY] Applied updates to {successful_updates} vehicles {antenna_info}")
    
    
    
    def _communicate_with_rl(self, vehicle_data: Dict) -> Dict:
        """RL communication with enhanced debugging and NO immediate parameter application"""
        if not self.rl_client:
            print("[RL DEBUG] No RL client connection")
            return {}
        
        # Initialize RL logging counter if not exists
        if not hasattr(self, 'rl_log_counter'):
            self.rl_log_counter = 0
        
        try:
            # Parameters for batch processing
            max_vehicles_per_message = 20
            vehicle_ids = list(vehicle_data.keys())
            num_messages = math.ceil(len(vehicle_ids) / max_vehicles_per_message)
            
            rl_response = {'vehicles': {}}
            
            for msg_idx in range(num_messages):
                start_idx = msg_idx * max_vehicles_per_message
                end_idx = min((msg_idx + 1) * max_vehicles_per_message, len(vehicle_ids))
                current_ids = vehicle_ids[start_idx:end_idx]
                
                message_data = {}
                for veh_id in current_ids:
                    current_data = vehicle_data[veh_id]
                    
                    # Ensure all required fields exist
                    if 'CBR' not in current_data or math.isnan(current_data.get('CBR', 0)):
                        current_data['CBR'] = 0
                    if 'SINR' not in current_data or math.isnan(current_data.get('SINR', 0)):
                        current_data['SINR'] = 0
                    if 'neighbors' not in current_data or current_data['neighbors'] is None:
                        current_data['neighbors'] = 0
                    
                    # Send front/rear-only weighted average power for sectoral antennas
                    
                    if veh_id in self.vehicles:
                        vehicle = self.vehicles[veh_id]
                        
                        # Use Coposite Value as the primary source (this is what gets updated by RL)
                
                        if ANTENNA_TYPE == "SECTORAL" and hasattr(vehicle, 'antenna_system'):
                            current_data['transmissionPower'] = vehicle.antenna_system.get_weighted_average_power()
                        else:
                            current_data['transmissionPower'] = getattr(vehicle, 'transmission_power', 20.0)
                        
                        if ANTENNA_TYPE == "SECTORAL" and hasattr(vehicle, 'antenna_system'):
                            # Add additional RL state information for sectoral antennas
                            rl_controlled_powers = vehicle.antenna_system.get_rl_controlled_power()
                            current_data['front_power'] = rl_controlled_powers.get('front', 20.0)
                            current_data['rear_power'] = rl_controlled_powers.get('rear', 20.0)
                            current_data['side_power_static'] = SIDE_ANTENNA_STATIC_POWER
                            
                            # Neighbor distribution in RL-controlled sectors
                            rl_neighbors = vehicle.antenna_system.rl_controlled_neighbor_distribution
                            current_data['front_neighbors'] = rl_neighbors.get('front', 0)
                            current_data['rear_neighbors'] = rl_neighbors.get('rear', 0)
                    else:
                        current_data['transmissionPower'] = current_data.get('transmissionPower', 20)
                    
                    # Add antenna type for RL agent information
                    current_data['antenna_type'] = ANTENNA_TYPE
                    current_data['rl_controlled_sectors'] = RL_CONTROLLED_SECTORS if ANTENNA_TYPE == "SECTORAL" else ["omnidirectional"]
                    
                    message_data[veh_id] = current_data
                
                # Debug logging for RL communication
                self.rl_log_counter += 1
                if RL_DEBUG_LOGGING and (self.rl_log_counter % RL_LOG_FREQUENCY == 0):
                    print(f"\n[RL DEBUG] === Communication #{self.rl_log_counter} ===")
                    print(f"[RL SEND] Vehicles: {len(message_data)} (batch {msg_idx + 1}/{num_messages})")
                    
                    # Log sample vehicle data
                    if message_data:
                        sample_vehicle = next(iter(message_data.keys()))
                        sample_data = message_data[sample_vehicle]
                        print(f"[RL SEND] Sample vehicle {sample_vehicle}:")
                        print(f"  CBR: {sample_data.get('CBR', 'N/A'):.3f}")
                        print(f"  SINR: {sample_data.get('SINR', 'N/A'):.1f} dB") 
                        print(f"  Neighbors: {sample_data.get('neighbors', 'N/A')}")
                        print(f"  Current Power: {sample_data.get('transmissionPower', 'N/A'):.1f} dBm")
                        print(f"  Current MCS: {sample_data.get('MCS', 'N/A')}")
                        print(f"  Current Beacon Rate: {sample_data.get('beaconRate', 'N/A'):.1f} Hz")
                        print(f"  Antenna Type: {sample_data.get('antenna_type', 'N/A')}")
                        if ANTENNA_TYPE == "SECTORAL":
                            print(f"  Front Power: {sample_data.get('front_power', 'N/A'):.1f} dBm")
                            print(f"  Rear Power: {sample_data.get('rear_power', 'N/A'):.1f} dBm")
                            print(f"  Front Neighbors: {sample_data.get('front_neighbors', 'N/A')}")
                            print(f"  Rear Neighbors: {sample_data.get('rear_neighbors', 'N/A')}")
                
                # Send/receive logic with enhanced debugging
                try:
                    message_json = json.dumps(message_data)
                    message_bytes = message_json.encode('utf-8')
                    
                    if RL_DEBUG_LOGGING and (self.rl_log_counter % RL_LOG_FREQUENCY == 0):
                        print(f"[RL DEBUG] Sending {len(message_bytes)} bytes")
                    
                except Exception as e:
                    print(f"[RL ERROR] JSON encoding failed: {e}")
                    continue
                
                if not self._check_rl_connection():
                    print("[RL ERROR] Failed to establish connection")
                    return {}
                
                try:
                    # Send message with length header
                    msg_length = len(message_bytes)
                    self.rl_client.sendall(msg_length.to_bytes(4, byteorder='little'))
                    self.rl_client.sendall(message_bytes)
                    
                    if RL_DEBUG_LOGGING and (self.rl_log_counter % RL_LOG_FREQUENCY == 0):
                        print(f"[RL DEBUG] Message sent successfully, waiting for response...")
                    
                    # Receive response with enhanced debugging
                    start_time = time.time()
                    response_length_bytes = b''
                    while len(response_length_bytes) < 4 and (time.time() - start_time) < 10:
                        try:
                            chunk = self.rl_client.recv(4 - len(response_length_bytes))
                            if chunk:
                                response_length_bytes += chunk
                            else:
                                time.sleep(0.01)
                        except socket.timeout:
                            time.sleep(0.01)
                    
                    if len(response_length_bytes) < 4:
                        print("[RL ERROR] No response header received from RL server")
                        return {}
                    
                    response_length = int.from_bytes(response_length_bytes, byteorder='little')
                    
                    if RL_DEBUG_LOGGING and (self.rl_log_counter % RL_LOG_FREQUENCY == 0):
                        print(f"[RL DEBUG] Expected response length: {response_length} bytes")
                    
                    response_data = b''
                    while len(response_data) < response_length and (time.time() - start_time) < 10:
                        try:
                            remaining = response_length - len(response_data)
                            chunk = self.rl_client.recv(min(remaining, 8192))
                            if chunk:
                                response_data += chunk
                            else:
                                time.sleep(0.01)
                        except socket.timeout:
                            time.sleep(0.01)
                    
                    if len(response_data) < response_length:
                        print(f"[RL ERROR] Incomplete response received: {len(response_data)}/{response_length} bytes")
                        return {}
                    
                    try:
                        response_json = response_data.decode('utf-8')
                        
                        # Enhanced debug logging for response
                        if RL_DEBUG_LOGGING and (self.rl_log_counter % RL_LOG_FREQUENCY == 0):
                            print(f"[RL DEBUG] Raw response received ({len(response_json)} chars): {response_json[:200]}...")
                        
                        partial_response = json.loads(response_json)
                        
                        # Enhanced debug logging for parsed response
                        if RL_DEBUG_LOGGING and (self.rl_log_counter % RL_LOG_FREQUENCY == 0):
                            print(f"[RL DEBUG] Parsed response type: {type(partial_response)}")
                            print(f"[RL DEBUG] Parsed response keys: {list(partial_response.keys()) if isinstance(partial_response, dict) else 'Not a dict'}")
                        
                        # Check if response has vehicles key
                        if isinstance(partial_response, dict) and 'vehicles' in partial_response:
                            vehicle_responses = partial_response['vehicles']
                            
                            if RL_DEBUG_LOGGING and (self.rl_log_counter % RL_LOG_FREQUENCY == 0):
                                print(f"[RL DEBUG] Found 'vehicles' key with {len(vehicle_responses)} responses")
                            
                            # Process each vehicle response
                            for veh_id, veh_response in vehicle_responses.items():
                                if RL_DEBUG_LOGGING and (self.rl_log_counter % RL_LOG_FREQUENCY == 0):
                                    print(f"[RL DEBUG] Processing response for vehicle {veh_id}: {veh_response}")
                                
                                # Validate response structure
                                if isinstance(veh_response, dict):
                                    # Check for required fields
                                    required_fields = ['transmissionPower', 'MCS', 'beaconRate']
                                    has_all_fields = all(field in veh_response for field in required_fields)
                                    
                                    if has_all_fields:
                                        rl_response['vehicles'][veh_id] = veh_response
                                        
                                        if RL_DEBUG_LOGGING and (self.rl_log_counter % RL_LOG_FREQUENCY == 0):
                                            print(f"[RL DEBUG] Successfully added response for {veh_id}")
                                    else:
                                        print(f"[RL ERROR] Vehicle {veh_id} response missing required fields: {veh_response}")
                                else:
                                    print(f"[RL ERROR] Vehicle {veh_id} response is not a dict: {type(veh_response)}")
                        
                        elif isinstance(partial_response, dict):
                            print(f"[RL ERROR] Response is dict but missing 'vehicles' key. Available keys: {list(partial_response.keys())}")
                        else:
                            print(f"[RL ERROR] Response is not a dict: {type(partial_response)}")
                        
                    except json.JSONDecodeError as e:
                        print(f"[RL ERROR] JSON decode error: {e}")
                        print(f"[RL ERROR] Raw response that failed: {response_data[:200]}")
                    except Exception as e:
                        print(f"[RL ERROR] Response parsing failed: {e}")
                        
                except Exception as e:
                    print(f"[RL ERROR] Communication with RL server failed: {e}")
                    return {}
            
            # Final debug logging summary with enhanced details
            if RL_DEBUG_LOGGING and self.rl_log_counter % RL_LOG_FREQUENCY == 0:
                total_responses = len(rl_response.get('vehicles', {}))
                print(f"[RL DEBUG] === Communication Complete ===")
                print(f"[RL SUMMARY] Total responses received: {total_responses}")
                
                if total_responses > 0:
                    print(f"[RL SUMMARY] Vehicle responses:")
                    for veh_id, response in rl_response['vehicles'].items():
                        print(f"  {veh_id}: Power={response.get('transmissionPower', 'N/A')}, MCS={response.get('MCS', 'N/A')}, Beacon={response.get('beaconRate', 'N/A')}")
                else:
                    print(f"[RL ERROR] No valid responses received from RL server!")
                    
                print(f"[RL NOTE] Parameter updates will be applied NEXT timestamp")
                print("="*50)
            
            return rl_response
            
        except Exception as e:
            print(f"[RL ERROR] Critical communication error: {e}")
            import traceback
            traceback.print_exc()
            return {}
    
    def _check_rl_connection(self):
        """Check and recreate RL connection if needed"""
        if not self.rl_client:
            try:
                self.rl_client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.rl_client.settimeout(30)
                self.rl_client.connect((self.rl_host, self.rl_port))
                return True
            except Exception as e:
                print(f"[RL WARNING] Connection failed: {e}")
                self.rl_client = None
                return False
        return True
    
    def _write_progressive_excel(self, time_idx: int, current_time: float):
        """Update single Excel file with current accumulated results"""
        if EXCEL_UPDATE_FREQUENCY <= 0 or not self.simulation_results:
            return
        
        try:
            df = pd.DataFrame(self.simulation_results)
            
            with pd.ExcelWriter(self.progressive_excel_file, engine='openpyxl') as writer:
                # Main results sheet
                df.to_excel(writer, sheet_name='Results', index=False)
                
                # Enhanced progress summary sheet
                progress_summary = {
                    'Metric': [
                        'Total_Data_Points', 'Current_Timestamp', 'Progress_Percent', 
                        'Avg_Throughput_Mbps', 'Avg_Latency_ms', 'Avg_PER', 'Avg_SINR_dB',
                        'Avg_Neighbors', 'Avg_CBR', 'Target_PER_Achievement_Percent',
                        'L3_Enabled', 'Avg_L3_PDR', 'Avg_Routing_Overhead',
                        'SDN_Enabled', 'Avg_Flow_Table_Size', 'Traffic_Engineering_Events'
                    ],
                    'Value': [
                        len(df),
                        current_time,
                        (time_idx + 1) / len(set(data['time'] for data in self.mobility_data)) * 100,
                        df['Throughput'].mean(),
                        df['Latency'].mean(),
                        df['PER'].mean(),
                        df['SINR'].mean(),
                        df['NeighborNumbers'].mean(),
                        df['CBR'].mean(),
                        df['TargetPER_Met'].value_counts().get('Yes', 0) / len(df) * 100,
                        self.config.enable_layer3,
                        df['L3_PDR'].mean() if 'L3_PDR' in df.columns else 0,
                        df['RoutingOverheadRatio'].mean() if 'RoutingOverheadRatio' in df.columns else 0,
                        self.config.enable_sdn,
                        df['FlowTableSize'].mean() if 'FlowTableSize' in df.columns else 0,
                        self.sdn_performance_monitor.get('traffic_engineering_events', 0)
                    ]
                }
                progress_df = pd.DataFrame(progress_summary)
                progress_df.to_excel(writer, sheet_name='Progress', index=False)
            
            print(f"[EXCEL UPDATE] Enhanced progressive Excel updated: {self.progressive_excel_file} (t={current_time:.1f}s, {len(df)} records)")
            
        except Exception as e:
            print(f"[WARNING] Failed to update progressive Excel at t={current_time:.1f}s: {e}")
    
    def save_results(self, filename: str = None):
        """ENHANCED: Save results with comprehensive IEEE 802.11bd statistics including PHY/MAC layer details"""
        if not filename:
            filename = self.final_excel_file
        
        if not self.simulation_results:
            print("[ERROR] No results to save.")
            return
        
        df = pd.DataFrame(self.simulation_results)
        
        with pd.ExcelWriter(filename, engine='openpyxl') as writer:
            # Main results sheet
            df.to_excel(writer, sheet_name='Detailed_Results', index=False)
            
            # ENHANCED: Comprehensive summary analysis with PHY/MAC layer breakdowns
            summary_stats = {
                'Metric': [
                    # Basic simulation info
                    'Total_Data_Points', 'Unique_Vehicles', 'Time_Points', 'Scenario_Type', 'Density_Category',
                    'Original_Duration_s', 'FCD_Reload_Count', 'Total_Extended_Duration_s', 'Vehicle_ID_Strategy',
                    
                    # SINR Statistics
                    'Avg_SINR_dB', 'Min_SINR_dB', 'Max_SINR_dB', 'Std_SINR_dB',
                    
                    # CBR Statistics
                    'Avg_CBR', 'Min_CBR', 'Max_CBR', 'Std_CBR',
                    
                    # Neighbor Statistics
                    'Avg_Neighbors', 'Min_Neighbors', 'Max_Neighbors', 'Std_Neighbors',
                    
                    # PDR Statistics (Packet Delivery Ratio)
                    'Avg_PDR_Percent', 'Min_PDR_Percent', 'Max_PDR_Percent', 'Std_PDR_Percent',
                    
                    # PER Statistics (Packet Error Rate)
                    'Avg_PER', 'Min_PER', 'Max_PER', 'Std_PER',
                    
                    # PHY Layer Throughput Statistics
                    'Avg_PHY_Throughput_Mbps', 'Min_PHY_Throughput_Mbps', 'Max_PHY_Throughput_Mbps', 'Std_PHY_Throughput_Mbps',
                    
                    # MAC Layer Throughput Statistics  
                    'Avg_MAC_Throughput_Mbps', 'Min_MAC_Throughput_Mbps', 'Max_MAC_Throughput_Mbps', 'Std_MAC_Throughput_Mbps',
                    
                    # Overall Throughput Statistics
                    'Avg_Throughput_Mbps', 'Min_Throughput_Mbps', 'Max_Throughput_Mbps', 'Std_Throughput_Mbps',
                    
                    # PHY Layer Latency Statistics
                    'Avg_PHY_Latency_ms', 'Min_PHY_Latency_ms', 'Max_PHY_Latency_ms', 'Std_PHY_Latency_ms',
                    
                    # MAC Layer Latency Statistics
                    'Avg_MAC_Latency_ms', 'Min_MAC_Latency_ms', 'Max_MAC_Latency_ms', 'Std_MAC_Latency_ms',
                    
                    # Total End-to-End Latency Statistics
                    'Avg_Total_Latency_ms', 'Min_Total_Latency_ms', 'Max_Total_Latency_ms', 'Std_Total_Latency_ms',
                    
                    # L3/SDN Latency Statistics
                    'Avg_L3_Delay_ms', 'Min_L3_Delay_ms', 'Max_L3_Delay_ms', 'Std_L3_Delay_ms',
                    'Avg_SDN_Delay_ms', 'Min_SDN_Delay_ms', 'Max_SDN_Delay_ms', 'Std_SDN_Delay_ms',
                    
                    # MAC Efficiency Statistics
                    'Avg_MAC_Efficiency', 'Min_MAC_Efficiency', 'Max_MAC_Efficiency', 'Std_MAC_Efficiency',
                    
                    # Collision Probability Statistics
                    'Avg_Collision_Prob', 'Min_Collision_Prob', 'Max_Collision_Prob', 'Std_Collision_Prob',
                    
                    # Communication Range Statistics
                    'Avg_CommRange_m', 'Min_CommRange_m', 'Max_CommRange_m', 'Std_CommRange_m',
                    
                    # IEEE 802.11bd Performance Indicators
                    'Throughput_Improvement_Factor', 'Target_PER_Achievement_Percent', 'Target_PDR_Achievement_Percent',
                    
                    # MCS Distribution Analysis
                    'Most_Used_MCS', 'Avg_MCS', 'Min_MCS', 'Max_MCS',
                    
                    # Power Control Statistics
                    'Avg_TX_Power_dBm', 'Min_TX_Power_dBm', 'Max_TX_Power_dBm', 'Std_TX_Power_dBm',
                    
                    # Beacon Rate Statistics
                    'Avg_Beacon_Rate_Hz', 'Min_Beacon_Rate_Hz', 'Max_Beacon_Rate_Hz', 'Std_Beacon_Rate_Hz',
                    
                    # Data Rate Statistics
                    'Avg_PHY_DataRate_Mbps', 'Min_PHY_DataRate_Mbps', 'Max_PHY_DataRate_Mbps', 'Std_PHY_DataRate_Mbps'
                ],
                'Value': [
                    # Basic simulation info
                    len(df), df['VehicleID'].nunique(), df['Timestamp'].nunique(),
                    self.scenario_info.get('scenario_type', 'Unknown'), self.scenario_info.get('density_category', 'Unknown'),
                    self.original_simulation_duration, self.fcd_reload_count, self.total_simulation_duration, self.fcd_reload_strategy,
                    
                    # SINR Statistics
                    df['SINR'].mean(), df['SINR'].min(), df['SINR'].max(), df['SINR'].std(),
                    
                    # CBR Statistics
                    df['CBR'].mean(), df['CBR'].min(), df['CBR'].max(), df['CBR'].std(),
                    
                    # Neighbor Statistics
                    df['NeighborNumbers'].mean(), df['NeighborNumbers'].min(), df['NeighborNumbers'].max(), df['NeighborNumbers'].std(),
                    
                    # PDR Statistics
                    df['PDR'].mean() * 100, df['PDR'].min() * 100, df['PDR'].max() * 100, df['PDR'].std() * 100,
                    
                    # PER Statistics
                    df['PER'].mean(), df['PER'].min(), df['PER'].max(), df['PER'].std(),
                    
                    # PHY Layer Throughput Statistics
                    df['PHYThroughput'].mean() if 'PHYThroughput' in df.columns else 0,
                    df['PHYThroughput'].min() if 'PHYThroughput' in df.columns else 0,
                    df['PHYThroughput'].max() if 'PHYThroughput' in df.columns else 0,
                    df['PHYThroughput'].std() if 'PHYThroughput' in df.columns else 0,
                    
                    # MAC Layer Throughput Statistics
                    df['MACThroughput'].mean() if 'MACThroughput' in df.columns else df['Throughput'].mean(),
                    df['MACThroughput'].min() if 'MACThroughput' in df.columns else df['Throughput'].min(),
                    df['MACThroughput'].max() if 'MACThroughput' in df.columns else df['Throughput'].max(),
                    df['MACThroughput'].std() if 'MACThroughput' in df.columns else df['Throughput'].std(),
                    
                    # Overall Throughput Statistics
                    df['Throughput'].mean(), df['Throughput'].min(), df['Throughput'].max(), df['Throughput'].std(),
                    
                    # PHY Layer Latency Statistics
                    df['PHY_Latency_ms'].mean() if 'PHY_Latency_ms' in df.columns else (df['Latency'] * 0.2).mean(),
                    df['PHY_Latency_ms'].min() if 'PHY_Latency_ms' in df.columns else (df['Latency'] * 0.2).min(),
                    df['PHY_Latency_ms'].max() if 'PHY_Latency_ms' in df.columns else (df['Latency'] * 0.2).max(),
                    df['PHY_Latency_ms'].std() if 'PHY_Latency_ms' in df.columns else (df['Latency'] * 0.2).std(),
                    
                    # MAC Layer Latency Statistics
                    df['MAC_Latency_ms'].mean() if 'MAC_Latency_ms' in df.columns else (df['Latency'] * 0.8).mean(),
                    df['MAC_Latency_ms'].min() if 'MAC_Latency_ms' in df.columns else (df['Latency'] * 0.8).min(),
                    df['MAC_Latency_ms'].max() if 'MAC_Latency_ms' in df.columns else (df['Latency'] * 0.8).max(),
                    df['MAC_Latency_ms'].std() if 'MAC_Latency_ms' in df.columns else (df['Latency'] * 0.8).std(),
                    
                    # Total End-to-End Latency Statistics
                    df['Latency'].mean(), df['Latency'].min(), df['Latency'].max(), df['Latency'].std(),
                    
                    # L3/SDN Latency Statistics
                    df['L3_Routing_Delay_ms'].mean() if 'L3_Routing_Delay_ms' in df.columns else 0,
                    df['L3_Routing_Delay_ms'].min() if 'L3_Routing_Delay_ms' in df.columns else 0,
                    df['L3_Routing_Delay_ms'].max() if 'L3_Routing_Delay_ms' in df.columns else 0,
                    df['L3_Routing_Delay_ms'].std() if 'L3_Routing_Delay_ms' in df.columns else 0,
                    df['SDN_Processing_Delay_ms'].mean() if 'SDN_Processing_Delay_ms' in df.columns else 0,
                    df['SDN_Processing_Delay_ms'].min() if 'SDN_Processing_Delay_ms' in df.columns else 0,
                    df['SDN_Processing_Delay_ms'].max() if 'SDN_Processing_Delay_ms' in df.columns else 0,
                    df['SDN_Processing_Delay_ms'].std() if 'SDN_Processing_Delay_ms' in df.columns else 0,
                    
                    # MAC Efficiency Statistics
                    df['MACEfficiency'].mean(), df['MACEfficiency'].min(), df['MACEfficiency'].max(), df['MACEfficiency'].std(),
                    
                    # Collision Probability Statistics
                    df['CollisionProb'].mean(), df['CollisionProb'].min(), df['CollisionProb'].max(), df['CollisionProb'].std(),
                    
                    # Communication Range Statistics
                    df['CommRange'].mean(), df['CommRange'].min(), df['CommRange'].max(), df['CommRange'].std(),
                    
                    # IEEE 802.11bd Performance Indicators
                    df['ThroughputImprovement'].mean() if 'ThroughputImprovement' in df.columns else 1.0,
                    df['TargetPER_Met'].value_counts().get('Yes', 0) / len(df) * 100,
                    df['TargetPDR_Met'].value_counts().get('Yes', 0) / len(df) * 100,
                    
                    # MCS Distribution Analysis
                    df['MCS'].mode().iloc[0] if not df['MCS'].mode().empty else 'N/A',
                    df['MCS'].mean(), df['MCS'].min(), df['MCS'].max(),
                    
                    # Power Control Statistics
                    df['PowerTx'].mean(), df['PowerTx'].min(), df['PowerTx'].max(), df['PowerTx'].std(),
                    
                    # Beacon Rate Statistics
                    df['BeaconRate'].mean(), df['BeaconRate'].min(), df['BeaconRate'].max(), df['BeaconRate'].std(),
                    
                    # Data Rate Statistics
                    df['PHYDataRate'].mean(), df['PHYDataRate'].min(), df['PHYDataRate'].max(), df['PHYDataRate'].std()
                ]
            }
            summary_df = pd.DataFrame(summary_stats)
            summary_df.to_excel(writer, sheet_name='Performance_Summary', index=False)
            
            # Layer 3 specific analysis (if enabled)
            if self.config.enable_layer3 and 'L3_PDR' in df.columns:
                l3_analysis = {
                    'Routing_Protocol': [self.config.routing_protocol],
                    'Average_L3_PDR': [df['L3_PDR'].mean()],
                    'L3_PDR_Std': [df['L3_PDR'].std()],
                    'Route_Discovery_Success_Rate': [df['RouteDiscoverySuccess'].sum() / max(1, df['RouteDiscoveryAttempts'].sum()) if 'RouteDiscoverySuccess' in df.columns else 0],
                    'Average_Hop_Count': [df['HopCount'].mean() if 'HopCount' in df.columns else 0],
                    'Routing_Overhead_Avg': [df['RoutingOverheadRatio'].mean() if 'RoutingOverheadRatio' in df.columns else 0],
                    'Packet_Forwarding_Efficiency': [df['PacketsForwarded'].sum() / max(1, df['PacketsGenerated'].sum()) if 'PacketsForwarded' in df.columns else 0],
                    'End_to_End_Delay_Avg_ms': [df['EndToEndDelay_ms'].mean() if 'EndToEndDelay_ms' in df.columns else 0]
                }
                l3_analysis_df = pd.DataFrame(l3_analysis)
                l3_analysis_df.to_excel(writer, sheet_name='L3_Performance_Analysis', index=False)
            
            # SDN specific analysis (if enabled)
            if self.config.enable_sdn and 'FlowTableSize' in df.columns:
                sdn_analysis = {
                    'Controller_Type': [self.config.sdn_controller_type],
                    'Average_Flow_Table_Size': [df['FlowTableSize'].mean()],
                    'Max_Flow_Table_Size': [df['FlowTableSize'].max()],
                    'Flow_Table_Utilization': [df['ActiveFlows'].sum() / max(1, df['FlowTableSize'].sum()) if 'ActiveFlows' in df.columns else 0],
                    'Average_Controller_Latency_ms': [df['ControllerLatency_ms'].mean() if 'ControllerLatency_ms' in df.columns else 0],
                    'Traffic_Engineering_Events': [self.sdn_performance_monitor.get('traffic_engineering_events', 0)]
                }
                sdn_analysis_df = pd.DataFrame(sdn_analysis)
                sdn_analysis_df.to_excel(writer, sheet_name='SDN_Performance_Analysis', index=False)
            
            # Continue with other analysis sheets from script 2...
            # [Add more analysis sheets as needed]
        
        print(f"[INFO] Enhanced IEEE 802.11bd results with comprehensive analysis saved to {filename}")
        if self.fcd_reload_count > 1:
            print(f"[INFO] Results include {self.fcd_reload_count} episodes with detailed layer-by-layer analysis")
        return filename
    
    def cleanup(self):
        """Enhanced cleanup with live plot management and proper L3 cleanup"""
        if self.rl_client:
            try:
                self.rl_client.close()
                print("[RL] Connection closed")
            except:
                pass
        
        # Cleanup SDN controller
        if self.sdn_controller:
            print("[SDN] Controller cleaned up")
        
        # FIXED: Cleanup routing protocols properly
        if self.config.enable_layer3:
            for vehicle in self.vehicles.values():
                if vehicle.routing_protocol:
                    try:
                        # Use universal cleanup method
                        self._cleanup_routing_protocol(vehicle, time.time())
                    except Exception as e:
                        print(f"[WARNING] Failed to cleanup routing protocol for {vehicle.vehicle_id}: {e}")
            print(f"[L3] {self.config.routing_protocol} routing protocol cleaned up")
        
        # Cleanup live plot manager
        if hasattr(self, 'live_plot_manager') and self.live_plot_manager:
            self.live_plot_manager.close_all_plots()
            print("[LIVE PLOTS] Live plot manager cleaned up")
        
        # Cleanup standard visualization
        if hasattr(self, 'visualizer') and self.visualizer:
            self.visualizer.close()
            print("[VISUALIZER] Standard visualization cleaned up")
            
    def save_attack_results(self, filename: str = None):
        """Save attack simulation results and detection dataset"""
        if not self.attack_manager or not ENABLE_ATTACK_SIMULATION:
            print("[INFO] Attack simulation not enabled")
            return
        
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"VANET_DoS_DDoS_Dataset_{timestamp}.xlsx"
        
        with pd.ExcelWriter(filename, engine='openpyxl') as writer:
            
            # 1. Attack Detection Dataset (Main ML Dataset)
            if self.attack_dataset:
                attack_df = pd.DataFrame(self.attack_dataset)
                attack_df.to_excel(writer, sheet_name='Attack_Detection_Dataset', index=False)
            
            # 2. Attack Summary
            attack_summary = self.attack_manager.get_attack_summary()
            summary_data = []
            for key, value in attack_summary.items():
                summary_data.append({'Metric': key, 'Value': str(value)})
            
            if summary_data:
                summary_df = pd.DataFrame(summary_data)
                summary_df.to_excel(writer, sheet_name='Attack_Summary', index=False)
            
            # 3. Attacker vs Normal Vehicle Comparison
            comparison_data = []
            for vehicle_id, vehicle in self.vehicles.items():
                metrics = self.attack_manager.get_attack_metrics(vehicle_id)
                comparison_data.append({
                    'Vehicle_ID': vehicle_id,
                    'Is_Attacker': metrics.is_attacker,
                    'Attack_Type': metrics.attack_type.value,
                    'Attack_Intensity': metrics.attack_intensity,
                    'Beacon_Rate_Variance': metrics.beacon_rate_variance,
                    'TX_Power_Anomaly': metrics.tx_power_anomaly,
                    'Network_Impact_Score': metrics.neighbor_disruption_ratio,
                    'Anomaly_Score': metrics.detection_features.get('anomaly_score', 0.0)
                })
            
            if comparison_data:
                comparison_df = pd.DataFrame(comparison_data)
                comparison_df.to_excel(writer, sheet_name='Attacker_Comparison', index=False)
            
            # 4. Feature Importance Analysis
            if self.attack_dataset:
                feature_columns = [col for col in attack_df.columns if col not in ['vehicle_id', 'is_attacker', 'attack_type', 'timestamp']]
                
                # Calculate feature statistics by attack type
                feature_stats = []
                for feature in feature_columns:
                    normal_values = attack_df[attack_df['is_attacker'] == False][feature].values
                    attack_values = attack_df[attack_df['is_attacker'] == True][feature].values
                    
                    if len(normal_values) > 0 and len(attack_values) > 0:
                        feature_stats.append({
                            'Feature': feature,
                            'Normal_Mean': np.mean(normal_values),
                            'Normal_Std': np.std(normal_values),
                            'Attack_Mean': np.mean(attack_values),
                            'Attack_Std': np.std(attack_values),
                            'Difference_Score': abs(np.mean(attack_values) - np.mean(normal_values)) / (np.std(normal_values) + 1e-6)
                        })
                
                if feature_stats:
                    feature_df = pd.DataFrame(feature_stats)
                    feature_df = feature_df.sort_values('Difference_Score', ascending=False)
                    feature_df.to_excel(writer, sheet_name='Feature_Importance', index=False)
            
            # 5. Attack Timeline Analysis
            timeline_data = []
            for result in self.simulation_results:
                if result.get('VehicleID') in self.attack_manager.attackers:
                    timeline_data.append({
                        'Timestamp': result['Timestamp'],
                        'Vehicle_ID': result['VehicleID'],
                        'Beacon_Rate': result.get('BeaconRate', 0),
                        'TX_Power': result.get('PowerTx', 0),
                        'Throughput': result.get('Throughput', 0),
                        'CBR': result.get('CBR', 0),
                        'SINR': result.get('SINR', 0),
                        'PER': result.get('PER', 0),
                        'Attack_Active': 1  # Indicates attack period
                    })
            
            if timeline_data:
                timeline_df = pd.DataFrame(timeline_data)
                timeline_df.to_excel(writer, sheet_name='Attack_Timeline', index=False)
        
        print(f"[INFO] Attack simulation results and ML dataset saved to {filename}")
        return filename
    
    def _update_all_routing_protocols(self, current_vehicles: Set[str], current_time: float):
        """Comprehensive routing protocol update with proper parameter passing"""
        if not self.config.enable_layer3:
            return
        
        for vehicle_id in current_vehicles:
            if vehicle_id not in self.vehicles:
                continue
            
            vehicle = self.vehicles[vehicle_id]
            if not vehicle.routing_protocol:
                continue
            
            try:
                # Update routing protocol state
                self._update_routing_protocol_state(vehicle, current_time)
                
                # Protocol-specific periodic tasks
                if isinstance(vehicle.routing_protocol, OLSRRoutingProtocol):
                    self._update_olsr_protocol(vehicle, current_time)
                elif isinstance(vehicle.routing_protocol, AODVRoutingProtocol):
                    self._update_aodv_protocol(vehicle, current_time)
                elif isinstance(vehicle.routing_protocol, GeographicRoutingProtocol):
                    self._update_geographic_protocol(vehicle, current_time)
                    
            except Exception as e:
                print(f"[ERROR] Failed to update routing protocol for {vehicle_id}: {e}")

    def _update_olsr_protocol(self, vehicle: 'VehicleState', current_time: float):
        """Update OLSR protocol with proper timing"""
        protocol = vehicle.routing_protocol
        
        # Generate HELLO if needed
        if current_time - protocol.last_hello_time >= protocol.hello_interval:
            hello_packet = protocol.generate_hello_packet(current_time)
            if hello_packet:
                self._broadcast_routing_packet(vehicle.vehicle_id, hello_packet, current_time)
        
        # Generate TC if needed and we're an MPR
        if (current_time - protocol.last_tc_time >= protocol.tc_interval and 
            protocol.mpr_selector_set):
            tc_packet = protocol.generate_tc_packet(current_time)
            if tc_packet:
                self._broadcast_routing_packet(vehicle.vehicle_id, tc_packet, current_time)
        
        # Cleanup expired entries
        protocol.cleanup_expired_entries(current_time)
    
    def _update_aodv_protocol(self, vehicle: 'VehicleState', current_time: float):
        """Update AODV protocol with proper timing"""
        protocol = vehicle.routing_protocol
        
        # Generate HELLO if needed
        hello_packet = protocol.generate_hello_packet(current_time)
        if hello_packet:
            self._broadcast_routing_packet(vehicle.vehicle_id, hello_packet, current_time)
        
        # Cleanup expired routes
        protocol.cleanup_expired_routes(current_time)
    
    def _update_geographic_protocol(self, vehicle: 'VehicleState', current_time: float):
        """Update Geographic protocol with proper timing"""
        protocol = vehicle.routing_protocol
        
        # Get current position
        vehicle_data = next((data for data in self.mobility_data 
                           if data['time'] == current_time and data['id'] == vehicle.vehicle_id), None)
        if vehicle_data:
            current_pos = (vehicle_data['x'], vehicle_data['y'])
            beacon = protocol.generate_position_beacon(
                current_pos, current_time,
                vehicle_data.get('speed', 0), vehicle_data.get('angle', 0)
            )
            if beacon:
                self._broadcast_routing_packet(vehicle.vehicle_id, beacon, current_time)
        
        # Cleanup expired positions
        protocol.cleanup_expired_positions(current_time)
        
    def _safe_call_routing_method(self, protocol, method_name: str, *args, **kwargs):
        """Safely call routing protocol method with error handling"""
        try:
            if hasattr(protocol, method_name):
                method = getattr(protocol, method_name)
                return method(*args, **kwargs)
            else:
                print(f"[WARNING] Protocol {type(protocol).__name__} missing method: {method_name}")
                return None
        except Exception as e:
            print(f"[ERROR] Failed to call {method_name} on {type(protocol).__name__}: {e}")
            return None
    
class VANETVisualizer:
    """Real-time VANET simulation visualizer for Spyder"""
    
    def __init__(self, config: SimulationConfig, scenario_info: Dict, enable_animation: bool = True):
        self.config = config
        self.scenario_info = scenario_info
        self.enable_animation = enable_animation
        
        # Visualization settings
        self.update_interval = 100  # milliseconds
        self.show_range = True
        self.show_connections = True
        self.show_attack_info = ENABLE_ATTACK_SIMULATION
        self.max_history = 50  # Vehicle trail length
        
        # Color schemes
        self.mcs_colors = plt.cm.viridis(np.linspace(0, 1, 11))  # 0-10 MCS levels
        self.attack_colors = {
            'NORMAL': 'lightblue',
            'BEACON_FLOODING': 'red',
            'HIGH_POWER_JAMMING': 'orange', 
            'ASYNC_BEACON': 'purple',
            'COMBINED': 'darkred'
        }
        
        # Beacon rate size mapping (radius multiplier)
        self.beacon_size_map = {
            (0, 5): 0.3,     # Very low beacon rate
            (5, 10): 0.5,    # Low beacon rate
            (10, 20): 0.7,   # Normal beacon rate
            (20, 50): 1.0,   # High beacon rate
            (50, 100): 1.3,  # Very high beacon rate (flooding)
            (100, 200): 1.6  # Extreme beacon rate
        }
        
        # Data storage
        self.vehicle_positions = {}
        self.vehicle_history = {}
        self.current_time = 0
        self.performance_history = []
        
        # Initialize plots
        self._setup_plots()
        
        print("[VISUALIZER] Real-time VANET visualization initialized")
    
    def _setup_plots(self):
        """Setup the matplotlib figure and subplots"""
        # Create figure with subplots
        self.fig = plt.figure(figsize=(16, 10))
        self.fig.suptitle('IEEE 802.11bd VANET Simulation - Real-time Visualization', fontsize=14, fontweight='bold')
        
        # Create grid layout
        gs = gridspec.GridSpec(3, 4, figure=self.fig, height_ratios=[3, 1, 1], width_ratios=[3, 1, 1, 1])
        
        # Main road visualization
        self.ax_main = self.fig.add_subplot(gs[:, :2])
        self.ax_main.set_title('Vehicle Positions and Communication')
        self.ax_main.set_xlabel('X Position (m)')
        self.ax_main.set_ylabel('Y Position (m)')
        self.ax_main.grid(True, alpha=0.3)
        self.ax_main.set_aspect('equal')
        
        # MCS distribution
        self.ax_mcs = self.fig.add_subplot(gs[0, 2])
        self.ax_mcs.set_title('MCS Distribution')
        self.ax_mcs.set_xlabel('MCS Level')
        self.ax_mcs.set_ylabel('Vehicle Count')
        
        # Performance metrics
        self.ax_perf = self.fig.add_subplot(gs[0, 3])
        self.ax_perf.set_title('Avg Performance')
        
        # Beacon rate distribution
        self.ax_beacon = self.fig.add_subplot(gs[1, 2])
        self.ax_beacon.set_title('Beacon Rate Dist.')
        self.ax_beacon.set_xlabel('Beacon Rate (Hz)')
        self.ax_beacon.set_ylabel('Count')
        
        # Attack status (if enabled)
        if self.show_attack_info:
            self.ax_attack = self.fig.add_subplot(gs[1, 3])
            self.ax_attack.set_title('Attack Status')
        else:
            self.ax_info = self.fig.add_subplot(gs[1, 3])
            self.ax_info.set_title('Simulation Info')
        
        # Network statistics
        self.ax_network = self.fig.add_subplot(gs[2, 2:])
        self.ax_network.set_title('Network Statistics Over Time')
        self.ax_network.set_xlabel('Time (s)')
        
        # Create legends
        self._create_legends()
        
        # Set up road boundaries based on scenario
        if self.scenario_info:
            x_range = self.scenario_info.get('x_range', (0, 1000))
            y_range = self.scenario_info.get('y_range', (0, 100))
            
            # Add some padding
            x_padding = (x_range[1] - x_range[0]) * 0.1
            y_padding = (y_range[1] - y_range[0]) * 0.1
            
            self.ax_main.set_xlim(x_range[0] - x_padding, x_range[1] + x_padding)
            self.ax_main.set_ylim(y_range[0] - y_padding, y_range[1] + y_padding)
            
            # Draw road boundaries
            road_width = y_range[1] - y_range[0]
            for i in range(0, int(road_width // 3.5)):  # Lane markings every 3.5m
                y_pos = y_range[0] + i * 3.5
                self.ax_main.axhline(y=y_pos, color='yellow', linestyle='--', alpha=0.5, linewidth=1)
        
        plt.tight_layout()
        
        # Enable interactive mode for real-time updates
        plt.ion()
        self.fig.show()
    
    def _create_legends(self):
        """Create color legends for MCS and other indicators"""
        # MCS Color Legend
        mcs_legend_elements = []
        for i in range(11):  # MCS 0-10
            color = self.mcs_colors[i]
            mcs_legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', 
                                                markerfacecolor=color, markersize=8, label=f'MCS {i}'))
        
        # Add legend to MCS subplot
        self.ax_mcs.legend(handles=mcs_legend_elements[:6], loc='upper right', fontsize=8)
        
        # Attack type legend (if enabled)
        if self.show_attack_info:
            attack_legend_elements = []
            for attack_type, color in self.attack_colors.items():
                attack_legend_elements.append(plt.Line2D([0], [0], marker='s', color='w',
                                                        markerfacecolor=color, markersize=8, 
                                                        label=attack_type.replace('_', ' ')))
            
            # Add attack legend to main plot
            self.ax_main.legend(handles=attack_legend_elements, loc='upper left', fontsize=8)
    
    def _get_beacon_size(self, beacon_rate: float) -> float:
        """Get marker size based on beacon rate"""
        for (min_rate, max_rate), size in self.beacon_size_map.items():
            if min_rate <= beacon_rate < max_rate:
                return size * 30  # Base marker size
        return 50  # Default size
    
    def _get_vehicle_color(self, vehicle_id: str, mcs: int, attack_manager=None) -> str:
        """Get vehicle color based on MCS and attack status"""
        if self.show_attack_info and attack_manager:
            if attack_manager.is_attacker(vehicle_id):
                attack_metrics = attack_manager.get_attack_metrics(vehicle_id)
                return self.attack_colors.get(attack_metrics.attack_type.value, 'red')
        
        # Normal vehicle - color by MCS
        mcs_index = max(0, min(10, int(mcs)))
        return self.mcs_colors[mcs_index]
    
    def update_visualization(self, current_time: float, vehicles: Dict, attack_manager=None):
        """Update the visualization with current simulation state"""
        self.current_time = current_time
        
        # Clear previous plots
        self.ax_main.clear()
        self.ax_mcs.clear()
        self.ax_beacon.clear()
        self.ax_perf.clear()
        if hasattr(self, 'ax_attack'):
            self.ax_attack.clear()
        
        # Setup main plot again
        self.ax_main.set_title(f'Vehicle Positions at t={current_time:.1f}s')
        self.ax_main.set_xlabel('X Position (m)')
        self.ax_main.set_ylabel('Y Position (m)')
        self.ax_main.grid(True, alpha=0.3)
        
        # Collect data for analysis
        vehicle_data = []
        mcs_counts = [0] * 11
        beacon_rates = []
        throughputs = []
        cbrs = []
        sinrs = []
        attack_counts = {'NORMAL': 0, 'BEACON_FLOODING': 0, 'HIGH_POWER_JAMMING': 0, 'ASYNC_BEACON': 0}
        
        # Draw vehicles
        for vehicle_id, vehicle in vehicles.items():
            if not hasattr(vehicle, 'x') or vehicle.x is None:
                continue
                
            # Get vehicle properties
            x, y = vehicle.x, vehicle.y
            mcs = getattr(vehicle, 'mcs', 1)
            beacon_rate = getattr(vehicle, 'beacon_rate', 10.0)
            comm_range = getattr(vehicle, 'comm_range', 100)
            tx_power = getattr(vehicle, 'transmission_power', 20)
            
            # Get performance metrics
            throughput = getattr(vehicle, 'current_throughput', 0) / 1e6  # Convert to Mbps
            cbr = getattr(vehicle, 'current_cbr', 0)
            sinr = getattr(vehicle, 'current_snr', 0)
            
            # Collect data
            vehicle_data.append({
                'id': vehicle_id, 'x': x, 'y': y, 'mcs': mcs, 
                'beacon_rate': beacon_rate, 'throughput': throughput,
                'cbr': cbr, 'sinr': sinr
            })
            
            mcs_counts[max(0, min(10, int(mcs)))] += 1
            beacon_rates.append(beacon_rate)
            throughputs.append(throughput)
            cbrs.append(cbr)
            sinrs.append(sinr)
            
            # Get vehicle color and size
            color = self._get_vehicle_color(vehicle_id, mcs, attack_manager)
            size = self._get_beacon_size(beacon_rate)
            
            # Track attack status
            if attack_manager and attack_manager.is_attacker(vehicle_id):
                attack_metrics = attack_manager.get_attack_metrics(vehicle_id)
                attack_type = attack_metrics.attack_type.value
                attack_counts[attack_type] = attack_counts.get(attack_type, 0) + 1
            else:
                attack_counts['NORMAL'] += 1
            
            # Draw vehicle
            self.ax_main.scatter(x, y, c=[color], s=size, alpha=0.8, edgecolors='black', linewidth=0.5)
            
            # Draw communication range
            if self.show_range:
                range_circle = plt.Circle((x, y), comm_range, fill=False, 
                                        color=color, alpha=0.3, linestyle='--', linewidth=1)
                self.ax_main.add_patch(range_circle)
            
            # Add vehicle ID label for selected vehicles
            if len(vehicles) <= 20:  # Only show labels if not too crowded
                self.ax_main.annotate(vehicle_id[-3:], (x, y), xytext=(5, 5), 
                                    textcoords='offset points', fontsize=6, alpha=0.7)
        
        # Draw connections between neighbors
        if self.show_connections and len(vehicles) <= 30:  # Avoid clutter with too many vehicles
            for vehicle_id, vehicle in vehicles.items():
                if hasattr(vehicle, 'neighbors') and vehicle.neighbors:
                    for neighbor in vehicle.neighbors[:5]:  # Limit to 5 closest neighbors
                        neighbor_id = neighbor['id']
                        if neighbor_id in vehicles:
                            neighbor_vehicle = vehicles[neighbor_id]
                            if hasattr(neighbor_vehicle, 'x'):
                                self.ax_main.plot([vehicle.x, neighbor_vehicle.x], 
                                                [vehicle.y, neighbor_vehicle.y], 
                                                'gray', alpha=0.2, linewidth=0.5)
        
        # Update MCS distribution
        self.ax_mcs.bar(range(11), mcs_counts, color=self.mcs_colors)
        self.ax_mcs.set_xlabel('MCS Level')
        self.ax_mcs.set_ylabel('Count')
        self.ax_mcs.set_xticks(range(11))
        
        # Update performance metrics
        if throughputs:
            perf_labels = ['Avg Tput\n(Mbps)', 'Avg CBR', 'Avg SINR\n(dB)']
            perf_values = [np.mean(throughputs), np.mean(cbrs), np.mean(sinrs)]
            colors = ['green', 'orange', 'blue']
            
            bars = self.ax_perf.bar(range(3), perf_values, color=colors, alpha=0.7)
            self.ax_perf.set_xticks(range(3))
            self.ax_perf.set_xticklabels(perf_labels, fontsize=8)
            
            # Add value labels on bars
            for bar, value in zip(bars, perf_values):
                height = bar.get_height()
                self.ax_perf.text(bar.get_x() + bar.get_width()/2., height + height*0.05,
                                f'{value:.2f}', ha='center', va='bottom', fontsize=8)
        
        # Update beacon rate distribution
        if beacon_rates:
            self.ax_beacon.hist(beacon_rates, bins=10, alpha=0.7, color='skyblue', edgecolor='black')
            self.ax_beacon.set_xlabel('Beacon Rate (Hz)')
            self.ax_beacon.set_ylabel('Count')
        
        # Update attack status
        if self.show_attack_info and hasattr(self, 'ax_attack'):
            attack_types = list(attack_counts.keys())
            attack_values = list(attack_counts.values())
            colors = [self.attack_colors.get(attack_type, 'gray') for attack_type in attack_types]
            
            self.ax_attack.pie(attack_values, labels=[t.replace('_', '\n') for t in attack_types], 
                             colors=colors, autopct='%1.0f', startangle=90, textprops={'fontsize': 8})
        
        # Update network statistics over time
        if hasattr(self, 'performance_history'):
            self.performance_history.append({
                'time': current_time,
                'avg_throughput': np.mean(throughputs) if throughputs else 0,
                'avg_cbr': np.mean(cbrs) if cbrs else 0,
                'avg_sinr': np.mean(sinrs) if sinrs else 0,
                'vehicle_count': len(vehicles)
            })
            
            # Keep only recent history
            if len(self.performance_history) > 100:
                self.performance_history = self.performance_history[-100:]
            
            if len(self.performance_history) > 1:
                times = [h['time'] for h in self.performance_history]
                throughputs_hist = [h['avg_throughput'] for h in self.performance_history]
                cbrs_hist = [h['avg_cbr'] for h in self.performance_history]
                
                self.ax_network.clear()
                self.ax_network.plot(times, throughputs_hist, 'g-', label='Avg Throughput (Mbps)', linewidth=2)
                
                # Secondary y-axis for CBR
                ax2 = self.ax_network.twinx()
                ax2.plot(times, cbrs_hist, 'r-', label='Avg CBR', linewidth=2)
                
                self.ax_network.set_xlabel('Time (s)')
                self.ax_network.set_ylabel('Throughput (Mbps)', color='g')
                ax2.set_ylabel('CBR', color='r')
                self.ax_network.grid(True, alpha=0.3)
                
                # Legends
                lines1, labels1 = self.ax_network.get_legend_handles_labels()
                lines2, labels2 = ax2.get_legend_handles_labels()
                self.ax_network.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=8)
        
        # Add simulation info text
        info_text = f"Time: {current_time:.1f}s\n"
        info_text += f"Vehicles: {len(vehicles)}\n"
        info_text += f"Avg MCS: {np.mean([v['mcs'] for v in vehicle_data]):.1f}\n"
        if throughputs:
            info_text += f"Total Tput: {sum(throughputs):.1f} Mbps"
        
        self.ax_main.text(0.02, 0.98, info_text, transform=self.ax_main.transAxes, 
                         verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                         fontsize=9)
        
        # Update display
        plt.tight_layout()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        
        # Small delay to control update rate
        plt.pause(0.01)
    
    def save_current_frame(self, filename: str = None):
        """Save current visualization frame"""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"vanet_visualization_{self.current_time:.1f}s_{timestamp}.png"
        
        self.fig.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"[VISUALIZER] Frame saved: {filename}")
    
    def close(self):
        """Close the visualization"""
        plt.ioff()
        plt.close(self.fig)
        print("[VISUALIZER] Visualization closed")
        
# ===============================================================================
# LIVE PLOTTING
# ===============================================================================


class LivePlotManager:
    """Working manager for multiple live plot windows"""
    
    def __init__(self, config: SimulationConfig, scenario_info: Dict):
        self.config = config
        self.scenario_info = scenario_info
        self.output_dir = VISUALIZATION_OUTPUT_DIR
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize plot windows
        self.plot_windows = {}
        self.current_time = 0
        self.frame_counter = 0
        
        # Data storage for live updates
        self.performance_history = []
        self.l3_path_history = []
        self.network_topology_history = []
        
        # Current data cache
        self.current_vehicles = {}
        self.current_attack_manager = None
        
        # Setup matplotlib for interactive mode
        plt.ion()
        plt.style.use('default')
        
        self._initialize_plot_windows()
        
        print(f"[LIVE PLOTS] Initialized {len(self.plot_windows)} live plot windows")
    
    def _initialize_plot_windows(self):
        """Initialize separate plot windows with working interactive mode"""
        
        # 1. Main Network Topology Plot
        if True:  # Always create topology plot
            self.plot_windows['topology'] = self._create_topology_window()
        
        # 2. Performance Metrics Plot
        if True:  # Always create performance plot
            self.plot_windows['performance'] = self._create_performance_window()
        
        # 3. L3 Path Visualization Plot
        if L3_PATH_VISUALIZATION and self.config.enable_layer3:
            self.plot_windows['l3_paths'] = self._create_l3_path_window()
        
        # 4. Antenna Pattern Plot
        if ANTENNA_TYPE == "SECTORAL":
            self.plot_windows['antenna'] = self._create_antenna_window()
        
        # Force initial draw
        for window in self.plot_windows.values():
            window['fig'].canvas.draw()
            window['fig'].canvas.flush_events()
        
        print(f"[LIVE PLOTS] Successfully initialized {len(self.plot_windows)} windows")
    
    def _create_topology_window(self):
        """Create main network topology window"""
        fig = plt.figure(figsize=(12, 8))
        fig.suptitle('VANET Network Topology - Live View', fontsize=14, fontweight='bold')
        
        ax = fig.add_subplot(111)
        ax.set_title('Vehicle Positions and Communication Links')
        ax.set_xlabel('X Position (m)')
        ax.set_ylabel('Y Position (m)')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        
        # Set initial bounds from scenario
        if self.scenario_info:
            x_range = self.scenario_info.get('x_range', (0, 1000))
            y_range = self.scenario_info.get('y_range', (0, 100))
            
            x_padding = (x_range[1] - x_range[0]) * 0.1
            y_padding = (y_range[1] - y_range[0]) * 0.1
            
            ax.set_xlim(x_range[0] - x_padding, x_range[1] + x_padding)
            ax.set_ylim(y_range[0] - y_padding, y_range[1] + y_padding)
        else:
            # Default bounds
            ax.set_xlim(-100, 1100)
            ax.set_ylim(-50, 150)
        
        # Position window
        mngr = fig.canvas.manager
        mngr.window.wm_geometry("+50+50")
        
        plt.show(block=False)
        
        return {'fig': fig, 'ax': ax}
    
    def _create_performance_window(self):
        """Create performance metrics window"""
        fig = plt.figure(figsize=(12, 8))
        fig.suptitle('Performance Metrics - Live View', fontsize=14, fontweight='bold')
        
        # Create 2x2 subplots
        ax1 = plt.subplot(2, 2, 1)
        ax1.set_title('Average Throughput')
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Throughput (Mbps)')
        ax1.grid(True, alpha=0.3)
        
        ax2 = plt.subplot(2, 2, 2)
        ax2.set_title('Average CBR')
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('CBR')
        ax2.grid(True, alpha=0.3)
        
        ax3 = plt.subplot(2, 2, 3)
        ax3.set_title('Average SINR')
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('SINR (dB)')
        ax3.grid(True, alpha=0.3)
        
        ax4 = plt.subplot(2, 2, 4)
        ax4.set_title('Packet Delivery Ratio')
        ax4.set_xlabel('Time (s)')
        ax4.set_ylabel('PDR')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Position window
        mngr = fig.canvas.manager
        mngr.window.wm_geometry("+800+50")
        
        plt.show(block=False)
        
        return {'fig': fig, 'axes': [ax1, ax2, ax3, ax4]}
    
    def _create_l3_path_window(self):
        """Create L3 path visualization window"""
        fig = plt.figure(figsize=(16, 6))
        fig.suptitle('Layer 3 Path Visualization - Live View', fontsize=14, fontweight='bold')
        
        ax1 = plt.subplot(1, 2, 1)
        ax1.set_title('Active Routes and Paths')
        ax1.set_xlabel('X Position (m)')
        ax1.set_ylabel('Y Position (m)')
        ax1.grid(True, alpha=0.3)
        ax1.set_aspect('equal')
        
        ax2 = plt.subplot(1, 2, 2)
        ax2.set_title('Routing Performance Metrics')
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Metrics')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Position window
        mngr = fig.canvas.manager
        mngr.window.wm_geometry("+50+500")
        
        plt.show(block=False)
        
        return {'fig': fig, 'ax_paths': ax1, 'ax_metrics': ax2}
    
    def _create_antenna_window(self):
        """Create antenna pattern visualization window"""
        fig = plt.figure(figsize=(14, 6))
        fig.suptitle('Sectoral Antenna Patterns - Live View', fontsize=14, fontweight='bold')
        
        # Polar plot for antenna pattern
        ax1 = fig.add_subplot(121, projection='polar')
        ax1.set_title('Antenna Radiation Pattern')
        ax1.set_theta_zero_location('N')
        ax1.set_theta_direction(1)
        
        ax2 = fig.add_subplot(122)
        ax2.set_title('Sector Status and Power')
        ax2.set_xlabel('Antenna Sector')
        ax2.set_ylabel('Effective Power (dBm)')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Position window
        mngr = fig.canvas.manager
        mngr.window.wm_geometry("+800+500")
        
        plt.show(block=False)
        
        return {'fig': fig, 'ax_polar': ax1, 'ax_status': ax2}
    
    def _draw_antenna_pattern(self, ax, x: float, y: float, vehicle_heading: float, 
                            vehicle_id: str, vehicles: Dict):
        """Draw sectoral antenna pattern - CORRECTED for vehicle orientation"""
        # Create antenna configuration for this vehicle
        antenna_config = AntennaConfiguration(AntennaType.SECTORAL)
        antenna_system = SectoralAntennaSystem(antenna_config)
        
        # Get enabled sectors
        enabled_sectors = antenna_system.get_communication_sectors()
        
        for sector in enabled_sectors:
            sector_config = antenna_config.sectoral_config[sector.value]
            
            # CORRECTED: Calculate sector direction based on vehicle heading
            sector_relative_angle = antenna_system.sector_relative_angles[sector]
            sector_absolute_angle = (vehicle_heading + sector_relative_angle) % 360
            
            # Convert to radians for plotting
            start_angle = math.radians(sector_absolute_angle - sector_config.beamwidth_deg/2)
            end_angle = math.radians(sector_absolute_angle + sector_config.beamwidth_deg/2)
            
            # Calculate effective range based on power
            base_range = 100  # Base range in meters
            power_factor = sector_config.power_dbm / 20.0  # Normalize to 20 dBm
            effective_range = base_range * power_factor
            
            # Create sector arc
            angles = np.linspace(start_angle, end_angle, 20)
            sector_x = x + effective_range * np.cos(angles)
            sector_y = y + effective_range * np.sin(angles)
            
            # Close the sector
            sector_x = np.append(sector_x, x)
            sector_y = np.append(sector_y, y)
            sector_x = np.append(sector_x, sector_x[0])
            sector_y = np.append(sector_y, sector_y[0])
            
            # Color based on sector and power level
            sector_colors = {
                'front': 'green',
                'rear': 'blue', 
                'left': 'orange',
                'right': 'red'
            }
            base_color = sector_colors.get(sector.value, 'gray')
            
            # Adjust alpha based on power
            alpha = min(0.6, sector_config.power_dbm / 30.0)
            
            ax.fill(sector_x, sector_y, color=base_color, alpha=alpha, 
                   edgecolor=base_color, linewidth=1)
            
            # Add sector label
            label_angle = math.radians(sector_absolute_angle)
            label_x = x + (effective_range * 0.7) * math.cos(label_angle)
            label_y = y + (effective_range * 0.7) * math.sin(label_angle)
            ax.text(label_x, label_y, sector.value.upper()[:1], fontsize=8, 
                   ha='center', va='center', weight='bold',
                   bbox=dict(boxstyle='circle,pad=0.1', facecolor='white', alpha=0.8))
    
    def update_topology_plot(self, current_time: float, vehicles: Dict, attack_manager=None):
        """Update network topology plot with CORRECTED antenna orientation"""
        if 'topology' not in self.plot_windows:
            return
            
        window = self.plot_windows['topology']
        ax = window['ax']
        fig = window['fig']
        
        # Clear and redraw
        ax.clear()
        ax.set_title(f'Vehicle Positions at t={current_time:.1f}s')
        ax.set_xlabel('X Position (m)')
        ax.set_ylabel('Y Position (m)')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        
        # Reset bounds
        if self.scenario_info:
            x_range = self.scenario_info.get('x_range', (0, 1000))
            y_range = self.scenario_info.get('y_range', (0, 100))
            x_padding = (x_range[1] - x_range[0]) * 0.1
            y_padding = (y_range[1] - y_range[0]) * 0.1
            ax.set_xlim(x_range[0] - x_padding, x_range[1] + x_padding)
            ax.set_ylim(y_range[0] - y_padding, y_range[1] + y_padding)
        
        if not vehicles:
            ax.text(0.5, 0.5, 'No vehicles data', transform=ax.transAxes, 
                   ha='center', va='center', fontsize=16)
        else:
            vehicle_count = 0
            # Draw vehicles
            for vehicle_id, vehicle in vehicles.items():
                if not hasattr(vehicle, 'x') or vehicle.x is None:
                    continue
                
                vehicle_count += 1
                x, y = vehicle.x, vehicle.y
                
                # CORRECTED: Get vehicle heading from current mobility data
                vehicle_data = next((data for data in self.mobility_data 
                                   if data['time'] == current_time and data['id'] == vehicle_id), None)
                vehicle_heading = vehicle_data.get('angle', 0) if vehicle_data else 0
                
                # Determine color based on attack status
                if attack_manager and attack_manager.is_attacker(vehicle_id):
                    color = 'red'
                    marker = 's'  # Square for attackers
                    size = 120
                else:
                    # Color by MCS level
                    mcs = getattr(vehicle, 'mcs', 1)
                    color = plt.cm.viridis(mcs / 10.0)  # Normalize MCS 0-10
                    marker = 'o'  # Circle for normal
                    size = 80
                
                # Draw vehicle
                ax.scatter(x, y, c=[color], s=size, marker=marker, alpha=0.8, 
                          edgecolors='black', linewidth=1, zorder=3)
                
                # Draw vehicle heading indicator (arrow)
                arrow_length = 20  # meters
                arrow_x = arrow_length * math.cos(math.radians(vehicle_heading))
                arrow_y = arrow_length * math.sin(math.radians(vehicle_heading))
                ax.arrow(x, y, arrow_x, arrow_y, head_width=8, head_length=6, 
                        fc='black', ec='black', alpha=0.7, zorder=4)
                
                # Draw antenna pattern for selected vehicles with CORRECTED orientation
                if ANTENNA_TYPE == "SECTORAL" and vehicle_count <= 5:  # Show pattern for first 5 vehicles
                    self._draw_antenna_pattern(ax, x, y, vehicle_heading, vehicle_id, vehicles)
                
                # Draw communication range for selected vehicles
                elif vehicle_count <= 10:  # Show range for first 10 vehicles only
                    comm_range = getattr(vehicle, 'comm_range', 100)
                    circle = plt.Circle((x, y), comm_range, fill=False, 
                                      color='blue', alpha=0.2, linestyle='--', linewidth=1)
                    ax.add_patch(circle)
                
                # Draw connections to neighbors with sector information
                if hasattr(vehicle, 'neighbors') and vehicle.neighbors and len(vehicles) <= 20:
                    for neighbor in vehicle.neighbors[:3]:  # Max 3 connections per vehicle
                        neighbor_id = neighbor['id']
                        if neighbor_id in vehicles:
                            neighbor_vehicle = vehicles[neighbor_id]
                            if hasattr(neighbor_vehicle, 'x'):
                                # Color connection based on which sector the neighbor is in
                                neighbor_sector = neighbor.get('neighbor_sector', 'unknown')
                                sector_colors = {
                                    'front': 'green',
                                    'rear': 'blue',
                                    'left': 'orange', 
                                    'right': 'red'
                                }
                                connection_color = sector_colors.get(neighbor_sector, 'gray')
                                
                                ax.plot([x, neighbor_vehicle.x], [y, neighbor_vehicle.y], 
                                       color=connection_color, alpha=0.6, linewidth=2, zorder=2)
                
                # Add vehicle ID label for small numbers
                if len(vehicles) <= 15:
                    # Add heading info to label
                    label_text = f"{vehicle_id[-3:]}\n{vehicle_heading:.0f}°"
                    ax.annotate(label_text, (x, y), xytext=(5, 5), 
                               textcoords='offset points', fontsize=7, alpha=0.8)
            
            print(f"[TOPOLOGY] Drew {vehicle_count} vehicles with CORRECTED orientation at t={current_time:.1f}s")
        
        # Add info text with antenna configuration
        info_text = f"Time: {current_time:.1f}s\nVehicles: {len(vehicles)}\nAntenna: {ANTENNA_TYPE}"
        if ANTENNA_TYPE == "SECTORAL":
            info_text += f"\nFront: {SECTORAL_ANTENNA_CONFIG['front']['enabled']}"
            info_text += f"\nRear: {SECTORAL_ANTENNA_CONFIG['rear']['enabled']}"
            info_text += f"\nLeft: {SECTORAL_ANTENNA_CONFIG['left']['enabled']}"
            info_text += f"\nRight: {SECTORAL_ANTENNA_CONFIG['right']['enabled']}"
        
        if hasattr(self, 'background_traffic_manager') and self.background_traffic_manager:
            stats = self.background_traffic_manager.get_traffic_statistics()
            info_text += f"\nBG Load: {stats.get('current_network_load_mbps', 0):.1f} Mbps"
        
        ax.text(0.02, 0.98, info_text, transform=ax.transAxes, 
               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
               fontsize=9)
        
        # Force update
        fig.canvas.draw()
        fig.canvas.flush_events()
        plt.pause(0.001)  # Small pause to allow GUI update
    
    def update_performance_plot(self, current_time: float, vehicles: Dict):
        """Update performance metrics plot with working real-time updates"""
        if 'performance' not in self.plot_windows:
            return
            
        window = self.plot_windows['performance']
        axes = window['axes']
        fig = window['fig']
        
        if not vehicles:
            for ax in axes:
                ax.clear()
                ax.text(0.5, 0.5, 'No data', transform=ax.transAxes, ha='center', va='center')
            fig.canvas.draw()
            return
        
        # Collect current metrics
        throughputs = []
        cbrs = []
        sinrs = []
        pdrs = []
        
        for vehicle in vehicles.values():
            if hasattr(vehicle, 'current_throughput'):
                throughputs.append(getattr(vehicle, 'current_throughput', 0) / 1e6)  # Convert to Mbps
            if hasattr(vehicle, 'current_cbr'):
                cbrs.append(getattr(vehicle, 'current_cbr', 0))
            if hasattr(vehicle, 'current_snr'):
                sinrs.append(getattr(vehicle, 'current_snr', 0))
            if hasattr(vehicle, 'current_pdr'):
                pdrs.append(getattr(vehicle, 'current_pdr', 0))
        
        # Store history
        if throughputs and cbrs and sinrs and pdrs:
            self.performance_history.append({
                'time': current_time,
                'avg_throughput': np.mean(throughputs),
                'avg_cbr': np.mean(cbrs),
                'avg_sinr': np.mean(sinrs),
                'avg_pdr': np.mean(pdrs)
            })
        
        # Keep only recent history (last 200 points)
        if len(self.performance_history) > 200:
            self.performance_history = self.performance_history[-200:]
        
        # Update plots
        if len(self.performance_history) >= 2:
            times = [h['time'] for h in self.performance_history]
            
            # Clear all axes
            for ax in axes:
                ax.clear()
                ax.grid(True, alpha=0.3)
            
            # Plot 1: Throughput
            axes[0].plot(times, [h['avg_throughput'] for h in self.performance_history], 
                        'g-', linewidth=2, label='Avg Throughput')
            axes[0].set_title('Average Throughput')
            axes[0].set_ylabel('Throughput (Mbps)')
            axes[0].set_xlabel('Time (s)')
            
            # Plot 2: CBR
            axes[1].plot(times, [h['avg_cbr'] for h in self.performance_history], 
                        'r-', linewidth=2, label='Avg CBR')
            axes[1].set_title('Average CBR')
            axes[1].set_ylabel('CBR')
            axes[1].set_xlabel('Time (s)')
            
            # Plot 3: SINR
            axes[2].plot(times, [h['avg_sinr'] for h in self.performance_history], 
                        'b-', linewidth=2, label='Avg SINR')
            axes[2].set_title('Average SINR')
            axes[2].set_ylabel('SINR (dB)')
            axes[2].set_xlabel('Time (s)')
            
            # Plot 4: PDR
            axes[3].plot(times, [h['avg_pdr'] for h in self.performance_history], 
                        'm-', linewidth=2, label='Avg PDR')
            axes[3].set_title('Packet Delivery Ratio')
            axes[3].set_ylabel('PDR')
            axes[3].set_xlabel('Time (s)')
            
            print(f"[PERFORMANCE] Updated plots with {len(self.performance_history)} history points")
        
        plt.tight_layout()
        fig.canvas.draw()
        fig.canvas.flush_events()
        plt.pause(0.001)
    
    def update_l3_path_plot(self, current_time: float, vehicles: Dict):
        """Update L3 path visualization with working updates"""
        if 'l3_paths' not in self.plot_windows:
            return
        
        window = self.plot_windows['l3_paths']
        ax_paths = window['ax_paths']
        ax_metrics = window['ax_metrics']
        fig = window['fig']
        
        # Clear path plot
        ax_paths.clear()
        ax_paths.set_title(f'Active Layer 3 Routes at t={current_time:.1f}s')
        ax_paths.set_xlabel('X Position (m)')
        ax_paths.set_ylabel('Y Position (m)')
        ax_paths.grid(True, alpha=0.3)
        ax_paths.set_aspect('equal')
        
        if not vehicles:
            ax_paths.text(0.5, 0.5, 'No vehicles', transform=ax_paths.transAxes, ha='center', va='center')
            fig.canvas.draw()
            return
        
        route_count = 0
        total_packets_generated = 0
        total_packets_delivered = 0
        
        # Draw vehicles and routes
        for vehicle_id, vehicle in vehicles.items():
            if not hasattr(vehicle, 'x') or vehicle.x is None:
                continue
            
            x, y = vehicle.x, vehicle.y
            
            # Draw vehicle
            ax_paths.scatter(x, y, c='blue', s=80, alpha=0.8, edgecolors='black')
            
            # Add vehicle label
            ax_paths.annotate(vehicle_id[-3:], (x, y), xytext=(3, 3), 
                             textcoords='offset points', fontsize=8)
            
            # Draw routing table entries
            if (hasattr(vehicle, 'routing_protocol') and vehicle.routing_protocol and
                hasattr(vehicle.routing_protocol, 'routing_table')):
                
                for dest, route_entry in vehicle.routing_protocol.routing_table.items():
                    if dest in vehicles and hasattr(vehicles[dest], 'x'):
                        dest_vehicle = vehicles[dest]
                        
                        # Draw route line
                        ax_paths.plot([x, dest_vehicle.x], [y, dest_vehicle.y], 
                                     'g--', alpha=0.6, linewidth=2)
                        route_count += 1
                        
                        # Add hop count label
                        mid_x = (x + dest_vehicle.x) / 2
                        mid_y = (y + dest_vehicle.y) / 2
                        ax_paths.text(mid_x, mid_y, f"H{route_entry.hop_count}", 
                                     fontsize=8, alpha=0.8, 
                                     bbox=dict(boxstyle='round,pad=0.2', facecolor='yellow', alpha=0.7))
            
            # Collect packet statistics
            if hasattr(vehicle, 'packet_counters'):
                total_packets_generated += vehicle.packet_counters.get('generated', 0)
                total_packets_delivered += vehicle.packet_counters.get('delivered', 0)
        
        # Calculate L3 PDR
        l3_pdr = total_packets_delivered / max(1, total_packets_generated)
        
        # Store L3 metrics
        self.l3_path_history.append({
            'time': current_time,
            'route_count': route_count,
            'l3_pdr': l3_pdr,
            'total_generated': total_packets_generated,
            'total_delivered': total_packets_delivered
        })
        
        # Keep recent history
        if len(self.l3_path_history) > 200:
            self.l3_path_history = self.l3_path_history[-200:]
        
        # Update metrics plot
        if len(self.l3_path_history) >= 2:
            ax_metrics.clear()
            ax_metrics.set_title('Layer 3 Performance Metrics')
            ax_metrics.set_xlabel('Time (s)')
            ax_metrics.grid(True, alpha=0.3)
            
            times = [h['time'] for h in self.l3_path_history]
            
            # Plot L3 PDR and route count
            ax_metrics.plot(times, [h['l3_pdr'] for h in self.l3_path_history], 
                           'g-', label='L3 PDR', linewidth=2)
            
            # Secondary axis for route count
            ax2 = ax_metrics.twinx()
            ax2.plot(times, [h['route_count'] for h in self.l3_path_history], 
                    'r-', label='Route Count', linewidth=2)
            ax2.set_ylabel('Route Count', color='r')
            
            ax_metrics.set_ylabel('L3 PDR', color='g')
            ax_metrics.legend(loc='upper left')
            ax2.legend(loc='upper right')
        
        # Add info text
        info_text = f"Routes: {route_count}\nL3 PDR: {l3_pdr:.3f}\nGenerated: {total_packets_generated}\nDelivered: {total_packets_delivered}"
        ax_paths.text(0.02, 0.98, info_text, transform=ax_paths.transAxes, 
                     verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                     fontsize=9)
        
        fig.canvas.draw()
        fig.canvas.flush_events()
        plt.pause(0.001)
    
    def update_all_plots(self, current_time: float, vehicles: Dict, attack_manager=None):
        """Update all live plots with working real-time updates"""
        self.current_time = current_time
        self.frame_counter += 1
        self.current_vehicles = vehicles
        self.current_attack_manager = attack_manager
        
        try:
            # Update topology plot
            self.update_topology_plot(current_time, vehicles, attack_manager)
            
            # Update performance plot  
            self.update_performance_plot(current_time, vehicles)
            
            # Update L3 path plot if enabled
            if L3_PATH_VISUALIZATION and self.config.enable_layer3:
                self.update_l3_path_plot(current_time, vehicles)
            
            # Save frames periodically
            if SAVE_PLOT_FRAMES and self.frame_counter % 50 == 0:
                self.save_current_frames()
            
            print(f"[LIVE PLOTS] Updated all plots at t={current_time:.1f}s (frame {self.frame_counter})")
            
        except Exception as e:
            print(f"[LIVE PLOTS ERROR] Failed to update plots: {e}")
    
    def save_current_frames(self, timestamp: str = None):
        """Save current frames from all windows"""
        if not SAVE_PLOT_FRAMES:
            return
        
        if not timestamp:
            timestamp = f"{self.current_time:.1f}s_{datetime.now().strftime('%H%M%S')}"
        
        for plot_name, window in self.plot_windows.items():
            try:
                filename = os.path.join(self.output_dir, f"{plot_name}_{timestamp}.png")
                window['fig'].savefig(filename, dpi=150, bbox_inches='tight')
            except Exception as e:
                print(f"[LIVE PLOTS] Failed to save {plot_name}: {e}")
        
        print(f"[LIVE PLOTS] Saved frames at t={self.current_time:.1f}s")
    
    def close_all_plots(self):
        """Close all plot windows"""
        plt.ioff()
        for plot_name, window in self.plot_windows.items():
            try:
                plt.close(window['fig'])
            except:
                pass
        self.plot_windows.clear()
        print("[LIVE PLOTS] All plot windows closed")

def main():
    """Enhanced main function with Layer 3 and SDN capabilities"""
    
    # Validate FCD file
    if not os.path.exists(FCD_FILE):
        print(f"[ERROR] FCD file not found: {FCD_FILE}")
        print(f"[INFO] Please update the FCD_FILE variable in the configuration section to point to your FCD XML file.")
        return
    
    print("="*120)
    print("IEEE 802.11bd VANET SIMULATION WITH LAYER 3 STACK AND SDN CAPABILITIES")
    print("ENHANCED: Complete networking stack with routing protocols and centralized control")
    print("FEATURES: FCD reloading, Layer 3 routing, SDN controller, packet simulation, QoS management")
    print("="*120)
    print(f"FCD File: {FCD_FILE}")
    print(f"FCD Reload Count: {FCD_RELOAD_COUNT}x")
    print(f"Vehicle ID Strategy: {FCD_RELOAD_VEHICLE_ID_STRATEGY}")
    print(f"RL Optimization: {'Enabled' if ENABLE_RL else 'Disabled'}")
    
    # Layer 3 and SDN configuration display
    print(f"\n[LAYER 3 NETWORKING CONFIGURATION]")
    print(f" Layer 3 Stack: {'Enabled' if ENABLE_LAYER3 else 'Disabled'}")
    if ENABLE_LAYER3:
        print(f"  - Routing Protocol: {ROUTING_PROTOCOL}")
        print(f"  - Multi-hop Communication: {'Enabled' if ENABLE_MULTI_HOP else 'Disabled'}")
        print(f"  - Max Hop Count: {MAX_HOP_COUNT}")
        print(f"  - Route Discovery Timeout: {ROUTE_DISCOVERY_TIMEOUT}s")
        print(f"  - Hello Interval: {HELLO_INTERVAL}s")
        print(f"  - Topology Update Interval: {TOPOLOGY_UPDATE_INTERVAL}s")
    
    print(f"\n[SDN CONFIGURATION]")
    print(f" SDN Controller: {'Enabled' if ENABLE_SDN else 'Disabled'}")
    if ENABLE_SDN:
        print(f"  - Controller Type: {SDN_CONTROLLER_TYPE}")
        print(f"  - Control Protocol: {SDN_CONTROL_PROTOCOL}")
        print(f"  - Flow Table Size: {FLOW_TABLE_SIZE}")
        print(f"  - Flow Rule Timeout: {FLOW_RULE_TIMEOUT}s")
        print(f"  - QoS Management: {'Enabled' if ENABLE_QOS_MANAGEMENT else 'Disabled'}")
        print(f"  - Traffic Engineering: {'Enabled' if ENABLE_TRAFFIC_ENGINEERING else 'Disabled'}")
    
    print(f"\n[PACKET SIMULATION CONFIGURATION]")
    print(f" Packet Simulation: {'Enabled' if ENABLE_PACKET_SIMULATION else 'Disabled'}")
    if ENABLE_PACKET_SIMULATION:
        print(f"  - Packet Generation Rate: {PACKET_GENERATION_RATE} packets/second")
        print(f"  - Average Packet Size: {PACKET_SIZE_BYTES} bytes")
        print(f"  - Application Types: {', '.join(APPLICATION_TYPES)}")
        print(f"  - QoS Classes: {', '.join(QOS_CLASSES)}")
        print(f"  - Traffic Patterns: {', '.join(TRAFFIC_PATTERNS)}")
    
    # FCD reloading information
    if FCD_RELOAD_COUNT > 1:
        print(f"\n[FCD RELOADING CONFIGURATION]")
        print(f" FCD data will be reloaded {FCD_RELOAD_COUNT} times")
        print(f" Effective simulation duration will be {FCD_RELOAD_COUNT}x longer")
        print(f" Vehicle ID strategy: {FCD_RELOAD_VEHICLE_ID_STRATEGY}")
        print(f" Episode-by-episode analysis will be included in results")
        if FCD_RELOAD_VEHICLE_ID_STRATEGY == "suffix":
            print(f"  - Vehicles in episode 1: original IDs (e.g., 'vehicle1')")
            print(f"  - Vehicles in episode 2: 'vehicle1_ep2', 'vehicle2_ep2', etc.")
            print(f"  - Each episode treated as independent learning experience")
        else:
            print(f"  - Same vehicle IDs across all episodes (continuous learning)")
            print(f"  - Vehicles 'continue' their learning across episode boundaries")
        
        estimated_duration = 10000 * FCD_RELOAD_COUNT
        estimated_training_steps = estimated_duration * 45 / 5
        print(f" Estimated total duration: ~{estimated_duration:,} seconds")
        print(f" Estimated training steps: ~{estimated_training_steps:,.0f} steps")
        print(f" Training adequacy: {'EXCELLENT for advanced RL' if estimated_training_steps > 25000 else 'GOOD for basic RL' if estimated_training_steps > 15000 else 'MINIMAL - consider more reloads'}")
    else:
        print(f"\n[SINGLE RUN CONFIGURATION]")
        print(f" Using original FCD data only (no reloading)")
        print(f" Standard simulation duration")
        if ENABLE_RL and (ENABLE_LAYER3 or ENABLE_SDN):
            print(f"   For advanced RL with Layer 3/SDN, consider setting FCD_RELOAD_COUNT > 3")
    
    print(f"\n[ENHANCED SIMULATION CAPABILITIES]")
    print(f" IEEE 802.11bd PHY/MAC layer with all enhancements")
    print(f" Layer 2: Enhanced MAC with neighbor impact modeling")
    if ENABLE_LAYER3:
        print(f" Layer 3: {ROUTING_PROTOCOL} routing protocol with multi-hop support")
        print(f"  - Route discovery and maintenance")
        print(f"  - Neighbor discovery and topology management")
        print(f"  - Packet forwarding and delivery tracking")
    if ENABLE_SDN:
        print(f" SDN Control Plane: {SDN_CONTROLLER_TYPE} controller")
        print(f"  - Centralized network topology management")
        print(f"  - Flow-based packet forwarding")
        print(f"  - Dynamic traffic engineering and optimization")
        print(f"  - QoS policy enforcement")
    if ENABLE_PACKET_SIMULATION:
        print(f" Application Layer: Multi-application packet generation")
        print(f"  - Safety, infotainment, and sensing applications")
        print(f"  - QoS-aware packet classification and processing")
        print(f"  - End-to-end performance tracking")
    
    print(f"\n[IEEE 802.11bd ENHANCEMENTS]")
    print(f"   LDPC coding: {'Enabled' if ENABLE_LDPC else 'Disabled'} (1-4 dB SNR gain)")
    print(f"   Midambles for channel tracking: {'Enabled' if ENABLE_MIDAMBLES else 'Disabled'}")
    print(f"   DCM (Dual Carrier Modulation): {'Enabled' if ENABLE_DCM else 'Disabled'}")
    print(f"   Extended Range Mode: {'Enabled' if ENABLE_EXTENDED_RANGE else 'Disabled'}")
    print(f"   MIMO-STBC: {'Enabled' if ENABLE_MIMO_STBC else 'Disabled'}")
    
    print(f"\n[ENHANCED PERFORMANCE METRICS]")
    print(f"   PHY/MAC metrics: BER, SER, PER, throughput, latency, MAC efficiency")
    print(f"   Layer 3 metrics: PDR, routing overhead, route discovery success, hop count")
    print(f"   SDN metrics: flow installation time, controller latency, traffic engineering")
    print(f"   QoS metrics: per-class performance, violation rates, queue utilization")
    print(f"   Network metrics: topology statistics, convergence time, reachability")
    
    print(f"\n[REAL-TIME OUTPUT CONFIGURATION]")
    print(f"   Real-time CSV: {'Enabled' if ENABLE_REALTIME_CSV else 'Disabled'}")
    if ENABLE_REALTIME_CSV:
        print(f"    - Update frequency: Every {CSV_UPDATE_FREQUENCY} timestamp(s)")
        print(f"    - Enhanced headers with Layer 3 and SDN metrics")
    print(f"   Progressive Excel: {'Enabled' if EXCEL_UPDATE_FREQUENCY > 0 else 'Disabled'}")
    if EXCEL_UPDATE_FREQUENCY > 0:
        print(f"    - Update frequency: Every {EXCEL_UPDATE_FREQUENCY} timestamp(s)")
        print(f"    - Multi-sheet analysis with network topology")
    
    print("="*120)
    
    try:
        # Create enhanced IEEE 802.11bd simulator with Layer 3 and SDN
        config = SimulationConfig()
        simulator = VANET_IEEE80211bd_L3_SDN_Simulator(
            config, FCD_FILE, ENABLE_RL, RL_HOST, RL_PORT
        )
        
        print(f"\n[ENHANCED SIMULATION INITIALIZATION COMPLETE]")
        if FCD_RELOAD_COUNT > 1:
            print(f" FCD reloading configured: {FCD_RELOAD_COUNT}x multiplier")
            print(f" Original duration: {simulator.original_simulation_duration:.0f}s")
            print(f" Extended duration: {simulator.total_simulation_duration:.0f}s")
            print(f" Training benefit: {simulator.total_simulation_duration/simulator.original_simulation_duration:.1f}x more data")
        
        print(f" IEEE 802.11bd compliance: Validated")
        print(f" Layer 3 networking: {'Configured with ' + config.routing_protocol if config.enable_layer3 else 'Disabled'}")
        print(f" SDN controller: {'Initialized as ' + config.sdn_controller_type if config.enable_sdn else 'Disabled'}")
        print(f" Packet simulation: {'Enabled with ' + str(len(config.application_types)) + ' application types' if config.enable_packet_simulation else 'Disabled'}")
        print(f" Enhanced output configuration: Ready")
        if ENABLE_RL:
            print(f" RL optimization: {'Connected' if simulator.rl_client else 'Failed to connect'}")
        
        # Run enhanced simulation
        print(f"\n[STARTING ENHANCED SIMULATION]")
        print(f"Press Ctrl+C to interrupt simulation")
        results = simulator.run_simulation()
        
        # Save enhanced results
        output_file = simulator.save_results(OUTPUT_FILENAME)
        
        
        print("="*120)
        print("ENHANCED IEEE 802.11bd SIMULATION WITH LAYER 3 AND SDN COMPLETED SUCCESSFULLY")
        print(f"Comprehensive Excel results with multi-layer analysis saved to: {output_file}")
        if ENABLE_REALTIME_CSV:
            print(f"Real-time CSV data with enhanced metrics available in: {simulator.realtime_csv_file}")
        
        # NEW: Save attack results if attack simulation is enabled
        if ENABLE_ATTACK_SIMULATION:
            attack_output_file = simulator.save_attack_results()
            print(f"[ATTACK RESULTS] Attack detection dataset saved to: {attack_output_file}")
        
        # Enhanced summary reporting
        if FCD_RELOAD_COUNT > 1:
            print(f"\n[RELOADING SUMMARY]")
            print(f"Original FCD duration: {simulator.original_simulation_duration:.0f} seconds")
            print(f"Total simulation duration: {simulator.total_simulation_duration:.0f} seconds")
            print(f"Training data multiplier: {FCD_RELOAD_COUNT}x")
            print(f"Total data points generated: {len(results):,}")
            print(f"Episode strategy: {FCD_RELOAD_VEHICLE_ID_STRATEGY}")
            
            if results:
                df = pd.DataFrame(results)
                if 'Episode' in df.columns:
                    episode_summary = df.groupby('Episode').agg({
                        'VehicleID': 'nunique',
                        'Throughput': 'mean',
                        'PDR': 'mean',
                        'SINR': 'mean',
                        'CBR': 'mean',
                        'L3_PDR': 'mean' if 'L3_PDR' in df.columns else 'count',
                        'FlowTableSize': 'mean' if 'FlowTableSize' in df.columns else 'count'
                    }).round(3)
                    
                    print(f"\n[EPISODE PERFORMANCE SUMMARY]")
                    for episode, row in episode_summary.iterrows():
                        base_info = f"Episode {episode}: {row['VehicleID']} vehicles, Throughput={row['Throughput']:.2f}Mbps, PDR={row['PDR']:.3f}"
                        l3_info = f", L3_PDR={row['L3_PDR']:.3f}" if 'L3_PDR' in row else ""
                        sdn_info = f", AvgFlows={row['FlowTableSize']:.1f}" if 'FlowTableSize' in row else ""
                        print(base_info + l3_info + sdn_info)
        
        print("="*120)
        
        # Enhanced validation summary
        if results:
            df = pd.DataFrame(results)
            
            # Basic validation
            neighbor_cbr_corr = df['NeighborNumbers'].corr(df['CBR'])
            neighbor_per_corr = df['NeighborNumbers'].corr(df['PER'])
            neighbor_throughput_corr = df['NeighborNumbers'].corr(df['Throughput'])
            
            print("[ENHANCED VALIDATION RESULTS]")
            print(f"  Neighbor-CBR Correlation: {neighbor_cbr_corr:.3f} (Expected: >0.3)")
            print(f"  Neighbor-PER Correlation: {neighbor_per_corr:.3f} (Expected: >0.2)")
            print(f"  Neighbor-Throughput Correlation: {neighbor_throughput_corr:.3f} (Expected: <-0.2)")
            
            if neighbor_cbr_corr > 0.3 and neighbor_per_corr > 0.2 and neighbor_throughput_corr < -0.2:
                print("    PHY/MAC VALIDATION PASSED: Neighbor count properly impacts performance")
            else:
                print("     PHY/MAC VALIDATION WARNING: Check neighbor impact calculations")
            
            # Layer 3 validation
            if config.enable_layer3 and 'L3_PDR' in df.columns:
                l3_pdr_values = df['L3_PDR'].dropna()
                if len(l3_pdr_values) > 0:
                    avg_l3_pdr = l3_pdr_values.mean()
                    l3_pdr_variation = l3_pdr_values.std()
                    packets_generated = df['PacketsGenerated'].sum()
                    packets_delivered = (df['PacketsGenerated'] * df['L3_PDR']).sum()
                    
                    print(f"  Layer 3 Average PDR: {avg_l3_pdr:.3f}")
                    print(f"  Layer 3 PDR Variation: {l3_pdr_variation:.3f}")
                    print(f"  Total L3 Packets: Generated={packets_generated}, Delivered={packets_delivered:.0f}")
                    
                    if avg_l3_pdr > 0.7 and l3_pdr_variation < 0.3:
                        print("    LAYER 3 VALIDATION PASSED: Routing protocol functioning properly")
                    elif avg_l3_pdr > 0.5:
                        print("     LAYER 3 VALIDATION WARNING: Routing performance could be improved")
                    else:
                        print("    LAYER 3 VALIDATION FAILED: Check routing protocol implementation")
            
            # SDN validation
            if config.enable_sdn and 'FlowTableSize' in df.columns:
                flow_table_values = df['FlowTableSize'].dropna()
                if len(flow_table_values) > 0:
                    avg_flows = flow_table_values.mean()
                    max_flows = flow_table_values.max()
                    flow_utilization = df['ActiveFlows'].sum() / max(1, df['FlowTableSize'].sum()) if 'ActiveFlows' in df.columns else 0
                    
                    print(f"  SDN Average Flow Table Size: {avg_flows:.1f}")
                    print(f"  SDN Max Flow Table Size: {max_flows}")
                    print(f"  SDN Flow Utilization: {flow_utilization:.3f}")
                    
                    if avg_flows > 0 and flow_utilization > 0.1:
                        print("    SDN VALIDATION PASSED: Controller and flow management functioning")
                    elif avg_flows > 0:
                        print("     SDN VALIDATION WARNING: Low flow utilization detected")
                    else:
                        print("    SDN VALIDATION FAILED: Check SDN controller implementation")
            
            # Performance variation check
            throughput_cv = df['Throughput'].std() / df['Throughput'].mean()
            per_cv = df['PER'].std() / df['PER'].mean()
            
            print(f"  Throughput Coefficient of Variation: {throughput_cv:.3f}")
            print(f"  PER Coefficient of Variation: {per_cv:.3f}")
            
            if throughput_cv > 0.1 and per_cv > 0.1:
                print("    PERFORMANCE VARIATION: Realistic diversity in results")
            else:
                print("     LOW VARIATION: Check if calculations are too static")
            
            # FCD reloading effectiveness
            if FCD_RELOAD_COUNT > 1:
                print(f"\n[FCD RELOADING EFFECTIVENESS]")
                total_training_data = len(results)
                original_estimate = 10000 * 45
                data_multiplier = total_training_data / original_estimate
                print(f"  Data points generated: {total_training_data:,}")
                print(f"  Data multiplication factor: {data_multiplier:.1f}x")
                print(f"  Training adequacy: {'EXCELLENT' if total_training_data > 1000000 else 'GOOD' if total_training_data > 500000 else 'ADEQUATE'}")
                
                if 'Episode' in df.columns:
                    episode_counts = df['Episode'].value_counts().sort_index()
                    episode_consistency = episode_counts.std() / episode_counts.mean()
                    print(f"  Episode data consistency: {episode_consistency:.3f} (lower is better)")
                    if episode_consistency < 0.1:
                        print("    EPISODE CONSISTENCY: Balanced data across episodes")
                    else:
                        print("     EPISODE IMBALANCE: Some episodes have different data volumes")
        
        print("="*120)
        print("[ENHANCED SIMULATION SUMMARY]")
        print("    IEEE 802.11bd PHY/MAC layer with all enhancements")
        print("    Fixed identical performance results issue")
        print("    Enhanced wireless communication theory compliance")
        print("    Improved neighbor impact modeling")
        if config.enable_layer3:
            print(f"    Layer 3 {config.routing_protocol} routing protocol implemented")
            print("    Multi-hop communication and route management")
        if config.enable_sdn:
            print(f"    SDN {config.sdn_controller_type} controller implemented")
            print("    Flow-based forwarding and traffic engineering")
        if config.enable_packet_simulation:
            print("    Application-layer packet generation and QoS management")
        print("    Comprehensive statistical validation")
        print("    Enhanced Excel output with multi-layer analysis")
        print("    Maintained all existing working functionality")
        if FCD_RELOAD_COUNT > 1:
            print(f"    FCD reloading capability ({FCD_RELOAD_COUNT}x duration extension)")
            print("    Episode-by-episode analysis and tracking")
        print("="*120)
        
        # Training recommendations for enhanced RL
        if ENABLE_RL and results:
            estimated_training_steps = len(results) / 5
            print(f"[ENHANCED RL TRAINING RECOMMENDATIONS]")
            print(f"  Estimated training steps: {estimated_training_steps:,.0f}")
            
            feature_complexity = 1.0
            if config.enable_layer3:
                feature_complexity += 0.5  # Additional complexity for routing
            if config.enable_sdn:
                feature_complexity += 0.5  # Additional complexity for SDN
            if config.enable_packet_simulation:
                feature_complexity += 0.3  # Additional complexity for applications
            
            required_steps = 15000 * feature_complexity
            
            if estimated_training_steps < required_steps:
                print(f"    RECOMMENDATION: Consider increasing FCD_RELOAD_COUNT for complex multi-layer training")
                recommended_reloads = max(3, int(required_steps * 5 / len(results)) + 1)
                print(f"     Suggested FCD_RELOAD_COUNT: {recommended_reloads}")
                print(f"     Feature complexity factor: {feature_complexity:.1f}x")
            elif estimated_training_steps < required_steps * 2:
                print("    TRAINING ADEQUACY: Good for enhanced RL with Layer 3/SDN features")
            else:
                print("    TRAINING ADEQUACY: Excellent for advanced multi-agent RL with full protocol stack")
            
            if FCD_RELOAD_COUNT > 1:
                print(f"    TRAINING PROGRESSION: Monitor episode performance in Excel output")
                print(f"     Expected learning: PHY optimization → Layer 3 efficiency → SDN coordination")
                if config.enable_layer3 and config.enable_sdn:
                    print(f"     Advanced objective: Joint PHY/MAC/L3/SDN optimization")
    
    except Exception as e:
        print(f"[ERROR] Enhanced simulation failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if 'simulator' in locals():
            simulator.cleanup()

if __name__ == "__main__":
    main()
