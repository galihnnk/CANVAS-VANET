# Comprehensive Adaptive Network for VANET Simulation (CANVAS)

A comprehensive Python-based VANET (Vehicular Ad-hoc Network) simulation framework featuring IEEE 802.11bd wireless communication, multiple routing protocols, SDN capabilities, reinforcement learning optimization, and advanced attack simulation.

## Overview

CANVAS is a state-of-the-art VANET simulation platform that combines realistic vehicular mobility (via SUMO), advanced wireless communication modeling, and cutting-edge network optimization techniques. It provides researchers and engineers with a comprehensive tool for evaluating VANET protocols, security mechanisms, and performance optimization strategies.

## Key Features

### Core Networking
- **IEEE 802.11bd Compliance**: Complete PHY/MAC layer implementation with LDPC, DCM, MIMO-STBC
- **Multi-Protocol Support**: AODV, OLSR, Geographic routing with realistic protocol behavior
- **SDN Integration**: Software-Defined Networking with centralized/distributed controllers
- **QoS Management**: Multi-class service differentiation and traffic engineering

### Advanced Optimization
- **Reinforcement Learning**: Real-time parameter optimization for transmission power, MCS, and beacon rates
- **Adaptive Antennas**: Omnidirectional and sectoral antenna systems with RL-controlled power allocation
- **Traffic Engineering**: Dynamic load balancing and congestion management

### Security & Attack Simulation
- **Attack Vectors**: Beacon flooding, high-power jamming, asynchronous beacon attacks
- **Detection Dataset**: ML-ready feature extraction for attack detection research
- **Network Resilience**: Comprehensive attack impact analysis and mitigation strategies

### Realistic Environment Modeling
- **SUMO Integration**: Uses real FCD (Floating Car Data) from SUMO traffic simulations
- **Background Traffic**: Realistic management and infotainment traffic patterns
- **Interference Modeling**: Multi-source interference with hidden terminal effects
- **Mobility Prediction**: Speed and heading-aware routing decisions

### Analysis & Visualization
- **Real-time Monitoring**: Live network topology and performance visualization
- **Comprehensive Metrics**: PHY/MAC/Network layer KPIs with detailed breakdowns
- **Multi-format Output**: Excel, CSV with episode-by-episode analysis
- **Statistical Validation**: Automated correlation analysis and performance validation

## Getting Started

### Prerequisites

```bash
# Python 3.8 or higher
python --version

# Required packages
pip install numpy pandas matplotlib networkx openpyxl
pip install scipy xmltodict

# For RL integration (optional)
pip install torch gymnasium

# For advanced visualization (optional)
pip install plotly seaborn
```

### Installation

1. Clone the repository
```bash
git clone https://github.com/galihnnk/CANVAS-VANET.git
cd CANVAS-VANET
```

2. Install dependencies
```bash
pip install -r requirements.txt
```

3. Prepare SUMO FCD data
```bash
# Generate FCD output from SUMO
sumo -c your_scenario.sumocfg --fcd-output fcd-output.xml
```

### Quick Start

1. Configure the simulation (edit configuration section in the script):

```python
# Basic Configuration
FCD_FILE = "your-fcd-output.xml"
ENABLE_RL = True
ENABLE_LAYER3 = True
ROUTING_PROTOCOL = "AODV"
ANTENNA_TYPE = "SECTORAL"
```

2. Run the simulation:
```bash
python canvas_simulation.py
```

3. View results:
   - Excel files with comprehensive analysis
   - Real-time CSV for monitoring
   - Attack detection datasets (if enabled)

## Configuration Guide

### Core Settings

| Parameter | Description | Options |
|-----------|-------------|---------|
| FCD_FILE | Path to SUMO FCD XML file | Any valid XML path |
| ENABLE_RL | Reinforcement Learning optimization | True/False |
| ENABLE_LAYER3 | Layer 3 routing protocols | True/False |
| ROUTING_PROTOCOL | Routing algorithm | "AODV", "OLSR", "GEOGRAPHIC" |
| ENABLE_SDN | Software-Defined Networking | True/False |
| ANTENNA_TYPE | Antenna configuration | "OMNIDIRECTIONAL", "SECTORAL" |

### Antenna Configuration

```python
# Sectoral Antenna (RL-optimized front/rear, static sides)
SECTORAL_ANTENNA_CONFIG = {
    "front": {"power_dbm": 12.0, "gain_db": 8.0, "beamwidth_deg": 60},
    "rear": {"power_dbm": 8.0, "gain_db": 8.0, "beamwidth_deg": 60},
    "left": {"power_dbm": 3.0, "gain_db": 5.0, "beamwidth_deg": 90},
    "right": {"power_dbm": 3.0, "gain_db": 5.0, "beamwidth_deg": 90}
}
```

### Attack Simulation

```python
# Enable DoS/DDoS attack simulation
ENABLE_ATTACK_SIMULATION = True
ATTACK_TYPE = "COMBINED"  # or "BEACON_FLOODING", "HIGH_POWER_JAMMING"
NUMBER_OF_ATTACKERS = 5
GENERATE_ATTACK_DATASET = True  # For ML research
```

### Extended Training

```python
# FCD Data Reloading for Extended Training
FCD_RELOAD_COUNT = 3  # 3x simulation duration
FCD_RELOAD_VEHICLE_ID_STRATEGY = "suffix"  # or "reuse"
```

## Project Structure

```
CANVAS-VANET/
├── canvas_simulation.py      # Main simulation script
├── requirements.txt          # Python dependencies
├── README.md                # This file
├── examples/                # Example configurations
│   ├── highway_scenario.py
│   ├── urban_intersection.py
│   └── attack_simulation.py
├── docs/                    # Documentation
│   ├── API_Reference.md
│   ├── Protocol_Details.md
│   └── Performance_Tuning.md
├── results/                 # Output directory
│   ├── excel_outputs/
│   ├── csv_realtime/
│   └── attack_datasets/
└── visualization/           # Visualization outputs
    ├── plots/
    └── animations/
```

## Output Files

### Performance Analysis
- **Main Results**: Training-conventionalsac-alternative-45cars-10000seconds.xlsx
  - Multi-sheet analysis with PHY/MAC/Network metrics
  - Episode-by-episode performance tracking
  - Statistical summaries and correlations

### Real-time Monitoring
- **Live CSV**: Real-time performance data with 100+ metrics
- **Progressive Excel**: Continuously updated results during simulation

### Security Analysis
- **Attack Dataset**: ML-ready feature vectors for attack detection
- **Impact Analysis**: Network performance under various attack scenarios

## Usage Examples

### Basic VANET Simulation
```python
# Configure for basic highway simulation
FCD_FILE = "highway_scenario.xml"
ENABLE_RL = False
ROUTING_PROTOCOL = "AODV"
ANTENNA_TYPE = "OMNIDIRECTIONAL"
```

### RL-Optimized Sectoral Antennas
```python
# Advanced setup with RL optimization
ENABLE_RL = True
ANTENNA_TYPE = "SECTORAL"
RL_CONTROLLED_SECTORS = ['front', 'rear']
RL_HOST = '127.0.0.1'
RL_PORT = 5000
```

### Attack Simulation Research
```python
# Security research configuration
ENABLE_ATTACK_SIMULATION = True
ATTACK_TYPE = "COMBINED"
NUMBER_OF_ATTACKERS = 10
GENERATE_ATTACK_DATASET = True
```

### Extended Training Dataset
```python
# Generate large training datasets
FCD_RELOAD_COUNT = 5  # 5x longer simulation
FCD_RELOAD_VEHICLE_ID_STRATEGY = "suffix"
ENABLE_REALTIME_CSV = True
```

## Research Applications

- **Protocol Evaluation**: Compare routing protocols under various conditions
- **Security Research**: Analyze attack vectors and detection mechanisms
- **RL Optimization**: Develop adaptive network optimization algorithms
- **Performance Analysis**: Study VANET behavior in realistic scenarios
- **Traffic Engineering**: Optimize network resource allocation
- **Antenna Design**: Evaluate directional vs omnidirectional antenna systems

## Contributing

We welcome contributions! Please see our Contributing Guidelines for details.

### Development Setup
```bash
# Fork and clone the repository
git clone https://github.com/galihnnk/CANVAS-VANET.git

# Create a development branch
git checkout -b feature/your-feature-name

# Make your changes and test
python -m pytest tests/

# Submit a pull request
```

## Documentation

- **API Reference**: Detailed class and method documentation
- **Protocol Details**: Routing protocol implementations
- **Performance Tuning**: Optimization guidelines
- **Attack Simulation**: Security testing guide

## Troubleshooting

### Common Issues

**FCD File Not Found**
```bash
[ERROR] FCD file not found: your-file.xml
```
*Solution*: Ensure the FCD file path is correct and the file exists.

**RL Connection Failed**
```bash
[RL ERROR] Failed to connect to RL server
```
*Solution*: Start your RL server before running the simulation or disable RL.

**Memory Issues with Large Simulations**
- Reduce FCD_RELOAD_COUNT
- Disable real-time visualization
- Use smaller FCD files

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- **SUMO Traffic Simulator**: Eclipse SUMO team for traffic simulation framework
- **IEEE 802.11bd Standard**: For wireless communication specifications
- **Research Community**: VANET researchers for protocol implementations and validation

## Contact

- **Project Maintainer**: Galih Nugraha Nurkahfi
- **Issues**: Please use GitHub Issues for bug reports and feature requests
- **Discussions**: Use GitHub Discussions for questions and ideas

## Citation

If you use CANVAS in your research, please cite:

```bibtex
@software{canvas_vanet_2025,
  title={CANVAS: Comprehensive Adaptive Network for VANET Simulation},
  author={Galih Nugraha Nurkahfi},
  year={2025},
  url={https://github.com/galihnnk/CANVAS-VANET},
  note={IEEE 802.11bd VANET Simulation Framework with RL Optimization}
}
```

---

**Star this repository if CANVAS helps your research!**
