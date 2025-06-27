Comprehensive Adaptive Network for VANET Simulation (CANVAS)
A comprehensive Python-based VANET (Vehicular Ad-hoc Network) simulation framework featuring IEEE 802.11bd wireless communication, multiple routing protocols, SDN capabilities, reinforcement learning optimization, and advanced attack simulation.
🎯 Overview
CANVAS is a state-of-the-art VANET simulation platform that combines realistic vehicular mobility (via SUMO), advanced wireless communication modeling, and cutting-edge network optimization techniques. It provides researchers and engineers with a comprehensive tool for evaluating VANET protocols, security mechanisms, and performance optimization strategies.
✨ Key Features
🔥 Core Networking

IEEE 802.11bd Compliance: Complete PHY/MAC layer implementation with LDPC, DCM, MIMO-STBC
Multi-Protocol Support: AODV, OLSR, Geographic routing with realistic protocol behavior
SDN Integration: Software-Defined Networking with centralized/distributed controllers
QoS Management: Multi-class service differentiation and traffic engineering

🤖 Advanced Optimization

Reinforcement Learning: Real-time parameter optimization for transmission power, MCS, and beacon rates
Adaptive Antennas: Omnidirectional and sectoral antenna systems with RL-controlled power allocation
Traffic Engineering: Dynamic load balancing and congestion management

🛡️ Security & Attack Simulation

Attack Vectors: Beacon flooding, high-power jamming, asynchronous beacon attacks
Detection Dataset: ML-ready feature extraction for attack detection research
Network Resilience: Comprehensive attack impact analysis and mitigation strategies

📊 Realistic Environment Modeling

SUMO Integration: Uses real FCD (Floating Car Data) from SUMO traffic simulations
Background Traffic: Realistic management and infotainment traffic patterns
Interference Modeling: Multi-source interference with hidden terminal effects
Mobility Prediction: Speed and heading-aware routing decisions

📈 Analysis & Visualization

Real-time Monitoring: Live network topology and performance visualization
Comprehensive Metrics: PHY/MAC/Network layer KPIs with detailed breakdowns
Multi-format Output: Excel, CSV with episode-by-episode analysis
Statistical Validation: Automated correlation analysis and performance validation

🚀 Getting Started
Prerequisites
bash# Python 3.8 or higher
python --version

# Required packages
pip install numpy pandas matplotlib networkx openpyxl
pip install scipy xmltodict

# For RL integration (optional)
pip install torch gymnasium

# For advanced visualization (optional)
pip install plotly seaborn
Installation

Clone the repository

bashgit clone https://github.com/galihnnk/CANVAS-VANET.git
cd CANVAS-VANET

Install dependencies

bashpip install -r requirements.txt

🔍 Real-time Monitoring

Live CSV: Real-time performance data with 100+ metrics
Progressive Excel: Continuously updated results during simulation

🔬 Research Applications

Protocol Evaluation: Compare routing protocols under various conditions
Security Research: Analyze attack vectors and detection mechanisms
RL Optimization: Develop adaptive network optimization algorithms
Performance Analysis: Study VANET behavior in realistic scenarios
Traffic Engineering: Optimize network resource allocation
Antenna Design: Evaluate directional vs omnidirectional antenna systems

🙏 Acknowledgments

SUMO Traffic Simulator: Eclipse SUMO team for traffic simulation framework
IEEE 802.11bd Standard: For wireless communication specifications
Research Community: VANET researchers for protocol implementations and validation

📧 Contact

Project Maintainer: Galih Nugraha Nurkahfi
Issues: Please use GitHub Issues for bug reports and feature requests
Discussions: Use GitHub Discussions for questions and ideas

 Citation
If you use CANVAS in your research, please cite:
bibtex@software{canvas_vanet_2024,
  title={CANVAS: Comprehensive Adaptive Network for VANET Simulation},
  author={Galih Nugraha Nurkahfi},
  year={2025},
  url={https://github.com/yourusername/CANVAS-VANET},
  note={IEEE 802.11bd VANET Simulation Framework with RL Optimization}
}
