"""
Quick Setup Script for AI Traffic Control System
This script creates all necessary files and directory structure automatically.
Run this first to set up your project!
"""

import os

def create_directory_structure():
    """Create project directories"""
    directories = ['network', 'src', 'models']
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    print("✓ Directories created")

def create_network_files():
    """Create SUMO network files"""
    
    # intersection.net.xml - Simplified version
    net_xml = """<?xml version="1.0" encoding="UTF-8"?>
<net version="1.16" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
    <edge id="north_in" from="n" to="center" numLanes="2"/>
    <edge id="south_in" from="s" to="center" numLanes="2"/>
    <edge id="east_in" from="e" to="center" numLanes="2"/>
    <edge id="west_in" from="w" to="center" numLanes="2"/>
    <edge id="north_out" from="center" to="n" numLanes="2"/>
    <edge id="south_out" from="center" to="s" numLanes="2"/>
    <edge id="east_out" from="center" to="e" numLanes="2"/>
    <edge id="west_out" from="center" to="w" numLanes="2"/>
    <tlLogic id="center" type="actuated" programID="0">
        <phase duration="31" state="GGGGrrrrGGGGrrrr"/>
        <phase duration="4" state="yyyyrrrryyyyrrrr"/>
        <phase duration="31" state="rrrrGGGGrrrrGGGG"/>
        <phase duration="4" state="rrrryyyyrrrrryyyy"/>
    </tlLogic>
</net>"""
    
    # routes.rou.xml
    routes_xml = """<?xml version="1.0" encoding="UTF-8"?>
<routes>
    <vType id="car" accel="2.6" decel="4.5" sigma="0.5" length="4.5" minGap="2.5" 
           maxSpeed="30" guiShape="passenger/sedan" carFollowModel="IDM"/>
    <vType id="truck" accel="1.8" decel="3.5" sigma="0.5" length="7.5" minGap="3.0" 
           maxSpeed="25" guiShape="truck" carFollowModel="IDM"/>
    
    <route id="north_south" edges="north_in south_out"/>
    <route id="south_north" edges="south_in north_out"/>
    <route id="east_west" edges="east_in west_out"/>
    <route id="west_east" edges="west_in east_out"/>
    
    <flow id="f_ns" type="car" route="north_south" begin="0" end="3600" vehsPerHour="400"/>
    <flow id="f_sn" type="car" route="south_north" begin="0" end="3600" vehsPerHour="380"/>
    <flow id="f_ew" type="car" route="east_west" begin="0" end="3600" vehsPerHour="420"/>
    <flow id="f_we" type="truck" route="west_east" begin="0" end="3600" vehsPerHour="200"/>
</routes>"""
    
    # additional.add.xml
    additional_xml = """<?xml version="1.0" encoding="UTF-8"?>
<additional>
    <inductionLoop id="det_n" lane="north_in_0" pos="50" freq="1" file="NUL"/>
    <inductionLoop id="det_s" lane="south_in_0" pos="50" freq="1" file="NUL"/>
    <inductionLoop id="det_e" lane="east_in_0" pos="50" freq="1" file="NUL"/>
    <inductionLoop id="det_w" lane="west_in_0" pos="50" freq="1" file="NUL"/>
</additional>"""
    
    # simulation.sumocfg
    config_xml = """<?xml version="1.0" encoding="UTF-8"?>
<configuration>
    <input>
        <net-file value="intersection.net.xml"/>
        <route-files value="routes.rou.xml"/>
        <additional-files value="additional.add.xml"/>
    </input>
    <time>
        <begin value="0"/>
        <end value="3600"/>
    </time>
</configuration>"""
    
    # Write files
    with open('network/intersection.net.xml', 'w') as f:
        f.write(net_xml)
    with open('network/routes.rou.xml', 'w') as f:
        f.write(routes_xml)
    with open('network/additional.add.xml', 'w') as f:
        f.write(additional_xml)
    with open('network/simulation.sumocfg', 'w') as f:
        f.write(config_xml)
    
    print("✓ Network files created")

def create_readme():
    """Create README with instructions"""
    readme = """# AI Traffic Control System - Quick Start

## Installation

## Usage

### 1. Train the AI Agent

### 2. Test with Visualization

## Features
-  Deep Q-Learning for adaptive signal control
-  VAC (Vehicle Actuated Control) timing
-  Bidirectional double lanes
-  Realistic vehicle models with collision avoidance
-  Obstacle detection and slowdown
-  Continuous traffic simulation
"""
    
    with open('README.md', 'w') as f:
        f.write(readme)
    
    print("✓ README created")

def create_requirements_txt():
    """Create requirements.txt"""
    requirements = """torch>=1.9.0
numpy>=1.21.0
matplotlib>=3.4.0
gym>=0.21.0
pandas>=1.3.0
"""
    
    with open('requirements.txt', 'w') as f:
        f.write(requirements)
    
    print("✓ requirements.txt created")

def main():
    """Main setup function"""
    print("=" * 50)
    print("AI Traffic Control System - Quick Setup")
    print("=" * 50)
    print()
    
    create_directory_structure()
    create_network_files()
    create_readme()
    create_requirements_txt()
    
    print()
    print("=" * 50)
    print(" Setup Complete!")
    print("=" * 50)
    print()
    print("Next Steps:")
    print("1. Install dependencies: pip install -r requirements.txt")
    print("2. Install SUMO from: https://sumo.dlr.de/docs/Downloads.php")
    print("3. I will send you the Python files next")
    print()

if __name__ == '__main__':
    main()
