import xml.etree.ElementTree as ET
import os

print("Auto-detecting edge names from generated network...")

# Parse the network file
tree = ET.parse('network/intersection.net.xml')
root = tree.getroot()

# Get all edge IDs
edges = []
for edge in root.findall('edge'):
    edge_id = edge.get('id')
    if edge_id and not edge_id.startswith(':'):  # Skip internal edges
        edges.append(edge_id)

print(f"Found {len(edges)} edges:")
for e in edges:
    print(f"  - {e}")

# Create simple routes using first available edges
if len(edges) >= 2:
    routes_xml = f'''<?xml version="1.0" encoding="UTF-8"?>
<routes>
    <vType id="car" accel="2.6" decel="4.5" sigma="0.5" length="4.5" minGap="2.5" 
           maxSpeed="13.89" guiShape="passenger/sedan" carFollowModel="IDM"/>
    
    <!-- Auto-generated routes using detected edges -->
    <route id="r0" edges="{edges[0]} {edges[1] if len(edges) > 1 else edges[0]}"/>
    <route id="r1" edges="{edges[1] if len(edges) > 1 else edges[0]} {edges[0]}"/>
'''
    
    # Add more routes if more edges exist
    if len(edges) >= 4:
        routes_xml += f'''    <route id="r2" edges="{edges[2]} {edges[3]}"/>
    <route id="r3" edges="{edges[3]} {edges[2]}"/>
'''
    
    # Add flows
    routes_xml += '''    
    <flow id="f0" type="car" route="r0" begin="0" end="3600" vehsPerHour="300"/>
    <flow id="f1" type="car" route="r1" begin="0" end="3600" vehsPerHour="300"/>
'''
    
    if len(edges) >= 4:
        routes_xml += '''    <flow id="f2" type="car" route="r2" begin="0" end="3600" vehsPerHour="300"/>
    <flow id="f3" type="car" route="r3" begin="0" end="3600" vehsPerHour="300"/>
'''
    
    routes_xml += '</routes>'
    
    # Write corrected routes file
    with open('network/routes.rou.xml', 'w', encoding='utf-8') as f:
        f.write(routes_xml)
    
    print("\n[OK] Routes file updated with correct edge names!")
    print("\nNow run: python src\\train.py")
else:
    print("\n[ERROR] Not enough edges found in network file")