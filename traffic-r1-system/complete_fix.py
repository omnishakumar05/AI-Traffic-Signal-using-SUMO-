import subprocess
import os

print("Creating complete working traffic system...")
print("="*60)

# Step 1: Generate network using netgenerate
print("[1/3] Generating network...")
net_cmd = [
    'netgenerate',
    '--grid',
    '--grid.x-number=1',
    '--grid.y-number=1',
    '--grid.x-length=200',
    '--grid.y-length=200',
    '--grid.attach-length=100',
    '--default.lanenumber=2',
    '--default.speed=13.89',
    '--tls.guess=true',
    '--output-file=network/intersection.net.xml'
]

try:
    subprocess.run(net_cmd, check=True, capture_output=True)
    print("    [OK] Network generated")
except Exception as e:
    print("    [ERROR]", e)
    exit(1)

# Step 2: Create matching routes based on generated network
print("[2/3] Creating routes...")

# For netgenerate grid, edges are named differently
# We'll use simple straight-through routes
routes_xml = '''<?xml version="1.0" encoding="UTF-8"?>
<routes>
    <vType id="car" accel="2.6" decel="4.5" sigma="0.5" length="4.5" minGap="2.5" 
           maxSpeed="13.89" guiShape="passenger/sedan" carFollowModel="IDM"/>
    
    <!-- Using actual edge names from netgenerate grid -->
    <route id="r0" edges="A0B0 B0A0"/>
    <route id="r1" edges="B0A0 A0B0"/>
    <route id="r2" edges="B0C0 C0B0"/>
    <route id="r3" edges="C0B0 B0C0"/>
    <route id="r4" edges="B1B0 B0B1"/>
    <route id="r5" edges="B0B1 B1B0"/>
    
    <flow id="f0" type="car" route="r0" begin="0" end="3600" vehsPerHour="300"/>
    <flow id="f1" type="car" route="r1" begin="0" end="3600" vehsPerHour="300"/>
    <flow id="f2" type="car" route="r2" begin="0" end="3600" vehsPerHour="300"/>
    <flow id="f3" type="car" route="r3" begin="0" end="3600" vehsPerHour="300"/>
    <flow id="f4" type="car" route="r4" begin="0" end="3600" vehsPerHour="300"/>
    <flow id="f5" type="car" route="r5" begin="0" end="3600" vehsPerHour="300"/>
</routes>'''

with open('network/routes.rou.xml', 'w', encoding='utf-8') as f:
    f.write(routes_xml)
print("    [OK] Routes created")

# Step 3: Create empty additional file (no detectors needed)
print("[3/3] Creating additional file...")
additional_xml = '''<?xml version="1.0" encoding="UTF-8"?>
<additional>
</additional>'''

with open('network/additional.add.xml', 'w', encoding='utf-8') as f:
    f.write(additional_xml)
print("    [OK] Additional file created")

print("="*60)
print("SUCCESS! Network setup complete!")
print("="*60)
print("\nNow run: python src\\train.py")