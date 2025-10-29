import xml.etree.ElementTree as ET
import os

print("Enhancing SUMO visualization with realistic elements...")
print("="*60)

# Create polygon (buildings, trees, etc.) file
polygons_xml = '''<?xml version="1.0" encoding="UTF-8"?>
<additional>
    <!-- Buildings around intersection -->
    <poly id="building_1" color="0.8,0.7,0.6" fill="1" layer="-1" 
          shape="120.00,120.00 180.00,120.00 180.00,180.00 120.00,180.00"/>
    <poly id="building_2" color="0.7,0.6,0.5" fill="1" layer="-1" 
          shape="-180.00,120.00 -120.00,120.00 -120.00,180.00 -180.00,180.00"/>
    <poly id="building_3" color="0.75,0.65,0.55" fill="1" layer="-1" 
          shape="120.00,-180.00 180.00,-180.00 180.00,-120.00 120.00,-120.00"/>
    <poly id="building_4" color="0.8,0.7,0.6" fill="1" layer="-1" 
          shape="-180.00,-180.00 -120.00,-180.00 -120.00,-120.00 -180.00,-120.00"/>
    
    <!-- Trees along roads - North side -->
    <poi id="tree_n1" color="0.2,0.6,0.2" layer="2" x="-30.00" y="100.00" type="tree"/>
    <poi id="tree_n2" color="0.2,0.6,0.2" layer="2" x="-20.00" y="110.00" type="tree"/>
    <poi id="tree_n3" color="0.2,0.6,0.2" layer="2" x="-10.00" y="120.00" type="tree"/>
    <poi id="tree_n4" color="0.2,0.6,0.2" layer="2" x="10.00" y="120.00" type="tree"/>
    <poi id="tree_n5" color="0.2,0.6,0.2" layer="2" x="20.00" y="110.00" type="tree"/>
    <poi id="tree_n6" color="0.2,0.6,0.2" layer="2" x="30.00" y="100.00" type="tree"/>
    
    <!-- Trees along roads - South side -->
    <poi id="tree_s1" color="0.2,0.6,0.2" layer="2" x="-30.00" y="-100.00" type="tree"/>
    <poi id="tree_s2" color="0.2,0.6,0.2" layer="2" x="-20.00" y="-110.00" type="tree"/>
    <poi id="tree_s3" color="0.2,0.6,0.2" layer="2" x="-10.00" y="-120.00" type="tree"/>
    <poi id="tree_s4" color="0.2,0.6,0.2" layer="2" x="10.00" y="-120.00" type="tree"/>
    <poi id="tree_s5" color="0.2,0.6,0.2" layer="2" x="20.00" y="-110.00" type="tree"/>
    <poi id="tree_s6" color="0.2,0.6,0.2" layer="2" x="30.00" y="-100.00" type="tree"/>
    
    <!-- Trees along roads - East side -->
    <poi id="tree_e1" color="0.2,0.6,0.2" layer="2" x="100.00" y="30.00" type="tree"/>
    <poi id="tree_e2" color="0.2,0.6,0.2" layer="2" x="110.00" y="20.00" type="tree"/>
    <poi id="tree_e3" color="0.2,0.6,0.2" layer="2" x="120.00" y="10.00" type="tree"/>
    <poi id="tree_e4" color="0.2,0.6,0.2" layer="2" x="120.00" y="-10.00" type="tree"/>
    <poi id="tree_e5" color="0.2,0.6,0.2" layer="2" x="110.00" y="-20.00" type="tree"/>
    <poi id="tree_e6" color="0.2,0.6,0.2" layer="2" x="100.00" y="-30.00" type="tree"/>
    
    <!-- Trees along roads - West side -->
    <poi id="tree_w1" color="0.2,0.6,0.2" layer="2" x="-100.00" y="30.00" type="tree"/>
    <poi id="tree_w2" color="0.2,0.6,0.2" layer="2" x="-110.00" y="20.00" type="tree"/>
    <poi id="tree_w3" color="0.2,0.6,0.2" layer="2" x="-120.00" y="10.00" type="tree"/>
    <poi id="tree_w4" color="0.2,0.6,0.2" layer="2" x="-120.00" y="-10.00" type="tree"/>
    <poi id="tree_w5" color="0.2,0.6,0.2" layer="2" x="-110.00" y="-20.00" type="tree"/>
    <poi id="tree_w6" color="0.2,0.6,0.2" layer="2" x="-100.00" y="-30.00" type="tree"/>
    
    <!-- Parking lots -->
    <poly id="parking_1" color="0.5,0.5,0.5" fill="1" layer="-1" 
          shape="80.00,80.00 110.00,80.00 110.00,110.00 80.00,110.00"/>
    <poly id="parking_2" color="0.5,0.5,0.5" fill="1" layer="-1" 
          shape="-110.00,80.00 -80.00,80.00 -80.00,110.00 -110.00,110.00"/>
    
    <!-- Green spaces / parks -->
    <poly id="park_1" color="0.3,0.7,0.3" fill="1" layer="-2" 
          shape="60.00,60.00 80.00,60.00 80.00,80.00 60.00,80.00"/>
    <poly id="park_2" color="0.3,0.7,0.3" fill="1" layer="-2" 
          shape="-80.00,-80.00 -60.00,-80.00 -60.00,-60.00 -80.00,-60.00"/>
    
    <!-- Street lights -->
    <poi id="light_1" color="1.0,1.0,0.5" layer="3" x="15.00" y="15.00" type="streetlight"/>
    <poi id="light_2" color="1.0,1.0,0.5" layer="3" x="-15.00" y="15.00" type="streetlight"/>
    <poi id="light_3" color="1.0,1.0,0.5" layer="3" x="15.00" y="-15.00" type="streetlight"/>
    <poi id="light_4" color="1.0,1.0,0.5" layer="3" x="-15.00" y="-15.00" type="streetlight"/>
    
</additional>'''

# Save polygon file
with open('network/polygons.poly.xml', 'w', encoding='utf-8') as f:
    f.write(polygons_xml)
print("[OK] Created polygons file with buildings and trees")

# Create SUMO view settings for better visuals
viewsettings_xml = '''<viewsettings>
    <scheme name="realistic">
        <background backgroundColor="0.82,0.90,0.95"/>
        <delay value="100"/>
        <vehicles vehicleQuality="2" vehicleSize="1.0" showBlinker="1"/>
        <edges laneEdgeMode="0" showLinkDecals="1" showLane2Lane="1"/>
        <buildings fill="1"/>
        <pois fill="1" minSize="10.0"/>
    </scheme>
</viewsettings>'''

with open('network/view.settings.xml', 'w', encoding='utf-8') as f:
    f.write(viewsettings_xml)
print("[OK] Created view settings for better visuals")

# Update simulation config to include polygons
config_xml = '''<?xml version="1.0" encoding="UTF-8"?>
<configuration>
    <input>
        <net-file value="intersection.net.xml"/>
        <route-files value="routes.rou.xml"/>
        <additional-files value="additional.add.xml,polygons.poly.xml"/>
        <gui-settings-file value="view.settings.xml"/>
    </input>
    <time>
        <begin value="0"/>
        <end value="3600"/>
    </time>
    <gui_only>
        <gui-settings-file value="view.settings.xml"/>
    </gui_only>
</configuration>'''

with open('network/simulation.sumocfg', 'w', encoding='utf-8') as f:
    f.write(config_xml)
print("[OK] Updated simulation config to include visual elements")

print("="*60)
print("Enhancement complete!")
print("="*60)
print("\nRealistic elements added:")
print("- 4 Buildings around intersection")
print("- 24 Trees along all roads")
print("- 2 Parking lots")
print("- 2 Green park areas")
print("- 4 Street lights")
print("- Custom color scheme and background")
print("\nRun 'python src/test.py' to see the enhanced visuals!")
