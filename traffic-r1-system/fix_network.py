import os

# Complete working network file
net_xml = '''<?xml version="1.0" encoding="UTF-8"?>
<net version="1.16">
    <location netOffset="0.00,0.00" convBoundary="-200.00,-200.00,200.00,200.00"/>
    
    <edge id="north_in" from="n" to="center">
        <lane id="north_in_0" index="0" speed="13.89" length="192.80" shape="-1.60,200.00 -1.60,7.20"/>
        <lane id="north_in_1" index="1" speed="13.89" length="192.80" shape="1.60,200.00 1.60,7.20"/>
    </edge>
    <edge id="south_in" from="s" to="center">
        <lane id="south_in_0" index="0" speed="13.89" length="192.80" shape="1.60,-200.00 1.60,-7.20"/>
        <lane id="south_in_1" index="1" speed="13.89" length="192.80" shape="-1.60,-200.00 -1.60,-7.20"/>
    </edge>
    <edge id="east_in" from="e" to="center">
        <lane id="east_in_0" index="0" speed="13.89" length="192.80" shape="200.00,1.60 7.20,1.60"/>
        <lane id="east_in_1" index="1" speed="13.89" length="192.80" shape="200.00,-1.60 7.20,-1.60"/>
    </edge>
    <edge id="west_in" from="w" to="center">
        <lane id="west_in_0" index="0" speed="13.89" length="192.80" shape="-200.00,-1.60 -7.20,-1.60"/>
        <lane id="west_in_1" index="1" speed="13.89" length="192.80" shape="-200.00,1.60 -7.20,1.60"/>
    </edge>
    <edge id="north_out" from="center" to="n">
        <lane id="north_out_0" index="0" speed="13.89" length="192.80" shape="1.60,7.20 1.60,200.00"/>
        <lane id="north_out_1" index="1" speed="13.89" length="192.80" shape="-1.60,7.20 -1.60,200.00"/>
    </edge>
    <edge id="south_out" from="center" to="s">
        <lane id="south_out_0" index="0" speed="13.89" length="192.80" shape="-1.60,-7.20 -1.60,-200.00"/>
        <lane id="south_out_1" index="1" speed="13.89" length="192.80" shape="1.60,-7.20 1.60,-200.00"/>
    </edge>
    <edge id="east_out" from="center" to="e">
        <lane id="east_out_0" index="0" speed="13.89" length="192.80" shape="7.20,-1.60 200.00,-1.60"/>
        <lane id="east_out_1" index="1" speed="13.89" length="192.80" shape="7.20,1.60 200.00,1.60"/>
    </edge>
    <edge id="west_out" from="center" to="w">
        <lane id="west_out_0" index="0" speed="13.89" length="192.80" shape="-7.20,1.60 -200.00,1.60"/>
        <lane id="west_out_1" index="1" speed="13.89" length="192.80" shape="-7.20,-1.60 -200.00,-1.60"/>
    </edge>
    
    <junction id="center" type="traffic_light" x="0.00" y="0.00" incLanes="north_in_0 north_in_1 south_in_0 south_in_1 east_in_0 east_in_1 west_in_0 west_in_1" intLanes="" shape="-3.20,7.20 3.20,7.20 3.20,-7.20 -3.20,-7.20 -7.20,-3.20 -7.20,3.20 7.20,3.20 7.20,-3.20"/>
    <junction id="n" type="dead_end" x="0.00" y="200.00" incLanes="north_out_0 north_out_1" intLanes=""/>
    <junction id="s" type="dead_end" x="0.00" y="-200.00" incLanes="south_out_0 south_out_1" intLanes=""/>
    <junction id="e" type="dead_end" x="200.00" y="0.00" incLanes="east_out_0 east_out_1" intLanes=""/>
    <junction id="w" type="dead_end" x="-200.00" y="0.00" incLanes="west_out_0 west_out_1" intLanes=""/>
    
    <connection from="north_in" to="south_out" fromLane="0" toLane="0" via=":center_0_0" tl="center" linkIndex="0" dir="s" state="O"/>
    <connection from="north_in" to="south_out" fromLane="1" toLane="1" via=":center_0_1" tl="center" linkIndex="1" dir="s" state="O"/>
    <connection from="south_in" to="north_out" fromLane="0" toLane="0" via=":center_1_0" tl="center" linkIndex="2" dir="s" state="O"/>
    <connection from="south_in" to="north_out" fromLane="1" toLane="1" via=":center_1_1" tl="center" linkIndex="3" dir="s" state="O"/>
    <connection from="east_in" to="west_out" fromLane="0" toLane="0" via=":center_2_0" tl="center" linkIndex="4" dir="s" state="o"/>
    <connection from="east_in" to="west_out" fromLane="1" toLane="1" via=":center_2_1" tl="center" linkIndex="5" dir="s" state="o"/>
    <connection from="west_in" to="east_out" fromLane="0" toLane="0" via=":center_3_0" tl="center" linkIndex="6" dir="s" state="o"/>
    <connection from="west_in" to="east_out" fromLane="1" toLane="1" via=":center_3_1" tl="center" linkIndex="7" dir="s" state="o"/>
    
    <tlLogic id="center" type="static" programID="0">
        <phase duration="31" state="GGGGrrrrGGGGrrrr"/>
        <phase duration="4" state="yyyyrrrryyyyrrrr"/>
        <phase duration="31" state="rrrrGGGGrrrrGGGG"/>
        <phase duration="4" state="rrrryyyyrrrrryyyy"/>
    </tlLogic>
</net>'''

with open('network/intersection.net.xml', 'w', encoding='utf-8') as f:
    f.write(net_xml)

print("[OK] Fixed network file created!")
print("Now run: python src\\train.py")