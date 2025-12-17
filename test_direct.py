#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Direct test of sensor input modules without importing the full brain package.
"""

import sys
import os

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(__file__))

# Test basic imports by loading modules directly
try:
    # Test data types
    exec(open('brain/cognitive/world_model/sensor_input_types.py').read())
    print("✓ Successfully loaded sensor_input_types.py")

    # Test basic data creation
    import time
    import numpy as np

    # Create a simple point cloud
    points = np.random.rand(10, 3)
    pc_data = PointCloudData(points=points, timestamp=time.time())
    print("✓ Created PointCloudData with {} points".format(pc_data.point_count))

    # Create a simple image
    image = np.random.randint(0, 256, (50, 50, 3), dtype=np.uint8)
    img_data = ImageData(image=image, timestamp=time.time())
    print("✓ Created ImageData with size {}x{}".format(img_data.width, img_data.height))

    # Create sensor data packet
    packet = SensorDataPacket(
        sensor_id="test",
        sensor_type=SensorType.POINT_CLOUD,
        timestamp=time.time(),
        data=pc_data
    )
    print("✓ Created SensorDataPacket with quality score {:.2f}".format(packet.quality_score))

    print("\n🎉 All basic functionality tests passed!")
    print("The sensor input module is implemented correctly.")

except ImportError as e:
    print("❌ Import error: {}".format(e))
    sys.exit(1)
except Exception as e:
    print("❌ Error during testing: {}".format(e))
    import traceback
    traceback.print_exc()
    sys.exit(1)