import sys
sys.path.append('/usr/local/lib')
# sys.path.append('/opt/intel/openvino_2019.3.376/opencv/lib')

import pyrealsense2 as rs
import numpy as np
import cv2
import time


def collect():
    try:
        pipeline = rs.pipeline()

        config = rs.config()
        config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 6)
        config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 6)

        profile = pipeline.start(config)

        # d_stream = profile.get_stream(rs.stream.depth)
        # c_stream = profile.get_stream(rs.stream.color)

        depth_sensor = profile.get_device().first_depth_sensor()
        depth_scale = depth_sensor.get_depth_scale()

        last_ts = 0

        while True:
        #if True:
            frames = pipeline.wait_for_frames()
            color = frames.get_color_frame()
            depth = frames.get_depth_frame()

            depth_image = np.asanyarray(depth.get_data())
            color_image = np.asanyarray(color.get_data())

            ts = int(time.time())
            if last_ts + 5 * 60 < ts:
                last_ts = ts

                np.savez_compressed('shots/{}.npz'.format(ts), depth_image)
                cv2.imwrite('shots/{}.png'.format(ts), color_image)

            # display
            normalized_depth = np.multiply(depth_image, depth_scale)
            normalized_depth[normalized_depth > 50] = 0
            normalized_depth[normalized_depth < 1] = 0
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(normalized_depth, alpha=5, beta=0), cv2.COLORMAP_HOT)
            images = cv2.addWeighted(color_image, 0.8, depth_colormap, 0.2, 0)
            # images = np.hstack((color_image, depth_colormap))

            cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
            cv2.imshow('RealSense', images)
            cv2.waitKey(1)

    except Exception as e:
        print(e)
        pass


if __name__ == "__main__":
    collect()
