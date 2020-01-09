#####################################################
## librealsense tutorial #1 - Accessing depth data ##
#####################################################

import sys
sys.path.append('/usr/local/lib')

# First import the library
import pyrealsense2 as rs
import numpy as np
import cv2
import imutils
import time

net = cv2.dnn.readNetFromCaffe('deploy.prototxt', 'deploy.caffemodel')

#def normalize_plane()

    # normalized_depth[range(220)] = np.zeros(640)  # vnp.ones(640) * 25
    # normalized_depth[range(350, 480)] = np.zeros(640)  # np.ones(640) * 25

    # np.savez_compressed('depth_image_{}.npz'.format(ts), depth_image)



def main_rect(frame):
    inWidth = 640
    inHeight = 480
    WHRatio = inWidth / float(inHeight)
    inScaleFactor = 0.007843
    meanVal = 127.5
    thr = 0.2

    blob = cv2.dnn.blobFromImage(frame, inScaleFactor, (inWidth, inHeight), (meanVal, meanVal, meanVal), False)
    net.setInput(blob)
    detections = net.forward()

    cols = frame.shape[1]
    rows = frame.shape[0]

    if cols / float(rows) > WHRatio:
        cropSize = (int(rows * WHRatio), rows)
    else:
        cropSize = (cols, int(cols / WHRatio))

    y1 = int((rows - cropSize[1]) / 2)
    y2 = y1 + cropSize[1]
    x1 = int((cols - cropSize[0]) / 2)
    x2 = x1 + cropSize[0]
    frame = frame[y1:y2, x1:x2]

    cols = frame.shape[1]
    rows = frame.shape[0]
    center = cols / 2, rows / 2

    main_rect = ((0, 0), (0, 0))
    min_dist = 10000

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        class_id = int(detections[0, 0, i, 1])
        if confidence > thr and class_id == 7:

            xLeftBottom = int(detections[0, 0, i, 3] * cols)
            yLeftBottom = int(detections[0, 0, i, 4] * rows)
            xRightTop = int(detections[0, 0, i, 5] * cols)
            yRightTop = int(detections[0, 0, i, 6] * rows)

            obj_rect = ((xLeftBottom, yLeftBottom), (xRightTop, yRightTop))

            # cv.rectangle(frame, obj_rect[0], obj_rect[1], (0, 255, 0))

            obj_center = (xRightTop + xLeftBottom) / 2, (yRightTop + yLeftBottom) / 2

            dist = np.sqrt((center[0] - obj_center[0]) ** 2 + (center[1] - obj_center[1]) ** 2)

            if dist < min_dist:
                min_dist = dist
                main_rect = obj_rect

    #return ((main_rect[0][0] + x1, main_rect[0][1] + y1), (main_rect[1][0] + x1, main_rect[1][1] + y1))
    return main_rect

def detect(color_image, depth_image, depth_scale):#, plane):

    # Переводим из милиметров в метры
    normalized_depth = np.multiply(depth_image, 0.001)
    # Больше 10 метров - зануляем
    normalized_depth[normalized_depth > 50] = 0
    # Меньше метра - зануляем
    normalized_depth[normalized_depth < 1] = 0

    # normalized_depth[range(220)] = np.zeros(640)  # vnp.ones(640) * 25
    # normalized_depth[range(350, 480)] = np.zeros(640)  # np.ones(640) * 25
    # np.savez_compressed('plane.npz', plane=normalized_depth)

    # normalized_depth[abs(normalized_depth - plane) < 1] = 0


    max = np.amax(depth_image)
    min = np.amin(depth_image)


    # normalized_depth[normalized_depth < 650] = 0

    # scale = 2.56 / max


    depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(normalized_depth, alpha=5, beta=0), cv2.COLORMAP_HOT)
    # depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.01, beta=0), cv2.COLORMAP_HOT)

    gray = cv2.cvtColor(depth_colormap, cv2.COLOR_BGR2GRAY)  # convert to grey scale
    gray = cv2.bilateralFilter(gray, 11, 10, 10)  # Blur to reduce noise
    edged = cv2.Canny(gray, 50, 150)  # Perform Edge detection

    # cnts = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # cnts = imutils.grab_contours(cnts)
    # cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:10]
    #
    # for c in cnts:
    #     # approximate the contour
    #     peri = cv2.arcLength(c, True)
    #     approx = cv2.approxPolyDP(c, 0.018 * peri, True)
    #     # print(">> ", len(approx))
    #
    #     cv2.drawContours(color_image, [c], -1, (0, 255, 255), 1)
    #     cv2.drawContours(color_image, [approx], -1, (0, 0, 255), 1)

    # Stack both images horizontally
    #images = np.hstack((color_image, depth_colormap))


    gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
    backtorgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    images = cv2.addWeighted(backtorgb, 0.5, depth_colormap, 0.5, 0)

    car_rect = main_rect(color_image)
    cv2.rectangle(images, car_rect[0], car_rect[1], (0, 255, 0), 2)


    car_center_x = int((car_rect[0][0] + car_rect[1][0]) / 2)
    car_center_y = int((car_rect[0][1] + car_rect[1][1]) / 2)

    if car_center_x > 0 and car_center_y > 0:
        d = 5
        # depth1 = depth_image[car_center_x, car_center_y].astype(float)
        # depth2 = depth_image[car_center_x - d, car_center_y].astype(float)
        # depth3 = depth_image[car_center_x + d, car_center_y].astype(float)
        # depth4 = depth_image[car_center_x, car_center_y + d].astype(float)
        # depth5 = depth_image[car_center_x, car_center_y + d].astype(float)
        # depth_arr = [depth1, depth2, depth3, depth4, depth5]
        # depth_non_zero = max(depth_arr)
        depth_arr = normalized_depth[car_center_x - d:car_center_x + 1, car_center_y - d: car_center_y + d]
        depth_non_zero = depth_arr[np.nonzero(depth_arr)]

        # depth = (depth1 + depth2 + depth3 + depth4 + depth5) / 5
        if depth_non_zero.size > 0:
            depth_max = np.median(depth_non_zero)
            distance = depth_max #* depth_scale
            print("Distance (m): ", distance)

            cv2.putText(images, "{}".format(distance), (0, 50), 0, 2, 255)

        cv2.rectangle(images, (car_center_x - d, car_center_y - d), (car_center_x + d, car_center_y + d), (255, 0, 0), 2)


    # Show images
    cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
    cv2.imshow('RealSense', images)
    cv2.waitKey(1)


def realcam():
    #plane_file = np.load('plane.npz')
    #plane = plane_file['plane']

    try:
        # Create a context object. This object owns the handles to all connected realsense devices
        pipeline = rs.pipeline()

        config = rs.config()
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        # config.enable_stream(rs.stream.depth, 1280, 960, rs.format.z16, 10)
        # config.enable_stream(rs.stream.color, 1280, 960, rs.format.bgr8, 10)

        profile = pipeline.start(config)

        depth_sensor = profile.get_device().first_depth_sensor()
        depth_scale = depth_sensor.get_depth_scale()
        print("Depth Scale is: ", depth_scale)

        last_ts = 0

        while True:
            # This call waits until a new coherent set of frames is available on a device
            # Calls to get_frame_data(...) and get_frame_timestamp(...) on a device will return stable values until wait_for_frames(...) is called
            frames = pipeline.wait_for_frames()
            color = frames.get_color_frame()
            depth = frames.get_depth_frame()
            if not depth: continue

            depth_image = np.asanyarray(depth.get_data())
            color_image = np.asanyarray(color.get_data())

            #ts = time.time().asInt()
            #if ts % 15 == 0 and last_ts + 15 < ts:
            #    last_ts = ts

                # np.savez_compressed('depth_image_{}.npz'.format(ts), depth_image)
                # cv2.imwrite('color_image_{}.png'.format(ts), color_image)


            detect(color_image, depth_image, depth_scale) #, plane)

    except Exception as e:
        print(e)
        pass


def simulate(color_filename, depth_filename):
    plane_file = np.load('plane.npz')
    plane = plane_file['plane']

    color_image = cv2.imread('{}.png'.format(color_filename))
    data = np.load('{}.npz'.format(depth_filename))
    depth_image = data['arr_0']

    detect(color_image, depth_image, 0.001)#, plane)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    if len(sys.argv) == 3:
        simulate(sys.argv[1], sys.argv[2])
    else:
        realcam()


