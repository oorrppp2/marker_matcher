#!/usr/bin/python
#-*- encoding: utf8 -*-

import rospy
import actionlib
import numpy as np
import math

import tf2_ros
import cv2

from geometry_msgs.msg import PoseStamped, Quaternion
from tf2_geometry_msgs import PoseStamped as TF2PoseStamped
from qrcode_detector_ros.msg import Result
from tf.transformations import quaternion_from_euler, quaternion_multiply

from std_msgs.msg import Empty, String, Bool, Header, Float64
from sensor_msgs.msg import CompressedImage, Image
from sensor_msgs.msg import PointCloud2, PointField
from cv_bridge import CvBridge

class MarkerMatcher:
    def __init__(self):
        self.result_pub = rospy.Publisher("/marker_matcher_result", Image, queue_size=1)

        self.image_raw_sub = rospy.Subscriber("/camera/color/image_raw", Image, self.handle_image_sub)
        # self.depth_raw_sub = rospy.Subscriber("/camera/depth/image_rect_raw", Image, self.handle_depth_sub)

        self.reference_img = cv2.imread("/home/jyp/catkin_ws/src/marker_matcher/reference_img.jpg")

        self.br = CvBridge()

    def handle_image_sub(self, msg):
        image = self.br.imgmsg_to_cv2(msg)

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blr = cv2.GaussianBlur(gray, (0, 0), 1)
        circles = cv2.HoughCircles(blr, cv2.HOUGH_GRADIENT, 1, 50,
                                param1=150, param2=40, minRadius=20, maxRadius=80)
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        rgb_image = cv2.normalize(rgb_image, None, 0, 255, cv2.NORM_MINMAX)
        dst = rgb_image.copy()
        if circles is not None: # 원이 검출 됬으면
            for i in range(circles.shape[1]): # 원의 개수 만큼 반복문
                cx, cy, radius = circles[0][i] # 중심좌표, 반지름 정보 얻기

                x1 = int(cx - radius)
                y1 = int(cy - radius)
                x2 = int(cx + radius)
                y2 = int(cy + radius)
                radius = int(radius)

                crop = dst[y1:y2, x1:x2, :] # 크롭 영역 생성
                ch, cw = crop.shape[:2] # 크롭 영상의 세로, 가로 정보 획득

                # Region of "Red" over region of cropped image > 0.5
                region_of_circle = np.pi * (radius ** 2)
                region_of_red = np.where( (crop[:,:,2] > 140) & (crop[:,:,0] < 100) & (crop[:,:,1] < 100), crop[:,:,0], 0)
                region_of_circle = region_of_red.shape[0] * region_of_red.shape[1]
                region_of_red = len(region_of_red[region_of_red > 0])

                # print(float(region_of_red) / float(region_of_circle))

                if float(region_of_red) / float(region_of_circle) > 0.5:
                    cv2.circle(rgb_image, (cx, cy), int(radius), (0, 0, 255), 2, cv2.LINE_AA) # 얻은 정보로 원 그리기
        # print("="*30)
        self.result_pub.publish(self.br.cv2_to_imgmsg(rgb_image))

    def handle_depth_sub(self, msg):
        self.depth = self.br.imgmsg_to_cv2(msg)
        print("depth shape : ", self.depth.shape)
        


if __name__ == '__main__':
    rospy.init_node('MarkerMatcher')
    server = MarkerMatcher()
    rospy.spin()