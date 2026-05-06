#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
import cv2
import numpy as np
from PIL import Image as PILImage
from PIL import ImageDraw, ImageFont
import serial

class ShapeSubscriber(Node):
    def __init__(self):
        super().__init__('shape_subscriber')

        self.sub = self.create_subscription(
            Image, '/camera/image_raw', self.image_callback, 10)
        
        self.pub_image = self.create_publisher(Image, 'shape_image', 10)
        self.pub_result = self.create_publisher(String, 'shape_result', 10)

        self.cv_bridge = CvBridge()
        self.last_uart_frame = None
        self.shape_protocol_map = {
            "圆形": ("CIRCLE", "A", "1"),
            "三角形": ("TRIANGLE", "B", "1"),
            "矩形": ("RECT", "C", "1"),
            "五边形": ("PENTAGON", "D", "1"),
        }

        try:
            self.ser = serial.Serial(
                port="/dev/ttyUSB0",
                baudrate=9600,
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
                timeout=1
            )
            self.get_logger().info("✅ 视觉串口已连接，协议输出已启用")
        except Exception as e:
            self.ser = None
            self.get_logger().error(f"❌ 串口打开失败: {str(e)}")

        self.get_logger().info("✅ 抗干扰粗黑空心图形识别启动（零误识别）")

    def image_callback(self, msg):
        try:
            frame = self.cv_bridge.imgmsg_to_cv2(msg, 'bgr8')
            result_name, center = self.detect_target_shape(frame)

            out_msg = self.cv_bridge.cv2_to_imgmsg(frame, 'bgr8')
            self.pub_image.publish(out_msg)

            if result_name:
                self.pub_result.publish(String(data=f"{result_name} 中心:{center}"))
                self.send_shape_uart(result_name)
                self.get_logger().info(f"✅ 识别成功：{result_name}")
            else:
                self.send_shape_uart(None)

        except Exception as e:
            self.get_logger().error(str(e))

    def send_shape_uart(self, shape_name):
        if self.ser is None or not self.ser.is_open:
            return

        if shape_name in self.shape_protocol_map:
            shape, target, valid = self.shape_protocol_map[shape_name]
        else:
            shape, target, valid = ("NONE", "X", "0")

        frame = f"${shape},{target},{valid}\r\n"
        if frame == self.last_uart_frame:
            return

        try:
            self.ser.write(frame.encode("ascii"))
            self.last_uart_frame = frame
        except Exception as e:
            self.get_logger().error(f"串口发送失败: {str(e)}")

    def detect_target_shape(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape

        # 1. 【优化】更严格的二值化：只提取纯黑色粗线条
        # 先固定阈值过滤黑色，再用自适应处理暗光
        _, binary_fixed = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY_INV)
        binary_adaptive = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 51, 5)
        # 两种二值化结果取交集，既过滤环境杂色，又适配暗光
        binary = cv2.bitwise_and(binary_fixed, binary_adaptive)

        # 2. 强化轮廓，去除细碎噪点
        kernel = np.ones((5,5), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=3)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)

        # 3. 层级轮廓：只找有内孔的双层轮廓
        contours, hierarchy = cv2.findContours(
            binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

        shape_name = None
        cx, cy = 0, 0
        found = False

        for i in range(len(contours)):
            cnt = contours[i]
            area = cv2.contourArea(cnt)

            # ---------------- 过滤1：面积范围（只认你这种大尺寸图形） ----------------
            # 你的图形在画面里的面积占比，大概在1%~30%之间，超出的直接过滤
            if not (0.01 * h*w < area < 0.3 * h*w):
                continue

            # ---------------- 过滤2：必须是双层空心轮廓 ----------------
            if hierarchy[0][i][2] == -1:
                continue
            inner_idx = hierarchy[0][i][2]
            inner_cnt = contours[inner_idx] if inner_idx < len(contours) else None
            if inner_cnt is None or cv2.contourArea(inner_cnt) < 0.1 * area:
                continue  # 内轮廓面积太小，不是目标的空心结构

            # ---------------- 过滤3：轮廓必须是凸的，且长宽比正常 ----------------
            # 目标图形都是凸多边形，环境里的凹形/不规则轮廓直接过滤
            hull = cv2.convexHull(cnt)
            hull_area = cv2.contourArea(hull)
            solidity = area / hull_area if hull_area !=0 else 0
            if solidity < 0.85:  # 凸包面积比，越接近1越规则
                continue

            # 外接矩形长宽比，过滤长条形干扰
            x, y, w_rect, h_rect = cv2.boundingRect(cnt)
            aspect_ratio = w_rect / h_rect
            if aspect_ratio < 0.5 or aspect_ratio > 2.0:  # 太宽或太窄的轮廓过滤
                continue

            # ---------------- 过滤4：中心点必须是白色（严格版） ----------------
            M = cv2.moments(cnt)
            if M["m00"] == 0:
                continue
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])

            roi = 25
            x1, y1 = max(0, cx-roi), max(0, cy-roi)
            x2, y2 = min(w, cx+roi), min(h, cy+roi)
            center_patch = gray[y1:y2, x1:x2]
            if np.mean(center_patch) < 150:  # 暗光环境下的白色标准，可调
                continue

            # ---------------- 过滤5：图形分类，只认规则形状 ----------------
            epsilon = 0.03 * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)
            sides = len(approx)

            if sides == 3:
                shape_name = "三角形"
            elif sides == 4:
                shape_name = "矩形"
            elif sides == 5:
                shape_name = "五边形"
            else:
                # 圆形的圆度范围，过滤不规则圆形
                peri = cv2.arcLength(cnt, True)
                circularity = 4 * np.pi * area / (peri ** 2) if peri != 0 else 0
                if 0.8 < circularity < 1.15:
                    shape_name = "圆形"

            if shape_name:
                # 绘制内外轮廓
                cv2.drawContours(frame, [cnt], -1, (255,0,0), 3)
                cv2.drawContours(frame, [inner_cnt], -1, (255,0,0), 2)
                cv2.circle(frame, (cx, cy), 8, (0,0,255), -1)
                frame = self.put_chinese(frame, shape_name, (cx-40, cy-30))
                found = True
                break

        return (shape_name, (cx, cy)) if found else (None, None)

    def put_chinese(self, img, text, pos):
        img_pil = PILImage.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc", 28)
        except:
            font = ImageFont.load_default()
        draw.text(pos, text, font=font, fill=(0,255,0))
        return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

    def destroy_node(self):
        if self.ser and self.ser.is_open:
            self.ser.close()
        super().destroy_node()

def main():
    rclpy.init()
    node = ShapeSubscriber()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
