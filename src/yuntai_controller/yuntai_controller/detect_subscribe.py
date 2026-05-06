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
import struct

class ShapeSubscriber(Node):
    def __init__(self):
        super().__init__('shape_subscriber')

        # 1. 订阅相机图像
        self.sub = self.create_subscription(
            Image, '/camera/image_raw', self.image_callback, 10)
        
        # 2. 发布处理后的图像
        self.pub_image = self.create_publisher(Image, 'shape_image', 10)
        # 3. 发布识别结果+偏差
        self.pub_result = self.create_publisher(String, 'shape_result', 10)

        self.cv_bridge = CvBridge()

        # ===================== 串口初始化（树莓派 -> 云台驱动板） =====================
        try:
            self.ser = serial.Serial(
                port="/dev/ttyUSB0",    # 树莓派硬件串口
                baudrate=115200,        # 波特率（和驱动板一致）
                timeout=1
            )
            self.get_logger().info("✅ 串口连接成功！云台控制已就绪")
        except Exception as e:
            self.get_logger().error(f"❌ 串口打开失败：{str(e)}")
            self.ser = None

        # 云台控制参数（比例系数，调这个控制云台速度）
        self.Kp = 0.3  # 数值越大，云台转动越快
        self.get_logger().info("✅ 图形识别+偏差计算+云台控制节点启动！")

    # ===================== 你的偏差计算函数 =====================
    def calculate_deviation(self, target_center, frame_shape):
        h, w = frame_shape[:2]
        img_center_x = w // 2
        img_center_y = h // 2
        dx = target_center[0] - img_center_x
        dy = target_center[1] - img_center_y
        return dx, dy, img_center_x, img_center_y

    # ===================== 串口发送云台控制指令 =====================
    def send_gimbal_command(self, dx, dy):
        if self.ser is None or not self.ser.is_open:
            return

        # 1. 比例控制：把像素偏差转换成云台运动速度（核心！）
        speed_x = int(dx * self.Kp)
        speed_y = int(dy * self.Kp)

        # 2. 限制速度范围（防止电机过载）
        speed_x = np.clip(speed_x, -255, 255)
        speed_y = np.clip(speed_y, -255, 255)

        # 3. 自定义通信协议（帧头+数据+校验+帧尾，稳定不丢包）
        frame_head = 0x55
        frame_tail = 0xBB
        # 打包数据：X速度 Y速度
        data = struct.pack('<hh', speed_x, speed_y)
        check_sum = sum(data) & 0xFF  # 简单校验

        # 4. 拼接完整帧并发送
        full_frame = bytes([frame_head]) + data + bytes([check_sum, frame_tail])
        self.ser.write(full_frame)

        self.get_logger().info(f"🎮 云台指令：X速度={speed_x} | Y速度={speed_y}")

    def image_callback(self, msg):
        try:
            frame = self.cv_bridge.imgmsg_to_cv2(msg, 'bgr8')
            shape_name, target_center = self.detect_target_shape(frame)

            # 识别成功 → 计算偏差 + 发送云台指令
            if shape_name and target_center:
                dx, dy, img_cx, img_cy = self.calculate_deviation(target_center, frame.shape)
                
                # 绘制可视化
                cv2.circle(frame, (img_cx, img_cy), 6, (255,0,0), -1)
                cv2.line(frame, target_center, (img_cx, img_cy), (0,255,255), 2)
                cv2.putText(frame, f"DX:{dx} DY:{dy}", (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)

                # 发布结果
                result_msg = String()
                result_msg.data = f"{shape_name} | X:{dx} Y:{dy}"
                self.pub_result.publish(result_msg)

                # 发送指令控制云台
                self.send_gimbal_command(dx, dy)

            # 发布处理后的图像
            out_msg = self.cv_bridge.cv2_to_imgmsg(frame, 'bgr8')
            self.pub_image.publish(out_msg)

        except Exception as e:
            self.get_logger().error(str(e))

    # ===================== 抗干扰图形识别（保留最优逻辑） =====================
    def detect_target_shape(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape

        _, binary_fixed = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY_INV)
        binary_adaptive = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 51, 5)
        binary = cv2.bitwise_and(binary_fixed, binary_adaptive)

        kernel = np.ones((5,5), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=3)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)

        contours, hierarchy = cv2.findContours(
            binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

        shape_name = None
        cx, cy = 0, 0
        found = False

        for i in range(len(contours)):
            cnt = contours[i]
            area = cv2.contourArea(cnt)
            if not (0.01 * h*w < area < 0.3 * h*w):
                continue
            if hierarchy[0][i][2] == -1:
                continue
            inner_idx = hierarchy[0][i][2]
            inner_cnt = contours[inner_idx] if inner_idx < len(contours) else None
            if inner_cnt is None or cv2.contourArea(inner_cnt) < 0.1 * area:
                continue

            hull = cv2.convexHull(cnt)
            hull_area = cv2.contourArea(hull)
            solidity = area / hull_area if hull_area !=0 else 0
            if solidity < 0.85:
                continue

            x, y, w_rect, h_rect = cv2.boundingRect(cnt)
            aspect_ratio = w_rect / h_rect
            if aspect_ratio < 0.5 or aspect_ratio > 2.0:
                continue

            M = cv2.moments(cnt)
            if M["m00"] == 0:
                continue
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])

            roi = 25
            x1, y1 = max(0, cx-roi), max(0, cy-roi)
            x2, y2 = min(w, cx+roi), min(h, cy+roi)
            center_patch = gray[y1:y2, x1:x2]
            if np.mean(center_patch) < 150:
                continue

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
                peri = cv2.arcLength(cnt, True)
                circularity = 4 * np.pi * area / (peri ** 2) if peri != 0 else 0
                if 0.8 < circularity < 1.15:
                    shape_name = "圆形"

            if shape_name:
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
        # 关闭串口
        if self.ser and self.ser.is_open:
            self.ser.close()
        super().destroy_node()

def main():
    rclpy.init()
    node = ShapeSubscriber()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
