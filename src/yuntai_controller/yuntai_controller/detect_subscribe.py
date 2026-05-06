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

        # 1. 订阅相机图像
        self.sub = self.create_subscription(
            Image, '/camera/image_raw', self.image_callback, 10)
        
        # 2. 发布处理后的图像
        self.pub_image = self.create_publisher(Image, 'shape_image', 10)
        # 3. 发布识别结果+偏差
        self.pub_result = self.create_publisher(String, 'shape_result', 10)

        self.cv_bridge = CvBridge()

        self.declare_parameter('debug_serial_port', '/dev/ttyGS0')
        self.declare_parameter('debug_serial_baudrate', 115200)
        self.declare_parameter('shape_serial_port', '/dev/ttyS3')
        self.declare_parameter('shape_serial_baudrate', 115200)

        self.debug_serial_port = self.get_parameter(
            'debug_serial_port').get_parameter_value().string_value
        self.debug_serial_baudrate = self.get_parameter(
            'debug_serial_baudrate').get_parameter_value().integer_value
        self.shape_serial_port = self.get_parameter(
            'shape_serial_port').get_parameter_value().string_value
        self.shape_serial_baudrate = self.get_parameter(
            'shape_serial_baudrate').get_parameter_value().integer_value

        self.debug_ser = self.open_serial(
            self.debug_serial_port,
            self.debug_serial_baudrate,
            '调试串口'
        )
        self.shape_ser = self.open_serial(
            self.shape_serial_port,
            self.shape_serial_baudrate,
            '图形结果串口'
        )

        self.shape_serial_map = {
            '三角形': 'TRIANGLE',
            '矩形': 'RECTANGLE',
            '五边形': 'PENTAGON',
            '圆形': 'CIRCLE',
        }

        self.get_logger().info("✅ 图形识别+双串口输出节点启动！")

    # ===================== 你的偏差计算函数 =====================
    def calculate_deviation(self, target_center, frame_shape):
        h, w = frame_shape[:2]
        img_center_x = w // 2
        img_center_y = h // 2
        dx = target_center[0] - img_center_x
        dy = target_center[1] - img_center_y
        return dx, dy, img_center_x, img_center_y

    def open_serial(self, port, baudrate, label):
        try:
            ser = serial.Serial(
                port=port,
                baudrate=baudrate,
                timeout=1
            )
            self.get_logger().info(
                f"✅ {label}已连接: {port} @ {baudrate}"
            )
            return ser
        except Exception as e:
            self.get_logger().error(
                f"❌ {label}打开失败({port}): {str(e)}"
            )
            return None

    def write_serial_line(self, ser, payload, label):
        if ser is None or not ser.is_open:
            return

        try:
            ser.write((payload + '\r\n').encode('utf-8'))
        except Exception as e:
            self.get_logger().error(f"{label}发送失败: {str(e)}")

    def send_debug_offset(self, dx, dy):
        payload = f"DX:{dx},DY:{dy}"
        self.write_serial_line(self.debug_ser, payload, '调试串口')

    def send_shape_result_serial(self, shape_name):
        shape_code = self.shape_serial_map.get(shape_name, 'NONE')
        payload = f"SHAPE:{shape_code}"
        self.write_serial_line(self.shape_ser, payload, '图形结果串口')

    def image_callback(self, msg):
        try:
            frame = self.cv_bridge.imgmsg_to_cv2(msg, 'bgr8')
            shape_name, target_center = self.detect_target_shape(frame)

            result_msg = String()

            # 识别成功 → 计算偏差 + 双串口发送
            if shape_name and target_center:
                dx, dy, img_cx, img_cy = self.calculate_deviation(target_center, frame.shape)
                
                # 绘制可视化
                cv2.circle(frame, (img_cx, img_cy), 6, (255,0,0), -1)
                cv2.line(frame, target_center, (img_cx, img_cy), (0,255,255), 2)
                cv2.putText(frame, f"DX:{dx} DY:{dy}", (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)

                # 发布结果
                result_msg.data = f"{shape_name} | X:{dx} Y:{dy}"
                self.pub_result.publish(result_msg)

                # 调试串口发偏差，业务串口发形状
                self.send_debug_offset(dx, dy)
                self.send_shape_result_serial(shape_name)
            else:
                result_msg.data = "未识别到目标"
                self.pub_result.publish(result_msg)
                self.send_shape_result_serial(None)

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
        if self.debug_ser and self.debug_ser.is_open:
            self.debug_ser.close()
        if self.shape_ser and self.shape_ser.is_open:
            self.shape_ser.close()
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
