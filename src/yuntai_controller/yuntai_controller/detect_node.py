#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

class CamPublisher(Node):
    def __init__(self):
        super().__init__('cam_publisher')
        
        # 创建发布者
        self.pub = self.create_publisher(Image, 'camera/image_raw', 10)
        
        # 定时器：10Hz (0.1秒)
        self.timer = self.create_timer(0.1, self.publish_frame)
        
        # 初始化工具
        self.cv_bridge = CvBridge()
        
        # 打开摄像头（根据你的实际情况改索引：0, 1, 2...）
        self.cap = cv2.VideoCapture(2)
        
        # 检查摄像头
        if not self.cap.isOpened():
            self.get_logger().error("❌ 无法打开摄像头！请检查设备索引或连接。")
            raise RuntimeError("摄像头打开失败")
        
        self.get_logger().info("✅ 图像发布节点已启动！正在发布到 /camera/image_raw")

    def publish_frame(self):
        ret, frame = self.cap.read()
        if ret:
            try:
                # 转换成ROS消息并发布
                img_msg = self.cv_bridge.cv2_to_imgmsg(frame, 'bgr8')
                self.pub.publish(img_msg)
            except Exception as e:
                self.get_logger().error(f"发布错误: {e}")
        else:
            self.get_logger().warn("⚠️ 无法读取摄像头帧")

    def destroy_node(self):
        if self.cap.isOpened():
            self.cap.release()
            self.get_logger().info("📹 摄像头已释放")
        super().destroy_node()

def main():
    rclpy.init()
    node = None
    try:
        node = CamPublisher()
        rclpy.spin(node)
    except Exception as e:
        print(f"错误: {e}")
    finally:
        if node:
            node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
