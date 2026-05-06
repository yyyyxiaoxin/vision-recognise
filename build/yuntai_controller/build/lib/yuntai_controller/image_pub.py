import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2                              # OpenCV图像处理库


class ImagePublisherNode(Node):
    def __init__(self,name):
        super().__init__(name)
        self.pub = self.create_publisher(Image,"camera/image_raw",10)
        self.timer = self.create_timer(0.1,self.listener_cb)
        self.cap = cv2.VideoCapture(2)
        self.cv_bridge = CvBridge()

    def listener_cb(self):
        ret, frame = self.cap.read()
        if ret == True:
            self.pub.publish(self.cv_bridge.cv2_to_imgmsg(frame,'bgr8'))
            
        self.get_logger().info('Publishing video frame')

def main():
    rclpy.init()
    node = ImagePublisherNode("topic_cam_pub")
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()