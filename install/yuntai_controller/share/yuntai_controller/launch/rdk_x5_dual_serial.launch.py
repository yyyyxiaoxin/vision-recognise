from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    camera_device = LaunchConfiguration('camera_device')
    debug_serial_port = LaunchConfiguration('debug_serial_port')
    debug_serial_baudrate = LaunchConfiguration('debug_serial_baudrate')
    shape_serial_port = LaunchConfiguration('shape_serial_port')
    shape_serial_baudrate = LaunchConfiguration('shape_serial_baudrate')

    return LaunchDescription([
        DeclareLaunchArgument(
            'camera_device',
            default_value='2',
            description='OpenCV camera index used by detect_node',
        ),
        DeclareLaunchArgument(
            'debug_serial_port',
            default_value='/dev/ttyGS0',
            description='RDK X5 Micro-USB debug serial device',
        ),
        DeclareLaunchArgument(
            'debug_serial_baudrate',
            default_value='115200',
            description='Baudrate for the Micro-USB debug serial',
        ),
        DeclareLaunchArgument(
            'shape_serial_port',
            default_value='/dev/ttyS3',
            description='Serial device used to send shape recognition results',
        ),
        DeclareLaunchArgument(
            'shape_serial_baudrate',
            default_value='115200',
            description='Baudrate for the shape result serial port',
        ),
        Node(
            package='yuntai_controller',
            executable='detect_node',
            name='cam_publisher',
            output='screen',
            parameters=[{
                'camera_device': camera_device,
            }],
        ),
        Node(
            package='yuntai_controller',
            executable='detect_subscribe',
            name='shape_subscriber',
            output='screen',
            parameters=[{
                'debug_serial_port': debug_serial_port,
                'debug_serial_baudrate': debug_serial_baudrate,
                'shape_serial_port': shape_serial_port,
                'shape_serial_baudrate': shape_serial_baudrate,
            }],
        ),
        Node(
            package='rqt_image_view',
            executable='rqt_image_view',
            name='shape_image_view',
            output='screen',
            arguments=['/shape_image'],
        ),
    ])
