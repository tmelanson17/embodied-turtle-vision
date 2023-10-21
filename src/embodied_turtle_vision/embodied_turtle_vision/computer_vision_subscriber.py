#!/home/noisebridge/miniconda3/envs/ros-embodied-turtle-vision/bin/python
"""Run inference on RGB image data published to a ROS2 topic.

"""
import cv2
import numpy as np
import rclpy
import torch
import math

from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Int32

BBOX_COLOR_RGB = (0, 255, 255)
BBOX_WIDTH = 1
DEBUG = True


class ComputerVisionSubscriber(Node):

    def __init__(self, topic: str, n_fps: float, target_class: int = 0):
        """Constructor.

        Subscribe to ROS2 topic `topic`, get new data for running inference
        at `n_fps` frames per second.

        Args:
            topic (str): ROS2 topic to subscribe to.
            n_fps (float): Topic image sample rate frames per second.
            target_class (int): Detection class to filter for. Default is 0 
                (human).

        """
        super().__init__('computer_vision_subscriber')

        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5m', pretrained=True)
        self.bridge = CvBridge()
        self.subscription = self.create_subscription(
            Image, topic, self.listener_callback, 10)
        if DEBUG:
            self.debug_image_publisher = self.create_publisher(Image, '/debug/image', 10)
        self.control_publisher = self.create_publisher(Int32, '/ai/follow', 10)
        self.timer_period = 1 / n_fps
        self.timer = self.create_timer(self.timer_period, self.run_inference)
        self.target_class = target_class
        self.prev_result = None

    def listener_callback(self, msg):
        self.latest_img = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        return

    def run_inference(self):
        if hasattr(self, 'latest_img'):
            results = self.model(self.latest_img).pandas().xyxy[0]
            # Filter results to only include the target class
            results = results[results['class'] == self.target_class]
            results = results[results['confidence'] > 0.6]
            # TODO: Filter a result that's within N pixels of the previous result
            if self.prev_result is None and not results.empty:
                self.prev_result = results.iloc[0]
            elif self.prev_result is not None:
                best_result = None
                for _, r in results.iterrows():
                    center_results_x = math.sqrt((r['xmax'] - r['xmin'])**2)
                    center_results_y = math.sqrt((r['ymax'] - r['ymin'])**2)
                    center_current_result_x = math.sqrt((self.prev_result['xmax'] - self.prev_result['xmin'])**2)
                    center_current_result_y = math.sqrt((self.prev_result['ymax'] - self.prev_result['ymin'])**2) 
                    delta = center_current_result_x - center_results_x
                    arbitrary_distance_x = 100
                    if delta <= arbitrary_distance_x:
                        best_result = r
                if best_result is not None:
                    self.prev_result = r
            if DEBUG and self.prev_result is not None:
                debug_img = self.latest_img.copy()
                cv2.rectangle(
                    debug_img, 
		    (int(self.prev_result['xmin']), int(self.prev_result['ymin'])), 
		    (int(self.prev_result['xmax']), int(self.prev_result['ymax'])), 
		    BBOX_COLOR_RGB,
		    BBOX_WIDTH)

                debug_msg = self.bridge.cv2_to_imgmsg(debug_img, 'bgr8')
                self.debug_image_publisher.publish(debug_msg)

            if not results.empty:
                areas = (results['xmax'] - results['xmin']) * (results['ymax'] - results['ymin'])
                max_area_idx = areas.idxmax()
                bbox = results.loc[max_area_idx, ['xmin', 'ymin', 'xmax', 'ymax']]
                mid_x = ((bbox['xmax'] + bbox['xmin']) / 2) / self.latest_img.shape[1]
                mid_x = min(0.999, max(0, mid_x))
                value_to_publish = int(mid_x * 3)  # Using x-axis value. Change to mid_y for y-axis.
            else:
                value_to_publish = 3    
            self.control_publisher.publish(Int32(data=value_to_publish))

        return


def main(args=None):
    rclpy.init(args=args)
    topic = '/oakd/rgb/preview/image_raw'
    n_fps = 5
    target_class = 0
    cv_subscriber = ComputerVisionSubscriber(topic, n_fps, target_class)
    rclpy.spin(cv_subscriber)
    cv_subscriber.destroy_node()
    rclpy.shutdown()
    return


if __name__ == '__main__':
    main()

