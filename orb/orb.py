from ament_index_python.packages import get_package_share_directory
import cv2
from cv_bridge import CvBridge
import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from geometry_msgs.msg import Pose, PoseArray
from sensor_msgs.msg import Image
from std_msgs.msg import Float32
from scipy.spatial.transform import Rotation
import os
import orbslam2
import time

class OrbNode(Node):
    def __init__(self):
        super().__init__('orb_node')
        self.bridge = CvBridge()

        settings_file = os.path.join(get_package_share_directory('orb'), 'config', 'uffda.yaml')
        self.slam = orbslam2.System("/tmp/ORB_SLAM2/Vocabulary/ORBvoc.txt", settings_file, orbslam2.Sensor.MONOCULAR)
        self.slam.set_use_viewer(False)
        self.slam.initialize()

        self.cam_sub = self.create_subscription(Image, "/BlueROV2/video", self.camera_cb, 1)
        self.cam_pub = self.create_publisher(Image, "/monovo/debug", 1)
        self.path_publisher = self.create_publisher(PoseArray, '/monovo/path', 1)
        self.confidence_publisher = self.create_publisher(Float32, '/monovo/ba_confidence', 1)

        self.tracked_pixels_2d = np.empty((0, 2))
        self.tracked_points_3d = np.empty((0, 3))

        # append-only, growing every tracked frame (mirrors the dpvo node's /dpvo/path
        # contract); get_trajectory_points() returns the full history each call, so we
        # only append the tail that's new since the last callback
        self.path = PoseArray()
        self.path.header.frame_id = 'map'
        self.trajectory_len = 0


    def camera_cb(self, msg: Image):
        bgr = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        ts = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        self.slam.process_image_mono(bgr, time.monotonic())

        features = np.array(self.slam.get_tracked_features(), dtype=np.float64)
        if features.size:
            self.tracked_pixels_2d = features[:, 0:2]
            self.tracked_points_3d = features[:, 2:5]
        else:
            self.tracked_pixels_2d = np.empty((0, 2))
            self.tracked_points_3d = np.empty((0, 3))

        annotated = bgr.copy()
        for x, y in self.tracked_pixels_2d:
            cv2.circle(annotated, (int(round(x)), int(round(y))), 2, (0, 255, 0), -1)

        out_msg = self.bridge.cv2_to_imgmsg(annotated, encoding='passthrough')
        out_msg.encoding = 'bgr8'
        out_msg.header = msg.header
        self.cam_pub.publish(out_msg)

        # confidence: matched/total feature ratio, gated to 0.0 when tracking isn't OK
        # so a lost/uninitialized tracker can't report a misleadingly high ratio
        num_features = self.slam.get_num_features()
        if self.slam.get_tracking_state() == orbslam2.TrackingState.OK and num_features > 0:
            confidence = np.sqrt(self.slam.get_num_matched_features() / num_features)
        else:
            confidence = 0.0
        self.confidence_publisher.publish(Float32(data=float(confidence)))

        # trajectory poses are already camera-to-world (world_T_cam) in ORB_SLAM2's
        # OpenCV/optical convention, matching the dpvo node's raw /dpvo/path poses --
        # leave the ROS-frame conversion to main.py, don't re-convert here
        trajectory = self.slam.get_trajectory_points()
        for _, r00, r01, r02, tx, r10, r11, r12, ty, r20, r21, r22, tz in trajectory[self.trajectory_len:]:
            rotation_matrix = np.array([[r00, r01, r02], [r10, r11, r12], [r20, r21, r22]])
            ros_pose = Pose()
            ros_pose.position.x, ros_pose.position.y, ros_pose.position.z = (float(v) for v in (tx, ty, tz))
            qx, qy, qz, qw = Rotation.from_matrix(rotation_matrix).as_quat()
            ros_pose.orientation.x, ros_pose.orientation.y, ros_pose.orientation.z, ros_pose.orientation.w = (
                float(qx), float(qy), float(qz), float(qw)
            )
            self.path.poses.append(ros_pose)
        self.trajectory_len = len(trajectory)

        self.path.header.stamp = msg.header.stamp
        self.path_publisher.publish(self.path)


def main(args=None):
    rclpy.init(args=args)
    node = OrbNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
