# Copyright 2023, Evan Palmer
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

import time
from abc import ABC, abstractmethod
from collections import deque
from typing import Any, Deque

import cv2
import numpy as np
import math
import rclpy
import tf2_geometry_msgs  # noqa
from cv_bridge import CvBridge
from geometry_msgs.msg import (
    Pose,
    PoseStamped,
    PoseWithCovarianceStamped,
    TwistStamped,
    TwistWithCovarianceStamped,
)
from nav_msgs.msg import Odometry
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
    qos_profile_system_default,
    qos_profile_sensor_data,
)
from scipy.spatial.transform import Rotation as R
from sensor_msgs.msg import CameraInfo, Image
from tf2_ros import TransformException  # type: ignore
from tf2_ros import Time
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from tf2_ros import TransformBroadcaster
from geometry_msgs.msg import TransformStamped


class Localizer(Node, ABC):
    """Base class for implementing a visual localization interface."""

    MAP_FRAME = "map"
    MAP_NED_FRAME = "map_ned"
    BASE_LINK_FRAME = "base_link"
    BASE_LINK_FRD_FRAME = "base_link_frd"
    CAMERA_FRAME = "camera_link"

    def __init__(self, node_name: str) -> None:
        """Create a new localizer.

        Args:
            node_name: The name of the ROS 2 node.
        """
        Node.__init__(self, node_name)
        ABC.__init__(self)

        self.declare_parameter("update_rate", 30.0)

        # Provide access to TF2
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Publish the current state at the provided rate. Note that, if the localizer
        # receives state messages at a lower rate, the state will be published at the
        # rate at which it is received (basically just a low-pass filter). The reason
        # for applying a filter is to ensure that high-frequency state updates don't
        # overload the FCU.
        self._state = None
        self._update_rate = 1 / (
            self.get_parameter("update_rate").get_parameter_value().double_value
        )
        self._last_update = time.time()
        self.update_state_timer = self.create_timer(
            self._update_rate, self._publish_wrapper, MutuallyExclusiveCallbackGroup()
        )

    @property
    def state(self) -> Any:
        """Get the current state obtained by a localizer.

        Returns:
            The current state.
        """
        return self._state

    @state.setter
    def state(self, state: Any) -> None:
        """Set the current state to be published by the EKF.

        Args:
            state: The current state.
        """
        self._last_update = time.time()
        self._state = state

    def _publish_wrapper(self) -> None:
        """Publish the state at the max allowable frequency.

        If the state hasn't been updated since the last loop iteration, don't publish
        the state again: only publish a state once.
        """
        if self.state is None or time.time() - self._last_update > self._update_rate:
            return

        self.publish()

    @abstractmethod
    def publish(self) -> None:
        """Publish the state to the ArduSub EKF.

        This is automatically called by the localizer timer and should not be called
        manually.
        """
        ...


class PoseLocalizer(Localizer):
    """Interface for sending pose estimates to the ArduSub EKF."""

    def __init__(self, node_name: str) -> None:
        """Create a new pose localizer.

        Args:
            node_name: The name of the localizer node.
        """
        super().__init__(node_name)

        # Poses are sent to the ArduPilot EKF
        self.vision_pose_pub = self.create_publisher(
            PoseStamped,
            "/mavros/vision_pose/pose",
            qos_profile_system_default,
        )
        self.vision_pose_cov_pub = self.create_publisher(
            PoseWithCovarianceStamped,
            "/mavros/vision_pose/pose_cov",
            qos_profile_system_default,
        )

    def publish(self) -> None:
        """Publish a pose message to the ArduSub EKF."""
        if isinstance(self.state, PoseStamped):
            self.vision_pose_pub.publish(self.state)
        elif isinstance(self.state, PoseWithCovarianceStamped):
            self.vision_pose_cov_pub.publish(self.state)
        else:
            raise TypeError(
                "Invalid state type provided for publishing. Expected one of"
                f" {PoseStamped.__name__}, {PoseWithCovarianceStamped.__name__}: got"
                f" {self.state.__class__.__name__}"
            )


class TwistLocalizer(Localizer):
    """Interface for sending pose estimates to the ArduSub EKF."""

    def __init__(self, node_name: str) -> None:
        """Create a new pose localizer.

        Args:
            node_name: The name of the localizer node.
        """
        super().__init__(node_name)

        # Twists are sent to the ArduPilot EKF
        self.vision_speed_pub = self.create_publisher(
            TwistStamped, "/mavros/vision_speed/speed", qos_profile_system_default
        )
        self.vision_speed_cov_pub = self.create_publisher(
            TwistWithCovarianceStamped,
            "/mavros/vision_speed/speed_cov",
            qos_profile_system_default,
        )

    def publish(self) -> None:
        """Publish a twist message to the ArduSub EKF."""
        if isinstance(self.state, TwistStamped):
            self.vision_speed_pub.publish(self.state)
        elif isinstance(self.state, TwistWithCovarianceStamped):
            self.vision_speed_cov_pub.publish(self.state)
        else:
            raise TypeError(
                "Invalid state type provided for publishing. Expected one of"
                f" {TwistStamped.__name__}, {TwistWithCovarianceStamped.__name__}: got"
                f" {self.state.__class__.__name__}"
            )


class ArucoMarkerLocalizer(PoseLocalizer):
    """Performs localization using ArUco markers."""

    ARUCO_MARKER_TYPES = [
        cv2.aruco.DICT_4X4_50,
        cv2.aruco.DICT_4X4_100,
        cv2.aruco.DICT_4X4_250,
        cv2.aruco.DICT_4X4_1000,
        cv2.aruco.DICT_5X5_50,
        cv2.aruco.DICT_5X5_100,
        cv2.aruco.DICT_5X5_250,
        cv2.aruco.DICT_5X5_1000,
        cv2.aruco.DICT_6X6_50,
        cv2.aruco.DICT_6X6_100,
        cv2.aruco.DICT_6X6_250,
        cv2.aruco.DICT_6X6_1000,
        cv2.aruco.DICT_7X7_50,
        cv2.aruco.DICT_7X7_100,
        cv2.aruco.DICT_7X7_250,
        cv2.aruco.DICT_7X7_1000,
        cv2.aruco.DICT_ARUCO_ORIGINAL,
    ]

    def __init__(self) -> None:
        """Create a new ArUco marker localizer."""
        super().__init__("aruco_marker_localizer")

        self.bridge = CvBridge()
        self.camera_info: CameraInfo | None = None

        self.camera_info_sub = self.create_subscription(
            CameraInfo,
            "/camera/camera_info",
            self.get_camera_info_cb,
            QoSProfile(
                reliability=ReliabilityPolicy.RELIABLE,
                durability=DurabilityPolicy.TRANSIENT_LOCAL,
                history=HistoryPolicy.KEEP_LAST,
                depth=1,
            ),
        )
        self.camera_sub = self.create_subscription(
            Image,
            "/camera/image_raw",
            self.update_pose_cb,
            qos_profile_sensor_data,
        )

         # Publisher to send the image with detected markers
        self.image_pub = self.create_publisher(
            Image,
            "/camera/image_with_markers",
            qos_profile_sensor_data,
        )

    def get_camera_info_cb(self, info: CameraInfo) -> None:
        """Get the camera info from the camera.

        Args:
            info: The camera meta information.
        """
        self.camera_info = info

    def detect_markers(self, frame: np.ndarray) -> tuple[Any, Any] | None:
        """Detect any ArUco markers in the frame.

        All markers in a frame should be the same type of ArUco marker
        (e.g., 4x4 50) if multiple are expected to be in-frame.

        Args:
            frame: The video frame containing ArUco markers.

        Returns:
            A list of marker corners and IDs. If no markers were found, returns None.
        """
        # Check each tag type, breaking when we find one that works
        for tag_type in self.ARUCO_MARKER_TYPES:
            aruco_dict = cv2.aruco.Dictionary_get(tag_type)
            aruco_params = cv2.aruco.DetectorParameters_create()
            aruco_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
            # Create the ArUco detector with the desired dictionary and parameters
            # aruco_dict = cv2.aruco.getPredefinedDictionary(tag_type)
            # aruco_params = cv2.aruco.DetectorParameters()
            # aruco_detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)

            try:
                corners, ids, _ = cv2.aruco.detectMarkers(
                    frame, aruco_dict, parameters=aruco_params)
                # Detect markers in the frame
                # corners, ids, _ = aruco_detector.detectMarkers(frame)

                if ids is not None and len(ids) > 0:
                    return corners, ids

            except Exception as e:
                print(f"Error during marker detection: {e}")

        # Nothing was found
        return None

    def get_camera_pose(self, frame: np.ndarray) -> tuple[Any, Any, int] | None:
        """Get the pose of the camera relative to any ArUco markers detected.

        If multiple markers are detected, then the "largest" marker will be used to
        determine the pose of the camera.

        Args:
            frame: The camera frame containing ArUco markers.

        Returns:
            The rotation vector and translation vector of the camera in the marker
            frame and the ID of the marker detected. If no marker was detected,
            returns None.
        """
        # Wait to process frames until we get the camera meta info
        if self.camera_info is None:
            return None

        # Convert to greyscale image then try to detect the tag(s)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detection = self.detect_markers(gray)

        if detection is None:
            return None

        corners, ids = detection

        # If there are multiple markers, get the marker with the "longest" side, where
        # "longest" should be interpreted as the relative size in the image
        side_lengths = [
            abs(corner[0][0][0] - corner[0][2][0])
            + abs(corner[0][0][1] - corner[0][2][1])
            for corner in corners
        ]

        min_side_idx = side_lengths.index(max(side_lengths))
        # min_marker_id = int(ids[min_side_idx])
        min_marker_id = int(0)

        camera_matrix = np.array(self.camera_info.k, dtype=np.float64).reshape(3, 3)
        projection_matrix = np.array(self.camera_info.d, dtype=np.float64).reshape(1, 5)

        # Get the estimated pose
        # marker_size = 0.5
        marker_size = 0.5
        marker_points = np.array([[-marker_size / 2.0, marker_size / 2.0, 0],
                                [marker_size / 2.0, marker_size / 2.0, 0],
                                [marker_size / 2.0, -marker_size / 2.0, 0],
                                [-marker_size / 2.0, -marker_size / 2.0, 0]], dtype=np.float32)

        # rot_vec, trans_vec, _ = cv2.aruco.estimatePoseSingleMarkers(
        #     corners[min_side_idx], marker_size, camera_matrix, projection_matrix
        # )
        # solvePnP returns the rotation and translation vectors
        retval, rot_vec, trans_vec = cv2.solvePnP(objectPoints=marker_points, imagePoints=corners[min_side_idx],
                                                  cameraMatrix=camera_matrix, distCoeffs=projection_matrix,
                                                  flags=cv2.SOLVEPNP_IPPE_SQUARE)
        rot_vec = rot_vec.reshape(3, 1)
        trans_vec = trans_vec.reshape(3, 1)

        # ------------------------------------------------------------------------------------------------------------
        # Define a rotation matrix to fix the pose
        # R_correction = np.array([
        #     [0, 0, 1],
        #     [0, 1, 0],
        #     [-1, 0, 0]
        # ], dtype=np.float32)
        R_correction = np.array([
            [0, 0, 1],
            [0, -1, 0],
            [1, 0, 0]
        ], dtype=np.float32)

        # Convert rvec to a rotation matrix
        R_marker, _ = cv2.Rodrigues(rot_vec)

        # Apply the correction
        R_fixed = R_marker @ R_correction

        # Convert back to rvec
        rot_vec, _ = cv2.Rodrigues(R_fixed)
        # ------------------------------------------------------------------------------------------------------------

        # Draw detected markers on the frame
        cv2.aruco.drawDetectedMarkers(frame, corners, ids)
        cv2.drawFrameAxes(frame, camera_matrix, projection_matrix, rot_vec, trans_vec, 0.5, 5)
        # Publish the image with detected markers
        try:
            ros_image = self.bridge.cv2_to_imgmsg(frame, encoding="rgb8")
            self.image_pub.publish(ros_image)
        except Exception as e:
            self.get_logger().error(f"Failed to publish image: {str(e)}")

        return rot_vec, trans_vec, min_marker_id

    def update_pose_cb(self, frame: Image) -> None:
        """Get the camera pose relative to the marker and send to the ArduSub EKF.

        Args:
            frame: The BlueROV2 camera frame.
        """
        # Get the pose of the camera in the `marker` frame
        camera_pose = self.get_camera_pose(self.bridge.imgmsg_to_cv2(frame))

        # If there was no marker in the image, exit early
        if camera_pose is None:
            self.get_logger().debug(
                "An ArUco marker could not be detected in the current image"
            )
            return

        rot_vec, trans_vec, marker_id = camera_pose

        # Convert the pose into a PoseStamped message
        pose = PoseStamped()

        pose.header.frame_id = f"marker_{marker_id}"
        pose.header.stamp = self.get_clock().now().to_msg()

        (
            pose.pose.position.x,
            pose.pose.position.y,
            pose.pose.position.z,
        ) = trans_vec.squeeze()

        rot_mat, _ = cv2.Rodrigues(rot_vec)

        (
            pose.pose.orientation.x,
            pose.pose.orientation.y,
            pose.pose.orientation.z,
            pose.pose.orientation.w,
        ) = R.from_matrix(rot_mat).as_quat()

        # Transform the pose from the `marker` frame to the `map` frame
        try:
            pose = self.tf_buffer.transform(pose, self.MAP_FRAME)
        except TransformException as e:
            self.get_logger().warning(
                f"Could not transform from frame marker_{marker_id} to map: {e}"
            )
            return

        # The pose now represents the transformation from the map frame to the
        # camera frame, but we need to publish the transformation from the map frame
        # to the base_link frame

        # Start by getting the camera to base_link transform
        try:
            tf_camera_to_base = self.tf_buffer.lookup_transform(
                self.CAMERA_FRAME, self.BASE_LINK_FRAME, Time()
            )
        except TransformException as e:
            self.get_logger().warning(f"Could not access transform: {e}")
            return

        # Convert the tf into a homogeneous tf matrix
        tf_camera_to_base_mat = np.eye(4)
        tf_camera_to_base_mat[:3, :3] = R.from_quat(
            [
                tf_camera_to_base.transform.rotation.x,
                tf_camera_to_base.transform.rotation.y,
                tf_camera_to_base.transform.rotation.z,
                tf_camera_to_base.transform.rotation.w,
            ]
        ).as_matrix()
        tf_camera_to_base_mat[:3, 3] = np.array(
            [
                tf_camera_to_base.transform.translation.x,
                tf_camera_to_base.transform.translation.y,
                tf_camera_to_base.transform.translation.z,
            ]
        )

        # Convert the pose back into a matrix
        tf_map_to_camera_mat = np.eye(4)
        tf_map_to_camera_mat[:3, :3] = R.from_quat(
            [
                pose.pose.orientation.x,  # type: ignore
                pose.pose.orientation.y,  # type: ignore
                pose.pose.orientation.z,  # type: ignore
                pose.pose.orientation.w,  # type: ignore
            ]
        ).as_matrix()
        tf_map_to_camera_mat[:3, 3] = np.array(
            [
                pose.pose.position.x,  # type: ignore
                pose.pose.position.y,  # type: ignore
                pose.pose.position.z,  # type: ignore
            ]
        )

        # Swap
        # swap_axes_mat = np.array([
        #     [0,  0,  -1,  -.21],  # x
        #     [0,  1,  0,  0],  # y
        #     [1,  0,  0,  .067],  # z
        #     [0,  0,  0,  1],  # homogeneous coordinate unchanged
        # ])
        swap_axes_mat = np.array([
            [ 0,  0, 1, 0],  # x
            [ 1, 0, 0, 0],  # y
            [0,  1, 0, 0],  # z
            [ 0,  0, 0, 1],  # homogeneous coordinate unchanged
        ], dtype=np.float32)

        # Calculate the new transform
        # tf_map_to_base_mat = tf_camera_to_base_mat @ tf_map_to_camera_mat

        # Calculate the new transform with the axis swap
        tf_map_to_base_mat = swap_axes_mat @ tf_camera_to_base_mat @ tf_map_to_camera_mat

        # Update the pose using the new transform
        (
            pose.pose.position.x,  # type: ignore
            pose.pose.position.y,  # type: ignore
            pose.pose.position.z,  # type: ignore
        ) = tf_map_to_base_mat[:3, 3]

        pose.pose.position.z = -pose.pose.position.z

        (
            pose.pose.orientation.x,  # type: ignore
            pose.pose.orientation.y,  # type: ignore
            pose.pose.orientation.z,  # type: ignore
            pose.pose.orientation.w,  # type: ignore
        ) = R.from_matrix(tf_map_to_base_mat[:3, :3]).as_quat()

        # Euler angle format in radians
        roll_x, pitch_y, yaw_z = self.euler_from_quaternion(
            pose.pose.orientation.x, pose.pose.orientation.y,
            pose.pose.orientation.z, pose.pose.orientation.w)
        print(f"X: {pose.pose.position.x} | Y: {pose.pose.position.y} | Z: {pose.pose.position.z}")
        print(f"radians - roll_x: {roll_x} | pitch_y: {pitch_y} | yaw_z: {yaw_z}")
        roll_x = math.degrees(roll_x)
        pitch_y = math.degrees(pitch_y)
        yaw_z = math.degrees(yaw_z)
        print(f"degrees - roll_x: {roll_x} | pitch_y: {pitch_y} | yaw_z: {yaw_z}")

        # print(f'pose: {pose}')

        self.state = pose

    def euler_from_quaternion(self, x, y, z, w):
        """
        Convert a quaternion into euler angles (roll, pitch, yaw)
        roll is rotation around x in radians (counterclockwise)
        pitch is rotation around y in radians (counterclockwise)
        yaw is rotation around z in radians (counterclockwise)
        """
        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        roll_x = math.atan2(t0, t1)
            
        t2 = +2.0 * (w * y - z * x)
        t2 = +1.0 if t2 > +1.0 else t2
        t2 = -1.0 if t2 < -1.0 else t2
        pitch_y = math.asin(t2)
            
        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw_z = math.atan2(t3, t4)
            
        return roll_x, pitch_y, yaw_z # in radians


class QualisysLocalizer(PoseLocalizer):
    """Localize the BlueROV2 using the Qualisys motion capture system."""

    def __init__(self) -> None:
        """Create a new Qualisys motion capture localizer."""
        super().__init__("qualisys_localizer")

        self.declare_parameter("body", "bluerov")
        self.declare_parameter("filter_len", 20)

        body = self.get_parameter("body").get_parameter_value().string_value
        filter_len = (
            self.get_parameter("filter_len").get_parameter_value().integer_value
        )

        self.mocap_sub = self.create_subscription(
            PoseStamped,
            f"/blue/mocap/qualisys/{body}",
            self.update_pose_cb,
            qos_profile_sensor_data,
        )

        # Publish to the MoCap interface instead of the default pose interface
        self.mocap_pose_pub = self.create_publisher(
            PoseStamped,
            "/mavros/mocap/pose",
            qos_profile_system_default,
        )

        # Store the pose information in a buffer and apply an LWMA filter to it
        self.pose_buffer: Deque[np.ndarray] = deque(maxlen=filter_len)

    @staticmethod
    def check_isnan(pose: PoseStamped) -> bool:
        """Check if a pose message has NaN values.

        NaN values are not uncommon when dealing with MoCap data.

        Args:
            pose: The message to check for NaN values.

        Returns:
            Whether or not the message has any NaN values.
        """
        # Check the position
        if np.isnan(
            np.min(
                np.array(
                    [pose.pose.position.x, pose.pose.position.y, pose.pose.position.z]
                )
            )
        ):
            return False

        # Check the orientation
        if np.isnan(
            np.min(
                np.array(
                    [
                        pose.pose.orientation.x,
                        pose.pose.orientation.y,
                        pose.pose.orientation.z,
                        pose.pose.orientation.w,
                    ]
                )
            )
        ):
            return False

        return True

    def publish(self) -> None:
        """Publish the current MoCap state.

        This overrides the default PoseLocalizer publish interface to send the pose
        state information to the MAVROS MoCap plugin.
        """
        self.mocap_pose_pub.publish(self.state)

    def update_pose_cb(self, pose: PoseStamped) -> None:
        """Proxy the pose to the ArduSub EKF.

        We need to do some filtering here to handle the noise from the measurements.
        The filter that we apply in this case is the LWMA filter.

        Args:
            pose: The pose of the BlueROV2 identified by the motion capture system.
        """
        # Check if any of the values in the array are NaN; if they are, then
        # discard the reading
        if not self.check_isnan(pose):
            return

        def pose_to_array(pose: Pose) -> np.ndarray:
            ar = np.zeros(6)
            ar[:3] = [pose.position.x, pose.position.y, pose.position.z]
            ar[3:] = R.from_quat(
                [
                    pose.orientation.x,
                    pose.orientation.y,
                    pose.orientation.z,
                    pose.orientation.w,
                ]
            ).as_euler("xyz")

            return ar

        # Convert the pose message into an array for filtering
        pose_ar = pose_to_array(pose.pose)

        # Add the pose to the circular buffer
        self.pose_buffer.append(pose_ar)

        # Wait until our buffer is full to start publishing the state information
        if len(self.pose_buffer) < self.pose_buffer.maxlen:  # type: ignore
            return

        def lwma(measurements: np.ndarray) -> np.ndarray:
            # Get the linear weights
            weights = np.arange(len(measurements)) + 1

            # Apply the LWMA filter and return
            return np.array(
                [
                    np.sum(np.prod(np.vstack((axis, weights)), axis=0))
                    / np.sum(weights)
                    for axis in measurements.T
                ]
            )

        filtered_pose_ar = lwma(np.array(self.pose_buffer))

        def array_to_pose(ar: np.ndarray) -> Pose:
            pose = Pose()
            pose.position.x, pose.position.y, pose.position.z = ar[:3]
            (
                pose.orientation.x,
                pose.orientation.y,
                pose.orientation.z,
                pose.orientation.w,
            ) = R.from_euler("xyz", ar[3:]).as_quat()
            return pose

        # Update the pose to be the new filtered pose
        pose.pose = array_to_pose(filtered_pose_ar)

        self.state = pose


class GazeboLocalizer(PoseLocalizer):
    """Localize the BlueROV2 using the Gazebo ground-truth data."""

    def __init__(self) -> None:
        """Create a new Gazebo localizer."""
        super().__init__("gazebo_localizer")

        # We need to know the topic to stream from
        self.declare_parameter("gazebo_odom_topic", "")

        # Subscribe to that topic so that we can proxy messages to the ArduSub EKF
        odom_topic = (
            self.get_parameter("gazebo_odom_topic").get_parameter_value().string_value
        )
        self.odom_sub = self.create_subscription(
            Odometry, odom_topic, self.update_odom_cb, qos_profile_sensor_data
        )

    def update_odom_cb(self, msg: Odometry) -> None:
        """Proxy the pose data from the Gazebo odometry ground-truth data.

        Args:
            msg: The Gazebo ground-truth odometry for the BlueROV2.
        """
        pose_cov = PoseWithCovarianceStamped()
        pose_cov.header = msg.header
        pose_cov.pose = msg.pose

        self.state = pose_cov


def main_aruco(args: list[str] | None = None):
    """Run the ArUco marker detector."""
    rclpy.init(args=args)

    node = ArucoMarkerLocalizer()
    executor = MultiThreadedExecutor()
    rclpy.spin(node, executor)

    node.destroy_node()
    rclpy.shutdown()


def main_qualisys(args: list[str] | None = None):
    """Run the Qualisys localizer."""
    rclpy.init(args=args)

    node = QualisysLocalizer()
    executor = MultiThreadedExecutor()
    rclpy.spin(node, executor)

    node.destroy_node()
    rclpy.shutdown()


def main_gazebo(args: list[str] | None = None):
    """Run the Gazebo localizer."""
    rclpy.init(args=args)

    node = GazeboLocalizer()
    executor = MultiThreadedExecutor()
    rclpy.spin(node, executor)

    node.destroy_node()
    rclpy.shutdown()