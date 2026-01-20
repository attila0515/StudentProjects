import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge
from geometry_msgs.msg import Twist
from geometry_msgs.msg import PoseStamped

from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import tensorflow as tf
import cv2
import numpy as np
import threading
import time
import os
from pathlib import Path

# Disable GPU to avoid CUDA errors in WSL2
os.environ["CUDA_VISIBLE_DEVICES"] = ""

class ImageSubscriber(Node):
    def __init__(self):
        super().__init__('image_subscriber')

        # Set image size
        self.image_size = 24

        model_path = Path.home() / "ros2_ws" / "src" / "Week-1-8-Cognitive-robotics" / "turtlebot3_mogi_py" / "network_model" / "model.best.keras"

        print("TensorFlow version: %s" % tf.__version__)
        print("Keras version: %s" % tf.keras.__version__)
        print("CNN model: %s" % model_path)

        # Load model
        self.model = load_model(model_path, compile=True)
        self.model.summary()

        self.last_time = time.time()
        self.recent_predictions = []  # Store recent predictions for smoothing
        self.max_predictions = 10  # Restored to original for stability
        self.last_angular_z = 0.0  # Store last angular velocity to maintain direction
        self.last_linear_x = 0.12  # Restored speed
        self.consecutive_nothing_count = 0  # Track consecutive "Nothing" predictions
        self.last_non_nothing_prediction = 0  # Store last non-"Nothing" prediction (default to Forward)

        # Variables for continuous cmd_vel publishing
        self.cmd_vel_msg = Twist()
        self.cmd_vel_msg.linear.x = 0.12  # Default forward speed
        self.cmd_vel_msg.angular.z = 0.0  # Default straight direction
        self.cmd_vel_lock = threading.Lock()  # Lock for thread safety

        self.subscription = self.create_subscription(
            CompressedImage,
            'image_raw/compressed',
            self.image_callback,
            1
        )

        # Subscribe to /turtlebot3/pose to log the robot's position in Gazebo
        self.pose_subscription = self.create_subscription(
            PoseStamped,
            '/turtlebot3/pose',
            self.pose_callback,
            10
        )

        self.publisher = self.create_publisher(Twist, 'cmd_vel', 10)
        
        self.bridge = CvBridge()
        
        self.latest_frame = None
        self.frame_lock = threading.Lock()
        
        self.running = True

        # Start threads for spinning and continuous cmd_vel publishing
        self.spin_thread = threading.Thread(target=self.spin_thread_func)
        self.spin_thread.start()
        self.cmd_vel_thread = threading.Thread(target=self.cmd_vel_publish_thread)
        self.cmd_vel_thread.start()

    def spin_thread_func(self):
        """Separate thread function for rclpy spinning."""
        while rclpy.ok() and self.running:
            rclpy.spin_once(self, timeout_sec=0.05)

    def cmd_vel_publish_thread(self):
        """Separate thread to continuously publish cmd_vel messages."""
        rate = self.create_rate(50)  # 50 Hz publishing rate
        while rclpy.ok() and self.running:
            with self.cmd_vel_lock:
                msg = Twist()
                msg.linear.x = self.cmd_vel_msg.linear.x
                msg.linear.y = self.cmd_vel_msg.linear.y
                msg.linear.z = self.cmd_vel_msg.linear.z
                msg.angular.x = self.cmd_vel_msg.angular.x
                msg.angular.y = self.cmd_vel_msg.angular.y
                msg.angular.z = self.cmd_vel_msg.angular.z
                self.publisher.publish(msg)
                print(f"Cmd_vel thread publishing: linear.x={msg.linear.x}, angular.z={msg.angular.z}")
            rate.sleep()

    def pose_callback(self, msg):
        """Callback to log the robot's position in Gazebo."""
        position = msg.pose.position
        print(f"Robot position in Gazebo: x={position.x}, y={position.y}, z={position.z}")

    def image_callback(self, msg):
        """Callback function to receive and store the latest frame."""
        with self.frame_lock:
            self.latest_frame = self.bridge.compressed_imgmsg_to_cv2(msg, desired_encoding="bgr8")

    def display_image(self):
        cv2.namedWindow("CNN input", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("CNN input", 200, 200)

        while rclpy.ok():
            if self.latest_frame is not None:
                cnn_input_display = self.process_image(self.latest_frame)
                cv2.imshow("CNN input", cnn_input_display)
                cv2.waitKey(1)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.stop_robot()
                self.running = False
                break

        cv2.destroyAllWindows()
        self.running = False

    def process_image(self, img):
        # Convert to grayscale for thresholding
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        print("Image shape:", gray.shape)  # Log image shape to verify resolution
        
        # Apply slightly wider trapezoid mask
        masked_image, mask = self.apply_polygon_mask(gray)
        
        # Log intensity statistics of the masked image
        print(f"Masked image min: {masked_image.min()}, max: {masked_image.max()}, mean: {masked_image.mean()}")
        
        # Restored original adaptive thresholding
        binary = cv2.adaptiveThreshold(masked_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 51, 15)
        
        # Fallback to fixed threshold with slightly lower value
        binary_fixed = self.threshold_binary(masked_image, thresh=(40, 255))
        binary = cv2.bitwise_or(binary, binary_fixed)  # Combine both for robustness
        
        # Invert the image (white lines become black)
        binary_inverted = cv2.bitwise_not(binary)
        
        # Apply morphological operation to connect dashed lines
        kernel = np.ones((10, 10), np.uint8)
        binary_inverted = cv2.dilate(binary_inverted, kernel, iterations=3)
        
        # Fallback: If binary_inverted is completely white, use the original binary image
        if np.sum(binary_inverted) > (binary_inverted.size * 255 * 0.95):  # If mostly white
            print("Binary inverted is mostly white, reverting to binary image")
            binary_inverted = cv2.bitwise_not(binary)  # Revert to binary without dilation
        
        # Resize and prepare for CNN
        image = cv2.resize(binary_inverted, (self.image_size, self.image_size))
        image = img_to_array(image)
        image = np.array(image, dtype="float") / 255.0
        image = np.repeat(image, 3, axis=2)  # Convert to 3 channels (RGB)
        image = image.reshape(-1, self.image_size, self.image_size, 3)
        
        # Debug: Display the CNN input image (scale back to [0, 255] for visualization)
        cnn_input_display = (image[0] * 255).astype(np.uint8)  # Scale and convert to uint8
        cnn_input_display = cv2.resize(cnn_input_display, (200, 200))  # Resize for visibility
        
        # Run prediction on CPU
        prediction_probs = self.model(image, training=False)[0]  # Get probabilities for logging
        prediction = np.argmax(prediction_probs)
        
        # Log prediction probabilities for debugging
        print(f"Prediction probabilities: Forward={prediction_probs[0]:.3f}, Left={prediction_probs[1]:.3f}, Right={prediction_probs[2]:.3f}, Nothing={prediction_probs[3]:.3f}")

        # Track consecutive "Nothing" predictions
        if prediction == 3:
            self.consecutive_nothing_count += 1
        else:
            self.consecutive_nothing_count = 0
            self.last_non_nothing_prediction = prediction  # Update last non-"Nothing" prediction
        if self.consecutive_nothing_count < 5:
            effective_prediction = self.last_non_nothing_prediction
        else:
            effective_prediction = smoothed_prediction if smoothed_prediction != 3 else self.last_non_nothing_prediction

        print(f"Consecutive Nothing Predictions: {self.consecutive_nothing_count}, Last Non-Nothing Prediction: {self.last_non_nothing_prediction}")

        # Smooth predictions by taking the most common recent prediction
        self.recent_predictions.append(prediction)
        if len(self.recent_predictions) > self.max_predictions:
            self.recent_predictions.pop(0)
        
        # Use the most frequent prediction to avoid stopping on gaps
        smoothed_prediction = max(set(self.recent_predictions), key=self.recent_predictions.count)
        print(f"Raw Prediction: {prediction}, Smoothed Prediction: {smoothed_prediction}, Recent Predictions: {self.recent_predictions}")

        # Movement logic: Always move forward, adjust direction based on prediction
        effective_prediction = smoothed_prediction if smoothed_prediction != 3 else self.last_non_nothing_prediction

        with self.cmd_vel_lock:
            if effective_prediction == 0:  # Forward
                self.cmd_vel_msg.angular.z = 0.0
                self.cmd_vel_msg.linear.x = 0.12
                self.last_angular_z = self.cmd_vel_msg.angular.z
            elif effective_prediction == 1:  # Left
                self.cmd_vel_msg.angular.z = -0.2  # Élesebb kanyar
                self.cmd_vel_msg.linear.x = 0.12
                self.last_angular_z = self.cmd_vel_msg.angular.z
            elif effective_prediction == 2:  # Right
                self.cmd_vel_msg.angular.z = 0.2   # Élesebb kanyar
                self.cmd_vel_msg.linear.x = 0.12
                self.last_angular_z = self.cmd_vel_msg.angular.z
            else:  # Fallback to last known direction
                self.cmd_vel_msg.angular.z = self.last_angular_z * 0.9  # Tompított ismétlés
                self.cmd_vel_msg.linear.x = 0.07  # Óvatosabb előrehaladás

        print(f"Updated cmd_vel: linear.x={self.cmd_vel_msg.linear.x}, angular.z={self.cmd_vel_msg.angular.z}")

        print("Elapsed time %.3f" % (time.time()-self.last_time))
        self.last_time = time.time()

        return cnn_input_display

    def threshold_binary(self, img, thresh=(40, 255)):
        binary = np.zeros_like(img)
        binary[(img >= thresh[0]) & (img <= thresh[1])] = 1
        return binary * 255

    def apply_polygon_mask(self, img):
        mask = np.zeros_like(img)
        ignore_mask_color = 255
        imshape = img.shape
        # Slightly wider trapezoid mask to improve turn detection
        vertices = np.array([[(imshape[1]*0.05, imshape[0]), (imshape[1]*0.3, imshape[0]*0.4), 
                              (imshape[1]*0.7, imshape[0]*0.4), (imshape[1]*0.95, imshape[0])]], dtype=np.int32)
        cv2.fillPoly(mask, vertices, ignore_mask_color)
        masked_image = cv2.bitwise_and(img, mask)
        return masked_image, mask

    def stop_robot(self):
        with self.cmd_vel_lock:
            self.cmd_vel_msg.linear.x = 0.0
            self.cmd_vel_msg.linear.y = 0.0
            self.cmd_vel_msg.linear.z = 0.0
            self.cmd_vel_msg.angular.x = 0.0
            self.cmd_vel_msg.angular.y = 0.0
            self.cmd_vel_msg.angular.z = 0.0
        self.publisher.publish(self.cmd_vel_msg)

    def stop(self):
        self.running = False
        self.spin_thread.join()
        self.cmd_vel_thread.join()

def main(args=None):
    print("OpenCV version: %s" % cv2.__version__)
    rclpy.init(args=args)
    node = ImageSubscriber()
    try:
        node.display_image()
    except KeyboardInterrupt:
        pass
    finally:
        node.stop()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
