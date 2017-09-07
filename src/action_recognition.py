#!/usr/bin/env python
"""
A ROS node to recognize actions by using the lstm method described in https://blog.coast.ai/five-video-classification-methods-implemented-in-keras-and-tensorflow-99cad29cc0b5.

Also, this node is to run the classification continuously. The main code is acquired from https://github.com/harvitronix/five-video-classification-methods.

Author:
    Cagatay Odabasi -- cagatay.odabasi@ipa.fraunhofer.de
"""

# ROS
import rospy

import rospkg

from sensor_msgs.msg import Image

from std_msgs.msg import String

from cv_bridge import CvBridge, CvBridgeError

from action_recognition.recognizer import Recognizer

import cv2

class ActionRecognitionNode(object):
    """docstring for ActionRecognitionNode."""
    def __init__(self):
        super(ActionRecognitionNode, self).__init__()

        # init the node
        rospy.init_node('action_recognition_node', anonymous=False)

        # Get the parameters
        (camera_namespace, model_name, no_of_frames) = self.get_parameters()

        rospack = rospkg.RosPack()

        print rospack.get_path('action_recognition')

        # Create Detector
        self._recognizer = Recognizer(\
            rospack.get_path('action_recognition') + '/' + 'src' + \
            model_name, no_of_frames)

        self._bridge = CvBridge()

        # Advertise the result of Object Detector
        self.pub_detections = rospy.Publisher('/action_classification/action_class', \
            String, queue_size=1)

        # Subscribe to the face positions
        self.sub_rgb = rospy.Subscriber(camera_namespace,\
            Image, self.rgb_callback, queue_size=1, buff_size=2**24)

        # spin
        rospy.spin()

    def get_parameters(self):
        """
        Gets the necessary parameters from parameter server

        Args:

        Returns:
        (tuple) (camera_namespace, model_name)

        """

        camera_namespace = rospy.get_param("~camera_namespace")

        model_name = rospy.get_param("~model_name")

        (no_of_frames) = rospy.get_param("~no_of_frames")

        return (camera_namespace, model_name, no_of_frames)


    def shutdown(self):
        """
        Shuts down the node
        """
        rospy.signal_shutdown("See ya!")

    def rgb_callback(self, data):
        """
        Callback for RGB images
        """
        try:
            # Conver image to numpy array
            cv_image = self._bridge.imgmsg_to_cv2(data, "bgr8")

            cv2.imshow("cagaty",cv_image)

            cv2.waitKey(1)



            self._recognizer.run(cv_image)

            # Publish the messages
            #self.pub_detections.publish(msg)

        except CvBridgeError as e:
            print(e)

def main():
    """ main function
    """
    node = ActionRecognitionNode()

if __name__ == '__main__':
    main()
