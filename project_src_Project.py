#!/usr/bin/env python
# This final piece fo skeleton code will be centred around gettign the students to follow a colour and stop upon sight of another one.

from __future__ import division
import cv2
import numpy as np
import rospy
import sys
import actionlib
import yaml
import time

from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from geometry_msgs.msg import Twist, Vector3
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from math import pi
from actionlib_msgs.msg import *
from geometry_msgs.msg import Pose, Point, Quaternion


sensitivity = 20

#Identify rooms with green circles
class greenIdentifier():
    #Subscribe to the robot's camera and call the callback method
    def __init__(self):

        self.bridge = CvBridge()

        rospy.Subscriber('camera/rgb/image_raw', Image, self.callback, queue_size=1)

    def callback(self, data):
        #Defining global variables
        #green_flag is a Boolean variable used to mark whether green is found
        #list_green_area is used to store the green circle area
        #list_image is used to store the green image
        global green_image
        global green_flag
        global list_green_area
        global list_image
        #print('start')
        #green_shape1 to define the shape of the green area
        self.green_shape1 = ''



        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, 'bgr8')
        except CvBridgeError as e:
            print(e)
        #Define the green hsv threshold range
        hsv_green_lower = np.array([60 - sensitivity, 0, 0], dtype = "uint8")
        hsv_green_upper = np.array([60 + sensitivity, 255, 255], dtype = "uint8")

        # print(hsv_green_lower)
        # print(hsv_green_upper)

        #Convert rgb image to hsv image, override values ​​outside the green threshold range with a mask
        hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
        mask_green = cv2.inRange(hsv, hsv_green_lower, hsv_green_upper)
        # output = cv2.bitwise_and(hsv, hsv, mask=mask_green)
    	# cv2.imshow("images", np.hstack([hsv, output]))
        # cv2.waitKey(1)
        # print(hsv)
        # print(mask_green)
        #Find the contours of the green area
        contours_green = cv2.findContours(mask_green, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[-2]
        #print(contours_green)
        #Assign the length of the contours to green_length
        self.green_length = len(contours_green)
        #Traverse the green area contours
        #calculates a contour perimeter or a curve length
        #draw a polygonal curve(s), approx's value is the corner number of the green area
        for cnt in range(self.green_length):
            epsilon = 0.01 * cv2.arcLength(contours_green[cnt], True)
            approx = cv2.approxPolyDP(contours_green[cnt], epsilon, True)

            corners = len(approx)

            if (corners) >= 10:
                self.green_shape1 = 'circle'

        #print('mid')
        #When the green area is circle
        if self.green_shape1 == 'circle':
            if (self.green_length) > 0:
                #print('111weqwe')
                #The robot turns around, it will recognize multiple green areas, find the max green area
                #cv2.moments () returns the calculated moments as a dictionary
                #The function cv2.minEnclosingCircle () can help us find the circumscribed circle of an object
                #It is the smallest of all circles that can contain objects.
                green_flag = True
                c = max(contours_green, key = cv2.contourArea)
                M = cv2.moments(c)
                ((x,y), radius) = cv2.minEnclosingCircle(c)
                #Calculate the area of ​​the green area
                green_area = cv2.contourArea(c)
                imask = mask_green > 0
                #Create a numpy matrix and store the images captured by the robot in this matrix
                #Assign the value of the image to green[imask]
                green = np.zeros_like(cv_image, np.uint8)
                green[imask] = cv_image[imask]
                #Add the value of the green area area and the green image to the corresponding list
                list_green_area.append(green_area)
                list_image.append(green)


class cluedoIdentifier():



    def __init__(self):


        self.bridge = CvBridge()

        rospy.Subscriber('camera/rgb/image_raw', Image, self.callback)

        #rospy.Subscriber('cameras/left_hand_camera/image', Image, image_callback)


    def callback(self, data):

        global image1

        global image0

        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, 'bgr8')
        except CvBridgeError as e:
            print(e)

        image0 = cv_image

        hsv_orange_lower = np.array([8, 160, 160])
        hsv_orange_upper = np.array([10, 175, 175])

        hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)

        mask_orange = cv2.inRange(hsv, hsv_orange_lower, hsv_orange_upper)

        contours_orange = cv2.findContours(mask_orange, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[-2]

        self.orange_length = len(contours_orange)
        #Set four different variables for four different templates
        template1 = cv2.imread('scarlet.png')
        template2 = cv2.imread('plum.png')
        template3 = cv2.imread('peacock.png')
        template4 = cv2.imread('mustard.png')
        templates = [template1, template2, template3, template4]
        #Perform template matching
        num = 1
        for template in templates:
            #Change the size of the template
            template1 = cv2.resize(template, (50,50), interpolation=cv2.INTER_CUBIC)
            template_gray = cv2.cvtColor(template1, cv2.COLOR_BGR2GRAY)
            image_gray = cv2.cvtColor(image0, cv2.COLOR_BGR2GRAY)
            #cv2.imwrite('123.png', image_gray)

            img2 = image_gray.copy()
            #cv2.imwrite('cluedo_character.png', img2)
            #Set patch's weight, height and coordinates of the upper-left corner of the patch
            w, h = template_gray.shape[::-1]
            result = cv2.matchTemplate(img2, template_gray, cv2.TM_SQDIFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
            threshold = 0.7
            #print(result)
            loc = np.where(result >= threshold)

            print(len(loc))
            for pt in zip(*loc[::-1]):
                success0 = cv2.rectangle(image, pt, (pt[0]+w, pt[1]+h), (7,249,151),2)

            if len(loc) > 0:
                if num == 1:
                    fo = open('cluedo_character.txt', 'w')
                    fo.write('scarlet')
                elif num == 2:
                    fo = open('cluedo_character.txt', 'w')
                    fo.write('plum')
                elif num == 3:
                    fo = open('cluedo_character.txt', 'w')
                    fo.write('peacock')
                elif num == 4:
                    fo = open('cluedo_character.txt', 'w')
                    fo.write('mustard')

class GoToPose():
    def __init__(self):

        self.goal_sent = False

	# What to do if shut down (e.g. Ctrl-C or failure)
	rospy.on_shutdown(self.shutdown)

	# Tell the action client that we want to spin a thread by default
	self.move_base = actionlib.SimpleActionClient("move_base", MoveBaseAction)
	rospy.loginfo("Wait for the action server to come up")

	# Allow up to 5 seconds for the action server to come up
	self.move_base.wait_for_server(rospy.Duration(5))

    def goto(self, pos, quat):

        # Send a goal
        self.goal_sent = True
	goal = MoveBaseGoal()
	goal.target_pose.header.frame_id = 'map'
	goal.target_pose.header.stamp = rospy.Time.now()
        goal.target_pose.pose = Pose(Point(pos['x'], pos['y'], 0.000),
                                     Quaternion(quat['r1'], quat['r2'], quat['r3'], quat['r4']))

	# Start moving
        self.move_base.send_goal(goal)

	# Allow TurtleBot up to 60 seconds to complete task
	success = self.move_base.wait_for_result(rospy.Duration(60))

        state = self.move_base.get_state()
        result = False

        if success and state == GoalStatus.SUCCEEDED:
            # We made it!
            result = True
            time.sleep(7.0)
        else:
            self.move_base.cancel_goal()

        self.goal_sent = False
        return result

    def shutdown(self):
        if self.goal_sent:
            self.move_base.cancel_goal()
        rospy.loginfo("Stop")
        rospy.sleep(1)

# def main(args):
#
#     rospy.init_node('image_converter', anonymous = True)
#     cI = colourIdentifier()
#
#     try:
#         rospy.spin()
#     except KeyboardInterrupt:
#         print('shutting down')
#
#     cv2.destroyAllWindows()


if __name__ == '__main__':
    #Initialize the value of the variable, like the greenIdentifier class
    image1 = None

    image =  None

    list_green_area = []
    list_image = []
    list_orange_area = []
    list_orange_image = []

    green_image = None

    green_flag = False
    image = None

    #Start a node
    #Define GoToPose class, Publisher, Rate and velocity in linear and angular
    rospy.init_node('nav_test', anonymous=True)
    navigator = GoToPose()
    pub = rospy.Publisher('mobile_base/commands/velocity', Twist, queue_size = 10)
    rate = rospy.Rate(10)
    desired_velocity = Twist()
    #Open the file contains the four points' coordinates
    with open("input_points.yaml",'r') as stream:
        points = yaml.safe_load(stream)

    x = points['starting_point'][0]
    y = points['starting_point'][1]
    theta = pi / 2
    #Initialize the values ​​of four points
    position_room1_entrance = {'x' : points['room1_entrance_xy'][0], 'y' : points['room1_entrance_xy'][1]}
    position_room1_centre = {'x' : points['room1_centre_xy'][0], 'y' : points['room1_centre_xy'][1]}
    position_room2_entrance = {'x' : points['room2_entrance_xy'][0], 'y' : points['room2_entrance_xy'][1]}
    position_room2_centre = {'x' : points['room2_centre_xy'][0], 'y' : points['room2_centre_xy'][1]}
    quaternion = {'r1' : 0.000, 'r2' : 0.000, 'r3' : np.sin(theta/2.0), 'r4' : np.cos(theta/2.0)}
    #quaternion1 = {'r1' : -1.43, 'r2' : 1.15, 'r3' : np.sin(theta/2.0), 'r4' : np.cos(theta/2.0)}
    # -2.44 0.175
    rospy.loginfo("Go to (%s, %s) pose", position_room1_entrance['x'], position_room1_entrance['y'])
    success = navigator.goto(position_room1_entrance, quaternion)
    #If success equals True
    if success:
        #Robot rotates one circle
        desired_velocity.linear.x = 0
        desired_velocity.angular.z = pi / 5

        for i in range(100):
            pub.publish(desired_velocity)
            gI = greenIdentifier()
            rate.sleep()
            #print(green_max_area, red_max_area)
            #print(red_shape)

        desired_velocity.linear.x = 0
        desired_velocity.angular.z = 0

        if green_flag == True:
            #Take the maximum value of the green area and its index
            #Save the max green area as 'green_circle.png' file
            max_area = max(list_green_area)
            max_area_index = list_green_area.index(max_area)
            green1 = list_image[max_area_index]
            cv2.imwrite('green_circle.png', green1)
            cv2.destroyWindow('Image window')
            print(position_room1_centre)
            success = navigator.goto(position_room1_centre, quaternion)
            print(success)
            #If the robot arrives the centre of the green room
            if success:
                position7 = {'x' : -2.30, 'y' : 6.50}

                success1 = navigator.goto(position7, quaternion)
                print(success1)
                #Robot rotates to face the image on the wall
                desired_velocity.linear.x = 0
                desired_velocity.angular.z = pi / 5

                for i in range(25):
                    pub.publish(desired_velocity)
                    #print('start3')
                    rate.sleep()

                desired_velocity.linear.x = 0.5
                desired_velocity.angular.z = 0

                for i in range(20):
                    pub.publish(desired_velocity)
                    rate.sleep()

                desired_velocity.linear.x = 0
                desired_velocity.angular.z = pi / 5

                for i in range(100):
                    pub.publish(desired_velocity)
                    rate.sleep()

                for i in range(40):
                    pub.publish(desired_velocity)
                    rate.sleep()

                desired_velocity.linear.x = 0
                desired_velocity.angular.z = 0
                #Save the recognized cluedo image
                cI = cluedoIdentifier()
                cv2.imwrite('cluedo_character.png', image0)

                desired_velocity.linear.x = 0
                desired_velocity.angular.z = 0

        else:
            success = navigator.goto(position_room2_entrance, quaternion)
            if success:
                desired_velocity.linear.x = 0
                desired_velocity.angular.z = pi / 10

                for i in range(200):
                    pub.publish(desired_velocity)
                    gI = greenIdentifier()
                    rate.sleep()

                if green_flag == True:
                    success = navigator.goto(position_room2_centre, quaternion)

                    if success:

                        desired_velocity.linear.x = 0
                        desired_velocity.angular.z = -pi / 5

                        for i in range(25):
                            pub.publish(desired_velocity)
                            #print('start3')
                            rate.sleep()

                        desired_velocity.linear.x = 0.5
                        desired_velocity.angular.z = 0

                        for i in range(20):
                            pub.publish(desired_velocity)
                            rate.sleep()

                        desired_velocity.linear.x = 0
                        desired_velocity.angular.z = -pi / 5

                        for i in range(100):
                            pub.publish(desired_velocity)
                            rate.sleep()

                        for i in range(40):
                            pub.publish(desired_velocity)
                            rate.sleep()

                        desired_velocity.linear.x = 0
                        desired_velocity.angular.z = 0

                        cI = cluedoIdentifier()
                        cv2.imwrite('cluedo_character.png', image0)

                        desired_velocity.linear.x = 0
                        desired_velocity.angular.z = 0
                else:
                    rospy.loginfo("there is no green entrance")

            try:
                rospy.spin()
            except KeyboardInterrupt:
                print('shutting down')

            cv2.destroyAllWindows()
