#!/usr/bin/env python3
# Bourli Eftychia
########################### Libraries ###############################
import sys
import math
import rospy
import moveit_commander
import geometry_msgs.msg
import matplotlib.pyplot as plt # type: ignore
from cmath import pi
from nav_msgs.msg import Odometry 
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from gazebo_msgs.msg import LinkStates
from tf.transformations import euler_from_quaternion
from tf.transformations import quaternion_from_euler
from moveit_commander.conversions import pose_to_list

###########################  Globals  ###############################
MAX_LINEAR_SPEED = 0.2        # Max Linear speed
MAX_ANGULAR_SPEED = 0.5       # Max Angular speed
GOAL_THRESHOLD = 0.9          # Distance threshold to start slow down
DOOR_THRESHOLD = 0.85         # Distance threshold of the gasp of the door 
current_position = [0, 0, 0]  # [x, y, theta]
final_goal = [0,0]            # [x, y]
goal = [0,0]                  # [x, y]
point_after_door = [0, 0]     # [x, y]
point_before_door = [0, 0]    # [x, y]
obstacles = []                # Array with obstacles
plot_angular_speed = []       # For the plots 
plot_real_angular_speed = []  # For the plots 
plot_linear_speed = []        # For the plots 
plot_real_linear_speed = []   # For the plots 
door = []                     # The two sides from the door opening [right_side_of_the_door, left_side_of_the_door]
finished = False              # Flag indicates that a goal has been achieved
door_case = None              # Case of door (indoor/outdoor,opens to the left/right)
flag_stop = 0                 # Flag that works kinda like a break point
flag_door_found = 0           # This flag prevent find_door(msg) to calculate again and again the door
#####################################################################

# This function puts the robot to sleep for 1 sec
def wakeup(): 
    waketime = 1
    move = Twist()
    start = rospy.get_time()
    uptime = rospy.get_time()

    while  (uptime-start < waketime):
        rate = rospy.Rate(10)
        rate.sleep()
        move.linear.x = 0
        move.angular.z = 0
        pub.publish(move)
        uptime = rospy.get_time()

# This function take the angular and linear velocities of jackal from gazebo
def get_u(msg):
    global realu, realw
    realw = msg.twist.twist.angular.z
    realu = msg.twist.twist.linear.x

# This function take the pose and yaw of jackal from gazebo
def get_jackal_orientation(msg):
    global current_position

    # Find the index of 'base_link' in the 'name' field of the message
    try:
        link_index = msg.name.index('jackal_kinova::base_link')
    except ValueError:
        rospy.logwarn("base_link not found in link states")
        return

    # Extract position and orientation of 'base_link'
    base_pose = msg.pose[link_index]
    x = base_pose.position.x
    y = base_pose.position.y
    quaternion = (
        base_pose.orientation.x,
        base_pose.orientation.y,
        base_pose.orientation.z,
        base_pose.orientation.w
    )
    _, _, yaw = euler_from_quaternion(quaternion)

    # Update current position
    current_position[0] = round(x,3)
    current_position[1] = round(y,3)
    current_position[2] = round(yaw,3)

# This function finds the 2 most close to the robot obstacles in 1 meter distance
def get_scan(msg):
    global obstacles, current_position

    obstacles = []  # Clear previous obstacles
    curr_pos = [0,0]
    curr_pos[0] = current_position[0]
    curr_pos[1] = current_position[1]
    angle_min = msg.angle_min
    angle_increment = msg.angle_increment
    scan = msg.ranges
    x = 0.0
    y = 0.0
    obstacle1 = [0,0]
    obstacle2 = [0,0]
    minDistance = 100
    second_minDistance = 100
    current_obstacle = [0,0]
    flag = 0
    for i in range(len(scan)):
        if scan[i] < 1:
            flag = 1
            # Convert polar coordinates to Cartesian coordinates
            angle = angle_min + i * angle_increment
            x = scan[i] * math.cos(angle)
            y = scan[i] * math.sin(angle)  
           
            current_obstacle = [x,y]
            current_distance = math.sqrt((current_obstacle[0])**2 + (current_obstacle[1])**2)

            if current_distance < minDistance:
                obstacle1 = current_obstacle
                minDistance = current_distance
            elif current_distance < second_minDistance:
                obstacle2 = current_obstacle
                second_minDistance = current_distance

    if flag == 1 :
        obstacles.append(obstacle1)
        obstacles.append(obstacle2)

#####################################################################################################################
####################################  Dynamic Potential Field Algorythm   ###########################################

def attractive_potential(cur_pos, goal_pos):
    force = 1000
    attractive_vector = [force * (goal_pos[0] - cur_pos[0]),force * (goal_pos[1] - cur_pos[1])]
    return attractive_vector

def repulsive_potential(cur_pos, obstacle):
    force = -300
    repulsive_vector = [force / (force*(obstacle[0] - cur_pos[0])),force / (force*(obstacle[1] - cur_pos[1]))]
    return repulsive_vector

def gradient_descent():
    global goal, obstacles, current_position
    
    final_vector = attractive_potential(current_position, goal)

    if len(obstacles) > 0:
        for obstacle in obstacles:
            repulsive_vector = repulsive_potential(current_position, obstacle)
            final_vector[0] += repulsive_vector[0]
            final_vector[1] += repulsive_vector[1]

    return final_vector

def delta_theta():
    global current_position
    
    # Compute the final vector representing the direction the robot should move towards
    final_vector = gradient_descent()
    final_vector_orientation = math.atan2(final_vector[1], final_vector[0])
    current_orientation = current_position[2]
    angle_difference = final_vector_orientation - current_orientation

    # Normalize the angle to be between -pi and pi
    while angle_difference > math.pi:
        angle_difference -= 2 * math.pi
    while angle_difference < -math.pi:
        angle_difference += 2 * math.pi

    return angle_difference

def command_move():
    global goal, current_position

    linear_speed = 0
    dtheta = round(delta_theta(),4)

    # Adjust linear and angular speed based on gradient descent 
    # Angular
    k = 1.2
    move.angular.z = max(min(dtheta*k, MAX_ANGULAR_SPEED), -MAX_ANGULAR_SPEED)  
    angular_speed = max(min(dtheta*k, MAX_ANGULAR_SPEED), -MAX_ANGULAR_SPEED)
    plot_angular_speed.append(angular_speed)
    
    # Linear
    distance_to_goal = math.sqrt((goal[0] - current_position[0])**2 + (goal[1] - current_position[1])**2)
    if distance_to_goal < GOAL_THRESHOLD:
        move.linear.x = min(MAX_LINEAR_SPEED * distance_to_goal / GOAL_THRESHOLD, MAX_LINEAR_SPEED)
        linear_speed = min(MAX_LINEAR_SPEED * distance_to_goal / GOAL_THRESHOLD, MAX_LINEAR_SPEED)
    else:
        move.linear.x = MAX_LINEAR_SPEED
        linear_speed = MAX_LINEAR_SPEED

    # For the plots
    plot_linear_speed.append(linear_speed)
    plot_real_angular_speed.append(realw)
    plot_real_linear_speed.append(realu)
    
    pub.publish(move)

def potential_navigation():
    global current_position, goal, finished, obstacles

    # The Time the robot must remain stationary to be considered to have reached the target
    idle_time_threshold = 5  
    idle_time = 0
    finished = False
            
    distance_from_goal = math.sqrt((goal[0] - current_position[0])**2 + (goal[1] - current_position[1])**2)

    while not finished and not rospy.is_shutdown() and distance_from_goal > 0.2:
        scan_msg = rospy.wait_for_message("/scan", LaserScan)
        get_scan(scan_msg)
        command_move()
        rate.sleep()

        # If the robot is not moving, we increase the immobility time. Otherwise, we reset it to zero.
        if move.linear.x < 0.04 and move.angular.z < 0.2:
            idle_time += 1
        else:
            idle_time = 0

        # If the idle time exceeds the idle_time_threshold, we consider the robot to have reached the target
        if idle_time >= idle_time_threshold:
            finished = True

        if len(obstacles)>0:
            obstacle = obstacles[0]
            distance_from_obstacle = math.sqrt((obstacle[0])**2 + (obstacle[1])**2)
            distance_from_goal = math.sqrt((goal[0] - current_position[0])**2 + (goal[1] - current_position[1])**2)

            if distance_from_obstacle < 0.35 and distance_from_goal < 0.3:
                finished = True

        distance_from_goal = math.sqrt((goal[0] - current_position[0])**2 + (goal[1] - current_position[1])**2)

#####################################################################################################################
#############################################  Navigation Through Door  #############################################

# This function finds the door in the room. NOTE: The robot must be able to see through the door opening to locate it with the laser scan
def find_door(msg):
    global door, goal, current_position, door_case, flag_stop, flag_door_found

    scan = msg.ranges
    angle_min = msg.angle_min
    angle_increment = msg.angle_increment
    
    
    if flag_door_found == 0:
        door = []
        min_left = []
        min_right = []
        min_left_distance = float('inf')
        min_right_distance = float('inf')
        left_dots = []
        right_dots = []
        auxiliary_point_left = []
        auxiliary_point_right = []

        x = 0
        y = 0
        check = 0
        obstacle = 0
        flag_inf = 0 
        leftSide = 0
        rightSide = 0
        flagLeftWall = 0
        previous_right_obstacle = 0
        previous_left_obstacle = 0
        distance_to_obstacle = 0
        distance_to_previous_obstacle = 0
        flag_stop = 0


        for i in range(len(scan)):
            # Convert polar coordinates to Cartesian coordinates
            angle = angle_min + i * angle_increment
            x = scan[i] * math.cos(angle)
            y = scan[i] * math.sin(angle) 

            # find the right side of the door's gap
            if 1 < scan[i] and flagLeftWall == 0:
                if (math.isinf(x) or math.isinf(y)) and previous_right_obstacle != 0:
                    flag_inf = 1
                    x = round(previous_right_obstacle[0],3)
                    y = round(previous_right_obstacle[1],3)
                    rightSide = [x,y]
                    door.append(rightSide)
                    flagLeftWall = 1 

                elif not(math.isinf(x) or math.isinf(y)):
                    obstacle = [x,y]
                    distance_to_obstacle = math.sqrt((obstacle[0] - current_position[0])**2 + (obstacle[1] - current_position[1])**2) 

                    if previous_right_obstacle == 0:
                        previous_right_obstacle = [x,y] 
                        distance_to_previous_obstacle = math.sqrt((previous_right_obstacle[0] - current_position[0])**2 + (previous_right_obstacle[1] - current_position[1])**2) 
                    elif (abs(distance_to_obstacle - distance_to_previous_obstacle)) > 1.5:
                        x = round(previous_right_obstacle[0],3)
                        y = round(previous_right_obstacle[1],3)
                        rightSide = [x,y]
                        door.append(rightSide)
                        flagLeftWall = 1    

                    previous_right_obstacle = [x,y] 
                    distance_to_previous_obstacle = math.sqrt((previous_right_obstacle[0] - current_position[0])**2 + (previous_right_obstacle[1] - current_position[1])**2)

                    if (abs(distance_to_obstacle - distance_to_previous_obstacle)) < 1.5:
                        right_dots.append(obstacle)
                
            # find the left side of the door's gap
            elif 1 < scan[i] and flagLeftWall == 1 :
                if flag_stop == 0:
                    if not(math.isinf(x) or math.isinf(y)) and flag_inf == 1:
                        leftSide = [round(x,3),round(y,3)]
                        door.append(leftSide)
                        flag_stop = 1
                       
                    elif not(math.isinf(x) or math.isinf(y)) and flag_inf == 0:
                        obstacle = [x,y]
                        distance_to_obstacle = math.sqrt((obstacle[0] - current_position[0])**2 + (obstacle[1] - current_position[1])**2)
                        if previous_left_obstacle == 0:
                            previous_left_obstacle = [x,y] 
                            distance_to_previous_obstacle = math.sqrt((previous_left_obstacle[0] - current_position[0])**2 + (previous_left_obstacle[1] - current_position[1])**2)
                        elif (abs(distance_to_obstacle - distance_to_previous_obstacle)) > 1:
                            leftSide = [round(x,3),round(y,3)]
                            door.append(leftSide)
                            flag_stop = 1
                
                obstacle = [x,y]
                left_dots.append(obstacle)
                previous_left_obstacle = [x,y] 
                distance_to_previous_obstacle = math.sqrt((previous_left_obstacle[0] - current_position[0])**2 + (previous_left_obstacle[1] - current_position[1])**2) 

        if len(door) == 2 :
            for dot in right_dots:
                robot_dot = distance(current_position,dot)
                right_side_door_dot = distance(door[0],dot)
                if right_side_door_dot < 2:
                    if robot_dot < min_right_distance:
                        min_right = dot
                        min_right_distance = robot_dot
                if 0.9 < right_side_door_dot < 1.3:
                    wall_right = dot
                if 0.2 < right_side_door_dot < 0.3:
                    auxiliary_point_right = dot

            for dot in left_dots:
                robot_dot = abs(distance(current_position,dot))
                left_side_door_dot = abs(distance(door[1],dot))
                if left_side_door_dot < 2:
                    if robot_dot < min_left_distance:
                        min_left = dot
                        min_left_distance = robot_dot
                if 0.9 < left_side_door_dot < 1.3:
                    wall_left = dot
                if 0.2 < left_side_door_dot < 0.3:
                    auxiliary_point_left = dot    
            #print("door = ", door)
            find_door_case(min_left, min_right, wall_left, wall_right, auxiliary_point_left, auxiliary_point_right)

            if door_case != None:
                flag_door_found = 1

# This function recognize if the door is Indoor or Outdoor and if the door opens to the left or to the right
def find_door_case(min_left, min_right, wall_left, wall_right, auxiliary_point_left, auxiliary_point_right):
    global current_position, door_case, door, point_after_door, point_before_door

    door_left = door[1]
    door_right = door[0]
    wall_left = [round(wall_left[0],3),round(wall_left[1],3)]
    wall_right = [round(wall_right[0],3),round(wall_right[1],3)]

    # Calculate the line e1 passing through wall_left and wall_right
    a1, b1 = calculate_line(wall_left, wall_right)  
    a1 = round(a1,3)
    b1 = round(b1,3)
    
    tolerance = 0.18
    
    if a1 == float('inf'):  # Special case where the line is vertical
        # e1 is the vertical line representing the wall : x = b1
        # I compare the x of the points to see in witch side of the line they are.
        # If both robot and point are lower or higher they are on the same half-plane so we have an Indoor 
        # If they are on saparate half-planes that means we have an Outdoor
        # And by knowing witch side of the door isn't on the same line with the wall I know if the door opens to the left or to the right
        if current_position[0] < b1:
            if abs(door_left[0]-b1) < tolerance and abs(door_right[0]-b1) < tolerance:

                if abs(min_left[0]-b1) < tolerance and abs(min_right[0]-b1) < tolerance:
                    door_case = 5   # The door is open
                elif abs(min_left[0]-b1) < tolerance and min_right[0] < b1:
                    door_case = 4   # Indoor to the right
                    if distance(door[1],min_right) < DOOR_THRESHOLD:
                        door = [min_right, door[1]]   
                elif min_left[0] < b1 and abs(min_right[0]-b1) < tolerance:
                    door_case = 3   # Indoor to the left
                    if distance(door[0],min_left) < DOOR_THRESHOLD:
                        door = [min_left, door[0]]
                elif abs(min_left[0]-b1) < tolerance and min_right[0] > b1:
                    door_case = 2   # Outdoor to the right
                elif min_left[0] > b1 and abs(min_right[0]-b1) < tolerance:
                    door_case = 1   # Outdoor to the left

            elif abs(door_left[0]-b1) < tolerance and door_right[0] < b1:
                door_case = 4   # Indoor to the right
            elif door_left[0] < b1 and abs(door_right[0]-b1) < tolerance:
                door_case = 3   # Indoor to the left
            elif abs(door_left[0]-b1) < tolerance and door_right[0] > b1:
                door_case = 2   # Outdoor to the right
            elif door_left[0] > b1 and abs(door_right[0]-b1) < tolerance:
                door_case = 1   # Outdoor to the left   

        elif current_position[0] > b1:
            if abs(door_left[0]-b1) < tolerance and abs(door_right[0]-b1) < tolerance:

                if abs(min_left[0]-b1) < tolerance and abs(min_right[0]-b1) < tolerance:
                    door_case = 5   # The door is open
                elif abs(min_left[0]-b1) < tolerance and min_right[0] < b1:
                    door_case = 2   # Outdoor to the right
                    if distance(door[1],min_right) < DOOR_THRESHOLD:
                        door = [min_right, door[1]]   
                elif min_left[0] < b1 and abs(min_right[0]-b1) < tolerance:
                    door_case = 1   # Outdoor to the left
                    if distance(door[0],min_left) < DOOR_THRESHOLD:
                        door = [min_left, door[0]]
                elif abs(min_left[0]-b1) < tolerance and min_right[0] > b1:
                    door_case = 4   # Indoor to the right
                elif min_left[0] > b1 and abs(min_right[0]-b1) < tolerance:
                    door_case = 3   # Indoor to the left

            elif abs(door_left[0]-b1) < tolerance and door_right[0] < b1:
                door_case = 2   # Outdoor to the right
            elif door_left[0] < b1 and abs(door_right[0]-b1) < tolerance:
                door_case = 1   # Outdoor to the left
            elif abs(door_left[0]-b1) < tolerance and door_right[0] > b1:
                door_case = 4   # Indoor to the right
            elif door_left[0] > b1 and abs(door_right[0]-b1) < tolerance:
                door_case = 3   # Indoor to the left   
    else:
        # e1 is the line representing the wall : y = a1 * x + b1
        # Given a point[xp, yp]. 
        # If we substitute the coordinates of this point into the equation of the line e1: y = a1x + b1 
        # and (yp - a1 *xp - b1 = 0), then the point lies on the line. 
        # If the result is a positive number, the point lies in the positive half-plane divided by the line, 
        # whereas if it returns a negative number, the point lies in the negative half-plane.
        # If both robot and point are lower or higher they are on the same half-plane so we have an Indoor 
        # If they are on saparate half-planes that means we have an Outdoor
        # And by knowing witch side of the door isn't on the same line with the wall I know if the door opens to the left or to the right

        result_robot = current_position[1] - a1 * current_position[0] - b1
        result_door_left = door_left[1] - a1 * door_left[0] - b1
        result_door_rigth = door_right[1] - a1 * door_right[0] - b1

        if result_robot < 0:
            if -tolerance < result_door_left < tolerance and -tolerance < result_door_rigth < tolerance:
                result_min_left = min_left[1] - a1 * min_left[0] - b1
                result_min_right = min_right[1] - a1 * min_right[0] - b1

                if -tolerance < result_min_left < tolerance and -tolerance < result_min_right < tolerance:
                    door_case = 5   # The door is completely open
                elif -tolerance < result_min_left < tolerance and result_min_right < 0 :
                    door_case = 4   # Indoor to the right
                    if distance(door[1],min_right) < DOOR_THRESHOLD:
                        door = [min_right, door[1]] 
                elif result_min_left < 0 and -tolerance < result_min_right < tolerance:
                    door_case = 3   # Indoor to the left
                    if distance(door[0],min_left) < DOOR_THRESHOLD:
                        door = [min_left, door[0]]
                elif -tolerance < result_min_left < tolerance and result_min_right > 0 :
                    door_case = 2   # Outdoor to the right
                elif result_min_left > 0 and -tolerance < result_min_right < tolerance:
                    door_case = 1   # Outdoor to the left

            elif -tolerance < result_door_left < tolerance and result_door_rigth < 0 : 
                door_case = 4   # Indoor to the right
            elif result_door_left < 0 and -tolerance < result_door_rigth < tolerance:
                door_case = 3   # Indoor to the left
            elif -tolerance < result_door_left < tolerance and result_door_rigth > 0:
                door_case = 2   # Outdoor to the right
            elif result_door_left > 0 and -tolerance < result_door_rigth < tolerance:
                door_case = 1   # Outdoor to the left 

        elif result_robot > 0:
            if -tolerance < result_door_left < tolerance and -tolerance < result_door_rigth < tolerance:
                result_min_left = min_left[1] - a1 * min_left[0] - b1
                result_min_right = min_right[1] - a1 * min_right[0] - b1

                if -tolerance < result_min_left < tolerance and -tolerance < result_min_right < tolerance:
                    door_case = 5   # The door is completely open
                elif -tolerance < result_min_left < tolerance and result_min_right < 0 :
                    door_case = 2   # Outdoor to the right
                    if distance(door[1],min_right) < DOOR_THRESHOLD:
                        door = [min_right, door[1]] 
                elif result_min_left < 0 and -tolerance < result_min_right < tolerance:
                    door_case = 1   # Outdoor to the left   
                    if distance(door[0],min_left) < DOOR_THRESHOLD:
                        door = [min_left, door[0]]
                elif -tolerance < result_min_left < tolerance and result_min_right > 0 :
                    door_case = 4   # Indoor to the right
                elif result_min_left > 0 and -tolerance< result_min_right < tolerance:
                    door_case = 3   # Indoor to the left   

            elif -tolerance < result_door_left < tolerance and result_door_rigth < 0 : 
                door_case = 2   # Outdoor to the right
            elif result_door_left < 0 and -tolerance < result_door_rigth < tolerance:
                door_case = 1   # Outdoor to the left
            elif -tolerance < result_door_left < tolerance and result_door_rigth > 0:
                door_case = 3   # Indoor  to the right
            elif result_door_left > 0 and -tolerance < result_door_rigth < tolerance:
                door_case = 4   # Indoor  to the left 

    # Calculate the line e2 passing through the door
    if door_case == 1 or door_case == 3:
        a2, b2 = calculate_line(door[1], auxiliary_point_left)  
        a2 = round(a2,3)
        b2 = round(b2,3)
        frame1 = door[0]
        frame2 = find_intersection(a1, b1, a2, b2)

        point = find_mid_vertical_segment(frame1, frame2, 0.20)
        point_before_door = point[0]

        mid_vertical_segment = find_mid_vertical_segment(frame1, frame2, 1)
        point_after_door = mid_vertical_segment[1]
        
    elif door_case == 2 or door_case == 4:
        a2, b2 = calculate_line(door[0], auxiliary_point_right)  
        a2 = round(a2,3)
        b2 = round(b2,3)
        frame1 = door[1]
        frame2 = find_intersection(a1, b1, a2, b2)

        point = find_mid_vertical_segment(frame1, frame2, 0.20)
        point_before_door = point[1]
        
        mid_vertical_segment = find_mid_vertical_segment(frame1, frame2, 1)
        point_after_door = mid_vertical_segment[0]
    
def check_for_partial_open_door():
    global door, door_case, goal, point_before_door,point_after_door
    
    if len(door) == 2:
        # Calculate the distance between the two sides of the door 
        distance_between_sides = math.dist(door[0], door[1])
        
        if distance_between_sides < DOOR_THRESHOLD:
            print("There is a half-open door at : ",door)
            print("I will move closer to open the half-open door")

            # Move closer to the half-open door
            before_door = find_mid_vertical_segment(door[0], door[1], 0.35)
            goal = before_door[0]
            turn_robot(goal)
            print("point before door = ", goal)
            potential_navigation()
            if door_case == 3:
                turn_robot(door[1])
            elif door_case == 4:
                turn_robot(door[0])

            # Open the door
            print("I will open that door")
            open_the_door()

            # Go throuth the door
            print("I will go through that door")
            print("point after door = ", point_after_door)
            if door_case == 3:
                goal = point_before_door
                potential_navigation()
            elif door_case == 4:
                goal = point_before_door
                potential_navigation()
            turn_robot(point_after_door)
            goal = point_after_door
            potential_navigation()
        else:
            print("There is an open door at ",door)
            print("I will go through that door")

            # Go throuth the door
            if door_case == 5:
                before_door = find_mid_vertical_segment(door[0], door[1], 0.4)
                goal = before_door[0]
                potential_navigation()
                after_door = find_mid_vertical_segment(door[0], door[1], 1.5)
                goal = after_door[1]
                turn_robot(point_after_door)
                potential_navigation()
            else:
                goal = point_before_door
                potential_navigation()
                turn_robot(point_after_door)
                goal = point_after_door
                potential_navigation()
    else:
        print("I can't find door.")

def turn_robot(goal_point):
    global goal, current_position

    tolerance = 0.01
    current_orientation = current_position[2] 
    new_orientation = math.atan2(goal_point[1], goal_point[0])

    while new_orientation > math.pi:
        new_orientation -= 2 * math.pi
    while new_orientation < -math.pi:
        new_orientation += 2 * math.pi

    move = Twist()
    pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
    rate = rospy.Rate(10) 

    while True:
        angle_difference = new_orientation - current_position[2]
        
        while angle_difference > math.pi:
            angle_difference -= 2 * math.pi
        while angle_difference < -math.pi:
            angle_difference += 2 * math.pi

        if abs(angle_difference) < tolerance:
            break

        k = 1.6    
        angular_speed = max(min(angle_difference*k, MAX_ANGULAR_SPEED), -MAX_ANGULAR_SPEED)
        move.angular.z = angular_speed 
        pub.publish(move)
        rate.sleep()

    move.angular.z = 0
    pub.publish(move)

##########################################################################################################################
#################### Auxiliary functions for calculations of geometric points, lines, distances, etc. #################### 
# This function calculates the distance between two points
def distance(a,b):
    distance_between_a_and_b = math.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2) 
    return distance_between_a_and_b

# This function calculates the slope (a) and y-intercept (b) of the line passing through the two points
def calculate_line(point1, point2):
    x1, y1 = point1
    x2, y2 = point2
    if round(x2 - x1,2) == 0:
        a = float('inf')
        b = x1
    else:
        a = (y2 - y1) / (x2 - x1)
        b = y1 - a * x1

    return a, b

# This function calculates the midpoint of a line segment
def find_midpoint(a, b): 
    if a is None or b is None:
        raise ValueError("One of the points is None")
    
    x1, y1 = a
    x2, y2 = b
    
    xm = (x1 + x2) / 2
    ym = (y1 + y2) / 2
    
    return [xm, ym]

# This function calculates the perpendicular line to a line segment
def find_perpendicular(a, b):
    if a is None or b is None:
        raise ValueError("One of the points is None")
    
    # Compute the direction vector from A to B
    xa, ya = a
    xb, yb = b

    # Perpendicular vector to AB
    # Swap x and y and negate one of them to get a perpendicular vector
    dx = xb - xa
    dy = yb - ya

    # Get the perpendicular vector by rotating 90 degrees
    perp = (-dy, dx)

    # Normalize the perpendicular vector to make it a unit vector
    length = math.sqrt(perp[0]**2 + perp[1]**2)
    perp_unit = [perp[0] / length, perp[1] / length]
    
    return perp_unit

# This function calculates the mid vertical line to a line segment
def find_mid_vertical_segment(a, b, offset):
    if a is None or b is None:
        raise ValueError("One of the points is None")
    
    # Calculate the midpoint
    midpoint = find_midpoint(a, b)
    
    # Get the perpendicular unit vector
    perp_unit = find_perpendicular(a, b)
    
    # Calculate points C and D 0.4 meter in opposite directions along the perpendicular line
    xm, ym = midpoint
    
    c = [xm + offset * perp_unit[0], ym + offset * perp_unit[1]]
    d = [xm - offset * perp_unit[0], ym - offset * perp_unit[1]]
    
    return [c, d]

def find_intersection(a1, b1, a2, b2):
    # Check if the first line is vertical
    if a1 == float('inf'):
        if a2 == float('inf'):
            # Both lines are vertical
            if b1 == b2:
                return (b1, float('inf'))  # Identical lines
            else:
                return None  # Parallel and do not intersect
        else:
            # The first line is vertical and the second is not
            x = b1
            y = a2 * x + b2
            return (x, y)

    # Check if the second line is vertical
    if a2 == float('inf'):
        # The second line is vertical and the first is not
        x = b2
        y = a1 * x + b1
        return (x, y)

    # Check if the lines are parallel
    if a1 == a2:
        return None  # The lines are parallel and do not intersect
    
    # Calculating the x-intercept
    x = (b2 - b1) / (a1 - a2)
    
    # Calculating the y-intercept
    y = a1 * x + b1
    
    return (x, y)

################################################################################################
#####################################  Gen3 Lite Control  ######################################

def all_close(goal, actual, tolerance):
    """
    Convenience method for testing if a list of values are within a tolerance of their counterparts in another list
    @param: goal       A list of floats, a Pose or a PoseStamped
    @param: actual     A list of floats, a Pose or a PoseStamped
    @param: tolerance  A float
    @returns: bool
    """
    all_equal = True
    if type(goal) is list:
        for index in range(len(goal)):
            if abs(actual[index] - goal[index]) > tolerance:
                return False

    elif type(goal) is geometry_msgs.msg.PoseStamped:
        return all_close(goal.pose, actual.pose, tolerance)

    elif type(goal) is geometry_msgs.msg.Pose:
        return all_close(pose_to_list(goal), pose_to_list(actual), tolerance)

    return True

class MoveItContext(object):
    def __init__(self):
        moveit_commander.roscpp_initialize(sys.argv)

        robot = moveit_commander.RobotCommander()
        scene = moveit_commander.PlanningSceneInterface()
        group_name = "arm"
        move_group = moveit_commander.MoveGroupCommander(group_name)

        planning_frame = move_group.get_planning_frame()
        end_effector_link = move_group.get_end_effector_link()
        group_names = robot.get_group_names()

        self.box_name = ''
        self.robot = robot
        self.scene = scene
        self.move_group = move_group
        self.planning_frame = planning_frame
        self.end_effector_link = end_effector_link
        self.group_names = group_names

        self.move_group.set_goal_orientation_tolerance(0.5)
        self.move_group.set_goal_position_tolerance(0.05)

    # Step 0 ##########################################################
    # For all cases
    def change_start_position(self):
        print("step 0")

        joint_goal = self.move_group.get_current_joint_values()
        joint_goal[4] = joint_goal[4] + (60.0 * 3.14159265359 / 180.0)  
        joint_goal[2] = joint_goal[2] - (60.0 * 3.14159265359 / 180.0)   

        self.move_group.go(joint_goal, wait=True)
        self.move_group.stop()
        
    # For all cases
    def unchange_start_position(self):
        print("step -0")

        joint_goal = self.move_group.get_current_joint_values()
        joint_goal[4] = joint_goal[4] - (60.0 * 3.14159265359 / 180.0)  
        joint_goal[2] = joint_goal[2] + (60.0 * 3.14159265359 / 180.0)   

        self.move_group.go(joint_goal, wait=True)
        self.move_group.stop()

    # Step 1 ##########################################################
    #                             Outdoor 
    # case 1 : Outdoor to the left
    def rotate_joints_step1_left(self):
        print("step 1 left")

        joint_goal = self.move_group.get_current_joint_values()
        joint_goal[4] = joint_goal[4] + (15.0 * 3.14159265359 / 180.0)
        joint_goal[2] = joint_goal[2] + (60.0 * 3.14159265359 / 180.0)   
        joint_goal[0] = joint_goal[0] + (10.0 * 3.14159265359 / 180.0)

        self.move_group.go(joint_goal, wait=True)
        self.move_group.stop()

    # case 2 : Outdoor to the right
    def rotate_joints_step1_right(self):
        print("step 1 right")

        joint_goal = self.move_group.get_current_joint_values()
        joint_goal[4] = joint_goal[4] - (15.0 * 3.14159265359 / 180.0)
        joint_goal[2] = joint_goal[2] + (60.0 * 3.14159265359 / 180.0)      
        joint_goal[0] = joint_goal[0] - (10.0 * 3.14159265359 / 180.0)
        
        self.move_group.go(joint_goal, wait=True)
        self.move_group.stop()
    #                             Indoor 
    # case 3 : Indoor to the left
    def rotate_joints_step1_left_indoor(self):
        print("step 1 left indoor")

        joint_goal = self.move_group.get_current_joint_values()
        joint_goal[4] = joint_goal[4] - (10.0 * 3.14159265359 / 180.0)
        joint_goal[2] = joint_goal[2] + (60.0 * 3.14159265359 / 180.0)      
        joint_goal[0] = joint_goal[0] - (2.0 * 3.14159265359 / 180.0)
    
        self.move_group.go(joint_goal, wait=True)
        self.move_group.stop()

    # case 4 : Indoor to the right
    def rotate_joints_step1_right_indoor(self):
        print("step 1 right indoor")

        joint_goal = self.move_group.get_current_joint_values()
        joint_goal[4] = joint_goal[4] + (10.0 * 3.14159265359 / 180.0)
        joint_goal[2] = joint_goal[2] + (60.0 * 3.14159265359 / 180.0)   
        joint_goal[0] = joint_goal[0] + (2.0 * 3.14159265359 / 180.0)
        
        self.move_group.go(joint_goal, wait=True)
        self.move_group.stop()

    # Step 2 ##########################################################
    # For all cases    
    def rotate_joints_step2(self):
        print("step 2")

        joint_goal = self.move_group.get_current_joint_values()
        joint_goal[2] = joint_goal[2] - (40.0 * 3.14159265359 / 180.0) 
        joint_goal[1] = joint_goal[1] - (40.0 * 3.14159265359 / 180.0)  

        self.move_group.go(joint_goal, wait=True)
        self.move_group.stop()

    # Step 3 ##########################################################
    #                             Outdoor 
    # case 1 : Outdoor to the left
    def rotate_joints_step3_left(self):
        print("step 3 left")

        joint_goal = self.move_group.get_current_joint_values()
        joint_goal[4] = joint_goal[4] + (20.0 * 3.14159265359 / 180.0)
        joint_goal[0] = joint_goal[0] + (55.0 * 3.14159265359 / 180.0)

        self.move_group.go(joint_goal, wait=True)
        self.move_group.stop()

    # case 2 : Outdoor to the right
    def rotate_joints_step3_right(self):
        print("step 3 right")

        joint_goal = self.move_group.get_current_joint_values()
        joint_goal[4] = joint_goal[4] - (25.0 * 3.14159265359 / 180.0)
        joint_goal[0] = joint_goal[0] - (55.0 * 3.14159265359 / 180.0)

        self.move_group.go(joint_goal, wait=True)
        self.move_group.stop()
    #                             Indoor 
    # case 3 : Indoor to the left
    def rotate_joints_step3_left_indoor(self):
        print("step 3 left indoor")

        joint_goal = self.move_group.get_current_joint_values()
        joint_goal[4] = joint_goal[4] + (40.0 * 3.14159265359 / 180.0)
        joint_goal[0] = joint_goal[0] + (55.0 * 3.14159265359 / 180.0)

        self.move_group.go(joint_goal, wait=True)
        self.move_group.stop()

    # case 4 : Indoor to the right
    def rotate_joints_step3_right_indoor(self):
        print("step 3 right indoor")

        joint_goal = self.move_group.get_current_joint_values()
        joint_goal[4] = joint_goal[4] - (40.0 * 3.14159265359 / 180.0)
        joint_goal[0] = joint_goal[0] - (55.0 * 3.14159265359 / 180.0)

        self.move_group.go(joint_goal, wait=True)
        self.move_group.stop()
        
    # Step 4 ##########################################################
    #                             Outdoor 
    # case 1 : Outdoor to the left
    def rotate_joints_step4_left(self):
        print("step 4 left")

        joint_goal = self.move_group.get_current_joint_values()
        joint_goal[4] = joint_goal[4] - (35.0 * 3.14159265359 / 180.0)
        joint_goal[2] = joint_goal[2] - (20.0 * 3.14159265359 / 180.0)  
        joint_goal[1] = joint_goal[1] + (40.0 * 3.14159265359 / 180.0)  
        joint_goal[0] = joint_goal[0] - (65.0 * 3.14159265359 / 180.0) 

        self.move_group.go(joint_goal, wait=True)
        self.move_group.stop()
    
    # case 2 : Outdoor to the right
    def rotate_joints_step4_right(self):
        print("step 4 right")

        joint_goal = self.move_group.get_current_joint_values()
        joint_goal[4] = joint_goal[4] + (35.0 * 3.14159265359 / 180.0) 
        joint_goal[2] = joint_goal[2] - (20.0 * 3.14159265359 / 180.0)
        joint_goal[1] = joint_goal[1] + (40.0 * 3.14159265359 / 180.0) 
        joint_goal[0] = joint_goal[0] + (65.0 * 3.14159265359 / 180.0) 
        
        self.move_group.go(joint_goal, wait=True)
        self.move_group.stop()
    #                             Idoor 
    # case 3 : Indoor to the left
    def rotate_joints_step4_left_indoor(self):
        print("step 4 left indoor")

        joint_goal = self.move_group.get_current_joint_values()
        joint_goal[4] = joint_goal[4] - (25.0 * 3.14159265359 / 180.0)
        joint_goal[2] = joint_goal[2] - (35.0 * 3.14159265359 / 180.0)  
        joint_goal[1] = joint_goal[1] + (40.0 * 3.14159265359 / 180.0)  
        joint_goal[0] = joint_goal[0] - (53.0 * 3.14159265359 / 180.0) 

        self.move_group.go(joint_goal, wait=True)
        self.move_group.stop()
    
    # case 4 : Indoor to the right
    def rotate_joints_step4_right_indoor(self):
        print("step 4 right indoor")

        joint_goal = self.move_group.get_current_joint_values()
        joint_goal[4] = joint_goal[4] + (25.0 * 3.14159265359 / 180.0) 
        joint_goal[2] = joint_goal[2] - (35.0 * 3.14159265359 / 180.0)
        joint_goal[1] = joint_goal[1] + (40.0 * 3.14159265359 / 180.0) 
        joint_goal[0] = joint_goal[0] + (53.0 * 3.14159265359 / 180.0) 
        
        self.move_group.go(joint_goal, wait=True)
        self.move_group.stop()

def open_the_door():
    global door, door_case

    moveit_context = MoveItContext()

    if door_case == 1: # Outdoor to the left
        moveit_context.rotate_joints_step1_left()
        moveit_context.rotate_joints_step2()
        moveit_context.rotate_joints_step3_left()
        moveit_context.rotate_joints_step4_left()
    elif door_case == 2: # Outdoor to the right
        moveit_context.rotate_joints_step1_right()
        moveit_context.rotate_joints_step2()
        moveit_context.rotate_joints_step3_right()
        moveit_context.rotate_joints_step4_right()
    elif door_case == 3: # Indoor to the left
        moveit_context.rotate_joints_step1_left_indoor()
        moveit_context.rotate_joints_step2()
        moveit_context.rotate_joints_step3_left_indoor()
        moveit_context.rotate_joints_step4_left_indoor()
    elif door_case == 4: # Indoor to the right
        moveit_context.rotate_joints_step1_right_indoor()
        moveit_context.rotate_joints_step2()
        moveit_context.rotate_joints_step3_right_indoor()
        moveit_context.rotate_joints_step4_right_indoor()


################################################################################################
##########################################  Graphics  ##########################################
def plot_velocities():
    plt.title('The angular velocity of Robot')
    plt.xlabel('Time(sec)')
    plt.ylabel('Angular velocity(rad/sec)')
    plt.plot(plot_angular_speed, 'blue', linewidth = 1, label = 'Expected' )
    plt.plot(plot_real_angular_speed,'red', linewidth = 0.5, label = 'Robot' )
    plt.legend()
    plt.show()
    plt.clf()

    plt.title('The linear velocity of Robot')
    plt.xlabel('Time(sec)')
    plt.ylabel('Velocity(m/sec)')
    plt.plot(plot_linear_speed, 'blue', linewidth = 1 , label = 'Expected')
    plt.plot(plot_real_linear_speed, 'red', linewidth = 0.5, label = 'Robot' )
    plt.legend()
    plt.show()
    plt.clf() 

################################################################################################
##########################################    Main    ##########################################
if __name__ == '__main__':
    rospy.init_node('topic_mover')
    rospy.loginfo("Jackal has started !!!")

    # Connections with ros topics
    pub = rospy.Publisher('/jackal_velocity_controller/cmd_vel', Twist, queue_size=10)
    sub = rospy.Subscriber('/jackal_velocity_controller/odom',Odometry, get_u)
    sub2 = rospy.Subscriber('/gazebo/link_states', LinkStates, get_jackal_orientation)
    scan1 = rospy.Subscriber('/scan', LaserScan, find_door)
    scan2=rospy.Subscriber('/scan', LaserScan, get_scan)
    rate = rospy.Rate(100)
    move = Twist()

    wakeup()
    
    #if door_case != None:
    while door_case == None:
        pass

    print("door = ", door)
    print("door_case = ", door_case)
    moveit_context = MoveItContext()
    moveit_context.change_start_position()
    check_for_partial_open_door()
   
    rospy.loginfo("Jackal has finished!!!")
    plot_velocities()