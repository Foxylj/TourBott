# TourBott
USC EE434 Capstone Project. 

The purpose of this project is to devise a solution to navigating around an indoor space, and to take visitors or workers to the correct destination, with minimal error and difficulties. 

ROS Packages:
mpu_6050_driver
robot_setup_tf
jetson_nano_bot
rplidar_ros 
hector_slam 

Important Files:
RemoteController.ino ~ robot controlls
~/catkin_ws/src/mpu_6050_driver/scripts/imu_node.py
~/catkin_ws/src/mpu_6050_driver/scripts/tf_broadcaster_imu.py
~/catkin_ws/src/mpu_6050_driver/scripts/registers.py
~/catkin_ws/src/jetson_nano_bot/launch/jetson_nano_bot.launch


Run the Program:
roslaunch jetson_nano_bot.launch
