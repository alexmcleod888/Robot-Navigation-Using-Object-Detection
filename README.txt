Author: Alex McLeod
Purpose: Code that uses a virtual robot (LoCoBot) camera and cocoMobilenet model to detect and avoid simulated objects in Gazebo simulated environment. 
LINK TO PROJECT PORTFOLIO: https://1drv.ms/f/s!ArmUv9st2PG-jc4LVLTEVWI__0uR1w?e=zkUeHI 

Requirements"
- Ubuntu 16.04 must be used.
- install pyrobot and simulated LoCoBot for Gazebo simulation environment
	- https://pyrobot.org/docs/software
	- ensure simulated verson is installed, python 2 (python 2.7) version of pyrobot is used and interbotix 
	  locobot hardware version used.
- numpy 1.16.6
- tensorflow 2.1.0
Note: 
- new world file "ground.world" included in folder. Replace default ground.world file with   
  this file at:
	low_cost_ws/src/pyrobot/robots/LoCoBot/locobot_gazebo/worlds
  This will load simulation with simulated objects.

How To Run:

1. Load pyrobot python2 virtual environment:
	load_pyrobot_env
2. To start robot run script "startRobot.sh"
	bash startRobot.sh
3. In seperate terminal load the same environment and run python code "RobotObjDetectionTest.py"
	load_pyrobot_env
	python RobotObjDetection.py
	
