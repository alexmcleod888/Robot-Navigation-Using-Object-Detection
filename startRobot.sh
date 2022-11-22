#!/bin/bash

source devel/setup.bash
roslaunch locobot_control main.launch use_base:=true use_sim:=true use_rviz:=false use_arm:=true use_camera:=true
