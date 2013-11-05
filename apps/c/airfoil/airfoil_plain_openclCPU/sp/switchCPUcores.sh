#!/bin/bash

if [ -z $1 ]; then
  echo "Specify On/Off option: 0 or 1"
  exit;
fi


for i in {16..31} 
do
  #echo "/sys/devices/system/cpu/cpu$i/online"
  if [ $1 -eq 0 ]; then
    sudo sh -c "echo 0 > /sys/devices/system/cpu/cpu$i/online"
  fi
  if [ $1 -eq 1 ]; then
    sudo sh -c "echo 1 > /sys/devices/system/cpu/cpu$i/online"
  fi
 
  
done
