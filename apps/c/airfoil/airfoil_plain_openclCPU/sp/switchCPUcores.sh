#!/bin/bash

#if [ -z $1 ]; then
#  echo "Specify number of virtual cores, eg. 32 for a 2 socket 8 core/CPU system"
#  exit;
#fi
if [ -z $1 ]; then
  echo "Specify On/Off option: 0 or 1"
  exit;
fi

#let "LOW = $1 / 2"
#let "HIGH = $1 - 1"
#for i in `seq $LOW $HIGH` 
for i in 6 7 8 9 10 11 18 19 20 21 22 23 
do
  echo "$i"
  #echo "/sys/devices/system/cpu/cpu$i/online"
  if [ $1 -eq 0 ]; then
    sudo sh -c "echo 0 > /sys/devices/system/cpu/cpu$i/online"
  fi
  if [ $1 -eq 1 ]; then
    sudo sh -c "echo 1 > /sys/devices/system/cpu/cpu$i/online"
  fi
done
