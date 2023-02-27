#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 21 13:47:29 2023
@author: william_hess
"""
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
#%% Retainer variables
Mag_list = []
vect_list = []
unit_vect_list =[]
#%% Data in, need to fix pathing for git
points = pd.read_excel("Sus_Arm_points.xlsx").set_index("Points")
WC = pd.read_excel("Sus_Wheel_center.xlsx").set_index("Points")
#%% Data Lists
WC_list = list(WC)
WC_axis_list = list(WC.index.values)
arm_list = list(points)
point_list = list(points.index.values)
#%% Analysis vectors and unit vectors
for i in range(0, len(arm_list)):
    vect = [points[arm_list[i]][point_list[3]] - points[arm_list[i]][point_list[0]], points[arm_list[i]][point_list[4]] - points[arm_list[i]][point_list[1]], points[arm_list[i]][point_list[5]] - points[arm_list[i]][point_list[2]]]
    mag = np.sqrt(np.square(vect[0]) + np.square(vect[1]) + np.square(vect[2]))
    unit_vect = vect/mag
    vect_list.append(vect)
    Mag_list.append(mag)
    unit_vect_list.append(unit_vect)
#%% Analysis and vectors for moment arms

#%% Formating data
vect_frame = pd.DataFrame(vect_list).transpose().set_axis(['X_vector', 'Y_vector', 'Z_vector'])
vect_frame.columns = arm_list
unit_vect_frame = pd.DataFrame(unit_vect_list).transpose().set_axis(['Xu', 'Yu', 'Zu'])
unit_vect_frame.columns = arm_list
mag_frame = pd.DataFrame(Mag_list, index = arm_list).transpose().set_axis(["Arm_Length"])
vect_data = pd.concat([points, mag_frame, vect_frame, unit_vect_frame])
#%% Visual Graph of arms and points
fig = plt.figure(num = 1, clear = True)
ax = fig.add_subplot(1,1,1, projection='3d')
for i in range(0, len(arm_list)): #Arms and Points
    x = np.array([points[arm_list[i]][point_list[3]], points[arm_list[i]][point_list[0]]])
    y = np.array([points[arm_list[i]][point_list[4]], points[arm_list[i]][point_list[1]]])
    z = np.array([points[arm_list[i]][point_list[5]], points[arm_list[i]][point_list[2]]])
    ax.scatter(x,y,z, c='red', s=20)
    ax.plot(x,y,z, color='k')
for i in range(0, len(WC_list)): #Wheel Centers
    x = WC[WC_list[i]][WC_axis_list[0]]
    y = WC[WC_list[i]][WC_axis_list[1]]
    z = WC[WC_list[i]][WC_axis_list[2]]
    ax.scatter(x,y,z, c='green', s = 100)
ax.scatter(0,0,0, color = 'b', label = 'Origin') # Origin
plt.show()