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
WC_vect_list = []
moment_list = []
TP_moment_arm_list = []
#%% Data in, need to fix pathing for git
points = pd.read_excel("Sus_Arm_points.xlsx").set_index("Points")
WC = pd.read_excel("Sus_Wheel_center.xlsx").set_index("Points")
TP = pd.read_excel("Sus_Center_of_tire_patch.xlsx").set_index("Points")
#%% Data Lists
WC_list = list(WC)
WC_axis_list = list(WC.index.values)
TP_list = list(TP)
TP_axis_list = list(TP.index.values)
arm_list = list(points)
FR_list = arm_list[0:6]
FL_list = arm_list[6:12]
RR_list = arm_list[12:18]
RL_list = arm_list[18:24]
vect_name_list = ['X_vector', 'Y_vector', 'Z_vector']
WC_name_list = ['rx', 'ry', 'rz']
unit_vect_name_list = ['Xu', 'Yu', 'Zu']
moment_name_list = ["Mx", "My", "Mz"]
TP_name_list = ["Rx", "Ry", "Rz"]
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
def WC_moment_arm(Alist, WClist_pos): # lengths
    for i in range(0, len(Alist)):
        vecti = [points[Alist[i]][0] - WC[WClist_pos][0], points[Alist[i]][1] - WC[WClist_pos][1], points[Alist[i]][2] - WC[WClist_pos][2]]
        WC_vect_list.append(vecti)
    return WC_vect_list
WC_moment_arm(FR_list, "FR WC")
WC_moment_arm(FL_list, "FL WC")
WC_moment_arm(RR_list, "RR WC")
WC_moment_arm(RL_list, "RL WC")
def WC_graphing(Alist, WClist_pos): #graphing
    for i in range(0, len(Alist)):
        x = np.array([points[Alist[i]][0], WC[WClist_pos][0]])
        y = np.array([points[Alist[i]][1], WC[WClist_pos][1]])
        z = np.array([points[Alist[i]][2], WC[WClist_pos][2]])
        ax.plot(x,y,z, color = 'b')
#%% Formating vector data
vect_frame = pd.DataFrame(vect_list).transpose().set_axis(vect_name_list)
vect_frame.columns = arm_list
WC_vect_frame = pd.DataFrame(WC_vect_list).transpose().set_axis(WC_name_list)
WC_vect_frame.columns = arm_list
unit_vect_frame = pd.DataFrame(unit_vect_list).transpose().set_axis(unit_vect_name_list)
unit_vect_frame.columns = arm_list
mag_frame = pd.DataFrame(Mag_list, index = arm_list).transpose().set_axis(["Arm_Length"])
vect_data = pd.concat([points, mag_frame, vect_frame, unit_vect_frame, WC_vect_frame])
#%% Moment EQs: 
for i in range(0, len(arm_list)): #moment arms for points
    moment_calcs = [((unit_vect_frame[arm_list[i]][2])*(WC_vect_frame[arm_list[i]][1])) - ((unit_vect_frame[arm_list[i]][1])*(WC_vect_frame[arm_list[i]][2])), ((unit_vect_frame[arm_list[i]][2])*(WC_vect_frame[arm_list[i]][0])) - ((unit_vect_frame[arm_list[i]][0])*(WC_vect_frame[arm_list[i]][2])), ((unit_vect_frame[arm_list[i]][1])*(WC_vect_frame[arm_list[i]][0])) - ((unit_vect_frame[arm_list[i]][0])*(WC_vect_frame[arm_list[i]][1]))]
    moment_list.append(moment_calcs)
moment_frame = pd.DataFrame(moment_list).transpose().set_axis(moment_name_list)
moment_frame.columns = arm_list
for i in range(0, len(TP_list)): #moment arms for Tire patch to wheel center
    TP_moment_arm = [WC[WC_list[i]][0] - TP[TP_list[i]][0], WC[WC_list[i]][1] - TP[TP_list[i]][1], WC[WC_list[i]][2] - TP[TP_list[i]][2]]
    TP_moment_arm_list.append(TP_moment_arm)
TP_frame = pd.DataFrame(TP_moment_arm_list).transpose().set_axis(TP_name_list)
TP_frame.columns = TP_list
#%% Matrix Data getting [A] matrix 
matrix_data = pd.concat([unit_vect_frame, moment_frame])
FR_matrix = matrix_data.iloc[:,0:6]
FL_matrix = matrix_data.iloc[:,6:12]
RR_matrix = matrix_data.iloc[:,12:18]
RL_matrix = matrix_data.iloc[:,18:24]
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
    ax.scatter(x,y,z, c='green', s = 40)
for i in range(0, len(TP_list)): #Tire Patch Centers
    x = TP[TP_list[i]][TP_axis_list[0]]
    y = TP[TP_list[i]][TP_axis_list[1]]
    z = TP[TP_list[i]][TP_axis_list[2]]
    ax.scatter(x,y,z, c='darkorange', s = 40)
WC_graphing(FR_list, "FR WC")
WC_graphing(FL_list, "FL WC")
WC_graphing(RR_list, "RR WC")
WC_graphing(RL_list, "RL WC")
#colours for legend hidden in origin
ax.scatter(0,0,0, color = 'red', label = 'Suspension Points', s = 20, zorder = 2)
ax.scatter(0,0,0, color = 'green', label = 'Wheel Centers', s = 20, zorder = 1)
ax.scatter(0,0,0, color = 'darkorange', label = 'Contact Patch Center', s = 20, zorder = 1)
ax.plot([0,0],[0,0], [0,0], color = 'k', label = "Suspension Arms")
ax.plot([0,0],[0,0], [0,0], color = 'b', label = "Moment Arms")
# Origin also hides other points
ax.scatter(0,0,0, color = 'magenta', label = 'Origin', s = 40, zorder = 500) # Origin
ax.legend(loc = 'best')
ax.set(xlabel = "X (in)", ylabel = 'Y (in)', zlabel = 'Z (in)', title = "Geometry Setup Check")
plt.show()