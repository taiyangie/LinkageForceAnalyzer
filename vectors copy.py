#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 18 23:09:13 2023

@author: william_hess
"""
#%%
import pandas as pd
from mpl_toolkits.mplot3d import proj3d
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch
import numpy as np
from mpl_toolkits.mplot3d import proj3d
#%% lists
Mag_list = [] # holding list for magnitude fectore
vect_list = [] #Vector holding list
unit_vect_list =[] #unit vector holding list
WC_vect_list = [] # Wheel Center holding list
moment_list = [] #moment holding list
TP_moment_arm_list = [] #moment arms for Tire patch to wheel center holding list
PCR_list = []
Fmax_list = []
Tube_mass_list = []
#%% data
points = pd.read_excel("Sus_Arm_points.xlsx").set_index("Points") # inboard and outboard points of the suspention 
WC = pd.read_excel("Sus_Wheel_center.xlsx").set_index("Points") # points for wheel centers of each wheel
TP = pd.read_excel("Sus_Center_of_tire_patch.xlsx").set_index("Points") #Tire Patch Center data
#%%
WC_list = list(WC) # list of column names of Wheel centers
WC_axis_list = list(WC.index.values) # list of index values for wheel centers
TP_list = list(TP) # list of column names of Wheel centers
TP_axis_list = list(TP.index.values) # List of index values for the tire patch
arm_list = list(points) # list of the name of every suspention arm 24 in total
FR_list = arm_list[0:6] # list of arms on the Front Right of the Car
FL_list = arm_list[6:12] # list of arms on the Front LEft of the car
RR_list = arm_list[12:18] # list of arms on the Rear Right of the car
RL_list = arm_list[18:24] # list of arms on the Rear Left of the car
vect_name_list = ['X_vector', 'Y_vector', 'Z_vector'] # list for vector index
WC_name_list = ['rx', 'ry', 'rz'] #list for Wheel moment center index
unit_vect_name_list = ['Xu', 'Yu', 'Zu'] # list for unit vector index
moment_name_list = ["Mx", "My", "Mz"] # list for moment index
TP_name_list = ["Rx", "Ry", "Rz"] # list of tire patch moment for index assignment
LA_name_list = ['Fx LA', 'Fy LA', "Fz LA"] #List for index of Linear Acceleration forces
BR_name_list = ['Fx BR', 'Fy BR', "Fz BR"]
SSC_name_list = ['Fx SSC', 'Fy SSC', "Fz SSC"]
B_name_list = ['Fx B', 'Fy B', "Fz B"]
tire_names = ["FR", "FL", "RR", "RL"] # List of the names of tires based on lication
arm_names = ["TR", "LCAF", "LCAR", "UCAF", "UCAR", "PR"] #Name of arms without the Wheel position
point_list = list(points.index.values) # list of index of the points data
#%%
#%% Analysis and vectors for moment arms
def WC_moment_arm(Alist, WClist_pos): # lengths, calculates moment arms for all of the suspention points to the Wheel Center
    for i in range(0, len(Alist)):
        vecti = [points[Alist[i]][0] - WC[WClist_pos][0], points[Alist[i]][1] - WC[WClist_pos][1], points[Alist[i]][2] - WC[WClist_pos][2]] #vector of the wheel center to suspention point
        WC_vect_list.append(vecti) #adds point to the list
    return WC_vect_list #outputs list
WC_moment_arm(FR_list, "FR WC") #Runs calculation for wheel center moment arms to the outboard suspention points on the Front Right of the car
WC_moment_arm(FL_list, "FL WC") #Runs calculation for wheel center moment arms to the outboard suspention points on the Front Left of the car
WC_moment_arm(RR_list, "RR WC") #Runs calculation for wheel center moment arms to the outboard suspention points on the Rear Right of the car
WC_moment_arm(RL_list, "RL WC") #Runs calculation for wheel center moment arms to the outboard suspention points on the Rear Left of the car

def WC_graphing(Alist, WClist_pos): #graphing the blue lines on the geometry check showing the moment arms 
    for i in range(0, len(Alist)):
        x = np.array([points[Alist[i]][0], WC[WClist_pos][0]])
        y = np.array([points[Alist[i]][1], WC[WClist_pos][1]])
        z = np.array([points[Alist[i]][2], WC[WClist_pos][2]])
        ax.plot(x,y,z, color = 'b') #plots in 3d in Blue
        
class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        super().__init__((0,0), (0,0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def do_3d_projection(self, renderer=None):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))

        return np.min(zs)
#%%
fig1 = plt.figure(num = 1, clear = True)
ax = fig1.add_subplot(1,1,1, projection='3d')
for i in range(0, len(arm_list)): #Arms and Points
    x = np.array([points[arm_list[i]][point_list[0]], points[arm_list[i]][point_list[3]]])
    y = np.array([points[arm_list[i]][point_list[1]], points[arm_list[i]][point_list[4]]])
    z = np.array([points[arm_list[i]][point_list[2]], points[arm_list[i]][point_list[5]]])
    ax.scatter(x,y,z, c='red', s=20)
    arrow_prop_dict = dict(mutation_scale=15, arrowstyle='-|>', color='k', shrinkA=0, shrinkB=0)
    a = Arrow3D(x,y,z, **arrow_prop_dict)
    ax.add_artist(a)
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

# Origin also hides other points
ax.scatter(0,0,0, color = 'magenta', label = 'Origin', s = 40, zorder = 500) # Origin
ax.legend(loc = 'best')
ax.set(xlabel = "X (in)", ylabel = 'Y (in)', zlabel = 'Z (in)', title = "Arm Vectors")
fig1.set_facecolor('grey')
ax.set_facecolor('grey')
plt.show(block=True) 
        
        
        
        
        