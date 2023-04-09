#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 29 01:31:40 2023

@author: william_hess
Same as SUS_calcs2 but without GUI Output
"""
import pandas as pd
from tkinter import *
from tkinter import ttk
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk)
import matplotlib as mpl
#%% Retainer variables
Mag_list = []
vect_list = []
unit_vect_list =[]
WC_vect_list = []
moment_list = []
TP_moment_arm_list = []
#%% Data in, maybe need to fix pathing for git
points = pd.read_excel("Sus_Arm_points.xlsx").set_index("Points")
WC = pd.read_excel("Sus_Wheel_center.xlsx").set_index("Points")
TP = pd.read_excel("Sus_Center_of_tire_patch.xlsx").set_index("Points")
WD = pd.read_excel("Sus_Wheel_center.xlsx")
LandB_var = pd.read_excel("variables.xlsx")
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
LA_name_list = ['Fx LA', 'Fy LA', "Fz LA"]
tire_names = ["FR", "FL", "RR", "RL"]
arm_names = ["TR", "LCAF", "LCAR", "UCAF", "UCAR", "PR"]
point_list = list(points.index.values)
Wb = TP["RR TP"][0] - TP["FR TP"][0] # finds Wheel Base
Tw = abs(TP["FR TP"][1]) + abs(TP["FR TP"][1]) #Finds Track Width
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
for i in range(0, len(arm_list)): #moment arms for points to Wheel Center
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
A_list = [FR_matrix, FL_matrix, RR_matrix, RL_matrix]
#%% Visual Graph of arms, points, moment arms, and centers
fig1 = plt.figure(num = 1, clear = True)
ax = fig1.add_subplot(1,1,1, projection='3d')
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
plt.close()
plt.show(block=True)
#%% Functions
def getInputLandB():
    We = WeV.get()
    F = FV.get()
    R = RV.get()
    mu = muV.get()
    h = hV.get()
    r = rV.get()
    c = cV.get()
    b = bV.get()
    g = gV.get()
    root.destroy()
    global paramsLandB
    paramsLandB = [We, F, R, mu, h, r, c, b, g]

def betterGUI(frs, i): 
    ttk.Label(mainframe, text=frs["Description"][i]).grid(column=1, row=i, sticky=W)
    globals()[frs["var_name"][i] + 'V'] = DoubleVar()
    Var_entry = ttk.Entry(mainframe, width=7, textvariable= globals()[frs["var_name"][i] + 'V'])
    Var_entry.grid(column=2, row=i, sticky=(W, E))
    ttk.Label(mainframe, text=frs["unit"][i]).grid(column=3, row=i, sticky=W)
    
def ForceParameterGenerator(var, cmd):
    for i in range(0, len(var)):
        betterGUI(var, i)
    ttk.Button(mainframe, text="Calculate", command=cmd).grid(column = 3, row = i+1)
    canvas = FigureCanvasTkAgg(fig1, master=mainframe)
    canvas.draw()
    canvas.get_tk_widget().grid(column=4, row = 1, rowspan=i, sticky = (NE, SW))
    for child in mainframe.winfo_children(): 
        child.grid_configure(padx=5, pady=5)
#%% L and B GUI
root = Tk()  
root.title("LINEAR ACCELERATION and BRAKING PERFORMANCE parameters")
mainframe = ttk.Frame(root, padding="2 4 12 12")
mainframe.grid(column=0, row=0, sticky=(N, W, E, S))
root.columnconfigure(0, weight=1)
root.rowconfigure(0, weight=1)
ForceParameterGenerator(LandB_var, getInputLandB)
root.bind("<Return>", getInputLandB) 
root.mainloop()
#%% Functions for producing B matrix and final Force numbers
#Function can be used to extract the raw B matrix for testing
def Force_to_BMatrix(F_list, index_list, Fcol, TP_R): #List of forces for a specific load case, XYZ index names, column_names, Tire Patch - wheel center moment arm matrix, names of moment arm columns
    mom_list = []
    F_data = pd.DataFrame(F_list).transpose().set_axis(index_list) #turns forces into callable matrix
    F_data.columns = Fcol #assigns colum names for easier calling and readibility
    Rcol = list(TP_R)
    for i in range(0, len(Fcol)):
        Mx = ((F_data[Fcol[i]][2])*(TP_R[Rcol[i]][1])) - ((F_data[Fcol[i]][1])*(TP_R[Rcol[i]][2])) #moment in X for all points on wheels
        My = ((F_data[Fcol[i]][0])*(TP_R[Rcol[i]][2])) - ((F_data[Fcol[i]][2])*(TP_R[Rcol[i]][0])) #moment in Y for all points on wheels
        Mz = ((F_data[Fcol[i]][1])*(TP_R[Rcol[i]][0])) - ((F_data[Fcol[i]][0])*(TP_R[Rcol[i]][1])) #moment in Z for all points on wheels
        M_vect = [Mx, My, Mz] #list of Moments for a given wheel
        mom_list.append(M_vect) #List of list of moments
    M_data = pd.DataFrame(mom_list).transpose().set_axis(moment_name_list) # creates moment dataframe
    M_data.columns = Fcol # give dataframe same column names as F_data to allow matrix to concat
    B_data = pd.concat([F_data, M_data]) #forms B matrix for all wheels with each column being the B matrix for a wheel
    B_data.columns = tire_names # Gives columns new names for just wheel
    return B_data

def WForce_to_AForce(F_list, index_list, Fcol, TP_R, A): #List of forces for a specific load case, XYZ index names, column_names, Tire Patch - wheel center moment arm matrix, names of moment arm columns
    mom_list = [] #storage for moment list
    armF_list = [] #list of forces in each arm
    F_data = pd.DataFrame(F_list).transpose().set_axis(index_list) #turns forces into callable matrix
    F_data.columns = Fcol #assigns colum names for easier calling and readibility
    Rcol = list(TP_R)
    for i in range(0, len(Fcol)):
        Mx = ((F_data[Fcol[i]][2])*(TP_R[Rcol[i]][1])) - ((F_data[Fcol[i]][1])*(TP_R[Rcol[i]][2])) #moment in X for all points on wheels
        My = ((F_data[Fcol[i]][0])*(TP_R[Rcol[i]][2])) - ((F_data[Fcol[i]][2])*(TP_R[Rcol[i]][0])) #moment in Y for all points on wheels
        Mz = ((F_data[Fcol[i]][1])*(TP_R[Rcol[i]][0])) - ((F_data[Fcol[i]][0])*(TP_R[Rcol[i]][1])) #moment in Z for all points on wheels
        M_vect = [Mx, My, Mz] #list of Moments for a given wheel
        mom_list.append(M_vect) #List of list of moments
    M_data = pd.DataFrame(mom_list).transpose().set_axis(moment_name_list) # creates moment dataframe
    M_data.columns = Fcol # give dataframe same column names as F_data to allow matrix to concat
    B_data = pd.concat([F_data, M_data]) #forms B matrix for all wheels with each column being the B matrix for a wheel
    B_data.columns = tire_names # Gives columns new names for just wheel
    for i in range(0, len(tire_names)): #iterates over matrix multiplication
        x_vect = list(np.dot(np.linalg.pinv(A[i]), B_data[tire_names[i]])) #matrix multiplies B matrix by the inverse of the A matrix to get X matrix
        armF_list.append(x_vect) #puts X matrix in a list
    armF = pd.DataFrame(armF_list).transpose().set_axis(arm_names) #Dataframe made of X matrixes, gives all forces in all arms in all tires and names index to arm names
    armF.columns = tire_names #assigns all arms proper tire
    return armF #returns this matrix

# Here make Force to Factor of Safety

#%% Graph functions

def Pos_neg_Cmap_scale(FList, j): #scaling for diverging colormap
    if FList[j] > 0: #positive
       cm_num = (FList[j] / max(FList)) / 2
       return cm_num + 0.5
    elif FList[j] < 0: 
       cm_num = (FList[j] / min(FList)) / 2
       return 0.5 - cm_num
   
def ABS_scale(FList, j): 
    abs_list = np.absolute(np.array(FList))
    cm_num = abs(FList[j]) / max(abs_list)
    return cm_num
    

def colour_arms_graph_TC(Flist, Scale ,num, title, cmp):
    fig = plt.figure(num = num, clear = True)
    ax = fig.add_subplot(1,1,1, projection='3d')
    cmap = mpl.colormaps[cmp]
    norm = mpl.colors.Normalize(vmin = min(Flist), vmax = max(Flist))
    fig.set_facecolor('grey')
    ax.set_facecolor('grey')
    cbar = fig.colorbar(mpl.cm.ScalarMappable(norm = norm, cmap = cmap), ax = ax, label = "Force (lbs)", ticks = [min(Flist), (min(Flist) + ((min(Flist) + max(Flist))/2)) / 2 , ((min(Flist) + max(Flist))/2), (max(Flist) + ((min(Flist) + max(Flist))/2)) / 2, max(Flist)])
    cbar.ax.set_yticklabels([str(round(min(Flist), 1)), str(round((min(Flist)/2), 1)),  '0', str(round((max(Flist)/2), 1)),  str(round(max(Flist), 1))])
    ax.set(xlabel = "X (in)", ylabel = 'Y (in)', zlabel = 'Z (in)', title = title)
    ax.scatter(0,0,0, color = 'magenta', label = 'Origin', s = 40, zorder = 500) # Origin
    ax.legend(loc = 'best')
    for i in range(0, len(arm_list)): #Arms and Points
        x = np.array([points[arm_list[i]][point_list[3]], points[arm_list[i]][point_list[0]]])
        y = np.array([points[arm_list[i]][point_list[4]], points[arm_list[i]][point_list[1]]])
        z = np.array([points[arm_list[i]][point_list[5]], points[arm_list[i]][point_list[2]]])
        ax.scatter(x,y,z, c='k', s=20)
        ax.plot(x,y,z, color= cmap(Scale(Flist, i)))
    plt.show(block = True)
    plt.close()
    return fig

def colour_arms_graph(Flist, Scale ,num, title, cmp):
    fig = plt.figure(num = num, clear = True)
    ax = fig.add_subplot(1,1,1, projection='3d')
    cmap = mpl.colormaps[cmp]
    norm = mpl.colors.Normalize(vmin = min(Flist), vmax = max(Flist))
    fig.set_facecolor('grey')
    ax.set_facecolor('grey')
    fig.colorbar(mpl.cm.ScalarMappable(norm = norm, cmap = cmap), ax = ax, label = "Force (lbs)")
    ax.set(xlabel = "X (in)", ylabel = 'Y (in)', zlabel = 'Z (in)', title = title)
    ax.scatter(0,0,0, color = 'magenta', label = 'Origin', s = 40, zorder = 500) # Origin
    ax.legend(loc = 'best')
    for i in range(0, len(arm_list)): #Arms and Points
        x = np.array([points[arm_list[i]][point_list[3]], points[arm_list[i]][point_list[0]]])
        y = np.array([points[arm_list[i]][point_list[4]], points[arm_list[i]][point_list[1]]])
        z = np.array([points[arm_list[i]][point_list[5]], points[arm_list[i]][point_list[2]]])
        ax.scatter(x,y,z, c='k', s=20)
        ax.plot(x,y,z, color= cmap(Scale(Flist, i)))
    plt.show(block = True)
    plt.close()
    return fig
#%% Inputs and calculating other constants
[TR, LCAF, LCAR, UCAF, UCAR, PR] = [0, 1, 2, 3, 4, 5] # assigns a variable name that is the same as the index position so that a Force case can be looked up by tire position and arm name to get a specific force
[W, F, R, mu, h, r, c, b, g_lin] = paramsLandB #links used varibales from GUI inputs to math for forces
Wb = TP["RR TP"][0] - TP["FR TP"][0] # finds Wheel Base
Tw = abs(TP["FR TP"][1]) + abs(TP["FR TP"][1]) #Finds Track Width
#%% Calculating Linear Acc Forces at contact patch
FR_LA = [0, 0, ((W * ((c/Wb) - g_lin*(h/Wb))) / 2)] #XYZ order for forces in each arm
FL_LA = [0, 0, ((W * ((c/Wb) - g_lin*(h/Wb))) / 2)]
RR_LA = [((((mu*W*b)/Wb) / (1-(h/Wb)*mu)) / 2), 0, ((W * ((b/Wb) + g_lin*(h/Wb))) / 2)]
RL_LA = [((((mu*W*b)/Wb) / (1-(h/Wb)*mu)) / 2), 0, ((W * ((b/Wb) + g_lin*(h/Wb))) / 2)]
Force_LA = [FR_LA, FL_LA, RR_LA, RL_LA] # List of Force lists for each tire
LA_col = ['FR_LA', 'FL_LA', 'RR_LA', 'RL_LA'] #List to help call columns for math
#%% Linear acceleration forces at Arms Viaual
LA_Forces = WForce_to_AForce(Force_LA, LA_name_list, LA_col, TP_frame, A_list) #Wheel Force to arm Force
LA_Forces_List = list(LA_Forces['FR']) + list(LA_Forces['FL']) + list(LA_Forces['RR']) + list(LA_Forces['RL'])
LA_Pos_Neg = colour_arms_graph_TC(LA_Forces_List, Pos_neg_Cmap_scale, 2, "Linear Acceleration Positive Negitive Force Visual", 'seismic')
LA_ABS = colour_arms_graph(np.absolute(LA_Forces_List), ABS_scale, 3, "Linear Acceleration Absolute Value Force Visual", 'jet')
#LA_Forces.to_excel("Force_due_to_Linear_Acceleration_of_" + str(g_lin) + "_G.xlsx")