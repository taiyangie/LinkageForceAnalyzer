#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 15 22:38:51 2023

@author: william_hess

better visuals than Test 2, should be final version, gives report of cases at the end
"""
import pandas as pd
from tkinter import *
from tkinter import ttk
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk)
import matplotlib as mpl
#%% Retainer variables
Mag_list = [] # holding list for magnitude fectore
vect_list = [] #Vector holding list
unit_vect_list =[] #unit vector holding list
WC_vect_list = [] # Wheel Center holding list
moment_list = [] #moment holding list
TP_moment_arm_list = [] #moment arms for Tire patch to wheel center holding list
PCR_list = []
Fmax_list = []
Tube_mass_list = []
#%% Data in, maybe need to fix pathing for git
points = pd.read_excel("Sus_Arm_points.xlsx").set_index("Points") # inboard and outboard points of the suspention 
WC = pd.read_excel("Sus_Wheel_center.xlsx").set_index("Points") # points for wheel centers of each wheel
TP = pd.read_excel("Sus_Center_of_tire_patch.xlsx").set_index("Points") #Tire Patch Center data
LandB_var = pd.read_excel("variables.xlsx") # list of variables for Linear and Breaking forces
WD = pd.read_excel("Sus_Wheel_center.xlsx")
SSC_var = pd.read_excel("Variable_SSC.xlsx")
Arm_mat = pd.read_excel("Arm_Materials.xlsx").set_index("Points")
#%% Data Lists
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
#%% Analysis vectors and unit vectors
for i in range(0, len(arm_list)): #iterates over all suspention arms
    vect = [points[arm_list[i]][point_list[3]] - points[arm_list[i]][point_list[0]], points[arm_list[i]][point_list[4]] - points[arm_list[i]][point_list[1]], points[arm_list[i]][point_list[5]] - points[arm_list[i]][point_list[2]]] # calculates vector from the outboard of the car to the inboard
    mag = np.sqrt(np.square(vect[0]) + np.square(vect[1]) + np.square(vect[2])) # Finds magnitude of this vector
    unit_vect = vect/mag # makes unit vector
    vect_list.append(vect) # adds vector to list
    Mag_list.append(mag) # adds magnitude vector to list
    unit_vect_list.append(unit_vect) # adds unit vector to list
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
#%% Compression Calculations
for i in range(len(arm_list)): 
    P = ((np.pi ** 2) * (Arm_mat[arm_list[i]][5]) * (Arm_mat[arm_list[i]][4])) / (((Arm_mat[arm_list[i]][8]) * (mag_frame[arm_list[i]][0])) ** 2)
    Fmax = (Arm_mat[arm_list[i]][7]) * (Arm_mat[arm_list[i]][3])
    Tube_mass = (mag_frame[arm_list[i]][0]) * (Arm_mat[arm_list[i]][3]) * (Arm_mat[arm_list[i]][9])
    PCR_list.append(P)
    Fmax_list.append(Fmax)
    Tube_mass_list.append(Tube_mass)
PCR_frame = pd.DataFrame(PCR_list, index = arm_list).transpose().set_axis(["Critical Load (Buckling)"])    
Fmax_frame = pd.DataFrame(Fmax_list, index = arm_list).transpose().set_axis(["Max Force (Tension)"])
TMass_frame = pd.DataFrame(Tube_mass_list, index = arm_list).transpose().set_axis(["Tube Mass (lbs)"])
Fos_data = pd.concat([PCR_frame, Fmax_frame])
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
fig1.set_facecolor('grey')
ax.set_facecolor('grey')
plt.close()
plt.show(block=True)
#%% Functions
def getInputLandB():
    We = WeV.get() #Vehicle weight
    F = FV.get() #Front weight distribution 
    R = RV.get() # Rear weight distribution
    mu = muV.get() # tire to road maximum coef of friction
    h = hV.get() # center of gravity height
    r = rV.get() #
    g = gV.get()
    Dx = DxV.get()
    G_bump = G_bumpV.get()
    Version_number = Version_numberV.get()
    root.destroy()
    global paramsLandB
    paramsLandB = [We, F, R, mu, h, r, g, Dx, G_bump, Version_number]
    
def getInputSSC():
    V = VV.get() #Velocity
    T_rad = T_radV.get() #Turn Radii
    Zrf = ZrfV.get() # Front Roll Center Height
    Zrr = ZrrV.get() # Rear Roll center Height
    K_f = K_fV.get() # Front Roll Rate
    K_r = K_rV.get() #Rear Roll Rate
    Theta_bank = Theta_bankV.get()
    root.destroy()
    global paramsSSC
    paramsSSC = [V, T_rad, Zrf, Zrr, K_f, K_r, Theta_bank]

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

def Fos_maker(FList, name): #Makes an FOS, if force is positive / compression uses buckling equation, if negitive
    Fos = []
    for i in range(0, len(arm_list)): 
        if FList[arm_list[i]][0] > 0: #positive / Compression
            Fos_num = ((Fos_data[arm_list[i]][0]) / Arm_mat[arm_list[i]][3]) / (abs(FList[arm_list[i]][0]) / Arm_mat[arm_list[i]][3])
            Fos.append(Fos_num)
        elif FList[arm_list[i]][0] < 0: #negitive / Tension
            Fos_num = (Fos_data[arm_list[i]][1]) / (abs(FList[arm_list[i]][0]))
            Fos.append(Fos_num)
    Fos_out = pd.DataFrame(Fos, index = arm_list).transpose().set_axis([name]) #Dataframe made of X matrixes, gives all forces in all arms in all tires and names index to arm names
    return Fos_out #returns this matrix

#%% Graph functions

def Pos_neg_Cmap_scale(FList, j): #scaling for diverging colormap
    if FList[j] > 0: #positive
       cm_num = (FList[j] / max(FList)) / 2
       return cm_num + 0.5
    elif FList[j] < 0: 
       cm_num = (FList[j] / min(FList)) / 2
       return 0.5 - cm_num

def FOS_scale(Flist, j):
    if Flist[j] >= 10: 
        cm_num = 0.99
        return cm_num
    else: 
        cm_num2 = Flist[j] / 10
        return cm_num2
 

def colour_arms_graph_TC(Flist, Scale ,num, title, cmp):
    fig = plt.figure(num = num, clear = True)
    ax = fig.add_subplot(1,1,1, projection='3d')
    cmap = mpl.colormaps[cmp]
    norm = mpl.colors.Normalize(vmin = min(Flist), vmax = max(Flist))
    fig.set_facecolor('grey')
    ax.set_facecolor('grey')
    cbar = fig.colorbar(mpl.cm.ScalarMappable(norm = norm, cmap = cmap), ax = ax, label = "<-- Extention | Compression -->", ticks = [min(Flist), (min(Flist) + ((min(Flist) + max(Flist))/2)) / 2 , ((min(Flist) + max(Flist))/2), (max(Flist) + ((min(Flist) + max(Flist))/2)) / 2, max(Flist)])
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

def colour_arms_graph_FOS(Flist, Scale ,num, title, cmp):
    fig = plt.figure(num = num, clear = True)
    ax = fig.add_subplot(1,1,1, projection='3d')
    cmap = mpl.colormaps[cmp]
    norm = mpl.colors.Normalize(vmin = 0, vmax = 10)
    fig.set_facecolor('grey')
    ax.set_facecolor('grey')
    cbar = fig.colorbar(mpl.cm.ScalarMappable(norm = norm, cmap = cmap), ax = ax, label = "Factor of Safety", ticks = [0, min(Flist), 10])
    cbar.ax.set_yticklabels(['0', "min:" + str(round(min(Flist), 1)), "10+ FOS"])
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
[We, F, R, mu, h, r, g_lin, Dx, G_bump, Version_number] = paramsLandB #links used varibales from GUI inputs to math for forces
Wfs = We*(F/100) #Front static weight
Wrs = We*(R/100) #Rear stati wright
Wb = TP["RR TP"][0] - TP["FR TP"][0] # finds Wheel Base
Twf = abs(WC["FR WC"][1]) + abs(WC["FL WC"][1]) #Finds front Track Width (Front wheel center to front wheel center)
Twr = abs(WC["RR WC"][1]) + abs(WC["RL WC"][1]) #Finds Rear Track Width (Front wheel Center to front wheel center)
c = (F/100) * Wb #Cg to rear axle
b = Wb - c #Cg to Front axle
faxle_z = WC["FR WC"][2]
raxle_z = WC["RR WC"][2]
#%% Calculating Linear Acc Forces at contact patch
FR_LA = [0, 0, ((We * ((c/Wb) - g_lin*(h/Wb))) / 2)] #XYZ order for forces in each arm
FL_LA = [0, 0, ((We * ((c/Wb) - g_lin*(h/Wb))) / 2)]
RR_LA = [((((mu*We*b)/Wb) / (1-(h/Wb)*mu)) / 2), 0, ((We * ((b/Wb) + g_lin*(h/Wb))) / 2)]
RL_LA = [((((mu*We*b)/Wb) / (1-(h/Wb)*mu)) / 2), 0, ((We * ((b/Wb) + g_lin*(h/Wb))) / 2)]
Force_LA = [FR_LA, FL_LA, RR_LA, RL_LA] # List of Force lists for each tire
LA_col = ['FR_LA', 'FL_LA', 'RR_LA', 'RL_LA'] #List to help call columns for math
LA_Forces = WForce_to_AForce(Force_LA, LA_name_list, LA_col, TP_frame, A_list) #Wheel Force to arm Force
LA_Forces_List = list(LA_Forces['FR']) + list(LA_Forces['FL']) + list(LA_Forces['RR']) + list(LA_Forces['RL'])
LA_Forces_Frame = pd.DataFrame(LA_Forces_List, index = arm_list).transpose().set_axis(["Linear Acceleration of " + str(g_lin) +"G (lbf)"])
LA_Fos = Fos_maker(LA_Forces_Frame, "Linear Acceleration of " + str(g_lin) +" G (FOS)")
#%% Linear acceleration forces at Arms Viaual
LA_Pos_Neg = colour_arms_graph_TC(LA_Forces_List, Pos_neg_Cmap_scale, 2, "Linear Acceleration of " + str(g_lin) + "G Positive Negitive Force Visual", 'seismic')
LA_FOS_Test = colour_arms_graph_FOS(list(LA_Fos.transpose()[LA_Fos.index.values[0]]), FOS_scale, 3, "Linear Acceleration of " + str(g_lin) + "G: FOS Plot", 'jet_r')
#%% Calculating Breaking forces
FR_BR = [(mu * (Wfs + ((We*Dx*h)/Wb))) / 2, 0, (Wfs + ((We*Dx*h)/Wb)) / 2] #XYZ order
FL_BR = [(mu * (Wfs + ((We*Dx*h)/Wb))) / 2, 0, (Wfs + ((We*Dx*h)/Wb)) / 2]
RR_BR = [(mu * (Wrs - ((We*Dx*h)/Wb))) / 2, 0, (Wrs - ((We*Dx*h)/Wb)) / 2]
RL_BR = [(mu * (Wrs - ((We*Dx*h)/Wb))) / 2, 0, (Wrs - ((We*Dx*h)/Wb)) / 2]
Force_BR = [FR_BR, FL_BR, RR_BR, RL_BR]
BR_col = ['FR_BR', 'FL_BR', 'RR_BR', 'RL_BR']
BR_Forces = WForce_to_AForce(Force_BR, BR_name_list, BR_col, TP_frame, A_list)
BR_Forces_List = list(BR_Forces['FR']) + list(BR_Forces['FL']) + list(BR_Forces['RR']) + list(BR_Forces['RL'])
BR_Forces_Frame = pd.DataFrame(BR_Forces_List, index = arm_list).transpose().set_axis(["Breaking at " + str(Dx) + "G (lbf)"])
BR_Fos = Fos_maker(BR_Forces_Frame, "Breaking at" + str(Dx) + "G (FOS)")
#%% Breaking Visualisation 
BR_Pos_Neg = colour_arms_graph_TC(BR_Forces_List, Pos_neg_Cmap_scale, 4, "Breaking Deceleration of " + str(Dx) + "G Positive Negitive Force Visual", 'seismic')
BR_FOS = colour_arms_graph_FOS(list(BR_Fos.transpose()[BR_Fos.index.values[0]]), FOS_scale, 3, "Breaking Deceleration of " + str(Dx) + "G: FOS Plot", 'jet_r')
#%% Bump case
FR_B = [0, 0, G_bump * (Wfs/2)]
FL_B = [0, 0, G_bump * (Wfs/2)]
RR_B = [0, 0, G_bump * (Wrs/2)]
RL_B = [0, 0, G_bump * (Wrs/2)]
Force_B = [FR_B, FL_B, RR_B, RL_B]
B_col = ['FR_B', 'FL_B', 'RR_L', 'RL_B']
B_Forces = WForce_to_AForce(Force_B, B_name_list, B_col, TP_frame, A_list)
B_Forces_List = list(B_Forces['FR']) + list(B_Forces['FL']) + list(B_Forces['RR']) + list(B_Forces['RL'])
B_Forces_Frame = pd.DataFrame(B_Forces_List, index = arm_list).transpose().set_axis(["Bump at " + str(G_bump) + "G (lbf)"])
B_Fos = Fos_maker(B_Forces_Frame, "Bump at" + str(G_bump) + "G (FOS)")
#%% Bump Graphs
B_Pos_Neg = colour_arms_graph_TC(B_Forces_List, Pos_neg_Cmap_scale, 4, "Bump of " + str(G_bump) + "G Positive Negitive Force Visual", 'seismic')
B_FOS = colour_arms_graph_FOS(list(B_Fos.transpose()[B_Fos.index.values[0]]), FOS_scale, 3, "Bump of " + str(G_bump) + "G: FOS Plot", 'jet_r')
#%% Steady State Cornering GUI
root = Tk()  
root.title("SSC parameters")
mainframe = ttk.Frame(root, padding="2 4 12 12")
mainframe.grid(column=0, row=0, sticky=(N, W, E, S))
root.columnconfigure(0, weight=1)
root.rowconfigure(0, weight=1)
ForceParameterGenerator(SSC_var, getInputSSC)
root.bind("<Return>", getInputSSC) 
root.mainloop()
#%% Inputs and calculating other constraints
[V, T_rad, Zrf, Zrr, K_f, K_r, Theta_bank] = paramsSSC
gravity = 9.80665
lateral_accel_horizontal = (V**2) / (T_rad * gravity)
lateral_accel = lateral_accel_horizontal * np.cos(Theta_bank) - np.sin(Theta_bank) # in g
effective_weight = We * (lateral_accel_horizontal * np.sin(Theta_bank) + np.cos(Theta_bank)) # in lbs
effective_axle_mass_f = effective_weight * c / Wb # kg
effective_axle_mass_r = effective_weight * b / Wb # kg
slope = (Zrr - Zrf) / Wb
height_under_cg = Zrf + (slope * b)
cg_to_rollaxis = h - height_under_cg
roll_gradient = -We * gravity * cg_to_rollaxis / (K_f + K_r)
f_lateral_load_transfer = lateral_accel * (We / Twf) * ((cg_to_rollaxis * K_f) / (K_f + K_r)) + (c / Wb) * Zrf
r_lateral_load_transfer = lateral_accel * (We / Twr) * ((cg_to_rollaxis * K_r) / (K_f + K_r)) + (b / Wb) * Zrr
fr_static_wt = Wfs/2
fl_static_wt = Wfs/2
rr_static_wt = Wrs/2
rl_static_wt = Wrs/2
SSC_Fzfr = fr_static_wt - f_lateral_load_transfer
SSC_Fzfl = fl_static_wt + f_lateral_load_transfer
SSC_Fzrr = rr_static_wt - r_lateral_load_transfer
SSC_Fzrl = rl_static_wt + r_lateral_load_transfer
#%% Steady State Cornering
FR_SSC = [0, lateral_accel * SSC_Fzfr, SSC_Fzfr] #XYZ
FL_SSC = [0, lateral_accel * SSC_Fzfl, SSC_Fzfl]
RR_SSC = [0, lateral_accel * SSC_Fzrr, SSC_Fzrr]
RL_SSC = [0, lateral_accel * SSC_Fzrl, SSC_Fzrl]
Force_SSC = [FR_SSC, FL_SSC, RR_SSC, RL_SSC]
SSC_col = ['FR_SSC', 'FL_SSC', 'RR_SSC', 'RL_SSC']
SSC_Forces = WForce_to_AForce(Force_SSC, SSC_name_list, SSC_col, TP_frame, A_list)
SSC_Forces_List = list(SSC_Forces['FR']) + list(SSC_Forces['FL']) + list(SSC_Forces['RR']) + list(SSC_Forces['RL'])
SSC_Forces_Frame = pd.DataFrame(SSC_Forces_List, index = arm_list).transpose().set_axis(["Steady State Cornering at " + str(round(lateral_accel, 1)) + " G (lbf)"])
SSC_Fos = Fos_maker(SSC_Forces_Frame, "Steady State Cornering at " + str(round(lateral_accel, 1)) + "G (FOS)")
#%% Graphing SSC
SSC_Pos_Neg = colour_arms_graph_TC(SSC_Forces_List, Pos_neg_Cmap_scale, 4, "Steady State Cornering at " + str(round(lateral_accel, 1)) + "G (lbf)", 'seismic')
BR_FOS = colour_arms_graph_FOS(list(SSC_Fos.transpose()[SSC_Fos.index.values[0]]), FOS_scale, 3, "Steady State Cornering at" + str(round(lateral_accel, 1)) + " G (FOS)", 'jet_r')
#%% Run imputs
param_names = pd.concat([LandB_var["Description"], SSC_var["Description"]])
units_names = pd.concat([LandB_var["unit"], SSC_var["unit"]])
var_val_list = paramsLandB + paramsSSC #+other load cases when they become available
var_value_frame = pd.DataFrame(var_val_list, index = list(param_names.index.values))
var_value_frame.columns = ["Value"]
var_value = pd.concat([param_names, var_value_frame, units_names], axis=1)
#%% other params
Backend_variable = [Wfs, Wrs, Wb, Twf, Twr, c, b]  
Backend_Description = ["Front static weight", "Rear static weight", "Wheel Base",  "Front Track Width", "Rear Track Width",  "Center of Gravity to Rear Axle",  "Center of Gravity to Front Axle"]
Backend_units = ["lbs", "lbs", "in", "in", "in", "in", "in"]
Backend_frame = pd.DataFrame([Backend_Description, Backend_variable, Backend_units]).transpose()
Backend_frame.columns = (list(var_value))
#%% End Spreadsheets: Gives a final report in a spreadsheet

Arm_info = pd.concat([mag_frame, Arm_mat, TMass_frame, Fos_data]) # gives all information on arm material, lengths, and critical loads
Overview_Frame = pd.concat([LA_Forces_Frame, LA_Fos, BR_Forces_Frame, BR_Fos, B_Forces_Frame, B_Fos, SSC_Forces_Frame, SSC_Fos])
input_info = pd.concat([var_value, Backend_frame]).set_index("Description")

with pd.ExcelWriter('Summary_report_' + str(Version_number) +'.xlsx') as writer:
    Arm_info.to_excel(writer, sheet_name='Arm_setup_information')
    Overview_Frame.to_excel(writer, sheet_name='Cases_and_FOS')
    input_info.to_excel(writer, sheet_name = "Model Input Values")
    points.to_excel(writer, sheet_name = "Model Suspension Points")
    WC.to_excel(writer, sheet_name = "Model Wheel Center Points")
    TP.to_excel(writer, sheet_name = "Model Tire Patch Center Points")





