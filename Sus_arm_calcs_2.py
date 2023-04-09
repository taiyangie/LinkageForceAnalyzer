#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  4 01:53:27 2023

@author: william_hess
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
#%% Data in, maybe need to fix pathing for git
points = pd.read_excel("Sus_Arm_points.xlsx").set_index("Points") # inboard and outboard points of the suspention 
WC = pd.read_excel("Sus_Wheel_center.xlsx").set_index("Points") # points for wheel centers of each wheel
TP = pd.read_excel("Sus_Center_of_tire_patch.xlsx").set_index("Points") #Tire Patch Center data
LandB_var = pd.read_excel("variables.xlsx") # list of variables for Linear and Breaking forces
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
tire_names = ["FR", "FL", "RR", "RL"] # List of the names of tires based on lication
arm_names = ["TR", "LCAF", "LCAR", "UCAF", "UCAR", "PR"] #Name of arms without the Wheel position
point_list = list(points.index.values) # list of index of the points data
Wb = TP["RR TP"][0] - TP["FR TP"][0] # finds Wheel Base
Tw = abs(TP["FR TP"][1]) + abs(TP["FR TP"][1]) #Finds Track Width
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
#%% Formating vector data, turns saved vectors into Dataframes
vect_frame = pd.DataFrame(vect_list).transpose().set_axis(vect_name_list)
vect_frame.columns = arm_list
WC_vect_frame = pd.DataFrame(WC_vect_list).transpose().set_axis(WC_name_list)
WC_vect_frame.columns = arm_list
unit_vect_frame = pd.DataFrame(unit_vect_list).transpose().set_axis(unit_vect_name_list)
unit_vect_frame.columns = arm_list
mag_frame = pd.DataFrame(Mag_list, index = arm_list).transpose().set_axis(["Arm_Length"])
vect_data = pd.concat([points, mag_frame, vect_frame, unit_vect_frame, WC_vect_frame])
#%% Moment EQs: 
for i in range(0, len(arm_list)): #moment arms for unit vectors to Wheel Center
    moment_calcs = [((unit_vect_frame[arm_list[i]][2])*(WC_vect_frame[arm_list[i]][1])) - ((unit_vect_frame[arm_list[i]][1])*(WC_vect_frame[arm_list[i]][2])), ((unit_vect_frame[arm_list[i]][2])*(WC_vect_frame[arm_list[i]][0])) - ((unit_vect_frame[arm_list[i]][0])*(WC_vect_frame[arm_list[i]][2])), ((unit_vect_frame[arm_list[i]][1])*(WC_vect_frame[arm_list[i]][0])) - ((unit_vect_frame[arm_list[i]][0])*(WC_vect_frame[arm_list[i]][1]))]
    moment_list.append(moment_calcs)
moment_frame = pd.DataFrame(moment_list).transpose().set_axis(moment_name_list)
moment_frame.columns = arm_list
for i in range(0, len(TP_list)): #moment arms for Tire patch to wheel center
    TP_moment_arm = [WC[WC_list[i]][0] - TP[TP_list[i]][0], WC[WC_list[i]][1] - TP[TP_list[i]][1], WC[WC_list[i]][2] - TP[TP_list[i]][2]]
    TP_moment_arm_list.append(TP_moment_arm)
TP_frame = pd.DataFrame(TP_moment_arm_list).transpose().set_axis(TP_name_list)
TP_frame.columns = TP_list
#%% Matrix Data getting [A] matrix for each of the quarter car assemblies 
matrix_data = pd.concat([unit_vect_frame, moment_frame])
FR_matrix = matrix_data.iloc[:,0:6]
FL_matrix = matrix_data.iloc[:,6:12]
RR_matrix = matrix_data.iloc[:,12:18]
RL_matrix = matrix_data.iloc[:,18:24]
A_list = [FR_matrix, FL_matrix, RR_matrix, RL_matrix] #List of quarter Car assemblies
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
WC_graphing(FR_list, "FR WC") # graphs Blue wheel center moment arms
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
plt.show(block=True) #block makes it so that the current interactive window has to close before opening a different interactive window
#%% Functions
def getInputLandB(): # only way to call Entry data is to use, this is for the Linear Acc and Breaking Calculations
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
    global paramsLandB #saves called variables to a global variable 
    paramsLandB = [We, F, R, mu, h, r, c, b, g] # populates paramsLand B

def betterGUI(frs, i): #takes variables from the LandB var and creates text and entery variables and displays units, globals allows for this list to iterable as a variable name
    ttk.Label(mainframe, text=frs["Description"][i]).grid(column=1, row=i, sticky=W)
    globals()[frs["var_name"][i] + 'V'] = DoubleVar() #makes entery a float
    Var_entry = ttk.Entry(mainframe, width=7, textvariable= globals()[frs["var_name"][i] + 'V']) #assigns iterated variable as the callable entery allowing to ger with .get() in a different function
    Var_entry.grid(column=2, row=i, sticky=(W, E))
    ttk.Label(mainframe, text=frs["unit"][i]).grid(column=3, row=i, sticky=W)
    
def ForceParameterGenerator(var, cmd): #creates GUI with the Geometry check and Calculate button able to support mutiple cases
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
   
def ABS_scale(FList, j): #Graph Scaling on an Absolute value scale
    abs_list = np.absolute(np.array(FList))
    cm_num = abs(FList[j]) / max(abs_list)
    return cm_num
    

def colour_arms_graph(Flist, Scale ,num, title, cmp): #Graphs 3d with colour map
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
    plt.close()
    plt.show(block = True)
    return fig #fig can be used in another canvas to embed into a GUI

#%% GUI Output functions
def close(): #Destroys root on command
    root.destroy()

def Output_GUI(Fig, cmd): #makes Graph in GUI output
    canvas = FigureCanvasTkAgg(Fig, master=mainframe)
    canvas.draw()
    canvas.get_tk_widget().grid(column=1, row = 1, rowspan=i, sticky = (NE, SW))
    ttk.Button(mainframe, text="Finished Visual Analysis", command=cmd).grid(column = 2, row = 2)
    for child in mainframe.winfo_children(): 
        child.grid_configure(padx=5, pady=5)
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
LA_Pos_Neg = colour_arms_graph(LA_Forces_List, Pos_neg_Cmap_scale, 2, "Linear Acceleration", 'seismic')
LA_ABS = colour_arms_graph(np.absolute(LA_Forces_List), ABS_scale, 3, "Linear Acceleration", 'jet')
LA_Forces.to_excel("Force_due_to_Linear_Acceleration_of_" + str(g_lin) + "_G.xlsx")
#%% GUI output
root = Tk()  
root.title("LINEAR ACCELERATION, Positive and Negitive Forces Visual")
mainframe = ttk.Frame(root, padding="2 4 12 12")
mainframe.grid(column=0, row=0, sticky=(N))
root.columnconfigure(0, weight=1)
root.rowconfigure(0, weight=1)
Output_GUI(LA_Pos_Neg, close)
root.bind("<Return>", getInputLandB) 
root.mainloop()

root = Tk()  
root.title("LINEAR ACCELERATION, Absolute Value Visual")
mainframe = ttk.Frame(root, padding="2 4 12 12")
mainframe.grid(column=0, row=0, sticky=(N))
root.columnconfigure(0, weight=1)
root.rowconfigure(0, weight=1)
Output_GUI(LA_ABS, close)
root.bind("<Return>", getInputLandB) 
root.mainloop()
















