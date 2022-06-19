#!/usr/bin/env python3

# -*- coding: utf-8 -*-
import numpy as np
import apfel as ap
import matplotlib.pyplot as plt
import os #os.system("pause")
from os import listdir
from os.path import isfile, join
import sys

import shutil as sh
import My_Functions as mf

#=======================================================================================
#========================= Cargamos los datos del fichero ==============================
#=======================================================================================

# Dir_Py_File  = os.path.dirname(os.path.realpath('__file__'))


Path_Py_File   = os.path.dirname(os.path.abspath(__file__))
Path_Prev_Dir  = os.path.sep.join(Path_Py_File.split(os.path.sep)[:-1])
# print(Path_Prev_Dir)
# input('Press enter to continue')
Dir_Data_Folder_fcc = join(Path_Prev_Dir,'data/datfcc')
Dir_Data_Folder_lhec = join(Path_Prev_Dir,'data/datlhec')

files_names_fcc = [f for f in listdir(Dir_Data_Folder_fcc) if isfile(join(Dir_Data_Folder_fcc, f))]
files_names_lhec = [f for f in listdir(Dir_Data_Folder_lhec) if isfile(join(Dir_Data_Folder_lhec, f))]
files_names_all = files_names_fcc + files_names_lhec
 
## we load the data

DataInfo_labels_all = []  #  [Ep, Ee, eCharge, ePol, Lumifb, lhapdf, NC, CC, itarg, erunco]
DataInfo_all      = []  
Q2_data_all       = []
x_data_all        = []
y_data_all        = []
F2_data_all       = []
sigrNC_data_all   = []
etot_data_all     = []
index_Q2_all      = []

for i in range(len(files_names_fcc)):
    
    # Read the data documents
    DataInfo_labels, DataInfo, Q2_data, x_data, y_data, F2_data, sigrNC_data, etot_data = mf.ff_read_datfcc(join(Dir_Data_Folder_fcc,files_names_fcc[i]))
    
    # Save the data readed
    DataInfo_labels_all.append(DataInfo_labels)
    DataInfo_all.append(DataInfo)
    Q2_data_all.append(Q2_data)
    x_data_all.append(x_data)
    y_data_all.append(y_data)
    F2_data_all.append(F2_data)
    sigrNC_data_all.append(sigrNC_data)
    etot_data_all.append(etot_data)

    # Find the index where we change the value os Q2
    index_Q2_all.append(mf.ff_index_Qs(Q2_data))


for i in range(len(files_names_lhec)):
    
    # Read the data documents
    DataInfo_labels, DataInfo, Q2_data, x_data, y_data, F2_data, sigrNC_data, etot_data = mf.ff_read_datlhec(join(Dir_Data_Folder_lhec,files_names_lhec[i]))
    
    # Save the data readed
    DataInfo_labels_all.append(DataInfo_labels)
    DataInfo_all.append(DataInfo)
    Q2_data_all.append(Q2_data)
    x_data_all.append(x_data)
    y_data_all.append(y_data)
    F2_data_all.append(F2_data)
    sigrNC_data_all.append(sigrNC_data)
    etot_data_all.append(etot_data)

    # Find the index where we change the value os Q2
    index_Q2_all.append(mf.ff_index_Qs(Q2_data))

print('Read succesfull')
# input('Press enter to continue')

## Output Folder
if os.path.exists('Output_APFEL'):
    print('You are going to delate the directory "Output_APFEL" ')
    Yes_or_No = input('Are you sure? (Type Y to continue) ')
    if Yes_or_No == 'Y':
        sh.rmtree('Output_APFEL')
        print('The directory "Output_APFEL" was deleted')
    else:
        sys.exit('Error, you decided not to delete the directory')

os.makedirs('Output_APFEL')



# Parameter from the bibliografi of the PDF set
PDFSet = 'PDF4LHC21_40'
x_min_LHC21 = 0.101563e-05
Q_0_LHC21   = 0.140010e+01
replicas    = 41

for k in range(len(files_names_all)): #[value_imput]:#range(len(files_names_all)):
    which_file = k
    print(' Data File: ')
    print(files_names_all[which_file])
    
    x_data = x_data_all[which_file]
    y_data = y_data_all[which_file]
    Q2_data =  Q2_data_all[which_file]
    index_Q2 = index_Q2_all[which_file]
    file_name = files_names_all[which_file]
    sigrNC_data = sigrNC_data_all[which_file]
    etot_data = etot_data_all[which_file]

    mf.ff_sigr_apfel_reps(x_data_all[which_file]  , y_data_all[which_file]   , Q2_data_all[which_file],
                       sigrNC_data_all[which_file], etot_data_all[which_file], index_Q2_all[which_file],
                       files_names_all[which_file], PDFSet                  , x_min_LHC21,
                       Q_0_LHC21                  , replicas)




# os.system("shutdown.exe /h") # Hiberna el ordenador al acabar
