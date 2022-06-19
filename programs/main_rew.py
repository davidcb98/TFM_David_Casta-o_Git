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
#============================  Read the necessary files  ===============================
#=======================================================================================

Dir_Py_File  = os.path.dirname(os.path.realpath('__file__'))

Dir_Data_Folder = join(Dir_Py_File,'Output_APFEL/')

files_names_all = [f for f in listdir(Dir_Data_Folder) if isfile(join(Dir_Data_Folder, f))]

files_lhec_160 = []
files_lhec_760 = []
files_fcc_720  = []
files_fcc_5060 = []

#print(files_names_all)
#for i in range(len(files_names_all)):
#    print(files_names_all[i][len(files_names_all[i])-14:len(files_names_all[i])-8])
#input()


for i in range(len(files_names_all)):
    if files_names_all[i][:7] == 'datlhec':
        if files_names_all[i][len(files_names_all[i])-14:len(files_names_all[i])-8] == 'random':
            if files_names_all[i][7] == '1':
                files_lhec_160.append(files_names_all[i])
            else:
                files_lhec_760.append(files_names_all[i])
    else:
        if files_names_all[i][len(files_names_all[i])-14:len(files_names_all[i])-8] == 'random':
            if files_names_all[i][6] == '7':
                files_fcc_720.append(files_names_all[i])
            else:
                files_fcc_5060.append(files_names_all[i])

# print(files_fcc_5060)
# input()

Q2_data_lhec_160, x_data_lhec_160, y_data_lhec_160, sigm_r_data_lhec_160, s_sigm_r_data_lhec_160, sigm_r_APFEL_lhec_160  = mf.ff_read_output_APFEL(Dir_Data_Folder,files_lhec_160)

Q2_data_lhec_760, x_data_lhec_760, y_data_lhec_760, sigm_r_data_lhec_760, s_sigm_r_data_lhec_760, sigm_r_APFEL_lhec_760  = mf.ff_read_output_APFEL(Dir_Data_Folder,files_lhec_760)

Q2_data_fcc_720, x_data_fcc_720, y_data_fcc_720, sigm_r_data_fcc_720, s_sigm_r_data_fcc_720, sigm_r_APFEL_fcc_720  = mf.ff_read_output_APFEL(Dir_Data_Folder,files_fcc_720)

Q2_data_fcc_5060, x_data_fcc_5060, y_data_fcc_5060, sigm_r_data_fcc_5060, s_sigm_r_data_fcc_5060, sigm_r_APFEL_fcc_5060  = mf.ff_read_output_APFEL(Dir_Data_Folder,files_fcc_5060)

Q2_data       = [Q2_data_lhec_160      , Q2_data_lhec_760      , Q2_data_fcc_720      , Q2_data_fcc_5060]
x_data        = [x_data_lhec_160       , x_data_lhec_760       , x_data_fcc_720       , x_data_fcc_5060]
y_data        = [y_data_lhec_160       , y_data_lhec_760       , y_data_fcc_720       , y_data_fcc_5060]
sigm_r_data   = [sigm_r_data_lhec_160  , sigm_r_data_lhec_760  , sigm_r_data_fcc_720  , sigm_r_data_fcc_5060]
s_sigm_r_data = [s_sigm_r_data_lhec_160, s_sigm_r_data_lhec_760, s_sigm_r_data_fcc_720, s_sigm_r_data_fcc_5060]
sigm_r_APFEL  = [sigm_r_APFEL_lhec_160 , sigm_r_APFEL_lhec_760 , sigm_r_APFEL_fcc_720 , sigm_r_APFEL_fcc_5060]
files_names_aux = ['LHeC_160', 'LHeC_760', 'FCC_720','FCC_5060']

'''
sigm_r_APFEL
    Cada elemento de sigms_APFEL es una lista con los 2N+1 valores de las
    secciones eficaces para cada valor de (x,Q2), esto es:
    f_sigms_APFEL_xQ2 = f_sigms_APFEL[i]
    donde
    f_sigms_APFEL_xQ2 = [sigm_0, sigm_{+1}, sigm_{-1},...., sigm_{+N}, sigm_{-N}] '''
    
#=======================================================================================
#========================= Select the range os values of x  ============================
#=======================================================================================

# All vaules: x_min = 1e-10, x_max =1

# x_min = 1e-10
# x_max = 1

x_min = float(input('x_min = '))
x_max = float(input('x_max = '))

Q2_data_cut, x_data_cut, y_data_cut, sigm_r_data_cut, s_sigm_r_data_cut, sigm_r_APFEL_cut = mf.ff_cut_x_rew(Q2_data, x_data, y_data, sigm_r_data, s_sigm_r_data, sigm_r_APFEL, x_min, x_max)

if os.path.exists('Output_rew'):
    print('You are going to delate the directory "Output_rew" ')
    Yes_or_No = input('Are you sure? (Type Y to continue) ')
    if Yes_or_No == 'Y':
        sh.rmtree('Output_rew')
        print('The directory "Output_rew" was deleted')
    else:
        sys.exit('Error, you decided not to delete the directory')

os.makedirs('Output_rew')
os.makedirs('Output_rew/Q2-x-y-SigmNew-SigmRep')
os.makedirs('Output_rew/SigmData')
os.makedirs('Output_rew/files_names')
os.makedirs('Output_rew/s_SigmData')
os.makedirs('Output_rew/chi2_randoms')
os.makedirs('Output_rew/Neff_P')
print('The directory "Output_rew" was created')


N_copias = 10000
d_chi2 = 10       # Tolerance

for i in range(len(files_names_aux)):
    print('\n')
    print('=========================================')
    print(files_names_aux[i])
    mf.ff_reweighting(Q2_data_cut[i]    , x_data_cut[i]       , y_data_cut[i]      , files_names_aux[i],
                      sigm_r_data_cut[i], s_sigm_r_data_cut[i], sigm_r_APFEL_cut[i], N_copias , d_chi2)

np.savetxt('Output_rew/files_names/'+'files_lhec_160',files_lhec_160, fmt="%s")
np.savetxt('Output_rew/files_names/'+'files_lhec_760',files_lhec_760, fmt="%s")
np.savetxt('Output_rew/files_names/'+'files_fcc_160',files_fcc_720, fmt="%s")
np.savetxt('Output_rew/files_names/'+'files_fcc_5060',files_fcc_5060, fmt="%s")

