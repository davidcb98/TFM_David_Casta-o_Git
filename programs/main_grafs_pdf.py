#!/usr/bin/env python3

# -*- coding: utf-8 -*-
import numpy as np
#import apfel as ap
import matplotlib.pyplot as plt
import os #os.system("pause")
from os import listdir
from os.path import isfile, join
import sys

import shutil as sh
import My_Functions as mf

import lhapdf as lha


#=======================================================================================
#=========================   We read the neccesary files   =============================
#=======================================================================================

# we read the data

Dir_Py_File  = os.path.dirname(os.path.realpath('__file__'))

Dir_Data_Folders = [join(Dir_Py_File,'Output_rew/Q2-x-y-SigmNew-SigmRep/'), 
                    join(Dir_Py_File,'Output_rew/SigmData/'), 
                    join(Dir_Py_File,'Output_rew/s_SigmData/'), 
                    join(Dir_Py_File,'Output_rew/chi2_randoms'),
                    join(Dir_Py_File,'Output_rew/Neff_P')]

files_names_SigmNew    = [f for f in listdir(Dir_Data_Folders[0]) if isfile(join(Dir_Data_Folders[0], f))]
files_names_SigmData   = [f for f in listdir(Dir_Data_Folders[1]) if isfile(join(Dir_Data_Folders[1], f))]
files_names_s_SigmData = [f for f in listdir(Dir_Data_Folders[2]) if isfile(join(Dir_Data_Folders[2], f))]
files_names_chi2       = [f for f in listdir(Dir_Data_Folders[3]) if isfile(join(Dir_Data_Folders[3], f))]
files_names_Neff_P     = [f for f in listdir(Dir_Data_Folders[4]) if isfile(join(Dir_Data_Folders[4], f))]

files_names = [files_names_SigmNew, files_names_SigmData, files_names_s_SigmData, files_names_chi2, files_names_Neff_P]


files_lhec_160 = []
files_lhec_760 = []
files_fcc_720  = []
files_fcc_5060 = []

for i in range(len(files_names)):
    for j in range(len(files_names[i])):
        if files_names[i][j][:4] == 'LHeC':
            if files_names[i][j][5] == '1':
                files_lhec_160.append(files_names[i][j])
            else:
                files_lhec_760.append(files_names[i][j])
        else:
            if files_names[i][j][4] == '7':
                files_fcc_720.append(files_names[i][j])
            else:
                files_fcc_5060.append(files_names[i][j])

files_names = [files_lhec_160, files_lhec_760, files_fcc_720, files_fcc_5060]
files_names_all   = ['LHeC_160', 'LHeC_760','FCC_720','FCC_5060']

Q2_data_all             = []
x_data_all              = []
# y_data_all              = []

# sigm_r_ones_all         = []
# s_sigm_r_ones_all       = []

# sigm_r_new_all          = []
# s_sigm_r_new_all        = []

# sigm_rep_menor_chi2_all = []
# sigm_r_APFEL_all        = []

# sigm_r_data_all         = []
# s_sigm_r_data_all       = []

index_Q2_all            = []

chi2_all                = []
# chi2_pesados_all        = []

# Neff_P_all              = []  # Cada lista son N_copias listas con N_eig valores
random_list_all         = []

for i in range(len(files_names)):
    data = mf.ff_read_output_rew(join(Dir_Data_Folders[0],files_names[i][0]))

    Q2_data_all.append(data[0])
    x_data_all.append(data[1])
    
    index_Q2_all.append(mf.ff_index_Qs(data[0]))
    random_list_all.append(np.loadtxt(join(Dir_Data_Folders[3],files_names[i][3]),dtype = float)[2:,:].T.tolist())
    
    chi2_all.append(np.loadtxt(join(Dir_Data_Folders[3],files_names[i][3]),dtype = float)[0,:].tolist())

    # y_data_all.append(data[2])
    # sigm_r_ones_all.append(data[3])
    # s_sigm_r_ones_all.append(data[4])
   
    # sigm_r_new_all.append(data[5])
    # s_sigm_r_new_all.append(data[6])
    
    #sigm_rep_menor_chi2_all.append(data[7])
    # sigm_r_APFEL_all.append(data[8])

    # sigm_r_data_all.append(np.loadtxt(join(Dir_Data_Folders[1],files_names[i][1]),dtype = float).tolist())
    # s_sigm_r_data_all.append(np.loadtxt(join(Dir_Data_Folders[2],files_names[i][2]),dtype = float).tolist())
    
    # chi2_pesados_all.append(np.loadtxt(join(Dir_Data_Folders[3],files_names[i][3]),dtype = float)[1,:].tolist())
    
    # Neff_P_all.append(np.loadtxt(join(Dir_Data_Folders[4],files_names[i][4]), dtype = float).tolist())



# files_names_data = []
# files_names_data.append(np.loadtxt('Output_rew/files_names/'+'files_lhec_160', dtype = str).tolist())
# files_names_data.append(np.loadtxt('Output_rew/files_names/'+'files_lhec_760', dtype = str).tolist())
# files_names_data.append(np.loadtxt('Output_rew/files_names/'+'files_fcc_160', dtype = str).tolist())
# files_names_data.append(np.loadtxt('Output_rew/files_names/'+'files_fcc_5060', dtype = str).tolist())
# 
# 
# 
# print(files_names_data)
# for i in range(len(sigm_r_data_all)):
#     if type(sigm_r_data_all[i][0]) != list:
#         sigm_r_data_all[i] = [sigm_r_data_all[i]]
#         s_sigm_r_data_all[i] = [s_sigm_r_data_all[i]]
#         
#         files_names_data[i] = [files_names_data[i]]
#         


#===============================================================================
# Directory to save plots
#===============================================================================
folder_out = 'Output_grafs_pdfs'

if os.path.exists(folder_out):
    print('You are going to delate the directory ',folder_out)
    Yes_or_No = input('Are you sure? (Type Y to continue) ')
    if Yes_or_No == 'Y':
        sh.rmtree(folder_out)
        print('The directory ',folder_out, ' was deleted')
    else:
        sys.exit('Error, you decided not to delete the directory')

os.makedirs(folder_out)
print('The directory ', folder_out, ' was created')
print('================================= \n')

#===============================================================================
# LHAPDF
#===============================================================================

# print(lha.availablePDFSets()) # See the avaliable pdfSets

print(lha.getPDFSet("PDF4LHC21_40").description)
print('================================= \n ')

pdfsets = lha.mkPDFs("PDF4LHC21_40") #  we do the calls with pdfsets[i].xfxQ2(flavour, x, Q^2)
print('================================= \n')

# print(pdfsets[0].xfxQ2(1,1e-5,5))
'''
Note: xfxQ2(flavour, x, Q^2)

For PDF4LHC21 we have:
    1   2   3   4   5   21
    d   u   s   c   b   g
the anti-flavours with the - (-1, -2,...)
'''

omega_k_ones_all = []
omega_k_all      = []

N_rep_LHC21 = 41
d_chi2      = 10
N_copias    = 10000

flavs = [1, -1, 21]
flavs_labels = [r'u', r'$\bar{u}$', r'g']


# Compute the value of xf(x,Q^2) for each flavour with the 41 replicas os PDF4LHC21

xf_ones_all = [] # xf_ones_all[flav][file]
xf_new_all  = [] # xf_new_all[flav][file]

print('Computing xf(x,Q^2)')
for flav in flavs:
    
    xf_ones_flav_all = []
    xf_new_flav_all      = []
    for i in range(len(files_names_all)):
        print('Flavour: ',flav,'|| File: ', files_names_all[i])
        xf_ones, xf_new = mf.ff_rew_xf(Q2_data_all[i], x_data_all[i], pdfsets, flav, random_list_all[i], chi2_all[i] , d_chi2, N_copias)
        
        xf_ones_flav_all.append(xf_ones)
        xf_new_flav_all.append(xf_new)

    xf_ones_all.append(xf_ones_flav_all)
    xf_new_all.append(xf_new_flav_all)

print('================================= ')

#===============================================================================
# plots
#===============================================================================

j = 0
colors = ['r','k','b','g','c','m','y'] # k, r
for k in range(len(files_names_all)):
    which_file = k
    print('\n====================================')
    print(' Data File: ')
    print(files_names_all[which_file])
    # print('Neff = ', Neff_P_all[which_file][0], ' P = ', Neff_P_all[which_file][1])

    for i in range(1,len(index_Q2_all[which_file])-1):
        Q2 = Q2_data_all[which_file][index_Q2_all[which_file][i]]

        # Plots
        plt.figure(j)

        fig, ax = plt.subplots()
    
        for l in range(len(flavs)):
            ax.plot(x_data_all[which_file][index_Q2_all[which_file][i]:index_Q2_all[which_file][i+1]],
                xf_new_all[l][which_file][index_Q2_all[which_file][i]:index_Q2_all[which_file][i+1]],
                '.-'+colors[l], markersize = 10,
                label = flavs_labels[l])
        
        for l in range(len(flavs)):
            ax.plot(x_data_all[which_file][index_Q2_all[which_file][i]:index_Q2_all[which_file][i+1]],
                xf_ones_all[l][which_file][index_Q2_all[which_file][i]:index_Q2_all[which_file][i+1]],
                'x-'+colors[len(flavs) + l], markersize = 10,
                label = r'$\omega_k = 1$, '+ flavs_labels[l])

        plt.xscale('log')
        plt.legend(loc = 'best', fontsize = 16)
        plt.title(r'$Q^2$ = ' + str(Q2) + r' GeV', fontsize = 17)
        plt.xlabel(r'x',fontsize = 18)
        plt.ylabel(r'$xf(x,Q^2)$', fontsize = 19)
        plt.xticks(fontsize = 19)        
               
        if files_names_all[which_file] == 'LHeC_160' and Q2 == 500:
            # print('yep')

            plt.xscale('log')
            ax.xaxis.set_major_locator(plt.MaxNLocator(4))
            plt.xticks(fontsize = 19)
            #plt.xticks(fontsize = 40)
        
        plt.yticks(fontsize = 19)
       
        plt.tight_layout()
        plt.savefig(folder_out+'/Fig_'+str(files_names_all[which_file])+'_Q2_'+str(int(Q2)))
        plt.close(j) 
        j += 1


