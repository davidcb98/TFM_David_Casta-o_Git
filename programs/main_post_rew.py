#!/usr/bin/env python3

# -*- coding: utf-8 -*-
import numpy as np
import apfel as ap
import matplotlib.pyplot as plt
import os #os.system("pause")
from os import listdir
from os.path import isfile, join

import My_Functions as mf
import sys
import shutil as sh
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
'''
print(files_lhec_160)
print(files_lhec_760)
print(files_fcc_720)
print(files_fcc_5060)
input('Press enter to continue')
'''
files_names = [files_lhec_160, files_lhec_760, files_fcc_720, files_fcc_5060]
files_names_all   = ['LHeC_160', 'LHeC_760','FCC_720','FCC_5060']

Q2_data_all             = []
x_data_all              = []
y_data_all              = []

sigm_r_ones_all         = []
s_sigm_r_ones_all       = []

sigm_r_new_all          = []
s_sigm_r_new_all        = []

sigm_rep_menor_chi2_all = []
sigm_r_APFEL_all        = []

sigm_r_data_all         = []
s_sigm_r_data_all       = []

index_Q2_all            = []

chi2_all                = []
chi2_pesados_all        = []

Neff_P_all              = []  # Cada lista son N_copias listas con N_eig valores
random_list_all         = []

for i in range(len(files_names)):
    data = mf.ff_read_output_rew(join(Dir_Data_Folders[0],files_names[i][0]))

    Q2_data_all.append(data[0])
    x_data_all.append(data[1])
    y_data_all.append(data[2])
    
    sigm_r_ones_all.append(data[3])
    s_sigm_r_ones_all.append(data[4])
   
    sigm_r_new_all.append(data[5])
    s_sigm_r_new_all.append(data[6])
    
    sigm_rep_menor_chi2_all.append(data[7])
    sigm_r_APFEL_all.append(data[8])

    index_Q2_all.append(mf.ff_index_Qs(data[0]))

    sigm_r_data_all.append(np.loadtxt(join(Dir_Data_Folders[1],files_names[i][1]),dtype = float).tolist())
    
    s_sigm_r_data_all.append(np.loadtxt(join(Dir_Data_Folders[2],files_names[i][2]),dtype = float).tolist())
    
    chi2_all.append(np.loadtxt(join(Dir_Data_Folders[3],files_names[i][3]),dtype = float)[0,:].tolist())
    chi2_pesados_all.append(np.loadtxt(join(Dir_Data_Folders[3],files_names[i][3]),dtype = float)[1,:].tolist())
    random_list_all.append(np.loadtxt(join(Dir_Data_Folders[3],files_names[i][3]),dtype = float)[2:,:].T.tolist())
    
    Neff_P_all.append(np.loadtxt(join(Dir_Data_Folders[4],files_names[i][4]), dtype = float).tolist())



files_names_data = []
files_names_data.append(np.loadtxt('Output_rew/files_names/'+'files_lhec_160', dtype = str).tolist())
files_names_data.append(np.loadtxt('Output_rew/files_names/'+'files_lhec_760', dtype = str).tolist())
files_names_data.append(np.loadtxt('Output_rew/files_names/'+'files_fcc_160', dtype = str).tolist())
files_names_data.append(np.loadtxt('Output_rew/files_names/'+'files_fcc_5060', dtype = str).tolist())



print(files_names_data)
for i in range(len(sigm_r_data_all)):
    if type(sigm_r_data_all[i][0]) != list:
        sigm_r_data_all[i] = [sigm_r_data_all[i]]
        s_sigm_r_data_all[i] = [s_sigm_r_data_all[i]]
        
        files_names_data[i] = [files_names_data[i]]
        


#===============================================================================
# Directory to save plots
#===============================================================================

if os.path.exists('Output_rew_graf'):
    print('You are going to delate the directory "Output_rew_graf" ')
    Yes_or_No = input('Are you sure? (Type Y to continue) ')
    if Yes_or_No == 'Y':
        sh.rmtree('Output_rew_graf')
        print('The directory "Output_rew_graf" was deleted')
    else:
        sys.exit('Error, you decided not to delete the directory')

os.makedirs('Output_rew_graf')
print('The directory "Output_rew_graf" was created')

#===============================================================================
# plots
#===============================================================================

j = 0
colors = ['b','g','c','m','y','r','k'] # k, r
for k in range(len(files_names_all)):
    which_file = k
    print('\n ====================================')
    print(' Data File: ')
    print(files_names_all[which_file])
    print('Neff = ', Neff_P_all[which_file][0], ' P = ', Neff_P_all[which_file][1])

    # print(sigm_r_APFEL_all[which_file])
    # input('Press enter to continue')

    for i in range(1,len(index_Q2_all[which_file])-1):
        Q2 = Q2_data_all[which_file][index_Q2_all[which_file][i]]

        # Plots
        plt.figure(j)

        fig, ax = plt.subplots()

        ax.plot(x_data_all[which_file][index_Q2_all[which_file][i]:index_Q2_all[which_file][i+1]],
            sigm_r_APFEL_all[which_file][0][index_Q2_all[which_file][i]:index_Q2_all[which_file][i+1]],
            'x-m', markersize = 10,
            label = r'APFEL $\sigma_0$')
        
        ax.errorbar(x_data_all[which_file][index_Q2_all[which_file][i]:index_Q2_all[which_file][i+1]],
            sigm_r_ones_all[which_file][index_Q2_all[which_file][i]:index_Q2_all[which_file][i+1]],
            yerr = s_sigm_r_ones_all[which_file][index_Q2_all[which_file][i]:index_Q2_all[which_file][i+1]],
            fmt = '.-r', markersize = 10,
            label = r'$\omega_k = 1$')
        '''
        plt.plot(x_data_all[which_file][index_Q2_all[which_file][i]:index_Q2_all[which_file][i+1]],
            sigm_rep_menor_chi2_all[which_file][index_Q2_all[which_file][i]:index_Q2_all[which_file][i+1]],
            'x-m', markersize = 10,
             label = r'Mayor $\chi^2$')
        '''

        ax.errorbar(x_data_all[which_file][index_Q2_all[which_file][i]:index_Q2_all[which_file][i+1]],
            sigm_r_new_all[which_file][index_Q2_all[which_file][i]:index_Q2_all[which_file][i+1]],
            yerr = s_sigm_r_new_all[which_file][index_Q2_all[which_file][i]:index_Q2_all[which_file][i+1]],
            fmt = '.-k', markersize = 10,
            label = r'Reweighting')

        
        '''    
        plt.plot(x_data_all[which_file][index_Q2_all[which_file][i]:index_Q2_all[which_file][i+1]],
            sigm_r_new_all[which_file][index_Q2_all[which_file][i]:index_Q2_all[which_file][i+1]],
            '.k', 
            label = 'reweighting')
        '''
        for l in range(len(sigm_r_data_all[which_file])):
            ax.errorbar(x_data_all[which_file][index_Q2_all[which_file][i]:index_Q2_all[which_file][i+1]],
                sigm_r_data_all[which_file][l][index_Q2_all[which_file][i]:index_Q2_all[which_file][i+1]],
                yerr = s_sigm_r_data_all[which_file][l][index_Q2_all[which_file][i]:index_Q2_all[which_file][i+1]],
                fmt = '.-'+colors[l],markersize = 10,
                label = files_names_data[k][l])
            

        # plt.title(str(files_names_all[which_file][:-4]))

        plt.xscale('log')
        plt.legend(loc = 'best', fontsize = 16)
        plt.title(r'$Q^2$ = ' + str(Q2) + r' GeV', fontsize = 17)
        plt.xlabel(r'x',fontsize = 18)
        plt.ylabel(r'$\sigma_r$', fontsize = 19)
        plt.xticks(fontsize = 19)        
               
        if files_names_all[which_file] == 'LHeC_160' and Q2 == 500:
            print('yep')

            plt.xscale('log')
            ax.xaxis.set_major_locator(plt.MaxNLocator(4))
            plt.xticks(fontsize = 19)
            #plt.xticks(fontsize = 40)
        
        plt.yticks(fontsize = 19)
       
        plt.tight_layout()
        plt.savefig('Output_rew_graf/Fig_'+str(files_names_all[which_file])+'_Q2_'+str(int(Q2)))
        plt.close(j) 
        j += 1


for i in range(len(chi2_all)):
    print('Max chi2 pesado: ', max(chi2_pesados_all[i]))
    plt.figure(j)
    fig, ax = plt.subplots()
    plt.title(str(files_names_all[i]))
    plt.hist(chi2_all[i],bins = 100, label = r'Sin pesar')
    plt.hist(chi2_pesados_all[i], bins = 10, label = r'Pesados')
    plt.xlabel(r'$\chi^2$',fontsize = 17)
    plt.yscale('log')
    plt.legend(loc = 'best', fontsize = 16)
    plt.tight_layout()

    plt.savefig('Output_rew_graf/chi2_'+str(files_names_all[i]))
    plt.close(j)
    j +=1

#========================================================================================
# Datos salida
#========================================================================================

file_out = open('Output_rew_graf/Datos_out.txt', 'w',encoding="utf-8")

file_out.write('{0:^13} {1:^13} {2:^13} {3:^13} {4:^13} \n'
        .format('Files', 'Neff','Penalty',r'min($\chi^2$)',r'max($\chi^2_{weig}$)'))

for i in range(len(files_names_all)):
    file_out.write('{0:13} {1:13.6f} {2:13.6f} {3:13.6f} {4:13.6f}  \n'
            .format(files_names_all[i], Neff_P_all[i][0], Neff_P_all[i][1],min(chi2_all[i]),max(chi2_pesados_all[i])))
    # fsalida.write('%d  %10.4f\n' % (i, np.exp(i)))
                
file_out.close()

