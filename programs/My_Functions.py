#!/usr/bin/env python3

# -*- coding: utf-8 -*-

import numpy as np
import apfel as ap
import random as rd

from os.path import isfile, join
#==========================================================================
# Functions to read archives
#==========================================================================

def ff_read_datlhec(f_name):
    ''' Funtion to read archives datlhec
        Returns: f_DataInfo_labels, f_DataInfo, f_Q2_data, 
                 f_x_data, f_y_data, f_F2_data, f_sigrNC_data, f_etot_data
    '''

    # We read the parameters -> (sqrt(s), e charge, reduced, e polatity)
    f_DataInfo_labels = ['sqrt(s)','e charge','reduced','e polarity']
    
    f_data = open(f_name, 'r')
    
    f_lines = f_data.readlines() # Leemos el fichero línea a línea
    f_DataInfo = f_lines[9].split(',')[:4]
    f_DataInfo[0] = f_DataInfo[0][12:]
    f_DataInfo = np.array(f_DataInfo, dtype = float)
   
    ''' To know where we star to read the data, we search the * '''
    f_i = 0
    for f_line in f_lines:
        if f_line.split()[0] == '*':
            f_i+=1
            break
        f_i+=1
    f_data.close()
        
    # we read the data
    f_data = np.loadtxt(f_name,dtype = str,skiprows = f_i)
    ''' *  q2    x    y    thetae    thetaj    f2    sigrNC    nevent    estat    eunco
       esyst    etot    eelen    ethee    ehadr    radco    egamp    effic    enois '''
    
    # Bucle para cargarse el punto del final de la columna de Q^2
    f_Q2_list = []
    for i in range(len(f_data)):
        f_str_aux = ''
        for j in range(len(f_data[i,0])-1):
            f_str_aux += f_data[i,0][j]
        f_Q2_list.append(f_str_aux)
    
    f_Q2_data     = np.array(f_Q2_list,   dtype = float)
    f_x_data      = np.array(f_data[:,1],dtype = float)
    f_y_data      = np.array(f_data[:,2],dtype = float)
    f_F2_data     = np.array(f_data[:,5],dtype = float)
    f_sigrNC_data = np.array(f_data[:,6],dtype = float)
    f_etot_data   = np.array(f_data[:,11],dtype = float) ## Incertidumbre de las sigma
    return f_DataInfo_labels, f_DataInfo, f_Q2_data, f_x_data, f_y_data, f_F2_data, f_sigrNC_data, f_etot_data

def ff_read_datfcc(f_name):
    ''' Funtion to read archives datfcc
        Returns: f_DataInfo_labels, f_DataInfo, f_Q2_data, 
                 f_x_data, f_y_data, f_F2_data, f_sigrNC_data, f_etot_data
    '''
    # We read the parameters -> (sqrt(s), e charge, reduced, e polatity)

    f_data = open(f_name, 'r')
    
    f_lines = f_data.readlines() # Leemos el fichero línea a línea
    f_DataInfo_labels = f_lines[0].split()
    f_DataInfo = np.array(f_lines[1].split(), dtype = float)
    
    
    f_data.close()

    # we read the data
    f_data = np.loadtxt(f_name,dtype = str,skiprows = 5)
    ''' *  q2    x    y    thetae    thetaj    f2    sigrNC    nevent    estat    eunco
       esyst    etot    eelen    ethee    ehadr    radco    egamp    effic    enois '''
    
    # Bucle para cargarse el punto del final de la columna de Q^2
    f_Q2_list = []
    for i in range(len(f_data)):
        f_str_aux = ''
        for j in range(len(f_data[i,0])-1):
            f_str_aux += f_data[i,0][j]
        f_Q2_list.append(f_str_aux)
    
    f_Q2_data     = np.array(f_Q2_list,   dtype = float)
    f_x_data      = np.array(f_data[:,1],dtype = float)
    f_y_data      = np.array(f_data[:,2],dtype = float)
    f_F2_data     = np.array(f_data[:,5],dtype = float)
    f_sigrNC_data = np.array(f_data[:,6],dtype = float)
    f_etot_data   = np.array(f_data[:,11],dtype = float) ## Incertidumbre de las sigma

    return f_DataInfo_labels, f_DataInfo, f_Q2_data, f_x_data, f_y_data, f_F2_data, f_sigrNC_data, f_etot_data


# Archives imput rew (outputs of APFEL)

def ff_read_output_APFEL_aux(file_name):
    f_data = np.loadtxt(file_name,dtype = float)

    f_Q2_data      = f_data[0,:].tolist()
    f_x_data       = f_data[1,:].tolist()
    f_y_data       = f_data[2,:].tolist()
    f_sigm_r_data  = f_data[3,:]
    f_etot_data    = f_data[4,:]
    f_sigm_r_APFEL = f_data[5:,:].T
    
    f_s_sigm_r_data = f_sigm_r_data*f_etot_data/100
    return f_Q2_data, f_x_data, f_y_data, f_sigm_r_data.tolist(), f_s_sigm_r_data.tolist(), f_sigm_r_APFEL.tolist()


def ff_read_output_APFEL(Dir_Data_Folder,files_names):
    ''' Read the output of the program main_APFEL.py, that is, the imput of main_rew.py '''
    Q2_data_all       = []
    x_data_all        = []
    y_data_all        = []
    sigm_r_data_all   = []
    s_sigm_r_data_all = []
    sigm_r_APFEL_all  = []

    for i in range(len(files_names)):

        Q2_data, x_data, y_data, sigm_r_data, s_sigm_r_data, sigm_r_APFEL = ff_read_output_APFEL_aux(join(Dir_Data_Folder,files_names[i]))

        Q2_data_all.append(Q2_data)
        x_data_all.append(x_data)
        y_data_all.append(y_data)
        sigm_r_data_all.append(sigm_r_data)
        s_sigm_r_data_all.append(s_sigm_r_data)
        sigm_r_APFEL_all.append(sigm_r_APFEL)
    
        if i >0:
            assert Q2_data_all[i-1] == Q2_data_all[i]
            assert x_data_all[i-1] == x_data_all[i]
            assert y_data_all[i-1] == y_data_all[i]
            assert sigm_r_APFEL_all[i-1] == sigm_r_APFEL_all[i]

    return Q2_data_all[0], x_data_all[0], y_data_all[0], sigm_r_data_all, s_sigm_r_data_all, sigm_r_APFEL_all[0]

# Read outputs reweighting

def ff_read_output_rew(file_name):
    ''' Read the output of the program main_rew.py, that is, the imput of main_post_rew.py '''
    f_data = np.loadtxt(file_name,dtype = float)

    f_Q2_data             = f_data[0,:].tolist()
    f_x_data              = f_data[1,:].tolist()
    f_y_data              = f_data[2,:].tolist()
    
    f_sigm_r_ones         = f_data[3,:].tolist()
    f_s_sigm_r_ones       = f_data[4,:].tolist()
    
    f_sigm_r_rew          = f_data[5,:].tolist()
    f_s_sigm_r_new        = f_data[6,:].tolist()
    
    f_sigm_rep_menor_chi2 = f_data[7,:].tolist()
    f_sigm_r_APFEL        = f_data[8:,:].tolist()
    # print(len(f_sigm_r_APFEL))
 
    return [f_Q2_data, f_x_data, f_y_data,f_sigm_r_ones, f_s_sigm_r_ones, f_sigm_r_rew, f_s_sigm_r_new, f_sigm_rep_menor_chi2, f_sigm_r_APFEL]





#==========================================================================
# Functions to APFEL calculos
#==========================================================================

def ff_sigma_r_NC(f_x,f_y):
    ''' This function returns the REDUCED cross section in DIS
        It calculates the cross section fron the PDFs F_2 and F_L
        It is necessary to import APFEL as ap '''
    # print(f_x,f_y,ap.F2total(f_x),ap.FLtotal(f_x))
    return ap.F2total(f_x) - f_y**2*ap.FLtotal(f_x)/(1+(1-f_y))**2

def ff_sigma_NC(f_x,f_y,f_Q2,f_alpha):
    ''' This function returns the TOTAL cross section in DIS.
        It calculates the cross section fron the PDFs F_2 and F_L
        It is necessary to import APFEL as ap '''

    f_k = ((2*np.pi*f_alpha**2*(1+(1-f_y))**2)/(f_Q2**2*f_x))
    return ff_sigma_r_NC(f_x,f_y) * f_k

def ff_index_Qs(f_Q2_data):
    ''' This function finds the index in which we change from one Q to
        another.
        Note: our array of Q's has the form [2,2,2,2,5,5,5,5,5,...]
    '''
    f_aux = f_Q2_data[0]
    f_index_Q2 =[0]
    for i in range(len(f_Q2_data)):
        if f_Q2_data[i] != f_aux:
            f_aux = f_Q2_data[i]
            f_index_Q2.append(i)

    f_index_Q2.append(len(f_Q2_data))

    return f_index_Q2

def ff_cut_small_x(f_x_data, f_y_data, f_sigrNC_data, f_etot_data, f_index_Q2, f_x_min, f_i):
    ''' We select the values of x > x_min '''
    f_x = []
    f_y = []
    f_sigm = []
    f_etot = []
    for k in range(f_index_Q2[f_i+1] - f_index_Q2[f_i]):

        if f_x_data[f_index_Q2[f_i] + k] > f_x_min:
            f_x.append(f_x_data[f_index_Q2[f_i] + k])
            f_y.append(f_y_data[f_index_Q2[f_i] + k])
            f_sigm.append(f_sigrNC_data[f_index_Q2[f_i] + k])
            f_etot.append(f_etot_data[f_index_Q2[f_i] + k])
    return f_x, f_y, f_sigm, f_etot

def ff_apfel_evol(f_PDFSet,f_Q,f_Q_0,f_x_min,f_rep_i):
    ''' Do the APFEL evolution
    '''
    
    ap.SetPDFSet(f_PDFSet) # XMin: 0.101563E-05 , XMax: 0.100000E+01
    ap.SetProcessDIS('NC')
    ap.SetQLimits(f_Q_0,250) # El default es Q_0 = 0.5 GeV
        
    ap.SetReplica(f_rep_i)

    ap.SetNumberOfGrids(3)
    ap.SetGridParameters(1, 100, 3, f_x_min)
    ap.SetGridParameters(2, 50, 5, 1e-1)
    ap.SetGridParameters(3, 40, 5, 8e-1)
    
    ap.InitializeAPFEL_DIS()
    ap.ComputeStructureFunctionsAPFEL(f_Q_0,f_Q)
        
    return 


def ff_sigr_apfel_reps(f_x_data, f_y_data, f_Q2_data, f_sigrNC_data, f_etot_data, f_index_Q2, f_file_name, f_PDFSet, f_x_min,f_Q_0,f_replicas):
    '''
    f_x_data = x_data_all[which_file]
    f_y_data = y_data_all[which_file]
    f_Q2_data =  Q2_data_all[which_file]
    f_index_Q2 = index_Q2_all[which_file]
    f_file_name = files_names_all[which_file]
    f_sigrNC_data = sigrNC_data_all[which_file]
    f_etot_data = etot_data_all[which_file]
    '''
    f_sigm_r_APFEL = []
    f_Q2_list      = []
    f_x_list       = []
    f_y_list       = []
    f_sigrNC_list  = []
    f_etot_list    = []

    for i in range(f_replicas):
        f_sigm_r_APFEL.append([])

    for i in range(1,len(f_index_Q2)-1):
    
        # print(Q2_data_fcc[which_file][index_Q2_fcc[which_file][i]:index_Q2_fcc[which_file][i+1]])
        f_Q2 = f_Q2_data[f_index_Q2[i]]
        f_Q = np.sqrt(f_Q2) # GeV
   
        ## PDFSet 1

        f_x, f_y, f_sigrNC, f_etot = ff_cut_small_x(f_x_data, f_y_data,f_sigrNC_data, f_etot_data, f_index_Q2, f_x_min, i)
                
        for l in range(f_replicas):
            print(f_PDFSet)
            print('Escala final: Q = ',f_Q, ' ,Q2 = ', f_Q2)
            print('Replica: ', l)
            ff_apfel_evol(f_PDFSet,f_Q,f_Q_0,f_x_min,l)    
            f_sigm_r_APFEL[l] += [ff_sigma_r_NC(f_x[j],f_y[j]) for j in range(len(f_x))]

        f_x_list += f_x
        f_y_list += f_y
        f_Q2_list += [f_Q2 for m in range(len(f_x))]
        f_sigrNC_list += f_sigrNC
        f_etot_list += f_etot
    
    ## print(f_etot_list)
    ## Save the results in a txt
    f_sigm_r_APFEL.insert(0,f_etot_list)
    f_sigm_r_APFEL.insert(0,f_sigrNC_list)
    f_sigm_r_APFEL.insert(0,f_y_list)
    f_sigm_r_APFEL.insert(0,f_x_list)
    f_sigm_r_APFEL.insert(0,f_Q2_list)
    
    ''' OJO!!! 
        Se guarda por FILAS:
        Q2
        x
        y
        sigrNC
        etot
        sigma replica 0
        sigma replica 1
        ...... 
    '''
    np.savetxt('Output_APFEL/'+f_file_name[:-4]+'_out.txt',f_sigm_r_APFEL)
    return


#===========================================================================================
# Reweighting
#===========================================================================================


def ff_cut_x_rew(f_Q2_data, f_x_data, f_y_data, f_sigm_r_data, f_s_sigm_r_data, f_sigm_r_APFEL, f_x_min, f_x_max):
    f_Q2_data_cut       = [[],[],[],[]]
    f_x_data_cut        = [[],[],[],[]]
    f_y_data_cut        = [[],[],[],[]]
    f_sigm_r_data_cut   = [[],[],[],[]]
    f_s_sigm_r_data_cut = [[],[],[],[]]
    f_sigm_r_APFEL_cut  = [[],[],[],[]]
    
    f_sigm_r_APFEL = [np.array(f_sigm_r_APFEL[i]).T.tolist() for i in range(len(f_sigm_r_APFEL))]
    
    for i in range(len(f_x_data)):
        
        f_sigm_r_data_aux = []
        f_s_sigm_r_data_aux = []
        for k in range(len(f_sigm_r_data[i])):
            f_sigm_r_data_aux.append([])
            f_s_sigm_r_data_aux.append([])
        
        f_sigm_r_APFEL_aux = []
        for k in range(len(f_sigm_r_APFEL[i])):
            f_sigm_r_APFEL_aux.append([])


        for j in range(len(f_x_data[i])):
            if f_x_data[i][j] > f_x_min and f_x_data[i][j] < f_x_max:
    
                f_Q2_data_cut[i].append(f_Q2_data[i][j])
                f_x_data_cut[i].append(f_x_data[i][j])
                f_y_data_cut[i].append(f_y_data[i][j])
                
                for k in range(len(f_sigm_r_data[i])):
                    f_sigm_r_data_aux[k].append(f_sigm_r_data[i][k][j])
                    f_s_sigm_r_data_aux[k].append(f_s_sigm_r_data[i][k][j])
                
                for k in range(len(f_sigm_r_APFEL[i])):
                    f_sigm_r_APFEL_aux[k].append(f_sigm_r_APFEL[i][k][j])
                
        for k in range(len(f_sigm_r_data[i])):
            f_sigm_r_data_cut[i].append(f_sigm_r_data_aux[k])
            f_s_sigm_r_data_cut[i].append(f_s_sigm_r_data_aux[k])
                
        for k in range(len(f_sigm_r_APFEL[i])):
            f_sigm_r_APFEL_cut[i].append(f_sigm_r_APFEL_aux[k])
        
    f_sigm_r_APFEL_cut = [np.array(f_sigm_r_APFEL_cut[i]).T.tolist() for i in range(len(f_sigm_r_APFEL_cut))]
    
    return f_Q2_data_cut, f_x_data_cut, f_y_data_cut, f_sigm_r_data_cut, f_s_sigm_r_data_cut, f_sigm_r_APFEL_cut

def ff_sigm_r_k_xQ2_aux(f_sigma, f_random_list):
    aux = 0
    j = 0
    for i in range(1,len(f_sigma),2):
        #aux += rd.gauss(0,1)*(f_sigma[i]-f_sigma[i+1])/2
        aux += f_random_list[j]*(f_sigma[i]-f_sigma[i+1])/2
        j +=1
    return aux

def ff_sigm_r_k_xQ2(f_sigm_r_APFEL_xQ2,f_random_list,f_N_copias): # f_seed
    '''
    Genera N_rep replicas de las sigma a partir de los 2N+1 valores 
    de f_sigms_APFEL_xQ2. Estos 2N+1 valores son para un punto (x,Q2).
    
    f_sigm_r_APFEL_xQ2 = [sigm_0, sigm_{+1}, sigm_{-1},...., sigm_{+N}, sigm_{-N}]
    '''
    # rd.seed(f_seed)
    f_sigm_r_k_xQ2 = [f_sigm_r_APFEL_xQ2[0] + ff_sigm_r_k_xQ2_aux(f_sigm_r_APFEL_xQ2, f_random_list[i])  for i in range(f_N_copias)]
    return f_sigm_r_k_xQ2

def ff_sigm_r_k(f_sigm_r_APFEL, f_random_list, f_N_copias):
    '''
    - Input:
        Cada elemento de f_sigms_APFEL es una lista con los 2N+1 valores de las 
        secciones eficaces para cada valor de (x,Q2), esto es:
            f_sigm_r_APFEL_xQ2 = f_sigm_r_APFEL[i]
        donde 
            f_sigm_r_APFEL_xQ2 = [sigm_0, sigm_{+1}, sigm_{-1},...., sigm_{+N}, sigm_{-N}]

    - Output:
        Cada elemento de f_sigm_r_k es una lista con f_N_rep valores (replicas) para cada valor
        de (x,Q2), es decir, cada una de estas listas es para un (x,Q2) particular
    '''
    #f_random_list = [[rd.gauss(0,1) for i in range(int((len(f_sigm_r_APFEL[0])-1)/2))] for j in range(f_N_copias)]
    f_sigm_r_k = [ff_sigm_r_k_xQ2(f_sigm_r_APFEL[i], f_random_list, f_N_copias) for i in range(len(f_sigm_r_APFEL))]
    return f_sigm_r_k #, f_random_list

def ff_chi2_rew_aux(f_sigm_r_k_i,f_sigm_exp, f_s_sigm_exp):

    assert len(f_sigm_r_k_i) == len(f_sigm_exp[0])
    
    f_chi2_aux = 0
    for j in range(len(f_sigm_exp)):
        for k in range(len(f_sigm_exp[0])):
             f_chi2_aux += (f_sigm_r_k_i[k] - f_sigm_exp[j][k])**2/f_s_sigm_exp[j][k]**2

    return f_chi2_aux

def ff_chi2_rew(f_sigm_r_k, f_sigm_exp, f_s_sigm_exp): ## f_s_sigm_exp = etot
    '''
    - Imput:
        - f_sigm_r_k es la salida de la función ff_sigm_r_k
        - f_sigm_exp son lo valores experimentales (es una lista de varias listas)
        - f_s_sigm_exp incertidumbres experimentales

    - Output:
        Esta función devuelve una lista de listas f_chi2_k con f_N_rep valores de chi2 cada una,
        uno para cada réplica

    - Nota: Recordar que cada lista de f_sigm_r_k es una lista de f_N_rep valores y tenemos una lista por
            cada valor de (x,Q2). Lo que hace esta función es calcular el chi2 para cada replica, es decir,
            cogemos el primer valor de cada lista y calculamos un chi2, cogemos el segundo y calculamos otro,..
    '''
    f_sigm_r_k_np = np.array(f_sigm_r_k)
    f_chi2_k = [ff_chi2_rew_aux(f_sigm_r_k_np[:,i], f_sigm_exp, f_s_sigm_exp) 
                for i in range(len(f_sigm_r_k[0]))]
    return f_chi2_k 

def ff_omega_k(f_chi2_k, f_d_chi2,f_N_copias):
    ''' - Output:
            f_omega_k es una lista con f_N_rep valores, uno por replica. '''
    f_aux = 0
    
    for j in range(len(f_chi2_k)):
        # f_aux += f_chi2_k[j]**((f_N_data-1)/2)*np.exp(-f_chi2_k[j]/(2*f_d_chi2), dtype = np.float128)
        f_aux += np.exp(-f_chi2_k[j]/(2*f_d_chi2), dtype = np.float128)
    # print('===',f_aux)

    # f_omega_k = [f_chi2_k[i]**((f_N_data-1)/2)*np.exp(- f_chi2_k[i]/(2*f_d_chi2), dtype = np.float128)*f_N_rep/f_aux  
    #             for i in range(len(f_chi2_k))] 
    f_omega_k = [np.exp(- f_chi2_k[i]/(2*f_d_chi2), dtype = np.float128)*f_N_copias/f_aux
                 for i in range(len(f_chi2_k))]

    return f_omega_k 
   
def ff_sigm_r_new_aux(f_sigm_r_k_i, f_omega_k, f_N_copias):
    f_sigm_r_new_i = sum(np.array(f_sigm_r_k_i)*np.array(f_omega_k))/f_N_copias
    return f_sigm_r_new_i

def ff_sigm_r_new(f_sigm_r_k, f_omega_k, f_N_copias):
    f_sigm_r_new = [ ff_sigm_r_new_aux(f_sigm_r_k[i], f_omega_k,f_N_copias) 
            for i in range(len(f_sigm_r_k))]
    return f_sigm_r_new

def ff_s_sigm_r_new_aux(f_sigm_r_k_i, f_sigm_r_new_i, f_omega_k, f_N_copias):
    #print(f_sigm_r_k_i)
    #print(f_sigm_r_new_i)
    #input('Press enter to continue')
    f_s_sigm_r_new_i = np.sqrt(sum(np.array(f_omega_k)*(np.array(f_sigm_r_k_i) - np.array([f_sigm_r_new_i for i in range(len(f_sigm_r_k_i))]))**2)/f_N_copias)
    return f_s_sigm_r_new_i

def ff_s_sigm_r_new(f_sigm_r_k,f_sigm_r_new,f_omega_k,f_N_copias):
    f_s_sigm_r_new = [ ff_s_sigm_r_new_aux(f_sigm_r_k[i],f_sigm_r_new[i] , f_omega_k,f_N_copias)
                        for i in range(len(f_sigm_r_k))]
    return f_s_sigm_r_new
    
def ff_Neff(f_omega_k,f_N_copias):
    aux = 0
    for i in range(f_N_copias):
        aux += f_omega_k[i] * np.log(f_N_copias/f_omega_k[i])
    aux = aux/f_N_copias
    return np.exp(aux)

def ff_penalty(f_omega_k, f_random_list, f_N_copias, f_d_chi2 ,f_N_eig):
    
    aux1 = 0
    for i in range(f_N_eig):
    
        aux2 = 0
        for k in range(f_N_copias):
            aux2 += f_omega_k[k] * f_random_list[k][i]
        
        aux1 += aux2**2
    
    return aux1*f_d_chi2/(f_N_copias**2)


def ff_reweighting(f_Q2_data, f_x_data, f_y_data, f_file_name, f_sigm_r_data,
                   f_s_sigm_r_data, f_sigm_r_APFEL, f_N_copias, f_d_chi2):
    '''
    sigm_r_APFEL
        Cada elemento de sigms_APFEL es una lista con los 2N+1 valores de las
        secciones eficaces para cada valor de (x,Q2), esto es:
        f_sigm_r_APFEL_xQ2 = f_sigm_r_APFEL[i]
        donde
        f_sigm_r_APFEL_xQ2 = [sigm_0, sigm_{+1}, sigm_{-1},...., sigm_{+N}, sigm_{-N}] '''
     
   
    f_random_list = [[rd.gauss(0,1) for i in range(int((len(f_sigm_r_APFEL[0])-1)/2))] for j in range(f_N_copias)]

    # f_sigm_r_k, f_random_list = ff_sigm_r_k(f_sigm_r_APFEL, f_N_copias)
    f_sigm_r_k = ff_sigm_r_k(f_sigm_r_APFEL, f_random_list ,f_N_copias)

    f_chi2_k = ff_chi2_rew(f_sigm_r_k, f_sigm_r_data, f_s_sigm_r_data)
    print('Minimo valor de chi^2: ', min(f_chi2_k))

    #f_N_data = len(f_sigm_r_data)*len(f_sigm_r_data[0])
    f_omega_k_ones = [1 for i in range(len(f_chi2_k))]
    f_omega_k = ff_omega_k(f_chi2_k, f_d_chi2, f_N_copias)

    f_sigm_r_ones = ff_sigm_r_new(f_sigm_r_k, f_omega_k_ones, f_N_copias)
    f_sigm_r_new = ff_sigm_r_new(f_sigm_r_k, f_omega_k, f_N_copias)

    print('Numero de punto (x,Q^2): ', len(f_sigm_r_new))
    print('Numero de pseudo datos: ', len(f_sigm_r_data)*len(f_sigm_r_data[0]))
    print('Pesos: min_chi2:',f_omega_k[np.argmin(f_chi2_k)], '||  max_chi2: ', f_omega_k[np.argmax(f_chi2_k)])

    f_s_sigm_r_ones = ff_s_sigm_r_new(f_sigm_r_k, f_sigm_r_ones, f_omega_k_ones, f_N_copias)
    f_s_sigm_r_new = ff_s_sigm_r_new(f_sigm_r_k, f_sigm_r_new, f_omega_k, f_N_copias)
    

    f_Neff    = ff_Neff(f_omega_k, f_N_copias)
    print('Neff = ', f_Neff)
    f_penalty = ff_penalty(f_omega_k, f_random_list, f_N_copias, f_d_chi2,int((len(f_sigm_r_APFEL[0])-1)/2))
    print('P = ', f_penalty)
    

    ## Save the results in a txt
    f_sigm_r_APFEL = (np.array(f_sigm_r_APFEL).T).tolist()

    f_sigm_r_APFEL.insert(0,np.array(f_sigm_r_k).T.tolist()[np.argmax(f_chi2_k)])
    f_sigm_r_APFEL.insert(0,f_s_sigm_r_new)
    f_sigm_r_APFEL.insert(0,f_sigm_r_new)
    f_sigm_r_APFEL.insert(0,f_s_sigm_r_ones)
    f_sigm_r_APFEL.insert(0,f_sigm_r_ones)
    f_sigm_r_APFEL.insert(0,f_y_data)
    f_sigm_r_APFEL.insert(0,f_x_data)
    f_sigm_r_APFEL.insert(0,f_Q2_data)
    
    ''' OJO!!! 
        Se guarda por FILAS, no COLUMNAS
        Q2
        x
        y
        sigm_r_new
        s_sigm_r_new
        sigm_r_APFEL[replica menos chi2]
        sigma replica 0
        sigma replica 1
        ...... 
    '''
    np.savetxt('Output_rew/Q2-x-y-SigmNew-SigmRep/'+f_file_name+'_Q2-x-y-SigmNew-SigmRep'+'_'+str(f_N_copias)+'_out.txt',f_sigm_r_APFEL)
    np.savetxt('Output_rew/SigmData/'+f_file_name+'_SigmData.txt',f_sigm_r_data)
    np.savetxt('Output_rew/s_SigmData/'+f_file_name+'_s_SigmData.txt',f_s_sigm_r_data)
    
    f_random_list = np.array(f_random_list).T.tolist()
    f_random_list.insert(0,np.array(f_chi2_k)*np.array(f_omega_k)/f_N_copias)
    f_random_list.insert(0,f_chi2_k)

    np.savetxt('Output_rew/chi2_randoms/'+f_file_name+'_chi2_randoms.txt',f_random_list)
    np.savetxt('Output_rew/Neff_P/'+f_file_name+'_Neff_P.txt',[f_Neff,f_penalty])

