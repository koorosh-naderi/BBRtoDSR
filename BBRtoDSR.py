# -*- coding: utf-8 -*-
"""
Created on Tue Apr 29 16:42:12 2025

@author: NADERIK1
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import minimize
from statistics import linear_regression
import math
#import os
#import csv

# Function to create plots
def create_plot(data):
    fig, ax = plt.subplots()
    ax.plot(data.iloc[:, 0], data.iloc[:, -3],label='Estimated', linestyle='-')
    ax.plot(data.iloc[:, 0], data.iloc[:, -6],label='Measured', linestyle=':', marker='o')
    ax.set_title('Plot of Stiffness vs Time')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Stiffness (MPa)')
    ax.set_xscale('log')
    ax.set_yscale('log')
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels)
    return fig

def find_bracketing_rows(df, column_name, target_value):
    # Sort the DataFrame by the specified column
    df_sorted = df.sort_values(by=column_name)
    
    # Initialize variables for bracketing
    lower_row = None
    upper_row = None

    # Iterate through the sorted DataFrame to find bracketing rows
    for index, row in df_sorted.iterrows():
        value = row[column_name]
        if value < target_value:
            lower_row = row
        elif value > target_value and upper_row is None:
            upper_row = row
            break

    # If both lower and upper rows are found, create a new DataFrame
    if lower_row is not None and upper_row is not None:
        return pd.DataFrame([lower_row, upper_row])
    else:
        # If no bracketing rows found, find the two closest rows
        df_sorted['distance'] = abs(df_sorted[column_name] - target_value)
        closest_rows = df_sorted.nsmallest(2, 'distance')
        closest_rows = closest_rows.drop(columns='distance')  # Drop the distance column if you don't want it
        return closest_rows

def stiffness(T1):
    list1 = allresults[allresults['Temperature (C)']==T1].iloc[:,1:4]
    mat = np.array([
                    [1,np.log10(8),(np.log10(8))**2],
                    [1,np.log10(15),(np.log10(15))**2],
                    [1,np.log10(30),(np.log10(30))**2],
                    [1,np.log10(60),(np.log10(60))**2],
                    [1,np.log10(120),(np.log10(120))**2],
                    [1,np.log10(240),(np.log10(240))**2]]).T
    list3 = 10**(list1@mat)
    return list3

def function(T1,T2,a):
    list1 = allresults[allresults['Temperature (C)']==T1].iloc[:,1:4]
    list2 = allresults[allresults['Temperature (C)']==T2].iloc[:,1:4]

    mat = np.array([
                    [1,np.log10(8),(np.log10(8))**2],
                    [1,np.log10(15),(np.log10(15))**2],
                    [1,np.log10(30),(np.log10(30))**2],
                    [1,np.log10(60),(np.log10(60))**2],
                    [1,np.log10(120),(np.log10(120))**2],
                    [1,np.log10(240),(np.log10(240))**2]]).T

    list3 = 10**(list1@mat)
    list4 = 10**(list2@mat)

    
    logt = [np.log10(8),np.log10(15),np.log10(30),np.log10(60),np.log10(120),np.log10(240)]
    logtReduced = logt + [-a+x for x in logt]
    logS = list(np.log10(list3.iloc[0,:]))
    logS.extend(list(np.log10(list4.iloc[0,:])))
    slope3, intercept3, r_value3, p_value3, std_err3 = stats.linregress(logtReduced, logS)
    return 1-abs(r_value3)
            
def function_to_minimize(x):
    return function(fixed_T1, fixed_T2, x[0])

def gpl_minimize(args):
    m, logD0, logD1 = args
    creep_comp_calc = (10**logD0)+(10**logD1)*reduced_time**m
    return sum((np.log10(creep_compliance) - np.log10(creep_comp_calc))**2)

def ca_minimize(args):
    beta, logOmegaC = args
    G_calc_CA = 1000*(1+(10**logOmegaC/reduced_omega)**beta)**(-1/beta)
    return sum((np.log10(G_calc_CA) - np.log10(dynamic_shear_modulus))**2)

def T_fatigue_minimize(T):
    omega_red_T_fatigue = 10*10**(slope4*(1/(T+273.15)-1/(273.15+allresults['Temperature (C)'][0])))
    phase_fatigue = 90/(1+(omega_red_T_fatigue/(10**result_CA.x[1]))**result_CA.x[0])
    G_fatigue = 1000*1000*(1+(10**result_CA.x[1]/omega_red_T_fatigue)**result_CA.x[0])**(-1/result_CA.x[0])
    return (5000-G_fatigue*np.sin(np.radians(phase_fatigue)))**2

# Streamlit app layout
st.title("BBR Data Processor (alpha release)")
st.image("BBRtoDSR.jpeg")
st.write("© 2025 [Koorosh Naderi](https://www.linkedin.com/in/koorosh-naderi/)")
st.write("A minimum of two CSV files is required for analysis.")

# Session state to track uploaded files
if 'uploaded_files' not in st.session_state:
    st.session_state.uploaded_files = []

# File uploader
uploaded_files = st.file_uploader("Choose CSV files", accept_multiple_files=True, type='csv')

st.write("Please exercise caution when using the results, as this method attempts to extrapolate the BBR results beyond the ranges measured by the device. Additionally, several assumptions have been made, including Arrhenius-type temperature dependence, the validity of Generalized Power Law behavior for creep compliance at low temperatures, the incompressibility of binder (with a Poisson's ratio of 0.5), and the applicability of the CA model for complex modulus and phase angle master curves. These assumptions may significantly deviate from the true behavior of the material.")

# If new files are uploaded, clear previous analysis
if uploaded_files:
    st.session_state.uploaded_files = uploaded_files

# Button to clear analysis
if st.button("Clear Analysis"):
    st.session_state.uploaded_files = []
    st.write("Analysis has been cleared.")

allresults = pd.DataFrame(columns=['Temperature (C)','A','B','C','S(60)','m-value(60)'])

# Perform analysis if there are uploaded files
if st.session_state.uploaded_files:
    dataframes = []
    for uploaded_file in uploaded_files:
        
        df = pd.read_csv(uploaded_file,
                header=None,engine='python',names=range(1,6))
        
        info = df.iloc[0:9,0:2]
        data = df.iloc[9:,:].dropna(axis=1)
        data.columns = data.iloc[0]
        data = data[1:]
        data.reset_index(drop=True,inplace=True)
        data = data.rename_axis(None, axis=1)
        BeamSpan = np.float64(info[2][6])/1000
        BeamWidth = np.float64(info[2][7])/1000
        BeamThickness = np.float64(info[2][8])/1000
        data['Stiffness (MPa)'] = 1/1000000*(np.float64(data['Force (mN)'])*BeamSpan**3)/(np.float64(data['Deflection (mm)'])*4*BeamWidth*BeamThickness**3)
        data['log(t)'] = np.log10(np.float64(data['Time (s)']))
        data['log(S)'] = np.log10(data['Stiffness (MPa)'])
        
        results = data[data['Time (s)'].isin(['8','15','30','60','120','240'])]
        model = np.poly1d(np.polyfit(results['log(t)'], results['log(S)'], 2))
        
        data['Sc (MPa)'] = 10**model(np.float64(data['log(t)']))
        data['Percent diff'] = (data['Stiffness (MPa)']-data['Sc (MPa)'])/data['Stiffness (MPa)']*100
        data['m-value'] = abs(2*model.coefficients[0]*data['log(t)']+model.coefficients[1])
        
        
        
        results = data[data['Time (s)'].isin(['8','15','30','60','120','240'])]
        
        
        
        # Display the uploaded dataframe
        st.write(f"**Data from {uploaded_file.name}:**")
        st.dataframe(results)
        
        temperature = np.float64(info[2][4])
        
        if abs(np.float64(results['Temperature (C)']).mean()-np.float64(info[2][4]))>0.1 :
            print(f"Temperature Control was not correct and the value considered is the test temperature not the intended temperature of {info[2][4]}.")
            temperature = np.float64(results['Temperature (C)']).mean()
        
        allresults.loc[len(allresults)] = [temperature,
                                           model.coefficients[2],
                                           model.coefficients[1],
                                           model.coefficients[0],
                                           10**(model(np.log10(60))),
                                           abs(2*model.coefficients[0]*np.log10(60)+model.coefficients[1])]
        
        
        # Calculate and display the mean of the first column
        #mean_value = df.iloc[:, 0].mean()
        #st.write(f"Mean of first column: {mean_value}")

        #Create and display plot
        fig = create_plot(results)
        st.pyplot(fig)
        


     

    #
if st.button("Print Results"):
    st.write("Results printed to console!")
    allresults.sort_values('Temperature (C)',
                                        axis=0,
                                        ascending=False,inplace=True)
    allresults.reset_index(drop=True, inplace=True)
    st.dataframe(allresults)
    
    
    
    if len(allresults)>=2:
            
            ddf1 = find_bracketing_rows(allresults,'m-value(60)',0.3)
            ddf2 = find_bracketing_rows(allresults,'S(60)',300)
            slope1, intercept1, r_value1, p_value1, std_err1 = stats.linregress(ddf1['m-value(60)'], ddf1['Temperature (C)'])
            slope2, intercept2, r_value2, p_value2, std_err2 = stats.linregress(np.log(ddf2['S(60)']), ddf2['Temperature (C)'])
            T_s = round(-10+slope2*np.log(300)+intercept2,1)
            T_m = round(-10+slope1*0.3+intercept1,1)
            Delta_Tc = round(T_s - T_m,1)
            st.write(f"**$T_{{{'c,S'}}}$: {T_s} °C**")
            st.write(f"**$T_{{{'c,m'}}}$: {T_m} °C**")
            st.write(f"**$Delta T_{'c'}$: {Delta_Tc} °C**")
            st.write(f"**Reference Temperature, $T_{{{'ref'}}}$: {allresults['Temperature (C)'][0]} °C**")           
                      
            
            a_T_list = []
            Temperature_list = [allresults['Temperature (C)'][0]]
            reduced_time_list = [8,15,30,60,120,240]
            stiffness_list = list(stiffness(allresults['Temperature (C)'][0]).iloc[0,:])
            
            fig1, ax1 = plt.subplots()
            
            ax1.plot([8,15,30,60,120,240], stiffness(allresults['Temperature (C)'][0]).iloc[0,:],label=allresults['Temperature (C)'][0], 
                     linestyle='-',
                     marker='o')
            
            for i in range(1,len(allresults)):
                fixed_T1 = allresults['Temperature (C)'][i-1]
                fixed_T2 = allresults['Temperature (C)'][i]
                initial_x = [np.log10(7200/60)*(1/(fixed_T2+273.15)-1/(fixed_T1+273.15))/(1/(-10+fixed_T1+273.15)-1/(fixed_T1+273.15))]
                result = minimize(function_to_minimize, initial_x)
                
                
                a_T_list.append(result.x[0])
                Temperature_list.append(allresults['Temperature (C)'][i])
                reduced_time_list.extend([8,15,30,60,120,240]/(10**np.cumsum(a_T_list)[i-1]))
                stiffness_list.extend(stiffness(allresults['Temperature (C)'][i]).iloc[0,:])
                
                
                st.write(f"**logaT={allresults['Temperature (C)'][i]}: {round(np.cumsum(a_T_list)[i-1],2)}**")
                
                
                
                ax1.plot([8,15,30,60,120,240]/(10**np.cumsum(a_T_list)[i-1]),
                         stiffness(allresults['Temperature (C)'][i]).iloc[0,:],
                                   label=allresults['Temperature (C)'][i], 
                                   linestyle='-',marker='o')
            
            Trans_temp = np.array([1/(273.15 + x)-1/(273.15+Temperature_list[0]) for x in Temperature_list])
            logaT_arr = np.insert(np.cumsum(a_T_list),0,0,axis=0)
            
            slope4, intercept4 = linear_regression(Trans_temp, logaT_arr, proportional=True)
            
            fig4, ax4 = plt.subplots()
            ax4.plot(np.array(Temperature_list), 10**logaT_arr, label='Shift Factors', 
                     linestyle='None',
                     marker='o')
                     
            ax4.plot(np.array(Temperature_list),10**(slope4*Trans_temp),label='Arrhenius Model',linestyle='-',marker='None')
            
            ax4.set_title('Shift Factor vs Temperature')
            ax4.set_xlabel('Temperature (°C)')
            ax4.set_ylabel('Shift Factor')
            ax4.set_yscale('log')
            plt.figtext(0.50, 0.60, f'ln$a_{"T"}$ = ($E_{"a"}$/R)(1/T-1/$T_{{{"ref"}}}$)' )
            handles4, labels4 = ax4.get_legend_handles_labels()
            ax4.legend(handles4, labels4)
            st.pyplot(fig4)
            
            st.write(f"**$E_{'a'}$: {round(slope4*np.log(10)*8.314462618/1000,3)} kJ/mol**")
            
            
            
            ax1.set_title('Plot of Stiffness Mastercurve vs Reduced Time')
            ax1.set_xlabel('Time (s)')
            ax1.set_ylabel('Stiffness (MPa)')
            ax1.set_xscale('log')
            ax1.set_yscale('log')
            handles1, labels1 = ax1.get_legend_handles_labels()
            ax1.legend(handles1, labels1)
            st.pyplot(fig1)
            
            
            creep_comp_list = [1/i for i in stiffness_list]
            reduced_time = np.array(reduced_time_list)
            creep_compliance = np.array(creep_comp_list)

            
            
            initial_data = [0.3,-2,-3]
            
            result_gpl = minimize(gpl_minimize, initial_data)
            
            newtime = 10**np.linspace(np.log10(reduced_time).min(),np.log10(reduced_time).max(),50)
            newcreepcom = 10**result_gpl.x[1] + 10**result_gpl.x[2] * newtime**result_gpl.x[0]
            
            
            st.write(f"**m: {round(result_gpl.x[0],3)}**")
            st.write(f"**log$D_{0}$: {round(result_gpl.x[1],3)}**")
            st.write(f"**log$D_{1}$: {round(result_gpl.x[2],3)}**")
            
            
            fig2, ax2 = plt.subplots()
            ax2.plot(reduced_time_list, creep_comp_list,label='Master Curve', 
                     linestyle='None',
                     marker='o')
            ax2.plot(newtime,newcreepcom,label='GPL Model',linestyle='-',marker='None')
            ax2.set_title('Plot of Creep Compliance vs Reduced Time')
            ax2.set_xlabel('Time (s)')
            ax2.set_ylabel('Creep Compliance (1/MPa)')
            ax2.set_xscale('log')
            ax2.set_yscale('log')
            plt.figtext(0.20, 0.60, 'D(t) = $D_{0}$ + $D_{1}$.$t^m$' )
            handles2, labels2 = ax2.get_legend_handles_labels()
            ax2.legend(handles2, labels2)
            st.pyplot(fig2)

            reduced_omega = 2/(np.pi*reduced_time)
            storage_compliance = (10**result_gpl.x[1]) + (10**result_gpl.x[2]) * math.gamma(1+result_gpl.x[0]) * (reduced_omega)**(-result_gpl.x[0]) * np.cos(result_gpl.x[0] * np.pi/2)
            loss_compliance = (10**result_gpl.x[2]) * math.gamma(1+result_gpl.x[0]) * (reduced_omega)**(-result_gpl.x[0]) * np.sin(result_gpl.x[0] * np.pi/2)
            dynamic_compliance = (storage_compliance**2 + loss_compliance**2)**0.5
            dynamic_modulus = 1/dynamic_compliance
            dynamic_shear_modulus = dynamic_modulus/(2*(1+0.5))
            
            initial_data_CA = [0.1,-3]
            result_CA = minimize(ca_minimize, initial_data_CA)
            
            newomega = 10**np.linspace(np.log10(reduced_omega).min(),np.log10(reduced_omega).max(),50)
            newG_CA = 1000*(1+(10**result_CA.x[1]/newomega)**result_CA.x[0])**(-1/result_CA.x[0])
            
            st.write(f"**β: {round(result_CA.x[0],3)}**")
            st.write(f"**log$ω_{'C'}$: {round(result_CA.x[1],3)}**")
            
            
            
            fig5, ax5 = plt.subplots()
            ax5.plot(reduced_omega, dynamic_shear_modulus,label='Master Curve', 
                     linestyle='None',
                     marker='o')
            ax5.plot(newomega,newG_CA,label='CA Model',linestyle='-',marker='None')
            ax5.set_title('Plot of G* vs Reduced Angular Frequency')
            ax5.set_xlabel('ω (Rad/s)')
            ax5.set_ylabel('G* (MPa)')
            ax5.set_xscale('log')
            ax5.set_yscale('log')
            plt.figtext(0.20, 0.60, f'G* = $G_{"g"}$[1+($ω_{"C"}$/ω$)^{"β"}$$]^{{{"(-1/β)"}}}$')
            handles5, labels5 = ax5.get_legend_handles_labels()
            ax5.legend(handles5, labels5)
            st.pyplot(fig5)
            
            omega_GR = 0.005
            omega_GR_reduced = 0.005 * 10**(slope4*(1/(15+273.15)-1/(273.15+allresults['Temperature (C)'][0])))
            phase_GR = 90/(1+(omega_GR_reduced/(10**result_CA.x[1]))**result_CA.x[0])
            G_GR = 1000*1000*(1+(10**result_CA.x[1]/omega_GR_reduced)**result_CA.x[0])**(-1/result_CA.x[0])
            
            G_R = (G_GR/(np.sin(np.radians(phase_GR))))*(np.cos(np.radians(phase_GR)))**2

            st.write(f"**G-R: {round(G_R,0)} kPa**")
            st.write(f"**$G^{'*'}_{{{'G-R'}}}$: {round(G_GR,0)} kPa**")
            st.write(f"**$δ_{{{'G-R'}}}$: {round(phase_GR,0)} °**")

            st.markdown("""---""")

            initial_data_T_fatigue = [22]
            result_T_fatigue = minimize(T_fatigue_minimize, initial_data_T_fatigue)    
            
            st.write(f"**$T_{{{'Fatigue'}}}$: {round(result_T_fatigue.x[0],1)} °C**")
    
    
    
    
    
    
    
    
    # This can be modified to save to a file
