# -*- coding: utf-8 -*-
"""
Created on Tue Apr 29 16:42:12 2025

@author: NADERIK1
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FFMpegWriter
from scipy import stats
from scipy.optimize import minimize
from statistics import linear_regression
import math
import tempfile
import os
#import csv

# Function to create plots
def create_plot(data):
    fig, ax = plt.subplots(dpi=900)
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

def T_fatigue_6000_minimize(T):
    omega_red_T_fatigue = 10*10**(slope4*(1/(T+273.15)-1/(273.15+allresults['Temperature (C)'][0])))
    phase_fatigue = 90/(1+(omega_red_T_fatigue/(10**result_CA.x[1]))**result_CA.x[0])
    G_fatigue = 1000*1000*(1+(10**result_CA.x[1]/omega_red_T_fatigue)**result_CA.x[0])**(-1/result_CA.x[0])
    return (6000-G_fatigue*np.sin(np.radians(phase_fatigue)))**2

def T_pavel_kriz(T):
    omega_red_T_pavel_kriz = 10*10**(slope4*(1/(T+273.15)-1/(273.15+allresults['Temperature (C)'][0])))
    phase_pavel_kriz = 90/(1+(omega_red_T_pavel_kriz/(10**result_CA.x[1]))**result_CA.x[0])
    G_pavel_kriz = 1000*1000*(1+(10**result_CA.x[1]/omega_red_T_pavel_kriz)**result_CA.x[0])**(-1/result_CA.x[0])
    return (8967-G_pavel_kriz)**2

def count_lines(file):
    try:
        # Attempt to read the file content
        content = file.getvalue()
        try:
            decoded_content = content.decode("utf-8")
        except UnicodeDecodeError:
            # Fallback to a different encoding if UTF-8 fails
            decoded_content = content.decode("ISO-8859-1")  # or "latin1" or "cp1252"
        
        lines = decoded_content.splitlines()
        return len(lines)
    except Exception as e:
        st.error(f"An error occurred: {e}")
        return None

# Streamlit app layout
st.title("BBR Data Processor (beta release)")

st.logo(
    "icon.png", size="large"
)


# Create a sidebar
st.sidebar.header("⌨ BBRtoDSR")
st.sidebar.write("""
    This app reads CSV files from Bending Beam Rheometer tests and attempts to transform the data into dynamic shear results typically obtained from a Dynamic Shear Rheometer device. 
    By applying various rheological models and functions, it calculates different rheological parameters and indices.
""")

st.sidebar.markdown("""---""")
st.sidebar.subheader("⚠ Important Information")
# Add some descriptive text in the sidebar
st.sidebar.write("""
    Before use, please ensure that the test data are accurate and reproducible.
    Please exercise caution when using the results of this tool, 
    as this method attempts to extrapolate the Bending Beam Rheometer results beyond the ranges measured by the device. 
    Additionally, several assumptions have been made, 
    including Arrhenius-type temperature dependence, 
    the validity of [Generalized Power Law behavior for creep compliance at low temperatures](https://www.fhwa.dot.gov/publications/research/infrastructure/pavements/ltpp/10035/009.cfm), 
    the temperature and frequency independence of Poisson's ratio, 
    and the applicability of the [Christensen–Anderson (CA) model](https://doi.org/10.1080/14680629.2016.1267448) for complex modulus and phase angle master curves. 
    These assumptions may significantly deviate from the true behavior of the material!
""")
st.sidebar.markdown("""---""")

# Add the slider with a custom class for styling
poissons_ratio = st.sidebar.slider(
    "Select Poisson's Ratio:",
    min_value=0.25,  # Minimum value
    max_value=0.5,   # Maximum value
    value=0.50,      # Default value
    step=0.05,       # Step value for the slider
    format="%.2f",   # Format the displayed value
    help="Adjust the Poisson's ratio between 0.25 and 0.5"  # Optional help text
)

st.sidebar.markdown("""---""")



# Add a footer or additional text
st.sidebar.markdown("✉ Contact Me")
st.sidebar.write("For more information, please reach out to me at: [koorosh.naderi@colas.com](mailto:koorosh.naderi@colas.com)")

st.sidebar.markdown("""
<style>
.sidebar .sidebar-content {
    background-color: #f0f0f0;
    border-radius: 5px;
    padding: 10px;
    width: 650px
}

.stSlider {
        width: 80%;  /* Adjust this percentage to change slider width */
        margin: 0 auto;  /* Center the slider */
    }

</style>
""", unsafe_allow_html=True)

st.image("BBRtoDSRv1.jpeg")
st.write("© 2025 [Koorosh Naderi](https://www.linkedin.com/in/koorosh-naderi/)")
st.write("""A minimum of two CSV (XLSM) files is required for analysis. Please note that only CSV files from Cannon® Instrument Company can be read by the app (BBRw versions 1.34 and 1.35 were tested). 
XLSM files from Universal Test Module version 2.3.0.5 of PaveTest® have also been tested.""")

# Session state to track uploaded files
if 'uploaded_files' not in st.session_state:
    st.session_state.uploaded_files = []

# File uploader
uploaded_files = st.file_uploader("Choose files (CSV or XLSM)", accept_multiple_files=True, type=['csv', 'xlsm'])

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
    
    for uploaded_file in uploaded_files:
        file_type = uploaded_file.name.split('.')[-1].lower()
        if file_type == 'csv':
            num_lines = count_lines(uploaded_file)
            if num_lines>44:
                try:
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
                    temperature = np.float64(info[2][4])
                except:
                    st.error("The file is not compatible. num_lines>44")
            
            elif num_lines==44:
                
                try:
                    df = pd.read_csv(uploaded_file, header=None,engine='python', encoding='unicode_escape',skiprows=38)
                    uploaded_file.seek(0)
                    dfhead = pd.read_csv(uploaded_file,header=None,engine='python', encoding='unicode_escape',nrows=36,index_col=False)
                    
                    data = pd.DataFrame()
                    data['Time (s)'] = df[0]
                    data['Force (mN)'] = df[1]
                    data['Deflection (mm)'] = df[2]
                    data['Temperature (C)'] = dfhead.iloc[6,1]
                    data['Stiffness (MPa)'] = df[3]
                    data['log(t)'] = np.log10(np.float64(data['Time (s)']))
                    data['log(S)'] = np.log10(data['Stiffness (MPa)'])
                    data['Sc (MPa)'] = df[4]
                    data['Percent diff'] = (data['Stiffness (MPa)']-data['Sc (MPa)'])/data['Stiffness (MPa)']*100
                    data['m-value'] = df[6]
                    results = data
                    model = np.poly1d(np.polyfit(results['log(t)'], results['log(S)'], 2))
                    temperature = np.float64(dfhead.iloc[6,1])
                except:
                    st.error("The file is not compatible. num_lines=44")
            else:
                st.error("The file is not compatible. num_lines not number")
            
            
            # Display the uploaded dataframe
            st.write(f"**Data from {uploaded_file.name}:**")
            
            try:
                st.dataframe(results, hide_index = True)
                
                if abs(np.float64(results['Temperature (C)']).mean()-temperature)>0.1 :
                    st.write(f"Temperature Control was not correct and the value considered is the test temperature not the intended temperature of {info[2][4]}.")
                    temperature = np.float64(results['Temperature (C)']).mean()
                
                allresults.loc[len(allresults)] = [temperature,
                                                   model.coefficients[2],
                                                   model.coefficients[1],
                                                   model.coefficients[0],
                                                   10**(model(np.log10(60))),
                                                   abs(2*model.coefficients[0]*np.log10(60)+model.coefficients[1])]
                fig = create_plot(results)
                st.pyplot(fig)
            except:
                st.write("There was an error reading the file.")
        
        elif file_type == 'xlsm':    
            # XLSM handling
            sheet_name = 'BBR Results'
            start_row = 100
            
            try:
                df = pd.read_excel(uploaded_file, sheet_name=sheet_name, engine="openpyxl", skiprows=start_row-1,header=None)
                full_sheet = pd.read_excel(uploaded_file, sheet_name=sheet_name, engine="openpyxl", header=None)
                    
                data = pd.DataFrame()
                data['Time (s)'] = df[0]
                data['Force (mN)'] = df[1]*1000
                data['Deflection (mm)'] = df[2]
                data['Temperature (C)'] = df[3]
                
                BeamSpan = np.float64(full_sheet.iloc[29,2])/1000
                BeamWidth = np.float64(full_sheet.iloc[19,8])/1000
                BeamThickness = np.float64(full_sheet.iloc[20,8])/1000
                temperature = np.float64(full_sheet.iloc[28,2])
                
                data['Stiffness (MPa)'] = 1/1000000*(np.float64(data['Force (mN)'])*BeamSpan**3)/(np.float64(data['Deflection (mm)'])*4*BeamWidth*BeamThickness**3)
                data['log(t)'] = np.log10(np.float64(data['Time (s)']))
                data['log(S)'] = np.log10(data['Stiffness (MPa)'])
            
                results = data[data['Time (s)'].isin([8,15,30,60,120,240])]
                model = np.poly1d(np.polyfit(results['log(t)'], results['log(S)'], 2))
            
                data['Sc (MPa)'] = 10**model(np.float64(data['log(t)']))
                data['Percent diff'] = (data['Stiffness (MPa)']-data['Sc (MPa)'])/data['Stiffness (MPa)']*100
                data['m-value'] = abs(2*model.coefficients[0]*data['log(t)']+model.coefficients[1])
                
                results = data[data['Time (s)'].isin([8,15,30,60,120,240])]
                
                
            except Exception as e:
                st.error(f"Error reading XLSM file: {e}")
            
            # Display the uploaded dataframe
            st.write(f"**Data from {uploaded_file.name}:**")
            
            try:
                st.dataframe(results, hide_index = True)
                
                if abs(np.float64(results['Temperature (C)']).mean()-temperature)>0.1 :
                    st.write(f"Temperature Control was not correct and the value considered is the test temperature not the intended temperature of {full_sheet.iloc[28,2]}.")
                    temperature = np.float64(results['Temperature (C)']).mean()
                
                allresults.loc[len(allresults)] = [temperature,
                                                   model.coefficients[2],
                                                   model.coefficients[1],
                                                   model.coefficients[0],
                                                   10**(model(np.log10(60))),
                                                   abs(2*model.coefficients[0]*np.log10(60)+model.coefficients[1])]
                fig = create_plot(results)
                st.pyplot(fig)
            except:
                st.write("There was an error reading the file.")
        else:
            st.error("Unsupported file type") 

    #
if st.button("Print Results"):
    st.markdown("""---""")
    st.subheader("Low Temperature Properties")
    allresults.sort_values('Temperature (C)',
                                        axis=0,
                                        ascending=False,inplace=True)
    allresults.reset_index(drop=True, inplace=True)
    st.dataframe(allresults , hide_index = True)
    
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
            st.write(f"**$Δ T_{'c'}$: {Delta_Tc} °C**")

            st.markdown("""---""")

        
            st.subheader(f"**Time-temperature Superposition, Shift factor using Arrhenius law**")
            st.write(f"**Reference Temperature, $T_{{{'ref'}}}$: {allresults['Temperature (C)'][0]} °C**")           
                      
            a_T_list = []
            Temperature_list = [allresults['Temperature (C)'][0]]
            reduced_time_list = [8,15,30,60,120,240]
            stiffness_list = list(stiffness(allresults['Temperature (C)'][0]).iloc[0,:])
            
            fig1, ax1 = plt.subplots()
            
            ax1.plot([8,15,30,60,120,240], stiffness(allresults['Temperature (C)'][0]).iloc[0,:],label=f'{allresults['Temperature (C)'][0]} °C', 
                     linestyle='-', linewidth=5.0, alpha=0.4,
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
                
                
                st.write(f"**$loga_{{{'T ='}{allresults['Temperature (C)'][i]}{'°C'}}}$: {round(np.cumsum(a_T_list)[i-1],2)}**")
                
                
                
                ax1.plot([8,15,30,60,120,240]/(10**np.cumsum(a_T_list)[i-1]),
                         stiffness(allresults['Temperature (C)'][i]).iloc[0,:],
                                   label=f'{allresults['Temperature (C)'][i]} °C', 
                                   linestyle='-',marker='o', linewidth=5.0, alpha=0.4)
            
            Trans_temp = np.array([1/(273.15 + x)-1/(273.15+Temperature_list[0]) for x in Temperature_list])
            logaT_arr = np.insert(np.cumsum(a_T_list),0,0,axis=0)
            
            slope4, intercept4 = linear_regression(Trans_temp, logaT_arr, proportional=True)

            predicted_logaT_arr = [slope4 * xi + intercept4 for xi in Trans_temp]
            
            rss = sum((yi - y_pred) ** 2 for yi, y_pred in zip(logaT_arr, predicted_logaT_arr))
            mean_y = sum(logaT_arr) / len(logaT_arr)
            tss = sum((yi - mean_y) ** 2 for yi in logaT_arr)
            r_squared_Arrhenius = 1 - rss / tss

            fig4, ax4 = plt.subplots()
            ax4.plot(np.array(Temperature_list), 10**logaT_arr, label='Shift Factors', 
                     linestyle='None',
                     marker='o')
                     
            ax4.plot(np.array(Temperature_list),10**(slope4*Trans_temp),label='Arrhenius Model',linestyle='-',marker='None')
            
            ax4.set_title('Shift Factor vs Temperature')
            ax4.set_xlabel('Temperature (°C)')
            ax4.set_ylabel('Shift Factor')
            ax4.set_yscale('log')
        
            plt.figtext(0.50, 0.60, f'ln$a_{"T"}$ = ($E_{"a"}$/R)(1/$T$-1/$T_{{{"ref"}}}$)' )
            plt.figtext(0.50, 0.55, f'$r^{2}$ = {round(r_squared_Arrhenius,3)}' )
        
            handles4, labels4 = ax4.get_legend_handles_labels()
            ax4.legend(handles4, labels4)
            st.pyplot(fig4)
            
            st.write(f"**$E_{'a'}$: {round(slope4*np.log(10)*8.314462618/1000,3)} kJ/mol**")
            st.write(f"**R is the universal gas constant which is equal to 8.31446261815324 $J$⋅$K^{{{'−1'}}}$⋅$mol^{{{'−1'}}}$**")
            st.write(f"**Please note that the temperature is converted to Kelvin, and 'ln' in the function refers to the natural logarithm.**")
            st.write(f"**An $r^{2}$ value below 0.98 raises concerns and warrants rechecking the BBR test data.**")
        
            
            st.markdown("""---""")
            
            ax1.set_title('Plot of Stiffness Mastercurve vs Reduced Time')
            ax1.set_xlabel('Time (s)')
            ax1.set_ylabel('Stiffness (MPa)')
            ax1.set_xscale('log')
            ax1.set_yscale('log')
            handles1, labels1 = ax1.get_legend_handles_labels()
            ax1.legend(handles1, labels1)
            st.pyplot(fig1)

            st.markdown("""---""")
            st.subheader(f"**Creep Compliance Mastercurve, Generalized Power Law (GPL)**")
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

            st.markdown("""---""")
            st.subheader(f"**Complex Modulus Mastercurve, Christensen–Anderson (CA) Model**")

            reduced_omega = 2/(np.pi*reduced_time)
            storage_compliance = (10**result_gpl.x[1]) + (10**result_gpl.x[2]) * math.gamma(1+result_gpl.x[0]) * (reduced_omega)**(-result_gpl.x[0]) * np.cos(result_gpl.x[0] * np.pi/2)
            loss_compliance = (10**result_gpl.x[2]) * math.gamma(1+result_gpl.x[0]) * (reduced_omega)**(-result_gpl.x[0]) * np.sin(result_gpl.x[0] * np.pi/2)
            dynamic_compliance = (storage_compliance**2 + loss_compliance**2)**0.5
            dynamic_modulus = 1/dynamic_compliance
            dynamic_shear_modulus = dynamic_modulus/(2*(1+poissons_ratio))
            
            initial_data_CA = [0.1,-3]
            result_CA = minimize(ca_minimize, initial_data_CA)
            
            newomega = 10**np.linspace(np.log10(reduced_omega).min(),np.log10(reduced_omega).max(),50)
            newG_CA = 1000*(1+(10**result_CA.x[1]/newomega)**result_CA.x[0])**(-1/result_CA.x[0])
            newphase_CA = 90/(1+(newomega/(10**result_CA.x[1]))**result_CA.x[0])
            
            st.write(f"**β: {round(result_CA.x[0],3)}**")
            st.write(f"**log$ω_{'C'}$: {round(result_CA.x[1],3)} at $T_{{{"ref"}}}$**")
        
            logomega_C_zero = np.log10((10**result_CA.x[1])*(10**(slope4*(1/(0+273.15)-1/(273.15+allresults['Temperature (C)'][0])))))

            st.write(f"**log$ω_{'C'}$: {round(logomega_C_zero,3)} at 0°C**")
                        
            fig5, ax5 = plt.subplots()
            ax5.plot(reduced_omega, dynamic_shear_modulus,label='Master Curve', 
                     linestyle='None',
                     marker='o')
            ax5.plot(newomega,newG_CA,label='CA Model',linestyle='-',marker='None')
            ax5.set_title('Plot of |G*| vs Reduced Angular Frequency')
            ax5.set_xlabel('ω (Rad/s)')
            ax5.set_ylabel('|G*| (MPa)')
            ax5.set_xscale('log')
            ax5.set_yscale('log')
            plt.figtext(0.20, 0.60, f'|G*| = $G_{"g"}$[1+($ω_{"C"}$/ω$)^{"β"}$$]^{{{"(-1/β)"}}}$')
            handles5, labels5 = ax5.get_legend_handles_labels()
            ax5.legend(handles5, labels5)
            st.pyplot(fig5)
        
            st.write(f"**Rheological Index: {round(np.log10(2)/result_CA.x[0],2)}**")
            st.write(f"**The glassy modulus ($G_{'g'}$) was assumed to be a constant value of 1 GPa.**")
        

            st.markdown("""---""")
            st.subheader(f"**Glover-Rowe Parameter, Cracking Performance**")

        
            omega_GR = 0.005
            omega_GR_reduced = omega_GR * 10**(slope4*(1/(15+273.15)-1/(273.15+allresults['Temperature (C)'][0])))
            phase_GR = 90/(1+(omega_GR_reduced/(10**result_CA.x[1]))**result_CA.x[0])
            G_GR = 1000*1000*(1+(10**result_CA.x[1]/omega_GR_reduced)**result_CA.x[0])**(-1/result_CA.x[0])
            
            G_R = (G_GR/(np.sin(np.radians(phase_GR))))*(np.cos(np.radians(phase_GR)))**2

            st.write(f"**$G-R$: {round(G_R,0)} kPa**")
            st.write(f"**$|G^{'*'}|_{{{'G-R'}}}$: {round(G_GR,0)} kPa**")
            st.write(f"**$δ_{{{'G-R'}}}$: {round(phase_GR,0)} °**")

            fig6, ax6 = plt.subplots()
            ax6.plot(phase_GR, G_GR, label='G-R Parameter', 
                     linestyle='None',
                     marker='o')
            ax6.plot(np.arange(1,89,1),180*np.sin(np.radians(np.arange(1,89,1)))/np.cos(np.radians(np.arange(1,89,1)))**2,label='G-R = 180 kPa',linestyle='-',marker='None',c='green')
            ax6.plot(np.arange(1,89,1),600*np.sin(np.radians(np.arange(1,89,1)))/np.cos(np.radians(np.arange(1,89,1)))**2,label='G-R = 600 kPa',linestyle='-',marker='None',c='red')
            ax6.set_title('Black Diagram')
            ax6.set_xlabel('Phase Angle (°)')
            ax6.set_ylabel('|G*| (kPa)')
            ax6.set_yscale('log')
            ax6.set_ylim(top=1e6)
            ax6.set_xlim(0,90)
            handles6, labels6 = ax6.get_legend_handles_labels()
            ax6.legend(handles6, labels6)
            st.pyplot(fig6)

            st.write(f"**The Glover-Rowe parameter, which is based on the DSR Fn proposed by [Glover et al.](https://rosap.ntl.bts.gov/view/dot/81884), was originally developed to relate dynamic rheological properties in shear mode to tensile failure strain (ductility) in tension mode for pure or unmodified bitumen (asphalt binder). If the tested sample is modified, however, care should be taken, as high G-R values do not necessarily correlate with low ductility values.**")
        
            st.markdown("""---""")

            initial_data_T_fatigue = [22]
            result_T_fatigue = minimize(T_fatigue_minimize, initial_data_T_fatigue)    

            st.subheader(f"**Fatigue Cracking Criteria**")
            st.write(f"**$T_{{{'G"=5000kPa'}}}$: {round(result_T_fatigue.x[0],1)} °C**")

            initial_data_T_fatigue_6000 = [22]
            result_T_fatigue_6000 = minimize(T_fatigue_6000_minimize, initial_data_T_fatigue_6000)

            st.write(f"**$T_{{{'G"=6000kPa'}}}$: {round(result_T_fatigue_6000.x[0],1)} °C**")

            omega_fatigue6_superpave = 10
            omega_fatigue6_superpave_reduced = omega_fatigue6_superpave * 10**(slope4*(1/(result_T_fatigue_6000.x[0]+273.15)-1/(273.15+allresults['Temperature (C)'][0])))
            phase_fatigue6 = 90/(1+(omega_fatigue6_superpave_reduced/(10**result_CA.x[1]))**result_CA.x[0])

            st.write(f"**$δ_{{{'G"=6000kPa'}}}$: {round(phase_fatigue6,1)} °**")
        
            Temperature_fatigue_list = np.array([4, 7, 10, 13, 16, 19, 22, 25, 28, 31, 34, 37, 40])
            Omega_fatigue_list = 10 * 10**(slope4*(1/(Temperature_fatigue_list+273.15)-1/(273.15+allresults['Temperature (C)'][0])))
            phase_fatigue_list = 90/(1+(Omega_fatigue_list/(10**result_CA.x[1]))**result_CA.x[0])
            G_fatigue_list = 1000*1000*(1+(10**result_CA.x[1]/Omega_fatigue_list)**result_CA.x[0])**(-1/result_CA.x[0])
            G_storage_fatigue_list = G_fatigue_list * np.cos(np.radians(phase_fatigue_list))
            G_loss_fatigue_list = G_fatigue_list * np.sin(np.radians(phase_fatigue_list))
            
            fatigue_list = pd.DataFrame(columns=['Temperature (°C)','Phase Angle (°)','|G*| (kPa)',"G' (kPa)",'G" (kPa)'])
            fatigue_list['Temperature (°C)'] = Temperature_fatigue_list
            fatigue_list['Phase Angle (°)'] = np.round(phase_fatigue_list,1)
            fatigue_list['|G*| (kPa)'] = np.round(G_fatigue_list,0)
            fatigue_list["G' (kPa)"] = np.round(G_storage_fatigue_list,0)
            fatigue_list['G" (kPa)'] = np.round(G_loss_fatigue_list,0)

            st.dataframe(fatigue_list, hide_index = True)

            fig7, ax7 = plt.subplots()
            ax7.plot(phase_fatigue_list, G_fatigue_list, label='Superpave Fatigue Points, ω = 10 Rad/s', 
                     linestyle='None',
                     marker='o', alpha=0.7, markersize=2, c='red')
            ax7.plot(np.arange(1,89,1),5000/np.sin(np.radians(np.arange(1,89,1))),label='|G*|sinδ = 5000 kPa',linestyle='--',marker='None',c='black')
            ax7.plot(np.arange(1,89,1),6000/np.sin(np.radians(np.arange(1,89,1))),label='|G*|sinδ = 6000 kPa',linestyle='-.',marker='None',c='black',alpha=0.3)
            ax7.set_title('Black Diagram')
            ax7.set_xlabel('Phase Angle (°)')
            ax7.set_ylabel('|G*| (kPa)')
            ax7.set_yscale('log')
            ax7.set_ylim(1,1e6)
            ax7.set_xlim(0,90)
            handles7, labels7 = ax7.get_legend_handles_labels()
            ax7.legend(handles7, labels7)
            for x, y, z in zip(phase_fatigue_list, G_fatigue_list, Temperature_fatigue_list):
                ax7.text(x, y, f"{z}°C", fontsize=7)
            st.pyplot(fig7)


            st.markdown("""---""")
            st.subheader(f"**Pavel-Kriz Phase Angle, [Detection of Phase Incompatible Binders](https://trid.trb.org/View/2344464/)**")

            initial_data_T_pavel_kriz = [22]
            result_T_pavel_kriz = minimize(T_pavel_kriz, initial_data_T_pavel_kriz)
        
            Omega_pavel_kriz = 10 * 10**(slope4*(1/(result_T_pavel_kriz.x[0]+273.15)-1/(273.15+allresults['Temperature (C)'][0])))
            phase_pavel_kriz = 90/(1+(Omega_pavel_kriz/(10**result_CA.x[1]))**result_CA.x[0])

            st.write(f"ω = 10 Rad/s")
            st.write(f"T = {round(result_T_pavel_kriz.x[0],1)} °C")
            st.write(f"**$δ_{{{'|G*|=8967kPa'}}}$: {round(phase_pavel_kriz,1)} °**")


            fig8, ax8 = plt.subplots()
            ax8.plot(newphase_CA, 1000*newG_CA, label='CA Model Points', 
                     linestyle='--',
                     marker='', alpha=0.6)
            ax8.plot(phase_pavel_kriz, 8967, label='Pavel-Kriz Point', 
                     linestyle='None',
                     marker='o', alpha=0.6)
            ax8.vlines(x=42,ymin=8967,ymax=1e6,linestyle='dashdot',colors='black',label='Pavel-Kriz Criteria')
            ax8.hlines(y=8967,xmin=0,xmax=90,linestyle='dashdot',colors='black')
            ax8.set_title('Black Diagram')
            ax8.set_xlabel('Phase Angle (°)')
            ax8.set_ylabel('|G*| (kPa)')
            ax8.set_yscale('log')
            ax8.set_ylim(1,1e6)
            ax8.set_xlim(0,90)
            handles8, labels8 = ax8.get_legend_handles_labels()
            ax8.legend(handles8, labels8)
            st.pyplot(fig8)



            st.markdown("""---""")
            st.subheader(f"**Animated Master Curve Shifting**")

            fig9, ax9 = plt.subplots()
            ax9.set_yscale('log')
            
            movingshifts = np.insert(np.cumsum(a_T_list),0,0)
            movingtime = np.array(np.log10([8, 15, 30, 60, 120, 240]))
            
            lines = [ax9.plot([], [], label=f'T = {round(tem,2)} °C', antialiased=True, alpha = 0.4, linewidth = 5)[0] for tem in allresults['Temperature (C)']]
            
            # Initialization function
            def init():
                for line in lines:
                    line.set_data([], [])
                return lines
                
            # Animation function
            def animate(i):
                for idx, shift in enumerate(movingshifts):
                    shifted_time = movingtime - (0.01 * i * shift)
                    mod = stiffness(allresults['Temperature (C)'][idx]).iloc[0,:]
                    lines[idx].set_data(shifted_time, mod)
                return lines

            ax9.set_title('Plot of Stiffness Mastercurve vs Reduced Time')
            ax9.set_xlabel('Log Time (s)')
            ax9.set_ylabel('Stiffness (MPa)')
            handles9, labels9 = ax9.get_legend_handles_labels()
            ax9.legend(handles9, labels9)
            
            ax9.set_xlim(round(movingtime.min()-movingshifts.max(),0)-1, round(movingtime.max(),0)+1)
            ax9.set_ylim(10**(round(np.log10(stiffness(allresults['Temperature (C)'][0]).iloc[0,:].min()),0)-1), 
                         10**(round(np.log10(stiffness(allresults['Temperature (C)'][0]).iloc[0,:].max()),0)+1))

            with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as fp:
                fname = fp.name
            
            ani = animation.FuncAnimation(fig9, animate, init_func=init, frames=100, interval=5, blit=True)

            writer = FFMpegWriter(fps=25,
                      metadata=dict(artist='Your Name'),
                      codec='libx264',
                      extra_args=['-pix_fmt', 'yuv420p'])

            ani.save(fname, writer=writer, dpi=200)
        
            plt.close(fig9)
            
            with open(fname, "rb") as f:
                video_bytes = f.read()
            os.remove(fname)  # Clean up       
            
            st.video(video_bytes)
    else:
        st.write("Upload a minimum of two CSV files for further analysis.")
    
    
    # This can be modified to save to a file




















































