# -*- coding: utf-8 -*-
"""
Created on Tue Apr 29 16:42:12 2025

@author: NADERIK1
"""

#import libraries
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

#magic numbers
BBR_TIMES = np.array([8,15,30,60,120,240])
LOG_BBR_TIMES = np.log10(BBR_TIMES)

POLY_MATRIX = np.vstack([
    np.ones(len(BBR_TIMES)),
    LOG_BBR_TIMES,
    LOG_BBR_TIMES**2
])

GAS_CONSTANT = 8.31446261815324
DEFAULT_GLASSY_MODULUS_MPA = 1000
KELVIN_OFFSET = 273.15
M_VALUE_LIMIT = 0.3
STIFFNESS_LIMIT = 300
REFERENCE_BBR_TIME = 60
FATIGUE_LIMIT = 5000
FATIGUE_LIMIT_ALT = 6000
PAVEL_KRIZ_MODULUS = 6000 / np.sin(np.deg2rad(42))
DISPLAY_DPI = 450


def find_bracketing_rows(df, column_name, target_value):
    # Sort the DataFrame by the specified column
    df_sorted = df.sort_values(by=column_name)  
    # Initialize variables for bracketing
    lower_row = None
    upper_row = None
    # Iterate through the sorted DataFrame to find bracketing rows
    for _, row in df_sorted.iterrows():
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


def calculate_low_temperature_properties(allresults):
    ddf1 = find_bracketing_rows(
        allresults,
        'm-value(60)',
        M_VALUE_LIMIT)

    ddf2 = find_bracketing_rows(
        allresults,
        'S(60)',
        STIFFNESS_LIMIT)

    slope1, intercept1, _, _, _ = stats.linregress(
        ddf1['m-value(60)'],
        ddf1['Temperature (C)'])

    slope2, intercept2, _, _, _ = stats.linregress(
        np.log(ddf2['S(60)']),
        ddf2['Temperature (C)'])

    T_s = round(
        -10 + slope2 * np.log(STIFFNESS_LIMIT)
        + intercept2,
        1)

    T_m = round(
        -10 + slope1 * M_VALUE_LIMIT
        + intercept1,
        1)

    Delta_Tc = round(
        T_s - T_m,
        1)

    return {
        "Tc_S": T_s,
        "Tc_m": T_m,
        "Delta_Tc": Delta_Tc,
        "slope1": slope1,
        "intercept1": intercept1,
        "slope2": slope2,
        "intercept2": intercept2,
        "ddf1": ddf1,
        "ddf2": ddf2
    }


def celsius_to_kelvin(temp_c):
    return temp_c + KELVIN_OFFSET

def fit_bbr_curve(data):
    data = data.copy()
    data['Time (s)'] = pd.to_numeric(data['Time (s)'], errors='coerce')
    data['log(t)'] = np.log10(
        np.float64(data['Time (s)']))
    data['log(S)'] = np.log10(
        data['Stiffness (MPa)'])
    results = data[
    data['Time (s)'].isin(
            BBR_TIMES)]
    model = np.poly1d(
        np.polyfit(
            results['log(t)'],
            results['log(S)'],
            2))
    data['Sc (MPa)'] = 10**(
    model(data['log(t)']))
    data['Percent diff'] = (
        data['Stiffness (MPa)']
        - data['Sc (MPa)']
    ) / data['Stiffness (MPa)'] * 100
    data['m-value'] = np.abs(
        2*model.coefficients[0]
        * data['log(t)']
        + model.coefficients[1])
    results = data[data['Time (s)'].isin(BBR_TIMES)]
    return data, results, model


def compute_tts(allresults):

    a_T_list = []
    master_curve_series = []
    shift_data = []

    temperatures = [allresults['Temperature (C)'][0]]

    reduced_time_list = BBR_TIMES.tolist()

    stiffness_list = list(
        stiffness(
            allresults['Temperature (C)'][0],
            allresults
        ).iloc[0,:]
    )

    master_curve_series.append({
        "temperature": allresults['Temperature (C)'][0],
        "time": BBR_TIMES,
        "stiffness": stiffness(
            allresults['Temperature (C)'][0],
            allresults
        ).iloc[0,:]
    })

    
    for i in range(1,len(allresults)):
        fixed_T1 = allresults['Temperature (C)'][i-1]
        fixed_T2 = allresults['Temperature (C)'][i]
        initial_x = [np.log10(7200/60)*(1/celsius_to_kelvin(fixed_T2)-1/celsius_to_kelvin(fixed_T1))/(
            1/(-10+celsius_to_kelvin(fixed_T1))-1/(celsius_to_kelvin(fixed_T1)))]
        result = minimize(shift_factor_objective,
                          initial_x,
                          args=(fixed_T1, fixed_T2, allresults))
        
        
        a_T_list.append(result.x[0])
        temperatures.append(allresults['Temperature (C)'][i])
        reduced_time_list.extend(BBR_TIMES/(10**np.cumsum(a_T_list)[i-1]))
        stiffness_list.extend(stiffness(allresults['Temperature (C)'][i], allresults).iloc[0,:])
        
        shift_data.append({
            "temperature": allresults['Temperature (C)'][i],
            "shift_factor": result.x[0],
            "cumulative_shift": np.cumsum(a_T_list)[i-1]
        })
        
        
        master_curve_series.append({
            "temperature": allresults['Temperature (C)'][i],
            "time": BBR_TIMES/(10**np.cumsum(a_T_list)[i-1]),
            "stiffness": stiffness(
                allresults['Temperature (C)'][i],
                allresults
            ).iloc[0,:]
        })
    inverse_temperature_difference = np.array([1/celsius_to_kelvin(x)-1/celsius_to_kelvin(temperatures[0]) for x in temperatures])
    logaT_arr = np.insert(np.cumsum(a_T_list),0,0,axis=0)
    
    slope4, _ = linear_regression(inverse_temperature_difference, logaT_arr, proportional=True)

    predicted_logaT_arr = [slope4 * xi for xi in inverse_temperature_difference]
    
    rss = sum((yi - y_pred) ** 2 for yi, y_pred in zip(logaT_arr, predicted_logaT_arr))
    mean_y = sum(logaT_arr) / len(logaT_arr)
    tss = sum((yi - mean_y) ** 2 for yi in logaT_arr)
    r_squared_Arrhenius = 1 - rss / tss
    
    shift_factor_values = 10**logaT_arr
    arrhenius_values = 10**(slope4 * inverse_temperature_difference)

    return {
    "a_T_list": a_T_list,
    "temperatures": temperatures,
    "reduced_time_list": reduced_time_list,
    "stiffness_list": stiffness_list,
    "shift_data": shift_data,
    "master_curve_series": master_curve_series,
    "slope4": slope4,
    "r_squared_Arrhenius": r_squared_Arrhenius,
    "shift_factor_values": shift_factor_values,
    "arrhenius_values": arrhenius_values
            }

def compute_gpl(
    reduced_time_list,
    stiffness_list
):

    creep_comp_list = [1/i for i in stiffness_list]

    reduced_time = np.array(reduced_time_list)

    creep_compliance = np.array(
        creep_comp_list
    )

    initial_data = [0.3, -2, -3]

    result_gpl = minimize(
        gpl_objective,
        initial_data,
        args=(
            reduced_time,
            creep_compliance
        )
    )

    newtime = 10**np.linspace(
        np.log10(reduced_time).min(),
        np.log10(reduced_time).max(),
        50
    )
    
    predicted_creep_compliance = (
    10**result_gpl.x[1]
    + 10**result_gpl.x[2]
    * reduced_time**result_gpl.x[0])

    newcreepcom = (
        10**result_gpl.x[1]
        + 10**result_gpl.x[2]
        * newtime**result_gpl.x[0]
    )
    
    rss = np.sum(
    (
        np.log10(creep_compliance)
        - np.log10(predicted_creep_compliance)
    )**2
    )
    
    tss = np.sum(
        (
            np.log10(creep_compliance)
            - np.mean(
                np.log10(creep_compliance)
            )
        )**2
    )
    
    r2_gpl = 1 - rss / tss
    rmse_log = np.sqrt(
    np.mean(
        (
            np.log10(creep_compliance)
            - np.log10(predicted_creep_compliance)
        )**2
    )
    )
    

    return {
        "m": result_gpl.x[0],
        "logD0": result_gpl.x[1],
        "logD1": result_gpl.x[2],
        "creep_comp_list": creep_comp_list,
        "reduced_time": reduced_time,
        "creep_compliance": creep_compliance,
        "newtime": newtime,
        "newcreepcom": newcreepcom,
        "result": result_gpl,
        "success": result_gpl.success,
        "message": result_gpl.message,
        "r2": r2_gpl,
        "rmse_log": rmse_log,
        "objective_value": result_gpl.fun
        
    }

def compute_ca(
    reduced_time,
    logD0,
    logD1,
    m,
    poissons_ratio,
    glassy_modulus
              ):
    
    reduced_omega = 2/(np.pi*reduced_time)
    storage_compliance = (10**logD0) + (10**logD1) * math.gamma(1+m) * (reduced_omega)**(-m) * np.cos(m * np.pi/2)
    loss_compliance = (10**logD1) * math.gamma(1+m) * (reduced_omega)**(-m) * np.sin(m * np.pi/2)
    dynamic_compliance = (storage_compliance**2 + loss_compliance**2)**0.5
    dynamic_modulus = 1/dynamic_compliance
    dynamic_shear_modulus = dynamic_modulus/(2*(1+poissons_ratio))
    
    initial_data_CA = [0.1,-3]
    
    result_CA = minimize(
                        ca_objective,
                        initial_data_CA,
                        args=(
                            reduced_omega,
                            dynamic_shear_modulus,
                            glassy_modulus
                    
                            ),
                        
                    )
    
    newomega = 10**np.linspace(np.log10(reduced_omega).min(),np.log10(reduced_omega).max(),50)
    newG_CA = glassy_modulus*(1+(10**result_CA.x[1]/newomega)**result_CA.x[0])**(-1/result_CA.x[0])
    newphase_CA = 90/(1+(newomega/(10**result_CA.x[1]))**result_CA.x[0])
    
    predicted_dynamic_shear_modulus = (
    glassy_modulus
    * (
        1 +
        (10**result_CA.x[1]/reduced_omega)
        **result_CA.x[0]
    )**(-1/result_CA.x[0])
    )
    
    rss = np.sum(
    (
        np.log10(dynamic_shear_modulus)
        -
        np.log10(predicted_dynamic_shear_modulus)
    )**2
    )
    
    tss = np.sum(
    (
        np.log10(dynamic_shear_modulus)
        -
        np.mean(
            np.log10(dynamic_shear_modulus)
        )
    )**2
    )
    
    r2_ca = 1 - rss/tss
    rmse_ca = np.sqrt(
    np.mean(
        (
            np.log10(dynamic_shear_modulus)
            -
            np.log10(predicted_dynamic_shear_modulus)
        )**2
    )
    )
    
    
    return {
    "beta": result_CA.x[0],
    "logOmegaC": result_CA.x[1],
    "reduced_omega": reduced_omega,
    "dynamic_shear_modulus": dynamic_shear_modulus,
    "newomega": newomega,
    "newG_CA": newG_CA,
    "newphase_CA": newphase_CA,
    "result": result_CA,
    "r2": r2_ca,
    "rmse": rmse_ca,
    "success": result_CA.success,
    "message": result_CA.message,
}



# Function to create plots-------------------------------------------------------------------
def create_plot(data):
    fig, ax = plt.subplots(dpi=DISPLAY_DPI)
    ax.plot(data['Time (s)'], data['Sc (MPa)'],label='Estimated', linestyle='-')
    ax.plot(data['Time (s)'], data['Stiffness (MPa)'],label='Measured', linestyle=':', marker='o')
    ax.set_title('Plot of Stiffness vs Time')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Stiffness (MPa)')
    ax.set_xscale('log')
    ax.set_yscale('log')
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels)
    return fig

def create_arrhenius_plot(
    temperature_list,
    shift_factor_values,
    arrhenius_values,
    r_squared_arrhenius
):

    fig, ax = plt.subplots()

    ax.plot(
        np.array(temperature_list),
        shift_factor_values,
        label='Shift Factors',
        linestyle='None',
        marker='o'
    )

    ax.plot(
        np.array(temperature_list),
        arrhenius_values,
        label='Arrhenius Model',
        linestyle='-',
        marker='None'
    )

    ax.set_title('Shift Factor vs Temperature')
    ax.set_xlabel('Temperature (°C)')
    ax.set_ylabel('Shift Factor')
    ax.set_yscale('log')

    fig.text(
        0.50,
        0.60,
        'ln$a_T$ = ($E_a$/R)(1/$T$-1/$T_{ref}$)'
    )

    fig.text(
        0.50,
        0.55,
        f'$r^2$ = {round(r_squared_arrhenius, 3)}'
    )

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels)

    return fig

def create_master_curve_plot(master_curve_series):

    fig, ax = plt.subplots()

    for curve in master_curve_series:

        ax.plot(
            curve["time"],
            curve["stiffness"],
            label=f'{curve["temperature"]} °C',
            linestyle='-',
            linewidth=5.0,
            alpha=0.4,
            marker='o'
        )

    ax.set_title('Plot of Stiffness Master Curve vs Reduced Time')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Stiffness (MPa)')
    ax.set_xscale('log')
    ax.set_yscale('log')

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels)

    return fig


def create_gpl_plot(
    reduced_time_list,
    creep_comp_list,
    newtime,
    newcreepcom,
    r2_gpl,
    rmse_log
):

    fig, ax = plt.subplots()

    ax.plot(
        reduced_time_list,
        creep_comp_list,
        label='Master Curve',
        linestyle='None',
        marker='o'
    )

    ax.plot(
        newtime,
        newcreepcom,
        label='GPL Model',
        linestyle='-',
        marker='None'
    )

    ax.set_title(
        'Plot of Creep Compliance vs Reduced Time'
    )

    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Creep Compliance (1/MPa)')

    ax.set_xscale('log')
    ax.set_yscale('log')

    fig.text(
        0.20,
        0.60,
        'D(t) = $D_{0}$ + $D_{1}$.$t^m$'
    )
    fig.text(
    0.20,
    0.55,
    f'$R^2$ = {r2_gpl:.4f}'
    )

    fig.text(
    0.20,
    0.50,
    f'RMSE(log) = {rmse_log:.4f}'
    )

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels)

    return fig

def create_ca_plot(
    reduced_omega,
    dynamic_shear_modulus,
    newomega,
    newG_CA,
    r2_ca,
    rmse_ca
):

    fig, ax = plt.subplots()

    ax.plot(
        reduced_omega,
        dynamic_shear_modulus,
        label='Master Curve',
        linestyle='None',
        marker='o'
    )

    ax.plot(
        newomega,
        newG_CA,
        label='CA Model',
        linestyle='-',
        marker='None'
    )

    ax.set_title(
        'Plot of |G*| vs Reduced Angular Frequency'
    )

    ax.set_xlabel('ω (Rad/s)')
    ax.set_ylabel('|G*| (MPa)')

    ax.set_xscale('log')
    ax.set_yscale('log')

    fig.text(
        0.20,
        0.65,
        '|G*| = $G_{g}$[1+($ω_{C}$/ω)$^{β}$]$^{(-1/β)}$'
    )
    
    fig.text(
    0.20,
    0.60,
    f'$R^2$ = {r2_ca:.4f}'
    )

    fig.text(
    0.20,
    0.55,
    f'RMSE(log) = {rmse_ca:.4f}'
    )

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels)

    return fig

#-----------------------------------------------------------------



def stiffness(T1, allresults):
    coeffs = allresults.loc[
        allresults['Temperature (C)'] == T1,
        ['A', 'B', 'C']]
    return 10**(coeffs @ POLY_MATRIX)

def shift_factor_error(T1, T2, a, allresults):
    coeffs_T1 = allresults.loc[
        allresults['Temperature (C)'] == T1,
        ['A', 'B', 'C']]
    
    coeffs_T2 = allresults.loc[
        allresults['Temperature (C)'] == T2,
        ['A', 'B', 'C']]

    stiffness_T1 = 10**(coeffs_T1@ POLY_MATRIX)
    stiffness_T2 = 10**(coeffs_T2@ POLY_MATRIX)
    
    logt = LOG_BBR_TIMES

    logtReduced = np.concatenate([
        logt,
        logt - a
    ])
    
    logS = list(np.log10(stiffness_T1.iloc[0,:]))
    logS.extend(list(np.log10(stiffness_T2.iloc[0,:])))
    _, _, r_value3, _, _ = stats.linregress(logtReduced, logS)
    return 1-abs(r_value3)
            
def shift_factor_objective(
    x,
    T1,
    T2,
    allresults):
    return shift_factor_error(
        T1,
        T2,
        x[0],
        allresults)

def gpl_objective(
    params,
    reduced_time,
    creep_compliance):
    m, logD0, logD1 = params

    creep_comp_calc = (
        10**logD0
        + 10**logD1 * reduced_time**m)

    return np.sum(
        (np.log10(creep_compliance)
            - np.log10(creep_comp_calc)
        )**2)

def ca_objective(
    params,
    reduced_omega,
    dynamic_shear_modulus,
    glassy_modulus):
    beta, logOmegaC = params

    G_calc_CA = (
        glassy_modulus
        * (
            1
            + (10**logOmegaC/reduced_omega)**beta
        )**(-1/beta))

    return np.sum(
        (np.log10(G_calc_CA)
            - np.log10(dynamic_shear_modulus))**2)

def fatigue_objective(
    T,
    target_value,
    slope4,
    Tref,
    beta,
    logOmegaC,
    glassy_modulus):
    T = T[0]

    omega_red = (
        10
        * 10**(
            slope4 *
            (
                1/celsius_to_kelvin(T)
                - 1/celsius_to_kelvin(Tref)
            )))

    phase = (90 / (1 + (omega_red / (10**logOmegaC))**beta))

    G = (1000*glassy_modulus * (1 + (10**logOmegaC/omega_red)**beta)**(-1/beta))

    return (target_value - G*np.sin(np.radians(phase)))**2

def pavel_kriz_objective(
    T,
    slope4,
    Tref,
    beta,
    logOmegaC,
    glassy_modulus
                ):
    T = T[0]
    omega_red_T_pavel_kriz = 10*10**(slope4*(1/celsius_to_kelvin(T)-1/celsius_to_kelvin(Tref)))
    #phase_pavel_kriz = 90/(1+(omega_red_T_pavel_kriz/(10**result_CA.x[1]))**result_CA.x[0])
    G_pavel_kriz = 1000*glassy_modulus*(1+(10**logOmegaC/omega_red_T_pavel_kriz)**beta)**(-1/beta)
    return (PAVEL_KRIZ_MODULUS - G_pavel_kriz)**2

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
st.title("BBRtoDSR Data Processor (beta release)")

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

#optimize_glassy_modulus = st.sidebar.checkbox(
#    f"Optimize glassy shear modulus ($G_{'g'}$)",
#    value=False
#)

glassy_modulus = st.sidebar.number_input(
    f"Glassy Shear Modulus ($G_{'g'}$) in MPa",
    min_value=200.0,
    max_value=2000.0,
    value=float(DEFAULT_GLASSY_MODULUS_MPA),
    step=100.0,
    help="Glassy shear modulus used in the Christensen-Anderson model."
)

generate_animation = st.sidebar.checkbox(
    "Generate master curve animation"
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
                    
                    data, results, model = fit_bbr_curve(data)
                    
                    temperature = np.float64(info[2][4])
                except Exception as e:
                    st.error(f"The file is not compatible: {e}. num_lines>44")
            
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
                    
                    data, results, model = fit_bbr_curve(data)
                    
                    temperature = np.float64(dfhead.iloc[6,1])
                except Exception as e:
                    st.error(f"The file is not compatible: {e}. num_lines=44")
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
                                                   10**(model(np.log10(REFERENCE_BBR_TIME))),
                                                   abs(2*model.coefficients[0]*np.log10(REFERENCE_BBR_TIME)+model.coefficients[1])]
                fig = create_plot(results)
                st.pyplot(fig)
            except Exception as e:
                st.write(f"There was an error reading the file: {e}. CSV file type.")
        
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
                
                data, results, model = fit_bbr_curve(data)
                
                
            except Exception as e:
                st.error(f"Error reading XLSM file: {e}.")
            
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
            except Exception as e:
                st.write(f"There was an error reading the file: {e}.")
        else:
            st.error("Unsupported file type.") 

    #
if st.button("Give me the DSR Results !"):
    st.markdown("""---""")
    st.subheader("Low Temperature Properties")
    allresults.sort_values('Temperature (C)',
                                        axis=0,
                                        ascending=False,inplace=True)
    allresults.reset_index(drop=True, inplace=True)
    st.dataframe(allresults , hide_index = True)
    
    if len(allresults)>=2:
            
            low_temp = calculate_low_temperature_properties(
                allresults
            )
            
            T_s = low_temp["Tc_S"]
            T_m = low_temp["Tc_m"]
            Delta_Tc = low_temp["Delta_Tc"]
            
            st.write(f"**$T_{{{'c,S'}}}$: {T_s} °C**")
            st.write(f"**$T_{{{'c,m'}}}$: {T_m} °C**")
            st.write(f"**$Δ T_{'c'}$: {Delta_Tc} °C**")

            st.markdown("""---""")
        
            st.subheader("**Time-temperature Superposition, Shift factor using Arrhenius law**")
            
            st.write(f"**Reference Temperature, $T_{{{'ref'}}}$: {allresults['Temperature (C)'][0]} °C**")           
           
            tts = compute_tts(allresults)
            
            a_T_list = tts["a_T_list"]
            temperatures = tts["temperatures"]         
            reduced_time_list = tts["reduced_time_list"]
            stiffness_list = tts["stiffness_list"]
            shift_data = tts["shift_data"]
            master_curve_series = tts["master_curve_series"]
            slope4 = tts["slope4"]
            r_squared_Arrhenius = tts["r_squared_Arrhenius"]
            shift_factor_values = tts["shift_factor_values"]
            arrhenius_values = tts["arrhenius_values"]
            
            activation_energy = (
                                    tts["slope4"]
                                    * np.log(10)
                                    * GAS_CONSTANT
                                    / 1000
                                )
           

            for item in shift_data:

                st.write(
                    f"**$loga_{{T={item['temperature']:.1f}°C}}$: "
                    f"{item['cumulative_shift']:.2f}**"
                )
                
            
            arrhenius_fig = create_arrhenius_plot(
                                            temperatures,
                                            shift_factor_values,
                                            arrhenius_values,
                                            r_squared_Arrhenius
                                        )
            st.pyplot(arrhenius_fig)
            
            st.write(f"**$E_{'a'}$: {round(activation_energy, 3)} kJ/mol**")
            st.write(f"**R is the universal gas constant which is equal to 8.31446261815324 $J$⋅$K^{{{'−1'}}}$⋅$mol^{{{'−1'}}}$**")
            st.write("**Please note that the temperature is converted to Kelvin, and 'ln' in the function refers to the natural logarithm.**")
            st.write(f"**An $r^{2}$ value below 0.98 raises concerns and warrants rechecking the BBR test data.**")
            
            st.markdown("""---""")
            
            master_curve_fig = create_master_curve_plot(master_curve_series)
            st.pyplot(master_curve_fig)

            st.markdown("""---""")
            st.subheader("**Creep Compliance Master Curve, Generalized Power Law (GPL)**")
            
            gpl = compute_gpl(
                                reduced_time_list,
                                stiffness_list
                             )
            m = gpl["m"]
            logD0 = gpl["logD0"]
            logD1 = gpl["logD1"]
            
            creep_comp_list = gpl["creep_comp_list"]
            reduced_time = gpl["reduced_time"]
            creep_compliance = gpl["creep_compliance"]
            newtime = gpl["newtime"]
            newcreepcom = gpl["newcreepcom"]
            
            r2_gpl = gpl["r2"]
            rmse_log = gpl["rmse_log"]
            
            
            if not gpl["success"] and r2_gpl < 0.99:
                st.warning(f"GPL optimizer message: {gpl['message']}")
            
            
            st.write(f"**m: {round(m,3)}**")
            st.write(f"**log$D_{0}$: {round(logD0,3)}**")
            st.write(f"**log$D_{1}$: {round(logD1,3)}**")

            gpl_fig = create_gpl_plot(
                    reduced_time_list,
                    creep_comp_list,
                    newtime,
                    newcreepcom,
                    round(r2_gpl, 4),
                    round(rmse_log, 4)
                )
                
            st.pyplot(gpl_fig)


            st.markdown("""---""")
            st.subheader("**Complex Modulus Master Curve, Christensen–Anderson (CA) Model**")
            
            ca = compute_ca(
                reduced_time,
                logD0,
                logD1,
                m,
                poissons_ratio,
                glassy_modulus
            )
            
            beta = ca["beta"]
            logOmegaC = ca["logOmegaC"]
            reduced_omega = ca["reduced_omega"]
            dynamic_shear_modulus = ca["dynamic_shear_modulus"]
            newomega = ca["newomega"]
            newG_CA = ca["newG_CA"]
            newphase_CA = ca["newphase_CA"]
            r2_CA = ca["r2"]
            rmse_CA = ca["rmse"]
            
            
            logomega_C_zero = np.log10((10**logOmegaC)*(10**(slope4*(1/celsius_to_kelvin(0)-1/celsius_to_kelvin(allresults['Temperature (C)'][0])))))
            
            if not ca["success"] and r2_CA < 0.99:
                st.warning(f"CA optimizer message: {ca['message']}")
            
            
            st.write(f"**β: {round(beta,3)}**")
            st.write(f"**log$ω_{'C'}$: {round(logOmegaC,3)} at $T_{{{"ref"}}}$**")
        

            st.write(f"**log$ω_{'C'}$: {round(logomega_C_zero,3)} at 0°C**")
            st.write(f"**Rheological Index: {round(np.log10(2)/beta, 2)}**")
                        
            
            ca_fig = create_ca_plot(
                reduced_omega,
                dynamic_shear_modulus,
                newomega,
                newG_CA,
                r2_CA,
                rmse_CA
            )
            
            st.pyplot(ca_fig)

        
            
            st.write(f"**The glassy modulus ($G_{'g'}$) was assumed to be a constant value of {glassy_modulus/1000: .1f} GPa.**")
        

            st.markdown("""---""")
            st.subheader("**Glover-Rowe Parameter, Cracking Performance**")

        
            omega_GR = 0.005
            omega_GR_reduced = omega_GR * 10**(slope4*(1/celsius_to_kelvin(15)-1/celsius_to_kelvin(allresults['Temperature (C)'][0])))
            phase_GR = 90/(1+(omega_GR_reduced/(10**logOmegaC))**beta)
            G_GR = 1000*glassy_modulus*(1+(10**logOmegaC/omega_GR_reduced)**beta)**(-1/beta)
            
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

            st.write("**The Glover-Rowe parameter, which is based on the DSR Fn proposed by [Glover et al.](https://rosap.ntl.bts.gov/view/dot/81884), was originally developed to relate dynamic rheological properties in shear mode to tensile failure strain (ductility) in tension mode for pure or unmodified bitumen (asphalt binder). If the tested sample is modified, however, care should be taken, as high G-R values do not necessarily correlate with low ductility values.**")
        
            st.markdown("""---""")
            
            
            result_T_fatigue = minimize(
                fatigue_objective,
                [22],
                args=(
                    FATIGUE_LIMIT,
                    slope4,
                    allresults['Temperature (C)'][0],
                    beta,
                    logOmegaC,
                    glassy_modulus
                )
            )

            st.subheader("**Fatigue Cracking Criteria**")
            st.write(f"**$T_{{{'G"=5000kPa'}}}$: {round(result_T_fatigue.x[0],1)} °C**")

            result_T_fatigue_6000 = minimize(
                                            fatigue_objective,
                                            [22],
                                            args=(
                                                FATIGUE_LIMIT_ALT,
                                                slope4,
                                                allresults['Temperature (C)'][0],
                                                beta,
                                                logOmegaC,
                                                glassy_modulus
                                            )
                                        )

            st.write(f"**$T_{{{'G"=6000kPa'}}}$: {round(result_T_fatigue_6000.x[0],1)} °C**")

            omega_fatigue6_superpave = 10
            omega_fatigue6_superpave_reduced = omega_fatigue6_superpave * 10**(slope4*(1/celsius_to_kelvin(result_T_fatigue_6000.x[0])-1/celsius_to_kelvin(allresults['Temperature (C)'][0])))
            phase_fatigue6 = 90/(1+(omega_fatigue6_superpave_reduced/(10**logOmegaC))**beta)

            st.write(f"**$δ_{{{'G"=6000kPa'}}}$: {round(phase_fatigue6,1)} °**")
        
            Temperature_fatigue_list = np.array([4, 7, 10, 13, 16, 19, 22, 25, 28, 31, 34, 37, 40])
            Omega_fatigue_list = 10 * 10**(slope4*(1/celsius_to_kelvin(Temperature_fatigue_list)-1/celsius_to_kelvin(allresults['Temperature (C)'][0])))
            phase_fatigue_list = 90/(1+(Omega_fatigue_list/(10**logOmegaC))**beta)
            G_fatigue_list = 1000*glassy_modulus*(1+(10**logOmegaC/Omega_fatigue_list)**beta)**(-1/beta)
            G_storage_fatigue_list = G_fatigue_list * np.cos(np.radians(phase_fatigue_list))
            G_loss_fatigue_list = G_fatigue_list * np.sin(np.radians(phase_fatigue_list))
            
            fatigue_list = pd.DataFrame(columns=['Temperature (°C)','Phase Angle (°)','|G*| (kPa)',"G'=|G*|cosẟ (kPa)",'G"=|G*|sinẟ (kPa)'])
            fatigue_list['Temperature (°C)'] = Temperature_fatigue_list
            fatigue_list['Phase Angle (°)'] = np.round(phase_fatigue_list,1)
            fatigue_list['|G*| (kPa)'] = np.round(G_fatigue_list,0)
            fatigue_list["G'=|G*|cosẟ (kPa)"] = np.round(G_storage_fatigue_list,0)
            fatigue_list['G"=|G*|sinẟ (kPa)'] = np.round(G_loss_fatigue_list,0)

            st.dataframe(fatigue_list, hide_index = True)

            fig7, ax7 = plt.subplots()
            ax7.plot(phase_fatigue_list, G_fatigue_list, label='Superpave Fatigue Points, ω = 10 Rad/s', 
                     linestyle='None',
                     marker='o', alpha=0.7, markersize=2, c='red')
            ax7.plot(np.arange(1,89,1),FATIGUE_LIMIT/np.sin(np.radians(np.arange(1,89,1))),label='|G*|sinδ = 5000 kPa',linestyle='--',marker='None',c='black')
            ax7.plot(np.arange(1,89,1),FATIGUE_LIMIT_ALT/np.sin(np.radians(np.arange(1,89,1))),label='|G*|sinδ = 6000 kPa',linestyle='-.',marker='None',c='black',alpha=0.3)
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
            st.subheader("**Pavel-Kriz Phase Angle, [Detection of Phase Incompatible Binders](https://trid.trb.org/View/2344464/)**")

            initial_data_T_pavel_kriz = [22]
            
            
            result_T_pavel_kriz = minimize(
                                            pavel_kriz_objective,
                                            [22],
                                            args=(
                                                slope4,
                                                allresults['Temperature (C)'][0],
                                                beta,
                                                logOmegaC,
                                                glassy_modulus
                                            )
                                        )
        
            Omega_pavel_kriz = 10 * 10**(slope4*(1/celsius_to_kelvin(result_T_pavel_kriz.x[0])-1/celsius_to_kelvin(allresults['Temperature (C)'][0])))
            phase_pavel_kriz = 90/(1+(Omega_pavel_kriz/(10**logOmegaC))**beta)

            st.write("ω = 10 Rad/s")
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
            
            
            if generate_animation:

                st.markdown("""---""")
                st.subheader("**Animated Master Curve Shifting**")
    
                fig9, ax9 = plt.subplots()
                ax9.set_yscale('log')
                
                movingshifts = np.insert(np.cumsum(a_T_list),0,0)
                movingtime = np.array(LOG_BBR_TIMES)
                
                lines = [ax9.plot([], [], label=f'T = {round(tem,2)} °C', antialiased=True, alpha = 0.4, linewidth = 5)[0] for tem in allresults['Temperature (C)']]
                
                # Initialization function
                def init():
                    for line in lines:
                        line.set_data([], [])
                    return lines
                
                stiffness_cache = []
                
                for temp in allresults['Temperature (C)']:
                    stiffness_cache.append(
                        stiffness(temp, allresults).iloc[0,:])
                
                # Animation function
                def animate(i):
                    for idx, shift in enumerate(movingshifts):
                        shifted_time = movingtime - (0.01 * i * shift)
                        mod = stiffness_cache[idx]
                        lines[idx].set_data(shifted_time, mod)
                    return lines
    
                ax9.set_title('Plot of Stiffness Master Curve vs Reduced Time')
                ax9.set_xlabel('Log Time (s)')
                ax9.set_ylabel('Stiffness (MPa)')
                handles9, labels9 = ax9.get_legend_handles_labels()
                ax9.legend(handles9, labels9)
                
                ax9.set_xlim(round(movingtime.min()-movingshifts.max(),0)-1, round(movingtime.max(),0)+1)
                ax9.set_ylim(10**(round(np.log10(stiffness(allresults['Temperature (C)'][0], allresults).iloc[0,:].min()),0)-1), 
                             10**(round(np.log10(stiffness(allresults['Temperature (C)'][0], allresults).iloc[0,:].max()),0)+1))
    
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





















































