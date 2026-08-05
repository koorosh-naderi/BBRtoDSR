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
from dataclasses import dataclass

#magic numbers

#Seconds
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
#MPa
STIFFNESS_LIMIT = 300

#Seconds
REFERENCE_BBR_TIME = 60

#kPa
FATIGUE_LIMIT = 5000
FATIGUE_LIMIT_ALT = 6000

#kPa
PAVEL_KRIZ_MODULUS_ORIGINAL = 6000 / np.sin(np.deg2rad(42))
PAVEL_KRIZ_MODULUS_10MPA = 10000

DISPLAY_DPI = 450

RI_LINES = np.arange(1,4.5,0.5)

REPLICATE_TEMPERATURE_TOLERANCE = 0.1


plt.rcParams.update({
    'figure.facecolor': '#F8F8F8',
    'axes.facecolor': '#F8F8F8'
})

#-----------------------------------------------------------------

@dataclass
class BBRLoadResult:
    bbr_data: pd.DataFrame
    bbr_fit_points: pd.DataFrame
    bbr_fit_model: np.poly1d
    target_temperature: float
    actual_temperature: float

def create_load_result(
    bbr_data,
    bbr_fit_points,
    bbr_fit_model,
    target_temperature,
    actual_temperature
):
    return BBRLoadResult(
        bbr_data=bbr_data,
        bbr_fit_points=bbr_fit_points,
        bbr_fit_model=bbr_fit_model,
        target_temperature=target_temperature,
        actual_temperature=actual_temperature
    )


def load_csv(uploaded_file):
    uploaded_file.seek(0)
    num_lines = count_lines(uploaded_file)
    

    if num_lines>44:
        
        df = pd.read_csv(uploaded_file,
            header=None,engine='python',names=range(1,6))
    
        info = df.iloc[0:9,0:2]
        bbr_data = df.iloc[9:,:].dropna(axis=1)
        bbr_data.columns = bbr_data.iloc[0]
        bbr_data = bbr_data[1:]
        bbr_data.reset_index(drop=True,inplace=True)
        bbr_data = bbr_data.rename_axis(None, axis=1)
        beam_span = np.float64(info[2][6])/1000
        beam_width = np.float64(info[2][7])/1000
        beam_thickness = np.float64(info[2][8])/1000
        
        force = np.float64(bbr_data['Force (mN)'])
        deflection = np.float64(bbr_data['Deflection (mm)'])
        
        bbr_data['Stiffness (MPa)'] = np.nan

        mask = deflection > 0
        
        bbr_data.loc[mask, 'Stiffness (MPa)'] = (
            1e-6
            * force[mask]
            * beam_span**3
            / (
                deflection[mask]
                * 4
                * beam_width
                * beam_thickness**3
            )
        )
        
        
        #bbr_data['Stiffness (MPa)'] = 1/1000000*(np.float64(bbr_data['Force (mN)'])*beam_span**3)/(np.float64(bbr_data['Deflection (mm)'])*4*beam_width*beam_thickness**3)
        
        target_temperature = np.float64(info[2][4])
        
        bbr_data, bbr_fit_points, bbr_fit_model  = fit_bbr_curve(bbr_data)
        
        
        actual_temperature = np.float64(bbr_fit_points['Temperature (C)']).mean()
            
    elif num_lines==44:    
       
        df = pd.read_csv(uploaded_file, header=None,engine='python', encoding='unicode_escape',skiprows=38)
        uploaded_file.seek(0)
        dfhead = pd.read_csv(uploaded_file,header=None,engine='python', encoding='unicode_escape',nrows=36,index_col=False)
        
        bbr_data = pd.DataFrame()
        bbr_data['Time (s)'] = df[0]
        bbr_data['Force (mN)'] = df[1]
        bbr_data['Deflection (mm)'] = df[2]
        bbr_data['Temperature (C)'] = dfhead.iloc[6,1]
        bbr_data['Stiffness (MPa)'] = df[3]
        target_temperature = np.float64(dfhead.iloc[6,1])
        
        bbr_data, bbr_fit_points, bbr_fit_model = fit_bbr_curve(bbr_data)
        
        actual_temperature = np.float64(bbr_fit_points['Temperature (C)']).mean()
        
    else:
        raise ValueError(
        "Unsupported CSV format."
        )
        
    return create_load_result(
                                        bbr_data,
                                        bbr_fit_points,
                                        bbr_fit_model,
                                        target_temperature,
                                        actual_temperature
                                    )


def load_xlsm(uploaded_file):
    # XLSM handling
    uploaded_file.seek(0)
    sheet_name = 'BBR Results'
    start_row = 100

    df = pd.read_excel(uploaded_file, sheet_name=sheet_name, engine="openpyxl", skiprows=start_row-1,header=None)
    full_sheet = pd.read_excel(uploaded_file, sheet_name=sheet_name, engine="openpyxl", header=None)
        
    bbr_data = pd.DataFrame()
    bbr_data['Time (s)'] = df[0]
    bbr_data['Force (mN)'] = df[1]*1000
    bbr_data['Deflection (mm)'] = df[2]
    bbr_data['Temperature (C)'] = df[3]
    
    beam_span = np.float64(full_sheet.iloc[29,2])/1000
    beam_width = np.float64(full_sheet.iloc[19,8])/1000
    beam_thickness = np.float64(full_sheet.iloc[20,8])/1000
    target_temperature = np.float64(full_sheet.iloc[28,2])
    
    
    force = np.float64(bbr_data['Force (mN)'])
    deflection = np.float64(bbr_data['Deflection (mm)'])
    
    bbr_data['Stiffness (MPa)'] = np.nan

    mask = deflection > 0
    
    bbr_data.loc[mask, 'Stiffness (MPa)'] = (
        1e-6
        * force[mask]
        * beam_span**3
        / (
            deflection[mask]
            * 4
            * beam_width
            * beam_thickness**3
        )
    )
    
    #bbr_data['Stiffness (MPa)'] = 1/1000000*(np.float64(bbr_data['Force (mN)'])*beam_span**3)/(np.float64(bbr_data['Deflection (mm)'])*4*beam_width*beam_thickness**3)
    
    bbr_data, bbr_fit_points, bbr_fit_model = fit_bbr_curve(bbr_data)
    
    actual_temperature = np.float64(bbr_fit_points['Temperature (C)']).mean()
    
    return create_load_result(
                                bbr_data,
                                bbr_fit_points,
                                bbr_fit_model,
                                target_temperature,
                                actual_temperature
                                )


def load_bbr_file(uploaded_file):

    extension = uploaded_file.name.split(".")[-1].lower()

    if extension == "csv":
        return load_csv(uploaded_file)

    elif extension == "xlsm":
        return load_xlsm(uploaded_file)

    raise ValueError(
        f"Unsupported file type: {extension}"
    )



#-----------------------------------------------------------------
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


def calculate_low_temperature_properties(bbr_temperature_results):
    m_value_bracketing_rows = find_bracketing_rows(
        bbr_temperature_results,
        'm-value(60)',
        M_VALUE_LIMIT)

    stiffness_bracketing_rows = find_bracketing_rows(
        bbr_temperature_results,
        'S(60)',
        STIFFNESS_LIMIT)

    m_value_slope, m_value_intercept, _, _, _ = stats.linregress(
        m_value_bracketing_rows['m-value(60)'],
        m_value_bracketing_rows['Temperature (C)'])

    stiffness_slope, stiffness_intercept, _, _, _ = stats.linregress(
        np.log(stiffness_bracketing_rows['S(60)']),
        stiffness_bracketing_rows['Temperature (C)'])

    T_s = round(
        -10 + stiffness_slope * np.log(STIFFNESS_LIMIT)
        + stiffness_intercept,
        1)

    T_m = round(
        -10 + m_value_slope * M_VALUE_LIMIT
        + m_value_intercept,
        1)

    Delta_Tc = round(
        T_s - T_m,
        1)

    return {
        "Tc_S": T_s,
        "Tc_m": T_m,
        "Delta_Tc": Delta_Tc,
        "m_value_slope": m_value_slope,
        "m_value_intercept": m_value_intercept,
        "stiffness_slope": stiffness_slope,
        "stiffness_intercept": stiffness_intercept,
        "m_value_bracketing_rows": m_value_bracketing_rows,
        "stiffness_bracketing_rows": stiffness_bracketing_rows
    }


def celsius_to_kelvin(temp_c):
    return temp_c + KELVIN_OFFSET

def fit_bbr_curve(bbr_data):
    bbr_data = bbr_data.copy()
    bbr_data['Time (s)'] = pd.to_numeric(bbr_data['Time (s)'], errors='coerce')
    
    
    time_values = np.float64(bbr_data['Time (s)'])
    stiffness_values = np.float64(
        bbr_data['Stiffness (MPa)']
    )
    
    bbr_data['log(t)'] = np.nan

    time_mask = time_values > 0
    
    bbr_data.loc[time_mask, 'log(t)'] = np.log10(
        time_values[time_mask]
    )
    
    bbr_data['log(S)'] = np.nan

    stiffness_mask = stiffness_values > 0
    
    bbr_data.loc[stiffness_mask, 'log(S)'] = np.log10(
        stiffness_values[stiffness_mask]
    )
    
    
    
    #bbr_data['log(t)'] = np.log10(
    #    np.float64(bbr_data['Time (s)']))
    #bbr_data['log(S)'] = np.log10(
    #    bbr_data['Stiffness (MPa)'])
    
    bbr_fit_points = bbr_data[
    bbr_data['Time (s)'].isin(
            BBR_TIMES)]
    
    bbr_fit_points = (
        bbr_fit_points
        .replace(
            [np.inf, -np.inf],
            np.nan
        )
        .dropna(
            subset=[
                'log(t)',
                'log(S)'
            ]
        )
    )
    
    #expected_points = len(BBR_TIMES)
    actual_points = len(bbr_fit_points)

    if actual_points < 3:
        raise ValueError(
            f"Only {actual_points} valid BBR points were found. "
            "At least 3 points are required for a quadratic fit."
        )
        
    #fit_warning = None

    #if actual_points < expected_points:
    #    fit_warning = (
    #        f"Only {actual_points} of {expected_points} expected "
    #        "BBR time points were available."
    #    )
    
    bbr_fit_model = np.poly1d(
        np.polyfit(
            bbr_fit_points['log(t)'],
            bbr_fit_points['log(S)'],
            2))
    bbr_data['Sc (MPa)'] = 10**(
    bbr_fit_model(bbr_data['log(t)']))
    bbr_data['Percent diff'] = (
        bbr_data['Stiffness (MPa)']
        - bbr_data['Sc (MPa)']
    ) / bbr_data['Stiffness (MPa)'] * 100
    bbr_data['m-value'] = np.abs(
        2*bbr_fit_model.coefficients[0]
        * bbr_data['log(t)']
        + bbr_fit_model.coefficients[1])
    bbr_fit_points = bbr_data[bbr_data['Time (s)'].isin(BBR_TIMES)]
    return bbr_data, bbr_fit_points, bbr_fit_model

@dataclass
class TTSResult:
    a_T_list: list
    temperatures: list
    reduced_time_list: list
    stiffness_list: list
    shift_data: list
    master_curve_series: list
    arrhenius_slope: float
    r_squared_Arrhenius: float
    shift_factor_values: np.ndarray
    arrhenius_values: np.ndarray
    warnings: list

def compute_tts(bbr_temperature_results):

    a_T_list = []
    master_curve_series = []
    shift_data = []
    warnings = []
    
    reference_temperature = (
    bbr_temperature_results.loc[0, 'Temperature (C)']
    )

    temperatures = [reference_temperature]

    reduced_time_list = BBR_TIMES.tolist()

    stiffness_list = list(
        calculate_stiffness_curve(
            reference_temperature,
            bbr_temperature_results
        ).iloc[0,:]
    )

    master_curve_series.append({
        "temperature": reference_temperature,
        "time": BBR_TIMES,
        "stiffness": calculate_stiffness_curve(
            reference_temperature,
            bbr_temperature_results
        ).iloc[0,:]
    })

    
    for i in range(1,len(bbr_temperature_results)):
        fixed_T1 = bbr_temperature_results.loc[i-1, 'Temperature (C)']
        fixed_T2 = bbr_temperature_results.loc[i, 'Temperature (C)']
        
        current_temperature = (
                                bbr_temperature_results.loc[i, 'Temperature (C)']
                                )
        
        initial_x = [np.log10(7200/60)*(1/celsius_to_kelvin(fixed_T2)-1/celsius_to_kelvin(fixed_T1))/(
            1/(-10+celsius_to_kelvin(fixed_T1))-1/(celsius_to_kelvin(fixed_T1)))]
        result = minimize(shift_factor_objective,
                          initial_x,
                          args=(fixed_T1, fixed_T2, bbr_temperature_results), bounds=[(-10, 10)])
        
        if not result.success:

            warnings.append(
                f"Shift factor optimization failed between "
                f"{fixed_T1:.1f}°C and {fixed_T2:.1f}°C"
            )
        
        
        a_T_list.append(result.x[0])
        temperatures.append(current_temperature)
        
        cumulative_shift = np.cumsum(a_T_list)[i-1]
        
        reduced_time_list.extend(BBR_TIMES/(10**cumulative_shift))
        stiffness_list.extend(calculate_stiffness_curve(current_temperature, bbr_temperature_results).iloc[0,:])
        
        shift_data.append({
            "temperature": current_temperature,
            "shift_factor": result.x[0],
            "cumulative_shift": cumulative_shift
        })
        
        
        master_curve_series.append({
            "temperature": current_temperature,
            "time": BBR_TIMES/(10**cumulative_shift),
            "stiffness": calculate_stiffness_curve(
                current_temperature,
                bbr_temperature_results
            ).iloc[0,:]
        })
    inverse_temperature_difference = np.array([1/celsius_to_kelvin(x)-1/celsius_to_kelvin(temperatures[0]) for x in temperatures])
    logaT_arr = np.insert(np.cumsum(a_T_list),0,0,axis=0)
    
    arrhenius_slope, _ = linear_regression(inverse_temperature_difference, logaT_arr, proportional=True)

    predicted_logaT_arr = [arrhenius_slope * xi for xi in inverse_temperature_difference]
    
    rss = sum((yi - y_pred) ** 2 for yi, y_pred in zip(logaT_arr, predicted_logaT_arr))
    mean_y = sum(logaT_arr) / len(logaT_arr)
    tss = sum((yi - mean_y) ** 2 for yi in logaT_arr)
    if np.isclose(tss, 0):
        r_squared_Arrhenius = np.nan
    else:
        r_squared_Arrhenius = 1 - rss/tss
    #r_squared_Arrhenius = 1 - rss / tss
    
    shift_factor_values = 10**logaT_arr
    arrhenius_values = 10**(arrhenius_slope * inverse_temperature_difference)

    return TTSResult(
    a_T_list=a_T_list,
    temperatures=temperatures,
    reduced_time_list=reduced_time_list,
    stiffness_list=stiffness_list,
    shift_data=shift_data,
    master_curve_series=master_curve_series,
    arrhenius_slope=arrhenius_slope,
    r_squared_Arrhenius=r_squared_Arrhenius,
    shift_factor_values=shift_factor_values,
    arrhenius_values=arrhenius_values,
    warnings=warnings
                    )

   

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
    
    if np.isclose(tss, 0):
        r2_gpl = np.nan
    else:
        r2_gpl = 1 - rss/tss
    
    
    #r2_gpl = 1 - rss / tss
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
    glassy_modulus,
    optimize_glassy_modulus=False
              ):
    
    reduced_omega = 2/(np.pi*reduced_time)
    storage_compliance = (10**logD0) + (10**logD1) * math.gamma(1+m) * (reduced_omega)**(-m) * np.cos(m * np.pi/2)
    loss_compliance = (10**logD1) * math.gamma(1+m) * (reduced_omega)**(-m) * np.sin(m * np.pi/2)
    dynamic_compliance = (storage_compliance**2 + loss_compliance**2)**0.5
    dynamic_modulus = 1/dynamic_compliance
    dynamic_shear_modulus = dynamic_modulus/(2*(1+poissons_ratio))
    

    if optimize_glassy_modulus:

        initial_data_CA = [
            0.1,
            -3,
            np.log10(glassy_modulus)
        ]
    
        result_CA = minimize(
            ca_objective_opt,
            initial_data_CA,
            args=(
                reduced_omega,
                dynamic_shear_modulus
            ),
            bounds=[
                (0.01, 5),
                (-10, 10),
                (2.3, 4)
            ]
        )

    else:

        initial_data_CA = [0.1,-3]
    
        result_CA = minimize(
            ca_objective,
            initial_data_CA,
            args=(
                reduced_omega,
                dynamic_shear_modulus,
                glassy_modulus
            ),
            bounds=[(0.01, 5), (-10, 10)]
        )
    
    if optimize_glassy_modulus:
        beta = result_CA.x[0]
        logOmegaC = result_CA.x[1]
        logGg = result_CA.x[2]
        fitted_Gg = 10**logGg
    else:
        beta = result_CA.x[0]
        logOmegaC = result_CA.x[1]
        fitted_Gg = glassy_modulus
    
    
    
    newomega = 10**np.linspace(np.log10(reduced_omega).min(),np.log10(reduced_omega).max(),50)
    newG_CA = fitted_Gg*(1+(10**logOmegaC/newomega)**beta)**(-1/beta)
    newphase_CA = 90/(1+(newomega/(10**logOmegaC))**beta)
    
    predicted_dynamic_shear_modulus = (
    fitted_Gg
    * (
        1 +
        (10**logOmegaC/reduced_omega)
        **beta
    )**(-1/beta)
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
    
    if np.isclose(tss, 0):
        r2_ca = np.nan
    else:
        r2_ca = 1 - rss/tss
    
    
    #r2_ca = 1 - rss/tss
    
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
    "glassy_modulus": fitted_Gg
}


def compute_gr(
    arrhenius_slope,
    Tref,
    beta,
    logOmegaC,
    glassy_modulus
):
    
    omega_gr = 0.005

    omega_gr_reduced = (
        omega_gr
        * 10**(
            arrhenius_slope
            * (
                1/celsius_to_kelvin(15)
                - 1/celsius_to_kelvin(Tref)
            )
        )
    )

    phase_gr = (90/(1 + (omega_gr_reduced/(10**logOmegaC))**beta))

    g_gr = (
        1000
        * glassy_modulus
        * (
            1
            +
            (
                10**logOmegaC
                /
                omega_gr_reduced
            )**beta
        )**(-1/beta)
    )

    g_r = (
        g_gr
        / np.sin(np.radians(phase_gr))
    ) * (
        np.cos(np.radians(phase_gr))
    )**2

    return {
        "omega_gr_reduced": omega_gr_reduced,
        "phase_gr": phase_gr,
        "g_gr": g_gr,
        "g_r": g_r
    }


def compute_fatigue(
    arrhenius_slope,
    Tref,
    beta,
    logOmegaC,
    glassy_modulus
):

    result_T_fatigue = minimize(
        fatigue_objective,
        [22],
        args=(
            FATIGUE_LIMIT,
            arrhenius_slope,
            Tref,
            beta,
            logOmegaC,
            glassy_modulus
        )
    )

    result_T_fatigue_6000 = minimize(
        fatigue_objective,
        [22],
        args=(
            FATIGUE_LIMIT_ALT,
            arrhenius_slope,
            Tref,
            beta,
            logOmegaC,
            glassy_modulus
        )
    )

    omega_fatigue6_superpave = 10

    omega_fatigue6_superpave_reduced = (
        omega_fatigue6_superpave
        * 10**(
            arrhenius_slope
            * (
                1/celsius_to_kelvin(
                    result_T_fatigue_6000.x[0]
                )
                - 1/celsius_to_kelvin(Tref)
            )
        )
    )

    phase_fatigue6 = (
        90
        /
        (
            1
            +
            (
                omega_fatigue6_superpave_reduced
                /
                (10**logOmegaC)
            )**beta
        )
    )

    Temperature_fatigue_list = np.array(
        [4, 7, 10, 13, 16, 19, 22, 25, 28, 31, 34, 37, 40]
    )

    Omega_fatigue_list = (
        10
        * 10**(
            arrhenius_slope
            * (
                1/celsius_to_kelvin(
                    Temperature_fatigue_list
                )
                - 1/celsius_to_kelvin(Tref)
            )
        )
    )

    phase_fatigue_list = (
        90
        /
        (
            1
            +
            (
                Omega_fatigue_list
                /
                (10**logOmegaC)
            )**beta
        )
    )

    G_fatigue_list = (
        1000
        * glassy_modulus
        * (
            1
            +
            (
                10**logOmegaC
                /
                Omega_fatigue_list
            )**beta
        )**(-1/beta)
    )

    G_storage_fatigue_list = (
        G_fatigue_list
        * np.cos(
            np.radians(
                phase_fatigue_list
            )
        )
    )

    G_loss_fatigue_list = (
        G_fatigue_list
        * np.sin(
            np.radians(
                phase_fatigue_list
            )
        )
    )
    
    fatigue_table = pd.DataFrame(
    columns=[
        'Temperature (°C)',
        'Phase Angle (°)',
        '|G*| (kPa)',
        "G'=|G*|cosδ (kPa)",
        'G"=|G*|sinδ (kPa)'
    ]
    )
    
    fatigue_table['Temperature (°C)'] = Temperature_fatigue_list
    
    fatigue_table['Phase Angle (°)'] = np.round(
        phase_fatigue_list,
        1
    )
    
    fatigue_table['|G*| (kPa)'] = np.round(
        G_fatigue_list,
        0
    )
    
    fatigue_table["G'=|G*|cosδ (kPa)"] = np.round(
        G_storage_fatigue_list,
        0
    )
    
    fatigue_table['G"=|G*|sinδ (kPa)'] = np.round(
        G_loss_fatigue_list,
        0
    )

    return {
        "T5000": result_T_fatigue.x[0],
        "T6000": result_T_fatigue_6000.x[0],
        "phase6000": phase_fatigue6,
        "Temperature_fatigue_list": Temperature_fatigue_list,
        "phase_fatigue_list": phase_fatigue_list,
        "G_fatigue_list": G_fatigue_list,
        "G_storage_fatigue_list": G_storage_fatigue_list,
        "G_loss_fatigue_list": G_loss_fatigue_list,
        "fatigue_table": fatigue_table,
    }

def compute_pavel_kriz(
    arrhenius_slope,
    Tref,
    beta,
    logOmegaC,
    glassy_modulus,
    target_modulus
):

    result_T_pavel_kriz = minimize(
        pavel_kriz_objective,
        [22],
        args=(
            arrhenius_slope,
            Tref,
            beta,
            logOmegaC,
            glassy_modulus,
            target_modulus
        )
    )

    temperature = result_T_pavel_kriz.x[0]

    omega = (
        10
        * 10**(
            arrhenius_slope
            * (
                1/celsius_to_kelvin(temperature)
                - 1/celsius_to_kelvin(Tref)
            )
        )
    )

    phase = (
        90
        /
        (
            1
            +
            (
                omega
                /
                (10**logOmegaC)
            )**beta
        )
    )

    return {
        "temperature": temperature,
        "phase": phase,   
    }

# Function to create plots-------------------------------------------------------------------

def add_reference_temperature_label(
    ax,
    reference_temperature
):
    ax.text(
        0.03,
        0.97,
        f"$T_{{{'ref'}}}$ = {reference_temperature:.1f} °C",
        transform=ax.transAxes,
        verticalalignment='top',
        bbox=dict(
            facecolor='white',
            alpha=0.8
        )
    )


def create_bbr_fit_plot(bbr_fit_points):
    fig, ax = plt.subplots(dpi=DISPLAY_DPI)
    
    
    ax.plot(bbr_fit_points['Time (s)'], bbr_fit_points['Sc (MPa)'],label='Estimated', linestyle='-', c='#634B71')
    ax.plot(bbr_fit_points['Time (s)'], bbr_fit_points['Stiffness (MPa)'],label='Measured', linestyle=':', marker='o', c='#EE7402', alpha=0.5)
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

    fig, ax = plt.subplots(dpi=DISPLAY_DPI)
    


    ax.plot(
        np.array(temperature_list),
        shift_factor_values,
        label='Shift Factors',
        linestyle='None',
        marker='o', c='#634B71'
    )

    ax.plot(
        np.array(temperature_list),
        arrhenius_values,
        label='Arrhenius Model',
        linestyle='-',
        marker='None',
        c='#EE7402'
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


def create_low_temperature_properties_plot(
    bbr_temperature_results,
    low_temp
):

    fig, (ax1, ax2) = plt.subplots(
        1, 2,
        figsize=(10,5),
        dpi=DISPLAY_DPI
    )

    # ------------------------
    # Stiffness criterion
    # ------------------------

    ax1.plot(
        bbr_temperature_results['Temperature (C)'],
        bbr_temperature_results['S(60)'],
        label='Measured',ls='-',marker='o',c='#C0CE2E'
    )


    ax1.axhline(
        STIFFNESS_LIMIT,
        ls='--',
        c='k'
    )

    ax1.axvline(
        low_temp["Tc_S"]+10,
        ls=':',
        c='#EE7402',
        label='S(60)=300 MPa'
    )

    ax1.plot(
        low_temp["Tc_S"]+10,
        STIFFNESS_LIMIT,marker='o',
        c='#EE7402'
    )

    ax1.set_yscale('log')
    ax1.set_xlabel('Test Temperature (°C)')
    ax1.set_ylabel('S(60) (MPa)')
    ax1.set_title(f'$T_{{{"c,S"}}}$ = {low_temp["Tc_S"]:.1f} °C')

    # ------------------------
    # m-value criterion
    # ------------------------

    ax2.plot(
        bbr_temperature_results['Temperature (C)'],
        bbr_temperature_results['m-value(60)'],
        label='Measured',ls='-',marker='o',c='#A3186D'
    )


    ax2.axhline(
        M_VALUE_LIMIT,
        ls='--',
        c='k'
    )

    ax2.axvline(
        low_temp["Tc_m"]+10,
        ls=':',
        c='#634B71',
        label='m-value(60)=0.300'
    )

    ax2.plot(
        low_temp["Tc_m"]+10,
        M_VALUE_LIMIT,marker='o',
        c='#634B71'
    )

    ax2.set_xlabel('Test Temperature (°C)')
    ax2.set_ylabel('m-value(60)')
    ax2.set_title(f'$T_{{{"c,m"}}}$ = {low_temp["Tc_m"]:.1f} °C')

    fig.suptitle(
        f'$ΔT_c$ = {low_temp["Delta_Tc"]:.1f} °C'
    )

    ax1.legend()
    ax2.legend()

    return fig


def create_master_curve_plot(master_curve_series,
                             reference_temperature):

    fig, ax = plt.subplots(dpi=DISPLAY_DPI)

    for curve in master_curve_series:

        ax.plot(
            curve["time"],
            curve["stiffness"],
            label=f'{round(curve["temperature"], 1)} °C',
            linestyle='-',
            linewidth=5.0,
            alpha=0.4,
            marker='o'
        )

    
    ax.set_title('Stiffness Master Curve')
    add_reference_temperature_label(ax, reference_temperature)
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
    rmse_log,
    reference_temperature
):

    fig, ax = plt.subplots(dpi=DISPLAY_DPI)

    ax.plot(
        reduced_time_list,
        creep_comp_list,
        label='Master Curve',
        linestyle='None',
        marker='o', c='#634B71'
    )

    ax.plot(
        newtime,
        newcreepcom,
        label='GPL Model',
        linestyle='-',
        marker='None',
        c='#EE7402'
    )
    
    ax.set_title('Creep Compliance Master Curve')
    add_reference_temperature_label(ax, reference_temperature)

    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Creep Compliance (1/MPa)')

    ax.set_xscale('log')
    ax.set_yscale('log')

    fig.text(
        0.20,
        0.60,
        '$D$($t$) = $D_{0}$ + $D_{1}$.$t^m$'
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
    rmse_ca,
    reference_temperature
):

    fig, ax = plt.subplots(dpi=DISPLAY_DPI)

    ax.plot(
        reduced_omega,
        dynamic_shear_modulus,
        label='Master Curve',
        linestyle='None',
        marker='o', c='#634B71'
    )

    ax.plot(
        newomega,
        newG_CA,
        label='CA Model',
        linestyle='-',
        marker='None',
        c='#EE7402'
    )

    
    ax.set_title('Complex Modulus Master Curve')
    add_reference_temperature_label(ax, reference_temperature)

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


def create_glover_rowe_plot(
    phase_gr,
    g_gr,
    show_RI_contours=True
):
    
    fig, ax = plt.subplots(dpi=DISPLAY_DPI)

    # RI lines
    if show_RI_contours:

        x = np.arange(-10, 10.5, 0.01)

        for i in RI_LINES:

            phase_x = (
                90 /
                (
                    1 +
                    ((10**x)/1)**(np.log10(2)/i)
                )
            )
    
            g_x = (
                1e6
                * (
                    1 +
                    (1/10**x)**(np.log10(2)/i)
                )**(i/-np.log10(2))
            )
    
            ax.plot(
                phase_x,
                g_x,
                color='lightgray',
                ls='--',
                alpha=0.3,
                zorder=1
            )
    
            target_phase = 40 + 2*(i + 0.25)
    
            idx = np.argmin(
                np.abs(
                    phase_x - target_phase
                )
            )
    
            dy = (
                np.log10(g_x[idx+1])
                -
                np.log10(g_x[idx-1])
            )
    
            dx = (
                phase_x[idx+1]
                -
                phase_x[idx-1]
            )
    
            ANGLE_CORRECTION = 7
    
            angle = (
                ANGLE_CORRECTION
                * np.degrees(
                    np.arctan(dy/dx)
                )
            )
    
            ax.text(
                phase_x[idx],
                g_x[idx],
                f"R={i:.1f}",
                rotation=angle,
                rotation_mode='anchor',
                ha='center',
                va='center',
                fontsize=8,
                color='gray',
                alpha=0.3,
                zorder=3,
                bbox=dict(
                    facecolor='#F8F8F8',
                    alpha=0.5,
                    edgecolor='none'
                )
            )

    # Existing G-R graphics

    ax.plot(
        phase_gr,
        g_gr,
        label='G-R Parameter',
        linestyle='None',
        marker='o',
        zorder=100, c='#634B71'
    )

    ax.plot(
        np.arange(1,89,1),
        180*np.sin(np.radians(np.arange(1,89,1)))
        /np.cos(np.radians(np.arange(1,89,1)))**2,
        label='G-R = 180 kPa',
        linestyle='-',
        c='#C0CE2E',
        zorder=2
    )

    ax.plot(
        np.arange(1,89,1),
        600*np.sin(np.radians(np.arange(1,89,1)))
        /np.cos(np.radians(np.arange(1,89,1)))**2,
        label='G-R = 600 kPa',
        linestyle='-',
        c='#A3186D',
        zorder=2
    )

    ax.set_title('Black Diagram')
    ax.set_xlabel('Phase Angle (°)')
    ax.set_ylabel('|G*| (kPa)')

    ax.set_xlim(0,90)
    ax.set_ylim(0.01,1e6)

    ax.set_yscale('log')

    ax.legend()

    return fig


def create_fatigue_plot(
    phase_fatigue_list,
    G_fatigue_list,
    Temperature_fatigue_list
):

    fig, ax = plt.subplots(dpi=DISPLAY_DPI)

    ax.plot(
        phase_fatigue_list,
        G_fatigue_list,
        label='Superpave Fatigue Points, ω = 10 Rad/s',
        linestyle='None',
        marker='o',
        alpha=0.7,
        markersize=2,
        c='#EE7402'
    )

    ax.plot(
        np.arange(1,89,1),
        FATIGUE_LIMIT
        / np.sin(
            np.radians(
                np.arange(1,89,1)
            )
        ),
        label='|G*|sinδ = 5000 kPa',
        linestyle='--',
        c='black'
    )

    ax.plot(
        np.arange(1,89,1),
        FATIGUE_LIMIT_ALT
        / np.sin(
            np.radians(
                np.arange(1,89,1)
            )
        ),
        label='|G*|sinδ = 6000 kPa',
        linestyle='-.',
        c='black',
        alpha=0.3
    )

    ax.set_title('Black Diagram')

    ax.set_xlabel('Phase Angle (°)')
    ax.set_ylabel('|G*| (kPa)')

    ax.set_yscale('log')

    ax.set_ylim(1,1e6)

    ax.set_xlim(0,90)

    for x, y, z in zip(
        phase_fatigue_list,
        G_fatigue_list,
        Temperature_fatigue_list
    ):
        ax.text(
            x,
            y,
            f"{z}°C",
            fontsize=7
        )

    ax.legend()

    return fig

def create_pavel_kriz_plot(
    newphase_CA,
    newG_CA,
    phase_pavel_kriz,
    pavel_kriz_modulus
):

    fig, ax = plt.subplots(dpi=DISPLAY_DPI)

    ax.plot(
        newphase_CA,
        1000 * newG_CA,
        label='CA Model Points',
        linestyle='--',
        marker='',
        alpha=0.6,
        c='#634B71'
    )

    ax.plot(
        phase_pavel_kriz,
        pavel_kriz_modulus,
        label='Pavel-Kriz Point',
        linestyle='None',
        marker='o',
        alpha=0.6,
        zorder=100, c='#EE7402'
    )

    ax.vlines(
        x=42,
        ymin=pavel_kriz_modulus,
        ymax=1e6,
        linestyle='dashdot',
        colors='black',
        label='Pavel-Kriz Criteria'
    )

    ax.hlines(
        y=pavel_kriz_modulus,
        xmin=0,
        xmax=90,
        linestyle='dashdot',
        colors='black'
    )

    ax.set_title('Black Diagram')

    ax.set_xlabel('Phase Angle (°)')
    ax.set_ylabel('|G*| (kPa)')

    ax.set_yscale('log')

    ax.set_ylim(1, 1e6)
    ax.set_xlim(0, 90)

    ax.legend()

    return fig



def create_master_curve_animation_figure(
                            temperatures,
                            movingtime,
                            movingshifts,
                            ymin,
                            ymax,
                            reference_temperature
                           ):

    fig, ax = plt.subplots(dpi=200)
    
    fig.subplots_adjust(bottom=0.22)

    ax.set_yscale('log')

    lines = [
        ax.plot(
            [],
            [],
            label=f'T = {round(temp,1)} °C',
            antialiased=True,
            alpha=0.4,
            linewidth=5
        )[0]
        for temp in temperatures
    ]

    ax.set_title(
        'Plot of Stiffness Master Curve vs Reduced Time'
    )
    
    add_reference_temperature_label(ax, reference_temperature)

    ax.set_xlabel('Log Time (s)')
    ax.set_ylabel('Stiffness (MPa)')
    
    ax.set_xlim(round(movingtime.min()-movingshifts.max(),0)-1, round(movingtime.max(),0)+1)
    ax.set_ylim(
    10**(round(np.log10(ymin),0)-1),
    10**(round(np.log10(ymax),0)+1)
                )
    ax.legend(
            loc="lower left",
            framealpha=0.8
        )


    return fig, lines


def create_master_curve_animation(
    fig,
    lines,
    movingtime,
    movingshifts,
    stiffness_cache
):

    def init():

        for line in lines:
            line.set_data([], [])

        return lines

    def animate(i):

        for idx, shift in enumerate(
            movingshifts
        ):

            shifted_time = (
                movingtime
                - (i/149) * shift
            )

            lines[idx].set_data(
                shifted_time,
                stiffness_cache[idx]
            )

        return lines

    return animation.FuncAnimation(
        fig,
        animate,
        init_func=init,
        frames=150,
        interval=5,
        blit=True,
        
    )

def render_animation_to_video(
                                ani,
                                fig,
                                dpi=200
                            ):

    with tempfile.NamedTemporaryFile(
        suffix=".mp4",
        delete=False
    ) as fp:

        fname = fp.name

    writer = FFMpegWriter(
        fps=75,
        metadata=dict(artist='Koorosh Naderi'),
        codec='libx264',
        extra_args=['-pix_fmt', 'yuv420p']
    )

    ani.save(
        fname,
        writer=writer,
        dpi=dpi
    )
    
    plt.close(fig)

    with open(fname, "rb") as f:
        video_bytes = f.read()

    os.remove(fname)

    return video_bytes




#-----------------------------------------------------------------



def calculate_stiffness_curve(T1, bbr_temperature_results):
    coeffs = bbr_temperature_results.loc[
        bbr_temperature_results['Temperature (C)'] == T1,
        ['A', 'B', 'C']]
    return 10**(coeffs @ POLY_MATRIX)

def build_stiffness_cache(bbr_temperature_results):
    stiffness_cache = []

    for temp in bbr_temperature_results['Temperature (C)']:
        stiffness_cache.append(
                               calculate_stiffness_curve(temp, bbr_temperature_results)
                               .iloc[0, :]
                               .to_numpy()
                              )

    return stiffness_cache


def shift_factor_error(T1, T2, a, bbr_temperature_results):
    coeffs_T1 = bbr_temperature_results.loc[
        bbr_temperature_results['Temperature (C)'] == T1,
        ['A', 'B', 'C']]
    
    coeffs_T2 = bbr_temperature_results.loc[
        bbr_temperature_results['Temperature (C)'] == T2,
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
    bbr_temperature_results):
    return shift_factor_error(
        T1,
        T2,
        x[0],
        bbr_temperature_results)

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

def ca_objective_opt(
    params,
    reduced_omega,
    dynamic_shear_modulus
):
    
    beta, logOmegaC, logGg = params
    
    Gg = 10**logGg

    G_calc_CA = (
        Gg
        * (
            1 +
            (10**logOmegaC / reduced_omega)**beta
        )**(-1/beta)
    )

    return np.sum(
        (
            np.log10(G_calc_CA)
            -
            np.log10(dynamic_shear_modulus)
        )**2
    )


def fatigue_objective(
    T,
    target_value,
    arrhenius_slope,
    Tref,
    beta,
    logOmegaC,
    glassy_modulus):
    T = T[0]

    omega_red = (
        10
        * 10**(
            arrhenius_slope *
            (
                1/celsius_to_kelvin(T)
                - 1/celsius_to_kelvin(Tref)
            )))

    phase = (90 / (1 + (omega_red / (10**logOmegaC))**beta))

    G = (1000*glassy_modulus * (1 + (10**logOmegaC/omega_red)**beta)**(-1/beta))

    return (target_value - G*np.sin(np.radians(phase)))**2

def pavel_kriz_objective(
    T,
    arrhenius_slope,
    Tref,
    beta,
    logOmegaC,
    glassy_modulus,
    target_modulus
                ):
    T = T[0]
    omega_red_T_pavel_kriz = 10*10**(arrhenius_slope*(1/celsius_to_kelvin(T)-1/celsius_to_kelvin(Tref)))
    #phase_pavel_kriz = 90/(1+(omega_red_T_pavel_kriz/(10**result_CA.x[1]))**result_CA.x[0])
    G_pavel_kriz = 1000*glassy_modulus*(1+(10**logOmegaC/omega_red_T_pavel_kriz)**beta)**(-1/beta)
    return (target_modulus - G_pavel_kriz)**2

def count_lines(file):
    try:
        content = file.getvalue()

        try:
            decoded_content = content.decode("utf-8")

        except UnicodeDecodeError:
            decoded_content = content.decode("ISO-8859-1")

        return len(decoded_content.splitlines())

    except Exception as e:
        raise ValueError(
            f"Unable to count lines: {e}"
        )

def prepare_animation_data(
    bbr_temperature_results,
    a_T_list
):
    return {
        "moving_shifts":
            np.insert(np.cumsum(a_T_list), 0, 0),
        "moving_time":
            np.array(LOG_BBR_TIMES),
        "stiffness_cache":
            build_stiffness_cache(bbr_temperature_results)
    }


def merge_replicate_temperatures(
    bbr_temperature_results,
    bbr_curve_results,
    tolerance=REPLICATE_TEMPERATURE_TOLERANCE
):
    
    messages = []
    sorted_curves = sorted(
        bbr_curve_results,
        key=lambda x: x["temperature"]
    )

    groups = []

    for curve in sorted_curves:

        temp = curve["temperature"]

        placed = False

        for group in groups:

            group_mean = np.mean(
                [x["temperature"] for x in group]
            )

            if abs(temp - group_mean) <= tolerance:

                group.append(curve)
                placed = True
                break

        if not placed:

            groups.append([curve])

    merged_rows = []

    for group in groups:

        temperatures = [
            x["temperature"]
            for x in group
        ]

        representative_temperature = np.mean(
            temperatures
        )

        # -------------------------
        # Single test: keep original
        # -------------------------

        if len(group) == 1:

            temp = temperatures[0]

            original_row = (
                bbr_temperature_results[
                    np.isclose(
                        bbr_temperature_results[
                            'Temperature (C)'
                        ],
                        temp,
                        atol=1e-6
                    )
                ]
                .iloc[0]
                .to_dict()
            )

            merged_rows.append(
                original_row
            )

            continue

        # -------------------------
        # Replicates: average S(t)
        # -------------------------


        combined = []

        for item in group:
        
            df = item["fit_points"][
                ["Time (s)", "Stiffness (MPa)"]
            ].copy()
        
            combined.append(df)
        
        all_points = pd.concat(combined)
        
        messages.append(
            f"{len(group)} replicate tests near "
            f"{representative_temperature:.2f} °C "
            f"were averaged before analysis."
        )

        mean_curve = (
                        all_points
                        .groupby("Time (s)")
                        ["Stiffness (MPa)"]
                        .mean()
                        .reset_index()
                        )
        
        if len(mean_curve) < 3:
            raise ValueError(
                f"Insufficient data points after averaging "
                f"replicates near "
                f"{representative_temperature:.2f} °C."
            )
        
        poly = np.poly1d(
            np.polyfit(
                np.log10(mean_curve["Time (s)"]),
                np.log10(mean_curve["Stiffness (MPa)"]),
                2
            )
        )

        

        s60 = 10**(
            poly(
                np.log10(
                    REFERENCE_BBR_TIME
                )
            )
        )

        m60 = abs(
            2
            * poly.coefficients[0]
            * np.log10(
                REFERENCE_BBR_TIME
            )
            +
            poly.coefficients[1]
        )

        merged_rows.append(
            {
                'Temperature (C)': representative_temperature,
                'A': poly.coefficients[2],
                'B': poly.coefficients[1],
                'C': poly.coefficients[0],
                'S(60)': s60,
                'm-value(60)': m60
            }
        )

    return pd.DataFrame(
        merged_rows
    ), messages




# Streamlit app layout
st.title("BBRtoDSR Rheological Analysis Tool (Beta)")

st.logo("icon.png", size="large")


# Create a sidebar
st.sidebar.header("⌨ BBRtoDSR")
st.sidebar.write("""
    This application reads Bending Beam Rheometer (BBR) test data and estimates dynamic shear rheological properties typically obtained using a Dynamic Shear Rheometer (DSR). 
    By applying a series of rheological models, the software calculates a range of rheological parameters and performance-related indices.
""")

st.sidebar.markdown("""---""")
st.sidebar.subheader("⚠ Important Information")
# Add some descriptive text in the sidebar
st.sidebar.write("""
    Before using this application, ensure that the test data are accurate, valid, and reproducible.
    Please exercise caution when interpreting the results, as this methodology extrapolates BBR measurements beyond the range directly measured by the instrument. 
    Additionally, several assumptions have been made, 
    including Arrhenius-type temperature dependence, 
    the validity of [Generalized Power Law behavior for creep compliance at low temperatures](https://www.fhwa.dot.gov/publications/research/infrastructure/pavements/ltpp/10035/009.cfm), 
    the temperature and frequency independence of Poisson's ratio, 
    and the applicability of the [Christensen–Anderson (CA) model](https://doi.org/10.1080/14680629.2016.1267448) for complex modulus and phase angle master curves. 
    These assumptions may not accurately represent the true rheological behavior of all materials and should be considered when interpreting the results.
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

optimize_glassy_modulus = st.sidebar.checkbox(
    f"Optimize glassy shear modulus ($G_{'g'}$)",
    value=False,
    help=(
    f"Optimizes $G_{'g'}$ together with β and log$ω_{'C'}$. "
    f"If unsure, leave unchecked and use a fixed $G_{'g'}$ value.")
)

glassy_modulus = st.sidebar.number_input(
    f"Glassy Shear Modulus ($G_{'g'}$) in MPa",
    min_value=200.0,
    max_value=2000.0,
    value=float(DEFAULT_GLASSY_MODULUS_MPA),
    step=100.0,
    help="Glassy shear modulus used in the Christensen-Anderson model."
)

pavel_kriz_modulus = st.sidebar.selectbox(
    "Pavel-Kriz Modulus Criterion",
    options=[
        PAVEL_KRIZ_MODULUS_ORIGINAL,
        PAVEL_KRIZ_MODULUS_10MPA
    ],
    format_func=lambda x: (
        "|G*| = 8967 kPa (original)"
        if np.isclose(
            x,
            PAVEL_KRIZ_MODULUS_ORIGINAL
        )
        else "|G*| = 10000 kPa (10 MPa)"
    )
)

st.sidebar.markdown("""---""")

generate_animation = st.sidebar.checkbox(
    "Generate master curve animation",
    value=True
)

show_RI_contours = st.sidebar.checkbox(
    "Show Rheological Index contours on the black diagram",
    value=True
)

st.sidebar.markdown("""---""")



# Add a footer or additional text
st.sidebar.markdown("✉ Contact Me")
st.sidebar.write("For questions, suggestions, or bug reports, please contact: [koorosh.naderi@colas.com](mailto:koorosh.naderi@colas.com)")

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









tab_data, tab_lowtemp, tab_tts, tab_dsr, tab_performance, tab_animation = st.tabs([
    "Data",
    "Low Temperature Properties",
    "TTS",
    "DSR Transformation",
    "Intermediate Temperature Performance",
    "Animation"
])


with tab_data:

    st.write("""Data from at least two distinct BBR test temperatures are required for analysis. 
             Currently, CSV files generated by Cannon® Instrument Company software are supported (BBRw versions 1.34 and 1.35 have been tested). 
             XLSM files generated by PaveTest® Universal Test Module v2.3.0.5 have also been tested successfully.""")
    
    # Session state to track uploaded files
    if 'uploaded_files' not in st.session_state:
        st.session_state.uploaded_files = []
        
    if 'uploader_key' not in st.session_state:
        st.session_state.uploader_key = 0
        
    if "analysis_complete" not in st.session_state:
        st.session_state.analysis_complete = False
    
    
    
    
    
    # File uploader
    uploaded_files = st.file_uploader("Choose files (CSV or XLSM)", 
                                      accept_multiple_files=True, 
                                      type=['csv', 'xlsm'],
                                      key=f"uploader_{st.session_state.uploader_key}"
                                      )
    
    # If new files are uploaded, clear previous analysis
    if uploaded_files != st.session_state.uploaded_files:
        st.session_state.uploaded_files = uploaded_files
        st.session_state.analysis_complete = False
    
    # Button to clear analysis
    if st.button("Clear loaded results"):
        st.session_state.uploaded_files = []
        st.session_state.uploader_key += 1
        st.session_state.analysis_complete = False
        st.rerun()
    
    bbr_temperature_results = pd.DataFrame(columns=['Temperature (C)','A','B','C','S(60)','m-value(60)'])
    
    bbr_curve_results = []
    
    # Perform analysis if there are uploaded files
    if st.session_state.uploaded_files:
        
        for uploaded_file in uploaded_files:
            try:
    
                loaded = load_bbr_file(uploaded_file)
            
                bbr_data = loaded.bbr_data
                bbr_fit_points = loaded.bbr_fit_points
                bbr_fit_model = loaded.bbr_fit_model
            
                target_temperature = loaded.target_temperature
                actual_temperature = loaded.actual_temperature
    
            except Exception as e:
    
                st.error(
                    f"Error reading {uploaded_file.name}: {e}"
                )
    
                continue
            
            
            st.write(f"**Data from {uploaded_file.name}:**")
            st.dataframe(bbr_fit_points, hide_index = True)
            fig = create_bbr_fit_plot(bbr_fit_points)
            st.pyplot(fig)
            
            analysis_temperature = target_temperature
            
            if abs(actual_temperature - target_temperature) > 0.1:
                st.warning("""Temperature control was outside the acceptable tolerance. The target and measured temperatures differed by more than 0.1°C. 
                         The measured temperature will therefore be used as the analysis temperature.""")
                analysis_temperature = round(actual_temperature,2)
            
            bbr_temperature_results.loc[len(bbr_temperature_results)] = [
            analysis_temperature,
            bbr_fit_model.coefficients[2],
            bbr_fit_model.coefficients[1],
            bbr_fit_model.coefficients[0],
            10**(bbr_fit_model(np.log10(REFERENCE_BBR_TIME))),
            abs(
            2 * bbr_fit_model.coefficients[0]
            * np.log10(REFERENCE_BBR_TIME)
            + bbr_fit_model.coefficients[1]
            )
            ]
            st.success(
        f"Added {uploaded_file.name} at {analysis_temperature:.2f} °C")
            #st.markdown("""---""")
            
            bbr_curve_results.append(
                                        {
                                            "temperature": analysis_temperature,
                                            "fit_points": bbr_fit_points.copy()
                                        }
                                    )
    
    
            
    st.session_state.bbr_temperature_results = (
    bbr_temperature_results.copy()
    )
    
    st.session_state.bbr_curve_results = (
        bbr_curve_results.copy()
    )
    
    
  
        
if st.button("Generate DSR Results"):
    
    st.session_state.analysis_complete = True
  
if st.session_state.analysis_complete:
    
    try:
        
        bbr_temperature_results, replicate_messages = (
            merge_replicate_temperatures(
                st.session_state.bbr_temperature_results,
                st.session_state.bbr_curve_results,
                tolerance=REPLICATE_TEMPERATURE_TOLERANCE
            )
        )
        
    except Exception as e:
        
        st.error(str(e))
        st.stop()
    
    bbr_temperature_results = (
        bbr_temperature_results
        .sort_values("Temperature (C)", ascending=False)
        .reset_index(drop=True)
    )
    
    for msg in replicate_messages:
        st.info(msg)
    
    if len(bbr_temperature_results) < 2:
        
        remaining_temps = (
            bbr_temperature_results["Temperature (C)"]
            .round(2)
            .tolist()
        )
        
        st.error(
            "After merging replicate temperatures, "
            f"only {len(remaining_temps)} distinct temperature(s) remain: "
            f"{remaining_temps}. "
            "At least two distinct temperatures are required."
        )
        
        st.stop()
    
    st.success(
    "Analysis successfully started. Navigate through the tabs to review them."
    )
    
    
    with tab_lowtemp:
    
         
        st.subheader("Low Temperature Properties")
        
        bbr_temperature_results.reset_index(drop=True, inplace=True)
        st.dataframe(bbr_temperature_results , hide_index = True)
        
        reference_temperature = (
        bbr_temperature_results.loc[0, 'Temperature (C)'])
        
        analysis_temps = np.sort(
            bbr_temperature_results['Temperature (C)'].to_numpy()
        )
        
        
        if len(bbr_temperature_results)<2:
            
            st.warning("Upload at least two BBR datasets obtained at distinct test temperatures.")
            
        elif np.ptp(analysis_temps) < 1:
        
            st.warning(
                "The temperature range is small. Arrhenius fitting may be less reliable."
            )
    
        else:
            low_temp = calculate_low_temperature_properties(
                bbr_temperature_results
            )
            
            T_s = low_temp["Tc_S"]
            T_m = low_temp["Tc_m"]
            Delta_Tc = low_temp["Delta_Tc"]
            
            low_temp_fig = create_low_temperature_properties_plot(
                bbr_temperature_results,
                low_temp
            )
            
            st.pyplot(low_temp_fig)
            
                        
            col_tcs, col_tcm, col_deltatc = st.columns(3)
            
            with col_tcs:
                st.metric(f"$T_{{{'c,S'}}}$", f"{T_s:.1f} °C",border=True)
            with col_tcm:
                st.metric(f"$T_{{{'c,m'}}}$", f"{T_m:.1f} °C",border=True)
            with col_deltatc:
                st.metric(f"$Δ T_{'c'}$", f"{Delta_Tc} °C",border=True)
            
            if Delta_Tc > -2.5:
                col_deltatc.success("✅️ Low embrittlement risk")
            
            elif Delta_Tc > -5:
                col_deltatc.warning("⚠️ Intermediate concern")
            
            else:
                col_deltatc.error("❌ High embrittlement risk")
        
        st.markdown("""---""")        
        
        st.write(f"**Please note that $T_{{{'c,S'}}}$ and $T_{{{'c,m'}}}$ are critical temperatures that have been shifted by −10 °C from the test temperature. "
                 "This adjustment converts the 60 s test condition to an equivalent 7,200 s stiffness, which was the original criterion for resistance to single-event low-temperature cracking. "
                 f"Also note that $ΔT_{'c'}$ was introduced to help detect highly aged or waxy bitumen samples. "
                 f"However, $ΔT_{'c'}$ can be low for polymer-modified binders; a low $ΔT_{'c'}$ does not necessarily indicate a high susceptibility to cracking.**")

        
        
    with tab_tts:

        st.subheader("**Time-Temperature Superposition (TTS) Using the Arrhenius Model**")
        
        st.write(f"**Reference Temperature, $T_{{{'ref'}}}$: {reference_temperature} °C**")           
       
        tts = compute_tts(bbr_temperature_results)
        
        for msg in tts.warnings:
            st.warning(msg)
        
        
        a_T_list = tts.a_T_list
        temperatures = tts.temperatures
        reduced_time_list = tts.reduced_time_list
        stiffness_list = tts.stiffness_list
        shift_data = tts.shift_data
        master_curve_series = tts.master_curve_series
        arrhenius_slope = tts.arrhenius_slope
        r_squared_Arrhenius = tts.r_squared_Arrhenius
        shift_factor_values = tts.shift_factor_values
        arrhenius_values = tts.arrhenius_values
        
        activation_energy = (
                                tts.arrhenius_slope
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
        st.write(f"**An $r^{2}$ value below 0.98 may indicate inconsistency in the BBR data and should prompt further review of the test results.**")
    
        st.markdown("""---""")
    
        master_curve_fig = create_master_curve_plot(master_curve_series, reference_temperature)
        st.pyplot(master_curve_fig)
        
    with tab_dsr:
    
        

        #st.markdown("""---""")
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
                round(rmse_log, 4),
                reference_temperature
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
            glassy_modulus,
            optimize_glassy_modulus
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
        glassy_modulus_CA = ca["glassy_modulus"]
        
        
        logomega_C_zero = np.log10((10**logOmegaC)*(10**(arrhenius_slope*(1/celsius_to_kelvin(0)-1/celsius_to_kelvin(reference_temperature)))))
        
        if not ca["success"] and r2_CA < 0.99:
            st.warning(f"CA optimizer message: {ca['message']}")
        
        
        st.write(f"**β: {round(beta,3)}**")
        st.write(f"**log$ω_{'C'}$: {round(logOmegaC,3)} at $T_{{{"ref"}}}$**")
        st.write(
            f"**$G_g$: {glassy_modulus_CA:.0f} MPa**"
        )
        
        st.write(f"**log$ω_{'C'}$: {round(logomega_C_zero,3)} at 0°C**")
        st.write(f"**Rheological Index: {round(np.log10(2)/beta, 2)}**")
                    
        
        ca_fig = create_ca_plot(
            reduced_omega,
            dynamic_shear_modulus,
            newomega,
            newG_CA,
            r2_CA,
            rmse_CA,
            reference_temperature
        )
        
        st.pyplot(ca_fig)

        if optimize_glassy_modulus:
            st.write(
                f"**The glassy modulus ($G_g$) was optimized as part of the CA model fit and converged to "
                f"{glassy_modulus_CA/1000:.2f} GPa.**"
            )
        else:
            st.write(
                f"**The glassy modulus ($G_g$) was fixed at "
                f"{glassy_modulus_CA/1000:.2f} GPa during the CA model fit.**"
            )
    
    #st.markdown("""---""")
        
    with tab_performance:
    
        st.subheader("**Glover-Rowe Parameter, Cracking Performance**")

    
        gr = compute_gr(
            arrhenius_slope,
            reference_temperature,
            beta,
            logOmegaC,
            glassy_modulus_CA
        )
        
        phase_gr = gr["phase_gr"]
        g_gr = gr["g_gr"]
        g_r = gr["g_r"]

        st.write(f"**$G-R$: {round(g_r,0)} kPa**")
        st.write(f"**$|G^{'*'}|_{{{'G-R'}}}$: {round(g_gr,0)} kPa**")
        st.write(f"**$δ_{{{'G-R'}}}$: {round(phase_gr,0)} °**")
        
        gr_fig = create_glover_rowe_plot(
            phase_gr,
            g_gr,
            show_RI_contours
        )
        
        st.pyplot(gr_fig)

        st.write("""**The Glover-Rowe parameter, which is based on the DSR Fn proposed by [Glover et al.](https://rosap.ntl.bts.gov/view/dot/81884), was originally developed to relate dynamic rheological properties in shear mode to tensile failure strain (ductility) in tension mode for pure or unmodified bitumen (asphalt binder). 
                 For polymer-modified binders, caution is advised, as elevated Glover-Rowe values do not necessarily correspond to reduced ductility or increased cracking susceptibility.**""")
    
        st.markdown("""---""")
        
        fatigue = compute_fatigue(
        arrhenius_slope,
        reference_temperature,
        beta,
        logOmegaC,
        glassy_modulus_CA
    )
    
        T5000 = fatigue["T5000"]
        T6000 = fatigue["T6000"]
        phase_fatigue6 = fatigue["phase6000"]
        
        Temperature_fatigue_list = fatigue["Temperature_fatigue_list"]
        phase_fatigue_list = fatigue["phase_fatigue_list"]
        G_fatigue_list = fatigue["G_fatigue_list"]
        
        G_storage_fatigue_list = fatigue["G_storage_fatigue_list"]
        G_loss_fatigue_list = fatigue["G_loss_fatigue_list"]
            

        st.subheader("**Fatigue Cracking Criteria**")
        st.write(f"**$T_{{{'G"=5000kPa'}}}$: {round(T5000,1)} °C**")


        st.write(f"**$T_{{{'G"=6000kPa'}}}$: {round(T6000,1)} °C**")


        st.write(f"**$δ_{{{'G"=6000kPa'}}}$: {round(phase_fatigue6,1)} °**")
    
        
        
        fatigue_table = fatigue["fatigue_table"]

        st.dataframe(fatigue_table, hide_index = True)

        
        
        fatigue_fig = create_fatigue_plot(
            phase_fatigue_list,
            G_fatigue_list,
            Temperature_fatigue_list
        )
        
        st.pyplot(fatigue_fig)

        st.markdown("""---""")
        st.subheader("**Pavel-Kriz Phase Angle, [Detection of Phase Incompatible Binders](https://trid.trb.org/View/2344464/)**")

        pavel_kriz = compute_pavel_kriz(
            arrhenius_slope,
            reference_temperature,
            beta,
            logOmegaC,
            glassy_modulus_CA,
            pavel_kriz_modulus
        )
        
        temperature_pavel_kriz = pavel_kriz["temperature"]
        phase_pavel_kriz = pavel_kriz["phase"]


        st.write("ω = 10 Rad/s")
        st.write(f"T = {round(temperature_pavel_kriz,1)} °C")
        
        st.write(
            f"**$δ_{{{'|G*|=' + f'{pavel_kriz_modulus:.0f}' + 'kPa'}}}$: "
            f"{round(phase_pavel_kriz,1)} °**"
        )
        
        
        pavel_kriz_fig = create_pavel_kriz_plot(
            newphase_CA,
            newG_CA,
            phase_pavel_kriz,
            pavel_kriz_modulus
        )
        
        st.pyplot(pavel_kriz_fig)
        
    with tab_animation:
    
        if generate_animation:

            
            st.subheader("**Animated Master Curve Shifting**")

            
            animation_data = prepare_animation_data(
                                                        bbr_temperature_results,
                                                        a_T_list
                                                    )
                                                    
            movingshifts = animation_data["moving_shifts"]
            movingtime = animation_data["moving_time"]
            stiffness_cache = animation_data["stiffness_cache"]
            
            ymin = min(np.min(x) for x in stiffness_cache)
            ymax = max(np.max(x) for x in stiffness_cache)
            
            animation_fig, lines = create_master_curve_animation_figure(
                                                                                        temperatures,
                                                                                        movingtime,
                                                                                        movingshifts,
                                                                                        ymin,
                                                                                        ymax,
                                                                                        reference_temperature
                                                                                        )
            
            ani = create_master_curve_animation(
                                                animation_fig,
                                                lines,
                                                movingtime,
                                                movingshifts,
                                                stiffness_cache
                                                )


            video_bytes = render_animation_to_video(
                ani,
                animation_fig
            )
            
            st.video(video_bytes)
