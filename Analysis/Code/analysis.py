
import numpy as np
from matplotlib import font_manager
import matplotlib.ticker as tck
from itertools import permutations

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'

import pprint as pp
import matplotlib.pyplot as plt
import scipy.signal as signal
import os
from scipy.stats import norm
import ast
import datetime
#import mne

#from tsfresh import extract_features, extract_relevant_features, select_features
#from tsfresh.utilities.dataframe_functions import impute
#from tsfresh.feature_extraction import ComprehensiveFCParameters
from scipy.fft import fft, fftfreq
import cmath
from scipy.signal import find_peaks
import pywt
from skimage.restoration import denoise_wavelet
from scipy.interpolate import krogh_interpolate
from scipy.stats import f_oneway
from collections import OrderedDict 
from statsmodels.stats.anova import AnovaRM
from statsmodels.stats.multitest import multipletests
from scipy import stats
from factor_analyzer import FactorAnalyzer
from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity
from factor_analyzer.factor_analyzer import calculate_kmo
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import pingouin as pg
import scipy.integrate as integrate
from clean import *
import shutil
from itertools import combinations
import scikit_posthocs as sp
from statsmodels.formula.api import ols

from sklearn.linear_model import LinearRegression
import scipy.stats as stats
from statsmodels.stats.anova import AnovaRM
import warnings

plt.rcParams['axes.edgecolor']='#333F4B'
plt.rcParams['axes.linewidth']=0.8
plt.rcParams['xtick.color']='#333F4B'
plt.rcParams['ytick.color']='#333F4B'
plt.rcParams['text.color']='#333F4B'
plt.rcParams['figure.figsize'] = [15, 5]

BAR_COLORS = ['steelblue',
              'deepskyblue',            
              'lightskyblue',
              'slategrey',
              'darkseagreen',
              'darkorange',
              'indigo',             
              'black',
              'darkturquoise',
              ]

colors_dict = {'B':'gold',
    'C':'saddlebrown',
    'D':'slategrey',
    'E':'deepskyblue',
    'F':'darkseagreen',
    'G':'darkorange',
    'H':'indigo',
    'I':'black',
    'K':'darkturquoise'}


drink_dict = {'B':'Alcohol',
        'C':'Caffiene',
        'D':'Rhodiola',
        'E':'Valerian',
        'F':'F-Calm',
        'G':'G-Lift',
        'H':'H-Boost',
        'I':'I-Focus',
        'K':'K-Numuti Unwind',
        'L':'L-Numuti Thrive',
        'M':'M-Redbull',
        'N':'N-Goodmind',
        'O':'O-Calm V2',
        'P':'P-Lift V2',
        'Q':'Q-Boost V2',
        'R':'R-Focus',
        'T':'T-Viridian',
        'X':'X-Calm V2 Retest',
        }

# control_key = {'B':'A',
#             'C':'A',
#             'D':'A',
#             'E':'A',
#             'F':'J',
#             'G':'J',
#             'H':'J',
#             'I':'A',
#             'K':'J',
#             'L':'J',
#             'M':'J',
#             'N':'J',
#             'O':'S',
#             'P':'S',
#             'Q':'S',
#             'R':'S'}

control_key = {'O':'S',
            'T':'S',
            'X':'S',
            'F':'J'}

#control_groups = ['A','J','S']
control_groups = ['S','J']

# treatment_key = {'A':['B','C','D','E','I'],
#                     'J':['F','G','H','K','L','M','N'],
#                     'S':['O','P','Q','R']}

treatment_key = {'S':['O','R','T','X'],
                 'J':['F',]}

#time_series_groups = ['A','B','C','D','E','I','K','F','G','H','J','K','L','M','N','P','Q','R','S']
time_series_groups = ['O','S','T','X','F','J']  

##### For custom
#drink_dict = {'K':'K-Numuti Unwind',
#        }

#control = 'B'
#treatment = 'L'

#control_key = {treatment:control,}

#control_groups = [control]

#treatment_key = {control:[treatment]}

#time_series_groups = [treatment,control] 

def get_files(folder, tags=['',]):
    return [f for f in os.listdir(folder) if not f.startswith('.') and all(x in f for x in tags)] 

def get_validity(data, threshold):
    mask = (data != 0)&(data < threshold)
    valid = sum(1 for value in mask if value)/len(mask)*100
    return valid, mask


def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = signal.filtfilt(b, a, data,method="gust")
    return y

def butter_highpass_filter(data, cutoff, fs, order=5):
    b, a = butter_highpass(cutoff, fs, order=order)
    y = signal.filtfilt(b, a, data,method="gust")
    return y

def butter_lowpass(cutoff, fs, order=5):
    return signal.butter(order, cutoff, fs=fs, btype='low', analog=False)

def butter_highpass(cutoff, fs, order=5):
    return signal.butter(order, cutoff, fs=fs, btype='high', analog=False)


def derivative(x,y,n):
    x_range = np.linspace(x[0],x[-1],1000)
    y_spl = UnivariateSpline(x,y,s=0,k=4)
    y_spl_2d = y_spl.derivative(n=n)
    return x_range,y_spl_2d(x_range)


def moving_average(a, n) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

    
def get_effect_size(treatment, control):
    d1 = treatment
    d2 = control
    # calculate the size of samples
    try:
        n1, n2 = len(d1), len(d2)
        # calculate the variance of the samples
        s1, s2 = np.var(d1, ddof=1), np.var(d2, ddof=1)
        # calculate the pooled standard deviation
        s = np.sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2))
        # calculate the means of the samples
        u1, u2 = np.mean(d1), np.mean(d2)
        # calculate the effect size
        cohensD = (u1 - u2) / s
        hedgesG = cohensD*(1-3/(4*(n1+n2)-9))
        return cohensD, hedgesG
    except:
        return np.nan, np.nan
    # calculate the size of samples
    n1, n2 = len(d1), len(d2)
    # calculate the variance of the samples
    s1, s2 = np.var(d1, ddof=1), np.var(d2, ddof=1)
    # calculate the pooled standard deviation
    s = np.sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2))
    # calculate the means of the samples
    u1, u2 = np.mean(d1), np.mean(d2)
    # calculate the effect size
    return (u1 - u2) / s


def merge_dict(dict1, dict2):
    return(dict2.update(dict1))

def percentiles_df(in_df, ind, cols):    
    #Define variables
    calc = pd.DataFrame()   
    
    col_means = {x:in_df[x].mean() for x in cols}
    col_stds = {x:in_df[x].std() for x in cols}

    for index, row in in_df.iterrows():
        _per = {f"{x}":100*norm.cdf((row[x]-col_means[x])/col_stds[x]) for x in cols}
        _calc = {f'{ind}':row[ind],}
        _calc.update(_per)       
        calc = pd.concat([calc, pd.DataFrame([_calc])], ignore_index=True)

    return calc

def cuboid_data(size=(1,1,1)):
    # code taken from
    # https://stackoverflow.com/a/35978146/4124317
    # suppose axis direction: x: to left; y: to inside; z: to upper
    # get the length, width, and height
    l, w, h = size
    o = [0,0,0]
    x = [[o[0], o[0] + l, o[0] + l, o[0], o[0]],  
         [o[0], o[0] + l, o[0] + l, o[0], o[0]],  
         [o[0], o[0] + l, o[0] + l, o[0], o[0]],  
         [o[0], o[0] + l, o[0] + l, o[0], o[0]]]  
    y = [[o[1], o[1], o[1] + w, o[1] + w, o[1]],  
         [o[1], o[1], o[1] + w, o[1] + w, o[1]],  
         [o[1], o[1], o[1], o[1], o[1]],          
         [o[1] + w, o[1] + w, o[1] + w, o[1] + w, o[1] + w]]   
    z = [[o[2], o[2], o[2], o[2], o[2]],                       
         [o[2] + h, o[2] + h, o[2] + h, o[2] + h, o[2] + h],   
         [o[2], o[2], o[2] + h, o[2] + h, o[2]],               
         [o[2], o[2], o[2] + h, o[2] + h, o[2]]]               
    return np.array(x), np.array(y), np.array(z)

def plotCubeAt(pos=(0,0,0), size=(1,1,1), ax=None,**kwargs):
    # Plotting a cube element at position pos
    if ax !=None:
        X, Y, Z = cuboid_data(size)
        ax.plot_surface(X, Y, Z, rstride=1, cstride=1, **kwargs)

        
class Participant:   
    ID = None
    drink = None
    #Flags
    isSignals= 0
    hasChannels = 0
    isData = 0
    isDataComplete = 0
    isSignalsComplete = 0
    isSignalsOverflow = 0
    isGoNoGo = 0
    isMAS = 0

    #Process Flags
    isCleaned = 0

    #Data
    metadata = {}
    date = None
    session = None
    device = None
    freq = {0:0,}
    signalDuration = 0
    dataDuration = 0
    signals = pd.DataFrame()
    cleanSignals = pd.DataFrame() 
    taskStatus = pd.DataFrame()
    status = 0
    sample = pd.DataFrame()
    
    #Tasks
    EVM = pd.DataFrame()
    tasksInfo = pd.DataFrame()
    psychoPyFiles = pd.DataFrame()
    task_list_all = ['benchmarking', 'waterBreak_01', 'MAS_01', 'manualCapture_01', 'step_01', 'waterBreak_02', 'emotionRecognition_01', 'digitSpan_01', 'digitSymbol_01', 'memoryEncoding_01', 'goNoGo_01', 'manualCapture_02', 'Exposure', 'break_01', 'manualCapture_03', 'waterBreak_03', 'break_02', 'break_03', 'MAS_02', 'manualCapture_04', 'step_02', 'waterBreak_04', 'recall_01', 'emotionRecognition_02', 'digitSpan_02', 'digitSymbol_02', 'memoryEncoding_02', 'goNoGo_02', 'waterBreak_05', 'tsst_01', 'MAS_03', 'manualCapture_05']
    task_list_tasks = ['waterBreak_01', 'MAS_01', 'manualCapture_01', 'step_01', 'waterBreak_02', 'emotionRecognition_01', 'digitSpan_01', 'digitSymbol_01', 'memoryEncoding_01', 'goNoGo_01', 'manualCapture_02', 'Exposure', 'break_01', 'manualCapture_03', 'waterBreak_03', 'break_02', 'break_03', 'MAS_02', 'manualCapture_04', 'step_02', 'waterBreak_04', 'recall_01', 'emotionRecognition_02', 'digitSpan_02', 'digitSymbol_02', 'memoryEncoding_02', 'goNoGo_02', 'waterBreak_05', 'tsst_01', 'MAS_03', 'manualCapture_05']
    tasks = {t:pd.DataFrame() for t in task_list_all}
    
    #Neuropsych Data
    emotionaRecognition = pd.DataFrame()
    goNoGo = pd.DataFrame()
    MAS = pd.DataFrame()

    psychoPyResults = pd.DataFrame(columns=['Condition','Task','Factor','Result'])
 
sig_threshold = 0.028
sample_threshold = 15


def get_MAS_timings():
    print_status('Now Running:','MAS Timings')
    in_path = f'C:/Users/ashra/Desktop/Distell/Working Files/'
    participants = get_files(in_path)
    participants = [p for p in participants if '.' not in p]

    condition_key = {'Baseline':1,
                    'TreatmentTasksStart':2,
                    'TreatmentTasksEnd':3,                   
                    }
    condition_key = {value:key for key,value in condition_key.items()}

    try:
        results = pd.read_csv(f'C:/Users/ashra/Desktop/Distell/Working Files/MAS_Timings.csv')
        old = results['Participant'].drop_duplicates().to_list()
        participants = [p for p in participants if p not in old]
    except:
        print('Existing file not found')
        results = pd.DataFrame()

    for p in participants:
        try:
            P = pd.read_pickle(f'C:/Users/ashra/Desktop/Distell/Results/{p}/{p}_Object.pickle')
            tasks = P.tasksInfo[['Task','Start','End', 'Duration']]
            tasks = tasks.loc[tasks['Task'].str.contains('MAS')]
            tasks = tasks.sort_values(by='Start')
            tasks['Start']=tasks['Start']-tasks['Start'].values[0]
            result = dict()
            result['Participant']=P.ID
            for index,row in tasks.iterrows():       
                result[condition_key[int(row['Task'].split('_')[1])]]=row['Start']   
            results = pd.concat([results, pd.DataFrame([result])], ignore_index=True)
            #(f'Completed: {p}')
        except:
            print(f'Failed: {p}')

    results = results.set_index('Participant')
    results.to_csv(f'C:/Users/ashra/Desktop/Distell/Working Files/MAS_Timings.csv')


def get_POMS_timings():
    print_status('Now Running:','POMS Timings')
    in_path = f'C:/Users/ashra/Desktop/Distell/Working Files/'
    participants = get_files(in_path)
    participants = [p for p in participants if '.' not in p]

    condition_key = {'Baseline':1,
                    'TreatmentTasksStart':2,
                    'TreatmentTasksEnd':3,                   
                    }
    condition_key = {value:key for key,value in condition_key.items()}

    try:
        results = pd.read_csv(f'C:/Users/ashra/Desktop/Distell/Working Files/POMS_Timings.csv')
        old = results['Participant'].drop_duplicates().to_list()
        participants = [p for p in participants if p not in old]
    except:
        print('Existing file not found')
        results = pd.DataFrame()

    for p in participants:
        try:
            P = pd.read_pickle(f'C:/Users/ashra/Desktop/Distell/Results/{p}/{p}_Object.pickle')
            tasks = P.tasksInfo[['Task','Start','End', 'Duration']]
            tasks = tasks.loc[tasks['Task'].str.contains('POMS')]
            tasks = tasks.sort_values(by='Start')
            tasks['Start']=tasks['Start']-tasks['Start'].values[0]
            result = dict()
            result['Participant']=P.ID
            for index,row in tasks.iterrows():       
                result[condition_key[int(row['Task'].split('_')[1])]]=row['Start']   
            results = pd.concat([results, pd.DataFrame([result])], ignore_index=True)
            #(f'Completed: {p}')
        except:
            print(f'Failed: {p}')

    results = results.set_index('Participant')
    results.to_csv(f'C:/Users/ashra/Desktop/Distell/Working Files/POMS_Timings.csv')


def get_manual_timings():
    print_status('Now Running:','Manual Timings')
    in_path = f'C:/Users/ashra/Desktop/Distell/Working Files/'
    participants = get_files(in_path)
    participants = [p for p in participants if '.' not in p]

    condition_key = {'Baseline':1,
                         'TreatmentBeforeTasks':4,
                         'TreatmentAfterTasks':5,
                          'BeforeTreatment':2,
                          'TreatmentInitial':3,                    
                         }
                 
    condition_key = {value:key for key,value in condition_key.items()}

    try:
        results = pd.read_csv(f'C:/Users/ashra/Desktop/Distell/Working Files/manualCapture_Timings.csv')
        old = results['Participant'].drop_duplicates().to_list()
        participants = [p for p in participants if p not in old]
    except:
        print('Existing file not found')
        results = pd.DataFrame()

    for p in participants:
        try:
            P = pd.read_pickle(f'C:/Users/ashra/Desktop/Distell/Results/{p}/{p}_Object.pickle')
            tasks = P.tasksInfo[['Task','Start','End', 'Duration']]
            tasks = tasks.loc[tasks['Task'].str.contains('manualCapture')]
            tasks = tasks.sort_values(by='Start')
            tasks['Start']=tasks['Start']-tasks['Start'].values[0]
            result = dict()
            result['Participant']=P.ID
            for index,row in tasks.iterrows():       
                result[condition_key[int(row['Task'].split('_')[1])]]=row['Start']   
            results = pd.concat([results, pd.DataFrame([result])], ignore_index=True)
            #print(f'Completed: {p}')
        except:
            print(f'Failed: {p}')

    results = results.set_index('Participant')
    results.to_csv(f'C:/Users/ashra/Desktop/Distell/Working Files/manualCapture_Timings.csv')

def get_tsst_timings():
    print_status('Now Running:','TSST Timings')
    in_path = f'C:/Users/ashra/Desktop/Distell/Results/'
    participants = get_files(in_path)
    participants = [p for p in participants if '.' not in p]


    try:
        results = pd.read_csv(f'C:/Users/ashra/Desktop/Distell/Working Files/tsst_Timings.csv')
        old = results['Participant'].drop_duplicates().to_list()
        participants = [p for p in participants if p not in old]
    except:
        print('Existing file not found')
        results = pd.DataFrame()

    for p in participants:
        try:
            P = pd.read_pickle(f'C:/Users/ashra/Desktop/Distell/Results/{p}/{p}_Object.pickle')
            tasks = P.tasksInfo[['Task','Start','End', 'Duration']]
            ### Time relative to ManualCapture
            tasks1 = tasks.loc[tasks['Task'].str.contains('manualCapture')]
            tasks1 = tasks1.sort_values(by='Start')
            t0= tasks1['Start'].values[0]
        
            tasks = tasks.loc[tasks['Task'].str.contains('tsst')]
            tasks = tasks.sort_values(by='Start')
            tasks['StartMC']=tasks['Start']-t0
        
            result = dict()
            result['Participant']=P.ID
            result['StartMC']=tasks['StartMC'].values[0]
            result['Start']=tasks['Start'].values[0] 
            results = pd.concat([results, pd.DataFrame([result])], ignore_index=True)
            #print(f'Completed: {p}')
        except:
            print(f'Failed: {p}')

    results = results.set_index('Participant')
    results.to_csv(f'C:/Users/ashra/Desktop/Distell/Working Files/tsst_Timings.csv')


def get_tool_efficacy4(control, treatment):    
    result = {}
    isNormal = False
    isUseful= False
    
    try:    
        if (len(control)>sample_threshold) & (len(treatment)>sample_threshold):
            bayesTest = pg.ttest(control, treatment, correction=True, paired = True)
            result['Bayes'] = float(bayesTest['BF10'].values[0])
            result['EffSize'] = float(bayesTest['cohen-d'].values[0])
            result['Mean'] = treatment.mean()-control.mean()
            result['pValue']=float(bayesTest['p-val'].values[0])
            isUseful = True if float(bayesTest['BF10'].values[0])>=sig_threshold else False
    except:
        result['Bayes']=np.nan
        result['EffSize']=np.nan
        result['Mean']=np.nan
        result['pValue']=np.nan
    
    normal = pg.multivariate_normality(np.column_stack((control,treatment))).pval
    result['Normality']=normal
    result['isUseful']=isUseful
    result['nC']=len(control)
    result['nT']=len(treatment)
    
    return result

def find_effective_tools(tools, df, control_tag='Baseline', treatment_tag='Treatment', grouping_variable=None):
    # Create a DataFrame to store results with predefined columns
    effective_tools = pd.DataFrame(columns=['Tool', 'Group', 'isUseful'])
    
    # Cache columns to avoid repeated access
    columns = df.columns
    
    # Get unique values of the grouping variable to iterate over
    if grouping_variable:
        grouping_variables = df[grouping_variable].drop_duplicates().tolist()
    else:
        grouping_variables = [None]  # Use None if no grouping is specified

    # Loop over each tool to assess its effectiveness
    for i, tool in enumerate(tools):
        # Find columns related to the current tool and specific tags
        cols = [c for c in columns if tool in c]
        try:
            control_col = next(c for c in cols if control_tag in c)
            treatment_cols = [c for c in cols if treatment_tag in c and tool in c]

            # Evaluate each treatment column within the context of each group
            for treatment_col in treatment_cols:
                for group in grouping_variables:
                    covariates = get_covariates(group, tool)  # Assume this is defined elsewhere

                    # Ensure that the remaining covariates are all present in the dataset
                    covariates = [c for c in covariates if (c in df.columns) and (len(df.loc[df[grouping_variable]==group][c].dropna())>10)]

                    # Filter data for current group and drop rows with missing covariates
                    group_data = df[df[grouping_variable] == group].dropna(subset=covariates)

                    # Prepare control and treatment data
                    control_data = group_data[[control_col] + covariates].rename(columns={control_col: 'Value'})
                    control_data['Condition'] = 0

                    treatment_data = group_data[[treatment_col] + covariates].rename(columns={treatment_col: 'Value'})
                    treatment_data['Condition'] = 1

                    # Combine control and treatment data
                    combined_data = pd.concat([control_data, treatment_data])

                    # Check for valid data presence
                    if not combined_data.empty:
                        efficacy = get_tool_valid(combined_data, tool, 'Value', covariates=covariates)
                        efficacy.update({'Tool': tool, 'Method': 'PairedLinearRegression', 'Group': group})
                        effective_tools = pd.concat([effective_tools, pd.DataFrame([efficacy])], ignore_index=True)
                        print(f'{group} complete')
                    else:
                        print_error('Missing:',group)
        except StopIteration:
            print(f'Excluded: {tool}')

        # Print progress
        print(f'{tool} {((i + 1) / len(tools)) * 100:.2f}% complete')

    return effective_tools


axes_covariates = pd.read_csv(f'C:/Users/ashra/Desktop/Distell/Working Files/Axes_covariates.csv').dropna()
tools_covariates = pd.read_csv(f'C:/Users/ashra/Desktop/Distell/Working Files/Tool_Axis_Key.csv')
groups_covariates = pd.read_csv(f'C:/Users/ashra/Desktop/Distell/Working Files/MatchedCheck.csv')
devices_covariates = pd.read_csv(f'C:/Users/ashra/Desktop/Distell/Working Files/Device_Variables.csv')['Tool'].tolist()

def get_covariates(group,tool, mode='Axes'):
    group_covariates = []
    axis_covariates = []

    try:
        if mode == 'Axes':
            axis = tools_covariates.loc[tools_covariates['Tool']==tool]['Axis'].values[0]
            axis_covariates = axes_covariates.loc[axes_covariates['Axis'].str.contains(axis[1:-1])]['Trait'].tolist()

            covariates = list(axis_covariates)
        else:
            group_covariates = groups_covariates.loc[(groups_covariates['Group']==group) & (groups_covariates['pValue']<0.05)]['Tool'].tolist()
            covariates = list(group_covariates)
    except:
        try:
            group_covariates = groups_covariates.loc[(groups_covariates['Group']==group) & (groups_covariates['pValue']<0.05)]['Tool'].tolist()
            covariates = list(group_covariates)
        except:
            print_error('Please add covariates for',tool)
            covariates = []

    if tool in devices_covariates:
        covariates = covariates + ['Device']

    if len(covariates)==0:
        print_error('Please add covariates for',tool)

    if group in ['A','B','C','D','E','I','K','F','G','H','J','L','M','N']:
        covariates = [c for c in covariates if 'PSQI' not in c]

    return list(OrderedDict.fromkeys(covariates))


def get_tool_valid(df, tool, dv_col, control_group=0, treatment_group=1, grouping_factor='Condition', covariates=None):
    result = dict()
    ex_list = ['Group','Residual','Drink','Condition']
    control = df.loc[df[grouping_factor]==control_group][dv_col].dropna().to_numpy()
    treatment = df.loc[df[grouping_factor]==treatment_group][dv_col].dropna().to_numpy()
    result['nC']=len(control)
    result['nT']=len(treatment)
    result['meanC']=control.mean()
    result['meanT']=treatment.mean()

    if tool == 'emotionRecognition_Sad_Response_AlphaPSDInaccurate':
        x =1 

    if (result['nC']>sample_threshold) & (result['nT']>sample_threshold):

        between = grouping_factor
        if covariates is None:
            covariates = []

        iv = 'Condition'
        dv = dv_col
        
        ### ANCOVA
        continue_flag = True
        n = 0
        while continue_flag:
            df.loc[df['Condition']==1, 'Group']='Treatment'
            df.loc[df['Condition']==0, 'Group']='Control'

            if len(covariates):
                original_covariates = covariates                
                res = pg.ancova(data=df, dv=dv, covar=covariates, between='Group' )
                ### Identify relevent covariates
                covariates = [] # Reset covariates
                for covariate in res['Source'].tolist():
                    pval = res.loc[res['Source']==covariate]['p-unc'].values[0]
                    ## Check if significant
                    pvals = []
                    if (pval <0.05) and (covariate not in ex_list):
                        other_covariates = [x for x in original_covariates if x!=covariate]
                        ## Check if correlated with other cvs
                        for other in other_covariates:
                            corr = pg.corr(df[covariate],df[other],method='spearman')
                            corr_pval = corr['p-val'].values[0]
                            # If covariates are correlated, add covariate with the highest p-val
                            if corr_pval <0.001:
                                other_pval = res.loc[res['Source']==other]['p-unc'].values[0]
                                if pval<other_pval:
                                    covariates=covariates+[covariate,]
                                else:
                                    covariates=covariates+[other,]
                            else:
                                covariates=covariates+[covariate,]

                # Final list of significant covariates
                covariates = get_unique(covariates)

                # Get new results with only relevant covariates
                if len(covariates):
                    res = pg.ancova(data=df, dv=dv, covar=covariates, between='Group' )
                    res = res.replace('Group','Condition')
                    punc = res['p-unc'][0]
                    holms2 = multipletests(res['p-unc'],alpha=0.05, method='holm')[1][0]
                    pvals = [res.loc[res['Source']==i]['p-unc'].values[0] for i in covariates]
                else:
                    pvals = [0,]
                    res = pg.anova(data=df, dv=dv, between='Group' )
                    res = res.replace('Group','Condition')
                    punc = res['p-unc'][0]
                    holms2 = res['p-unc'].values[0]      
                n +=1
            else:
                pvals = [0,]
                res = pg.anova(data=df, dv=dv, between='Group' )
                res = res.replace('Group','Condition')
                punc = res['p-unc'][0]
                holms2 = res['p-unc'].values[0]

            continue_flag = any(i>0.05 for i in pvals)

        res2 = res.copy()
        #pp.pprint(f'>>> n={n}')
        #pp.pprint(res2)

        ### Regression
        data= df[covariates+[iv]+[dv]].dropna()
        predictors = data[covariates+[iv]]
        res = pg.linear_regression(predictors,data[dv], add_intercept=False,relimp=True,as_dataframe=True)
        if len(covariates):
            holms = multipletests(res['pval'],alpha=0.05, method='holm')[1][len(predictors.columns)-1]
            r=res['relimp'].values[len(predictors.columns)-1]
            m = np.sign(res['coef'].values[len(predictors.columns)-1])
            highest = res.sort_values(by='relimp_perc',ascending=False)['names'].values[0]
            for c in res['names'].tolist():
                _pval = res2.loc[res2['Source']==c]['p-unc'].values[0]
                result[f'pVal-{c}']=_pval
                result[f'Eff-{c}']=res.loc[res['names']==c]['relimp_perc'].values[0]
        else:
            holms = res['pval'][0]
            r=res.relimp[0]
            m = np.sign(res.coef[0])
            highest = np.nan

        ### T-Test
        control = df.loc[df[grouping_factor]==control_group][dv_col].to_numpy()
        treatment = df.loc[df[grouping_factor]==treatment_group][dv_col].to_numpy()

        pVTTest = pg.ttest(control, treatment, correction=True, paired = True)['p-val'].values[0]

        result['EffSize'] = r
        result['Sign'] = m
        #isUseful = True if holms2<=sig_threshold else False
        #result['isUseful'] = isUseful

        result['pV-TTest']=pVTTest
        result['pV-ANCOVA']=holms2
        result['pV-unc']=punc
        result['MainEffect']=highest
        #result['Normality']=normal
        result['Covariates']=covariates

    return result


def get_significance_old(group_data: dict, cluster: str, groups = [], paired = False, one_sample = False):
    """Analysis of significance between arrays

    Args:
        data: A dictionary containg arrays, with keys set as group name
        cluster: The label attatched to this analysis
        groups: A list of all the groups
        paired: indicating is paired sample across groups

    Returns:
        dataframe: containing results
    """


    if not isinstance(groups, list) and groups is not None:
        raise TypeError("Expected 'groups' to be a list or None, got {}".format(type(groups).__name__))

    if len(groups) == 0:
        groups = list(group_data.keys())

    if len(groups) == 1:
        print(f'Failed to get stats for {cluster}: Only one group ({groups})')
        return pd.DataFrame()
    else:
        sample_check = True
        for i in groups:
            if len(group_data[i])<3:
                sample_check = False
                control = i
                nC = len(group_data[i])
            if len(group_data[i])==3:
                group_data[i] = np.append(group_data[i],0)

                

        significance = pd.DataFrame()

        if sample_check:

            data = pd.DataFrame()
            for key,value in group_data.items():
                res = pd.DataFrame()
                res['Group']=[key]*len(value)
                res['Value']=value
                data = pd.concat([data,res])

            ### Test for parametric or non-parametric
            normality = pg.normality(data=data, dv='Value', group='Group')
            homoscedasticity = pg.homoscedasticity(data=data, dv='Value', group='Group')
            normal = all(normality['normal'])
            equal_var = all(homoscedasticity['equal_var'])
            
            if len(groups)>2:            
                if equal_var:
                    aov = pg.anova(dv='Value', between='Group', data=data,
                        detailed=True)
                    pval = aov['p-unc'].values[0]
                    test_type = 'ANOVA'
                    
                    # Check if residuals are normally distributed
                    model = ols('Value ~ C(Group)', data=data).fit()
                    data['Residuals'] = model.resid
                    normality_test_results = pg.normality(data['Residuals'])
                    residuals_normal = all(normality_test_results['normal'])

                    if not residuals_normal:
                        kru = pg.kruskal(dv='Value', between='Group', data=data)
                        pval = kru['p-unc'].values[0]
                        test_type = 'Kruskal-Wallis'
        
                else:
                    wel = pg.welch_anova(dv='Value', between='Group', data=data)
                    pval = wel['p-unc'].values[0]
                    test_type = 'ANOVA Welch'

                sig = True if pval<0.05 else False  
                result = {'Groups':'Group',
                                'pValue':pval,
                                'Type':test_type,
                                'nC':'-',
                                'nT':'-',
                                'Cluster':cluster}
                result_df = pd.DataFrame([result], index=[0]) #, index=[0])  # Creating a DataFrame with a single row
                significance = pd.concat([significance, result_df])

                if sig:
                    if normal:
                        if equal_var:
                            ph = pg.pairwise_tukey(dv='Value', between='Group', data=data, effsize = 'r')
                            ph['Control']= ph['A']
                            ph['Treatment']= ph['B']
                            ph['pval']=ph['p-tukey']
                            test_type = 'Post-Hoc Tukey'
                        else:
                            ph = pg.pairwise_gameshowell(dv='Value', between='Group', data=data, effsize='r')
                            ph['Control']= ph['A']
                            ph['Treatment']= ph['B']
                            test_type = 'Post-Hoc Games'
                    else:
                        #What to do?
                        res = sp.posthoc_dunn(data, val_col='Value', group_col= 'Group', p_adjust = 'holm')
                        test_type = 'Post-Hoc Dunn'
                        groups = res.columns
                        pair_permutations = set('--'.join(sorted([p1, p2])) for p1, p2 in permutations(groups, 2))
                        ph = pd.DataFrame()
                        for pair in pair_permutations:
                            control = pair.split('--')[0]
                            treatment = pair.split('--')[1]
                            _ph = pd.DataFrame()
                            _ph['Control']=[control]
                            _ph['Treatment']=[treatment]
                            _ph['pval']=res.loc[control,treatment]
                            ph = pd.concat([ph,_ph])

                    
                    for index, row in ph.iterrows():
                        groups = [f"{row['Control']}",f"{row['Treatment']}"]
                        groups.sort()
                        result = {'Groups':(' and ').join(groups),
                            'pValue':row['pval'],
                            'Control':row['Control'],
                            'Treatment':row['Treatment'],
                            'Type':test_type,
                            'nC':len(data.loc[data['Group']==row['Control']]),
                            'nT':len(data.loc[data['Group']==row['Treatment']]),
                            'Cluster':cluster}
                        result_df = pd.DataFrame([result], index=[0]) #, index=[0])  # Creating a DataFrame with a single row
                        significance = pd.concat([significance, result_df])
            else:
                treatment = data.loc[data['Group']==groups[0]]['Value'].values
                control = data.loc[data['Group']==groups[1]]['Value'].values
                
                if paired:
                    if normal:
                        res = pg.ttest(control,treatment, paired=True)
                        test_type = 'Paired T-Test'
                    else:
                        with warnings.catch_warnings(record=True) as caught_warnings:
                            warnings.simplefilter("always")  # Catch any warnings
                            res = pg.wilcoxon(control, treatment)
                            test_type = 'Wilcoxin'
                            # Check if any caught warnings are UserWarning
                            user_warning_issued = any(issubclass(w.category, UserWarning) for w in caught_warnings)
                            
                            if user_warning_issued:
                                print("UserWarning detected, proceeding with bootstrap analysis.")
                                # Assume pre_scores and post_scores are defined or obtained as needed
                                res = bootstrap_test(control,treatment)
                                test_type = 'Bootstrap'
                    
                elif normal:
                    res = pg.ttest(control,treatment, paired=False)
                    test_type = 'Independent T-Test'
                else:
                    res = pg.mwu(control,treatment)
                    test_type = 'Mann-Whitney U'
                    
                groups = [f'{groups[1]}',f'{groups[0]}']
                groups.sort()

                result = {'Groups':(' and ').join(groups),
                        'Control':groups[1],
                        'Treatment':groups[0],
                        'pValue':res['p-val'].values[0],
                        'Type':test_type,
                        'nC':len(control),
                        'nT':len(treatment),
                        'Cluster':cluster}
                result_df = pd.DataFrame([result], index=[0])   # Creating a DataFrame with a single row
                significance = pd.concat([significance, result_df])
        else:
            result = {'Groups':(' and ').join(groups),
                        'pValue':1,
                        'Control':control,
                        'Treatment':'Not enough samples',
                        'nC':nC,
                        'nT':'Not enough samples',
                        'Cluster':cluster}
            result_df = pd.DataFrame([result], index=[0]) #, index=[0])  # Creating a DataFrame with a single row
            significance = pd.concat([significance, result_df])

        return significance



def get_repeated_valid(df, tool, dv_col, control_group, treatment_group, grouping_factor='Drink', covariates=None):
    print(f'Validating {tool}: {treatment_group} vs {control_group}')
    result = dict()
    result['Tool']=tool
    result['Method']='LinearRegression'
    result['Group']=treatment_group
    result['Control']=control_group
    ex_list = ['Group','Residual','Drink','Condition']
    control = df.loc[df[grouping_factor]==control_group][dv_col].dropna().astype(float).to_numpy()
    treatment = df.loc[df[grouping_factor]==treatment_group][dv_col].dropna().astype(float).to_numpy()
    result['nC']=len(control)
    result['nT']=len(treatment)
    result['meanC']=control.mean()
    result['meanT']=treatment.mean()

    if (len(control)>sample_threshold) & (len(treatment)>sample_threshold):

        if covariates is None:
            covariates = get_covariates(control_group,tool) + get_covariates(treatment_group,tool)
            covariates = list(OrderedDict.fromkeys(covariates))

        if (treatment_group in ['A','B','C','D','E','I','K','F','G','H','J','L','M','N']) or (control_group in ['A','B','C','D','E','I','K','F','G','H','J','L','M','N']):
            covariates = [c for c in covariates if 'PSQI' not in c]

        iv = 'Condition'
        dv = dv_col
        
        continue_flag = True
        n = 0
        while continue_flag:
            if len(covariates):
                original_covariates = covariates   
                data = df.loc[(df[grouping_factor]==control_group) | (df[grouping_factor]==treatment_group) ]
                res = pg.ancova(data=data, dv=dv, covar=covariates, between=grouping_factor )
                ### Identify relevent covariates
                covariates = [] # Reset covariates
                for covariate in res['Source'].tolist():
                    pval = res.loc[res['Source']==covariate]['p-unc'].values[0]
                    ## Check if significant
                    pvals = []
                    if (pval <0.05) and (covariate not in ex_list):
                        other_covariates = [x for x in original_covariates if x!=covariate]
                        ## Check if correlated with other cvs
                        for other in other_covariates:
                            corr = pg.corr(df[covariate],df[other],method='spearman')
                            corr_pval = corr['p-val'].values[0]
                            # If covariates are correlated, add covariate with the highest p-val
                            if corr_pval <0.001:
                                other_pval = res.loc[res['Source']==other]['p-unc'].values[0]
                                if pval<other_pval:
                                    covariates=covariates+[covariate,]
                                else:
                                    covariates=covariates+[other,]
                            else:
                                covariates=covariates+[covariate,]

                # Final list of significant covariates
                covariates = get_unique(covariates)

                if len(covariates):
                    data = df.loc[(df[grouping_factor]==control_group) | (df[grouping_factor]==treatment_group) ]
                    res = pg.ancova(data=data, dv=dv, covar=covariates, between=grouping_factor )
                    res = res.replace('Drink','Condition')
                    punc = res['p-unc'][0]
                    holms2 = multipletests(res['p-unc'],alpha=0.05, method='holm')[1][0]                   
                    pvals = [res.loc[res['Source']==i]['p-unc'].values[0] for i in covariates]
                else:
                    pvals = [0,]
                    res = pg.anova(data=data, dv=dv, between=grouping_factor )
                    res = res.replace('Drink','Condition')
                    punc = res['p-unc'][0]
                    holms2 = res['p-unc'].values[0]
                n += 1
            else:
                pvals = [0,]
                res = pg.anova(data=data, dv=dv, between=grouping_factor )
                res = res.replace('Drink','Condition')
                punc = res['p-unc'][0]
                holms2 = res['p-unc'].values[0]

            continue_flag = any(i>0.05 for i in pvals)

        res2 = res.copy() 
        #pp.pprint(f'>>> n={n}')
        #pp.pprint(res2)

        #Regression  
        data = df.loc[(df[grouping_factor]==control_group) | (df[grouping_factor]==treatment_group) ]

        # Create the 'Condition' column separately
        condition = pd.Series(
            data[grouping_factor].apply(
                lambda x: 1 if x == treatment_group else 0
            ),
            index=data.index,
            name='Condition'
        )
        # Combine the 'Condition' column with the original DataFrame
        data = pd.concat([data, condition], axis=1)
        # Defragment the DataFrame if necessary
        data = data.copy()


        data= data[covariates+[iv]+[dv]+[grouping_factor]].dropna()       
        predictors = data[covariates+[iv]]
        res = pg.linear_regression(predictors,data[dv], add_intercept=False,relimp=True,as_dataframe=True)

        holms = multipletests(res['pval'],alpha=0.05, method='holm')[1][len(predictors.columns)-1]
        r=res['relimp'].values[len(predictors.columns)-1]
        m = np.sign(res['coef'].values[len(predictors.columns)-1])
        highest = res.sort_values(by='relimp_perc',ascending=False)['names'].values[0]

        for c in res['names'].tolist():
            _pval = res2.loc[res2['Source']==c]['p-unc'].values[0]
            result[f'pVal-{c}']=_pval
            result[f'Coeff-{c}']=res.loc[res['names']==c]['coef'].values[0]


        pVTTest = pg.ttest(control, treatment, correction=True, paired = False)['p-val'].values[0]
        
        result['EffSize'] = r
        result['Sign'] = m
        #isUseful = True if holms2<=sig_threshold else False
        #result['isUseful'] = isUseful

        result['pV-TTest']=pVTTest
        result['pV-ANCOVA']=holms2
        result['pV-unc']=punc
        result['MainEffect']=highest
        #result['Normality']=normal
        result['Covariates']=covariates
        
    return result


def calculate_vif(df):
    """Calculate Variance Inflation Factor for each feature in the DataFrame."""
    vif_data = pd.DataFrame()
    vif_data["feature"] = df.columns
    vif_data["VIF"] = [
        1 / (1 - LinearRegression().fit(df.drop(col, axis=1), df[col]).score(df.drop(col, axis=1), df[col]))
        for col in df.columns
    ]
    return vif_data


def identify_significant_covariates(data, covariates, dependent_var):
    """
    Identify significant covariates by fitting a model and checking significance.
    
    Args:
        data: DataFrame containing the data
        covariates: List of covariate column names
        dependent_var: Name of the dependent variable column
    
    Returns:
        List of significant covariates
    """
    significant_covariates = []
    for covariate in covariates:
        if covariate in data.columns:
            res = pg.linear_regression(data[covariate].values.reshape(-1, 1), data[dependent_var].values, remove_na=True)
            p_value = res['pval'].values[0]
            if p_value < 0.05:
                significant_covariates.append(covariate)
    
    # Check for multicollinearity
    if len(significant_covariates) > 1:
        cov_data = data[significant_covariates]
        vif_data = calculate_vif(cov_data.dropna())
        high_vif_features = vif_data[vif_data['VIF'] > 5]['feature'].tolist()
        for feature in high_vif_features:
            if feature in significant_covariates:
                significant_covariates.remove(feature)
    
    return significant_covariates


def check_assumptions(df, dv, gv):
    """
    Check normality and homoscedasticity assumptions for parametric tests.

    Args:
        df: DataFrame containing the data
        dv: The dependent variable column name
        gv: The grouping variable column name

    Returns:
        Tuple: (is_normal, is_homoscedastic)
    """
    normality = pg.normality(data=df, dv=dv, group=gv)
    homoscedasticity = pg.homoscedasticity(data=df, dv=dv, group=gv)
    is_normal = all(normality['normal'])
    is_homoscedastic = all(homoscedasticity['equal_var'])
    return is_normal, is_homoscedastic


def perform_parametric_tests(df, dv, gv, significant_covariates, cluster):
    """
    Perform parametric tests (ANOVA, ANCOVA, t-tests).

    Args:
        df: DataFrame containing the data
        dv: The dependent variable column name
        gv: The grouping variable column name
        significant_covariates: List of significant covariates
        cluster: The label attached to this analysis

    Returns:
        DataFrame containing results of the parametric tests
    """
    significance = pd.DataFrame()
    groups = df[gv].unique()

    if significant_covariates:
        # ANCOVA
        covariate_str = ' + '.join(significant_covariates)
        aov = pg.ancova(data=df, dv=dv, covar=significant_covariates, between=gv)
        pval = aov['p-unc'].values[0]
        test_type = 'ANCOVA'
    else:
        # ANOVA
        aov = pg.anova(dv=dv, between=gv, data=df, detailed=True)
        pval = aov['p-unc'].values[0]
        test_type = 'ANOVA'

        # Check if residuals are normally distributed
        model = ols(f'{dv} ~ C({gv})', data=df).fit()
        df['Residuals'] = model.resid
        normality_test_results = pg.normality(df['Residuals'])
        residuals_normal = all(normality_test_results['normal'])

        if not residuals_normal:
            # Switch to non-parametric test if needed
            return perform_non_parametric_tests(df, dv, gv, cluster)

    sig = True if pval < 0.05 else False
    result = {'Groups': ' and '.join(map(str, groups)),
              'pValue': pval,
              'Type': test_type,
              'nC': df[df[gv] == groups[0]][dv].count(),
              'nT': df[df[gv] == groups[1]][dv].count(),
              'Cluster': cluster,
              'Significant Covariates': ', '.join(significant_covariates)}
    result_df = pd.DataFrame([result], index=[0])
    significance = pd.concat([significance, result_df])

    return significance


def perform_non_parametric_tests(df, dv, gv, cluster):
    """
    Perform non-parametric tests (Kruskal-Wallis, Mann-Whitney U).

    Args:
        df: DataFrame containing the data
        dv: The dependent variable column name
        gv: The grouping variable column name
        cluster: The label attached to this analysis

    Returns:
        DataFrame containing results of the non-parametric tests
    """
    significance = pd.DataFrame()
    groups = df[gv].unique()

    # Kruskal-Wallis test
    kruskal = pg.kruskal(data=df, dv=dv, between=gv)
    pval = kruskal['p-unc'].values[0]
    test_type = 'Kruskal-Wallis'

    sig = True if pval < 0.05 else False
    result = {'Groups': ' and '.join(map(str, groups)),
              'pValue': pval,
              'Type': test_type,
              'nC': df[df[gv] == groups[0]][dv].count(),
              'nT': df[df[gv] == groups[1]][dv].count(),
              'Cluster': cluster}
    result_df = pd.DataFrame([result], index=[0])
    significance = pd.concat([significance, result_df])

    return significance


def select_test_and_analyze(df, dv, gv, covariates, cluster):
    """
    Select the appropriate test based on data characteristics and perform the analysis.

    Args:
        df: DataFrame containing the data
        dv: The dependent variable column name
        gv: The grouping variable column name
        covariates: A list of covariate column names
        cluster: The label attached to this analysis

    Returns:
        DataFrame containing results of the analysis
    """
    is_normal, is_homoscedastic = check_assumptions(df, dv, gv)
    significant_covariates = []

    if covariates:
        significant_covariates = identify_significant_covariates(df, covariates, dv)

    perform_parametric_tests(df, dv, gv, significant_covariates, cluster)
    # if is_normal and is_homoscedastic:
    #     # Perform parametric tests
    #     return perform_parametric_tests(df, dv, gv, significant_covariates, cluster)
    # else:
    #     # Perform non-parametric tests
    #     return perform_non_parametric_tests(df, dv, gv, cluster)


def get_significance(df: pd.DataFrame, dv: str, gv: str, covariates=None, cluster=''):
    """
    Analysis of significance between groups using specified dependent variable and covariates.

    Args:
        df: DataFrame containing the data
        dv: The dependent variable column name
        gv: The grouping variable column name
        covariates: A list of covariate column names (optional)
        cluster: The label attached to this analysis

    Returns:
        DataFrame containing results
    """
    df = df[[dv,gv]+covariates]
    df = df.replace('-','', regex=True)
    groups = df[gv].unique()

    if len(groups) < 2:
        print(f'Failed to get stats for {cluster}: Less than two groups provided ({groups})')
        return pd.DataFrame()

    sample_check = all(df[df[gv] == group][dv].count() >= 3 for group in groups)

    if not sample_check:
        print(f'Failed to get stats for {cluster}: One or more groups have less than 3 samples')
        return pd.DataFrame()

    return select_test_and_analyze(df, dv, gv, covariates, cluster)


def check_tool_covariates(tools):
    ##### Check for new covariates
    checked = pd.read_csv(f'C:/Users/ashra/Desktop/Distell/Working Files/Tool_Axis_Key.csv')['Tool'].drop_duplicates().tolist()
    unchecked = [x for x in tools if x not in checked]
    if unchecked:
        print_error('Missing','Tool Covariates')
        _=[print(f'{x}') for x in unchecked]



def bootstrap_test(pre, post, n_bootstrap=10000):
    observed_difference = np.mean(post) - np.mean(pre)
    combined = np.concatenate([pre, post])
    count = 0

    for _ in range(n_bootstrap):
        # Resampling without replacement to maintain data integrity
        np.random.shuffle(combined)
        new_pre = combined[:len(pre)]
        new_post = combined[len(pre):]
        new_difference = np.mean(new_post) - np.mean(new_pre)
        
        # Count how often the bootstrap difference is greater than the observed
        if np.abs(new_difference) >= np.abs(observed_difference):
            count += 1

    # P-value estimation
    p_value = count / n_bootstrap

    result = pd.DataFrame()
    result['p-val']=[p_value]
    result['n_bootstrap']=[n_bootstrap]

    return result  

def calculate_outliers(df):
    outlier_info = {}
    
    for column in df.columns:
        if pd.api.types.is_numeric_dtype(df[column]):
            # Calculate Q1 and Q3
            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)
            IQR = Q3 - Q1
            
            # Define bounds for outliers
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # Determine outliers
            outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
            percent_outliers = (len(outliers) / len(df[column])) * 100
            
            # Store the result
            outlier_info[column] = percent_outliers
    
    # Return a DataFrame for better visualization
    return pd.DataFrame.from_dict(outlier_info, orient='index', columns=['Percent Outliers'])



def pipeline(out_path, control_key, control_groups, treatment_key, time_series_groups):

    def pairedttest(data):
        #############################
        ###### Paired T-Test Data
        #############################
        results = pd.DataFrame(columns = ['Tool','Group','nC','nT','EffSize','Sign','pV-TTest','pV-unc','pV-ANCOVA','MainEffect'])
        
        _data = data.loc[data['Drink'].isin(time_series_groups)]
        results = pd.concat([results,find_effective_tools(tools,_data, control_tag = 'Baseline', treatment_tag = 'Treatment', grouping_variable = 'Drink')])

        print('Completed: Validation Stage 0')
        return results
    

    working_path = f'C:/Users/ashra/Desktop/Distell/Working Files/'

        # Load the data
    df = pd.read_csv('C:/Users/ashra/Desktop/Distell/Results/EVM_Results.csv', low_memory=False)

    # Filter the DataFrame based on the 'Drink' column
    df = df.loc[df['Drink'].isin(time_series_groups)]

    # Define replacement key
    replacement_key = {
        'BC:33:AC:AB:B3:DB': 1,
        'BC:33:AC:AB:B3:C8': 2,
        'BC:33:AC:AB:B4:51': 3,
        '#NAME': np.nan,
        '?': np.nan
    }
    # Replace values according to the replacement_key
    df.replace(replacement_key, inplace=True)

    # Convert columns to numeric except specified ones
    cols = df.columns.difference(['Participant', 'Number', 'Drink', 'Device'])
    df[cols] = df[cols].apply(pd.to_numeric, errors='coerce')

    # Filter valid columns with sufficient non-null entries
    count_threshold = 10
    valid_cols = [
        c for c in cols if df.groupby('Drink')[c].apply(lambda x: x.notna().sum() > count_threshold).all()
    ]

    # Keep only valid columns
    df = df[['Participant', 'Number', 'Drink', 'Device'] + valid_cols]

    #Defragment df
    df = df.copy()
    # Add task-set inhibition calculations
    try:
        df['psy_goNoGo_Baseline_TSI'] = df['psy_goNoGo_Baseline_Task3_RTAccurate'] - df['psy_goNoGo_Baseline_Task1_RTAccurate']
        df['psy_goNoGo_Treatment_TSI'] = df['psy_goNoGo_Treatment_Task3_RTAccurate'] - df['psy_goNoGo_Treatment_Task1_RTAccurate']
    except KeyError as e:
        pass

    #Add TSST SCL and Phasic/Vagal Response calculations

    #### Add TSST SCL
    df.loc[:, 'psy_tsst_E2.0_SCLResponse']=(df['psy_tsst_E2.0_MeanSCL']-df['psy_tsst_E1.0_MeanSCL'])/df['psy_tsst_E1.0_MeanSCL']
    df.loc[:, 'psy_tsst_E3.0_SCLResponse']=(df['psy_tsst_E3.0_MeanSCL']-df['psy_tsst_E1.0_MeanSCL'])/df['psy_tsst_E1.0_MeanSCL']
    df.loc[:, 'psy_tsst_E4.0_SCLResponse']=(df['psy_tsst_E4.0_MeanSCL']-df['psy_tsst_E1.0_MeanSCL'])/df['psy_tsst_E1.0_MeanSCL']

    #### Add TSST SCL
    df.loc[:, 'psy_tsst_E2.0_PhasicResponse']=(df['psy_tsst_E2.0_PPM']-df['psy_tsst_E1.0_PPM'])/df['psy_tsst_E1.0_PPM']
    df.loc[:, 'psy_tsst_E3.0_PhasicResponse']=(df['psy_tsst_E3.0_PPM']-df['psy_tsst_E1.0_PPM'])/df['psy_tsst_E1.0_PPM']
    df.loc[:, 'psy_tsst_E4.0_PhasicResponse']=(df['psy_tsst_E4.0_PPM']-df['psy_tsst_E1.0_PPM'])/df['psy_tsst_E1.0_PPM']

    df.loc[:, 'psy_tsst_E2.0_VagalResponse']=(df['psy_tsst_E2.0_HFPower']-df['psy_tsst_E1.0_HFPower'])/df['psy_tsst_E1.0_HFPower']
    df.loc[:, 'psy_tsst_E3.0_VagalResponse']=(df['psy_tsst_E3.0_HFPower']-df['psy_tsst_E1.0_HFPower'])/df['psy_tsst_E1.0_HFPower']
    df.loc[:, 'psy_tsst_E4.0_VagalResponse']=(df['psy_tsst_E4.0_HFPower']-df['psy_tsst_E1.0_HFPower'])/df['psy_tsst_E1.0_HFPower']

    # Adjust cortisol to covariate
    df['STQ_Cortisol'] = df['psy_manualCapture_Baseline_Cortisol_Abs']

    # Plot Correlation
    corr_columns = [c for c in df.columns if all(x not in c for x in ['Unnamed', 'Participant', 'Drink', 'Device'])]
    df[corr_columns].corr().to_csv(f'{out_path}Raw_Corr.csv')

    #Defragment df
    df = df.copy()

    # Prepare for Repeated Tools
    columns = [c for c in df.columns if 'psy' in c and all(excl not in c for excl in ['recall', '-Score', 'manual', 'MAS', 'POMS', 'PSQI', 'ECG', 'EDA'])]

    # Filter columns with sufficient non-null values
    participants = len(df['Participant'])
    columns = [c for c in columns if df[c].notna().sum() > (0.5 * participants)]

    # Prepare data
    data = df[columns + ['Drink', 'Participant', 'Device', 'BMI', 'Day', 'Number', 'StartTime']]

    # Rename columns to move condition to the second field
    new_cols = {
        c: '_'.join([c.split('_')[2], c.split('_')[1]] + c.split('_')[3:]) for c in columns
    }
    data.rename(columns=new_cols, inplace=True)

    # Identify tools and prepare tool list
    tools = [t for t in data.columns if 'Baseline' in t or 'Treatment' in t]
    tools = list(OrderedDict.fromkeys(['_'.join(t.split('_')[1:]) for t in tools]))

    # Save tools data
    tools_df = pd.DataFrame({'Tools': tools + [c for c in df.columns if 'tsst' in c]})
    tools_df.to_csv(f'{out_path}Tools_List.csv', index=False)

    # Calculate and save means grouped by 'Drink'
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    numeric_data = data[numeric_cols].apply(lambda x: pd.to_numeric(x, errors='coerce'))
    mean_results = numeric_data.groupby(data['Drink']).mean()
    mean_results.to_csv(f'{out_path}EVM_Results_Means.csv')

    # Check for outliers
    outlier_report = calculate_outliers(numeric_data)
    outlier_report.to_csv(f'{out_path}Outlier_Report.csv')

    # Save validation preparation
    data.to_csv(f'{out_path}Tool_Validation_Prep.csv', index=False)

    # Prepare effects data
    effects_data = df[['Drink', 'Participant', 'Device', 'BMI', 'Number', 'StartTime']]

    # Add covariate data to effects
    stq_cols = [c for c in df.columns if 'STQ' in c or ('PSQI' in c and 'Overall' in c)]
    effects_data = pd.concat([effects_data, df[stq_cols]], axis=1)
    effects_data.to_csv(f'{out_path}EVM_Effects.csv', index=False)
    data = pd.concat([data, df[stq_cols]], axis=1)

    # Print completion message
    print('Completed: Validation Stage Prep')


    MAS_Timings = pd.read_csv(f'{working_path}MAS_Timings.csv', index_col='Participant')
    POMS_Timings = pd.read_csv(f'{working_path}POMS_Timings.csv', index_col='Participant')
    manualCapture_Timings = pd.read_csv(f'{working_path}manualCapture_Timings.csv', index_col='Participant')

    grouping_variables = [g for g in control_key.keys()]
    information = list(effects_data.columns)

    in_path = f'C:/Users/ashra/Desktop/Distell/Results/'

    ########### GPT


    # path = in_path + file
    # df = pd.read_csv(path, low_memory = False)
    # df = df.loc[df['Drink'].isin(time_series_groups)]
    
    
    # replacement_key = {'BC:33:AC:AB:B3:DB':1,
    #                     'BC:33:AC:AB:B3:C8':2,
    #                     'BC:33:AC:AB:B4:51':3,
    #                     '#NAME':np.nan,
    #                     '?':np.nan}
    # df = df.replace(replacement_key)

    # cols = df.columns.difference(['Participant', 'Number', 'Drink', 'Device'])
    # df[cols] = df[cols].apply(pd.to_numeric, errors='coerce')
    # # Group by 'Drink' and check for each column
    # count_threshold = 10
    # valid_cols = []

    # for c in cols:
    #     # Group by 'Drink' and count non-null values for each column
    #     valid = df.groupby('Drink')[c].apply(lambda x: x.notna().sum() > count_threshold).all()
    #     if valid:
    #         valid_cols.append(c)

    # # Drop columns that don't meet the criteria
    # df = df[['Participant', 'Number', 'Drink', 'Device'] + valid_cols]

    # #### Add task-set-inhibiion
    # try:
    #     df.loc[:, 'psy_goNoGo_Baseline_TSI']=df['psy_goNoGo_Baseline_Task3_RTAccurate']-df['psy_goNoGo_Baseline_Task1_RTAccurate']
    #     df.loc[:, 'psy_goNoGo_Treatment_TSI']=df['psy_goNoGo_Treatment_Task3_RTAccurate']-df['psy_goNoGo_Treatment_Task1_RTAccurate']
    # except:
    #     print_error('Failed to find','goNoGo')
    

    # #### Add TSST SCL
    # df.loc[:, 'psy_tsst_E2.0_SCLResponse']=(df['psy_tsst_E2.0_MeanSCL']-df['psy_tsst_E1.0_MeanSCL'])/df['psy_tsst_E1.0_MeanSCL']
    # df.loc[:, 'psy_tsst_E3.0_SCLResponse']=(df['psy_tsst_E3.0_MeanSCL']-df['psy_tsst_E1.0_MeanSCL'])/df['psy_tsst_E1.0_MeanSCL']
    # df.loc[:, 'psy_tsst_E4.0_SCLResponse']=(df['psy_tsst_E4.0_MeanSCL']-df['psy_tsst_E1.0_MeanSCL'])/df['psy_tsst_E1.0_MeanSCL']

    # #### Add TSST SCL
    # df.loc[:, 'psy_tsst_E2.0_PhasicResponse']=(df['psy_tsst_E2.0_PPM']-df['psy_tsst_E1.0_PPM'])/df['psy_tsst_E1.0_PPM']
    # df.loc[:, 'psy_tsst_E3.0_PhasicResponse']=(df['psy_tsst_E3.0_PPM']-df['psy_tsst_E1.0_PPM'])/df['psy_tsst_E1.0_PPM']
    # df.loc[:, 'psy_tsst_E4.0_PhasicResponse']=(df['psy_tsst_E4.0_PPM']-df['psy_tsst_E1.0_PPM'])/df['psy_tsst_E1.0_PPM']

    # df.loc[:, 'psy_tsst_E2.0_VagalResponse']=(df['psy_tsst_E2.0_HFPower']-df['psy_tsst_E1.0_HFPower'])/df['psy_tsst_E1.0_HFPower']
    # df.loc[:, 'psy_tsst_E3.0_VagalResponse']=(df['psy_tsst_E3.0_HFPower']-df['psy_tsst_E1.0_HFPower'])/df['psy_tsst_E1.0_HFPower']
    # df.loc[:, 'psy_tsst_E4.0_VagalResponse']=(df['psy_tsst_E4.0_HFPower']-df['psy_tsst_E1.0_HFPower'])/df['psy_tsst_E1.0_HFPower']

    #     ### Adjust cortisol to covariate
    # df['STQ_Cortisol']=df['psy_manualCapture_Baseline_Cortisol_Abs']

    # ### Plot Correlation
    # corr_columns = [c for c in df.columns if all(x not in c for x in ['Unnamed','Participant','Drink','Device'])]
    # corr_data = df[corr_columns]
    # corr_data.corr().to_csv(f'{out_path}Raw_Corr.csv')

    # ###############################
    # ### Prepare for Repeated Tools
    # ###############################

    # # Adjust these when new tools are added
    # columns = df.columns
    # columns = [c for c in columns if 'psy' in c]
    # columns = [c for c in columns if 'recall' not in c]
    # columns = [c for c in columns if '-Score' not in c]
    # columns = [c for c in columns if 'manual' not in c]
    # columns = [c for c in columns if 'MAS' not in c]
    # columns = [c for c in columns if 'POMS' not in c]
    # columns = [c for c in columns if 'PSQI' not in c]
    # columns = [c for c in columns if 'ECG' not in c]
    # columns = [c for c in columns if 'EDA' not in c]

    # participants = len(df['Participant'])

    # columns = [c for c in columns if (len(df[c].dropna())>(0.5*participants))]

    # data = df[columns]

    # ## Move condition to second field
    # new_cols = {}
    # for c in columns:
    #     fields = c.split('_')
    #     new_fields = [fields[2],]+[fields[1],]+fields[3:]
    #     new_cols[c]='_'.join(new_fields)            
    # data.rename(columns = new_cols, inplace = True)

    # tools = data.columns
    # tools = [t for t in tools if ('Baseline' in t) or ('Treatment' in t)]
    # tools_new = ['_'.join(t.split('_')[1:]) for t in tools]
    # tools= list(OrderedDict.fromkeys(tools_new))
    # data['Drink']=df['Drink']
    # data['Participant']=df['Participant']
    # data['Device']=df['Device']
    # data['BMI']=df['BMI']
    # data['Day']=df['Day']
    # data['Number']=df['Number']
    # data['StartTime']=df['StartTime']

    # # First, identify which columns are numeric
    # numeric_cols = data.apply(lambda col: pd.to_numeric(col, errors='coerce')).notnull().any()

    # # Filter the DataFrame to keep only numeric columns
    # numeric_data = data.loc[:, numeric_cols]
    # numeric_data = numeric_data.applymap(lambda x: pd.NA if isinstance(x, str) else x)
    # numeric_data['Drink']=data['Drink']

    # # Now perform the grouping and mean calculation
    # mean_results = numeric_data.groupby('Drink').mean()

    # # Save the results to a CSV file
    # mean_results.to_csv(f'{out_path}EVM_Results_Means.csv')

    # # Check for outliers
    # outlier_report = calculate_outliers(numeric_data)
    # outlier_report.to_csv(f'{out_path}Outlier_Report.csv')

    # data.to_csv(f'{out_path}Tool_Validation_Prep.csv')

    # tools_df=pd.DataFrame()
    # tsst_tools =  [c for c in df.columns if ('tsst' in c)]
    # tools_df.loc[:, 'Tools']=tools+tsst_tools
    
    # tools_df.to_csv(f'{out_path}Tools_List.csv')

    # print('Completed: Validation Stage Prep')

    # #### What is this effects_data? Something that ends up as EVM_Effects
    # effects_data = pd.DataFrame()
    # effects_data['Drink']=df['Drink']
    # effects_data['Participant']=df['Participant']
    # effects_data['Device']=df['Device']
    # effects_data['BMI']=df['BMI']
    # effects_data['Number']=df['Number']
    # effects_data['StartTime']=df['StartTime']
    
    # ### Add in covariate data to Effects
    # stq_cols = [c for c in df.columns if ('STQ' in c) or (('PSQI' in c) and ('Overall' in c))]

    # # Concatenate all selected columns at once
    # effects_data = pd.concat(effects_data,[df[stq_cols]], axis=1)
    # data = pd.concat([data, df[stq_cols]], axis=1)

    
    # effects_data.to_csv(f'{out_path}EVM_Effects.csv')
    
    ############################
    #####  Paired T-Test Data
    ############################
    
    # print_status('Beginning','Paired T-Test')
    # results = pairedttest(data)    
    # check_tool_covariates(results['Tool'].drop_duplicates().tolist())

    # results.to_csv(f'{out_path}Tool_Validation_Stage0.csv')
    # print_status('Completed', 'Paired T-Test')

    ###########################
    ####  Independent T-Test Data
    ###########################
    results = pd.read_csv(f'{in_path}Tool_Validation_Stage0.csv')
    effects_data = pd.read_csv(f'{in_path}EVM_Effects.csv')

    results = pd.read_csv(f'{out_path}Tool_Validation_Stage0.csv')
    print_status('Beginning', 'Independent T-Test')

    #### Neuropsych Repeated Measures
    print_step('Beginning', 'Neuropsych')
    paired_tools = results['Tool'].drop_duplicates()#.loc[results['isUseful']==True]['Tool'].drop_duplicates()


    #results = pd.DataFrame()
    for p in paired_tools:
        control_col = [c for c in data.columns if (p in c) and ('Baseline' in c)][0]
        treatment_col = [c for c in data.columns if (p in c) and ('Treatment' in c)][0]

        effect = data.copy()
        effect.loc[:, 'Effect'] = effect[treatment_col]-effect[control_col]
        effects_data.loc[:, f'{p}'] = effect[treatment_col]-effect[control_col]
        effects_data.to_csv(f'{out_path}EVM_Effects.csv')

    ### Do a final check on data:
    check = []
    for index, row in effects_data.iterrows():
        result = {}
        result['Participant']= row['Participant']
        result['OpenSignalsData'] = 0 if np.isnan(row['Device']) else 1
        result['STQ'] = 0 if np.isnan(row['STQ_Active']) else 1
        try:
            result['PSQI'] = 0 if np.isnan(row['PSQI_Overall']) else 1
        except:
            pass
        result['Cortisol'] = 0 if np.isnan(row['STQ_Cortisol']) else 1
        result['PsychoPy'] = 0 if np.isnan(row['emotionRecognition_Overall_Accuracy']) else 1
        check.append(result)

    check = pd.DataFrame(check)
    check.to_csv(f'{out_path}Final_Check.csv')
    
    #results = pd.DataFrame()
    for p in paired_tools:
        effect = effects_data[information+[p]]
        for g in grouping_variables:
            result = get_repeated_valid(effect,p,p,control_key[g],g)
            results = pd.concat([results, pd.DataFrame([result])], ignore_index=True)

    results = results.copy()
    effects_data = effects_data.copy()
    effects_data.to_csv(f'{out_path}EVM_Effects.csv')

    print_step('Completed', 'Neuropsych')

    ### Recall
    #Get racall col
    print_step('Beginning', 'Recall')

    rc = [c for c in df.columns if 'recall' in c][0]
    effects_data.loc[:, f'Recall'] = df[rc]
    effects_data.to_csv(f'{out_path}EVM_Effects.csv')
    
    fields = rc.split('_')
    new_fields = [fields[1],]+fields[3:]
    new_rc='_'.join(new_fields)
    df.rename(columns = {rc:new_rc}, inplace = True)
    result = dict()
    for g in grouping_variables:
        result = get_repeated_valid(df,'Recall', new_rc,control_key[g],g)
        results = pd.concat([results, pd.DataFrame([result])], ignore_index=True)
    
    print_step('Completed', 'Recall')

    ### TSST Results
    print_step('Beginning', 'TSST')

    cols =  [c for c in df.columns if ('tsst' in c)]
    check_tool_covariates(cols)

    for c in cols:
        effects_data.loc[:, f'{"_".join(c.split("_"))}'] = df[c]
        for g in grouping_variables:
            result = get_repeated_valid(df,c,c,control_key[g],g)
            results = pd.concat([results, pd.DataFrame([result])], ignore_index=True)
    print_step('Completed', 'TSST')

    results.to_csv(f'{out_path}Tool_Validation_Stage1.csv')
    effects_data.to_csv(f'{out_path}EVM_Effects.csv')
    results = results.copy()
    effects_data = effects_data.copy()   
    print('Completed: Validation Stage 1')  
    print_status('Completed', 'Independent T-Test')
    ############################
    ##### Time Series Data
    ############################
    results = pd.read_csv(f'{out_path}Tool_Validation_Stage1.csv')
    effects_data = pd.read_csv(f'{out_path}EVM_Effects.csv')
    print_status('Beginning', 'Time Series')

    icolumns = df.columns
    columns_list = dict()
    columns_list['MAS'] = [c for c in icolumns if '-Score' in c and ('Overall' not in c) and ('MAS' in c)]
    columns_list['Manual'] = [c for c in icolumns if ('manual' in c) and ('Overall' not in c) and ('Abs' not in c)]
    columns_list['POMS'] = [c for c in icolumns if (('-Score' in c) or ('_TMD' in c)) and ('Overall' not in c) and ('POMS' in c)]

    condition_list = {'MAS':{'Baseline':0,
                        'TreatmentTasksStart':72,
                        'TreatmentTasksEnd':120,                   
                        },
                      'Manual':{'Baseline':0,
                    'TreatmentBeforeTasks':3,
                    'TreatmentAfterTasks':4,
                        'BeforeTreatment':1,
                        'TreatmentInitial':2,                    
                    },
                    'POMS':{'Baseline':0,
                        'TreatmentTasksStart':72,
                        'TreatmentTasksEnd':120,                   
                        },}

    timings_list = {'MAS':MAS_Timings,
                    'Manual':manualCapture_Timings,
                    'POMS':POMS_Timings}

    exlist = ['E1','E2','E3','E4','E5','E6','E7','E8','E9']

    cv_cols = [c for c in df.columns if ('STQ' in c) 
             or ('BMI' in c)
             or ('StartT' in c)
             or ('Cort' in c)
             or ('Devi' in c)
             or (('PSQI' in c) and ('Overall' in c))]
    

    ##### For each type of data
    time_series_tools = []

    for key in columns_list:
        columns = columns_list[key]
        data = df[columns]
        new_cols = {}
        conditions = []
        #print(f'Key:{key}')
    
        #Rename columns
        for c in columns:
            if any(x in c for x in exlist):
                fields = c.split('_')
                conditions = conditions + [fields[2]]
                new_fields = [fields[2],]+[fields[1],]+fields[3:]
                new_cols[c]='_'.join(new_fields)
            else:
                fields = c.split('_')
                conditions = conditions + [fields[2]]
                new_fields = [fields[2],]+fields[3:]
                new_cols[c]='_'.join(new_fields)
        data.rename(columns = new_cols, inplace = True)
    
        #Reassemble df
        data['Drink']=df['Drink']
        data['Participant']=df['Participant']
        split = pd.DataFrame()
        conditions= list(OrderedDict.fromkeys(conditions))
        condition_key = condition_list[key] 
    
        for c in conditions:
            cols = [x for x in data.columns if c in x]
            for i in cols:
                result = pd.DataFrame()
                condition = data[['Participant','Drink',i]]
                result['Participant'] = condition['Participant']
                result['Drink']=condition['Drink']
                result['Tool']='_'.join(i.split('_')[1:])
                result['Value']=condition[i]
                condition = condition.set_index('Participant')
                for index,row in condition.iterrows():
                    try:
                        result.loc[result['Participant']==index, 'Time']=timings_list[key].loc[index][c]/60
                    except:
                        result.loc[result['Participant']==index, 'Time']=timings_list[key][c].mean()/60
                split = pd.concat([split, result])

        if len(split):
       
            tools = split['Tool'].drop_duplicates().dropna().tolist()
        
            # Only work on first epoch data
            exlist2 = ['E2','E3','E4','E5','E6','E7','E8','E9']
            tools = [t for t in tools if not(any(x in t for x in exlist2))]
            #print(f'Tools:{tools}')
        
            for t in tools:
                    
                    #print(f'Now Working: {t}')
                    result = {}
                    data = split.loc[(split['Tool']==t)].dropna()# & (split['Drink']==g)
                    
                    dresults = pd.DataFrame()
                    participants = data.loc[data['Participant'].str.contains('|'.join(time_series_groups))]['Participant'].drop_duplicates()
                    #participants = data['Participant'].drop_duplicates()
                    tn =t.replace('.','_')
                    tn =t.replace('/','_')
                    
                    path = f'{out_path}tool_results/'
                    os.makedirs(path, exist_ok=True)
                    data.to_csv(f'{path}{tn}.csv')
                    for p in participants:
                        dresult=dict()
                        pdata = data.loc[data['Participant']==p].sort_values(by='Time')
                        x = pdata['Time'].to_numpy()
                        y = pdata['Value'].to_numpy()

                        dresult['Participant'] = pdata['Participant'].values[0]
                        dresult['Drink']=pdata['Drink'].values[0]

                        if y.sum():
                            if len(y)==len(conditions):
                                y = y-y[0] #Correct for baseline
                                
                                dresult['Effect']=integrate.trapezoid(y,x=x)                              
                                if ~(np.isnan(y).any()):
                                    dresult['Trend'] = np.polyfit(x, y, 1)[0]
                                    effects_data.loc[effects_data['Participant']==p, f'{t}_Trend'] = dresult['Trend']
                                dresult['Range'] = y[-1]-y[0]                           
                                effects_data.loc[effects_data['Participant']==p, f'{t}_Effect'] = dresult['Effect']
                                effects_data.loc[effects_data['Participant']==p, f'{t}_Range'] = dresult['Range']

                                for c in cv_cols:
                                        dresult[c]=df.loc[df['Participant']==p][c].values[0]

                                dresults = pd.concat([dresults, pd.DataFrame([dresult])], ignore_index=True)
                    
                        else:
                            pass
                            
                    if t=='manualCapture_E1.0_MeanHR':
                        dresults.to_csv(f'{out_path}meanHR.csv')

                    for g in grouping_variables:
                        dvs = ['Effect','Trend','Range']
                        for dv in dvs:
                            try:
                                time_series_tools.append(f'{t}_{dv}')
                                result = get_repeated_valid(dresults,f'{t}_{dv}',dv,control_key[g],g)
                                results = pd.concat([results, pd.DataFrame([result])], ignore_index=True)
                            except:
                                print(f'Failed to validate: {t} {g} {dv}')
    
    ################# It looks like you have to run this to completion, it will fail, and then add covariates for the following
    time_series_tools_df = pd.DataFrame({'Tool':time_series_tools})
    time_series_tools_df.to_csv(f'{out_path}Time_Series_Tools.csv')
    print_status('Completed', 'Time Series')


    for index, row in tools_covariates.iterrows():
        tool = row['Tool']
        results.loc[results['Tool']==tool, 'Axis']=row['Axis']

    results.to_csv(f'{out_path}Tool_Validation_Stage2.csv')
    effects_data.to_csv(f'{out_path}EVM_Effects.csv')
    #effects_data.corr().to_csv(f'{out_path}Effects_Corr.csv')

    #effects_data.groupby('Drink').mean().to_csv(f'{out_path}EVM_Effects_Means.csv')
    print('Completed: Validation Stage 2')

    allsig = pd.read_csv(f'{out_path}Tool_Validation_Stage2.csv')

    #Tools
    #Get tools in results
    tools = allsig.loc[allsig['pV-ANCOVA']<0.05]['Tool'].drop_duplicates()
    #Filter our non-significant results
    allsig = allsig.loc[allsig['Tool'].isin(tools)]
    allsig = allsig.loc[allsig['Method']=='LinearRegression']
    allsig.loc[:,'Value']=allsig['EffSize']*np.sign(allsig['Sign'])
    allsig['Value']=allsig['Value']*(1-allsig['pV-ANCOVA'])
    allsig['Value']=allsig['Value']*100
    allsig.to_csv(f'{out_path}Tool_Validation_Stage3.csv')

    ##################################################################################

    print('Completed: Pipeline')
    ############################################################################################
    ############################################################################################


def matched():
    '''
    Basic T-Test to check if tools are valid
    '''
   

    # Load data from a specified CSV file into a DataFrame
    data = pd.read_csv(f'C:/Users/ashra/Desktop/Distell/Results/EVM_Effects.csv')
    
    # Create a copy of the DataFrame to work on
    dresults = data.copy()

    # Identify columns related to specific tools (STQ, BMI, StartT) using list comprehension
    tools = [c for c in dresults.columns if ('STQ' in c) 
             or ('BMI' in c)
             or ('StartT' in c)
             or ('PSQI') in c]
    
    # Define a list of variables by which data will be grouped
    grouping_variables = ['F','O','T','X']
    
    # Dictionary mapping each group to a control group
    control_key = {'B':'A', 'C':'A', 'D':'A', 'E':'A', 'F':'J', 'G':'J', 'O':'S','P':'S','Q':'S','R':'S','T':'S','X':'S'}
    # Initialize an empty DataFrame to store results
    results = pd.DataFrame()

    # Nested loops to process data for each tool and grouping variable
    for t in tools:
        for g in grouping_variables:
        
            # Extract control group data for the current tool and drop missing values
            control = dresults.loc[dresults['Drink']==control_key[g]][t].dropna().to_numpy()
            
            # Extract treatment group data for the current tool and drop missing values
            treatment = dresults.loc[dresults['Drink']==g][t].dropna().to_numpy()

            # Perform a Bayesian t-test between control and treatment groups
            bayesTest = pg.ttest(control, treatment, correction=True, paired = False)
            
            # Collect results into a dictionary
            result=dict()
            result['Bayes'] = float(bayesTest['BF10'].values[0])  # Bayesian Factor
            result['EffSize'] = float(bayesTest['cohen-d'].values[0])  # Effect size
            result['Mean'] = treatment.mean()-control.mean()  # Mean difference
            isUseful = True if float(bayesTest['BF10'].values[0]) >= 3 else False  # Evaluate usefulness

            # Additional details for results
            result['isUseful'] = isUseful
            result['Tool'] = f'{t}'
            result['Method'] = 'IndependentTTest'
            result['Group'] = g
            result['nC'] = len(control)  # Sample size of control group
            result['nT'] = len(treatment)  # Sample size of treatment group
            result['pValue'] = float(bayesTest['p-val'].values[0])  # P-value from the test
            
            # Append the result dictionary as a DataFrame to the results DataFrame
            results = pd.concat([results, pd.DataFrame([result])], ignore_index=True)
        
    # Save the final results to a CSV file
    results.to_csv(f'C:/Users/ashra/Desktop/Distell/Results/MatchedCheck.csv')


def individual_analysis(batch_name, groups):
    ###############
    ## FIRST RUN THE WHOLE THING TO UPDATE EVM_EFFECTS ##
    ## Be sure to comment out paired t-test data so that new Effects file is used

    ###############
    #### First copy effects file
    os.makedirs(f'C:/Users/ashra/Desktop/Distell/Analyses/{batch_name}/', exist_ok=True)
    shutil.copy(f'C:/Users/ashra/Desktop/Distell/Results/EVM_Effects.csv', 
                f'C:/Users/ashra/Desktop/Distell/Analyses/{batch_name}/EVM_Effects.csv')

    #### Perform group vs group analysis
    analyses = []
    for combo in combinations(groups, 2):
        control = combo[0]
        treatment = combo[1]
        control_key = {treatment:control,}
        control_groups = [control]
        treatment_key = {control:[treatment]}
        time_series_groups = [treatment,control]
        analysis = f'{treatment}vs{control}'
        analyses.append(analysis) 

        print(f'Beginning {analysis}')

        path = f'C:/Users/ashra/Desktop/Distell/Analyses/{batch_name}/{analysis}/'
        os.makedirs(path, exist_ok=True)

        pipeline(path, control_key, control_groups, treatment_key, time_series_groups)
        print(f'Completed {analysis}')


def build_results(batch_name, groups):
    #### Combine analyses
    

    analyses = [f'{combo[1]}vs{combo[0]}' for combo in combinations(groups, 2)]
    all_results = [pd.read_csv(f'C:/Users/ashra/Desktop/Distell/Analyses/{batch_name}/{analysis}/Tool_Validation_Stage2.csv') for analysis in analyses]
    all_results = pd.concat(all_results)

    drink_dict = {'A':'Water 2022',
        'B':'Alcohol',
        'C':'Caffiene',
        'D':'Rhodiola',
        'E':'Valerian',
        'F':'CalmV1',
        'G':'G-Lift',
        'H':'H-Boost',
        'I':'I-Focus',
        'K':'K-Numuti Unwind',
        'L':'L-Numuti Thrive',
        'M':'M-Redbull',
        'N':'N-Goodmind',
        'O':'CalmV2',
        'P':'LiftV2',
        'Q':'BoostV2',
        'R':'FocusV2',
        'S':'Water 2024',
        'X':'CalmV2Rework',
        'T':'Viridian',
        'J':'Water 2023'
        }
    
    library = pd.read_csv(f'C:/Users/ashra/Desktop/Distell/Analyses/Library.csv')
    covariate_columns = [c for c in all_results.columns if 'Coeff-' in c]
    keep = ['Axis','Tool', 'Control', 'meanC', 'Group', 'meanT', 'Sign', 'nC', 'nT', 'EffSize', 'pV-ANCOVA', 'MainEffect', 'Covariates'] + covariate_columns
    all_results = all_results[keep]
    
    for key, value in drink_dict.items():
        all_results.loc[all_results['Control']==key, 'Control'] = value
        all_results.loc[all_results['Group']==key, 'Group'] = value

    all_results['Sign']=np.sign(all_results['meanT']-all_results['meanC'])
    all_results = all_results.loc[all_results['pV-ANCOVA']<0.05]
    all_results = all_results.loc[(all_results['nC']>19) & (all_results['nT']>19)]
    all_results = all_results.loc[all_results['MainEffect']=='Condition']
    all_results.loc[:, 'NValue']= all_results['EffSize']*(1-all_results['pV-ANCOVA'])*100*all_results['Sign']
    all_results = all_results.sort_values(by=['Axis','Tool'])
    all_results['Min Sample'] = np.where(all_results['nC'] > all_results['nT'], all_results['nT'], all_results['nC'])
    all_results['Significance Shorthand'] = np.where(all_results['pV-ANCOVA'] > 0.001, '=' + np.round(all_results['pV-ANCOVA'],3).astype(str), '<0.001')
    all_results['Significance Footnote'] = all_results['Group'] + ' relative to ' + all_results['Control'] + ', p' + all_results['Significance Shorthand'] + ', (N=' + all_results['Min Sample'].astype(str) + ')'
    all_results['Direction'] = np.where(all_results['Sign']<0, ' decreases ', ' increases ')
    all_results = pd.merge(all_results, library[['Tool','Mechanism','Metric']], left_on='Tool', right_on='Tool', how='left')
    all_results['Interpretation'] = all_results['Group'] + all_results['Direction'] + all_results['Metric'] + ' relative to '+all_results['Control'] + ' through ' + all_results['Mechanism'] + ' by ' + np.round(all_results['NValue'],2).astype(str) + ' points' 

    all_results.to_excel(f'C:/Users/ashra/Desktop/Distell/Analyses/{batch_name}/All_Results.xlsx')

    print(f'Complete')
    
    # Clean results file

def build_results3(batch_name, groups):
    #### Combine analyses
    

    analyses = [f'{combo[1]}vs{combo[0]}' for combo in combinations(groups, 2)]
    all_results = [pd.read_csv(f'C:/Users/ashra/Desktop/Distell/Analyses/{batch_name}/{analysis}/Tool_Validation_Stage2.csv') for analysis in analyses]
    all_results = pd.concat(all_results)

    drink_dict = {'A':'Water 2022',
        'B':'Alcohol',
        'C':'Caffiene',
        'D':'Rhodiola',
        'E':'Valerian',
        'F':'CalmV1',
        'G':'G-Lift',
        'H':'H-Boost',
        'I':'I-Focus',
        'K':'K-Numuti Unwind',
        'L':'L-Numuti Thrive',
        'M':'M-Redbull',
        'N':'N-Goodmind',
        'O':'CalmV2',
        'P':'LiftV2',
        'Q':'BoostV2',
        'R':'FocusV2',
        'S':'Water 2024',
        'X':'CalmV2Rework',
        'T':'Viridian',
        'J':'Water 2023'
        }
    
    library = pd.read_csv(f'C:/Users/ashra/Desktop/Distell/Analyses/Library.csv')
    covariate_columns = [c for c in all_results.columns if 'Coeff-' in c]
    keep = ['Axis','Tool', 'Control', 'meanC', 'Group', 'meanT', 'Sign', 'nC', 'nT', 'EffSize','pV-TTest', 'pV-ANCOVA', 'MainEffect', 'Covariates'] + covariate_columns
    all_results = all_results[keep]
    
    for key, value in drink_dict.items():
        all_results.loc[all_results['Control']==key, 'Control'] = value
        all_results.loc[all_results['Group']==key, 'Group'] = value

    all_results['Sign']=np.sign(all_results['meanT']-all_results['meanC'])
    #all_results = all_results.loc[all_results['pV-ANCOVA']<0.05]
    all_results = all_results.loc[(all_results['nC']>19) & (all_results['nT']>19)]
    all_results = all_results.loc[all_results['MainEffect']=='Condition']
    all_results.loc[:, 'NValue']= all_results['EffSize']*(1-all_results['pV-ANCOVA'])*100*all_results['Sign']
    all_results = all_results.sort_values(by=['Axis','Tool'])
    all_results['Min Sample'] = np.where(all_results['nC'] > all_results['nT'], all_results['nT'], all_results['nC'])
    all_results['Significance Shorthand'] = np.where(all_results['pV-ANCOVA'] > 0.001, '=' + np.round(all_results['pV-ANCOVA'],3).astype(str), '<0.001')
    all_results['Significance Footnote'] = all_results['Group'] + ' relative to ' + all_results['Control'] + ', p' + all_results['Significance Shorthand'] + ', (N=' + all_results['Min Sample'].astype(str) + ')'
    all_results['Direction'] = np.where(all_results['Sign']<0, ' decreases ', ' increases ')
    all_results = pd.merge(all_results, library[['Tool','Mechanism','Metric']], left_on='Tool', right_on='Tool', how='left')
    all_results['Interpretation'] = all_results['Group'] + all_results['Direction'] + all_results['Metric'] + ' relative to '+all_results['Control'] + ' through ' + all_results['Mechanism'] + ' by ' + np.round(all_results['NValue'],2).astype(str) + ' points' 
    all_results['Diff']=all_results['pV-ANCOVA']<all_results['pV-TTest']
    all_results.to_excel(f'C:/Users/ashra/Desktop/Distell/Analyses/{batch_name}/All_Results_Non.xlsx')
    
    print(f'Complete')


def build_results2(batch_name, groups):
    
    effects = pd.read_csv(f'C:/Users/ashra/Desktop/Distell/Analyses/{batch_name}/EVM_Effects.csv')
    effects = effects.set_index('Drink')
    library = pd.read_csv(f'C:/Users/ashra/Desktop/Distell/Analyses/Library.csv').set_index('Tool').to_dict(orient='index')

    #tools = pd.read_excel(f'C:/Users/ashra/Desktop/Distell/Analyses/{batch_name}/All_Results.xlsx')

    tools = effects.columns.difference(['Participant', 'Number', 'Drink', 'Device'])

    colors = {'O':'deepskyblue',
              'S':'grey',
              'T':'cyan',
              'X':'steelblue',
              'F':'lightskyblue',}
    
    drink_dict = {'A':'Water 2022',
        'B':'Alcohol',
        'C':'Caffiene',
        'D':'Rhodiola',
        'E':'Valerian',
        'F':'CalmV1',
        'G':'G-Lift',
        'H':'H-Boost',
        'I':'I-Focus',
        'K':'K-Numuti Unwind',
        'L':'L-Numuti Thrive',
        'M':'M-Redbull',
        'N':'N-Goodmind',
        'O':'CalmV2',
        'P':'LiftV2',
        'Q':'BoostV2',
        'R':'FocusV2',
        'S':'Water 2024',
        'X':'CalmV2Rework',
        'T':'Viridian',
        'J':'Water 2023'
        }

    #### Plot box and whisker
    results = []

    for tool in tools:
        stats = {}
        for drink in groups:
            stats[drink]=effects.loc[drink][tool].dropna().to_numpy()
        results.append(get_significance(stats,tool,groups))


    all_results = pd.concat(results)

    all_results.to_excel(f'C:/Users/ashra/Desktop/Distell/Analyses/{batch_name}/All_Results2.xlsx')
    
    # for key, value in drink_dict.items():
    #     all_results.loc[all_results['Control']==key, 'Control'] = value
    #     all_results.loc[all_results['Group']==key, 'Group'] = value

    # ####
    # # Swap out water so it is always the control group
    # ####

    # all_results['Sign']=np.sign(all_results['meanT']-all_results['meanC'])
    # all_results = all_results.loc[all_results['pV-ANCOVA']<0.05]
    # all_results = all_results.loc[(all_results['nC']>19) & (all_results['nT']>19)]
    # all_results.loc[:, 'NValue']= all_results['EffSize']*(1-all_results['pV-ANCOVA'])*100*all_results['Sign']
    # all_results = all_results.sort_values(by=['Axis','Tool'])
    # all_results['Min Sample'] = np.where(all_results['nC'] > all_results['nT'], all_results['nT'], all_results['nC'])
    # all_results['Significance Shorthand'] = np.where(all_results['pV-ANCOVA'] > 0.001, '=' + np.round(all_results['pV-ANCOVA'],3).astype(str), '<0.001')
    # all_results['Significance Footnote'] = all_results['Group'] + ' relative to ' + all_results['Control'] + ', p' + all_results['Significance Shorthand'] + ', (N=' + all_results['Min Sample'].astype(str) + ')'
    # all_results['Direction'] = np.where(all_results['Sign']<0, ' decreases ', ' increases ')
    #all_results = pd.merge(all_results, library[['Tool','Mechanism','Metric']], left_on='Tool', right_on='Tool', how='left')
    #all_results['Interpretation'] = all_results['Group'] + all_results['Direction'] + all_results['Metric'] + ' relative to '+all_results['Control'] + ' through ' + all_results['Mechanism'] + ' by ' + np.round(all_results['NValue'],2).astype(str) + ' points' 

    all_results.to_excel(f'C:/Users/ashra/Desktop/Distell/Analyses/{batch_name}/All_Results2.xlsx')

    print(f'Complete')

    pass


def build_results4(batch_name, groups):
    
    effects = pd.read_csv(f'C:/Users/ashra/Desktop/Distell/Analyses/{batch_name}/EVM_Effects.csv')
    library = pd.read_csv(f'C:/Users/ashra/Desktop/Distell/Analyses/Library.csv').set_index('Tool').to_dict(orient='index')

    #tools = pd.read_excel(f'C:/Users/ashra/Desktop/Distell/Analyses/{batch_name}/All_Results.xlsx')

    tools = effects.columns.difference(['Participant', 'Number', 'Drink', 'Device','BMI'])

    colors = {'O':'deepskyblue',
              'S':'grey',
              'T':'cyan',
              'X':'steelblue',
              'F':'lightskyblue',}
    
    drink_dict = {'A':'Water 2022',
        'B':'Alcohol',
        'C':'Caffiene',
        'D':'Rhodiola',
        'E':'Valerian',
        'F':'CalmV1',
        'G':'G-Lift',
        'H':'H-Boost',
        'I':'I-Focus',
        'K':'K-Numuti Unwind',
        'L':'L-Numuti Thrive',
        'M':'M-Redbull',
        'N':'N-Goodmind',
        'O':'CalmV2',
        'P':'LiftV2',
        'Q':'BoostV2',
        'R':'FocusV2',
        'S':'Water 2024',
        'X':'CalmV2Rework',
        'T':'Viridian',
        'J':'Water 2023'
        }

    #### Plot box and whisker
    results = []
    effects = effects.loc[effects['Drink'].isin(groups)]
    for tool in tools:
        covariates = []
        for drink in groups:
            covariates.append(get_covariates(drink,tool))
            if (drink in ['A','B','C','D','E','I','K','F','G','H','J','L','M','N']):
                covariates = [c for c in covariates if 'PSQI' not in c]

        covariates = list(OrderedDict.fromkeys(flatten(covariates)))
        results.append(get_significance(effects,dv=tool,gv='Drink',covariates=covariates, cluster=tool))
            

        

    all_results = pd.concat(results)
    all_results.to_excel(f'C:/Users/ashra/Desktop/Distell/Analyses/{batch_name}/All_Results4.xlsx')
    

    print(f'Complete')

    pass


def build_visuals(batch_name, groups):
    
    effects = pd.read_csv(f'C:/Users/ashra/Desktop/Distell/Analyses/{batch_name}/EVM_Effects.csv')
    effects = effects.set_index('Drink')
    library = pd.read_csv(f'C:/Users/ashra/Desktop/Distell/Analyses/Library.csv').set_index('Tool').to_dict(orient='index')

    tools = pd.read_excel(f'C:/Users/ashra/Desktop/Distell/Analyses/{batch_name}/All_Results.xlsx')

    tools = tools['Tool'].unique()

    colors = {'O':'cornflowerblue',
              'S':'grey',
              'T':'turquoise',
              'X':'steelblue',
              'F':'lightskyblue',}
    
    drink_dict = {'A':'Water 2022',
        'B':'Alcohol',
        'C':'Caffiene',
        'D':'Rhodiola',
        'E':'Valerian',
        'F':'CalmV1',
        'G':'G-Lift',
        'H':'H-Boost',
        'I':'I-Focus',
        'K':'K-Numuti Unwind',
        'L':'L-Numuti Thrive',
        'M':'M-Redbull',
        'N':'N-Goodmind',
        'O':'Calm',
        'P':'LiftV2',
        'Q':'BoostV2',
        'R':'FocusV2',
        'S':'Control',
        'X':'Calm Retest',
        'T':'Viridian',
        'J':'Water 2023'
        }

    #### Plot box and whisker
    groups = ['S','O','X','T']
    for tool in tools:
        path = f'C:/Users/ashra/Desktop/Distell/Analyses/{batch_name}/BoxAndWhiskerData/{tool}/'
        os.makedirs(path, exist_ok=True)
        try:
            data = pd.DataFrame()

            x_data = groups
            y_data = dict()
            
            for drink in groups:
                values = effects.loc[drink][tool].tolist()
                values.extend([np.nan] * (50 - len(values)))
                data[drink_dict[drink]]=values
                y_data[drink] = dict()
                y_data[drink]['values'] =effects.loc[drink][tool].dropna().to_numpy()
                y_data[drink]['label'] = f'{drink_dict[drink]}'# (n={len(effects.loc[drink][tool].dropna().to_numpy())})'
                y_data[drink]['color'] = colors[drink]

            multi_box(f'{path}',
                    x_data,
                    y_data,
                    xlabel = '',
                    ylabel = library[tool]['Mechanism'],
                    title = library[tool]['Metric'],
                    )
            
            tool = tool.replace('/','')
            print(f'{path}{tool}.csv')       
            data.to_csv(f'{path}{tool}.csv')

        except:
            print(f'Failed: {tool}')

    pass

drink_dict = {'B':'Alcohol',
        'C':'Caffiene',
        'D':'Rhodiola',
        'E':'Valerian',
        'F':'F-Calm',
        'G':'G-Lift',
        'H':'H-Boost',
        'I':'I-Focus',
        'K':'K-Numuti Unwind',
        'L':'L-Numuti Thrive',
        'M':'M-Redbull',
        'N':'N-Goodmind',
        'O':'O-Calm V2',
        'P':'P-Lift V2',
        'Q':'Q-Boost V2',
        'R':'R-Focus'
        }

def final_analysis():
    batches = {'Calm2V2':['O','X','T','S'],}
    #batches = {'CalmVCalm':['O','X'],}
    #batches = {'Calm2V2':['O','S','T','X','F'],}
    #batches = {'ControlCalmV2':['S','J'],}
    #batches = {'Water':['A','S','J'],}

    for batch, conditions in batches.items():
        #individual_analysis(batch, conditions)
        #build_results2(batch, conditions)
        #build_results3(batch, conditions)
        build_results4(batch, conditions)
        #build_visuals(batch, conditions)


def std_axes(ax, spines=[], annotation_color = 'white' , x_ticks=None, x_labels=None, xlabel=None, ylabel=None, title=None, ylim=None, ymin = None, ymax= None, titlef=12, tickf=6, labelf=8, second = False ):   
    # change the style of the axis spines
    plt.rcParams['font.family'] = "Century Gothic"
    #plt.rcParams['font.sans-serif'] = font1.get_name()
    #plt.rcParams['text.color'] = 'white'

    all = ['top','bottom','left','right']
    for spine in all:
        ax.spines[spine].set_linewidth(0.8)
        if spine in spines:
            #ax.spines[spine].set_position(('outward',0.5))
            ax.spines[spine].set_color(annotation_color)
        else:
            ax.spines[spine].set_visible(False)
   

    ax.set_xticks(x_ticks) if x_ticks is not None else None
    ax.set_xticklabels(x_labels, fontsize=tickf, rotation=45, ha = 'right', color= annotation_color) if x_labels is not None else None
    ax.tick_params(axis='both', labelsize=tickf, color= annotation_color, labelcolor= annotation_color)
    
    ax.set_xlabel(xlabel,fontsize=labelf, color=annotation_color) if xlabel is not None else None
    ax.set_ylabel(ylabel,fontsize=labelf, color=annotation_color, labelpad = 20) if ylabel is not None else None
    ax.set_ylabel(ylabel,fontsize=labelf, color=annotation_color, rotation=270, labelpad = 20) if ylabel is not None and second is True else None
    ax.set_title(title, fontsize=titlef, color=annotation_color) if title is not None else None 
    
    if ylim is not None:
        ax.set_ylim(0,ylim)
    elif ymin is not None:
        ax.set_ylim(ymin,ymax)  
    else:
        pass
    
    #ax.yaxis.grid()
    return ax


def multi_box(out_folder, x_data, y_data, xlabel = '', ylabel='', title = '', tags =['',], highlight = {'':'',}, ylim =None, footnote = None, footnoteLines=None, tag = None, minorTicks=None):
    """
        Generates and saves boxplot visualizations for given datasets.

        This function creates boxplots for multiple datasets, allowing customization of labels, titles, colors, and other plot features. It saves the generated plots to specified output folders, with options for additional tagging and minor tick adjustments.

        Parameters:
        - out_folder (str): The path to the output folder where the plots will be saved.
        - x_data (list): The x-axis labels for the boxplot.
        - y_data (dict): A dictionary where keys are groups and values contain 'values' (list of data points), 'label' (str for the legend), and 'color' (str for the box color).
        - xlabel (str, optional): The label for the x-axis. Default is an empty string.
        - ylabel (str, optional): The label for the y-axis. Default is an empty string.
        - title (str, optional): The title of the plot. Default is an empty string.
        - tags (list, optional): A list of tags for categorizing plots. Default is a list with an empty string.
        - highlight (dict, optional): A dictionary to specify elements to highlight, with keys as element identifiers and values as styles/colors. Default is an empty dictionary.
        - ylim (tuple, optional): A tuple specifying the limits for the y-axis (min, max). Default is None.
        - footnote (str, optional): Text for a footnote to be added to the plot. Default is None.
        - footnoteLines (int, optional): The number of lines for the footnote. This parameter is currently unused in the function. Default is None.
        - tag (str, optional): A specific tag to append to the filename for the saved plot. Default is None.
        - minorTicks (bool, optional): Whether to include minor ticks on the y-axis. If True, minor ticks are added. Default is None.

        Returns:
        - None: The function saves the generated plots to files and does not return any value.

        Notes:
        - The function saves two sets of plots: one set with the 'box' prefix and another with the 'sig' prefix in the filenames. The distinction or purpose of these sets should be clarified as needed.
        - The function makes use of `plt.figtext` for adding footnotes, but it seems to be misplaced outside of the plot generation context (after `plt.close()`), which might not work as intended.
        - Ensure the 'out_folder' exists or has the appropriate permissions for file operations.
        - The function prints progress and completion messages to the console.
    """
    out_path = f"{out_folder}"
    os.makedirs(out_path, exist_ok=True)    
    
    print(f">> Running: Plotting Data for {title}")

    x = np.arange(len(x_data))  # the label locations
    
    fig = plt.figure()
    
    ax1 = std_axes(fig.add_subplot(111),
                spines=['left','bottom'],
                annotation_color = 'black',
                xlabel=xlabel,
                ylabel=ylabel,
                title=title,
                tickf=18,
                titlef=18,
                labelf=18
                )

    data = []
    labels = []
    colors = []
    for y in y_data:
        og = y_data[y]['values']
        og = [i for i in og if i is not np.nan]
        data = data + [og]
        labels = labels + [y_data[y]['label']]
        colors = colors + [y_data[y]['color']]


    bp = ax1.boxplot(data, patch_artist = True, manage_ticks = True, labels=labels, showfliers=False,)#,width, color= y_data[y]['bar_color'], label = y_data[y]['label'])

    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
 
    # changing color and linewidth of
    # whiskers
    for whisker in bp['whiskers']:
        whisker.set(color ='black',
                    linewidth = 1.5,
                    linestyle =":")
 
    # changing color and linewidth of
    # caps
    for cap in bp['caps']:
        cap.set(color ='black',
                linewidth = 2)
 
    # changing color and linewidth of
    # medians
    for median in bp['medians']:
        median.set(color ='black',
                    linewidth = 3)

    if minorTicks is not None:
        ax1.yaxis.set_minor_locator(tck.AutoMinorLocator())

    plt.xticks(rotation=45)

    if tag is not None:
        plt.savefig(f"{out_path}plot_box_{title}_{tag}.png", dpi=300, transparent=True,bbox_inches="tight" )
    else:
        plt.savefig(f"{out_path}plot_box_{title}.png", dpi=300, transparent=True,bbox_inches="tight" )

    plt.close()
    print(f">>> Plotted: {title}")





def main():

    ### Run first when adding new results to study
    # get_MAS_timings()
    # get_POMS_timings()
    # get_manual_timings()
    # get_tsst_timings()
    
    # Not sure about this
    #matched()

    # Run this first as first analysis step
    #pipeline(f'C:/Users/ashra/Desktop/Distell/Results/', control_key, control_groups, treatment_key, time_series_groups)

    # Once the above has run, run this after commenting out paired T-Test data
    final_analysis()

    ### Not in use
    #gfm_plot(f'C:/Users/ashra/Desktop/Distell/Results/')
    #individual_analysis()

if __name__ == "__main__":
    main()