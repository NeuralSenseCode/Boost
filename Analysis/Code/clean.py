import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
import os
import numpy as np
import pprint
import sys
import shutil
import itertools
import matplotlib.pyplot as plt
import scipy.signal as signal
from scipy.stats import norm
from matplotlib.animation import FuncAnimation as fa
import pprint as pp
import plot
from scipy.stats import ttest_ind
from collections import OrderedDict 
import plotly.express as px
from plotly import graph_objects as go
from plotly import subplots as sbp
#dfs = {f"{f[:-4]}": pd.read_csv(f, low_memory = False) for f in os.listdir('./') if '.csv' in f}
#ext = {key : value[['Row','High Engagement']].dropna() for key,value in dfs.items()}

try:
    import msvcrt
    def get_key(z):
        print()
        print('-- Exception Raised --')
        print(f'>> Warning due to {z}')
        print('-- Press any key to continue...--')
        #msvcrt.getch()
        print()
except:
    pass

def merge_dict(add, og):
    return(og.update(add))


def get_unique(x):
    return list(OrderedDict.fromkeys(x))

def get_effect(d1,d2):
        cD, hG = get_effect_size(d1, d2)

        try:
            if cD <0:
                cD_Pol = 'Decrease'
            else:
                cD_Pol = 'Increase'

            if hG <0:
                hG_Pol = 'Decrease'
            else:
                hG_Pol = 'Increase'

            cD = np.abs(cD)
            hG = np.abs(hG)


            if cD<0.2:
                cDE = 'Negligible'
            if cD>=0.2 and cD<0.5:
                cDE= 'Small-to-Medium'
            if cD>=0.5 and cD<0.8:
                cDE= 'Medium-to-Large'
            if cD>=0.8:
                cDE= 'Large'

            if hG<0.2:
                hGE = 'Negligible'
            if hG>=0.2 and hG<0.5:
                hGE= 'Small-to-Medium'
            if hG>=0.5 and hG<0.8:
                hGE= 'Medium-to-Large'
            if hG>=0.8:
                hGE= 'Large'
            return f"{cDE} {cD_Pol}", f"{hGE} {hG_Pol}"
        except:
            return np.nan, np.nan



def get_effect_size(d1, d2):
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


def print_status(text, item,err=0):

    if err==0:
        pre = "#####" if ("Fail" in text) or ("Fail" in item)  else ">>>>>"
        print(f"{pre} {text} : {item}")
        return None
    
    else:
        pre = "#####"
        if ("Fail" in text) or ("Fail" in item):
            print(f"{pre} {text} : {item}")
            return None
    

def print_step(text, item,err=0):
    if err==0:
        pre = "###" if ("Fail" in text) or ("Fail" in item)  else ">>>"
        print(f"{pre} {text} : {item}")
        return None
    
    else:
        pre = "###"
        if ("Fail" in text) or ("Fail" in item):
            print(f"{pre} {text} : {item}")
            return None
    

def print_error(text, item):
    print(f"########   ERROR   #######")
    print(f"### {text} : {item}")

def header(text):
    print("-"*len(text))
    print(text)
    print("-"*len(text))

def sub_header(text):
    print(text)
    print("-"*len(text))

def cleanup():
    files = [f for f in os.listdir('./infiles') if not f.startswith('.')]
    scenes = []
    # alpha = input("please enter the alpha column name: \n")
    # engagement = input("please enter the engagement column name: \n")
    # workload = input("please enter the workload column name: \n")
    # active = input("please enter the aos active column name: \n")
    # gazed = input("please enter the aos gazed at column name: \n")
    # scene = input("please enter the scene 1 column name: \n")

    row = "Row" 
    alpha = "Frontal Asymmetry Alpha"
    engagement = "High Engagement"
    workload = "Workload Average"
    active = "AOIs active"
    gazed = "AOIs gazed at"
    scene = "Scene1 active on TasteOfNo_CocaCola_30s"

    scenes.append(row)
    scenes.append(alpha)
    #scenes.append(engagement)
    #scenes.append(workload)
    scenes.append(active)
    scenes.append(gazed)
    scenes.append('Scene')


    for file in files:
        try:
            df = pd.read_csv('./infiles/' + file, header=1)
            #df = df.drop_duplicates([engagement, workload, alpha], keep='last')
            df = df.drop_duplicates([alpha], keep='last')
            df['Scene'] = np.nan

            index = df.columns.get_loc(scene)
            for col in range(index, len(df.columns)):
                columns = df.columns
                if columns[col] != active:
                    df['Scene'] = df['Scene'].combine_first(df[columns[col]])
                else:
                    break

            clean = df[scenes]
            clean.replace(to_replace=-99999,value=' ')
            clean.to_csv(r'./outfiles/' + 'clean' + file , index=False, header=True)

            print(f"> Cleaned: {file}")
        except:
            print('File: ' + file + ' has failed due to missing data.')


def joined():
    files = [f for f in os.listdir('./outfiles') if not f.startswith('.')] 
    #labels = ['Frontal Asymmetry', 'High Engagement', 'Workload', 'AOIs Active', 'AOI Gaze', 'Scene']
    labels = ['Row','Frontal Asymmetry', 'AOIs Active', 'AOI Gaze', 'Scene']
    all_data = pd.DataFrame(columns = labels )
    res = 1
    
    for file in files:
        print(f"> Collected: {file}")
        #_data = pd.read_csv(f"./outfiles/{file}", header=0, usecols=[0,1,2,3,4,5], names = labels)
        _data = pd.read_csv(f"./outfiles/{file}", header=0, usecols=[0,1,2,3,4], names = labels)
        _data.insert(0, 'Respondant',res)
        all_data = pd.concat([all_data, _data])
        res = res + 1
    pprint.pprint(len(all_data))
    all_data.to_csv('results.csv')


def join_to_df(in_folder, tags =['',]):
    print('> Now Running: Join_to_df')
    files = get_files(in_folder,tags)
    all_data = pd.DataFrame()
    for file in files:
            _path = f"{in_folder}{file}"
            _data = pd.read_excel(_path, header=0, index_col = 0)
            #_data = pd.read_csv(_path, header=0)
            all_data = pd.concat([all_data, _data])
            print(f"> Completed: {file}")

    print('> Completed: Join_to_df')
    return all_data


def get_files(folder, tags=['',]):
    return [f for f in os.listdir(folder) if not f.startswith('.') and all(x in f for x in tags)] 


def bin_data(file, x, y, inc):
    df= pd.read_csv(file, header=0)
    max_val = df[x].max()
    bins = np.arange(0,max_val, inc)
    ind = np.digitize(df[x],bins)
    df = df.groupby(ind).mean().reset_index()
    df[x] = df[x]/256
    return df[[x,y]]


def combine_files(in_folder, results_folder, axis = 0):
    ##Combine all data
    files = get_files(in_folder)
    all_data = pd.DataFrame()
    
    for file in files:
        print(f"> Joined: {file}")
        _data = pd.read_csv(f"{results_folder}{file}", header=0)
        all_data = pd.concat([all_data, _data], axis = axis)

    pprint.pprint(len(all_data))
    all_data.to_csv(f"{results_folder}combined_{data}.csv")  

 
def remove(folder):  
    header("> Now Running: Cleanup Trash ")
    if os.path.exists(folder):
        try:
            shutil.rmtree(folder)
            print(f"> Removed Results: {folder}")
        except OSError as error:
            print(f"> ##### Error cleaning outfile directory: {error} ##### ")


def flatten(t):
    return [item for sublist in t for item in sublist]


def join_scene_results(results_folder):
    header("> Running: Combining Scene Data")
    
    out_path = f"{results_folder}/summary"
    files = [f for f in get_files(out_path) if 'scenes_' in f]
    labels = ['Ad', 'Scene']
    sorts = ['Scene', 'Brand', 'Starting', 'Stopping']

    all_data = pd.DataFrame(columns = labels)
    
    for file in files:
        _data = pd.read_csv(out_path + '/' + file, header=0, index_col = 0)
        all_data = pd.merge(all_data, _data, how = "outer", on = labels)
    
    for sort in sorts:
        _data = all_data.loc[all_data['Scene'].str.contains(sort)]
        _data.to_csv(f"{out_path}/combined_{sort}_results.csv")
        pprint.pprint(len(_data))
 

def batch_scenes_GSR(in_folder, out_folder, results_folder, scene_tags):
    header("> Running: Extracting GSR")
    #Define variables
    calc_col = ['Ad','Scene','Avg GSR']
    calc = pd.DataFrame(columns = calc_col)
    scalc = pd.DataFrame(columns = calc_col)

    row = "Row" 
    data = "Peak detected (binary)"
    
    keep = []
    keep.append(row)
    keep.append(data)
    

    #Get Ad names
    dirs = get_files(in_folder)
    #For each Ad..
    for dir in dirs:
            print(f"> Now Working: {dir}")
            out_path = out_folder +dir + '/GSR'
            os.makedirs(out_path, exist_ok=True)

            #Extract data from files
            files = get_files(in_folder + dir + '/GSR')
            for file in files:
                try:
                    df = pd.read_csv(in_folder + dir + '/GSR/' + file, header=1, low_memory = False)
                    scenes = [s for s in df.columns if any(x in s for x in scene_tags)]

                    clean = df[keep+scenes] 
                    clean = clean.replace(to_replace=-99999,value=np.nan)
                    clean = clean.dropna(subset=[data])
                    clean.to_csv(out_path + '/clean_' + file , index=False, header=True)
                    print(f"> Cleaned: {file}")
                except OSError as error:
                    print(f">##### Error cleaning outfile {file} : {error} ##### ")
                except:
                    print(f"> #####  Error cleaning outfile {file} : Missing")
            
            #Combine data per ad
            files = get_files(out_path)
            labels = ['Row',data]
            labels = labels + scenes
            all_data = pd.DataFrame(columns = labels)
            res = 1
            for file in files:
                try:
                    _data = pd.read_csv(out_path + '/' + file, header=0, names = labels)
                    _data.insert(0, 'Respondant',res)
                    _data.insert(0, 'Ad', dir)
                    all_data = pd.concat([all_data, _data])
                    res = res + 1
                    print(f"> Collected: {file}")

                    for scn in scenes:
                        try:
                            _scenes = [data, scn]
                            _sdata = _data[_scenes]
                            _sdata = _sdata.dropna(subset=[scn])[data]
                            _mean = _sdata.mean()

                            scalc = scalc.append({'Ad':dir, 'Scene':scn, 'Avg GSR':_mean}, ignore_index = True)
                        except :
                            print(f">##### Error scraping outfile file: Missing Data #####")

                except OSError as error:
                    print(f">    Error joining outfile file: {error}")
            pprint.pprint(len(all_data))
            all_data.to_csv(results_folder+dir+'_'+data+'.csv')
            
            #Calculate proportions

            for scn in scenes:
                try:
                    _scenes = [data, scn]
                    _data = all_data[_scenes]
                    _data = _data.dropna(subset=[scn])[data]
                    _mean = _data.mean()

                    calc = calc.append({'Ad':dir, 'Scene':scn, 'Avg GSR':_mean}, ignore_index = True)
                except OSError or ValueError or ZeroDivisionError as error:
                    print(f">##### Error joining outfile file: {error} #####")
                    
            print(f"> Got Scenes: {file}")
    
    os.makedirs(f"{results_folder}summary", exist_ok=True)
    calc.to_csv(f"{results_folder}summary/scenes_{data}.csv")
    scalc.to_csv(f"{results_folder}summary/scenes_{data}_all.csv")
    pprint.pprint(len(calc))

        
def extract_AOI(in_folder, out_folder, results_folder, head = 8):
    header("> Running: Extracting AOI")

    out_path = out_folder
    calc_col = ['Ad','TTFF','DOAF','FFD','AOI Count']
    calc = pd.DataFrame(columns = calc_col)

    category = "Stimulus Label"
    AOI = "AOI Label" 
    dur = "Duration (ms)"
    TTFF = "TTFF (AOI)"
    DOAF = "Duration of average fixation"
    FFD =  "Fixation Duration"
    FC = "Fixation count"

    keep = []
    keep.append(category)
    keep.append(AOI)
    #keep.append(dur)
    #keep.append(TTFF)
    #keep.append(DOAF)
    keep.append(FFD)
    #keep.append(FC)
    
    files = get_files(in_folder)

    for file in files:
        try:
            df = pd.read_csv(in_folder + file, header= head)
            clean = df[keep] 
            clean = clean.dropna(subset=[category])
            clean.to_csv(out_path + '/clean_' + file , index=False, header=True)
            print(f"> Cleaned: {file}")

        except OSError as error:
            print(f">##### Error cleaning outfile {file} : {error} ##### ")
    
    files = get_files(out_path)

    #for file in files:
    #    try:
    #        df = pd.read_csv(out_path + file, header=0)

    #        ads = df[category].drop_duplicates()
    #        for a in ads:
    #            _data = df.loc[df[category].str.contains(a)]
                
    #            _dur = _data[dur].sum()
    #            _TTFF = _data[TTFF].sum()/_dur*100
    #            _DOAF = _data[DOAF].sum()/_dur*100
    #            _FFD  = _data[FFD].sum()/_dur*100
    #            _AOI = len(_data[AOI])

    #            calc = calc.append({'Ad':a,
    #                                'TTFF':_TTFF,
    #                                'DOAF':_DOAF,
    #                                'FFD':_FFD,
    #                                'AOI Count':_AOI,
    #                              }, ignore_index = True)

    #            _data = _data.append({' ':' ',
    #                                'TTFF':_TTFF,
    #                                'DOAF':_DOAF,
    #                                'FFD':_FFD,
    #                                'AOI Count':_AOI,
    #                              }, ignore_index = True)

    #            _data.to_csv(f"{results_folder}{a}_summary.csv")

    #            print(f"> Calculated: {a}")
    #    except OSError as error:
    #        print(f">##### Error cleaning outfile {file} : {error} ##### ")

    print(f"> Completed: {file}")

    calc.to_csv(f"{results_folder}AOI_summary.csv")
    pprint.pprint(len(calc))


def extract_cols(in_folder, out_folder, keep, head = 8, tags = ['',]):
    out_path = f"{out_folder}"
    os.makedirs(out_path, exist_ok=True)    
    print("> Running: Extracting AOI")
    
    files = get_files(in_folder, tags)

    for file in files:
        try:
            df = pd.read_csv(in_folder + file, header= head)
            clean = df[keep] 
            clean = clean.dropna(subset=[keep[0]])
            clean.to_csv(out_path + '/clean_' + file , index=False, header=True)
            print(f"> Cleaned: {file}")
            pprint.pprint(clean.shape)

        except OSError as error:
            print(f">##### Error cleaning outfile {file} : {error} ##### ")
    

def batch_AOI_engagement(in_folder, out_folder, results_folder):
    header("> Running: Extracting AOI Engagement")
    #Define variables
    calc_col = ['Ad','AOI','Eng Mean','Eng Disengaged Prop', 'Eng Low Prop', 'Eng High Prop','Eng Count']
    calc = pd.DataFrame(columns = calc_col)

    data = "High Engagement"
    AOI = "AOIs gazed at"
    
    keep = []
    keep.append(AOI)
    keep.append(data)
    
    scenes = []

    #Get Ad names
    dirs = get_files(in_folder)
    #For each Ad..
    for dir in dirs:
            print(f"> Now Working: {dir}")
            out_path = out_folder +dir + '/Engagement'
            os.makedirs(out_path, exist_ok=True)

            #Extract data from files
            files = get_files(in_folder + dir + '/Eng and WL')
            for file in files:
                try:
                    df = pd.read_csv(in_folder + dir + '/Eng and WL/' + file, header=1)
                    clean = df[keep] 
                    clean = clean.replace(to_replace=-99999,value=np.nan)
                    clean = clean.dropna(subset=[data, AOI])
                    clean.to_csv(out_path + '/clean_' + file , index=False, header=True)
                    print(f"> Cleaned: {file}")
                except OSError as error:
                    print(f">##### Error cleaning outfile {file} : {error} ##### ")
            
            #Combine data per ad
            files = get_files(out_path)
            labels = [AOI,data]
            all_data = pd.DataFrame(columns = labels)
            res = 1
            for file in files:
                try:
                    _data = pd.read_csv(out_path + '/' + file, header=0, names = labels)
                    _data.insert(0, 'Respondant',res)
                    _data.insert(0, 'Ad', dir)
                    all_data = pd.concat([all_data, _data])
                    res = res + 1
                    print(f"> Collected: {file}")
                except OSError as error:
                    print(f">    Error joining outfile file: {error}")
            pprint.pprint(len(all_data))
            all_data.to_csv(results_folder+dir+'_engagement.csv')
            
            AOIs = all_data[AOI].drop_duplicates()
            AOIs = AOIs.dropna()

            #Calculate proportions
            for aoi in AOIs:
                try:
                    _data = all_data.loc[all_data[AOI].str.contains(aoi)]
                    _data = _data.dropna()

                    _data = _data[data]
                    _mean = _data.mean()

                    _disengaged = sum(1 for item in _data if item < 0.4)
                    _low = sum(1 for item in _data if item > 0.4 and item < 0.7)
                    _high = sum(1 for item in _data if item > 0.7)

                    _count = len(_data)
                    if _count>0:
                        _disengaged = _disengaged/_count*100
                        _low = _low/_count*100
                        _high = _high/_count*100
                    else: 
                        _count = 0
                        _disengaged = 0
                        _low = 0
                        _high = 0

                    calc = calc.append({'Ad':dir, 'AOI':aoi, 'Eng Mean':_mean, 'Eng Disengaged Prop': _disengaged, 'Eng Low Prop':_low, 'Eng High Prop' :_high , 'Eng Count':_count}, ignore_index = True)
                except OSError or ValueError or ZeroDivisionError as error:
                    print(f">##### Error joining outfile file: {error} #####")
                    
            print(f"> Got Scenes: {file}")
    
    os.makedirs(f"{results_folder}summary", exist_ok=True)
    calc.to_csv(f"{results_folder}summary/AOI_engagement.csv")
    pprint.pprint(len(calc))


def batch_AOI_alpha(in_folder, out_folder, results_folder):
    header("> Running: Extracting AOI Engagement")
    #Define variables
    calc_col = ['Ad','AOI','Alpha Mean','Alpha Prop','Alpha Count']
    calc = pd.DataFrame(columns = calc_col)

    data = "Frontal Asymmetry Alpha"
    AOI = "AOIs gazed at"
    
    keep = []
    keep.append(AOI)
    keep.append(data)
    
    #Get Ad names
    dirs = get_files(in_folder)
    #For each Ad..
    for dir in dirs:
            print(f"> Now Working: {dir}")
            out_path = out_folder +dir + '/Alpha'
            os.makedirs(out_path, exist_ok=True)

            #Extract data from files
            files = get_files(in_folder + dir + '/Alpha')
            for file in files:
                try:
                    df = pd.read_csv(in_folder + dir + '/Alpha/' + file, header=1)
                    clean = df[keep] 
                    clean = clean.replace(to_replace=-99999,value=np.nan)
                    clean = clean.dropna(subset=[data, AOI])
                    clean.to_csv(out_path + '/clean_' + file , index=False, header=True)
                    print(f"> Cleaned: {file}")
                except OSError as error:
                    print(f">##### Error cleaning outfile {file} : {error} ##### ")
            
            #Combine data per ad
            files = get_files(out_path)
            labels = [AOI,data]
            all_data = pd.DataFrame(columns = labels)
            res = 1
            for file in files:
                try:
                    _data = pd.read_csv(out_path + '/' + file, header=0, names = labels)
                    _data.insert(0, 'Respondant',res)
                    _data.insert(0, 'Ad', dir)
                    all_data = pd.concat([all_data, _data])
                    res = res + 1
                    print(f"> Collected: {file}")
                except OSError as error:
                    print(f">    Error joining outfile file: {error}")
            pprint.pprint(len(all_data))
            all_data.to_csv(results_folder+dir+'_engagement.csv')
            
            AOIs = all_data[AOI].drop_duplicates()
            AOIs = AOIs.dropna()

            #Calculate proportions
            for aoi in AOIs:
                try:
                    _data = all_data.loc[all_data[AOI].str.contains(aoi)]
                    _data = _data.dropna()

                    _data = _data[data]

                    _mean = _data.mean()
                    _pos = sum(1 for item in _data if item>0)
                    _count = len(_data)
                    if _count>0:
                        _prop = _pos/_count*100
                    else: 
                        _count = 0
                        _prop = 0
                    calc = calc.append({'Ad':dir, 'AOI':aoi, 'Alpha Mean':_mean, 'Alpha Prop':_prop, 'Alpha Count':_count}, ignore_index = True)
                except OSError or ValueError or ZeroDivisionError as error:
                    print(f">##### Error joining outfile file: {error} #####")
                    
            print(f"> Got Scenes: {file}")
    
    os.makedirs(f"{results_folder}summary", exist_ok=True)
    calc.to_csv(f"{results_folder}summary/AOI_alpha.csv")
    pprint.pprint(len(calc))


def batch_AOI_workload(in_folder, out_folder, results_folder):
    header("> Running: Extracting AOI Workload")
    #Define variables
    calc_col = ['Ad','AOI','WL Mean','WL Low Prop', 'WL Optimal Prop', 'WL Overworked Prop','WL Count']
    calc = pd.DataFrame(columns = calc_col)

    data = "Workload Average"
    AOI = "AOIs gazed at"
    
    keep = []
    keep.append(AOI)
    keep.append(data)

    #Get Ad names
    dirs = get_files(in_folder)
    #For each Ad..
    for dir in dirs:
            print(f"> Now Working: {dir}")
            out_path = out_folder +dir + '/Workload'
            os.makedirs(out_path, exist_ok=True)

            #Extract data from files
            files = get_files(in_folder + dir + '/Eng and WL')
            for file in files:
                try:
                    df = pd.read_csv(in_folder + dir + '/Eng and WL/' + file, header=1)
                    clean = df[keep] 
                    clean = clean.replace(to_replace=-99999,value=np.nan)
                    clean = clean.dropna(subset=[data, AOI])
                    clean.to_csv(out_path + '/clean_' + file , index=False, header=True)
                    print(f"> Cleaned: {file}")
                except OSError as error:
                    print(f">##### Error cleaning outfile {file} : {error} ##### ")
            
            #Combine data per ad
            files = get_files(out_path)
            labels = [AOI,data]
            all_data = pd.DataFrame(columns = labels)
            res = 1
            for file in files:
                try:
                    _data = pd.read_csv(out_path + '/' + file, header=0, names = labels)
                    _data.insert(0, 'Respondant',res)
                    _data.insert(0, 'Ad', dir)
                    all_data = pd.concat([all_data, _data])
                    res = res + 1
                    print(f"> Collected: {file}")
                except OSError as error:
                    print(f">    Error joining outfile file: {error}")
            pprint.pprint(len(all_data))
            all_data.to_csv(results_folder+dir+'_workload.csv')
            
            AOIs = all_data[AOI].drop_duplicates()
            AOIs = AOIs.dropna()

            #Calculate proportions
            for aoi in AOIs:
                try:
                    val = all_data[AOI].str.contains(aoi)
                    _data = all_data.loc[all_data[AOI].str.contains(aoi)]
                    _data = _data.dropna()

                    _data = _data[data]
                    _mean = _data.mean()

                    _low = sum(1 for item in _data if item < 0.4)
                    _optimal = sum(1 for item in _data if item > 0.4 and item < 0.6)
                    _overworked = sum(1 for item in _data if item > 0.6)

                    _count = len(_data)
                    if _count>0:
                        _low = _low/_count*100
                        _optimal = _optimal/_count*100
                        _overworked = _overworked/_count*100
                    else: 
                        _count = 0
                        _low = 0
                        _optimal = 0
                        _overworked = 0

                    calc = calc.append({'Ad':dir, 'AOI':aoi, 'WL Mean':_mean, 'WL Low Prop': _low, 'WL Optimal Prop':_optimal, 'WL Overworked Prop':_overworked, 'WL Count':_count}, ignore_index = True)
                except OSError or ValueError or ZeroDivisionError as error:
                    print(f">##### Error joining outfile file: {error} #####")
                    
            print(f"> Got Scenes: {file}")
    
    os.makedirs(f"{results_folder}summary", exist_ok=True)
    calc.to_csv(f"{results_folder}summary/AOI_workload.csv")
    pprint.pprint(len(calc))


def merge_files(in_folder, out_folder, labels, title='', tags = ['',]): #make sure files are in csv format, will write to xlsx
    header(f"> Running: Merging Data by {tags}")
    
    out_path = f"{out_folder}"
    os.makedirs(out_path, exist_ok=True)

    files = get_files(in_folder, tags)

    all_data = pd.DataFrame(columns = labels)
    
    for file in files:
        _data = pd.read_csv(in_folder + '/' + file, header=0, index_col = 0)
        all_data = pd.merge(all_data, _data, how = "outer", on = labels)

    #all_data = all_data.replace(to_replace='#DIV/0!',value=np.nan)
    #all_data = all_data.dropna(subset = ['Eng Mean','Ratio of FFD/TTFF'])
    #all_data = all_data.groupby('AOI').mean().reset_index()
    #all_data = all_data.dropna(subset=['Parent Label'])
    #all_data = all_data.drop_duplicates('First fixation duration')
    all_data.to_excel(out_path + title + '.xlsx')
    pprint.pprint(len(all_data))


def extract_markers(in_folder, out_folder, col, markers, tags =['',]):
    name = '_'.join(markers+[tags[0]])
    print(f"> Running: Extracting Data for {name}")

    out_path = f"{out_folder}"
    os.makedirs(out_path, exist_ok=True)
    files = get_files(in_folder,tags)
    all_data = pd.DataFrame()

    for (file,mark) in itertools.product(files,markers):
            _data = pd.read_csv(in_folder + '/' + file, header=0)
            _data = _data.loc[_data[col].str.contains(mark)]
            all_data = pd.concat([all_data, _data])
            print(f"> Completed: {file}")

    all_data = all_data.drop_duplicates()
    all_data.to_csv(out_path + '/markers_' + name + '.csv')
    pprint.pprint(all_data.shape)


def sort_col(in_folder, out_folder, col1, tags=['',]):
    out_path = f"{out_folder}"
    os.makedirs(out_path, exist_ok=True)    
    name = '_'.join([col1]+[tags[0]])
    print(f"> Running: Sorting Data by {name}")
    all_data = join_to_df(in_folder,tags)   
    all_data = all_data.sort_values(col1)
    #all_data = all_data.sort_values(col1,ascending=False)
    all_data.to_csv(out_path + '/sort_' + name + '.csv')
    pprint.pprint(all_data.shape)


def drop_val(in_folder, out_folder, col1, tags =['',]):
    out_path = f"{out_folder}"
    os.makedirs(out_path, exist_ok=True)    
    name = '_'.join([col1]+[tags[0]])
    print(f"> Running: Dropping Data for {name}")
    all_data = join_to_df(in_folder,tags)
    all_data = all_data.drop_duplicates(subset= [col1], keep='first' )  
    all_data.to_csv(out_path + '/drop_' + name + '.csv')
    pprint.pprint(all_data.shape)


def drop_na(in_folder, out_folder, tags =['',], drop = None):
    out_path = f"{out_folder}"
    os.makedirs(out_path, exist_ok=True)    
    name = '_'.join([tags[0]])
    print(f"> Running: Dropping Data for {name}")
    all_data = join_to_df(in_folder,tags)
    all_data = all_data.dropna(subset=drop)
    all_data.to_csv(out_path + '/dropna_' + name + '.csv')
    pprint.pprint(all_data.shape)


def mean_val(in_folder, out_folder, col1, markers = [''], tags =['',]):
    out_path = f"{out_folder}"
    os.makedirs(out_path, exist_ok=True)    
    
    name = '_'.join([col1]+[tags[0]])
    print(f"> Running: Averaging Data for {name}")

    all_data = join_to_df(in_folder,tags)
    #all_data = str_extract_rows(all_data, xcol, '_')
    all_data_mean = all_data.groupby(col1).mean()
    mean_col = all_data_mean.columns
    all_data_mean = pd.merge(all_data_mean,all_data, how='left', on=col1)
    all_data_mean = all_data_mean.drop_duplicates(subset=[col1])
    
    all_data_mean.to_csv(out_path + '/mean_' + name + '.csv')
    pprint.pprint(all_data.shape)


def keep_col(in_folder, out_folder, cols, tags =['',]):
    out_path = f"{out_folder}"
    os.makedirs(out_path, exist_ok=True)    
    
    name = '_'.join(cols+[tags[0]])
    print(f"> Running: Keeping Data for {name}")

    all_data = join_to_df(in_folder,tags)
    all_data = all_data[cols]
    all_data = all_data.dropna()
    all_data.to_csv(out_path + '/keep_' + name + '.csv')
    pprint.pprint(all_data.shape)

def percentiles(in_folder, out_folder, cols, calc_col = ['',], title='', tags =['',]):
    header(f"> Running: Calculating Percentiles")
    
    #Define variables
    calc = pd.DataFrame()

    out_path = f"{out_folder}"
    os.makedirs(out_path, exist_ok=True)    
    
    df = join_to_df(in_folder,tags)
    col_means = {x:df[x].mean() for x in cols}
    col_stds = {x:df[x].std() for x in cols}

    for index, row in df.iterrows():
        _per = {x:100*norm.cdf((row[x]-col_means[x])/col_stds[x]) for x in cols}
        _calc = {'Ad':row['Ad'], 'Composite': np.array(list(_per.values())).mean()}
        if 'Scene' in calc_col:
            _calc['Scene']= row['Scene']
        _calc.update(_per)       
        calc = pd.concat([calc, pd.DataFrame([_calc])], ignore_index=True)

    calc.to_excel(f"{out_path}{title}.xlsx")
    pp.pprint(len(calc))
    pp.pprint(calc)
    print("> Completed: Percentiles")

def percentiles_df(in_df, ind, cols):
    header(f"> Running: Calculating Percentiles")
    
    #Define variables
    calc = pd.DataFrame()   
    
    col_means = {x:in_df[x].mean() for x in cols}
    col_stds = {x:in_df[x].std() for x in cols}

    for index, row in in_df.iterrows():
        _per = {f"Percentile_{x}":100*norm.cdf((row[x]-col_means[x])/col_stds[x]) for x in cols}
        _calc = {f'{ind}':row[ind],}
        _calc.update(_per)       
        calc = pd.concat([calc, pd.DataFrame([_calc])], ignore_index=True)

    print("> Completed: Percentiles")
    return calc


def plot_bar(in_folder, out_folder, xcol, ycol, bar_color='#dadfe2', xlabel = '', ylabel = '', title = '', tags =['',], highlight = {'':'',} ):
    out_path = f"{out_folder}"
    os.makedirs(out_path, exist_ok=True)    
    
    name = '_'.join(tags)
    print(f"> Running: Plotting Data for {name}")

    all_data = join_to_df(in_folder,tags)
    all_data = all_data.replace(to_replace=np.nan,value=0)
    all_data = str_extract_rows(all_data, xcol, '_')

    # set the style of the axes and the text color
    plt.rcParams['axes.edgecolor']='#333F4B'
    plt.rcParams['axes.linewidth']=0.8
    plt.rcParams['xtick.color']='#333F4B'
    plt.rcParams['ytick.color']='#333F4B'
    plt.rcParams['text.color']='#333F4B'

    fig = plt.figure()
    ax = fig.add_subplot(111)

    # change the style of the axis spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.spines['left'].set_position(('outward', 8))
    ax.spines['bottom'].set_position(('outward', 5))
    #ax = fig.add_axes([0,0,1,1])
    axes = plt.gca()
    axes.yaxis.grid()

    x_data = all_data[xcol].values
    y_data = all_data[ycol].values
    bar = ax.bar(x_data,y_data, color= bar_color)

    for i in highlight.items():
        try:
            key = np.where(x_data==i[0])[0][0]
            bar[key].set_color(i[1])
        except:
            break

    plt.xticks(rotation=45, ha = 'right')
    plt.xticks(fontsize=6)
    plt.yticks(fontsize=6)
    plt.xlabel(xlabel,fontsize=10)
    plt.ylabel(ylabel,fontsize=10)
    plt.title(title, fontsize=12)
    #plt.ylim(0,900)



    plt.tight_layout()
    plt.savefig(out_path + '/bar_' + name + '.png', dpi=300)

    print(f"> Plotted: {name}")


def group_by_range(df, col1, min, max, inc):
    df.sort_values(col1)
    bins = np.arange(min,max,inc)
    ind = np.digitize(df[col1],bins)
    return df.groupby(ind).mean().reset_index()


def plot_scatter(in_folder, out_folder, xcol, ycol, zcol, bar_color='#dadfe2', xlabel = '', ylabel = '', zlabel='', title = '', tags =['',], highlight = {'':'',} ):
    out_path = f"{out_folder}"
    os.makedirs(out_path, exist_ok=True)    
    
    name = '_'.join(tags)
    print(f"> Running: Plotting Data for {name}")

    all_data= pd.read_csv(f"{in_folder}VSP_NoMusic.csv", header=0)
    #all_data = all_data.replace(to_replace=np.nan,value=0)
    #all_data = str_extract_rows(all_data, xcol, '_')

    # set the style of the axes and the text color
    plt.rcParams['axes.edgecolor']='#333F4B'
    plt.rcParams['axes.linewidth']=0.8
    plt.rcParams['xtick.color']='#333F4B'
    plt.rcParams['ytick.color']='#333F4B'
    plt.rcParams['text.color']='#333F4B'

    fig = plt.figure()
    ax = fig.add_subplot(111,projection = '3d')

    # change the style of the axis spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.spines['left'].set_position(('outward', 8))
    ax.spines['bottom'].set_position(('outward', 5))
    #ax = fig.add_axes([0,0,1,1])
    #axes = plt.gca()
    #axes.yaxis.grid()

    x_data = all_data[xcol].values
    y_data = all_data[ycol].values
    z_data = all_data[zcol].values
    exp = all_data['Experience'].values

    #ax.scatter(x_data,y_data, z_data, color= '#e73e77')

    #all_data= pd.read_csv(f"{in_folder}VSP_Music.csv", header=0)

    #x_data = all_data[xcol].values
    #y_data = all_data[ycol].values
    #z_data = all_data[zcol].values

    #ax.scatter(x_data,y_data, z_data, color= '#a25133')

    #plt.xticks(rotation=45, ha = 'right')
    plt.xticks(fontsize=6)
    plt.yticks(fontsize=6)
    #plt.zticks(fontsize=6)
    plt.xlabel(xlabel,fontsize=10)
    plt.ylabel(ylabel,fontsize=10)
    #plt.zlabel(zlabel,fontsize=10)
    plt.title(title, fontsize=12)

    x,y,z = [],[],[]

    def animate(i):
        x.append(x_data[i])
        y.append(y_data[i])
        z.append(z_data[i])
        ax.scatter(x,y,z, color= 'blue')

    ani = fa(fig=fig, func = animate, interval = 1)
    #ani.show

    plt.show()
    #plt.tight_layout()
    #plt.savefig(out_path + '/bar_' + name + '.png', dpi=300)

    print(f"> Plotted: {name}")


def line(x,title,results_folder,ys = {},legend = False):
    out_path = f"{results_folder}"
    os.makedirs(results_folder, exist_ok=True) 
    
    plt.rcParams['axes.edgecolor']='#333F4B'
    plt.rcParams['axes.linewidth']=0.8
    plt.rcParams['xtick.color']='#333F4B'
    plt.rcParams['ytick.color']='#333F4B'
    plt.rcParams['text.color']='#333F4B'

    fig = plt.figure()

    ax = fig.add_subplot(111)

    # change the style of the axis spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.spines['left'].set_position(('outward', 8))
    ax.spines['bottom'].set_position(('outward', 5))

    axes = plt.gca()
    axes.get_xaxis().set_visible(False)

    plt.ylabel('Proportion (%)',fontsize=10)
    #plt.title(title, fontsize=12)
    plt.xlim(0,x.max())
    plt.ylim(0,100)

    for label,y in ys.items():
        plt.plot(x,y,label = label)
    #if legend:
    #    plt.legend()

    plt.savefig(results_folder + '/line_' + title + '.png', dpi=300)
    #plt.show()
    plt.close()

    print(f"> Plotted Line: {title}")


def split_ads(in_folder, out_folder, header_row):
    header("> Running: Splitting Ads")
    out_path = f"{out_folder}"
    os.makedirs(out_path, exist_ok=True)    
    respondants = get_files(in_folder)
    for r in respondants:
        print(f"> Splitting: {r}")
        head = pd.read_csv(f"{in_folder}{r}", header=None, low_memory = False)[:26]
        df = pd.read_csv(f"{in_folder}{r}", header=header_row, low_memory = False)
        ads = df['SourceStimuliName'].drop_duplicates()
       

        [os.makedirs(f"{out_path}{ad}/Alpha", exist_ok=True) for ad in ads]
        [os.makedirs(f"{out_path}{ad}/Eng and WL", exist_ok=True) for ad in ads]
        #[os.makedirs(f"{out_path}{ad}/Metadata", exist_ok=True) for ad in ads]
        for ad in ads:
            _data = df[df['SourceStimuliName']== ad]
            _data['Row'] = range(len(_data))
            col = [c for c in _data.columns if (' on ' not in c) or (' on ' in c and ad in c)]
            final = _data[col]                         
            final.to_csv(f"{out_path}{ad}/Alpha/{r}",index = False)
            final.to_csv(f"{out_path}{ad}/Eng and WL/{r}",index = False)
            #head.to_csv(f"{out_path}{ad}/Metadata/{r}", index = False)
            print(f">>  Split: {ad}")
    print(f">Completed: Splitting Ads")


def filter(data, freqn, type):
    y = data
    b, a = signal.butter(8, freqn,type)
    yfs = signal.filtfilt(b, a, y, padlen=10)
    return yfs

if __name__ == "__main__":
    main()


