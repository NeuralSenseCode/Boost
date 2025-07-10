from nslib import *
import ast
import pickle
import sys, os
import datetime
import difflib 
import warnings
import ast
import re
from scipy.signal import butter, filtfilt, welch, find_peaks
import json
import matplotlib.pyplot as plt
import scipy.interpolate
from scipy.signal import detrend



warnings.simplefilter(action='ignore', category=FutureWarning)

task_list_tasks = ['blankScreen_00', 'BP_00', 'Exposure', 'passiveViewing_01', 'blankScreen_01','BP_01', 'blankScreen_02','BP_02', 'blankScreen_03','BP_03', 'blankScreen_04', 'BP_04','BP_05','BP_06', 'blankScreen_05','Saliva_00','simpleRT_00','simpleRT_01','Exercise']

def butter_lowpass_filter(data, cutoff, fs, order=4):
    """
    # Arguments:
    data: np.array containing datapoint
    cutoff: cutoff frequency in Hz
    fs: sampling frequency in Hz
    order: butterworth filter order

    # Note:
    Higher order filters have a steeper cutoff but may reduce filter stability

    """
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = filtfilt(b, a, data,method="pad")
    return y


def butter_highpass_filter(data, cutoff, fs, order=4):
    """
    # Arguments:
    data: np.array containing datapoint
    cutoff: cutoff frequency in Hz
    fs: sampling frequency in Hz
    order: butterworth filter order

    # 
    """
    b, a = butter_highpass(cutoff, fs, order=order)
    y = filtfilt(b, a, data,method="pad")
    return y


def butter_bandpass_filter(data, cutoffs, fs, order=4):
    """
    # Arguments:
    data: np.array containing datapoint
    cutoffs: tuple containing cutoff frequencies in Hz
    fs: sampling frequency in Hz
    order: butterworth filter order

    # 
    """
    b, a = butter_bandpass(cutoffs, fs, order=order)
    y = filtfilt(b, a, data,method="pad")
    return y


def butter_lowpass(cutoff, fs, order=2):
    return butter(order, cutoff, fs=fs, btype='low', analog=False)


def butter_highpass(cutoff, fs, order=2):
    return butter(order, cutoff, fs=fs, btype='high', analog=False)


def butter_bandpass(cutoffs, fs, order=2):
    return butter(order, cutoffs, fs=fs, btype='band', analog=False)


def compute_welch_psd(data: list, fs: int):
    """
    # Arguments:
    data: np.array of datapoints
    fs: sampling frequency in Hz
    window_period: window period in milliseconds

    # Notes:
    A longer window period leads to higher frequncy resolution, but lower stability and confidence in result
    """
    freqs, psd = welch(data, fs, nperseg=np.round(len(data)*0.8,0))

    total_power = np.array(psd).sum()
    return total_power

def compute_welch_psd_range(data: list, fs: int, range: tuple):
    """
    # Arguments:
    data: np.array of datapoints
    fs: sampling frequency in Hz
    window_period: window period in milliseconds

    # Notes:
    A longer window period leads to higher frequncy resolution, but lower stability and confidence in result
    """
    freqs, psd = welch(data, fs, nperseg=np.round(len(data)*0.8,0), window='hann')

    # Find indices where frequency is within the specified range
    valid_indices = (freqs >= range[0]) & (freqs <= range[1])

    # Sum the PSD values within the range
    psd_sum = np.sum(psd[valid_indices])

    return psd_sum



class Participant:
    def __init__(self,p):
        self.ID = p
        self.drink = None

    def update(self):
        pass


def get_participants(in_folder, out_folder, participant, ver):
    participants = participant if participant[0] else get_files(in_folder)
    participants = [p for p in participants if 'cort' not in p]
    cleaned = get_clean_log(out_folder)
    plist = cleaned['Participant'].tolist()
    flist = []
    for p in participants:
        if p in plist:
            if cleaned.loc[cleaned['Participant']==p]['Version'].tolist()[0] < ver:
                flist.append(p)
        else:
            flist.append(p)        
    #participants = [p for p in participants if (((p in cleaned['Participant'].tolist()) and (cleaned.loc[cleaned['Participant']==p]['Version'].tolist()[0] < ver)) or (p not in cleaned['Participant'].tolist()))]
    return flist


def update_clean_log(ID, ver, out_folder):
    _clean = {'Participant':ID,
                 'Version':ver
        }
    cleaned_log = pd.DataFrame(_clean, index=[0])

    if os.path.exists(f"{out_folder}clean_log.csv"):
        old_log = pd.read_csv(f"{out_folder}clean_log.csv")
        #If participant already has data in clean log
        if ID in old_log['Participant'].tolist():
            for key, value in _clean.items():
                old_log.loc[old_log['Participant']==ID, key] = value
            new_log = old_log
        else:
            new_log = pd.concat([old_log, pd.DataFrame([_clean])], ignore_index=True)
        new_log.to_csv(f"{out_folder}clean_log.csv",index=False)
    else:
        cleaned_log.to_csv(f"{out_folder}clean_log.csv",index=False)


def get_clean_log(out_folder):
    if os.path.exists(f"{out_folder}clean_log.csv"):
            cleaned = pd.read_csv(f"{out_folder}clean_log.csv")
    else:
        cleaned = pd.DataFrame(columns = ['Participant','Version'])
    return cleaned


def get_pickle(in_folder, out_folder,p):
    '''
    Checks in the out_folder if the pickle exists
    Returns it if yes, or creates a new one
    '''
    print_step('Finding Pickle',p)

    path = f"{out_folder}{p}/"
    try:
        p_handle = open(f"{path}{p}_Object.pickle" , "rb")
        P = pickle.load(p_handle)
        print_status('Loaded',p)       
    except:
        os.makedirs(path, exist_ok=True)
        p_handle = open(f"{path}{p}_Object.pickle" , "rb")
        P=Participant(p)
        P.drink = P.ID[-1]
        with open(f"{path}{P.ID}_Object.pickle", 'wb') as f:
                # Pickle the 'data' dictionary using the highest protocol available.
                pickle.dump(P, f, pickle.HIGHEST_PROTOCOL)
        print_status('Created',p)
    return P,p_handle


def save_pickle(out_folder,P):
    '''
    Saves the new pickle to the output folder
    '''
    print_step('Saving Pickle',P.ID)
    path = f"{out_folder}{P.ID}/"
    with open(f"{path}{P.ID}_Object.pickle", 'wb') as f:
                # Pickle the 'data' dictionary using the highest protocol available.
                pickle.dump(P, f, pickle.HIGHEST_PROTOCOL)


def backup_files(out_folder):
    '''
    Saves the new pickle to the output folder
    '''
    path = f"{out_folder}/"

    backups = ['EVM_Results','clean_log','psychoPy_log']
    for b in backups:
        print_step('Backing Up',b)
        df = pd.read_csv(f'{path}{b}.csv')
        if len(df.columns):
            df.to_csv(f'{path}{b}_Backup.csv', index=False)
        else:
            print_status('File Corrupt',b)


def update_organise(self):
    self.organised = 0

def organise_data(in_folder, out_folder,P):
    '''
    Organise the data folder so that all files are named correctly and data folder is structured as required

    '''

    in_path = f"{in_folder}{P.ID}/"
    out_path = f"{out_folder}{P.ID}/"
    os.makedirs(out_path, exist_ok=True)

    #collapse all folder
    collapse_folder(in_path)  




def update_clean(self):
     #Flags
    self.isSignals= 0
    self.hasChannels = 0
    self.isData = 0
    self.isDataComplete = 0
    self.isSignalsComplete = 0
    self.isSignalsOverflow = 0
    #Process Flags
    self.isCleaned = 0

    #Data
    self.metadata = {}
    self.date = None
    self.session = None
    self.device = None
    self.freq = dict()
    self.signalDuration = 0
    self.dataDuration = 0
    self.signals = pd.DataFrame()
    self.status = 0
    self.sample = pd.DataFrame()
    
    #Tasks
    self.EVM = pd.DataFrame()
    self.tasksInfo = pd.DataFrame()
    self.BP = pd.DataFrame()
    self.psychoPyResults = pd.DataFrame()


def clean_data(in_folder, out_folder, pickle_folder, P):
    '''
    This function cleans and combines all the OpenSignals data from one participant into a single file

    Parameters
    __________
    in_folder: str
        Folder containing directories for each participant
        Must contain ONLY particpant 
    out_folder: str
        Folder in which a directory will be created per participant
        
    TODO:
    - Add in check for duplicate participant IDs
    '''
    print_step('Cleaning OpenSignals',P.ID)
    
    #Define variables
    metadata_line = 1
    data_row = 3
    clean_log = pd.DataFrame()
    metadata_log = pd.DataFrame()

    errors = []
    in_path = f"{in_folder}{P.ID}/"
    out_path = f"{out_folder}{P.ID}/"
    pickle_path = f"{out_folder}{P.ID}/"
    os.makedirs(out_path, exist_ok=True)  
    os.makedirs(pickle_path, exist_ok=True) 
        
    ###### OpenSignals Data
        
    ##Extract data from files
    files = get_files(in_path,tags=['.txt','SCH']) #Get list of all respondent files
    files = [f for f in files if '_convert' not in f] #Remove 'convert' files
       
    #For each file
    file_num = 0
    for f in files:                   
        try:
            path = in_path + f                 
            #Get Metadata
            with open(path, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f):
                    if i == metadata_line:
                        metadata_str = line[2:].strip()
                        metadata = json.loads(metadata_str)
                        break
            #Set P info 
            P.device = list(metadata.keys())[0]
            P.metadata = metadata[P.device]

            # Rename channel
            try:
                index_to_replace = P.metadata['sensor'].index('CUSTOM/0.5/1.0/V')  # find the index
                P.metadata['sensor'][index_to_replace] = "EDA"
            except:
                pass

            # Continue
            P.date = P.metadata['date']
            P.freq[file_num] = P.metadata['sampling rate']
            start_time = get_sec(P.metadata['time'])

            #Contingency for less than 9 channels of data
            rangeMax = len(P.metadata['channels'])
            cols = [f"{P.metadata['sensor'][i]}_{P.metadata['label'][i]}" for i in range(rangeMax)]
            usecolsData = list(range(2,len(cols)+2))

            #Read data
            df = pd.read_csv(path, header=None,skiprows=3, names=cols, low_memory = False, delimiter = r'\s+', usecols=usecolsData)
            df['Freq']=P.freq[file_num]
            df['Channels']=len(cols)
            df['File']=file_num
            #create absolute time column
            df['t'] = df.index*(1/df['Freq'])+start_time
                
            #Set flags
            P.signals = pd.concat([P.signals,df])
            P.isSignals = 1
            P.signalDuration+=df['t'].max()-df['t'].min()
                
            #Update Metadata
            dur = str(datetime.timedelta(seconds=df['t'].max()-df['t'].min()))                
            log_data = P.metadata #Add data to log
            log_data['Duration']=df['t'].max()-df['t'].min()
            metadata_log = pd.concat([metadata_log,pd.DataFrame([log_data])], ignore_index=True)
            file_num = file_num+1
            print_status('Extracted OpenSignals',path)
        except Exception as z:
            log_data = {'Duration':np.nan,'ID':np.nan,'channels':np.nan,'column':np.nan,'comments':np.nan,'convertedValues':np.nan,'date':np.nan,'device':np.nan,'device connection':np.nan,'device name':np.nan,'digital IO':np.nan,'firmware version':np.nan,'keywords':np.nan,'label':np.nan,'mode':np.nan,'position':np.nan,'resolution':np.nan,'sampling rate':np.nan,'sensor':np.nan,'sleeve color':np.nan,'special':np.nan,'sync interval':np.nan,'time':np.nan,}
            metadata_log = pd.concat([metadata_log,pd.DataFrame([log_data])], ignore_index=True)
            print_status('Failed OpenSignals',f"{path} - {z!r}")
            errors = errors + [f"Failed OpenSignals: {path} - {z!r}",]
                
    metadata_log.to_csv(f"{pickle_path}{P.ID}_metadata_log.csv",index=False)
        
    #Total Participant Data
    if P.isSignals:
        P.signals = P.signals.drop_duplicates(subset=['t'])
        #P.signals.to_csv(f"{out_path}{P.ID}_signals.csv",index=False)
        P.isSignalsOverflow = 1 if P.signalDuration > 9000 else 0
        P.sample = P.signals[P.signals['t'] % 60 == 0]
                
    ###### PsychoPy Data
    print_step('Cleaning PsychoPy',P.ID)
    
    #Extract data from files
    files = get_files(in_path,tags=['.csv','EVM']) #Get list of all respondent files
    files = [f for f in files if 'Question' not in pd.read_csv(f'{in_path}{f}').columns]
            
    versions = []
    psychoPyFiles = pd.DataFrame()
    #to account for all the diffrent Psychopy filenames
    if len(files):
        for f in files:
            _data = f.split('_')
            task_i = 2 if len(_data[1])<3 else 1   
            if len(_data[task_i+1])<3:        
                task = '_'.join([_data[task_i],_data[task_i+1]])
                date_i = task_i+2       
            else:
                task = _data[task_i]
                date_i = task_i+1
            version_i = date_i+1
            version = _data[version_i].split('.csv')[0]    
            if len(_data)== version_i+1:
                versionNumber = '00'
            else:
                versionNumber = _data[version_i+1].split('.csv')[0]
                if len(versionNumber)==1:
                    versionNumber = '0'+versionNumber
        
            _calc = {'Participant':_data[0][:4],
                        'File':f,
                        'Task':task ,
                        'Date': _data[date_i],
                        'Version': version ,
                        'Version Number':int(versionNumber) ,}
    
            psychoPyFiles = pd.concat([psychoPyFiles,pd.DataFrame([_calc])], ignore_index=True)
                    
        versions = psychoPyFiles['Version'].drop_duplicates().tolist()
        #take data only from the latest version for each version
        latest = []
        for v in versions:
            _data = psychoPyFiles.loc[psychoPyFiles['Version'].str.contains(v)]
            _data = _data.loc[_data['Version Number'].idxmax()]['File']
            latest = latest+[_data]
                
        files = list(dict.fromkeys(latest))
        #For each file
        for f in files:
            try:
                path = in_path + f                                                 
                df = pd.read_csv(path, low_memory = False, decimal=',')


                ##### Correct Saliva naming
                renames = {'00':'00'}

                try:
                    for bp,sv in renames.items():
                        idx = df.index[df['Task'] == f'BP_{bp}']
                        if not idx.empty:
                            start_idx = idx[1]
                            mask_after = df.loc[start_idx+1:start_idx + 2, 'Event'].str.contains('Saliva', na=False)
                            if mask_after.all():
                                df.loc[start_idx + 1:start_idx + 2, 'Task'] = f'Saliva_{sv}'
                                df.loc[start_idx + 1, 'Event'] = 'start'
                                df.loc[start_idx + 2, 'Event'] = 'end'
                except Exception as z:
                    print_status('Failed Saliva Rename',f"{path} - {z!r}")
                    errors = errors + [f"Failed Saliva Rename: {path} - {z!r}",]

                P.EVM = pd.concat([P.EVM,df])
                #Set flags
                if len(P.EVM)>1:
                    P.isData = 1             
                #Update Metadata               
                print_status('Cleaned',{path})
            except Exception as z:
                print_status('Failed Psycho',f"{path} - {z!r}")
                errors = errors + [f"Failed Psycho: {path} - {z!r}",]
         
        #Total Participant Data
        if P.isData:
            P.EVM = P.EVM.drop_duplicates(subset=['Time','Event','Task',])
            
            P.EVM['Time']=P.EVM['Time'].str.replace(',','.')
            P.EVM['Task']=P.EVM['Task'].replace(to_replace=np.nan,value='benchmarking')
            P.EVM['t']=[get_sec(i) for i in P.EVM['Time'].tolist()]
            #P.EVM.to_csv(f"{out_path}{P.ID}_EVM.csv",index=False)
            
            dur = P.EVM['t'].max()-P.EVM['t'].min()
            P.dataDuration = dur
            #Correct benchmarking
            P.EVM.loc[P.EVM['Task'] == 'benchmarking', 'Task'] = (P.EVM.loc[P.EVM['Task'] == 'benchmarking', 'Event'].apply(lambda event: '_'.join(event.split('_')[:2])))
            P.EVM['Event'] = P.EVM['Event'].apply(lambda x: x.split('_')[-1])
            
            # Extract BP
            blood_pressure = P.EVM.copy()
            blood_pressure['Task']=blood_pressure['Task'].shift(1)
            blood_pressure = blood_pressure.loc[blood_pressure['Event']=='BP']
            blood_pressure.to_csv(f'{in_path}{P.ID}_BP.csv')

            # Clean up P.EVM
            P.EVM = P.EVM.loc[~P.EVM['Event'].str.contains('BP', na=False)]

            #Correct blankScreen
            if len(P.EVM.loc[P.EVM['Task'] == 'blankScreen_00'])==4:            
                P.EVM.loc[P.EVM['Task'] == 'blankScreen_00', 'Task'] = ['blankScreen_NS','blankScreen_NS','blankScreen_00','blankScreen_00']

            #Discard skipped tasks and add task durations
            P.tasksInfo = P.EVM

            for t in task_list_tasks:
                _data = P.tasksInfo.loc[P.tasksInfo['Task']==t]
                _versions = _data['date'].drop_duplicates()
                threshold = 0 if 'Saliva' in t else 3
            
                for v in _versions:
                        _vdata = _data.loc[_data['date']==v]
                        _vstart = _vdata.loc[_vdata['Event']=='start']['t'].tolist()[0]
                        try:
                            _vend = _vdata.loc[_vdata['Event']=='end']['t'].tolist()[0]
                            _vdur= _vend-_vstart
                            if _vdur < threshold:
                                pass
                            else:
                                P.tasksInfo.loc[(P.tasksInfo['Task']==t) & (P.tasksInfo['date']==v), 'Start'] = _vstart
                                P.tasksInfo.loc[(P.tasksInfo['Task']==t) & (P.tasksInfo['date']==v), 'End'] = _vend
                                P.tasksInfo.loc[(P.tasksInfo['Task']==t) & (P.tasksInfo['date']==v), 'Duration'] = _vdur
                        except Exception as z:
                            print_status('DNF',f"{t} in Version {v}")
                            errors = errors + [f"DNF: {t} in Version {v}",]
                            
            P.tasksInfo = P.tasksInfo.dropna(subset=['Duration'])
            P.tasksInfo = P.tasksInfo.drop_duplicates(subset='Task')
            P.tasksInfo = P.tasksInfo.drop(columns=['Event',]).reset_index(drop=True)
            P.tasksInfo.to_csv(f"{pickle_path}{P.ID}_EVM_log.csv",index=False)

            ### Check for missing data
            print_step('Checking Tasks',P.ID)
            tasks = list(dict.fromkeys(P.tasksInfo['Task'].tolist()))
            

            ### Begin opensignals
            total_expected_length = 0
            total_actual_length = 0


            for t in tasks:
                _start = P.tasksInfo.loc[(P.tasksInfo['Task']==t)]['Start'].tolist()[0]
                _end = P.tasksInfo.loc[(P.tasksInfo['Task']==t)]['End'].tolist()[0]
                _dur = P.tasksInfo.loc[(P.tasksInfo['Task']==t)]['Duration'].tolist()[0]
                try:                         
                    _actual_datapoints = P.signals.loc[(P.signals['t']>=_start)&(P.signals['t']<_end)]                        
                    _expected_length = (_dur)*_actual_datapoints['Freq'].tolist()[0]                             
                    _actual_length = len(_actual_datapoints)                             
                except Exception as z:
                    print_status('Missing Signals',t)
                    _actual_length = 0
                    try:
                        _expected_length = (_dur)*P.freq[0]
                    except:
                        _expected_length = (_dur)*400
                    errors = errors + [f"Missing Signals: {t} - {z!r}"]

                total_expected_length+=_expected_length
                total_actual_length+=_actual_length

                try:
                    _clean = {'Participant':P.ID,
                                'Task':t,
                                'Status': _actual_length/_expected_length*100
                                }
                except Exception as z:
                    print_status('Duration Error',f"{t}")
                    _clean = {'Participant':P.ID,
                                'Task':t,
                                'Status': np.nan
                                }
                    
            P.status = total_actual_length/total_expected_length*100
        else:
            P.status = 0       
   
    ### End of file scanning
    ### Set Flags

    P.isSignalsComplete = 1 if P.status >90 else 0
    P.isSignalsOverflow = 1 if P.signalDuration > 10000 else 0
    
    if P.isData:
        P.isDataComplete= 1 if len(P.tasksInfo['Task'])==35 else 0
    else: 
        P.isDataComplete= 0

    if P.isSignals:
        P.hasChannels = float(P.signals['Channels'].mean())           
        _Task = []
        for t in P.sample['t'].tolist():
            try:
                _task = P.tasksInfo.loc[(P.tasksInfo['Start']<=t) & (P.tasksInfo['End']>t)]['Task'].tolist()[0]
            except:
                _task = None
                pass
            _Task.append(_task)
        P.sample.insert(len(P.sample.columns),'Task',_Task)
        P.sample.to_csv(f"{pickle_path}{P.ID}_sample.csv",index=False)
    
    P.isCleaned = 1
    clean = {'Participant':P.ID,
                'Date':P.date,
                'Session':P.session,
                'OpenSignalsData':P.isSignals,
                'Complete':P.isSignalsComplete,
                'Status':P.status,
                'Channels':P.hasChannels,
                'Overflow':P.isSignalsOverflow,
                'Duration':P.signalDuration,
                'PsychoPy':P.isData,
                'PsychoPyComplete':P.isDataComplete,
                'Errors':';'.join(errors)
                }
    clean_log = pd.concat([clean_log,pd.DataFrame([clean])], ignore_index=True)

    if os.path.exists(f"{out_folder}clean_log.csv"):
        old_log = pd.read_csv(f"{out_folder}clean_log.csv")
        #If participant already has data in clean log
        if P.ID in old_log['Participant'].tolist():
            for key, value in clean.items():
                old_log.loc[old_log['Participant']==P.ID, key] = value
            new_log = old_log
        else:
            clean_log = pd.concat([clean_log,pd.DataFrame([clean])], ignore_index=True)
            new_log = pd.concat([old_log, pd.DataFrame([clean])], ignore_index=True)
        new_log.to_csv(f"{out_folder}clean_log.csv",index=False)
    else:
        clean_log.to_csv(f"{out_folder}clean_log.csv",index=False)
    
    try:
        print_status(f"Collected {P.ID}",f"{int(P.status)}%")
    except:
        print_status('### Some things failed',P.ID)
    return P
        

def update_goNoGo(self):
    self.cntGoNoGoFiles = 0
    self.cntGoNoGoPlaced = 0
    self.cntGoNoGoProcessed = 0

        
def get_goNoGo(in_folder, out_folder,P):
    '''
    This function cleans and combines all the OpenSignals data from one participant into a single file

    Parameters
    __________
    in_folder: str
        Folder containing directories for each participant
        Must contain ONLY particpant 
    out_folder: str
        Folder in which a directory will be created per participant
        
    TODO:
    - Add in check for duplicate participant IDs
    '''
    
    #Define variables
    psychoPy_log = pd.DataFrame()

    errors = []
    in_path = f"{in_folder}{P.ID}/"
    out_path = f"{out_folder}{P.ID}/"
    os.makedirs(out_path, exist_ok=True) 
    
    EVM_Results = pd.DataFrame()
    EVM_task = 'goNoGo'

    ##Reset Flags
    P.cntGoNoGoFiles = 0
    P.cntGoNoGoPlaced = 0
    P.cntGoNoGoProcessed = 0
    
    taskData=pd.DataFrame()
    print_step(f'Collecting {EVM_task}',P.ID)

    ##### GoNoGo Data
    ##Extract data from files
    files = get_files(in_path,tags=['.csv', EVM_task]) #Get list of all goNogo tasks
    
    #For each file
    file_num = 0
    for f in files:                   
        try:
            path = in_path + f                 
            f=open(path)                    
            df = pd.read_csv(path, low_memory = False)
            df=df[['Task','Event','Time','Response_Time','Accurate','Expected']]
            df['Path']=path
            df['File']=file_num

            ##Adjust data
            #Rename columns
            df. rename(columns = {'Task':'Trial'}, inplace = True)
            #Replace entries
            df.replace('Task_1','Task1',inplace=True)
            df.replace('Task_2','Task2',inplace=True)
            df.replace('Task_1_lag','Task3',inplace=True)
            #create absolute time column
            df['Time']=df['Time'].str.replace(',','.')
            df['t']=[get_sec(i) for i in df['Time'].tolist()]

            ##Update Counters
            file_num = file_num+1
            print_status(f'Extracted {EVM_task}',path)

            ##Contextualise Data
            #set task from EVM list
            try:
                start_time = df['t'].tolist()[0]
                end_time = df['t'].tolist()[-1]
                task = P.tasksInfo.loc[(P.tasksInfo['Start']<=start_time) & (P.tasksInfo['End']>=end_time)]['Task'].tolist()[0]
                df['Task']=task
                print_status('Placed Data',task)

                #########################
                P.cntGoNoGoPlaced += 1
            except IndexError:
                print_status('Unplaced Data',path)
            except Exception as z:
                print_status(f'Failed Placing {EVM_task} Data',f"{path} - {z!r}")
                errors = errors + [f"Failed Placing {EVM_task} Data: {path} - {z!r}",]
                          
            ##Update Participant Data
            taskData = pd.concat([taskData,df])

            #########################
            P.cntGoNoGoFiles += 1
        
        ## Account for Issues
        except KeyError as z:
            if any('webcam' in c for c in df.columns):
                npath = path.replace('_goNoGo','_tsst')
                df.to_csv(npath, index=False)
                print_status('Found TSST',f"{path} - saved as {npath}")
                errors = errors + [f"Failed GoNoGo: {path} - saved as {npath}",]
            else:
                print_status(f'Failed {EVM_task}',f"{path} - {z!r}")
                errors = errors + [f"Failed {EVM_task}: {path} - {z!r}",]
        except Exception as z:
            print_status(f'Failed {EVM_task}',f"{path} - {z!r}")
            errors = errors + [f"Failed {EVM_task}: {path} - {z!r}",]
    
    if P.cntGoNoGoPlaced:
        ## Set condition
        taskData = taskData.dropna(subset=['Task'])
        taskData.loc[taskData['Task'].str.contains('_01'), 'Condition']='Baseline'
        taskData.loc[taskData['Task'].str.contains('_02'), 'Condition']='Treatment'

        ###Process GoNoGo Data 
        ## Initialise row entry
        calc = {'Participant':P.ID,
                'Drink':P.drink,}

        ## Results
        if P.cntGoNoGoPlaced:
            #for each condition
            conditions = taskData['Condition'].drop_duplicates()
            for condition in conditions:
                _condition = taskData.loc[taskData['Condition']==condition]

                #Update row entry
                _accuracy = sum(1 for item in _condition['Accurate'] if item is True)/len(_condition['Accurate'])
                _rt = _condition.loc[(_condition['Expected']=='space')]['Response_Time'].mean()
                _calc = {f"psy_{EVM_task}_{condition}_Overall_Accuracy":_accuracy,
                    f"psy_{EVM_task}_{condition}_Overall_RT":_rt,
                    }
                merge_dict(_calc,calc)

                #########################
                P.cntGoNoGoProcessed += 1

                trials = _condition['Trial'].drop_duplicates()
                for trial in trials:
                    _trial = _condition.loc[_condition['Trial']==trial]
                    _accuracy = sum(1 for item in _trial['Accurate'] if item is True)/len(_trial['Accurate'])
                    _rt = _trial['Response_Time'].mean()
                    _rtT = _trial.loc[(_trial['Expected']=='space')&(_trial['Accurate']==True)]['Response_Time'].mean()
                    _rtF = _trial.loc[(_trial['Expected']=='space')&(_trial['Accurate']==False)]['Response_Time'].mean()

                    _calc = {f"psy_{EVM_task}_{condition}_{trial}_Accuracy":_accuracy,
                             f"psy_{EVM_task}_{condition}_{trial}_RT":_rt,
                             f"psy_{EVM_task}_{condition}_{trial}_RTAccurate":_rtT,
                             f"psy_{EVM_task}_{condition}_{trial}_RTInaccurate":_rtF,
                            }
                    merge_dict(_calc,calc)
    
        ### Update Participant Results
        P.psychoPyResults = pd.concat([P.psychoPyResults, pd.DataFrame([calc])], ignore_index=True)
    
        ### Update EVM Results File
        if os.path.exists(f"{out_folder}EVM_Results.csv"):
            old_log = pd.read_csv(f"{out_folder}EVM_Results.csv")
            #If participant already has data in clean log
            if P.ID in old_log['Participant'].tolist():
                for key, value in calc.items():
                    old_log.loc[old_log['Participant']==P.ID, key] = value
                new_log = old_log
            else:
                new_log = pd.concat([old_log, pd.DataFrame([calc])], ignore_index=True)
            new_log.to_csv(f"{out_folder}EVM_Results.csv",index=False)
        else:
            EVM_Results = pd.DataFrame()
            EVM_Results = pd.concat([EVM_Results, pd.DataFrame([calc])], ignore_index=True)
            EVM_Results.to_csv(f"{out_folder}EVM_Results.csv",index=False)

    ### Update Log File

    log = {'Participant':P.ID,
            'Date':P.date,
            'Status':P.status,      
            'PsychoPy':P.isData,
            'PsychoPyComplete':P.isDataComplete,
            'Task':EVM_task,
            'Files':P.cntGoNoGoFiles,
            'Placed':P.cntGoNoGoPlaced,
            'Processed':P.cntGoNoGoProcessed,
            'Errors':';'.join(errors)
            }

    if os.path.exists(f"{out_folder}psychoPy_log.csv"):
        old_log = pd.read_csv(f"{out_folder}psychoPy_log.csv")
        #If participant already has data in clean log
        if (P.ID in old_log['Participant'].tolist()) and (old_log.loc[old_log['Participant']==P.ID]['Task'].str.contains(EVM_task).any()):
            for key, value in log.items():
                old_log.loc[(old_log['Participant']==P.ID)&(old_log['Task'].str.contains(EVM_task)), key] = value
            new_log = old_log
        else:
            new_log = pd.concat([old_log, pd.DataFrame([log])], ignore_index=True)
        new_log.to_csv(f"{out_folder}psychoPy_log.csv",index=False)
    else:
        psychoPy_log = pd.DataFrame()
        psychoPy_log = pd.concat([psychoPy_log, pd.DataFrame([log])], ignore_index=True)
        psychoPy_log.to_csv(f"{out_folder}psychoPy_log.csv",index=False)
    
    #End

    #############################
    print_status(f"Collected {EVM_task}",f"{int(P.cntGoNoGoProcessed/2*100)}%")  
    return P

def update_simpleRT(self):
    self.cntSimpleRTFiles = 0
    self.cntSimpleRTPlaced = 0
    self.cntSimpleRTProcessed = 0

        
def get_simpleRT(in_folder, out_folder,P):
    '''
    This function cleans and combines all the OpenSignals data from one participant into a single file

    Parameters
    __________
    in_folder: str
        Folder containing directories for each participant
        Must contain ONLY particpant 
    out_folder: str
        Folder in which a directory will be created per participant
        
    TODO:
    - Add in check for duplicate participant IDs
    '''
    
    #Define variables
    psychoPy_log = pd.DataFrame()

    errors = []
    in_path = f"{in_folder}{P.ID}/"
    out_path = f"{out_folder}{P.ID}/"
    os.makedirs(out_path, exist_ok=True) 
    
    EVM_Results = pd.DataFrame()
    EVM_task = 'simpleRT'

    ##Reset Flags
    P.cntSimpleRTFiles = 0
    P.cntSimpleRTPlaced = 0
    P.cntSimpleRTProcessed = 0
    
    taskData=pd.DataFrame()
    print_step(f'Collecting {EVM_task}',P.ID)

    ##### GoNoGo Data
    ##Extract data from files
    files = get_files(in_path,tags=['.csv', EVM_task]) #Get list of all goNogo tasks
    
    #For each file
    file_num = 0
    for file in files:                   
        try:
            path = in_path + file                
            f=open(path)                    
            df = pd.read_csv(path, low_memory = False)
            df=df[['Time','Response_Time','Response']]
            df['Path']=path
            df['File']=file_num

            df['Time']=df['Time'].str.replace(',','.')
            df['t']=[get_sec(i) for i in df['Time'].tolist()]
            df['Task'] = '_'.join(file.split('_')[1:3])
            ##Update Counters
            file_num = file_num+1
            print_status(f'Extracted {EVM_task}',path)
            P.cntSimpleRTPlaced += 1                          
            ##Update Participant Data
            taskData = pd.concat([taskData,df])

            #########################
            P.cntSimpleRTFiles += 1
        
        ## Account for Issues
        except Exception as z:
            print_status(f'Failed {EVM_task}',f"{path} - {z!r}")
            errors = errors + [f"Failed {EVM_task}: {path} - {z!r}",]
    
    if P.cntSimpleRTPlaced:
        ## Set condition
        taskData.loc[taskData['Task'].str.contains('_00'), 'Condition']='Baseline'
        taskData.loc[taskData['Task'].str.contains('_01'), 'Condition']='Fatigue'

        ###Process simpleRT Data 
        ## Initialise row entry
        calc = {'Participant':P.ID,
                'Drink':P.drink,}

        ## Results
        if P.cntSimpleRTPlaced:
            #for each condition
            conditions = taskData['Condition'].drop_duplicates()
            for condition in conditions:
                _condition = taskData.loc[taskData['Condition']==condition]
                _calc = dict()
                length = len(_condition)
                ### Check for bad data
                false_starts = len(_condition.loc[_condition['Response']=='FALSE START'])
                _calc[f"psy_{EVM_task}_{condition}_FalseStarts"]=false_starts
                _condition = _condition.loc[(_condition['Response_Time']>0.05)
                                            & (_condition['Response_Time']<1)]
                
                ## Givern 75 responses, get length of valid data
                valid_data = len(_condition)/length*100

                
                if valid_data>10:
                    _rt = _condition['Response_Time'].mean() # Get mean
                    _calc[f"psy_{EVM_task}_{condition}_RT"]=_rt
                    if valid_data>50: #Accept data if greater than 60% valid
                        #Update row entry
                        _sa,_ = np.polyfit(_condition['t'], _condition['Response_Time'], 1) #Get gradient of line of best fit for sustained attention
                        _calc[f"psy_{EVM_task}_{condition}_SustainedAttention"]=_sa
                    else:
                        print_status(f'Failed {condition} SA',f"Only {valid_data} percent valid")    
                    merge_dict(_calc,calc)
                    #########################
                    P.cntSimpleRTProcessed += 1
                else:
                    print_status(f'Failed {condition} RT',f"Only {valid_data} percent valid")

                        # Update timings
                taskName = _condition['Task'].values[0]
                timings = {'Task':taskName,
                        'Start':_condition.t.values[0],
                        'End':_condition.t.values[-1],
                        'Duration':_condition.t.values[-1] - _condition.t.values[0]}
                P.tasksInfo = P.tasksInfo.reset_index(drop=True)
                if taskName in P.tasksInfo['Task'].tolist():
                    P.tasksInfo.loc[P.tasksInfo['Task']==taskName, 'Start'] = _condition.t.values[0]
                    P.tasksInfo.loc[P.tasksInfo['Task']==taskName, 'End'] = _condition.t.values[-1]
                    P.tasksInfo.loc[P.tasksInfo['Task']==taskName, 'Duration'] = _condition.t.values[-1] - _condition.t.values[0]
                else:
                    P.tasksInfo.loc[len(P.tasksInfo)] = timings
                P.tasksInfo = P.tasksInfo.sort_values('Start')
                P.tasksInfo.to_csv(f"{out_path}{P.ID}_EVM_log.csv",index=False)
    
        ### Update Participant Results
        P.psychoPyResults = pd.concat([P.psychoPyResults, pd.DataFrame([calc])], ignore_index=True)
    
        ### Update EVM Results File
        if os.path.exists(f"{out_folder}EVM_Results.csv"):
            old_log = pd.read_csv(f"{out_folder}EVM_Results.csv")
            #If participant already has data in clean log
            if P.ID in old_log['Participant'].tolist():
                for key, value in calc.items():
                    old_log.loc[old_log['Participant']==P.ID, key] = value
                new_log = old_log
            else:
                new_log = pd.concat([old_log, pd.DataFrame([calc])], ignore_index=True)
            new_log.to_csv(f"{out_folder}EVM_Results.csv",index=False)
        else:
            EVM_Results = pd.DataFrame()
            EVM_Results = pd.concat([EVM_Results, pd.DataFrame([calc])], ignore_index=True)
            EVM_Results.to_csv(f"{out_folder}EVM_Results.csv",index=False)

    ### Update Log File

    log = {'Participant':P.ID,
            'Date':P.date,
            'Status':P.status,      
            'PsychoPy':P.isData,
            'PsychoPyComplete':P.isDataComplete,
            'Task':EVM_task,
            'Files':P.cntSimpleRTFiles,
            'Placed':P.cntSimpleRTPlaced,
            'Processed':P.cntSimpleRTProcessed,
            'Errors':';'.join(errors)
            }

    if os.path.exists(f"{out_folder}psychoPy_log.csv"):
        old_log = pd.read_csv(f"{out_folder}psychoPy_log.csv")
        #If participant already has data in clean log
        if (P.ID in old_log['Participant'].tolist()) and (old_log.loc[old_log['Participant']==P.ID]['Task'].str.contains(EVM_task).any()):
            for key, value in log.items():
                old_log.loc[(old_log['Participant']==P.ID)&(old_log['Task'].str.contains(EVM_task)), key] = value
            new_log = old_log
        else:
            new_log = pd.concat([old_log, pd.DataFrame([log])], ignore_index=True)
        new_log.to_csv(f"{out_folder}psychoPy_log.csv",index=False)
    else:
        psychoPy_log = pd.DataFrame()
        psychoPy_log = pd.concat([psychoPy_log, pd.DataFrame([log])], ignore_index=True)
        psychoPy_log.to_csv(f"{out_folder}psychoPy_log.csv",index=False)
    
    #End

    #############################
    print_status(f"Collected {EVM_task}",f"{int(P.cntSimpleRTProcessed/2*100)}%")  
    return P


def update_Cycle(self):
    self.cntCycleFiles = 0
    self.cntCyclePlaced = 0
    self.cntCycleProcessed = 0

        
def get_Cycle(in_folder, out_folder,P):
    '''
    This function cleans and combines all the OpenSignals data from one participant into a single file

    Parameters
    __________
    in_folder: str
        Folder containing directories for each participant
        Must contain ONLY particpant 
    out_folder: str
        Folder in which a directory will be created per participant
        
    TODO:
    - Add in check for duplicate participant IDs
    '''
    
    #Define variables
    psychoPy_log = pd.DataFrame()

    errors = []
    in_path = f"Analysis/Raw/Cycle/"
    out_path = f"{out_folder}{P.ID}/"
    os.makedirs(out_path, exist_ok=True) 
    
    EVM_Results = pd.DataFrame()
    EVM_task = 'Cycle'

    ##Reset Flags
    P.cntCycleFiles = 0
    P.cntCyclePlaced = 0
    P.cntCycleProcessed = 0
    
    taskData=pd.DataFrame()
    print_step(f'Collecting {EVM_task}',P.ID)

    ##### GoNoGo Data
    ##Extract data from files
    files = get_files(in_path,tags=['.csv', P.ID]) #Get list of all goNogo tasks
    
    #For each file
    file_num = 0
    for f in files:                   
        try:
            path = in_path + f                 
            f=open(path)                    
            df = pd.read_csv(path, low_memory = False)
            df['Path']=path
            df['File']=file_num
            ##Update Counters
            file_num = file_num+1
            print_status(f'Extracted {EVM_task}',path)
            P.cntCyclePlaced += 1                          
            ##Update Participant Data
            taskData = pd.concat([taskData,df])

            #########################
            P.cntCycleFiles += 1
        
        ## Account for Issues
        except Exception as z:
            print_status(f'Failed {EVM_task}',f"{path} - {z!r}")
            errors = errors + [f"Failed {EVM_task}: {path} - {z!r}",]

    if file_num>1:
        print_status(f'Warning:',f"Found more than one cycle file")
    
    if P.cntCyclePlaced:
        ## Set condition
        ###Process Cycle Data 
        ## Initialise row entry
        calc = {'Participant':P.ID,
                'Drink':P.drink,}

        ## Results
        if P.cntCyclePlaced:
            #for each condition
            total_time = taskData['Time (seconds)'].values[-1]
            auc_data = taskData.dropna(subset=['Time (seconds)','Watts'])
            total_power = np.trapz(auc_data['Watts'],x=auc_data['Time (seconds)'])/3600
            average_watts = taskData['Watts'].mean()
            average_strokes = taskData['Stroke Rate'].mean()

            ### Check for bad data
            calc[f"psy_{EVM_task}_CycleTime"]=total_time
            calc[f"psy_{EVM_task}_TotalPower"]=total_power
            calc[f"psy_{EVM_task}_AverageWatts"]=average_watts
            calc[f"psy_{EVM_task}_AverageStrokes"]=average_strokes

            #########################
            P.cntCycleProcessed += 1

    
        ### Update Participant Results
        P.psychoPyResults = pd.concat([P.psychoPyResults, pd.DataFrame([calc])], ignore_index=True)
    
        ### Update EVM Results File
        if os.path.exists(f"{out_folder}EVM_Results.csv"):
            old_log = pd.read_csv(f"{out_folder}EVM_Results.csv")
            #If participant already has data in clean log
            if P.ID in old_log['Participant'].tolist():
                for key, value in calc.items():
                    old_log.loc[old_log['Participant']==P.ID, key] = value
                new_log = old_log
            else:
                new_log = pd.concat([old_log, pd.DataFrame([calc])], ignore_index=True)
            new_log.to_csv(f"{out_folder}EVM_Results.csv",index=False)
        else:
            EVM_Results = pd.DataFrame()
            EVM_Results = pd.concat([EVM_Results, pd.DataFrame([calc])], ignore_index=True)
            EVM_Results.to_csv(f"{out_folder}EVM_Results.csv",index=False)



    ### Update Log File

    log = {'Participant':P.ID,
            'Date':P.date,
            'Status':P.status,      
            'PsychoPy':P.isData,
            'PsychoPyComplete':P.isDataComplete,
            'Task':EVM_task,
            'Files':P.cntCycleFiles,
            'Placed':P.cntCyclePlaced,
            'Processed':P.cntCycleProcessed,
            'Errors':';'.join(errors)
            }

    if os.path.exists(f"{out_folder}psychoPy_log.csv"):
        old_log = pd.read_csv(f"{out_folder}psychoPy_log.csv")
        #If participant already has data in clean log
        if (P.ID in old_log['Participant'].tolist()) and (old_log.loc[old_log['Participant']==P.ID]['Task'].str.contains(EVM_task).any()):
            for key, value in log.items():
                old_log.loc[(old_log['Participant']==P.ID)&(old_log['Task'].str.contains(EVM_task)), key] = value
            new_log = old_log
        else:
            new_log = pd.concat([old_log, pd.DataFrame([log])], ignore_index=True)
        new_log.to_csv(f"{out_folder}psychoPy_log.csv",index=False)
    else:
        psychoPy_log = pd.DataFrame()
        psychoPy_log = pd.concat([psychoPy_log, pd.DataFrame([log])], ignore_index=True)
        psychoPy_log.to_csv(f"{out_folder}psychoPy_log.csv",index=False)
    
    #End

    #############################
    print_status(f"Collected {EVM_task}",f"{int(P.cntCycleProcessed*100)}%")  
    return P


def update_RAT(self):
    self.cntRATFiles = 0
    self.cntRATPlaced = 0
    self.cntRATProcessed = 0


def get_RAT(in_folder, out_folder,P):
    '''
    This function collects all RAT data from participant

    Parameters
    __________
    in_folder: str
        Folder containing directories for each participant
        Must contain ONLY particpant 
    out_folder: str
        Folder in which a directory will be created per participant
        
    TODO:
    - Add in check for duplicate participant IDs
    '''
    
    #Define variables
    psychoPy_log = pd.DataFrame()

    errors = []
    in_path = f"{in_folder}{P.ID}/"
    out_path = f"{out_folder}{P.ID}/"
    os.makedirs(out_path, exist_ok=True) 
    
    EVM_Results = pd.DataFrame()
    EVM_task = 'RAT'
    # client = OpenAI()
    # model="gpt-4-0613"
    # messages=[{"role": "system", "content": "You are a simple model that takes in two words seperated by a space. \
    #     You then assess if those two words make a compound phrase or a compound word. \
    #     The compound word or phrase has to be valid and common in the english language, and can include a hyphenated word. \
    #     Your output is a simple '1', if the words form a compound word or phrase, and '0' if the words do not.\
    #     The words may not necessarily be in the right order"},]
    
    #If the output is 1, please also provide the typical way in which this compound word or phrase is written, and a single paragraph capturing its definition and context
    


    ##Reset Flags
    P.cntRATFiles = 0
    P.cntRATPlaced = 0
    P.cntRATProcessed = 0
    
    taskData=pd.DataFrame()
    print_step(f'Collecting {EVM_task}',P.ID)

    ##### GoNoGo Data
    ##Extract data from files
    files = get_files(in_path,tags=['.csv', EVM_task]) #Get list of all goNogo tasks
    
    #For each file
    file_num = 0
    for f in files:                   
        try:
            path = in_path + f                 
            f=open(path)                    
            df = pd.read_csv(path, low_memory = False)
            df=df[['Time','Response','Accuracy','Target']]
            df['Path']=path
            df['File']=file_num

            ##Adjust data
            df = df.drop_duplicates(subset=['Target',])
            #create absolute time column
            df['Time']=df['Time'].str.replace(',','.')
            df['t']=[get_sec(i) for i in df['Time'].tolist()]
            ##Update Counters
            file_num = file_num+1
            print_status(f'Extracted {EVM_task}',path)

            ##Contextualise Data
            #set task from EVM list
            try:
                start_time = df['t'].tolist()[0]
                end_time = df['t'].tolist()[-1]
                task = P.tasksInfo.loc[(P.tasksInfo['Start']<=start_time) & (P.tasksInfo['End']>=end_time)]['Task'].tolist()[0]
                df['Task']=task
                print_status('Placed Data',task)

                #########################
                P.cntRATPlaced += 1
            except IndexError:
                print_status('Unplaced Data',path)
            except Exception as z:
                print_status(f'Failed Placing {EVM_task} Data',f"{path} - {z!r}")
                errors = errors + [f"Failed Placing {EVM_task} Data: {path} - {z!r}",]
                          
            ##Update Participant Data
            taskData = pd.concat([taskData,df])

            #########################
            P.cntRATFiles += 1
        
        ## Account for Issues
        except Exception as z:
            print_status(f'Failed {EVM_task}',f"{path} - {z!r}")
            errors = errors + [f"Failed {EVM_task}: {path} - {z!r}",]
    
    if P.cntRATPlaced:
        ## Set condition
        taskData = taskData.dropna(subset=['Task'])
        taskData.loc[taskData['Task'].str.contains('_01'), 'Condition']='Baseline'
        taskData.loc[taskData['Task'].str.contains('_02'), 'Condition']='Treatment'

        ###Process RAT Data 
        ## Initialise row entry
        calc = {'Participant':P.ID,
                'Drink':P.drink,}

        ## Results
        if P.cntRATPlaced:
            #for each condition
            conditions = taskData['Condition'].drop_duplicates()
            spell = SpellChecker()
            for condition in conditions:
                _condition = taskData.loc[taskData['Condition']==condition]
                
                score = 0
                ### Check for bad data
                for index, row in _condition.iterrows():
                    if row['Accuracy']:
                        score += 3
                    else:
                        word = row['Response']
                        # Apply spellcheck to word
                        correct = spell.correction(word)
                        targets = ast.literal_eval(row['Target'])

                        for target in targets:
                            specific_message = {"role": "user", "content":f'{correct} {target}'}
                            specific_messages = messages
                            specific_messages.append(specific_message)

                            completion = client.chat.completions.create(
                                model=model,
                                messages=specific_messages
                                )
                            
                            point = float(completion.choices[0].message.content)
                            #print(f'{correct} {target}, {point}')
                            score += point

                _calc = dict()
                _calc[f"psy_{EVM_task}_{condition}_Score"]=score
                merge_dict(_calc,calc)

                #########################
                P.cntRATProcessed += 1
    
        ### Update Participant Results
        P.psychoPyResults = pd.concat([P.psychoPyResults, pd.DataFrame([calc])], ignore_index=True)
    
        ### Update EVM Results File
        if os.path.exists(f"{out_folder}EVM_Results.csv"):
            old_log = pd.read_csv(f"{out_folder}EVM_Results.csv")
            #If participant already has data in clean log
            if P.ID in old_log['Participant'].tolist():
                for key, value in calc.items():
                    old_log.loc[old_log['Participant']==P.ID, key] = value
                new_log = old_log
            else:
                new_log = pd.concat([old_log, pd.DataFrame([calc])], ignore_index=True)
            new_log.to_csv(f"{out_folder}EVM_Results.csv",index=False)
        else:
            EVM_Results = pd.DataFrame()
            EVM_Results = pd.concat([EVM_Results, pd.DataFrame([calc])], ignore_index=True)
            EVM_Results.to_csv(f"{out_folder}EVM_Results.csv",index=False)

    ### Update Log File

    log = {'Participant':P.ID,
            'Date':P.date,
            'Status':P.status,      
            'PsychoPy':P.isData,
            'PsychoPyComplete':P.isDataComplete,
            'Task':EVM_task,
            'Files':P.cntRATFiles,
            'Placed':P.cntRATPlaced,
            'Processed':P.cntRATProcessed,
            'Errors':';'.join(errors)
            }

    if os.path.exists(f"{out_folder}psychoPy_log.csv"):
        old_log = pd.read_csv(f"{out_folder}psychoPy_log.csv")
        #If participant already has data in clean log
        if (P.ID in old_log['Participant'].tolist()) and (old_log.loc[old_log['Participant']==P.ID]['Task'].str.contains(EVM_task).any()):
            for key, value in log.items():
                old_log.loc[(old_log['Participant']==P.ID)&(old_log['Task'].str.contains(EVM_task)), key] = value
            new_log = old_log
        else:
            new_log = pd.concat([old_log, pd.DataFrame([log])], ignore_index=True)
        new_log.to_csv(f"{out_folder}psychoPy_log.csv",index=False)
    else:
        psychoPy_log = pd.DataFrame()
        psychoPy_log = pd.concat([psychoPy_log, pd.DataFrame([log])], ignore_index=True)
        psychoPy_log.to_csv(f"{out_folder}psychoPy_log.csv",index=False)
    
    #End

    #############################
    print_status(f"Collected {EVM_task}",f"{int(P.cntRATProcessed/2*100)}%")  
    return P
    

def update_VGT(self):
    self.cntVGTFiles = 0
    self.cntVGTPlaced = 0
    self.cntVGTProcessed = 0


def get_VGT(in_folder, out_folder,P):
    '''
    This function collects all VGT data from participant

    Parameters
    __________
    in_folder: str
        Folder containing directories for each participant
        Must contain ONLY particpant 
    out_folder: str
        Folder in which a directory will be created per participant
        
    TODO:
    - Add in check for duplicate participant IDs
    '''

    '''
    plan:
    First, collect all responses from all participants. 
    Then, for each stim, do a count on each response. Use the reciprocal of this as an index for creativity
    For each response, this can be used to measure creativity. A second measure can be used as a sense check.

    '''
    
    #Define variables
    psychoPy_log = pd.DataFrame()

    errors = []
    in_path = f"{in_folder}{P.ID}/"
    out_path = f"{out_folder}{P.ID}/"
    os.makedirs(out_path, exist_ok=True) 
    
    EVM_Results = pd.DataFrame()
    EVM_task = 'VGT'
    # client = OpenAI()
    # model="gpt-4-0613"
    # prompts = {'creativity':"You are a simple model that asesses the results of a verb generation task. \
    #           In this task, participants are shown a noun, and are then asked to come up with a use, verb or action that is connected to or associated with the noun.\
    #           The task is meant to assess divergent thinking.\
    #           You will receive a participants response to this task in the form of two inputs seperated by a ':'. The first is the noun, and the second is the participants response as either a word or a phrase that describes a use, verb or action that can be associated with the target. \
    #           Please give the response a score on a scale of 0 to 10. \
    #           The score should also refect the unusualness of the response generated reflecting divergent thinking, with a higher score reflecting a more unusual answer. \
    #           It should also relfect the latent semantic distance between the noun and the response, with a higher score reflecting a higher latent semantic distance \
    #           Your output is simply a single number, the score. \
    #           ",
    #           'accuracy':
    #          "You are a simple model that asesses the results of a verb generation task. \
    #           In this task, participants are shown a noun, and are then asked to come up with a use, verb or action that is connected to or associated with the noun.\
    #           The task is meant to assess divergent thinking.\
    #           You will receive a participants response to this task in the form of two inputs seperated by a ':'. The first is the noun, and the second is the participants response as either a word or a phrase that describes a use, verb or action that can be associated with the target. \
    #           Please give the response a score on a scale of 0 to 10. \
    #           This score should reflect how accurately and relevantly the participant can generate responses that are appropriate to the presented noun, with a higher score reflecting a higher accuracy \
    #           It should reflect the linguistic accuracy or correctness of the association in relation to the noun given, with a higher score reflecting a higher accuracy. \
    #           Your output is simply a single number, the score. \
    #             ", }
    
    def clean_convert_and_spellcheck(input_string):
        # Check if input_string is not a string
        if not isinstance(input_string, str):
            return []
        
        # Initialize the spell checker
        spell = SpellChecker()
        
        # Remove all unwanted characters except alphanumeric, commas, and spaces
        cleaned_string = re.sub(r'[^\w\s,]', '', input_string)
        
        # Split the string by commas to form a list
        items = cleaned_string.split(',')
        
        # Process each item: strip spaces, convert to lowercase, and spell check
        cleaned_list = []
        for item in items:
            item = item.strip().lower()  # Strip and lower case
            if item:  # Ensure the item is not empty
                # Spellcheck and correct each word within the item
                words = item.split()
                corrected_words = [spell.correction(word) for word in words]
                corrected_item = ' '.join(corrected_words)
                cleaned_list.append(corrected_item)
        
        return cleaned_list

    ##Reset Flags
    P.cntVGTFiles = 0
    P.cntVGTPlaced = 0
    P.cntVGTProcessed = 0
    
    taskData=pd.DataFrame()
    print_step(f'Collecting {EVM_task}',P.ID)

    ##### GoNoGo Data
    ##Extract data from files
    files = get_files(in_path,tags=['.csv', EVM_task]) #Get list of all goNogo tasks
    
    #For each file
    file_num = 0
    for f in files:                   
        try:
            path = in_path + f                 
            f=open(path)                    
            df = pd.read_csv(path, low_memory = False)
            df=df[['Time','Stim','Response']]
            df['Path']=path
            df['File']=file_num

            ##Adjust data
            df = df.drop_duplicates(subset=['Stim',])
            #create absolute time column
            df['Time']=df['Time'].str.replace(',','.')
            df['t']=[get_sec(i) for i in df['Time'].tolist()]
            ##Update Counters
            file_num = file_num+1
            print_status(f'Extracted {EVM_task}',path)

            ##Contextualise Data
            #set task from EVM list
            try:
                start_time = df['t'].tolist()[0]
                end_time = df['t'].tolist()[-1]
                task = P.tasksInfo.loc[(P.tasksInfo['Start']<=start_time) & (P.tasksInfo['End']>=end_time)]['Task'].tolist()[0]
                df['Task']=task
                print_status('Placed Data',task)

                #########################
                P.cntVGTPlaced += 1
            except IndexError:
                print_status('Unplaced Data',path)
            except Exception as z:
                print_status(f'Failed Placing {EVM_task} Data',f"{path} - {z!r}")
                errors = errors + [f"Failed Placing {EVM_task} Data: {path} - {z!r}",]
                          
            ##Update Participant Data
            taskData = pd.concat([taskData,df])

            #########################
            P.cntVGTFiles += 1
        
        ## Account for Issues
        except Exception as z:
            print_status(f'Failed {EVM_task}',f"{path} - {z!r}")
            errors = errors + [f"Failed {EVM_task}: {path} - {z!r}",]
    
    if P.cntVGTPlaced:
        ## Set condition
        taskData = taskData.dropna(subset=['Task'])
        taskData.loc[taskData['Task'].str.contains('_01'), 'Condition']='Baseline'
        taskData.loc[taskData['Task'].str.contains('_02'), 'Condition']='Treatment'

        ###Process VGT Data 
        ## Initialise row entry
        calc = {'Participant':P.ID,
                'Drink':P.drink,}

        ## Results
        if P.cntVGTPlaced:
            #for each condition
            conditions = taskData['Condition'].drop_duplicates()
            spell = SpellChecker()
            for condition in conditions:
                _condition = taskData.loc[taskData['Condition']==condition]
                
                accuracy = 0
                creativity = 0
                ### Check for bad data
                for index, row in _condition.iterrows():
                    responses = clean_convert_and_spellcheck(row['Response'])
                    # Apply spellcheck to word

                    # #Accuracy
                    # for response in responses:
                    #     specific_message = [{"role": "system", "content":prompts['accuracy']},
                    #                         {"role": "user", "content":f"{row['Stim']}:{response}"}]
                      
                    #     point = 0
                    #     for i in range(3):
                    #         completion = client.chat.completions.create(
                    #             model=model,
                    #             messages=specific_message
                    #             )
                    #         point += float(completion.choices[0].message.content)
                        
                    #     accuracy += point/3

                    # #Creativity
                    # for response in responses:
                    #     specific_message = [{"role": "system", "content":prompts['creativity']},
                    #                         {"role": "user", "content":f"{row['Stim']}:{response}"}]

                    #     point = 0
                    #     for i in range(3):
                    #         completion = client.chat.completions.create(
                    #             model=model,
                    #             messages=specific_message
                    #             )
                    #         point += float(completion.choices[0].message.content)
                        
                    #     creativity += point/3

                _calc = dict()
                _calc[f"psy_{EVM_task}_{condition}_Count"]=len(responses)
                #_calc[f"psy_{EVM_task}_{condition}_Accuracy"]=accuracy
                #_calc[f"psy_{EVM_task}_{condition}_Creativity"]=creativity
                merge_dict(_calc,calc)

                #########################
                P.cntVGTProcessed += 1
    
        ### Update Participant Results
        P.psychoPyResults = pd.concat([P.psychoPyResults, pd.DataFrame([calc])], ignore_index=True)
    
        ### Update EVM Results File
        if os.path.exists(f"{out_folder}EVM_Results.csv"):
            old_log = pd.read_csv(f"{out_folder}EVM_Results.csv")
            #If participant already has data in clean log
            if P.ID in old_log['Participant'].tolist():
                for key, value in calc.items():
                    old_log.loc[old_log['Participant']==P.ID, key] = value
                new_log = old_log
            else:
                new_log = pd.concat([old_log, pd.DataFrame([calc])], ignore_index=True)
            new_log.to_csv(f"{out_folder}EVM_Results.csv",index=False)
        else:
            EVM_Results = pd.DataFrame()
            EVM_Results = pd.concat([EVM_Results, pd.DataFrame([calc])], ignore_index=True)
            EVM_Results.to_csv(f"{out_folder}EVM_Results.csv",index=False)

    ### Update Log File

    log = {'Participant':P.ID,
            'Date':P.date,
            'Status':P.status,      
            'PsychoPy':P.isData,
            'PsychoPyComplete':P.isDataComplete,
            'Task':EVM_task,
            'Files':P.cntVGTFiles,
            'Placed':P.cntVGTPlaced,
            'Processed':P.cntVGTProcessed,
            'Errors':';'.join(errors)
            }

    if os.path.exists(f"{out_folder}psychoPy_log.csv"):
        old_log = pd.read_csv(f"{out_folder}psychoPy_log.csv")
        #If participant already has data in clean log
        if (P.ID in old_log['Participant'].tolist()) and (old_log.loc[old_log['Participant']==P.ID]['Task'].str.contains(EVM_task).any()):
            for key, value in log.items():
                old_log.loc[(old_log['Participant']==P.ID)&(old_log['Task'].str.contains(EVM_task)), key] = value
            new_log = old_log
        else:
            new_log = pd.concat([old_log, pd.DataFrame([log])], ignore_index=True)
        new_log.to_csv(f"{out_folder}psychoPy_log.csv",index=False)
    else:
        psychoPy_log = pd.DataFrame()
        psychoPy_log = pd.concat([psychoPy_log, pd.DataFrame([log])], ignore_index=True)
        psychoPy_log.to_csv(f"{out_folder}psychoPy_log.csv",index=False)
    
    #End

    #############################
    print_status(f"Collected {EVM_task}",f"{int(P.cntVGTProcessed/2*100)}%")  
    return P


def update_emotionRecognition(self):
    self.cntEmotionRecognitionFiles = 0
    self.cntEmotionRecognitionPlaced = 0
    self.cntEmotionRecognitionProcessed = 0


def get_emotionRecognition(in_folder, out_folder,P):
    '''
    This function cleans and combines all the OpenSignals data from one participant into a single file

    Parameters
    __________
    in_folder: str
        Folder containing directories for each participant
        Must contain ONLY particpant 
    out_folder: str
        Folder in which a directory will be created per participant
        
    TODO:
    - Add in check for duplicate participant IDs
    '''
    
    
    #Define variables
    psychoPy_log = pd.DataFrame()

    errors = []
    in_path = f"{in_folder}{P.ID}/"
    out_path = f"{out_folder}{P.ID}/"
    os.makedirs(out_path, exist_ok=True) 
    
    EVM_Results = pd.DataFrame()
    EVM_task = 'emotionRecognition'

    ##Reset Flags
    P.cntEmotionRecognitionFiles = 0
    P.cntEmotionRecognitionPlaced = 0
    P.cntEmotionRecognitionProcessed = 0
    
    taskData=pd.DataFrame()
    print_step(f'Collecting {EVM_task}',P.ID)
    ##### EmotionRecognition Data
    ##Extract data from files
    files = get_files(in_path,tags=['.csv', EVM_task]) #Get list of all goNogo tasks
    #For each file
    file_num = 0
    for f in files:                   
        try:
            path = in_path + f                 
            f=open(path)                    
            df = pd.read_csv(path, low_memory = False)
            df=df[['Emotion','Time','Response_Time','Choice']]
            df['Path']=path
            df['File']=file_num

            ##Adjust data
            #Replace entries
            df.replace('Fearful','Fear',inplace=True)
            df.loc[df['Emotion']==df['Choice'],'Accurate']=True
            df.loc[df['Emotion']!=df['Choice'],'Accurate']=False

            #create absolute time column
            df['Time']=df['Time'].str.replace(',','.')
            df['t']=[get_sec(i) for i in df['Time'].tolist()]

            ##Update Counters
            file_num = file_num+1
            print_status(f'Extracted {EVM_task}',path)

            ##Contextualise Data
            #set task from EVM list
            try:
                start_time = df['t'].tolist()[0]
                end_time = df['t'].tolist()[-1]
                task = P.tasksInfo.loc[(P.tasksInfo['Start']<=start_time) & (P.tasksInfo['End']>=end_time)]['Task'].tolist()[0]
                df['Task']=task
                print_status('Placed Data',task)

                #########################
                P.cntEmotionRecognitionPlaced += 1
            except IndexError:
                print_status('Unplaced Data',path)
            except Exception as z:
                print_status(f'Failed Placing {EVM_task} Data',f"{path} - {z!r}")
                errors = errors + [f"Failed Placing {EVM_task} Data: {path} - {z!r}",]
                          
            ##Update Participant Data
            taskData = pd.concat([taskData,df])

            #########################
            P.cntEmotionRecognitionFiles += 1
        
        ## Account for Issues
        except Exception as z:
            print_status(f'Failed {EVM_task}',f"{path} - {z!r}")
            errors = errors + [f"Failed {EVM_task}: {path} - {z!r}",]
            
    if P.cntEmotionRecognitionPlaced:
        ## Set condition
        taskData = taskData.dropna(subset=['Task'])
        taskData.loc[taskData['Task'].str.contains('_01'), 'Condition']='Baseline'
        taskData.loc[taskData['Task'].str.contains('_02'), 'Condition']='Treatment'

        ###Process GoNoGo Data 
        ## Initialise row entry
        calc = {'Participant':P.ID,
                'Drink':P.drink,}

        ## Results
        taskData = taskData.loc[taskData['Response_Time']>=0.2]

        if P.cntEmotionRecognitionPlaced:
            #for each condition
            conditions = taskData['Condition'].drop_duplicates()

            for condition in conditions:
                _condition = taskData.loc[taskData['Condition']==condition]

                #Update row entry
                _accuracy = sum(1 for item in _condition['Accurate'] if item is True)/len(_condition['Accurate'])
                _rt = _condition.loc[_condition['Response_Time']>=0.2]['Response_Time'].mean()
                _calc = {f"psy_{EVM_task}_{condition}_Overall_Accuracy":_accuracy,
                    f"psy_{EVM_task}_{condition}_Overall_RT":_rt,
                    }
                merge_dict(_calc,calc)

                #########################
                P.cntEmotionRecognitionProcessed += 1

                emotions = _condition['Emotion'].drop_duplicates()
                for emotion in emotions:
                    _emotion = _condition.loc[_condition['Emotion']==emotion]
                    _accuracy = sum(1 for item in _emotion['Accurate'] if item is True)/len(_emotion['Accurate'])
                    _rt = _emotion.loc[(_emotion['Response_Time']>=0.2)]['Response_Time'].mean()
                    _rtT = _emotion.loc[(_emotion['Response_Time']>=0.2)&(_emotion['Accurate']==True)]['Response_Time'].mean()
                    _rtF = _emotion.loc[(_emotion['Response_Time']>=0.2)&(_emotion['Accurate']==False)]['Response_Time'].mean()

                    _calc = {f"psy_{EVM_task}_{condition}_{emotion}_Accuracy":_accuracy,
                             f"psy_{EVM_task}_{condition}_{emotion}_RT":_rt,
                             f"psy_{EVM_task}_{condition}_{emotion}_RTAccurate":_rtT,
                             f"psy_{EVM_task}_{condition}_{emotion}_RTInaccurate":_rtF,
                            }
                    merge_dict(_calc,calc)
    
        ### Update Participant Results
        P.psychoPyResults = pd.concat([P.psychoPyResults, pd.DataFrame([calc])], ignore_index=True)
    
        ### Update EVM Results File
        if os.path.exists(f"{out_folder}EVM_Results.csv"):
            old_log = pd.read_csv(f"{out_folder}EVM_Results.csv")
            #If participant already has data in clean log
            if P.ID in old_log['Participant'].tolist():
                for key, value in calc.items():
                    old_log.loc[old_log['Participant']==P.ID, key] = value
                new_log = old_log
            else:
                new_log = pd.concat([old_log, pd.DataFrame([calc])], ignore_index=True)
            new_log.to_csv(f"{out_folder}EVM_Results.csv",index=False)
        else:
            EVM_Results = pd.DataFrame()
            EVM_Results = pd.concat([EVM_Results, pd.DataFrame([calc])], ignore_index=True)
            EVM_Results.to_csv(f"{out_folder}EVM_Results.csv",index=False)

    ### Update Log File

    log = {'Participant':P.ID,
            'Date':P.date,
            'Status':P.status,      
            'PsychoPy':P.isData,
            'PsychoPyComplete':P.isDataComplete,
            'Task':EVM_task,
            'Files':P.cntEmotionRecognitionFiles,
            'Placed':P.cntEmotionRecognitionPlaced,
            'Processed':P.cntEmotionRecognitionProcessed,
            'Errors':';'.join(errors)
            }

    if os.path.exists(f"{out_folder}psychoPy_log.csv"):
        old_log = pd.read_csv(f"{out_folder}psychoPy_log.csv")
        #If participant already has data in clean log
        if (P.ID in old_log['Participant'].tolist()) and (old_log.loc[old_log['Participant']==P.ID]['Task'].str.contains(EVM_task).any()):
            for key, value in log.items():
                old_log.loc[(old_log['Participant']==P.ID)&(old_log['Task'].str.contains(EVM_task)), key] = value
            new_log = old_log
        else:
            new_log = pd.concat([old_log, pd.DataFrame([log])], ignore_index=True)
        new_log.to_csv(f"{out_folder}psychoPy_log.csv",index=False)
    else:
        psychoPy_log = pd.DataFrame()
        psychoPy_log = pd.concat([psychoPy_log, pd.DataFrame([log])], ignore_index=True)
        psychoPy_log.to_csv(f"{out_folder}psychoPy_log.csv",index=False)
    
    #End

    #############################
    print_status(f"Collected {EVM_task}",f"{int(P.cntEmotionRecognitionProcessed/2*100)}%")  
    return P


def update_digitSpan(self):
    self.cntDigitSpanFiles = 0
    self.cntDigitSpanPlaced = 0
    self.cntDigitSpanProcessed = 0


def get_digitSpan(in_folder, out_folder,P):
    '''
    This function cleans and combines all the OpenSignals data from one participant into a single file

    Parameters
    __________
    in_folder: str
        Folder containing directories for each participant
        Must contain ONLY particpant 
    out_folder: str
        Folder in which a directory will be created per participant
        
    TODO:
    - Add in check for duplicate participant IDs
    '''
    
    
    #Define variables
    psychoPy_log = pd.DataFrame()

    errors = []
    in_path = f"{in_folder}{P.ID}/"
    out_path = f"{out_folder}{P.ID}/"
    os.makedirs(out_path, exist_ok=True) 
    
    EVM_Results = pd.DataFrame()
    EVM_task = 'digitSpan'

    ##Reset Flags
    P.cntDigitSpanFiles = 0
    P.cntDigitSpanPlaced = 0
    P.cntDigitSpanProcessed = 0
    
    taskData=pd.DataFrame()
    print_step(f'Collecting {EVM_task}',P.ID)
    ##### EmotionRecognition Data
    ##Extract data from files
    files = get_files(in_path,tags=['.csv', EVM_task]) #Get list of all goNogo tasks
    #For each file
    file_num = 0
    for f in files:                   
        try:
            path = in_path + f                 
            f=open(path)                    
            df = pd.read_csv(path, low_memory = False)
            df=df[['Accuracy','Time','Response_Time','Digit_Span']]
            df['Path']=path
            df['File']=file_num

            ##Adjust data
            df. rename(columns = {'Accuracy':'Accurate'}, inplace = True)
            #create absolute time column
            df['Time']=df['Time'].str.replace(',','.')
            df['t']=[get_sec(i) for i in df['Time'].tolist()]

            ##Update Counters
            file_num = file_num+1
            print_status(f'Extracted {EVM_task}',path)

            ##Contextualise Data
            #set task from EVM list
            try:
                start_time = df['t'].tolist()[0]
                end_time = df['t'].tolist()[-1]
                task = P.tasksInfo.loc[(P.tasksInfo['Start']<=start_time) & (P.tasksInfo['End']>=end_time)]['Task'].tolist()[0]
                df['Task']=task
                print_status('Placed Data',task)

                #########################
                P.cntDigitSpanPlaced += 1
            except IndexError:
                print_status('Unplaced Data',path)
            except Exception as z:
                print_status(f'Failed Placing {EVM_task} Data',f"{path} - {z!r}")
                errors = errors + [f"Failed Placing {EVM_task} Data: {path} - {z!r}",]
                          
            ##Update Participant Data
            taskData = pd.concat([taskData,df])

            #########################
            P.cntDigitSpanFiles += 1
        
        ## Account for Issues
        except Exception as z:
            print_status(f'Failed {EVM_task}',f"{path} - {z!r}")
            errors = errors + [f"Failed {EVM_task}: {path} - {z!r}",]
    
    if P.cntDigitSpanPlaced:
        ## Set condition
        taskData = taskData.dropna(subset=['Task'])
        taskData.loc[taskData['Task'].str.contains('_01'), 'Condition']='Baseline'
        taskData.loc[taskData['Task'].str.contains('_02'), 'Condition']='Treatment'

        ###Process DigitSpan Data 
        ## Initialise row entry
        calc = {'Participant':P.ID,
                'Drink':P.drink,}

        ## Results
        if P.cntDigitSpanPlaced:
            #for each condition
            conditions = taskData['Condition'].drop_duplicates()
            for condition in conditions:
                _condition = taskData.loc[taskData['Condition']==condition]

                #Update row entry
                _score = sum(1 for item in _condition['Accurate'] if item is True)
                _performance = _condition['Digit_Span']/_condition['Response_Time']
                _performance = _performance.mean()
            
                _calc = {f"psy_{EVM_task}_{condition}_Overall_Score":_score,
                    f"psy_{EVM_task}_{condition}_Overall_Performance":_performance,
                    }
                merge_dict(_calc,calc)

                #########################
                P.cntDigitSpanProcessed += 1
    
        ### Update Participant Results
        P.psychoPyResults = pd.concat([P.psychoPyResults, pd.DataFrame([calc])], ignore_index=True)
    
        ### Update EVM Results File
        if os.path.exists(f"{out_folder}EVM_Results.csv"):
            old_log = pd.read_csv(f"{out_folder}EVM_Results.csv")
            #If participant already has data in clean log
            if P.ID in old_log['Participant'].tolist():
                for key, value in calc.items():
                    old_log.loc[old_log['Participant']==P.ID, key] = value
                new_log = old_log
            else:
                new_log = pd.concat([old_log, pd.DataFrame([calc])], ignore_index=True)
            new_log.to_csv(f"{out_folder}EVM_Results.csv",index=False)
        else:
            EVM_Results = pd.DataFrame()
            EVM_Results = pd.concat([EVM_Results, pd.DataFrame([calc])], ignore_index=True)
            EVM_Results.to_csv(f"{out_folder}EVM_Results.csv",index=False)

    ### Update Log File

    log = {'Participant':P.ID,
            'Date':P.date,
            'Status':P.status,      
            'PsychoPy':P.isData,
            'PsychoPyComplete':P.isDataComplete,
            'Task':EVM_task,
            'Files':P.cntDigitSpanFiles,
            'Placed':P.cntDigitSpanPlaced,
            'Processed':P.cntDigitSpanProcessed,
            'Errors':';'.join(errors)
            }

    if os.path.exists(f"{out_folder}psychoPy_log.csv"):
        old_log = pd.read_csv(f"{out_folder}psychoPy_log.csv")
        #If participant already has data in clean log
        if (P.ID in old_log['Participant'].tolist()) and (old_log.loc[old_log['Participant']==P.ID]['Task'].str.contains(EVM_task).any()):
            for key, value in log.items():
                old_log.loc[(old_log['Participant']==P.ID)&(old_log['Task'].str.contains(EVM_task)), key] = value
            new_log = old_log
        else:
            new_log = pd.concat([old_log, pd.DataFrame([log])], ignore_index=True)
        new_log.to_csv(f"{out_folder}psychoPy_log.csv",index=False)
    else:
        psychoPy_log = pd.DataFrame()
        psychoPy_log = pd.concat([psychoPy_log, pd.DataFrame([log])], ignore_index=True)
        psychoPy_log.to_csv(f"{out_folder}psychoPy_log.csv",index=False)
    
    #End

    #############################
    print_status(f"Collected {EVM_task}",f"{int(P.cntDigitSpanProcessed/2*100)}%")  
    return P


def update_digitSymbol(self):
    self.cntDigitSymbolFiles = 0
    self.cntDigitSymbolPlaced = 0
    self.cntDigitSymbolProcessed = 0


def get_digitSymbol(in_folder, out_folder,P):
    '''
    This function cleans and combines all the OpenSignals data from one participant into a single file

    Parameters
    __________
    in_folder: str
        Folder containing directories for each participant
        Must contain ONLY particpant 
    out_folder: str
        Folder in which a directory will be created per participant
        
    TODO:
    - Add in check for duplicate participant IDs
    '''
    
    
    #Define variables
    psychoPy_log = pd.DataFrame()

    errors = []
    in_path = f"{in_folder}{P.ID}/"
    out_path = f"{out_folder}{P.ID}/"
    os.makedirs(out_path, exist_ok=True) 
    
    EVM_Results = pd.DataFrame()
    EVM_task = 'digitSymbol'

    ##Reset Flags
    P.cntDigitSymbolFiles = 0
    P.cntDigitSymbolPlaced = 0
    P.cntDigitSymbolProcessed = 0
    
    taskData=pd.DataFrame()
    print_step(f'Collecting {EVM_task}',P.ID)
    ##### EmotionRecognition Data
    ##Extract data from files
    files = get_files(in_path,tags=['.csv', EVM_task]) #Get list of all goNogo tasks
    #For each file
    file_num = 0
    for f in files:                   
        try:
            path = in_path + f                 
            f=open(path)                    
            df = pd.read_csv(path, low_memory = False)
            df=df[['Accuracy','Time','Response_Time','Correct_Key']]
            df['Path']=path
            df['File']=file_num

            ##Adjust data
            df. rename(columns = {'Accuracy':'Accurate'}, inplace = True)
            #create absolute time column
            df['Time']=df['Time'].str.replace(',','.')
            df['t']=[get_sec(i) for i in df['Time'].tolist()]

            ##Update Counters
            file_num = file_num+1
            print_status(f'Extracted {EVM_task}',path)

            ##Contextualise Data
            #set task from EVM list
            try:
                start_time = df['t'].tolist()[0]
                end_time = df['t'].tolist()[-1]
                task = P.tasksInfo.loc[(P.tasksInfo['Start']<=start_time) & (P.tasksInfo['End']>=end_time)]['Task'].tolist()[0]
                df['Task']=task
                print_status('Placed Data',task)

                #########################
                P.cntDigitSymbolPlaced += 1
            except IndexError:
                print_status('Unplaced Data',path)
            except Exception as z:
                print_status(f'Failed Placing {EVM_task} Data',f"{path} - {z!r}")
                errors = errors + [f"Failed Placing {EVM_task} Data: {path} - {z!r}",]
                          
            ##Update Participant Data
            taskData = pd.concat([taskData,df])

            #########################
            P.cntDigitSymbolFiles += 1
        
        ## Account for Issues
        except Exception as z:
            print_status(f'Failed {EVM_task}',f"{path} - {z!r}")
            errors = errors + [f"Failed {EVM_task}: {path} - {z!r}",]
            
    if P.cntDigitSymbolPlaced:
        ## Set condition
        taskData = taskData.dropna(subset=['Task'])
        taskData.loc[taskData['Task'].str.contains('_01'), 'Condition']='Baseline'
        taskData.loc[taskData['Task'].str.contains('_02'), 'Condition']='Treatment'

        ###Process GoNoGo Data 
        ## Initialise row entry
        calc = {'Participant':P.ID,
                'Drink':P.drink,}

        ## Results
        if P.cntDigitSymbolPlaced:
            #for each condition
            conditions = taskData['Condition'].drop_duplicates()
            for condition in conditions:
                _condition = taskData.loc[taskData['Condition']==condition]
                #Update row entry
                _accuracy = sum(1 for item in _condition['Accurate'] if item == 1)/len(_condition['Accurate'])
                _score = sum(1 for item in _condition['Accurate'] if item == 1)
                _performanceData = _condition[pd.to_numeric(_condition['Response_Time'], errors='coerce').notnull()]

                #Remove consecutive repeats
                _performanceData.insert(0,'Mask',_performanceData['Correct_Key'].shift(1) == _performanceData['Correct_Key'])
                while _performanceData['Mask'].any():
                    _performanceData = _performanceData.loc[_performanceData['Correct_Key'].shift(1) != _performanceData['Correct_Key']]
                    _performanceData['Mask']=_performanceData['Correct_Key'].shift(1) == _performanceData['Correct_Key']

                _performance = _performanceData['Accurate']/_performanceData['Response_Time'].astype(float)
                _performance = _performance.mean()
            
                _calc = {f"psy_{EVM_task}_{condition}_Overall_Accuracy":_accuracy,
                         f"psy_{EVM_task}_{condition}_Overall_Score":_score,
                        f"psy_{EVM_task}_{condition}_Overall_Performance":_performance,
                        }
                merge_dict(_calc,calc)

                #########################
                P.cntDigitSymbolProcessed += 1
    
        ### Update Participant Results
        P.psychoPyResults = pd.concat([P.psychoPyResults, pd.DataFrame([calc])], ignore_index=True)
    
        ### Update EVM Results File
        if os.path.exists(f"{out_folder}EVM_Results.csv"):
            old_log = pd.read_csv(f"{out_folder}EVM_Results.csv")
            #If participant already has data in clean log
            if P.ID in old_log['Participant'].tolist():
                for key, value in calc.items():
                    old_log.loc[old_log['Participant']==P.ID, key] = value
                new_log = old_log
            else:
                new_log = pd.concat([old_log, pd.DataFrame([calc])], ignore_index=True)
            new_log.to_csv(f"{out_folder}EVM_Results.csv",index=False)
        else:
            EVM_Results = pd.DataFrame()
            EVM_Results = pd.concat([EVM_Results, pd.DataFrame([calc])], ignore_index=True)
            EVM_Results.to_csv(f"{out_folder}EVM_Results.csv",index=False)

    ### Update Log File

    log = {'Participant':P.ID,
            'Date':P.date,
            'Status':P.status,      
            'PsychoPy':P.isData,
            'PsychoPyComplete':P.isDataComplete,
            'Task':EVM_task,
            'Files':P.cntDigitSymbolFiles,
            'Placed':P.cntDigitSymbolPlaced,
            'Processed':P.cntDigitSymbolProcessed,
            'Errors':';'.join(errors)
            }

    if os.path.exists(f"{out_folder}psychoPy_log.csv"):
        old_log = pd.read_csv(f"{out_folder}psychoPy_log.csv")
        #If participant already has data in clean log
        if (P.ID in old_log['Participant'].tolist()) and (old_log.loc[old_log['Participant']==P.ID]['Task'].str.contains(EVM_task).any()):
            for key, value in log.items():
                old_log.loc[(old_log['Participant']==P.ID)&(old_log['Task'].str.contains(EVM_task)), key] = value
            new_log = old_log
        else:
            new_log = pd.concat([old_log, pd.DataFrame([log])], ignore_index=True)
        new_log.to_csv(f"{out_folder}psychoPy_log.csv",index=False)
    else:
        psychoPy_log = pd.DataFrame()
        psychoPy_log = pd.concat([psychoPy_log, pd.DataFrame([log])], ignore_index=True)
        psychoPy_log.to_csv(f"{out_folder}psychoPy_log.csv",index=False)
    
    #End

    #############################
    print_status(f"Collected {EVM_task}",f"{int(P.cntDigitSymbolProcessed/2*100)}%")  
    return P


def update_memoryEncoding(self):
    self.cntMemoryEncodingFiles = 0
    self.cntMemoryEncodingPlaced = 0
    self.cntMemoryEncodingProcessed = 0


def get_memoryEncoding(in_folder, out_folder,P):
    '''
    This function cleans and combines all the OpenSignals data from one participant into a single file

    Parameters
    __________
    in_folder: str
        Folder containing directories for each participant
        Must contain ONLY particpant 
    out_folder: str
        Folder in which a directory will be created per participant
        
    TODO:
    - Add in check for duplicate participant IDs
    '''
    
    
    #Define variables
    psychoPy_log = pd.DataFrame()

    errors = []
    in_path = f"{in_folder}{P.ID}/"
    out_path = f"{out_folder}{P.ID}/"
    os.makedirs(out_path, exist_ok=True) 
    
    EVM_Results = pd.DataFrame()
    EVM_task = 'memoryEncoding'

    ##Reset Flags
    P.cntMemoryEncodingFiles = 0
    P.cntMemoryEncodingPlaced = 0
    P.cntMemoryEncodingProcessed = 0
    
    taskData=pd.DataFrame()
    print_step(f'Collecting {EVM_task}',P.ID)
    ##### EmotionRecognition Data
    ##Extract data from files
    files = get_files(in_path,tags=['.csv', EVM_task]) #Get list of all goNogo tasks
    #For each file
    file_num = 0
    for f in files:                   
        try:
            path = in_path + f                 
            f=open(path)                    
            df = pd.read_csv(path, low_memory = False)
            df=df[['Accuracy','Time','Response']]
            df['Path']=path
            df['File']=file_num

            ##Adjust data
            df. rename(columns = {'Accuracy':'Accurate'}, inplace = True)
            #create absolute time column
            df['Time']=df['Time'].str.replace(',','.')
            df['t']=[get_sec(i) for i in df['Time'].tolist()]

            ##Update Counters
            file_num = file_num+1
            print_status(f'Extracted {EVM_task}',path)

            ##Contextualise Data
            #set task from EVM list
            try:
                start_time = df['t'].tolist()[0]
                end_time = df['t'].tolist()[-1]
                task = P.tasksInfo.loc[(P.tasksInfo['Start']<=start_time) & (P.tasksInfo['End']>=end_time)]['Task'].tolist()[0]
                df['Task']=task
                print_status('Placed Data',task)

                #########################
                P.cntMemoryEncodingPlaced += 1
            except IndexError:
                print_status('Unplaced Data',path)
            except Exception as z:
                print_status(f'Failed Placing {EVM_task} Data',f"{path} - {z!r}")
                errors = errors + [f"Failed Placing {EVM_task} Data: {path} - {z!r}",]
                          
            ##Update Participant Data
            taskData = pd.concat([taskData,df])

            #########################
            P.cntMemoryEncodingFiles += 1
        
        ## Account for Issues
        except Exception as z:
            print_status(f'Failed {EVM_task}',f"{path} - {z!r}")
            errors = errors + [f"Failed {EVM_task}: {path} - {z!r}",]
            
    if P.cntMemoryEncodingPlaced:
        ## Set condition
        taskData = taskData.dropna(subset=['Task'])
        taskData.loc[taskData['Task'].str.contains('_01'), 'Condition']='Baseline'
        taskData.loc[taskData['Task'].str.contains('_02'), 'Condition']='Treatment'

        ###Process GoNoGo Data 
        ## Initialise row entry
        calc = {'Participant':P.ID,
                'Drink':P.drink,}

        ## Results
        if P.cntMemoryEncodingPlaced:
            #for each condition
            conditions = taskData['Condition'].drop_duplicates()
            for condition in conditions:
                _condition = taskData.loc[taskData['Condition']==condition]
                #Update row entry
                if len(_condition['Accurate'])>1:
                    _score = sum(1 for item in _condition['Accurate'] if item is True)

                    _calc = {f"psy_{EVM_task}_{condition}_Overall_Score":_score,
                            }
                else:
                    _calc = {f"psy_{EVM_task}_{condition}_Overall_Score":np.nan,
                            }
                merge_dict(_calc,calc)

                #########################
                P.cntMemoryEncodingProcessed += 1
    
        ### Update Participant Results
        P.psychoPyResults = pd.concat([P.psychoPyResults, pd.DataFrame([calc])], ignore_index=True)
    
        ### Update EVM Results File
        if os.path.exists(f"{out_folder}EVM_Results.csv"):
            old_log = pd.read_csv(f"{out_folder}EVM_Results.csv")
            #If participant already has data in clean log
            if P.ID in old_log['Participant'].tolist():
                for key, value in calc.items():
                    old_log.loc[old_log['Participant']==P.ID, key] = value
                new_log = old_log
            else:
                new_log = pd.concat([old_log, pd.DataFrame([calc])], ignore_index=True)
            new_log.to_csv(f"{out_folder}EVM_Results.csv",index=False)
        else:
            EVM_Results = pd.DataFrame()
            EVM_Results = pd.concat([EVM_Results, pd.DataFrame([calc])], ignore_index=True)
            EVM_Results.to_csv(f"{out_folder}EVM_Results.csv",index=False)

        ### Update Log File

    log = {'Participant':P.ID,
            'Date':P.date,
            'Status':P.status,      
            'PsychoPy':P.isData,
            'PsychoPyComplete':P.isDataComplete,
            'Task':EVM_task,
            'Files':P.cntMemoryEncodingFiles,
            'Placed':P.cntMemoryEncodingPlaced,
            'Processed':P.cntMemoryEncodingProcessed,
            'Errors':';'.join(errors)
            }

    if os.path.exists(f"{out_folder}psychoPy_log.csv"):
        old_log = pd.read_csv(f"{out_folder}psychoPy_log.csv")
        #If participant already has data in clean log
        if (P.ID in old_log['Participant'].tolist()) and (old_log.loc[old_log['Participant']==P.ID]['Task'].str.contains(EVM_task).any()):
            for key, value in log.items():
                old_log.loc[(old_log['Participant']==P.ID)&(old_log['Task'].str.contains(EVM_task)), key] = value
            new_log = old_log
        else:
            new_log = pd.concat([old_log, pd.DataFrame([log])], ignore_index=True)
        new_log.to_csv(f"{out_folder}psychoPy_log.csv",index=False)
    else:
        psychoPy_log = pd.DataFrame()
        psychoPy_log = pd.concat([psychoPy_log, pd.DataFrame([log])], ignore_index=True)
        psychoPy_log.to_csv(f"{out_folder}psychoPy_log.csv",index=False)
    
    #End

    #############################
    print_status(f"Collected {EVM_task}",f"{int(P.cntMemoryEncodingProcessed/2*100)}%")  
    return P


def update_recall(self):
    self.cntRecallFiles = 0
    self.cntRecallPlaced = 0
    self.cntRecallProcessed = 0


def get_recall(in_folder, out_folder,P):
    '''
    This function cleans and combines all the OpenSignals data from one participant into a single file

    Parameters
    __________
    in_folder: str
        Folder containing directories for each participant
        Must contain ONLY particpant 
    out_folder: str
        Folder in which a directory will be created per participant
        
    TODO:
    - Add in check for duplicate participant IDs
    '''
    
    
    #Define variables
    psychoPy_log = pd.DataFrame()

    errors = []
    in_path = f"{in_folder}{P.ID}/"
    out_path = f"{out_folder}{P.ID}/"
    os.makedirs(out_path, exist_ok=True) 
    
    EVM_Results = pd.DataFrame()
    EVM_task = 'recall'

    ##Reset Flags
    P.cntRecallFiles = 0
    P.cntRecallPlaced = 0
    P.cntRecallProcessed = 0
    
    taskData=pd.DataFrame()
    print_step(f'Collecting {EVM_task}',P.ID)
    ##### EmotionRecognition Data
    ##Extract data from files
    files = get_files(in_path,tags=['.csv', EVM_task]) #Get list of all goNogo tasks
    #For each file
    file_num = 0
    for f in files:                   
        try:
            path = in_path + f                 
            f=open(path)                    
            df = pd.read_csv(path, low_memory = False)
            df=df[['Accuracy','Time']]
            df['Path']=path
            df['File']=file_num

            ##Adjust data
            df. rename(columns = {'Accuracy':'Accurate'}, inplace = True)
            #create absolute time column
            df['Time']=df['Time'].str.replace(',','.')
            df['t']=[get_sec(i) for i in df['Time'].tolist()]

            ##Update Counters
            file_num = file_num+1
            print_status(f'Extracted {EVM_task}',path)

            ##Contextualise Data
            #set task from EVM list
            try:
                start_time = df['t'].tolist()[0]
                end_time = df['t'].tolist()[-1]
                task = P.tasksInfo.loc[(P.tasksInfo['Start']<=start_time) & (P.tasksInfo['End']>=end_time)]['Task'].tolist()[0]
                df['Task']=task
                print_status('Placed Data',task)

                #########################
                P.cntRecallPlaced += 1
            except IndexError:
                print_status('Unplaced Data',path)
            except Exception as z:
                print_status(f'Failed Placing {EVM_task} Data',f"{path} - {z!r}")
                errors = errors + [f"Failed Placing {EVM_task} Data: {path} - {z!r}",]
                          
            ##Update Participant Data
            taskData = pd.concat([taskData,df])

            #########################
            P.cntRecallFiles += 1
        
        ## Account for Issues
        except Exception as z:
            print_status(f'Failed {EVM_task}',f"{path} - {z!r}")
            errors = errors + [f"Failed {EVM_task}: {path} - {z!r}",]
            
    if P.cntRecallPlaced:
        ## Set condition
        taskData = taskData.dropna(subset=['Task'])
        taskData.loc[taskData['Task'].str.contains('_01'), 'Condition']='Baseline'
        taskData.loc[taskData['Task'].str.contains('_02'), 'Condition']='Treatment'

        ###Process GoNoGo Data 
        ## Initialise row entry
        calc = {'Participant':P.ID,
                'Drink':P.drink,}

        ## Results
        if P.cntRecallPlaced:
            #for each condition
            conditions = taskData['Condition'].drop_duplicates()
            for condition in conditions:
                _condition = taskData.loc[taskData['Condition']==condition]
                #Update row entry
                if len(_condition['Accurate'])>1:
                    _score = sum(1 for item in _condition['Accurate'] if item is True)

                    _calc = {f"psy_{EVM_task}_{condition}_Overall_Score":_score,
                            }
                else:
                    _calc = {f"psy_{EVM_task}_{condition}_Overall_Score":np.nan,
                            }
                merge_dict(_calc,calc)

                #########################
                P.cntRecallProcessed += 1
    
        ### Update Participant Results
        P.psychoPyResults = pd.concat([P.psychoPyResults, pd.DataFrame([calc])], ignore_index=True)
    
        ### Update EVM Results File
        if os.path.exists(f"{out_folder}EVM_Results.csv"):
            old_log = pd.read_csv(f"{out_folder}EVM_Results.csv")
            #If participant already has data in clean log
            if P.ID in old_log['Participant'].tolist():
                for key, value in calc.items():
                    old_log.loc[old_log['Participant']==P.ID, key] = value
                new_log = old_log
            else:
                new_log = pd.concat([old_log, pd.DataFrame([calc])], ignore_index=True)
            new_log.to_csv(f"{out_folder}EVM_Results.csv",index=False)
        else:
            EVM_Results = pd.DataFrame()
            EVM_Results = pd.concat([EVM_Results, pd.DataFrame([calc])], ignore_index=True)
            EVM_Results.to_csv(f"{out_folder}EVM_Results.csv",index=False)

    ### Update Log File

    log = {'Participant':P.ID,
            'Date':P.date,
            'Status':P.status,      
            'PsychoPy':P.isData,
            'PsychoPyComplete':P.isDataComplete,
            'Task':EVM_task,
            'Files':P.cntRecallFiles,
            'Placed':P.cntRecallPlaced,
            'Processed':P.cntRecallProcessed,
            'Errors':';'.join(errors)
            }

    if os.path.exists(f"{out_folder}psychoPy_log.csv"):
        old_log = pd.read_csv(f"{out_folder}psychoPy_log.csv")
        #If participant already has data in clean log
        if (P.ID in old_log['Participant'].tolist()) and (old_log.loc[old_log['Participant']==P.ID]['Task'].str.contains(EVM_task).any()):
            for key, value in log.items():
                old_log.loc[(old_log['Participant']==P.ID)&(old_log['Task'].str.contains(EVM_task)), key] = value
            new_log = old_log
        else:
            new_log = pd.concat([old_log, pd.DataFrame([log])], ignore_index=True)
        new_log.to_csv(f"{out_folder}psychoPy_log.csv",index=False)
    else:
        psychoPy_log = pd.DataFrame()
        psychoPy_log = pd.concat([psychoPy_log, pd.DataFrame([log])], ignore_index=True)
        psychoPy_log.to_csv(f"{out_folder}psychoPy_log.csv",index=False)
    
    #End

    #############################
    print_status(f"Collected {EVM_task}",f"{int(P.cntRecallProcessed*100)}%")  
    return P


def update_step(self):
    self.cntStepFiles = 0
    self.cntStepPlaced = 0
    self.cntStepProcessed = 0


def get_step(in_folder, out_folder,P):
    '''
    This function cleans and combines all the OpenSignals data from one participant into a single file

    Parameters
    __________
    in_folder: str
        Folder containing directories for each participant
        Must contain ONLY particpant 
    out_folder: str
        Folder in which a directory will be created per participant
        
    TODO:
    - Add in check for duplicate participant IDs
    '''
    
    
    #Define variables
    psychoPy_log = pd.DataFrame()

    errors = []
    in_path = f"{in_folder}{P.ID}/"
    out_path = f"{out_folder}{P.ID}/"
    os.makedirs(out_path, exist_ok=True) 
    
    EVM_Results = pd.DataFrame()
    EVM_task = 'step'

    ##Reset Flags
    P.cntStepFiles = 0
    P.cntStepPlaced = 0
    P.cntStepProcessed = 0
    
    taskData=pd.DataFrame()
    print_step(f'Collecting {EVM_task}',P.ID)
    ##### EmotionRecognition Data
    ##Extract data from files
    files = get_files(in_path,tags=['.csv', EVM_task,]) #Get list of all goNogo tasks
    files = files + get_files(in_path,tags=['.csv', 'manual',])

    #For each file
    file_num = 0
    for f in files:                   
        try:
            path = in_path + f                 
            f=open(path)                    
            df = pd.read_csv(path, low_memory = False)
            df=df[['Time', 'Event', 'Systolic', 'Diastolic']]
            df['Path']=path
            df['File']=file_num

            #create absolute time column
            df['Time']=df['Time'].str.replace(',','.')
            df['t']=[get_sec(i) for i in df['Time'].tolist()]

            ##Update Counters
            file_num = file_num+1
            print_status(f'Extracted {EVM_task}',path)

            ##Contextualise Data
            #set task from EVM list
            try:
                start_time = df['t'].tolist()[0]
                end_time = df['t'].tolist()[-1]
                task = P.tasksInfo.loc[(P.tasksInfo['Start']<=start_time) & (P.tasksInfo['End']>=end_time)]['Task'].tolist()[0]
                df['Task']=task
                print_status('Placed Data',task)

                #########################
                P.cntStepPlaced += 1
            except IndexError:
                print_status('Unplaced Data',path)
            except Exception as z:
                print_status(f'Failed Placing {EVM_task} Data',f"{path} - {z!r}")
                errors = errors + [f"Failed Placing {EVM_task} Data: {path} - {z!r}",]
                          
            ##Update Participant Data
            taskData = pd.concat([taskData,df])

            #########################
            P.cntStepFiles += 1
        
        ## Account for Issues
        except KeyError:
            pass
        except Exception as z:
            print_status(f'Failed {EVM_task}',f"{path} - {z!r}")
            errors = errors + [f"Failed {EVM_task}: {path} - {z!r}",]
    
    if P.cntStepPlaced:
        ## Adjust data
        first_start = taskData.loc[(taskData['Task']=='step_01')]['t'].min()
        second_start = taskData.loc[(taskData['Task']=='step_02')]['t'].min()
        
        taskData.loc[(taskData['Event']=='BP')&(taskData['t']<first_start)&(taskData['t']>(first_start-600)),'Task']='step_01'
        taskData.loc[(taskData['Event']=='BP')&(taskData['t']<second_start)&(taskData['t']>(second_start-600)),'Task']='step_02'
        
        taskData = taskData.loc[(taskData['Systolic']<900) & (taskData['Diastolic']<900)]

        ## Set condition
        taskData = taskData.dropna(subset=['Task'])
        taskData.loc[taskData['Task'].str.contains('_01'), 'Condition']='Baseline'
        taskData.loc[taskData['Task'].str.contains('_02'), 'Condition']='Treatment'

        ###Process GoNoGo Data 
        ## Initialise row entry
        calc = {'Participant':P.ID,
                'Drink':P.drink,}

        ## Results
        if P.cntStepPlaced:
            #for each condition
            conditions = taskData['Condition'].drop_duplicates()
            for condition in conditions:
                _condition = taskData.loc[taskData['Condition']==condition]
                metrics = ['Systolic','Diastolic']
                #Update row entry
                for m in metrics:
                    _data = _condition[['t',m]].dropna().sort_values('t')
                    _time = _data['t'].tolist()
                    _bp = _data[m].tolist()

                    if len(_data)==3:
                        _response = (_bp[1]-_bp[0])#/_bp[0])*100
                        try:
                            _recovery = (_bp[1]-_bp[2])#/(_bp[1]-_bp[0]))*100
                        except ZeroDivisionError:
                            _recovery = np.nan
                        _rate = (_bp[1]-_bp[2])/(_time[2]-_time[1])

                        _calc = {f"psy_{EVM_task}_{condition}_{m}_Response":_response,
                                 f"psy_{EVM_task}_{condition}_{m}_Recovery":_recovery,
                                 f"psy_{EVM_task}_{condition}_{m}_Rate":_rate
                                }
                        merge_dict(_calc,calc)
                    else:
                        print_status(f'Failed Step Results',f"{condition} - {m}")
                        errors = errors + [f'Failed Step Results {condition} - {m}',] 

                #########################
                P.cntStepProcessed += 1
    
        ### Update Participant Results
        P.psychoPyResults = pd.concat([P.psychoPyResults, pd.DataFrame([calc])], ignore_index=True)
    
        ### Update EVM Results File
        if os.path.exists(f"{out_folder}EVM_Results.csv"):
            old_log = pd.read_csv(f"{out_folder}EVM_Results.csv")
            #If participant already has data in clean log
            if P.ID in old_log['Participant'].tolist():
                for key, value in calc.items():
                    old_log.loc[old_log['Participant']==P.ID, key] = value
                new_log = old_log
            else:
                new_log = pd.concat([old_log, pd.DataFrame([calc])], ignore_index=True)
            new_log.to_csv(f"{out_folder}EVM_Results.csv",index=False)
        else:
            EVM_Results = pd.DataFrame()
            EVM_Results = pd.concat([EVM_Results, pd.DataFrame([calc])], ignore_index=True)
            EVM_Results.to_csv(f"{out_folder}EVM_Results.csv",index=False)

    ### Update Log File

    log = {'Participant':P.ID,
            'Date':P.date,
            'Status':P.status,      
            'PsychoPy':P.isData,
            'PsychoPyComplete':P.isDataComplete,
            'Task':EVM_task,
            'Files':P.cntStepFiles,
            'Placed':P.cntStepPlaced,
            'Processed':P.cntStepProcessed,
            'Errors':';'.join(errors)
            }

    if os.path.exists(f"{out_folder}psychoPy_log.csv"):
        old_log = pd.read_csv(f"{out_folder}psychoPy_log.csv")
        #If participant already has data in clean log
        if (P.ID in old_log['Participant'].tolist()) and (old_log.loc[old_log['Participant']==P.ID]['Task'].str.contains(EVM_task).any()):
            for key, value in log.items():
                old_log.loc[(old_log['Participant']==P.ID)&(old_log['Task'].str.contains(EVM_task)), key] = value
            new_log = old_log
        else:
            new_log = pd.concat([old_log, pd.DataFrame([log])], ignore_index=True)
        new_log.to_csv(f"{out_folder}psychoPy_log.csv",index=False)
    else:
        psychoPy_log = pd.DataFrame()
        psychoPy_log = pd.concat([psychoPy_log, pd.DataFrame([log])], ignore_index=True)
        psychoPy_log.to_csv(f"{out_folder}psychoPy_log.csv",index=False)
    
    #End

    #############################
    print_status(f"Collected {EVM_task}",f"{int(P.cntStepProcessed/2*100)}%")  
    return P


def update_MAS(self):
    self.cntMASFiles = 0
    self.cntMASPlaced = 0
    self.cntMASProcessed = 0


def get_MAS(in_folder, out_folder,P):
    '''
    This function cleans and combines all the OpenSignals data from one participant into a single file

    Parameters
    __________
    in_folder: str
        Folder containing directories for each participant
        Must contain ONLY particpant 
    out_folder: str
        Folder in which a directory will be created per participant
        
    TODO:
    - Add in check for duplicate participant IDs
    '''
    
    
    #Define variables
    psychoPy_log = pd.DataFrame()

    errors = []
    in_path = f"{in_folder}{P.ID}/"
    out_path = f"{out_folder}{P.ID}/"
    os.makedirs(out_path, exist_ok=True) 
    
    EVM_Results = pd.DataFrame()
    EVM_task = 'MAS'

    MAS_polarity = {
        'I understand the instructions':'P',
        'I feel calm':'N',
        'I feel alert':'P',
        'My senses feel heightened':'P',
        'I feel relaxed':'N',
        'I feel stressed':'P',
        'I cannot think clearly':'N',
        'I feel tense':'P',
        'I feel nervous':'P',
        'I feel energetic':'P',
        'I feel awake':'P',
        'I feel motivated':'P',
        'I feel enthusiastic':'P',
        'I can think easily':'P',
        'My mind feels slow':'N',
        'I feel mentally sharp':'P',
        'My thoughts are clear':'P',
        'I need silence to think':'P',
        'My mind is wandering':'P',
        'I feel focused':'P',
        'I am able to concentrate':'P',
        'I feel sociable':'P',
        'I want to be in the company of others':'P',
        'I feel shy':'P',
        'I feel withdrawn':'N',
        'I feel impulsive':'P',
        'I feel cautious':'P',
        'I am craving new sensations':'P',
        'I feel self concious':'P',
    }

    #MAS_axes = {
    #    'I understand the instructions':'None',
    #    'I feel calm':'Autonomic Arousal',
    #    'I feel alert':'Autonomic Arousal',
    #    'My senses feel heightened':'Autonomic Arousal',
    #    'I feel relaxed':'Autonomic Arousal',
    #    'I feel stressed':'HPA',
    #    'I cannot think clearly':'HPA',
    #    'I feel tense':'HPA',
    #    'I feel nervous':'HPA',
    #    'I feel energetic':'Energetic Arousal',
    #    'I feel awake':'Energetic Arousal',
    #    'I feel motivated':'Energetic Arousal',
    #    'I feel enthusiastic':'Energetic Arousal',
    #    'I can think easily':'Cognitive Performance',
    #    'My mind feels slow':'Cognitive Performance',
    #    'I feel mentally sharp':'Cognitive Performance',
    #    'My thoughts are clear':'Cognitive Performance',
    #    'I need silence to think':'Selective Attention',
    #    'My mind is wandering':'Selective Attention',
    #    'I feel focused':'Selective Attention',
    #    'I am able to concentrate':'Selective Attention',
    #    'I feel sociable':'Social Cognition and Affiliation',
    #    'I want to be in the company of others':'Social Cognition and Affiliation',
    #    'I feel shy':'Social Cognition and Affiliation',
    #    'I feel withdrawn':'Social Cognition and Affiliation',
    #    'I feel impulsive':'Behavioral Activation',
    #    'I feel cautious':'Behavioral Activation',
    #    'I am craving new sensations':'Behavioral Activation',
    #    'I feel self concious':'Behavioral Activation',
    #}

    MAS_Factors = {
        'I understand the instructions':'NA',
        'My mind feels slow':'F1',
        'I cannot think clearly':'F1',
        'I feel enthusiastic':'F3',
        'I feel energetic':'NA',
        'I feel motivated':'F1',
        'My thoughts are clear':'F1',
        'I feel focused':'F1',
        'I can think easily':'F1',
        'I feel alert':'F1',
        'I feel mentally sharp':'F1',
        'I feel awake':'F1',
        'I am able to concentrate':'F1',
        'I feel withdrawn':'NA',
        'I feel shy':'F2',
        'I feel nervous':'F2',
        'I feel stressed':'F2',
        'I feel tense':'F2',
        'I am craving new sensations':'F3',
        'I want to be in the company of others':'F3',
        'I feel sociable':'F3',
        'I feel relaxed':'F2',
        'I feel calm':'F2',
        'I feel cautious':'NA',
        'I feel self concious':'F2',
        'My mind is wandering':'NA',
        'I need silence to think':'NA',
        'I feel impulsive':'NA',
        'My senses feel heightened':'NA',
        }

    MAS_Loading = {
        'I understand the instructions':0,
        'My mind feels slow':0.640738183734838,
        'I cannot think clearly':0.614194191226753,
        'I feel withdrawn':0.418655754030587,
        'I feel sociable':0.604890937790026,
        'I feel enthusiastic':0.441340344366177,
        'I feel energetic':0.668010729058796,
        'My thoughts are clear':0.694840773342819,
        'I feel motivated':0.70313089800611,
        'I can think easily':0.749312403351462,
        'I feel focused':0.757461382410167,
        'I am able to concentrate':0.780695123500412,
        'I feel alert':0.790317065309408,
        'I feel mentally sharp':0.804525609713284,
        'I feel awake':0.813381401481584,
        'I feel relaxed':0.526774553799681,
        'I feel calm':0.492209125708346,
        'My mind is wandering':0.411796556682625,
        'I feel self concious':0.439091458813101,
        'I feel shy':0.608440970627443,
        'I feel stressed':0.683537095926097,
        'I feel tense':0.731823930882015,
        'I feel nervous':0.762158565022562,
        'I am craving new sensations':0.487996116405164,
        'I want to be in the company of others':0.504896705068384,
        'I need silence to think':0,
        'I feel cautious':0,
        'I feel impulsive':0,
        'My senses feel heightened':0,

        }

    factor_total = {
        'F1':84.35263514,
        'F2':46.55832257,
        'F3':20.39124104,
        'NA':1
        }

    ##Reset Flags
    P.cntMASFiles = 0
    P.cntMASPlaced = 0
    P.cntMASProcessed = 0
    
    taskData=pd.DataFrame()
    print_step(f'Collecting {EVM_task}',P.ID)
    ##### MAS Data
    ##Extract data from files
    files = get_files(in_path,tags=['.csv', EVM_task]) #Get list of all goNogo tasks
    #For each file
    file_num = 0
    for f in files:                   
        try:
            path = in_path + f                 
            f=open(path)                    
            df = pd.read_csv(path, low_memory = False)
            df=df[['Time','Question','Response']]
            df['Path']=path
            df['File']=file_num

            ##Adjust data
            for q in df['Question']:
                df.loc[df['Question'] == q, ['Polarity','Factor','Loading']] = MAS_polarity[q], MAS_Factors[q], MAS_Loading[q]

            #create absolute time column
            df['Time']=df['Time'].str.replace(',','.')
            df['t']=[get_sec(i) for i in df['Time'].tolist()]

            ##Update Counters
            file_num = file_num+1
            print_status(f'Extracted {EVM_task}',path)

            ##Contextualise Data
            #set task from EVM list
            try:
                start_time = df['t'].tolist()[0]
                end_time = df['t'].tolist()[-1]
                task = P.tasksInfo.loc[(P.tasksInfo['Start']<=start_time) & (P.tasksInfo['End']>=end_time)]['Task'].tolist()[0]
                df['Task']=task
                print_status('Placed Data',task)

                #########################
                P.cntMASPlaced += 1
            except IndexError:
                print_status('Unplaced Data',path)
            except Exception as z:
                print_status(f'Failed Placing {EVM_task} Data',f"{path} - {z!r}")
                errors = errors + [f"Failed Placing {EVM_task} Data: {path} - {z!r}",]
                          
            ##Update Participant Data
            taskData = pd.concat([taskData,df])

            #########################
            P.cntMASFiles += 1
        
        ## Account for Issues
        except Exception as z:
            print_status(f'Failed {EVM_task}',f"{path} - {z!r}")
            errors = errors + [f"Failed {EVM_task}: {path} - {z!r}",]
            
    if P.cntMASPlaced:
        ## Set condition
        taskData = taskData.dropna(subset=['Task'])
        taskData = taskData.dropna(subset=['Response',])
        taskData.loc[taskData['Task'].str.contains('_01'), 'Condition']='Baseline'
        taskData.loc[taskData['Task'].str.contains('_02'), 'Condition']='TreatmentTasksStart'
        taskData.loc[taskData['Task'].str.contains('_03'), 'Condition']='TreatmentTasksEnd'

        taskData.loc[taskData['Polarity']=='N','Corrected Score']=10-taskData.loc[taskData['Polarity']=='N']['Response']
        taskData.loc[taskData['Polarity']=='P','Corrected Score']=taskData.loc[taskData['Polarity']=='P']['Response']
        taskData['Weighted']=taskData['Corrected Score']*taskData['Loading']

        #print( taskData.loc[ taskData['Condition']=='Baseline'][['Question','Weighted','Factor']])
        ###Process GoNoGo Data 
        ## Initialise row entry
        calc = {'Participant':P.ID,
                'Drink':P.drink,}

        #Process MAS Data
        #for each condition
        conditions = taskData['Condition'].drop_duplicates()
        for condition in conditions:
            _condition = taskData.loc[taskData['Condition']==condition]
            #Update row entry
            questions = _condition['Question'].drop_duplicates()
            #axes = list(_condition['Axis'].drop_duplicates())
            factors = list(_condition['Factor'].drop_duplicates())
            factors = [f for f in factors if 'N' not in f]

            for q in questions:
                #_val = _condition.loc[_condition['Question']==q]['Corrected Score'].tolist()[0]
                _val = _condition.loc[_condition['Question']==q]['Response'].tolist()[0]
                
                try:
                    _calc= {f'psy_{EVM_task}_{condition}_{q}':_val,}
                    merge_dict(_calc,calc)
                except Exception as z:
                    print_status(f'Error {EVM_task} on {q}',f"{path} - {z!r}")
                    errors = errors + [f"Failed {EVM_task} Question {q}: {path} - {z!r}",]
            #for a in axes:
            #    #_calc= {f'psy_{EVM_task}_{condition}_{a}-Score':_condition.loc[_condition['Axis']==a]['Corrected Score'].mean()}
            #    _calc= {f'psy_{EVM_task}_{condition}_{a}-Score':_condition.loc[_condition['Axis']==a]['Response'].mean()}
            #    merge_dict(_calc,calc)

            for f in factors:
                #_calc= {f'psy_{EVM_task}_{condition}_{a}-Score':_condition.loc[_condition['Axis']==a]['Corrected Score'].mean()}
                _calc= {f'psy_{EVM_task}_{condition}_{f}-Score':_condition.loc[_condition['Factor']==f]['Weighted'].sum()/factor_total[f]}
                merge_dict(_calc,calc)
            #########################
            P.cntMASProcessed += 1
            
        ### Update Participant Results
        P.psychoPyResults = pd.concat([P.psychoPyResults, pd.DataFrame([calc])], ignore_index=True)
    
        ### Update EVM Results File
        if os.path.exists(f"{out_folder}EVM_Results.csv"):
            old_log = pd.read_csv(f"{out_folder}EVM_Results.csv")
            #If participant already has data in clean log
            if P.ID in old_log['Participant'].tolist():
                for key, value in calc.items():
                    old_log.loc[old_log['Participant']==P.ID, key] = value
                new_log = old_log
            else:
                new_log = pd.concat([old_log, pd.DataFrame([calc])], ignore_index=True)
            new_log.to_csv(f"{out_folder}EVM_Results.csv",index=False)
        else:
            EVM_Results = pd.DataFrame()
            EVM_Results = pd.concat([EVM_Results, pd.DataFrame([calc])], ignore_index=True)
            EVM_Results.to_csv(f"{out_folder}EVM_Results.csv",index=False)

    ### Update Log File

    log = {'Participant':P.ID,
            'Date':P.date,
            'Status':P.status,      
            'PsychoPy':P.isData,
            'PsychoPyComplete':P.isDataComplete,
            'Task':EVM_task,
            'Files':P.cntMASFiles,
            'Placed':P.cntMASPlaced,
            'Processed':P.cntMASProcessed,
            'Errors':';'.join(errors)
            }

    if os.path.exists(f"{out_folder}psychoPy_log.csv"):
        old_log = pd.read_csv(f"{out_folder}psychoPy_log.csv")
        #If participant already has data in clean log
        if (P.ID in old_log['Participant'].tolist()) and (old_log.loc[old_log['Participant']==P.ID]['Task'].str.contains(EVM_task).any()):
            for key, value in log.items():
                old_log.loc[(old_log['Participant']==P.ID)&(old_log['Task'].str.contains(EVM_task)), key] = value
            new_log = old_log
        else:
            new_log = pd.concat([old_log, pd.DataFrame([log])], ignore_index=True)
        new_log.to_csv(f"{out_folder}psychoPy_log.csv",index=False)
    else:
        psychoPy_log = pd.DataFrame()
        psychoPy_log = pd.concat([psychoPy_log, pd.DataFrame([log])], ignore_index=True)
        psychoPy_log.to_csv(f"{out_folder}psychoPy_log.csv",index=False)
    
    #End

    #############################
    print_status(f"Collected {EVM_task}",f"{int(P.cntMASProcessed/3*100)}%")  
    return P


def update_PSQI(self):
    self.cntPSQIFiles = 0
    self.cntPSQIPlaced = 0
    self.cntPSQIProcessed = 0


def get_PSQI(in_folder, out_folder,P):
    '''
    PSQI Data
    '''
    
    
    #Define variables
    psychoPy_log = pd.DataFrame()

    errors = []
    in_path = f"{in_folder}{P.ID}/"
    out_path = f"{out_folder}{P.ID}/"
    os.makedirs(out_path, exist_ok=True) 
    
    EVM_Results = pd.DataFrame()
    EVM_task = 'PSQI'

    def time_to_minute(time_str):
        # Split the time string into hours and minutes
        hours, minutes = map(int, time_str.split(':'))
        
        # Calculate total minutes
        total_minutes = hours * 60 + minutes
        
        return total_minutes

    def time_difference(time1, time2):
        # Convert each time to minutes
        hours1, minutes1 = map(int, time1.split(':'))
        total_minutes1 = hours1 * 60 + minutes1
        
        hours2, minutes2 = map(int, time2.split(':'))
        total_minutes2 = hours2 * 60 + minutes2
        
        # Calculate the difference in minutes
        total = total_minutes2 + (1440-total_minutes1)
        
        # Convert the difference back to hours
        difference_in_hours = total / 60
        
        return difference_in_hours

    ##Reset Flags
    P.cntPSQIFiles = 0
    P.cntPSQIPlaced = 0
    P.cntPSQIProcessed = 0
    
    taskData=pd.DataFrame()
    print_step(f'Collecting {EVM_task}',P.ID)
    ##### PSQI Data
    ##Extract data from files
    files = get_files(in_path,tags=['.csv', EVM_task]) #Get list of all goNogo tasks
    #For each file
    file_num = 0
    for f in files:                   
        try:
            path = in_path + f                 
            f=open(path)                    
            df = pd.read_csv(path, low_memory = False)
            df=df[['Time','QuestionNumber','Response']]
            df['Path']=path
            df['File']=file_num

            #create absolute time column
            df['Time']=df['Time'].str.replace(',','.')
            df['t']=[get_sec(i) for i in df['Time'].tolist()]

            ##Update Counters
            file_num = file_num+1
            print_status(f'Extracted {EVM_task}',path)
            P.cntPSQIPlaced = 1
            ##Contextualise Data
            # #set task from EVM list
            # try:
            #     start_time = df['t'].tolist()[0]
            #     end_time = df['t'].tolist()[-1]
            #     task = P.tasksInfo.loc[(P.tasksInfo['Start']<=start_time) & (P.tasksInfo['End']>=end_time)]['Task'].tolist()[0]
            #     df['Task']=task
            #     print_status('Placed Data',task)

            #     #########################
            #     P.cntPSQIPlaced += 1
            # except IndexError:
            #     print_status('Unplaced Data',path)
            # except Exception as z:
            #     print_status(f'Failed Placing {EVM_task} Data',f"{path} - {z!r}")
            #     errors = errors + [f"Failed Placing {EVM_task} Data: {path} - {z!r}",]
                          
            ##Update Participant Data
            taskData = pd.concat([taskData,df])

            #########################
            P.cntPSQIFiles += 1
        
        ## Account for Issues
        except Exception as z:
            print_status(f'Failed {EVM_task}',f"{path} - {z!r}")
            errors = errors + [f"Failed {EVM_task}: {path} - {z!r}",]
            
    if P.cntPSQIPlaced:
        ## Set condition
        taskData = taskData.dropna(subset=['Response',])
        answers = {row['QuestionNumber']:row['Response'] for index,row in taskData.iterrows()}

        ###Process PSQI Data 
        ## Initialise row entry
        calc = {'Participant':P.ID,
                'Drink':P.drink,}

        #Process PSQI Data
        #for each conditio
        ### Write Results

        factors = ['SleepQuality',
               'SleepLatency',
               'SleepDuration',
               'SleepEfficiency',
               'SleepDisturbances',
               'SleepMedication',
               'SleepDysfunction'
        ]

        ### Fix some answers
        for x in [1,3]:
            if len(str(answers[x]))<=2:
                answers[x] = f'{answers[x]}:00'
            elif len(str(answers[x]))==4:
                if isinstance(x,int):
                    if '.' in str(answers[x]):
                        answers[x] = str(answers[x]).replace('.',':')
                    else:
                        answers[x] = f'{str(answers[x])[0:2]}:{str(answers[x])[2:]}'
            elif len(str(answers[x]))==3:
                if isinstance(x,int):
                    if '.' in str(answers[x]):
                        answers[x] = str(answers[x]).replace('.',':')
                    else:
                        answers[x] = f'{str(answers[x])[0]}:{str(answers[x])[1:]}'

        if isinstance(answers[2],str):
            answers[2] = answers[2].replace(':','')
            answers[2] = answers[2].replace('::','')

        for x in [1,3]:
            answers[x] = answers[x].replace('::',':')
            if len(answers[x].split(':'))>2:
                answers[x] = ':'.join(answers[x].split(':')[0:2])

        if isinstance(answers[4],str):
            answers[4] = answers[4].replace(':','.')
     
        psqi = 0
        for factor in factors:

            result = None
            if factor == 'SleepDuration':
                sleep_duration = float(answers[4])
                if sleep_duration <5:
                    result = 3
                elif (sleep_duration>=5) & (sleep_duration<6):
                    result = 2
                elif (sleep_duration>=6) & (sleep_duration<7):
                    result = 1
                elif (sleep_duration>=7):
                    result = 0
            elif factor == 'SleepEfficiency':
                if (':' in answers[1]) and (':' in answers[3]):
                    time_in_bed = time_difference(answers[1],answers[3])
                    time_asleep = float(answers[4])
                    efficiency = time_asleep/time_in_bed*100
                    if efficiency <65:
                        result = 3
                    elif (efficiency>=65) & (efficiency<75):
                        result = 2
                    elif (efficiency>=75) & (efficiency<85):
                        result = 1
                    elif (efficiency>=85):
                        result = 0
                else:
                    print_status('Failed SleepEfficiency','Incorrect Format')
            elif factor == 'SleepLatency':
                latency = float(answers[2])
                if latency <= 15:
                    result = 0
                elif (latency>15) & (latency<31):
                    result = 1
                elif (latency>30) & (latency<61):
                    result = 2
                elif (latency>60):
                    result = 3
                result += float(answers[51])
            elif factor == 'SleepDisturbances':
                result = float(answers[52])
            elif factor == 'SleepMedication':
                result = float(answers[7])
            elif factor == 'SleepDysfunction':
                result = (float(answers[8]) + float(answers[9])/2)
            elif factor == 'SleepQuality':
                result = float(answers[6])
                
            _calc= {f'{EVM_task}_{factor}':result}
            merge_dict(_calc,calc)

            try:
                psqi += result
            except:
                print_status(f'Failed {factor}','Incorrect Format')

            P.cntPSQIProcessed += 1

        _calc= {f'{EVM_task}_Overall':(21-psqi)/21*100}
        merge_dict(_calc,calc)

        _calc={f'time_{EVM_task}-Duration':taskData.t.values[-1] - taskData.t.values[0]}
        merge_dict(_calc,calc)
            
        ### Update Participant Results
        P.psychoPyResults = pd.concat([P.psychoPyResults, pd.DataFrame([calc])], ignore_index=True)

        # # Update timings
        # taskName = 'PSQI'
        # timings = {'Task':taskName,
        #            'Start':taskData.t.values[0],
        #            'End':taskData.t.values[-1],
        #            'Duration':taskData.t.values[-1] - taskData.t.values[0]}
        # P.tasksInfo = P.tasksInfo.reset_index(drop=True)
        # if taskName in P.tasksInfo['Task'].tolist():
        #     P.tasksInfo.loc[P.tasksInfo['Task']==taskName] = timings
        # else:
        #     P.tasksInfo.loc[len(P.tasksInfo)] = timings
        # P.tasksInfo = P.tasksInfo.sort_values('Start')
        # P.tasksInfo.to_csv(f"{out_path}{P.ID}_EVM_log.csv",index=False)
    
        ### Update EVM Results File
        if os.path.exists(f"{out_folder}EVM_Results.csv"):
            old_log = pd.read_csv(f"{out_folder}EVM_Results.csv")
            #If participant already has data in clean log
            if P.ID in old_log['Participant'].tolist():
                for key, value in calc.items():
                    old_log.loc[old_log['Participant']==P.ID, key] = value
                new_log = old_log
            else:
                new_log = pd.concat([old_log, pd.DataFrame([calc])], ignore_index=True)
            new_log.to_csv(f"{out_folder}EVM_Results.csv",index=False)
        else:
            EVM_Results = pd.DataFrame()
            EVM_Results = pd.concat([EVM_Results, pd.DataFrame([calc])], ignore_index=True)
            EVM_Results.to_csv(f"{out_folder}EVM_Results.csv",index=False)

    ### Update Log File

    log = {'Participant':P.ID,
            'Date':P.date,
            'Status':P.status,      
            'PsychoPy':P.isData,
            'PsychoPyComplete':P.isDataComplete,
            'Task':EVM_task,
            'Files':P.cntPSQIFiles,
            'Placed':P.cntPSQIPlaced,
            'Processed':P.cntPSQIProcessed,
            'Errors':';'.join(errors)
            }

    if os.path.exists(f"{out_folder}psychoPy_log.csv"):
        old_log = pd.read_csv(f"{out_folder}psychoPy_log.csv")
        #If participant already has data in clean log
        if (P.ID in old_log['Participant'].tolist()) and (old_log.loc[old_log['Participant']==P.ID]['Task'].str.contains(EVM_task).any()):
            for key, value in log.items():
                old_log.loc[(old_log['Participant']==P.ID)&(old_log['Task'].str.contains(EVM_task)), key] = value
            new_log = old_log
        else:
            new_log = pd.concat([old_log, pd.DataFrame([log])], ignore_index=True)
        new_log.to_csv(f"{out_folder}psychoPy_log.csv",index=False)
    else:
        psychoPy_log = pd.DataFrame()
        psychoPy_log = pd.concat([psychoPy_log, pd.DataFrame([log])], ignore_index=True)
        psychoPy_log.to_csv(f"{out_folder}psychoPy_log.csv",index=False)
    
    #End

    #############################
    print_status(f"Collected {EVM_task}",f"{int(P.cntPSQIProcessed/7*100)}%")  
    return P


def update_POMS(self):
    self.cntPOMSFiles = 0
    self.cntPOMSPlaced = 0
    self.cntPOMSProcessed = 0


def get_POMS(in_folder, out_folder,P):
    '''
    This function cleans and combines all the OpenSignals data from one participant into a single file

    Parameters
    __________
    in_folder: str
        Folder containing directories for each participant
        Must contain ONLY particpant 
    out_folder: str
        Folder in which a directory will be created per participant
        
    TODO:
    - Add in check for duplicate participant IDs
    '''
    
    
    #Define variables
    psychoPy_log = pd.DataFrame()

    errors = []
    in_path = f"{in_folder}{P.ID}/"
    out_path = f"{out_folder}{P.ID}/"
    os.makedirs(out_path, exist_ok=True) 
    
    EVM_Results = pd.DataFrame()
    EVM_task = 'POMS'

    POMS_polarity ={
        'Tense': 'P',
        'Angry': 'P',
        'Worn Out': 'P',
        'Unhappy': 'P',
        'Proud': 'P',
        'Lively': 'P',
        'Confused': 'P',
        'Sad': 'P',
        'Active': 'P',
        'On-edge': 'P',
        'Grouchy': 'P',
        'Ashamed': 'N',
        'Energetic': 'P',
        'Hopeless': 'P',
        'Uneasy': 'P',
        'Restless': 'P',
        'Unable to concentrate': 'P',
        'Fatigued': 'P',
        'Capable': 'P',
        'Annoyed': 'P',
        'Discouraged': 'P',
        'Resentful': 'P',
        'Nervous': 'P',
        'Miserable': 'P',
        'Confident': 'P',
        'Unforgiving': 'P',
        'Exhausted': 'P',
        'Anxious': 'P',
        'Helpless': 'P',
        'Weary': 'P',
        'Satisfied': 'P',
        'Puzzled': 'P',
        'Furious': 'P',
        'Full of energy': 'P',
        'Worthless': 'P',
        'Forgetful': 'P',
        'Physically Energised': 'P',
        'Uncertain about things': 'P',
        'Very Tired': 'P',
        'Embarrassed': 'N'
    }

    POMS_Factors = {
        'Tense': 'TEN',
        'Angry': 'ANG',
        'Worn Out': 'FAT',
        'Unhappy': 'DEP',
        'Proud': 'ERA',
        'Lively': 'VIG',
        'Confused': 'CON',
        'Sad': 'DEP',
        'Active': 'VIG',
        'On-edge': 'TEN',
        'Grouchy': 'ANG',
        'Ashamed': 'ERA',  # This will be reverse-scored
        'Energetic': 'VIG',
        'Hopeless': 'DEP',
        'Uneasy': 'TEN',
        'Restless': 'TEN',
        'Unable to concentrate': 'CON',
        'Fatigued': 'FAT',
        'Capable': 'ERA',
        'Annoyed': 'ANG',
        'Discouraged': 'DEP',
        'Resentful': 'ANG',
        'Nervous': 'TEN',
        'Miserable': 'DEP',
        'Confident': 'ERA',
        'Unforgiving': 'ANG',
        'Exhausted': 'FAT',
        'Anxious': 'TEN',
        'Helpless': 'DEP',
        'Weary': 'FAT',
        'Satisfied': 'ERA',
        'Puzzled': 'CON',
        'Furious': 'ANG',
        'Full of energy': 'VIG',
        'Worthless': 'DEP',
        'Forgetful': 'CON',
        'Physically Energised': 'VIG',
        'Uncertain about things': 'CON',
        'Very Tired': 'FAT',
        'Embarrassed': 'ERA'
    }

    ##Reset Flags
    P.cntPOMSFiles = 0
    P.cntPOMSPlaced = 0
    P.cntPOMSProcessed = 0
    
    taskData=pd.DataFrame()
    print_step(f'Collecting {EVM_task}',P.ID)
    ##### POMS Data
    ##Extract data from files
    files = get_files(in_path,tags=['.csv', EVM_task]) #Get list of all goNogo tasks
    #For each file
    file_num = 0
    for f in files:                   
        try:
            path = in_path + f                 
            f=open(path)                    
            df = pd.read_csv(path, low_memory = False)
            df=df[['Time','Question','Response']]
            df['Path']=path
            df['File']=file_num

            ##Adjust data
            for q in df['Question']:
                df.loc[df['Question'] == q, ['Polarity','Factor']] = POMS_polarity[q], POMS_Factors[q],

            #create absolute time column
            df['Time']=df['Time'].str.replace(',','.')
            df['t']=[get_sec(i) for i in df['Time'].tolist()]

            ##Update Counters
            file_num = file_num+1
            print_status(f'Extracted {EVM_task}',path)

            ##Contextualise Data
            #set task from EVM list
            try:
                start_time = df['t'].tolist()[0]
                end_time = df['t'].tolist()[-1]
                task = P.tasksInfo.loc[(P.tasksInfo['Start']<=start_time) & (P.tasksInfo['End']>=end_time)]['Task'].tolist()[0]
                df['Task']=task
                print_status('Placed Data',task)

                #########################
                P.cntPOMSPlaced += 1
            except IndexError:
                print_status('Unplaced Data',path)
            except Exception as z:
                print_status(f'Failed Placing {EVM_task} Data',f"{path} - {z!r}")
                errors = errors + [f"Failed Placing {EVM_task} Data: {path} - {z!r}",]
                          
            ##Update Participant Data
            taskData = pd.concat([taskData,df])

            #########################
            P.cntPOMSFiles += 1
        
        ## Account for Issues
        except Exception as z:
            print_status(f'Failed {EVM_task}',f"{path} - {z!r}")
            errors = errors + [f"Failed {EVM_task}: {path} - {z!r}",]
            
    if P.cntPOMSPlaced:
        ## Set condition
        taskData = taskData.dropna(subset=['Task'])
        taskData = taskData.dropna(subset=['Response',])
        taskData.loc[taskData['Task'].str.contains('_01'), 'Condition']='Baseline'
        taskData.loc[taskData['Task'].str.contains('_02'), 'Condition']='TreatmentTasksStart'
        taskData.loc[taskData['Task'].str.contains('_03'), 'Condition']='TreatmentTasksEnd'

        taskData.loc[taskData['Polarity']=='N','Corrected Score']=4-taskData.loc[taskData['Polarity']=='N']['Response']
        taskData.loc[taskData['Polarity']=='P','Corrected Score']=taskData.loc[taskData['Polarity']=='P']['Response']

        ###Process GoNoGo Data 
        ## Initialise row entry
        calc = {'Participant':P.ID,
                'Drink':P.drink,}

        #Process POMS Data
        #for each condition
        conditions = taskData['Condition'].drop_duplicates()
        for condition in conditions:
            _condition = taskData.loc[taskData['Condition']==condition]
            #Update row entry
            questions = _condition['Question'].drop_duplicates()
            #axes = list(_condition['Axis'].drop_duplicates())
            factors = list(_condition['Factor'].drop_duplicates())

            for q in questions:
                _val = _condition.loc[_condition['Question']==q]['Corrected Score'].tolist()[0]
                
                try:
                    _calc= {f'psy_{EVM_task}_{condition}_{q}':_val,}
                    merge_dict(_calc,calc)
                except Exception as z:
                    print_status(f'Error {EVM_task} on {q}',f"{path} - {z!r}")
                    errors = errors + [f"Failed {EVM_task} Question {q}: {path} - {z!r}",]

            factor_scores = {}
            for f in factors:
                score = _condition.loc[_condition['Factor']==f]['Corrected Score'].mean()
                factor_scores[f]=score
                _calc= {f'psy_{EVM_task}_{condition}_{f}-Score':score}
                merge_dict(_calc,calc)

            tmd = (factor_scores['TEN']+factor_scores['DEP']+factor_scores['ANG']+factor_scores['FAT']+factor_scores['CON'])-(factor_scores['VIG']+factor_scores['ERA'])
            _calc= {f'psy_{EVM_task}_{condition}_TMD':tmd}
            merge_dict(_calc,calc)
            #########################
            P.cntPOMSProcessed += 1
            
            
        ### Update Participant Results
        P.psychoPyResults = pd.concat([P.psychoPyResults, pd.DataFrame([calc])], ignore_index=True)
    
        ### Update EVM Results File
        if os.path.exists(f"{out_folder}EVM_Results.csv"):
            old_log = pd.read_csv(f"{out_folder}EVM_Results.csv")
            #If participant already has data in clean log
            if P.ID in old_log['Participant'].tolist():
                for key, value in calc.items():
                    old_log.loc[old_log['Participant']==P.ID, key] = value
                new_log = old_log
            else:
                new_log = pd.concat([old_log, pd.DataFrame([calc])], ignore_index=True)
            new_log.to_csv(f"{out_folder}EVM_Results.csv",index=False)
        else:
            EVM_Results = pd.DataFrame()
            EVM_Results = pd.concat([EVM_Results, pd.DataFrame([calc])], ignore_index=True)
            EVM_Results.to_csv(f"{out_folder}EVM_Results.csv",index=False)

    ### Update Log File

    log = {'Participant':P.ID,
            'Date':P.date,
            'Status':P.status,      
            'PsychoPy':P.isData,
            'PsychoPyComplete':P.isDataComplete,
            'Task':EVM_task,
            'Files':P.cntPOMSFiles,
            'Placed':P.cntPOMSPlaced,
            'Processed':P.cntPOMSProcessed,
            'Errors':';'.join(errors)
            }

    if os.path.exists(f"{out_folder}psychoPy_log.csv"):
        old_log = pd.read_csv(f"{out_folder}psychoPy_log.csv")
        #If participant already has data in clean log
        if (P.ID in old_log['Participant'].tolist()) and (old_log.loc[old_log['Participant']==P.ID]['Task'].str.contains(EVM_task).any()):
            for key, value in log.items():
                old_log.loc[(old_log['Participant']==P.ID)&(old_log['Task'].str.contains(EVM_task)), key] = value
            new_log = old_log
        else:
            new_log = pd.concat([old_log, pd.DataFrame([log])], ignore_index=True)
        new_log.to_csv(f"{out_folder}psychoPy_log.csv",index=False)
    else:
        psychoPy_log = pd.DataFrame()
        psychoPy_log = pd.concat([psychoPy_log, pd.DataFrame([log])], ignore_index=True)
        psychoPy_log.to_csv(f"{out_folder}psychoPy_log.csv",index=False)
    
    #End

    #############################
    print_status(f"Collected {EVM_task}",f"{int(P.cntPOMSProcessed/3*100)}%")  
    return P


def update_BMSC(self):
    self.cntBMSCFiles = 0
    self.cntBMSCPlaced = 0
    self.cntBMSCProcessed = 0


def get_BMSC(in_folder, out_folder,P):
    '''
    This function cleans and combines all the OpenSignals data from one participant into a single file

    Parameters
    __________
    in_folder: str
        Folder containing directories for each participant
        Must contain ONLY particpant 
    out_folder: str
        Folder in which a directory will be created per participant
        
    TODO:
    - Add in check for duplicate participant IDs
    '''
    
    
    #Define variables
    psychoPy_log = pd.DataFrame()

    errors = []
    in_path = f"{in_folder}{P.ID}/"
    out_path = f"{out_folder}{P.ID}/"
    os.makedirs(out_path, exist_ok=True) 
    
    EVM_Results = pd.DataFrame()
    EVM_task = 'BMS_Comprehension'

    factors = {
            "Alert\nIsilumkiso\nWaaksaam": 'Vigour',
            "Angry\nUnomsindo\nKwaad": 'Anger',
            "Annoyed\nUkudikwa\nGeïrriteerd": 'Anger',
            "Anxious\nUvalo\nAngstig": 'Anxious',
            "Bitter\nInqala\nBitter": 'Anger',
            "Calm\nUzolile\nKalm": 'Calm',
            "Cheerful\nUvuyo\nVrolik": 'Happy',
            "Composed\nYakhiwe\nSaamgesteld": 'Calm',
            "Confused\nUxakekileyo\nVerward": 'Confused',
            "Contented\nWanelisekile\nTevrede": 'Happy',
            "Depressed\nQumbile\nDepressief": 'Depression',
            "Downhearted\nNdidakumbile\nNeerslagtig": 'Depression',
            "Energetic\nSemandleni\nEnergiek": 'Vigour',
            "Exhausted\nNdiniwe\nUitgeput": 'Fatigue',
            "Happy\nWonwabile\nGelukkig": 'Happy',
            "Lively\nWonwabile\nLewendig": 'Vigour',
            "Miserable\nUqumbile\nEllendig": 'Depression',
            "Nervous\nUvalo\nSenuweeagtig": 'Anxious',
            "Panicky\nPhakuzela\nPaniekerig": 'Anxious',
            "Relaxed\nKhululekile\nOntspanne": 'Calm',
            "Restful\nPhumlile\nRustig": 'Calm',
            "Satisfied\nWanelisekile\nTevrede": 'Happy',
            "Sleepy\nUyozela\nSlaperig": 'Fatigue',
            "Tired\nNdiniwe\nMoeg": 'Fatigue',
            "Uncertain\nUkungaqiniseki\nOnseker": 'Confused',
            "Unhappy\nAwonwabanga\nOngelukkig": 'Depression',
            "Worn-out\nUkudinwa\nAfgemat": 'Fatigue',
            "Worried\nUkhathazekile\nBekommerd": 'Anxious',
            "Mixed-up\nKuphithene\nVerward": 'Confused',
            "Muddled\nUphithene\nDeurmekaar": 'Confused',
            "Active\nEsebenzayo\nAktief": 'Vigour',
            "Bad tempered\nUmoya ocaphukileyo\nSlegte humeur": 'Anger'
    }

    ##Reset Flags
    P.cntBMSCFiles = 0
    P.cntBMSCPlaced = 0
    P.cntBMSCProcessed = 0
    
    taskData=pd.DataFrame()
    print_step(f'Collecting {EVM_task}',P.ID)
    ##### BMSC Data
    ##Extract data from files
    files = get_files(in_path,tags=['.csv', EVM_task]) #Get list of all goNogo tasks
    #For each file
    file_num = 0
    for f in files:                   
        try:
            path = in_path + f                 
            f=open(path)                    
            df = pd.read_csv(path, low_memory = False)
            df=df[['Time','Question','Response']]
            df['Path']=path
            df['File']=file_num

            ##Adjust data
            for q in df['Question']:
                df.loc[df['Question'] == q, 'Factor'] = factors[q],

            #create absolute time column
            df['Time']=df['Time'].str.replace(',','.')
            df['t']=[get_sec(i) for i in df['Time'].tolist()]

            ##Update Counters
            file_num = file_num+1
            print_status(f'Extracted {EVM_task}',path)
            P.cntBMSCPlaced += 1                          
            ##Update Participant Data
            taskData = pd.concat([taskData,df])

            #########################
            P.cntBMSCFiles += 1
        
        ## Account for Issues
        except Exception as z:
            print_status(f'Failed {EVM_task}',f"{path} - {z!r}")
            errors = errors + [f"Failed {EVM_task}: {path} - {z!r}",]
            
    if P.cntBMSCPlaced:
        ## Set condition
        taskData = taskData.dropna(subset=['Response',])

        taskData['Condition']= 'Baseline'
        # taskData.loc[taskData['Task'].str.contains('_00'), 'Condition']='Baseline'
        # taskData.loc[taskData['Task'].str.contains('_01'), 'Condition']='Passive'
        # taskData.loc[taskData['Task'].str.contains('_02'), 'Condition']='End'

        ###Process GoNoGo Data 
        ## Initialise row entry
        calc = {'Participant':P.ID,
                'Drink':P.drink,}

        #Process BMSC Data
        #for each condition
        conditions = taskData['Condition'].drop_duplicates()
        for condition in conditions:
            _condition = taskData.loc[taskData['Condition']==condition]
            #Update row entry
            questions = _condition['Question'].drop_duplicates()
            #axes = list(_condition['Axis'].drop_duplicates())
            factors = list(_condition['Factor'].drop_duplicates())

            # for q in questions:
            #     _val = _condition.loc[_condition['Question']==q]['Response'].tolist()[0]
                
                #Response for each question
                # try:
                #     _calc= {f'psy_{EVM_task}_{condition}_{q}':_val,}
                #     merge_dict(_calc,calc)
                # except Exception as z:
                #     print_status(f'Error {EVM_task} on {q}',f"{path} - {z!r}")
                #     errors = errors + [f"Failed {EVM_task} Question {q}: {path} - {z!r}",]

            factor_scores = {}
            for f in factors:
                score = _condition.loc[_condition['Factor']==f]['Response'].mean()
                factor_scores[f]=score
                _calc= {f'psy_{EVM_task}_{f}-Score':score}
                merge_dict(_calc,calc)

            _calc= {f'psy_Comprehension':taskData['Response'].mean()}
            merge_dict(_calc,calc)

            _calc={f'time_{EVM_task}-Duration':taskData.t.values[-1] - taskData.t.values[0]}
            merge_dict(_calc,calc)
            #########################
            P.cntBMSCProcessed += 1
            
            
        ### Update Participant Results
        P.psychoPyResults = pd.concat([P.psychoPyResults, pd.DataFrame([calc])], ignore_index=True)
    
        # Update timings
        taskName = 'BMSC'
        timings = {'Task':taskName,
                   'Start':taskData.t.values[0],
                   'End':taskData.t.values[-1],
                   'Duration':taskData.t.values[-1] - taskData.t.values[0]}
        P.tasksInfo = P.tasksInfo.reset_index(drop=True)
        if taskName in P.tasksInfo['Task'].tolist():
            P.tasksInfo.loc[P.tasksInfo['Task']==taskName] = timings
        else:
            P.tasksInfo.loc[len(P.tasksInfo)] = timings
        P.tasksInfo = P.tasksInfo.sort_values('Start')
        P.tasksInfo.to_csv(f"{out_path}{P.ID}_EVM_log.csv",index=False)

        ### Update EVM Results File
        if os.path.exists(f"{out_folder}EVM_Results.csv"):
            old_log = pd.read_csv(f"{out_folder}EVM_Results.csv")
            #If participant already has data in clean log
            if P.ID in old_log['Participant'].tolist():
                for key, value in calc.items():
                    old_log.loc[old_log['Participant']==P.ID, key] = value
                new_log = old_log
            else:
                new_log = pd.concat([old_log, pd.DataFrame([calc])], ignore_index=True)
            new_log.to_csv(f"{out_folder}EVM_Results.csv",index=False)
        else:
            EVM_Results = pd.DataFrame()
            EVM_Results = pd.concat([EVM_Results, pd.DataFrame([calc])], ignore_index=True)
            EVM_Results.to_csv(f"{out_folder}EVM_Results.csv",index=False)

    ### Update Log File

    log = {'Participant':P.ID,
            'Date':P.date,
            'Status':P.status,      
            'PsychoPy':P.isData,
            'PsychoPyComplete':P.isDataComplete,
            'Task':EVM_task,
            'Files':P.cntBMSCFiles,
            'Placed':P.cntBMSCPlaced,
            'Processed':P.cntBMSCProcessed,
            'Errors':';'.join(errors)
            }

    if os.path.exists(f"{out_folder}psychoPy_log.csv"):
        old_log = pd.read_csv(f"{out_folder}psychoPy_log.csv")
        #If participant already has data in clean log
        if (P.ID in old_log['Participant'].tolist()) and (old_log.loc[old_log['Participant']==P.ID]['Task'].str.contains(EVM_task).any()):
            for key, value in log.items():
                old_log.loc[(old_log['Participant']==P.ID)&(old_log['Task'].str.contains(EVM_task)), key] = value
            new_log = old_log
        else:
            new_log = pd.concat([old_log, pd.DataFrame([log])], ignore_index=True)
        new_log.to_csv(f"{out_folder}psychoPy_log.csv",index=False)
    else:
        psychoPy_log = pd.DataFrame()
        psychoPy_log = pd.concat([psychoPy_log, pd.DataFrame([log])], ignore_index=True)
        psychoPy_log.to_csv(f"{out_folder}psychoPy_log.csv",index=False)
    
    #End

    #############################
    print_status(f"Collected {EVM_task}",f"{int(P.cntBMSCProcessed*100)}%")  
    return P


def update_IFIS(self):
    self.cntIFISFiles = 0
    self.cntIFISPlaced = 0
    self.cntIFISProcessed = 0


def get_IFIS(in_folder, out_folder,P):
    '''
    This function cleans and combines all the OpenSignals data from one participant into a single file

    Parameters
    __________
    in_folder: str
        Folder containing directories for each participant
        Must contain ONLY particpant 
    out_folder: str
        Folder in which a directory will be created per participant
        
    TODO:
    - Add in check for duplicate participant IDs
    '''
    
    
    #Define variables
    psychoPy_log = pd.DataFrame()

    errors = []
    in_path = f"{in_folder}{P.ID}/"
    out_path = f"{out_folder}{P.ID}/"
    os.makedirs(out_path, exist_ok=True) 
    
    EVM_Results = pd.DataFrame()
    EVM_task = 'IFIS'

    ##Reset Flags
    P.cntIFISFiles = 0
    P.cntIFISPlaced = 0
    P.cntIFISProcessed = 0
    
    taskData=pd.DataFrame()
    print_step(f'Collecting {EVM_task}',P.ID)
    ##### IFIS Data
    ##Extract data from files
    files = get_files(in_path,tags=['.csv', EVM_task]) #Get list of all goNogo tasks
    #For each file
    file_num = 0
    for f in files:                   
        try:
            path = in_path + f                 
            f=open(path)                    
            df = pd.read_csv(path, low_memory = False)
            df=df[['Time','Question','Response']]
            df['Path']=path
            df['File']=file_num

            #create absolute time column
            df['Time']=df['Time'].str.replace(',','.')
            df['t']=[get_sec(i) for i in df['Time'].tolist()]

            ##Update Counters
            file_num = file_num+1
            print_status(f'Extracted {EVM_task}',path)
            P.cntIFISPlaced += 1                          
            ##Update Participant Data
            taskData = pd.concat([taskData,df])

            #########################
            P.cntIFISFiles += 1
        
        ## Account for Issues
        except Exception as z:
            print_status(f'Failed {EVM_task}',f"{path} - {z!r}")
            errors = errors + [f"Failed {EVM_task}: {path} - {z!r}",]
            
    if P.cntIFISPlaced:
        ## Set condition
        taskData = taskData.dropna(subset=['Response',])
        taskData['Condition']= 'Baseline'
        ###Process GoNoGo Data 
        ## Initialise row entry
        calc = {'Participant':P.ID,
                'Drink':P.drink,}

        #Process IFIS Data
        #for each condition
        conditions = taskData['Condition'].drop_duplicates()
        for condition in conditions:
            _condition = taskData.loc[taskData['Condition']==condition]
            #Update row entry
            score = _condition['Response'].mean()
            _calc= {f'psy_{EVM_task}_Fitness-Score':score}
            merge_dict(_calc,calc)

            _calc={f'time_{EVM_task}-Duration':taskData.t.values[-1] - taskData.t.values[0]}
            merge_dict(_calc,calc)
            #########################
            P.cntIFISProcessed += 1
        
            
        ### Update Participant Results
        P.psychoPyResults = pd.concat([P.psychoPyResults, pd.DataFrame([calc])], ignore_index=True)
    
        # Update timings
        taskName = 'IFIS'
        timings = {'Task':taskName,
                   'Start':taskData.t.values[0],
                   'End':taskData.t.values[-1],
                   'Duration':taskData.t.values[-1] - taskData.t.values[0]}
        P.tasksInfo = P.tasksInfo.reset_index(drop=True)
        if taskName in P.tasksInfo['Task'].tolist():
            P.tasksInfo.loc[P.tasksInfo['Task']==taskName] = timings
        else:
            P.tasksInfo.loc[len(P.tasksInfo)] = timings
        P.tasksInfo = P.tasksInfo.sort_values('Start')
        P.tasksInfo.to_csv(f"{out_path}{P.ID}_EVM_log.csv",index=False)

        ### Update EVM Results File
        if os.path.exists(f"{out_folder}EVM_Results.csv"):
            old_log = pd.read_csv(f"{out_folder}EVM_Results.csv")
            #If participant already has data in clean log
            if P.ID in old_log['Participant'].tolist():
                for key, value in calc.items():
                    old_log.loc[old_log['Participant']==P.ID, key] = value
                new_log = old_log
            else:
                new_log = pd.concat([old_log, pd.DataFrame([calc])], ignore_index=True)
            new_log.to_csv(f"{out_folder}EVM_Results.csv",index=False)
        else:
            EVM_Results = pd.DataFrame()
            EVM_Results = pd.concat([EVM_Results, pd.DataFrame([calc])], ignore_index=True)
            EVM_Results.to_csv(f"{out_folder}EVM_Results.csv",index=False)

    ### Update Log File

    log = {'Participant':P.ID,
            'Date':P.date,
            'Status':P.status,      
            'PsychoPy':P.isData,
            'PsychoPyComplete':P.isDataComplete,
            'Task':EVM_task,
            'Files':P.cntIFISFiles,
            'Placed':P.cntIFISPlaced,
            'Processed':P.cntIFISProcessed,
            'Errors':';'.join(errors)
            }

    if os.path.exists(f"{out_folder}psychoPy_log.csv"):
        old_log = pd.read_csv(f"{out_folder}psychoPy_log.csv")
        #If participant already has data in clean log
        if (P.ID in old_log['Participant'].tolist()) and (old_log.loc[old_log['Participant']==P.ID]['Task'].str.contains(EVM_task).any()):
            for key, value in log.items():
                old_log.loc[(old_log['Participant']==P.ID)&(old_log['Task'].str.contains(EVM_task)), key] = value
            new_log = old_log
        else:
            new_log = pd.concat([old_log, pd.DataFrame([log])], ignore_index=True)
        new_log.to_csv(f"{out_folder}psychoPy_log.csv",index=False)
    else:
        psychoPy_log = pd.DataFrame()
        psychoPy_log = pd.concat([psychoPy_log, pd.DataFrame([log])], ignore_index=True)
        psychoPy_log.to_csv(f"{out_folder}psychoPy_log.csv",index=False)
    
    #End

    #############################
    print_status(f"Collected {EVM_task}",f"{int(P.cntIFISProcessed*100)}%")  
    return P


def update_caffeine(self):
    self.cntcaffeineFiles = 0
    self.cntcaffeinePlaced = 0
    self.cntcaffeineProcessed = 0


def get_caffeine(in_folder, out_folder,P):
    '''
    This function cleans and combines all the OpenSignals data from one participant into a single file

    Parameters
    __________
    in_folder: str
        Folder containing directories for each participant
        Must contain ONLY particpant 
    out_folder: str
        Folder in which a directory will be created per participant
        
    TODO:
    - Add in check for duplicate participant IDs
    '''
    
    
    #Define variables
    psychoPy_log = pd.DataFrame()

    errors = []
    in_path = f"{in_folder}{P.ID}/"
    out_path = f"{out_folder}{P.ID}/"
    os.makedirs(out_path, exist_ok=True) 
    
    EVM_Results = pd.DataFrame()
    EVM_task = 'caffeine'


    replace = {
            "How often do you drink cafeine or a caffeinated beverage such as tea, coffee, Monster, Redbull or Coke":0,
            "How often do you drink caffeine or a caffeinated beverage such as tea, coffee, Monster, Redbull or Coke":0,
            "If you drink a caffeinated beverage such as tea, coffee, Monster, Redbull or Coke daily, how many times a day to you drink it?":1,
            "When was the last time you consumed a caffeinated beverage such as tea, coffee, Monster, Redbull or Coke":2
        }

    metric = {0:'Frequency',
              1:'DailyConsumption',
              2:'LastCup'}

    ##Reset Flags
    P.cntcaffeineFiles = 0
    P.cntcaffeinePlaced = 0
    P.cntcaffeineProcessed = 0
    
    taskData=pd.DataFrame()
    print_step(f'Collecting {EVM_task}',P.ID)
    ##### caffeine Data
    ##Extract data from files
    files = get_files(in_path,tags=['.csv', EVM_task]) #Get list of all goNogo tasks
    #For each file
    file_num = 0
    for f in files:                   
        try:
            path = in_path + f                 
            f=open(path)                    
            df = pd.read_csv(path, low_memory = False)
            df=df[['Time','Question','Response']]
            df['Path']=path
            df['File']=file_num

            df['Question']= df['Question'].replace(replace)

            #create absolute time column
            df['Time']=df['Time'].str.replace(',','.')
            df['t']=[get_sec(i) for i in df['Time'].tolist()]

            ##Update Counters
            file_num = file_num+1
            print_status(f'Extracted {EVM_task}',path)
            P.cntcaffeinePlaced += 1                          
            ##Update Participant Data
            taskData = pd.concat([taskData,df])

            #########################
            P.cntcaffeineFiles += 1
        
        ## Account for Issues
        except Exception as z:
            print_status(f'Failed {EVM_task}',f"{path} - {z!r}")
            errors = errors + [f"Failed {EVM_task}: {path} - {z!r}",]
            
    if P.cntcaffeinePlaced:
        ## Set condition
        taskData = taskData.dropna(subset=['Response',])
        taskData['Condition']= 'Baseline'
        ###Process GoNoGo Data 
        ## Initialise row entry
        calc = {'Participant':P.ID,
                'Drink':P.drink,}

        #Process caffeine Data
        #for each condition
        conditions = taskData['Condition'].drop_duplicates()
        for condition in conditions:
            _condition = taskData.loc[taskData['Condition']==condition]
            #Update row entry
            for question,data in _condition.groupby('Question'):
                score = data['Response'].mean()
                _calc= {f'psy_{EVM_task}_{metric[question]}':score}
                merge_dict(_calc,calc)


            _calc={f'time_{EVM_task}-Duration':taskData.t.values[-1] - taskData.t.values[0]}
            merge_dict(_calc,calc)
            #########################
            P.cntcaffeineProcessed += 1
        
            
        ### Update Participant Results
        P.psychoPyResults = pd.concat([P.psychoPyResults, pd.DataFrame([calc])], ignore_index=True)
    
        # Update timings
        taskName = 'caffeine'
        timings = {'Task':taskName,
                   'Start':taskData.t.values[0],
                   'End':taskData.t.values[-1],
                   'Duration':taskData.t.values[-1] - taskData.t.values[0]}
        P.tasksInfo = P.tasksInfo.reset_index(drop=True)
        if taskName in P.tasksInfo['Task'].tolist():
            P.tasksInfo.loc[P.tasksInfo['Task']==taskName] = timings
        else:
            P.tasksInfo.loc[len(P.tasksInfo)] = timings
        P.tasksInfo = P.tasksInfo.sort_values('Start')
        P.tasksInfo.to_csv(f"{out_path}{P.ID}_EVM_log.csv",index=False)

        ### Update EVM Results File
        if os.path.exists(f"{out_folder}EVM_Results.csv"):
            old_log = pd.read_csv(f"{out_folder}EVM_Results.csv")
            #If participant already has data in clean log
            if P.ID in old_log['Participant'].tolist():
                for key, value in calc.items():
                    old_log.loc[old_log['Participant']==P.ID, key] = value
                new_log = old_log
            else:
                new_log = pd.concat([old_log, pd.DataFrame([calc])], ignore_index=True)
            new_log.to_csv(f"{out_folder}EVM_Results.csv",index=False)
        else:
            EVM_Results = pd.DataFrame()
            EVM_Results = pd.concat([EVM_Results, pd.DataFrame([calc])], ignore_index=True)
            EVM_Results.to_csv(f"{out_folder}EVM_Results.csv",index=False)

    ### Update Log File

    log = {'Participant':P.ID,
            'Date':P.date,
            'Status':P.status,      
            'PsychoPy':P.isData,
            'PsychoPyComplete':P.isDataComplete,
            'Task':EVM_task,
            'Files':P.cntcaffeineFiles,
            'Placed':P.cntcaffeinePlaced,
            'Processed':P.cntcaffeineProcessed,
            'Errors':';'.join(errors)
            }

    if os.path.exists(f"{out_folder}psychoPy_log.csv"):
        old_log = pd.read_csv(f"{out_folder}psychoPy_log.csv")
        #If participant already has data in clean log
        if (P.ID in old_log['Participant'].tolist()) and (old_log.loc[old_log['Participant']==P.ID]['Task'].str.contains(EVM_task).any()):
            for key, value in log.items():
                old_log.loc[(old_log['Participant']==P.ID)&(old_log['Task'].str.contains(EVM_task)), key] = value
            new_log = old_log
        else:
            new_log = pd.concat([old_log, pd.DataFrame([log])], ignore_index=True)
        new_log.to_csv(f"{out_folder}psychoPy_log.csv",index=False)
    else:
        psychoPy_log = pd.DataFrame()
        psychoPy_log = pd.concat([psychoPy_log, pd.DataFrame([log])], ignore_index=True)
        psychoPy_log.to_csv(f"{out_folder}psychoPy_log.csv",index=False)
    
    #End

    #############################
    print_status(f"Collected {EVM_task}",f"{int(P.cntcaffeineProcessed*100)}%")  
    return P


def update_MAIA(self):
    self.cntMAIAFiles = 0
    self.cntMAIAPlaced = 0
    self.cntMAIAProcessed = 0


def get_MAIA(in_folder, out_folder,P):
    '''
    This function cleans and combines all the OpenSignals data from one participant into a single file

    Parameters
    __________
    in_folder: str
        Folder containing directories for each participant
        Must contain ONLY particpant 
    out_folder: str
        Folder in which a directory will be created per participant
        
    TODO:
    - Add in check for duplicate participant IDs
    '''
    
    
    #Define variables
    psychoPy_log = pd.DataFrame()

    errors = []
    in_path = f"{in_folder}{P.ID}/"
    out_path = f"{out_folder}{P.ID}/"
    os.makedirs(out_path, exist_ok=True) 
    
    EVM_Results = pd.DataFrame()
    EVM_task = 'MAIA'

    factors = {
            "When I am tense I notice where the tension is located in my body.":'Noticing',
            "I notice when I am uncomfortable in my body.":'Noticing',
            "I notice where in my body I am comfortable":'Noticing',
            "I notice changes in my breathing, such as whether it slows down or speeds up.":'Noticing',
            "I ignore physical tension or discomfort until they become more severe": 'NotDistracting',
            "I distract myself from sensations of discomfort.": 'NotDistracting',
            " When I feel pain or discomfort, I try to power through it.": 'NotDistracting',
            "I try to ignore pain.": 'NotDistracting',
            "When I feel unpleasant body sensations, I occupy myself with something else so I don't have to feel them.": 'NotDistracting',
            " When I feel physical pain, I become upset.": "NotWorrying",
            "I start to worry that something is wrong if I feel any discomfort.": "NotWorrying",
            " I can notice an unpleasant body sensation without worrying about it.": "NotWorrying",
            "I can stay calm and not worry when I have feelings of discomfort or pain.": "NotWorrying",
            "When I am in discomfort or pain I can't get it out of my mind.": "NotWorrying",
            "I can pay attention to my breath without being distracted by things happening around me.": "AttentionRegulation" ,
            "I can maintain awareness of my inner bodily sensations even when there is a lot going on around me.": "AttentionRegulation" ,
            "When I am in conversation with someone, I can pay attention to my posture.": "AttentionRegulation" ,
            "I can return awareness to my body if I am distracted.": "AttentionRegulation" ,
            " I can refocus my attention from thinking to sensing my body.": "AttentionRegulation" ,
            "I can maintain awareness of my whole body even when a part of me is in pain or discomfort.": "AttentionRegulation" ,
            "I am able to consciously focus on my body as a whole.": "AttentionRegulation" ,
            "I notice how my body changes when I am angry.": "EmotionalAwareness",
            "When something is wrong in my life I can feel it in my body.": "EmotionalAwareness",
            "I notice that my body feels different after a peaceful experience.": "EmotionalAwareness",
            "I notice that my breathing becomes free and easy when I feel comfortable.": "EmotionalAwareness",
            "I notice how my body changes when I feel happy / joyful.": "EmotionalAwareness",
            "When I feel overwhelmed I can find a calm place inside.": "SelfRegulation",
            "When I bring awareness to my body I feel a sense of calm.": "SelfRegulation",
            "I can use my breath to reduce tension.": "SelfRegulation",
            "When I am caught up in thoughts, I can calm my mind by focusing on my body/breathing.": "SelfRegulation",
            "I listen for information from my body about my emotional state.": "BodyListening",
            "When I am upset, I take time to explore how my body feels.": "BodyListening",
            "I listen to my body to inform me about what to do.": "BodyListening",
            "I am at home in my body.": "Trust",
            "I feel my body is a safe place.": "Trust",
            " I trust my body sensations.": "Trust"
    }

    ##Reset Flags
    P.cntMAIAFiles = 0
    P.cntMAIAPlaced = 0
    P.cntMAIAProcessed = 0
    
    taskData=pd.DataFrame()
    print_step(f'Collecting {EVM_task}',P.ID)
    ##### MAIA Data
    ##Extract data from files
    files = get_files(in_path,tags=['.csv', EVM_task]) #Get list of all goNogo tasks
    #For each file
    file_num = 0
    for f in files:                   
        try:
            path = in_path + f                 
            f=open(path)                    
            df = pd.read_csv(path, low_memory = False)
            df=df[['Time','Question','Response']]
            df['Path']=path
            df['File']=file_num

            ##Adjust data
            for q in df['Question']:
                try:
                    df.loc[df['Question'] == q, 'Factor'] = factors[q]
                except KeyError:
                    pass

            #create absolute time column
            df['Time']=df['Time'].str.replace(',','.')
            df['t']=[get_sec(i) for i in df['Time'].tolist()]

            ##Update Counters
            file_num = file_num+1
            print_status(f'Extracted {EVM_task}',path)
            P.cntMAIAPlaced += 1                          
            ##Update Participant Data
            taskData = pd.concat([taskData,df])

            #########################
            P.cntMAIAFiles += 1
        
        ## Account for Issues
        except Exception as z:
            print_status(f'Failed {EVM_task}',f"{path} - {z!r}")
            errors = errors + [f"Failed {EVM_task}: {path} - {z!r}",]
            
    if P.cntMAIAPlaced:
        ## Set condition
        taskData = taskData.dropna(subset=['Response','Factor'])

        taskData['Condition']= 'Baseline'
        ## Initialise row entry
        calc = {'Participant':P.ID,
                'Drink':P.drink,}

        #Process MAIA Data
        #for each condition
        conditions = taskData['Condition'].drop_duplicates()
        for condition in conditions:
            _condition = taskData.loc[taskData['Condition']==condition]
            #axes = list(_condition['Axis'].drop_duplicates())
            factors = list(_condition['Factor'].drop_duplicates())
            factor_scores = {}
            for f in factors:
                score = _condition.loc[_condition['Factor']==f]['Response'].mean()
                factor_scores[f]=score
                _calc= {f'psy_{EVM_task}_{f}-Score':score}
                merge_dict(_calc,calc)

            _calc={f'time_{EVM_task}-Duration':taskData.t.values[-1] - taskData.t.values[0]}
            merge_dict(_calc,calc)
            #########################
            P.cntMAIAProcessed += 1
            
            
        ### Update Participant Results
        P.psychoPyResults = pd.concat([P.psychoPyResults, pd.DataFrame([calc])], ignore_index=True)

        # Update timings
        taskName = 'MAIA'
        timings = {'Task':taskName,
                   'Start':taskData.t.values[0],
                   'End':taskData.t.values[-1],
                   'Duration':taskData.t.values[-1] - taskData.t.values[0]}
        P.tasksInfo = P.tasksInfo.reset_index(drop=True)
        if taskName in P.tasksInfo['Task'].tolist():
            P.tasksInfo.loc[P.tasksInfo['Task']==taskName] = timings
        else:
            P.tasksInfo.loc[len(P.tasksInfo)] = timings
        P.tasksInfo = P.tasksInfo.sort_values('Start')
        P.tasksInfo.to_csv(f"{out_path}{P.ID}_EVM_log.csv",index=False)
    
        ### Update EVM Results File
        if os.path.exists(f"{out_folder}EVM_Results.csv"):
            old_log = pd.read_csv(f"{out_folder}EVM_Results.csv")
            #If participant already has data in clean log
            if P.ID in old_log['Participant'].tolist():
                for key, value in calc.items():
                    old_log.loc[old_log['Participant']==P.ID, key] = value
                new_log = old_log
            else:
                new_log = pd.concat([old_log, pd.DataFrame([calc])], ignore_index=True)
            new_log.to_csv(f"{out_folder}EVM_Results.csv",index=False)
        else:
            EVM_Results = pd.DataFrame()
            EVM_Results = pd.concat([EVM_Results, pd.DataFrame([calc])], ignore_index=True)
            EVM_Results.to_csv(f"{out_folder}EVM_Results.csv",index=False)

    ### Update Log File

    log = {'Participant':P.ID,
            'Date':P.date,
            'Status':P.status,      
            'PsychoPy':P.isData,
            'PsychoPyComplete':P.isDataComplete,
            'Task':EVM_task,
            'Files':P.cntMAIAFiles,
            'Placed':P.cntMAIAPlaced,
            'Processed':P.cntMAIAProcessed,
            'Errors':';'.join(errors)
            }

    if os.path.exists(f"{out_folder}psychoPy_log.csv"):
        old_log = pd.read_csv(f"{out_folder}psychoPy_log.csv")
        #If participant already has data in clean log
        if (P.ID in old_log['Participant'].tolist()) and (old_log.loc[old_log['Participant']==P.ID]['Task'].str.contains(EVM_task).any()):
            for key, value in log.items():
                old_log.loc[(old_log['Participant']==P.ID)&(old_log['Task'].str.contains(EVM_task)), key] = value
            new_log = old_log
        else:
            new_log = pd.concat([old_log, pd.DataFrame([log])], ignore_index=True)
        new_log.to_csv(f"{out_folder}psychoPy_log.csv",index=False)
    else:
        psychoPy_log = pd.DataFrame()
        psychoPy_log = pd.concat([psychoPy_log, pd.DataFrame([log])], ignore_index=True)
        psychoPy_log.to_csv(f"{out_folder}psychoPy_log.csv",index=False)
    
    #End

    #############################
    print_status(f"Collected {EVM_task}",f"{int(P.cntMAIAProcessed*100)}%")  
    return P


def update_BISBAS(self):
    self.cntBISBASFiles = 0
    self.cntBISBASPlaced = 0
    self.cntBISBASProcessed = 0


def get_BISBAS(in_folder, out_folder,P):
    '''
    This function cleans and combines all the OpenSignals data from one participant into a single file

    Parameters
    __________
    in_folder: str
        Folder containing directories for each participant
        Must contain ONLY particpant 
    out_folder: str
        Folder in which a directory will be created per participant
        
    TODO:
    - Add in check for duplicate participant IDs
    '''
    
    
    #Define variables
    psychoPy_log = pd.DataFrame()

    errors = []
    in_path = f"{in_folder}{P.ID}/"
    out_path = f"{out_folder}{P.ID}/"
    os.makedirs(out_path, exist_ok=True) 
    
    EVM_Results = pd.DataFrame()
    EVM_task = 'BISBAS'

    factors = {
            "Even if something bad is happening to me, I rarely experience fear or nervousness":"BIS",
            "I go out of my way to get things I want":"Drive", 
            "When I'm doing well at something I love to keep at it":"RewardResponsiveness", 
            "I'm always willing to try something new if I think it will be fun":"FunSeeking", 
            "When I get something I want, I feel excited and energized":"RewardResponsiveness", 
            "Criticism or scolding hurts me quite a bit":"BIS", 
            "When I want something I usually go all-out to get it":"Drive", 
            "I will often do things for no reason than they might be fun":"FunSeeking", 
            "If I see a chance to get something I want I move in on it right away":"Drive", 
            "I feel pretty worried or upset when I think or know somebody is angry at me":"BIS", 
            "When I see an opportunity for something I like I get excited right away":"RewardResponsiveness", 
            "I often act on the spur of the moment":"FunSeeking", 
            "If I think something unpleasant is going to happen I usually get pretty “worked up”":"BIS", 
            "When good things happen to me, it affects me strongly":"RewardResponsiveness", 
            "I feel worried when I think I have done poorly at something important":"BIS", 
            "I crave excitement and new sensations":"FunSeeking", 
            "When I go after something, nothing can hold me back":"Drive", 
            "I have very few fears compared to my friends":"BIS", 
            "It would excite me to win a contest":"RewardResponsiveness", 
            "I worry about making mistakes":"BIS"
    }

    polarity = {
                "Even if something bad is happening to me, I rarely experience fear or nervousness":1,
                "I go out of my way to get things I want":-1, 
                "When I'm doing well at something I love to keep at it":-1, 
                "I'm always willing to try something new if I think it will be fun":-1, 
                "When I get something I want, I feel excited and energized":-1, 
                "Criticism or scolding hurts me quite a bit":-1, 
                "When I want something I usually go all-out to get it":-1, 
                "I will often do things for no reason than they might be fun":-1, 
                "If I see a chance to get something I want I move in on it right away":-1, 
                "I feel pretty worried or upset when I think or know somebody is angry at me":-1, 
                "When I see an opportunity for something I like I get excited right away":-1, 
                "I often act on the spur of the moment":-1, 
                "If I think something unpleasant is going to happen I usually get pretty “worked up”":-1, 
                "When good things happen to me, it affects me strongly":-1, 
                "I feel worried when I think I have done poorly at something important":-1, 
                "I crave excitement and new sensations":-1, 
                "When I go after something, nothing can hold me back":-1, 
                "I have very few fears compared to my friends":1, 
                "It would excite me to win a contest":-1, 
                "I worry about making mistakes":-1
    }

    ##Reset Flags
    P.cntBISBASFiles = 0
    P.cntBISBASPlaced = 0
    P.cntBISBASProcessed = 0
    
    taskData=pd.DataFrame()
    print_step(f'Collecting {EVM_task}',P.ID)
    ##### BISBAS Data
    ##Extract data from files
    files = get_files(in_path,tags=['.csv', EVM_task]) #Get list of all goNogo tasks
    #For each file
    file_num = 0
    for f in files:                   
        try:
            path = in_path + f                 
            f=open(path)                    
            df = pd.read_csv(path, low_memory = False)
            df=df[['Time','Question','Response']]
            df['Path']=path
            df['File']=file_num

            ##Adjust data
            for q in df['Question']:
                try:
                    df.loc[df['Question'] == q, 'Factor'] = factors[q]
                    df.loc[df['Question'] == q, 'Polarity'] = polarity[q]
                except KeyError:
                    pass

            #create absolute time column
            df['Time']=df['Time'].str.replace(',','.')
            df['t']=[get_sec(i) for i in df['Time'].tolist()]

            ##Update Counters
            file_num = file_num+1
            print_status(f'Extracted {EVM_task}',path)
            P.cntBISBASPlaced += 1                          
            ##Update Participant Data
            taskData = pd.concat([taskData,df])

            #########################
            P.cntBISBASFiles += 1
        
        ## Account for Issues
        except Exception as z:
            print_status(f'Failed {EVM_task}',f"{path} - {z!r}")
            errors = errors + [f"Failed {EVM_task}: {path} - {z!r}",]
            
    if P.cntBISBASPlaced:
        ## Set condition
        taskData = taskData.dropna(subset=['Response','Factor'])

        taskData.loc[taskData['Polarity']==1, 'Corrected'] = taskData.loc[taskData['Polarity']==1]['Response']
        taskData.loc[taskData['Polarity']==-1, 'Corrected'] = 10-taskData.loc[taskData['Polarity']==-1]['Response']

        taskData['Condition']= 'Baseline'
        ## Initialise row entry
        calc = {'Participant':P.ID,
                'Drink':P.drink,}

        #Process BISBAS Data
        #for each condition
        conditions = taskData['Condition'].drop_duplicates()
        for condition in conditions:
            _condition = taskData.loc[taskData['Condition']==condition]
            #axes = list(_condition['Axis'].drop_duplicates())
            factors = list(_condition['Factor'].drop_duplicates())
            factor_scores = {}
            for f in factors:
                score = _condition.loc[_condition['Factor']==f]['Corrected'].mean()
                factor_scores[f]=score
                _calc= {f'psy_{EVM_task}_{f}-Score':score}
                merge_dict(_calc,calc)

            #########################
            _calc={f'time_{EVM_task}-Duration':taskData.t.values[-1] - taskData.t.values[0]}
            merge_dict(_calc,calc)

            P.cntBISBASProcessed += 1
            
            
        ### Update Participant Results
        P.psychoPyResults = pd.concat([P.psychoPyResults, pd.DataFrame([calc])], ignore_index=True)
    
        # Update timings
        taskName = 'BISBAS'
        timings = {'Task':taskName,
                   'Start':taskData.t.values[0],
                   'End':taskData.t.values[-1],
                   'Duration':taskData.t.values[-1] - taskData.t.values[0]}
        P.tasksInfo = P.tasksInfo.reset_index(drop=True)
        if taskName in P.tasksInfo['Task'].tolist():
            P.tasksInfo.loc[P.tasksInfo['Task']==taskName] = timings
        else:
            P.tasksInfo.loc[len(P.tasksInfo)] = timings
        P.tasksInfo = P.tasksInfo.sort_values('Start')
        P.tasksInfo.to_csv(f"{out_path}{P.ID}_EVM_log.csv",index=False)

        ### Update EVM Results File
        if os.path.exists(f"{out_folder}EVM_Results.csv"):
            old_log = pd.read_csv(f"{out_folder}EVM_Results.csv")
            #If participant already has data in clean log
            if P.ID in old_log['Participant'].tolist():
                for key, value in calc.items():
                    old_log.loc[old_log['Participant']==P.ID, key] = value
                new_log = old_log
            else:
                new_log = pd.concat([old_log, pd.DataFrame([calc])], ignore_index=True)
            new_log.to_csv(f"{out_folder}EVM_Results.csv",index=False)
        else:
            EVM_Results = pd.DataFrame()
            EVM_Results = pd.concat([EVM_Results, pd.DataFrame([calc])], ignore_index=True)
            EVM_Results.to_csv(f"{out_folder}EVM_Results.csv",index=False)

    ### Update Log File

    log = {'Participant':P.ID,
            'Date':P.date,
            'Status':P.status,      
            'PsychoPy':P.isData,
            'PsychoPyComplete':P.isDataComplete,
            'Task':EVM_task,
            'Files':P.cntBISBASFiles,
            'Placed':P.cntBISBASPlaced,
            'Processed':P.cntBISBASProcessed,
            'Errors':';'.join(errors)
            }

    if os.path.exists(f"{out_folder}psychoPy_log.csv"):
        old_log = pd.read_csv(f"{out_folder}psychoPy_log.csv")
        #If participant already has data in clean log
        if (P.ID in old_log['Participant'].tolist()) and (old_log.loc[old_log['Participant']==P.ID]['Task'].str.contains(EVM_task).any()):
            for key, value in log.items():
                old_log.loc[(old_log['Participant']==P.ID)&(old_log['Task'].str.contains(EVM_task)), key] = value
            new_log = old_log
        else:
            new_log = pd.concat([old_log, pd.DataFrame([log])], ignore_index=True)
        new_log.to_csv(f"{out_folder}psychoPy_log.csv",index=False)
    else:
        psychoPy_log = pd.DataFrame()
        psychoPy_log = pd.concat([psychoPy_log, pd.DataFrame([log])], ignore_index=True)
        psychoPy_log.to_csv(f"{out_folder}psychoPy_log.csv",index=False)
    
    #End

    #############################
    print_status(f"Collected {EVM_task}",f"{int(P.cntBISBASProcessed*100)}%")  
    return P


def update_IAAS(self):
    self.cntIAASFiles = 0
    self.cntIAASPlaced = 0
    self.cntIAASProcessed = 0


def get_IAAS(in_folder, out_folder,P):
    '''
    This function cleans and combines all the OpenSignals data from one participant into a single file

    Parameters
    __________
    in_folder: str
        Folder containing directories for each participant
        Must contain ONLY particpant 
    out_folder: str
        Folder in which a directory will be created per participant
        
    TODO:
    - Add in check for duplicate participant IDs
    '''
    
    
    #Define variables
    psychoPy_log = pd.DataFrame()

    errors = []
    in_path = f"{in_folder}{P.ID}/"
    out_path = f"{out_folder}{P.ID}/"
    os.makedirs(out_path, exist_ok=True) 
    
    EVM_Results = pd.DataFrame()
    EVM_task = 'IAAS'

    factors = {
        "I was aware of the sensations occurring within my body.\nEk was bewus van die sensasies wat in my liggaam plaasgevind het.\nBendiziva imvakalelo ebezenzeka emzimbeni wam":'InteroceptiveAwareness',
        "I paid close attention to small changes in my bodily state.\nEk het aandag gegee aan klein veranderinge in my liggaamlike toestand.\nNdinike ingqalasela kutshintsho oluncinci kwimo yam yomzimba":'InteroceptiveAwareness',
        "I clearly felt the physical sensations in my body as they occurred.\nEk kon duidelik die fisiese sensasies in my liggaam voel soos dit plaasgevind het.\nNdaziva ngokucaceliyo iimvakalelo zomzimba emzimbeni wam njengoko zazisenzeka":'InteroceptiveAwareness',
        "I was attentive to the small details of my internal physical experience.\nEk het aandag gegee aan die detail van my interne, fisiese ervaring.\nNdandinikel’ ingqalelo kwiinkcukacha ezincinci zamava am angaphakathi emzimbini":'InteroceptiveAwareness',
        "I felt a strong connection with my body during this period.\nEk kon 'n sterk verbintenis met my liggaam voel tydens hierdie tydperk.\nNdaziva ndixhulumana ngamandla nomzimba wam ngelixesha":'InteroceptiveAwareness',
        "I noticed that my bodily sensations came and went.\nEk het opgemerk dat my liggaamlike sensasies gekom en gegaan het.\nNdiqaphele ukuba imvakalelo zam zomzimba ziyafika aiphinde zihamba":'InteroceptiveAwareness',
        "I explored my inner sensations as they unfolded.\nEk het my innerlike sensasies ondersoek soos dit ontvou het.\nNdajonga iimvakalelo zam zangaphakathi njengoko ziqhubeka":'InteroceptiveAwareness',
        "I experienced a heightened awareness of my internal physical state.\nEk het 'n verhoogde bewussyn van my interne fisiese toestand ervaar.\nNdiye ndafumana ulwazi olongezelelekileyo lwemeko yam yangaphakathi yomzimba":'InteroceptiveAwareness',
        "I sensed my body reacting to both internal changes and the surrounding environment.\nEk het gevoel hoe my liggaam reageer op interne veranderinge, sowel as die omgewing rondom my.\nNdawuva umzimba wam usabele kutshintsho lwangaphakathi kunye nokusingqongileyo":'InteroceptiveAwareness',
        "I felt that my body was present and active in the moment.\nEk het gevoel dat my liggaam teenwoordig en aktief was.\nNdawuva umzimba wam ephilile kwaye esebenze ngalo mzuzu":'InteroceptiveAwareness',
        "I experienced noticeable increase in my heart beat.\nEk het 'n merkbare toename in my hartklop ervaar.\nNdeva intliziyo ibetha ngasantya esiphezulu":"IncreaseHeartbeat",
        "I felt a pressure or heaviness in my chest.\nEk het 'n druk of swaarte in my bors gevoel.\nNdeva ucinezelelo esifubeni sam":"HeavinessChest",
        "I noticed tingling sensations in my fingertips or feet.\nEk het sensasies in my vingerpunte of voete opgemerk.\nNdeva ukuchukumiseka kwi zandla zam okanye ezinyweni":"TinglingFingertips",
        "I experienced moments where my breathing was short and quick.\nEk het oomblikke ervaar waar my asemhaling kort en vinnig was.\nBendine xesha apho ukuphefumla kwam kukufutshane kwaye kukhawulezile":"ShortnessBreath",
        "I felt dizzy or lightheaded.\nEk het duiselig of lighoofdig gevoel.\nndiziva ndinesiyezi":"LightDizzy",
        "I experienced a lot of energy as if my body wanted to move.\nEk het baie energie ervaar, so asof ek my liggaam wou beweeg.\nndiziva ndise mandleni oko ingathi umzimba ufuna ukushukuma":"PropensityMovement",
        "I noticed sensations in my stomach, such as butterflies or a knot.\nEk het 'n knoop of vlinders in my maag opgemerk.\nNdiye ndaphawula ukuziva esiswini sam, okufana namabhabhathane okanye iqgina":"StomachSensations",
        "I experienced dryness in my throat.\nEk het 'n droë keel ervaar.\nNdeva ukoma wmqaleni":"ThroatDryness",
        "I noticed increased sweating during this period.\nEk het opgemerk dat ek meer sweet tydens hierdie periode.\nndiqaphele ukonyuka kombilo ngelixesha":"IncreasedSweating",
        "I experienced tightness in my shoulders, tension in my forehead, or tension around my eyes.\nEk het stywe skouers, 'n gespanne voorkop, of gespanne oë ervaar.\nndiye ndava ukuxinana emagxeni am, ukuxinezeleka ebunzi, okanye ukuxinana kwamehlo am":"TightnessShoulders"
    }

    ##Reset Flags
    P.cntIAASFiles = 0
    P.cntIAASPlaced = 0
    P.cntIAASProcessed = 0
    
    taskData=pd.DataFrame()
    print_step(f'Collecting {EVM_task}',P.ID)
    ##### IAAS Data
    ##Extract data from files

    files = get_files(in_path,tags=['.csv', EVM_task]) #Get list of all goNogo tasks
    #For each file
    file_num = 0
    for file in files:                   
        try:
            path = in_path + file                 
            f=open(path)                    
            df = pd.read_csv(path, low_memory = False)
            df=df[['Time','Question','Response']]
            df['Path']=path
            df['File']=file_num

            ##Adjust data
            for q in df['Question']:
                try:
                    df.loc[df['Question'] == q, 'Factor'] = factors[q]
                except KeyError:
                    pass

            #create absolute time column
            df['Time']=df['Time'].str.replace(',','.')
            df['t']=[get_sec(i) for i in df['Time'].tolist()]
            df['Task'] = '_'.join(file.split('_')[1:3])

            ##Update Counters
            file_num = file_num+1
            print_status(f'Extracted {EVM_task}',path)
            P.cntIAASPlaced += 1                          
            ##Update Participant Data
            taskData = pd.concat([taskData,df])

            #########################
            P.cntIAASFiles += 1
        
        ## Account for Issues
        except Exception as z:
            print_status(f'Failed {EVM_task}',f"{path} - {z!r}")
            errors = errors + [f"Failed {EVM_task}: {path} - {z!r}",]
            
    if P.cntIAASPlaced:
        ## Set condition
        taskData = taskData.dropna(subset=['Response','Factor'])

        taskData.loc[taskData['Task'].str.contains('_00'), 'Condition']='Baseline'
        taskData.loc[taskData['Task'].str.contains('_01'), 'Condition']='PassiveEnd'

        ## Initialise row entry
        calc = {'Participant':P.ID,
                'Drink':P.drink,}

        #Process IAAS Data
        #for each condition
        conditions = taskData['Condition'].drop_duplicates()
        for condition in conditions:
            _condition = taskData.loc[taskData['Condition']==condition]
            #axes = list(_condition['Axis'].drop_duplicates())
            factors = list(_condition['Factor'].drop_duplicates())
            factor_scores = {}
            for f in factors:
                score = _condition.loc[_condition['Factor']==f]['Response'].mean()
                factor_scores[f]=score
                _calc= {f'psy_{EVM_task}_{condition}_{f}-Score':score}
                merge_dict(_calc,calc)

            _calc = {f'time_{EVM_task}_{condition}-Start':_condition['t'].values[0]}
            merge_dict(_calc,calc)

            _calc={f'time_{EVM_task}_{condition}-Duration':_condition.t.values[-1] - _condition.t.values[0]}
            merge_dict(_calc,calc)
            #########################
            P.cntIAASProcessed += 1

            # Update timings
            taskName = _condition['Task'].values[0]
            timings = {'Task':taskName,
                    'Start':_condition.t.values[0],
                    'End':_condition.t.values[-1],
                    'Duration':_condition.t.values[-1] - _condition.t.values[0]}
            P.tasksInfo = P.tasksInfo.reset_index(drop=True)
            if taskName in P.tasksInfo['Task'].tolist():
                P.tasksInfo.loc[P.tasksInfo['Task']==taskName, 'Start'] = _condition.t.values[0]
                P.tasksInfo.loc[P.tasksInfo['Task']==taskName, 'End'] = _condition.t.values[-1]
                P.tasksInfo.loc[P.tasksInfo['Task']==taskName, 'Duration'] = _condition.t.values[-1] - _condition.t.values[0]
            else:
                P.tasksInfo.loc[len(P.tasksInfo)] = timings
            P.tasksInfo = P.tasksInfo.sort_values('Start')
            P.tasksInfo.to_csv(f"{out_path}{P.ID}_EVM_log.csv",index=False)
            
            
        ### Update Participant Results
        P.psychoPyResults = pd.concat([P.psychoPyResults, pd.DataFrame([calc])], ignore_index=True)
    
        ### Update EVM Results File
        if os.path.exists(f"{out_folder}EVM_Results.csv"):
            old_log = pd.read_csv(f"{out_folder}EVM_Results.csv")
            #If participant already has data in clean log
            if P.ID in old_log['Participant'].tolist():
                for key, value in calc.items():
                    old_log.loc[old_log['Participant']==P.ID, key] = value
                new_log = old_log
            else:
                new_log = pd.concat([old_log, pd.DataFrame([calc])], ignore_index=True)
            new_log.to_csv(f"{out_folder}EVM_Results.csv",index=False)
        else:
            EVM_Results = pd.DataFrame()
            EVM_Results = pd.concat([EVM_Results, pd.DataFrame([calc])], ignore_index=True)
            EVM_Results.to_csv(f"{out_folder}EVM_Results.csv",index=False)

    ### Update Log File

    log = {'Participant':P.ID,
            'Date':P.date,
            'Status':P.status,      
            'PsychoPy':P.isData,
            'PsychoPyComplete':P.isDataComplete,
            'Task':EVM_task,
            'Files':P.cntIAASFiles,
            'Placed':P.cntIAASPlaced,
            'Processed':P.cntIAASProcessed,
            'Errors':';'.join(errors)
            }

    if os.path.exists(f"{out_folder}psychoPy_log.csv"):
        old_log = pd.read_csv(f"{out_folder}psychoPy_log.csv")
        #If participant already has data in clean log
        if (P.ID in old_log['Participant'].tolist()) and (old_log.loc[old_log['Participant']==P.ID]['Task'].str.contains(EVM_task).any()):
            for key, value in log.items():
                old_log.loc[(old_log['Participant']==P.ID)&(old_log['Task'].str.contains(EVM_task)), key] = value
            new_log = old_log
        else:
            new_log = pd.concat([old_log, pd.DataFrame([log])], ignore_index=True)
        new_log.to_csv(f"{out_folder}psychoPy_log.csv",index=False)
    else:
        psychoPy_log = pd.DataFrame()
        psychoPy_log = pd.concat([psychoPy_log, pd.DataFrame([log])], ignore_index=True)
        psychoPy_log.to_csv(f"{out_folder}psychoPy_log.csv",index=False)
    
    #End

    #############################
    print_status(f"Collected {EVM_task}",f"{int(P.cntIAASProcessed/2*100)}%")  
    return P



def update_STAI(self):
    self.cntSTAIFiles = 0
    self.cntSTAIPlaced = 0
    self.cntSTAIProcessed = 0


def get_STAI(in_folder, out_folder,P):
    '''
    This function cleans and combines all the OpenSignals data from one participant into a single file

    Parameters
    __________
    in_folder: str
        Folder containing directories for each participant
        Must contain ONLY particpant 
    out_folder: str
        Folder in which a directory will be created per participant
        
    TODO:
    - Add in check for duplicate participant IDs
    '''
    
    
    #Define variables
    psychoPy_log = pd.DataFrame()

    errors = []
    in_path = f"{in_folder}{P.ID}/"
    out_path = f"{out_folder}{P.ID}/"
    os.makedirs(out_path, exist_ok=True) 
    
    EVM_Results = pd.DataFrame()
    EVM_task = 'STAI'

    polarity = {
        "I feel pleasant": 1,
        "I feel nervous and restless": -1,
        "I feel satisfied with myself": 1,
        "I wish I could be as happy as others seem to be": -1,
        "I feel like a failure": -1,
        "I feel rested": 1,
        'I am cool calm and collected': 1,
        "I feel that difficulties are piling up so that I cannot overcome them": -1,
        "I worry too much over something that really doesn't matter": -1,
        "I am happy": 1,
        "I have disturbing thoughts": -1,
        "I lack self-confidence": -1,
        "I feel secure": 1,
        "I make decisions easily": 1,
        "I feel inadequate": -1,
        "I am content": 1,
        "Some unimportant thoughts run through my mind and bother me": -1,
        "I take disappointments so keenly that I can't put them out of my mind": -1,
        "I am a steady person": 1,
        "I get in a state of tension or turmoil as I think over my recent concerns and interests": -1
        }

    ##Reset Flags
    P.cntSTAIFiles = 0
    P.cntSTAIPlaced = 0
    P.cntSTAIProcessed = 0
    
    taskData=pd.DataFrame()
    print_step(f'Collecting {EVM_task}',P.ID)
    ##### STAI Data
    ##Extract data from files
    files = get_files(in_path,tags=['.csv', EVM_task]) #Get list of all goNogo tasks
    #For each file
    file_num = 0
    for f in files:                   
        try:
            path = in_path + f                 
            f=open(path)                    
            df = pd.read_csv(path, low_memory = False)
            df=df[['Time','Question','Response']]
            df['Path']=path
            df['File']=file_num

            ##Adjust data
            for q in df['Question']:
                try:
                    df.loc[df['Question'] == q, 'Polarity'] = polarity[q]
                except KeyError:
                    pass

            #create absolute time column
            df['Time']=df['Time'].str.replace(',','.')
            df['t']=[get_sec(i) for i in df['Time'].tolist()]

            ##Update Counters
            file_num = file_num+1
            print_status(f'Extracted {EVM_task}',path)
            P.cntSTAIPlaced += 1                          
            ##Update Participant Data
            taskData = pd.concat([taskData,df])

            #########################
            P.cntSTAIFiles += 1
        
        ## Account for Issues
        except Exception as z:
            print_status(f'Failed {EVM_task}',f"{path} - {z!r}")
            errors = errors + [f"Failed {EVM_task}: {path} - {z!r}",]
            
    if P.cntSTAIPlaced:
        ## Set condition
        taskData = taskData.dropna(subset=['Response','Polarity'])

        taskData.loc[taskData['Polarity']==1, 'Corrected'] = taskData.loc[taskData['Polarity']==1]['Response']
        taskData.loc[taskData['Polarity']==-1, 'Corrected'] = 10-taskData.loc[taskData['Polarity']==-1]['Response']
        taskData['Condition']= 'Baseline'
        ## Initialise row entry
        calc = {'Participant':P.ID,
                'Drink':P.drink,}

        #Process STAI Data
        #for each condition
        conditions = taskData['Condition'].drop_duplicates()
        for condition in conditions:
            _condition = taskData.loc[taskData['Condition']==condition]
            _calc= {f'psy_{EVM_task}_Trait-Score':_condition['Corrected'].mean()}
            merge_dict(_calc,calc)

            _calc={f'time_{EVM_task}-Duration':taskData.t.values[-1] - taskData.t.values[0]}
            merge_dict(_calc,calc)

            #########################
            P.cntSTAIProcessed += 1
            
            
        ### Update Participant Results
        P.psychoPyResults = pd.concat([P.psychoPyResults, pd.DataFrame([calc])], ignore_index=True)

        # Update timings
        taskName = 'STAI'
        timings = {'Task':taskName,
                'Start':taskData.t.values[0],
                'End':taskData.t.values[-1],
                'Duration':taskData.t.values[-1] - taskData.t.values[0]}
        P.tasksInfo = P.tasksInfo.reset_index(drop=True)
        if taskName in P.tasksInfo['Task'].tolist():
            P.tasksInfo.loc[P.tasksInfo['Task']==taskName] = timings
        else:
            P.tasksInfo.loc[len(P.tasksInfo)] = timings
        P.tasksInfo = P.tasksInfo.sort_values('Start')
        P.tasksInfo.to_csv(f"{out_path}{P.ID}_EVM_log.csv",index=False)
    
        ### Update EVM Results File
        if os.path.exists(f"{out_folder}EVM_Results.csv"):
            old_log = pd.read_csv(f"{out_folder}EVM_Results.csv")
            #If participant already has data in clean log
            if P.ID in old_log['Participant'].tolist():
                for key, value in calc.items():
                    old_log.loc[old_log['Participant']==P.ID, key] = value
                new_log = old_log
            else:
                new_log = pd.concat([old_log, pd.DataFrame([calc])], ignore_index=True)
            new_log.to_csv(f"{out_folder}EVM_Results.csv",index=False)
        else:
            EVM_Results = pd.DataFrame()
            EVM_Results = pd.concat([EVM_Results, pd.DataFrame([calc])], ignore_index=True)
            EVM_Results.to_csv(f"{out_folder}EVM_Results.csv",index=False)

    ### Update Log File

    log = {'Participant':P.ID,
            'Date':P.date,
            'Status':P.status,      
            'PsychoPy':P.isData,
            'PsychoPyComplete':P.isDataComplete,
            'Task':EVM_task,
            'Files':P.cntSTAIFiles,
            'Placed':P.cntSTAIPlaced,
            'Processed':P.cntSTAIProcessed,
            'Errors':';'.join(errors)
            }

    if os.path.exists(f"{out_folder}psychoPy_log.csv"):
        old_log = pd.read_csv(f"{out_folder}psychoPy_log.csv")
        #If participant already has data in clean log
        if (P.ID in old_log['Participant'].tolist()) and (old_log.loc[old_log['Participant']==P.ID]['Task'].str.contains(EVM_task).any()):
            for key, value in log.items():
                old_log.loc[(old_log['Participant']==P.ID)&(old_log['Task'].str.contains(EVM_task)), key] = value
            new_log = old_log
        else:
            new_log = pd.concat([old_log, pd.DataFrame([log])], ignore_index=True)
        new_log.to_csv(f"{out_folder}psychoPy_log.csv",index=False)
    else:
        psychoPy_log = pd.DataFrame()
        psychoPy_log = pd.concat([psychoPy_log, pd.DataFrame([log])], ignore_index=True)
        psychoPy_log.to_csv(f"{out_folder}psychoPy_log.csv",index=False)
    
    #End

    #############################
    print_status(f"Collected {EVM_task}",f"{int(P.cntSTAIProcessed*100)}%")  
    return P



def update_BMS(self):
    self.cntBMSFiles = 0
    self.cntBMSPlaced = 0
    self.cntBMSProcessed = 0


def get_BMS(in_folder, out_folder,P):
    '''
    This function cleans and combines all the OpenSignals data from one participant into a single file

    Parameters
    __________
    in_folder: str
        Folder containing directories for each participant
        Must contain ONLY particpant 
    out_folder: str
        Folder in which a directory will be created per participant
        
    TODO:
    - Add in check for duplicate participant IDs
    '''
    
    
    #Define variables
    psychoPy_log = pd.DataFrame()

    errors = []
    in_path = f"{in_folder}{P.ID}/"
    out_path = f"{out_folder}{P.ID}/"
    os.makedirs(out_path, exist_ok=True) 
    
    EVM_Results = pd.DataFrame()
    EVM_task = 'BMS'

    factors = {
            "Alert\nIsilumkiso\nWaaksaam": 'Vigour',
            "Angry\nUnomsindo\nKwaad": 'Anger',
            "Annoyed\nUkudikwa\nGeïrriteerd": 'Anger',
            "Anxious\nUvalo\nAngstig": 'Anxious',
            "Bitter\nInqala\nBitter": 'Anger',
            "Calm\nUzolile\nKalm": 'Calm',
            "Cheerful\nUvuyo\nVrolik": 'Happy',
            "Composed\nYakhiwe\nSaamgesteld": 'Calm',
            "Confused\nUxakekileyo\nVerward": 'Confused',
            "Contented\nWanelisekile\nTevrede": 'Happy',
            "Depressed\nQumbile\nDepressief": 'Depression',
            "Downhearted\nNdidakumbile\nNeerslagtig": 'Depression',
            "Energetic\nSemandleni\nEnergiek": 'Vigour',
            "Exhausted\nNdiniwe\nUitgeput": 'Fatigue',
            "Happy\nWonwabile\nGelukkig": 'Happy',
            "Lively\nWonwabile\nLewendig": 'Vigour',
            "Miserable\nUqumbile\nEllendig": 'Depression',
            "Nervous\nUvalo\nSenuweeagtig": 'Anxious',
            "Panicky\nPhakuzela\nPaniekerig": 'Anxious',
            "Relaxed\nKhululekile\nOntspanne": 'Calm',
            "Restful\nPhumlile\nRustig": 'Calm',
            "Satisfied\nWanelisekile\nTevrede": 'Happy',
            "Sleepy\nUyozela\nSlaperig": 'Fatigue',
            "Tired\nNdiniwe\nMoeg": 'Fatigue',
            "Uncertain\nUkungaqiniseki\nOnseker": 'Confused',
            "Unhappy\nAwonwabanga\nOngelukkig": 'Depression',
            "Worn-out\nUkudinwa\nAfgemat": 'Fatigue',
            "Worried\nUkhathazekile\nBekommerd": 'Anxious',
            "Mixed-up\nKuphithene\nVerward": 'Confused',
            "Muddled\nUphithene\nDeurmekaar": 'Confused',
            "Active\nEsebenzayo\nAktief": 'Vigour',
            "Bad tempered\nUmoya ocaphukileyo\nSlegte humeur": 'Anger',
            "I understand the instructions": np.nan,
    }

    ##Reset Flags
    P.cntBMSFiles = 0
    P.cntBMSPlaced = 0
    P.cntBMSProcessed = 0
    
    taskData=pd.DataFrame()
    print_step(f'Collecting {EVM_task}',P.ID)
    ##### BMS Data
    ##Extract data from files

    files = get_files(in_path,tags=['.csv',EVM_task])
    files = [f for f in files if 'Comp' not in f]
    files = [f for f in files if 'EVM' not in f]
    #For each file
    file_num = 0
    for file in files:                   
        try:
            path = in_path + file                
            f=open(path)                    
            df = pd.read_csv(path, low_memory = False)
            df=df[['Time','Question','Response']]
            df['Path']=path
            df['File']=file_num

            ##Adjust data
            for q in df['Question']:
                df.loc[df['Question'] == q, 'Factor'] = factors[q]

            #create absolute time column
            df['Time']=df['Time'].str.replace(',','.')
            df['t']=[get_sec(i) for i in df['Time'].tolist()]
            df['Task'] = '_'.join(file.split('_')[1:3])

            ##Update Counters
            file_num = file_num+1
            print_status(f'Extracted {EVM_task}',path)
            P.cntBMSPlaced += 1                          
            ##Update Participant Data
            taskData = pd.concat([taskData,df])

            #########################
            P.cntBMSFiles += 1
        
        ## Account for Issues
        except Exception as z:
            print_status(f'Failed {EVM_task}',f"{path} - {z!r}")
            errors = errors + [f"Failed {EVM_task}: {path} - {z!r}",]
            
    if P.cntBMSPlaced:
        ## Set condition
        taskData = taskData.dropna(subset=['Response','Factor'])

        taskData.loc[taskData['Task'].str.contains('_01'), 'Condition']='PassiveEnd'
        taskData.loc[taskData['Task'].str.contains('_02'), 'Condition']='PostExercise'

        if len(taskData.loc[taskData['Task'].str.contains('_NS')])>0:
            taskData.loc[taskData['Task'].str.contains('_NS'), 'Condition']='Baseline'
            taskData.loc[taskData['Task'].str.contains('_00'), 'Condition']='PreNeedstate'
        else:
            taskData.loc[taskData['Task'].str.contains('_00'), 'Condition']='Baseline'
            
        ###Process GoNoGo Data 
        ## Initialise row entry
        calc = {'Participant':P.ID,
                'Drink':P.drink,}

        #Process BMS Data
        #for each condition
        conditions = taskData['Condition'].drop_duplicates()
        for condition in conditions:
            _condition = taskData.loc[taskData['Condition']==condition]
            #axes = list(_condition['Axis'].drop_duplicates())
            factors = list(_condition['Factor'].drop_duplicates())
            factor_scores = {}
            for f in factors:
                score = _condition.loc[_condition['Factor']==f]['Response'].sum()
                factor_scores[f]=score
                _calc= {f'psy_{EVM_task}_{condition}_{f}-Score':score}
                merge_dict(_calc,calc)
            _calc = {f'time_{EVM_task}_{condition}_Start':_condition['t'].values[0]}
            merge_dict(_calc,calc)

            _calc={f'time_{EVM_task}_{condition}-Duration':_condition.t.values[-1] - _condition.t.values[0]}
            merge_dict(_calc,calc)
            #########################
            P.cntBMSProcessed += 1

            # Update timings
            taskName = _condition['Task'].values[0]
            timings = {'Task':taskName,
                    'Start':_condition.t.values[0],
                    'End':_condition.t.values[-1],
                    'Duration':_condition.t.values[-1] - _condition.t.values[0]}
            P.tasksInfo = P.tasksInfo.reset_index(drop=True)
            if taskName in P.tasksInfo['Task'].tolist():
                P.tasksInfo.loc[P.tasksInfo['Task']==taskName, 'Start'] = _condition.t.values[0]
                P.tasksInfo.loc[P.tasksInfo['Task']==taskName, 'End'] = _condition.t.values[-1]
                P.tasksInfo.loc[P.tasksInfo['Task']==taskName, 'Duration'] = _condition.t.values[-1] - _condition.t.values[0]
            else:
                P.tasksInfo.loc[len(P.tasksInfo)] = timings
            P.tasksInfo = P.tasksInfo.sort_values('Start')
            P.tasksInfo.to_csv(f"{out_path}{P.ID}_EVM_log.csv",index=False)
            
            
        ### Update Participant Results
        P.psychoPyResults = pd.concat([P.psychoPyResults, pd.DataFrame([calc])], ignore_index=True)
    
        ### Update EVM Results File
        if os.path.exists(f"{out_folder}EVM_Results.csv"):
            old_log = pd.read_csv(f"{out_folder}EVM_Results.csv")
            #If participant already has data in clean log
            if P.ID in old_log['Participant'].tolist():
                for key, value in calc.items():
                    old_log.loc[old_log['Participant']==P.ID, key] = value
                new_log = old_log
            else:
                new_log = pd.concat([old_log, pd.DataFrame([calc])], ignore_index=True)
            new_log.to_csv(f"{out_folder}EVM_Results.csv",index=False)
        else:
            EVM_Results = pd.DataFrame()
            EVM_Results = pd.concat([EVM_Results, pd.DataFrame([calc])], ignore_index=True)
            EVM_Results.to_csv(f"{out_folder}EVM_Results.csv",index=False)

    ### Update Log File

    log = {'Participant':P.ID,
            'Date':P.date,
            'Status':P.status,      
            'PsychoPy':P.isData,
            'PsychoPyComplete':P.isDataComplete,
            'Task':EVM_task,
            'Files':P.cntBMSFiles,
            'Placed':P.cntBMSPlaced,
            'Processed':P.cntBMSProcessed,
            'Errors':';'.join(errors)
            }

    if os.path.exists(f"{out_folder}psychoPy_log.csv"):
        old_log = pd.read_csv(f"{out_folder}psychoPy_log.csv")
        #If participant already has data in clean log
        if (P.ID in old_log['Participant'].tolist()) and (old_log.loc[old_log['Participant']==P.ID]['Task'].str.contains(EVM_task).any()):
            for key, value in log.items():
                old_log.loc[(old_log['Participant']==P.ID)&(old_log['Task'].str.contains(EVM_task)), key] = value
            new_log = old_log
        else:
            new_log = pd.concat([old_log, pd.DataFrame([log])], ignore_index=True)
        new_log.to_csv(f"{out_folder}psychoPy_log.csv",index=False)
    else:
        psychoPy_log = pd.DataFrame()
        psychoPy_log = pd.concat([psychoPy_log, pd.DataFrame([log])], ignore_index=True)
        psychoPy_log.to_csv(f"{out_folder}psychoPy_log.csv",index=False)
    
    #End

    #############################
    print_status(f"Collected {EVM_task}",f"{int(P.cntBMSProcessed/3*100)}%")  
    return P


def update_manual(self):
    self.cntManualFiles = 0
    self.cntManualPlaced = 0
    self.cntManualProcessed = 0
    self.BMI = None
    self.Weight = None
    self.Height = None
    self.BodyFat = None
    self.Musclemass = None
    self.VisceralFat = None


def get_manual(in_folder, out_folder,P):
    '''
    This function cleans and combines all the OpenSignals data from one participant into a single file

    Parameters
    __________
    in_folder: str
        Folder containing directories for each participant
        Must contain ONLY particpant 
    out_folder: str
        Folder in which a directory will be created per participant
        
    TODO:
    - Add in check for duplicate participant IDs
    '''
    
    
    #Define variables
    psychoPy_log = pd.DataFrame()

    errors = []
    in_path = f"{in_folder}{P.ID}/"
    out_path = f"{out_folder}{P.ID}/"
    os.makedirs(out_path, exist_ok=True) 
    
    EVM_Results = pd.DataFrame()
    EVM_task = 'manualCapture'

    ##Reset Flags
    P.cntManualFiles = 0
    P.cntManualPlaced = 0
    P.cntManualProcessed = 0
    
    taskData=pd.DataFrame()
    print_step(f'Collecting {EVM_task}',P.ID)
    ##### EmotionRecognition Data
    ##Extract data from files
    files = get_files(in_path,tags=['.csv', 'manual']) #Get list of all goNogo tasks
    #For each file
    file_num = 0
    for f in files:                   
        try:
            path = in_path + f                 
            f=open(path)                    
            df = pd.read_csv(path, low_memory = False)
            df=df[['Time', 'Event', 'Weight','Height','Bodyfat','Musclemass','Visceral','BMI']]
            df['Path']=path
            df['File']=file_num

            #create absolute time column
            df['Time']=df['Time'].str.replace(',','.')
            df['t']=[get_sec(i) for i in df['Time'].tolist()]

            ##Update Counters
            file_num = file_num+1
            print_status(f'Extracted {EVM_task}',path)
            P.cntManualPlaced += 1

            # ##Contextualise Data
            # #set task from EVM list
            
            # for index, row in df.iterrows():
            #     try:
            #         t = row['t']
            #         task = P.tasksInfo.loc[(P.tasksInfo['Start']<=t) & (P.tasksInfo['End']>=t)]['Task'].tolist()[0]
            #         df.loc[index, 'Task']=task
            #         print_status('Placed Data',task)

            #         #########################
            #         P.cntManualPlaced += 1
            #     except IndexError:
            #         print_status('Unplaced Data',path)
            #     except Exception as z:
            #         print_status(f'Failed Placing {EVM_task} Data',f"{path} - {z!r}")
            #         errors = errors + [f"Failed Placing {EVM_task} Data: {path} - {z!r}",]
                          
            ##Update Participant Data
            taskData = pd.concat([taskData,df])

            #########################
            P.cntManualFiles += 1
        
        ## Account for Issues
        except KeyError:
            try:
                df=df[['Time', 'Event','BMI','Weight','Height','Bodyfat','Musclemass','Visceral','BMI']]
                df['Path']=path
                df['File']=file_num

                #create absolute time column
                df['Time']=df['Time'].str.replace(',','.')
                df['t']=[get_sec(i) for i in df['Time'].tolist()]

                ##Update Counters
                file_num = file_num+1
                print_status(f'Extracted {EVM_task}',path)

                ##Update Participant Data
                taskData = pd.concat([taskData,df])

                #########################
                P.cntManualFiles += 1
            except:
                print_status(f'Found redundant',f"{path}")

        except Exception as z:
            print_status(f'Failed {EVM_task}',f"{path} - {z!r}")
            errors = errors + [f"Failed {EVM_task}: {path} - {z!r}",]
            
    if P.cntManualPlaced:
        ## Set condition

        ### Set delay time
        ind = pd.read_csv(f'{out_folder}file_check.csv')
        data = ind.loc[(ind['Number']==int(P.number)) & ((ind['Drink']==P.drink))]
        try:
            scheduled_time = get_sec(f"{data['Time'].values[0].split('-')[0].strip()}:00")
            actual_time = get_sec(data['Actual time'].values[0])
            P.delay = actual_time-scheduled_time
            P.cntManualProcessed += 1
        except:
            P.delay = np.nan
            print_status('Missing','Time Delay')

        try:
            P.Trial = data['Trial'].values[0]
            P.cntManualProcessed += 1
        except:
            print_status('Missing','Trial')



        ########

        try:
            BMI = taskData.loc[taskData['Event']=='BMI']['BMI'].tolist()
            BMI = [b for b in BMI if ((b<=50) and (b>0))]
            P.BMI = max(BMI)
            P.cntManualProcessed += 1
        except:
            print_status('Missing','BMI')

        try:
            Weight = taskData.loc[taskData['Event']=='BMI']['Weight'].tolist()
            Weight = [b for b in Weight if ((b<=200) and (b>0))]
            P.Weight = max(Weight)
            P.cntManualProcessed += 1
        except:
            print_status('Missing','Weight')

        try:
            Height = taskData.loc[taskData['Event']=='BMI']['Height'].tolist()
            Height = [b for b in Height if ((b<=300) and (b>0))]
            P.Height = max(Height)
            P.cntManualProcessed += 1
        except:
            print_status('Missing','Height')

        try:
            Bodyfat = taskData.loc[taskData['Event']=='BMI']['Bodyfat'].tolist()
            Bodyfat = [b for b in Bodyfat if ((b<=100) and (b>0))]
            P.BodyFat = max(Bodyfat)
            P.cntManualProcessed += 1
        except:
            print_status('Missing','BodyFat')

        try:
            Musclemass = taskData.loc[taskData['Event']=='BMI']['Musclemass'].tolist()
            Musclemass = [b for b in Musclemass if ((b<=100) and (b>0))]
            P.Musclemass = max(Musclemass)
            P.cntManualProcessed += 1
        except:
            print_status('Missing','Musclemass')

        try:
            Visceral = taskData.loc[taskData['Event']=='BMI']['Visceral'].tolist()
            Visceral = [b for b in Visceral if ((b<=10) and (b>0))]
            P.VisceralFat = max(Visceral)
            P.cntManualProcessed += 1
        except:
            print_status('Missing','Visceral')


        ################ Get demographic data

        ind = pd.read_csv(f'{out_folder}file_check.csv')
        try:
            Visceral = taskData.loc[taskData['Event']=='BMI']['Visceral'].tolist()
            Visceral = [b for b in Visceral if ((b<=10) and (b>0))]
            P.VisceralFat = max(Visceral)
            P.cntManualProcessed += 1
        except:
            print_status('Missing','Visceral')

        ## Initialise row entry
        calc = {'Participant':P.ID,
                'Drink':P.drink,
                'study_delay':P.delay,
                'manual_BMI':P.BMI,
                'manual_Weight':P.Weight,
                'manual_Height':P.Height,
                'manual_BodyFat':P.BodyFat,
                'manual_Musclemass':P.Musclemass,
                'manual_VisceralFat':P.VisceralFat}
        
        # Load blood pressure data
        BP = pd.read_csv(f'{in_path}{P.ID}_BP.csv')

        BP = BP.drop_duplicates(subset=['Task','Systolic','Diastolic'], keep='first')
        BP = BP.drop_duplicates(subset=['Task'], keep='first')

        # Extract 'Number' from the 'Task' column (e.g., "BP_05" → 5)
        BP['Task1'] = BP['Task'].replace('BP_NS','BP_08')
        BP['Number'] = BP['Task1'].apply(lambda x: int(x.split('_')[1]))

        # Set 'Number' as the index
        BP = BP.set_index('Number')

        # Baseline values
        baseline_systolic = BP.loc[0, 'Systolic']
        baseline_diastolic = BP.loc[0, 'Diastolic']

        # Extract first 5 measurements and compute stats
        try:
            for index, row in BP.iterrows():
                _calc = {
                    f'manual_{row["Task"]}_Systolic_Absolute': row.Systolic,
                    f'manual_{row["Task"]}_Diastolic_Absolute': row.Diastolic,
                }
                merge_dict(_calc, calc)
        except:
            print_status('Missing BP','1-4')

        # taskData = taskData.dropna(subset=['Task'])
        # taskData.loc[taskData['Task'].str.contains('_01'), 'Condition']='Baseline'
        # taskData.loc[taskData['Task'].str.contains('_02'), 'Condition']='BeforeTreatment'
        # taskData.loc[taskData['Task'].str.contains('_03'), 'Condition']='TreatmentInitial'
        # taskData.loc[taskData['Task'].str.contains('_04'), 'Condition']='TreatmentBeforeTasks'
        # taskData.loc[taskData['Task'].str.contains('_05'), 'Condition']='TreatmentAfterTasks'
    
      
        # nb = 0
        # ## Get Baseline
        # try:
        #     bSys = taskData.loc[(taskData['Condition']=='Baseline')& (taskData['Event']=='BP')]['Systolic'].tolist()[0]
        #     bDias = taskData.loc[(taskData['Condition']=='Baseline')& (taskData['Event']=='BP')]['Diastolic'].tolist()[0]
        # except Exception as z:
        #     print_status(f'Failed {EVM_task} Baseline',f"{z!r}")
        #     errors = errors + [f"Failed {EVM_task} Baseline:{z!r}",]
        #     nb = 1

        
        # ## Results
        # cort = pd.read_csv(f"{in_folder}cortisol.csv")
        # corti = 0
        # bCort = 0
        # cortR = 0

        #for each condition
        # conditions = taskData['Condition'].drop_duplicates()
        # for condition in conditions:
            
        #     _condition = taskData.loc[taskData['Condition']==condition]
        #     try:
        #         corti = cort.loc[cort['Participant']==P.ID][condition].tolist()[0]
        #         bCort = _cort = cort.loc[cort['Participant']==P.ID]['Baseline'].tolist()[0]
        #         cortR = (corti-bCort)#/bCort*100
        #         P.cntManualProcessed += 1
        #     except Exception as z:
        #         corti = np.nan
        #         print_status(f'Failed {EVM_task} Cort',f"{z!r}")
        #         errors = errors + [f"Failed {EVM_task} Cort:{z!r}",]

        # This was indented
        # _sys = _condition['Systolic'].dropna().tolist()[0]
        # _dias = _condition['Diastolic'].dropna().tolist()[0]
        # _sysR =  np.nan if nb else (_sys-bSys)#/bSys*100
        # _diasR = np.nan if nb else (_dias-bDias)#/bDias*100
        #Update row entry
        # _calc = {f"psy_{EVM_task}_{condition}_Systolic_Abs":_sys,
        #             f"psy_{EVM_task}_{condition}_Diastolic_Abs":_dias,
        #             f"psy_{EVM_task}_{condition}_Systolic_Relative":_sysR,
        #             f"psy_{EVM_task}_{condition}_Diastolic_Relative":_diasR,
        #             f"psy_{EVM_task}_{condition}_Cortisol_Abs":corti,
        #             f"psy_{EVM_task}_{condition}_Cortisol_Relative":cortR,
        #         }
        #merge_dict(_calc,calc)


        #########################
        #P.cntManualProcessed += 1

        ### Update Participant Results
        P.psychoPyResults = pd.concat([P.psychoPyResults, pd.DataFrame([calc])], ignore_index=True)
    
        ### Update EVM Results File
        if os.path.exists(f"{out_folder}EVM_Results.csv"):
            old_log = pd.read_csv(f"{out_folder}EVM_Results.csv")
            #If participant already has data in clean log
            if P.ID in old_log['Participant'].tolist():
                for key, value in calc.items():
                    old_log.loc[old_log['Participant']==P.ID, key] = value
                new_log = old_log
            else:
                new_log = pd.concat([old_log, pd.DataFrame([calc])], ignore_index=True)
            new_log.to_csv(f"{out_folder}EVM_Results.csv",index=False)
        else:
            EVM_Results = pd.DataFrame()
            EVM_Results = pd.concat([EVM_Results, pd.DataFrame([calc])], ignore_index=True)
            EVM_Results.to_csv(f"{out_folder}EVM_Results.csv",index=False)

    ### Update Log File

    log = {'Participant':P.ID,
            'Date':P.date,
            'Status':P.status,      
            'PsychoPy':P.isData,
            'PsychoPyComplete':P.isDataComplete,
            'Task':EVM_task,
            'Files':P.cntManualFiles,
            'Placed':P.cntManualPlaced,
            'Processed':P.cntManualProcessed,
            'Errors':';'.join(errors)
            }

    if os.path.exists(f"{out_folder}psychoPy_log.csv"):
        old_log = pd.read_csv(f"{out_folder}psychoPy_log.csv")
        #If participant already has data in clean log
        if (P.ID in old_log['Participant'].tolist()) and (old_log.loc[old_log['Participant']==P.ID]['Task'].str.contains(EVM_task).any()):
            for key, value in log.items():
                old_log.loc[(old_log['Participant']==P.ID)&(old_log['Task'].str.contains(EVM_task)), key] = value
            new_log = old_log
        else:
            new_log = pd.concat([old_log, pd.DataFrame([log])], ignore_index=True)
        new_log.to_csv(f"{out_folder}psychoPy_log.csv",index=False)
    else:
        psychoPy_log = pd.DataFrame()
        psychoPy_log = pd.concat([psychoPy_log, pd.DataFrame([log])], ignore_index=True)
        psychoPy_log.to_csv(f"{out_folder}psychoPy_log.csv",index=False)
    
    #End

    #############################
    print_status(f"Collected {EVM_task}",f"{int(P.cntManualProcessed/7*100)}%")  
    return P


def update_Temp(self):
    self.cntTempFiles = 0
    self.cntTempPlaced = 0
    self.cntTempProcessed = 0

def get_Temp(in_folder, out_folder,P):
    '''
    This function cleans up Temp data
    At this point, no files need to be read in. All data needed is contained in the participants pickle

    Parameters
    __________
    in_folder: str
        Folder containing directories for each participant
        Must contain ONLY particpant 
    out_folder: str
        Folder in which a directory will be created per participant
        
    TODO:
    - Add in check for duplicate participant IDs
    '''
    
    def adc_to_temperature_celsius(adc_value, vcc=3.0, n_bits=16):
        a0 = 1.12764514e-3
        a1 = 2.34282709e-4
        a2 = 8.77303013e-8
        R0 = 1e4  # 10kΩ resistor

        # Avoid division by zero
        ntc_voltage = (adc_value * vcc) / (2**n_bits)
        ntc_voltage = np.clip(ntc_voltage, 1e-6, vcc - 1e-6)

        ntc_resistance = R0 * ntc_voltage / (vcc - ntc_voltage)

        ln_r = np.log(ntc_resistance)
        inv_temp_kelvin = a0 + a1 * ln_r + a2 * (ln_r ** 3)
        temp_kelvin = 1 / inv_temp_kelvin
        temp_celsius = temp_kelvin - 273.15

        return np.round(temp_celsius, 1)
    
    #Define variables
    psychoPy_log = pd.DataFrame()

    errors = []
    out_path = f"{out_folder}{P.ID}/"
    os.makedirs(out_path, exist_ok=True) 
    
    EVM_Results = pd.DataFrame()
    EVM_task = 'Temp'

    ##Reset Flags
    P.cntTempFiles = 0
    P.cntTempPlaced = 0
    P.cntTempProcessed = 0
    
    print_step(f'Collecting {EVM_task}',P.ID)

    raw_data = pd.DataFrame()

    if P.isSignals:
        channels = [c for c in P.signals.columns if 'TEMP' in c]
        for c in channels:
            result = pd.DataFrame()
            _data = P.signals.dropna(subset=[c])
            result.loc[:, 'Freq']= _data['Freq']
            result.loc[:, 'File']= _data['File']
            result.loc[:, 't']= _data['t']
            result.loc[:, EVM_task]= adc_to_temperature_celsius(_data[c])
            raw_data = pd.concat([raw_data,result])
            P.cntTempFiles = 1

    ### Only work on task data

    tasks_of_interest = ['blank']
    if P.isSignals & P.isData:
        tasks = P.tasksInfo[['Task','Start','End']]
        tasks = tasks.loc[tasks['Task'].str.contains('|'.join(tasks_of_interest))]

        tasks = tasks.sort_values(by='Start')
        #### Do all the writing for each task
        results = pd.DataFrame()
        for task in tasks['Task']:
                try:
                    ### Locate task data               
                    info = tasks.loc[tasks['Task']==task]
                    min_t = int(info['Start'].to_numpy()[0])
                    max_t = int(info['End'].to_numpy()[0])

                    data = pd.DataFrame()
                    data = raw_data.loc[(raw_data['t']>=min_t) & (raw_data['t']<=max_t)]
                    data = data.sort_values(by='t')
                    data.loc[:, 'Task']=task
                    results = pd.concat([results, data], ignore_index=True)

                    P.cntTempPlaced +=1

                except Exception as z:
                        print_status(f'Failed Temp on {task}', f"{z!r}")
                        errors = errors + [f'Failed Temp on {task}', f"{z!r}",]


    if P.cntTempPlaced:
        ## Set condition
        calc = {'Participant':P.ID,
                'Drink':P.drink,}

        for task,df in results.groupby('Task'):
            try:
                _calc = {f"os_Temperature_{task}_mean":df['Temp'].mean()}
                merge_dict(_calc,calc)
                P.cntTempProcessed += 1

            except Exception as z:
                print_status(f'Failed Calcs on {task}', f"{z!r}")
                errors = errors + [f'Failed Calcs on {task}', f"{z!r}",]

        ### Update Participant Results
        P.psychoPyResults = pd.concat([P.psychoPyResults, pd.DataFrame([calc])], ignore_index=True)
    
        ### Update EVM Results File
        if os.path.exists(f"{out_folder}EVM_Results.csv"):
            old_log = pd.read_csv(f"{out_folder}EVM_Results.csv")
            #If participant already has data in clean log
            if P.ID in old_log['Participant'].tolist():
                for key, value in calc.items():
                    old_log.loc[old_log['Participant']==P.ID, key] = value
                new_log = old_log
            else:
                new_log = pd.concat([old_log, pd.DataFrame([calc])], ignore_index=True)
            new_log.to_csv(f"{out_folder}EVM_Results.csv",index=False)
        else:
            EVM_Results = pd.DataFrame()
            EVM_Results = pd.concat([EVM_Results, pd.DataFrame([calc])], ignore_index=True)
            EVM_Results.to_csv(f"{out_folder}EVM_Results.csv",index=False)

    ### Update Log File

    log = {'Participant':P.ID,
            'Date':P.date,
            'Status':P.status,      
            'PsychoPy':P.isData,
            'PsychoPyComplete':P.isDataComplete,
            'Task':EVM_task,
            'Files':P.cntTempFiles,
            'Placed':P.cntTempPlaced,
            'Processed':P.cntTempProcessed,
            'Errors':';'.join(errors)
            }

    if os.path.exists(f"{out_folder}psychoPy_log.csv"):
        old_log = pd.read_csv(f"{out_folder}psychoPy_log.csv")
        #If participant already has data in clean log
        if (P.ID in old_log['Participant'].tolist()) and (old_log.loc[old_log['Participant']==P.ID]['Task'].str.contains(EVM_task).any()):
            for key, value in log.items():
                old_log.loc[(old_log['Participant']==P.ID)&(old_log['Task'].str.contains(EVM_task)), key] = value
            new_log = old_log
        else:
            new_log = pd.concat([old_log, pd.DataFrame([log])], ignore_index=True)
        new_log.to_csv(f"{out_folder}psychoPy_log.csv",index=False)
    else:
        psychoPy_log = pd.DataFrame()
        psychoPy_log = pd.concat([psychoPy_log, pd.DataFrame([log])], ignore_index=True)
        psychoPy_log.to_csv(f"{out_folder}psychoPy_log.csv",index=False)

    #############################
    print_status(f"Collected {EVM_task}",f"{int(P.cntTempProcessed/6*100)}%")  
    return P


def update_ECG(self):
    self.cntECGFiles = 0
    self.cntECGPlaced = 0
    self.cntECGProcessed = 0


def get_ECG(in_folder, out_folder,P):
    '''
    This function cleans up ECG data, and then devides it into epochs. 
    At this point, no files need to be read in. All data needed is contained in the participants pickle


    Parameters
    __________
    in_folder: str
        Folder containing directories for each participant
        Must contain ONLY particpant 
    out_folder: str
        Folder in which a directory will be created per participant
        
    TODO:
    - Add in check for duplicate participant IDs
    '''
    
    
    #Define variables
    psychoPy_log = pd.DataFrame()

    errors = []
    out_path = f"{out_folder}{P.ID}/"
    os.makedirs(out_path, exist_ok=True) 
    
    EVM_Results = pd.DataFrame()
    EVM_task = 'ECG'

    ##Reset Flags
    P.cntECGFiles = 0
    P.cntECGPlaced = 0
    P.cntECGProcessed = 0
    
    print_step(f'Collecting {EVM_task}',P.ID)

    raw_data = pd.DataFrame()

    if P.isSignals:
        channels = [c for c in P.signals.columns if EVM_task in c]
        for c in channels:
            result = pd.DataFrame()
            _data = P.signals.dropna(subset=[c])
            result.loc[:, 'Freq']= _data['Freq']
            result.loc[:, 'File']= _data['File']
            result.loc[:, 't']= _data['t']
            result.loc[:, EVM_task]= _data[c]
            raw_data = pd.concat([raw_data,result])
            P.cntECGFiles = 1

    #initialise filter params
    order = 4
    cutoffs = (5,15)  # desired cutoff frequency of the filter, Hz
    epoch_length = 181
    hr_length = 10

    min_threshold = 50 #BPM
    max_threshold = 180 #BPM
    ##### Get Task Data
    tasks_of_interest = ['blank','simpleRT','Exercise']

    ### Only work on task data
    if P.isSignals & P.isData:
        tasks = P.tasksInfo[['Task','Start','End']]
        tasks = tasks.loc[tasks['Task'].str.contains('|'.join(tasks_of_interest))]

        tasks = tasks.sort_values(by='Start')
        #### Do all the writing for each task
        results = pd.DataFrame()
        for task in tasks['Task']:
                try:
                    ### Locate task data               
                    info = tasks.loc[tasks['Task']==task]
                    min_t = int(info['Start'].to_numpy()[0])
                    max_t = int(info['End'].to_numpy()[0])

                    data = pd.DataFrame()
                    data = raw_data.loc[(raw_data['t']>=min_t) & (raw_data['t']<=max_t)]
                    data = data.sort_values(by='t')
                    start_time = data['t'].values[0]
                    data.loc[:, 'TaskTime']=data['t']-start_time
                    data.loc[:, 'Peaks']=0
                    data.loc[:, 'Task']=task
            
                    dataFiles = data['File'].drop_duplicates().tolist()

                    ####### Get Peaks

                    ### For each dataFile, do things seperately, due to filtering and continuity effects
                    for f in dataFiles:
                        try:
                            file = data.loc[data['File']==f].sort_values('t')  
                            freq = file['Freq'].values[0]

                            ## Set Epochs
                            time = file['TaskTime'].to_numpy()
                            maxi = time.max()

                            bins = np.arange(0,maxi,hr_length)
                            hr_epoch_col = np.digitize(time,bins)

                            file.loc[:, 'HREpoch']=hr_epoch_col

                            gb = file.groupby('HREpoch')

                            #for each epoch
                            for x in gb.groups:

                                hr_epoch = gb.get_group(x)
                                hr_epoch = hr_epoch.dropna(subset=[EVM_task])
                    
                                if len(hr_epoch):
                                    raw = hr_epoch[EVM_task].to_numpy().astype(float)    
                                 
                                    ### Filter data based on findings from 'Stress detection using ECG and EMG signals: A comprehensive study'
                                    filtered = butter_bandpass_filter(raw, cutoffs, hr_epoch['Freq'].values[0], order)

                                    #Get Peak detection parameters
                                    height = filtered.std()*1.1
                                    dist = hr_epoch['Freq'].values[0]/(160/60)                    
                                    # Get peaks
                                    peaks, _ = find_peaks(filtered, height=height, distance=dist)
                                    peakT = hr_epoch.iloc[peaks]['TaskTime']
                                    signalT = hr_epoch.iloc[peaks]['t']

                                    file.loc[(file['TaskTime'].isin(peakT)), 'Peaks']= 1
                                    P.signals.loc[(P.signals['t'].isin(signalT)), 'HRPeaks']= 1
                                    #Setting a threshold of 50-180 BPM
                                    start_time = hr_epoch.TaskTime.values[0]
                                    end_time = hr_epoch.TaskTime.values[-1]
                                    if len(peaks)>5 and len(peaks)<18:                        
                                        file.loc[(file['TaskTime']>=start_time) & (file['TaskTime']<=end_time) , 'Trusted']= 1
                                    else:
                                        file.loc[(file['TaskTime']>=start_time) & (file['TaskTime']<=end_time) , 'Trusted']= 0

                                    # #Some code to plot the data
                                    # plt.figure(figsize=(12, 6))

                                    # # Plot the raw ECG data against time.
                                    # plt.plot(hr_epoch['t'], raw, label='Raw ECG Signal')

                                    # # Overlay red dots at the detected peaks.
                                    # # hr_epoch.iloc[peaks]['t'] extracts the time values corresponding to the detected peak indices.
                                    # # raw[peaks] gives the amplitude of the raw signal at those indices.
                                    # plt.plot(hr_epoch.iloc[peaks]['t'], raw[peaks], 'ro', label='Detected Peaks')

                                    # # Label the axes and add a title.
                                    # plt.xlabel('Time')
                                    # plt.ylabel('Amplitude')
                                    # plt.title('ECG Raw Data with Detected Peaks')

                                    # # Add a legend to differentiate the plotted data.
                                    # plt.legend()

                                    # # Display the plot.
                                    # plt.show()

                            P.cntECGPlaced += 1                                          
                            results = pd.concat([results, file], ignore_index=True)
                
                        except IndexError as z:
                                print_status(f'Failed HR Peak Detection on {task}', f"File {f}")
                                errors = errors + [f'Failed HR Peak Detection on {task}', f"File {f}"]

                except Exception as z:
                        print_status(f'Failed HR Peak Detection on {task}', f"{z!r}")
                        errors = errors + [f'Failed HR Peak Detection on {task}', f"{z!r}",]


    if P.cntECGPlaced:
        ## Set condition
        calc = {'Participant':P.ID,
                'Drink':P.drink,}

        for task, _data in results.groupby('Task'):
            # We divide the task up into 3-minute epochs, so we can assess HRV components over a fair duration.
            try:
                dur = _data['TaskTime'].values[-1] - _data['TaskTime'].values[0]
                bins = np.arange(0, dur, epoch_length)
                i = 1
                for b in bins:
                    _data.loc[_data['TaskTime'] >= b, 'Epoch'] = i
                    i += 1

                # Lists to collect metrics from each epoch for trend analysis.
                epoch_indices = []
                meanHR_list = []
                hfPower_list = []
                lfhfRatio_list = []
                rmssd_list = []

                for epoch_index, epoch in _data.groupby('Epoch'):
                    # Get timestamps of detected peaks (assume 'TaskTime' is in seconds)
                    peak_times = epoch.loc[epoch['Peaks'] > 0]['TaskTime'].values

                    # Compute RR intervals (time differences between successive peaks)
                    rr_intervals = np.diff(peak_times)

                    # Calculate instantaneous heart rate (in BPM) from RR intervals
                    instant_hr = 60 / rr_intervals

                    # Create a regular time grid between the first and last peak times (using detected peaks' times)
                    time_grid = np.linspace(peak_times[1], peak_times[-1], len(peak_times))

                    # Interpolate the instantaneous HR values on the regular time grid.
                    interp_func = scipy.interpolate.interp1d(peak_times[1:], instant_hr, kind='linear')
                    bpm = interp_func(time_grid)

                    # Optional: Uncomment to visually inspect the bpm series
                    # plt.plot(time_grid, bpm, label='Interpolated BPM')
                    # plt.xlabel('Time')
                    # plt.ylabel('BPM')
                    # plt.legend()
                    # plt.show()

                    if len(bpm) > 20:
                        # Calculate effective sampling frequency for the interpolated bpm signal.
                        dt = time_grid[1] - time_grid[0]  # Assumes time_grid is uniformly spaced.
                        bpm_fs = 1.0 / dt

                        # Detrend the bpm signal for spectral analysis.
                        bpm_detrended = detrend(bpm)

                        # Compute the mean heart rate.
                        MeanHR = bpm.mean()

                        # Compute power spectral density (PSD) in the low frequency (LF) and high frequency (HF) bands.
                        LF_psd = compute_welch_psd_range(bpm_detrended, bpm_fs, (0.04, 0.15))
                        HF_psd = compute_welch_psd_range(bpm_detrended, bpm_fs, (0.15, 0.4))
                        HFPower = HF_psd / (HF_psd + LF_psd) * 100

                        # Compute the LF/HF ratio (using natural log transformation).
                        try:
                            LFHFRatio = np.log(LF_psd / HF_psd)
                        except ZeroDivisionError:
                            LFHFRatio = np.nan

                        # ----- Correct RMSSD Computation -----
                        # RMSSD is defined as the square root of the mean squared differences of successive RR intervals.
                        if len(rr_intervals) > 1:
                            rr_diff = np.diff(rr_intervals)  # Successive differences of RR intervals.
                            RMSSD = np.sqrt(np.mean(rr_diff**2))
                        else:
                            RMSSD = np.nan

                        # ----- Compute HRGradient for the epoch -----
                        # HRGradient is defined as the slope of the trendline of BPM versus time.
                        if len(time_grid) > 1:
                            hr_gradient = np.polyfit(time_grid, bpm, 1)[0]
                        else:
                            hr_gradient = np.nan

                        # Save the computed metrics for the current epoch.
                        _calc = {
                            f'os_ECG_{task}_E{epoch_index}_MeanHR': MeanHR,
                            f'os_ECG_{task}_E{epoch_index}_HFPower': HFPower,
                            f'os_ECG_{task}_E{epoch_index}_LFHFRatio': LFHFRatio,
                            f'os_ECG_{task}_E{epoch_index}_RMSSD': RMSSD,
                            f'os_ECG_{task}_E{epoch_index}_HRGradient': hr_gradient,
                            f'os_ECG_{task}_E{epoch_index}_Reliability': epoch['Trusted'].sum()/len(epoch)*100,
                        }
                        merge_dict(_calc, calc)
                        P.cntECGProcessed += 1

                        # Append the epoch index and metrics to lists for trend analysis.
                        epoch_indices.append(epoch_index)
                        meanHR_list.append(MeanHR)
                        hfPower_list.append(HFPower)
                        lfhfRatio_list.append(LFHFRatio)
                        rmssd_list.append(RMSSD)

                # ----- Compute the gradient (slope) of the trendline for the entire task -----
                # Use the epoch index as the independent variable. A linear fit (1st degree polynomial) yields a slope.
                if len(epoch_indices) >= 2:
                    x = np.array(epoch_indices)
                    meanHR_gradient = np.polyfit(x, np.array(meanHR_list), 1)[0]
                    hfPower_gradient = np.polyfit(x, np.array(hfPower_list), 1)[0]
                    lfhf_gradient = np.polyfit(x, np.array(lfhfRatio_list), 1)[0]
                    rmssd_gradient = np.polyfit(x, np.array(rmssd_list), 1)[0]

                    # Save the computed trend gradients for the entire task.
                    calc[f'os_ECG_{task}_MeanHR_gradient'] = meanHR_gradient
                    calc[f'os_ECG_{task}_HFPower_gradient'] = hfPower_gradient
                    calc[f'os_ECG_{task}_LFHFRatio_gradient'] = lfhf_gradient
                    calc[f'os_ECG_{task}_RMSSD_gradient'] = rmssd_gradient


                
         
            except Exception as z:
                print_status(f'Failed HR Calcs on {task}', f"{z!r}")
                errors = errors + [f'Failed HR Calcs on {task}', f"{z!r}",]

        ### Update Participant Results
        P.psychoPyResults = pd.concat([P.psychoPyResults, pd.DataFrame([calc])], ignore_index=True)
    
        ### Update EVM Results File
        if os.path.exists(f"{out_folder}EVM_Results.csv"):
            old_log = pd.read_csv(f"{out_folder}EVM_Results.csv")
            #If participant already has data in clean log
            if P.ID in old_log['Participant'].tolist():
                for key, value in calc.items():
                    old_log.loc[old_log['Participant']==P.ID, key] = value
                new_log = old_log
            else:
                new_log = pd.concat([old_log, pd.DataFrame([calc])], ignore_index=True)
            new_log.to_csv(f"{out_folder}EVM_Results.csv",index=False)
        else:
            EVM_Results = pd.DataFrame()
            EVM_Results = pd.concat([EVM_Results, pd.DataFrame([calc])], ignore_index=True)
            EVM_Results.to_csv(f"{out_folder}EVM_Results.csv",index=False)

    ### Update Log File

    log = {'Participant':P.ID,
            'Date':P.date,
            'Status':P.status,      
            'PsychoPy':P.isData,
            'PsychoPyComplete':P.isDataComplete,
            'Task':EVM_task,
            'Files':P.cntECGFiles,
            'Placed':P.cntECGPlaced,
            'Processed':P.cntECGProcessed,
            'Errors':';'.join(errors)
            }

    if os.path.exists(f"{out_folder}psychoPy_log.csv"):
        old_log = pd.read_csv(f"{out_folder}psychoPy_log.csv")
        #If participant already has data in clean log
        if (P.ID in old_log['Participant'].tolist()) and (old_log.loc[old_log['Participant']==P.ID]['Task'].str.contains(EVM_task).any()):
            for key, value in log.items():
                old_log.loc[(old_log['Participant']==P.ID)&(old_log['Task'].str.contains(EVM_task)), key] = value
            new_log = old_log
        else:
            new_log = pd.concat([old_log, pd.DataFrame([log])], ignore_index=True)
        new_log.to_csv(f"{out_folder}psychoPy_log.csv",index=False)
    else:
        psychoPy_log = pd.DataFrame()
        psychoPy_log = pd.concat([psychoPy_log, pd.DataFrame([log])], ignore_index=True)
        psychoPy_log.to_csv(f"{out_folder}psychoPy_log.csv",index=False)
    
    #End

    #############################
    print_status(f"Collected {EVM_task}",f"{int(P.cntECGProcessed/2*100)}%")  
    return P


def update_EDA(self):
    self.cntEDAFiles = 0
    self.cntEDAPlaced = 0
    self.cntEDAProcessed = 0


def get_EDA(in_folder, out_folder,P):
    '''
    This function cleans up EDA data, and then devides it into epochs. 
    At this point, no files need to be read in. All data needed is contained in the participants pickle


    Parameters
    __________
    in_folder: str
        Folder containing directories for each participant
        Must contain ONLY particpant 
    out_folder: str
        Folder in which a directory will be created per participant
        
    TODO:
    - Add in check for duplicate participant IDs
    '''
    
    
    #Define variables
    psychoPy_log = pd.DataFrame()

    errors = []
    out_path = f"{out_folder}{P.ID}/"
    os.makedirs(out_path, exist_ok=True) 
    
    EVM_Results = pd.DataFrame()
    EVM_task = 'EDA'

    ##Reset Flags
    P.cntEDAFiles = 0
    P.cntEDAPlaced = 0
    P.cntEDAProcessed = 0

    def calculate_peaks_per_minute(df):
        # Ensure the TaskTime column is in seconds and sort by TaskTime if needed
        df = df.sort_values('TaskTime')
        
        # Detect peaks in the 'Filtered' EDA data
        peaks, _ = find_peaks(df['Peaks'])
        
        # Calculate the total duration of the data in minutes
        total_time_seconds = df['TaskTime'].iloc[-1] - df['TaskTime'].iloc[0]
        total_time_minutes = total_time_seconds / 60
        
        # Calculate peaks per minute
        if total_time_minutes > 0:
            peaks_per_minute = len(peaks) / total_time_minutes
        else:
            peaks_per_minute = 0  # Avoid division by zero
        
        return peaks_per_minute
    
    print_step(f'Collecting {EVM_task}',P.ID)

    raw_data = pd.DataFrame()

    if P.isSignals:
        channels = [c for c in P.signals.columns if EVM_task in c]
        for c in channels:
            result = pd.DataFrame()
            _data = P.signals.dropna(subset=[c])
            result.loc[:, 'Freq']= _data['Freq']
            result.loc[:, 'File']= _data['File']
            result.loc[:, 't']= _data['t']
            result.loc[:, EVM_task]= _data[c]
            raw_data = pd.concat([raw_data,result])
            P.cntEDAFiles = 1

    #initialise filter params
    order = 4
    cutoffs = (0.05,0.5)  # desired cutoff frequency of the filter, Hz
    lp_filter = 0.05
    epoch_length = 180

    ##### Get Task Data
    tasks_of_interest = ['blank','simpleRT','Exercise']

    ### Only work on task data
    if P.isSignals & P.isData:
        tasks = P.tasksInfo[['Task','Start','End']]
        tasks = tasks.loc[tasks['Task'].str.contains('|'.join(tasks_of_interest))]

        tasks = tasks.sort_values(by='Start')
        #### Do all the writing for each task
        results = pd.DataFrame()
        for task in tasks['Task']:
                try:               
                    info = tasks.loc[tasks['Task']==task]
                    min_t = int(info['Start'].to_numpy()[0])
                    max_t = int(info['End'].to_numpy()[0])

                    data = pd.DataFrame()
                    data = raw_data.loc[(raw_data['t']>=min_t) & (raw_data['t']<=max_t)]
                    data = data.sort_values(by='t')
                    start_time = data['t'].values[0]
                    data.loc[:, 'TaskTime']=data['t']-start_time
                    data.loc[:, 'Peaks']=0
                    data.loc[:, 'Task']=task
            
                    dataFiles = data['File'].drop_duplicates().tolist()
                    for f in dataFiles:
                        try:
                            file = data.loc[data['File']==f].sort_values('t')  
                            freq = file['Freq'].values[0]

                            ## Set Epochs
                            time = file['TaskTime'].to_numpy()
                            maxi = time.max()
                            raw = file[EVM_task].to_numpy()     
                            ### Filter data based on findings from 'Stress detection using EDA and EMG signals: A comprehensive study'
                            #filtered = butter_bandpass_filter(raw, cutoffs, freq, order)
                            filtered = butter_lowpass_filter(raw, lp_filter, freq, order)
                            peak_filter = butter_bandpass_filter(raw, cutoffs, freq, 2)
                            file.loc[:, 'Filtered']= filtered
                            file.loc[:, 'Peaks']= peak_filter
                            #file.loc[:, 'LFFiltered']= lp_filtered


                            P.cntEDAPlaced += 1
                            results = pd.concat([results, file], ignore_index=True)
                
                        except IndexError as z:
                                print_status(f'Failed EDA on {task}', f"File {f}")
                                errors = errors + [f'Failed EDA on {task}', f"File {f}"]

                except Exception as z:
                        print_status(f'Failed EDA on {task}', f"{z!r}")
                        errors = errors + [f'Failed EDA on {task}', f"{z!r}",]


    if P.cntEDAPlaced:
        ## Set condition
        calc = {'Participant':P.ID,
                'Drink':P.drink,} 


        for task, _data in results.groupby('Task'):
            # We divide the task up into 3-minute epochs, so we can assess HRV components over a fair duration.
            try:
                dur = _data['TaskTime'].values[-1] - _data['TaskTime'].values[0]
                bins = np.arange(0, dur, epoch_length)
                i = 1
                for b in bins:
                    _data.loc[_data['TaskTime'] >= b, 'Epoch'] = i
                    i += 1

                # Lists to collect metrics from each epoch for trend analysis.
                epoch_indices = []
                meanSCL_list = []
                ppm_list = []

                for epoch_index, epoch in _data.groupby('Epoch'):
                    eda = epoch['Filtered'].dropna().to_numpy()
                    meanSCL = eda.mean()
                    PPM = calculate_peaks_per_minute(epoch)

                    if len(eda) > 20:
                        

                        # ----- Compute HRGradient for the epoch -----
                        # HRGradient is defined as the slope of the trendline of BPM versus time.
                        scl_gradient = np.polyfit(epoch['t'], epoch['Filtered'], 1)[0]

                        # Save the computed metrics for the current epoch.
                        _calc = {
                            f'os_EDA_{task}_E{epoch_index}_MeanSCL': meanSCL,
                            f'os_EDA_{task}_E{epoch_index}_PPM': PPM,
                            f'os_EDA_{task}_E{epoch_index}_Gradient': scl_gradient,
                        }
                        merge_dict(_calc, calc)
                        P.cntEDAProcessed += 1

                        # Append the epoch index and metrics to lists for trend analysis.
                        epoch_indices.append(epoch_index)
                        meanSCL_list.append(meanSCL)
                        ppm_list.append(PPM)

                # ----- Compute the gradient (slope) of the trendline for the entire task -----
                # Use the epoch index as the independent variable. A linear fit (1st degree polynomial) yields a slope.
                if len(epoch_indices) >= 2:
                    x = np.array(epoch_indices)
                    meanSCL_gradient = np.polyfit(x, np.array(meanSCL_list), 1)[0]
                    PPM_gradient = np.polyfit(x, np.array(ppm_list), 1)[0]

                    _calc = {
                            f'os_EDA_{task}_MeanSCL_gradient': meanSCL_gradient,
                            f'os_EDA_{task}_PPM_gradient': PPM_gradient,
                        }
                    merge_dict(_calc, calc)

       
            except Exception as z:
                print_status(f'Failed Calcs on {t}', f"{z!r}")
                errors = errors + [f'Failed Calcs on {t}', f"{z!r}",]

        ### Update Participant Results
        P.psychoPyResults = pd.concat([P.psychoPyResults, pd.DataFrame([calc])], ignore_index=True)
    
        ### Update EVM Results File
        if os.path.exists(f"{out_folder}EVM_Results.csv"):
            old_log = pd.read_csv(f"{out_folder}EVM_Results.csv")
            #If participant already has data in clean log
            if P.ID in old_log['Participant'].tolist():
                for key, value in calc.items():
                    old_log.loc[old_log['Participant']==P.ID, key] = value
                new_log = old_log
            else:
                new_log = pd.concat([old_log, pd.DataFrame([calc])], ignore_index=True)
            new_log.to_csv(f"{out_folder}EVM_Results.csv",index=False)
        else:
            EVM_Results = pd.DataFrame()
            EVM_Results = pd.concat([EVM_Results, pd.DataFrame([calc])], ignore_index=True)
            EVM_Results.to_csv(f"{out_folder}EVM_Results.csv",index=False)

    ### Update Log File

    log = {'Participant':P.ID,
            'Date':P.date,
            'Status':P.status,      
            'PsychoPy':P.isData,
            'PsychoPyComplete':P.isDataComplete,
            'Task':EVM_task,
            'Files':P.cntEDAFiles,
            'Placed':P.cntEDAPlaced,
            'Processed':P.cntEDAProcessed,
            'Errors':';'.join(errors)
            }

    if os.path.exists(f"{out_folder}psychoPy_log.csv"):
        old_log = pd.read_csv(f"{out_folder}psychoPy_log.csv")
        #If participant already has data in clean log
        if (P.ID in old_log['Participant'].tolist()) and (old_log.loc[old_log['Participant']==P.ID]['Task'].str.contains(EVM_task).any()):
            for key, value in log.items():
                old_log.loc[(old_log['Participant']==P.ID)&(old_log['Task'].str.contains(EVM_task)), key] = value
            new_log = old_log
        else:
            new_log = pd.concat([old_log, pd.DataFrame([log])], ignore_index=True)
        new_log.to_csv(f"{out_folder}psychoPy_log.csv",index=False)
    else:
        psychoPy_log = pd.DataFrame()
        psychoPy_log = pd.concat([psychoPy_log, pd.DataFrame([log])], ignore_index=True)
        psychoPy_log.to_csv(f"{out_folder}psychoPy_log.csv",index=False)
    
    #End

    #############################
    print_status(f"Collected {EVM_task}",f"{int(P.cntEDAProcessed*100)}%")  
    return P


def update_NIRS(self):
    self.cntNIRSFiles = 0
    self.cntNIRSPlaced = 0
    self.cntNIRSProcessed = 0


def get_NIRS(in_folder, out_folder,P):
    '''
    This function cleans up NIRS data, and then devides it into epochs. 
    At this point, no files need to be read in. All data needed is contained in the participants pickle


    Parameters
    __________
    in_folder: str
        Folder containing directories for each participant
        Must contain ONLY particpant 
    out_folder: str
        Folder in which a directory will be created per participant
        
    TODO:
    - Add in check for duplicate participant IDs
    '''
    
    
    #Define variables
    psychoPy_log = pd.DataFrame()

    errors = []
    out_path = f"{out_folder}{P.ID}/"
    os.makedirs(out_path, exist_ok=True) 
    
    EVM_Results = pd.DataFrame()
    EVM_task = 'NIRS'

    ##Reset Flags
    P.cntNIRSFiles = 0
    P.cntNIRSPlaced = 0
    P.cntNIRSProcessed = 0
    
    print_step(f'Collecting {EVM_task}',P.ID)

    if P.isSignals:
        channels = [c for c in P.signals.columns if 'NIRS' in c]
    
        #initialise filter params
        order = 4
        cutoffs = (0.5,3)  # desired cutoff frequency of the filter, Hz
        epoch_length = 60
        hr_length = 10



    ### Print 

    ### Only work on task data
    if bool(P.status) & bool(P.isData):
            ##### Get Task Data
        tasks_of_interest = ['goNoGo','memory','digit','tsst','manual','emotion']
        tasks = P.tasksInfo[['Task','Start','End']]
        tasks = tasks.loc[tasks['Task'].str.contains('|'.join(tasks_of_interest))]

        ### Add Benchmarking Block
        _frameStart = P.EVM.loc[P.EVM['Event']=='Benchmark_Start']['t'].values[0]
        _frameEnd = P.tasksInfo.loc[P.tasksInfo['Task']=='Mouth_Open']['Start'].values[0]
        _frame = {'Task':'Benchmarking',
                     'Start':_frameStart,
                     'End':_frameEnd,
                     'Duration':_frameEnd-_frameStart}
        tasks = pd.concat([tasks, pd.DataFrame([_frame])], ignore_index=True)
        tasks = tasks.sort_values(by='Start')
        #### Do all the writing for each task
        results = pd.DataFrame()
        P.signals['HRPeaks'] = P.signals['HRPeaks'].replace(np.nan,0)
        for task in tasks['Task']:
                try:               
                    info = tasks.loc[tasks['Task']==task]
                    min_t = int(info['Start'].to_numpy()[0])
                    max_t = int(info['End'].to_numpy()[0])

                    data = P.signals.loc[(P.signals['t']>=min_t) & (P.signals['t']<=max_t)]
                    data = data[['NIRS_Right_Oxy',
                                'NIRS_Right_Deoxy',
                                'NIRS_Left_Oxy',
                                'NIRS_Left_Deoxy',
                                'Freq', 'Channels', 'File', 't','HRPeaks']]

                    ##### check this
                    data=data.dropna(subset = ['NIRS_Right_Oxy', 'NIRS_Right_Deoxy', 'NIRS_Left_Oxy', 'NIRS_Left_Deoxy',])

                    data = data.sort_values(by='t')
                    start_time = data['t'].values[0]
                    data.loc[:, 'TaskTime']=data['t']-start_time
                    data.loc[:, 'Task']=task
            
                    dataFiles = data['File'].drop_duplicates().tolist()
                    for f in dataFiles:
                        try:
                            file = data.loc[data['File']==f].sort_values('t')  
                            freq = file['Freq'].values[0]

                            ## Set Epochs
                            time = file['TaskTime'].to_numpy()
                            maxi = time.max()
                            for c in channels:
                                raw = file[c].to_numpy()
                                filtered = butter_bandpass_filter(raw, cutoffs, freq, order)
                                file.loc[:, f'Filtered_{c}']= filtered

                            P.cntNIRSPlaced += 1
                            results = pd.concat([results, file], ignore_index=True)
                
                        except IndexError as z:
                                print_status(f'Failed NIRS on {task}', f"File {f}")
                                errors = errors + [f'Failed NIRS on {task}', f"File {f}"]
                    #results = pd.concat([results, pd.DataFrame([data])], ignore_index=True)

                except Exception as z:
                        print_status(f'Failed NIRS on {task}', f"{z!r}")
                        errors = errors + [f'Failed NIRS on {task}', f"{z!r}",]
    
    if P.cntNIRSPlaced:
        ## Set condition
        calc = {'Participant':P.ID,
                'Drink':P.drink,}

        tb = results.groupby('Task')
        processed_results = pd.DataFrame()

        for t in tb.groups:
            try:         
                _data = tb.get_group(t)
                
                result = dict()
                try:
                    ts = t.split('_')[0]
                    cs = t.split('_')[1]
                    conds = {'01':'Baseline',
                                '02':'BeforeTreatment',
                                '03':'TreatmentInitial',
                                '04':'TreatmentBeforeTasks',
                                '05':'TreatmentAfterTasks'}

                    conds2 = {'01':'Baseline',
                                '02':'TreatmentTasksStart',
                                '03':'TreatmentTasksEnd',
                                }

                    if ts == 'manualCapture':
                        cond = conds[cs]
                    elif ts == 'MAS':                             
                        cond = conds2[cs]
                    else:                             
                        cond = 'Baseline' if int(t.split('_')[1])==1 else 'Treatment'
                except:
                    pass                           
                try:
                    tag = f'{ts}_{cond}'
                except:
                    tag = f'{ts}'

                times = _data.loc[_data['HRPeaks']>0]['t'].to_list()
                for i in range(1,len(times)):
                    _data.loc[(_data['t']>times[i-1]) & (_data['t']<=times[i]), 'Beat'] = i
                
                channels = [c for c in _data.columns if 'Filtered_NIRS' in c]
                hresult = dict()
                task_length = _data['HRPeaks'].sum()
                for s in ['Left','Right']:
                    chns = [c for c in channels if s in c]
                    _data[chns] = _data[chns].replace(0,np.nan)

                    for c in chns:
                        _data.loc[_data[c]>65400, c] = np.nan

                    cols = chns + ['Beat',]
                    data = _data[cols].dropna()
                    data_length = len(data['Beat'].drop_duplicates())

                    hresult[f'{s}_Valid'] = 1 if (data_length/task_length)>=0.9 else 0


                    red = [c for c in chns if 'Oxy' in c][0]
                    ir = [c for c in chns if 'Deoxy' in c][0]

                    v_avg = data.groupby('Beat').mean()[chns]
                    v_pp = data.groupby('Beat').max()-data.groupby('Beat').min()[chns]
                    R = (v_pp[red]*v_avg[ir])/(v_pp[ir]*v_avg[red])
                    spo2 = 140-18*R

                    hresult[s]=spo2

                if hresult['Left_Valid'] and hresult['Right_Valid']:
                    result['Task'] = f'{tag}'
                    result['SpO2Left']=hresult['Left'].mean()
                    result['SpO2Right']=hresult['Right'].mean()

                    processed_results = pd.concat([processed_results, pd.DataFrame([result])], ignore_index=True)
                    P.cntNIRSProcessed += 1
                else:
                    print_status(f'Invaliid data in {t}', f"invalid")

            except Exception as z:
                print_status(f'Failed Calcs on {t}', f"{z!r}")
                errors = errors + [f'Failed Calcs on {t}', f"{z!r}",]
   
        try:
            norm = processed_results.loc[(processed_results['Task']=='Benchmarking')]
            gb = processed_results.groupby('Task')
            for x in gb.groups:
                data = gb.get_group(x)
                left = (data['SpO2Left'].values[0]*85)/norm['SpO2Left'].values[0]
                right = (data['SpO2Right'].values[0]*85)/norm['SpO2Right'].values[0]
                asym = right - left
                tot = right + left

                _calc = {f"psy_{x}_SpO2Left":left,
                            f"psy_{x}_SpO2Right":right,
                            f"psy_{x}_SpO2Asymmetry":asym,
                            f"psy_{x}_SpO2Total":tot,
                            }
                merge_dict(_calc,calc)
        except IndexError as z:
                print_status(f'No norm data on', f"{P.ID}")
                errors = errors + [f'No norm data on', f"{P.ID}"]
        except Exception as z:
                pass


        ### Update Participant Results
        P.psychoPyResults = pd.concat([P.psychoPyResults, pd.DataFrame([calc])], ignore_index=True)
    
        ### Update EVM Results File
        if os.path.exists(f"{out_folder}EVM_Results.csv"):
            old_log = pd.read_csv(f"{out_folder}EVM_Results.csv")
            #If participant already has data in clean log
            if P.ID in old_log['Participant'].tolist():
                for key, value in calc.items():
                    old_log.loc[old_log['Participant']==P.ID, key] = value
                new_log = old_log
            else:
                new_log = pd.concat([old_log, pd.DataFrame([calc])], ignore_index=True)
            new_log.to_csv(f"{out_folder}EVM_Results.csv",index=False)
        else:
            EVM_Results = pd.DataFrame()
            EVM_Results = pd.concat([EVM_Results, pd.DataFrame([calc])], ignore_index=True)
            EVM_Results.to_csv(f"{out_folder}EVM_Results.csv",index=False)

    ### Update Log File

    log = {'Participant':P.ID,
            'Date':P.date,
            'Status':P.status,      
            'PsychoPy':P.isData,
            'PsychoPyComplete':P.isDataComplete,
            'Task':EVM_task,
            'Files':P.cntNIRSFiles,
            'Placed':P.cntNIRSPlaced,
            'Processed':P.cntNIRSProcessed,
            'Errors':';'.join(errors)
            }

    if os.path.exists(f"{out_folder}psychoPy_log.csv"):
        old_log = pd.read_csv(f"{out_folder}psychoPy_log.csv")
        #If participant already has data in clean log
        if (P.ID in old_log['Participant'].tolist()) and (old_log.loc[old_log['Participant']==P.ID]['Task'].str.contains(EVM_task).any()):
            for key, value in log.items():
                old_log.loc[(old_log['Participant']==P.ID)&(old_log['Task'].str.contains(EVM_task)), key] = value
            new_log = old_log
        else:
            new_log = pd.concat([old_log, pd.DataFrame([log])], ignore_index=True)
        new_log.to_csv(f"{out_folder}psychoPy_log.csv",index=False)
    else:
        psychoPy_log = pd.DataFrame()
        psychoPy_log = pd.concat([psychoPy_log, pd.DataFrame([log])], ignore_index=True)
        psychoPy_log.to_csv(f"{out_folder}psychoPy_log.csv",index=False)
    
    #End

    #############################
    print_status(f"Collected {EVM_task}",f"{int(P.cntNIRSProcessed/2*100)}%")  
    return P



def update_EEG_ER(self):
    self.cntEEG_ERFiles = 0
    self.cntEEG_ERPlaced = 0
    self.cntEEG_ERProcessed = 0


def get_EEG_ER(in_folder, out_folder,P):
    '''
    This function cleans up EEG_ER data, and then devides it into epochs. 
    At this point, no files need to be read in. All data needed is contained in the participants pickle


    Parameters
    __________
    in_folder: str
        Folder containing directories for each participant
        Must contain ONLY particpant 
    out_folder: str
        Folder in which a directory will be created per participant
        
    TODO:
    - Add in check for duplicate participant IDs
    '''
    
    
    #Define variables
    psychoPy_log = pd.DataFrame()

    errors = []
    in_path = f"{in_folder}{P.ID}/"
    out_path = f"{out_folder}{P.ID}/"
    os.makedirs(out_path, exist_ok=True) 
    
    EVM_Results = pd.DataFrame()
    signalType = 'EEG'
    EVM_task = 'emotionRecognition'

    ##Reset Flags
    P.cntEEG_ERFiles = 0
    P.cntEEG_ERPlaced = 0
    P.cntEEG_ERProcessed = 0
    
    print_step(f'Collecting {signalType}',P.ID)

    raw_data = pd.DataFrame()
    channels = [c for c in P.signals.columns if signalType in c]
    for c in channels:
        result = pd.DataFrame()
        _data = P.signals.dropna(subset=[c])
        result.loc[:, 'Freq']= _data['Freq']
        result.loc[:, 'File']= _data['File']
        result.loc[:, 't']= _data['t']
        result.loc[:, signalType]= _data[c]
        raw_data = pd.concat([raw_data,result])

    #initialise filter params
    order = 4
    cutoffs = (8,12)  # desired cutoff frequency of the filter, Hz
    epoch_half = 1

    ##### Get Task Data
    tasks_of_interest = ['emotion',]
    try:
        channel = [c for c in P.signals.columns if signalType in c][0]
        P.cntEEG_ERFiles = 1
    except:
        ###Add error stuff
        P.cntEEG_ERFiles = 0

    ### Print 

    ### Only work on task data
    if P.isSignals & P.isData:
        tasks = P.tasksInfo[['Task','Start','End']]
        tasks = tasks.loc[tasks['Task'].str.contains('|'.join(tasks_of_interest))]

        tasks = tasks.sort_values(by='Start')
        #### Do all the writing for each task
        task_files = get_files(in_path,tags=['.csv', 'emotionRecognition']) #Get list of all goNogo tasks
    
        ###### Build Task Files
        task_file = pd.DataFrame()
    
        for f in task_files:
            try:
                data = pd.read_csv(f'{in_path}{f}', low_memory = False)
                data['Time']=data['Time'].str.replace(',','.')
                data.loc[:, 't']=[get_sec(i) for i in data['Time'].tolist()]
                data.loc[:, 'StimTime'] = data['t']-data['Response_Time']
                t0 = data['t'].values[0]
                data.loc[:, 'Task'] = tasks.loc[(tasks['Start']<=t0) & (tasks['End']>=t0)]['Task'].values[0]
        
                task_file = pd.concat([task_file,data])
            except Exception as z:
                print_status(f'Failed EEG_ER on {task_file}', f"{z!r}")
                errors = errors + [f'Failed EEG_ER on {task_file}', f"{z!r}",]

        results = pd.DataFrame()
        for task in tasks['Task']:
                try:               
                    info = tasks.loc[tasks['Task']==task]
                    min_t = int(info['Start'].to_numpy()[0])
                    max_t = int(info['End'].to_numpy()[0])

                    data = pd.DataFrame()
                    data = raw_data.loc[(raw_data['t']>=min_t) & (raw_data['t']<=max_t)]

                    if len(data) > 500:
                        data = data.sort_values(by='t')
                        P.cntEEG_ERPlaced += 1

                        events = task_file.loc[task_file['Task']==task]
                        e = 0
                        trial = 0
                        eresults = pd.DataFrame()
                        for index, row in events.iterrows():
                            if row['Response_Time']>0.2: #Check if valid response
                                ##### Get exposure data
                                event_start = row['StimTime']-epoch_half
                                event_end = row['StimTime']+epoch_half
                   
                                epoch = data.loc[(data['t']>=event_start) & (data['t']<=event_end)]
                                raw = epoch[signalType].to_numpy().astype(float)

                                f=epoch['Freq'].values[0]
                                trash = ((raw < 100).sum() + (raw > 65435).sum())/len(raw)*100
                                length = len(raw)/(f*2*epoch_half)*100
                                


                                if  (len(raw)>100):   #check if valid datapoints greater than 90% and more than 90% of datapoints for epoch    
                                    filtered = raw-(65535/2)
                                    
                                    filtered = denoise_wavelet(raw, method='VisuShrink', mode='hard', 
                                                               wavelet_levels=6, wavelet='sym3', rescale_sigma='True')
                                    
                                    filtered = butter_bandpass_filter(filtered, cutoffs,f , order) 
                                    power = np.log(compute_welch_psd(filtered, f)) 
                                    
                                    eresult=dict()
                                    eresult['Participant'] = P.ID
                                    eresult['Task'] = task
                                    eresult['t']=event_start
                                    eresult['Epoch']=e
                                    eresult['Power']=power
                                    eresult['Trial']=trial
                                    eresult['Event']='Stim'
                                    eresult['Emotion']=row['Emotion']
                                    eresult['Accurate']=True if row['Emotion']==row['Choice'] else False
                                    eresult['Valid']=100-trash
                                    eresult['Signal']= length
                                    results = pd.concat([results, pd.DataFrame([eresult])], ignore_index=True)
                                e+=1
                    
                                ### Get Response Data                  
                                event_start = row['t']-epoch_half
                                event_end = row['t']+epoch_half

                                       
                                epoch = data.loc[(data['t']>=event_start) & (data['t']<=event_end)]
                                raw = epoch[signalType].to_numpy().astype(float)
                                f=epoch['Freq'].values[0]

                                trash = ((raw < 100).sum() + (raw > 65435).sum())/len(raw)*100
                                length = len(raw)/(f*2*epoch_half)*100

                                if  (len(raw)>100):   
                                    filtered = raw-(65535/2)
                                    filtered = denoise_wavelet(raw, method='VisuShrink', mode='hard', 
                                                               wavelet_levels=6, wavelet='sym3', rescale_sigma='True')
                                    filtered = butter_bandpass_filter(filtered, (8,12),f , order) 
                                    power = np.log(compute_welch_psd(filtered, f))
                                    
                                    eresult=dict()
                                    eresult['Participant'] = P.ID
                                    eresult['Task'] = task
                                    eresult['t']=event_start
                                    eresult['Epoch']=e
                                    eresult['Power']=power
                                    eresult['Trial']=trial
                                    eresult['Event']='Response'
                                    eresult['Emotion']=row['Emotion']               
                                    eresult['Accurate']=True if row['Emotion']==row['Choice'] else False
                                    eresult['Valid']=100-trash
                                    eresult['Signal']= length
                                    results = pd.concat([results, pd.DataFrame([eresult])], ignore_index=True)
                                e+=1
                                trial+=1
                                P.cntEEG_ERProcessed += 1


                except Exception as z:
                        print_status(f'Failed EEG_ER on {task}', f"{z!r}")
                        errors = errors + [f'Failed EEG_ER on {task}', f"{z!r}",]

        

        calc = {'Participant':P.ID,
                'Drink':P.drink,}

        if P.cntEEG_ERProcessed:
            results.loc[results['Task'].str.contains('_01'), 'Condition']='Baseline'
            results.loc[results['Task'].str.contains('_02'), 'Condition']='Treatment'
            results = results.loc[(results['Valid']>90) & (results['Signal']>90)]
            #for each condition

            conditions = results['Condition'].drop_duplicates()

            for condition in conditions:
                _condition = results.loc[results['Condition']==condition]

                events = _condition['Event'].drop_duplicates().tolist()

                for event in events:
                #Update row entry
                    _event = _condition.loc[_condition['Event']==event]

                    _calc = {f"psy_{EVM_task}_{condition}_Overall_{event}_AlphaPSD":_event['Power'].mean(),
                             f"psy_{EVM_task}_{condition}_Overall_{event}_AlphaPSDAccurate":_event.loc[_event['Accurate']==1]['Power'].mean(),
                             f"psy_{EVM_task}_{condition}_Overall_{event}_AlphaPSDInaccurate":_event.loc[_event['Accurate']==0]['Power'].mean(),
                        }
                    merge_dict(_calc,calc)

                    #########################

                    emotions = _event['Emotion'].drop_duplicates()
                    for emotion in emotions:
                        _emotion = _event.loc[_event['Emotion']==emotion]
                    
                        _calc = {f"psy_{EVM_task}_{condition}_{emotion}_{event}_AlphaPSD":_emotion['Power'].mean(),
                                 f"psy_{EVM_task}_{condition}_{emotion}_{event}_AlphaPSDAccurate":_emotion.loc[_emotion['Accurate']==1]['Power'].mean(),
                                 f"psy_{EVM_task}_{condition}_{emotion}_{event}_AlphaPSDInaccurate":_emotion.loc[_emotion['Accurate']==0]['Power'].mean(),
                                }
                        merge_dict(_calc,calc)

        ### Update Participant Results
        P.psychoPyResults = pd.concat([P.psychoPyResults, pd.DataFrame([calc])], ignore_index=True)
    
        ### Update EVM Results File
        if os.path.exists(f"{out_folder}EVM_Results.csv"):
            old_log = pd.read_csv(f"{out_folder}EVM_Results.csv")
            #If participant already has data in clean log
            if P.ID in old_log['Participant'].tolist():
                for key, value in calc.items():
                    old_log.loc[old_log['Participant']==P.ID, key] = value
                new_log = old_log
            else:
                new_log = pd.concat([old_log, pd.DataFrame([calc])], ignore_index=True)
            new_log.to_csv(f"{out_folder}EVM_Results.csv",index=False)
        else:
            EVM_Results = pd.DataFrame()
            EVM_Results = pd.concat([EVM_Results, pd.DataFrame([calc])], ignore_index=True)
            EVM_Results.to_csv(f"{out_folder}EVM_Results.csv",index=False)

    ### Update Log File

    log = {'Participant':P.ID,
            'Date':P.date,
            'Status':P.status,      
            'PsychoPy':P.isData,
            'PsychoPyComplete':P.isDataComplete,
            'Task':signalType,
            'Files':P.cntEEG_ERFiles,
            'Placed':P.cntEEG_ERPlaced,
            'Processed':P.cntEEG_ERProcessed,
            'Errors':';'.join(errors)
            }

    if os.path.exists(f"{out_folder}psychoPy_log.csv"):
        old_log = pd.read_csv(f"{out_folder}psychoPy_log.csv")
        #If participant already has data in clean log
        if (P.ID in old_log['Participant'].tolist()) and (old_log.loc[old_log['Participant']==P.ID]['Task'].str.contains(signalType).any()):
            for key, value in log.items():
                old_log.loc[(old_log['Participant']==P.ID)&(old_log['Task'].str.contains(signalType)), key] = value
            new_log = old_log
        else:
            new_log = pd.concat([old_log, pd.DataFrame([log])], ignore_index=True)
        new_log.to_csv(f"{out_folder}psychoPy_log.csv",index=False)
    else:
        psychoPy_log = pd.DataFrame()
        psychoPy_log = pd.concat([psychoPy_log, pd.DataFrame([log])], ignore_index=True)
        psychoPy_log.to_csv(f"{out_folder}psychoPy_log.csv",index=False)
    
    #End

    #############################
    print_status(f"Collected {signalType}",f"{int(P.cntEEG_ERProcessed/40*100)}%")  
    return P

def update_EEG_GNG(self):
    self.cntEEG_GNGFiles = 0
    self.cntEEG_GNGPlaced = 0
    self.cntEEG_GNGProcessed = 0


def get_EEG_GNG(in_folder, out_folder,P):
    '''
    This function cleans up EEG_GNG data, and then devides it into epochs. 
    At this point, no files need to be read in. All data needed is contained in the participants pickle


    Parameters
    __________
    in_folder: str
        Folder containing directories for each participant
        Must contain ONLY particpant 
    out_folder: str
        Folder in which a directory will be created per participant
        
    TODO:
    - Add in check for duplicate participant IDs
    '''
    
    
    #Define variables
    psychoPy_log = pd.DataFrame()

    errors = []
    in_path = f"{in_folder}{P.ID}/"
    out_path = f"{out_folder}{P.ID}/"
    os.makedirs(out_path, exist_ok=True) 
    
    EVM_Results = pd.DataFrame()
    signalType = 'EEG'
    EVM_task = 'goNoGo'

    ##Reset Flags
    P.cntEEG_GNGFiles = 0
    P.cntEEG_GNGPlaced = 0
    P.cntEEG_GNGProcessed = 0
    
    print_step(f'Collecting {signalType}',P.ID)

    raw_data = pd.DataFrame()
    channels = [c for c in P.signals.columns if signalType in c]
    for c in channels:
        result = pd.DataFrame()
        _data = P.signals.dropna(subset=[c])
        result.loc[:, 'Freq']= _data['Freq']
        result.loc[:, 'File']= _data['File']
        result.loc[:, 't']= _data['t']
        result.loc[:, signalType]= _data[c]
        raw_data = pd.concat([raw_data,result])

    #initialise filter params
    order = 4
    cutoffs = (8,12)  # desired cutoff frequency of the filter, Hz
    epoch_before = 0.1
    epoch_after = 0.5
    wavelet_level = 5

    ##### Get Task Data
    tasks_of_interest = [EVM_task,]
    try:
        channel = [c for c in P.signals.columns if signalType in c][0]
        P.cntEEG_GNGFiles = 1
    except:
        ###Add error stuff
        P.cntEEG_GNGFiles = 0


    ### Only work on task data
    if P.isSignals & P.isData:
        tasks = P.tasksInfo[['Task','Start','End']]
        tasks = tasks.loc[tasks['Task'].str.contains('|'.join(tasks_of_interest))]

        tasks = tasks.sort_values(by='Start')
        #### Do all the writing for each task
        task_files = get_files(in_path,tags=['.csv', EVM_task]) #Get list of all goNogo tasks
    
        ###### Build Task Files
        task_file = pd.DataFrame()
    
        for f in task_files:
            try:
                data = pd.read_csv(f'{in_path}{f}', low_memory = False)
                data['Time']=data['Time'].str.replace(',','.')
                data.loc[:, 't']=[get_sec(i) for i in data['Time'].tolist()]
                data.loc[:, 'StimTime'] = data['t']-data['Response_Time']
                data.loc[:, 'StimDur'] = data['t']-data['StimTime']
                data.loc[:, 'Part'] = data['Task']
                t0 = data['t'].values[0]
                data.loc[:, 'Task'] = tasks.loc[(tasks['Start']<=t0) & (tasks['End']>=t0)]['Task'].values[0]
                task_file = pd.concat([task_file,data])
            except KeyError:
                print_status('Failed',f'{f} incorrect')
            except IndexError:
                print_status('Failed',f'{f} could not find task')

        results = pd.DataFrame()
        valid = 0
        for task in tasks['Task']:
                try:               
                    info = tasks.loc[tasks['Task']==task]
                    min_t = int(info['Start'].to_numpy()[0])
                    max_t = int(info['End'].to_numpy()[0])

                    data = pd.DataFrame()
                    data = raw_data.loc[(raw_data['t']>=min_t) & (raw_data['t']<=max_t)]

                    if len(data) > 500:
                        data = data.sort_values(by='t')
                        P.cntEEG_GNGPlaced += 1

                        events = task_file.loc[task_file['Task']==task]
                        e = 0
                        trial = 0
                        eresults = pd.DataFrame()
                        for index, row in events.iterrows():
                            if row['Response_Time']>0.2: #Check if valid response
                                ##### Get exposure data
                                event_start = row['StimTime']-epoch_before
                                event_end = row['StimTime']+epoch_after
                   
                                epoch = data.loc[(data['t']>=event_start) & (data['t']<=event_end)]
                                raw = epoch[signalType].to_numpy().astype(float)

                                f=epoch['Freq'].values[0]
                                trash = ((raw < 100).sum() + (raw > 65435).sum())/len(raw)*100
                                length = len(raw)/(f*(epoch_before+epoch_after))*100
                                


                                if  (len(raw)>100):   #check if valid datapoints greater than 90% and more than 90% of datapoints for epoch    
                                    filtered = raw-(65535/2)
                                    
                                    filtered = denoise_wavelet(raw, method='VisuShrink', mode='hard', 
                                                               wavelet_levels=wavelet_level, wavelet='sym3', rescale_sigma='True')
                                    
                                    filtered = butter_bandpass_filter(filtered, cutoffs,f , order) 
                                    power = np.log(compute_welch_psd(filtered, f)) 
                                    
                                    eresult=dict()
                                    eresult['Participant'] = P.ID
                                    eresult['Task'] = task
                                    eresult['t']=event_start
                                    eresult['Epoch']=e
                                    eresult['Power']=power
                                    eresult['Trial']=trial
                                    eresult['Part']=row['Part'].replace('_','')
                                    eresult['Event']='Stim'
                                    eresult['Accurate']=row['Accurate']
                                    eresult['Valid']=100-trash
                                    eresult['Signal']= length
                                    results = pd.concat([results, pd.DataFrame([eresult])], ignore_index=True)
                                    valid +=1
                                e+=1
                    
                                ### Get Response Data                  
                                event_start = row['t']-epoch_before
                                event_end = row['t']+epoch_after

                                       
                                epoch = data.loc[(data['t']>=event_start) & (data['t']<=event_end)]
                                raw = epoch[signalType].to_numpy().astype(float)
                                f=epoch['Freq'].values[0]

                                trash = ((raw < 100).sum() + (raw > 65435).sum())/len(raw)*100
                                length = len(raw)/(f*2*(epoch_before+epoch_after))*100

                                if  (len(raw)>100):   
                                    filtered = raw-(65535/2)
                                    filtered = denoise_wavelet(raw, method='VisuShrink', mode='hard', 
                                                               wavelet_levels=wavelet_level, wavelet='sym3', rescale_sigma='True')
                                    
                                    filtered = butter_bandpass_filter(filtered, (8,12),f , order)
                                    
                                    power = np.log(compute_welch_psd(filtered, f))
                                    
                                    eresult=dict()
                                    eresult['Participant'] = P.ID
                                    eresult['Task'] = task
                                    eresult['t']=event_start
                                    eresult['Epoch']=e
                                    eresult['Power']=power
                                    eresult['Trial']=trial
                                    eresult['Part']=row['Part'].replace('_','')
                                    eresult['Event']='Response'
                                    eresult['Accurate']=row['Accurate']
                                    eresult['Valid']=100-trash
                                    eresult['Signal']= length
                                    results = pd.concat([results, pd.DataFrame([eresult])], ignore_index=True)
                                    valid +=1
                                e+=1
                                trial+=1
                                P.cntEEG_GNGProcessed += 1


                except Exception as z:
                        print_status(f'Failed EEG_GNG on {task}', f"{z!r}")
                        errors = errors + [f'Failed EEG_GNG on {task}', f"{z!r}",]

        

        calc = {'Participant':P.ID,
                'Drink':P.drink,}

        if P.cntEEG_GNGProcessed and valid:
            results.loc[results['Task'].str.contains('_01'), 'Condition']='Baseline'
            results.loc[results['Task'].str.contains('_02'), 'Condition']='Treatment'
            results = results.loc[(results['Valid']>90) & (results['Signal']>90)]
            #for each condition

            conditions = results['Condition'].drop_duplicates()

            for condition in conditions:
                _condition = results.loc[results['Condition']==condition]

                events = _condition['Event'].drop_duplicates().tolist()

                for event in events:
                #Update row entry
                    _event = _condition.loc[_condition['Event']==event]

                    _calc = {f"psy_{EVM_task}_{condition}_Overall_{event}_AlphaPSD":_event['Power'].mean(),
                             f"psy_{EVM_task}_{condition}_Overall_{event}_AlphaPSDAccurate":_event.loc[_event['Accurate']==1]['Power'].mean(),
                             f"psy_{EVM_task}_{condition}_Overall_{event}_AlphaPSDInaccurate":_event.loc[_event['Accurate']==0]['Power'].mean(),
                        }
                    merge_dict(_calc,calc)

                    #########################

                    parts = _event['Part'].drop_duplicates()
                    for part in parts:
                        _part = _event.loc[_event['Part']==part]
                    
                        _calc = {f"psy_{EVM_task}_{condition}_{part}_{event}_AlphaPSD":_part['Power'].mean(),
                                 f"psy_{EVM_task}_{condition}_{part}_{event}_AlphaPSDAccurate":_part.loc[_part['Accurate']==1]['Power'].mean(),
                                 f"psy_{EVM_task}_{condition}_{part}_{event}_AlphaPSDInaccurate":_part.loc[_part['Accurate']==0]['Power'].mean(),
                                }
                        merge_dict(_calc,calc)

        ### Update Participant Results
        P.psychoPyResults = pd.concat([P.psychoPyResults, pd.DataFrame([calc])], ignore_index=True)
    
        ### Update EVM Results File
        if os.path.exists(f"{out_folder}EVM_Results.csv"):
            old_log = pd.read_csv(f"{out_folder}EVM_Results.csv")
            #If participant already has data in clean log
            if P.ID in old_log['Participant'].tolist():
                for key, value in calc.items():
                    old_log.loc[old_log['Participant']==P.ID, key] = value
                new_log = old_log
            else:
                new_log = pd.concat([old_log, pd.DataFrame([calc])], ignore_index=True)
            new_log.to_csv(f"{out_folder}EVM_Results.csv",index=False)
        else:
            EVM_Results = pd.DataFrame()
            EVM_Results = pd.concat([EVM_Results, pd.DataFrame([calc])], ignore_index=True)
            EVM_Results.to_csv(f"{out_folder}EVM_Results.csv",index=False)

    ### Update Log File

    log = {'Participant':P.ID,
            'Date':P.date,
            'Status':P.status,      
            'PsychoPy':P.isData,
            'PsychoPyComplete':P.isDataComplete,
            'Task':signalType,
            'Files':P.cntEEG_GNGFiles,
            'Placed':P.cntEEG_GNGPlaced,
            'Processed':P.cntEEG_GNGProcessed,
            'Errors':';'.join(errors)
            }

    if os.path.exists(f"{out_folder}psychoPy_log.csv"):
        old_log = pd.read_csv(f"{out_folder}psychoPy_log.csv")
        #If participant already has data in clean log
        if (P.ID in old_log['Participant'].tolist()) and (old_log.loc[old_log['Participant']==P.ID]['Task'].str.contains(signalType).any()):
            for key, value in log.items():
                old_log.loc[(old_log['Participant']==P.ID)&(old_log['Task'].str.contains(signalType)), key] = value
            new_log = old_log
        else:
            new_log = pd.concat([old_log, pd.DataFrame([log])], ignore_index=True)
        new_log.to_csv(f"{out_folder}psychoPy_log.csv",index=False)
    else:
        psychoPy_log = pd.DataFrame()
        psychoPy_log = pd.concat([psychoPy_log, pd.DataFrame([log])], ignore_index=True)
        psychoPy_log.to_csv(f"{out_folder}psychoPy_log.csv",index=False)
    
    #End

    #############################
    print_status(f"Collected {signalType}",f"{int(P.cntEEG_GNGProcessed/40*100)}%")  
    return P


def update_STQ(self):
    self.cntSTQFiles = 0
    self.cntSTQProcessed = 0

def get_STQ(in_folder, out_folder,P):
    '''
    This function collects stq data

    Parameters
    __________
    in_folder: str
        Folder containing directories for each participant
        Must contain ONLY particpant 
    out_folder: str
        Folder in which a directory will be created per participant
        
    TODO:
    - Add in check for duplicate participant IDs
    '''

    #Define variables
    psychoPy_log = pd.DataFrame()

    errors = []
    in_path = f"{in_folder}{P.ID}/"
    out_path = f"{out_folder}{P.ID}/"
    os.makedirs(out_path, exist_ok=True) 
    
    EVM_Results = pd.DataFrame()
    EVM_task = 'STQ'
    
    STQ_Dict = pd.read_csv(f"C:/Users/ashra/Desktop/Distell/Raw/STQ_dict.csv")
    section_dict = {1:'Lifestyle',
                    2:'Personality',
                    3:'Introversion',
                    4:'BISBAS',
                    5:'ADHD',
                    6:'PSS',
                    7:'ANX',
                    }

    section_max = {1:5,
                   2:5,
                   3:5,
                   4:4,
                   5:4,
                   6:4,
                   7:4}

    section_shift = {1:0,
                    2:-10,
                    3:40,
                    4:0,
                    5:0,
                    6:0,
                    7:0,
                    }

    ##Reset Flags
    P.cntSTQFiles = 0
    P.cntSTQProcessed = 0
    
    taskData=pd.DataFrame()
    print_step(f'Collecting {EVM_task}',P.ID)
    ##### STQ Data
    ##Extract data from files
    try:
        files = get_files(in_path,tags=['.csv', EVM_task]) #Get list of all goNogo tasks
    except Exception as z:
        files=[]
        print_status(f'Error {EVM_task}',f"{P.ID}")
        errors = errors + [f'Error {EVM_task}',]
    #For each file
    file_num = 0
    for f in files:                   
        try:
            path = in_path + f                 
            f=open(path)                    
            df = pd.read_csv(path, low_memory = False)
            df['Path']=path
            df['File']=file_num

            ##Adjust data
            #for q in df['Question']:
                #df.loc[df['Question'] == q, ['Polarity','Factor','Loading']] = STQ_polarity[q], STQ_Factors[q], STQ_Loading[q]

            ##Update Counters
            file_num = file_num+1
            print_status(f'Extracted {EVM_task}',path)
              
            ##Update Participant Data
            taskData = pd.concat([taskData,df])

            #########################
            P.cntSTQFiles += 1
        
        ## Account for Issues
        except Exception as z:
            print_status(f'Failed {EVM_task}',f"{path} - {z!r}")
            errors = errors + [f"Failed {EVM_task}: {path} - {z!r}",]
            
    if P.cntSTQFiles:
        data = taskData.copy()
        data.loc[pd.to_numeric(data['Section'], errors='coerce').isnull(), 'Section']=0
        data.loc[:,'Section'] = data['Section'].astype('int')

        results = pd.DataFrame()

        boxes = ['hbox1','hbox2','hbox3','hbox4','hbox5',
                 'vbox1','vbox2','vbox3','vbox4','vbox5']
        calc = {'Participant':P.ID,
                'Drink':P.drink,}
        ### Some pseudo code
        sections = STQ_Dict['Section'].drop_duplicates().dropna().tolist()
        for s in sections:
            section = data.loc[data['Section']==s].dropna(subset=['Question',])
            questions = STQ_Dict.loc[(STQ_Dict['Section']==s) & (STQ_Dict['Attribute']!=np.nan)]['Question'].drop_duplicates().dropna().tolist()
            section['Question'] = section['Question'].apply(lambda x: difflib.get_close_matches(x, questions)[0])
            for q in questions:      
                question = section.loc[section['Question']==q]
                stq = STQ_Dict.loc[STQ_Dict['Question']==q]
                score=int([x[-1] for x in boxes if question[f'{x}_value'].values[0]==True][0])
                raw = score
                
                if stq['Direction'].values[0]=='R':
                    score = (section_max[s]+1-score)*stq['Weight'].values[0]
                else:
                    score = (score)*stq['Weight'].values[0]


                result = dict()
                result['Section']=s
                result['Question']=q
                result['Raw']=raw
                result['Score']=score
                result['Attribute']=stq['Attribute'].values[0]
                result['Method']=stq['Method'].values[0]
                results = pd.concat([results, pd.DataFrame([result])], ignore_index=True)
        
        results.to_csv('stq_test_raw.csv')
        data = results.copy()
        results = pd.DataFrame()
        
        for s in sections:
            section = data.loc[data['Section']==s]
            attributes = section['Attribute'].drop_duplicates().dropna().tolist()
            for a in attributes:
                attribute = data.loc[(data['Section']==s) & (data['Attribute']==a)]
            
                method = attribute['Method'].values[0]
                if method == 'M':
                    score = attribute['Score'].product()
                elif method== 'A':
                    score = attribute['Score'].sum()
                elif method== 'W':
                    score = attribute['Score'].sum()/(len(attribute)*section_max[s])*10

                score = score + section_shift[s]

                P.cntSTQProcessed += 1

                result = dict()
                result['Section'] = s
                result['Attribute'] = a
                result['Score'] = score
                results = pd.concat([results, pd.DataFrame([result])], ignore_index=True)

                try:
                    _calc= {f'STQ_{a}':score,}
                    merge_dict(_calc,calc)
                except Exception as z:
                    print_status(f'Error {EVM_task} on {q}',f"{path} - {z!r}")
                    errors = errors + [f"Failed {EVM_task} Question {q}: {path} - {z!r}",]

        results.to_csv('stq_test.csv')
            
            
        ### Update Participant Results
        P.psychoPyResults = pd.concat([P.psychoPyResults, pd.DataFrame([calc])], ignore_index=True)
    
        ### Update EVM Results File
        if os.path.exists(f"{out_folder}EVM_Results.csv"):
            old_log = pd.read_csv(f"{out_folder}EVM_Results.csv")
            #If participant already has data in clean log
            if P.ID in old_log['Participant'].tolist():
                for key, value in calc.items():
                    old_log.loc[old_log['Participant']==P.ID, key] = value
                new_log = old_log
            else:
                new_log = pd.concat([old_log, pd.DataFrame([calc])], ignore_index=True)
            new_log.to_csv(f"{out_folder}EVM_Results.csv",index=False)
        else:
            EVM_Results = pd.DataFrame()
            EVM_Results = pd.concat([EVM_Results, pd.DataFrame([calc])], ignore_index=True)
            EVM_Results.to_csv(f"{out_folder}EVM_Results.csv",index=False)

    ### Update Log File

    log = {'Participant':P.ID,
            'Date':P.date,
            'Status':P.status,      
            'PsychoPy':P.isData,
            'PsychoPyComplete':P.isDataComplete,
            'Task':EVM_task,
            'Files':P.cntSTQFiles,
            'Processed':P.cntSTQProcessed,
            'Errors':';'.join(errors)
            }

    if os.path.exists(f"{out_folder}psychoPy_log.csv"):
        old_log = pd.read_csv(f"{out_folder}psychoPy_log.csv")
        #If participant already has data in clean log
        if (P.ID in old_log['Participant'].tolist()) and (old_log.loc[old_log['Participant']==P.ID]['Task'].str.contains(EVM_task).any()):
            for key, value in log.items():
                old_log.loc[(old_log['Participant']==P.ID)&(old_log['Task'].str.contains(EVM_task)), key] = value
            new_log = old_log
        else:
            new_log = pd.concat([old_log, pd.DataFrame([log])], ignore_index=True)
        new_log.to_csv(f"{out_folder}psychoPy_log.csv",index=False)
    else:
        psychoPy_log = pd.DataFrame()
        psychoPy_log = pd.concat([psychoPy_log, pd.DataFrame([log])], ignore_index=True)
        psychoPy_log.to_csv(f"{out_folder}psychoPy_log.csv",index=False)
    
    #End

    #############################
    print_status(f"Collected {EVM_task}",f"{int(P.cntSTQProcessed/3*100)}%")  
    return P

def update_attributes(self):
    self.day = None
    self.time = None
    self.number = None

def get_attributes(in_folder, out_folder,P):
    '''
    This function cleans and combs through the OpenSignals and gets some metadata

    '''
    
    #Define variables
    psychoPy_log = pd.DataFrame()

    errors = []
    in_path = f"{in_folder}{P.ID}/"
    out_path = f"{out_folder}{P.ID}/"
    os.makedirs(out_path, exist_ok=True) 
    
    EVM_task = 'attributes'
    i = 0

    if P.isSignals:
        try:
            
            #cols = [f'{P.device}_{c}' for c in P.signals.columns if 'hSp' in c]
            date = datetime.datetime.strptime(P.metadata['date'],'%Y-%m-%d')
            day_of_year = date.timetuple().tm_yday  # returns 1 for January 1st
            P.day = day_of_year
            P.time = P.tasksInfo['Start'].min()
            P.number = P.ID[:-1]

            calc = {f"Participant":P.ID,
                    f"Number":P.number,
                    f"Drink":P.drink,
                    f"Device":P.device,
                    f'StartTime':P.time,
                    f'Day':P.day,
                    }
            i+=1
            ### Update Participant Results
            P.psychoPyResults = pd.concat([P.psychoPyResults, pd.DataFrame([calc])], ignore_index=True)
    
            ### Update EVM Results File
            if os.path.exists(f"{out_folder}EVM_Results.csv"):
                old_log = pd.read_csv(f"{out_folder}EVM_Results.csv")
                #If participant already has data in clean log
                if P.ID in old_log['Participant'].tolist():
                    for key, value in calc.items():
                        old_log.loc[old_log['Participant']==P.ID, key] = value
                    new_log = old_log
                else:
                    new_log = pd.concat([old_log, pd.DataFrame([calc])], ignore_index=True)
                new_log.to_csv(f"{out_folder}EVM_Results.csv",index=False)
            else:
                EVM_Results = pd.DataFrame()
                EVM_Results = pd.concat([EVM_Results, pd.DataFrame([calc])], ignore_index=True)
                EVM_Results.to_csv(f"{out_folder}EVM_Results.csv",index=False)
        except Exception as z:
            print_status(f'Failed {EVM_task}',f"{P.ID} - {z!r}")
            errors = errors + [f"Failed {EVM_task}: {P.ID} - {z!r}",]


    #############################
    print_status(f"Collected {EVM_task}",f"{i/1*100}%")  
    return P

def split_by_drink(in_folder, out_folder):
    '''
    This function cleans up EDA data, and then devides it into epochs. 
    At this point, no files need to be read in. All data needed is contained in the participants pickle


    Parameters
    __________
    in_folder: str
        Folder containing directories for each participant
        Must contain ONLY particpant 
    out_folder: str
        Folder in which a directory will be created per participant
        
    TODO:
    - Add in check for duplicate participant IDs
    '''
    
    
    #Define variable
    out_path = f"{out_folder}"
    os.makedirs(out_path, exist_ok=True) 
    
    print_step(f'Splitting by Drink','EVM_Results')

    df = pd.read_csv(f"{out_folder}EVM_Measurements_Aggeragated_Clean.csv")
    gb = df.groupby(['Drink'])
    for x in gb.groups:
        data = gb.get_group(x)
        drink = data['Drink'].drop_duplicates().tolist()[0]
        data.to_csv(f"{out_folder}EVM_Measurements_Aggeragated_Cleen_{drink}.csv")

    print_status(f'Completed Splitting','EVM_Results')  
##### EmotionRecognition Data


def get_schedule(in_folder, out_folder,p):
    '''
    This function cleans and combines all the OpenSignals data from one participant into a single file

    Parameters
    __________
    in_folder: str
        Folder containing directories for each participant
        Must contain ONLY particpant 
    out_folder: str
        Folder in which a directory will be created per participant
        
    TODO:
    - Add in check for duplicate participant IDs
    '''
    
    #Define variables
    dur = pd.DataFrame()

    in_path = f"{in_folder}{p}/"
    out_path = f"{out_folder}{p}/"
    os.makedirs(out_path, exist_ok=True) 
    
    EVM_task = 'attributes'


    try:
        dur = pd.read_csv(f'{out_path}{p}_EVM_log.csv')[['Task','Time','Duration']]
        dur.loc[:,'Participant']=p
        if os.path.exists(f"{out_folder}EVM_Schedule.csv"):
            results = pd.read_csv(f'{out_folder}EVM_Schedule.csv')
        else:
            results = pd.DataFrame()
        results = pd.concat([results,dur])
        results.to_csv(f'{out_folder}EVM_Schedule.csv')
    except Exception as z:
        print_status(f'Failed {EVM_task}',f"{p} - {z!r}")
        


    #############################
    print_status(f"Collected {EVM_task}",f"{p}")  
    return p


if __name__ == "__main__":
    main()