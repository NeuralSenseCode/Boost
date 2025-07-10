from process_lib import *
from nslib import *
import gc
from shutil import copyfile
gc.enable()



'''
This script contains the high level functions required to process and analyse the data gathered for the distell project

'''

in_folder = f"./Analysis/Raw/"
out_folder = f"./Analysis/Results/"
pickle_folder = f"./Analysis/Pickles/"

def check_files(in_folder, out_folder):

    #Get Individual Sheet
    individual_sheet = pd.read_csv(f'{in_folder}Individual Sheets/Individual Sheet.csv')
    #individual_sheet['Comment'] = (individual_sheet['Abnormal Activity'].fillna('').astype(str) +'\nCPT: ' +individual_sheet['CPT Observations'].fillna('').astype(str))
    individual_sheet = individual_sheet.dropna(subset = ['Participant stripped','Drink Allocation'])
    individual_sheet['Participant stripped']= individual_sheet['Participant stripped'].astype(int)
    individual_sheet['Drink Allocation']=individual_sheet['Drink Allocation'].astype(str)
    #Check for folders
    participant_data_path = f'{in_folder}Participants/'
    files = get_files(participant_data_path)

    #Collapse all folders
    for file in files:
        collapse_folder(f'{participant_data_path}{file}')
    
    numbers = drop_duplicates([f[:3] for f in files])
    letters = drop_duplicates([f[-1] for f in files])

    results = []
    for n in numbers:
        
        for letter in letters:
            result = {}
            result['Number']=n
            result['Drink']=letter
            
            # Collect data
            count = 0
            data_files = []
            individual_data = individual_sheet.loc[(individual_sheet['Participant stripped']==int(n))]
            individual_data.loc[:, 'date_ind'] = individual_data['Date Import'].apply(lambda x: int(x.split('-')[2]+x.split('-')[1]+x.split('-')[0]))
            individual_data.loc[:, 'date_new'] = individual_data['Date Import'].apply(lambda x: x.split('-')[2]+'-'+x.split('-')[1]+'-'+x.split('-')[0])
            
            if f'{n}{letter}' in files:
                data_files = get_files(f'{participant_data_path}{n}{letter}/')
                count = len(data_files)
                count_os = len([x for x in data_files if 'SCH' in x])
                try:
                    count_p = len(drop_duplicates([x.split('_')[0] for x in data_files if 'EVM' in x]))
                    result['Count Participants']=count_p
                except:
                    result['Count Participants']=np.nan
                    pass
                result['Count']=count
                
                result['OS Count']=count_os
                #Check for folders
                try:
                    data = pd.DataFrame()
                    data['files'] = [x for x in data_files if 'EVM' in x]
                    data['date'] = data['files'].apply(lambda x: int(x.split('_')[2].replace('-', '')))
                    data['time'] = data['files'].apply(lambda x: int(x.split('_')[3].split('.')[0].replace('h', '')))
                    data['time2'] = data['files'].apply(lambda x: f"{x.split('_')[3].split('.')[0].replace('h', ':')}:00")
                    data['number'] = data['files'].apply(lambda x: x.split('_')[-1].split('.')[0] if len(x.split('_'))>4 else np.nan)
                    
                    dates = len(data['date'].unique())
                    time = data.sort_values('time')['time2'].values[0]

                    result['Date_File'] = data['files'].apply(lambda x: x.split('_')[2]).values[0]
                    result['Number of Dates'] = dates
                    result['Actual time'] = time
                except:
                    print(f'>> Filename error in {file}')

            

            comments = individual_data['Abnormal Activity'].values
            comment = ''
            for i in range(len(comments)):
                comment = comment + f'{i}:{comments[i]},'

            # cpts = individual_data['CPT Observations'].values
            # cpt = ''
            # for i in range(len(cpts)):
            #     cpt = cpt + f'{i}:{cpts[i]},'

            try:
                data = individual_data.loc[individual_data['date_ind']==result['Date_File']]
                date = data['date_new'].values[0]
                time = data['Time'].values[0]
                researcher = individual_data['Researcher'].values[0]
                result['Date'] = date
                result['Time'] = time
                result['Researcher'] = researcher
            except:
                print(f'Could not find date for {n}{letter}')



            # Set error codes
            if f'{n}{letter}' not in files:
                result['Err: Incorrect Drink'] = 1
                
            else:
                if len(comments)>1:
                    result['Err: Multiple Drinks'] = 1
                if count<10:
                    result['Err: Missing Data'] = 1
                if count>50:
                    result['Err: Excess Data'] = 1
                if len([x for x in data_files if 'opensignal' in x])<2:
                    result['Err: Missing OpenSignals'] = 1
                if len([x for x in data_files if '.mp4' in x])>3:
                    result['Err: Surplus Video'] = 1
                if len([x for x in data_files if 'MAIA' in x])<1:
                    result['Err: Missing MAIA'] = 1
                if len([x for x in data_files if 'EVM' in x])<1:
                    result['Err: Missing EVM'] = 1

            try:      
                trial = individual_data['Trial'].values[0]
            except:
                trial = np.nan
                pass
            result['Trial'] = trial
            result['Comment'] = comment
            results.append(result)

    results = pd.DataFrame(results)
    results.to_csv(f'{out_folder}file_check.csv', index=False)

def prep_demographics(in_folder, out_folder):
    '''
    Prep a file for demographic data extraction
    
    '''
    df = pd.read_csv(f'{in_folder}Individual Sheets/EVM Timings Final G3.csv')
    df['Number']=df['Study ID (same as per previous rounds)'].apply(lambda x: x.split('-')[1])
    df = df.replace('Yes',1)
    df = df.replace('No',0)
    df['Gender (50/50)'] = df['Gender (50/50)'].replace('Male',0)
    df['Gender (50/50)'] = df['Gender (50/50)'].replace('Female',1)
    df = df.rename({'Age (23-45)':'Age',
                    'Gender (50/50)':'Gender',
                    },axis=1)
    df = df[['Number','Age','Gender']]
    df = df.to_csv(f'{out_folder}demographics_G3.csv')


def rename_files(in_folder, out_folder):

    participant_data_path = f'{in_folder}Participants/'
    files = get_files(participant_data_path)
    #Collapse all folders
    for p in files:
        in_path = f'{participant_data_path}{p}/'
        all = get_files(in_path,tags=['.csv']) #Get list of all tasks

        #Correct for STAI mistake
        for file in all:
            try:
                df = pd.read_csv(in_path + file)
            except pd.errors.ParserError:
                with open(in_path + file, 'r', encoding='utf-8') as f:
                    text = f.read()
                    # Replace the problematic substring
                    text_fixed = text.replace('"I am "cool, calm, and collected""','I am cool calm and collected')
                with open(in_path + file, 'w', encoding='utf-8') as f:
                    f.write(text_fixed)
            except:
                pass

        for file in all:
            names = ['IFIS',
                    'BMS_00',
                    'BMS_Comprehension',
                    'BMS_01',
                    'BMS_02',
                    'IAAS_00',
                    'IAAS_01',
                    'BMS_NS',
                    'PSQI',
                    'STAI',
                    'MAIA',
                    'dualTask',
                    'caffeine',
                    'simpleRT_00',
                    'simpleRT_01',
                    'openEnded']
            for name in names:
                if (name in file) and ('EVM' in file):
                    new_name = file.replace(f'_{name}','').replace('EVM',name)
                    os.rename(f'{in_path}{file}',f'{in_path}{new_name}')
                
      
def fix_error(in_folder, out_folder):

    backup_dir = f'Analysis/Raw/Backup'
    participants_dir = f'Analysis/Raw/Participants'

    # Iterate through each participant subdirectory in the Backup folder.
    for participant in os.listdir(backup_dir):
        backup_participant_path = os.path.join(backup_dir, participant)
        
        # Check that this is indeed a directory.
        if os.path.isdir(backup_participant_path):
            # Construct the corresponding participant directory in Participants.
            target_participant_path = os.path.join(participants_dir, participant)
            
            # Create the target directory if it doesn't exist.
            if not os.path.exists(target_participant_path):
                os.makedirs(target_participant_path)
            
            # Loop over each file in the participant's Backup directory.
            for file_name in os.listdir(backup_participant_path):
                source_file = os.path.join(backup_participant_path, file_name)
                
                # Ensure this is a file.
                if os.path.isfile(source_file):
                    # Only process files that contain "IAAS" in the name.
                    if "IAAS" in file_name:
                        # If the file name indicates it is in the old format (i.e. contains "EVM")
                        if "EVM" in file_name:
                            # Expected backup pattern:
                            # {participant}_EVM_{date}_{time}_IAAS_{number}.csv
                            parts = file_name.split('_')
                            try:
                                # Parse components:
                                participant_code = parts[0]        # e.g. "068E"
                                # parts[1] is "EVM" which we ignore.
                                date = parts[2]                    # e.g. "2025-03-11"
                                time = parts[3]                    # e.g. "08h33.18.943"
                                iaas = parts[4]                    # Should be "IAAS"
                                # Remove extension from the number portion.
                                number = os.path.splitext(parts[5])[0]  # e.g. "00"
                                
                                # Construct the new file name in the correct format:
                                # {participant}_{IAAS}_{number}_{date}_{time}.csv
                                new_file_name = f"{participant_code}_{iaas}_{number}_{date}_{time}.csv"
                            except Exception as e:
                                print(f"Error parsing filename {file_name}: {e}")
                                new_file_name = file_name  # fallback to original name if parsing fails
                        else:
                            new_file_name = file_name
                            
                        # Build the target path with the new filename.
                        target_file = os.path.join(target_participant_path, new_file_name)
                        shutil.copy2(source_file, target_file)
                        print(f"Copied and renamed: {source_file} -> {target_file}")

def process_data(in_folder, out_folder, pickle_folder, participant = [False,]):
    ver = 1.1
    header(f"Processing Data (Version {ver})")
    #### Get files to iterate over
    in_folder = f'{in_folder}Participants/'
    participants = get_participants(in_folder, out_folder,participant, ver)
   
    ### Work each participant
    for p in participants:
        header(f'Now Working {p}')

        try:
            path = f"{pickle_folder}{p}/"
            os.makedirs(path, exist_ok=True)

            # Open correct pickle file
            try:
                with open(f"{path}{p}_Object.pickle", 'rb') as f:
                    P = pickle.load(f)
                    print_status('Loaded',p)       
            except:           
                P=Participant(p)
                P.drink = P.ID[-1]

            ### Clean data
            #P.isCleaned = 0
            try:
                if not P.isCleaned:
                    P = clean_data(in_folder, out_folder, pickle_folder,P)
                    save_pickle(pickle_folder,P)
                else:
                    print_status('Skipped',f'{P.ID} clean')
            except AttributeError:
                Participant.update = update_clean
                P.update()
                print_status('Update',f'{P.ID} clean')
                P = clean_data(in_folder, out_folder, pickle_folder,P)
                save_pickle(pickle_folder,P)


            #### MUST COME AFTER THE CLEAN STUFF - nGet attributes data
            Participant.update = update_attributes
            P.update()
            print_status('Update',f'{P.ID} attributes')
            P = get_attributes(in_folder, out_folder,P)
            save_pickle(pickle_folder,P)

            #Get Manual data
            #P.cntManualProcessed = 0
            try:
                if not P.cntManualProcessed:
                    P = get_manual(in_folder, out_folder,P)
                else:
                    print_status('Skipped',f'{P.ID} manual')
            except AttributeError:
                Participant.update = update_manual
                P.update()
                print_status('Update',f'{P.ID} manual')
                P = get_manual(in_folder, out_folder,P)

            # Get Temp data
            # P.cntTempProcessed = 0
            try:
                if not P.cntTempProcessed:
                    P = get_Temp(in_folder, out_folder,P)
                else:
                    print_status('Skipped',f'{P.ID} Temp')
            except AttributeError:
                Participant.update = update_Temp
                P.update()
                print_status('Update',f'{P.ID} Temp')
                P = get_Temp(in_folder, out_folder,P)

            #Get BMS_Comp data
            #P.cntBMS_CompProcessed = 0 
            try:
                if not P.cntBMSCProcessed:
                    P = get_BMSC(in_folder, out_folder,P)
                else:
                    print_status('Skipped',f'{P.ID} BMS_Comp')
            except AttributeError:
                Participant.update = update_BMSC
                P.update()
                print_status('Update',f'{P.ID} BMS_Comp')
                P = get_BMSC(in_folder, out_folder,P)

            #Get PSQI data
            #P.cntPSQIProcessed = 0 
            try:
                if not P.cntPSQIProcessed:
                    P = get_PSQI(in_folder, out_folder,P)
                else:
                    print_status('Skipped',f'{P.ID} PSQI')
            except AttributeError:
                Participant.update = update_PSQI
                P.update()
                print_status('Update',f'{P.ID} PSQI')
                P = get_PSQI(in_folder, out_folder,P)

            #Get MAIA data
            # P.cntMAIAProcessed = 0 
            try:
                if not P.cntMAIAProcessed:
                    P = get_MAIA(in_folder, out_folder,P)
                else:
                    print_status('Skipped',f'{P.ID} MAIA')
            except AttributeError:
                Participant.update = update_MAIA
                P.update()
                print_status('Update',f'{P.ID} MAIA')
                P = get_MAIA(in_folder, out_folder,P)

            #Get IFIS data
            #P.cntIFISProcessed = 0 
            try:
                if not P.cntIFISProcessed:
                    P = get_IFIS(in_folder, out_folder,P)
                else:
                    print_status('Skipped',f'{P.ID} IFIS')
            except AttributeError:
                Participant.update = update_IFIS
                P.update()
                print_status('Update',f'{P.ID} IFIS')
                P = get_IFIS(in_folder, out_folder,P)


             #Get caffeine data
            #P.cntcaffeineProcessed = 0 
            try:
                if not P.cntcaffeineProcessed:
                    P = get_caffeine(in_folder, out_folder,P)
                else:
                    print_status('Skipped',f'{P.ID} caffeine')
            except AttributeError:
                Participant.update = update_caffeine
                P.update()
                print_status('Update',f'{P.ID} caffeine')
                P = get_caffeine(in_folder, out_folder,P)

            ##Get STAI data
            # P.cntSTAIProcessed = 0 
            try:
                if not P.cntSTAIProcessed:
                    P = get_STAI(in_folder, out_folder,P)
                else:
                    print_status('Skipped',f'{P.ID} STAI')
            except AttributeError:
                Participant.update = update_STAI
                P.update()
                print_status('Update',f'{P.ID} STAI')
                P = get_STAI(in_folder, out_folder,P)

            # #Get IAAS data
            #P.cntIAASProcessed = 0 
            try:
                if not P.cntIAASProcessed:
                    P = get_IAAS(in_folder, out_folder,P)
                else:
                    print_status('Skipped',f'{P.ID} IAAS')
            except AttributeError:
                Participant.update = update_IAAS
                P.update()
                print_status('Update',f'{P.ID} IAAS')
                P = get_IAAS(in_folder, out_folder,P)


            #Get BMS data
            #P.cntBMSProcessed = 0 
            try:
                if not P.cntBMSProcessed:
                    P = get_BMS(in_folder, out_folder,P)
                else:
                    print_status('Skipped',f'{P.ID} BMS')
            except AttributeError:
                Participant.update = update_BMS
                P.update()
                print_status('Update',f'{P.ID} BMS')
                P = get_BMS(in_folder, out_folder,P)

            #Get Cycle data
            # P.cntCycleProcessed = 0 
            try:
                if not P.cntCycleProcessed:
                    P = get_Cycle(in_folder, out_folder,P)
                else:
                    print_status('Skipped',f'{P.ID} Cycle')
            except AttributeError:
                Participant.update = update_Cycle
                P.update()
                print_status('Update',f'{P.ID} Cycle')
                P = get_Cycle(in_folder, out_folder,P)

            #Get simpleRT data
            #P.cntSimpleRTProcessed = 0 
            try:
                if not P.cntSimpleRTProcessed:
                    P = get_simpleRT(in_folder, out_folder,P)
                else:
                    print_status('Skipped',f'{P.ID} simpleRT')
            except AttributeError:
                Participant.update = update_simpleRT
                P.update()
                print_status('Update',f'{P.ID} simpleRT')
                P = get_simpleRT(in_folder, out_folder,P)
            
            # Get ECG data
            #P.cntECGProcessed = 0
            try:
                if not P.cntECGProcessed:
                    P = get_ECG(in_folder, out_folder,P)
                else:
                    print_status('Skipped',f'{P.ID} ECG')
            except AttributeError:
                Participant.update = update_ECG
                P.update()
                print_status('Update',f'{P.ID} ECG')
                P = get_ECG(in_folder, out_folder,P)
      

            # ###Get EDA data
            #P.cntEDAProcessed = 0
            try:
                if not P.cntEDAProcessed:
                    P = get_EDA(in_folder, out_folder,P)
                else:
                    print_status('Skipped',f'{P.ID} EDA')
            except AttributeError:
                Participant.update = update_EDA
                P.update()
                print_status('Update',f'{P.ID} EDA')
                P = get_EDA(in_folder, out_folder,P)

            with open(f"{path}{p}_Object.pickle", 'wb') as f:
                pickle.dump(P, f, pickle.HIGHEST_PROTOCOL)

            # Update Clean Log

            update_clean_log(P.ID, ver, out_folder)
            backup_files(out_folder)
            gc.collect()


        except Exception as z:
            update_clean_log(P.ID, 0, out_folder)
            print_status('###################  Failed Participant',f"{p} - {z!r}")


def build_results(in_folder, out_folder,tags=[None,]):
    
    participants = get_files(out_folder)
    all_results = pd.DataFrame()
    ### Work each participant
    for p in participants:
        header(f'Now Working {p}')
        #get_schedule(in_folder, out_folder,p)

        path = f"{out_folder}{p}/"
        
        try:
            with open(f"{path}{p}_Object.pickle", 'rb') as f:
                P = pickle.load(f)
                print_status('Loaded',p) 
                
                results = P.psychoPyResults
                rem = {}
                cnt = 0
                for col in results.columns:
                    try:
                        data = results[col].dropna().values[-1]
                        cnt += 1
                    except:
                        print(f'nothing for {col}')
                        pass
                    rem[col]=data
                
                if cnt > 20:
                    all_results = pd.concat([all_results, pd.DataFrame([rem])], ignore_index=True)

        except:
            print(f'No data for {p}')

    all_results.to_csv(f'{out_folder}EVM_Results.csv')
            

def build_old_results(out_folder,tags=[None,]):
    
    participants = get_files(out_folder)
    participants = [p for p in participants if 'cort' not in p]
    participants = [p for p in participants if ('A' in p) or ('F' in p) or ('G' in p) or ('H' in p) or ('I' in p)]
    participants = [p for p in participants if len(p)==4]

    all_results = pd.DataFrame()
    ### Work each participant
    for p in participants:
        header(f'Now Working {p}')
        #get_schedule(in_folder, out_folder,p)

        path = f"{out_folder}{p}/"
        os.makedirs(path, exist_ok=True)
        

        try:
            with open(f"{path}{p}_Object.pickle", 'rb') as f:
                P = pickle.load(f)
                print_status('Loaded',p) 
                
                results = P.psychoPyResults
                rem = {}
                cnt = 0
                for col in results.columns:
                    try:
                        data = results[col].dropna().values[-1]
                        cnt += 1
                    except:
                        print(f'nothing for {col}')
                        pass
                    rem[col]=data
                
                if cnt > 20:
                    all_results = pd.concat([all_results, pd.DataFrame([rem])], ignore_index=True)

        except:
            print(f'No data for {p}')

    current_results = pd.read_csv(f'{out_folder}EVM_Results.csv')
    all_results = pd.concat(current_results, all_results)

    all_results.to_csv(f'{out_folder}EVM_Results.csv')


def get_cort():
    df = pd.read_csv(f'C:/Users/ashra/Desktop/Distell/Cortisol Raw/Cortisol_Corrected.csv', header=0, low_memory = False)
    df = df.replace("<","",regex=True)

    print(df['Participant'])

    intervals = ['S1','S2','S3','S4','S5']

    calc = pd.DataFrame()
    for index, row in df.iterrows():
        print(row['Participant'])
        values = {i:0 for i in intervals}
        for interval in intervals:
            try:
                values[interval]=float(row[interval].split('(')[0])
            except:
                pass
    
        result = pd.DataFrame([{'Participant':row['Participant'].split(" ")[1],
                            'Drink':row['Participant'].split(" ")[1][-1],
                            'Baseline':values['S1'],
                            'BeforeTreatment':values['S2'],
                            'TreatmentInitial':values['S3'],
                            'TreatmentBeforeTasks':values['S4'],
                            'TreatmentAfterTasks':values['S5'],}])
        
        calc = pd.concat([calc, result])

    calc.to_csv(f'C:/Users/ashra/Desktop/Distell/Raw/cortisol.csv')


def main():
    ### Must rename files for Boost

    #check_files(in_folder, out_folder)
    #prep_demographics(in_folder, out_folder)
    #rename_files(in_folder, out_folder)


    # Step 2: Update respondents

    #### For one person
    # participants = [
    #                 '053E',
    #                 '058D',
    #                 '065D',
    #                 '068E',
    #                 '069E',
    #                 '073D',
    #                 '076F',
    #                 '083F',
    #                 '084E',
    # ]

    ### EDA

    # eda = ['079E',
    #     '077E',
    #     '081E',
    #     '088F',
    #     '053F',
    #     '088E',
    #     '079F',
    #     '048F',
    #     '069F',
    #     '048D',
    #     '085F',
    #     '051F',
    #     '087D',
    #     '069E',
    #     '083F',]

    #process_data(in_folder, out_folder, pickle_folder, participant = ['100G'])

    #For some people
    #participants = get_files(in_folder)
    # participants = [p for p in participants if 'cort' not in p]
    # participants = [p for p in participants if 'STQ' not in p]
    #participants = [p for p in participants if ('110' in p)]
    # #participants = [p for p in participants if ('O' in p) or ('S' in p) or ('Q' in p) or ('R' in p) or ('P' in p)]
    #participants = [p for p in participants if ('103O' in p) or ('J' in p)]
    # #participants = [p for p in participants if int(p[:3])>732]
    #process_data(in_folder, out_folder, participant = participants)

    #### For all people
    process_data(in_folder, out_folder, pickle_folder)
    #split_by_drink(in_folder, out_folder)

    #df = pd.read_csv(f"{out_folder}EVM_Results.csv")
    #cols = [c for c in df.columns if 'Unnamed' not in c]
    #df[cols].to_csv(f"{out_folder}EVM_Results.csv", index=True)

    #build_results(in_folder, out_folder)
    #build_old_results(out_folder)
   

if __name__ == "__main__":
    main()