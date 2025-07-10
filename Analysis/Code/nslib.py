import pandas as pd
import os
import numpy as np
import pprint
import shutil
import scipy.signal as signal
import re

try:
    def get_key(z):
        print()
        print('-- Exception Raised --')
        print(f'>> Warning due to {z!r}')
        print('-- Press any key to continue...--')
        print()
except:
    pass

def header(text):
    print("-"*len(text))
    print(text)
    print("-"*len(text))


def sub_header(text, item):
    print(f"> {text} : {item}")
    print("-"*len(text))

def print_status(text, item):
    print(f">>>>> {text} : {item}")

def print_step(text, item):
    print(f">>> {text} : {item}")

def drop_duplicates(lst):
    """
    Removes duplicate items from the list while preserving order.

    Parameters:
        lst (list): The list from which to remove duplicates.

    Returns:
        list: A new list containing the unique items in their original order.
    """
    seen = set()
    result = []
    for item in lst:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


def collapse_folder(root):
    """
    Moves all files from subfolders of the given root directory into the root folder,
    renaming files if necessary to avoid name collisions, and then removes the now empty subfolders.
    
    Parameters:
        root (str): The path to the root folder.
    """
    # Walk the directory tree bottom-up to ensure subdirectories are processed before their parents.
    for dirpath, dirnames, filenames in os.walk(root, topdown=False):
        # Skip the root folder itself.
        if dirpath == root:
            continue
        
        # Move each file to the root folder.
        for filename in filenames:
            src = os.path.join(dirpath, filename)
            dst = os.path.join(root, filename)
            
            # If a file with the same name exists in the root, rename the file.
            if os.path.exists(dst):
                base, ext = os.path.splitext(filename)
                i = 1
                new_filename = f"{base}_{i}{ext}"
                new_dst = os.path.join(root, new_filename)
                while os.path.exists(new_dst):
                    i += 1
                    new_filename = f"{base}_{i}{ext}"
                    new_dst = os.path.join(root, new_filename)
                dst = new_dst

            shutil.move(src, dst)
        
        # After moving the files, remove the directory.
        try:
            os.rmdir(dirpath)
        except OSError:
            # Directory not empty (perhaps it contained subdirectories that weren't empty)
            print(f"Could not remove directory (not empty): {dirpath}")


def clean_and_convert_list(input_string):
    # Remove all unwanted characters except alphanumeric and commas
    cleaned_string = re.sub(r'[^a-zA-Z0-9,]', '', input_string)
    
    # Split the string by commas to form a list
    items = cleaned_string.split(',')
    
    # Strip whitespace and remove any empty items
    cleaned_list = [item.strip() for item in items if item.strip()]
    
    return cleaned_list


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


def group_by_range(df, col1, min, max, inc):
    df.sort_values(col1)
    bins = np.arange(min,max,inc)
    ind = np.digitize(df[col1],bins)
    return df.groupby(ind).mean().reset_index()


def merge_dict(dict1, dict2):
    return(dict2.update(dict1))


def filter(data, freqn, type):
    y = data
    b, a = signal.butter(8, freqn,type)
    yfs = signal.filtfilt(b, a, y, padlen=10)
    return yfs

def get_sec(time_str):
    """Get seconds from time."""
    h, m, s = time_str.split(':')
    return int((int(h) * 3600 + int(m) * 60 + float(s)))


if __name__ == "__main__":
    pass

