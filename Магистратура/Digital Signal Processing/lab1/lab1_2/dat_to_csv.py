import pandas as pd
from sys import argv
from os.path import exists
from numpy import nan 
import re 

def get_unique(list_:list):
    result = []
    for element in list_:
        if element not in result:
            result.append(element)
    return(result)

def to_float(list_:list):
    result = []
    for i in range(len(list_)):
        try:
            result.append(float(list_[i]))
        except:
            result.append(list_[i])
    return(result)
        

def fill_by_nan(list_:list,heads:list):
    for i in range(len(list_)):
        keys = list_[i].keys()
        for head in heads:
            if head not in keys:
                list_[i][head] = nan

    
def main(file:str, separator:str):
    data = []
    with open(file, 'rt') as data_file:
        for index, line in enumerate(data_file):
            if index == 0 and line.split(' ')[0] != separator:
                Exception('Uncorrect separator value')
                return(None)
            splited_line = line.split(' ')
            if len(splited_line) <= 1:
                Exception(f'Invalid data in line: {index+1}\n \\t \ \"{line[:20]} ...\"')
            if splited_line[0] == separator:
                data.append({})
            keys = [splited_line[0]+f'_{i+1}' for i in range(len(splited_line)-1)]
            values =to_float([element for element in splited_line[1:]])
            for i in range(len(keys)):
                data[-1][keys[i]] = values[i]
        
        all_headers = []
        for element in data:
            all_headers += element.keys()
        all_headers = get_unique(all_headers)
        fill_by_nan(data, all_headers)
        frame = pd.DataFrame(data = data)
        frame.to_csv(re.sub('\.[^\.]*$','.csv',file),index = False)
        


        






if __name__ == '__main__':
    if len(argv) > 1 and exists(argv[1]):
        if len(argv) > 2:
            main(argv[1], argv[2])
        else:
            Exception('The name of the separating parameter was expected')
    else:
        Exception(f'The file does not exist or the path is specified incorrectly\n the value was received: {argv[1]}')
                

    
