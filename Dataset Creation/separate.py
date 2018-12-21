import os
import glob 
import shutil

# print(len(files_list))
current_path = './Dataset/'
files_list = [f for f in os.listdir(current_path)]
passport_path = '../Doc_Scaner/Passport/'
pan_licence = '../Doc_Scaner/Pan_Licence/'
calibri = 'calibri'
verdana = 'verdana'

for f in files_list:
    f1 = f.split('.')[0]
    if f1[-5] == 'l':
        print(f, '\n', f[-5])
        continue
    if calibri in f:
        shutil.move(current_path + f, pan_licence)
    if verdana in f:
        shutil.move(current_path + f, passport_path)
