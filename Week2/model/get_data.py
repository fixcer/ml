import os
import pandas as pd

data_folder = "../dataset/data_files"
folders = ["politics","tech"]

os.chdir(data_folder)

x = []
y = []

for i in folders:
    files = os.listdir(i)
    for text_file in files:
        file_path = i + "/" +text_file
        print("Reading file:", file_path)
        with open(file_path) as f:
            data = f.readlines()
        data = ' '.join(data)
        x.append(data)
        y.append(i)
   
data = {'text': x, 'category': y}       
df = pd.DataFrame(data)
print('Writing csv flie ...')
df.to_csv('../dataset.csv', index=False)
