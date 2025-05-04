#! /usr/bin/env python3

# script to append all the csv data files into 1
import os
import pandas as pd
from sklearn.utils import shuffle

# Use raw string or forward slashes for the path
dataPath = r"C:\Users\Abbas\Desktop\ids-project\src\Cortex_Guard"  # use your path
fileNames = [
    '02-14-2018/02-14-2018.csv', '02-15-2018/02-15-2018.csv', '02-16-2018/02-16-2018.csv',
    '02-22-2018/02-22-2018.csv', '02-23-2018/02-23-2018.csv', '03-01-2018/03-01-2018.csv', '03-02-2018/03-02-2018.csv'
]

# Check if the first file exists
first_file = os.path.join(dataPath, fileNames[0])
if not os.path.exists(first_file):
    raise FileNotFoundError(f"File not found: {first_file}")

# Read the first file
df = pd.read_csv(first_file)
print(f"Initial shape: {df.shape}")

# Append remaining files
for name in fileNames[1:]:
    fname = os.path.join(dataPath, name)
    if not os.path.exists(fname):
        print(f"Warning: File not found: {fname}")
        continue
    print(f"Appending: {fname}")
    df1 = pd.read_csv(fname)
    df = pd.concat([df, df1], ignore_index=True)

# Shuffle the combined dataframe
df = shuffle(df)
print(f"Final shape after shuffling: {df.shape}")

# Save multiclass file
print("Creating multi-class file")
outFile = os.path.join(dataPath, 'Combined/IDS-2018-multiclass')
df.to_csv(outFile + '.csv', index=False)
df.to_pickle(outFile + '.pickle')

# Create binary-class file
print("Creating binary-class file")
df['Label'] = df['Label'].map({
    'Benign': 0,
    'FTP-BruteForce': 1, 'SSH-Bruteforce': 1,
    'DoS attacks-GoldenEye': 1, 'DoS attacks-Slowloris': 1,
    'DoS attacks-SlowHTTPTest': 1, 'DoS attacks-Hulk': 1,
    'Brute Force -Web': 1, 'Brute Force -XSS': 1,
    'SQL Injection': 1, 'Infilteration': 1, 'Bot': 1
})
print(df['Label'][1:20])

outFile = os.path.join(dataPath, 'Combined/IDS-2018-binaryclass')
df.to_csv(outFile + '.csv', index=False)
df.to_pickle(outFile + '.pickle')

print("All done...")