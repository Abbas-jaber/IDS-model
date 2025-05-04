import csv
import sys
import datetime
from dateutil import parser
from collections import defaultdict
import os

def is_epoch(timestamp):
    """Check if a string is an epoch timestamp."""
    try:
        float_val = float(timestamp)
        return 1000000000 <= float_val <= 2000000000  # Rough range for valid epoch times
    except ValueError:
        return False

def cleanData(inFile, outFile, out_dir):
    count = 1
    stats = {}
    dropStats = defaultdict(int)
    print(f'Cleaning {inFile}')
    
    # Create output directory if it doesn't exist
    os.makedirs(out_dir, exist_ok=True)

    with open(inFile, 'r') as csvfile:
        data = csvfile.readlines()
        totalRows = len(data)
        print(f'Total rows read = {totalRows}')
        header = data[0]

        for line in data[1:]:
            line = line.strip()
            cols = line.split(',')
            key = cols[-1]

            if line.startswith('D') or 'Infinity' in line or 'infinity' in line:
                dropStats[key] += 1
                continue
            
            # Check if cols[2] is already an epoch timestamp
            if not is_epoch(cols[2]):
                try:
                    dt = parser.parse(cols[2])  # Convert date string to datetime object
                    epochs = (dt - datetime.datetime(1970, 1, 1)).total_seconds()
                    cols[2] = str(int(epochs))  # Convert to integer for consistency
                except Exception as e:
                    print(f"Skipping row due to date parsing error: {e}")
                    dropStats[key] += 1
                    continue

            line = ','.join(cols)
            count += 1

            if key in stats:
                stats[key].append(line)
            else:
                stats[key] = [line]

    # Define output file paths within the specified directory
    output_csv = os.path.join(out_dir, f"{outFile}.csv")
    output_stats = os.path.join(out_dir, f"{outFile}.stats")

    # Write cleaned data to output files
    with open(output_csv, 'w') as csvoutfile:
        csvoutfile.write(header)
        with open(output_stats, 'w') as fout:
            fout.write(f'Total Clean Rows = {count}; Dropped Rows = {totalRows - count}\n')
            for key in stats:
                fout.write(f'{key} = {len(stats[key])}\n')
                line = '\n'.join(stats[key])
                csvoutfile.write(f'{line}\n')
                # Write label-specific files
                label_out = os.path.join(out_dir, f"{outFile}-{key}.csv")
                with open(label_out, 'w') as labelOut:
                    labelOut.write(header)
                    labelOut.write(line)
            for key in dropStats:
                fout.write(f'Dropped {key} = {dropStats[key]}\n')

    print(f'All done writing {count} rows; dropped {totalRows - count} rows.')

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print('Usage: python clean2.py inputFile.csv outputFile outputDirectory')
    else:
        cleanData(sys.argv[1], sys.argv[2], sys.argv[3])