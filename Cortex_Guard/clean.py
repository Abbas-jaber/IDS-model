import csv
import sys
import datetime
from dateutil import parser
from collections import defaultdict
import os
import clean
import pandas as pd
import ollama

# Function to clean dataset
def cleanData(inFile, outFile):
    count = 1
    skipped_rows = 0
    stats = {}
    dropStats = defaultdict(int)
    print('Cleaning {}'.format(inFile))
     # Strip .csv if it exists to prevent double extension
    outFile = outFile.rstrip('.csv')  # Removes trailing .csv if present

    with open(inFile, 'r') as csvfile:
        data = csvfile.readlines()
        totalRows = len(data)
        print('Total rows read = {}'.format(totalRows))
        header = data[0]

        for line in data[1:]:
            line = line.strip()
            cols = line.split(',')
            key = cols[-1]  # Last column as grouping key

            # Drop invalid rows
            if line.startswith('D') or 'Infinity' in line or 'infinity' in line:
                dropStats[key] += 1
                continue

            # Convert date to Unix timestamp
            try:
                dt = parser.parse(cols[2])  # Attempt to parse the date
                epochs = (dt - datetime.datetime(1970, 1, 1)).total_seconds()
                cols[2] = str(epochs)
                line = ','.join(cols)
                count += 1

                if key in stats:
                    stats[key].append(line)
                else:
                    stats[key] = [line]
            except ValueError as e:
                dropStats[key] += 1
                skipped_rows += 1  # Increment skipped rows counter
                continue


    # Write cleaned data to files
    with open(outFile+".csv", 'w') as csvoutfile:
        csvoutfile.write(header)
        with open(outFile + ".stats", 'w') as fout:
            fout.write(f'Total Clean Rows = {count}; Dropped Rows = {totalRows - count}\n')
            for key in stats:
                fout.write(f'{key} = {len(stats[key])}\n')
                csvoutfile.write('\n'.join(stats[key]) + '\n')
                with open(f'{outFile}-{key}.csv', 'w') as labelOut:
                    labelOut.write(header)
                    labelOut.write('\n'.join(stats[key]))
            for key in dropStats:
                fout.write(f'Dropped {key} = {dropStats[key]}\n')

    print(f'All done writing {count} rows; Dropped {totalRows - count - skipped_rows} rows; Skipped {skipped_rows} rows')

# Function to clean all datasets in a directory
def cleanAllData():
    inputDataPath = '../Cortex_Guard'
    

    files = os.listdir(inputDataPath)
    for file in files:
        # Skip hidden files and directories
        if file.startswith('.') or os.path.isdir(file):
            continue
        
        # Process only .csv files
        if not file.lower().endswith('.csv'):
            print(f"Skipping non-CSV file: {file}")
            continue

        
        inputFile = os.path.join(inputDataPath, file)
        outputDataPath = '../Cortex_Guard/' + os.path.splitext(file)[0]
        outFile = os.path.join(outputDataPath, file)
        print("Folder " + outputDataPath + " Has been created successfully") 

        if not os.path.exists(outputDataPath):
            os.mkdir(outputDataPath)
        cleanData(inputFile, outFile)

# Function to generate chatbot responses
def ask_chatbot(question, context=""):
    prompt = f"{context}\n\nUser: {question}"
    try:
        response = ollama.generate(
            model='deepseek-r1:7b',
            prompt=prompt
        )
        return response['response']
    except Exception as e:
        return f"Chatbot error: {str(e)}"

# Function to describe the cleaning process
def clean_and_process_dataset():
    print("Dataset cleaning and processing complete!")
    return """
    The dataset was cleaned using the following steps:
    1. Rows starting with 'D' or containing 'Infinity' or 'infinity' were dropped.
    2. Dates in the third column were converted to Unix timestamps.
    3. The data was grouped by the last column (key) and saved into separate files.
    4. A binary-class version of the dataset was created by mapping labels to 0 (Benign) or 1 (Attack).
    """
def start_chatbot():
    """Handles chatbot interaction after cleaning."""
    cleaning_summary = clean_and_process_dataset()
    print("\nWhat would you like to know about the cleaning process?")
    print("Type 'exit' to quit.")
    return cleaning_summary  # Return the summary so it can be used later

# Main function to execute data cleaning and chatbot
def main():
    if len(sys.argv) < 2:
        print('Usage: python data_cleanup.py inputFile.csv outputFile')
    elif sys.argv[1] == 'all':
        cleanAllData()
    else:
        cleanData(sys.argv[1], sys.argv[2])
    # After cleaning, allow user interaction with the chatbot
    cleaning_summary = clean_and_process_dataset()

    print("\nWhat would you like to know about the cleaning process?")
    print("You can ask questions like:")
    print("- What steps were taken to clean the dataset?")
    print("- How many rows were dropped?")
    print("- What transformations were applied to the data?")
    print("Type 'exit' to quit.")

    start_chatbot()

    while True:
        try:
            user_input = input("\nYour question: ").strip()
            if user_input.lower() == 'exit':
                print("Exiting chatbot. Goodbye!")
                break
            if not user_input:  # Handle empty input
                continue
                
            response = ask_chatbot(user_input, cleaning_summary)
            print(f"\nChatbot: {response}")
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
