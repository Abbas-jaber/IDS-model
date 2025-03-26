# Introduction


In this project, I aim to explore the capabilities of deep-learning framework in detecting and classifying network intursion traffic with an eye towards designing a ML-based intrusion detection model.

# Dataset


Downloaded from: https://www.unb.ca/cic/datasets/ids-2018.html
contains: 7 csv preprocessed and labelled files, top feature selected files, original traffic data in pcap format and logs
used csv preprocessed and labelled files for this research project

## How to Get the Dataset

To download this dataset:

1. Install the AWS CLI, available on **Mac, Windows, and Linux**.
2. Run the following command:

   ```sh
   aws s3 sync --no-sign-request --region <your-region> "s3://cse-cic-ids2018/" dest-dir

### Data Cleanup
- Dropped rows with `Infinity` values.
- Removed repeated headers from some files.
- Converted timestamp values from `DD-MM-YYYY` format (e.g., `15-2-2018`) to **UNIX epoch** (since `01/01/1970`).
- Separated data based on attack types for each data file.
- **~20K rows** were removed as part of the data cleanup process.

### Cleanup Process:
1. **Initial Cleanup:**  
   - Each dataset undergoes an initial cleanup using its respective `clean.py` script.
   
2. **Combining Datasets:**  
   - After the initial cleanup, datasets are combined using `combine.py`.  

3. **Final Cleanup:**  
   - Once combined, `clean2.py` is applied to the **binary** and **multiclass** CSV files.

See the corresponding scripts (`clean.py`, `combine.py`, and `clean2.py`) for details on each phase.

## Dataset Summary Table

| File Name   | Traffic Type                  | # Samples  | # Dropped |
|------------|--------------------------------|------------|------------|
| 02-14-2018 | Benign                        | 663,808     | 3,818      |
|            | FTP-BruteForce                 | 193,354     | 6          |
|            | SSH-Bruteforce                 | 187,589     | 0          |
| **Total**   |                                | **1,044,752** | **3,824**  |
| 02-15-2018 | Benign                        | 988,050     | 8,027      |
|            | DoS attacks-GoldenEye          | 41,508      | 0          |
|            | DoS attacks-Slowloris          | 10,990      | 0          |
| **Total**   |                                | **1,040,549** | **8,027**  |
| 02-16-2018 | Benign                        | 446,772     | 0          |
|            | DoS attacks-SlowHTTPTest       | 139,890     | 0          |
|            | DoS attacks-Hulk               | 461,912     | 0          |
| **Total**   |                                | **1,048,575** | **1**      |
| 02-22-2018 | Benign                        | 1,042,603   | 5,610      |
|            | Brute Force -Web               | 249         | 0          |
|            | Brute Force -XSS               | 79          | 0          |
|            | SQL Injection                  | 34          | 0          |
| **Total**   |                                | **1,042,966** | **5,610**  |
| 02-23-2018 | Benign                        | 1,042,301   | 5,708      |
|            | Brute Force -Web               | 362         | 0          |
|            | Brute Force -XSS               | 151         | 0          |
|            | SQL Injection                  | 53          | 0          |
| **Total**   |                                | **1,042,868** | **5,708**  |
| 03-01-2018 | Benign                        | 235,778     | 2,259      |
|            | Infiltration                   | 92,403      | 660        |
| **Total**   |                                | **328,182**  | **2,944**  |
| 03-02-2018 | Benign                        | 758,334     | 4,050      |
|            | Bot                            | 286,191     | 0          |
| **Total**   |                                | **1,044,526** | **4,050**  |


