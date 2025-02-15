### **📘 Prediction Script**  
This document explains how to use the **prediction script** (`predict.py`) for running **DOHLCV-based predictions** on financial market data. It also details the optional alternative data format that will be automatically reformatted into DOHLCV before processing.

---

## **Overview**  
The `predict.py` script allows you to run predictions on DOHLCV data. This script:  
- Loads and validates DOHLCV data (Date, Open, High, Low, Close, Volume).  
- Ensures proper formatting and handles missing values.  
- Converts data in an alternative format (if applicable) to DOHLCV.  
- Runs predictions using a trained model.  
- Saves the results to a CSV file.  

---

## **Installation & Setup**  
### **1. Install Dependencies**  
Ensure all required dependencies are installed:  
```bash
pip install -r requirements.txt
```

---

## **Running the Prediction Script**  
The prediction script is designed to be run via the command line.  

### **1. Basic Usage**  
To run a prediction on a dataset:  
```bash
python predict.py --input path/to/dohlcv_data.csv
```  
By default, the predictions are saved to `predictions.csv`.  

### **2. Specify a Custom Output File**  
```bash
python predict.py --input path/to/dohlcv_data.csv --output results.csv
```  

### **3. Automating Predictions (Optional)**  
If you want to schedule predictions to run automatically, use cron jobs or shell scripts:  
```bash
0 9 * * * /usr/bin/python3 /path/to/predict.py --input /data/todays_data.csv --output /results/todays_predictions.csv
```  
This runs the script daily at 9 AM.  

For batch processing multiple files:  
```bash
for file in data/*.csv; do
    python predict.py --input "$file" --output "predictions/$(basename "$file" .csv)_predictions.csv"
done
```

---

## **Input & Output Data Format**  
### **Supported Input Formats**
The script supports two formats: **Standard DOHLCV format** and an **Alternative Format**.  
If the alternative format is provided, it will be **automatically reformatted into DOHLCV format** before processing.

### **📥 Standard Input: DOHLCV Data Format**  
This is the default expected format.  

| Date       | Open  | High  | Low   | Close | Volume |  
|------------|-------|-------|-------|-------|--------|  
| 2025-02-10 | 1.093 | 1.099 | 1.092 | 1.096 | 120000 |  
| 2025-02-11 | 1.096 | 1.103 | 1.095 | 1.101 | 125000 |  

### **📥 Alternative Input Format (Will Be Reformatted)**
If your data contains `mid_*`, `bid_*`, and `ask_*` price levels, it will be converted into the DOHLCV format.

#### **Example Alternative Format**
```
,time,volume,mid_o,mid_h,mid_l,mid_c,bid_o,bid_h,bid_l,bid_c,ask_o,ask_h,ask_l,ask_c
0,2018-01-01 22:00:00-05:00,35630,1.20039,1.20812,1.20019,1.2058,1.20009,1.20806,1.19975,1.2055,1.20069,1.20819,1.20051,1.2061
1,2018-01-02 22:00:00-05:00,31354,1.2058,1.20666,1.2001,1.20144,1.2055,1.20658,1.20002,1.20119,1.2061,1.20673,1.20018,1.2017
```

#### **How the Script Reformats Alternative Format**  
1. The `time` column is renamed to `Date`.  
2. The `mid_*` columns are renamed to DOHLCV equivalents (`Open`, `High`, `Low`, `Close`).  
3. The `volume` column is retained.  
4. The `bid_*` and `ask_*` columns are **dropped** as they are not needed.  

#### **📤 Reformatted Output (Converted to DOHLCV)**
| Date                        | Open   | High   | Low    | Close  | Volume |
|-----------------------------|--------|--------|--------|--------|--------|
| 2018-01-01 22:00:00-05:00  | 1.20039 | 1.20812 | 1.20019 | 1.20580 | 35630  |
| 2018-01-02 22:00:00-05:00  | 1.20580 | 1.20666 | 1.20010 | 1.20144 | 31354  |

If your data is already in DOHLCV format, **no reformatting will be applied**.

---

### **📤 Output: Prediction Format**  
The output file includes input data with predictions.

| Date       | Open  | High  | Low   | Close | Volume | Predictions | Confidence |  
|------------|-------|-------|-------|-------|--------|-------------|------------|  
| 2025-02-10 | 1.093 | 1.099 | 1.092 | 1.096 | 120000 | 1.095       | 0.98       |  
| 2025-02-11 | 1.096 | 1.103 | 1.095 | 1.101 | 125000 | 1.102       | 0.97       |  

---

## **Error Handling & Logging**  
- If the input file is missing, an error is raised.  
- If the `Date` column is missing or incorrectly formatted, an error is logged.  
- If DOHLCV records are invalid, errors are logged.  
- If fewer predictions are returned than input records, a warning is logged.  

Logs will appear in the terminal. Example:  
```
[ 2025-02-15 06:10:10 ] INFO - Loading data from dohlcv_data.csv...  
[ 2025-02-15 06:10:12 ] INFO - Successfully reformatted to DOHLCV format.  
[ 2025-02-15 06:10:14 ] WARNING - Mismatch: Input records (1847) and predictions (1825). Adjusting output...  
[ 2025-02-15 06:10:16 ] INFO - Predictions saved to predictions.csv  
```

---

## **Why a CLI-Based Script?**  
This script is designed as a command-line tool instead of a web-based API because:  
- It is easier to automate using cron jobs or batch scripts.  
- It processes data faster than a web-based API.  
- It is lightweight and runs on any system with Python installed.  
- It can be used for batch processing multiple files.  

If real-time predictions are needed, the script can be extended into a FastAPI or Flask-based API in the future.  

---

## **Notes**  
- Ensure the model is trained and stored in the appropriate location.  
- If errors occur, check the logging output for troubleshooting.  

---

## **Future Enhancements**  
- Add support for JSON input/output format.  
- Implement real-time inference via FastAPI.  Not sure this is needed for my purposes.
