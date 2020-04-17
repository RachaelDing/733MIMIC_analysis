import numpy as np
import pandas as pd
NUM_CHUNK = 34

# load labelevents table and get the number of each item
def count_label_events():
    label_events = pd.read_csv('LABEVENTS.csv.gz')
    label_counts = label_events.groupby('ITEMID').count().sort_values(by=["ROW_ID"],ascending =False)
    label_counts.to_csv("LabCounts.csv")
    print("LabCounts.csv saved.")

# load ouputevents table and get the number of each item
def count_output_events():
    output_events = pd.read_csv('OUTPUTEVENTS.csv.gz')
    output_counts = output_events.groupby('ITEMID').count().sort_values(by=["ROW_ID"],ascending =False)
    output_counts.to_csv("OutputCounts.csv")
    print("OutputCounts.csv saved.")

# load chartevents table and get the number of each item
def count_chart_events(): 
    filename = "CHARTEVENTS.csv.gz"
    chunksize = 10000000
    counter = 0
    # split chartevents table intp multiple chunks
    for chunk in pd.read_csv(filename, chunksize=chunksize):
        chunk.to_csv("ChartChunk"+str(counter)+".csv")
        print("ChartChunk"+str(counter)+" saved.")
        counter = counter+1
    NUM_CHUNK = counter

    # count the number of each item in each chunk and save the result
    counter = 0
    for chunk in pd.read_csv(filename, chunksize=chunksize):
        chunk_count = chunk.groupby('ITEMID').count()
        chunk_count.to_csv("ChunkCount"+str(counter)+".csv")
        counter = counter+1
        print("ChunkCount"+str(counter)+" saved.")

    # load all ChunkCount*.csv files and calculate the total count of each item
    filename = "ChunkCount0.csv"
    total_counts = pd.read_csv(filename)
    for counter in range(1,counter):
        filename = "ChunkCount"+str(counter)+".csv"
        temp = pd.read_csv(filename)
        total_counts = total_counts.append(temp)
        print("Finished loading ChunkCount"+str(counter))
    total_counts = total_counts.groupby("ITEMID").sum()
    total_counts = total_counts.sort_values(by=["ROW_ID"],ascending =False)
    total_counts.to_csv("ChunkTotalCount.csv")
    print("Saved the total count into ChunkTotalCount.csv")


def save_output_events():
    # load the ouput events items corresponding to SAP-II 
    filename = "OUTPUTEVENTS.csv.gz"
    item_list_sap = [40055, 43175, 40069, 40094, 40715, 40473, 40085, 40057, 40056, 40405, 40428, 40086, 40096, 40651, 226559, 226560, 226561, 226584, 226563, 226564, 226565, 226567, 226557, 226558, 227488, 227489]
    lab = pd.read_csv(filename)
    for ITEMID in item_list_sap:
        item= lab[lab["ITEMID"]== ITEMID][["ROW_ID","SUBJECT_ID","HADM_ID","ITEMID","CHARTTIME","VALUE","VALUEUOM"]]
        item.to_csv("rawTemp"+str(ITEMID)+".csv")
        print("Saved rawTemp"+str(ITEMID)+".csv")
        
    # load 2 ouput events items with least missing values
    item_list = [40055, 40069]
    lab = pd.read_csv(filename)
    for ITEMID in item_list:
        if ITEMID not in item_list_sap:
            item= lab[lab["ITEMID"]== ITEMID][["ROW_ID","SUBJECT_ID","HADM_ID","ITEMID","CHARTTIME","VALUE","VALUEUOM"]]
            item.to_csv("rawTemp"+str(ITEMID)+".csv")
            print("Saved rawTemp"+str(ITEMID)+".csv")

# load the lab events items corresponding to SAP-II             
def save_lab_sap_events():
    filename = "LABEVENTS.csv.gz"
    item_list_sap = [50821, 50816, 51006, 51300, 51301, 50882, 950824, 50983, 50822, 50971, 50885]
    lab = pd.read_csv(filename)
    for ITEMID in item_list_sap:
        item= lab[lab["ITEMID"]== ITEMID][["ROW_ID","SUBJECT_ID","HADM_ID","ITEMID","CHARTTIME","VALUE","VALUENUM","VALUEUOM"]]
        item.to_csv("rawTemp"+str(ITEMID)+".csv")
        print("Saved rawTemp"+str(ITEMID)+".csv")
        
# load 20 lab events items with least missing values
def save_lab_other_events():
    item_list_sap = [50821, 50816, 51006, 51300, 51301, 50882, 950824, 50983, 50822, 50971, 50885]
    item_list = [51221, 50912, 50902, 51265, 50868, 51222, 50931, 51249, 51279, 51248,
                 51006, 51301, 50882, 50983, 50822, 50971,50885, 50821, 50816]
    filename = "LABEVENTS.csv.gz"
    lab = pd.read_csv(filename)
    for ITEMID in item_list:
        if ITEMID not in item_list_sap:
            item= lab[lab["ITEMID"]== ITEMID][["ROW_ID","SUBJECT_ID","HADM_ID","ITEMID","CHARTTIME","VALUE","VALUENUM","VALUEUOM"]]
            item.to_csv("rawTemp"+str(ITEMID)+".csv")
            print("Saved rawTemp"+str(ITEMID)+".csv")
        

def save_chart_events():
    # load the chart events items corresponding to SAP-II 
    item_list_sap = [723, 454, 184, 223900, 223901, 220739,
            51, 442, 455, 6701, 220179, 220050,
            211, 220045, 678, 223761, 676, 223762,
            223835, 3420, 3422, 190]
    filename = "ChartChunk0.csv"
    for ITEMID in item_list_sap:
        item = pd.read_csv(filename)
        item = item[item["ITEMID"]== ITEMID][["ROW_ID","SUBJECT_ID","HADM_ID","ICUSTAY_ID","ITEMID","CHARTTIME","VALUE","VALUENUM","VALUEUOM"]]
        for counter in range(1, NUM_CHUNK):
            filename = "ChartChunk"+str(counter)+".csv"
            temp = pd.read_csv(filename)
            temp = temp[temp["ITEMID"]== ITEMID][["ROW_ID","SUBJECT_ID","HADM_ID","ICUSTAY_ID","ITEMID","CHARTTIME","VALUE","VALUENUM","VALUEUOM"]]
            item = item.append(temp)
        item.to_csv("rawTemp"+str(ITEMID)+".csv")
        print("Saved rawTemp"+str(ITEMID)+".csv")

    # load 50 chart events items with least missing values
    item_list = [646, 618, 212, 161, 128, 550, 1125, 159,
                 1484, 51, 8368, 52, 5815, 8549, 5820,8554,5819,8553,
                 834, 3450, 8518, 3603, 581, 3609, 8532,455, 8441, 456,
                 31, 5817, 8551, 113, 1703, 467, 80, 1337, 674, 432,
                 5813, 8547,617, 210, 637, 184, 723, 454, 198, 707,
                 704, 479, 54, 32, 547, 154,  676, 442, 678, 3420]
    filename = "ChartChunk0.csv"
    for ITEMID in item_list:
        if ITEMID not in item_list_sap:        
            item = pd.read_csv(filename)
            item = item[item["ITEMID"]== ITEMID][["ROW_ID","SUBJECT_ID","HADM_ID","ICUSTAY_ID","ITEMID","CHARTTIME","VALUE","VALUENUM","VALUEUOM"]]
            for counter in range(1, NUM_CHUNK):
                filename = "ChartChunk"+str(counter)+".csv"
                temp = pd.read_csv(filename)
                temp = temp[temp["ITEMID"]== ITEMID][["ROW_ID","SUBJECT_ID","HADM_ID","ICUSTAY_ID","ITEMID","CHARTTIME","VALUE","VALUENUM","VALUEUOM"]]
                item = item.append(temp)
            item.to_csv("rawTemp"+str(ITEMID)+".csv")
            print("Saved rawTemp"+str(ITEMID)+".csv")


if __name__ == "__main__":
    count_label_events()
    count_output_events()
    count_chart_events()
    save_output_events()
    print("Finish saving all output events.")
    save_lab_sap_events()
    save_lab_other_events()
    print("Finish saving all lab events.")
    save_chart_events()
