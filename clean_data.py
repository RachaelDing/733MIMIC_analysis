import pandas as pd
import numpy as np

# convert x to numeric, if x is string, convert it to None 
def to_numeric(x):
    try:
        return float(x)
    except:
        return None

# clean chart events items, return the processed item ids and the not processed item ids
def clean_chart_events():

    heart_rate = [211,220045]
    heart_rate_range = [0, 350]

    sys_bp = [51,442,455,6701,220179,220050]
    sys_bp_range = [0, 375]

    dias_bp = [8368,8440,8441,8555,220180,220051]
    dias_bp_range = [0, 300]

    mean_bp = [456,52,6702,443,220052,220181,225312]
    mean_bp_range = [14, 330]

    resp_rate = [615,618,220210,224690]
    resp_rate_range = [0, 300]

    f_temp = [223761,678]
    f_temp_range = [78.8, 113]

    c_temp = [223762,676]
    c_temp_range = [26, 45]

    SpO2 = [646,220277]
    SpO2_range = [0, 100]

    glucose = [807,811,1529,3745,3744,225664,220621,226537]
    glucose_range =[33, 2000]

    ph= [780, 860, 1126, 1673, 3839, 4202, 4753, 6003, 220274, 220734, 223830, 228243]
    ph_range = [6.3, 8.4]
    
    
    item_list_sap = [723, 454, 184, 223900, 223901, 220739,
            51, 442, 455, 6701, 220179, 220050,
            211, 220045, 678, 223761, 676, 223762,
            223835, 3420, 3422, 190]
    
    item_list_other = [646, 618, 212, 161, 128, 550, 1125, 159,
                 1484, 51, 8368, 52, 5815, 8549, 5820, 8554, 5819, 8553,
                 834, 3450, 8518, 3603, 581, 3609, 8532,455, 8441, 456,
                 31, 5817, 8551, 113, 1703, 467, 80, 1337, 674, 432,
                 5813, 8547,617, 210, 637, 184, 723, 454, 198, 707,
                 704, 479, 54, 32, 547, 154,  676, 442, 678, 3420,
                 220277, 220210, 615, 224690, 
                 8368, 8440, 8441, 8555, 220180, 220051,
                 226755, 227013, 3348, 2981,
                 807, 811, 1529, 3745, 3744, 225664, 220621, 226537,
                 780, 860, 1126, 1673, 3839, 4202, 4753, 6003, 220274, 220734, 223830, 228243]
    
    item_list = list(set(item_list_sap+item_list_other))
    print(item_list)
    
    pivot_items = [heart_rate, sys_bp, dias_bp, mean_bp, resp_rate, f_temp, c_temp, SpO2, glucose, ph]
    pivot_ranges = [heart_rate_range, sys_bp_range, dias_bp_range, mean_bp_range,
                    resp_rate_range, f_temp_range, c_temp_range, SpO2_range, glucose_range, ph_range]
    
    processed_items = []

    # FiO2
    FiO2 = [3420, 190, 3422, 223835]
    for itemid in FiO2:
        item = pd.read_csv("./raw_data/rawTemp"+str(itemid)+".csv")
        #print(item[item["VALUE"].isnull()].count())
        item["VALUE"] = item["VALUE"].apply(lambda x: None if ((x < 21 and x > 1) or x>100 or x<=0) else x) 
        item["VALUE"] = item["VALUE"].apply(lambda x: x*100 if (x <=1 and x > 0) else x)
        #print(item[item["VALUE"].isnull()].count())
        item = item[item["VALUE"].notnull()]       
        item["VALUE"] = item["VALUE"].apply(lambda x: x/100)
        item.to_csv("./remove_outlier/roTemp"+str(itemid)+".csv")
        print("Saving roTemp"+str(itemid)+".csv")
        processed_items.append(itemid)

    # crr -- capillary refill rate
    crr = [3348]
    for itemid in crr:
        item = pd.read_csv("./raw_data/rawTemp"+str(itemid)+".csv")
        item["VALUE"] = item["VALUE"].apply(lambda x: 1 if ((x == 'Normal <3 secs') or (x== 'Brisk')) else x)
        item["VALUE"] = item["VALUE"].apply(lambda x: 2 if ((x == 'Abnormal >3 secs') or (x== 'Delayed')) else x)
        item.to_csv("./remove_outlier/roTemp"+str(itemid)+".csv")
        print("Saving roTemp"+str(itemid)+".csv")
        processed_items.append(itemid)

        
    # pivot items
    for i, pivot_item in enumerate(pivot_items):
        for itemid in pivot_item:
            if itemid in item_list:
                item = pd.read_csv("./raw_data/rawTemp"+str(itemid)+".csv")
                pivot_range = pivot_ranges[i]
                #print(item[item["VALUE"].isnull()].count())
                item["VALUE"] = item["VALUE"].apply(lambda x: to_numeric(x))                
                item["VALUE"] = item["VALUE"].apply(lambda x: None if x < pivot_range[0] else x)
                item["VALUE"] = item["VALUE"].apply(lambda x: None if x > pivot_range[1] else x)
                #print(item[item["VALUE"].isnull()].count())
                #item = item[item["VALUE"].notnull()]
                item.to_csv("./remove_outlier/roTemp"+str(itemid)+".csv")
                print("Saving roTemp"+str(itemid)+".csv")
                processed_items.append(itemid)

    # return processed_items and raw_items
    processed_items_set = set(processed_items) 
    raw_items = [item for item in item_list if item not in processed_items_set]
    return(processed_items, raw_items)


# clean lab events items, return the processed item ids and the not processed item ids
def clean_lab_events():
    #item 950824 does not exist
    item_list_sap = [50821, 50816, 51006, 51300, 51301, 50882,  50983, 50822, 50971, 50885]
    
    item_list_other = [51221, 50912, 50902, 51265, 50868, 51222, 50931, 51249, 51279, 51248,
                 51006, 51301, 50882, 50983, 50822, 50971,50885, 50821, 50816]
    item_list = list(set(item_list_sap+item_list_other))
    
    items_to_check  = [50862, 50868, 51144, 50882, 50885, 50806, 50902, 50912,
                      50809, 50931, 50810, 51221, 50811, 51222, 50813, 51265, 
                      50822, 50971, 51275, 51237, 51274, 50824, 50983, 51006, 
                      51300, 51301]
    upper_ranges = {50862:10, 50868:10000, 51144:100, 50882:10000,
                    50885:150, 50806: 10000, 50902: 10000, 50912: 150,
                    50809:10000, 50931:10000, 50810:100, 51221:100, 
                    50811:50, 51222:50, 50813:50, 51265:10000,
                    50822:30, 50971:30, 51275:150, 51237:50,
                    51274:150, 50824:200, 50983:200, 51006:300,
                    51300:1000, 51301:1000}
    
    processed_items = [item for item in item_list if item in items_to_check]
    processed_items_set = set(processed_items) 
    print(processed_items)
    raw_items = [item for item in item_list if item not in processed_items_set]

    for itemid in processed_items:
        item = pd.read_csv("./raw_data/rawTemp"+str(itemid)+".csv")
        upper_range = upper_ranges[itemid]
        item["VALUE"] = item["VALUE"].apply(lambda x: to_numeric(x))   
        item["VALUE"] = item["VALUE"].apply(lambda x: None if (x > upper_range or x<0) else x)
        item = item[item["VALUE"].notnull()]
        item.to_csv("./remove_outlier/roTemp"+str(itemid)+".csv")
        print("Saving roTemp"+str(itemid)+".csv")
    return(processed_items, raw_items)

# clean output events items, return the processed item ids and the not processed item ids
def clean_output_events():
    processed_items = [227488, 40055, 43175, 40069, 40094, 40715, 40473, 40085, 40057, 40056, 40405, 40428, 40086, 40096, 40651, 
                            226559, 226560, 226561, 226584, 226563, 226564, 226565, 226567, 226557, 226558, 227489,
                            43053, 43171, 43173, 43333, 43347,
                            43348, 43355, 43365, 43373, 43374, 43379, 43380, 43431,
                            43519, 43522, 43537, 43576, 43583, 43589, 43638, 43654,
                            43811, 43812, 43856, 44706, 45304, 227519]
    output_range = [0, 1200]
    for itemid in processed_items:
        item = pd.read_csv("./raw_data/rawTemp"+str(itemid)+".csv")
        #print(item[item["VALUE"].isnull()].count())
        item["VALUE"] = item["VALUE"].apply(lambda x: None if x < output_range[0] else x)
        item["VALUE"] = item["VALUE"].apply(lambda x: None if x > output_range[1] else x)
        #print(item[item["VALUE"].isnull()].count())
        item = item[item["VALUE"].notnull()]
        item.to_csv("./remove_outlier/roTemp"+str(itemid)+".csv")
        print("Saving roTemp"+str(itemid)+".csv")
    return(processed_items, raw_items)


# save the item in processed_items and raw_items with ealiest timestamp for each admission
def save_earliest_items(processed_items, raw_items, table_name):
    for itemid in processed_items:
        item = pd.read_csv("./remove_outlier/roTemp"+str(itemid)+".csv", index_col=None).iloc[:, 1:]
        item = item[item["VALUE"].notnull()]
        #df = df.groupby(["SUBJECT_ID","HADM_ID"]).min(df["CHARTTIME"])
        #print(df)
        if len(item) > 0:
            min_time = item.groupby(["SUBJECT_ID","HADM_ID"])["CHARTTIME"].agg(['min'])
            min_time = min_time.rename(columns={"min":"CHARTTIME"})
            data = pd.merge(item, min_time, on=["SUBJECT_ID","HADM_ID","CHARTTIME"],how="inner")
        #print(data)
        data.to_csv("./earliest_"+table_name+"_items/item"+str(itemid)+".csv")
        print("Saving item"+str(itemid)+".csv")

    for itemid in raw_items:
        item = pd.read_csv("./raw_data/rawTemp"+str(itemid)+".csv")
        item = item[item["VALUE"].notnull()]
        #df = df.groupby(["SUBJECT_ID","HADM_ID"]).min(df["CHARTTIME"])
        #print(df)
        if len(item) > 0:
            min_time = item.groupby(["SUBJECT_ID","HADM_ID"])["CHARTTIME"].agg(['min'])
            min_time = min_time.rename(columns={"min":"CHARTTIME"})
            data = pd.merge(item, min_time, on=["SUBJECT_ID","HADM_ID","CHARTTIME"],how="inner")
        #print(data)
        data.to_csv("./earliest_"+table_name+"_items/item"+str(itemid)+".csv")
        print("Saving item"+str(itemid)+".csv")



# Extract needed coloums of proceesed SpO2 
def SpO2():
    SpO2_ids = [646, 220277]
    SpO2 = pd.read_csv("./remove_outlier/roTemp"+str(SpO2_ids[0])+".csv")
    SpO2_temp = pd.read_csv("./remove_outlier/roTemp"+str(SpO2_ids[1])+".csv")
    SpO2 = SpO2.append(SpO2_temp)
    SpO2 = SpO2[["HADM_ID", "CHARTTIME","VALUE", "ITEMID"]]
    SpO2.to_csv("./gru_features/SpO2.csv")


# Extract needed coloums of proceesed HR -- Heart Rate
def HR():
    HR_ids = [211, 220045]
    HR = pd.read_csv("./remove_outlier/roTemp"+str(HR_ids[0])+".csv")
    HR_temp = pd.read_csv("./remove_outlier/roTemp"+str(HR_ids[1])+".csv")
    HR = HR.append(HR_temp)
    HR = HR[["HADM_ID", "CHARTTIME","VALUE", "ITEMID"]]
    HR.to_csv("./gru_features/HR.csv")


# Extract needed coloums of proceesed RR -- Respiratory Rate
def RR():
    RR_ids = [618, 615, 220210, 224690]
    RR = pd.read_csv("./remove_outlier/roTemp"+str(RR_ids[0])+".csv")
    RR = RR[["HADM_ID", "CHARTTIME","VALUE", "ITEMID"]]
    for RR_id in RR_ids[1:]:
        RR_temp = pd.read_csv("./remove_outlier/roTemp"+str(RR_id)+".csv")
        RR_temp = RR_temp[["HADM_ID", "CHARTTIME","VALUE", "ITEMID"]]
        RR = RR.append(RR_temp)
    RR.to_csv("./gru_features/RR.csv")


# Extract needed coloums of proceesed SBP -- Systolic Blood Pressure
def SBP():
    sbp_ids = [51,442,455,6701,220179,220050]
    sbp = pd.read_csv("./remove_outlier/roTemp"+str(sbp_ids[0])+".csv")
    sbp = sbp[["HADM_ID", "CHARTTIME","VALUE", "ITEMID"]]
    for sbp_id in sbp_ids[1:]:
        sbp_temp = pd.read_csv("./remove_outlier/roTemp"+str(sbp_id)+".csv")
        sbp_temp = sbp_temp[["HADM_ID", "CHARTTIME","VALUE", "ITEMID"]]
        sbp = sbp.append(sbp_temp)
    sbp.to_csv("./gru_features/sbp.csv")


# Extract needed coloums of proceesed DBP -- Dias Blood Pressure
def DBP():
    dbp_ids = [8368,8440,8441,8555,220180,220051]
    dbp = pd.read_csv("./remove_outlier/roTemp"+str(dbp_ids[0])+".csv")
    dbp = dbp[["HADM_ID", "CHARTTIME","VALUE", "ITEMID"]] 
    for dbp_id in dbp_ids[1:]:
        dbp_temp = pd.read_csv("./remove_outlier/roTemp"+str(dbp_id)+".csv")
        dbp_temp = dbp_temp[["HADM_ID", "CHARTTIME","VALUE", "ITEMID"]]
        dbp = dbp.append(dbp_temp)
    dbp.to_csv("./gru_features/dbp.csv")    


# Convert Celcius temperature to Fahrenheit temperature and concate the two temperatures
# Extract needed coloums of the two temperature
def Temperature():    
    tempf_ids = [223761,678]
    tempc_ids = [223762,676]

    # Fahrenheit temperature
    tempf = pd.read_csv("./remove_outlier/roTemp"+str(tempf_ids[0])+".csv")
    tempf = tempf[["HADM_ID", "CHARTTIME","VALUE", "ITEMID"]]
    for tempf_id in tempf_ids[1:]:
        tempf_temp = pd.read_csv("./remove_outlier/roTemp"+str(tempf_id)+".csv")
        tempf_temp = tempf_temp[["HADM_ID", "CHARTTIME","VALUE", "ITEMID"]]
        tempf = tempf.append(tempf_temp)

    # Celcius temperature    
    tempc = pd.read_csv("./remove_outlier/roTemp"+str(tempc_ids[0])+".csv")
    tempc = tempc[["HADM_ID", "CHARTTIME","VALUE", "ITEMID"]]
    for tempc_id in tempc_ids[1:]:
        tempc_temp = pd.read_csv("./remove_outlier/roTemp"+str(tempc_id)+".csv")
        tempc_temp = tempc_temp[["HADM_ID", "CHARTTIME","VALUE", "ITEMID"]]
        tempc = tempc.append(tempc_temp)
        
    # Convert Celcius to Fahrenheit
    print(tempc.head())
    tempc["VALUE"] = tempc["VALUE"].apply(lambda x : x*1.8 + 32)
    print(tempc.head())
    tempf = tempf.append(tempc)
    tempf.to_csv("./gru_features/temp.csv")


# Extract needed coloums of raw TGCS -- GCS Total
def TGCS():
    tgcs_ids = [198, 226755, 227013]
    tgcs = pd.read_csv("./raw_data/rawTemp"+str(tgcs_ids[0])+".csv")
    tgcs = tgcs[["HADM_ID", "CHARTTIME","VALUE", "ITEMID"]]
    for tgcs_id in tgcs_ids[1:]:
        tgcs_temp = pd.read_csv("./raw_data/rawTemp"+str(tgcs_id)+".csv")
        tgcs_temp = tgcs_temp[["HADM_ID", "CHARTTIME","VALUE", "ITEMID"]]
        tgcs = tgcs.append(tgcs_temp)
    #print(tgcs[tgcs["VALUE"].isnull()].count())
    tgcs["VALUE"] = tgcs["VALUE"].apply(lambda x: to_numeric(x))
    tgcs["VALUE"] = tgcs["VALUE"].apply(lambda x: None if (x <0) else x)
    #print(tgcs[tgcs["VALUE"].isnull()].count())
    tgcs.to_csv("./gru_features/tgcs.csv")



# Extract needed coloums of processed fio2 and crr - Capillary Refill
# Extract needed coloums of raw uo - Urine Output
# Concatenate the 3 features
def CRR_UO_FiO2():
    # CRR 
    crr_ids = [3348]
    crr = pd.read_csv("./remove_outlier/roTemp"+str(crr_ids[0])+".csv")
    crr = crr[["HADM_ID", "CHARTTIME","VALUE", "ITEMID"]]

    # UO
    uo_ids = [43053, 43171, 43173, 43333, 43347,
              43348, 43355, 43365, 43373, 43374, 43379, 43380, 43431,
              43519, 43522, 43537, 43576, 43583, 43589, 43638, 43654,
              43811, 43812, 43856, 44706, 45304, 227519]
    uo = pd.read_csv("./remove_outlier/roTemp"+str(uo_ids[0])+".csv")
    uo = uo[["HADM_ID", "CHARTTIME","VALUE", "ITEMID"]]
    for uo_id in uo_ids[1:]:
        uo_temp = pd.read_csv("./remove_outlier/roTemp"+str(uo_id)+".csv") 
        uo_temp = uo_temp[["HADM_ID", "CHARTTIME","VALUE", "ITEMID"]]
        uo = uo.append(uo_temp)
    #print(uo[uo["VALUE"].isnull()].count())
    uo["VALUE"] = uo["VALUE"].apply(lambda x: to_numeric(x))
    uo["VALUE"] = uo["VALUE"].apply(lambda x: None if (x <0) else x)
    #print(uo[uo["VALUE"].isnull()].count())

    # FiO2
    FiO2_ids = [3420, 190, 3422, 223835]
    FiO2 = pd.read_csv("./remove_outlier/roTemp"+str(FiO2_ids[0])+".csv")
    FiO2 = FiO2[["HADM_ID", "CHARTTIME","VALUE", "ITEMID"]]
    for FiO2_id in FiO2_ids[1:]:
        FiO2_temp = pd.read_csv("./remove_outlier/roTemp"+str(FiO2_id)+".csv") 
        FiO2_temp = FiO2_temp[["HADM_ID", "CHARTTIME","VALUE", "ITEMID"]]
        FiO2 = FiO2.append(FiO2_temp)

    # Concatenate CRR, UO, FiO2
    crr_uo_fio2 = crr.append(uo).append(FiO2)
    crr_uo_fio2["VALUE"] = crr_uo_fio2["VALUE"].apply(lambda x: to_numeric(x))
    crr_uo_fio2.to_csv("./gru_features/crr_uo_fio2.csv")



# Extract needed coloums of processed gluecose
def Gluecose():
    glucose_ids = [807,811,1529,3745,3744,225664,220621,226537]
    glucose = pd.read_csv("./remove_outlier/roTemp"+str(glucose_ids[0])+".csv")
    glucose = glucose[["HADM_ID", "CHARTTIME","VALUE", "ITEMID"]]
    for glucose_id in glucose_ids[1:]:
        glucose_temp = pd.read_csv("./remove_outlier/roTemp"+str(glucose_id)+".csv")
        glucose_temp = glucose_temp[["HADM_ID", "CHARTTIME","VALUE", "ITEMID"]]
        glucose = glucose.append(glucose_temp)
    #print(glucose[glucose["VALUE"].isnull()].count())
    glucose["VALUE"] = glucose["VALUE"].apply(lambda x: to_numeric(x))
    glucose["VALUE"] = glucose["VALUE"].apply(lambda x: None if (x <0) else x)
    #print(glucose[glucose["VALUE"].isnull()].count())
    glucose.to_csv("./gru_features/glucose.csv")  


# Extract needed coloums of processed ph
def Ph():
    ph_ids = [780, 860, 1126, 1673, 3839, 4202, 4753, 6003, 220274, 220734, 223830, 228243]
    ph = pd.read_csv("./remove_outlier/roTemp"+str(ph_ids[0])+".csv")
    ph = ph[["HADM_ID", "CHARTTIME","VALUE", "ITEMID"]]
    for ph_id in ph_ids[1:]:
        ph_temp = pd.read_csv("./remove_outlier/roTemp"+str(ph_id)+".csv")
        ph_temp = ph_temp[["HADM_ID", "CHARTTIME","VALUE", "ITEMID"]]
        ph = ph.append(ph_temp)
    #print(ph[ph["VALUE"].isnull()].count())
    ph["VALUE"] = ph["VALUE"].apply(lambda x: to_numeric(x))
    ph["VALUE"] = ph["VALUE"].apply(lambda x: None if (x <0) else x)
    #print(ph[ph["VALUE"].isnull()].count())
    ph.to_csv("./gru_features/ph.csv")  


# Generate gru network features
def gru_features():
    SpO2()
    HR()
    RR()
    SBP()
    DBP()
    Temperature()
    CRR_UO_FiO2()
    TGCS()
    Gluecose()
    Ph()


if __name__ == "__main__":
    print("Processing chart events.")
    (processed_items, raw_items) = clean_chart_events()
    print("Saving chart events items with ealiest timestamp")
    save_earliest_items(processed_items, raw_items, "chart")

    print("Processing lab events.")
    (processed_items, raw_items) = clean_lab_events()
    print("Saving lab events items with ealiest timestamp")
    save_earliest_items(processed_items, raw_items, "lab")

    print("Processing output events.")
    (processed_items, raw_items) = clean_output_events()
    print("Saving output events items with ealiest timestamp")
    save_earliest_items(processed_items, raw_items, "output")

    print("Saving GRU network features.")
    gru_features()




