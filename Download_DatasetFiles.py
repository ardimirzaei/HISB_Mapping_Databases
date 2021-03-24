import pandas as pd
from Grab_gSheets_Functions import get_google_sheet, gsheet_to_df

# Requires Token.pickle and Credentials FIle

Dataset_list = ['HINTS_1',
'HINTS_2',
'HINTS_3',
'HINTS_PR',
'HINTS_4c1',
'HINTS_4c2',
'HINTS_4c3',
'HINTS_4c4',
'HINTS_FDA',
'HINTS_FDA_2',
'HINTS_5c1',
'HINTS_5c2',
'HINTS_5c3',
'CHIS_01',
'CHIS_03',
'CHIS_05',
'CHIS_07',
'CHIS_09',
'CHIS_11',
'CHIS_12',
'CHIS_13',
'CHIS_14',
'CHIS_15',
'CHIS_16',
'CHIS_17',
'CHIS_18',
'ANHCS',
'GSS']

SPREADSHEET_ID =  '' # INSERT GOOGLE SHEET ID HERE


for dataset in Dataset_list:
	    LoadedData =  gsheet_to_df(get_google_sheet(SPREADSHEET_ID, str(dataset+'!A:C')))
	    LoadedData['Coded'].replace('Unsure','unknown', inplace=True)
	    LoadedData['Coded'].replace('Unclassified', 'unknown', inplace=True)
	    LoadedData['Coded'].replace('Case', 'unknown', inplace=True)
	    LoadedData['Coded'].replace('Source', 'unknown', inplace=True)
	    LoadedData['Coded'].replace('EMR', 'unknown', inplace=True)
	    LoadedData['Coded'].replace('Gun', 'unknown', inplace=True)
	    LoadedData['Coded'].replace('TV', 'unknown', inplace=True)
	    LoadedData['Coded'].replace('Music', 'unknown', inplace=True)
	    LoadedData['Coded'].replace('Advertisement', 'unknown', inplace=True)
	    LoadedData['Coded'].replace('Computer', 'unknown', inplace=True)
	    LoadedData['Coded'].replace('Salience', 'unknown', inplace=True)
	    LoadedData['Coded'].replace('Art', 'unknown', inplace=True)
	    LoadedData['Coded'].replace('NA', 'unknown', inplace=True)
	    LoadedData['Coded'].replace('NA_Interaction', 'unknown', inplace=True)
	    LoadedData['Coded'].replace('NOENTRY', 'unknown', inplace=True)
	    
	    LoadedData.to_csv("Datasets/" +dataset+ ".csv")
	    print("Created Dataset : " + str(dataset))


HINTS_COMPLETE = [ # Take out the first dataset
'HINTS_2',
'HINTS_3',
'HINTS_PR',
'HINTS_4c1',
'HINTS_4c2',
'HINTS_4c3',
'HINTS_4c4',
'HINTS_FDA',
'HINTS_FDA_2',
'HINTS_5c1',
'HINTS_5c2',
'HINTS_5c3']


HINTS_ALL = pd.read_csv("Datasets/" + "HINTS_1"+ ".csv") 
for dataset in HINTS_COMPLETE:
	df = pd.read_csv("Datasets/" + dataset + ".csv")
	HINTS_ALL = HINTS_ALL.append(df, ignore_index=True)

HINTS_ALL.to_csv("Datasets/" + "HINTS_COMPLETE" + ".csv")
print("Created Dataset : HINTS_COMPLETE")

CHIS_COMPLETE = [ # Take out the first dataset
'CHIS_03',
'CHIS_05',
'CHIS_07',
'CHIS_09',
'CHIS_11',
'CHIS_12',
'CHIS_13',
'CHIS_14',
'CHIS_15',
'CHIS_16',
'CHIS_17',
'CHIS_18']

CHIS_ALL = pd.read_csv("Datasets/" + "CHIS_01" + ".csv") 
for dataset in CHIS_COMPLETE:
	df = pd.read_csv("Datasets/" +dataset+ ".csv")
	CHIS_ALL = CHIS_ALL.append(df, ignore_index=True)

CHIS_ALL.to_csv("Datasets/" + "CHIS_COMPLETE" + ".csv")
print("Created Dataset : CHIS_COMPLETE")
