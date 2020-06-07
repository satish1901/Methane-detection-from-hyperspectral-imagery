import pandas as pd
import json


#fname = "/home/bisque/Aerospace_project/without_negatives_annotation_plumes.json"
fname = "/home/satishkumar/skeleton-custom-mask-rcnn/boats/annotation_plumes.json"
df = pd.read_json(f"{fname}")
dfT = df.T
print(dfT.head(5))
print()
for c, column in enumerate(dfT.columns):
	print("{c}. column{column}")
# print("column values as series", df[column])
	# print("column as np ", df[column].values)

for i, idx in enumerate(dfT.index.values):
	print(f"{i}. index: {idx}")

# selection / slicing examples
dfT = dfT[dfT[colum].str.contains("some string")]
dfT = dfT[dft['shape_attribute'] == "some_condition"]

# prepare the final dataframe 
df_final = pd.DataFrame(columns=dfT.columns)
new_names = []
for file in files:
	# read jsons
	# extract the rows
	temp = dfT[slice]
	df_final = pd.concat([df_final, temp], axis=0)
	# some extra cleaning
	# e.g., df_final.reset_index()
	# get old_name e.g., tile0_data1.png	
	# old_names.append(old_name)
	# get new_name, e.g., image1.png
	
# saving complete data and assoc information
# append two additional columns at the same 'height'
df_final["old_names"] = pd.Series(data=old_names, index=df_final.index)
df_final["new_names"] = pd.Series(data=new_names, index=df_final.index)
# save
df_final.to_json("complete_data_w_exra_info.json")


# prepare to write json
df_final = df_final[[<ONLY THE COLUMNS FOR FINAL JSON: column1, column2, >]]
df_final = df_final.T
df_final.to_json("new_name_for_mrcnn.json")
