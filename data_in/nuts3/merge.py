import pandas as pd

# EUROSTAT table for postcode to nuts3 mapping. Taken on 18.09.2025 from https://gisco-services.ec.europa.eu/tercet/flat-files
df_eurostat = pd.read_csv(
    "data_in/nuts3/pc2024_NUTS-2024_v3.0/pc2024_DE_NUTS-2024_v1.0.csv",
    sep=";",
    converters={"CODE": str},
)
df_eurostat = df_eurostat.rename(columns={"NUTS3": "natcode_nuts3", "CODE": "postcode"})
# Remove '' around postcodes codes
df_eurostat["postcode"] = df_eurostat["postcode"].str.replace("'", "")
df_eurostat["natcode_nuts3"] = df_eurostat["natcode_nuts3"].str.replace("'", "")
print("Number of rows in postcode to nuts3 mapping:")
print(len(df_eurostat))
print(df_eurostat.head())
print(df_eurostat.info())

# opendatasoft for postcodes to AGS mapping taken on 18.09.2025 from https://public.opendatasoft.com/explore/dataset/georef-germany-postleitzahl/table/
df_soft = pd.read_csv(
    "data_in/nuts3/georef-germany-postleitzahl.csv",
    sep=";",
    converters={"Postleitzahl / Post code": str, "Kreis code": str},
)
df_soft = df_soft.rename(
    columns={
        "Postleitzahl / Post code": "postcode",
        "Kreis code": "ags_lk",
        "Kreis name": "name",
    }
)
df_soft = df_soft.drop(columns=["PLZ Name (short)", "Geometry", "geo_point_2d"])

# Merge
df_eurostat["postcode"] = df_eurostat["postcode"].astype(str).str.zfill(5)
df_soft["postcode"] = df_soft["postcode"].astype(str).str.zfill(5)

df_merge = df_eurostat.merge(
    df_soft[["postcode", "ags_lk", "name"]], on="postcode", how="left"
)

# select rows from df_merge with any NaN value
df_nan = df_merge[df_merge.isna().any(axis=1)]
df_merge = df_merge.dropna()
print(len(df_merge))

# from df_merge select only rows with unique ags_lk
df_merge = df_merge.drop_duplicates(subset=["ags_lk"])

# create a column called id_ags. Its values are are based on the ags_lk, if the value has zeros at the start, remove them. Add three zeros to the end of all of them.
df_merge["id_ags"] = df_merge["ags_lk"].str.lstrip("0") + "000"

# convert id_ags to integer
df_merge["id_ags"] = df_merge["id_ags"].astype(int)

print(df_merge.head())
print(len(df_merge))

# save file to csv in data_in/regional/nuts3_lk_2024.csv
df_merge.to_csv("data_in/regional/nuts3_lk_2024.csv", index=False)
