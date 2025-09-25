from disaggregator.opendata_api import database_raw
from disaggregator.utils import literal_converter, dict_region_code
import requests
import pandas as pd

# url = "https://api.opendata.ffe.de"
# params = {"id_opendata": 50, "internal_id_1": 1}

url = "https://api.opendata.ffe.de/demandregio/demandregio_spatial?id_spatial=12&year=2018"
url = "https://api.opendata.ffe.de/demandregio/demandregio_temporal?id_spatial=12&year=2018"
url = "https://api.opendata.ffe.de/demandregio/demandregio_temporal?id_temporal=28&year=2011&internal_id_1=0"
url = "https://api.opendata.ffe.de/demandregio/demandregio_temporal?id_temporal=30"


# check health
def health_check(url):
    # check if the API is available by making a GET request to the /health endpoint
    response = requests.get(url + "/health")
    if not response.status_code == 200:
        print(f"API not available. Status code: {response.status_code}")
        print(response.json())
        exit()

    print("API available.")


health_check(url)

query = "demandregio_temporal?id_temporal=30"

for i in range(2000, 2024):
    query = "demandregio_temporal?id_temporal=30&year_weather={}&year_base={}".format(
        i, i
    )
    df = database_raw(query, max_retries=3)
    if df == None:
        print(f"\nNo data for year {i}")
        continue

# check if there are duplicated id_region values
duplicates = df["id_region"].duplicated().sum()
duplicates


df = (
    df.assign(nuts3=lambda x: x.id_region.map(dict_region_code()))
    .loc[lambda x: (~(x.nuts3.isna()))]
    .set_index("nuts3")
    .sort_index(axis=0)
    .loc[:, "values"]
    .apply(literal_converter)
)

df_exp = pd.DataFrame(df.values.tolist(), index=df.index).astype(float)

# check with id_region values are duplicated
df.head(24)

print(df.head())
print("Length")
print(len(df))
