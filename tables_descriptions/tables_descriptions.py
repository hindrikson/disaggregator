from disaggregator import data
import json


dataset_spacial_id = {}  # there are around 35 datasets
dataset_temporal_id = {}  # there are around 35 datasets


# loop though the first 100 ids.
# if dimension is "spatial" concat the data json dictionary to spatial_description json file
def write_descriptions(dimension=None, max_n=100):
    descriptions = {}
    for i in range(0, max_n):
        json_data = data.description_data(id=i, dimension=dimension, max_retries=3)
        if json_data:  # only append if data is not empty
            descriptions[str(i)] = json_data
        else:
            print(f"No data for ID {i} in dimension {dimension}")
            # continue for the next id
            continue
    for key, value in descriptions.copy().items():
        if "message" in value:
            del descriptions[key]
    # write descriptions to file
    with open(f"{dimension}_descriptions.json", "w") as f:
        json.dump(descriptions, f)


def simple_description(data, name=None):
    simple_description = {}
    for key, value in data.items():
        try:
            print("Description for ID:", key)
            if value["oep_metadata"] == None:
                print(f"No 'oep_metadata' found for ID {key}")
                title = value.get("title", "No title available")
                simple_description[key] = {
                    "title": title,
                    "description": "No description available",
                }
                continue
            title = value["oep_metadata"]["title"]
            description = value["oep_metadata"]["description"]
            simple_description[key] = {"title": title, "description": description}
        except Exception as e:
            print(f"Error processing ID {key}: {e}")
    if name:
        with open(f"{name}_simple_description.json", "w", encoding="utf-8") as f:
            json.dump(simple_description, f, ensure_ascii=False, indent=4)
    return simple_description


write_descriptions(dimension="spatial", max_n=80)
write_descriptions(dimension="temporal", max_n=40)

# get the first entry in spatial and temporal datasets
# Load the file as a list of dictionaries

with open("./spatial_descriptions.json", "r", encoding="utf-8") as f:
    spatial_data = json.load(f)
with open("./temporal_descriptions.json", "r", encoding="utf-8") as f:
    temporal_data = json.load(f)

# print the contentents of a entry in a readable format
entry_id = "30"
print(f"Entry {entry_id} in temporal datasets:")
print(json.dumps(temporal_data[entry_id], indent=4, ensure_ascii=False))

spatial_simple = simple_description(spatial_data, name="spatial")
temporal_simple = simple_description(temporal_data, name="temporal")
