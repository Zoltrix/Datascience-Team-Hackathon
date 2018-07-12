import json
import pandas as pd
import numpy as np

#Read JSON data into the datastore variable
with open("data/train.json", 'r') as f:
        datastore = json.load(f)

#Use the new datastore datastructure
print(type(datastore[0]))

columns=['id', 'cuisine', 'ingredient']
df = pd.DataFrame(columns=columns)

for item in datastore:
   id = item['id']
   cuisine = item['cuisine']
   for ing in item['ingredients']:
       df.append([id, cuisine, ing], ignore_index=True)

print(df.head())


#df.append(item['id'],item['cuisine'],item['ingredients'])



