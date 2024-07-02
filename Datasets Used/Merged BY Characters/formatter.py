# correct the format of the data
import json
import re
# pretty print
import pprint
f = open("./The_Big_Bang_Theory.merged.json", "r")

data = f.read()

# removing the participants{inside data} from the data


data = re.sub(r', "participants": \{.*?\},', '', data)


# add a , before every {
data = re.sub(r'(?<=[^,])\{', r',{', data)


# removing the episode_id from the data
data = re.sub(r'"episode_id": \".*?\"', '', data)

# removing the (.*?) from the data
data = re.sub(r'\(.*?\)', '', data)

# replace \n with null
data = re.sub(r'\n', '', data)


# replacing {"id" : <number>
data = re.sub(r'{"id": \d+,', '', data)


# replacing "title": with "


data = re.sub(r'"title": "Scene:', '{"Scene": "', data)

# find the number of times the word "Scene" appears
n = data.count('"Scene": "')

for i in range(n):
    # add i:{ before first occurance of {"Scene"
    data = re.sub(r'{"Scene": "', '"'+str(i)+'":{ "Scene": "', data, 1)
    

    



# pprint.pprint(data)

with open('./123.txt', 'w') as outfile:
    outfile.write(data)


blabla = json.dumps(data)

# print(json)

# save the data in a new file

with open('./123.json', 'w') as outfile:
    json.dump(blabla, outfile)
