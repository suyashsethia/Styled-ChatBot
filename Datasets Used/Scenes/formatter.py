import re
import json
import os
import pprint

# list of all the files
files = []

# get all the files in the directory
files = os.listdir("./The_Big_Bang_Theory")

#sort files
files.sort(key=lambda f: int(re.sub('\D', '', f)))

# pprint.pprint(files)
final_data=[]
for file in files:
    
    # open the json file
    data=json.load(open("./The_Big_Bang_Theory/"+file))
    temp=data["Transcript"]
    # pprint.pprint(data["Episode Number"])
        
    final_data.extend(temp)
    
    
# remove the string in  between []

for i, sentence in enumerate(final_data):
    final_data[i]=re.sub(r'\[.*?\]', '', final_data[i])
    final_data[i]=final_data[i].replace('\u266a', '')
    final_data[i]="" if final_data[i].find(":") == -1 else final_data[i]
    


final_data=[x for x in final_data if x != '']
with open('The_Big_Bang_Theory.json', 'w') as outfile:
    json.dump(final_data, outfile)
    
    
# keep only those lines starting with Sheldon: and the line before it
only_sheldon=[]

for i, sentence in enumerate(final_data):
    if sentence.find("Sheldon:") != -1:
        # index1 = final_data[i-20:i].find(":")
        index2 = final_data[i].find(":")
        if(i-20>=0 and index2!=-1):
            only_sheldon.append({"dialog":[{"id":0, "sender":"Question", "text":'\n'.join(n for n in final_data[i-20:i])},{"id":1,"sender":final_data[i][:index2],"text":final_data[i][index2+1:]}]})
        
        else:
            only_sheldon.append({"dialog":[{"id":0, "sender":"Question", "text":final_data[i-1]},{"id":1,"sender":final_data[i][:index2],"text":final_data[i][index2+1:]}]})
    
with open("sheldon_chats_bigger_context.json", "w") as outfile:
    json.dump(only_sheldon, outfile)
    
    
    
    
        

