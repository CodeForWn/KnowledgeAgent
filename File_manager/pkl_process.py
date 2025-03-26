import pickle as pkl
import json
import os
from tqdm import tqdm
import chardet

root_path='./chaojubian/file_json/'
file_list=os.listdir(root_path)
db_map={}
db_map['db_id']=1
db_map_file_list=[]
fout_db=open('./file_db/1.pkl','wb')
for index,item in enumerate(file_list):
    with open(root_path+item,'r') as f:
        data=json.load(f)
    map={}
    file_list=[]
    file_list_map={}
    file_list_map['name']=item
    id=int(item[0])
    file_list_map['id']=id
    file=[]
    for line in data:
        map_file={}
        map_file['page']=int(line['section'][8])
        map_file['text']=line['text']
        file.append(map_file)
    file_list_map['file']=file
    file_list.append(file_list_map)
    db_map_file_list.append(file_list_map)
    map['file_list']=file_list
    map['db_id']=1
    fout=open(f'./file/1/{id}.pkl','wb')
    pkl.dump(map,fout)
    fout.close()
db_map['file_list']=db_map_file_list
pkl.dump(db_map,fout_db)
fout_db.close()