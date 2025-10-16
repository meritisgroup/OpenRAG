from backend.database.data_extraction import DocumentText
from backend.utils.splitter import TextSplitter

import json
import os

list_doc = os.listdir("data/databases/scaling")

with open("data/base_config_server.json", 'r') as f:
    config_server = json.load(f)

config_server["data_preprocessing"] = "md_without_images"
path_dir = "data/databases/scaling"


reset_preprocess = False

for i, doc in enumerate(list_doc):
    DocumentText(doc_index=i,path=os.path.join(path_dir, doc), config_server=config_server)
