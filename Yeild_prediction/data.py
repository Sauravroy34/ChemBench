import numpy as np 
import pandas as pd
import requests
import io
BH_PROMPT = """
You are an expert chemist, your task is to predict the yield of reaction using your experienced chemical yield prediction knowledge.
Please strictly follow the format, no other information can be provided. Given the SMILES string of a Buchwald-Hartwig reaction, 
the task focuses on predicting reaction yield, specifically wether a reaction is High-yielding(Yes) or Not High-yielding(No),based on the SMILES string representation of each Buchwald-Hartwig reaction. 
The reactants are separated by '.' and product are separated by '>>'. High-yielding reaction means the yield rate of the reaction is above 70. 
In comparison to the traditional Buchwald-Hartwig reaction, this study involved the addition of additives and aimed to evaluate the effects of isoxazole agment additives as Glorius fragment additives on the amination of various aryl and heteroaryl halides. 
You will be provided with several examples molecules. Please answer with only Yes or No.
"""


SUZUKI_PROMPT = """
You are an expert chemist, your task is to predict the yield of reaction using your experienced chemical yield prediction knowledge.
Please strictly follow the format, no other information can be provided. Given the SMILES string of a Suzuki-Miyaura reaction, 
the task focuses on predicting reaction yield, specifically wether a reaction is High-yielding(Yes) or Not High-yielding(No),based on the SMILES string representation of each Suzuki-Miyaura reaction. 
The reactants are separated by '.' and product are separated by '>>'. High-yielding reaction means the yield rate of the reaction is above 70. 
You will be provided with several examples molecules. Please answer with only Yes or No.
"""

BH_URL = "https://raw.githubusercontent.com/ChemFoundationModels/ChemLLMBench/main/data/yield_prediction/BH_dataset%20.npz"
BH_TEST_URL = "https://raw.githubusercontent.com/ChemFoundationModels/ChemLLMBench/main/data/yield_prediction/BH_sample_100_test.npZ"


SUZUKI = "https://raw.githubusercontent.com/ChemFoundationModels/ChemLLMBench/main/data/yield_prediction/Suzuki.npz"
SUZUKI_TEST = "https://raw.githubusercontent.com/ChemFoundationModels/ChemLLMBench/main/data/yield_prediction/SU_sample_100_test.npZ"





def _url_to_df(url):
    response = requests.get(url)
    
    if response.status_code == 200:
        with np.load(io.BytesIO(response.content), allow_pickle=True) as data:

            raw_data = data['data_df'] if "TEST" in url else data['data'] 
            return raw_data if "TEST" in url else raw_data[2]
    else:
        print(f"Failed to download file. Status code: {response.status_code}")
        return None


with np.load("BH_dataset .npz", allow_pickle=True) as data:
    # Extract the object array containing the dataframe
    raw_data = data["data"]
    
    # Use .item() to extract the actual pandas DataFrame from the numpy wrapper
    # print(raw_data.shape)
    print(type(raw_data[2]).__name__) 
    print(raw_data[2].keys())
    print(len(raw_data[2]["rsmi"])) # contains the juice 







# with np.load("SU_sample_100_test.npz", allow_pickle=True) as data:
#     # Extract the object array containing the dataframe
#     print(data)   
#     print(data["data_df"]) 
#     # print(data["data"].shape)
#     # # # Use .item() to extract the actual pandas DataFrame from the numpy wrapper
#     # # print(raw_data.shape)
#     # # print(raw_data[2]) # contains the juice 
#     # print(data["data"][2])

    
