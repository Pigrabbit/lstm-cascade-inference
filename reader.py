import scipy.io as sio
import numpy as np
import pandas as pd

NUM_MODEL = 104
NUM_WELL = 20

def read_dataset(data_dir):
    mat_content = sio.loadmat(data_dir)
    data = mat_content['en_d'][0, 0]
    well_dic = create_well_dic(data)
    return well_dic

def create_well_dic(data):
    well_dic = {}
    for well_index in range(NUM_WELL): # well, Producer P1-P20
        # 'model_num' => dataframe
        model_dic = {}
        well_key = 'P' + str(well_index+1)
        for model_index in range(NUM_MODEL): # model, model 1-104
            well_data = np.array([
                data['WOPR'][0,0][well_key][:,model_index],
                data['WBHP'][0,0][well_key][:,model_index],
                data['WWCT'][0,0][well_key][:,model_index],
                data['WWPR'][0,0][well_key][:,model_index]
            ])
            # col1: WOPR, col2: WBHP, col3: WWCT, col4: WWPR
            # row1: day1, ... row 498: day3648
            well_data = well_data.T
            df = pd.DataFrame(
                data=well_data,
                columns=['WOPR', 'WBHP', 'WWCT', 'WWPR']
            )
            model_dic[str(model_index+1)] = df
        
        well_dic[str(well_index+1)] = model_dic

    return well_dic