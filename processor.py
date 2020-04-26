import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

NUM_FEATURES = 4
NUM_FEATURES_USED = 2
INPUT_SEQUENCE = 5
FUTURE_TARGET = 0
STEP = 1

def remove_zero_wopr(well):
    for model_index in well:
        df = well[model_index]
        well[model_index] = df[df.WOPR != 0]

def serialize_well_df(well):
    serialized_well = np.empty((0, 4))
    min_list = []
    scale_list = []
    end_indice = []
    for model in well:
        model_value = well[str(model)].values
        # Todo: scale the model
        scaled_model, scaler_min, scaler_scale = scale_model(model_value)
        serialized_well = np.concatenate((serialized_well, scaled_model))
        min_list.append(scaler_min)
        scale_list.append(scaler_scale)
        num_timesteps = model_value.shape[0]
        if len(end_indice) == 0:
            end_indice.append(num_timesteps)
        else:
            end_indice.append(num_timesteps + end_indice[-1])

    return serialized_well, end_indice, min_list, scale_list

def scale_model(model):
    scaler = MinMaxScaler()
    transformed = scaler.fit_transform(model, model)
    return transformed, scaler.min_[0], scaler.scale_[0]

def multivariate_data(dataset, target, start_index, end_index, history_size,
                      target_size, step, single_step=False):
    data = []
    labels = []

    start_index = start_index + history_size
    if end_index is None:
        end_index = len(dataset) - target_size

    for i in range(start_index, end_index):
        indices = range(i-history_size, i, step)
        data.append(dataset[indices])

        if single_step:
            labels.append(target[i+target_size])
        else:
            labels.append(target[i:i+target_size])

    return np.array(data), np.array(labels)

def get_dataset(serialized_well, model_list, end_indice):
    dataset_x = np.empty((0, INPUT_SEQUENCE, NUM_FEATURES_USED))
    dataset_y = np.empty((0))

    for i in range(len(model_list)):
        start_index = end_indice[model_list[i]-1] if model_list[i] > 0 else 0 
        end_index = end_indice[model_list[i]]
        range_multi_x, range_multi_y = multivariate_data(
            serialized_well[start_index:end_index][:, :NUM_FEATURES_USED],
            serialized_well[start_index:end_index][:, 0],
            start_index = 0,
            end_index = end_index - start_index,
            history_size = INPUT_SEQUENCE,
            target_size = FUTURE_TARGET,
            step = STEP,
            single_step = True
        )
        dataset_x = np.concatenate((dataset_x, range_multi_x))
        dataset_y = np.concatenate((dataset_y, range_multi_y))

    return dataset_x, dataset_y