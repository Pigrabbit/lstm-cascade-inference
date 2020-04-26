import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import reader
import processor
import model
from os.path import dirname, join as pjoin
from datetime import datetime

DATA_PATH = 'data'   
DATA_FILE_NAME = 'Brugge_en_d.mat'

NUM_MODEL = 104
TRAIN_SPLIT = 90
BATCH_SIZE = 32
BUFFER_SIZE = 10000
EPOCHS = 20

TRUE_MODEL = 103
TARGET_WELL = '9'
OBSERVATION_DATE = 150
INFERENCE_GAUSSIAN_STD = 0.01

IMG_PATH = f"img\well_{TARGET_WELL}"
LOG_PATH = f"logs\well_{TARGET_WELL}"
RESULT_PATH = f"result\well_{TARGET_WELL}"

def split_train(test_model):
    model_list = list(range(NUM_MODEL))
    train = []
    test = [test_model]
    model_list.remove(test_model)
    for i in range(TRAIN_SPLIT):
        chosen = random.choice(model_list)
        train.append(chosen)
        model_list.remove(chosen)

    val = model_list
    return train, val, test

def save_train_history(history, title):
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(len(loss))

    fig_extension = "png"
    img_dir = pjoin(IMG_PATH, "history." + fig_extension)
    fig = plt.figure()
    plt.plot(epochs, loss, 'b', label='Training loss')
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.title(title)
    plt.legend()

    fig.savefig(img_dir, format=fig_extension, dpi=300)
    fig.clf()

def save_prediction(predicted, true, title, timestamp):
    fig_extension = "png"
    img_dir = pjoin(IMG_PATH, f"true{TRUE_MODEL}_{timestamp}_prediction." + fig_extension)
    fig = plt.figure()
    plt.plot(predicted, label='prediction')
    plt.plot(true, label='true')
    plt.title(title)
    plt.legend()

    fig.savefig(img_dir, format=fig_extension, dpi=300)
    fig.clf()

def cascade_inference(model, test_x, test_y, obs, gaussian_std):
    y_hat_list = []
    observed = test_y[:obs]
    y_hat_list.extend(observed)

    buffer = test_x[obs:obs+1]
    mu = gaussian_std

    for i in range(obs+1, test_x.shape[0]):
        y_hat = model.predict(buffer)
        predicted_val = y_hat[0, 0, 0]
        y_hat_list.append(predicted_val)
    
        buffer = np.delete(buffer, 0, 1)
        next_wbhp = test_x[i:i+1][0, 4, 1]
        predicted_array = np.array([[predicted_val, next_wbhp]])
    
        buffer = np.vstack((buffer[0], predicted_array))
        buffer = np.reshape(buffer, (1, 5, 2))
    
        mean_wopr = np.mean(buffer[0], axis=0)[0]
        mean_wbhp = np.mean(buffer[0], axis=0)[1]
        wopr_predicted_noise_added = np.random.normal(mean_wopr, mu, 1)
        wbhp_predicted_noise_added = np.random.normal(mean_wbhp, mu, 1)
    
        buffer[0, 4, 0] = wopr_predicted_noise_added
        buffer[0, 4, 1] = wbhp_predicted_noise_added

    return y_hat_list

if __name__ == "__main__" :
    random.seed(7)

    data_dir = pjoin(DATA_PATH, DATA_FILE_NAME)
    well_dic = reader.read_dataset(data_dir)

    # choose a well to learn
    well = well_dic[TARGET_WELL]
    processor.remove_zero_wopr(well)
    serialized_well, end_indice, min_list, scale_list = processor.serialize_well_df(well)
    print(serialized_well.shape)

    train_model_list, val_model_list, test_model_list = split_train(TRUE_MODEL)
    print(f"train model list: {train_model_list}")
    print(f"validation model list: {val_model_list}")
    print(f"test model list: {test_model_list}")

    train_x, train_y = processor.get_dataset(serialized_well, train_model_list, end_indice)
    val_x, val_y = processor.get_dataset(serialized_well, val_model_list, end_indice)
    test_x, test_y = processor.get_dataset(serialized_well, test_model_list, end_indice)

    train_data = tf.data.Dataset.from_tensor_slices((train_x, train_y))
    train_data = train_data.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()
    train_data = train_data.prefetch(1)

    val_data = tf.data.Dataset.from_tensor_slices((val_x, val_y))
    val_data = val_data.batch(BATCH_SIZE).repeat()

    lstm_model = model.get_model(params = {
        "input_shape": train_x.shape[-2:],
        "lstm1_units": 50,
        "lstm2_units": 50,
        "gaussian_std": 0.1,
        "dropout_rate": 0.2,
        "optimizer": "adam",
        "loss": "mean_squared_error"
        })

    es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', patience=6, verbose=1)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    # logdir = pjoin(LOG_PATH, 'scalars', timestamp)
    logdir = pjoin(LOG_PATH, 'scalars')
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)
    checkpoint_path = pjoin(LOG_PATH, "model.ckpt-{epoch:04d}")
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, save_weights_only=True, verbose=1)

    latest = tf.train.latest_checkpoint(LOG_PATH)
    print(f"latest checkpoint: {latest}")
    if latest != None:
        print("Restoring trained weights")
        lstm_model.load_weights(latest)
    else:
        history = lstm_model.fit(
            train_data,
            epochs = EPOCHS,
            steps_per_epoch = train_y.shape[0]//BATCH_SIZE,
            # steps_per_epoch = 160 // BATCH_SIZE,
            validation_data = val_data,
            validation_steps = val_y.shape[0]//BATCH_SIZE,
            use_multiprocessing = True,
            workers = 8,
            callbacks = [es, tensorboard_callback]
            # callbacks = [es, checkpoint_callback, tensorboard_callback]
        )
        print("training has been ended")

        lstm_model.save_weights(checkpoint_path.format(epoch=EPOCHS))
        print("model weights are saved")
        np.save(pjoin(RESULT_PATH, f"ground_truth_{timestamp}.npy"), test_y)
        print("Ground Truth is saved...")
        # save_train_history(history, 'Training and validation loss')

    ## Inference
    y_hat_list = cascade_inference(lstm_model, test_x, test_y, obs=OBSERVATION_DATE, gaussian_std=INFERENCE_GAUSSIAN_STD)
    # print(len(y_hat_list))
    # print(test_y.shape)
    print("Saving inference results...")
    np.save(pjoin(RESULT_PATH, f"inference_result_{timestamp}.npy"), y_hat_list)
    print("Saving prediction graph...")
    save_prediction(y_hat_list, test_y, f"Prediction {TARGET_WELL}, true model: {TRUE_MODEL}", timestamp)
