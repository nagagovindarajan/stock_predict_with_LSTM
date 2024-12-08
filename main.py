# -*- coding: UTF-8 -*-
"""
Main program: includes configuration, data reading, logging, plotting, model training, and prediction
"""

import pandas as pd
import numpy as np
import os
import sys
import time
import logging
from logging.handlers import RotatingFileHandler
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

frame = "pytorch"  # Options: "keras", "pytorch", "tensorflow"
if frame == "pytorch":
    from model.model_pytorch import train, predict, forecast_future
elif frame == "keras":
    from model.model_keras import train, predict
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'
elif frame == "tensorflow":
    from model.model_tensorflow import train, predict
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'    # There are many tf warnings under tf and keras, but they do not affect training
else:
    raise Exception("Wrong frame selection")

class Config:
    # Data parameters
    # Use multiple columns: For example, if your CSV has columns:
    # Date(0), Open(1), High(2), Low(3), Close(4), Volume(5)
    # and you want to predict the Close price
    feature_columns = [1, 2, 3, 4, 5]  # Open, High, Low, Close, Volume
    label_columns = [3]  # Predicting Close price (adjust index accordingly)
    label_in_feature_index = (lambda x,y: [x.index(i) for i in y])(feature_columns, label_columns)

    predict_day = 1

    # Network parameters
    input_size = len(feature_columns)
    output_size = len(label_columns)
    hidden_size = 256          # Increased hidden size
    lstm_layers = 3            # More layers
    dropout_rate = 0.3         # Slightly higher dropout
    time_step = 168            # Longer historical window

    # Training parameters
    do_train = True
    do_predict = True
    add_train = False
    shuffle_train_data = True
    use_cuda = False

    train_data_rate = 0.9
    valid_data_rate = 0.2  # Increased validation portion to get a better estimate of generalization

    batch_size = 64
    learning_rate = 0.001
    epoch = 150              # More epochs for deeper model
    patience = 20
    random_seed = 42

    do_continue_train = False
    continue_flag = ""

    debug_mode = False
    debug_num = 500

    used_frame = "pytorch"
    model_postfix = {"pytorch": ".pth", "keras": ".h5", "tensorflow": ".ckpt"}
    model_name = "improved_model_" + continue_flag + used_frame + model_postfix[used_frame]

    train_data_path = "./data/hourly_prices.csv"
    model_save_path = "./checkpoint/" + used_frame + "/"
    figure_save_path = "./figure/"
    log_save_path = "./log/"
    do_log_print_to_screen = True
    do_log_save_to_file = True
    do_figure_save = False
    do_train_visualized = False

    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)
    if not os.path.exists(figure_save_path):
        os.mkdir(figure_save_path)
    if do_train and (do_log_save_to_file or do_train_visualized):
        cur_time = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
        log_save_path = log_save_path + cur_time + '_' + used_frame + "/"
        os.makedirs(log_save_path)

class Data:
    def __init__(self, config):
        self.config = config
        self.data, self.data_column_name = self.read_data()

        self.data_num = self.data.shape[0]
        self.train_num = int(self.data_num * self.config.train_data_rate)

        self.mean = np.mean(self.data, axis=0)              # The mean and variance of the data
        self.std = np.std(self.data, axis=0)
        self.norm_data = (self.data - self.mean)/self.std   # Normalization, de-dimensionalization

        self.start_num_in_test = 0      # The data from the first few days in the test set will be deleted because it is not enough for one time_step

    def read_data(self):                # Read initial data
        if self.config.debug_mode:
            init_data = pd.read_csv(self.config.train_data_path, nrows=self.config.debug_num,
                                    usecols=self.config.feature_columns)
        else:
            init_data = pd.read_csv(self.config.train_data_path, usecols=self.config.feature_columns)
        return init_data.values, init_data.columns.tolist()     # .columns.tolist() is to get column names

    def get_train_and_valid_data(self):
        feature_data = self.norm_data[:self.train_num]
        label_data = self.norm_data[self.config.predict_day : self.config.predict_day + self.train_num,
                                    self.config.label_in_feature_index]    # Use data delayed by a few days as label

        if not self.config.do_continue_train:
            # In non-continuous training mode, every time_step rows of data will be used as a sample, two samples are staggered by one row, for example: rows 1-20, rows 2-21, ...
            train_x = [feature_data[i:i+self.config.time_step] for i in range(self.train_num-self.config.time_step)]
            train_y = [label_data[i:i+self.config.time_step] for i in range(self.train_num-self.config.time_step)]
        else:
            # In continuous training mode, every time_step rows of data will be used as a sample, two samples are staggered by time_step rows,
            # for example: rows 1-20, rows 21-40, ... to the end of the data, then rows 2-21, rows 22-41, ... to the end of the data, ...
            # This way, the final_state of the previous sample can be used as the init_state of the next sample, and cannot be shuffled
            # Currently, this project can only be used in pytorch's RNN series models
            train_x = [feature_data[start_index + i*self.config.time_step : start_index + (i+1)*self.config.time_step]
                       for start_index in range(self.config.time_step)
                       for i in range((self.train_num - start_index) // self.config.time_step)]
            train_y = [label_data[start_index + i*self.config.time_step : start_index + (i+1)*self.config.time_step]
                       for start_index in range(self.config.time_step)
                       for i in range((self.train_num - start_index) // self.config.time_step)]

        train_x, train_y = np.array(train_x), np.array(train_y)

        train_x, valid_x, train_y, valid_y = train_test_split(train_x, train_y, test_size=self.config.valid_data_rate,
                                                              random_state=self.config.random_seed,
                                                              shuffle=self.config.shuffle_train_data)   # Split training and validation sets, and shuffle
        return train_x, valid_x, train_y, valid_y

    def get_test_data(self, return_label_data=False):
        feature_data = self.norm_data[self.train_num:]
        sample_interval = min(feature_data.shape[0], self.config.time_step)     # Prevent time_step from being larger than the test set
        self.start_num_in_test = feature_data.shape[0] % sample_interval  # These days of data are not enough for one sample_interval
        time_step_size = feature_data.shape[0] // sample_interval

        # In the test data, every time_step rows of data will be used as a sample, two samples are staggered by time_step rows
        # For example: rows 1-20, rows 21-40, ... to the end of the data.
        test_x = [feature_data[self.start_num_in_test+i*sample_interval : self.start_num_in_test+(i+1)*sample_interval]
                   for i in range(time_step_size)]
        if return_label_data:       # In actual applications, the test set does not have label data
            label_data = self.norm_data[self.train_num + self.start_num_in_test:, self.config.label_in_feature_index]
            return np.array(test_x), label_data
        return np.array(test_x)

def load_logger(config):
    logger = logging.getLogger()
    logger.setLevel(level=logging.DEBUG)

    # StreamHandler
    if config.do_log_print_to_screen:
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setLevel(level=logging.INFO)
        formatter = logging.Formatter(datefmt='%Y/%m/%d %H:%M:%S',
                                      fmt='[ %(asctime)s ] %(message)s')
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

    # FileHandler
    if config.do_log_save_to_file:
        file_handler = RotatingFileHandler(config.log_save_path + "out.log", maxBytes=1024000, backupCount=5)
        file_handler.setLevel(level=logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        # Record config information to log file
        config_dict = {}
        for key in dir(config):
            if not key.startswith("_"):
                config_dict[key] = getattr(config, key)
        config_str = str(config_dict)
        config_list = config_str[1:-1].split(", '")
        config_save_str = "\nConfig:\n" + "\n'".join(config_list)
        logger.info(config_save_str)

    return logger

def draw(config: Config, origin_data: Data, logger, predict_norm_data: np.ndarray):
    label_data = origin_data.data[origin_data.train_num + origin_data.start_num_in_test : ,
                                            config.label_in_feature_index]
    predict_data = predict_norm_data * origin_data.std[config.label_in_feature_index] + \
                   origin_data.mean[config.label_in_feature_index]   # Restore data through saved mean and variance
    assert label_data.shape[0]==predict_data.shape[0], "The element number in origin and predicted data is different"

    label_name = [origin_data.data_column_name[i] for i in config.label_in_feature_index]
    label_column_num = len(config.label_columns)

    # label and predict are staggered by config.predict_day days
    # Below are two ways to calculate the norm loss, the results are the same, you can simply verify it
    # label_norm_data = origin_data.norm_data[origin_data.train_num + origin_data.start_num_in_test:,
    #              config.label_in_feature_index]
    # loss_norm = np.mean((label_norm_data[config.predict_day:] - predict_norm_data[:-config.predict_day]) ** 2, axis=0)
    # logger.info("The mean squared error of stock {} is ".format(label_name) + str(loss_norm))

    loss = np.mean((label_data[config.predict_day:] - predict_data[:-config.predict_day] ) ** 2, axis=0)
    loss_norm = loss/(origin_data.std[config.label_in_feature_index] ** 2)
    logger.info("The mean squared error of stock {} is ".format(label_name) + str(loss_norm))

    label_X = range(origin_data.data_num - origin_data.train_num - origin_data.start_num_in_test)
    predict_X = [ x + config.predict_day for x in label_X]

    if not sys.platform.startswith('linux'):    # No desktop Linux cannot output, if it is desktop Linux, such as Ubuntu, you can remove this line
        for i in range(label_column_num):
            plt.figure(i+1)                     # Plot predicted data
            plt.plot(label_X, label_data[:, i], label='label')
            plt.plot(predict_X, predict_data[:, i], label='predict')
            plt.title("Predict stock {} price with {}".format(label_name[i], config.used_frame))
            logger.info("The predicted stock {} for the next {} day(s) is: ".format(label_name[i], config.predict_day) +
                  str(np.squeeze(predict_data[-config.predict_day:, i])))
            if config.do_figure_save:
                plt.savefig(config.figure_save_path+"{}predict_{}_with_{}.png".format(config.continue_flag, label_name[i], config.used_frame))

        plt.show()

def main(config):
    logger = load_logger(config)
    try:
        np.random.seed(config.random_seed)  # Set random seed to ensure reproducibility
        data_gainer = Data(config)

        if config.do_train:
            train_X, valid_X, train_Y, valid_Y = data_gainer.get_train_and_valid_data()
            train(config, logger, [train_X, train_Y, valid_X, valid_Y])

        if config.do_predict:
            test_X, test_Y = data_gainer.get_test_data(return_label_data=True)
            pred_result = predict(config, test_X)       # This outputs un-restored normalized prediction data
            draw(config, data_gainer, logger, pred_result)

              # Predict future n steps
            future_steps = 10
            # Use the last sequence of the last segment of the test set as the initial input
            start_sequence = test_X[-1:,:,:]  # shape [1, seq_len, input_size]
            future_predictions = forecast_future(config, start_sequence, n_future_steps=future_steps)
            print("Future predictions shape:", future_predictions.shape)
            print("Future predictions:", future_predictions)
            # Step 1: Inverse transform the future predictions to original scale
            # Remember: norm_data = (data - mean) / std
            # So, data = norm_data * std + mean
            future_predictions_original = future_predictions * data_gainer.std[config.label_in_feature_index] + \
                                        data_gainer.mean[config.label_in_feature_index]

            # Also get the original test_Y for comparison (these are the known labels you have)
            test_Y_original = test_Y * data_gainer.std[config.label_in_feature_index] + data_gainer.mean[config.label_in_feature_index]

            # Let's assume we are plotting the first label column predicted.
            # If you predicted multiple columns, pick the column index you want to visualize.
            feature_idx_to_plot = 0  # For example, if you have multiple label columns, adjust accordingly.

            # Length of historical test data
            history_len = test_Y_original.shape[0]

            # We predicted future_steps steps ahead
            future_steps = future_predictions_original.shape[0]

            # Create a timeline: 
            # Historical part: from 0 to history_len-1
            # Future part: from history_len to history_len + future_steps - 1
            time_history = np.arange(history_len)
            time_future = np.arange(history_len, history_len + future_steps)

            plt.figure(figsize=(10, 5))
            plt.plot(time_history, test_Y_original[:, feature_idx_to_plot], label='Historical (True)', color='blue')
            plt.plot(time_future, future_predictions_original[:, feature_idx_to_plot], label='Forecast (Predicted)', color='red', linestyle='--')

            plt.title("Historical and Forecasted Future Values")
            plt.xlabel("Time Steps")
            plt.ylabel("Value")
            plt.legend()
            plt.grid(True)
            plt.show()

    except Exception:
        logger.error("Run Error", exc_info=True)


if __name__=="__main__":
    import argparse
    # argparse is convenient for inputting parameters from the command line, you can add more as needed
    parser = argparse.ArgumentParser()
    # parser.add_argument("-t", "--do_train", default=False, type=bool, help="whether to train")
    # parser.add_argument("-p", "--do_predict", default=True, type=bool, help="whether to train")
    # parser.add_argument("-b", "--batch_size", default=64, type=int, help="batch size")
    # parser.add_argument("-e", "--epoch", default=20, type=int, help="epochs num")
    args = parser.parse_args()

    con = Config()
    for key in dir(args):               # dir(args) function gets all attributes of args
        if not key.startswith("_"):     # Remove built-in attributes of args, such as __name__, etc.
            setattr(con, key, getattr(args, key))   # Assign attribute values to Config

    main(con)
