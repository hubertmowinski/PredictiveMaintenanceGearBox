#!/usr/bin/env python
# coding: utf-8

# In[2]:


from math import pi
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import sys

sns.set_style("whitegrid")
import pyarrow.parquet as pq
import pandas as pd
import numpy as np
from tsai.all import *
from tqdm import tqdm
import pickle

logging.basicConfig(level=logging.INFO)


# from keras.models import save_model
# from keras.callbacks import * # This object helps the model to train in a smarter way, avoiding overfitting
# from sklearn.model_selection import GridSearchCV, StratifiedKFold # Used to use Kfold to train our model


# In[3]:


# # This is NN LSTM Model creation
# def model_lstm(input_shape):
#     # The shape was explained above, must have this order
#     inp = Input(shape=(input_shape[1], input_shape[2],))
#     # This is the LSTM layer
#     # Bidirecional implies that the 160 chunks are calculated in both ways, 0 to 159 and 159 to zero
#     # although it appear that just 0 to 159 way matter, I have tested with and without, and tha later worked best
#     # 128 and 64 are the number of cells used, too many can overfit and too few can underfit
#     x = Bidirectional(LSTM(1, return_sequences=True))(inp)
#     # The second LSTM can give more fire power to the model, but can overfit it too
# #     x = Bidirectional(LSTM(64, return_sequences=True))(x)
# #     # A intermediate full connected (Dense) can help to deal with nonlinears outputs
# #     x = Dense(64, activation="relu")(x)
# #     # A binnary classification as this must finish with shape (1,)
# #     x = Dense(1, activation="sigmoid")(x)
#     model = Model(inputs=inp, outputs=x)
#     # Pay attention in the addition of matthews_correlation metric in the compilation, it is a success factor key
#     model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[tf.keras.metrics.AUC()])

#     return model


# In[4]:


def datetime_to_seconds(dt):
    return round(dt.microsecond * 1e-6 + dt.second + dt.minute * 60, 3)


def range1(start, end):
    return range(start, end+1)


def save_list_to_file(filename, list):
    with open(filename, "wb") as fp:
        pickle.dump(list, fp)


def list_from_file(filename):
    with open(filename, "rb") as fp:
        return pickle.load(fp)
# In[20]:


class Tacho:
    #Using slots to speed up code
    __slots__ = ('count_pulses', 'frequency', 'angular_velocity', 'name', 'column_name', 'resolution',
                 'time_sample', 'improve_low_velocity_reconstruction', 'column_to_write')

    def __init__(self, name, resolution, time_sample, improve_low_velocity_reconstruction):
        self.count_pulses = 0
        self.frequency = 0
        self.angular_velocity = 0
        self.name = name
        self.column_name = None
        self.resolution = resolution
        self.time_sample = time_sample
        self.improve_low_velocity_reconstruction = improve_low_velocity_reconstruction
        self.column_to_write = []

    def calculation_for_sample(self, impulse):
        if impulse == 1:
            # jezeli mamy 1 to znaczy wykonany zostal jeden pelny obrót i liczymy predkosc
            self.update_angular_velocity()
        elif impulse == 0:
            # jezeli mamy 0 to zliczamy time_sample sekundy bo co tyle są kolejne pomiary
            self.count_pulses_increment()
        self.update_column_to_write()

    def update_column(self, sim_id):
        #Added 100 pulses at begin becouse at begin we dont know how much pulses "0" were before and it is possible that
        # we will have velocity 3140 at begin and we want to avoid that
        self.count_pulses = 100
        self.frequency = 0
        self.angular_velocity = 0
        self.column_to_write = []
        self.column_name = self.name + str(sim_id)

    def update_angular_velocity(self):
        self.count_pulses += 1
        self.frequency = (1 / self.resolution) / (self.count_pulses * self.time_sample)
        self.angular_velocity = self.frequency * 2 * pi
        self.count_pulses = 1

    def count_pulses_increment(self):
        self.count_pulses += 1
        if self.improve_low_velocity_reconstruction:
            frequency_no_impulse = (1 / self.resolution) / (self.count_pulses * self.time_sample)
            angular_velocity_no_impulse = frequency_no_impulse * 2 * pi
            if angular_velocity_no_impulse < self.angular_velocity:
                self.angular_velocity = angular_velocity_no_impulse

    def update_column_to_write(self):
        self.column_to_write.append(self.angular_velocity)


def add_angular_velocity_calculated_from_pulses(simulation_data, number_of_columns_in_one_simulation, time_sample,
                                                resolution,
                                                improve_low_velocity_reconstruction=True):
    # Dane s ustawione w kolumnach, dlatego iterujemy po szerokosci, dodatkowo dane
    # dla jednej symulacji zajmuj number_of_columns_in_one_simulation kolumny dlatego dzielimy
    # przez liczbe kolumn i mamy liczbe symulacji
    # input: resolution: rozdzielczosc tachometru

    tachos = [
        Tacho('TachoDriveSchaftLow_', resolution, time_sample, improve_low_velocity_reconstruction),
        Tacho('TachoDriveSchaftHigh_', resolution, time_sample, improve_low_velocity_reconstruction),
        Tacho('TachoLoadSchaft_', resolution, time_sample, improve_low_velocity_reconstruction)
    ]

    for sim_id in range1(1, int(simulation_data.shape[1] / number_of_columns_in_one_simulation)):
        logger.debug(f"Converting tacho columns from impulses to angular velocity: sim id: {sim_id}")
        for tacho in tachos:
            tacho.update_column(sim_id)
            for sample_id in range(0, len(simulation_data)):
                logger.debug(f"Converting tacho columns from impulses to angular velocity: sample id: {sample_id}")
                tacho.calculation_for_sample(simulation_data[tacho.column_name][sample_id])

            simulation_data[tacho.column_name] = tacho.column_to_write
    return simulation_data


# In[7]:


def simulation_data_time_to_float(simulation_data, number_of_columns_in_one_simulation):
    # Dane s ustawione w kolumnach, dlatego iterujemy po szerokosci, dodatkowo dane
    # dla jednej symulacji zajmuj number_of_columns_in_one_simulation kolumny dlatego dzielimy
    # przez liczbe kolumn i mamy liczbe symulacji
    temp_float_time = []
    for sim_id in range1(1, int(simulation_data.shape[1] / number_of_columns_in_one_simulation)):
        time_column_name = 'Time_' + str(sim_id)
        for time_id in range(0, len(simulation_data)):
            temp_float_time.append(datetime_to_seconds(simulation_data[time_column_name][time_id]))
        simulation_data[time_column_name] = temp_float_time
        temp_float_time = []
    return simulation_data


# In[8]:


def print_angular_velocity_tacho(tacho_name, angular_velocity_name, data, number_of_columns_in_one_simulation):
    for sim_id in range1(1, int(data.shape[1] / number_of_columns_in_one_simulation)):
        tacho_column_name = tacho_name + str(sim_id)
        angular_velocity_column_name = angular_velocity_name + str(sim_id)
        time_name = "Time_" + str(sim_id)
        data_dict = {}
        data_dict[time_name] = data[time_name]
        data_dict[tacho_column_name] = data[tacho_column_name]
        data_dict[angular_velocity_column_name] = data[angular_velocity_column_name]
        simulation_data_to_show = pd.DataFrame(data_dict)
        #         print(simulation_data_to_show.head())
        new_data = simulation_data_to_show.set_index(time_name)

        sns.lineplot(data=new_data)
        plt.plot()
        # sns.lineplot(x = "Time_100", y = "TachoLoadSchaft_10", data = simulation_data_to_show)
    plt.show()

#         new_data.head()


def prepare_data_to_train(simulation_data, number_of_columns_in_one_simulation, fault_codes):
    #This functions gets 2d X data, takes only essensial data and converts it to 3d
    X = []
    Y = fault_codes.iloc[:, -1].values.tolist()
    for sim_id in range1(1, int(simulation_data.shape[1] / number_of_columns_in_one_simulation)):
        column_names = ["Vibration_" + str(sim_id), "TachoDriveSchaftHigh_" + str(sim_id),
                        "TachoDriveSchaftLow_" + str(sim_id), "TachoLoadSchaft_" + str(sim_id),
                        "Gear_" + str(sim_id), "Torque_" + str(sim_id), "Brake_" + str(sim_id)
                        ]
        X.append([])
        for column_id, column_name in enumerate(column_names):
            X[sim_id - 1].append([])
            X[sim_id - 1][column_id].extend(simulation_data[column_name].tolist())

    return X, Y

# In[9]:


# Simulations crucial parameters to proper working of this script
logger = logging.getLogger(__name__)
number_of_simulations = 208
number_of_columns_in_one_simulation = 11
time_sample_for_data = 0.001
resolution = 2
big_data_file_name = 'BigData_16_09.parquet'
fault_data_file_name = 'FaultCodes_16_09.csv'
file_for_train_data = 'train_data'
file_for_train_data_x = file_for_train_data + '_x'
file_for_train_data_y = file_for_train_data + '_y'

# In[21]:

logger.info("Reading data:" + big_data_file_name)
train_set = pq.read_pandas(big_data_file_name).to_pandas()
train_set.head()

logger.info("Converting time columns from data time to float")
train = simulation_data_time_to_float(train_set, number_of_columns_in_one_simulation)
logger.info("Converting tacho columns from impulses to angular velocity")
train = add_angular_velocity_calculated_from_pulses(train, number_of_columns_in_one_simulation, time_sample_for_data,
                                                    resolution)
# train.head()

# In[15]:

logger.info("Printing")
# print_angular_velocity_tacho("TachoDriveSchaftLow_", "AngularVelocityDriveLow_", train,
#                              number_of_columns_in_one_simulation)
#
# # In[ ]:
#
#
# print_angular_velocity_tacho("TachoLoadSchaft_", "AngularVelocityLoadSchaft_", train,
#                              number_of_columns_in_one_simulation)

logger.info("Reading fault data:" + big_data_file_name)
fault_data = pd.read_csv(fault_data_file_name)

# In[ ]:


with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
    print(fault_data)

logger.info("Reshape and prepare data to train")
X, Y = prepare_data_to_train(train, number_of_columns_in_one_simulation, fault_data)

logger.info(f"Save lists with train data to files: {file_for_train_data_x}, {file_for_train_data_y}")
save_list_to_file(file_for_train_data_x, X)
save_list_to_file(file_for_train_data_y, Y)

logger.info("Split data to train and valid")
splits = get_splits(Y, valid_size=.2, stratify=True, random_state=23, shuffle=True)

tfms  = [None, TSClassification()] # TSClassification == Categorize
batch_tfms = TSStandardize()
dls = get_ts_dls(X, Y, splits=splits, tfms=tfms, batch_tfms=batch_tfms, bs=[64, 128])
dls.show_batch(sharey=True)
learn = ts_learner(dls, metrics=accuracy)
learn.lr_find()
learn = ts_learner(dls, metrics=accuracy, cbs=ShowGraph())
learn.fit_one_cycle(10, lr_max=1e-3)

PATH = Path('./Multiclass.pkl')
PATH.parent.mkdir(parents=True, exist_ok=True)
learn.export(PATH)

interp = ClassificationInterpretation.from_learner(learn)
interp.plot_confusion_matrix()