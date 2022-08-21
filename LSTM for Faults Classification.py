#!/usr/bin/env python
# coding: utf-8

# In[2]:


from math import pi
import os
# import keras
# import keras.backend as K
# from keras.layers import *
# from keras.models import Sequential, Model
import gc
# from numba import jit
from IPython.display import display, clear_output
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import sys

sns.set_style("whitegrid")
import pyarrow.parquet as pq
import pandas as pd
import numpy as np
from tqdm import tqdm


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


# In[20]:


class Tacho:
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
        self.count_pulses = 0
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


# In[6]:

def prepare_data_to_train(simulation_data, number_of_columns_in_one_simulation, fault_codes):
    #This functions gets 2d X data, takes only essensial data and converts it to 3d
    X = []
    Y = []
    for sim_id in range(1, int(simulation_data.shape[1] / number_of_columns_in_one_simulation)):
        column_names = ["Vibration_" + str(sim_id), "TachoDriveSchaftHigh_" + str(sim_id),
                        "TachoDriveSchaftLow_" + str(sim_id), "TachoLoadSchaft_" + str(sim_id),
                        "Gear_" + str(sim_id), "Torque_" + str(sim_id), "Brake_" + str(sim_id)
                        ]

        for column_id, column_name in enumerate(column_names):
            X[sim_id][column_id] = simulation_data[column_name]

def add_angular_velocity_calculated_from_pulses(simulation_data, number_of_columns_in_one_simulation, time_sample,
                                                resolution,
                                                improve_low_velocity_reconstruction=True):
    # Dane s ustawione w kolumnach, dlatego iterujemy po szerokosci, dodatkowo dane
    # dla jednej symulacji zajmuj number_of_columns_in_one_simulation kolumny dlatego dzielimy
    # przez liczbe kolumn i mamy liczbe symulacji
    temp_drive_column = []
    temp_load_column = []
    drive_count = 0
    load_count = 0
    frequency_drive = 0
    frequency_load = 0
    angular_velocity_drive = 0
    angular_velocity_load = 0
    # input: resolution: rozdzielczosc tachometru

    tachos = [
        Tacho('TachoDriveSchaftLow_', resolution, time_sample, improve_low_velocity_reconstruction),
        Tacho('TachoDriveSchaftHigh_', resolution, time_sample, improve_low_velocity_reconstruction),
        Tacho('TachoLoadSchaft_', resolution, time_sample, improve_low_velocity_reconstruction)
    ]

    for sim_id in range(1, int(simulation_data.shape[1] / number_of_columns_in_one_simulation)):
        for tacho in tachos:
            tacho.update_column(sim_id)
            for sample_id in range(0, len(simulation_data)):
                tacho.calculation_for_sample(simulation_data[tacho.column_name][sample_id])
            simulation_data[tacho.column_name] = tacho.column_to_write
    return simulation_data


# In[7]:


def simulation_data_time_to_float(simulation_data, number_of_columns_in_one_simulation):
    # Dane s ustawione w kolumnach, dlatego iterujemy po szerokosci, dodatkowo dane
    # dla jednej symulacji zajmuj number_of_columns_in_one_simulation kolumny dlatego dzielimy
    # przez liczbe kolumn i mamy liczbe symulacji
    temp_float_time = []
    for sim_id in range(1, int(simulation_data.shape[1] / number_of_columns_in_one_simulation)):
        time_column_name = 'Time_' + str(sim_id)
        for time_id in range(0, len(simulation_data)):
            temp_float_time.append(datetime_to_seconds(simulation_data[time_column_name][time_id]))
        simulation_data[time_column_name] = temp_float_time
        temp_float_time = []
    return simulation_data


# In[8]:


def print_angular_velocity_tacho(tacho_name, angular_velocity_name, data, number_of_columns_in_one_simulation):
    for sim_id in range(1, int(data.shape[1] / number_of_columns_in_one_simulation)):
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
        plt.draw()
        # sns.lineplot(x = "Time_100", y = "TachoLoadSchaft_10", data = simulation_data_to_show)
    plt.show()

#         new_data.head()


# In[9]:


# Simulations crucial parameters to proper working of this script
number_of_simulations = 208
number_of_columns_in_one_simulation = 11
time_sample_for_data = 0.001
resolution = 2
big_data_file_name = 'BigData_16_09.parquet'

# In[21]:


train_set = pq.read_pandas(big_data_file_name).to_pandas()
train_set.head()


train = simulation_data_time_to_float(train_set, number_of_columns_in_one_simulation)

train = add_angular_velocity_calculated_from_pulses(train, number_of_columns_in_one_simulation, time_sample_for_data,
                                                    resolution)
train.head()

# In[15]:


print_angular_velocity_tacho("TachoDriveSchaftLow_", "AngularVelocityDriveLow_", train,
                             number_of_columns_in_one_simulation)

# In[ ]:


print_angular_velocity_tacho("TachoLoadSchaft_", "AngularVelocityLoadSchaft_", train,
                             number_of_columns_in_one_simulation)


# usuniecie kolumn z czasem
for sim_id in range(1, int(round((train.shape[1] / 4) + 1, 0))):
    train.drop(['Time_' + str(sim_id)], axis=1, inplace=True)

# In[ ]:

print(train)

# In[ ]:


X_train = train.values

# In[ ]:


type(X_train)

# In[ ]:


np.shape(X_train)

# In[ ]:


# Shape 0 jest nieparzysty, dlatego tutaj usuwamy ostatni rekord, żeby móc potem zrobić reshape
X_train = np.delete(X_train, -1, 0)

# In[ ]:


np.shape(X_train)

# In[ ]:


n_signals = 3
X = X_train.reshape((int(X_train.shape[1] / n_signals), X_train.shape[0], n_signals))
print(X)

# In[ ]:


fault_data = pd.read_csv('FaultCodes_16_09.csv')
y = np.array(fault_data.iloc[:, 0].values)
print(y)

# In[ ]:


with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
    print(fault_data)

# In[ ]:


# First, create a set of indexes of the 5 folds
N_SPLITS = 5
splits = list(StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=2019).split(X, y))
preds_val = []
y_val = []
# Then, iteract with each fold
# If you dont know, enumerate(['a', 'b', 'c']) returns [(0, 'a'), (1, 'b'), (2, 'c')]
for idx, (train_idx, val_idx) in enumerate(splits):
    K.clear_session()  # I dont know what it do, but I imagine that it "clear session" :)
    print("Beginning fold {}".format(idx + 1))
    # use the indexes to extract the folds in the train and validation data
    train_X, train_y, val_X, val_y = X[train_idx], y[train_idx], X[val_idx], y[val_idx]
    # instantiate the model for this fold
    model = model_lstm(train_X.shape)
    # This checkpoint helps to avoid overfitting. It just save the weights of the model if it delivered an
    # validation matthews_correlation greater than the last one.
    ckpt = ModelCheckpoint('weights_{}.h5'.format(idx), save_best_only=True, save_weights_only=True, verbose=1,
                           mode='max')
    # Train, train, train
    model.fit(train_X, train_y, batch_size=16, epochs=50, validation_data=[val_X, val_y], callbacks=[ckpt])
    # loads the best weights saved by the checkpoint
    model.load_weights('weights_{}.h5'.format(idx))
    # Add the predictions of the validation to the list preds_val
    preds_val.append(model.predict(val_X, batch_size=16))
    # and the val true y
    y_val.append(val_y)

# concatenates all and prints the shape    
preds_val = np.concatenate(preds_val)[..., 0]
y_val = np.concatenate(y_val)
preds_val.shape, y_val.shape


# In[ ]:


# model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[tf.keras.metrics.AUC()])


# In[ ]:


class SquareModelLSTM(nn.Module):
    def __init__(self, n_features, hidden_dim, n_outputs):
        super(SquareModelLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_features = n_features
        self.n_outputs = n_outputs
        self.hidden = None
        self.cell = None
        2
        # Simple LSTM
        self.basic_rnn = nn.LSTM(self.n_features,
                                 self.hidden_dim,
                                 batch_first=True)
        1
        # Classifier to produce as many logits as outputs
        self.classifier = nn.Linear(self.hidden_dim, self.n_outputs)

    def forward(self, X):
        # X is batch first (N, L, F)
        # output is (N, L, H)
        # final hidden state is (1, N, H)
        # final cell state is (1, N, H)
        batch_first_output, (self.hidden, self.cell) = \
            self.basic_rnn(X)
        2
        # only last item in sequence (N, 1, H)
        last_output = batch_first_output[:, -1]
        # classifier will output (N, 1, n_outputs)
        out = self.classifier(last_output)
        # final output is (N, n_outputs)
        return out.view(-1, self.n_outputs)


# In[ ]:


torch.manual_seed(21)
model = SquareModelLSTM(n_features=2, hidden_dim=2, n_outputs=1)
loss = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# In[ ]:


sbs_lstm = StepByStep(model, loss, optimizer)
sbs_lstm.set_loaders(train_loader, test_loader)
sbs_lstm.train(100)

# In[ ]:


fig = sbs_lstm.plot_losses()
# data_loader


# In[ ]:
