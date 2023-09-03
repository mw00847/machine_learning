#https://github.com/tensorflow/docs/blob/master/site/en/tutorials/structured_data/time_series.ipynb

#import libraries

import os
import datetime

import IPython
import IPython.display
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf


#unzip the dataset. uses tf.keras.utils.get_file

zip_path = tf.keras.utils.get_file(
    origin='https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip',
    fname='jena_climate_2009_2016.csv.zip',
    extract=True)

csv_path, _ = os.path.splitext(zip_path)

#read the csv file using pandas

df = pd.read_csv(csv_path)

#make hourly predictions

#slice [start:stop:step], starting from index 5 take every 6th record.
df = df[5::6]

#set the date time format
date_time=pd.to_datetime(df.pop('Date Time'), format='%d.%m.%Y %H:%M:%S')

#print the data frame
print(df.head())

#print the data fram headers
print(df.columns)

#plot the data, name the columns plot_cols and the features plot_features
plot_cols=['T (degC)', 'p (mbar)', 'rho (g/m**3)', 'wv (m/s)', 'max. wv (m/s)', 'wd (deg)']
plot_features=df[plot_cols]

#index the data frame by date time
plot_features.index=date_time

#plot a set data
plot_features=df[plot_cols][:480]
plot_features.index=date_time[:480]

#view the plot
#plt.figure(figsize=(12, 8))
plot_features.plot(subplots=True)
plt.show()


#set pandas to display all columns
pd.set_option('display.max_columns', None)


print("Statistics of the data")
print(df.describe().transpose())

#the wv (m/s) and max. wv (m/s) are -9999

#replace the -9999 with 0
df['wv (m/s)'].replace(-9999.0, 0.0, inplace=True)
df['max. wv (m/s)'].replace(-9999.0, 0.0, inplace=True)

#reprint df.describe
print("Statistics of the data after replacing -9999 with 0")
print(df.describe().transpose())




#the wd (deg) is in degrees, the data should be continuous
#plot the wd (deg) column
plt.hist2d(df['wd (deg)'], df['wv (m/s)'], bins=(50, 50), vmax=400)
plt.colorbar()
plt.xlabel('Wind Direction [deg]')
plt.ylabel('Wind Velocity [m/s]')
plt.show()

#convert the wind direction and velocity to a wind vector
wv = df.pop('wv (m/s)')
max_wv = df.pop('max. wv (m/s)')

# Convert to radians.
wd_rad = df.pop('wd (deg)')*np.pi / 180

# Calculate the wind x and y components.
df['Wx'] = wv*np.cos(wd_rad)
df['Wy'] = wv*np.sin(wd_rad)

# Calculate the max wind x and y components.
df['max Wx'] = max_wv*np.cos(wd_rad)
df['max Wy'] = max_wv*np.sin(wd_rad)

plt.hist2d(df['Wx'], df['Wy'], bins=(50, 50), vmax=400)
plt.colorbar()
plt.xlabel('Wind X [m/s]')
plt.ylabel('Wind Y [m/s]')
ax = plt.gca()
ax.axis('tight')

plt.show()