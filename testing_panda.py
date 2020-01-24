import pandas as pd
from datetime import  datetime
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix

# cant put a try catch block in lambda so here is a named function which does the trick
def save_division(a, b, c):
    try:
        return a / b
    except Exception as e:
        print(e)
        return c
# Load data
# TODO JOIN with CA_DATA
cols = ["trip_start_timestamp", "trip_miles", "trip_seconds", "fare", "payment_type", "pickup_community_area", "dropoff_community_area", "pickup_centroid_latitude", "pickup_centroid_longitude", "dropoff_centroid_latitude", "dropoff_centroid_longitude"]
df = pd.read_csv("train.csv", usecols=cols, delimiter=',')
df = df.sample(n=1000)  # Sample for better performance

# ____ CA_data ____
cols_ca = ["community_area_number", "community_area_name", "percent_of_housing_crowded", "percent_households_below_poverty", "percent_aged_over_15_unemployed", "percent_aged_over_24_without_high_school_diploma", "percent_aged_under_18_or_over_64", "per_capita_income", "hardship_index", "life_expectancy_1990", "life_expectancy_2000", "life_expectancy_2010", "predominant_non_english_language_percent", "avg_elec_usage_kwh", "avg_gas_usage_therms"]
df_ca = pd.read_csv("CA_data.csv", usecols=cols_ca, delimiter=',')
# Join df with CA_data on pickup_community_area and community_area_number
df_joined = pd.merge(left=df, right=df_ca, left_on=["pickup_community_area"], right_on=["community_area_number"], how='outer', sort=False)
print(list(df_joined))
print(df_joined[["community_area_number", "pickup_community_area", "trip_start_timestamp"]])

#
# # remove stupid character from datetime string
# df['trip_start_timestamp'] = df['trip_start_timestamp'].apply(lambda x :( x.replace('T', ' ')).replace('Z',''))
#
# # is time during rush hour
# # TODO check for public holidays
# # easier to read rush hour checker than the lambda expression. DT has to be a datetime object
# def is_in_rush_hour(dt):
#     if not isinstance(dt, datetime):
#         raise Exception("is_in_rush_hour excepts a datetime object as input")
#
#     if dt.strftime("%A") == 'Sunday' or dt.strftime("%A") == 'Saturday':
#         # saturday and sunday no rush hour
#         return False
#     else:
#         if datetime.strptime("07:00:00", "%H:%M:%S").time() <= dt.time() <= datetime.strptime("09:00:00", "%H:%M:%S").time():
#             return True
#         elif datetime.strptime("16:00:00", "%H:%M:%S").time() <= dt.time() <= datetime.strptime("18:00:00", "%H:%M:%S").time():
#             return True
#         else:
#             return False
#
# df['in_rush_hour'] = df['trip_start_timestamp'].apply(lambda d : is_in_rush_hour(datetime.strptime(d, "%Y-%m-%d %H:%M:%S")))
# # now remove trip start timestamp
# df = df.drop(['trip_start_timestamp'], axis=1)
#
# # compute average speed of trip and remove all entries above 120mph
# df['avg_speed'] = df.apply(lambda row: save_division(row['trip_miles'], (row['trip_seconds']+1), 0)*3600, axis=1)
# df = df.drop(df[df['avg_speed']> 120].index)
#
# # compute fare per mile - maybe in rush hour 1 mile is more expensive, drop every fare >1000$
# # TODO normalize with trip_seconds
# df['fare_per_mile'] = df.apply(lambda row : save_division(row['fare'], row['trip_miles']+1, 0), axis=1)
# # df = df.drop(df[df['fare_per_mile'] > 1000.0].index)
#
#
#
#
# # all NA community areas set to 0
# df['pickup_community_area'] = df['pickup_community_area'].fillna(0)
# df['dropoff_community_area'] = df['dropoff_community_area'].fillna(0)
# df['pickup_centroid_latitude'] = df['pickup_centroid_latitude'].fillna(0)
# df['pickup_centroid_longitude'] = df['pickup_centroid_longitude'].fillna(0)
# df['dropoff_centroid_latitude'] = df['dropoff_centroid_latitude'].fillna(0)
# df['dropoff_centroid_longitude'] = df['dropoff_centroid_longitude'].fillna(0)
#
# # move target value in_rush_hour to the back just for consistency
# df = df[['trip_seconds', 'trip_miles', 'pickup_community_area', 'dropoff_community_area', 'fare', 'payment_type', "pickup_centroid_latitude", "pickup_centroid_longitude", "dropoff_centroid_latitude", "dropoff_centroid_longitude", 'avg_speed', 'fare_per_mile', 'in_rush_hour']]
#
#
# print("Size before dropping all NA values: ", df.shape)
# df = df.dropna()
# print("Size after dropping all NA values: ", df.shape)
# # split dataset
# x1 = df.loc[df['in_rush_hour'] == True]
# x2 = df.loc[df['in_rush_hour'] == False]
#
