import pandas as pd
from datetime import  datetime
import matplotlib.pyplot as plt
import sklearn
import sklearn.model_selection

# cant put a try catch block in lambda so here is a named function which does the trick
def save_division(a, b, c):
    try:
        return a / b
    except Exception as e:
        print(e)
        return c
# Daten laden
cols = ["trip_start_timestamp", "trip_miles", "trip_seconds", "fare", "payment_type", "pickup_community_area", "dropoff_community_area"]
df = pd.read_csv("train.csv", usecols=cols, delimiter=',')
df = df.sample(n=5000) # Sample for better performance

# reformat the time stamp
remove_alpha = lambda x :( x.replace('T', ' ')).replace('Z','')
df['trip_start_timestamp'] = df['trip_start_timestamp'].apply(remove_alpha)
df['trip_start_time'] = df['trip_start_timestamp'].apply(lambda x : x[11:])
df['trip_start_date'] = df['trip_start_timestamp'].apply(lambda x : x[:10])

# is time during rush hour
# TODO check for public holidays
is_rush_hour = lambda x : ((datetime.strptime("07:00:00", "%H:%M:%S") < datetime.strptime(x[11:],"%H:%M:%S") < datetime.strptime("09:00:00","%H:%M:%S")) or (datetime.strptime("16:00:00", "%H:%M:%S") < datetime.strptime(x[11:],"%H:%M:%S") < datetime.strptime("18:00:00","%H:%M:%S"))) and not (datetime.strptime(x[:10], "%Y-%m-%d").strftime("%A") is 'Sunday') and not (datetime.strptime(x[:10], "%Y-%m-%d").strftime("%A") is 'Saturday')
df['in_rush_hour'] = df['trip_start_timestamp'].apply(is_rush_hour)

# compute average speed of trip and remove all entries above 120mph
df['avg_speed'] = df.apply(lambda row: save_division(row['trip_miles'], (row['trip_seconds']+1), 0)*3600, axis=1)
df = df.drop(df[df['avg_speed']> 120].index)

# compute fare per mile - maybe in rush hour 1 mile is more expensive, drop every fare >1000$
# TODO normalize with trip_seconds
df['fare_per_mile'] = df.apply(lambda row : save_division(row['fare'], row['trip_miles']+1, 0), axis=1)
# df = df.drop(df[df['fare_per_mile'] > 1000.0].index)

# split dataset
x1 = df.loc[df['in_rush_hour'] == True]
x2 = df.loc[df['in_rush_hour'] == False]

# avg speed ist schon aussagekräftig
def plt_avg_speed():
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.set_title("Average Speed")
    ax.hist(x1['avg_speed'], bins=range(50), alpha=0.25, color='b')
    ax.axvline(x1['avg_speed'].mean(), color='k', linestyle='dashed', linewidth=1)
    ax.axvline(x2['avg_speed'].mean(), color='r', linestyle='dashed', linewidth=1)
    ax.hist(x2['avg_speed'], bins=range(50), alpha=0.25, color='r')
    plt.show()

# fare ist nicht sehr unterschiedlich
def plt_fare():
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.set_title("Fares")
    ax.hist(x1['fare'], bins=range(50), alpha=0.25, color='b', label="Rush Hour")
    ax.axvline(x1['fare'].mean(), color='k', linestyle='dashed', linewidth=1, label="Rush Hour")
    ax.axvline(x2['fare'].mean(), color='r', linestyle='dashed', linewidth=1, label="No Rush Hour")
    ax.hist(x2['fare'], bins=range(50), alpha=0.25, color='r', label="No Rush Hour")
    plt.legend(loc='upper right')
    plt.show()

def plt_payment_type():
    plt.hist([x1['payment_type'], x2['payment_type']], log=True, label=['Rush Hour', 'No Rush Hour'])
    plt.legend(loc='upper right')
    plt.title("Payment Type")
    plt.show()

def plt_pickup_community():
    plt.hist([x1['pickup_community_area'], x2['pickup_community_area']], log=True, label=['Rush Hour', 'No Rush Hour'])
    plt.legend(loc='upper right')
    plt.title("Pickup Community Area")
    plt.show()

def plt_dropoff_community():
    plt.hist([x1['dropoff_community_area'], x2['dropoff_community_area']], log=True, label=['Rush Hour', 'No Rush Hour'])
    plt.legend(loc='upper right')
    plt.title("Dropoff Community Area")
    plt.show()

def plt_fare_per_mile():
    plt.hist([x1['fare_per_mile'], x2['fare_per_mile']], log=True, label=['Rush Hour', 'No Rush Hour'], bins=range(0,60))
    plt.legend(loc='upper right')
    plt.title("Fare Per Mile")
    plt.show()

plt_avg_speed()
plt_payment_type()
plt_fare()
plt_payment_type()
plt_pickup_community()
plt_dropoff_community()
plt_avg_speed()
plt_fare_per_mile()

print("CORRELATION TO IN_RUSH_HOUR BOOLEAN")

print(df.corr(method='kendall')['in_rush_hour'])
