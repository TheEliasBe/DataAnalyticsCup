import pandas as pd
from datetime import  datetime
import matplotlib.pyplot as plt
import sklearn
import sklearn.model_selection


cols = ["trip_start_timestamp", "trip_miles", "trip_seconds", "fare", "payment_type", "pickup_community_area", "dropoff_community_area"]
dtypes = {"trip_start_timestamp" : object, "trip_miles": int}
df = pd.read_csv("../../../../Programming/DataAnalyticsCup/train.csv", usecols=cols)
df = df.sample(n=50000)

remove_alpha = lambda x :( x.replace('T', ' ')).replace('Z','')
df['trip_start_timestamp'] = df['trip_start_timestamp'].apply(remove_alpha)
df['trip_start_time'] = df['trip_start_timestamp'].apply(lambda x : x[11:])
df['trip_start_date'] = df['trip_start_timestamp'].apply(lambda x : x[:10])
is_rush_hour = lambda x : ((datetime.strptime("07:00:00", "%H:%M:%S") < datetime.strptime(x[11:],"%H:%M:%S") < datetime.strptime("09:00:00","%H:%M:%S")) or (datetime.strptime("16:00:00", "%H:%M:%S") < datetime.strptime(x[11:],"%H:%M:%S") < datetime.strptime("18:00:00","%H:%M:%S"))) and not (datetime.strptime(x[:10], "%Y-%m-%d").strftime("%A") is 'Sunday') and not (datetime.strptime(x[:10], "%Y-%m-%d").strftime("%A") is 'Saturday')
df['in_rush_hour'] = df['trip_start_timestamp'].apply(is_rush_hour)
df['avg_speed'] = df.apply(lambda row: row['trip_miles'] / (row['trip_seconds']+1)*3600, axis=1)
sanitized_df = df.drop(df[df['avg_speed']> 120].index)

x1 = sanitized_df.loc[sanitized_df['in_rush_hour'] == True]
x2 = sanitized_df.loc[sanitized_df['in_rush_hour'] == False]

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

plt_avg_speed()
plt_payment_type()
plt_fare()
plt_payment_type()
plt_pickup_community()
plt_dropoff_community()

print("CORRELATION TO IN RUSH HOUR BOOLEAN")
print(sanitized_df.corr(method='kendall')['in_rush_hour'])
