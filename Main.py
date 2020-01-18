import pandas as pd
from datetime import  datetime
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix

# Can't put a try catch block in lambda so here is a named function which does the trick
def save_division(a, b, c):
    try:
        return a / b
    except Exception as e:
        print(e)
        return c


# Is time during rush hour?
# TODO check for public holidays
# Easier to read rush hour checker than the lambda expression. DT has to be a datetime object
def is_in_rush_hour(dt):
    if not isinstance(dt, datetime):
        raise Exception("is_in_rush_hour excepts a datetime object as input")

    if dt.strftime("%A") == 'Sunday' or dt.strftime("%A") == 'Saturday':
        # saturday and sunday no rush hour
        return False
    else:
        if datetime.strptime("07:00:00", "%H:%M:%S").time() <= dt.time() <= datetime.strptime("09:00:00","%H:%M:%S").time():
            return True
        elif datetime.strptime("16:00:00", "%H:%M:%S").time() <= dt.time() <= datetime.strptime("18:00:00","%H:%M:%S").time():
            return True
        else:
            return False


# Load data
# TODO JOIN with CA_DATA
cols = ["trip_start_timestamp", "trip_miles", "trip_seconds", "fare", "payment_type", "pickup_community_area", "dropoff_community_area", "pickup_centroid_latitude", "pickup_centroid_longitude", "dropoff_centroid_latitude", "dropoff_centroid_longitude"]
df = pd.read_csv("train.csv", usecols=cols, delimiter=',')
df = df.sample(n=150000)  # Sample for better performance

# Remove stupid character from datetime string
df['trip_start_timestamp'] = df['trip_start_timestamp'].apply(lambda x: (x.replace('T', ' ')).replace('Z', ''))

df['in_rush_hour'] = df['trip_start_timestamp'].apply(lambda d: is_in_rush_hour(datetime.strptime(d, "%Y-%m-%d %H:%M:%S")))

# Now remove trip start timestamp
df = df.drop(['trip_start_timestamp'], axis=1)

# ____ CA_data ____ => as data types may be changed we have to take care of "trip_start_timestamp" before
cols_ca = ["community_area_number", "community_area_name", "percent_of_housing_crowded", "percent_households_below_poverty", "percent_aged_over_15_unemployed", "percent_aged_over_24_without_high_school_diploma", "percent_aged_under_18_or_over_64", "per_capita_income", "hardship_index", "life_expectancy_1990", "life_expectancy_2000", "life_expectancy_2010", "predominant_non_english_language_percent", "avg_elec_usage_kwh", "avg_gas_usage_therms"]
df_ca = pd.read_csv("CA_data.csv", usecols=cols_ca, delimiter=',')

# Join df with CA_data on pickup_community_area and community_area_number
df = pd.merge(left=df, right=df_ca, left_on=["pickup_community_area"], right_on=["community_area_number"], how='outer', sort=False)

# df = df_joined  # this line can easily be made into a comment to test only the original data set
# ____ CA_data ____


# Compute average speed of trip and remove all entries above 120mph
df['avg_speed'] = df.apply(lambda row: save_division(row['trip_miles'], (row['trip_seconds']+1), 0)*3600, axis=1)
df = df.drop(df[df['avg_speed']> 120].index)

# Compute fare per mile - maybe in rush hour 1 mile is more expensive, drop every fare >1000$
# TODO normalize with trip_seconds
df['fare_per_mile'] = df.apply(lambda row : save_division(row['fare'], row['trip_miles']+1, 0), axis=1)
# df = df.drop(df[df['fare_per_mile'] > 1000.0].index)




# All NA community areas set to 0
df['pickup_community_area'] = df['pickup_community_area'].fillna(0)
df['dropoff_community_area'] = df['dropoff_community_area'].fillna(0)
df['pickup_centroid_latitude'] = df['pickup_centroid_latitude'].fillna(0)
df['pickup_centroid_longitude'] = df['pickup_centroid_longitude'].fillna(0)
df['dropoff_centroid_latitude'] = df['dropoff_centroid_latitude'].fillna(0)
df['dropoff_centroid_longitude'] = df['dropoff_centroid_longitude'].fillna(0)

# Move target value in_rush_hour to the back just for consistency
df = df[['trip_seconds', 'trip_miles', 'pickup_community_area', 'dropoff_community_area', 'fare', 'payment_type', "pickup_centroid_latitude", "pickup_centroid_longitude", "dropoff_centroid_latitude", "dropoff_centroid_longitude", 'avg_speed', 'fare_per_mile', "community_area_number", "percent_of_housing_crowded", "percent_households_below_poverty", "percent_aged_over_15_unemployed", "percent_aged_over_24_without_high_school_diploma", "percent_aged_under_18_or_over_64", "per_capita_income", "hardship_index", "life_expectancy_1990", "life_expectancy_2000", "life_expectancy_2010", "avg_elec_usage_kwh", "avg_gas_usage_therms", "in_rush_hour"]]


print("Size before dropping all NA values: ", df.shape)
df = df.dropna()
print("Size after dropping all NA values: ", df.shape)

# manually set type necessary for some values. Discretization: convert float to int for some values
df = df.astype({'in_rush_hour': 'bool', 'payment_type': 'str', 'avg_gas_usage_therms': 'int32', "avg_elec_usage_kwh": 'int32', 'trip_seconds': 'int32'})

# print("CORRELATION TO IN_RUSH_HOUR BOOLEAN")
corr = df.corr(method='pearson')['in_rush_hour'].apply(lambda x : abs(x))
print(corr[corr>0.03])
# TODO remove all attributes with |correlation| < 0.03 programmatically
df = df[['trip_seconds', 'trip_miles', 'fare', 'pickup_centroid_latitude', 'pickup_centroid_longitude', 'avg_speed', 'percent_households_below_poverty', 'percent_aged_over_15_unemployed', 'percent_aged_under_18_or_over_64', 'life_expectancy_1990', 'life_expectancy_2000', 'life_expectancy_2010', 'fare_per_mile', 'payment_type', 'in_rush_hour']]
# split dataset
x1 = df.loc[df['in_rush_hour'] == True]
x2 = df.loc[df['in_rush_hour'] == False]

# Avg speed is actually significant
def plt_avg_speed():
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.set_title("Average Speed")
    ax.hist(x1['avg_speed'], bins=range(50), alpha=0.25, color='b')
    ax.axvline(x1['avg_speed'].mean(), color='k', linestyle='dashed', linewidth=1)
    ax.axvline(x2['avg_speed'].mean(), color='r', linestyle='dashed', linewidth=1)
    ax.hist(x2['avg_speed'], bins=range(50), alpha=0.25, color='r')
    plt.show()

# Fare doesn't differ much
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
    plt.hist([x1['pickup_community_area'], x2['pickup_community_area']], log=False, bins=77, label=['Rush Hour', 'No Rush Hour'])
    plt.legend(loc='upper right')
    plt.title("Pickup Community Area")
    plt.xlim(60,77)
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

# Problem: the dots may cover other coloured dots; hard to identify with own eyes which colour predominates an area
def plt_pickup_coordinates():
    fig = plt.figure()
    ax = fig.add_subplot();
    ax.set_title("Geographic Coordinates")
    ax.scatter(x2['pickup_centroid_longitude'], x2['pickup_centroid_latitude'], c='b', s=3, alpha=0.5, label="No Rush Hour")
    ax.scatter(x1['pickup_centroid_longitude'], x1['pickup_centroid_latitude'], c='r', s=3, alpha=0.5, label="Rush Hour")
    plt.ylim(41.80,42.05)
    plt.xlim(-87.70, -87.55)
    plt.legend(loc='upper right')
    plt.show()

def plt_pickup_coordinates2():
    fig = plt.figure()
    ax = fig.add_subplot();
    ax.set_title("Geographic Coordinates")
    ax.scatter(x1['pickup_centroid_longitude'], x1['pickup_centroid_latitude'], c='r', s=3, alpha=0.5, label="Rush Hour")
    ax.scatter(x2['pickup_centroid_longitude'], x2['pickup_centroid_latitude'], c='b', s=3, alpha=0.5, label="No Rush Hour")

    plt.ylim(41.80,42.05)
    plt.xlim(-87.70, -87.55)
    plt.legend(loc='upper right')
    plt.show()

# plt_avg_speed()
# plt_payment_type()
# plt_fare()
# plt_payment_type()
# plt_pickup_community()
# plt_dropoff_community()
# plt_avg_speed()
# plt_fare_per_mile()
# plt_pickup_coordinates()
# plt_pickup_coordinates2()
#
# normalize the data. some classifier work better on normalized data
sc = StandardScaler()
df[['trip_seconds', 'trip_miles', 'fare', 'avg_speed', 'fare_per_mile']] = sc.fit_transform(df[['trip_seconds', 'trip_miles', 'fare', 'avg_speed', 'fare_per_mile']])

# encode the payment type as integer
le = LabelEncoder()
df['payment_type'] = le.fit_transform(df['payment_type'])


# split the data into train and test date
# TODO k-fold cross validation
training_set, validation_set = train_test_split(df, test_size = 0.2, random_state = 21)
X_train = training_set.iloc[:,0:-1].values
Y_train = training_set.iloc[:,-1].values
X_val = validation_set.iloc[:,0:-1].values
y_val = validation_set.iloc[:,-1].values

# define a measure for accuracy
def accuracy(confusion_matrix):
   diagonal_sum = confusion_matrix.trace()
   sum_of_all_elements = confusion_matrix.sum()
   return diagonal_sum / sum_of_all_elements

# create the neural network classifier
classifier = MLPClassifier(hidden_layer_sizes=(11,100,100,2), max_iter=300, activation = 'relu', solver='adam', random_state=1)
classifier.fit(X_train, Y_train)
y_pred = classifier.predict(X_val)
cm = confusion_matrix(y_pred, y_val)
print("Accuracy of MLPClassifier : ", accuracy(cm))