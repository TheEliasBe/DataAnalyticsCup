import pandas as pd
from datetime import  datetime
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, KFold
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
import numpy as np
import Plot
import geopy.distance

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
cols = ["trip_start_timestamp", "trip_miles", "trip_seconds", "fare", "payment_type", "pickup_community_area", "dropoff_community_area", "pickup_centroid_latitude", "pickup_centroid_longitude", "dropoff_centroid_latitude", "dropoff_centroid_longitude"]
df = pd.read_csv("train.csv", usecols=cols, delimiter=',')
df = df.sample(n=150000)

# Remove stupid character from datetime string
df['trip_start_timestamp'] = df['trip_start_timestamp'].apply(lambda x: (x.replace('T', ' ')).replace('Z', ''))

df['in_rush_hour'] = df['trip_start_timestamp'].apply(lambda d: is_in_rush_hour(datetime.strptime(d, "%Y-%m-%d %H:%M:%S")))

# Now remove trip start timestamp
df = df.drop(['trip_start_timestamp'], axis=1)

# ____ CA_data ____ => as data types may be changed we have to take care of "trip_start_timestamp" before
cols_ca = ["community_area_number", "community_area_name", "percent_of_housing_crowded", "percent_households_below_poverty", "percent_aged_over_15_unemployed", "percent_aged_over_24_without_high_school_diploma", "percent_aged_under_18_or_over_64", "per_capita_income", "hardship_index", "life_expectancy_1990", "life_expectancy_2000", "life_expectancy_2010", "predominant_non_english_language_percent", "avg_elec_usage_kwh", "avg_gas_usage_therms"]
df_ca = pd.read_csv("CA_data.csv", usecols=cols_ca, delimiter=',')

# Join df with CA_data on pickup_community_area and community_area_number
df_merged = pd.merge(left=df, right=df_ca, left_on=["pickup_community_area"], right_on=["community_area_number"], how='outer', sort=False)

# df = df_joined  # this line can easily be made into a comment to test only the original data set
# ____ CA_data ____

# for all instances with trip_miles = 0
# All NA community areas set to 0
df['pickup_community_area'] = df['pickup_community_area'].fillna(0)
df['dropoff_community_area'] = df['dropoff_community_area'].fillna(0)
df['pickup_centroid_latitude'] = df['pickup_centroid_latitude'].fillna(0)
df['pickup_centroid_longitude'] = df['pickup_centroid_longitude'].fillna(0)
df['dropoff_centroid_latitude'] = df['dropoff_centroid_latitude'].fillna(0)
df['dropoff_centroid_longitude'] = df['dropoff_centroid_longitude'].fillna(0)

total_direct = 0
total_driven = 0

def heuristic_number(row):
    global total_direct, total_driven
    if row['trip_miles'] > 0 and row['pickup_centroid_latitude'] > 0 and abs(row['dropoff_centroid_latitude']) > 0:
        total_direct += geopy.distance.geodesic((row["pickup_centroid_latitude"], row["pickup_centroid_longitude"]), (row["dropoff_centroid_latitude"], row['dropoff_centroid_longitude'])).miles
        total_driven += row['trip_miles']
        return row['trip_miles']
    else:
        return row['trip_miles']


def geo_heuristic(row, factor):
    if row['trip_miles'] == 0:
        heuristic_distance = geopy.distance.geodesic((row["pickup_centroid_latitude"], row["pickup_centroid_longitude"]), (row["dropoff_centroid_latitude"], row['dropoff_centroid_longitude'])).miles * factor
        return heuristic_distance
    else:
        return row['trip_miles']


# compute heurcistic factor
df['trip_miles']= df.apply(lambda row: heuristic_number(row), axis=1)
direct_to_driven_factor = total_driven/total_direct
# apply factor to all trip_miles = 0
print(df['trip_miles'])
df['trip_miles']= df.apply(lambda row: geo_heuristic(row, direct_to_driven_factor), axis=1)
print(df[df['trip_miles']>1000])

# Compute average speed of trip and remove all entries above 120mph
df['avg_speed'] = df.apply(lambda row: save_division(row['trip_miles'], (row['trip_seconds']+1), 0)*3600, axis=1)
df = df.drop(df[df['avg_speed']> 120].index)
print(df['avg_speed'])

# Compute fare per mile - maybe in rush hour 1 mile is more expensive, drop every fare >1000$
# TODO normalize with trip_seconds
df['fare_per_mile'] = df.apply(lambda row : save_division(row['fare'], row['trip_miles']+1, 0), axis=1)
# df = df.drop(df[df['fare_per_mile'] > 1000.0].index)

# Move target value in_rush_hour to the back just for consistency
# df = df[['trip_seconds', 'trip_miles', 'pickup_community_area', 'dropoff_community_area', 'fare', 'payment_type', "pickup_centroid_latitude", "pickup_centroid_longitude", "dropoff_centroid_latitude", "dropoff_centroid_longitude", 'avg_speed', 'fare_per_mile', "community_area_number", "percent_of_housing_crowded", "percent_households_below_poverty", "percent_aged_over_15_unemployed", "percent_aged_over_24_without_high_school_diploma", "percent_aged_under_18_or_over_64", "per_capita_income", "hardship_index", "life_expectancy_1990", "life_expectancy_2000", "life_expectancy_2010", "avg_elec_usage_kwh", "avg_gas_usage_therms", "in_rush_hour"]]
df = df[['trip_seconds', 'trip_miles', 'pickup_community_area', 'dropoff_community_area', 'fare', 'payment_type', "pickup_centroid_latitude", "pickup_centroid_longitude", "dropoff_centroid_latitude", "dropoff_centroid_longitude", 'avg_speed', 'fare_per_mile', "in_rush_hour"]]

print("Size before dropping all NA values: ", df.shape)
df = df.dropna()
print("Size after dropping all NA values: ", df.shape)

# manually set type necessary for some values. Discretization: convert float to int for some values
# df = df.astype({'in_rush_hour': 'bool', 'payment_type': 'str', 'avg_gas_usage_therms': 'int32', "avg_elec_usage_kwh": 'int32', 'trip_seconds': 'int32'})
df = df.astype({'in_rush_hour': 'bool', 'payment_type': 'str'})

# print("CORRELATION TO IN_RUSH_HOUR BOOLEAN")
corr = df.corr(method='pearson')['in_rush_hour']

# df = df[['trip_seconds', 'trip_miles', 'fare', 'pickup_centroid_latitude', 'pickup_centroid_longitude', 'avg_speed', 'percent_households_below_poverty', 'percent_aged_over_15_unemployed', 'percent_aged_under_18_or_over_64', 'life_expectancy_1990', 'life_expectancy_2000', 'life_expectancy_2010', 'fare_per_mile', 'payment_type', 'in_rush_hour']]
# df = df[['trip_seconds', 'trip_miles', 'fare', 'pickup_centroid_latitude', 'pickup_centroid_longitude', 'avg_speed', 'fare_per_mile', 'payment_type', 'in_rush_hour']]

for name, val in corr.items():
    if abs(val) < 0.03 and name != 'trip_seconds' and name != 'trip_miles' and name != 'fare' and name != 'avg_speed' \
            and name != 'fare_per_mile':
        df.drop(name, axis=1, inplace=True)


# normalize the data. some classifier work better on normalized data
sc = StandardScaler()
df[['trip_seconds', 'trip_miles', 'fare', 'avg_speed', 'fare_per_mile']] = sc.fit_transform(df[['trip_seconds', 'trip_miles', 'fare', 'avg_speed', 'fare_per_mile']])

# encode the payment type as integer
le = LabelEncoder()
df['payment_type'] = le.fit_transform(df['payment_type'])

# split the data into train and test date
# TODO k-fold cross validation
# kf = KFold(n_splits=5)
# KFold(n_splits=2, random_state=None, shuffle=False)
# X = df.drop(['in_rush_hour'], axis=1)
# y = df['in_rush_hour']
#
# for train_index, test_index in kf.split(X):
#     print("TRAIN:", train_index, "TEST:", test_index)
#     X_train, X_test = X[train_index], X[test_index]
#     y_train, y_test = y[train_index], y[test_index]

# load test.csv
# df_test = pd.read_csv('test.csv', usecols=cols)
#
# # compute average speed for test data
# df_test['avg_speed'] = df_test.apply(lambda row: save_division(row['trip_miles'], (row['trip_seconds']+1), 0)*3600, axis=1)
# df = df.drop(df[df['avg_speed']> 120].index)
#
# # compute fare per mile for test data
# df_test['fare_per_mile'] = df_test.apply(lambda row : save_division(row['fare'], row['trip_miles']+1, 0), axis=1)
# df_test = df_test[['trip_seconds', 'trip_miles', 'fare', 'pickup_centroid_latitude', 'pickup_centroid_longitude', 'avg_speed', 'fare_per_mile', 'payment_type']]
#
# # encode labels
# df_test['payment_type'] = le.fit_transform(df_test['payment_type'])
# df_test[['trip_seconds', 'trip_miles', 'fare', 'avg_speed', 'fare_per_mile']] = sc.fit_transform(df_test[['trip_seconds', 'trip_miles', 'fare', 'avg_speed', 'fare_per_mile']])
# # scale values
# print("Attributes used for testing : ", df_test.columns)
# X_test = df_test.to_numpy()

training_set, validation_set = train_test_split(df, test_size = 0.2, random_state = 2020)
X_train = training_set.iloc[:,0:-1].values
Y_train = training_set.iloc[:,-1].values
X_val = validation_set.iloc[:,0:-1].values
y_val = validation_set.iloc[:,-1].values


# define a measure for accuracy
def accuracy(confusion_matrix):
   diagonal_sum = confusion_matrix.trace()
   sum_of_all_elements = confusion_matrix.sum()
   return diagonal_sum / sum_of_all_elements


# create the neural network classifier - returns the numpy array of predicted values
def mlp(X_train, Y_train, X_val, y_val, training):
    classifier = MLPClassifier(hidden_layer_sizes=(8, 16, 8, 1), max_iter=600, activation='relu', solver='adam', random_state=1)
    classifier.fit(X_train, Y_train)
    y_pred = classifier.predict(X_val)
    if training:
        cm = confusion_matrix(y_pred, y_val)
        print("Accuracy of MLPClassifier : ", accuracy(cm))
        evaluation(y_pred, y_val)
        res = np.column_stack((X_val, y_val, y_pred))
        print(res.shape)
        res_df = pd.DataFrame(data=res)
        print(res_df)
        res_df = res_df.rename({0: 'trip_seconds', 1: 'trip_miles', 2: 'dropoff_community_area', 3: 'fare', 4: 'payment_type', 5: 'avg_speed', 6: 'fare_per_mile', 7: 'in_rush_hour', 8: 'prediction'}, axis=1)
        res_df[['trip_seconds', 'trip_miles', 'fare', 'avg_speed', 'fare_per_mile']] = sc.inverse_transform(res_df[['trip_seconds', 'trip_miles', 'fare', 'avg_speed', 'fare_per_mile']], copy=True)
        res_df = res_df.astype({'in_rush_hour': bool, 'prediction': bool})
        res_df_filtered = res_df[(res_df['in_rush_hour'] == True) & (res_df['prediction'] == False)]
        res_df_filtered.to_csv(path_or_buf="fn.csv" )
        return y_pred
    else:
        ids = np.arange(200001, int(200001+X_test.shape[0]))
        ids.astype(int)
        return np.stack((ids, y_pred))


# create adaboost classifier
def ada(X_train, Y_train, X_val, y_val):
    adaClassifier = AdaBoostClassifier(n_estimators=1000, learning_rate=1.0)
    adaClassifier.fit(X_train, Y_train)
    y_ada = adaClassifier.predict(X_val)
    # print(y_ada)
    print("Accuracy of AdaBoost : ", adaClassifier.score(X_val, y_val))
    evaluation(y_ada, y_val)
    return y_ada


def gradient(X_train, Y_train, X_val, y_val):
    clf = GradientBoostingClassifier(n_estimators=1000, learning_rate=1.0, max_depth=1, random_state=2020)
    clf.fit(X_train, Y_train)
    gradient_boost_predict = clf.predict(X_val)
    # print(gradient_boost_predict)
    cm = confusion_matrix(gradient_boost_predict, y_val)
    print("Accuracy of Gradient Boost Classifier : ", accuracy(cm))
    evaluation(gradient_boost_predict, y_val)
    return gradient_boost_predict


def blockchain_cloud_ai_predict_algo_top_secret(X_train, Y_train, X_val, y_val):
    y_pred_constant = np.full(y_val.shape, False, dtype=bool)
    cm = confusion_matrix(y_pred_constant, y_val)
    print("Accuracy of Gradient Boost Classifier : ", accuracy(cm))
    evaluation(y_pred_constant, y_val)
    return y_pred_constant


def evaluation(pred, truth):
    # Sensitivity = True Positive Rate
    pred = pred.astype(int)
    truth = truth.astype(int)

    tp = 0
    tn = 0
    fn = 0
    fp = 0
    for i in range(len(pred)):
        if pred[i] == truth[i] and pred[i] == 1:
            tp += 1
        elif pred[i] == truth[i] and pred[i] == 0:
            tn += 1
        elif pred[i] != truth[i] and pred[i] == 1:
            fp += 1
        else:
            fn += 1

    sensitivity = tp / (tp + fn)
    specificity = tn / (fp + tn)
    balanced_accuracy = (sensitivity + specificity) / 2

    print("TP: ", tp, " - TN: ", tn, " - FN: ", fn, " - FP: ", fp)
    print("Evaluation: ", balanced_accuracy)

res = mlp(X_train, Y_train, X_val, y_val, training=True)
res = res.transpose()
res = res.astype(int)
print(res.shape)
# np.savetxt('abgabe.csv', res, delimiter=',', newline='\n', fmt='%i', header='id,prediction', comments='')

