import pandas as pd
from datetime import  datetime
from sklearn.preprocessing import StandardScaler, LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
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

# TODO ALLE WEITERE PAYMENT TYPES HIER EINFÜGEN
enc = LabelBinarizer()
enc.fit(pd.DataFrame(['Cash', 'Credit Card', 'Dispute', 'Mobile', 'No Charge', 'Pcard', 'Prcard', 'Prepaid', 'Unknown']))


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


total_direct = 0 # saves the total mileage direst line from start coordinates to end coordinates
total_driven = 0 # saves total mileage of actually driven miles

def enhance_data(df, labelled):
    global total_direct, total_driven
    if labelled:
        # Remove stupid character from datetime string
        df['trip_start_timestamp'] = df['trip_start_timestamp'].apply(lambda x: (x.replace('T', ' ')).replace('Z', ''))
        # create training labels
        df['in_rush_hour'] = df['trip_start_timestamp'].apply(lambda d: is_in_rush_hour(datetime.strptime(d, "%Y-%m-%d %H:%M:%S")))
        # Now remove trip start timestamp
        df = df.drop(['trip_start_timestamp'], axis=1)


    # remove na values
    df['pickup_centroid_latitude'] = df['pickup_centroid_latitude'].fillna(0)
    df['pickup_centroid_longitude'] = df['pickup_centroid_longitude'].fillna(0)
    df['dropoff_centroid_latitude'] = df['dropoff_centroid_latitude'].fillna(0)
    df['dropoff_centroid_longitude'] = df['dropoff_centroid_longitude'].fillna(0)



    # here we compute the factor direct to driven miles
    def heuristic_number(row):
        global total_direct, total_driven
        if row['trip_miles'] > 0 and abs(row['pickup_centroid_latitude']) > 0 and abs(row['pickup_centroid_longitude']) > 0 and abs(row['dropoff_centroid_latitude']) > 0 and abs(row['dropoff_centroid_longitude']) > 0:
            total_direct += geopy.distance.geodesic((row["pickup_centroid_latitude"], row["pickup_centroid_longitude"]), (row["dropoff_centroid_latitude"], row['dropoff_centroid_longitude'])).miles
            total_driven += row['trip_miles']
            return row['trip_miles']
        else:
            return row['trip_miles']

    # here we compute the geo-distance for all instances with trip_miles == 0 and apply the factor calculated above
    def geo_heuristic(row, factor):
        global total_changed
        if row['trip_miles'] == 0 and abs(row['pickup_centroid_latitude']) > 0 and abs(row['pickup_centroid_longitude']) > 0 and abs(row['dropoff_centroid_latitude']) > 0 and abs(row['dropoff_centroid_longitude']) > 0:
            heuristic_distance = geopy.distance.geodesic((row["pickup_centroid_latitude"], row["pickup_centroid_longitude"]), (row["dropoff_centroid_latitude"], row['dropoff_centroid_longitude'])).miles * factor
            return heuristic_distance
        else:
            return row['trip_miles']


    df['trip_miles']= df.apply(lambda row: heuristic_number(row), axis=1)
    direct_to_driven_factor = total_driven/total_direct
    df['trip_miles']= df.apply(lambda row: geo_heuristic(row, direct_to_driven_factor), axis=1)

    # Compute average speed of trip and remove all entries above 120mph because those are very likely measurement errors
    df['avg_speed'] = df.apply(lambda row: save_division(row['trip_miles'], (row['trip_seconds']+1), 0)*3600, axis=1)
    df = df.drop(df[df['avg_speed']> 120].index)

    # Compute fare per mile - maybe in rush hour 1 mile is more expensive, drop every fare >1000$
    df['fare_per_mile'] = df.apply(lambda row : save_division(row['fare'], row['trip_miles']+1, 0), axis=1)
    df = df.drop(df[df['fare_per_mile'] > 1000.0].index)

    # Move target value in_rush_hour to the back just for consistency
    if labelled:
        df = df.astype({'in_rush_hour': 'bool', 'payment_type': 'str'})
        df = df[['trip_seconds', 'trip_miles', 'fare', 'avg_speed', 'fare_per_mile', 'payment_type', 'in_rush_hour']]
    else:
        df = df.astype({'payment_type': 'str'})
        df = df[['trip_seconds', 'trip_miles', 'fare', 'avg_speed', 'fare_per_mile', 'payment_type']]

    # TODO only remove NA values in attributes we actually need
    if labelled:
        print("Size before dropping all NA values: ", df.shape)
        df = df.dropna()
        print("Size after dropping all NA values: ", df.shape)
    else:
        pass
        df = df.fillna(0)

    # normalize the data. some classifier work better on normalized data
    sc = StandardScaler()
    df[['trip_seconds', 'trip_miles', 'fare', 'avg_speed', 'fare_per_mile']] = sc.fit_transform(df[['trip_seconds', 'trip_miles', 'fare', 'avg_speed', 'fare_per_mile']])

    # one hot encode payment_type
    df = df.reset_index()
    ohe_df = pd.DataFrame(enc.transform(df['payment_type']))
    df = pd.concat([df, ohe_df], axis=1)
    df = df.drop(['payment_type', 'index'], axis=1)

    return df


# define a measure for accuracy
def accuracy(confusion_matrix):
   diagonal_sum = confusion_matrix.trace()
   sum_of_all_elements = confusion_matrix.sum()
   return diagonal_sum / sum_of_all_elements


# create the neural network classifier - returns the numpy array of predicted values nad id when training
def mlp(X_train, Y_train, X_test, y_val, training):
    classifier = MLPClassifier(hidden_layer_sizes=(14, 200, 80, 1), max_iter=600, activation='relu', solver='adam', random_state=2020)
    classifier.fit(X_train, Y_train)
    y_pred = classifier.predict(X_test)
    if training:
        cm = confusion_matrix(y_pred, y_val)
        print("Accuracy of MLPClassifier : ", accuracy(cm))
        evaluation(y_pred, y_val)
        res = np.column_stack((X_val, y_val, y_pred))
        print(res.shape)
        res_df = pd.DataFrame(data=res)
        return y_pred
    else:
        ids = np.arange(200001, int(200001+X_test.shape[0]))
        ids.astype(int)
        return np.stack((ids, y_pred))


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


cols_train = ["trip_start_timestamp", "trip_miles", "trip_seconds", "fare", "payment_type", "pickup_centroid_latitude", "pickup_centroid_longitude", "dropoff_centroid_latitude", "dropoff_centroid_longitude"]
cols_test = ["trip_miles", "trip_seconds", "fare", "payment_type", "pickup_centroid_latitude", "pickup_centroid_longitude", "dropoff_centroid_latitude", "dropoff_centroid_longitude"]
# load csv files
df_test = pd.read_csv('test.csv', usecols=cols_test)
df_train = pd.read_csv("train.csv", usecols=cols_train, delimiter=',')
df_test = enhance_data(df_test, labelled=False)
df_train = enhance_data(df_train, labelled=True)

print("Attributes used for testing : ", df_test.columns)
print("Learning attributes : ",  df_train.columns)

training_set, validation_set = train_test_split(df_train, test_size = 0.05, random_state = 2020)
X_train = training_set.iloc[:,0:-1].values
Y_train = training_set.iloc[:,-1].values
X_val = validation_set.iloc[:,0:-1].values
y_val = validation_set.iloc[:,-1].values
X_Test = df_test.to_numpy()
output = mlp(X_train, Y_train, X_val, y_val, training=True)
output = np.transpose(output)
output.astype(int)
np.savetxt("predict.csv", output, delimiter=",", header='id, prediction', comments='', fmt='%i')
