import matplotlib.pyplot as plt


def plt_avg_speed(x1,x2):
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.set_title("Average Speed")
    ax.hist(x1['avg_speed'], bins=range(50), alpha=0.25, color='b')
    ax.axvline(x1['avg_speed'].mean(), color='k', linestyle='dashed', linewidth=1)
    ax.axvline(x2['avg_speed'].mean(), color='r', linestyle='dashed', linewidth=1)
    ax.hist(x2['avg_speed'], bins=range(50), alpha=0.25, color='r')
    plt.show()

# Fare doesn't differ much
def plt_fare(x1,x2):
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.set_title("Fares")
    ax.hist(x1['fare'], bins=range(50), alpha=0.25, color='b', label="Rush Hour")
    ax.axvline(x1['fare'].mean(), color='k', linestyle='dashed', linewidth=1, label="Rush Hour")
    ax.axvline(x2['fare'].mean(), color='r', linestyle='dashed', linewidth=1, label="No Rush Hour")
    ax.hist(x2['fare'], bins=range(50), alpha=0.25, color='r', label="No Rush Hour")
    plt.legend(loc='upper right')
    plt.show()

def plt_payment_type(x1,x2):
    plt.hist([x1['payment_type'], x2['payment_type']], log=True, label=['Rush Hour', 'No Rush Hour'])
    plt.legend(loc='upper right')
    plt.title("Payment Type")
    plt.show()

def plt_pickup_community(x1,x2):
    plt.hist([x1['pickup_community_area'], x2['pickup_community_area']], log=False, bins=77, label=['Rush Hour', 'No Rush Hour'])
    plt.legend(loc='upper right')
    plt.title("Pickup Community Area")
    plt.xlim(60,77)
    plt.show()

def plt_dropoff_community(x1,x2):
    plt.hist([x1['dropoff_community_area'], x2['dropoff_community_area']], log=True, label=['Rush Hour', 'No Rush Hour'])
    plt.legend(loc='upper right')
    plt.title("Dropoff Community Area")
    plt.show()

def plt_fare_per_mile(x1,x2):
    plt.hist([x1['fare_per_mile'], x2['fare_per_mile']], log=True, label=['Rush Hour', 'No Rush Hour'], bins=range(0,60))
    plt.legend(loc='upper right')
    plt.title("Fare Per Mile")
    plt.show()

# Problem: the dots may cover other coloured dots; hard to identify with own eyes which colour predominates an area
def plt_pickup_coordinates(x1,x2):
    fig = plt.figure()
    ax = fig.add_subplot();
    ax.set_title("Geographic Coordinates")
    ax.scatter(x2['pickup_centroid_longitude'], x2['pickup_centroid_latitude'], c='b', s=3, alpha=0.5, label="No Rush Hour")
    ax.scatter(x1['pickup_centroid_longitude'], x1['pickup_centroid_latitude'], c='r', s=3, alpha=0.5, label="Rush Hour")
    plt.ylim(41.80,42.05)
    plt.xlim(-87.70, -87.55)
    plt.legend(loc='upper right')
    plt.show()

def plt_pickup_coordinates2(x1,x2):
    fig = plt.figure()
    ax = fig.add_subplot();
    ax.set_title("Geographic Coordinates")
    ax.scatter(x1['pickup_centroid_longitude'], x1['pickup_centroid_latitude'], c='r', s=3, alpha=0.5, label="Rush Hour")
    ax.scatter(x2['pickup_centroid_longitude'], x2['pickup_centroid_latitude'], c='b', s=3, alpha=0.5, label="No Rush Hour")

    plt.ylim(41.80,42.05)
    plt.xlim(-87.70, -87.55)
    plt.legend(loc='upper right')
    plt.show()