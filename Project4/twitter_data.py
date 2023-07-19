import pytz
from sklearn.model_selection import train_test_split
import numpy as np
from datetime import datetime
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error as mse
directory = 'E:/Downloads/219Twitter/'
tweettags = ["gohawks", "gopatriots", "nfl", "patriots", "sb49", "superbowl"]
pst_tz = pytz.timezone('America/Los_Angeles')
tweetfeatures = ['Number of tweets', 'Total number of retweets', 'Total number of followers',
                 'Maximum number of followers', 'Hour of the day', 'Total tweets by Author',
                 'Author Passivity', 'Impression Count', 'Ranking Score', 'Mention Count',
                 'Graph Density', 'Avg Graph Degree', 'urls', 'hashtags']

default_feature_count = 5  # number of tweets, retweets, followers, max followers and hour of the day
extra_feature_count = len(tweetfeatures)
start_time = datetime(2015, 2, 1, 8, 0, 0).replace(tzinfo=pst_tz)
end_time = datetime(2015, 2, 1, 20, 0, 0).replace(tzinfo=pst_tz)

# q 4 and 5
tweettags = ["gohawks", "gopatriots", "nfl", "patriots", "sb49", "superbowl"]
window = 1
avg_past_features = False
args = {"extra_features": True, "window": window, "avg_past_features": avg_past_features, "scale_input": False}
utils_obj = tweets()
NUM_FEATURES = 11

for tag in tweettags:
    print("=" * 50)
    print("\nLinear Regression for ", tag)
    print("\nWindow = ", window)

    args["tweettags"] = [tag]
    features = utils_obj.get_features(args)
    features, test = train_test_split(features["features"], test_size=0.2, random_state=42, shuffle = True)

    retweets = features["retweets"].to_list()
    list_time = features["time"].dt.time.to_list()
    list_time_day = features["time"].dt.day.to_list()

    list_of_hour = []
    list_of_day = []
    list = ['followers', 'favorite_count', 'user_tweet_count', 'passivity', 'impression_count',
         'ranking_score', 'mention_count', 'density', 'degree', 'url_count',
         'hashtag_count']
    for i in range(len(list_time)):
        list_of_hour.append(list_time[i].hour)
    for k in range(len(list_time)):
        list_of_day.append(list_time_day[k])

    train_X = np.zeros((NUM_FEATURES + 2,len(list_of_hour)))
    train_X[0, :] = list_of_hour
    train_X[1, :] = list_of_day

    for j in range(NUM_FEATURES):
        train_X[j+2,:] = features[list[j]].to_list()

    #data preparation
    trainY = np.array(retweets).T
    trainX = np.array(train_X).T

    sc = StandardScaler()
    scaler = sc.fit(trainX)
    trainX = scaler.transform(trainX)
    krr = Ridge(alpha=1.0)
    krr.fit(trainX, trainY)
    #train dataset
    # mlp_reg = MLPRegressor(hidden_layer_sizes=(20,20),
    #                        max_iter=100, activation='relu',
    #                        solver='adam')
    #
    # mlp_reg.fit(trainX, trainY)

    #test dataset
    retweets = test["retweets"].to_list()
    list_time = test["time"].dt.time.to_list()
    list_time_day = test["time"].dt.day.to_list()
    list_of_hour = []
    list_of_day = []
    list = ['followers', 'favorite_count', 'user_tweet_count', 'passivity', 'impression_count',
            'ranking_score', 'mention_count', 'density', 'degree', 'url_count',
            'hashtag_count']
    for i in range(len(list_time)):
        list_of_hour.append(list_time[i].hour)
    for k in range(len(list_time)):
        list_of_day.append(list_time_day[k])


    testX = np.zeros((NUM_FEATURES + 2,len(list_of_hour)))
    testX[0, :] = list_of_hour
    testX[1, :] = list_of_day
    for j in range(NUM_FEATURES):
        testX[j+2,:] = test[list[j]].to_list()

    testY = np.array(retweets).T
    testX = np.array(testX).T
    testX = scaler.transform(testX)
    y_pred = krr.predict(testX)
    print("Root MSE for tag: ",tag)
    print(np.sqrt(mse(y_pred,testY)))

