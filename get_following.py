import pandas as pd
import tweepy
from itertools import zip_longest

# Load Twitter API key, it's assumed our file contains the consumer token on the first line and its secret on the second line
try:
    with open("api.txt", 'r') as f:
        lines = []
        for line in f:
                lines.append(line.strip())
    consumer_token = lines[0]
    consumer_secret = lines[1]
except FileNotFoundError as e:
    print("File not found: ",  e.filename)
    exit()

# Set up tweepy
auth = tweepy.AppAuthHandler(consumer_token, consumer_secret)
api = tweepy.API(auth)
api.wait_on_rate_limit = True
api.wait_on_rate_limit_notify = True

# Load preprocessed dataset from csv, engine='python' needed because there's a buffer overflow in pandas...
df = pd.read_csv("data.csv", encoding='utf8', engine='python')

print(df)

# Create dataframe to store retrieved data in
data = pd.DataFrame()

# Use Twitter API to get IDs of all users followed by and following the users in the dataset
for idx, user_id in enumerate(df["user_id"]):
   
    # Print progress
    if (idx % 10 == 0):
        print(str(round(idx*100/len(df["user_id"]))) + "% progress", end="\r", flush=True)

    try:
        following = api.friends_ids(user_id=user_id)
        followers = api.followers_ids(user_id=user_id)

    # Handle exception in case the user does not exist or user is protected
    except tweepy.error.TweepError as e:
        if (e.reason == "Not authorized."):
            continue
        print("Error: " + e)

    # Store retrieved data in dataframe, use list of lists because column needs to have the same length
    data["user_id"] = user_id
    data["following_list"] = [following]
    data["followers_list"] = [followers]

# Merge retrieved data with the dataset
data.to_csv("data_following_scraped.csv", index=None, header=True)
df = pd.merge(df, data, on='user_id')
df.to_csv("data_following.csv", index=None, header=True)
print(df)
