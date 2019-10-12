import pandas as pd
import tweepy
from itertools import zip_longest

# Helper method that returns a list in chunks
def grouper(iterable, n, fillvalue=None):
    "grouper('ABCDEFG', '3', 'x') --> ABC DEF Gxx"
    args = [iter(iterable)] * n
    return zip_longest(fillvalue=fillvalue, *args)

# Load Twitter API key
# It's assumed our file contains the consumer token on the first line and its secret on the second line
try:
    with open("api.txt", 'r') as f:
        lines = []
        for line in f:
                lines.append(line.strip())
    consumer_token = lines[0]
    consumer_secret = lines[1]
except FileNotFoundError:
     print("'%s' file not found" % filename)

# Set up tweepy
auth = tweepy.AppAuthHandler(consumer_token, consumer_secret)
api = tweepy.API(auth)
api.wait_on_rate_limit = True
api.wait_on_rate_limit_notify = True

# Load dataset from tsv
df = pd.read_csv("data-set-cresci-2018.tsv", sep="\t")

# Create new dataframe to store retrieved data in
users_df = pd.DataFrame()
users_df["user_id"] = ""
users_df["screen_name"] = ""
users_df["followers"] = ""
users_df["following"] = ""

# Use Twitter API to get usernames, follower and following counts, save in Pandas dataframe
for user_ids in grouper(df["user_id"][0:100], 100):
    # Group of user ids now looks like: (user_id0, user_id1, ..., user_id99)
    
    try:
        users = api.lookup_users(user_ids=user_ids)
    
    # Handle exception in case the user does not exist
    except tweepy.error.TweepError:
        print("User_ID not found: '%s'" % id)
    
    # Now iterate over the users and modify dataframe accordingly
    for idx,user in enumerate(users):
        users_df.at[idx, "user_id"] = user.id
        users_df.at[idx, "screen_name"] = user.screen_name
        users_df.at[idx, "followers"] = user.followers_count
        users_df.at[idx, "following"] = user.friends_count

df = pd.merge(df, users_df, on='user_id')
df.to_csv("data.csv", index = None, header=True)
print(df)
