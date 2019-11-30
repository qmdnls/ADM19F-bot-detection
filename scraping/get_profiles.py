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
df = pd.read_csv("data/data-set-cresci-2018.tsv", sep="\t")

# Create new list to store retrieved data in
data = []

# Use Twitter API to get usernames, follower and following counts, save in Pandas dataframe
for idx, user_ids in enumerate(grouper(df["user_id"], 100)):
    # Group of user ids now looks like: (user_id0, user_id1, ..., user_id99)
   
    # Print progress
    if (idx % 10 == 0):
        print(str(round(idx*100*100/len(df["user_id"]))) + "% progress", end="\r", flush=True)

    try:
        users = api.lookup_users(user_ids=user_ids)
   
    # Handle exception in case the user does not exist
    except tweepy.error.TweepError as e:
        print("Error: " + e)

    # Now iterate over the users and store retrieved data in dicts
    for user in users:
        userdata = dict(user_id=user.id,
                screen_name=user.screen_name,
                followers=user.followers_count,
                following=user.friends_count,
                name=user.name,
                location=user.location,
                url=user.url,
                description=user.description,
                protected=user.protected,
                listed_count=user.listed_count,
                favourites_count=user.favourites_count,
                statuses_count=user.statuses_count,
                created_at=user.created_at,
                default_profile=user.default_profile,
                default_profile_image=user.default_profile_image)
        data.append(userdata)

# Make our lists of dicts into a dataframe and merge it with the dataset
users_df = pd.DataFrame(data)
users_df.to_csv("data/data_scraped.csv", index=None, header=True)
df = pd.merge(df, users_df, on='user_id')
df.to_csv("data/data.csv", index=None, header=True)
print(df)
