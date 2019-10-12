import pandas as pd
import tweepy
from itertools import zip_longest

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

# Load dataset from tsv
df = pd.read_csv("data-set-cresci-2018.tsv", sep="\t")

# Create new empty column to store username in
df["username"] = ""
df["followers"] = ""
df["following"] = ""

# Use Twitter API to get usernames, follower and following counts, save in Pandas dataframe
for idx, id in enumerate(df["user_id"][0:5]):
    try:
        user = api.get_user(user_id=id)
        print(user.screen_name)
        print("Followers: " + str(user.followers_count))
        print("Following: " + str(user.friends_count))
    except tweepy.error.TweepError:
        print("User_ID not found: '%s'" % id)

print(api.rate_limit_status())
