import pandas as pd
import numpy as np
import twint
import tweepy

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

# Twitter API example
user = api.get_user('twitter')
print(user.screen_name)
print(user.followers_count)

# Load dataset from tsv
df = pd.read_csv("data-set-cresci-2018.tsv", sep="\t")

# Create new empty columns to store followers and following in
df["followers"] = ""
df["following"] = ""

# Configure twint
c = twint.Config()
c.Followers = True
c.Pandas = True
c.Hide_output = True

# Skip the first one for testing because it has too many followers/following...
for id in df["user_id"][1:]:
    c.User_id = id
    
    twint.run.Followers(c)

    # This will be empty if the user does not exist, in this case skip to the next user_id
    if (twint.storage.panda.Follow_df.empty):
        continue

    followers = twint.storage.panda.Follow_df["followers"].tolist()[0]

    twint.run.Following(c)
    following = twint.storage.panda.Follow_df["following"].tolist()[0]

    print("User_id: " + str(id))
    print(followers)
    print(following)
