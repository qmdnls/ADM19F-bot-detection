import pandas as pd
import tweepy
import ast
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

print("Reading data...")

# Load dataset from tsv
df = pd.read_csv("dataset.csv", encoding="utf8", engine="python")
following = [user_id for sublist in df['following_list'] for user_id in ast.literal_eval(sublist)[0]]
followers = [user_id for sublist in df['followers_list'] for user_id in ast.literal_eval(sublist)[0]]
ids = list(set(following + followers))
ids_length = len(ids)

# Create new list to store retrieved data in
data = []

print("Starting download...")

# Use Twitter API to get usernames, follower and following counts, save in Pandas dataframe
for idx, user_ids in enumerate(grouper(ids, 100)):
    # Group of user ids now looks like: (user_id0, user_id1, ..., user_id99)
   
    # Print progress
    if (idx % 10 == 0):
        print(idx*100, "of", ids_length, "rows processed", "(" + str(round(100*100*idx/ids_length)) + "%)" ,end="\r", flush=True)

    for attempt in range(10):
        try:
            users = api.lookup_users(user_ids=user_ids)
            
        # Handle exception in case the user does not exist etc.
        except tweepy.error.TweepError as e:
            print("Error: " + e, attempt)
            continue
        # Handle exception in case of connection timeouts or resets
        except (Timeout, SSLError, ReadTimeoutError, ConnectionError) as e:
            print("Error:", e)
            print("Reconnecting...", attempt)
            continue
        # If successful, break and do not retry
        else:
            break

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

    if (idx % 250 == 0):
        pd.DataFrame(data).to_csv("user_data_progress.csv", index=None, header=True)

# Make our lists of dicts into a dataframe and merge it with the dataset
users_df = pd.DataFrame(data)
users_df.to_csv("user_data.csv", index=None, header=True)
print(df)
