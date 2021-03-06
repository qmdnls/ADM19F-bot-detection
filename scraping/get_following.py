import pandas as pd
import tweepy
import time
from itertools import zip_longest
from ssl import SSLError
from requests.exceptions import Timeout, ConnectionError
from urllib3.exceptions import ReadTimeoutError

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
df = pd.read_csv("data/data.csv", encoding='utf8', engine='python')
df_length = len(df["user_id"])
print(df)

# Create dataframe to store retrieved data in
data = pd.DataFrame()
data["user_id"] = ""
data["following_list"] = ""
data["followers_list"] = ""


# Use Twitter API to get IDs of all users followed by and following the users in the dataset
for idx, user_id in enumerate(df["user_id"]):
   
    # Print progress
    print(idx, "of", df_length, "rows processed", end="\r", flush=True)

    # Keep track whether current user is protected
    protected = False

    # We attempt to connect up to 50 times in case of disconnects, resets etc.
    for attempt in range(50):
        # Get following and followers for user
        try:
            #print("try", idx, "attempt", attempt)
            following = api.friends_ids(user_id=user_id)
            followers = api.followers_ids(user_id=user_id)
        # Handle exception in case the user does not exist or user is protected
        except tweepy.error.TweepError as e:
            if (e.reason == "Not authorized."):
                protected = True
                break
            print("Error: ", e)
        # Handle exception in case of connection timeouts or resets
        except (Timeout, SSLError, ReadTimeoutError, ConnectionError) as e:
            print("Error:", e)
            print("Reconnecting...", attempt)
            continue
        # If successful, break and do not retry
        else:
            break

    # If user is protected, skip this user
    if (protected):
        continue

    # Store retrieved data in dataframe, use list of lists because column needs to have the same length
    data.at[idx,"user_id"] = user_id
    data.at[idx,"following_list"] = [following]
    data.at[idx,"followers_list"] = [followers]

    if (idx % 10 == 0):
        data.to_csv("data/data_following_progress.csv", index=None, header=True)

    # Wait one minute to avoid rate limit
    time.sleep(61)

# Merge retrieved data with the dataset
data.to_csv("data/data_following_scraped.csv", index=None, header=True)
df = pd.merge(df, data, on='user_id')
df.to_csv("data/data_following.csv", index=None, header=True)
print(df)
