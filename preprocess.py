import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset in chunks, returns list of chunks that we can iterate over
def load_dataset(filename):
    return pd.read_csv(filename, encoding='utf8', engine='python', chunksize=None)

print("Loading dataset...")
df = load_dataset("data/users.csv")

print("Preprocessing data...")

# Use only labelled accounts
df = df[pd.notnull(df['label'])]

# Create column for the account age in days
df['created_at'] = pd.to_datetime(df['created_at'])
df['account_age'] = time.time() / 60 / 60 /24 # Current time in days
df['account_age'] = df['account_age'] - df['created_at'].astype(int) / 10**9 / 60 / 60 / 24 # Convert nanoseconds to seconds, then to days. Subtract from current time to get age.

# Encode default_profile and default_profile_image numerically
df['default_profile'] = df['default_profile'].astype(int)
df['default_profile_image'] = df['default_profile_image'].astype(int)

# Convert labels to binary values where human = 0, bot = 1
df['label'] = df['label'].map({'human': 0, 'bot': 1})

# Drop columns that are no longer needed
df = df.drop(['created_at', 'description', 'followers_list', 'following_list', 'location', 'name', 'protected', 'screen_name', 'url', 'user_id'], axis=1)

print(df)

# Show correlation matrix
corr = df.corr()
corr.style.background_gradient(cmap='coolwarm').set_precision(2)
ax = sns.heatmap(corr,xticklabels=corr.columns.values,yticklabels=corr.columns.values)
plt.show()
