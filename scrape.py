import pandas as pd
import numpy as np
import twint

dataset_df = pd.read_csv("data-set-cresci-2018.tsv", sep="\t")

c = twint.Config()
c.Followers = True
c.Pandas = True
c.Hide_output = True


for id in dataset_df["user_id"]:
    c.User_id = id
    
    twint.run.Followers(c)

    if (twint.storage.panda.Follow_df.empty):
        continue

    df = twint.storage.panda.Follow_df
    followers = df["followers"].tolist()[0]

    twint.run.Following(c)
    df = twint.storage.panda.Follow_df
    following = df["following"].tolist()[0]

    print("User_id: " + str(id))
    print(followers)
    print(following)
