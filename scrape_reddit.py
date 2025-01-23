from time import sleep
import pandas as pd
import praw
from pathlib import Path

from tqdm import tqdm

ID = ID
SECRET = SECRET
USER_AGENT = USER_AGENT

reddit = praw.Reddit(
    client_id=ID,
    client_secret=SECRET,
    user_agent=USER_AGENT,
)

# each sub needs to have more than 1000 members and posts within the last year
lds_subs = ["lds", "latterdaysaints"]
other_lds_subs = ["ldssexuality", "unexpectedLDS", "Clean_LDS", "LDS_Dating", "NuancedLDS"]
mormon_subs = ["exmormon", "mormon", "MormonShrivel"]
# questionable: mormonpolitics? probably not, looks like mostly political speech by mormons
# mormonscholar not very active, same with ldsgamers  
christian_subs = ["Christianity", "Christian", "Christians", "OpenChristian", "Catholicism", "exchristian", "TrueChristian", "Anglicanism", "OrthodoxChristianity", "churchofchrist",]
other_religious_subs = ["atheism", "excatholic", "exchristian", "islam"]

christian_scrape = ["Christianity", "Christian", "Christians", "TrueChristian"]

posts = pd.DataFrame(columns=["Sub name", "type", "title", "id", "date", "text"])
comments = pd.DataFrame(columns=["Sub name", "type", "title", "id", "date", "text"])

# remove the two above, and make sure that both types can have all attributes I'm trying to store
texts = pd.DataFrame(columns=["Sub name", "type", "title", "id", "date", "text"])

for sub_name in christian_scrape:
    # sub_name = "latterdaysaints"
    sub = reddit.subreddit(sub_name)

    for submission in tqdm(sub.new(limit=1000)):
        # want to include the title, id, text, time, and sub name for each post
        # print(submission.title)
        # print(submission.id)
        # print(submission.selftext) # this can be '' if it's just a link or nothing is there
        # print(submission.created_utc)
        # create a pandas instance that stores all posts
        texts.loc[len(texts.index)] = [sub_name, "post", submission.title, submission.id, 
                                            submission.created_utc, submission.selftext]
        sleep(2)
        submission.comments.replace_more(limit=None)
        for comment in submission.comments.list():
            # print("comments")
            # # want to include the text, time, sub name, submission id for each comment
            # print(comment.body)
            # print(comment.created_utc)
            # print(comment.subreddit_id)
            texts.loc[len(texts.index)] = [sub_name, "comment", submission.title, comment.id, 
                                            comment.created_utc, comment.body]

texts.to_csv(Path("final_project/christian_reddit.csv"))
