import pandas as pd
from pathlib import Path
from tqdm import tqdm

df = pd.read_csv(Path("final_project/christian_reddit.csv"))

for i, row in tqdm(df.iterrows()):
    try:
        with open(Path(f'final_project/christian_corpus/christian_reddit/christian_reddit_{row["id"]}.txt'), 'w', encoding='utf-8') as f:    
            f.write(row["text"])
    except:
        with open(Path('csv2txt_errors.txt'), 'a') as f:
            print(row["id"], file=f)

# new_df = df["text"]
# print(new_df)
# new_df.to_csv(Path("final_project/reddit_text.csv"), sep=",", index=False)