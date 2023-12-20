from pathlib import Path
import re
from bs4 import BeautifulSoup as bs
import requests
import pandas as pd
from tqdm import tqdm

# from their website, go through all speakers and collect the text of every devotional
# start = "https://speeches.byu.edu/talks/?event=devotional" # this is the link to go back by date, doesn't work well
start = "https://speeches.byu.edu/speakers/"

page = requests.get(start)
soup = bs(page.content, "html.parser")
all_links = set()

links = soup.select(".archive-item__link")
speakers = [link["href"] for link in links]

speech_df = pd.DataFrame(columns=["type", "speaker", "title", "date", "text"])

for i, speaker_link in tqdm(enumerate(speakers)):
    # request every speech for a speaker
    # check that it's a devotional
    # take all the text from that page - big csv with everything for now
    speaker_page = requests.get(speaker_link)
    soup = bs(speaker_page.content, "html.parser")

    speeches = [link["href"] for link in soup.select(".card__header--reduced a")]
    
    for speech_link in speeches:
        try:
            speech_page = requests.get(speech_link)
            speech_soup = bs(speech_page.content, "html.parser")
            
            # if not a devotional, ignore it
            speech_type = speech_soup.find("a", class_="single-speech__type").get_text()
            if speech_type != "Devotional":
                continue

            title = speech_soup.find("h1", class_="single-speech__title").get_text()
            date = speech_soup.find("p", class_="single-speech__date").get_text()
            speaker = speech_soup.find("h2", class_="single-speech__speaker").get_text().strip()
            text_list = speech_soup.select("li , .single-speech__content p")
            
            # remove the ©.*All rights reserved last paragraph
            if re.search(r"©.*All rights reserved", text_list[-1].get_text()) is not None:
                text_list.pop()

            text = ""
            for para in text_list:
                # combine the text of all paragraphs
                text += para.get_text() + "\n"
            
            # get the date, speaker name, and all text
            speech_df.loc[len(speech_df.index)] = [speech_type, speaker, title, date, text]
        except Exception as e:
            # print(e)
            with open(Path("final_project/speech_errors.txt"), "a") as f:
                print(type(e), e, f"\n{title}\n", file=f)

# rename index column and write to csv
speech_df.index.name="Index"
speech_df.to_csv(Path("final_project/byu_speeches.csv"),index=True)