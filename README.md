# Detecting Semantic Differences between LDS and Christian Speech

## Abstract

This project seeks to apply semantic shift detection methods to speech of distinct religious groups. I compare speech between English-speaking members of the Church of Jesus Christ of Latter-day Saints and other English-speaking Christians. I present a new corpus of speech from these two groups, collected from LDS and Christian subreddits, and with LDS speech supplemented with religious speeches from Brigham Young University and the Church's semiannual General Conferences.

I use BERT to generate word embeddings contextualized at a sentence level for a subset of the corpus. The results of keyness analyses comparing LDS speech to general speech, Christian to general, and LDS and Christian to each other inform which words I focus on to determine if there are semantic differences. For each target word, I collect the embeddings and reduce them to two dimensions using Principal Component Analysis (PCA) to visually examine them and determine if there appear to be clusters indicating meaning correlating with religious group. I find that some words (*church, sin, mormon*) do have visible differences detectable using this method.
