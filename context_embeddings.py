# this should get contextual embeddings for every word? for just all of the keywords that I'm looking at? 
# https://medium.com/mlearning-ai/getting-contextualized-word-embeddings-with-bert-20798d8b43a4 
 
import pandas as pd
import numpy as np
import torch
from transformers import BertModel, BertTokenizer
from tqdm import tqdm
import spacy
from pathlib import Path

nlp = spacy.blank("en")
nlp.add_pipe("sentencizer")

# https://www.semanticscholar.org/reader/0cbd7b946983b807735dc0e990c359ab57e2ad82
# these guys ^^ use ELMo instead of BERt b/c it requires less computationally
model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states = True)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# likely want to generate embeddings for each word and store all of those with the word, and its lemma
# then will use the lemma column to compare it to others that match
# after generating all of them, can use kNN as suggested by the Ukrainian paper
# can visualize them by using PCA projections of the word


christian_df = pd.read_csv('final_project/christian_reddit.csv')
lds_df = pd.read_csv('final_project/lds_reddit.csv')

lds_texts = list(lds_df['text'].astype(str))
christian_texts = list(christian_df['text'].astype(str))

christian_sentences = list(pd.read_csv(Path('final_project/christian_sents.csv'))['text'])
lds_sentences = list(pd.read_csv(Path('final_project/lds_sents.csv'))['text'].astype(str))

# print(christian_sentences)
# print(lds_sentences)

# christian_sentences = []
# lds_sentences = []

# # for each text, split the text by sentences and put those all into a list
# for doc in tqdm(nlp.pipe(lds_texts)):
#     try:
#       for sent in doc.sents:
#         lds_sentences.append(sent.text)
#     except Exception as e:
#       with open(Path("final_project/lds_sents_errors.txt"), "a") as f:
#         print(type(e), e, f"\n{doc}\n", file=f)

# sent_df = pd.DataFrame(lds_sentences, columns = ['text'])
# sent_df.to_csv(Path('final_project/lds_sents.csv'), index_label='id')


# for doc in tqdm(nlp.pipe(christian_texts)):
#   for sent in doc.sents:
#     christian_sentences.append(sent.text)

# sent_df = pd.DataFrame(christian_sentences, columns = ['text'])
# sent_df.to_csv(Path('final_project/christian_sents.csv'), index_label='id')

print(len(lds_sentences))
print(len(christian_sentences))

tests = lds_sentences[:100]
encoded_input = tokenizer.batch_encode_plus(lds_sentences, add_special_tokens=True, 
                                            max_length=64, padding=True, 
                                            truncation=True,
                                            return_attention_mask=True, return_tensors='pt',
                                            )

with torch.no_grad():
    output_embeddings = model(encoded_input['input_ids'], encoded_input['attention_mask'])

hidden_states = output_embeddings[2]
print ("Number of layers:", len(hidden_states), "  (initial embeddings + 12 BERT layers)")
layer_i = 0
print ("Number of batches:", len(hidden_states[layer_i]))
batch_i = 0
print(tokenizer.convert_ids_to_tokens(encoded_input['input_ids'][0]))
print ("Number of tokens:", len(hidden_states[layer_i][batch_i])) # I'm going to somehow need to get the tokens attached back to their embeddings
token_i = 0
print ("Number of hidden units:", len(hidden_states[layer_i][batch_i][token_i]))

token_embeddings = torch.stack(hidden_states, dim=0)
token_embeddings.size()

token_embeddings = token_embeddings.permute(1,2,0,3)
token_embeddings.size()

token_vecs_sum = []
token_vecs_df = pd.DataFrame(columns=['token', 'embedding'])
for batch_i, batch in enumerate(token_embeddings):
  print(batch.size())
  tokenized_text = tokenizer.convert_ids_to_tokens(encoded_input['input_ids'][batch_i])
  # zip them together for each batch here, which should correspond to a sentence
  for token_i, token in enumerate(batch):
    sum_vec = torch.sum(token[-4:], dim=0)
    token_vecs_sum.append(sum_vec)
    if tokenized_text[token_i] != "[PAD]":
      token_vecs_df.loc[len(token_vecs_df.index)] = [tokenized_text[token_i], sum_vec]
print ('Shape is: %d x %d' % (len(token_vecs_sum), len(token_vecs_sum[0])))

token_vecs_df.to_csv(Path('final_project/lds_red_embeds.csv'))

encoded_input = tokenizer.batch_encode_plus(christian_sentences, add_special_tokens=True, 
                                            max_length=64, padding=True, 
                                            truncation=True,
                                            return_attention_mask=True, return_tensors='pt',
                                            )

with torch.no_grad():
    output_embeddings = model(encoded_input['input_ids'], encoded_input['attention_mask'])

hidden_states = output_embeddings[2]
print ("Number of layers:", len(hidden_states), "  (initial embeddings + 12 BERT layers)")
layer_i = 0
print ("Number of batches:", len(hidden_states[layer_i]))
batch_i = 0
print(tokenizer.convert_ids_to_tokens(encoded_input['input_ids'][0]))
print ("Number of tokens:", len(hidden_states[layer_i][batch_i])) # I'm going to somehow need to get the tokens attached back to their embeddings
token_i = 0
print ("Number of hidden units:", len(hidden_states[layer_i][batch_i][token_i]))

token_embeddings = torch.stack(hidden_states, dim=0)
token_embeddings.size()

token_embeddings = token_embeddings.permute(1,2,0,3)
token_embeddings.size()

token_vecs_sum = []
token_vecs_df = pd.DataFrame(columns=['token', 'embedding'])
for batch_i, batch in enumerate(token_embeddings):
  print(batch.size())
  tokenized_text = tokenizer.convert_ids_to_tokens(encoded_input['input_ids'][batch_i])
  # zip them together for each batch here, which should correspond to a sentence
  for token_i, token in enumerate(batch):
    sum_vec = torch.sum(token[-4:], dim=0)
    token_vecs_sum.append(sum_vec)
    if tokenized_text[token_i] != "[PAD]":
      token_vecs_df.loc[len(token_vecs_df.index)] = [tokenized_text[token_i], sum_vec]
print ('Shape is: %d x %d' % (len(token_vecs_sum), len(token_vecs_sum[0])))

token_vecs_df.to_csv(Path('final_project/christian_red_embeds.csv'))