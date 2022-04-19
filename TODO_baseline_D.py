import pandas as pd
import csv
import numpy as np

# printing data DELETE LATER
with open('data/original/english/WikiNews_Train.tsv', 'r', encoding='utf-8') as f:
    #dataframe = pd.read_csv(f, sep='\t')
    read_tsv = csv.reader(f, delimiter="\t")
    for row in read_tsv:
        #print(row)
        pass
#print(dataframe)

# importing data and saving to variable
with open('data/original/english/WikiNews_Train.tsv', 'r', encoding='utf-8') as f:
    df = pd.read_csv(f, sep='\t')
    df.columns = ['code', 'sentence', 'onset', 'offset', 'target', 'native', 'nonnative', 'labelnat', 'labelnon', 'bin', 'prob']

# extracting columns
target_words = df['target']
binary_scores = df['bin']
prob_scores = df['prob']

# calculate number of instances labeled with 1 
labeled1 = sum(binary_scores)
print('num instances labeled 1:', labeled1)

# calculate number of instances labeled with 0
labeled0 = sum(df['bin'] == 0)
print('num instances labeled 0:', labeled0)

# check whether this sums up to the number of entries
print(labeled1 + labeled0)
print(len(df))

# calculate statistics of probabilistic label 
min_prob = min(prob_scores)
max_prob = max(prob_scores)
median_prob = np.median(prob_scores)
mean_prob = np.mean(prob_scores)
std_prob = np.std(prob_scores)

# print results 
print('min:', min_prob)
print('max:', max_prob)
print('median:', median_prob)
print('mean:', mean_prob)
print('std:', std_prob)

# add column with number of tokens in the target 
df['targetlength'] = ''
for number in range(0,len(df)):
    targetlength = len(target_words[number].split(' '))
    df.at[number, 'targetlength'] = targetlength

# calculate statistics on target length
instances_multiple_tokens = sum(df['targetlength'] > 1)
max_target_length = max(df['targetlength'])

# print results
print('num instances with more than one token:', instances_multiple_tokens)
print('max num tokens for an instance:', max_target_length)