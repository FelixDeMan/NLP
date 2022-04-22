import pandas as pd
import csv
import numpy as np
from wordfreq import word_frequency
from wordfreq import zipf_frequency
import spacy
nlp = spacy.load("en_core_web_sm")
import matplotlib.pyplot as plt
import seaborn as sns

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
num_instances_multiple_tokens = sum(df['targetlength'] > 1)
max_target_length = max(df['targetlength'])

# print results
print('num instances with more than one token:', num_instances_multiple_tokens)
print('max num tokens for an instance:', max_target_length)

# extract only the targets consisting of one token with binary label 1 
df_single_complex = df[(df['targetlength']==1) & (df['bin']==1)]

# add column with word length 
df_single_complex['wordlength'] = df_single_complex['target'].str.len()

# iteratively create columns with word frequency and POS tag
word_freqs_list = []
word_freqs_log_list = []
pos_tag_list = []
for word in df_single_complex['target']:
    # extract word frequency: both linear and logarithmic scale
    word_freqs_list.append(word_frequency(word, 'en'))
    word_freqs_log_list.append(zipf_frequency(word, 'en'))
    # assign POS label
    pos_tag = nlp(word)[0].pos_
    pos_tag_list.append(str(pos_tag))
df_single_complex['wordfreq'] = word_freqs_list   
df_single_complex['wordfreqlog'] = word_freqs_log_list
df_single_complex['postag'] = pos_tag_list   

# frame final results 
df_results = df_single_complex[['target','prob','wordlength','wordfreq', 'wordfreqlog', 'postag']]

# compute Pearson correlation 
correlation_matrix = df_single_complex[['target','prob','wordlength','wordfreq', 'postag']].corr(method='pearson')
print(correlation_matrix)

# create scatter plot of word length vs complexity
sns.regplot(data=df_results, x='wordlength', y='prob', color=".4")
plt.title('Relation between word complexity and word length')
plt.xlabel('word length')
plt.ylabel('probabilistic complexity')
plt.show()

# create scatter plot of word frequency vs complexity
sns.regplot(data=df_results, x='wordfreqlog', y='prob')
plt.title('Logarithmic relation between word complexity and word frequency')
plt.xlabel('word frequency')
plt.ylabel('probabilistic complexity')
plt.show()

# create a boxplot of POS tag vs complexity 
sns.scatterplot(data=df_results, x='postag', y='prob', color="0.1")
plt.title('Relation between word complexity and POS tag')
plt.xlabel('POS tag')
plt.ylabel('probabilistic complexity')
plt.show()
