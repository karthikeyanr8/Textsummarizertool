# Install necessary libraries
!pip install gradio nltk beautifulsoup4 requests

import nltk
import requests
from nltk.tokenize import sent_tokenize, word_tokenize
nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords
from collections import defaultdict
from string import punctuation
from heapq import nlargest
from bs4 import BeautifulSoup
import gradio as gr

# Frequency summarizer class
class FrequencySummarizer:
    def __init__(self, min_cut=0.1, max_cut=0.9):
        """
        Initialize the text summarizer.
        Words that have a frequency term lower than min_cut
        or higher than max_cut will be ignored.
        """
        self._min_cut = min_cut
        self._max_cut = max_cut
        self._stopwords = set(stopwords.words('english') + list(punctuation))

    def _compute_frequencies(self, word_sent):
        """ 
        Compute the frequency of each word.
        Input: word_sent (list of tokenized sentences).
        Output: freq (dict where freq[w] is the frequency of w).
        """
        freq = defaultdict(int)
        for s in word_sent:
            for word in s:
                if word not in self._stopwords:
                    freq[word] += 1
        # Frequency normalization and filtering
        m = float(max(freq.values()))
        for w in list(freq.keys()):  # Use a list to prevent errors while modifying dict
            freq[w] = freq[w] / m
            if freq[w] >= self._max_cut or freq[w] <= self._min_cut:
                del freq[w]
        return freq

    def summarize(self, text, n):
        """
        Return a list of n sentences which represent the summary of the text.
        """
        sents = sent_tokenize(text)
        assert n <= len(sents)
        word_sent = [word_tokenize(s.lower()) for s in sents]
        self._freq = self._compute_frequencies(word_sent)
        ranking = defaultdict(int)
        for i, sent in enumerate(word_sent):
            for w in sent:
                if w in self._freq:
                    ranking[i] += self._freq[w]
        sents_idx = self._rank(ranking, n)
        return [sents[j] for j in sents_idx]

    def _rank(self, ranking, n):
        """ Return the first n sentences with highest ranking """
        return nlargest(n, ranking, key=ranking.get)

# Function to extract text and title from an article URL
def get_only_text(url):
    """ 
    Return the title and the text of the article at the specified URL
    """
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'lxml')  # Use 'lxml' as parser
    text = ' '.join(map(lambda p: p.text, soup.find_all('p')))
    return soup.title.text, text

# Function to get summaries for the articles
def get_article_summaries(article_url, num_sentences=3):
    """ 
    Get article summaries from a given article URL
    """
    title, text = get_only_text(article_url)  # Get article content
    fs = FrequencySummarizer()
    summary = fs.summarize(text, num_sentences)
    
    summary_text = f"Article Title: {title}\n"
    summary_text += "\nSummary:\n" + "\n".join([f"- {sentence}" for sentence in summary])
    return summary_text

# Gradio Interface function
def summarize_articles(article_url, num_sentences):
    return get_article_summaries(article_url, num_sentences)

# Gradio Interface
article_url_input = gr.Textbox(label="Article URL", value='https://timesofindia.indiatimes.com/sports/cricket/ipl/top-stories/bcci-announces-ipl-schedules-for-next-three-seasons/articleshow/115550810.cms')
num_sentences_input = gr.Slider(minimum=1, maximum=5, step=1, label="Number of Sentences for Summary", value=3)
output = gr.Textbox(label="Article Summaries", interactive=False)

# Create the Gradio interface
gr.Interface(fn=summarize_articles, 
             inputs=[article_url_input, num_sentences_input], 
             outputs=output,
             live=True).launch()
