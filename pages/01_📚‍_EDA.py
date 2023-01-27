import streamlit as st
import malaya
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import seaborn as sns
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
from wordcloud import WordCloud
from PIL import Image

st.title("Exploratory Data Analysis")
text_stats = st.container()
ngram_exp = st.container()
s_analysis = st.container()

@st.cache
def get_data():
    tweet = pd.read_csv('data/cleaneddataset.csv')
    return tweet

def plot_character_length_histogram(text, col):
    
    fig, ax = plt.subplots()
    ax.hist(text.str.len(), bins=10)
    ax.set_title("Number of characters in each headlines")
    ax.set_xlabel("Number of characters")
    ax.set_ylabel("Frequency")

    col.pyplot(fig)

def plot_word_number_histogram(text, col):

    fig = plt.figure()
    plt.hist(text.str.split().map(lambda x: len(x)), bins=10)
    plt.title("Number of words in each headlines")
    plt.xlabel("Number of words")
    plt.ylabel("Frequency")

    col.pyplot(fig)

def plot_top_words_barchart(text, col):
    
    new=text.str.split()
    new=new.values.tolist()
    corpus=[word for i in new for word in i]

    counter=Counter(corpus)
    most=counter.most_common()
    x, y=[], []
    for word,count in most[:40]:
        x.append(word)
        y.append(count)

    fig = plt.figure()
    sns.barplot(x=y,y=x)
    plt.title("Number of non-stopwords in the dataset")
    plt.ylabel("Non-stopwords")
    plt.xlabel("Frequency")
    col.pyplot(fig)

def plot_query_barchart(text, col):

    y_val = text.value_counts(sort=False)
    y=[]
    x = pd.unique(text)
    x = x.tolist()
    x_index = []

    for i in range(len(y_val)):
        y.append(y_val[i])

    for i in range(len(x)):
        x_index.append(i)

    fig = plt.figure()
    plt.bar(x_index,y)
    plt.title("Number of queries in the dataset")
    plt.xlabel("Query")
    plt.ylabel("Frequency")

    col.pyplot(fig)

def plot_top_ngrams_barchart(text, n):
    if n == 2:
        ayat = "bigram"
        st.write("Bar chart for the top bigram in the dataset")
    elif n == 3:
        ayat = "trigram"
        st.write("Bar chart for the top trigram in the dataset")

    new= text.str.split()
    new=new.values.tolist()
    corpus=[word for i in new for word in i]

    def _get_top_ngram(corpus, n=None):
        vec = CountVectorizer(ngram_range=(n, n)).fit(corpus)
        bag_of_words = vec.transform(corpus)
        sum_words = bag_of_words.sum(axis=0) 
        words_freq = [(word, sum_words[0, idx]) 
                      for word, idx in vec.vocabulary_.items()]
        words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
        return words_freq[:10]

    top_n_bigrams=_get_top_ngram(text,n)[:10]
    x,y=map(list,zip(*top_n_bigrams))

    fig = plt.figure()
    sns.barplot(x=y,y=x)
    title_ner = f"Top {ayat} in the dataset"
    plt.title(title_ner)
    plt.xlabel("Frequency")
    st.pyplot(fig)

def plot_sentiment_barchart(text, add_wordcloud=False):
    sentiment_desc_col, sentiment_graph_col = st.columns(2)

    model = malaya.sentiment.multinomial()
    tweet_sentiment = model.predict(text)

    positive_count = tweet_sentiment.count('positive')
    neutral_count = tweet_sentiment.count('neutral')
    negative_count = tweet_sentiment.count('negative')
    
    x_val=['positive', 'neutral', 'negative']
    y_val=[positive_count, neutral_count, negative_count]

    fig = plt.figure()
    plt.bar(x_val,y_val)
    plt.title("Number of each polarity in the dataset")
    plt.xlabel("Polarity")
    plt.ylabel("Frequency")

    sentiment_graph_col.pyplot(fig)
    sentiment_desc_col.markdown("- The number of positive tweet:")
    sentiment_desc_col.write(positive_count)
    sentiment_desc_col.markdown("- The number of neutral tweet:")
    sentiment_desc_col.write(neutral_count)
    sentiment_desc_col.markdown("- The number of negative tweet:")
    sentiment_desc_col.write(negative_count)

    if add_wordcloud:
        st.write("")
        st.header("Wordcloud")
        positive_tweet = []
        neutral_tweet = []
        negative_tweet = []

        for i in range(len(text)):
            if tweet_sentiment[i]=='positive':
                positive_tweet.append(text[i])
            elif tweet_sentiment[i]=='neutral':
                neutral_tweet.append(text[i])
            elif tweet_sentiment[i]=='negative':
                negative_tweet.append(text[i])

        mask = np.array(Image.open('data/Dialog.png'))

        wordcloud_positive = WordCloud(
            background_color='white',
            width=3000,
            height=2000,
            max_words=500,
            max_font_size=70, 
            colormap='summer',
            scale=3,
            collocations=False,
            mask=mask,
            contour_color='#023075',
            contour_width=1,
            random_state=1)
        
        wordcloud_neutral = WordCloud(
            background_color='white',
            width=3000,
            height=2000,
            max_words=500,
            max_font_size=70, 
            colormap='Wistia',
            scale=3,
            collocations=False,
            mask=mask,
            contour_color='#023075',
            contour_width=1,
            random_state=1)

        wordcloud_negative = WordCloud(
            background_color='white',
            width=3000,
            height=2000,
            max_words=500,
            max_font_size=70, 
            colormap='Reds',
            scale=3,
            collocations=False,
            mask=mask,
            contour_color='#023075',
            contour_width=1,
            random_state=1)

        pos_wordcloud=wordcloud_positive.generate(str(positive_tweet))
        neu_wordcloud=wordcloud_neutral.generate(str(neutral_tweet))
        neg_wordcloud=wordcloud_negative.generate(str(negative_tweet))

        pos_wordcloud_col, neu_wordcloud_col, neg_wordcloud_col = st.columns(3)

        fig_pos = plt.figure()
        plt.axis('off')
        plt.imshow(pos_wordcloud)
        pos_wordcloud_col.write("Positive")
        pos_wordcloud_col.pyplot(fig_pos)

        fig_neu = plt.figure()
        plt.axis('off')
        plt.imshow(neu_wordcloud)
        neu_wordcloud_col.write("Neutral")
        neu_wordcloud_col.pyplot(fig_neu)

        fig_neg = plt.figure()
        plt.axis('off')
        plt.imshow(neg_wordcloud)
        neg_wordcloud_col.write("Negative")
        neg_wordcloud_col.pyplot(fig_neg)

tweet = get_data()

with text_stats:
    st.write("First 10 tweets from the dataset:")
    st.write(tweet.head(10))
    st.write("Total of tweets in the dataset: ", len(tweet))
    
    st.subheader("Number of characters in each tweets")
    clh_desc_col, clh_col = st.columns(2)
    plot_character_length_histogram(tweet['tweet'], clh_col)
    clh_desc_col.markdown("- The histogram shows that tweets range from 20 to 400 characters.")
    clh_desc_col.markdown("- Most of the tweets contains around 100 characters.")
    st.write("")

    st.subheader("Number of words in each tweets")
    wnh_desc_col, wnh_col = st.columns(2)
    plot_word_number_histogram(tweet['tweet'], wnh_col)
    wnh_desc_col.markdown("- Based on the histogram, the number of words in tweets range from 5 to 60.")
    wnh_desc_col.markdown("- The highest frequency for the words is around 15 to 20 words.")
    st.write("")

    st.subheader("Bar chart for the top words in the dataset")
    twb_desc_col, twb_col = st.columns(2)
    plot_top_words_barchart(tweet['tweet'], twb_col)
    twb_desc_col.markdown("- The most common words use in the tweets are 'foodpanda' and 'tidak'.")
    st.write("")

    st.subheader("List of queries used to search the tweets")
    query_graph_col, query_list_col = st.columns(2)
    query_list_col.write(pd.unique(tweet['query']))
    plot_query_barchart(tweet['query'], query_graph_col)
    st.write("")

with ngram_exp:
    list_of_ngram = st.radio("Which ngram you want to see?", ('Bigram', 'Trigram'), horizontal=True)
    
    if list_of_ngram == 'Bigram':
        n_gram = 2
    elif list_of_ngram == 'Trigram':
        n_gram = 3

    plot_top_ngrams_barchart(tweet['tweet'], n_gram)
    st.write("")

with s_analysis:
    st.header("Sentiment analysis at sentence level")
    st.write("")

    tweet_list = tweet['tweet'].values.tolist()
    plot_sentiment_barchart(tweet_list, add_wordcloud=True)
    st.write("")