import re
import string
import pandas as pd
import malaya

def partial_clean(tweet):
    username = "@\S+"
    new_tweet = re.sub(username, ' ',tweet) # Remove @tags

    hashtag = "#\S+"
    new_tweet = re.sub(hashtag, ' ',new_tweet) # Remove #tags

    new_tweet = new_tweet.lower() # Lowercasing

    text_noise = "https?:\S+|http?:\S|[^A-Za-z0-9]+" 
    new_tweet = re.sub(text_noise, ' ', new_tweet) # Remove links

    new_tweet = new_tweet.translate(new_tweet.maketrans('','',string.punctuation)) # Remove Punctuation

    new_tweet = new_tweet.strip() # Remove white spaces

    return new_tweet

def malaya_preprocess(tweet):
    corrector = malaya.spell.probability()
    segmenter = malaya.segmentation.viterbi()
    preprocessing = malaya.preprocessing.preprocessing(normalize=[], annotate=[], speller=corrector, segmenter=segmenter)

    new_tweet = ' '.join(preprocessing.process(tweet))
    return new_tweet

def malaya_normalizer(tweet):
    corrector = malaya.spell.probability()
    normalizer = malaya.normalize.normalizer(corrector, date=False)
    new_tweet = normalizer.normalize(tweet, normalize_entity=False, normalize_telephone = False, normalize_date = False, normalize_time = False)

    return new_tweet['normalize']