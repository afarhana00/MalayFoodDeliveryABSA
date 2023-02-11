import streamlit as st
import pandas as pd
import malay_preprocess as mp 
from absa_predict import prediction

st.title('ABSA on Food Delivery Services')
predict_cont = st.container()

with predict_cont:

    with st.form("my_form"):
        sentences = st.text_input("Please insert sentence (in BM) related to food delivery services to predict the aspect term and its polarity.")

        submitted = st.form_submit_button("Submit")

    if sentences != "":
        clean_sentence = mp.partial_clean(sentences)
        clean_sentence = mp.malaya_preprocess(clean_sentence)
        clean_sentence = mp.malaya_normalizer(clean_sentence)
        aspect, sentiment, confidence = prediction(clean_sentence)
        
        if len(aspect)==0:
            ayat = "No related aspect found. Please type other sentence."
            ayat = '<p style="font-family:Calibri; color:#00fff2; font-size: 20px;">'+ayat+'</p>'
            st.markdown(ayat, unsafe_allow_html=True)

        else:
            data = {'Aspect': aspect,
                    'Sentiment': sentiment,
                    'Confidence': confidence}

            df = pd.DataFrame(data)

            st.table(df)
