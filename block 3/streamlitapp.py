import tensorflow as tf
#import tensorflow_datasets as tfds
import numpy as np
import os
from tqdm import tqdm
from transformers import pipeline, AutoConfig, AutoTokenizer, TFAutoModel
import streamlit as st


qa = pipeline("question-answering",
              model="/Users/retoheller/ml4ds_2020_g1/ml4ds_2020_g1/block_3/exercises/transformers",
              tokenizer="/Users/retoheller/ml4ds_2020_g1/ml4ds_2020_g1/block_3/exercises/transformers")

st.title('I answer your question')

st.subheader('Gib deinen Text ein:')

default_value_goes_here = """
Roger Federer has won a record eight Wimbledon men's singles titles, six Australian Open titles, five US Open titles (all consecutive, a record), and one French Open title. He is one of eight men to have achieved a Career Grand Slam. Federer has reached a record 31 men's singles Grand Slam finals, including 10 consecutively from the 2005 Wimbledon Championships to the 2007 US Open. Federer has also won a record six ATP Finals titles, 28 ATP Tour Masters 1000 titles, and a record 24 ATP Tour 500 titles. Federer was a member of Switzerland's winning Davis Cup team in 2014. He is also the only player after Jimmy Connors to have won 100 or more career singles titles, as well as to amass 1,200 wins in the Open Era.

"""
user_input = st.text_area("Gib hier deinen Text ein:", default_value_goes_here)

default_question = "How many Wimbledon men's singles titles has Roger Federer won?"
user_question = st.text_area("Gib hier deine Frage ein:", default_question)

if st.button("Answer my question"):
    answer = qa(question=user_question, context=user_input)
    st.write('Answer: %s' % answer["answer"])

