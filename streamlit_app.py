import torch
from rnn_summarizer_github import EncoderRNN, DecoderRNN, prod_evaluate
import streamlit as st
from gensim.summarization import summarize
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer

def main():
    decoderpath = 'rnn_models/decoder.pt'
    encoderpath = 'rnn_models/encoder.pt'
    # encoder = BertModel.from_pretrained('google/bert_uncased_L-4_H-512_A-8').to(device)

    encoder1 = torch.load(encoderpath,map_location=torch.device('cpu'))
    decoder1 = torch.load(decoderpath,map_location=torch.device('cpu'))
    st.title("Summary and Text Preprocessing")
    # activity1 = ["Summarize","Text Preprocessing"]
    # choice = st.sidebar.selectbox("Select Function",activity1)

    # encoder = BertModel.from_pretrained('google/bert_uncased_L-4_H-512_A-8').to(device)
    # tokenizer = BertTokenizer.from_pretrained('google/bert_uncased_L-4_H-512_A-8', do_lower=True)
    # encoder_config = BertConfig.from_pretrained("google/bert_uncased_L-4_H-512_A-8")
    # starting_word = tokenizer.vocab['[unused0]']
    # for params in encoder.parameters():
    #     params.require_grad = False




    st.subheader("Summary with NLP")
    raw_text = st.text_area(label = "enter text here", value = "main representative body british jews called wigan chairman dave whelan comments outrageous labelled apology halfhearted whelan set face football association charge responded controversy wigan appointment malky mackay manager telling guardian think jewish people chase money everybody else wigan owner since apologised offence caused facing critical situation club one latics shirt sponsors kitchen firm premier range announced breaking ties club due whelan appointment mackay subject fa investigation sending allegedly racist text messages iain moody former head recruitment cardiff dave whelan left jewish body outraged following comments aftermath malky mackay hiring board deputies british jews vicepresident jonathan arkush said statement dave whelan comments jews outrageous offensive bring club game disrepute halfhearted apology go far enough insult whole group people say would never insult hope ok need see proper apology full recognition offence caused whelan role chair football club responsibility set tone players supporters mackay appointed wigan boss week despite text email scandal racism antisemitism prevail pitch acceptable unchallenged boardroom taking matter football association kick", )
    summary_choice = st.selectbox("Summary Choice",["Genism","Sumy Lex Rank", "RNN"])
    if st.button("Generate Summary"):
        print("Button Pressed")
        if summary_choice == "Genism":
            summary_result = summarize(raw_text)
            st.write(summary_result)
        if summary_choice == "RNN":
            summary_result = prod_evaluate(raw_text, encoder1, decoder1)
            st.write(summary_result)
if __name__ == '__main__':
    main()
