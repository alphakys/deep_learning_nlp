from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from keras.preprocessing.text import text_to_word_sequence

raw_text = "A barber is a person. a barber is good person. a barber is huge person. he Knew A Secret! The Secret He Kept is huge secret. Huge secret. His barber kept his word. a barber kept his word. His barber kept his secret. But keeping and keeping such a huge secret to himself was driving the barber crazy. the barber went up a huge mountain."
from nltk.corpus import stopwords

stopword_list = stopwords('english')
stopword_list = stopwords.words('english')
stopword_list_set = set(stopword_list)
stopword_list.__len__
len(stopword_list)
len(stopword_list_set)
stopword_list = stopword_list_set
del stopword_list_set
sentences = sent_tokenize(raw_text)
sentences = sent_tokenize(raw_text).lower()
sentences = [sentence.lower() for sentence in sentences]
[senten for senten in sentences if senten not in stopword_list]
stopword_list
sentences
tokenized_sentences = [[word_tokenize(senten)] for senten in sentences]
tokenized_sentences = [[w for w in word_tokenize(senten) if w not in stopword_list] for senten in sentences]
tokenized_sentences = [[w for w in word_tokenize(senten) if w not in stopword_list] and len(w) > 2 for senten in
                       sentences]
tokenized_sentences = [[w for w in word_tokenize(senten) if w not in stopword_list and len(w) > 2] for senten in
                       sentences]
from keras.preprocessing.text import Tokenizer

keras_tokenizer = Tokenizer()
tokenized_sentences
keras_tokenizer.word_counts
keras_tokenizer = Tokenizer(num_words=6)
keras_tokenizer.fit_on_texts(tokenized_sentences)
keras_tokenizer.texts_to_sequences()
keras_tokenizer.texts_to_sequences(tokenized_sentences)
