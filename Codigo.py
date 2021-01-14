####################################################################Important####################################################################
#The following code was inspired by the articles below:
#1. https://medium.com/turing-talks/uma-an%C3%A1lise-de-dom-casmurro-com-nltk-343d72dd47a7
#2. https://towardsdatascience.com/generating-text-with-tensorflow-2-0-6a65c7bdc568
#Therefore, our work only adapts such content to Machado de Assis' opus, not owning or being proprietary of this part of the content. 
#Credit is given to the original authors.
##################################################################################################################################################

####################################################################Importante####################################################################
#O código a seguir foi inspirado pelo que é feito nos links:
#1. https://medium.com/turing-talks/uma-an%C3%A1lise-de-dom-casmurro-com-nltk-343d72dd47a7
#2. https://towardsdatascience.com/generating-text-with-tensorflow-2-0-6a65c7bdc568
#Assim, o presente trabalho apenas adapta tais conteúdos à obra de Machado de Assis, não sendo dono ou proprietário dessa parte do conteúdo. 
#O crédito é dados aos autores originais.
##################################################################################################################################################

#Importando a obra

import numpy as np
import pandas as pd
import tensorflow as tf
import nltk

nltk_id = 'machado'
nltk.download(nltk_id)

memorias_postumas = nltk.corpus.machado.raw('romance/marm05.txt')
memorias_postumas1 = memorias_postumas[2025:]

#Vendo a frequência

import re
nltk.download('stopwords')
nltk.download('punkt')
from nltk.probability import FreqDist
from nltk.tokenize import word_tokenize
def pre_process(text):
    letras_min =  re.findall(r'\b[A-zÀ-úü]+\b', text.lower())
    stopwords = nltk.corpus.stopwords.words('portuguese')
    stop = set(stopwords)
    sem_stopwords = [w for w in letras_min if w not in stop]
    texto_limpo = " ".join(sem_stopwords)
    return texto_limpo
text = pre_processamento(memorias_postumas1)
tokens = word_tokenize(text)
fd = FreqDist(tokens)
import matplotlib.pyplot as plt
plt.figure(figsize = (13, 8))
fd.plot(30, title = "Palavras Frequentes")

#Preparando

vocab = sorted(set(memorias_postumas1))
char_to_ind = {char:ind for ind,char in enumerate(vocab)}
ind_to_char = np.array(vocab)
encoded_text = np.array([char_to_ind[c] for c in memorias_postumas1])
linha = "Algum tempo hesitei se devia abrir estas memórias pelo princípio ou pelo fim, isto é, se poria em primeiro lugar o meu nascimento ou a minha morte."
len(linha)
print(round(0.9*len(encoded_text)))
text_as_int = encoded_text
tr_text = text_as_int[:round(0.85*len(encoded_text))] 
val_text = text_as_int[round(0.85*len(encoded_text)):] 
print(text_as_int.shape, tr_text.shape, val_text.shape)
def get_divisors(n):
    for i in range(1, int(n / 2) + 1):
        if n % i == 0:
            yield i
    yield n
print(list(get_divisors(n = 300629)))
batch_size = 67
buffer_size = 10000
embedding_dim = 96
epochs = 50
seq_length = 150
examples_per_epoch = len(memorias_postumas1)//seq_length
rnn_units = 1024
vocab_size = len(vocab)

tr_char_dataset = tf.data.Dataset.from_tensor_slices(tr_text)
val_char_dataset = tf.data.Dataset.from_tensor_slices(val_text)
tr_sequences = tr_char_dataset.batch(seq_length+1, drop_remainder=True)
val_sequences = val_char_dataset.batch(seq_length+1, drop_remainder=True)
def split_input_target(chunk):
    input_text = chunk[:-1]
    target_text = chunk[1:]
    return input_text, target_text
tr_dataset = tr_sequences.map(split_input_target).shuffle(buffer_size).batch(batch_size, drop_remainder=True)
val_dataset = val_sequences.map(split_input_target).shuffle(buffer_size).batch(batch_size, drop_remainder=True)
print(tr_dataset, val_dataset)

#Construindo a rede

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding,GRU,Dense

def build_model(vocab_size,embedding_dim,rnn_units,batch_size):

  model = Sequential()
  
  model.add(Embedding(vocab_size,embedding_dim,batch_input_shape = [batch_size,None]))
  model.add(GRU(rnn_units, return_sequences=True, stateful=True,recurrent_initializer="glorot_uniform",dropout=0.5))
  model.add(GRU(rnn_units, return_sequences=True, stateful=True,recurrent_initializer="glorot_uniform",dropout=0.5))
  model.add(Dense(vocab_size))

  return model

model = build_model(
    vocab_size = len(vocab),
    embedding_dim=embedding_dim,
    rnn_units=rnn_units,
    batch_size=batch_size)
    
model.summary()
for input_example_batch, target_example_batch in tr_dataset.take(1):
    example_batch_predictions = model(input_example_batch)
    print(example_batch_predictions.shape)
def loss(labels, logits):
    return tf.keras.losses.sparse_categorical_crossentropy(labels,    logits, from_logits=True)
example_batch_loss  = loss(target_example_batch, example_batch_predictions)
print("Loss:      ", example_batch_loss.numpy().mean())
    
optimizer = tf.keras.optimizers.Adam()
model.compile(optimizer=optimizer, loss=loss)
patience = 5
early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience)

#Treinando

history = model.fit(tr_dataset, epochs=epochs, callbacks=[early_stop] , validation_data=val_dataset)
print("Treinamento encerrado, não houve meçhora após {} épocas".format(patience))

pd.DataFrame(history.history).plot(figsize = (10,8))

model.save('Memórias_Póstumas.h5')

model = build_model(vocab_size, embedding_dim, rnn_units, batch_size=1)
model.load_weights('/content/drive/MyDrive/Machine Learning/Deep Learning/Tensorflow 2.0/NLP/Memóras_Póstumas4.h5') 
model.build(tf.TensorShape([1, None]))

#Gerando texto

def generate_text(model,start_seed,gen_size=500,temp=1.0):
  num_generate = gen_size
  input_eval = [char_to_ind[s] for s in start_seed]
  input_eval = tf.expand_dims(input_eval,0)

  text_generated = []

  temperature = temp

  model.reset_states()

  for i in range(0,num_generate):
    predictions = model(input_eval)

    predictions = tf.squeeze(predictions, 0)

    predictions = predictions/temperature

    predicted_id = tf.random.categorical(predictions,num_samples=1)[-1,0].numpy()

    input_eval = tf.expand_dims([predicted_id],0)

    text_generated.append(ind_to_char[predicted_id])

  return (start_seed + "".join(text_generated))
  
print(generate_text(model, start_seed="Brás", temp = 0.50, gen_size = 1000))
