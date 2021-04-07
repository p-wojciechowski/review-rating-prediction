# Notebooks

Here are notebooks where EDA and model selection are performed.

## EDA

There are 20491 hotel reviews together with ratings/stars (scale from 1 to 5) that were given along with the reviews. 

* 96% of reviews are shorter than 300 words (that fact will be used during preprocessing for determining padding length).
* Over 70% of ratings are 4-star and 5-star. For longest reviews (longer than 300 words), distribution of ratings is more balanced.
* Inspecting structure of reviews in terms of text sections' sentiments. Pre-trained sentiment analyzer [VADER](https://www.nltk.org/howto/sentiment.html) was used to find some patterns in which parts of a review are more positive or negative. (*e.g. 2-star reviews start with some negative conclusions, followed by some positive words of praise, and then probably explain the lower rating (mild negative sentiment at the end)*)
* It seems that each rating (number of given stars) category has its own pattern of sentiment values localization. Thus, information on the spatiality across review will be very important and solutions that lose it, like bag-of-words, shouldn't be used.

## Model selection

I decided to treat this task as a regression problem to keep the importance of ratings' order. Confusing a 5-star review with a 1-star one is a bigger mistake than confusing it with a 4-star one. In case of classification, these mistakes would be treated equally.

Review texts are tokenized and padded to a length of 300. Pre-train word vectors created with [the GloVe](https://nlp.stanford.edu/projects/glove/) algorithm (glove.6B.300d variant ) are used to create an embedding matrix for an embedding layer. 

Mean and median of Rating are used as baseline models.

### Dense network

Simple network to begin with. At the top of the Embedding layer, a dense layer is attached. First training resulted in oscillating `train loss` which indicates a too high learning rate. After adjusting it, model achieved convergence but with big overfitting.

### CNN

Convolutional layers used on text can replace forming n-grams and can extract patterns and contexts from text to some degree.

For searching optimal structure of the network and its hyperparameters I used [Keras Tuner](https://keras-team.github.io/keras-tuner/). It implements an efficient Bayesian search algorithm, which is much better than simple grid search or random search. 

### LSTM

LSTM is a variation of RNN which is more suited for extraction information from sequences (here, sequences of words). LSTMs are often used in NLP problems, so we gave it a try. Network with LSTM gave better results than CNN.

