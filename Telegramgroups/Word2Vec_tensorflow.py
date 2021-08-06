import io
import re
import string
import tensorflow as tf
import tqdm
import spacy
from tensorflow.keras import Model
from tensorflow.keras.layers import Dot, Embedding, Flatten
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
import NLP
import pandas as pd
from nltk import word_tokenize


nlp = spacy.load("de_core_news_sm")
posts = NLP.posts_prep_two


def word2vec():
    df = pd.DataFrame(list(posts.find()))
    for message in df.Message:
        word_tokens = word_tokenize(message)
        # Vocabulary
        vocab, index = {}, 1  # start indexing from 1
        vocab['<pad>'] = 0  # add a padding token
        for token in word_tokens:
            if token not in vocab:
                vocab[token] = index
                index += 1
        vocab_size = len(vocab)
        print(vocab)
        inverse_vocab = {index: token for token, index in vocab.items()}
        print(inverse_vocab)
        example_sequence = [vocab[word] for word in word_tokens]
        print(example_sequence)

        # Skip-Grams
        window_size = 2
        positive_skip_grams, _ = tf.keras.preprocessing.sequence.skipgrams(
            example_sequence,
            vocabulary_size=vocab_size,
            window_size=window_size,
            negative_samples=0)
        print(len(positive_skip_grams))

        # Test
        for target, context in positive_skip_grams[:5]:
            print(f"({target}, {context}): ({inverse_vocab[target]}, {inverse_vocab[context]})")

        # Get target and context words for one positive skip-gram.
        target_word, context_word = positive_skip_grams[0]

        # Set the number of negative samples per positive context.
        num_ns = 4

        context_class = tf.reshape(tf.constant(context_word, dtype="int64"), (1, 1))
        negative_sampling_candidates, _, _ = tf.random.log_uniform_candidate_sampler(
            true_classes=context_class,  # class that should be sampled as 'positive'
            num_true=1,  # each positive skip-gram has 1 positive context class
            num_sampled=num_ns,  # number of negative context words to sample
            unique=True,  # all the negative samples should be unique
            range_max=vocab_size,  # pick index of the samples from [0, vocab_size]
            seed=SEED,  # seed for reproducibility
            name="negative_sampling"  # name of this operation
        )
        print(negative_sampling_candidates)
        print([inverse_vocab[index.numpy()] for index in negative_sampling_candidates])

        # Add a dimension so you can use concatenation (on the next step).
        negative_sampling_candidates = tf.expand_dims(negative_sampling_candidates, 1)

        # Concat positive context word with negative sampled words.
        context = tf.concat([context_class, negative_sampling_candidates], 0)

        # Label first context word as 1 (positive) followed by num_ns 0s (negative).
        label = tf.constant([1] + [0] * num_ns, dtype="int64")

        # Reshape target to shape (1,) and context and label to (num_ns+1,).
        target = tf.squeeze(target_word)
        context = tf.squeeze(context)
        label = tf.squeeze(label)

        # Test
        print(f"target_index    : {target}")
        print(f"target_word     : {inverse_vocab[target_word]}")
        print(f"context_indices : {context}")
        print(f"context_words   : {[inverse_vocab[c.numpy()] for c in context]}")
        print(f"label           : {label}")


if __name__ == '__main__':
    word2vec()
