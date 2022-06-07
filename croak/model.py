import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import random
from pathlib import Path
import soundfile as sf

class AudioDataset:
    """ Open a folder full of OPUS audio files to use as training data for a neural network.
        This works by reading the audio, splitting it into frames, and presenting several frames at once as an example.
        The target of the network is to predict the next frame.
    """
    def __init__(self, root: Path, frame_width:int=128, chain_length:int=128):
        """ Open a new dataset.

        Parameters
        ==========
        root: Path
            The directory the OPUS files are stored in.
        frame_width: int
            The width of each frame in samples.
        chain_length: int
            The number of frames to use as input to the network.
        """
        self.root = root
        self.files = list(self.root.glob("**/*.opus"))
        self.frame_width = frame_width
        self.chain_length = chain_length
        self.buffer = np.zeros(0)
        self.cursor=0

    def generate_music_batch(self, batch_size:int=1):
        """ Generate a minibatch for training.

        Parameters
        ==========
        batch_size: int
            The number of examples to generate.
        """
        start, end = self.cursor, self.cursor + (self.frame_width * self.chain_length * batch_size)
        # Refill the buffer if we're at the end
        if self.buffer.size <= end:
            filename = random.choice(self.files)
            self.buffer = sf.read(filename)[0] # a numpy array of shape (n_samples, n_channels)
            self.cursor = 0
            assert self.buffer.size > end, "Buffer is too small to generate a batch of size {}".format(batch_size)
        # Extract the data
        data = self.buffer[start:end]
        # Reshape it to a batch of examples
        data = data.reshape((batch_size, self.chain_length, self.frame_width))
        # Split it into the inputs and output along the chain
        inputs = data[:, :-1]
        outputs = data[:, 1:]
        # Skip forward
        self.cursor = end
        # Return the data
        return (inputs, outputs)

class MusicModelWrapper:
    def __init__(self,
            window_count:int=100,
            embed_dim:int=128,
            num_heads:int=2,
            feed_forward_dim:int=128,
            window_size:int=128,
            num_layers:int=2,
        ):
        self.window_count = window_count
        self.window_size = window_size
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.feed_forward_dim = feed_forward_dim
        self.num_layers = num_layers
        self.model = self._create_model()
    
    def _create_model(self) -> keras.Model:
        inputs = layers.Input(shape=(self.window_count,), dtype=tf.int32)
        x = PositionEmbedding(self.window_count, self.embed_dim)(inputs)
        for _ in range(self.num_layers):
            x = TransformerBlock(self.embed_dim, self.num_heads, self.feed_forward_dim)(x)
        outputs = layers.Dense(self.window_size)(x)
        model = keras.Model(inputs=inputs, outputs=[outputs, x])
        model.compile(
            "adam", loss="mse", metrics=["mse"]
        )  # No loss and optimization based on word embeddings from transformer block
        return model
    
    def fit(self, dataset:AudioDataset, epochs:int=100, batch_size:int=32):
        self.model.fit(
            dataset,
            epochs=epochs,
            batch_size=batch_size
        )


class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = layers.MultiHeadAttention(num_heads, embed_dim)
        self.ffn = keras.Sequential(
            [layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim),]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs):
        input_shape = tf.shape(inputs)
        batch_size = input_shape[0]
        seq_len = input_shape[1]
        causal_mask = self.causal_attention_mask(batch_size, seq_len, seq_len, tf.bool)
        attention_output = self.att(inputs, inputs, attention_mask=causal_mask)
        attention_output = self.dropout1(attention_output)
        out1 = self.layernorm1(inputs + attention_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output)
        return self.layernorm2(out1 + ffn_output)


    @staticmethod
    def causal_attention_mask(batch_size, n_dest, n_src, dtype):
        """
        Mask the upper half of the dot product matrix in self attention.
        This prevents flow of information from future tokens to current token.
        1's in the lower triangle, counting from the lower right corner.
        """
        i = tf.range(n_dest)[:, None]
        j = tf.range(n_src)
        m = i >= j - n_src + n_dest
        mask = tf.cast(m, dtype)
        mask = tf.reshape(mask, [1, n_dest, n_src])
        mult = tf.concat(
            [tf.expand_dims(batch_size, -1), tf.constant([1, 1], dtype=tf.int32)], 0
        )
        return tf.tile(mask, mult)


class PositionEmbedding(layers.Layer):
    def __init__(self, maxlen, embed_dim):
        super(PositionEmbedding, self).__init__()
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=embed_dim)

    def call(self, x):
        maxlen = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        return x + self.pos_emb(positions)


class TextGenerator(keras.callbacks.Callback):
    """A callback to generate text from a trained model.
    1. Feed some starting prompt to the model
    2. Predict probabilities for the next token
    3. Sample the next token and add it to the next input

    Arguments:
        max_tokens: Integer, the number of tokens to be generated after prompt.
        start_tokens: List of integers, the token indices for the starting prompt.
        index_to_word: List of strings, obtained from the TextVectorization layer.
        top_k: Integer, sample from the `top_k` token predictions.
        print_every: Integer, print after this many epochs.
    """

    def __init__(
        self, max_tokens, start_tokens, index_to_word, top_k=10, print_every=1
    ):
        self.max_tokens = max_tokens
        self.start_tokens = start_tokens
        self.index_to_word = index_to_word
        self.print_every = print_every
        self.k = top_k

    def sample_from(self, logits):
        logits, indices = tf.math.top_k(logits, k=self.k, sorted=True)
        indices = np.asarray(indices).astype("int32")
        preds = keras.activations.softmax(tf.expand_dims(logits, 0))[0]
        preds = np.asarray(preds).astype("float32")
        return np.random.choice(indices, p=preds)

    def detokenize(self, number):
        return self.index_to_word[number]

    def on_epoch_end(self, epoch, logs=None):
        start_tokens = [_ for _ in self.start_tokens]
        if (epoch + 1) % self.print_every != 0:
            return
        num_tokens_generated = 0
        tokens_generated = []
        while num_tokens_generated <= self.max_tokens:
            pad_len = self.window_count - len(start_tokens)
            sample_index = len(start_tokens) - 1
            if pad_len < 0:
                x = start_tokens[:self.window_count]
                sample_index = self.window_count - 1
            elif pad_len > 0:
                x = start_tokens + [0] * pad_len
            else:
                x = start_tokens
            x = np.array([x])
            y, _ = self.model.predict(x)
            sample_token = self.sample_from(y[0][sample_index])
            tokens_generated.append(sample_token)
            start_tokens.append(sample_token)
            num_tokens_generated = len(tokens_generated)
        txt = "".join(
            [self.detokenize(_) for _ in self.start_tokens + tokens_generated]
        )
        print(f"generated text:\n{txt}\n")


if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--window-count", type=int, default=128)
    parser.add_argument("--window-size", type=int, default=128)
    parser.add_argument("--embed-dim", type=int, default=128)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    args = parser.parse_args()
    model = MusicModelWrapper(
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        feed_forward_dim=args.embed_dim, # Set the same for now
        dropout_rate=args.dropout_rate,
        num_layers=args.num_layers,
        window_count=args.window_count,
        window_size=args.window_size,
    )


# Tokenize starting prompt
word_to_index = {}
for index, word in enumerate(vocab):
    word_to_index[word] = index

start_prompt = "The Common Ovenbird is a"
start_tokens = [word_to_index.get(_, 1) for _ in start_prompt]
num_tokens_generated = 40
text_gen_callback = TextGenerator(num_tokens_generated, start_tokens, vocab)

model = create_model()

model.fit(text_ds, epochs=100, steps_per_epoch=1000, callbacks=[text_gen_callback])
