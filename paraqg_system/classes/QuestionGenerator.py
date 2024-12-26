from tensorflow.keras.models import load_model
import numpy as np
class QuestionGenerator:
    def __init__(self, model_path='classes/model/seq2seq_question_generation.h5', tokenizer_path='classes/model/tokenizer.pkl',max_len_context_path="classes/model/max_len_context.pkl",max_len_question_path="classes/model/max_len_question.pkl"):
        """
        Initializes the QuestionGenerator by loading the model and tokenizer.
        Args:
        - model_path: Path to the saved model.
        - tokenizer_path: Path to the saved tokenizer.
        """
        # Import the custom SparseMax layer
        from tensorflow.keras.layers import Layer
        import tensorflow as tf

        class SparseMax(Layer):
            def call(self, inputs):
                logits = inputs
                z_sorted = tf.sort(logits, direction='DESCENDING', axis=-1)
                z_cumsum = tf.cumsum(z_sorted, axis=-1)
                k = tf.range(1, tf.shape(logits)[-1] + 1, dtype=tf.float32)

                # Compute threshold
                threshold = (z_cumsum - 1) / k
                is_gt_threshold = z_sorted > threshold
                tau = tf.reduce_max(tf.where(is_gt_threshold, threshold, -1e10), axis=-1)

                # Adjust tau shape for broadcasting
                tau = tf.expand_dims(tau, axis=-1)

                # Sparsemax output
                output = tf.maximum(logits - tau, 0)
                return output

        # Load the model with the custom layer
        self.model = load_model(model_path, custom_objects={'SparseMax': SparseMax})

        # Load the tokenizer
        import pickle
        with open(tokenizer_path, 'rb') as f:
            self.tokenizer = pickle.load(f)

        # Load the tokenizer
        with open(max_len_context_path, 'rb') as f:
            self.max_len_context = pickle.load(f)

        # Load the tokenizer
        with open(max_len_question_path, 'rb') as f:
            self.max_len_question = pickle.load(f)
        # Retrieve special tokens
        self.start_token = "starttag"
        self.end_token = "endtag"

    def generate_question(self, context, max_len_question=50):
        """
        Generates a question based on the given context using the trained model.
        Args:
        - context: Input context as a string.
        - max_len_question: Maximum length for the generated question.
        Returns:
        - Generated question as a string.
        """
        # Preprocess context
        from tensorflow.keras.preprocessing.sequence import pad_sequences

        context_seq = self.tokenizer.texts_to_sequences([context])
        context_seq = pad_sequences(context_seq, maxlen=self.model.input_shape[0][1], padding='post')

        # Initialize decoder input with the start token
        decoder_input_seq = np.zeros((1, self.max_len_question))
        decoder_input_seq[0, 0] = self.tokenizer.word_index[self.start_token]

        # Initialize the generated question
        generated_question = []

        for t in range(1, self.max_len_question):
            # Predict next token
            predictions = self.model.predict([context_seq, decoder_input_seq])
            next_token = np.argmax(predictions[0, t - 1])

            # Append the token to the generated question
            if next_token == self.tokenizer.word_index[self.end_token]:
                break  # Stop if the end token is predicted
            generated_question.append(next_token)

            # Update the decoder input for the next step
            decoder_input_seq[0, t] = next_token

        # Convert token IDs back to words
        question = ' '.join([self.tokenizer.index_word[token] for token in generated_question if token in self.tokenizer.index_word])
        return question
