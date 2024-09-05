# Explanation
# Embeddings Visualization:

# embeddings.T: Transpose to make each row represent a different word and each column an embedding dimension.
# Aspect Ratio: aspect='auto' ensures that the plot is not distorted.
# Conv1D Output Visualization:

# conv1d_output.squeeze(axis=2): Remove the singleton dimension to plot 2D data where rows represent sequences and columns represent feature maps.
# Aspect Ratio: Again, aspect='auto' ensures proper scaling.
# Key Points
# Embeddings: Each word in your text is represented by a dense vector. The plot will show how each word is transformed into its embedding vector.
# Conv1D Output: The plot will show the transformed features of your sequences after applying the convolution operation. It displays how different features are extracted from the input text sequences.
# This approach should give you clear and informative visualizations of your text data before and after applying the Conv1D layer. If you encounter any issues with the plots or need further customization, feel free to ask!

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, MaxPooling1D, GlobalMaxPooling1D
from tensorflow.keras import backend as K

# Sample data
texts = ["I love machine learning", "Deep learning is amazing", "I enjoy natural language processing"]
labels = [1, 1, 1]  # Dummy labels

# Tokenization and padding
tokenizer = Tokenizer(num_words=1000)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
data = pad_sequences(sequences, maxlen=10)

# Define model
model = Sequential([
    Embedding(input_dim=1000, output_dim=50, input_length=10),  # Embedding layer
    Conv1D(filters=1, kernel_size=3, activation='relu', padding='valid')  # Conv1D layer
])

# Build the model
model.build((None, 10))  # Initialize the model

# Extract the embeddings and Conv1D layer outputs
embedding_layer = model.layers[0]
conv1d_layer = model.layers[1]
embeddings = embedding_layer.get_weights()[0]  # Shape: (vocab_size, embedding_dim)
conv1d_output = model.predict(data)  # Output shape: (batch_size, new_sequence_length, filters)

# Convert Conv1D output to numpy
conv1d_output = K.eval(K.constant(conv1d_output))

# Visualization
plt.figure(figsize=(16, 8))

# Plot embeddings
plt.subplot(1, 2, 1)
plt.imshow(embeddings.T, aspect='auto', cmap='viridis')
plt.title('Word Embeddings')
plt.colorbar()
plt.xlabel('Embedding Dimension')
plt.ylabel('Word Index')

# Plot Conv1D output
plt.subplot(1, 2, 2)
plt.imshow(conv1d_output.squeeze(axis=2), aspect='auto', cmap='viridis')  # Remove the channel dimension
plt.title('Conv1D Output')
plt.colorbar()
plt.xlabel('Feature Map Index')
plt.ylabel('Sequence Index')

plt.tight_layout()
plt.show()
