"""
Deep Learning Code Templates

These are interview-friendly templates. They show the standard patterns:
- preprocessing
- model definition
- compile/train/evaluate
- callbacks
- saving models

Most examples use TensorFlow/Keras because the course list you shared uses Keras.
Install if needed:
    pip install tensorflow scikit-learn pandas matplotlib
"""

# ============================================================
# 1. Tabular Binary Classification with ANN
# ============================================================

def tabular_binary_classification_template():
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import classification_report, confusion_matrix
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers

    # Example CSV format:
    # columns = feature1, feature2, ..., target
    df = pd.read_csv("data.csv")

    X = df.drop(columns=["target"])
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = keras.Sequential([
        layers.Input(shape=(X_train_scaled.shape[1],)),
        layers.Dense(128, activation="relu", kernel_initializer="he_normal"),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(64, activation="relu", kernel_initializer="he_normal"),
        layers.Dropout(0.2),
        layers.Dense(1, activation="sigmoid")
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=["accuracy", keras.metrics.AUC(name="auc")]
    )

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True
        )
    ]

    history = model.fit(
        X_train_scaled,
        y_train,
        validation_split=0.2,
        epochs=100,
        batch_size=32,
        callbacks=callbacks,
        verbose=1
    )

    test_loss, test_acc, test_auc = model.evaluate(X_test_scaled, y_test)
    print("Test loss:", test_loss)
    print("Test accuracy:", test_acc)
    print("Test AUC:", test_auc)

    y_prob = model.predict(X_test_scaled).ravel()
    y_pred = (y_prob >= 0.5).astype(int)

    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    model.save("ann_binary_model.keras")
    return model, history


# ============================================================
# 2. MNIST Digit Classification with ANN
# ============================================================

def mnist_ann_template():
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers

    (X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()

    # Images are 28x28 pixels. Scale to 0-1.
    X_train = X_train.astype("float32") / 255.0
    X_test = X_test.astype("float32") / 255.0

    model = keras.Sequential([
        layers.Input(shape=(28, 28)),
        layers.Flatten(),
        layers.Dense(256, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(128, activation="relu"),
        layers.Dense(10, activation="softmax")
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    model.fit(
        X_train,
        y_train,
        validation_split=0.1,
        epochs=10,
        batch_size=64
    )

    print(model.evaluate(X_test, y_test))
    return model


# ============================================================
# 3. Image Classification with CNN
# ============================================================

def image_cnn_template():
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers

    train_dir = "dataset/train"
    val_dir = "dataset/val"
    image_size = (224, 224)
    batch_size = 32

    train_ds = keras.utils.image_dataset_from_directory(
        train_dir,
        image_size=image_size,
        batch_size=batch_size
    )

    val_ds = keras.utils.image_dataset_from_directory(
        val_dir,
        image_size=image_size,
        batch_size=batch_size
    )

    num_classes = len(train_ds.class_names)

    data_augmentation = keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1)
    ])

    model = keras.Sequential([
        layers.Input(shape=(224, 224, 3)),
        data_augmentation,
        layers.Rescaling(1.0 / 255),

        layers.Conv2D(32, 3, padding="same", activation="relu"),
        layers.MaxPooling2D(),

        layers.Conv2D(64, 3, padding="same", activation="relu"),
        layers.MaxPooling2D(),

        layers.Conv2D(128, 3, padding="same", activation="relu"),
        layers.MaxPooling2D(),

        layers.Conv2D(256, 3, padding="same", activation="relu"),
        layers.GlobalAveragePooling2D(),

        layers.Dropout(0.4),
        layers.Dense(num_classes, activation="softmax")
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True
        )
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=30,
        callbacks=callbacks
    )

    model.save("cnn_image_classifier.keras")
    return model, history


# ============================================================
# 4. Transfer Learning with MobileNetV2
# ============================================================

def transfer_learning_template():
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers

    train_dir = "dataset/train"
    val_dir = "dataset/val"
    image_size = (224, 224)
    batch_size = 32

    train_ds = keras.utils.image_dataset_from_directory(
        train_dir,
        image_size=image_size,
        batch_size=batch_size
    )

    val_ds = keras.utils.image_dataset_from_directory(
        val_dir,
        image_size=image_size,
        batch_size=batch_size
    )

    num_classes = len(train_ds.class_names)

    base_model = keras.applications.MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights="imagenet"
    )
    base_model.trainable = False

    inputs = keras.Input(shape=(224, 224, 3))
    x = layers.RandomFlip("horizontal")(inputs)
    x = layers.RandomRotation(0.1)(x)
    x = keras.applications.mobilenet_v2.preprocess_input(x)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = keras.Model(inputs, outputs)

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    model.fit(train_ds, validation_data=val_ds, epochs=10)

    # Fine tuning: unfreeze top layers with a small learning rate.
    base_model.trainable = True
    for layer in base_model.layers[:-30]:
        layer.trainable = False

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-5),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    model.fit(train_ds, validation_data=val_ds, epochs=5)
    model.save("transfer_learning_model.keras")
    return model


# ============================================================
# 5. Text Classification with LSTM
# ============================================================

def lstm_text_classification_template():
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers

    texts = [
        "this movie is great",
        "this film is bad",
        "i loved the acting",
        "i hated the story"
    ]
    labels = [1, 0, 1, 0]

    max_tokens = 10000
    max_len = 100

    vectorizer = layers.TextVectorization(
        max_tokens=max_tokens,
        output_mode="int",
        output_sequence_length=max_len
    )
    vectorizer.adapt(texts)

    text_input = keras.Input(shape=(1,), dtype=tf.string)
    x = vectorizer(text_input)
    x = layers.Embedding(input_dim=max_tokens, output_dim=128)(x)
    x = layers.Bidirectional(layers.LSTM(64))(x)
    x = layers.Dropout(0.3)(x)
    output = layers.Dense(1, activation="sigmoid")(x)

    model = keras.Model(text_input, output)

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    model.fit(
        tf.constant(texts),
        tf.constant(labels),
        epochs=5,
        batch_size=2
    )

    prediction = model.predict(tf.constant(["the movie was wonderful"]))
    print(prediction)
    return model


# ============================================================
# 6. Scaled Dot-Product Attention from Scratch
# ============================================================

def scaled_dot_product_attention(q, k, v, mask=None):
    import tensorflow as tf

    scores = tf.matmul(q, k, transpose_b=True)
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scores = scores / tf.math.sqrt(dk)

    if mask is not None:
        scores = scores + (mask * -1e9)

    weights = tf.nn.softmax(scores, axis=-1)
    output = tf.matmul(weights, v)
    return output, weights


# ============================================================
# 7. Simple Transformer Encoder Block in Keras
# ============================================================

class TransformerEncoderBlock:
    """
    This class returns a Keras layer class when build_layer is called.
    It is written this way so the file can be skimmed easily for interviews.
    """

    @staticmethod
    def build_layer(embed_dim, num_heads, ff_dim, dropout_rate=0.1):
        import tensorflow as tf
        from tensorflow import keras
        from tensorflow.keras import layers

        class EncoderLayer(layers.Layer):
            def __init__(self):
                super().__init__()
                self.att = layers.MultiHeadAttention(
                    num_heads=num_heads,
                    key_dim=embed_dim // num_heads
                )
                self.ffn = keras.Sequential([
                    layers.Dense(ff_dim, activation="relu"),
                    layers.Dense(embed_dim)
                ])
                self.norm1 = layers.LayerNormalization(epsilon=1e-6)
                self.norm2 = layers.LayerNormalization(epsilon=1e-6)
                self.dropout1 = layers.Dropout(dropout_rate)
                self.dropout2 = layers.Dropout(dropout_rate)

            def call(self, inputs, training=False):
                attn_output = self.att(inputs, inputs)
                attn_output = self.dropout1(attn_output, training=training)
                out1 = self.norm1(inputs + attn_output)

                ffn_output = self.ffn(out1)
                ffn_output = self.dropout2(ffn_output, training=training)
                return self.norm2(out1 + ffn_output)

        return EncoderLayer()


# ============================================================
# 8. PyTorch Minimal Training Loop
# ============================================================

def pytorch_minimal_training_loop_template():
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    X = torch.randn(1000, 20)
    y = (torch.rand(1000) > 0.5).float().view(-1, 1)

    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    class MLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(20, 64),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, 1)
            )

        def forward(self, x):
            return self.net(x)

    model = MLP()
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(10):
        model.train()
        total_loss = 0.0

        for xb, yb in loader:
            logits = model(xb)
            loss = criterion(logits, yb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch + 1}, loss = {total_loss / len(loader):.4f}")

    torch.save(model.state_dict(), "pytorch_mlp.pt")
    return model


# ============================================================
# 9. Choosing Final Layer and Loss
# ============================================================

OUTPUT_LAYER_AND_LOSS = {
    "binary_classification": {
        "final_layer": 'Dense(1, activation="sigmoid")',
        "loss": "binary_crossentropy"
    },
    "multi_class_integer_labels": {
        "final_layer": 'Dense(num_classes, activation="softmax")',
        "loss": "sparse_categorical_crossentropy"
    },
    "multi_class_one_hot_labels": {
        "final_layer": 'Dense(num_classes, activation="softmax")',
        "loss": "categorical_crossentropy"
    },
    "multi_label_classification": {
        "final_layer": 'Dense(num_labels, activation="sigmoid")',
        "loss": "binary_crossentropy"
    },
    "regression": {
        "final_layer": "Dense(1)",
        "loss": "mse / mae / huber"
    }
}


if __name__ == "__main__":
    print("This file contains templates. Open it and copy the function you need.")
    for task, config in OUTPUT_LAYER_AND_LOSS.items():
        print(task, "->", config)
