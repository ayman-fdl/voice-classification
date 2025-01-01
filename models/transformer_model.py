class TransformerModel:
    def __init__(self, input_shape, num_classes):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = None

    def build_model(self):
        import tensorflow as tf
        from tensorflow.keras import layers

        inputs = layers.Input(shape=self.input_shape)
        x = layers.Conv1D(64, kernel_size=3, padding='same')(inputs)
        x = layers.MultiHeadAttention(num_heads=4, key_dim=64)(x, x)
        x = layers.GlobalAveragePooling1D()(x)
        x = layers.Dense(128, activation='relu')(x)
        outputs = layers.Dense(self.num_classes, activation='softmax')(x)

        self.model = tf.keras.Model(inputs, outputs)
        self.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    def train(self, x_train, y_train, epochs=10, batch_size=32, validation_data=None):
        self.model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=validation_data)