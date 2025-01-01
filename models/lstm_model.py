class LSTMModel:
    def __init__(self, input_shape, num_classes):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = None

    def build_model(self):
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, Dense, Dropout

        self.model = Sequential()
        self.model.add(LSTM(128, input_shape=self.input_shape, return_sequences=True))
        self.model.add(Dropout(0.5))
        self.model.add(LSTM(64))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(self.num_classes, activation='softmax'))

        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    def train(self, X_train, y_train, batch_size, epochs, validation_data):
        history = self.model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=validation_data)
        return history