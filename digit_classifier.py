from tensorflow.keras.layers import Dense,Input
from tensorflow.keras.models import Model
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.optimizers import Adam,SGD
from tensorflow.keras.metrics import SparseCategoricalAccuracy
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import load_model

class DigitClassifier:
    def __init__(self):
        self.model = None

    def build_model(self):
        input_layer = Input(shape=784)
        densed_layer_1 = Dense(512,activation='relu')(input_layer)
        densed_layer_2 = Dense(256,activation='relu')(densed_layer_1)
        densed_layer_3 = Dense(128, activation='relu')(densed_layer_2)
        densed_layer_4 = Dense(64,activation='relu')(densed_layer_3)
        densed_layer_5 = Dense(32,activation = 'relu')(densed_layer_4)
        densed_layer_6 = Dense(16,activation='relu')(densed_layer_5)
        output_layer = Dense(10, activation='softmax')(densed_layer_6)
        self.model = Model(input_layer,output_layer)
        print(self.model.summary())

        loss = SparseCategoricalCrossentropy()
        optimizer = Adam(learning_rate=1e-3)
        self.model.compile(loss = loss, optimizer = optimizer, metrics = [SparseCategoricalAccuracy()])

    def load_model(self):
        self.model = load_model('models/trung.hdf5')

    def save_model(self):
        self.model.save('models/trung.hdf5')

    def train(self, x, y, **kwargs):
        es = EarlyStopping(patience=5)
        self.model.fit(x,y,validation_split=0.2, batch_size = 32, epochs=50, callbacks=[es])

    def predict(self, x_test):
        """
        :param x_test: a numpy array with dimension (N,D)
        :return: a numpy array with dimension (N,)
        """
        # return 2 * np.ones(x_test.shape[0])  # delete this line and replace yours
        y_predict = self.model.predict(x_test,batch_size = 32)
        return y_predict.argmax(axis=1)
