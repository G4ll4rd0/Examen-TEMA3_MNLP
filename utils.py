import numpy as np
import seaborn as sns
from tensorflow.keras import Model
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from tensorflow.keras.layers import Input, Dense, Flatten, Conv1D, MaxPooling1D, Reshape, LSTM, TimeDistributed
from tensorflow.keras.callbacks import EarlyStopping

class TimeSeriesNetworks:
    #Initializer
    def __init__(self, X, y) -> None:
        self.X = X
        self.y = y
        self.models = {'cnn': {}, 'lstm': {}, 'cnn-lstm': {}}
        self.history = {'cnn': {}, 'lstm': {}, 'cnn-lstm': {}}
        self.maxAccuracy = {'cnn': {}, 'lstm': {}, 'cnn-lstm': {}}

    #Builds a Dense Block
    def denseBlock(self, x):
        for i in range(round(np.random.random()*10)):
            x = Dense(units = 100/(i+2), activation = 'relu')(x)
        return x
    
    #Builds a ConvBlock
    def convBlock(self, x):
        for i in range(round(np.random.random()*5)):
            x = Conv1D(128/(i+2), 1, activation='relu')(x)
        return x

    #Builds a LSTM Block
    def lstmBlock(self, x):
        for i in range(round(np.random.random()*10)):
            x = LSTM(50, activation='relu', return_sequences=True)(x)
        return x
    
    #Builds a TimeDistributed CNN Block
    def timeBlock(self, x):
        for i in range(round(np.random.random()*5)):
            x = TimeDistributed(Conv1D(64, 1, activation='relu'))(x)
        return x

    #Build CNN
    def cnn(self) -> Model:
        input_tensor = Input(shape = self.input)
        x = Flatten()(input_tensor)
        if len(self.input) == 1:
            x = Reshape(self.input + (1,))(x)
        else:
            x = Reshape(self.input)(x)
        x = Conv1D(128, 2, activation = 'relu')(x)
        x = self.convBlock(x)
        x = MaxPooling1D()(x)
        x = Flatten()(x)
        x = Dense(100, activation = 'relu')(x)
        x = self.denseBlock(x)
        output_tensor = Dense(4, activation = 'softmax')(x)
        model = Model(inputs = input_tensor, outputs = output_tensor)
        return model

    #Build LSTM
    def lstm(self) -> Model:
        input_tensor = Input(shape = self.input)
        x = Flatten()(input_tensor)
        if len(self.input) == 1:
            x = Reshape(self.input + (1,))(x)
        else:
            x = Reshape(self.input)(x)
        x = self.lstmBlock(x)
        x = LSTM(50, activation='relu')(x)
        output_tensor = Dense(4, activation = 'softmax')(x)
        model = Model(inputs = input_tensor, outputs = output_tensor)
        return model

    #Build CNN-LSTM
    def cnnLstm(self) -> Model:
        input_tensor = Input(shape = self.input)
        x = Flatten()(input_tensor)
        if len(self.input) < 3:
            x = Reshape((2, 41, 223))(x)
        else:
            x = Reshape(self.input)(x)
        x = TimeDistributed(Conv1D(64, 1, activation='relu'))(x)
        x = self.timeBlock(x)
        x = TimeDistributed(MaxPooling1D(pool_size=2))(x)
        x = TimeDistributed(Flatten())(x)
        x = self.lstmBlock(x)
        x = LSTM(50, activation='relu')(x)
        output_tensor = Dense(4, activation = 'softmax')(x)
        model = Model(inputs = input_tensor, outputs = output_tensor)
        return model

    #Build Models
    def build(self):
        self.input = self.X.shape[1:]
        for i in range(3):
            self.models['cnn'][i]       = self.cnn()
            self.models['lstm'][i]      = self.lstm()
            self.models['cnn-lstm'][i]  = self.cnnLstm()

    #Train Models
    def train(self):
        es = EarlyStopping(monitor = 'accuracy', patience = 10, restore_best_weights = True)
        
        #CNN
        for k, i in self.models['cnn'].items():
            i.compile(optimizer='adam', loss=['sparse_categorical_crossentropy'], metrics=['accuracy'])
            self.history['cnn'][k] = i.fit(self.X, self.y, epochs=1000, verbose = False, callbacks=[es])
        
        #LSTM
        for k, i in self.models['lstm'].items():
            i.compile(optimizer='RMSprop', loss=['sparse_categorical_crossentropy'], metrics=['accuracy'])
            self.history['lstm'][k] = i.fit(self.X, self.y, epochs=1000, verbose = False, callbacks=[es])
        
        #CNN-LSTM
        for k, i in self.models['cnn-lstm'].items():
            i.compile(optimizer='adam', loss=['sparse_categorical_crossentropy'], metrics=['accuracy'])
            self.history['cnn-lstm'][k] = i.fit(self.X, self.y, epochs=1000, verbose = False, callbacks=[es])
    
    #Provides best results
    def results(self):            
        #CNN
        for k, h in self.history['cnn'].items():
            self.maxAccuracy['cnn'][k] = max(h.history['accuracy'])
            
        #LSTM
        for k, h in self.history['lstm'].items():
            self.maxAccuracy['lstm'][k] = max(h.history['accuracy'])
            
        #CNN-LSTM
        for k, h in self.history['cnn-lstm'].items():
            self.maxAccuracy['cnn-lstm'][k] = max(h.history['accuracy'])

    #Function to execute to do all models
    def model(self):
        self.build()
        self.train()
        self.results()




def plot_confusion_matrix(true, pred):
    cm = confusion_matrix(true, pred, labels=true.unique())
    f, ax = plt.subplots(figsize =(7,7))
    sns.heatmap(cm, annot = True, linewidths=0.2, linecolor="black", fmt = ".0f", ax=ax, cmap="Purples")
    plt.xlabel("PREDICTED LABEL")
    plt.ylabel("TRUE LABEL")
    plt.title('Confusion Matrix for SVM Classifier')
    plt.show()