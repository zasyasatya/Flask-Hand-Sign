import tensorflow.keras

modelPath = 'D:/citra/handSign/Model/keras_model.h5'
labelPath = 'D:/citra/handSign/Model/labels.txt'

model = tensorflow.keras.models.load_model(modelPath)

model.compile(loss = 'categorical_crossentropy', optimizer = RMSprop(learning_rate = 0.001), metrics = ['accuracy'])

loss, acc = model.evaluate()
print("Restored model, accuracy: {:5.2f}%".format(100 * acc))