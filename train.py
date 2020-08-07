import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, BatchNormalization, Dropout, Flatten
from keras.optimizers import SGD

from keras.models import load_model
import h5py

from getData import readAllGames

if __name__ == '__main__':
	# Get x_train and y_train
	x_train, y_train = readAllGames("data/data.pgn", "data/stockfish.csv", max_boards=3000000, verbose=True)


	# Define MLP model
	model = Sequential()

	# Layer 1
	model.add(Dense(2048, input_shape=(64, 6)))
	model.add(Activation('elu'))
	model.add(BatchNormalization())
	model.add(Dropout(0.2))

	# Layer 2
	model.add(Dense(2048))
	model.add(Activation('elu'))
	model.add(BatchNormalization())
	model.add(Dropout(0.2))

	# Layer 3
	model.add(Dense(2048))
	model.add(Activation('elu'))
	model.add(BatchNormalization())
	model.add(Dropout(0.2))

	# Output layer
	model.add(Dense(1))

	# Compile model
	opt = SGD(lr=0.001, momentum=0.7, nesterov=True)

	model.compile(loss='mse', optimizer=opt, metrics=['mse'])

	model.summary()	

	# Train model
	model.fit(x_train, y_train, epochs=5, batch_size=256, validation_split=0.2)

	# Save model
	model.save('models/ChessAI_MLP.h5')