BIN = test
CC = g++
SRC = main.cpp ../lib/src/NeuralNetwork.cpp ../lib/src/Dataset.cpp

all: $(BIN)
$(BIN): $(SRC)
	$(CC) -o $(BIN) $(SRC)
run: $(BIN)
	./$(BIN)