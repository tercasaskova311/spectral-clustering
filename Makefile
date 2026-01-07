CC = mpicc
CFLAGS = -std=c99 -Wall -Wextra
LDFLAGS = -llapack -lblas -lm

SRC = main.c laplacian.c eigensolver.c kmeans.c
OBJ = $(SRC:.c=.o)

TARGET = spectral_mpi

all: $(TARGET)

$(TARGET): $(OBJ)
	$(CC) $(CFLAGS) -o $@ $(OBJ) $(LDFLAGS)

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(OBJ) $(TARGET)

.PHONY: all clean