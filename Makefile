CC = mpicc
CFLAGS = -std=c99 -Wall -Wextra
# For benchmarking, disable metrics with:
# make CFLAGS="-std=c99 -Wall -Wextra -DENABLE_METRICS=0"
LDFLAGS = -llapack -lblas -lm

SRC = main.c laplacian.c eigensolver.c kmeans.c metrics.c compute_similarity.c read_matrix_size.c
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