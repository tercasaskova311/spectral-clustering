#ifndef READ_MATRIX_SIZE_H
#define READ_MATRIX_SIZE_H

/* Detect the size of a square matrix stored as raw values in a text file. */
int get_square_matrix_size(const char *filename);

/* Detect the size (n, cols) of a feature matrix stored as rows of values. */
int get_feature_matrix_size(const char *filename, int *cols);

#endif
