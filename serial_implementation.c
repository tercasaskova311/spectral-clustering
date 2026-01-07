//pseudo code for serial implementation of spectral clustering - more just notes... for what we actually need to implement
int n;    // number of points
int k;    // number of clusters

double S[n][n];   // similarity matrix
double D[n];   // degree matrix (diagonal)
double L[n][n];   // Laplacian
double U[n][k];   // first k eigenvectors
double Y[n][k];   // row-embedded points (same as U)
int labels[n];    // cluster assignment
double eigenvalues[n];
double eigenvectors[n][n];


// INPUT: data points - assumed already assigned similarity values in S for the simplicity ....?// OUTPUT: cluster labels for each point
build_similarity_matrix(S);

//build degree matrix based on similarity matrix - sum of rows
build_degree_matrix(D, S);

//build Laplacian matrix L = D - S
build_laplacian(L, D, S);

//compute eigenvalues and eigenvectors of L - simply compute first n eigenvalues from L and then eigenvectors out of it
//Take the eigenvectors corresponding to the k smallest non-zero eigenvalues
eig_decompose(L, eigenvalues, eigenvectors);

//select k smallest eigenvectors → U = this step will fill U with the first k eigenvectors
select_k_smallest_eigenvectors(eigenvectors, eigenvalues, U, k);

kmeans(U, labels);

return labels;
