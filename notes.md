I did few changes:

### 1. **compute_similarity.c / compute_similarity.h** (NEW)
- Handles similarity matrix computation for non-square (feature) matrices.

**Key Function**:
void compute_similarity_matrix(double *X, double *S, int n, int m, double sigma)
- Computes Gaussian (RBF) similarity: `S[i,j] = exp(-||x_i - x_j||² / (2σ²))`
- Converts n×m feature matrix to n×n similarity matrix
- Uses sigma parameter for kernel bandwidth

## Enables processing of both:
- Pre-computed similarity matrices (square)
- Raw feature matrices (non-square) that need similarity computation

---

### 2.**read_matrix_size.c / read_matrix_size.h** (NEW)
**Purpose**: Separates matrix size detection logic from main code for clarity.

**Features**:
- `get_square_matrix_size()`: Counts total values and checks if √count is integer
- `get_feature_matrix_size()`: Counts rows and columns for rectangular matrices
- Returns -1 for square matrix detection if not square
- Improved error handling

- jsut to make main.c cleaner and more readable
---

### 3. **metrics.c / metrics.h** (NEW)
- clustering  evaluation.

**Func**:
```c
double cluster_similarity_score(double *S, int *labels, int n)
```

**Computation**:
- Calculates average intra-cluster similarity (points in same cluster)
- Calculates average inter-cluster similarity (points in different clusters)
- Returns ratio: intra/inter (higher = better clustering)

---


### Bug Missing Parameter Broadcasts
```
Message truncated; 200 bytes received but buffer size is 1
```

**Fix**: Added broadcasts for all parameters:
```c
MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
MPI_Bcast(&cols, 1, MPI_INT, 0, MPI_COMM_WORLD);
MPI_Bcast(&is_feature, 1, MPI_INT, 0, MPI_COMM_WORLD);
MPI_Bcast(&k, 1, MPI_INT, 0, MPI_COMM_WORLD);           // ADDED
MPI_Bcast(&clusters, 1, MPI_INT, 0, MPI_COMM_WORLD);    // ADDED
```
---

### Bug #4: Premature Memory Deallocation
**Location**: main.c

**Issue**: `S` (similarity matrix) was freed after Laplacian computation but needed later for `cluster_similarity_score()`.

**Fix**: Moved memory cleanup to end, after all computations:
```c
// Compute cluster score (needs S)
if(rank == 0){
    double score = cluster_similarity_score(S, labels, n);
    // ... print results ...
}

// NOW free everything
free(S);
free(degree);
if (rank == 0) free(L);  // Only rank 0 still has L
free(U); 
free(labels);
```

---

### Bug #5: Memory Leak on Non-Root Ranks
**Location**: main.c final cleanup

**Issue**: `L` was freed on non-root ranks earlier (line 114) but cleanup tried to free it again unconditionally.

**Fix**: Conditional free:
```c
if (rank == 0) free(L);  // Only rank 0 still has L allocated
```

---

## File Structure
```
spectral-clustering/
├── main.c                    # Main driver (cleaned up)
├── laplacian.c/h            # Matrix loading, degree, Laplacian
├── eigensolver.c/h          # LAPACK eigendecomposition
├── kmeans.c/h               # Parallel k-means
├── compute_similarity.c/h   # NEW: RBF kernel computation
├── read_matrix_size.c/h     # NEW: Input format detection
├── metrics.c/h              # NEW: Clustering quality metrics
├── Makefile
├── run_spectral.sh          # PBS job script
└── data/ans_batch/          # Test datasets
```

---
