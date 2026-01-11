#include "read_matrix_size.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

int get_square_matrix_size(const char *filename){
    FILE *f = fopen(filename,"r");
    if(!f){ perror("Failed to open file"); return -1; }

    double tmp;
    long long count = 0;
    while(fscanf(f, "%lf%*[, ]", &tmp) == 1) count++;
    fclose(f);

    int n = (int)(sqrt((double)count));
    if ((long long)n * n != count) return -1; // not square

    return n;
}

int get_feature_matrix_size(const char *filename, int *cols){
    FILE *f = fopen(filename,"r");
    if(!f){ perror("Failed to open file"); return -1; }

    int n=0;
    char line[8192];
    int c = -1;

    while(fgets(line, sizeof(line), f)){
        if(strlen(line) <= 1) continue;
        n++;
        if(c == -1){
            c = 0;
            char *p = line;
            double tmp;
            while(sscanf(p,"%lf%*[, ]",&tmp)==1){
                c++;
                while(*p && *p != ',' && *p != ' ' && *p != '\n') p++;
                while(*p == ',' || *p==' ') p++;
            }
        }
    }

    fclose(f);
    *cols = c;
    return n;
}
