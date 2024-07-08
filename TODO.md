## List of things to do to make the code more clean ##
1. Replace `DATA_TYPE` with a template parameter.
2. The Kmeans constructor should not accept a points struct, that should be 
   a parameter of `Kmeans::run()`.
3. Create a macro to make timing with `cudaEvent_t` less clunky.
4. Split the gemm and spmm distances strategies into two different member functions.
   They are different enough now to justify two different functions.
5. Make `Kmeans::init_centroids_rand()` work with a CSR `F` matrix.
6. Comprehensive integration tests that run all variants of K-means to convergence on different datasets and compare against RAFT.
