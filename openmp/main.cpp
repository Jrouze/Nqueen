// -----------------------------------------------------------------------------
//    Nqueens problem |  OpenMP version
// -----------------------------------------------------------------------------

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <omp.h>

using namespace std;

//#define _QUEENS_BLOCK_SIZE_   128
#define _EMPTY_      -1



typedef struct queen_root{
  unsigned int control;
  int8_t board[12];
} QueenRoot;

inline void prefixesHandleSol(QueenRoot *root_prefixes, unsigned int flag,
                              const int *board, int initialDepth, int num_sol)
{
  root_prefixes[num_sol].control = flag;
  for(int i = 0; i<initialDepth;++i)
    root_prefixes[num_sol].board[i] = board[i];
}

inline bool MCstillLegal(const int *board, const int r)
{
  // Check vertical
  for (int i = 0; i < r; ++i)
    if (board[i] == board[r]) return false;
  // Check diagonals
  int ld = board[r];  //left diagonal columns
  int rd = board[r];  // right diagonal columns
  for (int i = r-1; i >= 0; --i) {
    --ld; ++rd;
    if (board[i] == ld || board[i] == rd) return false;
  }
  return true;
}

#pragma omp declare target
bool queens_stillLegal(const int *board, const int r)
{
  bool safe = true;
  // Check vertical
  for (int i = 0; i < r; ++i)
    if (board[i] == board[r]) safe = false;
  // Check diagonals
  int ld = board[r];  //left diagonal columns
  int rd = board[r];  // right diagonal columns
  for (int i = r-1; i >= 0; --i) {
    --ld; ++rd;
    if (board[i] == ld || board[i] == rd) safe = false;
  }
  return safe;
}
#pragma omp end declare target


void BP_queens_root_dfs(int N, unsigned int nPreFixos, int depthPreFixos,
  const QueenRoot *__restrict root_prefixes,  unsigned long long *__restrict vector_of_tree_size,
  unsigned long long *__restrict sols, const int Queens_Block_Size, const int device) {
  if (device ==0) // ------------ If work done on GPU ------------
  #pragma omp target teams distribute parallel for thread_limit(Queens_Block_Size)
  for (int idx = 0; idx < nPreFixos; idx++) {
     unsigned int flag = 0;
     unsigned int bit_test = 0;
     int vertice[20];
     int N_l = N;
     int i, depth;
     unsigned long long  qtd_solutions_thread = 0ULL;
     int depthGlobal = depthPreFixos;
     unsigned long long tree_size = 0ULL;

#pragma unroll 2
    for (i = 0; i < N_l; ++i) {
      vertice[i] = _EMPTY_;
    }

    flag = root_prefixes[idx].control;

#pragma unroll 2
    for (i = 0; i < depthGlobal; ++i)
      vertice[i] = root_prefixes[idx].board[i];

    depth = depthGlobal;

    do {
      vertice[depth]++;
      bit_test = 0;
      bit_test |= (1<<vertice[depth]);
      if(vertice[depth] == N_l){
        vertice[depth] = _EMPTY_;
      } else if (!(flag & bit_test ) && queens_stillLegal(vertice, depth)){
        ++tree_size;
        flag |= (1ULL<<vertice[depth]);
        depth++;
        if (depth == N_l) { //sol
          ++qtd_solutions_thread;
        } else continue;
      } else continue;
      depth--;
      flag &= ~(1ULL<<vertice[depth]);
    } while(depth >= depthGlobal);

    sols[idx] = qtd_solutions_thread;
    vector_of_tree_size[idx] = tree_size;
  }//endif
  if (device ==1) // ------------ If work done on CPU ------------
  #pragma omp parallel for
  for (int idx = 0; idx < nPreFixos; idx++) {
     unsigned int flag = 0;
     unsigned int bit_test = 0;
     int vertice[20];
     int N_l = N;
     int i, depth;
     unsigned long long  qtd_solutions_thread = 0ULL;
     int depthGlobal = depthPreFixos;
     unsigned long long tree_size = 0ULL;

#pragma unroll 2
    for (i = 0; i < N_l; ++i) {
      vertice[i] = _EMPTY_;
    }

    flag = root_prefixes[idx].control;

#pragma unroll 2
    for (i = 0; i < depthGlobal; ++i)
      vertice[i] = root_prefixes[idx].board[i];

    depth = depthGlobal;

    do {
      vertice[depth]++;
      bit_test = 0;
      bit_test |= (1<<vertice[depth]);
      if(vertice[depth] == N_l){
        vertice[depth] = _EMPTY_;
      } else if (!(flag & bit_test ) && queens_stillLegal(vertice, depth)){
        ++tree_size;
        flag |= (1ULL<<vertice[depth]);
        depth++;
        if (depth == N_l) { //sol
          ++qtd_solutions_thread;
        } else continue;
      } else continue;
      depth--;
      flag &= ~(1ULL<<vertice[depth]);
    } while(depth >= depthGlobal);

    sols[idx] = qtd_solutions_thread;
    vector_of_tree_size[idx] = tree_size;
  }//endif
}//end kernel

void BP_queens_root_dfs_both(int N, unsigned int gpupart, int depthPreFixos,
  const QueenRoot *__restrict root_prefixes,  unsigned long long *__restrict vector_of_tree_size,
  unsigned long long *__restrict sols, const int Queens_Block_Size, int cpupart) {
   // ------------ If work done on GPU ------------
  #pragma omp target teams distribute parallel for thread_limit(Queens_Block_Size) nowait
  for (int idx = 0; idx < gpupart; idx++) {
     unsigned int flag = 0;
     unsigned int bit_test = 0;
     int vertice[20];
     int N_l = N;
     int i, depth;
     unsigned long long  qtd_solutions_thread = 0ULL;
     int depthGlobal = depthPreFixos;
     unsigned long long tree_size = 0ULL;
#pragma unroll 2
    for (i = 0; i < N_l; ++i) {
      vertice[i] = _EMPTY_;
    }
    flag = root_prefixes[idx].control;
#pragma unroll 2
    for (i = 0; i < depthGlobal; ++i)
      vertice[i] = root_prefixes[idx].board[i];
    depth = depthGlobal;
    do {
      vertice[depth]++;
      bit_test = 0;
      bit_test |= (1<<vertice[depth]);
      if(vertice[depth] == N_l){
        vertice[depth] = _EMPTY_;
      } else if (!(flag & bit_test ) && queens_stillLegal(vertice, depth)){
        ++tree_size;
        flag |= (1ULL<<vertice[depth]);
        depth++;
        if (depth == N_l) { //sol
          ++qtd_solutions_thread;
        } else continue;
      } else continue;
      depth--;
      flag &= ~(1ULL<<vertice[depth]);
    } while(depth >= depthGlobal);
    sols[idx] = qtd_solutions_thread;
    vector_of_tree_size[idx] = tree_size;
  }//endif
  // ------------ If work done on CPU ------------
  #pragma omp parallel for
  for (int idx = cpupart; idx < cpupart+gpupart; idx++) {
     unsigned int flag = 0;
     unsigned int bit_test = 0;
     int vertice[20];
     int N_l = N;
     int i, depth;
     unsigned long long  qtd_solutions_thread = 0ULL;
     int depthGlobal = depthPreFixos;
     unsigned long long tree_size = 0ULL;
#pragma unroll 2
    for (i = 0; i < N_l; ++i) {
      vertice[i] = _EMPTY_;
    }
    flag = root_prefixes[idx].control;
#pragma unroll 2
    for (i = 0; i < depthGlobal; ++i)
      vertice[i] = root_prefixes[idx].board[i];
    depth = depthGlobal;

    do {
      vertice[depth]++;
      bit_test = 0;
      bit_test |= (1<<vertice[depth]);
      if(vertice[depth] == N_l){
        vertice[depth] = _EMPTY_;
      } else if (!(flag & bit_test ) && queens_stillLegal(vertice, depth)){
        ++tree_size;
        flag |= (1ULL<<vertice[depth]);
        depth++;
        if (depth == N_l) { //sol
          ++qtd_solutions_thread;
        } else continue;
      } else continue;
      depth--;
      flag &= ~(1ULL<<vertice[depth]);
    } while(depth >= depthGlobal);

    sols[idx] = qtd_solutions_thread;
    vector_of_tree_size[idx] = tree_size;
  }//endif
}//end kernel

unsigned long long BP_queens_prefixes(int size, int initialDepth,
                                      unsigned long long *tree_size,
                                      QueenRoot *root_prefixes)
{
  unsigned int flag = 0;
  int bit_test = 0;
  int vertice[20];
  int i, nivel;
  unsigned long long local_tree = 0ULL;
  unsigned long long num_sol = 0;

  for (i = 0; i < size; ++i) {
    vertice[i] = -1;
  }

  nivel = 0;

  do{

    vertice[nivel]++;
    bit_test = 0;
    bit_test |= (1<<vertice[nivel]);

    if(vertice[nivel] == size){
      vertice[nivel] = _EMPTY_;
    }else if ( MCstillLegal(vertice, nivel) && !(flag &  bit_test ) ){ //is legal

      flag |= (1ULL<<vertice[nivel]);
      nivel++;
      ++local_tree;
      if (nivel == initialDepth){ //handle solution
        prefixesHandleSol(root_prefixes,flag,vertice,initialDepth,num_sol);
        num_sol++;
      }else continue;
    }else continue;

    nivel--;
    flag &= ~(1ULL<<vertice[nivel]);

  }while(nivel >= 0);

  *tree_size = local_tree;

  return num_sol;
}


void nqueens(short size, int initial_depth, unsigned int n_explorers, QueenRoot *root_prefixes_h ,
             unsigned long long *vector_of_tree_size_h, unsigned long long *sols_h, const int Queens_Block_Size,
             const int repeat, const int device)
{
  printf("\n### Regular BP-DFS search. ###\n");
  int cpupart,gpupart;
  if (device<=1){ // 0 if for GPU only, 1 is for CPU only
    cpupart=device*n_explorers;
    gpupart=(1-device)*n_explorers;
  }
  else{
    cpupart = n_explorers/2;
    gpupart = n_explorers-cpupart;
  }

  if(device==0){ // ------------ If work done on GPU ------------
#pragma omp target data map (to: root_prefixes_h[0:gpupart]) \
                        map (from: vector_of_tree_size_h[0:gpupart], \
                                   sols_h[0:gpupart])
  {
    for (int i = 0; i < repeat; i++)
      BP_queens_root_dfs(size, gpupart, initial_depth, root_prefixes_h, vector_of_tree_size_h,
                         sols_h, Queens_Block_Size, 0);
  }
}
  else if(device==1) // ------------ If work done on CPU ------------
    {
      for (int i = 0; i < repeat; i++)
        BP_queens_root_dfs(size, cpupart, initial_depth, root_prefixes_h, vector_of_tree_size_h,
                           sols_h, Queens_Block_Size, 1);
    }
  else{ // ------------ If work done on both ------------
#pragma omp target data map (to: root_prefixes_h[0:gpupart]) \
                          map (from: vector_of_tree_size_h[0:gpupart], \
                                     sols_h[0:gpupart])
    {
      for (int i = 0; i < repeat; i++)
      BP_queens_root_dfs_both(size, gpupart, initial_depth, root_prefixes_h, vector_of_tree_size_h,
                         sols_h, Queens_Block_Size, cpupart);
    }
  }
}

// -----------------------------------------------------------------------------
//                  Main function
// -----------------------------------------------------------------------------
int main(int argc, char *argv[]){
  short size=15;
  int initialDepth=7;
  int Queens_Block_Size=128;
  int repeat=1;
  int device=2;
  if (argc==6){
    size = atoi(argv[1]);  // 15 - 17 for a short run
    initialDepth = atoi(argv[2]); // 6 or 7
    Queens_Block_Size = atoi(argv[3]); // Block size (local size of nd_range)
    repeat = atoi(argv[4]); // kernel execution times
    device = atoi(argv[5]); // should be 0 for gpu, 1 for cpu, 2 for both
  }
  else {
    cout<<"Wrong number of arguments\nPlease give the following order of arguments\n\n"<<
    "Argument\t\t Information\t\t\t Default values\n"<<
    "Size of the problem\t 15 - 17 for a short run\t\t"<< size <<"\n"
    "InitialDepth\t\t 6 or 7\t\t\t\t\t"<< initialDepth <<"\n"
    "Blocksize\t\t Should be lower than 256\t\t"<< Queens_Block_Size <<"\n"
    "Number of repetition\t 1 for short \t\t\t\t"<< repeat <<"\n"
    "Device choice\t\t 0 for gpu, 1 for cpu, 2 for both \t"<< device <<"\n";

    cout<<"\nUsing default values instead\n";
  }
  printf("\n### Initial depth: %d - Size: %d:", initialDepth, size);

  unsigned long long tree_size = 0ULL;
  unsigned long long qtd_sols_global = 0ULL;
  unsigned int nMaxPrefixos = 75580635;

  QueenRoot* root_prefixes_h = (QueenRoot*)malloc(sizeof(QueenRoot)*nMaxPrefixos);
  unsigned long long *vector_of_tree_size_h = (unsigned long long*)malloc(sizeof(unsigned long long)*nMaxPrefixos);
  unsigned long long *solutions_h = (unsigned long long*)malloc(sizeof(unsigned long long)*nMaxPrefixos);

  if (root_prefixes_h == NULL || vector_of_tree_size_h == NULL || solutions_h == NULL) {
    printf("Error: host out of memory\n");
    if (root_prefixes_h) free(root_prefixes_h);
    if (vector_of_tree_size_h) free(vector_of_tree_size_h);
    if (solutions_h) free(solutions_h);
    return 1;
  }

  struct timespec start, stop;
  double elapsed;
// --------- Beginning of the search -----------
  clock_gettime(CLOCK_MONOTONIC,&start);

  //initial search, getting the tree root nodes for the gpu;
  unsigned long long n_explorers = BP_queens_prefixes(size, initialDepth, &tree_size, root_prefixes_h);

  //calling the gpu-based search
  nqueens(size, initialDepth, n_explorers, root_prefixes_h, vector_of_tree_size_h, solutions_h, Queens_Block_Size, repeat, device);

  for(unsigned long long i = 0; i<n_explorers;++i){
    if(solutions_h[i]>0)
      qtd_sols_global += solutions_h[i];
    if(vector_of_tree_size_h[i]>0)
      tree_size +=vector_of_tree_size_h[i];
  }

  clock_gettime(CLOCK_MONOTONIC,&stop);
  elapsed=(stop.tv_sec-start.tv_sec)+(stop.tv_nsec-start.tv_nsec)/1e9;
// --------- Searching process is done -----------
  printf("\nTree size: %llu", tree_size );
  printf("\nNumber of solutions found: %llu \nTree size: %llu\n", qtd_sols_global, tree_size );

  // Initial depth: 7 - Size: 15:
  // Tree size: 2466109
  // Number of solutions found: 2279184
  // Tree size: 171129071
  if (size == 15 && initialDepth == 7) {
    if (qtd_sols_global == 2279184 && tree_size == 171129071)
      printf("PASS\n");
    else
      printf("FAIL\n");
  }

// --------- Storing time mesurement -----------
  printf("\nElapsed time :\t %2.10f\n", elapsed);
  FILE *f;
  if (device==0) {
    f=fopen("data_omp_gpu.txt","a");}
  else if (device==1){
    f=fopen("data_omp_cpu.txt","a");}
  else{
    f=fopen("data_omp_both.txt","a");}
  fprintf(f,"%d %d %d %f \n", size, initialDepth, Queens_Block_Size, elapsed/repeat);

  free(root_prefixes_h);
  free(vector_of_tree_size_h);
  free(solutions_h);
  return 0;
}
