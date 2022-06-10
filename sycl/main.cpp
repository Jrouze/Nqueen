// -----------------------------------------------------------------------------
//    Nqueens problem |  OneAPI version using nd_range and buffers
// -----------------------------------------------------------------------------


#include <stdio.h>
#include <stdlib.h>
//#include "common.h"
#include <CL/sycl.hpp>

using namespace std;
using namespace cl::sycl;

#define _EMPTY_      -1

#define sycl_read access::mode::read
#define sycl_discard_write access::mode::write
#define dpc_rw access::mode::read_write

typedef struct queen_root{
  unsigned int control;
  int8_t board[12];
} QueenRoot;


inline void prefixesHandleSol(QueenRoot *root_prefixes, unsigned int flag,
                              const char *board, int initialDepth, int num_sol){
  root_prefixes[num_sol].control = flag;
  for(int i = 0; i<initialDepth;++i)
    root_prefixes[num_sol].board[i] = board[i];
}

inline bool MCstillLegal(const char *board, const int r){
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

bool queens_stillLegal(const char *board, const int r){
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


void BP_queens_root_dfs(nd_item<1> &item, int N, unsigned int nPreFixos, int depthPreFixos,
  const QueenRoot *__restrict root_prefixes, unsigned long long *__restrict vector_of_tree_size,
  unsigned long long *__restrict sols) {
  int idx = item.get_global_id(0);
  if (idx < nPreFixos) {
     unsigned int flag = 0;
     unsigned int bit_test = 0;
     char vertice[20];
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
}//kernel

unsigned long long BP_queens_prefixes(int size, int initialDepth,
                                      unsigned long long *tree_size,
                                      QueenRoot *root_prefixes){
  unsigned int flag = 0;
  int bit_test = 0;
  char vertice[20];
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
             unsigned long long *vector_of_tree_size_h, unsigned long long *sols_h,
             const int Queens_Block_Size, const int repeat, int device){
  int num_blocks = ceil((double)n_explorers/Queens_Block_Size);

  gpu_selector dev_sel_gpu;
  cpu_selector dev_sel_cpu;
  queue Qd(dev_sel_gpu);
  queue Qh(dev_sel_cpu);
  if (device<=1){
    queue q;
    if (device==0) {q=Qd;}
    else {q=Qh;}
    cout << "\nRunning on : "<< q.get_device().get_info<info::device::name>() << "\n";
    buffer<unsigned long long, 1> vector_of_tree_size_d (vector_of_tree_size_h, n_explorers);
    buffer<unsigned long long, 1> sols_d (sols_h, n_explorers);
    buffer<QueenRoot, 1> root_prefixes_d (root_prefixes_h, n_explorers);

    printf("\n### Regular BP-DFS search. ###\n");
    range<1> gws (num_blocks * Queens_Block_Size);
    range<1> lws (Queens_Block_Size);

    for (int i = 0; i < repeat; i++)
      q.submit([&] (handler &cgh) {
        auto root_prefixes = root_prefixes_d.get_access<sycl_read>(cgh);
        auto vector_of_tree_size = vector_of_tree_size_d.get_access<sycl_discard_write>(cgh);
        auto sols =  sols_d.get_access<sycl_discard_write>(cgh);
        cgh.parallel_for(nd_range<1>(gws, lws), [=] (nd_item<1> item) {
          BP_queens_root_dfs(item, size, n_explorers, initial_depth, root_prefixes.get_pointer(),
                             vector_of_tree_size.get_pointer(), sols.get_pointer());
       });
     });
  }
  else{
    cout << "\nRunning on : "<< Qd.get_device().get_info<info::device::name>() << "\n";
    cout << "Running on : "<< Qh.get_device().get_info<info::device::name>() << "\n";
    int cpupart = n_explorers/5;
    //cpupart-=cpupart%16;
    int gpupart = n_explorers-cpupart;

    buffer<unsigned long long, 1> vector_of_tree_size_D (vector_of_tree_size_h, gpupart);
    buffer<unsigned long long, 1> sols_D (sols_h, gpupart);
    buffer<QueenRoot, 1> root_prefixes_D (root_prefixes_h, gpupart);

    buffer<unsigned long long, 1> vector_of_tree_size_H (vector_of_tree_size_h+gpupart, cpupart);
    buffer<unsigned long long, 1> sols_H (sols_h+gpupart, cpupart);
    buffer<QueenRoot, 1> root_prefixes_H (root_prefixes_h+gpupart, cpupart);
    printf("\n### Regular BP-DFS search. ###\n");
    //range<1> gws_D (4*num_blocks/5 * Queens_Block_Size);
    //range<1> gws_H (num_blocks/5 * Queens_Block_Size);
    range<1> gws_D (ceil((double)gpupart/Queens_Block_Size) * Queens_Block_Size);
    range<1> gws_H (ceil((double)cpupart/Queens_Block_Size) * Queens_Block_Size);

    range<1> lws (Queens_Block_Size);

    for (int i = 0; i < repeat; i++){
      Qd.submit([&] (handler &cgh) {
        auto root_prefixes = root_prefixes_D.get_access<sycl_read>(cgh);
        auto vector_of_tree_size = vector_of_tree_size_D.get_access<sycl_discard_write>(cgh);
        auto sols =  sols_D.get_access<sycl_discard_write>(cgh);
        cgh.parallel_for(nd_range<1>(gws_D, lws), [=] (nd_item<1> item) {
          BP_queens_root_dfs(item, size, n_explorers, initial_depth, root_prefixes.get_pointer(),
                             vector_of_tree_size.get_pointer(), sols.get_pointer());
        });
      });
      Qh.submit([&] (handler &cgh) {
        auto root_prefixes = root_prefixes_H.get_access<sycl_read>(cgh);
        auto vector_of_tree_size = vector_of_tree_size_H.get_access<sycl_discard_write>(cgh);
        auto sols =  sols_H.get_access<sycl_discard_write>(cgh);
        cgh.parallel_for(nd_range<1>(gws_H, lws), [=] (nd_item<1> item) {
          BP_queens_root_dfs(item, size, n_explorers, initial_depth, root_prefixes.get_pointer(),
                             vector_of_tree_size.get_pointer(), sols.get_pointer());
        });
      });
    }
    Qd.wait_and_throw();
    Qh.wait_and_throw();
  }
}


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

  //initial search, getting the tree root nodes for the gpu;
  unsigned long long n_explorers = BP_queens_prefixes(size, initialDepth, &tree_size, root_prefixes_h);

  //calling the gpu-based search
  nqueens(size, initialDepth, n_explorers, root_prefixes_h, vector_of_tree_size_h,
     solutions_h, Queens_Block_Size, repeat, device);

  printf("\nTree size: %llu", tree_size );

  for(unsigned long long i = 0; i<n_explorers;++i){
    if(solutions_h[i]>0)
      qtd_sols_global += solutions_h[i];
    if(vector_of_tree_size_h[i]>0)
      tree_size +=vector_of_tree_size_h[i];
  }

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

  free(root_prefixes_h);
  free(vector_of_tree_size_h);
  free(solutions_h);
  return 0;
}
