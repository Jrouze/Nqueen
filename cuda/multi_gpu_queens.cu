#include <cuda.h>
#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <sys/time.h>
#define _QUEENS_BLOCK_SIZE_ 	128
#define _EMPTY_      -1
#define _MAX_DEPTH_ 12
#define _MAX_QUEEN_SIZE_ 32

double rtclock()
{
    struct timezone Tzp;
    struct timeval Tp;
    int stat;
    stat = gettimeofday (&Tp, &Tzp);
    if (stat != 0) printf("Error return from gettimeofday: %d",stat);
    return(Tp.tv_sec + Tp.tv_usec*1.0e-6);
}

//keeps the feasible, valid and incomplete solutions
typedef struct queen_root{
    unsigned int control;
    int8_t board[_MAX_DEPTH_];
} QueenRoot;


#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}


//this is to get the feasible-valid and incomplete solution into the data structure
inline void prefixesHandleSol(QueenRoot *root_prefixes,unsigned int flag,char *board,int initialDepth,int num_sol){

    root_prefixes[num_sol].control = flag;

    for(int i = 0; i<initialDepth;++i)
      root_prefixes[num_sol].board[i] = (char)board[i];
}

//verifies if the configuration is legal -- CPU
inline bool MCstillLegal(const char *board, const int r)
{

    int i;
    int ld;
    int rd;
  // Check vertical
  for ( i = 0; i < r; ++i)
    if (board[i] == board[r]) return false;
    // Check diagonals
    ld = board[r];  //left diagonal columns
    rd = board[r];  // right diagonal columns
    for ( i = r-1; i >= 0; --i) {
      --ld; ++rd;
      if (board[i] == ld || board[i] == rd) return false;
    }

    return true;
}

//verifies if the configuration is legal -- GPU (Not the same as on CPU)
__device__  bool GPU_queens_stillLegal(const char *board, const int r){

  bool safe = true;
  int i;
  register int ld;
  register int rd;
  // Check vertical
  for ( i = 0; i < r; ++i)
    if (board[i] == board[r]) safe = false;
    // Check diagonals
    ld = board[r];  //left diagonal columns
    rd = board[r];  // right diagonal columns
    for ( i = r-1; i >= 0; --i) {
      --ld; ++rd;
      if (board[i] == ld || board[i] == rd) safe = false;
    }

    return safe;
}

//the GPU Kernel
__global__ void BP_queens_root_dfs(int N, unsigned int nPreFixos, int depthPreFixos,
    QueenRoot *root_prefixes,unsigned long long int *vector_of_tree_size, unsigned long long int *sols){

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < nPreFixos) {
        unsigned int flag = 0;
        unsigned int bit_test = 0;
        char board[_MAX_QUEEN_SIZE_]; //keeps the solution
        int N_l = N;
        int i, depth;
        unsigned long long  qtd_solucoes_thread = 0ULL;
        int depthGlobal = depthPreFixos;
        unsigned long long int tree_size = 0ULL;

        for (i = 0; i < N_l; ++i) {
            board[i] = _EMPTY_;
        }

        flag = root_prefixes[idx].control;


        for (i = 0; i < depthGlobal; ++i)
            board[i] = root_prefixes[idx].board[i];

        depth=depthGlobal;

        do{

            board[depth]++;
            bit_test = 0;
            bit_test |= (1<<board[depth]);

            if(board[depth] == N_l){
                board[depth] = _EMPTY_;
                //if(block_ub > upper)   block_ub = upper;
            }else if (!(flag &  bit_test ) && GPU_queens_stillLegal(board, depth)){


                    flag |= (1ULL<<board[depth]);
                    tree_size++;
                    depth++;

                    if (depth == N_l) { //sol
                        ++qtd_solucoes_thread;
                    }else continue;
                }else continue;

            depth--;
            flag &= ~(1ULL<<board[depth]);

            }while(depth >= depthGlobal); //FIM DO DFS_BNB

        sols[idx] = qtd_solucoes_thread;
        vector_of_tree_size[idx] = tree_size;
    }//if
}//kernel
////////


//the partial seach on CPU -- finds feasible, valid and incomplete solutions
unsigned long long int BP_queens_prefixes(int size, int initialDepth ,unsigned long long *tree_size, QueenRoot *root_prefixes){

    unsigned int flag = 0;
    int bit_test = 0;
    char board[_MAX_QUEEN_SIZE_]; //the board wich keeps the solution at hand
    int i, depth; //the initial depth of the search
    unsigned long long int local_tree = 0ULL;
    unsigned long long int num_sol = 0;
   //register int custo = 0;


    /*initialization*/
    for (i = 0; i < size; ++i) { //
        board[i] = -1;
    }

    depth = 0;

    do{

        board[depth]++;
        bit_test = 0;
        bit_test |= (1<<board[depth]);


        if(board[depth] == size){
            board[depth] = _EMPTY_;
                //if(block_ub > upper)   block_ub = upper;
        }else if ( MCstillLegal(board, depth) && !(flag &  bit_test ) ){ //is legal

                flag |= (1ULL<<board[depth]);
                depth++;
                ++local_tree;
                if (depth == initialDepth){ //handle solution
                   prefixesHandleSol(root_prefixes,flag,board,initialDepth,num_sol);
                   num_sol++;
            }else continue;
        }else continue;

        depth--;
        flag &= ~(1ULL<<board[depth]);

    }while(depth >= 0);

    *tree_size = local_tree;

    return num_sol;
}


//CUDA memory manipulation and calling both searches

void GPU_call_cuda_queens(int size, int initial_depth, int block_size, bool set_cache, unsigned int n_explorers, QueenRoot *root_prefixes_h ,
	unsigned long long int *vector_of_tree_size_h, unsigned long long int *sols_h, int nb_streams_per_dev,  int max_devices)
    {
        int nb_devices = 1;
        gpuErrchk( cudaGetDeviceCount(&nb_devices) );
        nb_devices=min(nb_devices,max_devices);
        printf("\n### Number of devices:\t %d\n",nb_devices);

        //set cache config for each device (optional)
        if(set_cache){
            for(int i=0;i<nb_devices;i++){
                printf("\n ### nSeeting up the cache ###\n");
                gpuErrchk( cudaSetDevice(i) );
                gpuErrchk( cudaFuncSetCacheConfig(BP_queens_root_dfs,cudaFuncCachePreferL1) );
            }
        }

        //how many streams per device
        int nb_streams = nb_devices*nb_streams_per_dev;

        //explorers per stream
        unsigned int *nb_explorers_per_stream = (unsigned int*)malloc(nb_streams*sizeof(unsigned int));
        unsigned int *nb_explorers_prefix_sum = (unsigned int*)malloc(nb_streams*sizeof(unsigned int));

        unsigned int q = n_explorers/nb_streams;
        unsigned int r = n_explorers;
        for(int i=0;i<nb_streams;i++){
            nb_explorers_per_stream[i] = q;
            r -= nb_explorers_per_stream[i];
        }
        for(unsigned i=0;i<r;i++){
            nb_explorers_per_stream[i]++;
        }

        nb_explorers_prefix_sum[0]=0;
        for(int i=1;i<nb_streams;i++){
            nb_explorers_prefix_sum[i] = nb_explorers_prefix_sum[i-1] + nb_explorers_per_stream[i-1];
        }

        printf("prefix sum : \n");
        for(int i=0;i<nb_streams;i++){
            printf("%d ",nb_explorers_prefix_sum[i]);
        }
        printf("\n");

        //start parallel exploration on multiple devices / streams
        #pragma omp parallel for
        for(int i=0; i<nb_streams;i++)
        {
            gpuErrchk( cudaSetDevice(i/nb_streams_per_dev) );

            unsigned int n_explorers_local = nb_explorers_per_stream[i];
            unsigned int first_prefix = nb_explorers_prefix_sum[i];

            //one stream per thread
            cudaStream_t stream;
            gpuErrchk( cudaStreamCreate(&stream) );

            unsigned long long int *vector_of_tree_size_d;
            unsigned long long int *sols_d;
            QueenRoot *root_prefixes_d;

            gpuErrchk( cudaMalloc((void**) &vector_of_tree_size_d,n_explorers_local*sizeof(unsigned long long int)) );
            gpuErrchk( cudaMalloc((void**) &sols_d,n_explorers_local*sizeof(unsigned long long int)) );
            gpuErrchk( cudaMalloc((void**) &root_prefixes_d,n_explorers_local*sizeof(QueenRoot)) );


            // printf("hello %d %d %d \n",i,nb_explorers_per_stream[i],first_prefix);

            // gpuErrchk( cudaMemcpy(
            //     root_prefixes_d,
            //     root_prefixes_h+first_prefix,
            //     nb_explorers_per_stream[i] * sizeof(QueenRoot),
            //     cudaMemcpyHostToDevice
            // ) );
            gpuErrchk( cudaMemcpyAsync(
                root_prefixes_d,
                root_prefixes_h+first_prefix,
                n_explorers_local * sizeof(QueenRoot),
                cudaMemcpyHostToDevice,
                stream
            ) );

            gpuErrchk( cudaPeekAtLastError() );
            gpuErrchk( cudaDeviceSynchronize() );

            int num_blocks = ceil((double)n_explorers_local/block_size);

            BP_queens_root_dfs<<< num_blocks,block_size, 0, stream>>> (size,n_explorers_local,initial_depth,root_prefixes_d, vector_of_tree_size_d,sols_d);
            gpuErrchk( cudaPeekAtLastError() );
            gpuErrchk( cudaDeviceSynchronize() );

            gpuErrchk( cudaMemcpyAsync(vector_of_tree_size_h+first_prefix,vector_of_tree_size_d,n_explorers_local*sizeof(unsigned long long int),cudaMemcpyDeviceToHost,stream) );
            gpuErrchk( cudaMemcpyAsync(sols_h+first_prefix,sols_d,n_explorers_local*sizeof(unsigned long long int),cudaMemcpyDeviceToHost,stream) );

            gpuErrchk( cudaPeekAtLastError() );
            gpuErrchk( cudaDeviceSynchronize() );

            gpuErrchk( cudaFree(vector_of_tree_size_d) );
            gpuErrchk( cudaFree(sols_d) );
            gpuErrchk( cudaFree(root_prefixes_d) );
        }

    free(nb_explorers_per_stream);
    free(nb_explorers_prefix_sum);
}

double call_queens(int size, int initialDepth, int block_size, int set_cache, int nb_streams_per_dev, int max_devices){
    unsigned long long initial_tree_size = 0ULL;
    unsigned long long qtd_sols_global = 0ULL;
    unsigned long long gpu_tree_size = 0ULL;


    unsigned int nMaxPrefixos = 75580635;

    printf("\n### Queens size: %d, Initial depth: %d, Block size: %d", initialDepth, size, block_size);
    double initial_time = rtclock();

    QueenRoot* root_prefixes_h = (QueenRoot*)malloc(sizeof(QueenRoot)*nMaxPrefixos);
    unsigned long long int *vector_of_tree_size_h = (unsigned long long int*)malloc(sizeof(unsigned long long int)*nMaxPrefixos);
    unsigned long long int *solutions_h = (unsigned long long int*)malloc(sizeof(unsigned long long int)*nMaxPrefixos);

    //initial search, getting the tree root nodes for the gpu;
    unsigned long long n_explorers = BP_queens_prefixes((short)size, initialDepth ,&initial_tree_size, root_prefixes_h);

    //calling the gpu-based search

    GPU_call_cuda_queens(size, initialDepth, block_size, (bool)set_cache,n_explorers, root_prefixes_h ,vector_of_tree_size_h,
                            solutions_h, nb_streams_per_dev , max_devices);

    printf("\nInitial tree size: %llu", initial_tree_size );

    double final_time = rtclock();

    for(int i = 0; i<n_explorers;++i){
        if(solutions_h[i]>0)
            qtd_sols_global += solutions_h[i];
        if(vector_of_tree_size_h[i]>0)
            gpu_tree_size += vector_of_tree_size_h[i];
    }


    printf("\nGPU Tree size: %llu\nTotal tree size: %llu\nNumber of solutions found: %llu.\n", gpu_tree_size,(initial_tree_size+gpu_tree_size),qtd_sols_global );
    printf("\nElapsed total: %.3f\n", (final_time-initial_time));

    return (final_time-initial_time);
}


int main(int argc, char *argv[]){
  int initialDepth;
  int size;
  int block_size;
  int max_devices;
  int streams_per_dev = 1;

  if(argc!=5 && argc!=6){
      printf("provide arguments : ./queens size initialDepth block_size max_devices [streams_per_dev, default=1]");
      return -1;
  }
  if(argc==6)
      streams_per_dev = atoi(argv[5]);

  max_devices = atoi(argv[4]);
  block_size = atoi(argv[3]);
  initialDepth = atoi(argv[2]);
  size = atoi(argv[1]);

    auto time=call_queens(size, initialDepth, block_size, 0, streams_per_dev, max_devices);

    FILE *f;
    f=fopen("data_multi_cuda.txt","a");
    fprintf(f, "%d %d %d %f \n", size, initialDepth, block_size, time );

    return 0;
}
