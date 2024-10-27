#pragma once
#include "cuda_runtime.h"
#include <vector>
#include <cmath>

struct CudaLaunchSetup
{
	dim3 Grid3D, Block3D, Grid1D, Block1D;
	unsigned int thread_x = 8, thread_y = 8, thread_z = 8, thread_1D = 512;

	CudaLaunchSetup(unsigned int N, unsigned int nx = 1, unsigned int ny = 1, unsigned nz = 1)
	{
		Grid3D = dim3(
			(unsigned int)ceil((nx + 1.0) / thread_x),
			(unsigned int)ceil((ny + 1.0) / thread_y),
			(unsigned int)ceil((nz + 1.0) / thread_z));
		Block3D = dim3(thread_x, thread_y, thread_z);

		Grid1D = dim3((unsigned int)ceil((N + 0.0) / thread_1D));
		Block1D = thread_1D;

	};
};

struct SparseMatrixCuda
{
	/*		Compressed Sparse Row	 */

	int Nfull = 0;	// the input (linear) size of a matrix
	int nval = 0;	// number of non-zero elements
	int nrow = 0;	// number of rows
	size_t bytesVal = 0;
	size_t bytesCol = 0;
	size_t bytesRow = 0;
	double* val = nullptr;
	//double* aux = nullptr;
	int* col = nullptr;
	int* row = nullptr;

	SparseMatrixCuda() {};
	SparseMatrixCuda(int N, int nv, double* v, int* c, int* r) : Nfull(N), nval(nv)
	{
		nrow = N + 1;
		bytesVal = nval * sizeof(double);
		bytesCol = nval * sizeof(int);
		bytesRow = nrow * sizeof(int);

		cudaMalloc((void**)&val, sizeof(double) * nval);
		//cudaMalloc((void**)&aux, sizeof(double) * nval);
		cudaMalloc((void**)&col, sizeof(int) * nval);
		cudaMalloc((void**)&row, sizeof(int) * nrow);

		cudaMemcpy(val, v, bytesVal, cudaMemcpyHostToDevice);
		//cudaMemcpy(val, v, bytesVal, cudaMemcpyHostToDevice);
		cudaMemcpy(col, c, bytesCol, cudaMemcpyHostToDevice);
		cudaMemcpy(row, r, bytesRow, cudaMemcpyHostToDevice);
	}
	~SparseMatrixCuda() {};
};


struct CuGraph
{
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t graphExec = nullptr;
    std::vector<cudaGraphNode_t> Nodes;
    CuGraph()
    {
        cudaGraphCreate(&graph, 0);
    }

    //~CuGraph()
    //{
    //    cudaGraphExecDestroy(graphExec);
    //    cudaGraphDestroy(graph);
    //    graph = nullptr;
    //    graphExec = nullptr;
    //    Nodes.clear();
    //}

    void add_kernel_node(unsigned int threads_per_block, unsigned int num_blocks, void* kernel, void** args,
        unsigned int sbytes = 0, bool depend_on_previous = true)
    {
        Nodes.emplace_back();

        cudaKernelNodeParams param = { 0 };
        param.blockDim = threads_per_block;
        param.gridDim = num_blocks;
        param.func = kernel;
        param.kernelParams = args;
        param.sharedMemBytes = sbytes;

        size_t current = Nodes.size() - 1;
        if (depend_on_previous && current > 0)
            cudaGraphAddKernelNode(&Nodes.back(), graph, &Nodes[current - 1], 1, &param);
        else
            cudaGraphAddKernelNode(&Nodes.back(), graph, nullptr, 0, &param);


    }

    void add_copy_node(void* dst, const void* src, unsigned int Nbytes, cudaMemcpyKind direction, bool depend_on_previous = true)
    {
        Nodes.emplace_back();
        size_t current = Nodes.size() - 1;
        if (depend_on_previous && current > 0)
            cudaGraphAddMemcpyNode1D(&Nodes[current], graph, &Nodes[current - 1], 1, dst, src, Nbytes, direction);
        else
            cudaGraphAddMemcpyNode1D(&Nodes[current], graph, nullptr, 0, dst, src, Nbytes, direction);

    }

    void add_graph_as_node(cudaGraph_t& childGraph, bool depend_on_previous = true)
    {
        Nodes.emplace_back();
        size_t current = Nodes.size() - 1;
        if (depend_on_previous && current > 0)
            cudaGraphAddChildGraphNode(&Nodes[current], graph, &Nodes[current - 1], 1, childGraph);
        else
            cudaGraphAddChildGraphNode(&Nodes[current], graph, nullptr, 0, childGraph);
    }

    void add_graph_as_node(CuGraph& childGraph, bool depend_on_previous = true)
    {
        Nodes.emplace_back();
        size_t current = Nodes.size() - 1;
        if (depend_on_previous && current > 0)
            cudaGraphAddChildGraphNode(&Nodes[current], graph, &Nodes[current - 1], 1, childGraph.graph);
        else
            cudaGraphAddChildGraphNode(&Nodes[current], graph, nullptr, 0, childGraph.graph);
    }


    void instantiate()
    {
        cudaGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0);
    }

    void launch()
    {
        cudaGraphLaunch(graphExec, 0);
    }
};