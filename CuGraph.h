#pragma once
#include "cuda_runtime.h"
#include <vector>

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