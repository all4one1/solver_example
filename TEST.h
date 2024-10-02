#pragma once

#include "cublas_v2.h"
#include "cusparse.h"

double get_gpu_memory_used()
{
    size_t free_byte, total_byte;
    cudaError_t cuda_status = cudaMemGetInfo(&free_byte, &total_byte);

    if (cudaSuccess != cuda_status)
    {
        printf("Error: cudaMemGetInfo fails, %s \n", cudaGetErrorString(cuda_status));
        return -1;
    }
    double free_mb = (double)free_byte / 1024.0 / 1024.0;
    double total_mb = (double)total_byte / 1024.0 / 1024.0;
    double used_mb = total_mb - free_mb;

    return used_mb;
}

#define GPU_MEMORY cout << "GPU Memory: " << get_gpu_memory_used() << endl;

typedef struct VecStruct 
{
    cusparseDnVecDescr_t vec;
    double* ptr;
} Vec;

void cuSparse_test(int m, int nnz, int *d_A_rows, int * d_A_columns, double * d_A_values, double * d_L_values)
{
    int num_offsets = m + 1;
    Vec d_B, d_X, d_R, d_R_aux, d_P, d_T, d_tmp;

    cudaMalloc((void**)&d_B.ptr, m * sizeof(double));
    cudaMalloc((void**)&d_X.ptr, m * sizeof(double));
    cudaMalloc((void**)&d_R.ptr, m * sizeof(double));
    cudaMalloc((void**)&d_R_aux.ptr, m * sizeof(double));
    cudaMalloc((void**)&d_P.ptr, m * sizeof(double));
    cudaMalloc((void**)&d_T.ptr, m * sizeof(double));
    cudaMalloc((void**)&d_tmp.ptr, m * sizeof(double));

    cublasHandle_t   cublasHandle = NULL;
    cusparseHandle_t cusparseHandle = NULL;

    cublasCreate(&cublasHandle);
    cusparseCreate(&cusparseHandle);


    cusparseCreateDnVec(&d_B.vec, m, d_B.ptr, CUDA_R_64F);
    cusparseCreateDnVec(&d_X.vec, m, d_X.ptr, CUDA_R_64F);
    cusparseCreateDnVec(&d_R.vec, m, d_R.ptr, CUDA_R_64F);
    cusparseCreateDnVec(&d_R_aux.vec, m, d_R_aux.ptr, CUDA_R_64F);
    cusparseCreateDnVec(&d_P.vec, m, d_P.ptr, CUDA_R_64F);
    cusparseCreateDnVec(&d_T.vec, m, d_T.ptr, CUDA_R_64F);
    cusparseCreateDnVec(&d_tmp.vec, m, d_tmp.ptr, CUDA_R_64F);


    cusparseIndexBase_t  baseIdx = CUSPARSE_INDEX_BASE_ZERO;
    cusparseSpMatDescr_t matA, matL;
    int* d_L_rows = d_A_rows;
    int* d_L_columns = d_A_columns;
    cusparseFillMode_t   fill_lower = CUSPARSE_FILL_MODE_LOWER;
    cusparseDiagType_t   diag_non_unit = CUSPARSE_DIAG_TYPE_NON_UNIT;

    cusparseCreateCsr(&matA, m, m, nnz, d_A_rows,  d_A_columns, d_A_values, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,  baseIdx, CUDA_R_64F);
    cusparseCreateCsr(&matL, m, m, nnz, d_L_rows,  d_L_columns, d_L_values, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,  baseIdx, CUDA_R_64F);




    cusparseSpMatSetAttribute(matL, CUSPARSE_SPMAT_FILL_MODE, &fill_lower, sizeof(fill_lower));
    cusparseSpMatSetAttribute(matL, CUSPARSE_SPMAT_DIAG_TYPE, &diag_non_unit, sizeof(diag_non_unit));

    const double alpha = 0.75;
    size_t       bufferSizeMV;
    void* d_bufferMV;
    double       beta = 0.0;
    cusparseSpMV_bufferSize(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE,  &alpha, matA, d_X.vec, &beta, d_B.vec, CUDA_R_64F,    CUSPARSE_SPMV_ALG_DEFAULT, &bufferSizeMV);
    cudaMalloc(&d_bufferMV, bufferSizeMV);

    cusparseSpMV(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA, d_X.vec, &beta, d_B.vec, CUDA_R_64F,   CUSPARSE_SPMV_ALG_DEFAULT, d_bufferMV);
    cudaMemset(d_X.ptr, 0x0, m * sizeof(double));


    cusparseMatDescr_t descrM;
    csric02Info_t      infoM = NULL;
    int                bufferSizeIC = 0;
    void* d_bufferIC;
    cusparseCreateMatDescr(&descrM);
    cusparseSetMatIndexBase(descrM, baseIdx);
    cusparseSetMatType(descrM, CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatFillMode(descrM, CUSPARSE_FILL_MODE_LOWER);
    cusparseSetMatDiagType(descrM, CUSPARSE_DIAG_TYPE_NON_UNIT);
    cusparseCreateCsric02Info(&infoM);

    cusparseDcsric02_bufferSize(cusparseHandle, m, nnz, descrM, d_L_values, d_A_rows, d_A_columns, infoM, &bufferSizeIC);
    cudaMalloc(&d_bufferIC, bufferSizeIC);
    cusparseDcsric02_analysis(cusparseHandle, m, nnz, descrM, d_L_values, d_A_rows, d_A_columns, infoM, CUSPARSE_SOLVE_POLICY_NO_LEVEL, d_bufferIC);
    int structural_zero;
    cusparseXcsric02_zeroPivot(cusparseHandle, infoM, &structural_zero);


    cusparseDcsric02(cusparseHandle, m, nnz, descrM, d_L_values,  d_A_rows, d_A_columns, infoM, CUSPARSE_SOLVE_POLICY_NO_LEVEL, d_bufferIC);
    int numerical_zero;
    cusparseXcsric02_zeroPivot(cusparseHandle, infoM,  &numerical_zero);

    cusparseDestroyCsric02Info(infoM);
    cusparseDestroyMatDescr(descrM);
    cudaFree(d_bufferIC);

}