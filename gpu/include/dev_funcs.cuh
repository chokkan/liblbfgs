//Copyright (c) 2017 Franklin OKOLI
/* * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

#pragma once

// Cuda libs
#include <cuda_runtime.h>
#include <cusparse_v2.h>
#include <cusolverSp.h>
#include <helper_cuda.h>





// Thrust libs
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/extrema.h>
#include <thrust/copy.h>
#include <thrust/system_error.h>
#include <thrust/execution_policy.h>
#include <thrust/transform.h>
#include <thrust/tuple.h>
#include <thrust/transform_reduce.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/inner_product.h>

// C ++ standard libs
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <fstream>
#include "string.h"
#include <cstdlib>
#include <list>
#include <vector>
#include <iterator>
#include <iostream>
#include <algorithm>
#include <math.h>

#include "Utilities.cuh"
#include "cuda.h"



#define BLOCK_NUM	64
#define THREAD_NUM	512

//#define min(x,y) (x>y?x:y)
//#define N 33*1024

//#define ThreadPerBlock 256

//smallest multiple of threadsPerBlock that is greater than or equal to N
//#define blockPerGrid min(32 , (N+ThreadPerBlock-1) / ThreadPerBlock )
//Vector_Dot_Product <<<blockPerGrid , ThreadPerBlock >>> (V1_D , V2_D , V3_D ) ;

// function prototypes
void CALLsquareVector(int n, float *d_vec1, float *d_vec2, float * squaredResult);
float CALLreduction(int n, float * d_x);
float CALLdot(int n, float* oldvector, float* newvector);
void CALLcreateIdentity(float *devMatrix, int numR, int numC);
void CALLcreateZero(float *devMatrix, int numR, int numC);
void CALLselectColumnGPU(float *devMatrix, int numR, int numC, int index );
void CALLclipNegative(float *A, int N);
void CALLgetFixed(float *grad, float *x, int *IndexVector, int N);
void CALLsubtractVector(int n, float *d_vec1, float *d_vec2, float * subtractResult);
void CALLZeroIndexes(float *d_vec1, int *IndexVector, int N);
void CALLGetsubmatrix( float* AValues, int* AColptr, int* ARowInd, int* keptCols, int inColCount, float* resultValues ,int* resultColptr, int* resultRowInd);
void CALLinitDiagonalMatrix( float* csrVal,int* csrRowPtr,int* csrColInd,  float* scalars,   int nnz);
void CALLdiagMult_kernel( float* csrVal,  int* csrRowPtr ,int* csrColInds ,  float* diagVals,  int total_nnz); 
void CALLdiagMult_kernel2( float* cscVal,  int* cscColPtr,  float* diagVals,  int total_cols); 
void CALLChangeIndex(float *A, int N, int index, float value);
void CALLChangeIndex2(int *A, int N, int index, int value);
bool CALLcompareInt2Array(int *A, int N, int index, int value);
void CALLSelectIndexes(float *d_vec1, int *IndexVector, int N);
void CALLbindzeros(float *d_vec1, int *IndexVector1,int * IndexVector2, int N);
float CALLnorm2(int n,  float* newvector);
void CALLfindAlpha(int n,  float* x, int* Free, float* pvector, float* Alphacontainer);
void CALLupdateX(int n, float* d_p, float* d_alpha, float * d_x);
void CALLreleaseMinY(int n, float* y, int* Free, int * Bound);
bool CALLoptimalPt(int n, float* y);
float CALLmaxelement(int n, float* y, int index);
void CALLgetFreeIndex(int n,int* Free,float* prices,float* temp,int best_price_free_index, int* intvector);
int CALLreduction2(int n, int * d_x);
bool CALLcompareInt2Array(int *A, int N, int index, int value, int* emptyIntVec);
float CALLowlqn_x1norm(float* x,  int start,  int n );
void CALLowlqn_pseudo_gradient( float* pg,  float* x,  float* g,  int n,  float c,  int start,  int end );
void CALLowlqn_chooseorthant(float* wp,  float* xp, float* gp,  int n );
void CALLowlqn_project(float* d, float* sign, int start,  int end );
void CALLowlqn_constrain_searchdir(float* d, float* pg, int start,  int end );



















