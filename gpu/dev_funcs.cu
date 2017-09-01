
#include "include/dev_funcs.cuh"


// Sample kernels for device operations

// Some kernels may come from ideas gotten online, especially Stack overflow and OrangeOwlSolutions github page

// Feel free to add your own kernels or change as required

//Franklin OKOLI - 2017

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
 
__global__ void Getsubmatrix( float* AValues, int* AColptr, int* ARowInd, int* keptCols, int inColCount, float* resultValues ,int* resultColptr, int* resultRowInd)
{
	int cItr, colOffset, c2;
	
	//result->m = A->m;
	//result->n = inColCount;
	//result->flags = A->flags;
	//KeptCols = Ncols of result

			
	colOffset = 0;
	for( cItr=0; cItr<inColCount; cItr++ )
	{
		resultColptr[cItr] = colOffset;
		for( c2=AColptr[keptCols[cItr]]; c2<AColptr[keptCols[cItr]+1]; c2++ )
		{
			resultRowInd[colOffset] = ARowInd[c2];
			resultValues[colOffset] = AValues[c2];
			colOffset++;
		}
	}
	resultColptr[cItr] = colOffset;
}




// Struct to compute squared difference on a tuple, to be called by thrust
struct zdiffsq{
template <typename Tuple>
  __host__ __device__ float operator()(Tuple a)
  {
    float result = thrust::get<1>(a) - thrust::get<0>(a);
    return result*result;
  }
};

struct square { __host__ __device__ float operator()(float x) { return x * x; } };


//Obtained from equelle cuda backend, please reference equelle as the original creator, i just change to float since the original version if not templated for different types
__global__ void initDiagonalMatrix( float* csrVal,int* csrRowPtr,int* csrColInd,  float* scalars,   int nnz)
{

   int row    =  blockIdx.x * blockDim.x + threadIdx.x;  
    if ( row < nnz + 1) {
	csrRowPtr[row] = row;
	if ( row < nnz) {
	    csrVal[row] = scalars[row];
	    csrColInd[row] = row;
	}
    }
}



//Kernel filters specific column from a csr matrix by multiplying elements of values with a boolean in the diagValues vector[colIndice]
__global__ void diagMult_kernel( float* csrVal,  int* csrRowPtr,int* csrColInds ,float* diagVals,  int total_nnz) 
{


          int index = blockIdx.x * blockDim.x + threadIdx.x;
	  int stride = blockDim.x * gridDim.x;

	  for (int i = index; i < total_nnz; i += stride)
	    {
		csrVal[i] = csrVal[i] * diagVals[csrColInds[i]];
				
	    }
}





//Multiply a Matrix in CSC format with an array of booleans matrix with the effect of filtering this matrix to create a new matrix
__global__ void diagMult_kernel2( float* cscVal,  int* cscColPtr,  float* diagVals,  int total_cols) 
{
   int col    = blockIdx.x * blockDim.x + threadIdx.x; 
		         printf("We are in column =: %d\n", col);  
    if ( col < total_cols ) {
	for (int i = cscColPtr[col]; i < cscColPtr[col+1]; i++) {
	    cscVal[i] = diagVals[col] * cscVal[i];
	}
    }
}

// Get a logical vector from a real vector, you can change this functionto pass a val to it or a val vector instead of 1.0
__global__ void SelectIndexes(float *d_vec1, int *IndexVector, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < N)
        if(IndexVector[i] == 1)
            d_vec1[i] = 1.0;
        if(IndexVector[i] == 0)
            d_vec1[i] = 0.0;
}

// Used to get active (IndexVector1) and passive (IndexVector2) indices in an active set optimization, bind those solutions (d_vec1) that are less than zero
// Release solutions that are valid , we can then use this d_vec to filter columns that we want to be active 
//In a matrix-vector multiplication, i.e.  filtering operation
// We can add more constraints to the solution here, e.g limit the upper bound of the solution vector
__global__ void bindzeros(float *d_vec1, int *IndexVector1, int * IndexVector2, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < N)
        {
	if(d_vec1[i] <= 0 && IndexVector1[i] == 1 )
            {IndexVector1[i] = 0;}
        if(d_vec1[i] >  0 && IndexVector1[i] == 0 )
            {IndexVector1[i] = 1;}

        if(IndexVector1[i] == 0)
            {IndexVector2[i] = 1;}
	else if(IndexVector1[i] == 1)
	    {IndexVector2[i] = 0;}
	}

}

// Set all active (IndexVector1) and passive (IndexVector2) indices in an active set optimization to initial state
// At initial state all  solution indices are set to active
__global__ void releaseMinY(int ind, int *IndexVector1, int * IndexVector2, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < N && i == ind)
        {
	IndexVector1[i] = 1;
	IndexVector2[i] = 0;
	}

}




// Zero all indices of an index set
__global__ void ZeroIndexes(float *d_vec1, int *IndexVector, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < N)
      IndexVector[i] = 0;
}

// Get thee solution points in an active set optimization whose gradient  can improve the solution 
//when increased from zero
__global__ void getFixed(float *grad, float *x, int *IndexVector, int N){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < N)
        if(x[i] == 0  && grad[i] > 0)
            IndexVector[i] = 1;
}




//clip negative value
__global__ void clipNegative(float *A, int N){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < N && A[i] < 0)
        A[i] = 0;
}

//owlqn_fabs , Take absolute value of a vector starting from an index the unneeded indices are set to zero
__global__ void owlqn_fabs(float* A,  int idx,  int n ){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < n && i >= idx)
	{
	 A[i] = fabs(A[i]);
	}
    else if(i < n && i < idx)
	{
	 A[i] = 0;
	}
}

//owlqn_pseudo_gradient , pseudo gradient modified for GPU from lbfgs solver by chokkan 
//for solving 1-norm constrained problems

__global__ void owlqn_pseudo_gradient( float* pg, float* x,  float* g,  int n,  float c,  int start,  int end )
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    /* Compute the negative of gradients. */
    if(i < start) {pg[i] = g[i];}

    if(i >= end && i < n) {pg[i] = g[i];}

/* Compute the psuedo-gradients. */
    if(i >= start && i < end) 
  {
/* Differentiable. */
    if (x[i] < 0.) { pg[i] = g[i] - c;}
/* Differentiable. */
    else if (0. < x[i]) { pg[i] = g[i] + c;} 
    
    else {
            if (g[i] < -c) {
                /* Take the right partial derivative. */
                pg[i] = g[i] + c;
            } else if (c < g[i]) {
                /* Take the left partial derivative. */
                pg[i] = g[i] - c;
            } else {
                pg[i] = 0.;
            }
        }
   }
}

//Choose Orthant on gpu for new point using this kernel
__global__ void owlqn_chooseorthant(float* wp,  float* xp, float* gp,  int n ){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < n)
	{
	wp[i] = (xp[i] == 0.) ? -gp[i] : xp[i];
	}
    
}

// projection in only one orthant for 1-norm constrained lbfgs

__global__ void owlqn_project(float* d, float* sign, int start,  int end )
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if( i >= start && i < end)
	{
	if (d[i] * sign[i] <= 0) { d[i] = 0;}
	}    
}


// constrained search direction t for 1-norm constrained lbfgs
__global__ void owlqn_constrain_searchdir(float* d, float* pg, int start,  int end )
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if( i >= start && i < end)
	{
	if (d[i] * pg[i] >= 0) { d[i] = 0;}
	}    
}

    
__global__ void findAlpha(int N, float* x,int* Free, float* pvector,float* Alphacontainer)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float alpha_temp = 1.0;
    if(i < N && Free[i] == 1)        
	{
	      if (alpha_temp * pvector[i] + x[i] < 0) 
		{ 
		// If the current alpha would overshoot
		alpha_temp = -x[i]/pvector[i];
		Alphacontainer[i] = alpha_temp;

		}
	      else if (alpha_temp * pvector[i] + x[i] >= 0)
	      	{ 
		// If the current alpha would be normal
		Alphacontainer[i] = alpha_temp;
		}

	}
    else if(i < N && Free[i] == 0) 
	{

	Alphacontainer[i] = 0;

	} 
}




//Change value of float device vector at specific index 
__global__ void ChangeIndex(float *A, int N, int index, float value){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < N && i == index)
        A[i] = value;
}


//Change value of  int device vector at specific index 
__global__ void ChangeIndex2(int *A, int N, int index, int value){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < N && i == index)
        A[i] = value;
}

// Creating a diagonal binary matrix that can be used to multiply and select a column 

__global__ void selectColumnGPU(float *devMatrix, int numR, int numC, int index ) {

    int col =      blockDim.x*blockIdx.x + threadIdx.x;
    int row    =   blockDim.y*blockIdx.y + threadIdx.y;
    int idx = row * numR + col;

    if(col < numC && row < numR)
	{
		  if(row == index &&  col == index)
			      {devMatrix[idx] = 1.0;}

		
	}
}



// Creating a diagonal binary matrix that can be used to multiply and select a column 
__global__ void initIdentityGPU(float *devMatrix, int numR, int numC) {
    int col =      blockDim.x*blockIdx.x + threadIdx.x;
    int row    =   blockDim.y*blockIdx.y + threadIdx.y;
    int index = row * numR + col;

    if(col < numC && row < numR)
	{
		  if(row  == col)
			      {devMatrix[index] = 1.0;}
	          else
			      {devMatrix[index] = 0.0;}
		
	}
}



// Creating a zero matrix that can be used to nullify a matrix 
__global__ void initZeroGPU(float *devMatrix, int numR, int numC) {
    
    int col =      blockDim.x*blockIdx.x + threadIdx.x;
    int row    =   blockDim.y*blockIdx.y + threadIdx.y;
    int index = row * numR + col;


    if(col < numC && row < numR)
	{
		  if(row  == col)
			      {devMatrix[index] = 0.0;}
		  else
			      {devMatrix[index] = 0.0;}
		
	}
}




// Generic kernel, get the kernel id - use id to run operation on a single element

__global__ void myKernel(float * vector, int n)
{
	  int index = blockIdx.x * blockDim.x + threadIdx.x;
	  int stride = blockDim.x * gridDim.x;

	  for (int i = index; i < n; i += stride)
	    {
	
	     
		         printf("Fetched for idx=%d: %g\n", i, vector[i]);

		
	    }
  

}


// Compare an int vector to an int val, give result in a binary vector,
__global__ void compareInt(int*A, int n,int ind,int value, int* result)
{
	  int index = blockIdx.x * blockDim.x + threadIdx.x;
	  int stride = blockDim.x * gridDim.x;

	  for (int i = index; i < n; i += stride)
	    {
	
		if(i == ind && A[i] == value)
		{
		result[i] = 1;
		}
		else
		{
		result[i] = 0;
		}

		
	    }
  

}




// Compute 2-norm using thrust
__host__ __device__ float norm2(int n, thrust::device_vector<float> newvector)
	{
  float reduction = std::sqrt(thrust::transform_reduce(newvector.begin(), newvector.end(), square(), 0.0f, thrust::plus<float>()));
   return    reduction;
	}


// Update a solution vector given a search direction and a step size

__global__ void updateX(int n, float* d_p, float* d_alpha, float * d_x)
	{
	  int index = blockIdx.x * blockDim.x + threadIdx.x;
	  int stride = blockDim.x * gridDim.x;

	  for (int i = index; i < n; i += stride)
	    {

	    d_x[i] =  d_x[i] + d_p[i] * d_alpha[i];	
		
	    }
	    
	}


// Square each element in a vector
__global__ void squareVector(int n, float *d_vec1, float *d_vec2, float * squaredResult)
	{
	  int index = blockIdx.x * blockDim.x + threadIdx.x;
	  int stride = blockDim.x * gridDim.x;
	
	  for (int i = index; i < n; i += stride)
	    {
	    squaredResult[i] = d_vec1[i] * d_vec2[i];
	    }
	    
	}
	
	// element-wise subtraction 
__global__ void subtractVector(int n, float *d_vec1, float *d_vec2, float * subtractResult)
	{
	  int index = blockIdx.x * blockDim.x + threadIdx.x;
	  int stride = blockDim.x * gridDim.x;
	
	  for (int i = index; i < n; i += stride)
	    {
	    subtractResult[i] = d_vec1[i] - d_vec2[i];
	    }
	    
	}




// Square element wise between two vector ----> result[i] = vec1[i]*vec2[i]
void CALLsquareVector(int n, float *d_vec1, float *d_vec2, float * squaredResult)
{
squareVector<<<THREAD_NUM,BLOCK_NUM>>>(n, d_vec1, d_vec2,squaredResult);
gpuErrchk(cudaPeekAtLastError());
gpuErrchk(cudaDeviceSynchronize()); 

}

// Difference element wise between two vector ----> result[i] = vec1[i] - vec2[i]
void CALLsubtractVector(int n, float *d_vec1, float *d_vec2, float * subtractResult)
{
subtractVector<<<THREAD_NUM,BLOCK_NUM>>>(n, d_vec1, d_vec2,subtractResult);
gpuErrchk(cudaPeekAtLastError());
gpuErrchk(cudaDeviceSynchronize());  
}



void CALLupdateX(int n, float* d_p, float* d_alpha, float * d_x)
{
    updateX<<<THREAD_NUM,BLOCK_NUM>>>(n, d_p, d_alpha, d_x);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
}



// sum of all elements in a vector using thrust ----> result = sum( vec1[i] .... vec2[n])
float CALLreduction(int n, float * d_x)
{
    
    //myKernel<<<THREAD_NUM,BLOCK_NUM>>>(d_x, n);
    thrust::device_ptr<float> dev_ptr_x = thrust::device_pointer_cast(d_x);
    float result = thrust::reduce(dev_ptr_x, dev_ptr_x + n);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
    
    return result;
}

// sum of all elements in a for integers vector using thrust ----> result = sum( vec1[i] .... vec2[n])
int CALLreduction2(int n, int * d_x)
{
    
    //myKernel<<<THREAD_NUM,BLOCK_NUM>>>(d_x, n);
    thrust::device_ptr<int> dev_ptr_x = thrust::device_pointer_cast(d_x);
    int result = thrust::reduce(dev_ptr_x, dev_ptr_x + n);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
    
    return result;
}

// Dot  product between two vectors
float CALLdot(int n, float* oldvector, float* newvector)
{
    squareVector<<<THREAD_NUM,BLOCK_NUM>>>(n, oldvector, newvector,newvector);
    float result =  CALLreduction(n, newvector);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    return result;
}

// Norm2 of a device vector
float CALLnorm2(int n,  float* newvector)
{
    thrust::device_ptr<float> dev_ptr_new = thrust::device_pointer_cast(newvector);
    thrust::device_vector<float> vec(newvector, newvector + n); 
    float result =  norm2( n , vec );
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    return result;
}

void CALLfindAlpha(int n,  float* x, int* Free, float* pvector, float* Alphacontainer)
{
    findAlpha<<<THREAD_NUM,BLOCK_NUM>>>(n, x, Free,  pvector, Alphacontainer);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
}

void CALLreleaseMinY(int n, float* y, int* Free, int * Bound)
{
	// We first find the minimum element position and index with the help of thrust
    thrust::device_ptr<float> dp = thrust::device_pointer_cast(y);
    thrust::device_ptr<float> pos = thrust::min_element(dp, dp + n);
    int index = thrust::distance(dp, pos);
	// Then we make a swap the corresponding indices on the Free and Bound vectors
    releaseMinY<<<THREAD_NUM,BLOCK_NUM>>>(index, Free,  Bound,n);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
}

bool CALLoptimalPt(int n, float* y)
{
    bool isoptim = true;
	// We first find the minimum element position and index with the help of thrust
    thrust::device_ptr<float> dp = thrust::device_pointer_cast(y);
    thrust::device_ptr<float> pos = thrust::min_element(dp, dp + n);
    thrust::device_vector<float> vec(pos,pos + 1); 

	if(vec[0] < 0)
	{
	isoptim = false;
	return isoptim;
	}
     return isoptim;
}

float CALLmaxelement(int n, float* y, int index)
{
    float max_element = 0;
	// We first find the minimum element position and index with the help of thrust
    thrust::device_ptr<float> dp = thrust::device_pointer_cast(y);
    thrust::device_ptr<float> pos = thrust::max_element(dp, dp + n);
    thrust::device_vector<float> vec(pos,pos + 1);
    index = thrust::distance(dp, pos); 

     max_element = vec[0];
     return max_element;
}

bool CALLcompareInt2Array(int *A, int N, int index, int value, int* emptyIntVec)
{
    bool result = false;
    compareInt<<<THREAD_NUM,BLOCK_NUM>>>(A, N, index, value, emptyIntVec );
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
    int temp = CALLreduction2(N,emptyIntVec);

	if(temp > 0)
	{
	result = true;
	return result;
	}

	return result;
}

void CALLgetFreeIndex(int n,int* Free,float* prices,float* temp,int best_price_free_index, int* intvector)
{
	int one = 1;

	int price_index = 0;

	//define infinity
	float Infinity = std::numeric_limits<float>::infinity();
	//negate infinity
	Infinity = -Infinity;
	//Save data to temp and use temp for calculations
	gpuErrchk(cudaMemcpy(temp,prices, n *sizeof(float), cudaMemcpyDeviceToDevice)); 
	// Get the max element and its index
	CALLmaxelement(n,temp,price_index);
	 
	while( CALLcompareInt2Array(Free,n, price_index,one, intvector))
	{
	CALLChangeIndex(temp, n,price_index,Infinity);
	CALLmaxelement(n,temp,price_index);
	}

	best_price_free_index = price_index;

}



// Creates an Identity Matrix on GPU
void CALLcreateIdentity(float *devMatrix, int numR, int numC)
{

     dim3 dimBlock(1, 1);
     dim3 dimGrid(numR,numC);
     initIdentityGPU<<<dimGrid, dimBlock>>>(devMatrix, numR, numC);
     gpuErrchk(cudaPeekAtLastError());
     gpuErrchk(cudaDeviceSynchronize());
}

// Selects a column by changing filter index to 1
void CALLselectColumnGPU(float *devMatrix, int numR, int numC, int index )
{
	dim3 dimBlock(1, 1);
        dim3 dimGrid(numR,numC);
 	selectColumnGPU<<<dimGrid, dimBlock>>>(devMatrix, numR, numC,index);
       	gpuErrchk(cudaPeekAtLastError());
 	gpuErrchk(cudaDeviceSynchronize());
}

// Creates an Zero Matrix on GPU
void CALLcreateZero(float *devMatrix, int numR, int numC)
{
	dim3 dimBlock(1, 1);
        dim3 dimGrid(numR,numC);
 	initZeroGPU<<<dimGrid, dimBlock>>>(devMatrix, numR, numC);
       	gpuErrchk(cudaPeekAtLastError());
 	gpuErrchk(cudaDeviceSynchronize());
}


// Creates an CSR Diagonal Matrix on GPU at specified indices
void CALLinitDiagonalMatrix( float* csrVal,int* csrRowPtr,int* csrColInd,  float* scalars,   int nnz)
{
	dim3 dimBlock(512);
        dim3 dimGrid( (int)(( (nnz+1) + 512 - 1)/512) );
 	initDiagonalMatrix<<<dimGrid, dimBlock>>>( csrVal, csrRowPtr, csrColInd, scalars, nnz);
       	gpuErrchk(cudaPeekAtLastError());
 	gpuErrchk(cudaDeviceSynchronize());
}


// Multiplies a CSR Diagonal Matrix on GPU to CSR DENSE MATRIX at specified indices
void CALLdiagMult_kernel( float* csrVal,  int* csrRowPtr ,int* csrColInds ,  float* diagVals,  int total_nnz) 
{
 	diagMult_kernel<<<THREAD_NUM, BLOCK_NUM>>>( csrVal, csrRowPtr, csrColInds,  diagVals,  total_nnz);
       	gpuErrchk(cudaPeekAtLastError());
 	gpuErrchk(cudaDeviceSynchronize());
}


// Multiplies a CSC Matrix on GPU at specified indices with booleans leaving a filtering effect
 void CALLdiagMult_kernel2( float* cscVal,  int* cscColPtr,  float* diagVals,  int total_cols) 
{
	dim3 dimBlock(512);
        dim3 dimGrid( (int)(( total_cols + 512 - 1)/512) );
 	diagMult_kernel2<<<dimGrid, dimBlock>>>( cscVal, cscColPtr,  diagVals,  total_cols);
       	gpuErrchk(cudaPeekAtLastError());
 	gpuErrchk(cudaDeviceSynchronize());
}


// Sets elements in a vector less than zero to zero on GPU
void CALLclipNegative(float *A, int N)
{
    clipNegative<<<THREAD_NUM,BLOCK_NUM>>>(A,N);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
}

// Sets elements in a float vector at index to value on GPU
void CALLChangeIndex(float *A, int N, int index, float value)
{
    ChangeIndex<<<THREAD_NUM,BLOCK_NUM>>>(A,  N,  index,  value);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
}

// Sets elements in an int vector at index to value on GPU
void CALLChangeIndex2(int *A, int N, int index, int value)
{
    ChangeIndex2<<<THREAD_NUM,BLOCK_NUM>>>(A,  N,  index,  value);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
}

void CALLgetFixed(float *grad, float *x, int *IndexVector, int N)
{
    getFixed<<<THREAD_NUM,BLOCK_NUM>>>(grad, x, IndexVector, N);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
}

// Given a vector and a vector of indices, set vector[i] = 0 
void CALLZeroIndexes(float *d_vec1, int *IndexVector, int N)
{
    ZeroIndexes<<<THREAD_NUM,BLOCK_NUM>>>(d_vec1,IndexVector, N);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
}

// Given a vector and a vector of indices, set vector[indice[i]] = 0 
void CALLSelectIndexes(float *d_vec1, int *IndexVector, int N)
{
    SelectIndexes<<<THREAD_NUM,BLOCK_NUM>>>(d_vec1,IndexVector, N);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
}


// Given a vector and a vector of indices, set vector[indice[i]] = 0 
void CALLbindzeros(float *d_vec1, int *IndexVector1,int * IndexVector2, int N)
{
    bindzeros<<<THREAD_NUM,BLOCK_NUM>>>(d_vec1,IndexVector1,IndexVector2, N);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
}





// A kernel to copy a submatrix from a full matrix in CCS format by selecting specific columns, this kernel is not tested yet and might simply be wrong --> Adapted from tsnnls by Jason Cantarella (jason.cantarella@gmail.com) and Michael Piatek 

void CALLGetsubmatrix( float* AValues, int* AColptr, int* ARowInd, int* keptCols, int inColCount, float* resultValues ,int* resultColptr, int* resultRowInd)
{
    Getsubmatrix<<<THREAD_NUM,BLOCK_NUM>>>( AValues, AColptr, ARowInd, keptCols, inColCount,  resultValues , resultColptr, resultRowInd);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
}


float CALLowlqn_x1norm(float* x,  int start,  int n )
{
    float norm = 0.;
    owlqn_fabs<<<THREAD_NUM,BLOCK_NUM>>>(x, start, n );
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());   
    norm = CALLreduction(n,x); 
    return norm;
}


void CALLowlqn_pseudo_gradient( float* pg,  float* x,  float* g,  int n,  float c,  int start, const int end )
{

    owlqn_pseudo_gradient<<<THREAD_NUM,BLOCK_NUM>>>(  pg, x,  g,  n,  c,  start,  end );
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize()); 
  
}

void CALLowlqn_chooseorthant(float* wp,  float* xp, float* gp,  int n )
{
    owlqn_chooseorthant<<<THREAD_NUM,BLOCK_NUM>>>(  wp, xp,  gp,  n);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize()); 
  
}

void CALLowlqn_project(float* d, float* sign, int start,  int end )
{
    owlqn_project<<<THREAD_NUM,BLOCK_NUM>>>(  d, sign,  start,  end);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize()); 
  
}

void CALLowlqn_constrain_searchdir(float* d, float* pg, int start,  int end )
{
    owlqn_constrain_searchdir<<<THREAD_NUM,BLOCK_NUM>>>(  d, pg,  start,  end);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize()); 
  
}























pseudo_gradient
