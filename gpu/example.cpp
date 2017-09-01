#include <stdio.h>
#include <lbfgs.cuh>

class objective_function
{
protected:
    float *m_x;

public:
    objective_function() : m_x(NULL)
    {
    }

    virtual ~objective_function()
    {
        if (m_x != NULL) {
            lbfgs_free(m_x);
            m_x = NULL;
        }
    }

    int run(int N)
    {
        
	gpuErrchk(cudaMalloc(&d_A, N * sizeof(float)));
	gpuErrchk(cudaMalloc(&d_x, N * sizeof(float)));
	gpuErrchk(cudaMalloc(&d_y, N * sizeof(float)));

        /* Initialize the variables. */
        for (int i = 0;i < N;i += 2) {
            h_A[i] = -1.2;
            h_A[i+1] = 1.0;
        }
        gpuErrchk(cudaMemcpy(d_A, h_A, N*sizeof(float), cudaMemcpyHostToDevice));

        /*
            Start the L-BFGS optimization; this will invoke the callback functions
            evaluate() and progress() when necessary.
         */
        int ret = lbfgs(N, d_x, &fx, _evaluate, _progress, this, NULL);

        /* Report the result. */
        printf("L-BFGS optimization terminated with status code = %d\n", ret);
        printf("  fx = %f, x[0] = %f, x[1] = %f\n", fx, m_x[0], m_x[1]);

        return ret;
    }

protected:
	float fx;
       
    float * h_A,d_A,h_x,d_x , h_y,d_y, b;
    int* rowptr, colinds;
    static float _evaluate(
        void *instance,
        float *x,
        float *g,
        const int n,
        const float step
        )
    {
        return reinterpret_cast<objective_function*>(instance)->evaluate(x, g, n, step);
    }

    float evaluate(
        float *x,
        float *g,
        const int n,
        float step
        )
    {
        float fx = 0.0;
        // g =  -A^T(Ax - b)
        // f = sum(Ax - b)^2
        
        CALLf_gradf(x, g, n);

        for (int i = 0;i < n;i += 2) {
            lbfgsfloatval_t t1 = 1.0 - x[i];
            lbfgsfloatval_t t2 = 10.0 * (x[i+1] - x[i] * x[i]);
            g[i+1] = 20.0 * t2;
            g[i] = -2.0 * (x[i] * g[i+1] + t1);
            fx += t1 * t1 + t2 * t2;
        }
        return fx;
    }

    static int _progress(
        void *instance,
        const lbfgsfloatval_t *x,
        const lbfgsfloatval_t *g,
        const lbfgsfloatval_t fx,
        const lbfgsfloatval_t xnorm,
        const lbfgsfloatval_t gnorm,
        const lbfgsfloatval_t step,
        int n,
        int k,
        int ls
        )
    {
        return reinterpret_cast<objective_function*>(instance)->progress(x, g, fx, xnorm, gnorm, step, n, k, ls);
    }

    int progress(
        const lbfgsfloatval_t *x,
        const lbfgsfloatval_t *g,
        const lbfgsfloatval_t fx,
        const lbfgsfloatval_t xnorm,
        const lbfgsfloatval_t gnorm,
        const lbfgsfloatval_t step,
        int n,
        int k,
        int ls
        )
    {
        printf("Iteration %d:\n", k);
        printf("  fx = %f, x[0] = %f, x[1] = %f\n", fx, x[0], x[1]);
        printf("  xnorm = %f, gnorm = %f, step = %f\n", xnorm, gnorm, step);
        printf("\n");
        return 0;
    }
};



#define N   100

int main(int argc, char *argv)
{
    objective_function obj;
    return obj.run(N);
}
