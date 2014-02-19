package main

import (
	"../go"
	"fmt"
)

func evaluate(x, g []lbfgs.Float, _ lbfgs.Float) lbfgs.Float {
	fx := lbfgs.Float(0)
	for i := 0; i < len(x); i += 2 {
		t1 := lbfgs.Float(1 - x[i])
		t2 := 10 * (x[i+1] - x[i]*x[i])
		g[i+1] = 20 * t2
		g[i] = -2 * (x[i]*g[i+1] + t1)
		fx += t1*t1 + t2*t2
	}
	return fx
}

func progress(x, _ []lbfgs.Float, fx, xnorm, gnorm, step lbfgs.Float, k, _ int) int {
	fmt.Printf("Iteration %d:\n", k)
	fmt.Printf("  fx = %f, x[0] = %f, x[1] = %f\n", fx, x[0], x[1])
	fmt.Printf("  xnorm = %f, gnorm = %f, step = %f\n", xnorm, gnorm, step)
	fmt.Println()
	return 0
}

const N = 100

func main() {
	x := lbfgs.MakeSlice(N)
	defer lbfgs.FreeSlice(x)

	// Initialize the variables.
	for i := 0; i < N; i += 2 {
		x[i], x[i+1] = -1.2, 1
	}

	// Use pre-initialized default parameters.
	param := lbfgs.DefaultParam
	// param.LineSearch = lbfgs.BackTracking

	// Start the L-BFGS optimization; this will invoke evaluate() and
	// progress() when necessary.
	fx, err := lbfgs.Minimize(x, evaluate, progress, &param)

	// Report the result.
	status := "success"
	if err != nil {
		status = err.Error()
	}
	fmt.Printf("L-BFGS optimization terminated with status = %q\n", status)
	fmt.Printf("  fx = %f, x[0] = %f, x[1] = %f\n", fx, x[0], x[1])
}
