// Package lbfgs is a Go wrapper for libLBFGS by Naoaki Okazaki, a C
// port of the implementation of Limited-memory
// Broyden-Fletcher-Goldfarb-Shanno (L-BFGS) method written by Jorge
// Nocedal. [http://www.chokkan.org/software/liblbfgs]
//
// The L-BFGS method solves the unconstrainted minimization problem,
//
//     minimize F(x), x = (x1, x2, ..., xN),
//
// only if the objective function F(x) and its gradient G(x) are
// computable. The well-known Newton's method requires computation of
// the inverse of the hessian matrix of the objective
// function. However, the computational cost for the inverse hessian
// matrix is expensive especially when the objective function takes a
// large number of variables. The L-BFGS method iteratively finds a
// minimizer by approximating the inverse hessian matrix by
// information from last m iterations. This innovation saves the
// memory storage and computational time drastically for large-scaled
// problems.
package lbfgs

/*
#cgo LDFLAGS: -llbfgs
#include "cgo.h"
*/
import "C"

import (
	"errors"
	"fmt"
	"reflect"
	"unsafe"
)

// Float is the floating point number type used by libLBFGS, which
// allows the user to configure the precision of floating point
// numbers.
type Float C.lbfgsfloatval_t

// EvaluateFunc is the Go correpsondence of lbfgs_evaluate_t. It
// computes the objective function and gradient. The Minimize()
// function calls this to obtain the values of objective function and
// its gradients when needed. A client program must implement this
// function to evaluate the values of the objective function and its
// gradients, given current values of variables.
//
// x: a slice of the current values of variables; must not be
// modified.
//
// g: a pre-allocated slice to write the computed gradient;
// guarranteed to have the same length as x.
//
// step: the current step of the line search routine.
//
// The value of the objective function at point x should be returned.
type EvaluateFunc func(x, g []Float, step Float) Float

// ProgressFunc is the Go correpsondence of lbfgs_progress_t. It
// receives the progress of the optimization process. The Minimize()
// function calls this function for each iteration. Implementing this
// function, a client program can store or display the current
// progress of the optimization process.
//
// x: a slice of the current values of variables; must not be
// modified.
//
// g: a slice of the current gradient values of variables; must not be
// modified.
//
// fx: the current value of the objective function.
//
// xnorm: the Euclidean norm of the variables.
//
// gnorm: the current value of the objective function.
//
// step: the line-search step used for this iteration.
//
// k: the iteration count.
//
// ls: the number of evaluations called for this iteration.
//
// Return 0 if the optimization process should continue and a non-zero
// value if it should terminate.
type ProgressFunc func(x, g []Float, fx, xnorm, gnorm, step Float, k, ls int) int

// Silent is a silent ProgressFunc.
func Silent(_, _ []Float, _, _, _, _ Float, _, _ int) int {
	return 0
}

// EveryN transforms f to a ProgressFunc that calls f every n
// iterations and continues right-away in other cases.
func EveryN(n int, f ProgressFunc) ProgressFunc {
	return func(x, g []Float, fx, xnorm, gnorm, step Float, k, ls int) int {
		if k%n == 0 {
			return f(x, g, fx, xnorm, gnorm, step, k, ls)
		} else {
			return 0
		}
	}
}

// Param stores L-BFGS optimization parameters. A pre-initialized
// DefaultParam is also available. When customization is needed, the
// user is recommended to copy DefaultParam and modifiy the copy.
type Param struct {
	// The number of corrections to approximate the inverse hessian
	// matrix.
	//
	// The L-BFGS routine stores the computation results of previous m
	// iterations to approximate the inverse hessian matrix of the
	// current iteration. This parameter controls the size of the
	// limited memories (corrections). The default value is 6. Values
	// less than 3 are not recommended. Large values will result in
	// excessive computing time.
	M int
	// Epsilon for convergence test.
	//
	// This parameter determines the accuracy with which the solution is
	// to be found. A minimization terminates when ||g|| < epsilon *
	// max(1, ||x||), where ||.|| denotes the Euclidean (L2) norm. The
	// default value is 1e-5.
	Epsilon Float
	// Distance for delta-based convergence test.
	//
	// This parameter determines the distance, in iterations, to compute
	// the rate of decrease of the objective function. If the value of
	// this parameter is zero, the library does not perform the
	// delta-based convergence test. The default value is 0.
	Past int
	// Delta for convergence test.
	//
	// This parameter determines the minimum rate of decrease of the
	// objective function. The library stops iterations when the
	// following condition is met: (f' - f) / f < delta, where f' is the
	// objective value of past iterations ago, and f is the objective
	// value of the current iteration. The default value is 0.
	Delta Float
	// The maximum number of iterations.
	//
	// The lbfgs() function terminates an optimization process with
	// LBFGSERR_MAXIMUMITERATION status code when the iteration count
	// exceedes this parameter. Setting this parameter to zero continues
	// an optimization process until a convergence or error. The default
	// value is 0.
	MaxIterations int
	// The line search algorithm.
	//
	// This parameter specifies a line search algorithm to be used by
	// the L-BFGS routine.
	LineSearch int
	// The maximum number of trials for the line search.
	//
	// This parameter controls the number of function and gradients
	// evaluations per iteration for the line search routine. The
	// default value is 20.
	MaxLineSearch int
	// The minimum step of the line search routine.
	//
	// The default value is 1e-20. This value need not be modified
	// unless the exponents are too large for the machine being used, or
	// unless the problem is extremely badly scaled (in which case the
	// exponents should be increased).
	MinStep Float
	// The maximum step of the line search.
	//
	// The default value is 1e+20. This value need not be modified
	// unless the exponents are too large for the machine being used, or
	// unless the problem is extremely badly scaled (in which case the
	// exponents should be increased).
	MaxStep Float
	// A parameter to control the accuracy of the line search routine.
	//
	// The default value is 1e-4. This parameter should be greater than
	// zero and smaller than 0.5.
	Ftol Float
	// A coefficient for the Wolfe condition.
	//
	// This parameter is valid only when the backtracking line-search
	// algorithm is used with the Wolfe condition,
	// BackTrackingStrongWolfe or BackTrackingWolfe. The default value
	// is 0.9. This parameter should be greater the ftol parameter and
	// smaller than 1.0.
	Wolfe Float
	// A parameter to control the accuracy of the line search routine.
	//
	// The default value is 0.9. If the function and gradient
	// evaluations are inexpensive with respect to the cost of the
	// iteration (which is sometimes the case when solving very large
	// problems) it may be advantageous to set this parameter to a small
	// value. A typical small value is 0.1. This parameter shuold be
	// greater than the ftol parameter (1e-4) and smaller than 1.0.
	Gtol Float
	// The machine precision for floating-point values.
	//
	// This parameter must be a positive value set by a client program
	// to estimate the machine precision. The line search routine will
	// terminate with the status code (LBFGSERR_ROUNDING_ERROR) if the
	// relative width of the interval of uncertainty is less than this
	// parameter.
	Xtol Float
	// Coeefficient for the L1 norm of variables.
	//
	// This parameter should be set to zero for standard minimization
	// problems. Setting this parameter to a positive value activates
	// Orthant-Wise Limited-memory Quasi-Newton (OWL-QN) method, which
	// minimizes the objective function F(x) combined with the L1 norm
	// |x| of the variables, {F(x) + C |x|}. This parameter is the
	// coeefficient for the |x|, i.e., C. As the L1 norm |x| is not
	// differentiable at zero, the library modifies function and
	// gradient evaluations from a client program suitably; a client
	// program thus have only to return the function value F(x) and
	// gradients G(x) as usual. The default value is zero.
	OrthantWiseC Float
	// Start index for computing L1 norm of the variables.
	//
	// This parameter is valid only for OWL-QN method (i.e.,
	// orthantwise_c != 0). This parameter b (0 <= b < N) specifies the
	// index number from which the library computes the L1 norm of the
	// variables x, |x| := |x_{b}| + |x_{b+1}| + ... + |x_{N}| . In
	// other words, variables x_1, ..., x_{b-1} are not used for
	// computing the L1 norm. Setting b (0 < b < N), one can protect
	// variables, x_1, ..., x_{b-1} (e.g., a bias term of logistic
	// regression) from being regularized. The default value is zero.
	OrthantWiseStart int
	// End index for computing L1 norm of the variables.
	//
	// This parameter is valid only for OWL-QN method (i.e.,
	// orthantwise_c != 0). This parameter e (0 < e <= N) specifies the
	// index number at which the library stops computing the L1 norm of
	// the variables x,
	OrthantWiseEnd int
}

// DefaultParam is the default parameter. It is initialized in the
// following init().
var DefaultParam Param

func init() {
	var param C.lbfgs_parameter_t
	C.lbfgs_parameter_init(&param)
	setGoParam(&param, &DefaultParam)
}

func setGoParam(c *C.lbfgs_parameter_t, g *Param) {
	g.M = int(c.m)
	g.Epsilon = Float(c.epsilon)
	g.Past = int(c.past)
	g.Delta = Float(c.delta)
	g.MaxIterations = int(c.max_iterations)
	g.LineSearch = int(c.linesearch)
	g.MaxLineSearch = int(c.max_linesearch)
	g.MinStep = Float(c.min_step)
	g.MaxStep = Float(c.max_step)
	g.Ftol = Float(c.ftol)
	g.Wolfe = Float(c.wolfe)
	g.Gtol = Float(c.gtol)
	g.Xtol = Float(c.xtol)
	g.OrthantWiseC = Float(c.orthantwise_c)
	g.OrthantWiseStart = int(c.orthantwise_start)
	g.OrthantWiseEnd = int(c.orthantwise_end)
}

func setCParam(g *Param, c *C.lbfgs_parameter_t) {
	c.m = C.int(g.M)
	c.epsilon = C.lbfgsfloatval_t(g.Epsilon)
	c.past = C.int(g.Past)
	c.delta = C.lbfgsfloatval_t(g.Delta)
	c.max_iterations = C.int(g.MaxIterations)
	c.linesearch = C.int(g.LineSearch)
	c.max_linesearch = C.int(g.MaxLineSearch)
	c.min_step = C.lbfgsfloatval_t(g.MinStep)
	c.max_step = C.lbfgsfloatval_t(g.MaxStep)
	c.ftol = C.lbfgsfloatval_t(g.Ftol)
	c.wolfe = C.lbfgsfloatval_t(g.Wolfe)
	c.gtol = C.lbfgsfloatval_t(g.Gtol)
	c.xtol = C.lbfgsfloatval_t(g.Xtol)
	c.orthantwise_c = C.lbfgsfloatval_t(g.OrthantWiseC)
	c.orthantwise_start = C.int(g.OrthantWiseStart)
	c.orthantwise_end = C.int(g.OrthantWiseEnd)
}

// List of available line search algorithms.
const (
	// The default algorithm (MoreThuente method).
	Default = C.LBFGS_LINESEARCH_DEFAULT
	// MoreThuente method proposd by More and Thuente.
	MoreThuente = C.LBFGS_LINESEARCH_MORETHUENTE
	// Backtracking method with the Armijo condition.
	//
	// The backtracking method finds the step length such that it
	// satisfies the sufficient decrease (Armijo) condition,
	//
	// f(x + a * d) <= f(x) + lbfgs_parameter_t::ftol * a * g(x)^T d,
	//
	// where x is the current point, d is the current search direction,
	// and a is the step length.
	BackTrackingArmijo = C.LBFGS_LINESEARCH_BACKTRACKING_ARMIJO
	// The backtracking method with the defualt (regular Wolfe) condition.
	BackTracking = C.LBFGS_LINESEARCH_BACKTRACKING
	// Backtracking method with regular Wolfe condition.
	//
	// The backtracking method finds the step length such that it
	// satisfies both the Armijo condition (BackTrackingArmijo) and the
	// curvature condition,
	//
	// g(x + a * d)^T d >= lbfgs_parameter_t::wolfe * g(x)^T d,
	//
	// where x is the current point, d is the current search direction,
	// and a is the step length.
	BackTrackingWolfe = C.LBFGS_LINESEARCH_BACKTRACKING_WOLFE
	// Backtracking method with strong Wolfe condition.
	//
	// The backtracking method finds the step length such that it
	// satisfies both the Armijo condition (BackTrackingArmijo) and the
	// following condition,
	//
	// |g(x + a * d)^T d| <= lbfgs_parameter_t::wolfe * |g(x)^T d|,
	//
	// where x is the current point, d is the current search direction,
	// and a is the step length.
	BackTrackingStrongWolfe = C.LBFGS_LINESEARCH_BACKTRACKING_STRONG_WOLFE
)

// List of errors returned from Minimize().
var (
	AlreadyMinimized        = errors.New("the initial variables already minimize the objective function")
	UnknownError            = errors.New("unknown error")
	LogicError              = errors.New("logic error")
	OutOfMemory             = errors.New("insufficient memory")
	Cancelled               = errors.New("the minimization process has been canceled")
	InvalidN                = errors.New("invalid number of variables specified")
	InvalidNSSE             = errors.New("invalid number of variables (for SSE) specified")
	InvalidXSSE             = errors.New("the array x must be aligned to 16 (for SSE)")
	InvalidEpsilon          = errors.New("invalid parameter Param.Epsilon specified")
	InvalidTestPeriod       = errors.New("invalid parameter Param.Past specified")
	InvalidDelta            = errors.New("invalid parameter Param.Delta specified")
	InvalidLineSearch       = errors.New("invalid parameter Param.LineSearch specified")
	InvalidMinStep          = errors.New("invalid parameter Param.MinStep specified")
	InvalidMaxStep          = errors.New("invalid parameter Param.MaxStep specified")
	InvalidFtol             = errors.New("invalid parameter Param.Ftol specified")
	InvalidWolfe            = errors.New("invalid parameter Param.Wolfe specified")
	InvalidGtol             = errors.New("invalid parameter Param.Gtol specified")
	InvalidXtol             = errors.New("invalid parameter Param.Xtol specified")
	InvalidMaxLineSearch    = errors.New("invalid parameter Param.MaxLineSearch specified")
	InvalidOrthantWise      = errors.New("invalid parameter Param.OrthantWiseC specified")
	InvalidOrthantWiseStart = errors.New("invalid parameter Param.OrthantWiseStart specified")
	InvalidOrthantWiseEnd   = errors.New("invalid parameter Param.OrthantWiseEnd specified")
	OutOfInterval           = errors.New("the line-search step went out of the interval of uncertainty")
	IncorrectTMinMax        = errors.New("a logic error occurred; alternatively, the interval of uncertainty became too small")
	RoundingError           = errors.New("a rounding error occurred; alternatively, no line-search step satisfies the sufficient decrease and curvature conditions")
	MinimumStep             = errors.New("the line-search step became smaller than Param.MinStep")
	MaximumStep             = errors.New("the line-search step became larger than Param.MaxStep")
	MaximumLineSearch       = errors.New("the line-search routine reaches the maximum number of evaluations")
	MaximumIteration        = errors.New("the algorithm routine reaches the maximum number of iterations")
	WidthTooSmall           = errors.New("relative width of the interval of uncertainty is at most Param.Xtol")
	InvalidParameters       = errors.New("a logic error (negative line-search step) occurred")
	IncreaseGradient        = errors.New("the current search direction increases the objective function value")
)

var errs = map[C.int]error{
	C.LBFGS_ALREADY_MINIMIZED:            AlreadyMinimized,
	C.LBFGSERR_UNKNOWNERROR:              UnknownError,
	C.LBFGSERR_LOGICERROR:                LogicError,
	C.LBFGSERR_OUTOFMEMORY:               OutOfMemory,
	C.LBFGSERR_CANCELED:                  Cancelled,
	C.LBFGSERR_INVALID_N:                 InvalidN,
	C.LBFGSERR_INVALID_N_SSE:             InvalidNSSE,
	C.LBFGSERR_INVALID_X_SSE:             InvalidXSSE,
	C.LBFGSERR_INVALID_EPSILON:           InvalidEpsilon,
	C.LBFGSERR_INVALID_TESTPERIOD:        InvalidTestPeriod,
	C.LBFGSERR_INVALID_DELTA:             InvalidDelta,
	C.LBFGSERR_INVALID_LINESEARCH:        InvalidLineSearch,
	C.LBFGSERR_INVALID_MINSTEP:           InvalidMinStep,
	C.LBFGSERR_INVALID_MAXSTEP:           InvalidMaxStep,
	C.LBFGSERR_INVALID_FTOL:              InvalidFtol,
	C.LBFGSERR_INVALID_WOLFE:             InvalidWolfe,
	C.LBFGSERR_INVALID_GTOL:              InvalidGtol,
	C.LBFGSERR_INVALID_XTOL:              InvalidXtol,
	C.LBFGSERR_INVALID_MAXLINESEARCH:     InvalidMaxLineSearch,
	C.LBFGSERR_INVALID_ORTHANTWISE:       InvalidOrthantWise,
	C.LBFGSERR_INVALID_ORTHANTWISE_START: InvalidOrthantWiseStart,
	C.LBFGSERR_INVALID_ORTHANTWISE_END:   InvalidOrthantWiseEnd,
	C.LBFGSERR_OUTOFINTERVAL:             OutOfInterval,
	C.LBFGSERR_INCORRECT_TMINMAX:         IncorrectTMinMax,
	C.LBFGSERR_ROUNDING_ERROR:            RoundingError,
	C.LBFGSERR_MINIMUMSTEP:               MinimumStep,
	C.LBFGSERR_MAXIMUMSTEP:               MaximumStep,
	C.LBFGSERR_MAXIMUMLINESEARCH:         MaximumLineSearch,
	C.LBFGSERR_MAXIMUMITERATION:          MaximumIteration,
	C.LBFGSERR_WIDTHTOOSMALL:             WidthTooSmall,
	C.LBFGSERR_INVALIDPARAMETERS:         InvalidParameters,
	C.LBFGSERR_INCREASEGRADIENT:          IncreaseGradient,
}

// Minimize minimizes the objective function using L-BFGS.
//
// x: A slice of variables. A client program can set default values
// for the optimization and receive the optimization result through
// this slice. This slice must be allocated by MakeSlice (and later
// freed by FreeSlice()) for libLBFGS built with SSE/SSE2 optimization
// routine enabled. The library built without SSE/SSE2 optimization
// does not have such a requirement.
//
// e: A function to provide function and gradient evaluations given a
// current values of variables.
//
// p: A function to receive the progress (the number of iterations,
// the current value of the objective function) of the minimization
// process. Use Silent when a progress report is not necessary.
//
// Returns the final value of the objective function and any errors
// from the optimization process.
func Minimize(x []Float, e EvaluateFunc, p ProgressFunc, param *Param) (fx Float, err error) {
	var cParam C.lbfgs_parameter_t
	setCParam(param, &cParam)
	ret := C.goLbfgs(C.int(len(x)), (*C.lbfgsfloatval_t)(&x[0]), (*C.lbfgsfloatval_t)(&fx), unsafe.Pointer(&evaluateProgress{e, p}), &cParam)
	if ret != C.LBFGS_SUCCESS {
		err = errs[ret]
	}
	return
}

// MakeSlice allocates a slice of n elements. This function allocates
// an array of variables for the convenience of Minimize() function;
// the function has a requirement for a variable slice when libLBFGS
// is built with SSE/SSE2 optimization routines. A user does not have
// to use this function for libLBFGS built without SSE/SSE2
// optimization.
//
// A slice returned from this function is NOT managed by Go runtime
// and must be manually freed with FreeSlice().
func MakeSlice(n int) []Float {
	xPtr := C.lbfgs_malloc(C.int(n))
	if xPtr == nil {
		panic(fmt.Sprintf("failed to allocate a memory block of %d floats", n))
	}
	return wrapSlice(xPtr, n)
}

// FreeSlice frees a slice returned by MakeSlice().
func FreeSlice(s []Float) {
	xPtr := (*C.lbfgsfloatval_t)(&s[0])
	C.lbfgs_free(xPtr)
}

// wrapSlice wraps regions of memory (e.g. from lbfgs_malloc) into a
// Go slice.
func wrapSlice(p *C.lbfgsfloatval_t, n int) (s []Float) {
	header := (*reflect.SliceHeader)(unsafe.Pointer(&s))
	header.Cap = n
	header.Len = n
	header.Data = uintptr(unsafe.Pointer(p))
	return s
}

// evaluateProgress packs the two functions needed for
// optimization. It is sent as the "instance" data when calling the C
// lbfgs() function.
type evaluateProgress struct {
	Evaluate EvaluateFunc
	Progress ProgressFunc
}

//export goLbfgsEvaluate
func goLbfgsEvaluate(p unsafe.Pointer, xPtr, gPtr *C.lbfgsfloatval_t, n C.int, step C.lbfgsfloatval_t) C.lbfgsfloatval_t {
	return C.lbfgsfloatval_t(
		(*evaluateProgress)(p).Evaluate(
			wrapSlice(xPtr, int(n)), wrapSlice(gPtr, int(n)), Float(step)))
}

//export goLbfgsProgress
func goLbfgsProgress(p unsafe.Pointer, xPtr, gPtr *C.lbfgsfloatval_t, fx, xnorm, gnorm, step C.lbfgsfloatval_t, n, k, ls C.int) C.int {
	return C.int(
		(*evaluateProgress)(p).Progress(
			wrapSlice(xPtr, int(n)), wrapSlice(gPtr, int(n)), Float(fx),
			Float(xnorm), Float(gnorm), Float(step), int(k), int(ls)))
}
