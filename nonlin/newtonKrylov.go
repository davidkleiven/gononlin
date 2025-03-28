package nonlin

import (
	"log"
	"math"

	"gonum.org/v1/exp/linsolve"
	"gonum.org/v1/gonum/mat"
)

// NewtonKrylov solve a non-linear system of equation using the conjugate gradient
// Jacobian Free Newton Krylov method. It is Newton based method, but the Jacobian
// is never explicitly constructed, since we only needs its action onto a vector
// (e.g. Jv, where J is the jacobian matrix and v is a vector). This, makes the
// method memory efficient. The jacobian matrix along a particular direction is however
// estimates using finithe differences.
// Jv = (F(x + eps*v) - F(x))/eps
// where eps is the Stepsize. F is the non-linear (vector-valued) function of which
// one seeks the folusion F(x) = 0. The method will try at most Maxiter Newton steps
// before it terminates. In case, the maximum number of iterations is reached the Converged
// attribute of the Result struct will be false.
type NewtonKrylov struct {
	// Tolerance for convergence. The method terminates when InfNorm(F(x)) + InfNorm(dx)
	// is less than this value. Here, dx represents the amound the position changed on the
	// current Newton step
	Tol float64

	// Stepsize used in finite difference approximation of the Jacobian.
	StepSize float64

	// Maximum number of outer iterations (e.g. the number of Newton steps). If not given
	// a default value of 10000 will be used
	Maxiter int

	// Number of points used to approximate the jacobian. If not given (or set to zero)
	// a four point stencil will be used
	Stencil int

	// InnerMethod is the linear solver used in the "inner" iterations of the NewtonKrylov
	// method. If not given, GMRES with the default parameters in Gonum will be used
	InnerMethod linsolve.Method

	// Settings passed to linsolve.Iterative.
	// See https://godoc.org/github.com/gonum/exp/linsolve#Iterative
	// for further details
	InnerSettings *linsolve.Settings
}

// Solve solves the non-linear system of equations. The method terminates when
// the inifinity norm of F + the inifinity norm of dx is less than the tolerance.
// dx is the change in x between two successive iterations.
func (nk *NewtonKrylov) Solve(p Problem, x []float64) (Result, error) {
	var deriv DerivativeApprox
	switch nk.Stencil {
	case 0, 4:
		deriv = NewFourPoint(p.F, nk.StepSize)
	case 2:
		deriv = NewCentral(p.F, nk.StepSize)
	case 6:
		deriv = NewSixPoint(p.F, nk.StepSize)
	case 8:
		deriv = NewEightPoint(p.F, nk.StepSize)
	default:
		deriv = NewFourPoint(p.F, nk.StepSize)
	}
	deriv.X = x

	if nk.InnerMethod == nil {
		nk.InnerMethod = &linsolve.GMRES{}
	}

	if nk.Maxiter == 0 {
		nk.Maxiter = 10000
	}

	work := make([]float64, 2*len(x))
	f0 := work[:len(x)]
	f1 := work[len(x):]
	b := mat.NewVecDense(len(f0), nil)

	for iter := 0; iter < nk.Maxiter; iter++ {
		p.F(f0, x)
		for i := range f0 {
			b.SetVec(i, -f0[i])
		}
		res, err := linsolve.Iterative(&deriv, b, nk.InnerMethod, nk.InnerSettings)
		if err != nil {
			log.Printf("[ERROR] NewtonKrylov: %s\n", err)
			return Result{}, err
		}

		if InfNorm(f0)+mat.Norm(res.X, math.Inf(1)) < nk.Tol {
			return Result{
				X:         x,
				Converged: true,
				MaxF:      InfNorm(f0),
				F:         f0,
			}, nil
		}

		// Update x
		for i := range x {
			x[i] += res.X.AtVec(i)
		}

		p.F(f1, x)
		g0 := 0.0
		g1 := 0.0
		for i := range f0 {
			g0 += 0.5 * f0[i] * f0[i]
			g1 += 0.5 * f1[i] * f1[i]
		}
		lamb := g0 / (g1 + g0)
		lambMin := 0.1 // Numerical Recipies suggests 0.1 as lower cutoff
		if lamb < lambMin {
			lamb = lambMin
		}
		for i := range x {
			x[i] += (lamb - 1.0) * res.X.AtVec(i) // Subtract 1.0*dx since we already added that before
		}
	}
	return Result{
		X:         x,
		Converged: false,
		MaxF:      InfNorm(f0),
		F:         f0,
	}, nil
}
