package nonlin

import (
	"math"
)

// Problem to be solved
type Problem struct {
	// F is a function where we seek the solution of F(x) = 0. The value of the function
	// should be given to out. out will always be the same length as x. Note that F must
	// not modify x.
	F func(out, x []float64)
}

// Result is a result type returned by the solvers.
type Result struct {
	// X is the solution vector at termination
	X []float64

	// F is the value the function at termination
	F []float64

	// MaxF is the infinity norm of F
	MaxF float64

	// Converges is True if the convergence criteria are met, otherwise it is False
	Converged bool
}

// Settings is a type that holds parameters required by jfsj algorithm
type Settings struct {
	Tol     float64
	Maxiter int
}

func defaultSettings() *Settings {
	return &Settings{
		Tol:     1e-10,
		Maxiter: 100000,
	}
}

// identify returns the diagonal of the identity matrix
func identity(n int) []float64 {
	I := make([]float64, n)
	for i := range I {
		I[i] = 1.0
	}
	return I
}

func updateG(G []float64, dF []float64, dx []float64) {
	dFDotDx := 0.0
	dFGdF := 0.0
	dF4 := 0.0
	for i := range dx {
		dFDotDx += dF[i] * dx[i]
		dFGdF += dF[i] * G[i] * dF[i]
		dF4 += dF[i] * dF[i] * dF[i] * dF[i]
	}

	for i := range G {
		G[i] += (dFDotDx - dFGdF) * dF[i] * dF[i] / dF4
	}
}

// JFSJ applies the jacobian free singular jacobian algorithm to solve
// the non-linear system of equations
func JFSJ(p Problem, x []float64, settings *Settings) Result {
	G := identity(len(x))
	if settings == nil {
		settings = defaultSettings()
	}

	work := make([]float64, 3*len(x))
	p.F(work[:len(x)], x)

	var result Result
	for i := 0; i < settings.Maxiter; i++ {
		// On even iterations F is placed in work[:len(x)], on odd iterations
		// F is placed in work[len(x):2*len(x)]
		F := work[:len(x)]
		if i%2 == 1 {
			F = work[len(x) : 2*len(x)]
		}

		infNormDx := 0.0
		for j := range x {
			dx := -G[j] * F[j]
			x[j] += dx
			work[2*len(x)+j] = dx
			if math.Abs(dx) > infNormDx {
				infNormDx = math.Abs(dx)
			}
		}

		dFArray := work[:len(x)]
		if i%2 == 0 {
			F = work[len(x) : 2*len(x)]
			p.F(F, x)
		} else {
			F = work[:len(x)]
			dFArray = work[len(x) : 2*len(x)]
			p.F(F, x)
		}

		infNormDF := 0.0
		infNormF := 0.0
		for j := range x {
			dF := math.Abs(work[j] - work[j+len(x)])
			if dF > infNormDF {
				infNormDF = dF
			}

			if math.Abs(F[j]) > infNormF {
				infNormF = math.Abs(F[j])
			}
		}

		if infNormDx+infNormF < settings.Tol {
			result.X = x
			result.Converged = true
			result.F = F
			return result
		}

		if infNormDF > settings.Tol {
			sign := math.Pow(-1.0, float64(i))
			// Overwrite the part of the work array where F_k is stored with dF_k = F_{k+1} - F_k
			for j := range x {
				dFArray[j] = sign * (work[len(x)+j] - work[j])
			}
			updateG(G, dFArray, work[2*len(x):])
		}
	}

	result.X = x
	result.Converged = false
	result.F = work[:len(x)]
	return result
}
