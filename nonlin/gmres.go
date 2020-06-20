package nonlin

import "math"

// GMRES is a type that uses Generalize Minimzed Residual as an "inner" solver
// for in the NewtonSolver
type GMRES struct {
	StepSize float64
	MaxIter  int
	Tol      float64
}

// Solve solves the problem J*X = -F, which arises from the Newton's method when solving the non-linear
// system of non-linear equations F(x) = 0. J is the jacobian J_ij = dF_i/dx_j.
// The method follows closely the procedure on wikipedia. f0 contains the current value of the function
// (e.g. p.Eval(f0, x))
func (gmres *GMRES) Solve(p Problem, x []float64, f0 []float64, deriv DerivativeApprox) []float64 {
	work := make([]float64, 5*len(x))
	r0 := work[:len(x)]
	sn := work[len(x) : 2*len(x)]
	cs := work[2*len(x) : 3*len(x)]
	e1 := work[3*len(x) : 4*len(x)]
	beta := work[4*len(x):]
	e1[0] = 1.0

	Q := NewMatrix(len(x), gmres.MaxIter)
	H := NewMatrix(len(x), gmres.MaxIter)

	for i := range f0 {
		r0[i] = -f0[i]
	}
	resNorm := InfNorm(r0)
	beta[0] = resNorm
	for i := 0; i < len(x); i++ {
		Q.Set(i, 0, r0[i]/resNorm)
	}

	for iter := 0; iter < gmres.MaxIter; iter++ {
		arnoldi(deriv, x, &Q, iter, H.ColView(iter), Q.ColView(iter+1))
		applyGivensRotation(H.ColView(iter), cs, sn, iter)

		// Update residual vector
		beta[iter+1] = -sn[iter] * beta[iter]
		beta[iter] = cs[iter] * beta[iter]
		resNorm = math.Abs(beta[iter])

		if resNorm < gmres.Tol {
			return x
		}

		// TODO: Solve system Hy = beta

		// TODO: Update x
	}
	return x
}

// arnoldi performs arnoldi iterations
func arnoldi(deriv DerivativeApprox, x []float64, Q *Matrix, k int, h []float64, q []float64) {
	deriv.Eval(x, Q.ColView(k), q)
	for i := 0; i < k; i++ {
		h[i] = Q.DotColumn(q, i)
		for j := 0; j < len(q); j++ {
			q[j] = q[j] - h[i]*Q.At(j, i)
		}
	}
	h[k+1] = L2Norm(q)

	for i := range q {
		q[i] /= h[k+1]
	}
}

func givensRotation(v1 float64, v2 float64) (float64, float64) {
	if math.Abs(v1-1.0) < 1e-6 {
		return 0.0, 1.0
	}
	t := v1*v1 + v2*v2
	cs := math.Abs(v1) / t
	sn := cs * v2 / v1
	return cs, sn
}

func applyGivensRotation(h []float64, cs []float64, sn []float64, k int) {
	for i := 0; i < k-1; i++ {
		temp := cs[i]*h[i] + sn[i]*h[i+1]
		h[i+1] = -sn[i]*h[i] + cs[i]*h[i+1]
		h[i] = temp
	}

	// Update sine and cosine values for rotaitons
	cs[k], sn[k] = givensRotation(h[k], h[k+1])

	// eliminate H(i+1, i)
	h[k] = cs[k]*h[k] + sn[k]*h[k+1]
	h[k+1] = 0.0
}
