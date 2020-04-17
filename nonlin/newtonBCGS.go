package nonlin

import (
	"math"
)

// NewtonBCGS solve a non-linear system of equation using the conjugate gradient
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
type NewtonBCGS struct {
	// Tolerance for convergence
	Tol float64

	// Stepsize used in finite difference approximation of the Jacobian
	StepSize float64

	// Maximum number of iterations
	Maxiter int
}

func (bcgs *NewtonBCGS) solveDX(p Problem, x []float64, f0 []float64) []float64 {
	work := make([]float64, 8*len(x))
	dx := work[:len(x)]
	r := work[len(x) : 2*len(x)]
	xn := work[2*len(x) : 3*len(x)]
	f1 := work[3*len(x) : 4*len(x)]
	g := work[4*len(x) : 5*len(x)]
	r0 := work[5*len(x) : 6*len(x)]
	pVec := work[6*len(x) : 7*len(x)]
	s := work[7*len(x):]

	for i := range r {
		r0[i] = -f0[i]
	}
	copy(r, r0)

	rho := 1.0
	alpha := 1.0
	omega := 1.0
	for iter := 0; iter < len(x); iter++ {
		// Calculate new value for rho
		rhoNext := 0.0
		for i := range r0 {
			rhoNext += r0[i] * r[i]
		}
		beta := rhoNext * alpha / (rho * omega)
		for i := range pVec {
			pVec[i] = r[i] + beta*(pVec[i]-omega*g[i])
		}

		eps := bcgs.StepSize
		for i := range xn {
			xn[i] = x[i] + eps*pVec[i]
		}
		p.F(f1, xn)
		for i := range g {
			g[i] = (f1[i] - f0[i]) / eps
		}

		alpha = 0.0
		for i := range g {
			alpha += r0[i] * g[i]
		}

		alpha = rhoNext / alpha

		for i := range pVec {
			dx[i] += alpha * pVec[i]
		}

		if math.Abs(alpha)*InfNorm(pVec) < bcgs.Tol {
			return dx
		}

		for i := range s {
			s[i] = r[i] - alpha*g[i]
		}

		for i := range xn {
			xn[i] = x[i] + eps*s[i]
		}
		p.F(f1, xn)
		t := f1 // Overwrite t into f1, they are not needed simultaneously
		for i := range t {
			t[i] = (f1[i] - f0[i]) / eps
		}

		tDots := 0.0
		tDott := 0.0
		for i := range t {
			tDots += s[i] * t[i]
			tDott += t[i] * t[i]
		}

		omega = tDots / tDott

		for i := range dx {
			dx[i] += omega * s[i]
		}

		if math.Abs(omega)*InfNorm(s) < bcgs.Tol {
			return dx
		}

		for i := range r {
			r[i] = s[i] - omega*t[i]
		}

		rho = rhoNext
	}
	return dx
}

// Solve solves the non-linear system of equations. The method terminates when
// the inifinity norm of F + the inifinity norm of dx is less than the tolerance.
// dx is the change in x between two sucessive iterations.
func (bcgs *NewtonBCGS) Solve(p Problem, x []float64) Result {
	work := make([]float64, 2*len(x))
	f0 := work[:len(x)]
	f1 := work[len(x):]

	for iter := 0; iter < bcgs.Maxiter; iter++ {
		p.F(f0, x)
		dx := bcgs.solveDX(p, x, f0)

		if InfNorm(f0)+InfNorm(dx) < bcgs.Tol {
			return Result{
				X:         x,
				Converged: true,
				MaxF:      InfNorm(f0),
				F:         f0,
			}
		}

		// Update x
		for i := range dx {
			x[i] += dx[i]
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
			x[i] += (lamb - 1.0) * dx[i] // Subtract 1.0*dx since we already added that before
		}
	}
	return Result{
		X:         x,
		Converged: false,
		MaxF:      InfNorm(f0),
		F:         f0,
	}
}

// StepTuner is a type that automatically chooses a good stepsize, based on the history
// of moves
type StepTuner struct {
}
