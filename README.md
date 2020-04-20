# GoNonLin

[![Build Status](https://travis-ci.org/davidkleiven/gononlin.svg?branch=master)](https://travis-ci.org/davidkleiven/gononlin)

Package for solving non-linear systems of equations. The package implements Jacobian-Free Newton
Krylov method, where the Jacobian is approximated via finite differences. However, since the Krylov space methods only require the action of the Jacboian on a search direction *v*, the full
Jacobian matrix is never explicitly stored, which makes the technique very memory efficient.

# Example

```go
import (
	"fmt"
	"math"

	"github.com/davidkleiven/gononlin/nonlin"
)

func ExampleNewtonBCGS() {
	// This example shows how one can use NewtonBCGS to solve the
	// system of equations
	// (x-1)^2*(x - y) = 0
	// (x-2)^3*cos(2*x/y) = 0

	problem := nonlin.Problem{
		F: func(out, x []float64) {
			out[0] = math.Pow(x[0]-1.0, 2.0) * (x[0] - x[1])
			out[1] = math.Pow(x[1]-2.0, 3.0) * math.Cos(2.0*x[0]/x[1])
		},
	}

	solver := nonlin.NewtonBCGS{
		// Maximum number of Newton iterations
		Maxiter: 1000,

		// Stepsize used to appriximate jacobian with finite differences
		StepSize: 1e-2,

		// Tolerance for the solution
		Tol: 1e-7,
	}

	x0 := []float64{0.0, 3.0}
	res := solver.Solve(problem, x0)
	fmt.Printf("Root: (x, y) = (%.2f, %.2f)\n", res.X[0], res.X[1])
	fmt.Printf("Function value: (%.2f, %.2f)\n", res.F[0], res.F[1])

	// Output:
	//
	// Root: (x, y) = (1.00, 2.00)
	// Function value: (-0.00, 0.00)
}
```