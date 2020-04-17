package nonlin

import (
	"math"
	"testing"
)

func TestBicStab(t *testing.T) {
	for i, test := range []struct {
		F        func(out []float64, x []float64)
		Init     []float64
		Solution []float64
		Tol      float64
	}{
		{
			F: func(out []float64, x []float64) {
				out[0] = math.Pow(x[0]-1.0, 2.0) * (x[0] - x[1])
				out[1] = math.Pow(x[1]-2.0, 5.0) * math.Cos(2.0*x[0]/x[1])
			},
			Init:     []float64{0.0, 3.0},
			Solution: []float64{1.0, 2.0},
			Tol:      1e-2,
		},
		{
			F: func(out []float64, x []float64) {
				out[0] = math.Exp(x[0]) + x[1] - 1.0
				out[1] = math.Exp(x[1]) + x[0] - 1.0
			},
			Init:     []float64{-0.3, -0.5},
			Solution: []float64{0.0, 0.0},
			Tol:      1e-2,
		},
		// Rosenbrock function
		{
			F: func(out []float64, x []float64) {
				out[0] = -2.0*(1.0-x[0]) - 400.0*(x[1]-x[0]*x[0])*x[0]
				out[1] = 200.0 * (x[1] - x[0]*x[0])
			},
			Init:     []float64{-2.0, 3.5},
			Solution: []float64{1.0, 1.0},
			Tol:      1e-4,
		},
	} {
		solver := NewtonBCGS{
			Maxiter:  1000,
			StepSize: 1e-3,
			Tol:      1e-7,
		}

		p := Problem{
			F: test.F,
		}
		res := solver.Solve(p, test.Init)

		if !res.Converged {
			t.Errorf("Test #%d did not converged", i)
		}

		//fmt.Printf("%v\n", res.X)
		for j := range res.X {
			if math.Abs(res.X[j]-test.Solution[j]) > test.Tol {
				t.Errorf("Test #%d:\nExpected\n%v\nGot\n%v\n", i, test.Solution, res.X)
				break
			}
		}

		if math.Abs(res.MaxF) > solver.Tol {
			t.Errorf("Test #%d: Did find a root. Inf. norm: %e\n", i, res.MaxF)
		}
	}
}
