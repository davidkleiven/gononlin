package nonlin

import (
	"math"
	"testing"
)

func TestJFSJ(t *testing.T) {
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
			Tol:      1e-6,
		},
		{
			F: func(out []float64, x []float64) {
				out[0] = math.Pow(x[0]-1.0, 2.0) * (x[0] - x[1])
				out[1] = math.Pow(x[1]-2.0, 5.0) * math.Cos(2.0*x[0]/x[1])
			},
			Init:     []float64{0.5, 2.0},
			Solution: []float64{1.0, 2.0},
			Tol:      1e-6,
		},
		{
			F: func(out []float64, x []float64) {
				out[0] = 4.0*x[0] - 2.0*x[1] + x[0]*x[0] - 3.0
				out[1] = -x[0] + 4*x[1] - x[2] + x[1]*x[1] - 3.0
				out[2] = -2.0*x[1] + 4.0*x[2] + x[2]*x[2] - 3.0
			},
			Init:     []float64{-1.5, 0.0, -1.5},
			Solution: []float64{1.0, 1.0, 1.0},
			Tol:      1e-6,
		},
		{
			F: func(out []float64, x []float64) {
				out[0] = 2.0/(1.0+x[0]*x[0]) + math.Sin(x[1]-1.0) - 1.0
				out[1] = math.Sin(x[0]-1.0) + 2.0/(1.0+x[1]*x[1]) - 1.0
			},
			Init:     []float64{0.7, 0.7},
			Solution: []float64{1.0, 1.0},
			Tol:      1e-6,
		},
		{
			F: func(out []float64, x []float64) {
				out[0] = math.Exp(x[0]) + x[1] - 1.0
				out[1] = math.Exp(x[1]) + x[0] - 1.0
			},
			Init:     []float64{-0.5, -0.5},
			Solution: []float64{0.0, 0.0},
			Tol:      1e-6,
		},
	} {
		p := Problem{
			F: test.F,
		}

		res := JFSJ(p, test.Init, nil)
		if !res.Converged {
			t.Errorf("Test #%d: Did not converge\n", i)
		}

		for j := range res.X {
			if math.Abs(res.X[j]-test.Solution[j]) > test.Tol {
				t.Errorf("Test #%d:\nExpected\n%v\nGot\n%v\n", i, test.Solution, res.X)
				break
			}
		}
	}
}
