package nonlin

import (
	"math"
	"testing"
)

func TestDerivativeApprox(t *testing.T) {
	x := []float64{1.0, 2.0}
	v := []float64{0.2, 0.3}

	f := func(out []float64, x []float64) {
		out[0] = x[0] * x[1]
		out[1] = x[0] + x[1]
	}

	// Jacobian is given by [[x[1], x[0]], [1, 1]]
	expect := []float64{x[1]*v[0] + x[0]*v[1], v[0] + v[1]}

	for i, approx := range []DerivativeApprox{
		NewCentral(f, 1e-3),
		NewFourPoint(f, 1e-3),
		NewSixPoint(f, 1e-3),
		NewEightPoint(f, 1e-3),
	} {
		res := make([]float64, 2)
		approx.Eval(x, v, res)

		tol := 1e-8
		for j := range expect {
			if math.Abs(expect[j]-res[j]) > tol {
				t.Errorf("Test #%d: Expected\n%v\nGot\n%v\n", i, expect, res)
				break
			}
		}
	}
}
