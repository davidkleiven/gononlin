package nonlin

import "gonum.org/v1/gonum/mat"

// VecFunc is a type that describes a vector valued function that should
// be evaluated at point x. The result is placed in out.
type VecFunc func(out []float64, x []float64)

// DerivativeApprox is a type that is used to approximate the dot product between
// a search direction and the jacobian of a function
type DerivativeApprox struct {
	// Function of from which the jacobian should be calculated
	F VecFunc

	// Stepsize used to calculate the inner product between the jacobian and a vector v
	// J.dot(v) = (F(x + Eps*v) - F(x))/Eps
	Eps float64

	// X is the position at which the jacobian should be calculated
	X       []float64
	shifts  []float64
	weights []float64
}

// MulVecTo evaluates the dot product between the jacobian of the vector values function
// and a search direction v. x is the point at which the jacobian should be evaluated
// and the result is given in out
func (da *DerivativeApprox) MulVecTo(dst *mat.VecDense, trans bool, v mat.Vector) {
	if trans {
		panic("DerivApprox: Transpose not supported")
	}
	work := make([]float64, 2*len(da.X))
	xn := work[:len(da.X)]
	fVal := work[len(da.X):]

	// Make sure that out is cleared
	for i := range da.X {
		dst.SetVec(i, 0.0)
	}

	for i := range da.shifts {
		for j := range da.X {
			xn[j] = da.X[j] + da.shifts[i]*v.AtVec(j)*da.Eps
		}
		da.F(fVal, xn)

		for j := range da.X {
			dst.SetVec(j, dst.AtVec(j)+da.weights[i]*fVal[j]/da.Eps)
		}
	}
}

// NewCentral returns a derivative approximator of using the central
// differences.
// Jv = (F(x + eps*v) - F(x - eps*v)/(2*eps), where J is the jacobian
func NewCentral(F VecFunc, Eps float64) DerivativeApprox {
	return DerivativeApprox{
		F:       F,
		Eps:     Eps,
		shifts:  []float64{-1.0, 1.0},
		weights: []float64{-0.5, 0.5},
	}
}

// NewFourPoint returns a derivative approximator of using the central
// four point differences.
// Jv = (F(x-2*eps*v) - 8*F(x-eps*v) + 8*F(x+eps*v) - F(x+2*v))/(12*eps)
// where J is the jacobian
func NewFourPoint(F VecFunc, Eps float64) DerivativeApprox {
	return DerivativeApprox{
		F:       F,
		Eps:     Eps,
		shifts:  []float64{-2.0, -1.0, 1.0, 2.0},
		weights: []float64{1.0 / 12.0, -8.0 / 12.0, 8.0 / 12.0, -1.0 / 12.0},
	}
}

// NewSixPoint returns a derivative approximator of using the central
// six point differences.
// Jv = (-F(x-3*eps*v) + 9*F(x-2*eps*v) - 45*F(x-eps*v)
//
//	+45*F(x-eps*v) - 9*F(x+2*eps*v) + F(x+3*v))/(60*eps)
//
// where J is the jacobian
func NewSixPoint(F VecFunc, Eps float64) DerivativeApprox {
	return DerivativeApprox{
		F:       F,
		Eps:     Eps,
		shifts:  []float64{-3.0, -2.0, -1.0, 1.0, 2.0, 3.0},
		weights: []float64{-1.0 / 60.0, 9.0 / 60.0, -45.0 / 60.0, 45.0 / 60.0, -9.0 / 60.0, 1.0 / 60.0},
	}
}

// NewEightPoint returns a derivative approximator of using the central
// eight point differences.
// Jv = (-F(x-3*eps*v) + 9*F(x-2*eps*v) - 45*F(x-eps*v)
//
//	+45*F(x-eps*v) - 9*F(x+2*eps*v) + F(x+3*v))/(60*eps)
//
// where J is the jacobian
func NewEightPoint(F VecFunc, Eps float64) DerivativeApprox {
	return DerivativeApprox{
		F:       F,
		Eps:     Eps,
		shifts:  []float64{-4.0, -3.0, -2.0, -1.0, 1.0, 2.0, 3.0, 4.0},
		weights: []float64{3.0 / 840.0, -32.0 / 840.0, 168.0 / 840.0, -672.0 / 840.0, 672.0 / 840.0, -168.0 / 840.0, 32.0 / 840.0, -3.0 / 840.0},
	}
}
