package nonlin

import "math"

// InfNorm calculates the inifinity norm of the passed vector
func InfNorm(x []float64) float64 {
	value := 0.0
	for i := range x {
		if math.Abs(x[i]) > value {
			value = math.Abs(x[i])
		}
	}
	return value
}
