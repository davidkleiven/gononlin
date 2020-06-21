// main shows how one can use NewtonKrylov to solve the Burgers equation using
// implicit euler method
package main

import (
	"encoding/csv"
	"flag"
	"fmt"
	"math"
	"os"

	"github.com/davidkleiven/gononlin/nonlin"
)

func main() {
	dt := flag.Float64("dt", 0.001, "Timestep")
	nodes := flag.Int("nodes", 64, "Number of grid points")
	steps := flag.Int("steps", 10, "Number of timesteps to perform")
	fname := flag.String("out", "velocity.csv", "Solution will be written to this text file")
	flag.Parse()

	N := *nodes
	u := make([]float64, N)
	uCpy := make([]float64, N)
	for i := range u {
		x := float64(i) / float64(N)
		u[i] = math.Exp(-math.Pow((x-0.5)/0.1, 2.0))
	}
	copy(uCpy, u)

	problem := nonlin.Problem{
		F: func(out []float64, x []float64) {
			for i := range x {
				next := (i + 1) % N
				prev := i - 1
				if prev < 0 {
					prev += N
				}
				out[i] = x[i] - uCpy[i] + *dt*x[i]*(x[next]-x[prev])*0.5
			}
		},
	}

	solver := nonlin.NewtonKrylov{
		Tol:      1e-7,
		StepSize: 1e-3,
		Maxiter:  10000,
	}

	csvfile, err := os.Create(*fname)
	if err != nil {
		panic(err)
	}
	csvwriter := csv.NewWriter(csvfile)
	for i := 0; i < *steps; i++ {
		copy(uCpy, u)
		res := solver.Solve(problem, u)
		copy(u, res.X)

		row := make([]string, len(u))
		for j := range u {
			row[j] = fmt.Sprintf("%f", u[j])
		}
		csvwriter.Write(row)
	}
	csvwriter.Flush()
	csvfile.Close()
}
