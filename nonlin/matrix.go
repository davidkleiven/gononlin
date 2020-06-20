package nonlin

// Matrix is a matrix type
type Matrix struct {
	Data         []float64
	Nrows, Ncols int
}

// At returns the value at position
func (m *Matrix) At(i, j int) float64 {
	return m.Data[i*m.Nrows+j]
}

// Set sets element (i, j) to v
func (m *Matrix) Set(i, j int, v float64) {
	m.Data[i*m.Nrows+j] = v
}

// ColView returns a view of a column
func (m *Matrix) ColView(i int) []float64 {
	return m.Data[i*m.Nrows : (i+1)*m.Nrows]
}

// DotColumn performs the dot product between vec and a column
func (m *Matrix) DotColumn(vec []float64, col int) float64 {
	res := 0.0
	for i := 0; i < m.Nrows; i++ {
		res += m.At(i, col) * vec[i]
	}
	return res
}

// NewMatrix returns a new matrix type
func NewMatrix(nrows, ncols int) Matrix {
	return Matrix{
		Data:  make([]float64, nrows*ncols),
		Nrows: nrows,
		Ncols: ncols,
	}
}
