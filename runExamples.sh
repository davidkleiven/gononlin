echo "Running Burger's equation"
go build examples/burger/main.go
./main -dt=0.01 -nodes=100 -steps=10 -out=velocity.csv
rm main velocity.csv