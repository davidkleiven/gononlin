# Burger's Equation

The Burger's equation is given given by

![Burgers](fig/burgers.svg)

The implicit Euler discretized form is given by

![BurgersDisc](fig/burgersDiscretized.svg)

*n* denotes the time step, and *i* denotes the spatial position. At each time step a non-linear
system of equations needs to be solved for the unknown <i>u<sub>i</sub><sup>n+1</sup></i>.
In the example file *main.go* this is done by using *NewtonKrylov* method. The solution is shown 
below.

![solution](fig/velocityProfile.svg)

After a certain time a shock develops.