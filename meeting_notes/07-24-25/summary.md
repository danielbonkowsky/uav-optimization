Hey @Robert Heath,

Yesterday, we went over methods for optimizing flight path radius to maximize mean spectral efficiency. Because users are randomly distributed, mean SE is a function of random variables and Ibrahim helped me work through finding the expected value.

I'm working on cleaning up some of the notation, but then we can share the Overleaf with you so you can look at the in-depth problem formulation. In the end, we're considering joint optimization of three variables: the radius of the UAV's flight path, the location of the flight path, and the time division between the user-to-UAV and UAV-to-BS links.

Next week, I'm going to work on finding the optimizing the radius numerically with a Monte Carlo simulation, and on finding an analytical solution in the case of a single user. I'm leaving the location and timeshare values constant for now, but later we might consider an algorithm which iteratively optimizes all three.
