Today we made good progress towards identifying a good focus for our research. Literature already exists which addresses the problem of serving multiple ground users with a single fixed-wing UAV, but all of those papers have extremely simplified models for how the ground users are distributed.
- This paper assumes the users exist evenly spaced on a line
- This paper has the users evenly distributed on a square
- This paper assumes the users are always directly below the UAV
While the papers all make interesting optimizations, such as UAV trajectory or user scheduling, the user distributions make them a lot less realistic. UAV trajectory, energy efficiency, and user scheduling are all heavily impacted by how the ground users are distributed. We would like to look at different user distributions and see how the optimizations made in previous literature behave in more realistic scenarios.
1:43
Initially, we will look at the single user scenario in order to simply the user scheduling. We will consider different distributions and see how that affects the optimal circling radius of the UAV. Once we have the initial case worked out, we can move onto more users. This will require us to optimize user scheduling, and we can also investigate optimizing the center location of the UAV's flight path.
