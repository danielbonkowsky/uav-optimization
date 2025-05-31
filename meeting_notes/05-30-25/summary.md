Today we did more literature review, trying to narrow down what part of the fixed-wing UAV base station scenario to optimize for/explore.

- [This paper]( https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9057370) looks at a single fixed wing UAV serving multiple ground users and optimizes trajectory to achieve a theoretical minimum rate for each user
- These papers ([1](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=6965778), [2](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=7562472)) look at a fixed transmitter and receiver with the UAV in the middle acting as a relay. They optimize for energy efficiency and rate respectively

At the moment, we are still unsure what avenue to explore in this problem. We were thinking of trying to optimize flight path radius based on the distribution of ground users, and maybe considering the ground users as mobile actors as an extension. There's not very many papers that try to optimize coverage for fixed-wing UAVs, so there's a lot of variables we can try to look at.
