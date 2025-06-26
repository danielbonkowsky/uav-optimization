![system model](06-26-25/system-model.png)
Today we developed a system model that I'll use to do simulations, and that we'll use in the future to do more theoretical optimizations. The important aspects are
- Using a frequency-division duplexing system
- Decode-forward (T, 1-T) type time allocation between GU-Relay, Relay-BS links
- Considering multiple ground users in different distributions
  - Initially uniform, but later more exciting
- Considering uplink from ground user to UAV to base station
- The bits/s received by the UAV will be
![equation 1](06-26-25/eqn1.png)
- The bits/s received by the base station will be
![equation 2](06-26-25/eqn2.png)

With this problem set up, I'm planning to generate some plots optimize the UAV's velocity to maximize the minimum rate achieved over all the users.
