Hi Professor, here is the meeting summary
- We discussed the key concepts in the relay paper you have shared. I explained the basic received signal model, the rate calculations, namely what Daniel needs to understand key concepts in wireless signal model. He will look into the paper shared by you and me in more detail.
- We briefly went over the paper I shared.

Hi All, I just wanted to summarize the setup we discussed in the last meeting, which the paper I shared also has without ground users and a single fixed wing UAV. We have two cells with two BS and there  is a dead zone at the center. The system has two fixed winged UAVs, a trajectory over the dead zone (which can be optimized too). There are many many parameters to calculate the rate for a particular user in the dead zone. The rate can depend on
UE location in the dead zone
The position of the UAVs (Trajectory)
The speed of UAVs
The time share between relay links
XXX (if any other)
We can start with a toy scenario. We can write the rate expression for the users in the dead zone and evaluate it for a single free variable for example varying the user locations in the dead zone. As we have the ability and intuition from one dimensional cases, we can decide an interesting problem for our paper, and tackle it.
The expressions we need is already available in Prof. Heath's paper (https://jwcn-eurasipjournals.springeropen.com/counter/pdf/10.1155/2009/618787.pdf) and the formulation of basic scenario for UAV trajectory, speed, and so on is in this paper (https://ieeexplore.ieee.org/abstract/document/7562472). We can follow decode and forward approach for a single directional link from BS to the ground users (downlink) and get the rate as a function of a single fixed variable.