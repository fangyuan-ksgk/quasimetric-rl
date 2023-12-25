Conside the word 'Map' and the 'Path' on the 'Map'. Our intuition deceives us by adding a spatial symmetry: going from A to B is the exact reverse of going back to A from B. 

The flaw is obvious once we consider the temporal dimension. Traveling back to time A from time B is in NO WAY the easy reverse of traveling from time A to time B. Therefore, the symmetry between the 'Path' is not a general fact. 

Modeling Human decision in RL setting is essentially trying to find a 'Shortest-Path' between the initial state and the goal state, to evaluate distance metric requires acknowledging the asymmetry in this general 'distance' -- in the decision space (which includes temporal dimension). Quasi-metric RL does precisely this. 

Hand-design of reward function in RL is also a pain-in-the-ass. Generality is observed usually through letting the agent to merely explore instead of fixate on some sinle goal (which is usally poorly learned due to sparse reward). A destructive idea is to evalute the cost-to-go any state, from any state, Quasi-RL provides theoretical guarantee on picking the 'shortest-path' for any target state, starting from any state, without assuming symmetry. 

Great work done by MIT & UT Austin researchers!
@Tongzhou Wang @Antonio Torralba @Phillip Isola @Amy Zhang





