# Leveraging network topology for better fake account detection in social networks

This project is done as part of the course Advanced Data Mining in the fall of 2019 (currently a work-in-progress).

  
<p align="center"><img src="paper/FIG/social_graph.png?raw=true" width="350px" height="auto"></p>
  

**Abstract** Due to their popularity online social networks are a popular target for spam, scams, malware distribution and more recently state-actor propaganda. In this paper we review a number of recent approaches to fake account and bot classification. Based on this review and our experiments, we propose our own method which leverages the social graph's topology and differences in ego graphs of legitimate and fake user accounts to improve identification of the latter. We evaluate our approach against other common approaches on a real-world dataset of users of the social network Twitter.

Keywords: Fake account detection, social graph, network topology, neighborhood aggregation

## Dataset
4.6 million accounts  
~13,000 labelled accounts

<p align="center"><img src="fig/indegrees.png?raw=true" width="550px" height="auto"></p>
<p align="center"><img src="fig/outdegrees.png?raw=true" width="550px" height="auto"></p>
<p align="center"><img src="fig/centrality.png?raw=true" width="550px" height="auto"></p>

## Aggregate features

Some interesting aggregate distributions over a node's predecessors/successors in the social graph that we found are below. For the full version please read our paper.

<p align="center">Median reputation of predecessors:</p>
<p align="center"><img src="paper/FIG/reputation_pre.png?raw=true" width="550px" height="auto"></p>

<p align="center">Median out-degree of predecessors:</p>
<p align="center"><img src="paper/FIG/indegree_pre.png?raw=true" width="550px" height="auto"></p>

<p align="center">Median out-degree of predecessors:</p>
<p align="center"><img src="paper/FIG/indegree_succ.png?raw=true" width="550px" height="auto"></p>


## Building the paper
UNIX/Linux: type 'make' to build the paper/main.pdf file.  
Windows: use a Latex distribution to build the paper/main.pdf
