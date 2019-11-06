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

## Building the paper
UNIX/Linux: type 'make' to build the paper/main.pdf file.  
Windows: use a Latex distribution to build the paper/main.pdf
