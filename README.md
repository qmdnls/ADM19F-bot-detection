# Leveraging network topology for better fake account detection in social networks

This project is done as part of the course Advanced Data Mining in the fall of 2019 (currently a work-in-progress).

  
<p align="center"><img src="paper/FIG/social_graph.png?raw=true" width="350px" height="auto"></p>
  

**Abstract** Due to their popularity online social networks are a popular target for spam, scams, malware distribution and more recently state-actor propaganda. In this paper we review a number of recent approaches to fake account and bot classification. Based on this review and our experiments, we propose our own method which leverages the social graph's topology and differences in ego graphs of legitimate and fake user accounts to improve identification of the latter. We evaluate our approach against other common approaches on a real-world dataset of users of the social network Twitter.

Keywords: Fake account detection, social graph, network topology, neighborhood aggregation

## Requirements

Requires Python3 as well as a working installation of pip3 and venv for Python3 (on Ubuntu you can get this via "sudo apt install python3-venv"). Other requirements can be installed via the makefile.

## Building the project

Use the makefile to install the requirements. A Python3 virtualenv will be installed in ".env". 
See below for a description of what the makefile can do for you:

make help		show this message
make clean		remove intermediate files and clean the directory
make install		make a virtualenv in the base directory and install requirements
make paper.pdf		build the paper (recommended)
make pdf			compile the paper's .tex source using latexmk
make pdflatex		compile the paper's .tex source using pdflatex, might require multiple runs
make all.tar		creates a .tar ready for distribution

## Running the project and files in the project

In order to run the project start the virtualenv using the command "source .env/bin/activate". You can then run any of our project files using Python3. You can leave the virtual env using the command "deactivate".

Files:

baseline.py             Trains and evaluates the baseline models
rf.py                   Trains and evaluates our random forest classifier
rf\_minimal.py          Trains and evaluates a minimal random forest classifier using only three features
nn.py                   Trains and evaluates our neural network classifier (NF + GF)
nn\_baseline.py         Trains and evaluates the baseline neural network model
nn\_nf.py               Trains and evaluates our neural network classifier (only NF)
paper/                  Contains the LateX source files to build the paper
data/                   Contains datasets
data/train\_baseline.csv Dataset with the features retrieved using the Twitter API, used to train the baseline
data/train.csv          Dataset for neighborhood features
data/train\_graph.csv    Dataset containing all features used in the paper

## Dataset
4.6 million accounts
~13,000 labelled accounts

<p align="center"><img src="paper/FIG/indegrees.png?raw=true" width="550px" height="auto"></p>
<p align="center"><img src="paper/FIG/outdegrees.png?raw=true" width="550px" height="auto"></p>
<p align="center"><img src="paper/FIG/centrality.png?raw=true" width="550px" height="auto"></p>

## Aggregate features

Some interesting aggregate distributions over a node's predecessors/successors in the social graph that we found are below. For the full version please read our paper.

<p align="center"><img src="paper/FIG/reputation_pre.png?raw=true" width="550px" height="auto"></p>
<p align="center">Median reputation of predecessors</p>

<p align="center"><img src="paper/FIG/indegree_pre.png?raw=true" width="550px" height="auto"></p>
<p align="center">Median in-degree of predecessors</p>

<p align="center"><img src="paper/FIG/indegree_succ.png?raw=true" width="550px" height="auto"></p>
<p align="center">Median in-degree of successors</p>

## Building the paper
UNIX/Linux: type 'make' to build the paper/main.pdf file.  
Windows: use a Latex distribution to build the paper/main.pdf
