# Developing an intuitive causal graphical model GUI for students and junior researchers
Name: Jonah King  
MSc Program: Computer Science MSc  

# Papers:


## An Overview of the Methodologies of Causal Discovery:
Causal discovery: extracts causal direction between variables using data using either constraint based or score based methods  
Structural causal models (SCM) are made of three things  
1. Exogenous variables: determined by mechanisms extrinsic to the system interested in  
2. Endogenous variables: determined by variables intrinsic to the system  
3. Structural equations: relates value of target variable X with values of those with directed edges that end at X

Three possibilities in a causal system  
1. X causes Y  
2. Y directly or indirectly causes X  
3. There is a common cause, Z, of both X and Y  

## Methods and Tools for Causal Discovery and Causal Inference:
### Datasets:
**LUCAS (lung cancer simple set)**
- synthetic dataset with 12 binary variables, 2000 instances and 12 causal relationships  
- uses potential factors to develop lung cancer and other related factors  
- test if causal discovery methods recover the true causal structure

**SIDO (simple drug operation mechanism)**
- dataset with 4932 variables adn 12,678 instances  
- used to discover the causes for molecular activity in the descriotors
            
**Sachs**
- represents proteins and phospholipids present in human immune system cells  
- 11 discrete variables, 5400 instances, 17 causal relationships  
- aim to discovery connections between the molecules without needing physical intervention on them in a lab  
