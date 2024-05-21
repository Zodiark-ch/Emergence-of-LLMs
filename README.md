
## README
Emergence In LLMs (paper reference would come soon)
### Preparations
Datasets of OpenOrca and OpenHermes can be created by create_context_dataset.py 
Datasets for country, animal, and color categories exist in data\ 

Dependencies are listed in 'requirements.txt'.


### processing 
1. create datasets (only for natural sentences text) :
    run create_context_dataset.py 

2. Sample from LLMs :
    run generate_samples.py 

3. Estimating EI :
    run MINE_for_LLMs.py 

Note: I apologize for the numerous issues with the encapsulation of the code （I'll make it more refined soon）, which have led to many trivialities and manual adjustments that need to be made, such as obvious parameters like paths and token lengths. Because my experimental setup is severely limited— a single 3090 GPU and very restricted storage space (what can I say? -.-). As a result, I've had to incorporate many manual pluge-in throughout the experimental process (and I do not have much time to make it look more elegant). Fortunately, the code is very simple, so it shouldn't pose significant obstacles to replication and research. After all, what more can you expect from just a pair of hands and a 3090 GPU?
