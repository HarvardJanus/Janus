# Related Work Notes
For each possibly related work, we write down some notes that we deem important to know.
We categorize all the related work into 
* _WHAT_ paper: what is interpretability, what is considered interpretable, etc.
* _WHY_ paper: why do we want interpretability of black-box algorithms
* _HOW_ paper: different approaches to interpret/explain black-box algorithms
* _TOOL_ paper: tools that assist explanation of ML algorithms

We also have the following works that do not have notes:
* The Mythos of Model Interpretability (_WHAT_)
* Understanding Black-Box Predictions via Influence Functions (_HOW_)
* European Union Regulations on Algorithmic Decision-making and A 'Right to Explanation' (_WHY_)
* LAMP : Data Provenance for Graph Based Machine Learning Algorithms through Derivative Computation (_HOW_, _TOOL_)
* 'Why Should I Trust You ?'' Explaining the Predictions of Any Classifier (_HOW_)
* Diagnosing Machine Learning Pipelines with Fine-grained Lineage (_TOOL_)

## Program As Black-Box Explanations
- Need to identify which interpretable representation would be suitable to convey the local behavior of the model in an accurate and succinct manner, and existing model-agnostic approaches have focused only on (sparse) linear models.
- There are many more other representations: additive models, decision rules, trees, sets, and lists, etc..
- Problem is no single one of these representations, by itself, provides the necessary tradeoff between expressivity and interpretability; We don't really understand this tradeoff; it is also likely that different representations are appropriate for different kinds of users and domains.
- This paper proposes using programs to explain the local behavior of black-box systems, essentially **decompiling** the local behavior of any black-box complex systems.
- Why is this a good idea?
	1. Programming languages are designed to capture complex behavior using a high-level syntax that is both succinct and intuitive
	2. Programs can represent any Turing-complete behavior
	3. They can represent arbitrary combinations of multiple of these representations
	4. It is also possible to trade off the expressivity and the comprehensibility of the program
	5. We can apply research in program/software analysis to evaluate various aspects of complex systems
- Most of these existing interpretable models retain their readability when written as programs.
- Recently, decision lists and decision sets have been introduced as more comprehensible representations than decision trees, while being much more powerful that linear models.
- The key challenges is to actually synthesize the appropriate program, i.e. to make sure it is both a **good approximation** of the black-box model, and is **readable**.
- They use program induction, where programs are synthesized automatically to match some desired goal. 
- The goal is to find the optimial program _p* = argmin L(f, p, Z, T) + S(p)_, where f is the black-box system, p is the program, Z is a number of random perturbations of x weighted by T by their similarity to x, and p is the complexity of the program. We try to explain the individual prediction x.
- The combinatorial optimization is solved using _simulated annealing_.