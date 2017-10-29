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
* BadNets: Identifying Vulnerabilities in the Machine Learning Model Supply Chain (_WHY_)

## Program As Black-Box Explanations (_HOW_)
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

## ActiveClean: Interactive Data Cleaning For Statistical Modeling
- Many analysts do not approach cleaning as a one-shot pre-processing step, and instead, repeatedly alternate between cleaning and analysis, using the preliminary analysis on dirty data as a guide to help identify potential errors and design repairs
- However, for statistical models, iteratively cleaning some data and re-training on a partially clean dataset can lead to biases in even the simplest models.
- Statistical models face more dramatic sampling effects, and cleaning could result in a misleading trend, or even Simpson's paradox where aggregates over different populations of data can result in spurious relationships. Cleaning subsets of data to avoid the potentially expensive cleaning costs may be problematic.
- This paper focuses on two common operations that often require iterative cleaning: removing outliers and attribute transformation. Since these two types of errors do not affect the schema or leave any obvious signs of corruption (e.g., NULL values), model training may seemingly succeed–albeit with an inaccurate result.
- ActiveClean is a model training framework that allows for iterative data cleaning while preserving provable convergence properties. Applicable to *convex-loss models*. It is a *Machine-Learning-oriented iterative data cleaning framework*.
- Key insight: treat the cleaning and training iteration as a form of Stochastic Gradient Descent, an iterative optimization method, incrementally take gradient steps (cleaning a sample of records) towards the global solution (i.e., the clean model).
- Iterative data cleaning: the process of cleaning subsets of data, evaluating preliminary results, and then cleaning more data as necessary.
- Two goals:
	* Correctness. Will this clean-retrain loop converge to the intended result; ActiveClean provides an update algorithm with a monotone convergence guarantee where more cleaning is leads to a more accurate result in expectation.
	* Efficiency. How can we best make use of the existing data and analyst effort. Sampling approach is limited by the curse of dimensionality. Sampling further has a problem of scarcity, where errors that are rare may not show up in the sample.
- ActiveClean studies the problem of prioritizing modifications to *both features and labels* in existing examples. It handles incorrect values in any part (label or feature) of an example.
- The Update Problem is to update the model with a monotone convergence guarantee such that more cleaning implies a more accurate model.
- The prioritization problem is to select batches in such a way that the model converges in the fewest iterations possible.
- ActiveClean approximates the gradient from a sample of newly cleaned data, using a variation of gradient descent.
- Optimizations include:
	* If we know that a record is likely to be clean, we can move it from dirty set to clean set without having to sample it.
	* We can set the sampling probabilities p(.) to favor records that are likely to affect the model.
- Model is guaranteed to converge essentially with a rate proportional to the inaccuracy of the sample-based estimate.
- It is well-known that even for an arbitrary initialization, SGD makes significant progress in less than one epoch (a pass through the entire dataset).
- In the non-convex setting, ActiveClean will converge to the closest locally optimal value to the dirty model which is how we initialize ActiveClean.
- If corrupted records are relatively rare, sampling might be very inefficient -> error detection techniques: exact rule-based detector and approximate adaptive detector
- The sampling algorithm is designed to include records in each batch that are most valuable to the analyst’s model with a higher probability. The optimal sam- pling problem is defined as a search over sampling distributions to find the minimum variance sampling distribution.
- This sampling distribution prioritizes records with higher gradients, i.e., make a larger impact during optimization. The challenge is that this particular optimal distribution depends on knowing the clean value of a record.
- As the analyst cleans more data, we can build a model for how cleaned data relates to dirty data. By using the detector from the previous section to estimate the impact of data cleaning, we show that we can estimate the cleaned values.

## An Empirical Evaluation of the Comprehensibility of Decision Table, Tree, and Rule-Based Predictive Models

## Enslaving the algorithm: from a `right to an explanation' to a `right to better decisions'?
- Because ML algorithms are trained on historical data, they risk replicating unwanted historical patterns of unfairness and/or discrimination (e.g., gender equality, racial policing, luxury advertising).
- A severe obstacle to challenging such systems is that outputs, which translate with or without human intervention to decisions, are made not by humans or even human-legible rules, but by less scrutable mathematical techniques. This opacity has been described as creating a ``black box'' society.
- Two related provisions in Data Protection Directive (DPD):
	1. A ``significant'' decision could not be based solely on automated data processing.
	2. Users have rights to obtain information about whether and how their particular personal data was processed, i.e., they have the specific right to obtain ``knowledge of the logic involved in any automatic processing'' of their data.

## Rationalizing Neural Predictions
- Many recent advances in NLP problems have come from formulating and training expressive and elaborate neural models. The gains in accuracy have come at the cost of interpretability since complex neural models offer little transparency concerning their inner workings.
- This paper incorporates rationale generation as an integral part of the overall learning problem, but only limited to extractive (as opposed to abstractive) rationales.
- In NLP, rationales are simply subsets of the words from the input text that satisfy two key properties.
	1. Selected words represent short and coherent pieces of text.
	2. Selected words must alone suffice for prediction as a substitute of the original text
- In most practical applications, rationale generation must be learned entirely in an unsupervised manner.
-  The model is composed of two modular components: the generator and the encoder. Our generator specifies a distribution over possible rationales (extracted text) and the encoder maps any such text to task specific target values. They are trained jointly to minimize a cost function that favors short, concise rationales while enforcing that the rationales alone suffice for accurate prediction.
- Two domain for experimentation where ambiguity of what counts as a rationale in some contexts and performance of the task of selecting rationales is minimized: 
	1. Multi-aspect sentiment analysis
	2. The problem of retrieving similar questions
- Beyond learning to understand or further constrain the network to be directly interpretable, one can estimate interpretable proxies that approximate the network (e.g., if-then rule, decision trees from trained networks). 
- Ribeiro et al. (2016) propose a model-agnostic framework where the proxy model is learned only for the target sample (and its neighborhood) thus ensuring locally valid approximations.
- Attention based models offer another means to explicate the inner workings of neural models.
- In extractive rationale generation, our goal is to select a subset of the input sequence as a rationale.
- We can think of the generator as a tagging model where each word in the input receives a binary tag pertaining to whether it is selected to be included in the rationale.
- In our case, the generator is probabilistic and specifies a distribution over possible selections.
- The rationale is introduced as a latent variable, a constraint that guides how to interpret the input sequence.
- The encoder and generator are trained jointly, in an end-to-end fashion so as to function well together.
- Rationale for a given sequence x can be equivalently defined in terms of binary variables {z1,..., zl} where each zt = 0 or 1 indicates whether word xt is selected or not. This is because the rational is a subset of text from the original input.
- The generator can either be modeled as conditionally independent, i.e., each z is independent of other z's given the input string x. However, the model is unable to select phrases or refrain from selecting the same word again if already chosen.
- To solve the previous problem, we can use a dependent selection of words. Also introduce another hidden state s whose role is to couple the selections.
- Our generator and encoder are learned jointly to interact well but they are treated as independent units for modularity.
- They efficiently sample from the generator using doubly stocahstic gradient decent, employing a REINFORCE-style algorithm. Additional constraints on the generator output can be helpful in alleviating problems of exploring potentially a large space of possible rationales in terms of their interaction with the encoder.
- The model they employ is **recurrent convolution NN (RCNN)**, which is shown to work remarkably in clas- sification and retrieval applications, compared to e.g. CNNs and LSTMs.
- The model is evaluated on two NLP applications:
	1. Multi-aspect sentiment analysis on product reviews
	2. Similar text retrieval on AskUbuntu question answering forum




























