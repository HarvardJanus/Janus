## Classification of Approaches Used in ML Interpretability
- Transparency (as oppose to opacity or blackbox-ness): approaches that answer the question, _how does the model work?_
	1. Explain how the entire model works or **simulatability**: use an interpretable model to begin with, generates a model that “can be readily presented to the user with visual or textual artifacts”. For example, sparse linear models, decision trees, rule-based systems, however, they are actually not intrinsically interpretable, as sufficiently high-dimensional models, unwieldy rule lists, and deep decision trees could all be considered less transparent than comparatively compact neural networks.

	2. Explain individual components (e.g., parameters) or **decomposability**: inputs themselves be individually interpretable, disqualifying some models with highly engineered or anonymous features.

	3. Explain the training algorithm or **algorithmic transparency**: this is difficult, esp. for deep learning method. 

	*Even humans exhibit none of these forms of transparency.*

- Post-hoc explanations: approaches that answer the question, _what else can the model tell me?_ (While post-hoc interpretations often do not elucidate precisely how a model works, they may nonetheless confer useful information for practitioners and end users of machine learning. Humans can exhibit such transparency. **We can interpret opaque models after-the-fact, without sacrificing predictive performance.**)
3.2.1.)
	1. Natural language/text explanations:
		-- e.g., train one model to generate predictions and a separate model, such as a recurrent neural network language model, to generate an explanation
	2. Visualization of learned representations or models:
		-- e.g., visualize high-dimensional distributed representations with t-SNE, a technique that renders 2D visualizations in which nearby data points are likely to appear close together.
	3. Local explanations: explaining what a neural network depends on *locally*
		-- e.g., compute a saliency map
		-- e.g., learning a separate sparse linear model to explain the decisions of a local region near a particular point
	4. Explanations by example: report (in addition to predictions) which other examples the model considers to be most similar
		-- e.g., we can use the activations of the hidden layers to identify the k-nearest neighbors based on the proximity in the space learned by the model.

## Why interpretability
The demand for interpretability arises when there is a _mismatch_ between the formal objectives of supervised learning (test set predictive performance) and the real world costs in a deployment setting. 
Predictions alone and metrics calculated on these predictions do not suffice to characterize the model.

- Trust: in many case when the training and deployment objectives diverge, trust might denote confidence that the model will perform well with respect to the real objectives and scenarios. Another sense in which we might trust a machine learning model might be that we feel comfortable relinquishing control to it: we might care not only about how often a model is right but also _for which examples it is right_.
- Causality: The associations learned by supervised learning algorithms are not guaranteed to reflect causal relationships. There could always exist unobserved causes responsible for both associated variables, but by interpreting supervised learning models, we hope to generate hypotheses that scientists could then test experimentally.
- Transferability: transferring learned skills to unfamiliar situations.
- Informativeness: convey additional information to the human decision-maker; an interpretation may prove informative even without shedding light on a model’s inner workings (e.g., by providing similar training cases). 
- Fair and ethical decision-making: right to explanation, recidivism prediction, algorithmic decisions should be _contestable_, etc..

## Some other points:
- Linear models are not strictly more interpretable than deep neural networks
- When choosing between linear and deep models, we must often make a trade-off between algorithic transparency and decomposability. This is because deep neural networks tend to operate on raw or lightly processed features. So if nothing else, the features are intuitively meaningful, and post-hoc reasoning is sensible. However, in order to get comparable performance, linear models often must operate on heavily hand-engineered features.
- In some cases, transparency may be at odds with the broader objectives of AI; we should be careful when giving up predictive power, that the desire for transparency is justified and isn’t simply a concession to institutional biases against new methods.
－ Post-hoc interpretations can potentially mislead。




