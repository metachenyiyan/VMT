<a name="br1"></a> 

**Probabilistic Model for Infinite Memory LLM:**

**Variational Markov Transformer**

(theory part complete, experiment in progress, paper not finished yet)

Authors: Yiyan CHEN

**Abstract**

Current Long Language Models (LLMs) are constrained by limited window sizes and require substantial resources for training and inference. Addressing these limitations, we propose a novel approach that establishes a foundational probabilistic framework for modeling the distribution of sequences of arbitrary lengths, including the comprehensive corpus used to train LLMs.

Our methodology innovatively integrates Deep Hidden Markov Models (DHMMs) as the core probabilistic framework for representing sequence collections. Moving beyond the traditional HMM algorithm's reliance on discrete hidden states, we employ a novel combination of the Evidence Lower Bound (ELBO) objective from Variational Autoencoders (VAEs) with the Truncated Backpropagation Through Time (TBTT) algorithm in Recurrent Neural Networks (RNNs) for learning DHMMs. This integration leverages the strengths of VAEs in mitigating gradient explosion and the insignificance of initial hidden states, common challenges in RNNs. Simultaneously, the TBTT algorithm facilitates the learning of long-term dependencies by sharing hidden states across sequence segments. This synergistic approach enables our model to effectively capture complex sequence patterns, achieving theoretically infinite dependencies and advancing the capabilities of sequence modeling.

In addition, for language tasks, we use LLMs as pre-trained text encoders,  Our method significantly reduces the required input window size for LLMs during both training and inference phases, thereby decreasing the overall computational demands and resource consumption of LLM.

Code in progress, please go to PDF for full content: [preprint paper VMT](VMT.pdf)

