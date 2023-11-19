<a name="br1"></a> 

**Probabilistic Model for Infinite Memory LLM:**

**Variational Markov Transformer**

(theory part complete, experiment in progress, paper not finished yet, open for collaborations)

Authors: Yiyan CHEN

**Abstract**

Current Long Language Models (LLMs) are constrained by limited window sizes and

require substantial resources for training and inference. Addressing these limitations,

we propose a novel approach that establishes a foundational probabilistic framework

for modeling the distribution of sequences of arbitrary lengths, including the

comprehensive corpus used to train LLMs.

Our methodology integrates Deep Hidden Markov Models (DHMMs) as the core

probabilistic mechanism to represent the sequence collection. For text generation tasks,

we employ pre-trained LLMs as encoders for observed textual sequences. By optimizing

a lower bound of the log likelihood of randomly sampled continuous subsequences

from the training dataset, we derive multiple probabilistic models. These encompass

the prior distribution of hidden states across all sequences, the transition probabilities

between hidden states, and the likelihood of mapping hidden states to sequence tokens,

along with their posterior distributions.

This framework facilitates several practical applications during inference, such as

processing input sequences **without window size constraints**, accelerating sequence

generation, and enabling bidirectional (normal and reverse order) generation. Crucially,

our model achieves **theoretical inﬁnite memory through state transitions**, extending

beyond conventional window limitations. Moreover, it signiﬁcantly reduces the required

input window size for LLMs during both training and inference phases, thereby

decreasing the overall computational demands and resource consumption of LLMs.

This approach not only enhances the capabilities of existing LLMs but also paves the

way for more eﬃcient and versatile language modeling techniques, with potential

applications across various domains of natural language processing.

Code in progress, please go to PDF for full content: [preprint paper VMT](VMT.pdf)

