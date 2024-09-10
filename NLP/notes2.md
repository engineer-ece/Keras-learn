# Probabilistic Models (Introduction)

 - useful for ml & non-ml (eg. document retrieval)


 Probability Models Introduction - (Markov models)

 1900 - analyze patterns of vowel & consonants
 1948 - claude shannon - generate text
 HMMs (Hidden Markov Models) - speech recognition.


# Biological Sequences & NLP

DNA - build body parts
i.e (non-human one)

- developments in NLP - genomics (CNNs, RNNs, Transformers)

- alphafold 2 

Google's PageRank method

Lesson : Want to start a billon dollar business? just learn about Markov models

Outline
```````

1. Markov model / N-gram language model
2. Application : Article spinner (black hat SEO)
3. Application : Cipher decryption / Code-breaking
4. Later : machine learning and deep learning, which apply both vector models and probability models


# Markov Models Section Introduction

Black-scholes formula
Reinforcement learning - MDP
HMM (Hidden Markov Model) ()- sp, comp bio
MCMC (Markov Chain Monte Carlo) - numerical approximation 


# Section Outline

1. Markov property (requires knowledge of probability)

2. Markov model, state transition matrix

3. "Training" a Markov model (i.e. "learning")


Application

- building a text classifier using Bayes rule + Generating poetry

- There are 2 fundamentally different ways to apply ML

- Supervised : predict a target from a dataset of inputs + labels

- Unsupervised : learn structure of data, create new examples having the same structure.


# The Markov Property

sequence : {x1, x2, ... , XT}

probability of sequence : p(x1, x2, ... , XT)

forecast/predict : p(xt | xt-1, xt-2, ... x1)


x(t-2) -> x(t-1) -> x(t) -> x(t+1) -> x(t+2)

p(x1,...,xt ) = p(x1) prod p(x(t)|x(t-1))


x(t) is independent of x(t-2), x(t-3), ...

# What if the Markov Property were Not true?

2-variable example (Baye's rule)

p(x1, x2) = p(x1)p(x2|x1)
p(x1, x2, x3) = p(x1)p(x2,x3|x1)
p(x1, x2, x3) = p(x1)p(x2|x1)p(p3 | x2,x1)

# Chain Rule of Probability

1. T-variable expansion
2. In general, each term depends on all preceding terms


p(x1, ..., xT) = p(x1)p(x2|x1)p(x3|x1,x2)...p(xT|x(t-1),...,x1)  


# Why is the Markov Property Useful?
  model -> english -> 2000 words

  p(x10| x9, x8, ..., x1)

Each x has 2000 possible values of words

How many probabilities to estimate ? 2000x2000x2000 = 2000 ^ 10

# The Markov 'Assumption'

"I like green eggs and ham."
"I like to code in C++ and Python."

1. Do 'ham' and 'python' only depend on 'and'? no

remember - markov property is very useful (finance, RL, biology,...)
--------------------

# The Markov Model

1. Categorical symbols - states
2. Eg: weather : {sunny, rainy, cloudy}
3. parts-of-speech tags : {noun, verb, adjective, ...}

# Notation

1. We'll uses for 'state' (other sources might use z or x)

s(t) = s_t = state at time t

M = total no. of possible states (M = 3 for sunny/ rainy/ cloudy)

p(s_t = i ) means : probability that state at time t is 'i'

# State Distribution

We have p(s_t = 1), p(s_t = 2), ... , p(s_=M)

distribution

What is the probability that it will be rainy on sunday?

ans = p(S_sunday = rainy)

p(s_t = state distribution(length M vector))

# State Transitions

p(s_t = j | s_(t-1) = i)

probability that state at time t is j.

given that the state at time t-1 was i.


# State Transistion  Matrix

p(s_t = j | s_(t-1) = i), Vi = 1...M, j=1...M

first index = previous state,
second index = next state

A is a MxM matrix

In general,we could have A_(ij)(t)

time-homogeneous Markov process

# State Transition Diagram


# Initial State

The quick brown fox jumps over the lazy dog.

To quantify the probability of the first state in a sequence, we use the initial state distribuion


pi(i) = p(s_1= i) (for i=1...m)

Recap:

# Probability of a sequence
`````````````````````````

p(S_(1..T))

# Training a Markov Model

p(heads) ~ count(H)/total tosses

p('cat') ~ count(cat)/total word count

Estimating A & pi ['training']

pi_{i} = count(s_1 = i)/N

Aij = count(i->j)/count(i)

# Markov Model Summary

1. What we want to do : model a sequence of states
2. state transition matrix (A), iniital state distribution (pi)
3. Task #1: find the probability of a sequence
4. Task #2: given a dataset, find A and Pi
5. learning & training

--------

next video

# Probability Smoothing and Log-Probabilities

1. Modify how we estimate A and pi
2. Compute the probability of a sequence

# Probability of a Sequence

- Only involves multiplication
- train set different from test set

The whole thing becomes 0!

# Add-one smoothing

1. Give a small probability to every possible transition
2. Add a fake count of 1 for each (i,j) transition

Aij = (count(i->j) + 1 )/ (count(i) + M)

pi = count(s1= i ) + 1 / (N+M)

# Add-Epsilon Smoothing.

* we can make it more smooth (epsilon > 1) or less smooth (epsilon < 1)

Aij = (count(i->j) + eps) / (count(i)+eps*M)
pi_(i)  = count(s1=i) + eps / (N+ eps*M) 


* This involves multiplying many small numbers together

* common to use 20k - 50k vocabulary size in English

* As you multiply small numbers together, they approach 0! (0.1 * 0.1  = 0.001)

* when comparing 2 sentences, it not valid?

* Working with log probabilities

- Compute log probabilities insted
- we don't need the actual probability value,  
- since what we usually want to do is compare(e.g is one sequence more likely than another?)

- log - monotonically increasing function
- if A>B, then log(A) > log(B) (i.e. the 'winner' is preserved)

- Does it work? log_10(10^-10) = -10
- Does it work? log_10(10^-100) = -100

# Working with Log Probabilities

log p(S_{1,...,T}) = log pi_(si) + sum_{t=2}^T log(A_{st-1},st)

since log(AB) = log(A) + log(B)

Note : do not compute the product and then take the log

Also note: addition (+) is more efficient (faster) than multiplication (*)

# Building a Text Classifier Theory


Poem ----> Classifier ----> Is the poem by Edgar Allan Poe or Robert Frost?



Email --> classifier --> spam or not
movie --> classifier --> pos or neg


# Goal of this exercise

- accuracy - pretrained data

# Supervised or Unsupervised?

P(y|x) = (p(x|y)p(y))/p(x)

# Bayes classifier

Poems by robert frost ---> Markov Model --> (A0, pi0)

poem by edgar allan poe ---> markov model --> (A1,pi1)

-- New unknown text

p(x | author = frost ) | p(x| author = Poe)

# Applying Bayes' Rule

1. We have p(poem | author), but we want p(author | poem)
2. Why?
3. Then we can apply the following decision rule:

k* = arg max p(class = k|x)
          k 


p(author | poem) = p(poem | author) p(author) / p(poem)

# Simplifying the decision rule

k* = arg max [p(poem|author=k) p(author=k) / p(poem)]
          k 

k* = arg max [p(poem|author=k) p(author=k) ]
          k

k* = arg max log p(poem | author = k ) + log p(author = k)
          k 

# Maximum Likelihood


k* = arg max log p(poem | author = k)
          k 

# Recap

We train a separate Markov model for each class
Each model gives us p(x | class = k) for all k

General


# Text Classifier

1. loop through each file, save each line to a list (one line == one sample)

2. Save the labels too

3. Train-test split

4. Create a mapping from unique word to unique integer index

5. loop through data, tokenize each line (string split should suffice)

6. Assign each unique word a unique integer index ("mapping" == "dic")

1. Details

Convert each line of text (the samples) into integer lists
Train a Markov model for each class (Edgar Allar Poe/ Robert Frost)

2. Don't forget to use smoothing (e.g add-one smoothing)

3. Consider whether you need A and I , or log(A) and log(pi)

4. For bayes' rule, compute the priors : p(class= k)

5. Now you have everything you need to make a prediction.

- Write a function to compute the posterior for each class , given an input

- Take the argmax over the posteriors to get the predicted class

- Make predictions for both train and test sets

- compute accuracy for train/test

- Check for class imbalance

- If imbalanced, check confusion matrix and F1-score


Using - Markov model to generate text


classifying text : supervised learning ( we have targets/label)

generating text : unsupervised learning.


Bayes Classifier
````````````````
1. Discriminative Model p(y|x) - logistic regression, neural network

2. Generative Model p(x|y)


Sampling (drawing random numbers)

 * Sampling from N(0, 1) : np.random.randn()
 * Sampling from Bernoulli: np.random.choice([0,1] ) or np.random.choice(2)
 * sampling from categorical (e.g a 10 sided die) : np.random.choice(10)
 * Bernoulli with p(heads) = 0.8 (where heads = 1)
    np.random.choice([0,1] , p=[0.2,0.8])

 * Sampling words - suppose our vocabulary is ("cat", "dog", "mouse")
 * with probabilites 0.2, 0.5. 0.3
 * supposing we can map the words to ints (0,1,2) then it's easy: 
    np.random.choice(3, p = [0.2,0.5,0.3])
* Supposing we can map the words to ints (0,1,2), then it's easy: 
    np.random.choice(3, p=[0.2, 0.5, 0.3])
* There are many ways to do this? Please 'exercise' your coding skills.


# Extending the Markov Model

First Order Markov : p(st | s(t-1), s(t-2), ...) = p(st|s(t-1))

Second Order Markov : p(st | s(t-1), s(t-2),...) = p(st| s(t-1),s(t-2))

2nd 

Aijk = p(s_T = k | st-1 = j , st-2=i)

3D dimensional model

MxMxM = M^3


Preview / Foreshadowing

- In deep learning, there is no need to make the Markov assumption

- Magically, there is 'no' increase in computational cost when you add more previous words

Exercise_prompt

Why using Dictionaries?

* Sparsity!
* we use add-one smoothing since many possible transition don't appear in the training corpus

A1 = {
  'the' : {
      'cat': 0.05,
      'dog': 0.03,
      'mouse': 0.01
  },
  'a'L
}

Sampling from a probability dictionary

- How to sample
- This is part of your exercise

Example

-- cumulative sum


Alternative 

-- Conceptually simpler but less ideal : just convert the dictionary keys and values into list