# Cipher Decryption

A unique project

1. Probabilistic language modeling
2. Genetic algorithms

Applications : warfare, espionage

# Section Outline

1. What is a cipher ?
    - Encode and decode a message

2. Language modeling 
    - What is the probability of this sentence? (poetry, article spinning)

3.  Genetic algorithm / evolutionary algorithm
    - Optimization based on biological evolution.

4. Our decoded message should have the highest likelihood if the model is trained on the English language
    - A message not in English should have a smaller likelihood ("maximum likelihood")


1. Theory to Code

- Implement as much of the code as possible by yourself.
- Basic pattern - theory -> code

- The code is simply an example of how to translate the theory into a real computer program.


# Optional 

- This example will be more challenging than the rest of the examples

- A few students wanted more of a challenge, 
- even though I originally designed this course to be e


----

# Ciphers

1. Substitution Cipher
2. High - level picture
3. Example [Sender] [Sender]

Summary:


# Language Model Review


high probability to real words/sentence low probability to unreal words/sentence

N - grams and Markov Models

---- "CATS" -> C,A,T,S
       | 
       ------> CA, CT, TS
N = 1
N - 2  - unigram, bigram
N - 3

P(x_t | x_(t-1), x_(t-2), ... ) = p(x_t | x_(t-1))

What does it mean?

P(A|C) = how often does "A" follow "C"?
P(T|A) = how often does "T" follow "A"?

counting

p(x_t | x_(t-1)) = count { x(t-1) -> xt} / count{x(t-1)}

P(A|C ) = # of times "CA" appears in the dataset

Mini Quiz

* How many bigram probabilit of 26

V letters , v^2 bigrams


26*26 = 

joint:

p(AB) = p(B|A)p(A) ------ Marginal unigram
           |     
        conditioanl
         bigram


Chain Rule of Probability

p(ABC) = p(C | AB) p(B|A) p(A)
p(ABC) = p(C | b) p(B|A)p(A)

# Word of Any length

A word of length T

p(X1, x2, ..., xT ) = p(x1) proc(T)(t=2) p(xt | x(t-1))

# Probability of a Sentence

p(w1, w2, ... , Wn ) = prod(n=1)(N) 


# Why only letters?

# Add-one Smoothing 

p(x1, x2, ..., xT ) 

p(x_t | x ( t-1) ) = count { x(t-1)-> x(t) + 1} / count{x(t-1)+V}


practical issue. it goes to zero, 

use log-likelihood

log p(x1, x2, ... , Xt) = log p(x1) + sum(t=2)(T) log p(xt | x(t-1))


# Genetic algorithm

First -> a bit about genetics and evolution

parent - passes on DNA to children


DNA Replication
Mitosis

ATCG

Types of mistakes (mutations)

1. Substitution = ATCG -> ATTG
2. Insertion    = ATCG -> ATCTG
3. Deletion     = ATCG -> ATG 

probability of error is small


1. Numerical Optimization

|
|
|
| 
|      
|__________________________
