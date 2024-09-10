bayes theorem
multiplication rule of probability
independence of events

* random variable 
* its probability distribution
* mean & variance of probability distribution

* discrete probability distribution  - called -> Binomial distribution

* experiments having equally likely outcomes, unless stated otherwise

# 2. Conditional Probability

- discussed - methods of finding the probability of events.

$$
\displaystyle P(E|F) = 
\frac{n(E\cap F)}{n(F)}
= \frac{\frac{n(E\cap F)}{n(S)}}{\frac{n(F)}{n(S)}}

= \frac{P(E\cap F)}{P(F)}


$$

$$
\text{Note that (1) is valid only } P(F) \neq 0 (i.e.) \text{F} \neq φ
$$

Thus, we can define the conditional probability

### Definition 1: 

$$
\displaystyle P(E|F) =  \frac{P(E\cap F)}{P(F)} \text{ provided } P(F) \neq 0
$$

## Properties of conditional probability

### Property 1 :  $P(S|F) = P(F|F) = 1$

Let E and F be events of a sample space S of an experiment

$$
 P(S|F) = \frac{P(S \cap F)}{P(F)} = \frac{P(F)}{P(F)} = 1
$$

$$
 P(F|F) = \frac{P(F \cap F)}{P(F)} = \frac{P(F)}{P(F)} = 1
$$

### Property 2 : 

If A and B are any two events of a sample space S and F is an event of S such that P(F) != 0, then

$$
P((A \cup B) | F) = P(A | F ) + P(B | F) - P((A \cap B) | F)
$$

in particular, if A and B are disjoint events, 

$$

\begin{align*}
  
  P((AUB)|F) &= \frac{P[(A \cup B) \cap F]}{P(F)} \\ 
             &= \frac{P[(A \cap F) \cup (B \cap F)]}{P(F)}
\end{align*}

$$

- by distributive law of union of sets over intersection

$$
\begin{aligned}
  P((AUB)|F) &= \frac{ P(A \cap F) + P(B\cap F) - P(A \cap B \cap F) }{P(F)} \\  
  &= \frac{ P(A \cap F)}{P(F)} + \frac{ P(B \cap F)}{P(F)} - \frac{ P[(A \cap B)|  F]}{P(F)} \\
  &= P(A | F ) + P(B | F) - P((A \cap B) | F)
      
\end{aligned} $$

When A and B are disjoint events, then $P((A \cap B)|F) = 0$

$= P((A \cup B)|F) = P(A|F) + P(B|F)$


### Property 3 : $P(E' | F) = 1 - P(E|F)$

From Property 1 , we know that  $P(S|F) = 1$

= $P( E \cup E'|F ) = 1$ since $S = E \cup E'$

= $P(E|F) + P(E'|F) = 1 $ since E and E' are disjoint events

Thus, $P(E'|F) = 1 - P(E|F)$

------

### Example 1 : 

if $\displaystyle P(A) =  \frac{7}{13}$, $\displaystyle P(B) =  \frac{9}{13}$ and $\displaystyle P(A \cap B) =  \frac{4}{13}$, evaluate $P(A|B)$

### Solution :

$$
 \begin{align*}
 P(A|B)  &= \frac{P(A \cap B)}{P(B)} \\
              \\
         &= \Large  {\frac{\frac{4}{13}}{\frac{9}{13}}} \\ \\
         &= \frac{4}{9}
\end{align*}
$$

### Example 2 :

A family has two children. What is the probability that both the children are boys given that at least one of them is a boy?

### Solution :

Let b stand for boy and g for girl. 
The sample space of the experiment is 

$S = \{ (b,b), (g,b), (b,g), (g,g) \}$

Let E and F denote the following events : 

$E : \text{both the children are boys}$

$F : \text{at least one of the child is a boy }$

Then $E = \{ (b,b) \}$ and $F = \{ (b,b), (g,b), (b,g) \}$

Now $E \cap F = \{ (b,b) \}$

Thus $P(F) = \Large \frac{3}{4}$, $P(E \cap F) = \Large\frac{1}{4}$

Therefore 
$$
\begin{align*}
P(E|F) &= \frac{P(E \cap F)}{ P(F)} \\ \\
       &= \displaystyle \Large{\frac{\frac{1}{4}}{\frac{3}{4}}} \\ \\
       &= \frac{1}{3}
\end{align*}
$$

### Example 3 :

Ten cards numbered 1 to 10 are placed in a box, mixed up thorougly and then one card is drawn randomly. If it is known that the number on the drawn card is more than 3, what is the probability that it is an even number ? 

### Solution : 

Let A be the event ' the number on the card drawn is even ' and B  be the event the number on the card drawn is greater than 3. 
We find P(A|B).

Now, the sample space of the experiment is $S = \{1, 2, 3, 4, 5, 6, 7, 8, 9, 10\}$

Then $A = \{ 2, 4, 6, 8, 10 \}$, $B = \{ 4, 5, 6, 7, 8, 9, 10\}$ and $A \cap B = {4, 6, 8, 10}$

$P(A) = \frac{5}{10}, P(B) = \frac{7}{10}$ and  $P(A \cap B)= \frac{4}{10}$

$$
 \begin{align*}
  P(A|B) &= \frac{P(A \cap B)}{P(B)} \\
         &= \Large \frac{\frac{4}{10}}{\frac{7}{10}} \\
         &= \frac{4}{7}

 \end{align*}
$$

### Example 4 :

In a school, there are 1000 students, out of which 430 are girls. It is known that out of 430, 10% of the girls study in class XII.
What is the probability that a student chosen randomly studies in Class XII given that the choosen student is a girl?
