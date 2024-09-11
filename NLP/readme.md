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
\begin{align*}
  P((AUB)|F) &= \frac{ P(A \cap F) + P(B\cap F) - P(A \cap B \cap F) }{P(F)} \\  
  &= \frac{ P(A \cap F)}{P(F)} + \frac{ P(B \cap F)}{P(F)} - \frac{ P[(A \cap B)|  F]}{P(F)} \\
  &= P(A | F ) + P(B | F) - P((A \cap B) | F)
\end{align*}
$$

When A and B are disjoint events, then $P((A \cap B)|F) = 0$

$= P((A \cup B)|F) = P(A|F) + P(B|F)$


### Property 3 : $P(E' | F) = 1 - P(E|F)$

From Property 1 , we know that  $P(S|F) = 1$

= $P( E \cup E'|F ) = 1$ since $S = E \cup E'$

= $P(E|F) + P(E'|F) = 1$ since E and E' are disjoint events

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
P(E|F) &= \frac{P(E \cap F)}{P(F)} \\
       &= \displaystyle \Large\frac{\frac{1}{4}}{\frac{3}{4}} \\
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

### Solution : 

Let E denote the event that a student chosen randomly studies in Class XII and F be the event that the randomly chosen student is a girl. We have to find $P(E|F)$

Now 

$P(F)=\frac{430}{1000}=0.43$ and $P(E \cap F )=\frac{43}{1000}=0.043$ 10% 

Then

$$
\begin{align*}
P(E|F) &= \frac{P(E \cap F)}{P(F)} \\
       &= \frac{0.043}{0.43} \\
       &= 0.1
\end{align*}
$$

### Example 5 : 

A die is thrown three times. Events A and B are defined as below

A :  4 on the third throw

B : 6 on the first and 5 on the second throw

Find the probability of A given that B has already occurred.


### Solution:

The sample space has 216 outcome - for thrown 3 times

$P(S)=216$

$A = \{(1, 1, 4),
(1, 2, 4),
(1, 3, 4),
(1, 4, 4),
(1, 5, 4),
(1, 6, 4),
(2, 1, 4),
(2, 2, 4),
(2, 3, 4),
(2, 4, 4),
(2, 5, 4),
(2, 6, 4),
(3, 1, 4),
(3, 2, 4),
(3, 3, 4),
(3, 4, 4),
(3, 5, 4),
(3, 6, 4),
(4, 1, 4),
(4, 2, 4),
(4, 3, 4),
(4, 4, 4),
(4, 5, 4),
(4, 6, 4),
(5, 1, 4),
(5, 2, 4),
(5, 3, 4),
(5, 4, 4),
(5, 5, 4),
(5, 6, 4),
(6, 1, 4),
(6, 2, 4),
(6, 3, 4),
(6, 4, 4),
(6, 5, 4),
(6, 6, 4),\}$

$B=\{(6,5,1),(6,5,2),(6,5,3,),(6,5,4,),(6,5,5,),(6,5,6,)\}$ and
$(A \cap B)=\{(6,5,4)\}$


$P(B)=\Large\frac{6}{216}$ and $P(A \cap B)=\Large\frac{1}{216}$ 

Then 

$$
\begin{align*}
P(A|B)&=\frac{P(A \cap B)}{P(B)}\\
      &=\Large\frac{\frac{1}{216}}{\frac{6}{216}}\\
      &=\frac{1}{6}
\end{align*}
$$

### Example 6 : 

A die is thrown twice and the sum of the numbers appearing is observed to be 6. What is the conditional probability that the number 4 has appeared at least once?

### Solution:

Let E be the event that 'number 4 appears at least once',
    F be the event that 'the sume of the numbers appearing is 6'


The, 

$P(E)=\{(4,1),(4,2),(4,3),(4,4),(4,5),(4,6),(1,4),(2,4),(3,4),(5,4),(6,4)\}$

and $F=\{(1,5),(2,4),(3,3),(4,2),(5,1),\}$

We have 

$P(E)=\Large\frac{11}{36}$ and $P(F)=\Large\frac{5}{36}$

Also $E \cap F  = \{(2,4), (4,2) \}$

Therefore $P(E \cap F) = \Large\frac{2}{36}$

Hence, the required probability

$$
\begin{align*}
P(E|F)&=\frac{P(E \cap F)}{P(F)}\\
      &=\Large\frac{\frac{2}{36}}{\frac{5}{36}}\\
      &=\frac{2}{5}
\end{align*}
$$

### Example 7 :

Consider the experiment of tossing a coin. If the coin shows head, toss it again but if it shows tail, then throw a die.
Find the conditional probability of the event that `the die shows a number greater than 4' given that 'there is at least one tail'

# Solution : 

E = The die shows a number greater than 4

F = There is at least one tail

The outcomes of the experiment can be represented in following diagrammatic manner called the 'tree diagram'

The sample space of the experiment may be described as 

$S=\{(H,H),(H,T),(T,1),(T,2),(T,3),(T,4),(T,5),(T,6),\}$

Where (H,H) denotes that both the tosses result into head and
(T,i) denote the first toss result into a tail and the number i appeared on the die for 'i = 1,2,3,4,5,6'

Thus, the probabilities assigned to the 8 elementary events

F = {(H, T), (T, 1), (T, 2), (T, 3), (T, 4), (T, 5), (T, 6)}

E = {(T,5), (T,6)}

$E \cap F = \{ (T, 5), (T, 6)\}$