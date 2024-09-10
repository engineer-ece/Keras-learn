# Article Spinning Problem Description

What is it? Why should you care ?

How to Get Readers?

- Share with friends and family?
- Does not scale

Search Engine Optimization

How to Automatically Write Content

you'll be penalized, making your ranking worse (oops)

# N-Gram Models

first-order markov model

p(W_t | w(t-1) ) = count (w(t-1) -> wt) / count (w(t-1))

w(t-1) ---> w(t)

second-order markov model

p(w_t | w(t-1), w(t-2)) = count(w(t-2) --> w(t-1) --> w(t)) / count(w(t-2)--> w(t-1))

# Predicting the Middle Word

p(W_t | W(t-1), w(t+1))

# Estimating the Middle Word Distribution

Maximum likelihood estimate as usual

p(W(t) | W(t-1), W(t+1)) 
=  count(w(t-1)-->w(t)-->w(t+1))
   `````````````````````````````
   count(w(t-1)-->any-->w(t+1))


Does It work?

Production -> began        -> to
           -> capacity
           -> closer
           -> continued
           -> Facilities


# Article Spinner Exercise Prompt
```````````````````````````````

BBC news Data

-- TreebankWordDetokenizer