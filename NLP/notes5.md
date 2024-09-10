# Outline

ML section --> application
(eg. vector models & Markov models)

1. Application : Spam Detection
   Technique   : Naive Bayes

2. Application : Sentiment Analysis
   Technique   : Logistic Regression

3. Application : Latent Semantic Indexing (for SEO)
   Technique   : PCA / SVD

4. Application : Topic Modeling
   Technique   : Latent Dirichlet Allocation


# Application Over Technique

- NLP
- Naive Bayes
- Some application in mind
- Naive bayes - good for that application

-----

Supervised     

spam detection
sentiment analysis

Unsupervised

Topic modeling
Latent semantics analysis
Text summarization (but can be supervised)

# Spam detection - problem description

first part of this section : what is spam detection?
second part of this section : what is the solution?

what is spam detections?

- Best way to understand is by experience with email/text/messaging.

- Not every message you receive is legitimate

- Sometimes, you get unsolicited messages from unknown senders

- Their objectives:

    - sell something

    - install malware by convincing you to click a link

    - steal your credentials (bank, Facebook, etc)

Example : Nigerian Prince Scame

- They need to move a large sum of money urgently
- They might ask for your bank details to deposit the money
- Or ask for a small advance payment
- This is one example of a spam message that many people received by email

# Why do we want to detect spam?

- Isn't it obvious what is spam and not spam ? 
- No! Some people fall for it

- A great example of how ML is used for automation and efficiency

# Common Mistake

- I noticed a somewhat common mistake in V1 of this course
- Students were confused about where spam detection 'fits in' to an email application.

- You don't need to build a whole email service (e.g Gmail) or a whole email client (e.g Thunderbird) - that would take a whole team.

- That requires user interface work, database design, etc.

- You should be comfortable with building only a small part of a big app.

- Software engineers do that everyday in the real world!
    Use Q&A If you don't understand

1. Useful Context

Imagine you're a software engineer at Google working on Gmail
Your job is to write a function (not all of Gmail itself.)

def spam_detector (document):
    # ... do some work ...
    return 1 if spam else 0

This section: how do we write such a function?
Note 

# Naive Bayes Intuition

P(Y|X) = P(X|Y) P(Y) / sum(y) P(X|y)p(y)


P(Y|X) is a Distribution


--------------------

P(severse | BMI) =  P(BMI|severe) p(severe)
                    -----------------------
                    P(BMI|severe)P(severe) + P(BMI|mild)P(mild)

p(mild|BMI) = p(BMI | mild)p(mild)
              ----------------------------------------------
              P(BMI | severe)P(severe) + P(BMI|mild) P(mild)


