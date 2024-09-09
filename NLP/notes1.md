Tokenize, Stop word, Stemming and lemmation
N-Gram
Vector - tail->head = have direction & magnitude
Why are vector useful? 
    - Spam Detection
    - Organizing Documents
Bag of Words
    - Where is Bag of Words? Where is it Not Used?
Count Vector (Theory)
    - Text -> Vector (bag of words)
    - Documents
    - Data Format
        - Text (input) = i/p, Label (target) = o/p
        - Determining Vocabulary Size
        - Practical issues 
            - text - string
            - algorithn - words
            - tokenization - string to list of words
            - mapping in index

# CountVectorizer in Code(Scikit-learn) 

```code
vectorizer = CountVectorizer()
vectorizer.fit(list_of_documents_train)
Xtrain = vectorizer.transform(list_of_documents_train)
Xtrain = vectorizer.fit_transform(list_of_documents_train)
xtest = vectorizer.transform(list_of_documents_test)
```

Why Scipy instead of Numpy>
 - sparse matrices
 - N document, vocabulary size V 
 - Count data - matrix - size NxV
 - Most documents will not contain most words

Normalization
 - very long and very short document?
 - more words = higher counts 
 - 1. L2-norm 1
 - 2. divide by the sum 
 - probability model
 - 3. L1-norm - sum of all the element 1

Tokenizer
 - 1. Splitting String into words
 - "I like cat" -> s.split() -> ["I","like","cats"]
 - 2. Punctuation
 - 3. Tokenizing punctuation: "I hate cats?" -> ["I","hate","cats","?"]
 - 4. Note: Sklearn CountVectorizer ignores punctuation
 - 5. Back to String Split
 - 6. ["I","hate","cats"] or ["I","hate","Cats?"]
 
 - Casing
 - Accents
 - Character-Based Tokenization
   - Store Million words - [1 million dimension probability distribution]
   
   - word have meaning / contain lots of information
   
   - characters do not contain lots of information

- Subword-Based Tokenization
   
   - What if we didn't split "walking" into "walk"+"ing"?
   - walk, walks, walking, walked, etc.

Scikit-learn

Word based tokenization

  CountVectorizer(analyzer="word")

Character-Based Tokenization
  
  - CountVectorizer(analyzer="char")


  - For Beginners: How is Tokenization Done?


Stopwords

    - Why do we need stopwords ?
    
    - sucha as 'and', 'the', 'it' , 'is' ,

    - increase dimensionality is bad.

Distance Consideration
    - 

Stopwords in CountVectorizer

    - CountVectorizer(stop_words="english")
    - CountVectorizer(stop_words=list_of_user_defined_word)
    - CountVectorizer(stop_words=None) # default

NLTK Stopwords

    import nltk
    nltk.download('stopwords')
    from nltk.corpus import stopwords

    stopwords.words('english')
    stopwords.words('german')

Stemming & Lemmation

 - Walk, Walking, Walks, Walked
 - high-dimensional vectors
 - Practical issue : imagine we're building a search engine
   (like DuckDuckGo)
 - I search for "running" (What about 'ran' or 'run')

 - Stemming - very crude
 - lemmatization - uses actual rules of language


Stemming
 - SSES -> Remove ES
 - Bosses -> Boss
 - Replacement -> "Replac" (not a real word)

 from nltk.stem import PorterStemmer

 porter = PorterStemmer()

 porter.stem("walking")

Lemmatization

    lookup table/ table of rules
    
    Stemming - "better" -> "better"
    lemmatization - "better" -> "Good"

    was -> wa
    was/is -> be

    Mice -> mice
    Mice -> mouse

from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

nltk.download("wordnet")

lemmatizer = WordNetLemmatizer()
lemmatizer.lemmatize("mice")

lemmatizer.lemmatize("going")
lemmatizer.lemmatize("going", pos=wordnet.VERB)



Why does the part-of-speech tag matter?

'Donald Trump has a devoted following"


Vector Similarity
````````````````

Application : Article Spinning - auto generate content

Euclidean Distance  - normal distance formula

Angle Between 2 Vector - angle two vector

Cosine Similarity -  2 vector == 180 = -1 = least similar 

cosine distance =  1 - cosine similarity
(it not true distance)

dist = 1 - sim = 1 - 1 = 0

dist = 1 - sim = 1 - (-1) = 2

Which one should we use?
When are cosine distance and Euclidean distance equivalent?

 TF-IDF (Theory) 
 ```````````````

 What's wrong with the Count Vectorizer?

 - why don't we want to keep stopwords?
 - unlikely - nlp tasks
 - every document contains these words

 stopwords

 auto identity good or bad

 convert -> text into vectors by counting

 scale down word count

 TF - IDF ~~  Term Frequency/Document Frequency

 tfidf(t,d) = tf(t,d) x idf(t)

 tf(t, d) = # of times t appears in d
    
t = term
d = document

idf(t) = log (N/N(t))

tfidf(t,d) = tf(t,d) * idf(t)

why take the log?
  N/N(t) 


from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer()

Xtrain = tfidf.fit_transform(train_texts)
Xtest = tfidf.transform(test_texts)

# note : arguments exist for stopwords, tokenizer, strip accents, etc.

Recap  - TF-IDF works and what problems it solves
`````

= Term Frequency Variations

tf(t,d ) = count(t,d)/ sum(count(t',d))

tf(t,d) = log(1+count(t,d))

Inverse Document Frequency

- Smooth IDF -  idf(t) = log(N/N(t)+1) + 1
- idf max - idf(t) = log (max N(t)/n(T))




Why do we need it?

Text -> Vect

document-term matrix

row = document
column = term (size = #document x #terms)

Warning : Coding Skills Required


simple example
`````````````````

doc 1 = I like cats
doc 2 = i love cats
doc 3 = i love dogs

words in column

How can we do this programmatically?

current_idx = 0
word2idx = {}

for doc in documents:
  tokens = word_tokenize(doc)
  for token in tokens:
    if token not in word2idx:
        word2idx[token] = current_idx
        current_idx += 1



Dependencies
````````````

Word-to-index mapping ----> Count Vectorizer ----> TF-IDF

Train vs. Test
``````````````

* What to do with words in the test set but not train set?
* Method 1 : Ignore those words
* Why? Your ML model will be trained on the train set
* It wouldn't know what to do with other words
* method 2 : createa a special token/rare word
* Assign any rare word


Reverse Mapping (index-to-word)

==

-Neural Word Embeddings

Documents vs Words

- softmax, RNN

Sequences of Vectors - we have purpose-built models for sequences

Word2Vec (google)
Glove (stanford)

Word2Vec - embedding

The quick brown fox jumps over the lazy dog

Glove - Doesn't use neural network, but invented in the era of 'Deep NLP'

"Jumps" is 1 word away from 'fox' : score = 1/1
"Jumps" is 2 words away from 'brown' : score = 1/2

i.e. our "rating" are based on distance between words
. If you want to learn more about

1. What can we do with word vectors?

can convert a document into a vector(but not sparse like counting/TFIDF)

Embeddings are dense and low-dimensional (20, 50, 100, 300, ... << V)

Doc ---> "I", "like", "cats", "and", "dogs" ---> vec(i),vec(like), vec(cats) ---> average()


Word Analogies
``````````````

- We can do arithmetic on vectors (+ and -)
- king: Man :: ??? : woman
- Answer : Queen
- In math : King - Man ~ Queen - Woman
- In code : x = king - man + Woman
            Find the closest word vector to x
            The result will be Queen

- Neural Word Embeddings Demo

---------------------------



Vector models & Text preprocessing summary
``````````````````````````````````````````
- text -> vector
- preprocessing
  - tokenization
  - bag of words
  - stopwords
  - stemming and lemmatization

- counting
- tf-idf
- vector similarity / recommender system
- tf-idf from scratch
- word - to - index mapping
- neural word embeddings (word2vec, GloVe)
- word analogies
- words really represent concepts in multipe dimensions
- if concepts are numbers, then are thinking and reasoning simply mathematical operation on numbers?

Text Summarization Preview
``````````````````````````

Steps of a typical NLP analysis

- Get the text (strings)
- tokenize the text
- stopwords, stemming/lemmatization
- map tokens to integers
  - tabular ML works with numbers
  - a table of the format (documents x tokens)
  - need to know which column goes with which token!
- Convert text into count vectors / TF-idf
- do ml task (recommend, detect spam, summarize, topic model, ...)


Many Tools are english-centric

 - Japanese tokenizer 1.6


 Why should you learn the language?

 Eg - suppose text = "this movie is good", but target = "negative"

 - suppose model predicts "positive" (incorrect)
 - you can only diagnose the issue if you know the language.

 - Learning a language is a matter of taking a language course, not an NLP course!

The Key Points
``````````````
2 solutions : (1) build one yourself, (2) find one

Word Embeddings
```````````````

- How to find word vectors for neural networks?
- It really comes down to the same strategy we've discussed
- simply do a search to see if anyone has uploaded them publicly
- Not much i can do to help - I would just be doing the same search!
- What if no embeddings exits? same strategy applies!
- tokenize the text
- Multilingual models
  - active area of research
  - idea : embeddings consistent across languages
  - gpt3

--------------------------------------------------------------

Suggestion box








 





