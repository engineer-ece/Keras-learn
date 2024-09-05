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
'The


 





