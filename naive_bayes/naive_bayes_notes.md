# Naive Bayes
Naive bayes uses bayes-rule to calculate posterior probability densities using prior probability densities
The main assumption that naive bayes rules makes is that the features are conditionally independent of each other given the class.   
For example $P(rain|cloudy,windy) = P(rain|cloudy)*P(rain|windy)$ even though cloudy and windy will generally give us
higher probability of rain. Neverthless, this assumption generally works pretty good.
$$ P(y|x_1,x_2,x_3,...x_n) = \frac{P(y) P(x_1,x_2,...x_n|y)}{P(x_1,x_2,...,x_n)} $$
because of conditional independence assumption across features, the above equation can be written as,
$$ P(y|x_1,x_2,x_3,...x_n) = \frac{P(y) \prod_{i=1}^nP(x_i|y)}{P(x_1,x_2,....,x_n)} $$
$$ \Longrightarrow P(y|x_1,x_2,x_3,...x_n) \propto P(y) \prod_{i=1}^nP(x_i|y) $$

#### Preventing floating point underflow
As multiplying lots of probabilities can result in floating-point underflow, we can simply do sum of $\log$ and the class with highest relative value usually wins.
$$ \Longrightarrow P(y|x_1,x_2,x_3,...x_n) \propto \log(P(y)) + \log(\prod_{i=1}^nP(x_i|y)) $$
## Different types of Naive Bayes estimation
### Gaussian Naive Bayesian
  - Used when the features are distributed over a gaussian distribution with mean, $\mu_{iy}$, and std.dev, $\sigma_{iy}$ (where subscript $iy$ represents feature $i$ in class $y$)
  - Therefore, $P(x_i|y)$ has a gaussian pdf given as 
  $$ P(x_i \mid y) = \frac{1}{\sqrt{2\pi\sigma^2_{iy}}} \exp\left(-\frac{(x_i - \mu_{iy})^2}{2\sigma^2_{iy}}\right) $$
where $\mu_{iy}$ and $\sigma_{iy}$ are estimated from the training data

### Multinomial Naive Bayesian
  - Used when the  data/features are multinomially distributed and is most commonly used for text classification
  - In-order to estimate $P(x_i|y)$ a frequency count method is used
  $$ \hat{P}(x_i|y) = \frac{count(x_i, y)}{\sum_{w\in v}count(x,y)} $$
  where, 
  > $count(x_i, y)$ is the number of times a feature $i$ appeared in class $y$  
  > \sum_{w\in v}count(x,y) is the total count of the features in class $y$ with dataset length $v$  
    
one problem to note in estimating $\hat{P}(x_i|y)$ is that there is a chance that we might not see a given feature in our dataset. This means $\hat{P}(x_i|y)$ will be $0$. This implies $\prod_{i=1}^nP(x_i|y)$ will go to 0 because of this feature. To go around this, the estimated probability, $\hat{P}(x_i|y)$, can be modified using laplace smoothing
$$ \hat{P}(x_i|y) = \frac{count(x_i, y) + 1}{\sum_{w\in v}(count(x,y) + 1) } $$
$$ \Longrightarrow \hat{P}(x_i|y) = \frac{count(x_i, y) + 1}{\sum_{w\in v}(count(x,y))+ |v| } $$
assuming a small constant $\alpha = 0.001$ also called as smoothing parameter
$$ \hat{P}(x_i|y) = \frac{count(x_i, y) + \alpha}{\sum_{w\in v}(count(x,y))+ \alpha|v| } $$
This guarantees that the probability, $\hat{P}(x_i|y)$,  can never go to 0  and the result will be between relative frequency ($x_i/N$) and uniform probability ($1/v$) and it's a way of regularizing Naive Bayes.  
In the end, the estimated class can be obtained by evaluating below expression and selecting the class that has the biggest final value. 
$$  P(y=y_i|x_1,x_2,x_3,...x_n) \propto \log(P(y=y_i)) + \log(\prod_{i=1}^nP(x_i|y=y_i)) $$

### Complement Naive Bayesian
  - This is another version of Multinomaial naive bayes where the weights ($\hat{P}(x_i|y_i)$) for a given class are estimated using the word occurances in classes other than $y_i$
  - This helps in dealing with dataset that has clas imbalance and avoid skewed bias towards one class than the other
$$ \hat{P}(x_i|y_i) = \frac{count(x_i, y\neq y_i) + \alpha}{\sum_{w\in v}(count(x,y \neq y_i))+ \alpha|v| } $$

Now, the estimated class can be obtained by evaluating below expression and selecting the class that has the **lowest** final value. 
$$  P(y=y_i|x_1,x_2,x_3,...x_n) \propto \log(P(y=y_i)) - \log(\prod_{i=1}^nP(x_i|y \neq y_i)) $$

other types are **Bernoulli Naive Bayes**

## Text Classification
When it comes to Text classification things like stop words (at, and,...), punctuation, numbers, ... tend to skew the feature distribution. Unless the classification is about learning the grammar syntax of a given text, these things are unnecessary. So, a data preprocessing before estimating probabilities are performed. Types of preprocessing things are as follows, 
* Removing Stopwords
  - a, able, else, ever, etc
* Lemmatizing words
  - Grouping together different inflections of the same word as they mean the same thing
* n-grams 
  - instead of using unigram model we could use bi-gram or n-gram depending on the type of the task at hand 
* TF-IDF (Term Frequency-Inverse Document Frequency)
  - instead of simply taking the frequency of the words in a given class, we can model the contribution of a given word in class decision using TF-IDF
  - if a word appears too much in a  given dataset, then the IDF term will weigh this word low as this word won't be a good indicator in class decision making
  - This way we can significantly increase classification accuracy by weighing down words that appear much often (like, and/or/an/a...)
  $$ \hat{P}(x_i|y_i) = count(x_i, y_i) \log(\frac{N}{df_i})$$
  where,
  > $N$ is the number of documents  
  > $df_i$ is the number of documents that word i appeared in N documents

## References
  - **[Scikit_learn_naiveBayes](https://scikit-learn.org/stable/modules/naive_bayes.html)**
  - **[Stanford_Juarfsky_slides](https://web.stanford.edu/~jurafsky/slp3/slides/7_NB.pdf)**
  - **[CMU_EpXing_slides](https://www.cs.cmu.edu/~epxing/Class/10701-10s/Lecture/lecture5.pdf)**
  - **[Complement_Naive_Bayes](https://people.csail.mit.edu/jrennie/papers/icml03-nb.pdf)**
  - **[MNB_Text_Categorization_Revisited](https://www.cs.waikato.ac.nz/ml/publications/2004/kibriya_et_al_cr.pdf)**