# Naive Bayes
Naive bayes uses bayes-rule to calculate posterior probability densities using prior probability densities
The main assumption that naive bayes rules makes is that the features are conditionally independent of each other given the class.   
For example $P(rain|cloudy,windy) = P(rain|cloudy)*P(rain|windy)$ even though cloudy and windy will generally give us
higher probability of rain. Neverthless, this assumption generally works pretty good.
$$ P(y|x_1,x_2,x_3,...x_n) = \frac{P(y) P(x_1,x_2,...x_n|y)}{P(x_1,x_2,...,x_n)} $$
because of conditional independence assumption across features, the above equation can be written as,
$$ P(y|x_1,x_2,x_3,...x_n) = \frac{P(y) \prod_{i=1}^nP(x_i|y)}{P(x_1,x_2,....,x_n)} $$
$$ \Longrightarrow P(y|x_1,x_2,x_3,...x_n) \propto P(y) \prod_{i=1}^nP(x_i|y) $$

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

other types are **Complement Naive Bayes**, **Bernoulli Naive Bayes**

## Text Classification
When it comes to Text classification things like stop words (at, and,...), punctuation, numbers, ... tend to skew the feature distribution. Unless the classification is about learning the grammar syntax of a given text, these things are unnecessary. So, a data preprocessing before estimating probabilities are performed. Types of preprocessing things are as follows, 
* Removing Stopwords
  - a, able, else, ever, etc
* Lemmatizing words
  - Grouping together different inflections of the same word as they mean the same thing
* n-grams 
  - instead of using unigram model we could use bi-gram or n-gram depending on the type of the task at hand 
* TF-IDF
  - fgf

## References
  - **[Scikit_learn_naiveBayes](https://scikit-learn.org/stable/modules/naive_bayes.html)**
  - **[Stanford_Juarfsky_slides](https://web.stanford.edu/~jurafsky/slp3/slides/7_NB.pdf)**
  - **[CMU_EpXing_slides](https://www.cs.cmu.edu/~epxing/Class/10701-10s/Lecture/lecture5.pdf)**