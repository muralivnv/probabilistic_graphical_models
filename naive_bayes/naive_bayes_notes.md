## Naive Bayes
Naive bayes uses bayes-rule to calculate posterior probability densities using prior probability densities
The main assumption that naive bayes rules takes is that the features are conditionally independent of each other
for example $P(rain|cloudy,windy) = P(rain|cloudy)*P(rain|windy)$ even though cloudy and windy will generally give us
higher probability of rain. Neverthless, this assumption generally works pretty good.
$$ P(y|x_1,x_2,x_3,...x_n) = \frac{P(y) P(x_1,x_2,...x_n|y)}{P(x_1,x_2,...,x_n)} $$
because of conditional independence assumption across features, the above equation can be written as,
$$ P(y|x_1,x_2,x_3,...x_n) = \frac{P(y) \prod_{i=1}^nP(x_i|y)}{P(x_1,x_2,....,x_n)} $$

## Different types of Naive Bayes estimation
### **[Reference](https://scikit-learn.org/stable/modules/naive_bayes.html)**