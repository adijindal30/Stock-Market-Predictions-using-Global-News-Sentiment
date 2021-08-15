# Stock-Market-Predictions-using-Global-News-Sentiment

# ECO399A Project

## Prediction of the stock market using News

## Headline

#### Aditya Jindal | 17817048

#### Supervisor: Prof. Wasim Ahmad


## Abstract

##### From the age of Indutrialization, stock market predictionis an area of extreme

##### importance to an entire economy. The behavior of humaninvestors

##### determines the stock price, and a smart investor candetermine stock prices

##### by using openly available information to predict howthe market will act by the

##### change in human bhevaior. Financial news articlescan thus play a significant

##### role in influencing the stock movement as humans respondto the surrounding

##### information through social media, news, and othersocial interactions.

##### Previous research has suggested a relationship betweennews report and

##### stock price movement, as there is some time lag betweenwhen the news

##### article is released and when the market has movedto reflect this information.

##### Also, they are inherently volatile and reflect diversemacroeconomic and

##### microeconomic trends, which makes them difficult topredict.

##### In this undergraduate research project, we use naturallanguage processing in

##### conjunction with a deep neural network to predictthe daily change in the Dow

##### Jones Industrial Average price. Our model was slightlybetter than random

##### guessing for indicating whether the stock market wouldrise and fall on a given

##### day. In a nutshell, we built a text classificationsystem to categorize news

##### articles containing references to world news and predictthe impact of world

##### news on the stock price.


## Introduction

```
Much of the world’s wealth can be found in large financialmarkets, which allow
individuals and big/small organizations to purchaseshares of companies on a public
exchange. The price of a company’s shares is subjectto a wide range of
variables–some intuitive, i.e., systematic risk, andsome not, e.g., non-systematic
risk. The task of using these inputs to predict astock’s price movement is a
trillion-dollar industry.
For decades, only large financial corporations andexpert analysts have had access
to the data required for understanding financial markets.In the past decade,
however, publicly-provided market information hasenabled independent
investors(freelancers) to make much more intelligent,informed investment decisions.
For most of the stock market history, investment decisionshave been made by
educated and informed humans. Though the stock markethas displayed a
consistent overall trend for its existence, the resultsof attempts to ’beat’ the stock
market have been mixed. It is not clear which variablesinfluence stock prices and by
how much. Microeconomic trends and consumer preferencesoften determine the
price of a single stock, but these factors can bechallenging to predict; even experts
sometimes perform this task very poorly.
For this research project, we focus on predictingthe price of the Dow Jones
Industrial Average (DJIA), a price-weighted indexof thirty large American companies
representing the state of the domestic economy asa whole. Since significant
changes in the DJIA index are caused by factors thataffect each of its constituent
stocks, its price reflects macroeconomic trends. Previousresearch has shown that
macroeconomic trends can be predicted based on dailynews, and further research
shows that news headlines alone can be used to achievethe same results.
We have modelled a neural network that predicts thefloat amount by which the DJIA
index increases or decrease based on a textual inputconsisting of twenty-five news
headlines concatenated into one string per day.
```

## Related work

```
Past research work had used text drawn from varioussources to generate
predictions about stock market movement. Another researchwork used Twitter to
extract public opinion and was able to use a neuralnetwork to transform these
opinions into relatively accurate predictions aboutthe DJIA closing price. Other
leading researchers have experimented with event detectionalgorithms and other
natural language processing techniques to create richinput for neural networks to
learn their relationship to stock prices.
Others have focused on developing a neural networkarchitecture that can convert
these inputs into accurate predictions. Since macroeconomictrends are closely tied
to public opinion and sentiment, effective sentimentanalysis is critical for our
objective. One promising model uses a convolutionalneural network in conjunction
with a recurrent neural network to conduct sentimentanalysis of short texts. This
work applies to our objective because it extractssentiment from too brief documents
to contain many contexts, like news headlines.
```
## Dataset and Features

```
We used a publicly available dataset ofDow JonesIndustrial Average (DJIA) prices and
daily news articles available from Kaggle. The DJIAdata included the opening and
closing price of the DJIA, plus other metrics suchas its daily trading volume, daily
high, and daily low, for every market day from August8, 2008 through July 1, 2016.
This dataset can also be downloaded from Yahoo Finance.
```
```
Fig: An example of all the metrics included for one day in our dataset.
```

The news data included the 25 most popular news headlinesfor every day (including
weekends) in the same range, sourced from Reddit News.The 25 news headlines
for each day are those that received the most votesfrom Reddit users.

For example, corresponding to the price change on1st July, 2016, there are 25
global news headlines ranging from "The presidentof France says if Brexit won, so
can Donald Trump" to "Spain arrests three Pakistanisaccused of promoting
militancy".

Our data included 10% test set of our total data,dev set included 10% of our total
data, and finally train data included 80% of our totaldata.

Another key feature to each day in our dataset isthe previous 5 days’ stock price
changes. The intuition behind this feature is thatstock prices depend not only on the
daily news headlines, but also recent previous stockprice movements. The number
of previous stock price changes is a hyperparameterthat we tuned appropriately.‘
Thus, we implemented an LSTM to create a sequentialmodel in addition to
newsheadlines over time. Our workflow and architecutreis explained in the Methods
section of this paper.

For our news input data, we clean the data beforetraining our model. To do so, we
convert the headlines to lower case, replace contractions,remove unwanted
characters, and remove stop words that are irrelevantfor this natural-language


```
processing task. Next, we create embeddings, which are low-dimensional feature
vectors representing discrete data like strings or,in this case, news headlines. We
use pre-trained, 300-dimensional GloVe embeddingsfor all words with
corresponding GloVe embeddings and use random embeddingsof the same size for
words not found in GloVe’s vocabulary. The GloVe vocabularyis used to create a
vector space representation of words to capture semanticand syntactic meaning.
For more information on GloVe, see references.
```
```
To structure our previous 5 days’ stock price changes,we create a two-dimensional
matrix with rows that represent each day in our corpus.Every row then contains the
last 5 days’ stock price changes. Thus, we end upwith a (1988,5) NumPy matrix. For
the first 5 days, we zero-pad the non-existent entries.For instance, on July 1, 2016,
the first row of this input data would be like thisin the given figure.
```
```
Fig:This is a sample of the first 5 rows of our two-dimensionalarray. The first
row corresponds to the previous stock price changes on August 8, 2008 which is all
zeros since this is the beginning of our dataset.The sixth row corresponds to
August 13, 2008 which contains the five previous stockprice changes.
```
## Methods

```
In this study, supervised methods and econometricsmethods have been explored:
Linear Regression, Time Series such as AutoregressiveIntegrated Moving Average
(ARIMA), machine learning methods like SVM, Randomforest and advance deep
learning methods like sequence models including LSTM,etc.
```

```
I. A basic Linear regression model :
```
```
𝑌 = 𝑋β + ∈
```
```
We set this traditional linear regression model toset up baseline results for
our bilateral trade model, we further induce somedummy variables for the
same to see the better R^2 results.
```
```
II. ARIMA model:
```
### 𝑙𝑛(𝑌𝑡)= α + β 𝑙𝑛(𝑌𝑡− 1 ) + ε

```
To investigate further we introduce more lags termsin this model, to achieve
better relationships between time agnostic variables.
```
**III. Random forest:**

```
Random forest is another ensemble machine learningmodel builds upon
multiple decision trees and joins them together toget a more accurate and
consistent prediction. They assumed to be a good choicefor predictive
modeling because they are easy to understand due totheir tree type structure
and are also very robust. The basic goal of a decisiontree is to split a population
of data into smaller sections.
```
```
IV. Fully Connected, Feedforward Neural Networks usingLogarithmic
Features:
```

```
Since we assume no a priori knowledge on interactionsbetween features (except for
Gravity Model), we utilized a fully connected neuralnetwork. To find an optimal
architecture of the neural network, we will comparearchitectures with different layer
and node-setups
```
**V. Fully Connected, sequence LSTM models:**

```
We use this LSTM model of deep learning to capturethe time effect, this is similar to
the ARIMA model, but it's quite robust and can accountfor non-linear relationships
too unlike the basic time series model.
```
```
We are using stacked lstm , which is we take 2 ormore layers of LSTM one onto
another , which captures the relationship and capturesnon-linear behavior more
accurately.
```

The inputs to our neural network are sequential, aswe discussed in the above
section. More specifically, every day from August 8, 2008, to July 1, 2016, has 25
news headlines. Additionally, each input vector alsocontains the difference between
the DJIA stock price at time t and the DJIA stockprice at t− 1. In this model, we are
not predicting the actual stock price; instead, wepredict that stock price will rise or
fall for each day, i.e., the change in its price (apositive or negative difference). To
reduce the effect of Inflation, we also normalizethe data by subtracting the mean
and dividing by the standard deviation of the stockprice changes.

As our baseline method, we simplified this probleminto a pretty straightforward
sentiment analysis task. Since we are working withbrief news headlines, we used
Wang et al’s(refer below) approach to sentiment analysis,feeding sentence
embeddings into a CNN that takes the local featuresand inputs these results into an
RNN for sentiment analysis of short texts or newsheadlines. The output of the RNN
is fed into a fully connected network, which aimsto learn its relationship to
sentiment. The model architecture is shown in thebelow figure. We used the mean
squared error as our loss function and Adam optimizationalgorithm. This network is
fully implemented using Tensorflow-based Keras.


```
Model architecture
```
In this architecture, we apply windows and convolutionaloperations with different
matrices in the CNN while maintaining sequential informationin the texts. Then, the
RNN takes in these encoded features and learns long-termdependencies.

More specifically, for our RNN, we use a Long Short-TermMemory (LSTM) that
learns long-term dependencies. Rather than using atraditional feed-forward network,
an LSTM detects features from an input sequence andcarries the information over a
long distance. Our model is built on the Keras andscikit-learn machine learning
frameworks.

In addition to the sentiment analysis on the textualinputs, we also included the
change in the price of the stock market for the previoustwenty days. We feed the
previous twenty days’ price changes into an LSTM,which then becomes another
input to the fully connected network. This shouldhelp our model recognize long-term
trends in the DJIA’s index price.


## Experiments

Our baseline model relied on news data as the solesource of input. And to keep it
simple, we apply ridge regression and then an auto-regressivemodel. With this
model and the optimal hyperparameters found, we couldcorrectly predict whether
the stock market would rise/fall 54% of the time.This was our starting point. From
this baseline, we iterated through several experimentsto develop our final model.
Our hyperparameters include:
❖ Different learning rates.
❖ The value of p in dropout.
❖ The number of layers to include in the fully connectedlayer.
● The number of elements in the LSTM.
● The number of convolution layers on the textual data.
We used both grid search as well as random searchto find the best possible values
of these hyperparameters.
To put the performance of our model in perspectivewith other datasets, we created a
simple logistic regression model that uses NYC weatherdata to predict whether the
price of the DJIA rises or falls. With this simple(and logically unrelated) data, we
were able to predict positive/negative price movementsin the DJIA with 52%
accuracy–slightly better than flipping a coin.
We tried six different models, varying the input data(weather, news, and previous
stock prices) and the model architecture. Our bestaccuracy results came from the
most profound and most comprehensive networks, whichused more convolutional
and fully connected layers.


## Results

Using the previous stock price changes and globalnews headlines, we achieved
55.28% accuracy with a mean absolute error of 71.40,meaning that this model more
accurately predicted price changes than our baseline.

Further tuning our hyperparameters, we increased thedepth and width of our
network by using 3 Fully Connected(FC) layers, 2 convolutionallayers, and 128
hidden dimensions in the fully connected network tofurther increase our accuracy,
reaching 58.28%.

Our best model used 5 fully connected layers, 5 1Dconvolutions for the textual news
data , and 256 hidden dimensions in the fully connectedlayer, achieving 61.31%
accuracy in predicting a positive or negative changein the DJIA, a 7.03% increase of
our baseline model. Our model was well-fit to thedata, since the training error for our
our best model was around 58% while our test accuracywas 60.31%.

One notable trend throughout our experiments is thereasonably high mean absolute
error, which provides insight on the average magnitudeof error in our predictions
regardless of direction. Our model is best suitedfor a binary classification task of
predicting a rise/fall and struggles with determiningthe magnitude of stock price
changes.

```
Model Accuracy RMSE
```
```
Logistic regression 52.21% 1.
Ridgre regression 53.30% 1.
AR model*(Baseline) 54.1 % 1.
CNN and LSTM(no lag) 56.5% 0.
CNN and GRU(with lag) 58.5% 0.
CNN and LSTM(with lag) 61.4% 0.
```

```
Table:Different models and accuracy I tried along the project
```
## Conclusion

```
Advance machine learning models outperform traditionaleconomics models. Since
our data has both news headline and time dimensions,using the LSTM model
extended to a panel setting may be a better approachthat we can take in the future.
Since the neural network approach expects to showgood results with capturing
nonlinear interaction effects among other economicindicators, including many more
economic features may also be effective in furtherimproving our prediction.
Also, our model was able to outperform our baselineat the task of predicting whether
the DJIA would rise or fall each day. We started at54% accuracy and eventually got
to 61.31%–a marginal difference and still barely betterthan a coin flip. This model
didn't completely succeed at its task of predictingthe overall stock price movements.
However, it did far better than even many human analystscan do.
It was promising that our NN model reacted positivelyto the addition of historical
market price data, as that shows that the basic modelwith a textual news input can
be augmented and bolstered by building in relevantalternate input sources. This
makes sense because a typical stock market is subjectto many factors, any of which
can cause the market's price to swing wildly. Historicalprice data is just one, and
there are many others we could include in our model.Some exciting ideas include
tweets from the presidential Twitter account and priceinformation from foreign
markets. It could also be interesting to apply thismodel on a company-specific basis,
feeding it news headlines focused on a single company.We suspect that this data
would be even more relevant because it is directlyrelevant to the company's health
(which should be reflected in its stock price).
```
## Tools Used

##### 1) Pandas for data management

##### 2) Scikit-learn for linear regression/ kernels

##### 3) Matplotlib for plotting

##### 4) Keras for neural networks

##### 5) Numpy for data transformation


##### 6) Tensorflow for sequence models based stacked LSTM

##### 7) Jupyter Notebook for coding and plotting graphs


## References

1. Xiaojun Zeng Johan Bollen, Huina Mao. Twitter moodpredicts the stock
    market.Journal of Computational Science, 2(1):1–8,2011.
2. Ting Liu Junwen Duan Xiao Ding, Yue Zhang. Deep learningfor event-driven
    stock prediction. 2014.
3. Zhiyoung Luo Xingyou Wang, Weijie Jiang. Combinationof convolutional and
    recurrent neural network for sentiment analysis ofshort texts.COLING, 2016.
4. Aaron7sun. Daily news for stock market prediction,2016.
5. Christopher D. Manning Jeffrey Pennington, RichardSocher. Glove: Global
    vectors for word representation, 2014.
6. David Currie.
    Predicting-the-dow-jones-with-headlines.https://github.com/Currie32/
    Predicting-the-Dow-Jones-with-Headlines.
7. Open Source Francois Challet. Keras deep learningframework.
8. Open Source. scikit-learn machine learning framework.


