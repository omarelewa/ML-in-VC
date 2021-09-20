# CapitalVX: A machine learning model for startup selection and exit prediction

## Abstract

- This paper develops a machine- learning model called CapitalVX (for “Capital Venture eXchange”) to predict the outcomes for startups, i.e., whether they will exit successfully through an IPO or acquisition, fail, or remain private.

-

## Keywords

- Machine Learning
- Ensemble Models
- Exit Prediction
- Funding Prediction
- Private Equity
- Venture Capital
- Crunchbase
- USPTO
- Model Evaluation

## Previous Research

There is a growing literature on analyzing venture capital investments using alternate data or methodologies, such as machine learning.
Broadly put, these approaches extend and improve on traditional econometric approaches.
The literature may be broken down into two broad strands:

- (i) identifying successful investors.
- (ii) identifying successful startup investments.
There are of course, several taxonomies for this well-researched area, such as the excellent survey by Rin et al. (2013).25
In work associated with identifying successful investors, graph theory has become an important tool. One such approach relies on investor networks, and work by Glupker et al. (2019)16 has shown that network position determines the success rate of investors. The paper shows that it is in fact easier to predict unsuccessful investors. The study offers a two-step approach, cut by industry first, followed by community construction within industry, i.e., focus on the industry of the startup followed by the use of a machine-learning model. This combines financial data with graph- theoretic ideas and machine learning algorithms.
Gupta et al. (2015)17 developed InvestorRank, a method for identifying successful investors based on position in a network, such as being close to an exemplary investor or super-angel. The result of the research shows potential in discovering investors who will become successful. InvestorRank flags investors who follow a general trend of improvement when compared to their preceding snapshot based on a threshold. Bubna et al. (2020)7 analyze a VC network to uncover communities, i.e., small groups of VCs who tend to frequently work together. They show that startups funded by community VCs (as opposed to non-community VCs) tend to have higher probabilities of suc- cessful exits as well as faster exits. Geographical distance is a determinant in VCs working together and therefore is an indirect determinant of startup success, see Sorenson and Stuart (2001)28. Similarly, Adcock et al. (2012)1 analyze a bipartite investment network of investors and investees with links based on investments between them. Personal investors evidence the highest average clustering, tech companies the lowest, indicating that they choose to acquire small firms rather than invest in them.
Syndication of VC investments is also a driver of better performance by startups, as evidenced in the epochal paper by Brander et al. (2002).6 Their empirical analysis examines whether superior performance of syndicated startups comes from better venture selection or from monitoring post-selection, finding that the latter has more impact. Das et al. (2011)10 show that the selection effect positively impacts exit returns, whereas monitoring leads to higher prob- ability of exits, and faster exits. Bernstein et al. (2016)3 show that VCs who are connected to their startups via close physical proximity (direct flights) are able to tightly monitor their investments and it leads to better outcomes.
The other side of the coin from identifying successful investors is predicting which startups will see a positive exit. Antretter et al. (2019)2 analyze the concept of “online legitimacy” and demonstrate the power of machine-learning, by using Twitter as a data source, to distinguish between those ventures which are bound to fail and those which are not. Xu et al. (2017)33 compare their results against the Crunchbase database and aim to narrow the list of portfolio companies that a venture capital investor may consider, thereby offering results that are similar to the motivations of this paper. Srinivasan et al. (2014)30 provide further evidence that potential high-value investees can be identified through an analysis of company features.
In contrast, Dellermann et al. (2017)13 opt for the approach of combining machine learning with the traditional human approach in what is known as a hybrid intelligence method. Their observation is that humans fill in the gaps where machines fail, namely when it comes to “soft” information and navigating moments of “unknowable risk”.
Krishna et al. (2016)20 performs research similar to ours where funding and business data is pulled from sources such as Crunchbase and Tech Crunch while a combination of Random Forest, Bayesian Classifiers and other models are used to predict company outcomes. However, the key difference is that our analysis aims to enable investors to identify the next unicorn, whereas Krishna et al. (2016)20 attempts to assist private entities in securing future funding, as do Biesinger et al. (2020).5 Sharchilev et al. (2018)27 use Crunchbase and web-based information to achieve good accuracy in predicting follow-on funding, and we get similar levels of accuracy using Crunchbase data as well, though with different models. For exits via mergers and acquisitions, an excellent modeling analysis using Crunchbase data is undertaken in Xiang et al. (2012)32 who achieve metrics similar to ours. Overall, in contrast to the previous literature, this paper achieves high levels of accuracy on a much more extensive dataset, using data from all stages of the startup cycle. We also provide feature importance and have built the algorithm into a web system for exit prediction, as discussed later.

## Introduction

## Methodology

## Commentary
