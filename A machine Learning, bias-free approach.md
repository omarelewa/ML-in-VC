# A machine Learning, bias-free approach for predicting business success using Crunchbase data

## Highlights

- Forecasting a company’s success with supervised machine learning methods.
- Preventing look-ahead bias.
- Direct applicability as a decision support system.
- Promising results were obtained with the gradient boosting classifier.

Study uses data collected from Crunchbase and the final training set consists of 213171 companies.

However, we found that very often they were significantly biased by their use of data containing information that was a direct consequence of a company reaching some level of success (or failure). Such an approach is a classic example of the look-ahead bias.

## Keywords

- Startups
- Supervised Learning
- XGBoost
- Crunchbase
- Look-ahead bias

## Introduction

### Objectives and contributions

- In this study, we analyzed the problem of forecasting business ventures’ success with the use of supervised machine learning methods.
- The main objective of this paper was to conduct an experiment that would lead to developing an information system that would not be flawed with the aforementioned biases – one that could be applied in practice to predict business success.
- To the best of our knowledge, this is the first study that strongly focuses on the applicability of its results by reducing the number of biases introduced in the dataset. We achieved it by purposefully limiting the set of predictors to information known at the beginning of the company’s operations.
- Additional information about funding events, acquisitions, IPOs, and investors is held in respective tables. They include data about dates of such events, amount of collected funds, and investment type (seed, angel funding, series A, B, C, etc.).
- The people table describes individuals who are founders, investors, or employees of the organizations. The table includes the person’s name, gender, address, social media account links, organization, and position within the organization. Information about an individual’s education is held in the degrees table. Each entry might contain information about the subject of the degree, dates of matriculation and graduation, and the institution at which it was studied.
- Other tables, which were not used in the research, hold information about past jobs connecting organizations and people, parent organizations of those described in the organizations table, and industry events. There are also two tables containing descriptions of people and organizations.
We planned and conducted an experiment by crawling responses from the homepages of companies included in the organizations table with the status operating. The responses from the websites were collected using Python’s urllib library. The crawling script collects the HTTP response code of the website or the message of the exception that was raised by urllib during the processing of the request. The output of the script is stored in CSV files. Each row is indexed by the organization’s identifier in the Crunchbase dataset.
- Based on data collected in the experiment, we added a new column to the organizations table with a flag indicating whether the homepage was active or not. Organizations with HTTP response code 200 were assigned with 1 (active) in this column; all the other organizations were assigned with 0 (inactive) (Fielding & Reschke, 2014).
- Fig. 2(b) shows the ratio of companies with active and inactive homepages. We can see the minimum value for companies in the sixth year since the founding. After that, the ratio of companies with an active homepage rises. We can conclude that it takes five years for most companies to validate their idea on the market as shown in Fig. 2. 
- We selected 1995 as the lower bound of analysis to include the companies that operated during the dot-com bubble.
- Our dataset most likely does not include companies that operated in the past but have not survived.
- The oldest samples are biased towards companies that persisted through difficult times (possibly through several economic downturns).
- This will result in overestimating the sensitivity of trained models.
- However, it is reasonable to assume that after Crunchbase gained popularity, this bias reduced over time. The company mission is to be a "master database for companies". However, even with the assumption of almost full coverage of companies that have ever existed, we have to address the problem of data not being up-to-date. In particular, this is a very common situation for companies that have an ‘‘active’’ status in the database while not operating anymore

## Methodology

- The research was conducted using data from the Crunchbase database (www.crunchbase.com). Crunchbase is a platform with business information about private and public companies, founders or people in leadership positions, investors, and funding rounds (Crunchbase Inc., 2020). We applied for access to the Crunchbase database for this research. After the positive response to the request, we got access to the daily snapshots of the Crunchbase database. The data used in the research and experiments was obtained on March 10, 2020. In this section, we will further describe preparing the dataset used in the training machine learning models.
- The dataset provided by Crunchbase for research purposes consists of multiple tables that can be joined by unique identifiers. The simplified entity-relationship diagram (ERD) of Crunchbase tables is shown in Fig. 1.
The organizations table includes information about companies and investment funds. The table holds basic information such as name, HQ address, number of employees, website, social media links, email, and phone number. The summarized financial data include the number of funding rounds, the date of the last funding event, total funding, and the number of exits from investments. Crunchbase also keeps track of the status of the organization – active, closed, acquired, or ipo (public company). Each organization is also described by its primary role (company or investor) and the categories and subcategories that describe the industry it operates in.
- Using the date of company foundation in the dataset would result in a model favoring older organizations when predicting success. Our initial experiments confirmed that dates should be transformed into relative time ranges in order to avoid this scenario.
- The year of the company’s foundation was replaced with the number of years between the founder’s graduation and the company’s foundation.

## Factors for Success

- More recently, Huang et al. (2020) built a framework for assessing enterprise value using factors like number of patents, R&D employees, and share of owners among management. Enterprise valuation is an important metric that could be used to determine the success of a company.
- One of the popular milestones is the valuation of $1 billion. Privately-owned companies that reach this threshold are called unicorns, to signify the uniqueness of the achievement. The term was first used by venture capital investor Aileen Lee (2013) in her blog article and has since become popular in describing startups’ success.
- A startup is "a human institution designed to create a new product or service under conditions of extreme uncertainty" (Ries, 2011).
- It is industry knowledge that 9 out of 10 startups fail.
- Lussier (1995) used logistic regression to predict a young firm’s success using the data collected through surveys on US companies.
- Tomy and Pardede (2018) successfully used k-nearest neighbors (k-NN), naive Bayes, and support vector machine (SVM) algorithms (Tomy & Pardede, 2018). However, the datasets used in those works were relatively small, having 216 and 250 instances, respectively.
- Bento (2018) used the Crunchbase data about startups located in the US to predict acquisition or an initial public offering (IPO) using logistic regression, SVM, and random forest algorithms.
- Sharchilev et al. (2018) built a gradient boosted decision tree model predicting series A funding in the next year for companies that had already acquired seed or angel funding. They used the dataset collected from Crunchbase in monthly snapshots and enriched it with data from the LinkedIn profiles of people working at the companies.
- Crunchbase data was also used to predict investment behaviors modeled with graph methods. Yuxian and Yuan found that by using different link predictors like the shortest path in the graph or number of neighbors, it is possible to predict whether investors are going to invest (Yuxian & Yuan, 2013).
- Other alternative approaches to predicting business success include hybrid intelligence methods. Dellermann et al. proposed a framework that uses decisions made by machine learning algorithms using hard information (team size, entrepreneurial experience) and decisions made by a group of people. Both experts and non-experts would use their intuition, experience, and knowledge of the market to predict a startup’s success. The results would be then aggregated to generate the classification output (Dellermann et al., 2017).
- In many preceding studies that applied statistical methods, the dataset contained features that would not have been available at the time of the decision. Introducing this kind of look-ahead bias may lead to the final model not being useful in any real-world scenario.
- We observed such a vulnerability in the following works: Bento (2018), Dellermann et al. (2017), Krishna et al. (2016), Yuxian and Yuan (2013), and Xiang et al. (2012). It is also present in Sharchilev et al. (2018), a very thoroughly conducted study, where the authors emphasized their awareness of the potential harm that could be done through data leakage between different time periods. Nevertheless, they enriched the dataset with data gathered from Linkedin and web mentions dated long after the last sample from their original dataset. The negative impact of this operation on the performance of the final model is hard to predict.

## Commentary

