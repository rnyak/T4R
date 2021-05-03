First, the user interactions are grouped by sessions. All datasets provide session ids, except the ADRESSA dataset, for which we artificially split sessions to have a maximum idle time of 30 minutes between the user interactions.

Repeated user interactions on the same items within sessions are removed from news dataset, as they do not provide information gain. For the e-commerce domain, repeated interactions are common when users are comparing products and recommending items already interacted can be helpful from a user’s perspective e.g., as reminders [malte2020empirical, jannach2017session, lerche2016value, ren2019repeatnet]. Thus we remove consecutive interactions in the same items, but allow them to be interleaved, e.g. the sequence of interacted items 𝑎𝑎𝑎𝑏𝑐𝑐𝑎 becomes 𝑎𝑏𝑐𝑎. We remove sessions with length 1, and truncate sessions up to the maximum of 20 interactions.

The sessions are divided in time windows, according to the unit: one day for e-commerce datasets and one hour for the news datasets. The reason for is that interactions in the news domain are very biased toward recent items. For example, in G1 news, 50% and 90% of interactions are performed on articles published within up to 11 hours and 20hours, respectively. So, training those types of algorithms on a daily basis would not be effective for recommending fresh articles, as they would not be included in the train set.

We also explore the usage of side features by Transformers architectures (RQ3). The following table presents the additional features other than the item id and their feature engineering that were used to by our experiments to address RQ3, which explores different techniques to include side information into Transformers.
It is worthwhile to note that the YOOCHOOSE dataset have a single categorical feature (category), but it is inconsistent over time in the dataset. All interactions before day 84 (2014-06-23) have the same value for that feature, when many other categories are introduced. Under the incremental evaluation protocol, this drops significantly the model accuracy for the early subsequent days so we cannot use that feature for our purpose. As there was no other categorical feature left for the YOOCHOOSE dataset, we decided not including it for the analysis of RQ3.

<center>Table 3. Datasets feature engineering</center>


Features/Dataset| REES46 eCommerce | G1 news | Adressa news | Preprocessing techniques |
--- | --- | --- | --- |--- |
Categorical features| category, subcategory, brand | User context features: region, country, environment, device group, OS | category, subcategory, author and user context features: city, region, country, device, OS, referrer |Discrete encoding as contiguous ids |
Item recency features | item age in days (log) | item age in hours| |Standardization for the e-commerce datasets and GaussRank for the news datasets |
Temporal features |  | hour of the day, day of the week | .. | Cyclical continuous features (using sine and cosine|
Other continuous features |  price, relative price to the average of category |-| - |Standardization|

