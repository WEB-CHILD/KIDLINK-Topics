# Overview of model changes
This document works as an aid to what have been done to the model generation proces to create a fitting model. 

# First iteration
Added stopwords related to different languages and some stopwords related to web content, such as html, http, www.
Added stopwords related to Odd Presno - the inventer of KidLink
Added stopwords related to years and dates.
Minimum number of documents per topic was 30 when run the first time, which provided more than 3000 topics.
Changed minimum number of documents to 500, which provided only 11 topics. 
Changed minimum number of documents to 100, which provided 350 topics with 164K outliers
Added stopwords related to advanced search, such as 'advanced' and 'search' as these words are responsible for one of the biggest clusters in all iterations. 
Changed minimum number of documents to 80 to get number of outliers down. This creates 525 topics and 146K outliers
Changed amount of keywords to include to 50, instead of 10