Project : Expert assining system

Determining expert relevance with respect to the board subject and candidate's area of interest
It is basically a NLP project, problem statement given by "Ministry of DRDO".
We have used NLP Pipeline , python libraries(numpy, pandas...), TF-IDF vectorizer, cosine similarity to match profiles of expert to that of candidate's area of interest and board subject.
For that we have calculated 3 three scores:
1) Profile Score - which is a similarity score between an expert and candidate
2) Matching Score - similarity score between an expert and board subject
3) Relevancy Score - similarity score between profile score and matching score ...(This score shows which expert should interview which interview which candidate with respect to the board subject)  
