# series-classifier-ic
Research into a classification algorithm for time series by the Federal University of Minas Gerais (UFMG). In this work, we focus in the analysis of the relevance of the proposed data transformation in potentializing the effectiveness of time series classification.

# Overview
THe first transformation we apply to the original series is the Symbolic Aggregate Approximation (SAX), that aggregates to each point in the series the amplitude information. Next, we use a sliding window to group subsequent points, adding the information of temporality to each symbol of the new series. vizinhos. Using this transformation combined with the Nearest Neighbors algorithm, we can measure the effects of the transformation on the precision of the classifier.

The **full report** for the work can be seen [**here**](report.pdf).
 
