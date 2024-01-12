# ConSentiment: A Graph-Based Regularization Framework for Sentiment Analysis with Consensus and Uncertainty Reduction

In recent years, various methods for Aspect-Based Sentiment Analysis (ABSA) have been proposed, each with their own advantages and disadvantages. This has led to the exploration of ensemble strategies that combine these models. However, current methods face challenges in reaching an agreement among ABSA models. The main difficulty arises because each model might generate different aspects, some of which could be partly correct, and associate various sentiments with each aspect. This variation in output makes it complex to build a consensus. Moreover, ABSA tasks also encounters issues dealing with unclear aspects and sentiments, especially when exploring different domains and languages.

We introduces ConSentiment (Consensus-Based Sentiment Analysis) as a new approach to address these challenges. We propose ABSA ensemble learning principles, called Consensus Information Graph and Uncertainty Reduction, to find a consensus among ABSA base models.

- Firstly, ConSentiment suggests a consensus graph by connecting similar reviews, aspects, and predicted labels, emphasizing agreement among base models. The graph deals with uncertainty by also connecting reviews with similar aspects but different sentiments, effectively managing the ambiguity in ABSA tasks.
-  Secondly, ConSentiment introduces a graph-based ABSA regularization framework that prioritizes consensual information, where nodes with higher consensus more strongly share label information with neighbors, reducing uncertainty where base models disagree on sentiment.

Experimental evaluations across multi-domain and multi-lingual scenarios show that ConSentiment outperforms individual ABSA base models, including those based on BERT and GPT, as well as committee-based voting models.

### Acknowledgments

The authors of this work would like to thank the Center for Artificial Intelligence (C4AI-USP) and the support from the FAPESP grant 2019/07665-4 and from the IBM Corporation. Moreover, this work was supported by \textit{Ministério da Ciência, Tecnologia e Inovações}, with funds from Law 8248, of October 23, 1991, PPI-SOFTEX, coordinated by Softex and published Residence in TIC 13, DOU 01245.010222/2022-44.
