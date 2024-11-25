# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
Jacob Simpson created the model, is is a Categorical Naive Baybes using default hyperparameters
## Intended Use
Classify and predict salary based off many available features.
## Training Data
Is a subset, using the train_test_split() mehtod, of the data on census income available at https://archive.ics.uci.edu/dataset/20/census+income
## Evaluation Data
The other subset of above data, using 20% of reserved training data.
## Metrics
The model was evaluated with a F1 score also shwoing the precision and recall it recieved 
Precision: 0.6566 | Recall: 0.7352 | F1: 0.6937 overall.

## Ethical Considerations
This is cencus data, all PII is removed before I even had access to it. However in the unlikely event someone was maliciously looking for demographic data to make true but logically shaky claims it would be possible.
## Caveats and Recommendations
I have a whole career ahead of me to get better at deploying ML solutions, so my solution may not be the best possible predictions, because of my limited time for study I initially had an F1 score at .43. This was not good enough, but through further attempts I was able to hit .693 which I deemed close enough for the intents of this project. If I had more time I would run many more attemps and get a better feel for all the various methods.