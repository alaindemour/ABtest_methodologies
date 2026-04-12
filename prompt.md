Please genereate a small jupyter notebook  that can be used to generate a slide deck with generate_slides.sh, similar to presentation.ipynb, but this one is to explain the performance a a machine learning model used for fraud detection

The Model  is it and XGBoost classifier, the performance of the model is captured in the csv file in Fraud Detection - ac_thresholds_merged.csv, for each threeshold we could use based on the a score column which is just 1 - probability of fraud, it show the total count of True Posistive (TP), True Negative (TN), False Positive (FP) and False Negative (FN), as well is the Recall, Precision and False Positive Rate.

1. Have a first set of cells explaining the genereal setup:
   1. XGBoost binary classifier but with very unbalanced classes, explain that the prevlance reate is very low arond 0.08% as shown by that data
   2. The False Positive number as for all fraud model are computed with post hoc analytics to address the kown counter factial problem, explain wha the counter factual problem is
2. Then a few cells showing the current methodology to compute perforamnce which is using the ROC AUC, have a slide with the matplot lib figure showing the full ROC curve, the ROC AUC number,  the no skill diagonal, show the AUC on the matplot lib figure, but also the cutoff point where the model is opereating which is the first  5 row of the csv file, so a small sliverk, also show the confusion matrix with both absolute number and percentqage in each quandrant for the first row of the data set.
3. a call explainig hte limitaiton fo the ROC AUC single nunbers, and issues with extreme class unbalance
4. Then a few slides explaining the Precision Recall methdology, show the full Precision Recall curve, with the horizintal prevalance number , also show the are the model is actually operating at (first 5 rows of the data set)
5. Also explain that the Precision Recall is not perfect , and that a Lift curve can add some color to the analsyis, how the lift curve and explain how to interpret it
6. Then disxuss how to decide how a insult rate can be derived from all that that is how many good customer do we need to flag whether they succceed or not in exiting the flaggin, either by stepping up and just abandohing to catch a fix proporotion of all our frausster.
   
