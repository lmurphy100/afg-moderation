
# Koala Analysis
Koala was used for the predicition of several content labels:
The generation of the Weave report was done in `koala_weave.py`
## Koala Output

Koala outputs 1 of 9 content labels: 

* H - Hate Speech
* H2 - Hate Speech / Threatening
* HR - Harrassment
* OK - Safe
* S - Sexual Content
* S3 - Sexual Content / Minors
* SH - Self Harm
* V - Violence
* V2 - Violence / Graphic Imagery

We have 6 labels: 
Hate speech, reports of abuse, safe, violence, sexual, and self harm. 

## Label Mapping to Our Categories
| Koala Label | Our Label          |
|-------------|--------------------|
| H, H2       | hate speech        |
| HR, S3      | reports of abuse   |
| OK          | safe               |
| S           | sexual             |
| SH          | self harm          |
| V, V2       | violence           |

## Weave Results
### Accuracy Function

Input: Prompt
Output: Dictionary that is exported to Weave as additional columns

1. *Match:* Expected Label and Mapped Label match. 
2. *Predicted Label:* Raw output from Koala 
3. *Mapped Label:* Converted label per our categories
4. *True Label:* Expected label from responses.json

After running the 32 examples, it intialized a weave report.

Image of the report

The report was exported and used in `koala_Weave_Generation.py`


## Data Analysis
### __Confusion Matrix__
To measure the performance of Koala against our data, a confusion matrix measured:

* Precision - 
* Recall - 
* F1 Value -

Because Precision and Recall are complements to both shortcomings, and F1 Score is a compiled version of both, we'll use F1 Score as a metric for which categories our model is predicting well. 

![Generated Confusion Matrix](./images/confusion_matrix.png) 
### Outputted Classifcation Report


### Classification Table 
 
![Table](./images/table.png)

Besides the Confusion Matrix, we can look at the Receiver Operating Character (ROC) curve to
analyze how well our model distinguses between classes and the overall performance. 

Aggregated Probabilities: Because there is a class mismatch with Koala, we need to sum probabilities that correlate 
with the conversion table above. Then feed it into the Receiver Operating Characteristic (ROC) curve. (*Total probability <= 1*)

We use Micro-Averaging because our dataset classes are widely imbalanced. The Area Under Curve (AUC) metric indicates
how well our model distinguses between classes and the overall performance. 

![Micro-Averaged ROC Curve](./images/microaverage-ROC.png)

Due to the class mismatch and need to estimate, this chart is not entirely accurate. More data is needed 
to truly assess the AUC metric. 

## Koala Conclusion

Based on the data analysis above. The model is:

* **Strong** with identifying 'Safe' 
* **Good** with identifying 'Sexual content', 'Violence', 'Hate Speech', 'Self Harm'
* **Poor** with identifying 'Reports of Abuse'

Koala doesn't directly have a reports of abuse possibility (We had to supplement with Harrassment or Sexual/Minors), so this makes sense. 
If we wantesd to increase the F1 Score or Accuracy for ethier column, we'd need to ethier finetune Koala with our data in a format it can understand, or use a different model.
