# Koala and Detoxify: Content Label and Severity Level Analysis

I will refer to our dataset as responses.json

Input Guard Rails:

## Data Set

Input data is expected in `respones.json` in the following structure:

```json
{
 "expected": "<expected_label>",
 "question": "<prompt>",
 "prompt_number":"<prompt #>"
}
```




### Koala 
I used Koala for the predicition of content labels:

Koala outputs one of the nine content labels: 

1. H - Hate Speech
2. H2 - Hate Speech / Threatening
3. HR - Harrassment
4. OK - Safe
5. S - Sexual Content
6. S3 - Sexual Content / Minors
7. SH - Self Harm
8. V - Violence
9. V2 - Violence / Graphic Imagery

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

# Accuracy Function

Input: Prompt
Output: Dictionary that is exported to Weave as additional columns

1. Match: Expected Label and Mapped Label match. 
2. Predicted Label: Raw output from Koala 
3. Mapped Label: Converted label per our categories
4. True Label: Expected label from responses.json

After running the 32 examples





## Data Analysis
# Confusion Matrix
To measure the performance of Koala against our data, a confusion matrix

Measured 
1. Precision - 
2. Recall - 
3. F1 Value - 

![Generated Confusion Matrix](images/confusion_matrix.png) 