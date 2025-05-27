# Koala and Detoxify: Content Label and Severity Level Analysis

I will refer to our dataset as responses.json

Input Guard Rails:





# Koala 
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

Conversion To Our Labels
1. H -> hate speech
2. H2 -> hate speech
3. HR -> reports of abuse
4. OK -> safe
5. S -> sexual
6. S3 -> reports of abuse
7. 
7. V-> violence
8. V2 -> violence

### Accuracy Function

Input: Prompt

Output: Dictionary that is exported to Weave as additional columns

1. Match: Expected Label and Mapped Label match. 
2. Predicted Label: Raw output from Koala 
3. Mapped Label: Mapped label from the raw output 
4. True Label: Expected Label (From responses.json)

After running the 32 examples





## Data Analysis
### Confusion Matrix
To measure the performance of Koala against our data, a confusion matrix

Measured 
1. Precision - 
2. Recall - 
3. F1 Value - 



"H":"hate speech",
    "H2":"hate speech",
    "HR":"reports of abuse",
    "OK":"safe",
    "S":"sexual content",
    "S3":"reports of abuse",
    "SH":"N/A",
    "V":"violence",
    "V2":"violence",
