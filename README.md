# UIFTextAugmentation
UIF Text Augmentation Project

## Prerequisite

Below is the list of libraries that you need to install. Even though requirement.txt contains all the library list, you don't need to manually install all of them. 
  ```
  tensorflow(-gpu)==1.14
  tensorflow_hub
  pandas
  numpy
  sklearn
  ```
## Data
  
  Your .csv data should have below columns
  ```
  ID, Text, Label
  ```
  
  For the training, you can use convertDataFrame (included in Utility.py) to change the format to like below
  ```
  ID, Text, Class0, Class1, ..., ClassN
  ```

## Pretrained bert model
  
  You will find where to download the trained model in the folder. 

## How to use the code
  ```
  .\Code\TextAgumentation.py : Main text augmentation template. The default method is recursive synonym replacement method. 
  .\Code\UIFClassifierBert_Train.py : Slightly modified version of BERT based classifier. 
  .\Code\UIFClassifierBert_Predict.py : Predictor using the trained model from the above code.  
  .\Code\Utility.py: Contains various utility function including convert function.  
  ```
  So the flow should be like below:
  1. Prepare the training data with text augmentation: Feed your original text data (in the original column format) to generate augmented text data
  2. Convert the data to BERT supported format (using convertDataFrame)
  3. Train your model with the data
  4. Predict with the model
  5. Analyze the result with some functions in Utility (like ROCAnalysis)
  
  Have fun and leave comment if you have any questions
