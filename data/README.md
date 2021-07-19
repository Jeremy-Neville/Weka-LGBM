# What to do with the Data given

As shown within this folder, you can see that there is both a training and testing file. These two files were the files utilized to acquire the results displayed in the previous README. Within this README you will see how exactly to select the training data and how to select the testing data after the classifier has finished training the model with the training data. 

It is important to note that the data provided is not the only data set you can utilize. There are various training and testing data sets that you can utilize or create with the LGBM and CatBoost pyscripts.

## How to open and set the training data

![image](https://user-images.githubusercontent.com/49813790/126164732-1b5eaa44-4116-4157-a84d-a8562de29df6.png)
When you have opened Weka, you will see the default menu, first click explorer to get to the location where you can set the training data.



![openfile](https://user-images.githubusercontent.com/49813790/126163130-99c2990d-b1f2-4acc-89a2-f918fe511dcb.PNG)
As shown above, once you have done that you will be in the WekaExplorer, more specefically you will be in the preprocess tab. You will need to first click the open file button to be given a prompt where you will select what training set you will provide.


![openingtraining](https://user-images.githubusercontent.com/49813790/126164972-044fb25e-ab53-4d1a-ad60-fdb1dcac32a2.PNG)
As displayed above, the prompt will allow you to traverse your files to locate the training data set you want to provide. Select the training data you want to provide, in this instance I selected KDDTrain+. Once you have made your selection,click open.

![Openedtraining](https://user-images.githubusercontent.com/49813790/126165365-bfd783fc-57ec-4597-b1ed-5b2d945efb93.PNG)
After you clicked open, you will now be given this screen in the preprocess tab. Once you see this, you will see that the training data is now present within your preprocess tab indicating that you can now run classifications to train with the provided data.


## How to open and set the testing data

![setting](https://user-images.githubusercontent.com/49813790/126166555-c0347d90-f1de-456b-9d82-630412e2817f.PNG)

To be able to set the testing data you must first run a classification with the training data. Once you have finished running the model and are givien the results, first go to  supplied test set and click set. It will open a prompt similar to the prompt given when setting the training data.

![settingtesting](https://user-images.githubusercontent.com/49813790/126167387-3c57a864-cf3d-4897-8ede-eb8c21a49757.PNG)
Just like before, you will need to navigate to find where you placed your testing data file. Once you have located it you just select it and then select open.

![finalsetting](https://user-images.githubusercontent.com/49813790/126168020-24104024-477c-4449-bf63-87b3fd50396f.PNG)
Even after you set the testing data you will not see any difference with the window unlike when you selected your training data.


 As shown before, the process to set your testing and training data is not too difficult however it is documented just to ensure the testing data isn't provided prior to the training data.

