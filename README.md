# ScalableLab2

On this second lab of Scalable Machine Learning and Deep learning we fine tuned a Whisper model using the blog post on https://huggingface.co/blog/fine-tune-whisper for Galician, a language spoken on the north of Spain. We started by following the steps as prepared on the blog, and later made some improvements on feature extraction, training and UI. 


## Public URLs 

In this section, you can see the 3 links created in Lab 2, access and test them.

**Link 1: Microphone Interface Functionality**

https://huggingface.co/spaces/Victorlopo21/labScalabletask1


**Link 2: YouTube Video Interface Functionality**


https://huggingface.co/spaces/Victorlopo21/Lab2Youtube

*YouTube Link to make tests:* https://www.youtube.com/watch?v=Z2SjeZJZi6s&ab_channel=rimc7 

**Link 3: Audio File Interface Functionality**


https://huggingface.co/spaces/Victorlopo21/Lab2File


*Audio File:* The audio file to test this functionality can be found inside the folder *Resources*

## Description of the project


First of all we split the single Colab file into a pipeline:

**Feature Extraction** - Where the dataset and its processors are downloaded and the features are extracted.

**Training** - Where the features extracted and the whisper model are downloaded, then this model is trained. There are 2 versions (Model 1 and Model 2). Model 2 outperforms Model 1 (their parameters and metric results will be described below).

**Inference** - The gradio app UI that is run on the Hugging Face. The user can put the audio/youtube link/register on the microphone and the app will do the transcribtion, using the previosly traind model. Three version of this were built.


On the other hand to improve the predicitoin of our model we approached it in different ways:

**Focusing on the data**


By default we are processing the audios the model will use into 30 seconds. This means that if an audio found on the train/test dataset is shorter than that it will be padded with 0 (silence), if it is longer on the other hand it will be cropped. We wanted to analyse a little bit more the choice of this number, since it could have an influence on the final output. For this we created a dataset (array) with the measures of the lenght of all the audios found on the training set and computed some statistics on it, such as the maximum and the quartiles. This led us to the decision of using an audio lenght of 10 seconds. Unfortunately, even though the models trained on this new features and good results were obtained, we were not able to make inference with them, and since the transformers used in this project were very new, we could not find out what the problem was or how to fix (if it even was possible to do so). At the end we went back to training models on the original features extracted.


![image](https://user-images.githubusercontent.com/73105766/206696990-5eb3f944-485b-4c8d-8d62-459fb9c5b927.png)


**Focusing on the models**


After training the initial first model, we wanted to improve the performance, even though we could not change the architecture of our model. One thing we immediatly noticed is that since we were training the model on 1000 steps, on 750 steps it already had a better performance than later. One very probable explanation of this is that the model was overfitting. To solve this we wether needed to use less epochs or some reularizatoin technique. After some fine tuning and testing the best model we managed to achieve uses the following settings. Below we can see a summary of the two models tested during this Lab and their respective training parameters:

- Model 1 (Small whisper model)

	- Layers = 12
	- Width = 768
	- Heads = 12
	- Parameters = 244M
	- Steps = 1000 
	- Eval steps = 250
	- Learning rate = 1e-5



- Model 2 (Medium whisper model)

	- Layers = 24
	- Width = 1024
	- Heads = 16
	- Parameters = 769M
	- Steps = 500 
	- Eval steps = 250
	- Learning rate = 1e-5
	- Learning rate scheduler = cosine
	- Weight decay = 0.001


Model 1 is trained over the train set for 1000 steps (2.75 epochs) and is evaluated over the validation set after 250 steps. The model with the better results in the WER metric over the validation set is the one uploaded to hugging face and considered as the optimal. In our case, this would be the model after 750 steps.
Model 2 is much more complex than Model 1 and the number of parameters is higher. This increase in complexity increased also the training times, so the number of steps was reduced to 500 (limitation of resources). The validation is performed after 250 steps, so in the end we validate Model 2 twice. We have changed not only the architecture of the net, but also some training parameters such as the learning rate scheduler and the weight decay. The scheduler adjusts the learning rate between the iterations as the training progresses. The other parameter added is weight_decay. Weight decay is a regularization technique that is used in deep learning to reduce the complexity of a model and prevent overfitting. This regularization technique can be implemented by modifying the the update rule of the parameters so it is based not only on the training data but also on the weight decay term. 

The results of Model 1 over the training and validation sets are summarized in the following table:


![image](https://user-images.githubusercontent.com/73105766/206698063-1886cc10-2203-4405-8fc2-2bd8d240bc88.png)


The results of Model 2 over the training and validation sets are summarized in the following table:


![image](https://user-images.githubusercontent.com/73105766/206698201-76537697-f534-4f68-87c6-d799f7cd4ad8.png)


As we can see in the results, the combination of the medium whisper model with a learning rate scheduler and weight decay outperforms the initial Model 1. In this way, Model 2 has been used in the 3 Gradio applications that have been created.




At the end, most of the weight of the project building and maintaning fell on Huggingface, since that was our primary source for getting the transformers, saving our features and models, as well as running the gradio app.
