# Fine Tuning of a Whisper Model for Galician

On this second lab of Scalable Machine Learning and Deep learning I fine tuned a Whisper model using the blog post on https://huggingface.co/blog/fine-tune-whisper for Galician, a language spoken on the north of Spain. I started by following the steps as prepared on the blog, and later made some improvements on feature extraction, training and UI. 


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


First of all I split the single Colab file into a pipeline:

**Feature Extraction** - Where the dataset and its processors are downloaded and the features are extracted.

**Training** - Where the features extracted and the whisper model are downloaded, then this model is trained. There are 2 versions (Model 1 and Model 2). Model 2 outperforms Model 1 (their parameters and metric results will be described below).

**Inference** - The gradio app UI that is run on the Hugging Face. The user can put the audio/youtube link/register on the microphone and the app will do the transcribtion, using the previosly traind model. Three version of this were built.


**Model based Improvements**


After training the initial first model, I wanted to improve the performance, even though I could not change the architecture of the model. One thing I immediatly noticed is that since we were training the model on 1000 steps, on 750 steps it already had a better performance than later. One very probable explanation of this is that the model was overfitting. To solve this we wether needed to use less epochs or some reularizatoin technique. After some fine tuning and testing the best model we managed to achieve uses the following settings. Below we can see a summary of the two models tested during this Lab and their respective training parameters:

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



I trained Model 1 over the training set for 1000 steps, equivalent to 2.75 epochs, and evaluated its performance on the validation set every 250 steps. The model that showed the best results according to the Word Error Rate (WER) metric on the validation set was selected to be uploaded to Hugging Face, considered as optimal. In this case, it was the model after 750 steps.

Model 2 is significantly more complex than Model 1, with a higher number of parameters. This increase in complexity also led to longer training times, so I reduced the number of steps to 500 due to resource limitations. I performed the validation after every 250 steps, thus, in total, I validated Model 2 twice. I not only changed the network architecture but also some training parameters, such as the learning rate scheduler and weight decay. The scheduler adjusts the learning rate between iterations as training progresses. Another parameter I added was weight decay, a regularization technique used in deep learning to reduce the model's complexity and prevent overfitting. This technique is implemented by modifying the update rule for the parameters, so it is based not only on the training data but also on the weight decay term.
The results of Model 1 over the training and validation sets are summarized in the following table:


![image](https://user-images.githubusercontent.com/73105766/206698063-1886cc10-2203-4405-8fc2-2bd8d240bc88.png)


The results of Model 2 over the training and validation sets are summarized in the following table:


![image](https://user-images.githubusercontent.com/73105766/206698201-76537697-f534-4f68-87c6-d799f7cd4ad8.png)


As we can see in the results, the combination of the medium whisper model with a learning rate scheduler and weight decay outperforms the initial Model 1. In this way, Model 2 has been used in the 3 Gradio applications that have been created.




At the end, most of the weight of the project building and maintaning fell on Huggingface, since that was our primary source for getting the transformers, saving our features and models, as well as running the gradio app.
