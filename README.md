# BostonHacks-2024-Project
This project was designed as a method to show a practical use of our Deep Learning Neural Network that we designed. 



Mushroom Classifier:
We first transform the data such that it is all uniform and that the classifier cannot mess up when reading the data.
The mushroom Classifier uses a Deep Convolutional learning Neural Network to analyze an image, and then classify it. The input layer of the CNN has 3 different parameters, those being Red, Green and Blue. It splits into the 3 colors in order to be able to better classify how the pixels in the image work, and to form patterns between the pixel layout and color as well as the label of the mushroom. As the CNN goes through the layers, it gets more feature extraction, getting more complex data from each new layer (as more filters are added). Through four convolutions, it finally gets through all the inner layers, it finishes the feature extraction, getting as much complexity from the data as possible. Once the feature extraction is complete, the data is sent to the final output layer, and then the classification occurs. The binary classifier determines whether the image is a poisonous or edible mushroom through the calculations and feature patterns that the rest of the layers find. 

The model trains through "epochs" (iterations), where it runs multiple times. The dataset that is used for training and testing is randomized every time, in order to stop the model from just memorizing data, and not being adaptive. This leads to the CCR not being consistent, but making the model more reliable with new data. The model learns off the mistaken classifications it makes every time, and makes corrections to the features it prioritizes when classifying data. 

Issues with the model:

- Small dataset: the dataset we used is ~2000 images, which is small when compared to a Deep CNN which usually uses hundreds of thousands of data points. We did this due to a lack of good data that was available on the internet, and due to a lack of time, since a larger dataset requires more time to train. If we had a larger amount of time, I would've added another layer to further extract complexity (At 4-5, it should max out, as mushroom features are not enitrely too complex), and we'd be able to train more data for longer, leading to the model being more successful.

Overall comments:
The model used in the presentation had a CCR (Classifier Correction Rate) of 70.29%, which, for a small scale Deep CNN of this calibur, is pretty impressive and well made. The model is consistent, and is able to adapt to new data quickly, as well as quick to classify new data. 

The model is useful not just for mushrooms, as it can be trained on any dataset, in order to classify any two things. 


MongoDB:

User database and storage was mainly handled through MongoDB. To ensure that the files were stored and accessible by anyone who used the web application, we needed to make sure that all IP addresses were granted access to avoid any errors when the web application was used. The MongoDB stores an image_id, image_data, filename, and classification due to the machine learning model. The Web Application is what makes the MongoDB database accessible to the users. The Web Application provides a simple and efficient way of uploading images and associating them with the user.

Streamlit:

We built a localhost website on Streamlit to further show the model's use cases, but were unable to develop the website into a website that can be used online, as the model's size was too large to place into a Github Repository normally, and we ran out of time. 
