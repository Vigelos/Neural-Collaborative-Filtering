# Neural-Collaborative-Filtering
A simple implementation of the frames in the eassy Neural Collaborative Filtering(XiangNan He).<br><br>
There are many well-designed models that implenment Neural Collaborative Filtering, but mostly written in pytorch or tensorflow. And if you want to learn about the model or want to use a model that performs really well, it is recommended that you used those well-designed ones.<br><br>
This model is a simple implenmentation of the main frame and many details can be improved, but written totally in matlab and also the preformance is not that bad.<br><br>
**How to use these files↓**
* GMF_PreTraining.m: pre-train function for the GMF part in the NeuMF model
* LogLoss.m: use **LogLoss** to evaluate the preformance of the model
* Loss.m: it calculates the performance of **pure MF**
* MLP_PreTraining.m: pre-train function for the MLP part in the NeuMF model
* ***NCF.mlx: The Main function and entrance of all these files. Run this file to see how the NCF model works***
* NeuMF_training: optimization of **h** of the two pre-trained model after fusion
* ratings.csv: the data set used in this model,from movieLens

