# MSK_ML_gamma
#This file contains the code corresponding to the article "Generalisation and Robustness of Supervised Machine Learning Algorithms for Biomechanical Modelling."

#A HPC cluster was used to run the complete cross-validation runs which are not feasible to run on a normap laptop. However, few instances, can be easily done using the function "specific" and "specific_CV" in pytorch.py, respectively.

#Lastly, all the plots and statistics provided in the paper can be reproduced by simply running main.py. Note that the path need to be modified in main.py. 

#Test data for reproducing the results are provided here. The full training dataset can be obtained on request to vikranth.harthikotenagaraja@eng.ox.ac.uk

#The codes are developed using Python 3 with keras for ML.
#MSK_ML_beta #This file contains the code description used in the article: "Machine Learning for Optical Motion Capture-driven
Musculoskeletal Modeling from Inertial Motion Capture Data" #The codes are developed in python3 using standard python modules and keras for ML.

#The code here contains: 
#a) Pipeline to run ML methods such FFNN, RNN, and LM which can be easily extended for other methods. 
#b) We have used this pipeline to run cross-validation on cluster.  
#c) Final training and testing can be easily done on any laptop/computer. 
#d) Code for analysis and plotting is also provided.

#Code description
#pytorch_utilities.py -- contains function for various models (Linear, Neural Network, RNN, LSTM, GRU,....) and generate a file with hyperparameters choices 
#read_in_out.py -- contains classes to read input and output data and some other classes for handling the final trained models
#pytorch.py -- several functions to handle data, perform cross-validation, train model, forward pass, plot and analyse results, plot outputs
#find_best_model.py -- code for finding the best-fit model using the average validation accuracy. 
#main.py -- contains function to estimate plot results and compute statistics


#final trained NN model #Files are heavy to upload, can be downloaded from here: https://www.dropbox.com/sh/svuqdy597d6pg60/AACiWr-kVx_W0bU0AD3HWPNha?dl=0
#Test data for reproducing the results are provided here, 
Input data --  https://www.dropbox.com/sh/8isp6yl29np6ngo/AAAqRWIc8lTOtYieehUSEQN1a?dl=0
Output data -- https://www.dropbox.com/sh/7h1oncpyru9vupl/AACm0YPmlmdu2rlabUpynCW4a?dl=0
The full training can be obtained on request to vikranth.harthikotenagaraja@eng.ox.ac.uk

