The following four jupyter files,show examples of federated learning on a classification problem and a prediction problem applied on a dataset that is a record of appliance energy consumption in one year from 50 households. The consumption is recorded every 15 minutes. Each household has the following 10 appliance
records: air conditioner (AC), dish washer, washing machine, dryer, water heater, TV,
microwave, kettle, lighting, and refrigerator.  

We use an RNN and a LSTM as our two models of choice and compare how they do compared to each other bought on the training and test data.  

Our goal:  
(i) train a model that can predict the energy consumption of appliances in a household, and  
(ii) train another model that can classify the type of electricity appliances according to their energy consumption recordings.  

The models are built using keras and aggregated with the tensorflow_federated package.  


Important note: For files classification_RNN.ipynb and classification_lstm.ipynb, we simulate our training and test accuracy with confusion matrices and the labels of the matrices represents.  

(1) Class1=AC  
(2) Class2=Dish washer  
(3) Class3=Washing Machine	  
(4) Class4=Dryer  
(5) Class5=Water heater  
(6) Class6=TV	 
(7) Class7=Microwave  
(8) Class8=Kettle  
(9) Class9=Lighting  
(10) Class10=Refrigerator  
