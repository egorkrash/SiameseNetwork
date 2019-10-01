# SiameseNetwork
This model is designed to help just released apps to deal with a cold start. It uses app description to predict a set of queries which may be used when searching it. 
# Usage
## Set up for Linux/macOS machine
1. Clone repository:
	'''console
	$ git clone https://github.com/egorkrash/SiameseNetwork.git
	'''
2. Install requirements via pip:
        '''console
	$  pip install -r requirements.txt
	'''
	
Now you're ready to use!
Run the following command to get a detailed description for each parameter:<br/>
$ python run_model.py --help

## Make predictions
Put some text description into testdesc.txt and run this command:<br/>
$ python make_predictions.py

Predictions will be saved in testpreds.txt
Notice! In this case model loads from './weights/params_wval_9.pt'

Equivalently you can run model with your custom checkpoint and a new bank of possible queries as follows:<br/>
$ python run_model.py --make-predictions --load-model --checkpoint-path 'path to saved weights'

Thus, you can change the bank_queries.txt and the model will take into account new queries when making predictions
