Demo

A running demo of our model can be found here. 

Please download pretrained weights beforehand. 

Requirements

To install all requirements, run pip install -r requirements.txt

Data Format 

The code assumes the input text is in the following format:

Generation

Para-M: python generate.py --model_path /path/to/modeldir/ --model_type 'nomem' --beam 10 --source 
Para-M (mem): python generate.py --model_path /path/to/modeldir/ --model_type 'mem' --beam 10 --source 

Citing 

Link: https://arxiv.org/abs/2010.01486
