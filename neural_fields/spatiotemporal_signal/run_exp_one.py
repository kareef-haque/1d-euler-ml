from train import train_model
from neural_fields.spatiotemporal_signal.models import INR, SIREN, WIRE



if __name__ == '__main__':
    train_model(INR, model_name = 'neural_field', config  =r'.\neural_fields\spatiotemporal_signal\configs')
    train_model(SIREN, model_name='SIREN',config  =r'.\neural_fields\spatiotemporal_signal\configs')
    train_model(WIRE, model_name='WIRE', 
                config  =r'.\neural_fields\spatiotemporal_signal\configs')