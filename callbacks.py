from pytorch_lightning.callbacks import Callback

class MyPrintingCallback(Callback):
    def __init__(self):
        super().__init__()
        
    def on_train_start(self, trainer, pl_module):
        print('Starting to train!')
        
    def on_train_end(self, trainer, pl_module):
        print('Finished training!')  
