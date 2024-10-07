

from pytorch_lightning import LightningModule
import torch.nn as nn
import torch
from functools import partial
from typing import Optional
from warnings import warn

class myLightningModule(LightningModule):
    '''
    This training code follows the standard structure of Pytorch - lighthning. It's worth looking at their docs for a more in depth dive as to why it is this was
    '''
    
    def __init__(self,
                learning_rate,
                **kwargs,
                ):

        super().__init__()
        self.save_hyperparameters()
        self.loss=torch.nn.CrossEntropyLoss()
        self.GroundTruthWheel1=torch.randperm(26)
        self.GroundTruthWheel2=torch.randperm(26)
        self.GroundTruthWheel3=torch.randperm(26)
        #encode each as 1hot vector
        self.GroundTruthReflector=torch.randperm(26)
        self.GroundTruthReflector=torch.nn.functional.one_hot(self.GroundTruthReflector,num_classes=26).float()
        self.GroundTruthWheel1=torch.nn.functional.one_hot(self.GroundTruthWheel1,num_classes=26).float()
        self.GroundTruthWheel2=torch.nn.functional.one_hot(self.GroundTruthWheel2,num_classes=26).float()
        self.GroundTruthWheel3=torch.nn.functional.one_hot(self.GroundTruthWheel3,num_classes=26).float()


        self.solutionWheel1=nn.Parameter(26,26) #an assignment matrix representing each wheel
        self.solutionWheel2=nn.Parameter(26,26)
        self.solutionWheel3=nn.Parameter(26,26)



    def encode_message(self,message):
        #message is a batch of B,Seq,26 and is 1 hot encoded.
        #after each character in the sequence, wheel 1 will spin. 
        message =message.permute(1,0,2)
        for i in range(message.size(0)):
            #send first char (shape B,26) through wheel 1, wheel 2 , wheel 3 then back.
            wheel1Offset=i%26
            wheel2Offset=i//3 %26
            wheel3Offset=i//9 %26
            wheel1OffsetAssignment=torch.zeros(26,26)
            wheel2OffsetAssignment=torch.zeros(26,26)
            wheel3OffsetAssignment=torch.zeros(26,26)
            wheel1OffsetAssignment=wheel1OffsetAssignment.diagonal(offset=wheel1Offset).fill_(1)
            wheel2OffsetAssignment=wheel2OffsetAssignment.diagonal(offset=wheel2Offset).fill_(1)
            wheel3OffsetAssignment=wheel3OffsetAssignment.diagonal(offset=wheel3Offset).fill_(1)
            wheel1OffsetAssignment=wheel1OffsetAssignment.diagonal(offset=-26+wheel1Offset).fill_(1)
            wheel2OffsetAssignment=wheel2OffsetAssignment.diagonal(offset=-26+wheel2Offset).fill_(1)
            wheel3OffsetAssignment=wheel3OffsetAssignment.diagonal(offset=-26+wheel3Offset).fill_(1)
            GROUNDTRUTHWHEEL1=self.GroundTruthWheel1@wheel1OffsetAssignment
            GROUNDTRUTHWHEEL2=self.GroundTruthWheel2@wheel2OffsetAssignment
            GROUNDTRUTHWHEEL3=self.GroundTruthWheel3@wheel3OffsetAssignment
            message[i]=message[i]@GROUNDTRUTHWHEEL1
            message[i]=message[i]@GROUNDTRUTHWHEEL2
            message[i]=message[i]@GROUNDTRUTHWHEEL3
            message[i]=message[i]@self.reflector
            message[i]=message[i]@GROUNDTRUTHWHEEL3.T
            message[i]=message[i]@GROUNDTRUTHWHEEL2.T
            message[i]=message[i]@GROUNDTRUTHWHEEL1.T
            #spin wheels





    def forward(self,input):
        #This inference steps of a foward pass of the model 
        message =message.permute(1,0,2)
        for i in range(message.size(0)):
            wheel1Offset=i%26
            wheel2Offset=i//3 %26
            wheel3Offset=i//9 %26
            wheel1OffsetAssignment=torch.zeros(26,26)
            wheel2OffsetAssignment=torch.zeros(26,26)
            wheel3OffsetAssignment=torch.zeros(26,26)
            wheel1OffsetAssignment=wheel1OffsetAssignment.diagonal(offset=wheel1Offset).fill_(1)
            wheel2OffsetAssignment=wheel2OffsetAssignment.diagonal(offset=wheel2Offset).fill_(1)
            wheel3OffsetAssignment=wheel3OffsetAssignment.diagonal(offset=wheel3Offset).fill_(1)
            wheel1OffsetAssignment=wheel1OffsetAssignment.diagonal(offset=-26+wheel1Offset).fill_(1)
            wheel2OffsetAssignment=wheel2OffsetAssignment.diagonal(offset=-26+wheel2Offset).fill_(1)
            wheel3OffsetAssignment=wheel3OffsetAssignment.diagonal(offset=-26+wheel3Offset).fill_(1)
            WHEEL1=self.solutionWheel1@wheel1OffsetAssignment
            WHEEL2=self.solutionWheel2@wheel2OffsetAssignment
            WHEEL3=self.solutionWheel3@wheel3OffsetAssignment
            message[i]=message[i]@WHEEL1
            message[i]=message[i]@WHEEL2
            message[i]=message[i]@WHEEL3
            message[i]=message[i]@self.reflector
            message[i]=message[i]@WHEEL3.T
            message[i]=message[i]@WHEEL2.T
            message[i]=message[i]@WHEEL1.T


    def training_step(self, batch, batch_idx,optimizer_idx=0):
        #The batch is collated for you, so just seperate it here and calculate loss. 
        #By default, PTL handles optimization and scheduling and logging steps. so All you have to focus on is functionality. Here's an example...
        input=batch[0]
        GroundTruth=self.encode_message(input)
        prediction=self.forward(input)
        #normalize prediction to sum to one 
        prediction=torch.nn.functional.softmax(prediction,dim=-1)
        loss=self.loss(prediction,GroundTruth)
        
        #Logging is done through this module as follows.
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss
      
      
      
        #You could log here the val_loss, or just print something. 
        
    def configure_optimizers(self):
        #Automatically called by PL. So don't worry about calling it yourself. 
        #you'll notice that everything from the init function is stored under the self.hparams object 
        optimizerA = torch.optim.sgd(
            self.parameters(), lr=self.hparams.learning_rate)
        
        #Define scheduler here too if needed. 
        return [optimizerA]
