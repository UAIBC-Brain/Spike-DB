import torch
from scipy.io import loadmat
from einops import rearrange
from snn_layers import first_order_low_pass_layer, neuron_layer
from torch.utils.data import DataLoader, TensorDataset

class SNN_Model(torch.nn.Module):
    def __init__(self):   
        super().__init__()

        self.batchsize = 8  

        self.axon1 = first_order_low_pass_layer((240,), 25, self.batchsize, 4, True)
        self.snn1 = neuron_layer(240, 240, 25, self.batchsize, 4, True, False)

        self.axon2 = first_order_low_pass_layer((240,), 25, self.batchsize, 4, True)
        self.snn2 = neuron_layer(240, 240, 25, self.batchsize, 4, True, False)

        self.axon3 = first_order_low_pass_layer((240,), 25, self.batchsize, 4, True)
        self.snn3 = neuron_layer(240, 240, 25, self.batchsize, 4, True, False)

        self.dropout1 = torch.nn.Dropout(p=0.1, inplace=False)
        self.dropout2 = torch.nn.Dropout(p=0.1, inplace=False)

        self.linear = torch.nn.Linear(30 * 90, 2)

    def forward(self, inputs):    
        """
        :param inputs: [batch, input_size, t]
        :return:
        """
        inputs = rearrange(inputs, 'b c h -> b h c')  #  [batch, t, input_size]
        batch_size = inputs.size(0)     

        axon1_states = self.axon1.create_init_states()
        snn1_states = self.snn1.create_init_states()

        axon2_states = self.axon2.create_init_states()
        snn2_states = self.snn2.create_init_states()

        axon3_states = self.axon3.create_init_states()
        snn3_states = self.snn3.create_init_states()


        axon1_out, axon1_states = self.axon1(inputs, axon1_states)  
        spike_l1, snn1_states = self.snn1(axon1_out, snn1_states)  
        # print(f'Layer 1 output shape: {spike_l1.dtype}')
        drop_1 = self.dropout1(spike_l1)  


        axon2_out, axon2_states = self.axon2(drop_1, axon2_states)        
        spike_l2, snn2_states = self.snn2(axon2_out, snn2_states)        
        # print(f'Layer 2 output shape: {spike_l2.dtype}')
        drop_2 = self.dropout2(spike_l2)  


        axon3_out, axon3_states = self.axon3(drop_2, axon3_states)  
        spike_l3, snn3_states = self.snn3(axon3_out, snn3_states)  
        # print(f'Layer 3 output shape: {spike_l3.shape}')

        spike = spike_l1.permute(0, 2, 1)


        # spike_l3 = spike_l3.reshape(self.batchsize, -1)  
        # spike_l3 = spike_l3.reshape(batch_size, -1)     
        # print(f'Before reshape, spike_l3 shape: {spike_l3.dtype}')

        # spike_l3 = self.linear(spike_l3)  # 16,2    
        return spike
