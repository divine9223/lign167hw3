#!/usr/bin/env python
# coding: utf-8

# In[72]:


import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


######################################## BEGIN STARTER CODE ########################################

def relu(x):
	if x<0:
		return 0
	else:
		return x

def loss(y_predicted, y_observed):
	return (y_predicted - y_observed)**2


def mlp(x,W0,W1,W2):
	

	r0_0 = x*W0[0]
	r0_1 = x*W0[1]
	r0_2 = x*W0[2]
	r0 = np.array([r0_0,r0_1,r0_2])

	h0_0 = relu(r0_0)
	h0_1 = relu(r0_1)
	h0_2 = relu(r0_2)
	h0 = np.array([h0_0,h0_1,h0_2])

	

	r1_0 = h0_0*W1[0,0] + h0_1*W1[0,1]+ h0_2*W1[0,2]
	r1_1 = h0_0*W1[1,0] + h0_1*W1[1,1]+ h0_2*W1[1,2]
	r1_2 = h0_0*W1[2,0] + h0_1*W1[2,1]+ h0_2*W1[2,2]
	r1 = np.array([r1_0,r1_1,r1_2])

	h1_0 = relu(r1_0)
	h1_1 = relu(r1_1)
	h1_2 = relu(r1_2)
	h1 = np.array([h1_0,h1_1,h1_2])

	y_predicted = h1_0*W2[0] + h1_1*W2[1]+ h1_2*W2[2]

	variable_dict = {}
	variable_dict['x'] = x
	variable_dict['r0'] = r0
	variable_dict['h0'] = h0
	variable_dict['r1'] = r1
	variable_dict['h1'] = h1
	variable_dict['y_predicted'] = y_predicted

	return variable_dict


# x = 10
# W0 = np.array([1,2,3])
# W1 = np.array([[3,4,5],[-5,4,3],[3,4,1]])
# W2 = np.array([1,3,-3])






######################################## END STARTER CODE ########################################


# NOTICE: DO NOT EDIT FUNCTION SIGNATURES 
# PLEASE FILL IN FREE RESPONSE AND CODE IN THE PROVIDED SPACES



#PROBLEM 1
def d_loss_d_ypredicted(variable_dict,y_observed):
    # Retrieve the network's predicted value y_pred
    y_pred = variable_dict['y_predicted']
    # Calculate the derivative of the loss with respect to y_pred
    derivative = 2 * (y_pred - y_observed)
    return derivative



# #PROBLEM 2
def d_loss_d_W2(variable_dict,y_observed):
    h1 = variable_dict['h1']
    h1 = h1.reshape(1, 3)
    dloss_ypred = d_loss_d_ypredicted(variable_dict, y_observed)
    return np.dot(h1, dloss_ypred)
    


# #PROBLEM 3
def d_loss_d_h1(variable_dict,W2,y_observed):
    dloss_ypred = d_loss_d_ypredicted(variable_dict, y_observed)
    W2 = W2.copy()
    W2 = W2.reshape(1, 3)
    
    return np.dot(W2, dloss_ypred)


# #PROBLEM 4
def relu_derivative(x):
    if x > 0:
        return 1
    else:
        return 0


# #PROBLEM 5
def d_loss_d_r1(variable_dict,W2,y_observed):
    d_loss = d_loss_d_h1(variable_dict, W2, y_observed)
    r1 = variable_dict['r1']
    d_r1 = []
    for i in range(len(r1)):
        d_r1.append(relu_derivative(r1[i]))
    
        
    return d_loss * d_r1


#PROBLEM 6
def d_loss_d_W1(variable_dict,W2,y_observed):
    d_loss = d_loss_d_r1(variable_dict, W2, y_observed)
    h0 = variable_dict['h0']
    d_W1 = np.zeros((3, 3))
    for j in range(len(d_W1)):
        d_W1[j, :] = d_loss[0][j] * h0
    return d_W1

    
    
    


#PROBLEM 7
def d_loss_d_h0(variable_dict,W1,W2,y_observed):
    l = []
    for i in range(len(W1)):
        d_loss = d_loss_d_r1(variable_dict, W2, y_observed)
        d_loss = np.dot(d_loss, W1)
    return d_loss
        
    



#PROBLEM 8
def d_loss_d_r0(variable_dict,W1,W2,y_observed):
    # Retrieve the derivative of the loss with respect to h0 from Problem 7
    d_loss = d_loss_d_h0(variable_dict, W1, W2, y_observed)
    
    # Retrieve the network's value for the layer r0
    r0 = variable_dict['r0']
    
    # Calculate the derivative of ReLU for each element of r0
    relu_derivs = np.array([relu_derivative(value) for value in r0])
    
    # Compute the partial derivatives of the loss with respect to r0
    d_loss = d_loss * relu_derivs
    
    return d_loss
        
        


#PROBLEM 9
def d_loss_d_W0(variable_dict,W1,W2,y_observed):
    # Retrieve the derivative of the loss with respect to r0
    d_loss = d_loss_d_r0(variable_dict, W1, W2, y_observed)
    
    # Retrieve the input value x from variable_dict
    x = variable_dict['x']
    
    # The derivative of r0 with respect to W0 is simply the input x
    # Thus, the gradient of the loss with respect to W0 is the product of d_loss_d_r0 and x
    d_loss= d_loss* x
    
    return d_loss


#PROBLEM 10
class TorchMLP(nn.Module):
    def __init__(self, custom_weights=None):
        super(TorchMLP, self).__init__()
        
        # Define the layers
        self.fc1 = nn.Linear(1, 3)
        self.fc2 = nn.Linear(3, 3)
        self.fc3 = nn.Linear(3, 1)
        
        # If custom weights are provided, initialize the layers with these weights
        if custom_weights:
            with torch.no_grad():
                self.fc1.weight = nn.Parameter(custom_weights['fc1_weight'])
                self.fc2.weight = nn.Parameter(custom_weights['fc2_weight'])
                self.fc3.weight = nn.Parameter(custom_weights['fc3_weight'])
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x



    


# #PROBLEM 11
def torch_loss(y_predicted, y_observed):
    return (y_predicted - y_observed)**2


# #PROBLEM 12
def torch_compute_gradient(x, y_observed, model):
    # Set the model to training mode
    model.train()
    
    # Make sure the gradients are zeroed before the forward pass
    model.zero_grad()
    
    # Perform the forward pass
    y_pred = model(x)
    
    # Calculate the loss 
    loss = torch_loss(y_pred, y_observed)
    
    # Compute the gradient of the loss with respect to the model parameters
    loss.backward()
    
    # Return the model with updated gradients
    return model





# In[5]:


# variable_dict = mlp(x,W0,W1,W2)


# In[75]:


x = 10
W0 = np.array([1,2,3])
W1 = np.array([[3,4,5],[-5,4,3],[3,4,1]])
W2 = np.array([1,3,-3])
y_observed = 9


# In[76]:


d_loss_d_W0(variable_dict,W1,W2,y_observed)


# In[ ]:


# custom_weights = {
#     'fc1_weight': torch.tensor([[1], [2], [3]], dtype=torch.float),
#     'fc2_weight': torch.tensor([[3,4,5],[-5,4,3],[3,4,1]], dtype=torch.float),  # Example of random weights for the second layer
#     'fc3_weight': torch.tensor([[1,3,-3]], dtype=torch.float),
# }

# # Instantiate the model with the custom weights
# model = TorchMLP(custom_weights=custom_weights)

# # Print the model's state_dict
# print("Model's state_dict:")
# for param_tensor in model.state_dict():
#     print(param_tensor, "\t", model.state_dict()[param_tensor])

# # Test the model with a dummy input
# dummy_input = torch.tensor([[10.0]])  # Single feature input
# output = model(dummy_input)
# print("Output:", output)


# In[74]:


d_loss_d_W1(variable_dict,W2,y_observed)


# In[ ]:




