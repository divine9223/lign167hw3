import numpy as np
import torch
from torch import nn

#GPT Link: https://chat.openai.com/share/dbd4e556-61c3-4390-9534-f5ab471f4362

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
	# Retrieve the network’s predicted value ypred
    y_predicted = variable_dict['y_predicted']
    
    # Calculate the partial derivative of the loss with respect to y_predicted
    derivative = 2 * (y_predicted - y_observed)
    
    return derivative


#PROBLEM 2
def d_loss_d_W2(variable_dict,y_observed):
	# Call the previously defined function to get the derivative of the loss with respect to y_predicted
    d_loss_d_ypred = d_loss_d_ypredicted(variable_dict, y_observed)
    
    # Retrieve the network’s value for the layer h1
    h1 = variable_dict['h1']
    
    # Calculate the partial derivatives of the loss with respect to W2
    # We multiply the derivative of the loss with respect to y_predicted by each element of h1
    # to get the derivative with respect to each weight in W2
    gradient_W2 = d_loss_d_ypred * h1
    
    return gradient_W2


#PROBLEM 3
def d_loss_d_h1(variable_dict,W2,y_observed):
	# Calculate the derivative of the loss with respect to y_predicted
    d_loss_d_ypred = d_loss_d_ypredicted(variable_dict, y_observed)
    
    # Calculate the partial derivatives of the loss with respect to h1
    # This is done by multiplying the derivative of the loss with respect to y_predicted
    # by the corresponding weight from W2
    d_loss_d_h1_0 = d_loss_d_ypred * W2[0]
    d_loss_d_h1_1 = d_loss_d_ypred * W2[1]
    d_loss_d_h1_2 = d_loss_d_ypred * W2[2]
    
    # Combine these partial derivatives into a 1x3 NumPy array
    d_loss_d_h1 = np.array([d_loss_d_h1_0, d_loss_d_h1_1, d_loss_d_h1_2])
    
    return d_loss_d_h1


#PROBLEM 4
def relu_derivative(x):
    if x > 0:
        return 1
    else:
        return 0


#PROBLEM 5
def d_loss_d_r1(variable_dict,W2,y_observed):
	# Calculate the derivative of the loss with respect to h1
    d_loss_d_h1 = d_loss_d_h1(variable_dict, W2, y_observed)
    
    # Retrieve the network’s value for the layer r1
    r1 = variable_dict['r1']
    
    # Calculate the derivative of ReLU for each element of r1
    relu_derivatives = np.array([relu_derivative(r1_j) for r1_j in r1])
    
    # Calculate the partial derivatives of the loss with respect to r1
    # This is done by multiplying the derivative of the loss with respect to h1 by the derivative of ReLU at r1
    d_loss_d_r1 = d_loss_d_h1 * relu_derivatives
    
    return d_loss_d_r1


#PROBLEM 6
def d_loss_d_W1(variable_dict,W2,y_observed):
	# Calculate the derivative of the loss with respect to r1 (from Problem 5)
    d_loss_d_r1 = d_loss_d_r1(variable_dict, W2, y_observed)
    
    # Retrieve the network’s value for the layer h0
    h0 = variable_dict['h0']
    
    # Initialize the gradient matrix for W1
    gradient_W1 = np.zeros((3, 3))
    
    # Calculate the outer product of the derivative of the loss with respect to r1 and h0
    # for each row in W1
    for j in range(3):
        gradient_W1[j, :] = d_loss_d_r1[j] * h0
    
    return gradient_W1
#error in Problem 6 doesn't output code index error in line 160, needed the [0] before[j]


#PROBLEM 7
def d_loss_d_h0(variable_dict,W1,W2,y_observed):
    # Retrieve the derivative of the loss with respect to r1
    d_loss_d_r1 = d_loss_d_r1(variable_dict, W2, y_observed)
    
    # Calculate the partial derivatives of the loss with respect to h0
    # This requires summing over the contributions from each element of r1
    d_loss_d_h0 = np.dot(W1.T, d_loss_d_r1)
    
    return d_loss_d_h0
#error in Problem 7 as it used W1.T which is transpose



#PROBLEM 8
def d_loss_d_r0(variable_dict,W1,W2,y_observed):
    # Retrieve the derivative of the loss with respect to h0 from Problem 7
    d_loss_d_h0 = d_loss_d_h0(variable_dict, W1, W2, y_observed)
    
    # Retrieve the network's value for the layer r0
    r0 = variable_dict['r0']
    
    # Calculate the derivative of ReLU for each element of r0
    relu_derivs = np.array([relu_derivative(value) for value in r0])
    
    # Compute the partial derivatives of the loss with respect to r0
    d_loss_d_r0 = d_loss_d_h0 * relu_derivs
    
    return d_loss_d_r0


#PROBLEM 9
def d_loss_d_W0(variable_dict,W1,W2,y_observed):
    # Retrieve the derivative of the loss with respect to r0
    d_loss_d_r0 = d_loss_d_r0(variable_dict, W1, W2, y_observed)
    
    # Retrieve the input value x from variable_dict
    x = variable_dict['x']
    
    # The derivative of r0 with respect to W0 is simply the input x
    # Thus, the gradient of the loss with respect to W0 is the product of d_loss_d_r0 and x
    d_loss_d_W0 = d_loss_d_r0 * x
    
    return d_loss_d_W0


#PROBLEM 10
class TorchMLP(nn.Module):
    def __init__(self):
        super(TorchMLP, self).__init__()
        # Define the first layer with 1 input and 3 outputs
        self.layer1 = nn.Linear(1, 3)
        # Define the second layer with 3 inputs and 3 outputs
        self.layer2 = nn.Linear(3, 3)
        # Define the output layer with 3 inputs and 1 output
        self.output = nn.Linear(3, 1)


    def forward(self, x):
        # Apply the first layer and ReLU activation
        x = self.layer1(x)
        x = nn.functional.relu(x)
        # Apply the second layer and ReLU activation
        x = self.layer2(x)
        x = nn.functional.relu(x)
        # Apply the output layer
        x = self.output(x)
        return x


#PROBLEM 11
def torch_loss(y_predicted, y_observed):
    # Compute the Mean Squared Error loss
    loss = (y_predicted - y_observed)**2
    return loss


#PROBLEM 12
def torch_compute_gradient(x,y_observed,model):
    # Ensure model is in train mode, which is relevant if there are layers like dropout or batchnorm
    model.train()
    
    # Zero the gradients of all model parameters
    model.zero_grad()
    
    # Perform the forward pass
    y_predicted = model(x)
    
    # Compute the loss using the torch_loss function from Problem 11
    loss = torch_loss(y_predicted, y_observed)
    
    # Perform the backward pass to compute gradients
    loss.backward()
    
    # The gradients are now stored in the parameters of the model,
    # so we simply return the model
    return model


