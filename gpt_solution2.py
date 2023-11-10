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


#Same as solution 1
#PROBLEM 1
def d_loss_d_ypredicted(variable_dict,y_observed):
	# Retrieve the network’s predicted value ypred
    y_predicted = variable_dict['y_predicted']
    
    # Calculate the partial derivative of the loss with respect to y_predicted
    derivative = 2 * (y_predicted - y_observed)
    
    return derivative


#PROBLEM 2
def d_loss_d_W2(variable_dict,y_observed):
	# Retrieve the network’s value for the layer h1
    h1 = variable_dict['h1']
    
    # Call the previously defined function to get the derivative of the loss with respect to y_predicted
    d_loss_d_ypred = d_loss_d_ypredicted(variable_dict, y_observed)
    
    # Calculate the gradient of the loss with respect to each weight in W2
    # Each element of the gradient is the product of the derivative of the loss with respect to y_predicted
    # and the corresponding value from h1
    gradient_w2_0 = d_loss_d_ypred * h1[0]
    gradient_w2_1 = d_loss_d_ypred * h1[1]
    gradient_w2_2 = d_loss_d_ypred * h1[2]
    
    # Combine the gradients into a 1x3 NumPy array
    gradient_W2 = np.array([gradient_w2_0, gradient_w2_1, gradient_w2_2])
    
    return gradient_W2


#PROBLEM 3
def d_loss_d_h1(variable_dict,W2,y_observed):
	# Calculate the derivative of the loss with respect to y_predicted
    d_loss_d_ypred = d_loss_d_ypredicted(variable_dict, y_observed)
    
    # Calculate the partial derivatives of the loss with respect to each element of h1
    # This is the gradient of the loss with respect to y_predicted, scaled by each weight in W2
    d_loss_d_h1 = d_loss_d_ypred * W2
    
    return d_loss_d_h1


#same as solution 1
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
    # This is where the mistake was - we need to apply the derivative of ReLU to each element of r1
    d_relu_d_r1 = np.array([relu_derivative(r1_j) for r1_j in r1])
    
    # Calculate the partial derivatives of the loss with respect to r1
    d_loss_d_r1 = d_loss_d_h1 * d_relu_d_r1
    
    return d_loss_d_r1


#PROBLEM 6
def d_loss_d_W1(variable_dict,W2,y_observed):
	# Retrieve the derivative of the loss with respect to r1
    d_loss_d_r1 = d_loss_d_r1(variable_dict, W2, y_observed)
    
    # Retrieve the network’s value for the layer h0
    h0 = variable_dict['h0']
    
    # Compute the outer product of the derivative of the loss with respect to r1 and h0
    # This will give us the gradient of the loss with respect to each element of W1
    gradient_W1 = np.outer(d_loss_d_r1, h0)
    
    return gradient_W1
# the overshadowing variable name in line 149 can cause errors in the code

#PROBLEM 7
def d_loss_d_h0(variable_dict,W1,W2,y_observed):
	# Retrieve the derivative of the loss with respect to r1
    d_loss_d_r1 = d_loss_d_r1(variable_dict, W2, y_observed)
    
    # Retrieve the network's values for the layers r0 and h0
    r0 = variable_dict['r0']
    h0 = variable_dict['h0']
    
    # Calculate the derivative of ReLU for each element of r0
    relu_derivs = np.array([relu_derivative(value) for value in r0])
    
    # Compute the gradients with respect to h0
    # This is a sum over the contributions from the weights in W1 and the derivatives of r1
    # Adjusted by the derivative of the ReLU (which acts element-wise on r0)
    d_loss_d_h0 = np.dot(W1.T, d_loss_d_r1) * relu_derivs
    
    return d_loss_d_h0
#same as problem 6 in line 164, overshadowing variable name


#PROBLEM 8
def d_loss_d_r0(variable_dict,W1,W2,y_observed):
	# Retrieve the derivative of the loss with respect to r1
    d_loss_d_r1 = d_loss_d_r1(variable_dict, W2, y_observed)
    
    # Compute the gradient of r1 with respect to h0, which is just W1
    # Then compute the gradient of h0 with respect to r0, which is the derivative of ReLU
    # We then multiply these two gradients to get the gradient of the loss with respect to r0
    
    # Retrieve the network's value for the layer r0
    r0 = variable_dict['r0']
    
    # Calculate the derivative of ReLU for each element of r0
    relu_derivs = np.array([relu_derivative(r) for r in r0])
    
    # Compute the gradient of the loss with respect to r0 by backpropagating through W1
    d_loss_d_r0 = np.dot(d_loss_d_r1, W1) * relu_derivs
    
    return d_loss_d_r0
#needs to use the transpose in order to properly use the np.dot function

#same as solution 1
#PROBLEM 9
def d_loss_d_W0(variable_dict,W1,W2,y_observed):
	# Retrieve the derivative of the loss with respect to r0
    d_loss_d_r0 = d_loss_d_r0(variable_dict, W1, W2, y_observed)
    
    # Retrieve the input value x from variable_dict
    x = variable_dict['x']
    
    # Compute the gradient of the loss with respect to W0
    # Since the derivative of r0 with respect to W0 is the input x for each element,
    # we multiply the input x with the gradient of the loss with respect to r0 element-wise
    d_loss_d_W0 = d_loss_d_r0 * x
    
    return d_loss_d_W0


#PROBLEM 10
class TorchMLP(nn.Module):
    def __init__(self):
        super(TorchMLP, self).__init__()
        # Define the first layer with 1 input feature and 3 output features
        self.layer1 = nn.Linear(1, 3)
        # Define the second layer with 3 input features and 3 output features
        self.layer2 = nn.Linear(3, 3)
        # Define the output layer with 3 input features and 1 output feature
        self.output = nn.Linear(3, 1)

    def forward(self, x):
        # Ensure input x is a tensor with shape (n, 1) where n is the batch size
        x = x.view(-1, 1)  # Reshape x if necessary
        # Apply the first layer and ReLU activation
        x = F.relu(self.layer1(x))
        # Apply the second layer and ReLU activation
        x = F.relu(self.layer2(x))
        # Apply the output layer
        x = self.output(x)
        return x


#PROBLEM 11
def torch_loss(y_predicted, y_observed):
    # Define the MSE loss function
    loss_func = nn.MSELoss()
    # Compute the loss
    loss = loss_func(y_predicted, y_observed)
    return loss


#PROBLEM 12
def torch_compute_gradient(x,y_observed,model):
    # Ensure model is in train mode
    model.train()
    
    # Check if x needs to be reshaped to match the expected input shape (batch_size, num_features)
    if x.ndim == 1:
        x = x.view(1, -1)

    # Zero the gradients of all model parameters
    model.zero_grad()
    
    # Perform the forward pass
    y_predicted = model(x)
    
    # Compute the loss
    loss_function = nn.MSELoss()
    loss = loss_function(y_predicted, y_observed)
    
    # Perform the backward pass to compute gradients
    loss.backward()
    
    # The model parameters are now updated with the gradients
    return model


