# SEAF: Secure Evaluation on Activation Functions with Dynamic Precision for Secure Two-Party Inference

## Setup

For setup instructions, please refer to the README file in the `SCI` folder.

We successfully completed the compilation on Ubuntu 22.04.5 LTS with Intel(R) Xeon(R) Platinum 8475B.


## Code Structure

The code for this project is organized as follows:

- **/SEAF/SCI/tests**:  
  Contains all the code for SEAF, including specific implementations of activation functions and models.  

- **/SEAF/SCI/tests/activation**:  
  Includes the SEAF implementation of the GELU activation function. The main file for this is:  
  **`activation/bolt_gelu_new.cpp`**

- **/SEAF/SCI/tests/bert_bolt**:  
  Contains the implementation of the BERT model inference using Bolt + SEAF. The main file for this is:  
  **`bert_bolt_SEAF.cpp`**

- **/SEAF/SCI/tests/bert_iron**:  
  Also contains implementations of both the GELU activation function and BERT model inference using SEAF and Iron.  

- **New Activation Functions**:  
  The **/SEAF/SCI/tests** directory now includes four new activation functions, which are:  
  - ELU  
  - GELU  
  - Sigmoid  
  - Tanh  

##Running Tests

Run the unit tests in `SEAF/` folder of Generalized Geometric MPC Protocols as follows:
'''
./SCI/build/bin/BOLT_BERT_SEAF r=1 & ./SCI/build/bin/BOLT_BERT_SEAF r=2
'''
