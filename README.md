# SEAF: Secure Evaluation on Activation Functions with Dynamic Precision for Secure Two-Party Inference

**Reference Repository:**  
**Project webpage:** <[https://aka.ms/ezpc](https://github.com/Clive2312/EzPC/tree/bert/SCI)>

## Setup

For setup instructions, please refer to the README file located in the `SCI` folder.

We successfully completed the compilation on Ubuntu 22.04.5 LTS with Intel(R) Xeon(R) Platinum 8475B.


## Code Structure

The project is organized as follows:

- **/SEAF/SCI/tests**  
  Contains all SEAF-related code, including implementations of activation functions and models.

- **/SEAF/SCI/tests/bert_bolt**  
  Contains the implementation of BERT model inference using Bolt + SEAF. 


- **/SEAF/SCI/tests/bert_iron**  
  Contains the implementation of BERT model inference using Iron + SEAF. 

- **Our improved Activation Functions**  
  The **/SEAF/SCI/tests** directory now includes four SEAF version activation functions:  
  - ELU  
  - GELU  
  - Sigmoid  
  - Tanh  

## Running Tests

To run the unit tests in the `SEAF/` folder of Generalized Geometric MPC Protocols, use the following command:

```bash
./SCI/build/bin/BOLT_BERT_SEAF r=1 & ./SCI/build/bin/BOLT_BERT_SEAF r=2
```
