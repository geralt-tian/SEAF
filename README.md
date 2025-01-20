```markdown
# SEAF: Secure Evaluation on Activation Functions with Dynamic Precision for Secure Two-Party Inference

## Setup

For setup instructions, please refer to each component's README file.

Alternatively, you can use the **setup_env_and_build.sh** script. This script installs all necessary dependencies, builds each component, and creates a virtual environment named *mpc_venv* with all the required packages. 

- To perform a setup with default paths and settings, run:  
  ```bash
  ./setup_env_and_build.sh quick
  ```
- If you prefer to manually choose paths, run the script without the `quick` argument:  
  ```bash
  ./setup_env_and_build.sh
  ```

After setup, activate the virtual environment by running:  
```bash
source mpc_venv/bin/activate
```

## Code Structure

The code for this project is organized as follows:

- **/EzPC/SCI/tests**:  
  Contains all the code for SEAF, including specific implementations of activation functions and models.  

- **/EzPC/SCI/tests/activation**:  
  Includes the SEAF implementation of the GELU activation function. The main file for this is:  
  **`activation/bolt_gelu_new.cpp`**

- **/EzPC/SCI/tests/bert_bolt**:  
  Contains the implementation of the BERT model inference using Bolt + SEAF. The main file for this is:  
  **`bert_bolt_new.cpp`**

- **/EzPC/SCI/tests/bert_iron**:  
  Also contains implementations of both the GELU activation function and BERT model inference using SEAF and Iron.  

- **New Activation Functions**:  
  The **/EzPC/SCI/tests** directory now includes four new activation functions, which are:  
  - ELU  
  - GELU  
  - Sigmoid  
  - Tanh  
```