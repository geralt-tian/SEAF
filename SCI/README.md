# Secure and Correct Inference (SCI) Library

## Introduction
This directory contains the code for the Secure and Correct Inference (SCI) library from ["CrypTFlow2: Practical 2-Party Secure Inference"](https://eprint.iacr.org/2020/1002), ["SIRNN: A Math Library for Secure RNN Inference"](https://eprint.iacr.org/2021/459), and the secure floating-point library from ["SecFloat: Accurate Floating-Point meets Secure 2-Party Computation"](https://eprint.iacr.org/2022/) and ["Secure Floating-Point Training"](about:blank).

SEAF extends these frameworks by introducing efficient protocols for secure evaluation of activation functions with dynamic precision, ensuring both accuracy and privacy in machine learning inference tasks. This library is designed to enable privacy-preserving computation for a wide range of machine learning models, offering enhancements in performance and precision for secure two-party computation settings.

## Required Packages
 - g++ (version >= 8)
 - cmake
 - make
 - libgmp-dev
 - libmpfr-dev
 - libssl-dev  
 - SEAL 4.1.1
 - Eigen 3.3

SEAL can be installed as follows:

```
mkdir -p extern && cd extern
git clone --branch 4.1.1 https://github.com/microsoft/SEAL.git
```

Eigen can be installed as follows:

```
mkdir -p extern && cd extern
git clone https://github.com/libigl/eigen.git
cd eigen
mkdir build && cd build
cmake ..
sudo make install
```

The other packages can be installed directly using `sudo apt-get install <package>` on Linux.


## Compilation

To compile the library:

```
mkdir build && cd build
sudo cmake -DCMAKE_INSTALL_PREFIX=./install .. -DBUILD_TESTS=ON -DBUILD_NETWORKS=ON
cmake --build . --target install --parallel
```


## Running Tests

On successful compilation, the test and network binaries will be created in `build/bin/`.

Run the tests as follows to make sure everything works as intended:

`./<test> r=1 [port=port] & ./<test> r=2 [port=port]`

For the end to end inference like `./BOLT_BERT_SEAF` in the `SEAF/` folder of Generalized Geometric MPC Protocols, use the following command:

```bash
./SCI/build/bin/BOLT_BERT_SEAF r=1 & ./SCI/build/bin/BOLT_BERT_SEAF r=2
```

# Acknowledgements

This library includes code from the following external repositories:

 - [emp-toolkit/emp-tool](https://github.com/emp-toolkit/emp-tool/tree/c44566f40690d2f499aba4660f80223dc238eb03/emp-tool) for cryptographic tools and network I/O.
 - [emp-toolkit/emp-ot](https://github.com/emp-toolkit/emp-ot/tree/0f4a1e41a25cf1a034b5796752fde903a241f482/emp-ot) for Naor-Pinkas (base) OT and IKNP OT extension implementation.
 - [mc2-project/delphi](https://github.com/mc2-project/delphi/tree/de77cd7b896a2314fec205a8f67b257df46dd75c/rust/protocols-sys/c++/src/lib) for implementation of [Gazelle's](https://eprint.iacr.org/2018/073.pdf) algorithms for convolution and fully connected layers, which was majorly modified for better efficiency. 
 - [homenc/HElib](https://github.com/homenc/HElib/blob/6397b23e64c32fd6eab76bd7a08b95d8399503f4/src/NumbTh.h) for command-line argument parsing.