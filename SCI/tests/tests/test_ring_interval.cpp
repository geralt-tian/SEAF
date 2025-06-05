#include "LinearOT/linear-ot.h"
#include "utils/emp-tool.h"
#include <iostream>

#include "FloatingPoint/floating-point.h"
#include "FloatingPoint/fp-math.h"
#include <random>
#include <limits>
#include "float_utils.h"

#include "Millionaire/millionaire.h"
#include "Millionaire/millionaire_with_equality.h"
#include "Millionaire/equality.h"
#include "Math/math-functions.h"
#include "BuildingBlocks/truncation.h"
#include "BuildingBlocks/aux-protocols.h"
#include <chrono>
// #include <matplotlibcpp.h>
#include <cmath>
#include <fstream>
#include <string>
#include <vector>

using namespace sci;
using namespace std;
// namespace plt = matplotlibcpp;
// using namespace plt;
int party, port = 32000;
string address = "127.0.0.1";
IOPack *iopack;
OTPack *otpack;
LinearOT *prod;
XTProtocol *ext;

int bwL = 21; 
uint64_t mask_bwL = (bwL == 64 ? -1 : ((1ULL << bwL) - 1));
bool accumulate = true;  
bool precomputed_MSBs = false;
MultMode mode = MultMode::None; 

uint64_t f = 12;
uint64_t d = f + 2;

uint64_t alpha = 4 * pow(2, f);
Truncation *trunc_oracle;
AuxProtocols *aux;
MillionaireWithEquality *mill_eq;
Equality *eq;
MathFunctions *math;
int dim = pow(2, 20);
uint64_t acc = 2;
uint64_t init_input = 0; 
uint64_t step_size = 1;
uint64_t correct = 1;
void DReLU_Eq(uint64_t *inA, uint8_t *b, uint8_t *b_, int32_t dim, int32_t bwl)
{
    uint8_t *m = new uint8_t[dim];
    uint64_t *y = new uint64_t[dim];
    uint64_t mask_l_sub1 = ((bwl - 1) == 64) ? ~0ULL : (1ULL << (bwl - 1)) - 1;
    for (int i = 0; i < dim; i++)
    {
        m[i] = inA[i] >> (bwl - 1);
        y[i] = inA[i] & mask_l_sub1;
    }
    uint64_t *comp_eq_input = new uint64_t[dim];
    if (party == ALICE)
    {
        for (int i = 0; i < dim; i++)
        {
            comp_eq_input[i] = (mask_l_sub1 - y[i]) & mask_l_sub1;
        }
    }
    else
    {
        for (int i = 0; i < dim; i++)
        {
            comp_eq_input[i] = y[i] & mask_l_sub1;
        }
    }
    uint8_t *carry = new uint8_t[dim];
    uint8_t *res_eq = new uint8_t[dim];
    mill_eq->compare_with_eq(carry, res_eq, comp_eq_input, dim, bwl - 1, false);
    if (party == ALICE)
    {
        for (int i = 0; i < dim; i++)
        {
            b[i] = carry[i] ^ 1 ^ m[i];
            // b[i] = carry[i] ^ m[i];
        }
    }
    else
    {
        for (int i = 0; i < dim; i++)
        {
            b[i] = carry[i] ^ m[i];
        }
    }

    aux->AND(res_eq, m, b_, dim);
}
//////////////////////
///////////////////////////////
int two_comparisons(uint64_t *input_data)
{

    mill_eq = new MillionaireWithEquality(party, iopack, otpack);
    trunc_oracle = new Truncation(party, iopack, otpack);
    aux = new AuxProtocols(party, iopack, otpack);
    eq = new Equality(party, iopack, otpack);
    math = new MathFunctions(party, iopack, otpack); // Initialize the 'math' object
    uint8_t *carry = new uint8_t[dim];
    uint8_t *res_eq = new uint8_t[dim];
    uint64_t *comp_eq_input = new uint64_t[dim];
    uint64_t Comm_start = iopack->get_comm();

    math->DReLU(dim, comp_eq_input, res_eq, bwL, 0); // Use the 'math' object
    math->DReLU(dim, comp_eq_input, res_eq, bwL, 0);
    uint64_t Comm_end = iopack->get_comm();
    std::cout << "two_comparisons Comm = " << (Comm_end - Comm_start) / dim * 8 << std::endl;
    return 1;
}
int first_interval(uint64_t *input_data)
{
    mill_eq = new MillionaireWithEquality(party, iopack, otpack);
    trunc_oracle = new Truncation(party, iopack, otpack);
    aux = new AuxProtocols(party, iopack, otpack);
    eq = new Equality(party, iopack, otpack);
    ext = new XTProtocol(party, iopack, otpack);
    uint64_t *comp_eq_input = new uint64_t[dim];
    uint64_t *outtrunc = new uint64_t[dim];
    uint64_t *input_data1 = new uint64_t[dim];
    uint8_t *res_cmp = new uint8_t[dim];
    uint8_t *res_eq = new uint8_t[dim];
    // TR

    if (party == ALICE)
    {
        for (int i = 0; i < dim; i++)
        {
            input_data1[i] = (input_data[i] - alpha) & mask_bwL;
        }
    }
    uint64_t Comm_start = iopack->get_comm();
    // auto time_start = std::chrono::high_resolution_clock::now();
    uint64_t trun_start = iopack->get_comm();
    trunc_oracle->truncate_and_reduce(dim, input_data1, outtrunc, d, bwL);
    uint64_t trun_end = iopack->get_comm();
    // ERELU_EQ
    uint64_t mask_l_sub1 = ((bwL - d - 1) == 64) ? ~0ULL : (1ULL << (bwL - d - 1)) - 1;
    // auto time_start = std::chrono::high_resolution_clock::now();
    if (party == ALICE)
    {
        for (int i = 0; i < dim; i++)
        {
            comp_eq_input[i] = (mask_l_sub1 - outtrunc[i]) & mask_l_sub1;
        }
    }
    else
    {
        for (int i = 0; i < dim; i++)
        {
            comp_eq_input[i] = outtrunc[i] & mask_l_sub1;
        }
    }
    uint64_t compare_with_eq_start = iopack->get_comm();
    eq->check_equality(res_eq, comp_eq_input, dim, bwL - d);
    uint64_t compare_with_eq_end = iopack->get_comm();
    uint64_t Comm_end = iopack->get_comm();

    std::cout << "Comm = " << (Comm_end - Comm_start) / dim * 8 << std::endl;
    std::cout << "Truncation = " << (trun_end - trun_start) / dim * 8 << std::endl;
    std::cout << "Compare_with_eq = " << (compare_with_eq_end - compare_with_eq_start) / dim * 8 << std::endl;
    return 1;
}

int second_interval(uint64_t *input_data)
{
    mill_eq = new MillionaireWithEquality(party, iopack, otpack);
    trunc_oracle = new Truncation(party, iopack, otpack);
    aux = new AuxProtocols(party, iopack, otpack);
    eq = new Equality(party, iopack, otpack);
    ext = new XTProtocol(party, iopack, otpack);
    uint64_t *comp_eq_input = new uint64_t[dim];
    uint64_t *outtrunc = new uint64_t[dim];
    uint8_t *res_cmp = new uint8_t[dim];
    uint8_t *res_drelu_eq = new uint8_t[dim];
    uint8_t *res_eq = new uint8_t[dim];
    // TR
    uint64_t Comm_start = iopack->get_comm();
    uint64_t trun_start = iopack->get_comm();
    trunc_oracle->truncate_and_reduce(dim, input_data, outtrunc, d, bwL); // test comm
    uint64_t trun_end = iopack->get_comm();
    uint64_t DReLU_Eq_start = iopack->get_comm();
    DReLU_Eq(outtrunc, res_cmp, res_drelu_eq, dim, bwL - d);
    uint64_t DReLU_Eq_end = iopack->get_comm();
    uint64_t Comm_end = iopack->get_comm();
    std::cout << "Comm = " << (Comm_end - Comm_start) / dim * 8 << std::endl;
    std::cout << "Truncation = " << (trun_end - trun_start) / dim * 8 << std::endl;
    std::cout << "DReLU_Eq = " << (DReLU_Eq_end - DReLU_Eq_start) / dim * 8 << std::endl;
    return 1;
}

void third_interval(uint64_t *input_data, uint8_t *res_drelu_cmp, uint8_t *res_drelu_eq, uint8_t *res_eq)
{
    mill_eq = new MillionaireWithEquality(party, iopack, otpack);
    trunc_oracle = new Truncation(party, iopack, otpack);
    aux = new AuxProtocols(party, iopack, otpack);
    eq = new Equality(party, iopack, otpack);
    ext = new XTProtocol(party, iopack, otpack);
    uint64_t *comp_eq_input = new uint64_t[dim];
    uint64_t *outtrunc = new uint64_t[dim];
    uint8_t *res_cmp = new uint8_t[dim];
    uint64_t Comm_start = iopack->get_comm();
    auto time_start = std::chrono::high_resolution_clock::now();
    uint64_t trun_start = iopack->get_comm();
    trunc_oracle->truncate_and_reduce(dim, input_data, outtrunc, d, bwL); // test comm
    DReLU_Eq(outtrunc, res_drelu_cmp, res_drelu_eq, dim, bwL - d);
    uint64_t mask_l_sub1 = ((bwL - d) == 64) ? ~0ULL : (1ULL << (bwL - d)) - 1;
    if (party == ALICE)
    {
        for (int i = 0; i < dim; i++)
        {
            comp_eq_input[i] = (mask_l_sub1 + 1 - outtrunc[i]) & mask_l_sub1;
        }
    }
    else
    {
        for (int i = 0; i < dim; i++)
        {
            comp_eq_input[i] = outtrunc[i] & mask_l_sub1;
        }
    }
    eq->check_equality(res_eq, comp_eq_input, dim, bwL - d);
    uint64_t Comm_end = iopack->get_comm();
    std::cout << "Comm = " << (Comm_end - Comm_start) / dim * 8 << std::endl;
}

int main(int argc, char **argv)
{
    ArgMapping amap;

    amap.arg("r", party, "Role of party: ALICE = 1; BOB = 2");
    amap.arg("p", port, "Port Number");
    amap.arg("ip", address, "IP Address of server (ALICE)");
    amap.arg("m", precomputed_MSBs, "MSB_to_Wrap Optimization?");
    amap.arg("a", ::accumulate, "Accumulate?");
    amap.parse(argc, argv);

    iopack = new IOPack(party, port, "127.0.0.1");
    uint64_t *inA = new uint64_t[dim]; // Declare the variable "inA"
    otpack = new OTPack(iopack, party);
    prod = new LinearOT(party, iopack, otpack);

    uint64_t *inB = new uint64_t[dim]; // Declare the variable "inB"

    for (int i = 0; i < dim; i++)
    {
        inA[i] = (0 + i * 0) & mask_bwL;
        inB[i] = (init_input + i * step_size) & mask_bwL;
    }

    for (int p = 0; p < 1; p++)
    {
        std::cout << "Single Interval Test :" << std::endl;
        auto time_start = std::chrono::high_resolution_clock::now();
        if (party == ALICE)
        {

            first_interval(inA);
        }
        else
        {
            inB[0] = 2097152 - 17000;
            inB[1] = 2097152 - 1;
            inB[2] = 0;
            first_interval(inB);
        }
        auto time_end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(time_end - time_start).count();
        std::cout << "Single Interval Test Time elapsed: " << duration << " microseconds" << std::endl;
        std::cout << " " << std::endl;
        std::cout << "second_interval :" << std::endl;
        auto sectime_start = std::chrono::high_resolution_clock::now();
        if (party == ALICE)
        {
            second_interval(inA);
        }
        else
        {
            second_interval(inB);
        }
        auto sectime_end = std::chrono::high_resolution_clock::now();
        auto secduration = std::chrono::duration_cast<std::chrono::microseconds>(sectime_end - sectime_start).count();
        std::cout << "second_interval Time elapsed: " << secduration << " microseconds" << std::endl;
        std::cout << " " << std::endl;

        /////////////////
        std::cout << "third_interval :" << std::endl;
        uint8_t *res_drelu_cmp = new uint8_t[dim];
        uint8_t *res_drelu_eq = new uint8_t[dim];
        uint8_t *res_eq = new uint8_t[dim];
        auto thirdtime_start = std::chrono::high_resolution_clock::now();
        if (party == ALICE)
        {
            third_interval(inA, res_drelu_cmp, res_drelu_eq, res_eq);
        }
        else
        {
            third_interval(inB, res_drelu_cmp, res_drelu_eq, res_eq);
        }
        auto thirdtime_end = std::chrono::high_resolution_clock::now();
        auto thirdduration = std::chrono::duration_cast<std::chrono::microseconds>(thirdtime_end - thirdtime_start).count();
        std::cout << "third_interval Time elapsed: " << thirdduration << " microseconds" << std::endl;
        std::cout << " " << std::endl;
        // for (uint64_t i = 8; i < 9; i++)
        std::cout << "Double DReLU :" << std::endl;
        auto twocmptime_start = std::chrono::high_resolution_clock::now();
        if (party == ALICE)
        {

            two_comparisons(inA);
        }
        else
        {
            two_comparisons(inB);
        }
        auto twocmptime_end = std::chrono::high_resolution_clock::now();
        auto twocmpduration = std::chrono::duration_cast<std::chrono::microseconds>(twocmptime_end - twocmptime_start).count();
        std::cout << "Double DReLU Time elapsed: " << twocmpduration << " microseconds" << std::endl;
    }

    delete prod;
    delete[] inA; // Delete the variable "inA" to avoid memory leaks
    delete[] inB;
}