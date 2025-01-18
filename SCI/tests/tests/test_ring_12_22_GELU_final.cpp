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
#include "BuildingBlocks/truncation.h"
#include "BuildingBlocks/aux-protocols.h"
#include <chrono>
#include <matplotlibcpp.h>
#include <cmath>
#include <fstream>
#include <string>
#include <vector>

#define MAX_THREADS 4
using namespace sci;
using namespace std;
namespace plt = matplotlibcpp;
using namespace plt;
int party, port = 32000;
string address = "127.0.0.1";
IOPack *iopack;
OTPack *otpack;
// int num_threads = 4;
// IOPack *iopack[MAX_THREADS];
// OTPack *otpack[MAX_THREADS];
LinearOT *prod;
XTProtocol *ext;

int bwL = 21; 
uint64_t mask_bwL = (bwL == 64 ? -1 : ((1ULL << bwL) - 1));
int bwL_1 = bwL - 1;
uint64_t mask_bwL_1 = (bwL_1 == 64 ? -1 : ((1ULL << bwL_1) - 1));
bool signed_B = true;           
bool accumulate = true;        
bool precomputed_MSBs = false;  
MultMode mode = MultMode::None; 

// uint64_t la = 14;//la=5 f=5,la=14,f=12
uint64_t lb = 10;
uint64_t la = 10; // la=5 f=5,la=14,f=12
uint64_t f = 12;
uint64_t s = 6;
// int dim = 4096 * 8;
uint64_t dim = 1048576;
// int dim = 100;
uint64_t acc = 2;
// uint64_t init_input = 1032192-10;
uint64_t init_input = 2080768;
uint64_t step_size = 1;
uint64_t correct = 1;

uint64_t h = f + 2;
uint64_t d = f + 2;
uint64_t Tk = f - 1;
uint64_t alpha = 3.5 * pow(2, f);
uint64_t mask_lb = (lb == 64 ? -1 : ((1ULL << lb) - 1));
uint64_t mask_l_Tk = (bwL == 64 ? -1 : ((1ULL << (bwL - Tk)) - 1));
uint64_t mask_lah1 = ((la + h + 1) == 64 ? -1 : ((1ULL << (la + h + 1)) - 1));
uint64_t mask_lla = ((la + bwL) == 64 ? -1 : ((1ULL << (la + bwL)) - 1));
// uint64_t s = 7;
uint64_t mask_s = ((s) == 64 ? -1 : ((1ULL << (s)) - 1));
uint64_t mask_h = (h == 64) ? ~0ULL : (1ULL << h) - 1;
Truncation *trunc_oracle;
AuxProtocols *aux;
MillionaireWithEquality *mill_eq;
Equality *eq;
double calculate_GELU(uint64_t value)
{
    const int64_t shift_amount = 64 - bwL; 
    int64_t signed_value = static_cast<int64_t>(value << shift_amount) >> shift_amount;
    const double pow_2_f = static_cast<double>(1ULL << f);
    double x = static_cast<double>(signed_value) / pow_2_f;
    return 0.5 * x + 0.5 * x * std::erf(x / 1.414);
}

int64_t decode_ring(uint64_t input, uint64_t bw)
{
    uint64_t mask = (bw == 64) ? ~0ULL : (1ULL << bw) - 1;
    uint64_t half = 1ULL << (bw - 1);

    // std::cout << "input = " << input << std::endl;
    // std::cout << "half = " << half << std::endl;
    if (input == 1048576)
    {
        return 0;
    }
    if (input < half)
    {
        return input;
    }
    else
    {
        return -((1ULL << (bw)) - input);
    }
}

void assign_lower_h_bits(int32_t dim, uint64_t *inA, uint64_t *inB, uint64_t *inA_, uint64_t *inB_, int32_t h)
{
    // Create a mask that has the lowest h bits set to 1
    uint64_t mask = (h == 64) ? ~0ULL : (1ULL << h) - 1;

    // Assign the lower h bits from inA to inA_
    for (int i = 0; i < dim; i++)
    {
        inA_[i] = inA[i] & mask;
    }

    // Assign the lower h bits from inB to inB_
    for (int i = 0; i < dim; i++)
    {
        inB_[i] = inB[i] & mask;
    }
}

void select_share(uint8_t *sel, uint64_t *x, uint64_t *y, uint64_t *output, int32_t dim, int32_t h)
{
    // Create a mask that has the lowest h bits set to 1
    uint64_t mask = (h == 64) ? ~0ULL : (1ULL << h) - 1;
    uint64_t *mid = new uint64_t[dim];
    for (int i = 0; i < dim; i++)
    {
        mid[i] = (x[i] - y[i]) & mask;
    }

    aux->multiplexer(sel, mid, output, dim, h, h);

    for (int i = 0; i < dim; i++)
    {
        output[i] = (output[i] + y[i]) & mask;
    }
}

void DReLU_Eq(uint64_t *inA, uint8_t *b, uint8_t *b_, uint64_t dim, uint64_t bwl)
{
    uint8_t *m = new uint8_t[dim];
    uint64_t *y = new uint64_t[dim];
    uint64_t mask_l_sub1 = ((bwl - 1) == 64) ? ~0ULL : (1ULL << (bwl - 1)) - 1;
    for (int i = 0; i < dim; i++)
    {
        m[i] = inA[i] >> (bwl - 1);
        // std::cout << "m[" << i << "] = " << static_cast<int>(m[i]) << std::endl;
        y[i] = inA[i] & mask_l_sub1;
    }
    std::cout << "mask_l_sub1 = " << mask_l_sub1 << std::endl;
    uint64_t *comp_eq_input = new uint64_t[dim];

    if (party == ALICE)
    {
        for (int i = 0; i < dim; i++)
        {
            comp_eq_input[i] = (mask_l_sub1 - y[i]) & mask_l_sub1;
            // std::cout << "inA[" << i << "] = " << inA[i] << std::endl;
            // std::cout << "y[" << i << "] = " << y[i] << std::endl;
            // std::cout << "comp_eq_input[" << i << "] = " << comp_eq_input[i] << std::endl;
        }
    }
    else
    {
        for (int i = 0; i < dim; i++)
        {
            comp_eq_input[i] = y[i] & mask_l_sub1;
            // std::cout << "inA[" << i << "] = " << inA[i] << std::endl;
            // std::cout << "y[" << i << "] = " << y[i] << std::endl;
            // std::cout << "comp_eq_input[" << i << "] = " << comp_eq_input[i] << std::endl;
        }
    }

    uint8_t *carry = new uint8_t[dim];
    uint8_t *res_eq = new uint8_t[dim];
    mill_eq->compare_with_eq(carry, res_eq, comp_eq_input, dim, bwl - 1, false);
    for (int i = 0; i < dim; i++)
    {
        // std::cout << "carry[" << i << "] = " << static_cast<int>(carry[i]) << std::endl;
        // std::cout << "res_eq[" << i << "] = " << static_cast<int>(res_eq[i]) << std::endl;
    }
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

void third_interval(uint64_t *input_data, uint8_t *res_drelu_cmp, uint8_t *res_drelu_eq, uint8_t *res_eq)
{
    mill_eq = new MillionaireWithEquality(party, iopack, otpack);
    trunc_oracle = new Truncation(party, iopack, otpack);
    aux = new AuxProtocols(party, iopack, otpack);
    eq = new Equality(party, iopack, otpack);
    ext = new XTProtocol(party, iopack, otpack);
    uint64_t *comp_eq_input = new uint64_t[dim];
    uint64_t *outtrunc = new uint64_t[dim];
    // uint8_t *res_drelu_cmp = new uint8_t[dim];
    // uint8_t *res_drelu_eq = new uint8_t[dim];
    // uint8_t *res_eq = new uint8_t[dim];
    uint8_t *res_cmp = new uint8_t[dim];
    // TR
    uint64_t Comm_start = iopack->get_comm();
    auto time_start = std::chrono::high_resolution_clock::now();
    // if (party == ALICE)
    // {
    //     for (int i = 0; i < dim; i++)
    //     {
    //         input_data[i] = (input_data[i] - alpha) & mask_bwL;
    //     }
    // }
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
}

uint64_t computeULPErr(double calc, double actual, int SCALE)
{
    int64_t calc_fixed = (double(calc) * (1ULL << SCALE));
    int64_t actual_fixed = (double(actual) * (1ULL << SCALE));
    uint64_t ulp_err = (calc_fixed - actual_fixed) > 0
                           ? (calc_fixed - actual_fixed)
                           : (actual_fixed - calc_fixed);
    return ulp_err;
}

int init_test(uint64_t i, uint64_t j, uint64_t k, uint64_t l)
{

    uint64_t la = i;
    uint64_t lb = j;
    // la=5 f=5,la=14,f=12
    uint64_t s = k;
    uint64_t f = l;
    uint64_t h = f + 2;
    uint64_t d = f + 2;
    uint64_t Tk = f - 1;
    uint64_t alpha = 3.5 * pow(2, f);
    uint64_t mask_lb = (lb == 64 ? -1 : ((1ULL << lb) - 1));
    uint64_t mask_l_Tk = (bwL == 64 ? -1 : ((1ULL << (bwL - Tk)) - 1));
    uint64_t mask_lah1 = ((la + h + 1) == 64 ? -1 : ((1ULL << (la + h + 1)) - 1));
    uint64_t mask_lla = ((la + bwL) == 64 ? -1 : ((1ULL << (la + bwL)) - 1));
    // uint64_t s = 7;
    uint64_t mask_s = ((s) == 64 ? -1 : ((1ULL << (s)) - 1));
    uint64_t mask_h = (h == 64) ? ~0ULL : (1ULL << h) - 1;

    uint64_t *a_alice = new uint64_t[dim];
    uint64_t *b_alice = new uint64_t[dim];

    for (size_t i = 0; i < dim; i++)
    {
        a_alice[i] = 0;
        b_alice[i] = 0;
    }

    uint64_t **spec_a = new uint64_t *[dim];
    uint64_t *a_bob = new uint64_t[dim];
    uint64_t N = 1ULL << s; // LUT size
    ////////////////////////////////////////////
    std::vector<std::vector<uint64_t>> data;
    if (party == ALICE)
    {
        std::ifstream file("/home/ubuntu/EzPC/gelu_la_ld_s6.csv");
        if (!file.is_open())
        {
            std::cerr << "fail to open the file!" << std::endl;
            return 1;
        }

        std::string line;
        int target_line = 24 * (la - 2) + 2 * (lb - 1); 
        int current_line = 0;

        while (std::getline(file, line))
        {
            current_line++;

            if (current_line == target_line)
            {
                std::size_t start_pos = line.find("{{");
                std::size_t end_pos = line.find("}}");

                if (start_pos != std::string::npos && end_pos != std::string::npos)
                {

                    std::string data_part = line.substr(start_pos + 2, end_pos - start_pos - 2);


                    std::stringstream ss(data_part);
                    std::string pair_str;

                    while (std::getline(ss, pair_str, '}'))
                    {
  
                        std::size_t open_bracket_pos = pair_str.find('{');
                        if (open_bracket_pos != std::string::npos)
                        {
                            pair_str = pair_str.substr(open_bracket_pos + 1); 
                        }
                        std::stringstream pair_stream(pair_str);
                        std::string number_str;
                        std::vector<uint64_t> pair;
                        while (std::getline(pair_stream, number_str, ','))
                        {
                            if (!number_str.empty())
                            {
                                pair.push_back(static_cast<uint64_t>(std::stoull(number_str)));
                            }
                        }
                        if (pair.size() == 2)
                        {
                            data.push_back(pair);
                        }
                    }
                }
            }
        }

        file.close();
    }

    uint64_t comm_start = iopack->get_comm();
    auto time_start = chrono::high_resolution_clock::now();

    prod = new LinearOT(party, iopack, otpack);

    uint64_t *inA = new uint64_t[dim];
    uint64_t *inB = new uint64_t[dim];

    uint64_t *outax = new uint64_t[dim];
    for (int i = 0; i < dim; i++)
    {
        inA[i] = (0 + i * 0) & mask_bwL;
        inB[i] = (init_input + i * step_size) & mask_bwL;
    }
    uint8_t *outb = new uint8_t[dim];
    uint8_t *outb_star = new uint8_t[dim];
    uint8_t *outb_sharp = new uint8_t[dim];
    if (party == ALICE)
    {
        third_interval(inA, outb, outb_star, outb_sharp);
    }
    else
    {
        third_interval(inB, outb, outb_star, outb_sharp);
    }

    uint64_t *inA_h = new uint64_t[dim];
    uint64_t *inB_h = new uint64_t[dim];

    std::cout << "\n=========STEP3 use DRelu to learn [[b]]^B===========" << std::endl;

    uint8_t *Drelu = new uint8_t[dim];
    uint8_t *msbA = new uint8_t[dim];
    uint8_t *msbB = new uint8_t[dim];
    uint8_t *wrap = new uint8_t[dim];

    std::cout << "\n=========STEP4 use EMUX to learn [[|x|]]in L ring===========" << std::endl;
    uint64_t STEP4_comm_start = iopack->get_comm();
    aux = new AuxProtocols(party, iopack, otpack);
    uint64_t *EMUX_output_x = new uint64_t[dim];
    uint64_t *neg_inA = new uint64_t[dim];
    uint64_t *neg_inB = new uint64_t[dim];

    for (int i = 0; i < dim; i++)
    {
        Drelu[i] = outb[i];
    }
    if (party == ALICE)
    {
        for (int i = 0; i < dim; i++)
        {
            neg_inA[i] = ((-inA[i] - 1) & mask_bwL); 
        }
        select_share(Drelu, inA, neg_inA, EMUX_output_x, dim, bwL); // step 10
        // aux->multiplexerabs(Drelu, inA, EMUX_output_x, dim, bwL, bwL);
    }
    else
    {
        for (int i = 0; i < dim; i++)
        {
            neg_inB[i] = ((-inB[i]) & mask_bwL); // 
        }
        select_share(Drelu, inB, neg_inB, EMUX_output_x, dim, bwL);
        // aux->multiplexerabs(Drelu, inB, EMUX_output_x, dim, bwL, bwL);
    }
    uint64_t *EMUX_output_x1 = new uint64_t[dim];
    for (int i = 0; i < dim; i++)
    {
        EMUX_output_x1[i] = EMUX_output_x[i];
    }
    uint64_t STEP4_comm_end = iopack->get_comm();
    std::cout << "\n=========STEP7 extract the lower h bits===========" << std::endl;
    std::cout << "inB[" << 0 << "] = " << inB[0] << std::endl;
    assign_lower_h_bits(dim, inA, inB, inA_h, inB_h, h);

    // step6 check
    std::cout << "\n=========STEP7 get mid s bit for LUT===========" << std::endl;

    trunc_oracle = new Truncation(party, iopack, otpack);
    uint64_t *outtrunc = new uint64_t[dim];
    uint8_t *wrap_ = new uint8_t[dim];
    // if(acc==2){
    if (party == ALICE)
    {
        for (int i = 0; i < dim; i++)
        {
            wrap_[i] = (wrap[i] ^ Drelu[i] ^ 1) & 1;
        }
    }
    else
    {
        for (int i = 0; i < dim; i++)
        {
            wrap_[i] = (wrap[i] ^ Drelu[i]) & 1;
        }
    }

    ////////////////////////////////////////////////////////
    uint64_t *EMUX_output_x1_h = new uint64_t[dim];
    for (int i = 0; i < dim; i++)
    {
        // std::cout << "outtrunc[" << i << "] = " << outtrunc[i] << std::endl;
        EMUX_output_x1_h[i] = EMUX_output_x1[i] & mask_h;
    }
    uint64_t STEP5_comm_start = iopack->get_comm();
    if (party == sci::ALICE)
    {
        trunc_oracle->truncate_and_reduce(dim, EMUX_output_x1_h, outtrunc, h - s, h);
    }
    else
    {
        trunc_oracle->truncate_and_reduce(dim, EMUX_output_x1_h, outtrunc, h - s, h); 

    }
    uint64_t STEP5_comm_end = iopack->get_comm();
    std::cout << "\n=========STEP6 LookUp Table   ===========" << std::endl;

    //////////////////////////////////////////
    if (party == ALICE)
        for (int i = 0; i < dim; i++)
        {
            spec_a[i] = new uint64_t[N];
            for (int j = 0; j < N; j++)
            {
                spec_a[i][j] = data[j][0];
                // std::cout << "i = " << i << ", j = " << j << ", data = " << data[j][i] << std::endl;
            }
        }
    uint64_t *outtrunc1 = new uint64_t[dim];
    uint64_t *outtrunc_a = new uint64_t[dim];
    if (party == ALICE)
    {
        iopack->io->send_data(outtrunc, dim * sizeof(uint64_t));
    }
    else
    { // party == BOB
        iopack->io->recv_data(outtrunc1, dim * sizeof(uint64_t));

        for (int i = 0; i < dim; i++)
        {
            outtrunc_a[i] = (outtrunc[i] + outtrunc1[i]) & ((1ULL << s) - 1);
            // std::cout << "(outtrunc[i] + outtrunc1[i])" << i << "] = " << (outtrunc[i] + outtrunc1[i]) << std::endl;
            // std::cout << "((1ULL << s) - 1)" << ((1ULL << s) - 1) << std::endl;
        }

        for (int i = 0; i < dim; i++)
        {
            // std::cout << "outtrunc[" << i << "] = " << outtrunc[i] << std::endl;
            // std::cout << "outtrunc_a[" << i << "] = " << outtrunc_a[i] << std::endl;
        }
        // std::cout << "outtrunc_a[" << 0 << "] = " << outtrunc_a[0] << std::endl;
    }
    uint64_t STEP6_comm_start = iopack->get_comm();
    if (party == ALICE)
    {
        aux->lookup_table<uint64_t>(spec_a, nullptr, nullptr, dim, s, la); // step 12 lut
    }
    else
    {                                                                        // party == BOB
        aux->lookup_table<uint64_t>(nullptr, outtrunc_a, a_bob, dim, s, la); // 
    }
    if (party != ALICE)
        for (int i = 0; i < dim; i++)
        {
            // std::cout << "a_bob[" << i << "] = " << a_bob[i] << std::endl;
        }
    /////选择截距
    uint64_t **spec_b = new uint64_t *[dim];
    uint64_t *b_bob = new uint64_t[dim];
    if (party == ALICE)
        for (int i = 0; i < dim; i++)
        {
            spec_b[i] = new uint64_t[N];
            for (int j = 0; j < N; j++)
            {
                spec_b[i][j] = data[j][1];
            }
        }
    if (party == ALICE)
    {
        aux->lookup_table<uint64_t>(spec_b, nullptr, nullptr, dim, s, lb); // step 12 lut
    }
    else
    {                                                                        // party == BOB
        aux->lookup_table<uint64_t>(nullptr, outtrunc_a, b_bob, dim, s, lb); // 
    }
    if (party != ALICE)
        // std::cout << "b_bob[" << 0 << "] = " << b_bob[0] << std::endl;

        uint64_t STEP6_comm_end = iopack->get_comm();
    // cout << "LUT Bytes Sent: " << (comm_end_lut - comm_start_lut) << "bytes" << endl;

    ext = new XTProtocol(party, iopack, otpack);

    std::cout << "\n=========STEP7 multiplication to get a|x| l+la ===========" << std::endl;
    uint64_t STEP7_comm_start = iopack->get_comm();

    //   test_matrix_multiplication(inA, inB, outC, false);
    // test_matrix_multiplication(inA, inB, outC, true);
    uint8_t *msb1 = new uint8_t[dim];
    uint8_t *msb2 = new uint8_t[dim];
    for (int i = 0; i < dim; i++)
    {
        msb1[i] = 0;
        msb2[i] = 0;
    }
    if (correct == 1)
    {
        if (party == ALICE)
        {
            // std::cout << "inA_h[" << 0 << "] = " << inA_h[0] << std::endl;
            // std::cout << "a_alice[" << 1 << "] = " << a_alice[1] << std::endl;
            // std::cout << "EMUX_output_x[" << 1 << "] = " << EMUX_output_x[1] << std::endl;
            // prod->hadamard_product_MSB(dim, a_alice, EMUX_output_x, outax, la, bwL, la + bwL, true, true, mode, msb1, msb2);
            prod->hadamard_product(dim, a_alice, EMUX_output_x, outax, la, bwL, la + bwL, true, true, mode, msb1, msb2); // step 13 mul
            // std::cout << "outax[" << 0 << "] = " << outax[0] << std::endl;
        }
        else
        {
            // std::cout << "inB_h[" << 0 << "] = " << inB_h[0] << std::endl;
            // std::cout << "a_bob[" << 1 << "] = " << a_bob[1] << std::endl;
            // std::cout << "EMUX_output_x[" << 1 << "] = " << EMUX_output_x[1] << std::endl;
            // prod->hadamard_product_MSB(dim, a_bob, EMUX_output_x, outax, la, bwL, la + bwL, true, true, mode, msb1, msb2);
            prod->hadamard_product(dim, a_bob, EMUX_output_x, outax, la, bwL, la + bwL, true, true, mode, msb1, msb2);
            // std::cout << "outax[" << 0 << "] = " << outax[0] << std::endl;
        }
    }
    else
    {
        uint64_t *outuseless = new uint64_t[dim];
        if (party == ALICE)
        {
            trunc_oracle->unsigned_mul(dim, a_alice, EMUX_output_x, outuseless, la, bwL, la + bwL);
        }
        else
        {
            trunc_oracle->unsigned_mul(dim, a_bob, EMUX_output_x, outuseless, la, bwL, la + bwL);
        }
    }
    uint64_t STEP7_comm_end = iopack->get_comm();

    std::cout << "\n=========STEP8 ax truncate from l+la to l+1  ===========" << std::endl; // 
                                                                                            ////////////////////////////////////////////////////////
    uint8_t *msb_zero = new uint8_t[dim];
    for (int i = 0; i < dim; i++)
    {
        msb_zero[i] = 0;
    }
    uint64_t *mid_ax = new uint64_t[dim];
    uint64_t STEP8_comm_start = iopack->get_comm();
    if (acc == 1)
    {
        if (party == ALICE)
        {
            // trunc_oracle->truncate(dim, outax, mid_ax, la - 1, bwL + la, true, msb_zero);
            trunc_oracle->truncate_and_reduce(dim, outax, mid_ax, la - 1, bwL + la); // step 14 tr
        }
        else
        {
            // trunc_oracle->truncate(dim, outax, mid_ax, la - 1, bwL + la, true, msb_zero);
            trunc_oracle->truncate_and_reduce(dim, outax, mid_ax, la - 1, bwL + la);
        }
        for (int i = 0; i < dim; i++)
        {
            // std ::cout << "outax[" << i << "] = " << outax[i] << std::endl;
            // std::cout << "mid_ax[" << i << "] = " << mid_ax[i] << std::endl;
            outax[i] = mid_ax[i];
            // std ::cout << "outax[" << i << "] = " << outax[i] << std::endl;
        }
    }
    else
    {
        for (int i = 0; i < dim; i++)
        {
            outax[i] = (outax[i] >> (la - 1)) & mask_bwL;
        }
    }

    uint64_t STEP8_comm_end = iopack->get_comm();

    std::cout << "\n=========STEP11 d SExt with MSB from f+1 to l   ===========" << std::endl;
    uint64_t *b_SExt = new uint64_t[dim];

    uint8_t *msb_b_extend = new uint8_t[dim];

    uint64_t s_extend_comm_start = iopack->get_comm();
    if (party == ALICE)
    {
        for (int i = 0; i < dim; i++)
        {
            msb_b_extend[i] = 1;
            b_alice[i] = (b_alice[i] + 10) & mask_lb;
        }
        // ext->s_extend(dim, b_alice, b_SExt, lb, bwL, msb_b_extend);
        // std::cout << "b_alice[" << 0 << "] = " << b_alice[0] << std::endl;

        ext->s_extend_msb(dim, b_alice, b_SExt, lb, bwL, msb_b_extend); // step 17 s_extend
    }
    else
    {
        for (int i = 0; i < dim; i++)
        {
            msb_b_extend[i] = 1;
            b_bob[i] = (b_bob[i] - 10) & mask_lb;
        }
        // ext->s_extend(dim, b_bob, b_SExt, lb, bwL, msb_b_extend);
        // std::cout << "b_alice[" << 0 << "] = " << b_alice[0] << std::endl;

        ext->s_extend_msb(dim, b_bob, b_SExt, lb, bwL, msb_b_extend);
    }
    uint64_t s_extend_comm_end = iopack->get_comm();

    // std::cout << "b_SExt[" << 0 << "] = " << b_SExt[0] << std::endl;

    std::cout << "\n=========STEP12 Caculate z=ax+b   ===========" << std::endl;
    uint64_t *z = new uint64_t[dim];

    // for (int i = 0; i < dim; i++)
    // {
    //     std::cout << "b_SExt[" << i << "] = " << b_SExt[i] << std::endl;
    //     std::cout << "outax[" << i << "] = " << outax[i] << std::endl;
    // }
    // std::cout << "outax[" << 0 << "] = " << outax[0] << std::endl;
    uint64_t mask_d = (f + 1 == 64 ? -1 : ((1ULL << f + 1) - 1));
    for (int i = 0; i < dim; i++)
        z[i] = ((outax[i] + b_SExt[i] * static_cast<uint64_t>(std::pow(2, f - lb + 1))) & mask_bwL); // step 18 add

    std::cout << "\n=========STEP14 Drelu |x|-a  to learn b' ===========" << std::endl;
    uint8_t *Drelu_ = new uint8_t[dim];
    uint8_t *DreluMSB = new uint8_t[dim];
    uint64_t STEP14_comm_start = iopack->get_comm();
    uint64_t STEP14_comm_end = iopack->get_comm();
    std::cout << "\n=========STEP15 get x_half ===========" << std::endl;
    int64_t STEP15_comm_start = iopack->get_comm();
    // online
    uint64_t *xhalf = new uint64_t[dim];
    uint64_t *abs_xhalf = new uint64_t[dim];
    uint64_t *bitMul_wrap = new uint64_t[dim];
    uint64_t *out_last_bitwrap = new uint64_t[dim];
    // if (acc == 2)
    // {
    // std::cout << "acc == 2" << std::endl;
    if (party == ALICE)
    {
        // trunc_oracle->truncate(dim, inA, xhalf, 1, bwL, true, msbA);
        // uint8_t *msb_zero = new uint8_t[dim];

        aux->lastbit_MSB_to_Wrap_bitMul(dim, EMUX_output_x1, out_last_bitwrap, bwL);
        aux->clear_MSB_to_Wrap_bitMul(dim, EMUX_output_x1, msb_zero, bitMul_wrap, bwL);
        // std::cout << "bitMul_wrap[" << 0 << "] = " << bitMul_wrap[0] << std::endl;
        for (int i = 0; i < dim; i++)
        {
            abs_xhalf[i] = ((EMUX_output_x1[i] >> 1) - bitMul_wrap[i] * (uint64_t)pow(2, bwL - 1) + out_last_bitwrap[i]) & mask_bwL;
        }
    }
    else
    {
        // trunc_oracle->truncate(dim, inB, xhalf, 1, bwL, true, msbB);

        aux->lastbit_MSB_to_Wrap_bitMul(dim, EMUX_output_x1, out_last_bitwrap, bwL);
        aux->clear_MSB_to_Wrap_bitMul(dim, EMUX_output_x1, msb_zero, bitMul_wrap, bwL);
        // std::cout << "bitMul_wrap[" << 0 << "] = " << bitMul_wrap[0] << std::endl;
        for (int i = 0; i < dim; i++)
        {
            abs_xhalf[i] = ((EMUX_output_x1[i] >> 1) - bitMul_wrap[i] * (uint64_t)pow(2, bwL - 1) + out_last_bitwrap[i]) & mask_bwL;
        }
    }
    uint64_t *neg_abs_xhalf = new uint64_t[dim];
    for (int i = 0; i < dim; i++)
    {
        neg_abs_xhalf[i] = (-abs_xhalf[i]) & mask_bwL;
    }
    select_share(Drelu, abs_xhalf, neg_abs_xhalf, xhalf, dim, bwL); // step 22 ss
    // }
    // else
    // {
    //     if (party == ALICE)
    //     {
    //         for (int i = 0; i < dim; i++)
    //         {
    //             xhalf[i] = (inA[i] >> 1) & mask_bwL;
    //         }
    //     }
    //     else
    //     {
    //         for (int i = 0; i < dim; i++)
    //         {
    //             xhalf[i] = (inB[i] >> 1) & mask_bwL;
    //         }
    //     }
    // }

    for (int i = 0; i < dim; i++)
    {
        // std::cout << "xhalf[" << i << "] = " << xhalf[i] << std::endl;
        // std::cout << "abs_xhalf[" << i << "] = " << abs_xhalf[i] << std::endl;
    }

    int64_t STEP15_comm_end = iopack->get_comm();
    // std::cout << "xhalf[" << 0 << "] = " << xhalf[0] << std::endl;
    // std::cout << "abs_xhalf[" << 0 << "] = " << abs_xhalf[0] << std::endl;

    std::cout << "\n=========STEP16 get delta = z-x_half ===========" << std::endl;

    uint64_t *delta = new uint64_t[dim];
    for (int i = 0; i < dim; i++)
    {
        abs_xhalf[i] = (abs_xhalf[i]) & mask_bwL;
    }

    std::cout << "\n=========STEP17 |g|=delta_ + x_half ===========" << std::endl;
    // uint64_t *delta_ = new uint64_t[dim];
    // for (int i = 0; i < dim; i++)
    // {
    //     delta_[i] = 0;
    // }
    // aux->multiplexer(Drelu_, delta, delta_, dim, bwL, bwL);
    // std::cout << "MUX_output_u[" << 0 << "] =" << MUX_output_u[0] << std::endl;
    uint64_t *MUX_output_g = new uint64_t[dim];
    int64_t STEP21_comm_start = iopack->get_comm();

    for (int i = 0; i < dim; i++)
    {
        Drelu_[i] = (outb_star[i] + outb_sharp[i]) & 1;
        if (party == ALICE)
        {
            Drelu_[i] = Drelu_[i] ^ 1;
        }
    }

    select_share(Drelu_, abs_xhalf, z, MUX_output_g, dim, bwL); // step 20 ss

    int64_t STEP21_comm_end = iopack->get_comm();
    // for (int i = 0; i < dim; i++)
    // {
    //     std::cout << "outb[" << i << "] = " << static_cast<int>(outb[i]) << std::endl;
    //     std::cout << "outb_star[" << i << "] = " << static_cast<int>(outb_star[i]) << std::endl;
    //     std::cout << "outb_sharp[" << i << "] = " << static_cast<int>(outb_sharp[i]) << std::endl;
    //     std::cout << "Drelu_[" << i << "] = " << static_cast<int>(Drelu_[i]) << std::endl;
    //     // std::cout << "delta_[" << i << "] = " << delta_[i] << std::endl;
    //     std::cout << "z[" << i << "] = " << z[i] << std::endl;
    //     // MUX_output_g[i] = (delta_[i] + z[i]) & mask_bwL;abs_xhalf
    //     std::cout << "abs_xhalf[" << i << "] = " << abs_xhalf[i] << std::endl;
    //     std::cout << "neg_abs_xhalf[" << i << "] = " << neg_abs_xhalf[i] << std::endl;
    //     std::cout << "xhalf[" << i << "] = " << xhalf[i] << std::endl;
    //     std::cout << "MUX_output_g[" << i << "] = " << MUX_output_g[i] << std::endl;
    // }
    uint64_t comm_end = iopack->get_comm();
    std::cout << "\n=========STEP19 y = xhalf + u + v ===========" << std::endl;

    uint64_t *y = new uint64_t[dim];

    for (int i = 0; i < dim; i++)
    {
        y[i] = (xhalf[i] + MUX_output_g[i]) & mask_bwL;
    }
    auto time_end = chrono::high_resolution_clock::now();
    ////////////////////////////////////verfication
    // for (int i = 0; i < dim; i++)
    // {
    //     std::cout << "outb[" << i << "] = " << static_cast<int>(outb[i]) << std::endl;
    //     std::cout << "outb_star[" << i << "] = " << static_cast<int>(outb_star[i]) << std::endl;
    //     std::cout << "outb_sharp[" << i << "] = " << static_cast<int>(outb_sharp[i]) << std::endl;
    //     //outb_star[i] ^outb_sharp[i]
    //     std::cout << "total outb[" << i << "] = " << static_cast<int>((outb_star[i] + outb_sharp[i]) & 1) << std::endl;
    //     std::cout << "Drelu_[" << i << "] = " << static_cast<int>(Drelu_[i]) << std::endl;
    //     std::cout << "Drelu[" << i << "] = " << static_cast<int>(Drelu[i]) << std::endl;

    // }

    std::cout << "\n=========END verification ===========" << std::endl;
    if (party == ALICE)
    {
        iopack->io->send_data(y, dim * sizeof(uint64_t));
        uint64_t *comm = new uint64_t[1];
        comm[0] = (comm_end - comm_start) / dim * 8;
        iopack->io->send_data(comm, 1 * sizeof(uint64_t));
    }
    else
    {
        uint64_t *recv_y = new uint64_t[dim];
        iopack->io->recv_data(recv_y, dim * sizeof(uint64_t));
        // std::cout << "total y = y0 + y1 =  " << ((y[0] + recv_y[0]) & mask_bwL) << ", real num: " << (double)decode_ring((y[0] + recv_y[0])&mask_bwL,37) / f_pow << std::endl;

        // std::cout << "ax +b =  " << (((inA[0] + inB[0]) * a_bob[0] + b_bob[0]) & mask_bwL) << std::endl;
        // std::cout << "ax +b  >> 12=  " << ((((inA[0] + inB[0]) * a_bob[0] + b_bob[0]) & mask_bwL) >> 12) << std::endl;
        // std::cout << "The result should be calculate_GELU = " << calculate_GELU(inA[0] + inB[0]) << std::endl;
        std::vector<double> x_values, y_values;
        std::vector<double> x_real, y_real;
        double *ULPs = new double[dim];
        double f_pow = pow(2, f);
        int s_y = 12;
        for (int i = 0; i < dim; i++)
        {
            // std::cout << "dim [" << i << "]total y = y0 + y1 =  " << ((y[i] + recv_y[i]) & mask_bwL) << ", real num: " << (double)decode_ring((y[i] + recv_y[i]) & mask_bwL, bwL) / f_pow << std::endl;

            // std::cout << "ax +b =  " << (((inA[i] + inB[i]) * a_bob[i] + b_bob[i]) & mask_bwL) << std::endl;
            // std::cout << "ax +b  >> 12=  " << ((((inA[i] + inB[i]) * a_bob[i] + b_bob[i]) & mask_bwL) >> 12) << std::endl;
            // std::cout << "The result " << inA[i] + inB[i] << " should be calculate_GELU = " << calculate_GELU(inA[i] + inB[i]) << std::endl;
            // ULPs[i] = abs((((double)decode_ring((y[i] + recv_y[i]) & mask_bwL, bwL) / f_pow) - calculate_GELU(inA[i] + inB[i])) / 0.000244140625);
            ULPs[i] = computeULPErr(((double)decode_ring((y[i] + recv_y[i]) & mask_bwL, bwL) / f_pow), calculate_GELU(inA[i] + inB[i]), s_y);
            // std::cout << "The ULP is = " << ULPs[i] << std::endl;

            x_values.push_back((inA[i] + inB[i]) / (uint64_t)f_pow);
            y_values.push_back((double)decode_ring((y[i] + recv_y[i]) & mask_bwL, bwL) / (uint64_t)f_pow);
            x_real.push_back((inA[i] + inB[i]) / (uint64_t)f_pow);
            y_real.push_back(calculate_GELU(inA[i] + inB[i]));
        }

        double sum = 0.0;
        for (size_t i = 1; i < dim; ++i)
        {
            sum += (ULPs[i]);
            // std::cout << "ULPs[" << i << "] = " << ULPs[i] << std::endl;
        }
        double average = sum / static_cast<double>(dim);
        std::cout << "sum: " << sum << std::endl;
        std::cout << "static_cast<double>(dim): " << static_cast<double>(dim) << std::endl;
        double max_val = *std::max_element(ULPs + 1, ULPs + dim); 
        std::cout << "average: " << average << std::endl;
        std::cout << "max_val: " << max_val << std::endl;
        uint64_t *alice_comm = new uint64_t[1];
        iopack->io->recv_data(alice_comm, 1 * sizeof(uint64_t));

        uint64_t bob_comm = (comm_end - comm_start) / dim * 8;
        uint64_t total_comm = bob_comm + alice_comm[0];
        std::ofstream file("/home/ubuntu/EzPC/GELU_output_data.csv", std::ios_base::app);
        if (!file.is_open())
        {
            std::cerr << "Error: Could not open file for writing." << std::endl;
            return 1;
        }

        auto total_time = chrono::duration_cast<chrono::milliseconds>(time_end - time_start).count();
        file << la << "," << lb << ",  " << average << ",   " << max_val << "  , " << total_comm << ",    " << total_time << "\n";
    }

    ///////////

    // cout << "STEP3 MSBnew Bytes Sent: " << (STEP3_comm_end - STEP3_comm_start) / dim * 8 << " bits" << endl;
    cout << "STEP4 Select_share Bytes Sent: " << (STEP4_comm_end - STEP4_comm_start) / dim * 8 << " bits" << endl;
    cout << "STEP5 TR Bytes Sent: " << (STEP5_comm_end - STEP5_comm_start) / dim * 8 << " bits" << endl;
    // cout << "STEP6 LUT*2 Bytes Sent: " << (STEP6_comm_end - STEP6_comm_start) / dim * 8 << " bits" << endl;
    cout << "STEP7 hadamard_product Bytes Sent: " << (STEP7_comm_end - STEP7_comm_start) / dim * 8 << " bits" << endl;
    cout << "STEP8 truncate_and_reduce Bytes Sent: " << (STEP8_comm_end - STEP8_comm_start) / dim * 8 << " bits" << endl;
    std::cout << "s_extend_comm: " << (s_extend_comm_end - s_extend_comm_start) / dim * 8 << std::endl;
    cout << "STEP14 DRELUsec Bytes Sent: " << (STEP14_comm_end - STEP14_comm_start) / dim * 8 << " bits" << endl;
    cout << "STEP15 clear_MSB_to_Wrap_bitMul and one trunc Bytes Sent: " << (STEP15_comm_end - STEP15_comm_start) / dim * 8 << " bits" << endl;
    cout << "STEP21 select_share Bytes Sent: " << (STEP21_comm_end - STEP21_comm_start) / dim * 8 << " bits" << endl;
    // cout << "STEP3 Bytes Sent: " << (comm_end - comm_start) << "bytes" << endl;
    // cout << "STEP3 Bytes Sent: " << (comm_end - comm_start) << "bytes" << endl;
    // cout << "STEP3 Bytes Sent: " << (comm_end - comm_start) << "bytes" << endl;
    // cout << "STEP3 Bytes Sent: " << (comm_end - comm_start) << "bytes" << endl;
    // cout << "STEP3 Bytes Sent: " << (comm_end - comm_start) << "bytes" << endl;
    // uint64_t comm_end = iopack->get_comm();
    cout << "Total Bytes Sent: " << (comm_end - comm_start) / dim * 8 << " bits" << endl;

    cout << "Total time: "
         << chrono::duration_cast<chrono::milliseconds>(time_end - time_start).count()
         << " ms" << endl;
delete[] a_alice;
delete[] b_alice;
delete[] inA;
delete[] inB;
delete[] outax;
delete[] outb;
delete[] outb_star;
delete[] outb_sharp;
delete[] inA_h;
delete[] inB_h;
delete[] Drelu;
delete[] msbA;
delete[] msbB;
delete[] wrap;
delete[] EMUX_output_x;
delete[] neg_inA;
delete[] neg_inB;
delete[] EMUX_output_x1;
delete[] EMUX_output_x1_h;
delete[] outtrunc;
delete[] outtrunc1;
delete[] outtrunc_a;
delete[] a_bob;
delete[] b_bob;
delete[] msb1;
delete[] msb2;
delete[] msb_zero;
delete[] mid_ax;
delete[] b_SExt;
delete[] msb_b_extend;
delete[] z;
delete[] Drelu_;
delete[] DreluMSB;
delete[] xhalf;
delete[] abs_xhalf;
delete[] neg_abs_xhalf;
delete[] bitMul_wrap;
delete[] out_last_bitwrap;
delete[] delta;
delete[] MUX_output_g;
delete[] y;

if (party == ALICE) {
    for (int i = 0; i < dim; i++) {
        delete[] spec_a[i];
        delete[] spec_b[i];
    }
}
delete[] spec_a;
delete[] spec_b;

delete prod;
delete aux;
delete trunc_oracle;
delete ext;
    return 0;
}

int main(int argc, char **argv)
{
    ArgMapping amap;

    amap.arg("r", party, "Role of party: ALICE = 1; BOB = 2");
    amap.arg("p", port, "Port Number");
    amap.arg("ip", address, "IP Address of server (ALICE)");
    amap.arg("m", precomputed_MSBs, "MSB_to_Wrap Optimization?");
    amap.arg("a", ::accumulate, "Accumulate?");
    amap.arg("dim", dim, "Dimension parameter for accumulation");
    amap.arg("init_input", init_input, "init_input for accumulation");
    amap.arg("step_size", step_size, "step_size for accumulation");
    amap.arg("acc", acc, "acc=0 low, acc=1 general (default), acc =2 high");
    amap.arg("correct", correct, "correct=1 or communication=2");

    amap.parse(argc, argv);
    std::cout << "Parsed dimension (dim) = " << dim << std::endl;
    iopack = new IOPack(party, port, "127.0.0.1");
    otpack = new OTPack(iopack, party);

    // for (int i = 0; i < num_threads; i++)
    // {
    //     iopack[i] = new IOPack(party, port + i, address);
    //     if (i & 1)
    //     {
    //         otpack[i] = new OTPack(iopack[i], 3 - party);
    //     }
    //     else
    //     {
    //         otpack[i] = new OTPack(iopack[i], party);
    //     }
    // }

        std::vector<std::pair<uint64_t, uint64_t>> la_lb_pairs = {
//     {3, 3},
// {3, 4},
// {3, 5},
// {3, 6},
// {3, 7},
// {3, 8},
// {3, 9},
// {3, 10},
// {3, 11},
// {3, 12},
// {3, 13},
// {4, 3},
// {4, 4},
// {4, 5},
// {4, 6},
// {4, 7},
// {4, 8},
{4, 9},
{4, 10},
{4, 11},
// {4, 12},
// {4, 13},
// {5, 3},
// {5, 4},
// {5, 5},
// {5, 6},
// {5, 7},
// {5, 8},
// {5, 9},
// {5, 10},
{5, 11},
// {5, 12},
// {5, 13},
// {6, 3},
// {6, 4},
// {6, 5},
// {6, 6},
// {6, 7},
// {6, 8},
// {6, 9},
// {6, 10},
{6, 11},
{6, 12},
{6, 13},
// {7, 3},
// {7, 4},
// {7, 5},
// {7, 6},
// {7, 7},
// {7, 8},
// {7, 9},
// {7, 10},
// {7, 11},
// {7, 12},
// {7, 13},
// {8, 3},
// {8, 4},
// {8, 5},
// {8, 6},
// {8, 7},
// {8, 8},
// {8, 9},
// {8, 10},
// {8, 11},
// {8, 12},
{8, 13},
// {9, 3},
// {9, 4},
// {9, 5},
// {9, 6},
// {9, 7},
// {9, 8},
// {9, 9},
// {9, 10},
// {9, 11},
// {9, 12},
// {9, 13},
// {10, 3},
// {10, 4},
// {10, 5},
// {10, 6},
// {10, 7},
// {10, 8},
// {10, 9},
// {10, 10},
// {10, 11},
// {10, 12},
// {10, 13},
// {11, 3},
// {11, 4},
// {11, 5},
// {11, 6},
// {11, 7},
// {11, 8},
// {11, 9},
// {11, 10},
// {11, 11},
// {11, 12},
// {11, 13},
// {12, 3},
// {12, 4},
// {12, 5},
// {12, 6},
// {12, 7},
// {12, 8},
// {12, 9},
// {12, 10},
// {12, 11},
// {12, 12},
// {12, 13},
// {13, 3},
// {13, 4},
// {13, 5},
// {13, 6},
// {13, 7},
// {13, 8},
// {13, 9},
// {13, 10},
// {13, 11},
// {13, 12},
// {13, 13}
};
    // {8, 12}, {7, 12}, {6, 12}, {6, 11}, {5, 12}, {5, 10}, {4, 12}, {4, 10}};

    for (const auto &pair : la_lb_pairs)
    {
        uint64_t la = pair.first;
        uint64_t lb = pair.second;

        for (uint64_t s = 6; s < 7; s++)
        {
            for (uint64_t k = 12; k < 13; k++)
            {
                // if ((la <= k) & (lb <= k))
                // {
                    for (int i = 0; i < 1; i++)
                    {
                        init_test(la, lb, s, k);
                    }
                    // std::cout << "la = " << la << ", lb = " << lb << ", s = " << s << ", k = " << k << std::endl;
                // }
            }
        }
    }
    delete prod;
}
