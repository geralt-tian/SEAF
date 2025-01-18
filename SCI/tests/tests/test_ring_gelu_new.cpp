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
#include "BuildingBlocks/truncation.h"
#include "BuildingBlocks/aux-protocols.h"
#include <chrono>
#include <matplotlibcpp.h>
#include <cmath>
#include <fstream>
#include <string>
#include <vector>

using namespace sci;
using namespace std;
namespace plt = matplotlibcpp;
using namespace plt;
int party, port = 32000;
string address = "127.0.0.1";
IOPack *iopack;
OTPack *otpack;
LinearOT *prod;
XTProtocol *ext;

int bwL = 21; // 矩阵位宽
uint64_t mask_bwL = (bwL == 64 ? -1 : ((1ULL << bwL) - 1));
int bwL_1 = bwL - 1;
uint64_t mask_bwL_1 = (bwL_1 == 64 ? -1 : ((1ULL << bwL_1) - 1));
bool signed_B = true;           // 表示矩阵B是否为有符号数
bool accumulate = true;         // 决定是否累加结果
bool precomputed_MSBs = false;  // 决定是否预计算最高有效位
MultMode mode = MultMode::None; // 乘法模式

// uint64_t la = 14;//la=5 f=5,la=14,f=12
uint64_t lb = 10;
uint64_t la = 6; // la=5 f=5,la=14,f=12
uint64_t f = 11;
uint64_t s = 6;

// uint64_t h = f + 3;
uint64_t h = f + 2;
uint64_t Tk = f - 1;
uint64_t alpha = 4 * pow(2, f);
uint64_t mask_lb = (lb == 64 ? -1 : ((1ULL << lb) - 1));
uint64_t mask_l_Tk = (bwL == 64 ? -1 : ((1ULL << (bwL - Tk)) - 1));
uint64_t mask_lah1 = ((la + h + 1) == 64 ? -1 : ((1ULL << (la + h + 1)) - 1));
uint64_t mask_lla = ((la + bwL) == 64 ? -1 : ((1ULL << (la + bwL)) - 1));
// uint64_t s = 7;
uint64_t mask_s = ((s) == 64 ? -1 : ((1ULL << (s)) - 1));
uint64_t mask_h = (h == 64) ? ~0ULL : (1ULL << h) - 1;
// s = 5(低精度)，s = 6(高)， s = 7 与 s = 6 误差相差不大
Truncation *trunc_oracle;
AuxProtocols *aux;
MillionaireWithEquality *mill_eq;
Equality *eq;

int dim = 1000;
uint64_t acc = 2;
// uint64_t init_input = 2070769; //左边区间
uint64_t init_input = 2095151; // 中间区间
// uint64_t init_input = 0;  //右边区间
uint64_t step_size = 1;
uint64_t correct = 1;
double calculate_GELU(uint64_t value)
{
    // 假设 bwL 和 f 已经定义，分别表示总位宽和小数部分位数
    const int64_t shift_amount = 64 - bwL; // 计算需要左移的位数

    // 将无符号整数进行符号扩展
    int64_t signed_value = static_cast<int64_t>(value << shift_amount) >> shift_amount;

    // 将定点数转换为浮点数
    const double pow_2_f = static_cast<double>(1ULL << f);
    double x = static_cast<double>(signed_value) / pow_2_f;

    // 计算 GELU 函数值
    return 0.5 * x + 0.5 * x * std::erf(x / 1.414);
}
double calculate_ELU(uint64_t value, uint64_t f_ELU, double alpha = 1.0)
{
    // 计算需要左移的位数以进行符号扩展
    const int64_t shift_amount = 64 - bwL;

    // 将无符号整数进行符号扩展
    int64_t signed_value = static_cast<int64_t>(value << shift_amount) >> shift_amount;

    // 将定点数转换为浮点数
    const double pow_2_f = static_cast<double>(1ULL << f_ELU);
    double x = static_cast<double>(signed_value) / pow_2_f;

    // 计算 ELU 函数值
    if (x > 0.0)
    {
        return x;
    }
    else
    {
        return alpha * (std::exp(x) - 1.0);
    }
}

int64_t decode_ring(uint64_t input, uint64_t bw)
{
    // 从环上解码值
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

void assign_lower_h_bits(int32_t dim, uint64_t *inA, uint64_t *inB, uint64_t *input_lower_h, int32_t h)
{
    // Create a mask that has the lowest h bits set to 1
    uint64_t mask = (h == 64) ? ~0ULL : (1ULL << h) - 1;

    // Assign the lower h bits from inA to inA_

    if (party == ALICE) // Assign the lower h bits from inB to inB_
        for (int i = 0; i < dim; i++)
        {
            input_lower_h[i] = inA[i] & mask;
        }
    else
    {
        for (int i = 0; i < dim; i++)
        {
            input_lower_h[i] = inB[i] & mask;
        }
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

void EReLU_Eq(uint64_t *inA, uint8_t *b, uint8_t *b_, uint64_t dim, uint64_t bwl)
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

    mill_eq->compare_with_eq(carry, res_eq, comp_eq_input, dim, bwl);

    if (party == ALICE)
    {
        for (int i = 0; i < dim; i++)
        {
            b[i] = carry[i] ^ 1 ^ m[i];
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

void EReLU_Eq_eq0(uint64_t *inA, uint8_t *b, uint8_t *b_, uint8_t *beq0, uint64_t dim, uint64_t bwl)
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

    mill_eq->compare_with_eq(carry, res_eq, comp_eq_input, dim, bwl);

    if (party == ALICE)
    {
        for (int i = 0; i < dim; i++)
        {
            b[i] = carry[i] ^ 1 ^ m[i];
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

    eq->check_equality(beq0, inA, dim, bwl);
}
//////////////////////
// 初始化
///////////////////////////////

int init_test(uint64_t i, uint64_t j, uint64_t s_, uint64_t f_)
{

    uint64_t la = i;
    uint64_t lb = j;
    // la=5 f=5,la=14,f=12
    uint64_t s = s_;
    uint64_t f = f_;

    // uint64_t h = f + 3;
    uint64_t h = f + 2;
    uint64_t Tk = f - 1;
    uint64_t alpha = 3.5 * pow(2, f);
    uint64_t mask_lb = (lb == 64 ? -1 : ((1ULL << lb) - 1));
    uint64_t mask_l_Tk = (bwL == 64 ? -1 : ((1ULL << (bwL - Tk)) - 1));
    uint64_t mask_lah1 = ((la + h + 1) == 64 ? -1 : ((1ULL << (la + h + 1)) - 1));
    uint64_t mask_lla = ((la + bwL) == 64 ? -1 : ((1ULL << (la + bwL)) - 1));
    // uint64_t s = 7;
    uint64_t mask_s = ((s) == 64 ? -1 : ((1ULL << (s)) - 1));
    uint64_t mask_h = (h == 64) ? ~0ULL : (1ULL << h) - 1;
    uint64_t comm_start = iopack->get_comm();
    auto time_start = chrono::high_resolution_clock::now();

    uint64_t *inA = new uint64_t[dim];
    uint64_t *inB = new uint64_t[dim];

    uint64_t *outax = new uint64_t[dim];
    // la=6 lb=7
    // std::vector<std::vector<uint64_t>> data = {{1, 0}, {2, 0}, {4, 127}, {5, 127}, {7, 126}, {9, 125}, {10, 124}, {11, 123}, {12, 122}, {13, 121}, {15, 119}, {16, 117}, {16, 117}, {17, 116}, {18, 114}, {18, 114}, {19, 112}, {19, 112}, {20, 110}, {20, 110}, {20, 110}, {20, 110}, {20, 110}, {20, 110}, {20, 110}, {20, 110}, {20, 110}, {20, 110}, {19, 113}, {19, 113}, {19, 113}, {19, 113}, {19, 113}, {18, 117}, {18, 117}, {18, 117}, {18, 117}, {18, 117}, {17, 122}, {17, 122}, {17, 122}, {17, 122}, {17, 122}, {17, 122}, {17, 122}, {17, 122}, {16, 0}, {16, 0}, {16, 0}, {16, 0}, {16, 0}, {16, 0}, {16, 0}, {16, 0}, {16, 0}, {16, 0}, {16, 0}, {16, 0}, {16, 0}, {16, 0}, {16, 0}, {16, 0}, {16, 0}, {16, 0}};
    // la=6 lb=10
    // std::vector<std::vector<uint64_t>> data = {{1, 0}, {2, 1023}, {4, 1019}, {6, 1013}, {7, 1009}, {8, 1004}, {10, 992}, {11, 985}, {12, 977}, {14, 959}, {15, 949}, {16, 937}, {16, 937}, {17, 924}, {18, 910}, {18, 911}, {19, 895}, {19, 895}, {20, 877}, {20, 876}, {20, 876}, {20, 876}, {20, 877}, {20, 877}, {20, 877}, {20, 877}, {20, 877}, {20, 876}, {19, 904}, {19, 905}, {19, 905}, {19, 905}, {19, 905}, {18, 938}, {18, 938}, {18, 938}, {18, 938}, {18, 938}, {17, 976}, {17, 976}, {17, 976}, {17, 976}, {17, 976}, {17, 976}, {17, 976}, {17, 975}, {16, 1021}, {16, 1022}, {16, 1022}, {16, 1022}, {16, 1023}, {16, 1023}, {16, 1023}, {16, 1023}, {16, 1023}, {16, 0}, {16, 0}, {16, 0}, {16, 0}, {16, 0}, {16, 0}, {16, 0}, {16, 0}, {16, 0}};
    std::vector<std::vector<uint64_t>> data = {{3, 0}, {3, 1023}, {3, 1019}, {3, 1013}, {9, 1009}, {9, 1004}, {9, 992}, {9, 985}, {14, 977}, {14, 959}, {14, 949}, {14, 937}, {18, 937}, {18, 924}, {18, 910}, {18, 911}, {19, 895}, {19, 895}, {19, 877}, {19, 876}, {20, 876}, {20, 876}, {20, 877}, {20, 877}, {20, 877}, {20, 877}, {20, 877}, {20, 876}, {19, 904}, {19, 905}, {19, 905}, {19, 905}, {18, 905}, {18, 938}, {18, 938}, {18, 938}, {18, 938}, {18, 938}, {18, 976}, {18, 976}, {17, 976}, {17, 976}, {17, 976}, {17, 976}, {17, 976}, {17, 975}, {17, 1021}, {17, 1022}, {16, 1022}, {16, 1022}, {16, 1023}, {16, 1023}, {16, 1023}, {16, 1023}, {16, 1023}, {16, 0}, {16, 0}, {16, 0}, {16, 0}, {16, 0}, {16, 0}, {16, 0}, {16, 0}, {16, 0}};
    for (int i = 0; i < dim; i++)
    {
        inA[i] = (0 + i * 0) & mask_bwL;
        inB[i] = (init_input + i * step_size) & mask_bwL;
    }
    uint64_t *input_cut_h = new uint64_t[dim];
    // uint64_t *inB_cut_h = new uint64_t[dim];

    // std::cout << "\n=========STEP??  ===========" << std::endl;
    mill_eq = new MillionaireWithEquality(party, iopack, otpack);
    trunc_oracle = new Truncation(party, iopack, otpack);
    aux = new AuxProtocols(party, iopack, otpack);
    eq = new Equality(party, iopack, otpack);
    ext = new XTProtocol(party, iopack, otpack);
    // comp_eq前进行tr，截掉后14位，方便进行比较，将-alpha——0映射到-1
    uint64_t *comp_eq_input = new uint64_t[dim];
    uint64_t tr = f + 2;
    uint64_t mask_bwL_sub_tr = ((bwL - tr) == 64 ? -1 : ((1ULL << (bwL - tr)) - 1));
    uint64_t mask_bwL_sub_tr_sub_1 = ((bwL - tr - 1) == 64 ? -1 : ((1ULL << (bwL - tr - 1)) - 1));

    std::cout << "mask_bwL_sub_tr_sub_1 = " << mask_bwL_sub_tr_sub_1 << std::endl;
    uint64_t mask_TR_sub_1 = (bwL - h - 1 == 64) ? ~0ULL : (1ULL << bwL - h - 1) - 1;
    std::cout << "mask_TR_sub_1 = " << mask_TR_sub_1 << std::endl;
    uint64_t *eight_bit_wrap = new uint64_t[dim];

    uint64_t Comm_start = iopack->get_comm();

    uint64_t TR_wrap_start = iopack->get_comm();

    uint64_t *abs_x = new uint64_t[dim];
    uint64_t *neg_inA = new uint64_t[dim];
    uint64_t *neg_inB = new uint64_t[dim];
    uint8_t *Drelu = new uint8_t[dim];
    uint8_t *msbA = new uint8_t[dim];
    uint8_t *msbB = new uint8_t[dim];

    if (party == ALICE)
    {
        prod->aux->MSB(inA, msbA, dim, bwL);
    }
    else
    {
        prod->aux->MSB(inB, msbB, dim, bwL);
    }
    uint64_t STEP3_comm_end = iopack->get_comm();
    if (party == ALICE)
    {
        for (int i = 0; i < dim; i++)
        {
            Drelu[i] = msbA[i] ^ 1;
        }
    }
    else
    {
        for (int i = 0; i < dim; i++)
        {
            Drelu[i] = msbB[i];
        }
    }
    if (party == ALICE)
    {
        for (int i = 0; i < dim; i++)
        {
            neg_inA[i] = ((-inA[i]) & mask_bwL); // 取反
        }
        select_share(Drelu, inA, neg_inA, abs_x, dim, bwL);
    }
    else
    {
        for (int i = 0; i < dim; i++)
        {
            neg_inB[i] = ((-inB[i]) & mask_bwL); // 取反
        }
        select_share(Drelu, inB, neg_inB, abs_x, dim, bwL);
    }
    if (party == ALICE)
    {
        trunc_oracle->truncate_and_reduce_eight_bit_wrap(dim, inA, input_cut_h, eight_bit_wrap, tr, bwL);
        for (int i = 0; i < dim; i++)
        {
            comp_eq_input[i] = mask_bwL_sub_tr_sub_1 + 1 - (input_cut_h[i] & mask_bwL_sub_tr_sub_1);
            // std::cout << "input_cut_h[" << i << "] = " << (input_cut_h[i] & mask_bwL_sub_tr_sub_1) << std::endl;
            std::cout << "eight_bit_wrap[" << i << "] = " << eight_bit_wrap[i] << std::endl;
        }
    }
    else
    {
        trunc_oracle->truncate_and_reduce_eight_bit_wrap(dim, inB, input_cut_h, eight_bit_wrap, tr, bwL);
        for (int i = 0; i < dim; i++)
        {
            comp_eq_input[i] = input_cut_h[i] & mask_bwL_sub_tr_sub_1;
            // std::cout << "input_cut_h[" << i << "] = " << (input_cut_h[i] & mask_bwL_sub_tr_sub_1) << std::endl;
            std::cout << "eight_bit_wrap[" << i << "] = " << eight_bit_wrap[i] << std::endl;
            // eq_input[i] = TR_output[i] & mask_TR_sub_1;
        }
    }
    uint64_t TR_wrap_end = iopack->get_comm();
    // comp_eq 得到 eq=1时，值映射到-1，means在中间区间。否则，comp=0，为负值，comp=1为正。comp_eq还需要得到一个wrap
    uint8_t *res_cmp = new uint8_t[dim];
    uint8_t *res_eq_neg1 = new uint8_t[dim];
    uint8_t *res_eq_0 = new uint8_t[dim];

    // mill_eq->compare_with_eq(res_cmp, res_eq,  comp_eq_input, dim, bwL - h); // res_wrapcomp1是最右边的叶子节点
    uint64_t EReLU_Eq_start = iopack->get_comm();
    EReLU_Eq_eq0(input_cut_h, res_cmp, res_eq_neg1, res_eq_0, dim, bwL - h);
    uint64_t EReLU_Eq_end = iopack->get_comm();
    for (int i = 0; i < dim; i++)
    {
        std::cout << "res_cmp[" << i << "] = " << static_cast<int>(res_cmp[i]) << std::endl;
        std::cout << "res_eq_neg1[" << i << "] = " << static_cast<int>(res_eq_neg1[i]) << std::endl;
        // std::cout << "non_negative1_part[" << i << "] = " << non_negative1_part[i] << std::endl;
    }

    // tr 得到中间长度为s的index，进行LUT
    uint64_t *input_lower_h = new uint64_t[dim];
    uint64_t *outtrunc = new uint64_t[dim];

    assign_lower_h_bits(dim, abs_x, abs_x, input_lower_h, h);

    for (int i = 0; i < dim; i++)
    {
        std::cout << "input_lower_h[" << i << "] = " << input_lower_h[i] << std::endl;
    }

    trunc_oracle->truncate_and_reduce(dim, input_lower_h, outtrunc, h - s, h); // 这个不需要tr，可以本地截断加wrap，wrap上一步已经算好了

    // for (int i = 0; i < dim; i++)
    // {
    //     // outtrunc[i] = input_lower_h[i] & mask_s;
    //     outtrunc[i] = input_lower_h[i] >> (h - s);
    //     outtrunc[i] = (outtrunc[i] + eight_bit_wrap[i]) & mask_s;
    // }

    for (int i = 0; i < dim; i++)
    {
        std::cout << "***outtrunc[" << i << "] = " << outtrunc[i] << std::endl;
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
            std::cout << "outtrunc_a[" << i << "] = " << outtrunc_a[i] << std::endl;
        }
    }
    uint64_t N = 1ULL << s; // LUT size

    uint64_t **spec_a = new uint64_t *[dim]; // 查表前还要先给表赋值
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

    uint64_t two_LUT_start = iopack->get_comm();
    uint64_t *a_bob = new uint64_t[dim];
    if (party == ALICE)
    {
        aux->lookup_table<uint64_t>(spec_a, nullptr, nullptr, dim, s, la); // bw_xlut是outtrunc的位宽
    }
    else
    {                                                                        // party == BOB
        aux->lookup_table<uint64_t>(nullptr, outtrunc_a, a_bob, dim, s, la); // a_bob是查询到的斜率
    }

    uint64_t **spec_b = new uint64_t *[dim]; // 查表前还要先给表赋值
    if (party == ALICE)
        for (int i = 0; i < dim; i++)
        {
            spec_b[i] = new uint64_t[N];
            for (int j = 0; j < N; j++)
            {
                spec_b[i][j] = data[j][1];
            }
        }
    uint64_t *b_bob = new uint64_t[dim];
    // if (party == ALICE)
    //     for (int i = 0; i < dim; i++)
    //     {
    //         spec_b[i] = new uint64_t[N];
    //         for (int j = 0; j < N; j++)
    //         {
    //             spec_b[i][j] = data[j][1];
    //         }
    //     }
    if (party == ALICE)
    {
        aux->lookup_table<uint64_t>(spec_b, nullptr, nullptr, dim, s, lb);
    }
    else
    {                                                                        // party == BOB
        aux->lookup_table<uint64_t>(nullptr, outtrunc_a, b_bob, dim, s, lb); // b_bob是查询到的截距  重要问题，这里的outtrunc应该是两边share加起来，代码里只有Bob的outtrunc check
    }
    uint64_t two_LUT_end = iopack->get_comm();
    uint64_t *a_alice = new uint64_t[dim];
    uint64_t *b_alice = new uint64_t[dim];
    for (size_t i = 0; i < dim; i++)
    {
        a_alice[i] = 0;
        b_alice[i] = 0;
    }
    // 做乘法，已知msb
    uint8_t *msb1 = new uint8_t[dim];
    uint8_t *msb2 = new uint8_t[dim];
    for (int i = 0; i < dim; i++)
    {
        msb1[i] = 0;
        msb2[i] = 0;
    }
    // if (party == ALICE)
    // {
    //     msb1[i] = 0;
    //     msb2[i] = 0;
    // }
    uint64_t hadamard_product_start = iopack->get_comm();
    if (party == ALICE)
    {
        // prod->hadamard_product_MSB(dim, a_alice, inA, outax, la, bwL, la + bwL, true, true, mode, msb1, msb2);
        prod->hadamard_product(dim, a_alice, abs_x, outax, la, bwL, la + bwL, true, true, mode, msb1, msb2);
    }
    else
    {
        // prod->hadamard_product_MSB(dim, a_bob, inB, outax, la, bwL, la + bwL, true, true, mode, msb1, msb2);
        prod->hadamard_product(dim, a_bob, abs_x, outax, la, bwL, la + bwL, true, true, mode, msb1, msb2);
    }
    uint64_t hadamard_product_end = iopack->get_comm();
    uint64_t *mid_ax = new uint64_t[dim];
    uint64_t tr_start = iopack->get_comm();

    for (int i = 0; i < dim; i++)
    {
        std::cout << "a_bob[" << i << "] = " << a_bob[i] << std::endl;
        std::cout << "abs_x[" << i << "] = " << abs_x[i] << std::endl;
        std::cout << "outax[" << i << "] = " << outax[i] << std::endl;
    }
    // trunc_oracle->truncate_and_reduce(dim, outax, mid_ax, la - 1, bwL + la);

    for (int i = 0; i < dim; i++)
    {
        mid_ax[i] = (outax[i] >> (la - 1)) & mask_bwL;
    }

    uint64_t tr_end = iopack->get_comm();
    for (int i = 0; i < dim; i++)
    {
        // std ::cout << "outax[" << i << "] = " << outax[i] << std::endl;
        // std::cout << "mid_ax[" << i << "] = " << mid_ax[i] << std::endl;
        outax[i] = mid_ax[i];
        std ::cout << "outax[" << i << "] = " << outax[i] << std::endl;
    }
    uint64_t *b_SExt = new uint64_t[dim];

    uint8_t *msb_b_extend = new uint8_t[dim];

    uint64_t s_extend_comm_start = iopack->get_comm();
    std::cout << "\n=========STEP??  ===========" << std::endl;
    if (party == ALICE)
    {
        for (int i = 0; i < dim; i++)
        {
            msb_b_extend[i] = 1;
            b_alice[i] = (b_alice[i] + 10) & mask_lb;
        }
        ext->s_extend_msb(dim, b_alice, b_SExt, lb, bwL, msb_b_extend);
    }
    else
    {
        for (int i = 0; i < dim; i++)
        {
            msb_b_extend[i] = 1;
            b_bob[i] = (b_bob[i] - 10) & mask_lb;
        }
        ext->s_extend_msb(dim, b_bob, b_SExt, lb, bwL, msb_b_extend);
    }
    uint64_t s_extend_comm_end = iopack->get_comm();
    std::cout << "\n=========STEP??  ===========" << std::endl;
    uint64_t *z = new uint64_t[dim];
    for (int i = 0; i < dim; i++)
    {
        z[i] = ((outax[i] + b_SExt[i] * static_cast<uint64_t>(std::pow(2, f - lb + 1))) & mask_bwL);
        std::cout << "z[" << i << "] = " << z[i] << std::endl;
        std::cout << "outax[" << i << "] = " << outax[i] << std::endl;
        std::cout << "b_SExt[" << i << "] = " << b_SExt[i] << std::endl;
    }

    uint64_t *xhalf = new uint64_t[dim];
    uint64_t *abs_xhalf = new uint64_t[dim];
    uint64_t *bitMul_wrap = new uint64_t[dim];
    uint64_t *out_last_bitwrap = new uint64_t[dim];
    uint8_t *msb_zero = new uint8_t[dim];
    for (int i = 0; i < dim; i++)
    {
        msb_zero[i] = 0;
    }

    // std::cout << "acc == 2" << std::endl;
    if (party == ALICE)
    {
        // trunc_oracle->truncate(dim, inA, xhalf, 1, bwL, true, msbA);
        // uint8_t *msb_zero = new uint8_t[dim];

        aux->lastbit_MSB_to_Wrap_bitMul(dim, abs_x, out_last_bitwrap, bwL);
        aux->clear_MSB_to_Wrap_bitMul(dim, abs_x, msb_zero, bitMul_wrap, bwL);
        std::cout << "bitMul_wrap[" << 0 << "] = " << bitMul_wrap[0] << std::endl;
        for (int i = 0; i < dim; i++)
        {
            abs_xhalf[i] = ((abs_x[i] >> 1) - bitMul_wrap[i] * (uint64_t)pow(2, bwL - 1) + out_last_bitwrap[i]) & mask_bwL;
        }
    }
    else
    {
        aux->lastbit_MSB_to_Wrap_bitMul(dim, abs_x, out_last_bitwrap, bwL);
        aux->clear_MSB_to_Wrap_bitMul(dim, abs_x, msb_zero, bitMul_wrap, bwL);
        std::cout << "bitMul_wrap[" << 0 << "] = " << bitMul_wrap[0] << std::endl;
        for (int i = 0; i < dim; i++)
        {
            abs_xhalf[i] = ((abs_x[i] >> 1) - bitMul_wrap[i] * (uint64_t)pow(2, bwL - 1) + out_last_bitwrap[i]) & mask_bwL;
        }
    }
    uint64_t *neg_abs_xhalf = new uint64_t[dim];
    for (int i = 0; i < dim; i++)
    {
        neg_abs_xhalf[i] = -abs_xhalf[i] & mask_bwL;
    }
    select_share(Drelu, abs_xhalf, neg_abs_xhalf, xhalf, dim, bwL);

    // 判断区间

    uint8_t *z_or_negx = new uint8_t[dim];
    uint8_t *z_or_neghalfx_or_halfx = new uint8_t[dim];
    for (int i = 0; i < dim; i++)
    {
        z_or_negx[i] = res_eq_neg1[i] ^ res_eq_0[i];
        z_or_neghalfx_or_halfx[i] = res_eq_0[i] ^ res_cmp[i];
        std::cout << "res_eq_neg1[" << i << "] = " << static_cast<int>(res_eq_neg1[i]) << std::endl;
        std::cout << "res_eq_0[" << i << "] = " << static_cast<int>(res_eq_0[i]) << std::endl;
        std::cout << "res_cmp[" << i << "] = " << static_cast<int>(res_cmp[i]) << std::endl;
        std::cout << "z_or_negx[" << i << "] = " << static_cast<int>(z_or_negx[i]) << std::endl;
        std::cout << "z_or_neghalfx_or_halfx[" << i << "] = " << static_cast<int>(z_or_neghalfx_or_halfx[i]) << std::endl;
    }

    uint64_t *y = new uint64_t[dim];
    uint64_t *mid = new uint64_t[dim];
    // uint64_t postive_part = new uint64_t[dim];
    uint64_t two_select_share_start = iopack->get_comm();
    select_share(z_or_negx, z, neg_abs_xhalf, mid, dim, bwL);
    select_share(z_or_neghalfx_or_halfx, xhalf, mid, y, dim, bwL);

    for (int i = 0; i < dim; i++)
    {
        y[i] = (y[i] + xhalf[i]) & mask_bwL;
        std::cout << "z[" << i << "] = " << z[i] << std::endl;
        std::cout << "xhalf[" << i << "] = " << xhalf[i] << std::endl;
        std::cout << "y[" << i << "] = " << y[i] << std::endl;
    }

    uint64_t two_select_share_end = iopack->get_comm();

    uint64_t Comm_end = iopack->get_comm();

    if (party == ALICE)
    {
        iopack->io->send_data(y, dim * sizeof(uint64_t));
        // double Total_MSBytes_ALICE = static_cast<double>(comm_end - comm_start) / dim * 8;
        // iopack->io->send_data(&Total_MSBytes_ALICE, sizeof(double));
    }
    else
    {
        uint64_t *recv_y = new uint64_t[dim];
        iopack->io->recv_data(recv_y, dim * sizeof(uint64_t));
        // double recv_Total_MSBytes_ALICE;
        // iopack->io->recv_data(&recv_Total_MSBytes_ALICE, sizeof(double));
        double *ULPs = new double[dim];
        double f_pow = pow(2, f);
        for (int i = 0; i < dim; i++)
        {
            std::cout << "dim [" << i << "]total y = y0 + y1 =  " << ((y[i] + recv_y[i]) & mask_bwL) << ", real num: " << (double)decode_ring((y[i] + recv_y[i]) & mask_bwL, bwL) / f_pow << std::endl;

            // std::cout << "ax +b =  " << (((inA[i] + inB[i]) * a_bob[i] + b_bob[i]) & mask_bwL) << std::endl;
            // std::cout << "ax +b  >> 12=  " << ((((inA[i] + inB[i]) * a_bob[i] + b_bob[i]) & mask_bwL) >> 12) << std::endl;
            std::cout << "The result " << inA[i] + inB[i] << " should be calculate_GELU = " << calculate_GELU(inA[i] + inB[i]) << std::endl;
            ULPs[i] = abs((((double)decode_ring((y[i] + recv_y[i]) & mask_bwL, bwL) / f_pow) - calculate_GELU(inA[i] + inB[i])) / 0.000244140625);
            std::cout << "The ULP is = " << ULPs[i] << std::endl;
        }
        double sum = 0.0;
        for (size_t i = 0; i < dim; ++i) // 去掉了第一个ULP
        {
            sum += (ULPs[i]);
            // std::cout << "ULPs[" << i << "] = " << ULPs[i] << std::endl;
        }
        double average = 0.0;
        double max_val = 0.0;
        double min_val = 0.0;
        average = sum / static_cast<double>(dim);
        // std::cout << "sum: " << sum << std::endl;
        // std::cout << "static_cast<double>(dim): " << static_cast<double>(dim) << std::endl;
        max_val = *std::max_element(ULPs, ULPs + dim); // 去掉了第一个ULP
        min_val = *std::min_element(ULPs, ULPs + dim); // 去掉了第一个ULP
        std::cout << "average: " << average << std::endl;
        std::cout << "max_val: " << max_val << std::endl;
        std::cout << "min_val: " << min_val << std::endl;
    }
    cout << "TR_wrap Sent: " << (TR_wrap_end - TR_wrap_start) / dim * 8 << " bits" << endl;
    cout << "EReLU_Eq Sent: " << (EReLU_Eq_end - EReLU_Eq_start) / dim * 8 << " bits" << endl;
    cout << "Two LUT Sent: " << (two_LUT_end - two_LUT_start) / dim * 8 << " bits" << endl;
    cout << "Hadamard Product Sent: " << (hadamard_product_end - hadamard_product_start) / dim * 8 << " bits" << endl;
    cout << "S Extend Sent: " << (s_extend_comm_end - s_extend_comm_start) / dim * 8 << " bits" << endl;
    cout << "TR Sent: " << (tr_end - tr_start) / dim * 8 << " bits" << endl;
    cout << "Two Select Share Sent: " << (two_select_share_end - two_select_share_start) / dim * 8 << " bits" << endl;
    cout << "Total Bytes Sent: " << (Comm_end - Comm_start) / dim * 8 << " bits" << endl;
    // comp_eq w
    // AND - 128 - \eplison (128 - 20)
    // LUT - 256
    // LUT - s
    cout << "Total Bytes Sent: " << (Comm_end - Comm_start) / dim * 8 - 64 - 128 << " bits" << endl;
    delete[] inA;
    delete[] inB;
    delete[] outax;

    return 1;
}

int main(int argc, char **argv)
{
    ArgMapping amap;

    amap.arg("r", party, "Role of party: ALICE = 1; BOB = 2");
    amap.arg("p", port, "Port Number");
    amap.arg("ip", address, "IP Address of server (ALICE)");
    amap.arg("m", precomputed_MSBs, "MSB_to_Wrap Optimization?");
    amap.arg("a", ::accumulate, "Accumulate?");
    // amap.arg("dim", dim, "Dimension parameter for accumulation");
    // amap.arg("init_input", init_input, "init_input for accumulation");
    // amap.arg("step_size", step_size, "step_size for accumulation");
    // amap.arg("acc", acc, "acc=0 low, acc=1 general (default), acc =2 high");
    // amap.arg("correct", correct, "correct=1 or communication=2");

    amap.parse(argc, argv);

    // std::cout << "Parsed dimension (dim) = " << dim << std::endl;
    iopack = new IOPack(party, port, "127.0.0.1");
    otpack = new OTPack(iopack, party);
    prod = new LinearOT(party, iopack, otpack);

    if (party != ALICE)
    {
        const std::string filename = "/home/lzq/EzPC/SCI/tests/three_division_test_output.csv";
        std::ofstream csvFile;
        // 第一次访问，清空文件
        csvFile.open(filename, std::ios::out | std::ios::trunc);
        if (!csvFile.is_open())
        {
            std::cerr << "无法打开文件用于写入: " << filename << std::endl;
            return 1;
        }
        csvFile.close();
    }

    // for (uint64_t i = 8; i < 9; i++)
    for (uint64_t la = 6; la < 7; la++)
    {
        // for (uint64_t j = 8; j < 9; j++)
        for (uint64_t lb = 10; lb < 11; lb++)
        {
            for (uint64_t s = 6; s < 7; s++)
            {
                for (uint64_t k = 11; k < 12; k++)
                {
                    if ((la <= k) & (lb <= k))
                        init_test(la, lb, s, k);
                    // std::cout << "la=" << la << ",lb=" << lb << ",f=" << f << ",s=" << s << std::endl;
                }
            }
        }
    }

    delete prod;
}
