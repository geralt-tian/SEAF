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
uint64_t lb = 7;
uint64_t la = 6; // la=5 f=5,la=14,f=12
uint64_t f = 12;
uint64_t s = 6;

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
// s = 5(低精度)，s = 6(高)， s = 7 与 s = 6 误差相差不大
Truncation *trunc_oracle;
AuxProtocols *aux;
int dim = 14336;
uint64_t acc = 2;
uint64_t init_input = 2089984;
uint64_t step_size = 1;
uint64_t correct = 1;
// double calculate_GELU(uint64_t value)
// {
//     //     // 定义 2^37 和 2^12 的浮点值
//     const uint64_t sign_bit_mask = 1ULL << (bwL - 1); // 第 37 位的掩码
//     const double pow_2_21 = static_cast<double>(1ULL << bwL);
//     const double pow_2_12 = static_cast<double>(1ULL << f);

//     // 检查符号位（第 37 位）
//     if (value & sign_bit_mask)
//     {
//         // 如果符号位为 1，表示负数
//         value -= static_cast<uint64_t>(pow_2_21); // 减去 2^37
//     }
//     // 将值转换为浮点数
//     double x = static_cast<double>(value) / pow_2_12;
//     return 0.5 * x + 0.5 * x * std::erf(x / 1.414);
// }
double calculate_GELU(uint64_t value, uint64_t f_GELU)
{
    // 假设 bwL 和 f 已经定义，分别表示总位宽和小数部分位数
    const int64_t shift_amount = 64 - bwL; // 计算需要左移的位数

    // 将无符号整数进行符号扩展
    int64_t signed_value = static_cast<int64_t>(value << shift_amount) >> shift_amount;

    // 将定点数转换为浮点数
    const double pow_2_f = static_cast<double>(1ULL << f_GELU);
    double x = static_cast<double>(signed_value) / pow_2_f;

    // 计算 GELU 函数值
    return 0.5 * x + 0.5 * x * std::erf(x / 1.414);
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

//////////////////////
// 初始化
///////////////////////////////

int init_test(uint64_t i, uint64_t j, uint64_t k, uint64_t l)
{

    uint64_t la = i;
    uint64_t lb = j;
    // la=5 f=5,la=14,f=12
    uint64_t s = k;
    uint64_t f = l;

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
    for (int i = 0; i < dim; i++)
    {
        inA[i] = 0 + i * 0;
        inB[i] = init_input + i * step_size;
    }
    uint64_t *inA_h = new uint64_t[dim];
    uint64_t *inB_h = new uint64_t[dim];

    // std::cout << "\n=========STEP3 use DRelu to learn [[b]]^B===========" << std::endl;

    uint8_t *Drelu = new uint8_t[dim];
    uint8_t *msbA = new uint8_t[dim];
    uint8_t *msbB = new uint8_t[dim];
    uint8_t *wrap = new uint8_t[dim];
    uint64_t STEP3_comm_start = iopack->get_comm();
    // Drelu = MSB , Alice ^1
    if (party == ALICE)
    {
        prod->aux->MSBnew(inA, msbA, wrap, dim, bwL);
    }
    else
    {
        prod->aux->MSBnew(inB, msbB, wrap, dim, bwL);
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

    // for (int i = 0; i < dim; i++)
    // {
    //     std::cout << "wrap[" << i << "] = " << static_cast<int>(wrap[i]) << std::endl;
    // }

    // std::cout << "Drelu[" << 0 << "] = " << static_cast<int>(Drelu[0]) << std::endl;

    // std::cout << "\n=========STEP4 use EMUX to learn [[|x|]]in L ring===========" << std::endl;
    uint64_t STEP4_comm_start = iopack->get_comm();
    aux = new AuxProtocols(party, iopack, otpack);
    uint64_t *EMUX_output_x = new uint64_t[dim];
    uint64_t *neg_inA = new uint64_t[dim];
    uint64_t *neg_inB = new uint64_t[dim];

    if (party == ALICE)
    {
        for (int i = 0; i < dim; i++)
        {
            neg_inA[i] = ((-inA[i]) & mask_bwL); // 取反
        }
        select_share(Drelu, inA, neg_inA, EMUX_output_x, dim, bwL);
        // aux->multiplexerabs(Drelu, inA, EMUX_output_x, dim, bwL, bwL);
    }
    else
    {
        for (int i = 0; i < dim; i++)
        {
            neg_inB[i] = ((-inB[i]) & mask_bwL); // 取反
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
    // std::cout << "\n=========STEP7 extract the lower h bits===========" << std::endl;
    // std::cout << "inB[" << 0 << "] = " << inB[0] << std::endl;
    assign_lower_h_bits(dim, inA, inB, inA_h, inB_h, h);

    //////////////////////////////////////////////////////// general版本：直接截取，不用截断协议；高精度版本：使用截断协议
    // step6 check
    // std::cout << "\n=========STEP7 get mid s bit for LUT===========" << std::endl;

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
    uint64_t *EMUX_output_x1_h = new uint64_t[dim];
    for (int i = 0; i < dim; i++)
    {
        // std::cout << "outtrunc[" << i << "] = " << outtrunc[i] << std::endl;
        EMUX_output_x1_h[i] = EMUX_output_x1[i] & mask_h;
    }
    uint64_t STEP5_comm_start = iopack->get_comm();
    if (party == sci::ALICE)
    {
        trunc_oracle->truncate_and_reduce(dim, EMUX_output_x1_h, outtrunc, h - s, h); // shift=h-s,hypothesis s=7  truncate就是为了分组，截断后7位，为了前8位可以映射到对应的table
                                                                                      // std::cout << "outtrunc[" << 0 << "] = " << outtrunc[0] << std::endl;
    }
    else
    {
        trunc_oracle->truncate_and_reduce(dim, EMUX_output_x1_h, outtrunc, h - s, h); // shift=h-s,hypothesis s=7,outtrunc是0-127
                                                                                      // std::cout << "outtrunc[" << 0 << "] = " << outtrunc[0] << std::endl; // outtrunc是<i>，范围是0-127
    }
    uint64_t STEP5_comm_end = iopack->get_comm();
    // std::cout << "\n=========STEP6 LookUp Table   ===========" << std::endl;
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
    ////////////////////////////////////////////ALICE从csv文件中读取data，做自动化测试
    std::vector<std::vector<uint64_t>> data;
    if (party == ALICE)
    {
        std::ifstream file("/home/zhaoqian/EzPC/la_ld_s.csv");
        if (!file.is_open())
        {
            std::cerr << "fail to open the file!" << std::endl;
            return 1;
        }

        std::string line;
        int target_line = 28 * (la - 6) + 4 * (lb - 7) + 2 * (s - 6) + 2; // 目标行号（从0计数：0行是 la=6,ld=10，1行是数据行）计算一下行号
        int current_line = 0;

        // 定义存储数据的二维 vector

        // 逐行读取文件
        while (std::getline(file, line))
        {
            current_line++;

            // 如果当前行是目标行
            if (current_line == target_line)
            {
                // 定位到 "{{" 和 "}}" 来提取数据部分
                std::size_t start_pos = line.find("{{");
                std::size_t end_pos = line.find("}}");

                if (start_pos != std::string::npos && end_pos != std::string::npos)
                {
                    // 提取数据部分，去掉 "{{" 和 "}}"
                    std::string data_part = line.substr(start_pos + 2, end_pos - start_pos - 2);

                    // 使用 stringstream 分割每个 {a,b}
                    std::stringstream ss(data_part);
                    std::string pair_str;

                    while (std::getline(ss, pair_str, '}'))
                    {
                        // 找到 '{' 的位置，忽略前面的逗号或空白
                        std::size_t open_bracket_pos = pair_str.find('{');
                        if (open_bracket_pos != std::string::npos)
                        {
                            pair_str = pair_str.substr(open_bracket_pos + 1); // 获取 { 后面的内容
                        }

                        // 将每个数对 (a,b) 解析为两个数字
                        std::stringstream pair_stream(pair_str);
                        std::string number_str;
                        std::vector<uint64_t> pair;

                        // 提取逗号分隔的数字
                        while (std::getline(pair_stream, number_str, ','))
                        {
                            if (!number_str.empty())
                            {
                                pair.push_back(static_cast<uint64_t>(std::stoull(number_str)));
                            }
                        }

                        // 确保这对数是有效的二元组
                        if (pair.size() == 2)
                        {
                            data.push_back(pair);
                        }
                    }
                }
            }
        }
        // 关闭文件
        file.close();

        // 输出解析得到的数据
        std::cout << "读取的二维数组数据：" << std::endl;
        for (const auto &vec : data)
        {
            std::cout << "{" << vec[0] << ", " << vec[1] << "}, ";
        }
        std::cout << std::endl;
    }

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
        iopack->io->send_data(outtrunc, dim * sizeof(uint64_t)); // 计算通信的时候减掉这部分
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

        // for (int i = 0; i < dim; i++)
        // {
        //     std::cout << "outtrunc[" << i << "] = " << outtrunc[i] << std::endl;
        //     std::cout << "outtrunc_a[" << i << "] = " << outtrunc_a[i] << std::endl;
        // }
        // std::cout << "outtrunc_a[" << 0 << "] = " << outtrunc_a[0] << std::endl;
    }
    uint64_t STEP6_comm_start = iopack->get_comm();
    if (party == ALICE)
    {
        aux->lookup_table<uint64_t>(spec_a, nullptr, nullptr, dim, s, la); // bw_xlut是outtrunc的位宽
    }
    else
    {                                                                        // party == BOB
        aux->lookup_table<uint64_t>(nullptr, outtrunc_a, a_bob, dim, s, la); // a_bob是查询到的斜率
    }
    // if (party != ALICE)
    //     for (int i = 0; i < dim; i++)
    //     {
    //         // std::cout << "a_bob[" << i << "] = " << a_bob[i] << std::endl;
    //     }
    // /////选择截距
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
        aux->lookup_table<uint64_t>(spec_b, nullptr, nullptr, dim, s, lb);
    }
    else
    {                                                                        // party == BOB
        aux->lookup_table<uint64_t>(nullptr, outtrunc_a, b_bob, dim, s, lb); // b_bob是查询到的截距  重要问题，这里的outtrunc应该是两边share加起来，代码里只有Bob的outtrunc check
    }
    // if (party != ALICE)
    //     std::cout << "b_bob[" << 0 << "] = " << b_bob[0] << std::endl;

    uint64_t STEP6_comm_end = iopack->get_comm();
    // cout << "LUT Bytes Sent: " << (comm_end_lut - comm_start_lut) << "bytes" << endl;

    ext = new XTProtocol(party, iopack, otpack);

    // std::cout << "\n=========STEP7 multiplication to get a|x| l+la ===========" << std::endl;
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
            // std::cout << "a_alice[" << 0 << "] = " << a_alice[0] << std::endl;
            // std::cout << "EMUX_output_x[" << 0 << "] = " << EMUX_output_x[0] << std::endl;
            // prod->hadamard_product_MSB(dim, a_alice, EMUX_output_x, outax, la, bwL, la + bwL, true, true, mode, msb1, msb2);
            prod->hadamard_product(dim, a_alice, EMUX_output_x, outax, la, bwL, la + bwL, true, true, mode, msb1, msb2);
            // std::cout << "outax[" << 0 << "] = " << outax[0] << std::endl;
        }
        else
        {
            //     std::cout << "inB_h[" << 0 << "] = " << inB_h[0] << std::endl;
            //     std::cout << "a_bob[" << 0 << "] = " << a_bob[0] << std::endl;
            //     std::cout << "EMUX_output_x[" << 0 << "] = " << EMUX_output_x[0] << std::endl;
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

    // std::cout << "\n=========STEP8 ax truncate from l+la to l+1  ===========" << std::endl; // 跟协议对不上，这里直接得到了axl
    //////////////////////////////////////////////////////// general版本：直接截取，不用截断协议；高精度版本：使用截断协议
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
            trunc_oracle->truncate_and_reduce(dim, outax, mid_ax, la - 1, bwL + la);
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

    // std::cout << "\n=========STEP11 d SExt with MSB from f+1 to l   ===========" << std::endl;
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

        ext->s_extend_msb(dim, b_alice, b_SExt, lb, bwL, msb_b_extend);
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

    // std::cout << "\n=========STEP12 Caculate z=ax+b   ===========" << std::endl;
    uint64_t *z = new uint64_t[dim];

    // for (int i = 0; i < dim; i++)
    // {
    //     // std::cout << "b_SExt[" << i << "] = " << b_SExt[i] << std::endl;
    //     std::cout << "outax[" << i << "] = " << outax[i] << std::endl;
    // }
    // std::cout << "outax[" << 0 << "] = " << outax[0] << std::endl;
    uint64_t mask_d = (f + 1 == 64 ? -1 : ((1ULL << f + 1) - 1));
    for (int i = 0; i < dim; i++)
        z[i] = ((outax[i] + b_SExt[i] * static_cast<uint64_t>(std::pow(2, f - lb + 1))) & mask_bwL);

    // std::cout << "\n=========STEP14 Drelu |x|-a  to learn b' ===========" << std::endl;
    // 去掉13，修改14
    uint8_t *Drelu_ = new uint8_t[dim];
    uint8_t *DreluMSB = new uint8_t[dim];
    uint64_t STEP14_comm_start = iopack->get_comm();
    if (party == ALICE)
    {
        for (int i = 0; i < dim; i++)
        {
            EMUX_output_x[i] = (EMUX_output_x[i] - alpha) & mask_bwL;
            // std::cout << "EMUX_output_x[" << i << "] A =  " << EMUX_output_x[i] << std::endl;
            EMUX_output_x[i] = (EMUX_output_x[i] >> Tk) & mask_l_Tk;
            // std::cout << "EMUX_output_x[" << i << "] A trun =  " << EMUX_output_x[i] << std::endl;
        }
        prod->aux->MSBsec(EMUX_output_x, DreluMSB, dim, bwL - Tk);
    }
    else
    {
        for (int i = 0; i < dim; i++)
        {
            // std::cout << "EMUX_output_x[" << i << "] B =  " << EMUX_output_x[i] << std::endl;
            EMUX_output_x[i] = (EMUX_output_x[i] >> Tk) & mask_l_Tk;
            // std::cout << "EMUX_output_x[" << i << "] B trun =  " << EMUX_output_x[i] << std::endl;
        }

        // prod->aux->MSB(EMUX_output_x, DreluMSB, dim, bwL);
        prod->aux->MSBsec(EMUX_output_x, DreluMSB, dim, bwL - Tk);
    }

    if (party == ALICE)
    {
        for (int i = 0; i < dim; i++)
        {
            Drelu_[i] = DreluMSB[i] ^ 1;
        }
    }
    else
    {
        for (int i = 0; i < dim; i++)
        {
            Drelu_[i] = DreluMSB[i];
        }
    }
    uint64_t STEP14_comm_end = iopack->get_comm();
    // std::cout << "\n=========STEP15 get x_half ===========" << std::endl;
    int64_t STEP15_comm_start = iopack->get_comm();
    // online
    uint64_t *xhalf = new uint64_t[dim];
    uint64_t *abs_xhalf = new uint64_t[dim];
    uint64_t *bitMul_wrap = new uint64_t[dim];
    uint64_t *out_last_bitwrap = new uint64_t[dim];
    if (acc == 2)
    {
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
            neg_abs_xhalf[i] = -abs_xhalf[i] & mask_bwL;
        }
        select_share(Drelu, abs_xhalf, neg_abs_xhalf, xhalf, dim, bwL);
    }
    else
    {
        if (party == ALICE)
        {
            for (int i = 0; i < dim; i++)
            {
                xhalf[i] = (inA[i] >> 1) & mask_bwL;
            }
        }
        else
        {
            for (int i = 0; i < dim; i++)
            {
                xhalf[i] = (inB[i] >> 1) & mask_bwL;
            }
        }
    }

    // for (int i = 0; i < dim; i++)
    // {
    //     std::cout << "xhalf[" << i << "] = " << xhalf[i] << std::endl;
    //     std::cout << "abs_xhalf[" << i << "] = " << abs_xhalf[i] << std::endl;
    // }

    int64_t STEP15_comm_end = iopack->get_comm();
    // std::cout << "xhalf[" << 0 << "] = " << xhalf[0] << std::endl;
    // std::cout << "abs_xhalf[" << 0 << "] = " << abs_xhalf[0] << std::endl;

    // std::cout << "\n=========STEP16 get delta = z-x_half ===========" << std::endl;

    uint64_t *delta = new uint64_t[dim];
    for (int i = 0; i < dim; i++)
    {
        delta[i] = (abs_xhalf[i] - z[i]) & mask_bwL;
    }

    // std::cout << "\n=========STEP17 |g|=delta_ + x_half ===========" << std::endl;
    // uint64_t *delta_ = new uint64_t[dim];
    // for (int i = 0; i < dim; i++)
    // {
    //     delta_[i] = 0;
    // }
    // aux->multiplexer(Drelu_, delta, delta_, dim, bwL, bwL);
    // std::cout << "MUX_output_u[" << 0 << "] =" << MUX_output_u[0] << std::endl;
    uint64_t *MUX_output_g = new uint64_t[dim];
    int64_t STEP21_comm_start = iopack->get_comm();
    select_share(Drelu_, abs_xhalf, z, MUX_output_g, dim, bwL);
    int64_t STEP21_comm_end = iopack->get_comm();
    for (int i = 0; i < dim; i++)
    {
        // std::cout << "Drelu_[" << i << "] = " << static_cast<int>(Drelu_[i]) << std::endl;
        // std::cout << "delta_[" << i << "] = " << delta_[i] << std::endl;
        // std::cout << "z[" << i << "] = " << z[i] << std::endl;
        // MUX_output_g[i] = (delta_[i] + z[i]) & mask_bwL;

        // std::cout << "MUX_output_g[" << i << "] = " << MUX_output_g[i] << std::endl;
    }
    uint64_t comm_end = iopack->get_comm();
    // if (party == ALICE)
    // {
    //     iopack->io->send_data(Drelu_, dim * sizeof(uint8_t));
    // }
    // else
    // {
    //     uint8_t *recv_Drelu_ = new uint8_t[dim];
    //     iopack->io->recv_data(recv_Drelu_, dim * sizeof(uint8_t));
    //     for (int i = 0; i < dim; i++)
    //     {
    //         std::cout << "total Drelu_ [" << i << "]=  " << ((Drelu_[i] ^ recv_Drelu_[i])) << std::endl;
    //     }
    // }
    // std::cout << "\n=========STEP18 EMUX |g| to learn g ===========" << std::endl;

    // uint64_t *MUX_output_t = new uint64_t[dim];
    // aux->multiplexer(Drelu_, xhalf, MUX_output_t, dim, bwL, bwL);
    // uint64_t *EMUX_output_g = new uint64_t[dim];

    // // aux->multiplexerabs(Drelu, MUX_output_g, EMUX_output_g, dim, bwL, bwL);
    // if (party == ALICE)
    // {
    //     iopack->io->send_data(MUX_output_g, dim * sizeof(uint64_t));
    //     iopack->io->send_data(xhalf, dim * sizeof(uint64_t));
    // }
    // else
    // {
    //     uint64_t *recv_MUX_output_g = new uint64_t[dim];
    //     uint64_t *recv_xhalf = new uint64_t[dim];
    //     iopack->io->recv_data(recv_MUX_output_g, dim * sizeof(uint64_t));
    //     iopack->io->recv_data(recv_xhalf, dim * sizeof(uint64_t));
    //     for (int i = 0; i < dim; i++)
    //     {
    //         std::cout << "total MUX_output_g[" << i << "] = " << ((MUX_output_g[i] + recv_MUX_output_g[i]) & mask_bwL) << std::endl;
    //         std::cout << "total xhalf[" << i << "] = " << ((xhalf[i] + recv_xhalf[i]) & mask_bwL) << std::endl;
    //     }
    // }
    // std::cout << "\n=========STEP19 y = xhalf + u + v ===========" << std::endl;

    uint64_t *y = new uint64_t[dim];
    double average, max_val;
    for (int i = 0; i < dim; i++)
    {
        y[i] = (xhalf[i] + MUX_output_g[i]) & mask_bwL;
    }

    // std::cout << "\n=========END verification ===========" << std::endl;
    if (party == ALICE)
    {
        iopack->io->send_data(y, dim * sizeof(uint64_t));
        double Total_MSBytes_ALICE = static_cast<double>(comm_end - comm_start) / dim * 8;
        iopack->io->send_data(&Total_MSBytes_ALICE, sizeof(double));
        // double *Total_MSBytes_ALICE = new double[1];
        // Total_MSBytes_ALICE[0] = (comm_end - comm_start) / dim * 8;
        // iopack->io->send_data(Total_MSBytes_ALICE, dim * sizeof(double));
    }
    else
    {
        uint64_t *recv_y = new uint64_t[dim];
        // double *recv_Total_MSBytes_ALICE = new double[1];
        iopack->io->recv_data(recv_y, dim * sizeof(uint64_t));
        double recv_Total_MSBytes_ALICE;
        iopack->io->recv_data(&recv_Total_MSBytes_ALICE, sizeof(double));
        // iopack->io->recv_data(recv_Total_MSBytes_ALICE, sizeof(double));
        // std::cout << "total y = y0 + y1 =  " << ((y[0] + recv_y[0]) & mask_bwL) << ", real num: " << (double)decode_ring((y[0] + recv_y[0])&mask_bwL,37) / f_pow << std::endl;

        // std::cout << "ax +b =  " << (((inA[0] + inB[0]) * a_bob[0] + b_bob[0]) & mask_bwL) << std::endl;
        // std::cout << "ax +b  >> 12=  " << ((((inA[0] + inB[0]) * a_bob[0] + b_bob[0]) & mask_bwL) >> 12) << std::endl;
        // std::cout << "The result should be calculate_GELU = " << calculate_GELU(inA[0] + inB[0]) << std::endl;
        // std::vector<double> x_values, y_values;
        // std::vector<double> x_real, y_real;
        double *ULPs = new double[dim];
        double f_pow = pow(2, f);
        for (int i = 0; i < dim; i++)
        {
            // std::cout << "dim [" << i << "]total y = y0 + y1 =  " << ((y[i] + recv_y[i]) & mask_bwL) << ", real num: " << (double)decode_ring((y[i] + recv_y[i]) & mask_bwL, bwL) / f_pow << std::endl;

            // std::cout << "ax +b =  " << (((inA[i] + inB[i]) * a_bob[i] + b_bob[i]) & mask_bwL) << std::endl;
            // std::cout << "ax +b  >> 12=  " << ((((inA[i] + inB[i]) * a_bob[i] + b_bob[i]) & mask_bwL) >> 12) << std::endl;
            // std::cout << "The result " << inA[i] + inB[i] << " should be calculate_GELU = " << calculate_GELU(inA[i] + inB[i],f) << std::endl;
            ULPs[i] = abs((((double)decode_ring((y[i] + recv_y[i]) & mask_bwL, bwL) / f_pow) - calculate_GELU(inA[i] + inB[i], f)) / 0.000244140625);
            // std::cout << "The ULP is = " << ULPs[i] << std::endl;
        }

        double sum = 0.0;
        for (size_t i = 0; i < dim; ++i) // 去掉了第一个ULP
        {
            sum += (ULPs[i]);
            // std::cout << "ULPs[" << i << "] = " << ULPs[i] << std::endl;
        }
        double average = 0.0;
        double max_val = 0.0;
        average = sum / static_cast<double>(dim);
        // std::cout << "sum: " << sum << std::endl;
        // std::cout << "static_cast<double>(dim): " << static_cast<double>(dim) << std::endl;
        max_val = *std::max_element(ULPs, ULPs + dim); // 去掉了第一个ULP
        // std::cout << "average: " << average << std::endl;
        // std::cout << "max_val: " << max_val << std::endl;
        // 绘制曲线
        // plt::scatter(x_values, y_values, 2 , {{"color", "red"},{"marker", "."}});
        // plt::scatter(x_real, y_real, 1, {{"color", "blue"},{"marker", "."},{"edgecolors", "none"},
        //     {"alpha", "0.7"},
        //     {"label", "GELU"} });

        // 设置标题和标签
        // plt::title("Simple Line Plot");
        // plt::xlabel("x-axis");
        std::ofstream csvFile("/home/zhaoqian/EzPC/SCI/tests/auto_test_output.csv", std::ios::app);

        if (!csvFile.is_open())
        {
            std::cerr << "无法打开文件用于写入: auto_test_output.csv" << std::endl;
            return 1; // 或其他适当的错误处理
        }

        // 仅在文件为空时写入列名
        csvFile.seekp(0, std::ios::end);
        if (csvFile.tellp() == 0)
        {
            csvFile << "la,lb,f,s,average,max_val,"
                    << "Total_MSBytes,Total_time_ms\n";
        }

        // 示例计算（根据实际情况进行调整）
        double STEP3_MSBytes = (STEP3_comm_end - STEP3_comm_start) / dim * 8;
        double STEP4_MSBytes = (STEP4_comm_end - STEP4_comm_start) / dim * 8;
        double STEP5_MSBytes = (STEP5_comm_end - STEP5_comm_start) / dim * 8;
        double STEP6_MSBytes = (STEP6_comm_end - STEP6_comm_start) / dim * 8;
        double STEP7_MSBytes = (STEP7_comm_end - STEP7_comm_start) / dim * 8;
        double STEP8_MSBytes = (STEP8_comm_end - STEP8_comm_start) / dim * 8;
        double s_extend_comm_MSBytes = (s_extend_comm_end - s_extend_comm_start) / dim * 8;
        double STEP14_MSBytes = (STEP14_comm_end - STEP14_comm_start) / dim * 8;
        double STEP15_MSBytes = (STEP15_comm_end - STEP15_comm_start) / dim * 8;
        double STEP21_MSBytes = (STEP21_comm_end - STEP21_comm_start) / dim * 8;
        double Total_MSBytes_Bob = (comm_end - comm_start) / dim * 8;

        // 记录结束时间
        auto time_end = std::chrono::high_resolution_clock::now();
        double Total_time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(time_end - time_start).count();

        // 将数据写入 CSV 文件
        csvFile << la << ","
                << lb << ","
                << f << ","
                << s << ","
                << average << ","
                << max_val << ","
                << Total_MSBytes_Bob + recv_Total_MSBytes_ALICE << ","
                << Total_time_ms
                << "\n";

        // 关闭 CSV 文件
        csvFile.close();
    }
    auto time_end = chrono::high_resolution_clock::now();

    // 打开 CSV 文件（以追加模式打开）
    if (party != ALICE)
    {

        // 可选：在控制台输出确认信息
        std::cout << "数据已成功写入 auto_test_output.csv" << std::endl;
        ///////////输出时间和通信
        std::cout << "la=" << la << ",lb=" << lb << ",f=" << f << ",s=" << s << std::endl;
        cout << "STEP3 MSBnew Bytes Sent: " << (STEP3_comm_end - STEP3_comm_start) / dim * 8 << " bits" << endl;
        cout << "STEP4 Select_share Bytes Sent: " << (STEP4_comm_end - STEP4_comm_start) / dim * 8 << " bits" << endl;
        cout << "STEP5 TR Bytes Sent: " << (STEP5_comm_end - STEP5_comm_start) / dim * 8 << " bits" << endl;
        cout << "STEP6 LUT*2 Bytes Sent: " << (STEP6_comm_end - STEP6_comm_start) / dim * 8 << " bits" << endl;
        cout << "STEP7 hadamard_product Bytes Sent: " << (STEP7_comm_end - STEP7_comm_start) / dim * 8 << " bits" << endl;
        cout << "STEP8 truncate_and_reduce Bytes Sent: " << (STEP8_comm_end - STEP8_comm_start) / dim * 8 << " bits" << endl;
        cout << "s_extend_comm: " << (s_extend_comm_end - s_extend_comm_start) / dim * 8 << " bits" << endl;
        cout << "STEP14 DRELUsec Bytes Sent: " << (STEP14_comm_end - STEP14_comm_start) / dim * 8 << " bits" << endl;
        cout << "STEP15 clear_MSB_to_Wrap_bitMul and one trunc Bytes Sent: " << (STEP15_comm_end - STEP15_comm_start) / dim * 8 << " bits" << endl;
        cout << "STEP21 select_share Bytes Sent: " << (STEP21_comm_end - STEP21_comm_start) / dim * 8 << " bits" << endl;
        cout << "Total Bytes Sent: " << (comm_end - comm_start) / dim * 8 << " bits" << endl;

        cout << "Total time: "
             << chrono::duration_cast<chrono::milliseconds>(time_end - time_start).count()
             << " ms" << endl;
    }

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
        const std::string filename = "/home/zhaoqian/EzPC/SCI/tests/auto_test_output.csv";
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

    for (uint64_t i = 6; i < 14; i++)
    {

        for (uint64_t j = 7; j < 14; j++)
        {
            for (uint64_t k = 6; k < 8; k++)
            {
                for (uint64_t l = 11; l < 13; l++)
                {
                    if ((i <= l) & (j <= l))
                        init_test(i, j, k, l);
                    // std::cout << "la=" << la << ",lb=" << lb << ",f=" << f << ",s=" << s << std::endl;
                }
            }
        }
    }

    delete prod;
}
