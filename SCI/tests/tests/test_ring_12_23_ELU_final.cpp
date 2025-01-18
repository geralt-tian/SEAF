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
uint64_t la = 10; // la=5 f=5,la=14,f=12
uint64_t f = 12;
uint64_t s = 7;

uint64_t h = f + 3;
uint64_t d = f + 3;
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
MillionaireWithEquality *mill_eq;
Equality *eq;

// int dim = 4096 * 16;
uint64_t dim = 1048576;
// int dim = 1024;
uint64_t acc = 2;
// uint64_t init_input = 2097000; //左边区间
uint64_t init_input = 2064384; // 中间区间

// uint64_t init_input = 0;  //右边区间
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

uint64_t computeULPErr(double calc, double actual, int SCALE)
{
    int64_t calc_fixed = (double(calc) * (1ULL << SCALE));
    int64_t actual_fixed = (double(actual) * (1ULL << SCALE));
    uint64_t ulp_err = (calc_fixed - actual_fixed) > 0
                           ? (calc_fixed - actual_fixed)
                           : (actual_fixed - calc_fixed);
    return ulp_err;
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

void DReLU_Eq(uint64_t *inA, uint8_t *b, uint8_t *b_, uint64_t dim, int bwl)
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
  mill_eq->compare_with_eq(carry, res_eq, comp_eq_input, dim, bwl - 1,false);
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

int second_interval(uint64_t *input_data, uint8_t *res_drelu_cmp, uint8_t *res_drelu_eq)
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
    uint8_t *res_eq = new uint8_t[dim];
    // TR
    // uint64_t Comm_start = iopack->get_comm();
    // auto time_start = std::chrono::high_resolution_clock::now();
    // if (party == ALICE)
    // {
    //     for (int i = 0; i < dim; i++)
    //     {
    //         input_data[i] = (input_data[i] - alpha) & mask_bwL;
    //     }
    // }
    uint64_t trun_start = iopack->get_comm();
    trunc_oracle->truncate_and_reduce(dim, input_data, outtrunc, h, bwL); // test comm
    uint64_t trun_end = iopack->get_comm();
    // ERELU_EQ
    // auto time_start = std::chrono::high_resolution_clock::now();
    uint64_t DReLU_Eq_start = iopack->get_comm();
    DReLU_Eq(outtrunc, res_drelu_cmp, res_drelu_eq, dim, bwL - h);
    uint64_t DReLU_Eq_end = iopack->get_comm();
    // auto time_end = std::chrono::high_resolution_clock::now();

    // uint64_t addfor = static_cast<uint64_t>(pow(2, bwL - d));
    // for (int i = 0; i < dim; i++)
    // {
    //     comp_eq_input[i] = (addfor + outtrunc[i]) & mask_bwL; // 这里应该mod 多少？
    // }

    auto time_end = std::chrono::high_resolution_clock::now();
    uint64_t Comm_end = iopack->get_comm();
    // std::cout << "Comm = " << (Comm_end - Comm_start) / dim * 8 << std::endl;
    std::cout << "Truncation = " << (trun_end - trun_start) / dim * 8 << std::endl;
    std::cout << "DReLU_Eq = " << (DReLU_Eq_end - DReLU_Eq_start) / dim * 8 << std::endl;

    // auto duration = std::chrono::duration_cast<std::chrono::microseconds>(time_end - time_start).count();
    // std::cout << "Time elapsed: " << duration << " microseconds" << std::endl;
    // for (int i = 0; i < dim; i++)
    // {
    //     std::cout << "outtrunc[" << i << "] = " << outtrunc[i] << std::endl;
    //     std::cout << "res_cmp[" << i << "] = " << static_cast<int>(res_cmp[i]) << std::endl;
    //     std::cout << "res_drelu_eq[" << i << "] = " << static_cast<int>(res_drelu_eq[i]) << std::endl; // right
    // }

    return 1;
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

    uint64_t *a_alice = new uint64_t[dim];
    uint64_t *b_alice = new uint64_t[dim];

    for (size_t i = 0; i < dim; i++)
    {
        a_alice[i] = 0;
        b_alice[i] = 0;
    }
    
    // uint64_t **spec_a = new uint64_t *[dim];
    // uint64_t *a_bob = new uint64_t[dim];
    // uint64_t N = 1ULL << s; // LUT size
    ////////////////////////////////////////////ALICE从csv文件中读取data，做自动化测试
    std::vector<std::vector<uint64_t>> data;
    if (party == ALICE)
    {
        std::ifstream file("/home/ubuntu/EzPC/elu_la10_ld10_s7_test.csv");
        if (!file.is_open())
        {
            std::cerr << "fail to open the file!" << std::endl;
            return 1;
        }

        std::string line;
        int target_line = 24 * (la - 2) + 2 * (lb - 1); // 目标行号（从0计数：0行是 la=6,ld=10，1行是数据行）计算一下行号
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
    }
    uint64_t h = f + 3;
    uint64_t Tk = f - 1;
    uint64_t alpha = 8 * pow(2, f);
    uint64_t mask_lb = (lb == 64 ? -1 : ((1ULL << lb) - 1));
    uint64_t mask_l_Tk = (bwL == 64 ? -1 : ((1ULL << (bwL - Tk)) - 1));
    uint64_t mask_lah1 = ((la + h + 1) == 64 ? -1 : ((1ULL << (la + h + 1)) - 1));
    uint64_t mask_lla = ((la + bwL) == 64 ? -1 : ((1ULL << (la + bwL)) - 1));
    // uint64_t s = 7;
    uint64_t mask_s = ((s) == 64 ? -1 : ((1ULL << (s)) - 1));
    uint64_t mask_h = (h == 64) ? ~0ULL : (1ULL << h) - 1;
    prod = new LinearOT(party, iopack, otpack);
    uint64_t *inA = new uint64_t[dim];
    uint64_t *inB = new uint64_t[dim];

    uint64_t *outax = new uint64_t[dim];
    // la=8 lb=8
    // std::vector<std::vector<uint64_t>> data = {{1, 136}, {1, 136}, {3, 151}, {2, 143}, {2, 143}, {3, 150}, {1, 135}, {1, 135}, {1, 135}, {1, 135}, {1, 135}, {1, 135}, {2, 141}, {2, 141}, {4, 153}, {1, 134}, {1, 134}, {2, 140}, {1, 134}, {1, 134}, {1, 134}, {1, 134}, {1, 134}, {1, 134}, {0, 129}, {0, 129}, {1, 134}, {1, 134}, {1, 134}, {1, 134}, {0, 130}, {0, 130}, {3, 142}, {4, 146}, {4, 146}, {4, 146}, {2, 139}, {4, 146}, {4, 146}, {1, 137}, {7, 155}, {9, 161}, {9, 161}, {9, 161}, {11, 166}, {14, 173}, {14, 173}, {12, 169}, {19, 183}, {21, 187}, {25, 194}, {27, 197}, {31, 203}, {34, 207}, {37, 211}, {44, 219}, {51, 226}, {58, 232}, {65, 237}, {73, 242}, {83, 247}, {94, 251}, {105, 254}, {121, 0}};
    // la=10,ld=10,s=6
    // std::vector<std::vector<uint64_t>> data = {{10, 561}, {14, 577}, {13, 573}, {12, 569}, {12, 569}, {15, 580}, {15, 580}, {15, 580}, {15, 580}, {17, 587}, {20, 597}, {17, 587}, {21, 600}, {20, 597}, {19, 594}, {9, 564}, {27, 618}, {32, 633}, {31, 630}, {31, 630}, {35, 641}, {35, 641}, {40, 654}, {40, 654}, {44, 664}, {44, 664}, {49, 676}, {52, 683}, {56, 692}, {62, 705}, {63, 707}, {58, 697}, {72, 725}, {76, 733}, {83, 746}, {88, 755}, {92, 762}, {98, 772}, {106, 785}, {104, 782}, {118, 803}, {124, 812}, {132, 823}, {145, 840}, {149, 845}, {159, 857}, {175, 875}, {172, 872}, {195, 895}, {202, 902}, {219, 917}, {235, 930}, {251, 942}, {264, 951}, {285, 964}, {301, 973}, {321, 983}, {342, 992}, {366, 1001}, {389, 1008}, {409, 1013}, {441, 1019}, {466, 1022}, {499, 0}};
    // std::vector<std::vector<uint64_t>> data = {{72,725}, {76,733}, {83,746}, {88,755}, {92,762}, {98,772}, {106,785}, {104,782}, {118,803}, {124,812}, {132,823}, {145,840}, {149,845}, {159,857}, {175,875}, {172,872}, {195,895}, {202,902}, {219,917}, {235,930}, {251,942}, {264,951}, {285,964}, {301,973}, {321,983}, {342,992}, {366,1001}, {389,1008}, {409,1013}, {441,1019}, {466,1022}, {499,0}, {10,561}, {14,577}, {13,573}, {12,569}, {12,569}, {15,580}, {15,580}, {15,580}, {15,580}, {17,587}, {20,597}, {17,587}, {21,600}, {20,597}, {19,594}, {9,564}, {27,618}, {32,633}, {31,630}, {31,630}, {35,641}, {35,641}, {40,654}, {40,654}, {44,664}, {44,664}, {49,676}, {52,683}, {56,692}, {62,705}, {63,707}, {58,697}};

    // la=6 lb=6
    //  std::vector<std::vector<uint64_t>> data = {{1,40}, {1,40}, {3,55}, {2,47}, {2,47}, {3,54}, {1,39}, {1,39}, {1,39}, {1,39}, {3,52}, {2,45}, {2,45}, {3,51}, {1,38}, {1,38}, {1,38}, {1,38}, {1,38}, {2,43}, {2,43}, {3,48}, {4,53}, {1,37}, {1,37}, {1,37}, {1,37}, {1,37}, {1,37}, {2,41}, {3,45}, {1,37}, {0,33}, {4,48}, {0,33}, {2,40}, {0,33}, {3,43}, {3,43}, {2,40}, {0,34}, {0,34}, {4,45}, {1,37}, {3,42}, {3,42}, {2,40}, {0,36}, {5,46}, {6,48}, {6,48}, {6,48}, {8,51}, {8,51}, {12,56}, {12,56}, {9,53}, {12,56}, {16,59}, {19,61}, {19,61}, {24,63}, {29,0}, {30,0}};
    for (int i = 0; i < dim; i++)
    {
        inA[i] = (0 + i * 0) & mask_bwL;
        inB[i] = (init_input + i * step_size) & mask_bwL;
    }
    uint8_t *outb = new uint8_t[dim];
    uint8_t *outb_star = new uint8_t[dim];
    uint64_t comm_start = iopack->get_comm();
    auto time_start = std::chrono::high_resolution_clock::now();
    if (party == ALICE)
    {
        second_interval(inA, outb, outb_star); // step 5SSS
    }
    else
    {
        second_interval(inB, outb, outb_star);
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
    uint64_t tr = f + 3;
    uint64_t mask_bwL_sub_tr = ((bwL - tr) == 64 ? -1 : ((1ULL << (bwL - tr)) - 1));
    uint64_t mask_bwL_sub_tr_sub_1 = ((bwL - tr - 1) == 64 ? -1 : ((1ULL << (bwL - tr - 1)) - 1));

    std::cout << "mask_bwL_sub_tr_sub_1 = " << mask_bwL_sub_tr_sub_1 << std::endl;
    uint64_t mask_TR_sub_1 = (bwL - h - 1 == 64) ? ~0ULL : (1ULL << bwL - h - 1) - 1;
    std::cout << "mask_TR_sub_1 = " << mask_TR_sub_1 << std::endl;
    uint64_t *eight_bit_wrap = new uint64_t[dim];

    uint64_t TR_wrap_start = iopack->get_comm();
    // if (party == ALICE)
    // {
    //     // trunc_oracle->truncate_and_reduce_eight_bit_wrap(dim, inA, input_cut_h, eight_bit_wrap, tr, bwL); //step 7 tr
    //     trunc_oracle->truncate_and_reduce(dim, inB, input_cut_h, tr, bwL); //step 7 tr
    // }
    // else
    // {
    //     // trunc_oracle->truncate_and_reduce_eight_bit_wrap(dim, inB, input_cut_h, eight_bit_wrap, tr, bwL);
    //     trunc_oracle->truncate_and_reduce(dim, inB, input_cut_h, tr, bwL);
    // }
    uint64_t TR_wrap_end = iopack->get_comm();

    // tr 得到中间长度为s的index，进行LUT
    uint64_t *input_lower_h = new uint64_t[dim];
    uint64_t *outtrunc = new uint64_t[dim];
    uint64_t *inputinB = new uint64_t[dim];
    for (int i = 0; i < dim; i++)
    {
        inputinB[i] = inB[i] + 32768;
    }
    assign_lower_h_bits(dim, inA, inputinB, input_lower_h, h);

    for (int i = 0; i < dim; i++)
    {
        // std::cout << "input_lower_h[" << i << "] = " << input_lower_h[i] << std::endl;
    }

    trunc_oracle->truncate_and_reduce(dim, input_lower_h, outtrunc, h - s, h); // step 7 tr

    for (int i = 0; i < dim; i++)
    {
        // std::cout << "***outtrunc[" << i << "] = " << outtrunc[i] << std::endl;
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
            // std::cout << "outtrunc_a[" << i << "] = " << outtrunc_a[i] << std::endl;
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
        aux->lookup_table<uint64_t>(spec_a, nullptr, nullptr, dim, s, la); // step 8 lut
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
        aux->lookup_table<uint64_t>(spec_b, nullptr, nullptr, dim, s, lb); // step 8
    }
    else
    {                                                                        // party == BOB
        aux->lookup_table<uint64_t>(nullptr, outtrunc_a, b_bob, dim, s, lb); // b_bob是查询到的截距  重要问题，这里的outtrunc应该是两边share加起来，代码里只有Bob的outtrunc check
    }
    uint64_t two_LUT_end = iopack->get_comm();
    // uint64_t *a_alice = new uint64_t[dim];
    // uint64_t *b_alice = new uint64_t[dim];
    // for (size_t i = 0; i < dim; i++)
    // {
    //     a_alice[i] = 0;
    //     b_alice[i] = 0;
    // }
    // 做乘法，已知msb
    uint8_t *msb1 = new uint8_t[dim];
    uint8_t *msb2 = new uint8_t[dim];
    for (int i = 0; i < dim; i++)
    {
        msb1[i] = 0;
        msb2[i] = 0;
    }
    if (party == ALICE)
    {
        msb2[i] = 1;
    }
    uint64_t hadamard_product_start = iopack->get_comm();
    if (party == ALICE)
    {
        // prod->hadamard_product_MSB(dim, a_alice, inA, outax, la, bwL, la + bwL, true, true, mode, msb1, msb2);
        prod->hadamard_product(dim, a_alice, inA, outax, la, bwL, la + bwL, true, true, mode, msb1, msb2); // step 9 hadamard
    }
    else
    {
        // prod->hadamard_product_MSB(dim, a_bob, inB, outax, la, bwL, la + bwL, true, true, mode, msb1, msb2);
        prod->hadamard_product(dim, a_bob, inB, outax, la, bwL, la + bwL, true, true, mode, msb1, msb2);
    }
    uint64_t hadamard_product_end = iopack->get_comm();
    uint64_t *mid_ax = new uint64_t[dim];
    uint64_t tr_start = iopack->get_comm();

    trunc_oracle->truncate_and_reduce(dim, outax, mid_ax, la - 1, bwL + la); // step 10 tr

    // for (int i = 0; i < dim; i++)
    // {
    //     mid_ax[i] = (outax[i] >> (la - 1)) & mask_bwL;// step 10 tr
    // }

    uint64_t tr_end = iopack->get_comm();
    for (int i = 0; i < dim; i++)
    {
        // std ::cout << "outax[" << i << "] = " << outax[i] << std::endl;
        // std::cout << "mid_ax[" << i << "] = " << mid_ax[i] << std::endl;
        outax[i] = mid_ax[i];
        // std ::cout << "outax[" << i << "] = " << outax[i] << std::endl;
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
        ext->s_extend_msb(dim, b_alice, b_SExt, lb, bwL, msb_b_extend); // step 13 s_extend
    }
    else
    {
        for (int i = 0; i < dim; i++)
        {
            msb_b_extend[i] = 1;
            // std::cout << "b_bob[" << i << "] = " << b_bob[i] << std::endl;
            b_bob[i] = (b_bob[i] - 10) & mask_lb;
        }
        ext->s_extend_msb(dim, b_bob, b_SExt, lb, bwL, msb_b_extend);
    }
    uint64_t s_extend_comm_end = iopack->get_comm();
    std::cout << "\n=========STEP??  ===========" << std::endl;
    uint64_t *z = new uint64_t[dim];
    for (int i = 0; i < dim; i++)
    {
        z[i] = ((outax[i] + b_SExt[i] * static_cast<uint64_t>(std::pow(2, f - lb + 1))) & mask_bwL); // step 14
    }
    // if (party == ALICE)
    // {
    for (int i = 0; i < dim; i++)
    {
        //         std::cout << "b_alice[" << i << "] = " << b_alice[i] << std::endl;
        // std::cout << "outax[" << i << "] = " << outax[i] << std::endl;
        // std::cout << "b_SExt[" << i << "] = " << (b_SExt[i] * static_cast<uint64_t>(std::pow(2, f - lb + 1)) & mask_bwL) << std::endl;
    }
    // }
    // else
    // {
    //     for (int i = 0; i < dim; i++)
    //     {
    //         std::cout << "b_bob[" << i << "] = " << b_bob[i] << std::endl;
    //         std::cout << "outax[" << i << "] = " << outax[i] << std::endl;
    //         std::cout << "b_SExt[" << i << "] = " << (b_SExt[i] * static_cast<uint64_t>(std::pow(2, f - lb + 1)) & mask_bwL) << std::endl;
    //     }
    // }

    // 判断区间
    uint64_t *y = new uint64_t[dim];
    // uint64_t postive_part = new uint64_t[dim];
    uint64_t *non_negative1_part = new uint64_t[dim];

    uint64_t *neg1 = new uint64_t[dim];
    if (party == ALICE)
    {
        for (int i = 0; i < dim; i++)
        {
            neg1[i] = mask_bwL;
        }
    }
    else
    {
        for (int i = 0; i < dim; i++)
        {
            neg1[i] = 0;
        }
    }
    uint64_t two_select_share_start = iopack->get_comm();
    if (party == ALICE)
    {
        select_share(outb, inA, z, non_negative1_part, dim, bwL); // step 16
    }
    else
    {
        select_share(outb, inB, z, non_negative1_part, dim, bwL);
    }
    for (int i = 0; i < dim; i++)
    {

        // std::cout << "non_negative1_part[" << i << "] = " << non_negative1_part[i] << std::endl;
    }

    uint8_t *choose_negative_part = new uint8_t[dim];

    for (int i = 0; i < dim; i++)
    {
        choose_negative_part[i] = outb[i] ^ outb_star[i];
    }

    select_share(choose_negative_part, non_negative1_part, neg1, y, dim, bwL); // step 17
    uint64_t two_select_share_end = iopack->get_comm();

    uint64_t comm_end = iopack->get_comm();
    auto time_end = std::chrono::high_resolution_clock::now();
    // for (int i = 0; i < dim; i++)
    // {
    //     y[i] = (z[i] - postive_part[i]) & mask_bwL;
    // }
    for (int i = 0; i < dim; i++)
    {
        // std::cout << "outb[" << i << "] = " << static_cast<int>(outb[i]) << std::endl;
        // std::cout << "outb_star[" << i << "] = " << static_cast<int>(outb_star[i]) << std::endl;
        // std::cout << "choose_negative_part[" << i << "] = " << static_cast<int>(choose_negative_part[i]) << std::endl;
        // std::cout << "z[" << i << "] = " << z[i] << std::endl;
        // std::cout << "y[" << i << "] = " << y[i] << std::endl;
        // std::cout << "y[" << i << "] = " << y[i] / pow(2,20) << std::endl;
    }
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
        // double recv_Total_MSBytes_ALICE;
        // iopack->io->recv_data(&recv_Total_MSBytes_ALICE, sizeof(double));
        double *ULPs = new double[dim];
        double f_pow = pow(2, f);
        int s_y = 12;
        for (int i = 0; i < dim; i++)
        {
            // std::cout << "dim [" << i << "]total y = y0 + y1 =  " << ((y[i] + recv_y[i]) & mask_bwL) << ", real num: " << (double)decode_ring((y[i] + recv_y[i]) & mask_bwL, bwL) / f_pow << std::endl;
            // std::cout << "The result " << inA[i] + inB[i] << " should be calculate_ELU = " << calculate_ELU(inA[i] + inB[i], f) << std::endl;
            // ULPs[i] = abs((((double)decode_ring((y[i] + recv_y[i]) & mask_bwL, bwL) / f_pow) - calculate_ELU(inA[i] + inB[i], f)) / 0.000244140625);
            ULPs[i] = computeULPErr(((double)decode_ring((y[i] + recv_y[i]) & mask_bwL, bwL) / f_pow), calculate_ELU(inA[i] + inB[i],12), s_y);
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
        double min_val = 0.0;
        average = sum / static_cast<double>(dim);
        // std::cout << "sum: " << sum << std::endl;
        // std::cout << "static_cast<double>(dim): " << static_cast<double>(dim) << std::endl;
        max_val = *std::max_element(ULPs, ULPs + dim); // 去掉了第一个ULP
        min_val = *std::min_element(ULPs, ULPs + dim); // 去掉了第一个ULP
        std::cout << "average: " << average << std::endl;
        std::cout << "max_val: " << max_val << std::endl;
        std::cout << "min_val: " << min_val << std::endl;
        uint64_t *alice_comm = new uint64_t[1];
        iopack->io->recv_data(alice_comm, 1 * sizeof(uint64_t));

        uint64_t bob_comm = (comm_end - comm_start) / dim * 8;
        uint64_t total_comm = bob_comm + alice_comm[0];
        std::ofstream file("/home/ubuntu/EzPC/elu_output_data.csv", std::ios_base::app);

        // std::ofstream file("/home/lzq/EzPC/tanh_output_data.csv");
        if (!file.is_open())
        {
            std::cerr << "Error: Could not open file for writing." << std::endl;
            return 1;
        }

        // 写入CSV头
        // file << "la,ld,average ULP, MAX ULP , Total comm , Total time\n";
        auto total_time = chrono::duration_cast<chrono::milliseconds>(time_end - time_start).count();
        file << la << "," << lb << ",  " << average << ",   " << max_val << "  , " << total_comm << ",    " << total_time << "\n";
    }
    cout << "TR_wrap Sent: " << (TR_wrap_end - TR_wrap_start) / dim * 8 << " bits" << endl;
    cout << "Two LUT Sent: " << (two_LUT_end - two_LUT_start) / dim * 8 << " bits" << endl;
    cout << "Hadamard Product Sent: " << (hadamard_product_end - hadamard_product_start) / dim * 8 << " bits" << endl;
    cout << "S Extend Sent: " << (s_extend_comm_end - s_extend_comm_start) / dim * 8 << " bits" << endl;
    cout << "TR Sent: " << (tr_end - tr_start) / dim * 8 << " bits" << endl;
    cout << "Two Select Share Sent: " << (two_select_share_end - two_select_share_start) / dim * 8 << " bits" << endl;
    // cout << "Total Bytes Sent: " << (Comm_end - Comm_start) / dim * 8 << " bits" << endl;
    // comp_eq w
    // AND - 128 - \eplison (128 - 20)
    // LUT - 256
    // LUT - s
    // cout << "Total Bytes Sent: " << (Comm_end - Comm_start) / dim * 8 - 64 - 128 << " bits" << endl;
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(time_end - time_start).count();
    std::cout << "Time elapsed: " << duration << " ms" << std::endl;
delete[] a_alice;
delete[] b_alice;
delete[] inA;
delete[] inB;
delete[] outax;
delete[] outb;
delete[] outb_star;
delete[] input_cut_h;
delete[] comp_eq_input;
delete[] eight_bit_wrap;
delete[] input_lower_h;
delete[] outtrunc;
delete[] outtrunc1;
delete[] outtrunc_a;
delete[] a_bob;
delete[] b_bob;
delete[] msb1;
delete[] msb2;
delete[] mid_ax;
delete[] b_SExt;
delete[] msb_b_extend;
delete[] z;
delete[] non_negative1_part;
delete[] neg1;
delete[] choose_negative_part;
delete[] y;

// 二维动态数组释放（如 spec_a 和 spec_b）
if (party == ALICE) {
    for (int i = 0; i < dim; i++) {
        delete[] spec_a[i];
        delete[] spec_b[i];
    }
}
delete[] spec_a;
delete[] spec_b;

// 释放动态分配的对象
delete mill_eq;
delete trunc_oracle;
delete aux;
delete eq;
delete ext;
delete prod;

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
    // prod = new LinearOT(party, iopack, otpack);

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

        for (uint64_t s = 7; s < 8; s++)
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

    // for (uint64_t la = 10; la < 11; la++)
    // // for (uint64_t la = 6; la < 7; la++)
    // {
    //     for (uint64_t lb = 10; lb < 11; lb++)
    //     // for (uint64_t lb = 6; lb < 7; lb++)
    //     {
    //         for (uint64_t s = 7; s < 8; s++)
    //         {
    //             for (uint64_t k = 12; k < 13; k++)
    //             {
    //                 if ((la <= k) & (lb <= k))
    //                     init_test(la, lb, s, k);
    //                 // std::cout << "la=" << la << ",lb=" << lb << ",f=" << f << ",s=" << s << std::endl;
    //             }
    //         }
    //     }
    // }

    delete prod;
}