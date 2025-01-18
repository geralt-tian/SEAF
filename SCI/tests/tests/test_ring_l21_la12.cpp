/*
Authors: Deevashwer Rathee
Copyright:
Copyright (c) 2021 Microsoft Research
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

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

// 全局变量
// int dim1 = 1;
// int dim2 = 1;
// int dim3 = 1;
int bwL = 21; // 矩阵位宽
uint64_t mask_bwL = (bwL == 64 ? -1 : ((1ULL << bwL) - 1));

bool signed_B = true;           // 表示矩阵B是否为有符号数
bool accumulate = true;         // 决定是否累加结果
bool precomputed_MSBs = false;  // 决定是否预计算最高有效位
MultMode mode = MultMode::None; // 乘法模式

/////////////////////compare_with_eq
int bitlength = 64; // 假设每个数是32位
uint64_t h = 14;
// uint64_t la = 14;//la=5 f=5,la=14,f=12
uint64_t lb = 13;
uint64_t la = 13; // la=5 f=5,la=14,f=12
uint64_t f = 12;
uint64_t Tk = f-1;
uint64_t alpha = 14336;
    
uint64_t mask_l_Tk = (bwL == 64 ? -1 : ((1ULL << (bwL - Tk)) - 1));
uint64_t mask_lah1 = ((la + h + 1) == 64 ? -1 : ((1ULL << (la + h + 1)) - 1));
uint64_t mask_lla = ((la + bwL) == 64 ? -1 : ((1ULL << (la + bwL)) - 1));
uint64_t s = 7; // a=5 s=10,a=14,s=7
// uint64_t mask_bwL = (bwL == 64 ? -1 : ((1ULL << bwL) - 1));
Truncation *trunc_oracle;

//////////////////////////MUX
AuxProtocols *aux;

// double calculate_GELU(uint64_t value)
// {
//     // 定义 2^37 和 2^12 的浮点值
//     const uint64_t sign_bit_mask = 1ULL << 36; // 第 37 位的掩码
//     const double pow_2_37 = static_cast<double>(1ULL << 37);
//     const double pow_2_12 = static_cast<double>(1ULL << 12);

//     // 检查符号位（第 37 位）
//     if (value & sign_bit_mask)
//     {
//         // 如果符号位为 1，表示负数
//         value -= static_cast<uint64_t>(pow_2_37); // 减去 2^37
//     }
//     // 将值转换为浮点数
//     double x = static_cast<double>(value) / pow_2_12;
//     // 计算表达式
//     // double a = x - 4;
//     double a = x;
//     double tanh_part = std::tanh(0.7978845608 * a + 0.7978845608 * 0.044715 * std::pow(a, 3));
//     return 0.5 * a * (1 + tanh_part);
// }

double calculate_GELU(uint64_t value)
{
    //     // 定义 2^37 和 2^12 的浮点值
    const uint64_t sign_bit_mask = 1ULL << 20; // 第 37 位的掩码
    const double pow_2_21 = static_cast<double>(1ULL << 21);
    const double pow_2_12 = static_cast<double>(1ULL << 12);

    // 检查符号位（第 37 位）
    if (value & sign_bit_mask)
    {
        // 如果符号位为 1，表示负数
        value -= static_cast<uint64_t>(pow_2_21); // 减去 2^37
    }
    // 将值转换为浮点数
    double x = static_cast<double>(value) / pow_2_12;
    return 0.5 * x + 0.5 * x * std::erf(x / 1.414);
}

uint64_t decode_ring(uint64_t input, uint64_t bw)
{
    // 从环上解码值
    uint64_t mask = (bw == 64) ? ~0ULL : (1ULL << bw) - 1;
    uint64_t half = 1ULL << (bw - 1);

    // std::cout << "input = " << input << std::endl;
    // std::cout << "half = " << half << std::endl;
    if (input < half)
    {
        return input;
    }
    else
    {
        return (1ULL << (bw)) - input;
    }
}

// void assign_lower_h_bits(int32_t dim1, int32_t dim2, int32_t dim3, uint64_t *inA, uint64_t *inB, uint64_t *inA_, uint64_t *inB_, int32_t h)
// {
//     // Create a mask that has the lowest h bits set to 1
//     uint64_t mask = (h == 64) ? ~0ULL : (1ULL << h) - 1;

//     // Assign the lower h bits from inA to inA_
//     for (int i = 0; i < dim1; i++)
//     {
//         for (int j = 0; j < dim2; j++)
//         {
//             inA_[i * dim2 + j] = inA[i * dim2 + j] & mask;
//         }
//     }

//     // Assign the lower h bits from inB to inB_
//     for (int i = 0; i < dim2; i++)
//     {
//         for (int j = 0; j < dim3; j++)
//         {
//             inB_[i * dim3 + j] = inB[i * dim3 + j] & mask;
//         }
//     }
// }

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

//////////////////////
// 初始化
///////////////////////////////

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
    otpack = new OTPack(iopack, party);

    uint64_t comm_start = iopack->get_comm();
    auto time_start = chrono::high_resolution_clock::now();

    prod = new LinearOT(party, iopack, otpack);

    PRG128 prg; //(fix_key);
    int dim = 5120;
    uint64_t *inA = new uint64_t[dim]; // 1*100
    uint64_t *inB = new uint64_t[dim]; // 100*35

    uint64_t *outax = new uint64_t[dim];
    for (int i = 0; i < dim; i++)
    {
        inA[i] = 000 + i * 2;
        inB[i] = 000 + i * 2;
    }
    std::cout << "input inA[" << 0 << "] = " << inA[0] << std::endl;
    std::cout << "input inB[" << 0 << "] = " << inB[0] << std::endl;
    uint64_t *inA_h = new uint64_t[dim]; // 1*100
    uint64_t *inB_h = new uint64_t[dim]; // 100*35

    std::cout << "\n=========STEP4 use DRelu to learn [[b]]^B===========" << std::endl;

    uint8_t *Drelu = new uint8_t[dim];
    uint8_t *msbA = new uint8_t[dim];
    uint8_t *msbB = new uint8_t[dim];

    // Drelu = MSB , Alice ^1
    if (party == ALICE)
    {
        prod->aux->MSB(inA, msbA, dim, bwL);
    }
    else
    {
        prod->aux->MSB(inB, msbB, dim, bwL);
    }

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
    std::cout << "Drelu[" << 0 << "] = " << static_cast<int>(Drelu[0]) << std::endl;

    std::cout << "\n=========STEP5 use EMUX to learn [[|x|]]in L ring===========" << std::endl;
    aux = new AuxProtocols(party, iopack, otpack);
    uint64_t *EMUX_output_x = new uint64_t[dim];
    std::cout << "inA[" << 0 << "] = " << inA[0] << std::endl;
    std::cout << "inB[" << 0 << "] = " << inB[0] << std::endl;
    if (party == ALICE)
    {
        aux->multiplexerabs(Drelu, inA, EMUX_output_x, dim, bwL, bwL);
    }
    else
    {
        aux->multiplexerabs(Drelu, inB, EMUX_output_x, dim, bwL, bwL);
    }
    std::cout << "EMUX_output_x[" << 0 << "] = " << EMUX_output_x[0] << std::endl; // 目前的输出是16383+2**37

    std::cout << "\n=========STEP6 extract the lower h=14 bits===========" << std::endl;
    assign_lower_h_bits(dim, inA, inB, inA_h, inB_h, h);

    std::cout << "inA_h[" << 0 << "] = " << inA_h[0] << std::endl;
    std::cout << "inB_h[" << 0 << "] = " << inB_h[0] << std::endl;

    // step6 check

    std::cout << "\n=========STEP7 reduce and get [[x]]===========" << std::endl;
    uint64_t comm_start_tr = iopack->get_comm();
    trunc_oracle = new Truncation(party, iopack, otpack);
    uint64_t *outtrunc = new uint64_t[dim];
    if (party == sci::ALICE)
    {
        trunc_oracle->truncate_and_reduce(dim, inA_h, outtrunc, h - s, h); // shift=h-s,hypothesis s=7  truncate就是为了分组，截断后7位，为了前8位可以映射到对应的table
                                                                           // std::cout << "outtrunc[" << 0 << "] = " << outtrunc[0] << std::endl;
    }
    else
    {
        trunc_oracle->truncate_and_reduce(dim, inB_h, outtrunc, h - s, h); // shift=h-s,hypothesis s=7,outtrunc是0-127
                                                                           // std::cout << "outtrunc[" << 0 << "] = " << outtrunc[0] << std::endl; // outtrunc是<i>，范围是0-127
    }

    std::cout << std::dec << "outtrunc = " << outtrunc[0] << std::endl;
    uint64_t comm_end_tr = iopack->get_comm();
    std::cout << "TR Bytes Sent: " << (comm_end_tr - comm_start_tr) << "bytes" << std::endl;
    // step7 check
    uint64_t comm_start_lut = iopack->get_comm();

    std::cout << "\n=========STEP8 LookUp Table   ===========" << std::endl;
    // 重跑一个有256个的
    // std::vector<std::vector<uint64_t>> data = {{137438953471, 0}, {137438953470, 0}, {137438953470, 0}, {137438953470, 0}, {137438953470, 0}, {137438953469, 0}, {137438953469, 0}, {137438953468, 0}, {137438953468, 0}, {137438953467, 0}, {137438953467, 0}, {137438953466, 1}, {137438953465, 1}, {137438953465, 1}, {137438953464, 2}, {137438953463, 2}, {137438953462, 3}, {137438953460, 3}, {137438953459, 4}, {137438953458, 5}, {137438953456, 6}, {137438953454, 7}, {137438953453, 8}, {137438953450, 10}, {137438953448, 11}, {137438953446, 13}, {137438953443, 15}, {137438953440, 18}, {137438953437, 20}, {137438953434, 23}, {137438953431, 27}, {137438953427, 30}, {137438953423, 34}, {137438953418, 39}, {137438953414, 44}, {137438953409, 49}, {137438953404, 55}, {137438953398, 61}, {137438953392, 68}, {137438953386, 76}, {137438953379, 84}, {137438953372, 93}, {137438953365, 103}, {137438953357, 114}, {137438953349, 125}, {137438953340, 137}, {137438953331, 150}, {137438953322, 163}, {137438953312, 178}, {137438953302, 193}, {137438953292, 210}, {137438953281, 227}, {137438953269, 245}, {137438953258, 265}, {137438953246, 285}, {137438953234, 306}, {137438953221, 328}, {137438953208, 351}, {137438953195, 375}, {137438953182, 399}, {137438953168, 424}, {137438953155, 450}, {137438953141, 477}, {137438953127, 504}, {137438953113, 532}, {137438953099, 560}, {137438953086, 588}, {137438953072, 616}, {137438953059, 644}, {137438953046, 672}, {137438953033, 700}, {137438953021, 727}, {137438953010, 753}, {137438952999, 778}, {137438952988, 801}, {137438952979, 823}, {137438952970, 844}, {137438952963, 862}, {137438952956, 878}, {137438952951, 891}, {137438952947, 900}, {137438952945, 907}, {137438952944, 910}, {137438952944, 908}, {137438952947, 902}, {137438952951, 891}, {137438952957, 874}, {137438952965, 852}, {137438952975, 824}, {137438952988, 789}, {137438953003, 747}, {137438953020, 697}, {137438953040, 640}, {137438953063, 574}, {137438953088, 500}, {137438953116, 416}, {137438953147, 324}, {137438953181, 221}, {137438953218, 108}, {137438953258, 137438953457}, {137438953301, 137438953323}, {137438953347, 137438953178}, {137438953396, 137438953021}, {137438953448, 137438952853}, {31, 137438952673}, {90, 137438952482}, {151, 137438952278}, {216, 137438952062}, {283, 137438951834}, {354, 137438951594}, {427, 137438951342}, {503, 137438951078}, {582, 137438950803}, {663, 137438950515}, {747, 137438950216}, {834, 137438949906}, {922, 137438949586}, {1013, 137438949255}, {1105, 137438948913}, {1199, 137438948563}, {1295, 137438948203}, {1393, 137438947835}, {1491, 137438947459}, {1591, 137438947076}, {1692, 137438946686}, {1793, 137438946290}, {1895, 137438945889}, {1997, 137438945484}, {2099, 137438945075}, {2201, 137438944664}, {2303, 137438944250}, {2404, 137438943836}, {2505, 137438943421}, {2605, 137438943006}, {2703, 137438942593}, {2801, 137438942182}, {2897, 137438941775}, {2991, 137438941371}, {3083, 137438940972}, {3174, 137438940578}, {3262, 137438940191}, {3349, 137438939811}, {3433, 137438939439}, {3514, 137438939075}, {3593, 137438938720}, {3669, 137438938376}, {3742, 137438938041}, {3813, 137438937717}, {3880, 137438937405}, {3945, 137438937105}, {4006, 137438936816}, {4065, 137438936541}, {4120, 137438936278}, {4172, 137438936028}, {4221, 137438935792}, {4267, 137438935569}, {4310, 137438935360}, {4350, 137438935164}, {4387, 137438934982}, {4421, 137438934814}, {4452, 137438934659}, {4480, 137438934518}, {4505, 137438934390}, {4528, 137438934275}, {4548, 137438934172}, {4565, 137438934083}, {4580, 137438934005}, {4593, 137438933939}, {4603, 137438933885}, {4611, 137438933842}, {4617, 137438933809}, {4621, 137438933787}, {4624, 137438933774}, {4624, 137438933771}, {4623, 137438933777}, {4621, 137438933791}, {4617, 137438933812}, {4612, 137438933841}, {4605, 137438933877}, {4598, 137438933920}, {4589, 137438933968}, {4580, 137438934021}, {4569, 137438934080}, {4558, 137438934142}, {4547, 137438934209}, {4535, 137438934279}, {4522, 137438934352}, {4509, 137438934428}, {4496, 137438934506}, {4482, 137438934585}, {4469, 137438934667}, {4455, 137438934749}, {4441, 137438934832}, {4427, 137438934915}, {4413, 137438934998}, {4400, 137438935082}, {4386, 137438935164}, {4373, 137438935246}, {4360, 137438935328}, {4347, 137438935408}, {4334, 137438935486}, {4322, 137438935563}, {4310, 137438935639}, {4299, 137438935713}, {4287, 137438935785}, {4276, 137438935854}, {4266, 137438935922}, {4256, 137438935988}, {4246, 137438936051}, {4237, 137438936112}, {4228, 137438936171}, {4219, 137438936227}, {4211, 137438936281}, {4203, 137438936333}, {4196, 137438936383}, {4189, 137438936430}, {4182, 137438936475}, {4176, 137438936517}, {4170, 137438936558}, {4164, 137438936596}, {4159, 137438936632}, {4154, 137438936666}, {4150, 137438936698}, {4145, 137438936729}, {4141, 137438936757}, {4137, 137438936784}, {4134, 137438936808}, {4131, 137438936832}, {4128, 137438936853}, {4125, 137438936873}, {4122, 137438936892}, {4120, 137438936909}, {4118, 137438936925}, {4115, 137438936940}, {4114, 137438936954}, {4112, 137438936967}, {4110, 137438936978}, {4109, 137438936989}, {4108, 137438936999}, {4106, 137438937008}, {4105, 137438937016}, {4104, 137438937023}, {4103, 137438937030}, {4103, 137438937036}, {4102, 137438937042}, {4101, 137438937047}, {4101, 137438937051}, {4100, 137438937055}, {4100, 137438937059}, {4099, 137438937062}, {4099, 137438937065}, {4098, 137438937068}, {4098, 137438937070}, {4098, 137438937073}, {4098, 137438937074}, {4097, 137438937076}};
    // 128:

    // la=14 lb=14
    //  std::vector<std::vector<uint64_t>> data = {{0, 0}, {204, 16380}, {407, 16367}, {610, 16345}, {812, 16314}, {1013, 16275}, {1211, 16228}, {1407, 16172}, {1600, 16109}, {1790, 16038}, {1977, 15959}, {2160, 15874}, {2339, 15782}, {2514, 15684}, {2684, 15579}, {2850, 15470}, {3010, 15355}, {3165, 15236}, {3315, 15112}, {3459, 14985}, {3597, 14855}, {3729, 14722}, {3855, 14587}, {3975, 14450}, {4089, 14312}, {4197, 14173}, {4298, 14033}, {4394, 13894}, {4483, 13755}, {4566, 13617}, {4642, 13481}, {4713, 13346}, {4778, 13213}, {4837, 13083}, {4890, 12955}, {4938, 12830}, {4981, 12709}, {5018, 12590}, {5050, 12476}, {5078, 12365}, {5100, 12258}, {5119, 12155}, {5133, 12056}, {5143, 11962}, {5149, 11871}, {5151, 11785}, {5151, 11704}, {5147, 11626}, {5140, 11553}, {5130, 11483}, {5118, 11418}, {5104, 11356}, {5087, 11299}, {5069, 11245}, {5049, 11194}, {5027, 11147}, {5004, 11102}, {4980, 11061}, {4955, 11023}, {4929, 10987}, {4903, 10953}, {4876, 10922}, {4849, 10892}, {4821, 10864}, {4794, 10838}, {4766, 10813}, {4739, 10789}, {4711, 10767}, {4684, 10745}, {4658, 10723}, {4631, 10702}, {4606, 10682}, {4580, 10661}, {4556, 10641}, {4532, 10620}, {4509, 10599}, {4486, 10577}, {4464, 10555}, {4443, 10532}, {4423, 10509}, {4404, 10484}, {4385, 10459}, {4367, 10433}, {4350, 10406}, {4334, 10377}, {4318, 10348}, {4303, 10317}, {4289, 10286}, {4276, 10253}, {4263, 10218}, {4251, 10183}, {4240, 10146}, {4230, 10109}, {4220, 10070}, {4210, 10030}, {4202, 9988}, {4193, 9946}, {4186, 9902}, {4179, 9857}, {4172, 9812}, {4166, 9765}, {4160, 9717}, {4154, 9668}, {4149, 9619}, {4145, 9568}, {4141, 9516}, {4137, 9464}, {4133, 9411}, {4130, 9357}, {4126, 9303}, {4124, 9248}, {4121, 9192}, {4119, 9135}, {4116, 9079}, {4114, 9021}, {4113, 8963}, {4111, 8905}, {4109, 8846}, {4108, 8786}, {4107, 8727}, {4106, 8667}, {4105, 8606}, {4104, 8545}, {4103, 8484}, {4102, 8423}, {4101, 8362}, {4101, 8300}, {4100, 8238}};
    // la=13 lb=13
    // std::vector<std::vector<uint64_t>> data = {{38,0}, {144,8189}, {257,8182}, {353,8173}, {458,8160}, {554,8145}, {655,8126}, {747,8106}, {851,8080}, {940,8055}, {1036,8025}, {1129,7993}, {1214,7961}, {1305,7924}, {1387,7888}, {1470,7849}, {1540,7814}, {1621,7771}, {1694,7730}, {1763,7689}, {1835,7644}, {1893,7606}, {1957,7562}, {2021,7516}, {2069,7480}, {2124,7437}, {2177,7394}, {2223,7355}, {2263,7320}, {2306,7281}, {2339,7250}, {2370,7220}, {2408,7182}, {2437,7152}, {2453,7135}, {2485,7100}, {2500,7083}, {2514,7067}, {2535,7042}, {2544,7031}, {2556,7016}, {2560,7011}, {2570,6998}, {2573,6994}, {2575,6991}, {2576,6990}, {2575,6991}, {2572,6996}, {2568,7002}, {2562,7011}, {2556,7020}, {2548,7033}, {2539,7047}, {2530,7062}, {2519,7081}, {2508,7100}, {2496,7121}, {2484,7142}, {2471,7166}, {2458,7190}, {2445,7214}, {2431,7241}, {2418,7266}, {2404,7294}, {2390,7322}, {2376,7350}, {2363,7377}, {2349,7406}, {2336,7434}, {2322,7464}, {2309,7492}, {2297,7519}, {2284,7548}, {2272,7576}, {2260,7604}, {2249,7629}, {2238,7655}, {2227,7682}, {2217,7706}, {2207,7731}, {2197,7756}, {2188,7779}, {2179,7802}, {2171,7823}, {2163,7844}, {2156,7862}, {2148,7884}, {2142,7900}, {2135,7919}, {2129,7936}, {2123,7953}, {2118,7967}, {2113,7981}, {2108,7996}, {2103,8011}, {2099,8022}, {2095,8034}, {2091,8047}, {2088,8056}, {2085,8065}, {2082,8074}, {2079,8084}, {2076,8093}, {2074,8100}, {2072,8106}, {2070,8113}, {2068,8119}, {2066,8126}, {2064,8133}, {2063,8136}, {2061,8143}, {2060,8147}, {2059,8150}, {2058,8154}, {2057,8157}, {2056,8161}, {2055,8164}, {2055,8164}, {2054,8168}, {2053,8172}, {2053,8172}, {2052,8176}, {2052,8176}, {2051,8180}, {2051,8180}, {2051,8180}, {2050,8183}, {2050,8183}};

    // la=5
    //  std::vector<std::vector<uint64_t>> data = {{0, 0}, {12, 28}, {23, 15}, {2, 25}, {12, 26}, {21, 19}, {27, 4}, {31, 12}, {0, 13}, {30, 6}, {25, 23}, {16, 2}, {3, 6}, {18, 4}, {28, 27}, {2, 14}, {2, 27}, {29, 4}, {19, 8}, {3, 9}, {13, 7}, {17, 2}, {15, 27}, {7, 18}, {25, 8}, {5, 29}, {10, 17}, {10, 6}, {3, 27}, {22, 17}, {2, 9}, {9, 2}, {10, 29}, {5, 27}, {26, 27}, {10, 30}, {21, 5}, {26, 14}, {26, 28}, {22, 13}, {12, 2}, {31, 27}, {13, 24}, {23, 26}, {29, 31}, {31, 9}, {31, 24}, {27, 10}, {20, 1}, {10, 27}, {30, 26}, {16, 28}, {31, 3}, {13, 13}, {25, 26}, {3, 11}, {12, 30}, {20, 21}, {27, 15}, {1, 11}, {7, 9}, {12, 10}, {17, 12}, {21, 16}, {26, 22}, {30, 29}, {3, 5}, {7, 15}, {12, 25}, {18, 3}, {23, 14}, {30, 26}, {4, 5}, {12, 17}, {20, 28}, {29, 7}, {6, 17}, {16, 27}, {27, 4}, {7, 13}, {20, 20}, {1, 27}, {15, 1}, {30, 6}, {14, 9}, {30, 12}, {15, 13}, {1, 14}, {20, 13}, {7, 10}, {27, 7}, {16, 2}, {6, 29}, {28, 22}, {18, 14}, {10, 4}, {1, 26}, {26, 14}, {19, 1}, {12, 20}, {6, 5}, {0, 21}, {26, 4}, {21, 19}, {17, 0}, {13, 12}, {9, 24}, {5, 3}, {2, 13}, {30, 23}, {28, 0}, {25, 8}, {23, 15}, {20, 23}, {18, 29}, {17, 3}, {15, 9}, {13, 14}, {12, 18}, {11, 23}, {10, 27}, {9, 30}, {8, 1}, {7, 4}, {6, 7}, {5, 10}, {5, 12}, {4, 14}};
    // la=6  lb=13
    //  std::vector<std::vector<uint64_t>> data = {{0,0}, {1,16382}, {2,16374}, {3,16362}, {4,16344}, {4,16344}, {5,16321}, {6,16293}, {7,16260}, {7,16260}, {8,16221}, {9,16177}, {9,16177}, {10,16127}, {11,16070}, {11,16071}, {12,16008}, {13,15939}, {13,15939}, {14,15863}, {14,15864}, {15,15780}, {15,15780}, {16,15688}, {16,15688}, {17,15588}, {17,15587}, {17,15587}, {18,15475}, {18,15475}, {18,15475}, {19,15351}, {19,15350}, {19,15349}, {19,15350}, {19,15351}, {20,15207}, {20,15205}, {20,15204}, {20,15203}, {20,15203}, {20,15203}, {20,15203}, {20,15204}, {20,15204}, {20,15204}, {20,15205}, {20,15205}, {20,15206}, {20,15206}, {20,15206}, {20,15206}, {20,15205}, {20,15204}, {20,15203}, {20,15202}, {20,15200}, {19,15428}, {19,15429}, {19,15430}, {19,15431}, {19,15431}, {19,15431}, {19,15430}, {19,15429}, {19,15427}, {18,15692}, {18,15693}, {18,15694}, {18,15695}, {18,15696}, {18,15695}, {18,15695}, {18,15694}, {18,15693}, {18,15692}, {17,15996}, {17,15997}, {17,15999}, {17,16000}, {17,16001}, {17,16001}, {17,16002}, {17,16002}, {17,16001}, {17,16001}, {17,16000}, {17,15999}, {17,15998}, {17,15996}, {17,15995}, {17,15993}, {17,15991}, {16,16363}, {16,16365}, {16,16367}, {16,16368}, {16,16370}, {16,16371}, {16,16372}, {16,16373}, {16,16374}, {16,16375}, {16,16376}, {16,16377}, {16,16377}, {16,16378}, {16,16379}, {16,16379}, {16,16380}, {16,16380}, {16,16380}, {16,16381}, {16,16381}, {16,16381}, {16,16382}, {16,16382}, {16,16382}, {16,16382}, {16,16383}, {16,16383}, {16,16383}, {16,16383}, {16,16383}, {16,16383}, {16,16383}, {16,16383}, {16,16383}};
    // la=6 lb=10
    // std::vector<std::vector<uint64_t>> data = {{0, 0}, {1, 0}, {2, 1023}, {3, 1021}, {4, 1019}, {4, 1019}, {5, 1016}, {6, 1013}, {7, 1009}, {7, 1009}, {8, 1004}, {9, 998}, {9, 998}, {10, 992}, {11, 985}, {11, 985}, {12, 977}, {13, 968}, {13, 968}, {14, 959}, {14, 959}, {15, 948}, {15, 949}, {16, 937}, {16, 937}, {17, 924}, {17, 924}, {17, 924}, {18, 910}, {18, 910}, {18, 910}, {19, 895}, {19, 895}, {19, 895}, {19, 895}, {19, 895}, {20, 877}, {20, 877}, {20, 876}, {20, 876}, {20, 876}, {20, 876}, {20, 876}, {20, 876}, {20, 876}, {20, 877}, {20, 877}, {20, 877}, {20, 877}, {20, 877}, {20, 877}, {20, 877}, {20, 877}, {20, 877}, {20, 876}, {20, 876}, {20, 876}, {19, 904}, {19, 905}, {19, 905}, {19, 905}, {19, 905}, {19, 905}, {19, 905}, {19, 905}, {19, 904}, {18, 937}, {18, 938}, {18, 938}, {18, 938}, {18, 938}, {18, 938}, {18, 938}, {18, 938}, {18, 938}, {18, 937}, {17, 975}, {17, 976}, {17, 976}, {17, 976}, {17, 976}, {17, 976}, {17, 976}, {17, 976}, {17, 976}, {17, 976}, {17, 976}, {17, 976}, {17, 976}, {17, 976}, {17, 975}, {17, 975}, {17, 975}, {16, 1021}, {16, 1022}, {16, 1022}, {16, 1022}, {16, 1022}, {16, 1022}, {16, 1023}, {16, 1023}, {16, 1023}, {16, 1023}, {16, 1023}, {16, 1023}, {16, 1023}, {16, 1023}, {16, 1023}, {16, 1023}, {16, 1023}, {16, 0}, {16, 0}, {16, 0}, {16, 0}, {16, 0}, {16, 0}, {16, 0}, {16, 0}, {16, 0}, {16, 0}, {16, 0}, {16, 0}, {16, 0}, {16, 0}, {16, 0}, {16, 0}, {16, 0}, {16, 0}};
    
    
    
    //la=13 lb=13 avgULP
    // std::vector<std::vector<uint64_t>> data = {{36,0}, {143,8189}, {257,8182}, {353,8173}, {457,8160}, {559,8144}, {655,8126}, {751,8105}, {851,8080}, {940,8055}, {1036,8025}, {1126,7994}, {1214,7961}, {1305,7924}, {1387,7888}, {1472,7848}, {1538,7815}, {1621,7771}, {1699,7727}, {1763,7689}, {1835,7644}, {1893,7606}, {1957,7562}, {2010,7524}, {2065,7483}, {2124,7437}, {2166,7403}, {2223,7355}, {2263,7320}, {2307,7280}, {2340,7249}, {2371,7219}, {2407,7183}, {2436,7153}, {2453,7135}, {2485,7100}, {2500,7083}, {2525,7054}, {2535,7042}, {2544,7031}, {2556,7016}, {2560,7011}, {2570,6998}, {2573,6994}, {2575,6991}, {2576,6990}, {2575,6991}, {2572,6996}, {2568,7002}, {2562,7011}, {2556,7020}, {2548,7033}, {2539,7047}, {2530,7062}, {2519,7081}, {2508,7100}, {2496,7121}, {2484,7142}, {2471,7166}, {2458,7190}, {2445,7214}, {2431,7241}, {2418,7266}, {2404,7294}, {2390,7322}, {2376,7350}, {2363,7377}, {2349,7406}, {2336,7434}, {2322,7464}, {2309,7492}, {2297,7519}, {2284,7548}, {2272,7576}, {2260,7604}, {2249,7629}, {2238,7655}, {2227,7682}, {2217,7706}, {2207,7731}, {2197,7756}, {2188,7779}, {2179,7802}, {2171,7823}, {2163,7844}, {2156,7862}, {2148,7884}, {2142,7900}, {2135,7919}, {2129,7936}, {2123,7953}, {2118,7967}, {2113,7981}, {2108,7996}, {2103,8011}, {2099,8022}, {2095,8034}, {2091,8047}, {2088,8056}, {2085,8065}, {2082,8074}, {2079,8084}, {2076,8093}, {2074,8100}, {2072,8106}, {2070,8113}, {2068,8119}, {2066,8126}, {2064,8133}, {2063,8136}, {2061,8143}, {2060,8147}, {2059,8150}, {2058,8154}, {2057,8157}, {2056,8161}, {2055,8164}, {2055,8164}, {2054,8168}, {2053,8172}, {2053,8172}, {2052,8176}, {2052,8176}, {2051,8180}, {2051,8180}, {2051,8180}, {2050,8183}, {2050,8183}};
    
    //la=13 lb=13 maxULP
    std::vector<std::vector<uint64_t>> data = {{43,0}, {146,8189}, {258,8182}, {354,8173}, {458,8160}, {560,8144}, {646,8128}, {756,8104}, {844,8082}, {947,8053}, {1033,8026}, {1132,7992}, {1217,7960}, {1298,7927}, {1385,7889}, {1466,7851}, {1544,7812}, {1623,7770}, {1694,7730}, {1768,7686}, {1832,7646}, {1893,7606}, {1960,7560}, {2017,7519}, {2069,7480}, {2124,7437}, {2177,7394}, {2215,7362}, {2263,7320}, {2306,7281}, {2339,7250}, {2365,7225}, {2405,7185}, {2438,7151}, {2453,7135}, {2476,7110}, {2500,7083}, {2514,7067}, {2535,7042}, {2544,7031}, {2556,7016}, {2560,7011}, {2570,6998}, {2573,6994}, {2575,6991}, {2576,6990}, {2575,6991}, {2572,6996}, {2568,7002}, {2562,7011}, {2556,7020}, {2548,7033}, {2539,7047}, {2530,7062}, {2519,7081}, {2508,7100}, {2496,7121}, {2484,7142}, {2471,7166}, {2458,7190}, {2445,7214}, {2431,7241}, {2418,7266}, {2404,7294}, {2390,7322}, {2376,7350}, {2363,7377}, {2349,7406}, {2336,7434}, {2322,7464}, {2309,7492}, {2297,7519}, {2284,7548}, {2272,7576}, {2260,7604}, {2249,7629}, {2238,7655}, {2227,7682}, {2217,7706}, {2207,7731}, {2197,7756}, {2188,7779}, {2179,7802}, {2171,7823}, {2163,7844}, {2156,7862}, {2148,7884}, {2142,7900}, {2135,7919}, {2129,7936}, {2123,7953}, {2118,7967}, {2113,7981}, {2108,7996}, {2103,8011}, {2099,8022}, {2095,8034}, {2091,8047}, {2088,8056}, {2085,8065}, {2082,8074}, {2079,8084}, {2076,8093}, {2074,8100}, {2072,8106}, {2070,8113}, {2068,8119}, {2066,8126}, {2064,8133}, {2063,8136}, {2061,8143}, {2060,8147}, {2059,8150}, {2058,8154}, {2057,8157}, {2056,8161}, {2055,8164}, {2055,8164}, {2054,8168}, {2053,8172}, {2053,8172}, {2052,8176}, {2052,8176}, {2051,8180}, {2051,8180}, {2051,8180}, {2050,8183}, {2050,8183}};
    uint64_t *a_alice = new uint64_t[dim];
    uint64_t *b_alice = new uint64_t[dim];

    for (size_t i = 0; i < dim; i++)
    {
        a_alice[i] = 0;
        b_alice[i] = 0;
    }

    uint64_t **spec_a = new uint64_t *[dim];
    uint64_t *a_bob = new uint64_t[dim];
    uint64_t N = 1ULL << 7;

    for (int i = 0; i < dim; i++)
    {
        spec_a[i] = new uint64_t[N];
        for (int j = 0; j < N; j++)
        {
            spec_a[i][j] = data[j][0];
            // std::cout << "i = " << i << ", j = " << j << ", data = " << data[j][i] << std::endl;
        }
    }

    // for (int i = 0; i < dim; ++i) {
    //     std::cout << "spec[" << i << "]: ";
    //     for (int j = 0; j < N; ++j) {
    //         std::cout << spec[i][j] << " "; // Output the value in decimal format
    //     }
    //     std::cout << std::endl; // Move to the next line after each row
    // }
    uint64_t *outtrunc1 = new uint64_t[dim];
    for (size_t i = 0; i < dim; i++)
    {
        outtrunc1[0] = 0;
    }

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
            outtrunc_a[i] = (outtrunc[i] + outtrunc1[i]) & ((1ULL << 7) - 1);
        }

        for (int i = 0; i < dim; i++)
        {
            std::cout << "outtrunc[" << i << "] = " << outtrunc[i] << std::endl;
        }
        // std::cout << "outtrunc_a[" << 0 << "] = " << outtrunc_a[0] << std::endl;
    }
    if (party == ALICE)
    {
        aux->lookup_table<uint64_t>(spec_a, nullptr, nullptr, dim, 7, la); // bw_xlut是outtrunc的位宽
    }
    else
    {                                                                        // party == BOB
        aux->lookup_table<uint64_t>(nullptr, outtrunc_a, a_bob, dim, 7, la); // a_bob是查询到的斜率
    }
    if (party != ALICE)
        for (int i = 0; i < dim; i++)
        {
            std::cout << "a_bob[" << i << "] = " << a_bob[i] << std::endl;
        }
    /////选择截距
    uint64_t **spec_b = new uint64_t *[dim];
    uint64_t *b_bob = new uint64_t[dim];

    for (int i = 0; i < dim; i++)
    {
        spec_b[i] = new uint64_t[N];
        for (int j = 0; j < N; j++)
        {
            spec_b[i][j] = data[j][1];
            // std::cout << "i = " << i << ", j = " << j << ", data = " << data[j][i+1] << std::endl;
        }
    }
    // for (int i = 0; i < dim; ++i) {
    //     std::cout << "spec[" << i << "]: ";
    //     for (int j = 0; j < N; ++j) {
    //         std::cout << spec[i][j] << " "; // Output the value in decimal format
    //     }
    //     std::cout << std::endl; // Move to the next line after each row
    // }
    // for (int i; i < dim; i++)
    // {
    //     outtrunc[i] = outtrunc[i] - 16384;
    // }
    if (party == ALICE)
    {
        aux->lookup_table<uint64_t>(spec_b, nullptr, nullptr, dim, 7, lb);
    }
    else
    {                                                                        // party == BOB
        aux->lookup_table<uint64_t>(nullptr, outtrunc_a, b_bob, dim, 7, lb); // b_bob是查询到的截距  重要问题，这里的outtrunc应该是两边share加起来，代码里只有Bob的outtrunc check
    }
    if (party != ALICE)
        std::cout << "b_bob[" << 0 << "] = " << b_bob[0] << std::endl;

    uint64_t comm_end_lut = iopack->get_comm();
    cout << "LUT Bytes Sent: " << (comm_end_lut - comm_start_lut) << "bytes" << endl;
    //////////////////////step8
    std::cout << "\n=========STEP9 EMUX a  ===========" << std::endl;
    uint64_t *EMUX_output_a = new uint64_t[dim];
    std::cout << "inA[" << 0 << "] = " << inA[0] << std::endl;
    std::cout << "inB[" << 0 << "] = " << inB[0] << std::endl;
    if (party == ALICE)
    {
        aux->multiplexerabs(Drelu, a_alice, EMUX_output_a, dim, la, la);
    }
    else
    {
        aux->multiplexerabs(Drelu, a_bob, EMUX_output_a, dim, la, la);
    }
    for (int i = 0; i < dim; i++)
    {
        std::cout << "EMUX_output_a[" << i << "] = " << EMUX_output_a[i] << std::endl;
    }
    // std::cout << "EMUX_output_a[" << 0 << "] = " << EMUX_output_a[0] << std::endl; // 目前的输出是16383+2**37

    ext = new XTProtocol(party, iopack, otpack);
    uint64_t *zext_h = new uint64_t[dim];
    if (party == ALICE)
    {
        ext->z_extend(dim, inA_h, zext_h, h, h + 1, nullptr);
    }
    else
    {
        ext->z_extend(dim, inB_h, zext_h, h, h + 1, nullptr);
    }
    std::cout << "zext_h[" << 0 << "] = " << zext_h[0] << std::endl;

    std::cout << "\n=========STEP10 multiplication ===========" << std::endl;
    uint64_t comm_start_mult = iopack->get_comm();

    //   test_matrix_multiplication(inA, inB, outC, false);
    // test_matrix_multiplication(inA, inB, outC, true);

    if (party == ALICE)
    {
        std::cout << "inA_h[" << 0 << "] = " << inA_h[0] << std::endl;
        std::cout << "a_alice[" << 0 << "] = " << a_alice[0] << std::endl;

        prod->hadamard_product(dim, EMUX_output_a, EMUX_output_x, outax, la, bwL, la + bwL,
                               true, true, mode,
                               0, 0);
    }
    else
    {
        std::cout << "inB_h[" << 0 << "] = " << inB_h[0] << std::endl;
        std::cout << "a_bob[" << 0 << "] = " << a_bob[0] << std::endl;
        prod->hadamard_product(dim, EMUX_output_a, EMUX_output_x, outax, la, bwL, la + bwL,
                               true, true, mode,
                               0, 0);
    }
    /////////////////////////
    if (party == ALICE)
    {
        iopack->io->send_data(outax, dim * sizeof(uint64_t));
    }
    else
    {
        uint64_t *recv_outax = new uint64_t[dim];
        iopack->io->recv_data(recv_outax, dim * sizeof(uint64_t));

        for (int i = 0; i < dim; i++)
        {
            std::cout << "bwL =  " << bwL << std::endl;
            std::cout << "mask_lla =  " << mask_lla << std::endl;
            std::cout << "outax[" << i << "] =  " << (outax[i] & mask_lla) << std::endl;
            std::cout << "recv_outax[" << i << "] =  " << (recv_outax[i] & mask_lla) << std::endl;
            std::cout << "total outax[" << i << "] =  " << ((outax[i] + recv_outax[i]) & mask_lla) << std::endl;
        }
        // std::cout << "total outax =  " << ((outax[0] + recv_outax[0]) & mask_lah1) << std::endl;
    }
    // std::cout << "mask_lah_f = " << mask_lah_f << std::endl;
    /////////////////////////新增截断
    for (int i = 0; i < dim; i++)
    {
        // outax[i] = outax[i] & mask_bwL;
        // std::cout << "trunc outax[" << i << "] = " << outax[i] << std::endl;
        outax[i] = (outax[i] >> (la - 1)) & mask_bwL; //
        // std::cout << "trunc outax mask_16[" << i << "] = " << outax[i] << std::endl;
    }

    uint64_t comm_end_mult = iopack->get_comm();
    cout << "Mult Bytes Sent: " << (comm_end_mult - comm_start_mult) << "bytes" << endl;

    if (party == ALICE)
    {
        iopack->io->send_data(outax, dim * sizeof(uint64_t));
    }
    else
    {
        uint64_t *recv_outax = new uint64_t[dim];
        iopack->io->recv_data(recv_outax, dim * sizeof(uint64_t));
        for (int i = 0; i < dim; i++)
        {
            std::cout << "total outax[" << i << "] =  " << ((outax[i] + recv_outax[i]) & mask_bwL) << std::endl;
        }
    }
    std::cout << "\n=========STEP11 ax SExt with MSB  ===========" << std::endl;
    // uint64_t *ax_SExt = new uint64_t[dim];

    // if (party == ALICE)
    // {
    //     ext->s_extend(dim, outax, ax_SExt, la + h - f, bwL, msbA);
    // }
    // else
    // {
    //     ext->s_extend(dim, outax, ax_SExt, la + h - f, bwL, msbB);
    // }
    // std::cout << "ax_SExt[" << 0 << "] = " << ax_SExt[0] << std::endl;

    // if (party == ALICE)
    // {
    //     iopack->io->send_data(ax_SExt, dim * sizeof(uint64_t));
    // }
    // else
    // {
    //     uint64_t *recv_ax_SExt = new uint64_t[dim];
    //     iopack->io->recv_data(recv_ax_SExt, dim * sizeof(uint64_t));
    //     std::cout << "total ax_SExt =  " << ((ax_SExt[0] + recv_ax_SExt[0]) & mask_bwL) << std::endl;
    // }
    std::cout << "\n=========STEP12 d SExt with MSB   ===========" << std::endl;
    uint64_t *b_SExt = new uint64_t[dim];
    if (party == ALICE)
    {
        ext->s_extend(dim, b_alice, b_SExt, lb, bwL, msbA);
    }
    else
    {
        ext->s_extend(dim, b_bob, b_SExt, lb, bwL, msbB);
    }
    std::cout << "b_SExt[" << 0 << "] = " << b_SExt[0] << std::endl;

    if (party == ALICE)
    {
        iopack->io->send_data(b_SExt, dim * sizeof(uint64_t));
    }
    else
    {
        uint64_t *recv_b_SExt = new uint64_t[dim];
        iopack->io->recv_data(recv_b_SExt, dim * sizeof(uint64_t));
        for (int i = 0; i < dim; i++)
        {
            std::cout << "total b_SExt[" << i << "] =  " << ((b_SExt[i] + recv_b_SExt[i]) & mask_bwL) << std::endl;
        }
    }
    std::cout << "\n=========STEP13 Caculate z=ax+b   ===========" << std::endl;
    uint64_t *z = new uint64_t[dim];

    std::cout << "b_SExt[" << 0 << "] = " << b_SExt[0] << std::endl;
    std::cout << "outax[" << 0 << "] = " << outax[0] << std::endl;

    for (int i = 0; i < dim; i++)
        z[i] = ((outax[i] + b_SExt[i] * static_cast<uint64_t>(std::pow(2,f-lb+1))) & mask_bwL);

    std::cout << "z[" << 0 << "] = " << z[0] << std::endl;

    if (party == ALICE)
    {
        iopack->io->send_data(z, dim * sizeof(uint64_t));
    }
    else
    {
        uint64_t *recv_z = new uint64_t[dim];
        iopack->io->recv_data(recv_z, dim * sizeof(uint64_t));
        std::cout << "total recv_z =  " << ((z[0] + recv_z[0]) & mask_bwL) << std::endl;
    }

    std::cout << "\n=========STEP14 Drelu |x|-a  to learn b' ===========" << std::endl;

    uint8_t *Drelu_ = new uint8_t[dim];
    uint8_t *DreluMSB = new uint8_t[dim];
    // uint64_t fouty = 16384;
    // Drelu = MSB , Alice ^1

    // if (party == ALICE)
    // {
    //     for (int i = 0; i < dim; i++)
    //     {
    //         EMUX_output_x[i] = (EMUX_output_x[i] - 16384) & mask_bwL;
    //     }

    //      for (int i = 0; i < dim; i++)
    //     {
    //         EMUX_output_x[i] = EMUX_output_x[i] & mask_l_10;
    //     }
    //     prod->aux->MSB(EMUX_output_x, DreluMSB, dim, bwL-10);
    // }
    // else
    // {

    //     for (int i = 0; i < dim; i++)
    //     {
    //         EMUX_output_x[i] = EMUX_output_x[i] & mask_l_10;
    //     }
    //     prod->aux->MSB(EMUX_output_x, DreluMSB, dim, bwL-10);
    // }



    if (party == ALICE)
    {
        for (int i = 0; i < dim; i++)
        {
            EMUX_output_x[i] = (EMUX_output_x[i] - alpha) & mask_bwL;
            std::cout << "EMUX_output_x[i] A =  " << EMUX_output_x[i] << std::endl;

            EMUX_output_x[i] = (EMUX_output_x[i] >> Tk) & mask_l_Tk;

            std::cout << "EMUX_output_x[i] A trun =  " << EMUX_output_x[i] << std::endl;
        }

        // prod->aux->MSB(EMUX_output_x, DreluMSB, dim, bwL);
        prod->aux->MSB(EMUX_output_x, DreluMSB, dim, bwL - Tk);
    }
    else
    {
        for (int i = 0; i < dim; i++)
        {
            std::cout << "EMUX_output_x[i] B =  " << EMUX_output_x[i] << std::endl;
            EMUX_output_x[i] = (EMUX_output_x[i] >> Tk) & mask_l_Tk;
            std::cout << "EMUX_output_x[i] B trun =  " << EMUX_output_x[i] << std::endl;
        }

        // prod->aux->MSB(EMUX_output_x, DreluMSB, dim, bwL);
        prod->aux->MSB(EMUX_output_x, DreluMSB, dim, bwL - Tk);
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

    std::cout << "\n=========STEP15 AND ===========" << std::endl;
    // uint8_t *final_b = new uint8_t[dim];
    // uint8_t *Drelu_and = new uint8_t[dim];
    // if (party == ALICE)
    //     for (int i = 0; i < dim; i++)
    //     {
    //         Drelu_and[i] = Drelu_[i];
    //     }
    // aux->AND(Drelu, Drelu_, final_b, dim);

    for (int i = 0; i < dim; i++)
    {
        // std::cout << "\nDrelu[" << i << "] = " << static_cast<int>(Drelu[i]) << std::endl;
        // std::cout << "Drelu_[" << i << "] = " << static_cast<int>(Drelu_[i]) << std::endl;
        // std::cout << "final_b[" << i << "] = " << static_cast<int>(final_b[i]) << std::endl;
    }

    std::cout << "\n=========STEP16 MUX ===========" << std::endl;
    uint64_t comm_start_mux = iopack->get_comm();

    uint64_t *MUX_data1 = new uint64_t[dim];
    uint64_t *MUX_output_u = new uint64_t[dim];
    for (int i = 0; i < dim; i++)
    {
        MUX_output_u[i] = 0;
    }

    // MUX_data1[0] = z[0];
    // MUX_data1[0] = z[0];
    // if (party==ALICE)
    // MUX_sel[0]=MUX_sel[0]^1;

    // std::cout << "MUX_data1[" << 0 << "] =" << MUX_data1[0] << std::endl;

    // if (party == ALICE)
    //     final_b[0] = final_b[0] ^ 1;

    uint8_t *uinput = new uint8_t[dim];
    if (party == ALICE)
    {
        for (int i = 0; i < dim; i++)
        {
            uinput[i] = Drelu_[i] ^ 1;
        }
    }
    else
    {
        for (int i = 0; i < dim; i++)
        {
            uinput[i] = Drelu_[i];
        }
    }

    aux->multiplexer(uinput, z, MUX_output_u, dim, bwL, bwL);
    std::cout << "MUX_output_u[" << 0 << "] =" << MUX_output_u[0] << std::endl;

    uint64_t comm_end_mux = iopack->get_comm();
    std::cout << "MUX Bytes Sent: " << (comm_end_mux - comm_start_mux) << "bytes" << std::endl;

    std::cout << "\n=========STEP17 truncation with MSB ===========" << std::endl;
    uint64_t *xhalf = new uint64_t[dim];
    if (party == ALICE)
    {
        trunc_oracle->truncate(dim, inA, xhalf, 1, bwL, true, msbA);
    }
    else
    {
        trunc_oracle->truncate(dim, inB, xhalf, 1, bwL, true, msbB);
    }

    std::cout << "xhalf[" << 0 << "] =" << xhalf[0] << std::endl;

    std::cout << "\n=========STEP18 xhalf with final_b to learn v ===========" << std::endl;

    uint64_t *MUX_output_t = new uint64_t[dim];
    aux->multiplexer(Drelu_, xhalf, MUX_output_t, dim, bwL, bwL);

    // uint8_t *vinput = new uint8_t[dim];
    // if (party == ALICE)
    // {
    //     for (size_t i = 0; i < dim; i++)
    //     {
    //         vinput[i] = Drelu[i];
    //     }
    // }
    // else
    // {
    //     for (size_t i = 0; i < dim; i++)
    //     {
    //         vinput[i] = Drelu[i];
    //     }
    // }

    // std::cout << "vinput[" << 0 << "] = " << static_cast<int>(vinput[0]) << std::endl;

    uint64_t *MUX_output_v = new uint64_t[dim];

    aux->multiplexerabs(Drelu, MUX_output_t, MUX_output_v, dim, bwL, bwL);

    std::cout << "\n=========STEP19 y = xhalf + u + v ===========" << std::endl;

    std::cout << "uuu MUX_output_u =  " << MUX_output_u[0] << std::endl;
    if (party == ALICE)
    {
        iopack->io->send_data(MUX_output_u, dim * sizeof(uint64_t));
    }
    else
    {
        uint64_t *recv_MUX_output_u = new uint64_t[dim];
        iopack->io->recv_data(recv_MUX_output_u, dim * sizeof(uint64_t));

        for (int i = 0; i < dim; i++)
        {
            std::cout << "total MUX_output_u =  " << ((MUX_output_u[i] + recv_MUX_output_u[i]) & mask_bwL) << std::endl;
        }
    }
    std::cout << "vvv MUX_output_v =  " << MUX_output_v[0] << std::endl;
    if (party == ALICE)
    {
        iopack->io->send_data(MUX_output_v, dim * sizeof(uint64_t));
    }
    else
    {
        uint64_t *recv_MUX_output_v = new uint64_t[dim];
        iopack->io->recv_data(recv_MUX_output_v, dim * sizeof(uint64_t));
        std::cout << "total MUX_output_v =  " << ((MUX_output_v[0] + recv_MUX_output_v[0]) & mask_bwL) << std::endl;
    }
    std::cout << "xxx xhalf =  " << xhalf[0] << std::endl;
    if (party == ALICE)
    {
        iopack->io->send_data(xhalf, dim * sizeof(uint64_t));
    }
    else
    {
        uint64_t *recv_xhalf = new uint64_t[dim];
        iopack->io->recv_data(recv_xhalf, dim * sizeof(uint64_t));
        std::cout << "total xhalf =  " << ((xhalf[0] + recv_xhalf[0]) & mask_bwL) << std::endl;
    }
    uint64_t *y = new uint64_t[dim];
    for (int i = 0; i < dim; i++)
    {
        y[i] = (xhalf[i] + MUX_output_v[i] + MUX_output_u[i]) & mask_bwL;
        std::cout << "y[" << i << "] = " << y[i] << std::endl;
    }

    std::cout << "\n=========END verification ===========" << std::endl;
    if (party == ALICE)
    {
        iopack->io->send_data(y, dim * sizeof(uint64_t));
    }
    else
    {
        uint64_t *recv_y = new uint64_t[dim];
        iopack->io->recv_data(recv_y, dim * sizeof(uint64_t));
        // std::cout << "total y = y0 + y1 =  " << ((y[0] + recv_y[0]) & mask_bwL) << ", real num: " << (double)decode_ring((y[0] + recv_y[0])&mask_bwL,37) / 4096 << std::endl;

        // std::cout << "ax +b =  " << (((inA[0] + inB[0]) * a_bob[0] + b_bob[0]) & mask_bwL) << std::endl;
        // std::cout << "ax +b  >> 12=  " << ((((inA[0] + inB[0]) * a_bob[0] + b_bob[0]) & mask_bwL) >> 12) << std::endl;
        // std::cout << "The result should be calculate_GELU = " << calculate_GELU(inA[0] + inB[0]) << std::endl;
        std::vector<double> x_values, y_values;
        std::vector<double> x_real, y_real;
        double *ULPs = new double[dim];
        for (int i = 0; i < dim; i++)
        {
            std::cout << "total y = y0 + y1 =  " << ((y[i] + recv_y[i]) & mask_bwL) << ", real num: " << (double)decode_ring((y[i] + recv_y[i]) & mask_bwL, bwL) / 4096 << std::endl;

            // std::cout << "ax +b =  " << (((inA[i] + inB[i]) * a_bob[i] + b_bob[i]) & mask_bwL) << std::endl;
            // std::cout << "ax +b  >> 12=  " << ((((inA[i] + inB[i]) * a_bob[i] + b_bob[i]) & mask_bwL) >> 12) << std::endl;
            std::cout << "The result " << inA[i] + inB[i] << " should be calculate_GELU = " << calculate_GELU(inA[i] + inB[i]) << std::endl;
            ULPs[i] = abs((((double)decode_ring((y[i] + recv_y[i]) & mask_bwL, bwL) / 4096) - calculate_GELU(inA[i] + inB[i])) / 0.000244140625);
            std::cout << "The ULP is = " << ULPs[i] << std::endl;

            x_values.push_back((inA[i] + inB[i]) / 4096.0);
            y_values.push_back((double)decode_ring((y[i] + recv_y[i]) & mask_bwL, bwL) / 4096);
            x_real.push_back((inA[i] + inB[i]) / 4096.0);
            y_real.push_back(calculate_GELU(inA[i] + inB[i]));
        }

        double sum = 0.0;
        for (size_t i = 0; i < dim; ++i)
        {
            sum += (ULPs[i]);
            // std::cout << "ULPs[" << i << "] = " << ULPs[i] << std::endl;
        }
        double average = sum / static_cast<double>(dim);
        std::cout << "sum: " << sum << std::endl;
        std::cout << "static_cast<double>(dim): " << static_cast<double>(dim) << std::endl;
        double max_val = *std::max_element(ULPs, ULPs + dim);
        std::cout << "average: " << average << std::endl;
        std::cout << "max_val: " << max_val << std::endl;
        // 绘制曲线
        // plt::scatter(x_values, y_values, 2 , {{"color", "red"},{"marker", "."}});
        // plt::scatter(x_real, y_real, 1, {{"color", "blue"},{"marker", "."},{"edgecolors", "none"},
        //     {"alpha", "0.7"},
        //     {"label", "GELU"} });

        // 设置标题和标签
        // plt::title("Simple Line Plot");
        // plt::xlabel("x-axis");
        // plt::ylabel("y-axis");
        // // 显示图形
        // // plt::legend();
        // plt::save("/home/zhaoqian/EzPC/test.svg");
        // plt::show();
        std::ofstream file("/home/zhaoqian/EzPC/scatter_data.csv");
        if (!file.is_open())
        {
            std::cerr << "Error: Could not open file for writing." << std::endl;
            return 1;
        }

        // 写入CSV头
        file << "dataset,x,y\n";

        // 写入第一组数据（Decoded Ring）
        for (size_t i = 0; i < x_values.size(); ++i)
        {
            file << "Decoded Ring," << x_values[i] << "," << y_values[i] << "\n";
        }

        // 写入第二组数据（GELU）
        for (size_t i = 0; i < x_real.size(); ++i)
        {
            file << "GELU," << x_real[i] << "," << y_real[i] << "\n";
        }
    }

    ///////////输出时间和通信
    uint64_t comm_end = iopack->get_comm();
    cout << "Tptal Bytes Sent: " << (comm_end - comm_start) << "bytes" << endl;

    auto time_end = chrono::high_resolution_clock::now();
    cout << "Total time: "
         << chrono::duration_cast<chrono::milliseconds>(time_end - time_start).count()
         << " ms" << endl;
    delete[] inA;
    delete[] inB;
    delete[] outax;
    delete prod;
}
