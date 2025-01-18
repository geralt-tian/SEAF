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
using namespace sci;
using namespace std;

int party, port = 32000;
string address = "127.0.0.1";
IOPack *iopack;
OTPack *otpack;
LinearOT *prod;

// 全局变量
int dim1 = 1;
int dim2 = 1;
int dim3 = 1;
int bwA = 22;
int bwB = 15;
int bwC = bwA + bwB; // 矩阵位宽
uint64_t mask_bwC = (bwC == 64 ? -1 : ((1ULL << bwC) - 1));
bool signed_B = true;           // 表示矩阵B是否为有符号数
bool accumulate = true;         // 决定是否累加结果
bool precomputed_MSBs = false;  // 决定是否预计算最高有效位
MultMode mode = MultMode::None; // 乘法模式

/////////////////////compare_with_eq
int bitlength = 64; // 假设每个数是32位
int radix_base = 4; // 基数为4
uint64_t alpha = 4;
uint64_t alpha_ = 1ULL << 14;
uint64_t beta = alpha_ * 2;
uint64_t h = 15;

Truncation *trunc_oracle;
//////////////////////////MUX
AuxProtocols *aux;

double calculate_GELU(uint64_t value) {
    // 定义 2^37 和 2^12 的浮点值
    const uint64_t sign_bit_mask = 1ULL << 36;  // 第 37 位的掩码
    const double pow_2_37 = static_cast<double>(1ULL << 37);
    const double pow_2_12 = static_cast<double>(1ULL << 12);

    // 检查符号位（第 37 位）
    if (value & sign_bit_mask) {
        // 如果符号位为 1，表示负数
        value -= static_cast<uint64_t>(pow_2_37);  // 减去 2^37
    }
    // 将值转换为浮点数
    double x = static_cast<double>(value) / pow_2_12;
    // 计算表达式
    // double a = x - 4;
    double a=x;
    double tanh_part = std::tanh(0.7978845608 * a + 0.7978845608 * 0.044715 * std::pow(a, 3));
    return 0.5 * a * (1 + tanh_part);
}

void assign_lower_h_bits(int32_t dim1, int32_t dim2, int32_t dim3, uint64_t *inA, uint64_t *inB, uint64_t *inA_, uint64_t *inB_, int32_t h)
{
    // Create a mask that has the lowest h bits set to 1
    uint64_t mask = (h == 64) ? ~0ULL : (1ULL << h) - 1;

    // Assign the lower h bits from inA to inA_
    for (int i = 0; i < dim1; i++)
    {
        for (int j = 0; j < dim2; j++)
        {
            inA_[i * dim2 + j] = inA[i * dim2 + j] & mask;
        }
    }

    // Assign the lower h bits from inB to inB_
    for (int i = 0; i < dim2; i++)
    {
        for (int j = 0; j < dim3; j++)
        {
            inB_[i * dim3 + j] = inB[i * dim3 + j] & mask;
        }
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

    uint64_t *inA = new uint64_t[dim1 * dim2]; // 1*100
    uint64_t *inB = new uint64_t[dim2 * dim3]; // 100*35
    int dim = (::accumulate ? dim1 * dim3 : dim1 * dim2 * dim3);
    uint64_t *outax = new uint64_t[dim];
    // 使用PRG128分配并初始化随机矩阵inA和inB
    // prg.random_data(inA, dim1 * dim2 * sizeof(uint64_t));
    // prg.random_data(inB, dim2 * dim3 * sizeof(uint64_t));

    inA[0] = 10000; //(0-16383)
    inB[0] = 6383;
    std::cout << "input inA[" << 0 << "] = " << inA[0] << std::endl;
    std::cout << "input inB[" << 0 << "] = " << inB[0] << std::endl;
    /////////////step2 //check
    //   inA[0] = inA[0]+alpha_;
    //   inB[0] = inB[0]+alpha_;

    uint64_t *inA_ = new uint64_t[dim1 * dim2]; // 1*100
    uint64_t *inB_ = new uint64_t[dim2 * dim3]; // 100*35

    inA_[0] = (inA[0] + alpha_) & mask_bwC;
    inB_[0] = (inB[0]) & mask_bwC;
    std::cout<< "inB_[" << 0 << "] = " << inB_[0] << std::endl;
    // step5 //check
    uint64_t *inA_h = new uint64_t[dim1 * dim2]; // 1*100
    uint64_t *inB_h = new uint64_t[dim2 * dim3]; // 100*35
    std::cout<< "=========STEP5 extract the lower h bits==========="<<std::endl;
    assign_lower_h_bits(dim1, dim2, dim3, inA_, inB_, inA_h, inB_h, h);
    
    std::cout << "inA_h[" << 0 << "] = " << inA_h[0] << std::endl;
    std::cout << "inB_h[" << 0 << "] = " << inB_h[0] << std::endl;

    // step6 check

    std::cout<<"=========STEP6 Truncate_reduce==========="<<std::endl;
    uint64_t comm_start_tr = iopack->get_comm();
    trunc_oracle = new Truncation(party, iopack, otpack);
    uint64_t *outtrunc = new uint64_t[dim];
    if (party == sci::ALICE)
    {
        trunc_oracle->truncate_and_reduce(dim, inA_h, outtrunc, 7, 15); // shift=h-s,hypothesis s=8  truncate就是为了分组，截断后7位，为了前8位可以映射到对应的table
        //std::cout << "outtrunc[" << 0 << "] = " << outtrunc[0] << std::endl;
    }
    else
    {
        trunc_oracle->truncate_and_reduce(dim, inB_h, outtrunc, 7, 15);      // shift=h-s,hypothesis s=8,outtrunc是0-255
        //std::cout << "outtrunc[" << 0 << "] = " << outtrunc[0] << std::endl; // outtrunc是<i>，范围是0-255
    }
    
    std::cout << std::dec << "outtrunc = " << outtrunc[0] << std::endl;
    uint64_t comm_end_tr = iopack->get_comm();
    std::cout << "TR Bytes Sent: " << (comm_end_tr - comm_start_tr) << "bytes" << std::endl;
    // step7 check
    uint64_t comm_start_lut = iopack->get_comm();
    std::cout<<"=========STEP7 LookUp Table   ==========="<<std::endl;
    // 重跑一个有256个的
    std::vector<std::vector<uint64_t>> data = {{137438953471, 0}, {137438953470, 0}, {137438953470, 0}, {137438953470, 0}, {137438953470, 0}, {137438953469, 0}, {137438953469, 0}, {137438953468, 0}, {137438953468, 0}, {137438953467, 0}, {137438953467, 0}, {137438953466, 1}, {137438953465, 1}, {137438953465, 1}, {137438953464, 2}, {137438953463, 2}, {137438953462, 3}, {137438953460, 3}, {137438953459, 4}, {137438953458, 5}, {137438953456, 6}, {137438953454, 7}, {137438953453, 8}, {137438953450, 10}, {137438953448, 11}, {137438953446, 13}, {137438953443, 15}, {137438953440, 18}, {137438953437, 20}, {137438953434, 23}, {137438953431, 27}, {137438953427, 30}, {137438953423, 34}, {137438953418, 39}, {137438953414, 44}, {137438953409, 49}, {137438953404, 55}, {137438953398, 61}, {137438953392, 68}, {137438953386, 76}, {137438953379, 84}, {137438953372, 93}, {137438953365, 103}, {137438953357, 114}, {137438953349, 125}, {137438953340, 137}, {137438953331, 150}, {137438953322, 163}, {137438953312, 178}, {137438953302, 193}, {137438953292, 210}, {137438953281, 227}, {137438953269, 245}, {137438953258, 265}, {137438953246, 285}, {137438953234, 306}, {137438953221, 328}, {137438953208, 351}, {137438953195, 375}, {137438953182, 399}, {137438953168, 424}, {137438953155, 450}, {137438953141, 477}, {137438953127, 504}, {137438953113, 532}, {137438953099, 560}, {137438953086, 588}, {137438953072, 616}, {137438953059, 644}, {137438953046, 672}, {137438953033, 700}, {137438953021, 727}, {137438953010, 753}, {137438952999, 778}, {137438952988, 801}, {137438952979, 823}, {137438952970, 844}, {137438952963, 862}, {137438952956, 878}, {137438952951, 891}, {137438952947, 900}, {137438952945, 907}, {137438952944, 910}, {137438952944, 908}, {137438952947, 902}, {137438952951, 891}, {137438952957, 874}, {137438952965, 852}, {137438952975, 824}, {137438952988, 789}, {137438953003, 747}, {137438953020, 697}, {137438953040, 640}, {137438953063, 574}, {137438953088, 500}, {137438953116, 416}, {137438953147, 324}, {137438953181, 221}, {137438953218, 108}, {137438953258, 137438953457}, {137438953301, 137438953323}, {137438953347, 137438953178}, {137438953396, 137438953021}, {137438953448, 137438952853}, {31, 137438952673}, {90, 137438952482}, {151, 137438952278}, {216, 137438952062}, {283, 137438951834}, {354, 137438951594}, {427, 137438951342}, {503, 137438951078}, {582, 137438950803}, {663, 137438950515}, {747, 137438950216}, {834, 137438949906}, {922, 137438949586}, {1013, 137438949255}, {1105, 137438948913}, {1199, 137438948563}, {1295, 137438948203}, {1393, 137438947835}, {1491, 137438947459}, {1591, 137438947076}, {1692, 137438946686}, {1793, 137438946290}, {1895, 137438945889}, {1997, 137438945484}, {2099, 137438945075}, {2201, 137438944664}, {2303, 137438944250}, {2404, 137438943836}, {2505, 137438943421}, {2605, 137438943006}, {2703, 137438942593}, {2801, 137438942182}, {2897, 137438941775}, {2991, 137438941371}, {3083, 137438940972}, {3174, 137438940578}, {3262, 137438940191}, {3349, 137438939811}, {3433, 137438939439}, {3514, 137438939075}, {3593, 137438938720}, {3669, 137438938376}, {3742, 137438938041}, {3813, 137438937717}, {3880, 137438937405}, {3945, 137438937105}, {4006, 137438936816}, {4065, 137438936541}, {4120, 137438936278}, {4172, 137438936028}, {4221, 137438935792}, {4267, 137438935569}, {4310, 137438935360}, {4350, 137438935164}, {4387, 137438934982}, {4421, 137438934814}, {4452, 137438934659}, {4480, 137438934518}, {4505, 137438934390}, {4528, 137438934275}, {4548, 137438934172}, {4565, 137438934083}, {4580, 137438934005}, {4593, 137438933939}, {4603, 137438933885}, {4611, 137438933842}, {4617, 137438933809}, {4621, 137438933787}, {4624, 137438933774}, {4624, 137438933771}, {4623, 137438933777}, {4621, 137438933791}, {4617, 137438933812}, {4612, 137438933841}, {4605, 137438933877}, {4598, 137438933920}, {4589, 137438933968}, {4580, 137438934021}, {4569, 137438934080}, {4558, 137438934142}, {4547, 137438934209}, {4535, 137438934279}, {4522, 137438934352}, {4509, 137438934428}, {4496, 137438934506}, {4482, 137438934585}, {4469, 137438934667}, {4455, 137438934749}, {4441, 137438934832}, {4427, 137438934915}, {4413, 137438934998}, {4400, 137438935082}, {4386, 137438935164}, {4373, 137438935246}, {4360, 137438935328}, {4347, 137438935408}, {4334, 137438935486}, {4322, 137438935563}, {4310, 137438935639}, {4299, 137438935713}, {4287, 137438935785}, {4276, 137438935854}, {4266, 137438935922}, {4256, 137438935988}, {4246, 137438936051}, {4237, 137438936112}, {4228, 137438936171}, {4219, 137438936227}, {4211, 137438936281}, {4203, 137438936333}, {4196, 137438936383}, {4189, 137438936430}, {4182, 137438936475}, {4176, 137438936517}, {4170, 137438936558}, {4164, 137438936596}, {4159, 137438936632}, {4154, 137438936666}, {4150, 137438936698}, {4145, 137438936729}, {4141, 137438936757}, {4137, 137438936784}, {4134, 137438936808}, {4131, 137438936832}, {4128, 137438936853}, {4125, 137438936873}, {4122, 137438936892}, {4120, 137438936909}, {4118, 137438936925}, {4115, 137438936940}, {4114, 137438936954}, {4112, 137438936967}, {4110, 137438936978}, {4109, 137438936989}, {4108, 137438936999}, {4106, 137438937008}, {4105, 137438937016}, {4104, 137438937023}, {4103, 137438937030}, {4103, 137438937036}, {4102, 137438937042}, {4101, 137438937047}, {4101, 137438937051}, {4100, 137438937055}, {4100, 137438937059}, {4099, 137438937062}, {4099, 137438937065}, {4098, 137438937068}, {4098, 137438937070}, {4098, 137438937073}, {4098, 137438937074}, {4097, 137438937076}};
    
    int32_t T_size = sizeof(uint64_t) * 8;
    int bw_xlut = 8;
    int bw_ylut;
    aux = new AuxProtocols(party, iopack, otpack);
    if (T_size == 8)
        bw_ylut = 7;
    else
        bw_ylut = 29;
    uint64_t *a_alice = new uint64_t[dim];
    uint64_t *b_alice = new uint64_t[dim];
    a_alice[0] = 0;
    b_alice[0] = 0;
    uint64_t **spec_a = new uint64_t *[dim];
    uint64_t *a_bob = new uint64_t[dim];
    uint64_t N = 1ULL << bw_xlut;

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
        outtrunc1[0]=0;
    }
    
    uint64_t *outtrunc_a = new uint64_t[dim];
        if (party == ALICE)
    {
        iopack->io->send_data(outtrunc, dim * sizeof(uint64_t));
    }
    else
    {                                                                            // party == BOB
        iopack->io->recv_data(outtrunc1, dim * sizeof(uint64_t));
        outtrunc_a[0] = (outtrunc[0] + outtrunc1[0]) & ((1ULL<<8) - 1);
        std::cout << "outtrunc_a[" << 0 << "] = " << outtrunc_a[0] << std::endl;
    }
    if (party == ALICE)
    {
        aux->lookup_table<uint64_t>(spec_a, nullptr, nullptr, dim, bw_xlut, 37); // bw_xlut是outtrunc的位宽
    }
    else
    {                                                                            // party == BOB
        aux->lookup_table<uint64_t>(nullptr, outtrunc_a, a_bob, dim, bw_xlut, 37); // a_bob是查询到的斜率
    }
    if (party != ALICE)
        std::cout << "a_bob[" << 0 << "] = " << a_bob[0] << std::endl;

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
        aux->lookup_table<uint64_t>(spec_b, nullptr, nullptr, dim, bw_xlut, 37);
    }
    else
    {                                                                          // party == BOB
        aux->lookup_table<uint64_t>(nullptr, outtrunc_a, b_bob, dim, bw_xlut, 37); // b_bob是查询到的截距  重要问题，这里的outtrunc应该是两边share加起来，代码里只有Bob的outtrunc check
    }
    if (party != ALICE)
        std::cout << "b_bob[" << 0 << "] = " << b_bob[0] << std::endl;

    uint64_t comm_end_lut = iopack->get_comm();
    cout << "LUT Bytes Sent: " << (comm_end_lut - comm_start_lut) << "bytes" << endl;
    //////////////////////step8
    std::cout<<"=========STEP8 matrix_multiplication   ==========="<<std::endl;
    uint64_t comm_start_mult = iopack->get_comm();
    uint8_t *msbA = nullptr;
    uint8_t *msbB = nullptr;


    //   test_matrix_multiplication(inA, inB, outC, false);
    // test_matrix_multiplication(inA, inB, outC, true);
    
    if (party == ALICE)
    {
            if (precomputed_MSBs)
    { // 预计算MSB
        msbA = new uint8_t[dim1 * dim2];
        msbB = new uint8_t[dim2 * dim3];
        prod->aux->MSB(a_alice, msbA, dim1 * dim2, bwA);
        prod->aux->MSB(inA_h, msbB, dim2 * dim3, bwB);
    }
        std::cout << "inA_h[" << 0 << "] = " << inA_h[0] << std::endl;
        std::cout << "a_alice[" << 0 << "] = " << a_alice[0] << std::endl;

        prod->matrix_multiplication(dim1, dim2, dim3, a_alice,  inA_, outax, bwC, bwC, bwC+22,
                                    true, true, ::accumulate, mode,
                                    msbA, msbB);
    }
    else
    {
        std::cout << "inB_h[" << 0 << "] = " << inB_h[0] << std::endl;
        std::cout << "a_bob[" << 0 << "] = " << a_bob[0] << std::endl;
                    if (precomputed_MSBs)
    { // 预计算MSB
        msbA = new uint8_t[dim1 * dim2];
        msbB = new uint8_t[dim2 * dim3];
        prod->aux->MSB(a_bob, msbA, dim1 * dim2, bwA);
        prod->aux->MSB(inB_h, msbB, dim2 * dim3, bwB);
    }
        prod->matrix_multiplication(dim1, dim2, dim3, a_bob,  inB_h, outax, bwC, bwC, bwC+22,
                                    true, true, ::accumulate, mode,
                                    msbA, msbB);
    }
    /////////////////////////

    for (int i = 0; i < dim1 * dim3; i++)
    {
        outax[i] = outax[i] & mask_bwC;
        std::cout << "step8 outax[" << i << "] = " << outax[i] << std::endl;
    }

    uint64_t comm_end_mult = iopack->get_comm();
    cout << "Mult Bytes Sent: " << (comm_end_mult - comm_start_mult) << "bytes" << endl;
    /////////////////////////新增截断
    std::cout<<"=========STEP9 Truncation   ==========="<<std::endl;
    uint64_t *outB = new uint64_t[dim];
    trunc_oracle->truncate(dim, inA, outB, 12, 37, true, nullptr);


    for (size_t i = 0; i < dim; i++)
    {
        std::cout << "step9 Trunc_outB[" << i << "] = " << outB[i] << std::endl;
    }
    




    /////////////////////////////step9
    std::cout<<"=========STEP9 ADDITION <z>  ==========="<<std::endl;
    uint64_t *z = new uint64_t[dim];
    if (party == ALICE)
    {
        z[0] = ((outax[0] + b_alice[0]) & mask_bwC);
    }
    else
    {
        z[0] = (outax[0] + b_bob[0]) & mask_bwC;
    }
    std::cout << "z[" << 0 << "] = " << z[0] << std::endl;
    //////////////////////MillionaireWithEquality step10
    std::cout<<"=========STEP10 CMP   ==========="<<std::endl;


    // 比较两个数
    // ALICE 和 BOB 分别输入自己的数据
    uint64_t *local_data1 = new uint64_t[dim2 * dim3]; // 100*35
    uint64_t *local_data2 = new uint64_t[dim2 * dim3]; // 100*35
    if (party == ALICE)
    {
        for (int i = 0; i < dim; i++)
            local_data1[i] = inA_[i];
    }
    else
    {
        for (int i = 0; i < dim; i++)
            local_data1[i] = ((inB_[i]) - (1ULL << 15)) & mask_bwC; // 得设置环操作

    }
    std::cout << "local_data1[0]= " << local_data1[0] << std::endl;
    // 调用 compare_with_eq 函数进行比较

    uint8_t *msb = new uint8_t[dim];
    // 参与方分别传入自己的数据
    uint64_t comm_start_msb = iopack->get_comm();
    prod->aux->MSB(local_data1, msb, dim1 * dim2, bwC);
    std::cout << "msb[0] = " << static_cast<int>(msb[0]) << std::endl;
    if (party == ALICE)
        msb[0] = msb[0] ^ 1;
    uint64_t comm_end_msb = iopack->get_comm();
    // millionaire.compare_with_eq(res_cmp_b, res_eq_b, local_data1, 1, bitlength, true, radix_base);//line10  这里生成的可能是结果的share，真tm是
    
    std::cout << "msb[0] = " << static_cast<int>(msb[0]) << std::endl;
    // std::cout << "res_cmp_b[0] = " << static_cast<int>(res_eq_b[0]) << std::endl;
    //  输出比较结果

    cout << "MSB Bytes Sent: " << (comm_end_msb - comm_start_msb) << "bytes" << endl;

    ///////////step11

    if (party == ALICE)
    {
        for (int i = 0; i < dim; i++)
            local_data2[i] = inA_[i] & mask_bwC;
    }
    else
    {
        for (int i = 0; i < dim; i++)
            local_data2[i] = (inB_[i]) & mask_bwC;
        ; // 设置环操作
    }
    std::cout << "local_data2[0]=" << local_data2[0] << std::endl;
    uint8_t *msb_ = new uint8_t[dim];
    uint64_t comm_start_cmp = iopack->get_comm();


    prod->aux->MSB(local_data2, msb_, dim1 * dim2, bwC);
    if (party == ALICE)
    msb_[0] = msb_[0] ^ 1;

    std::cout << "msb_[0] = " << static_cast<int>(msb_[0]) << std::endl;

    // uint64_t comm_end_cmp = iopack->get_comm();
    // cout << "CMP Bytes Sent: " << (comm_end_cmp - comm_start_cmp) <<"bytes"<< endl;

    ////////////////////////////////step12
    std::cout<<"=========STEP12 MUX   ==========="<<std::endl;
    // aux = new AuxProtocols(party, iopack, otpack);
    uint64_t comm_start_mux = iopack->get_comm();
    uint8_t *MUX_sel = new uint8_t[dim1];
    int bw_x = 37, bw_y = 37;

    uint64_t *MUX_data1 = new uint64_t[dim1];
    uint64_t *MUX_output_u = new uint64_t[dim1];
    MUX_output_u[0] = 0;

    MUX_sel[0] = msb_[0] ^ msb[0];
    // MUX_data1[0] = z[0];
    MUX_data1[0] = z[0];
    // if (party==ALICE)
    // MUX_sel[0]=MUX_sel[0]^1;

    std::cout << "MUX_sel[" << 0 << "] = " << static_cast<int>(MUX_sel[0]) << std::endl;
    std::cout << "MUX_data1[" << 0 << "] =" << MUX_data1[0] << std::endl;

    aux->multiplexer(MUX_sel, z, MUX_output_u, dim1, bw_x, bw_y);
    std::cout << "MUX_output_u[" << 0 << "] =" << MUX_output_u[0] << std::endl;

    uint64_t comm_end_mux = iopack->get_comm();
    std::cout << "MUX Bytes Sent: " << (comm_end_mux - comm_start_mux) << "bytes" << std::endl;

    /////////step13
    uint64_t *MUX_output_v = new uint64_t[dim1];
    MUX_output_v[0] = 0;
    if (party == ALICE)
    {
        MUX_sel[0] = msb[0];
        std::cout << "MUX_sel[" << 0 << "] = " << static_cast<int>(MUX_sel[0]) << std::endl;
        aux->multiplexer(MUX_sel, inA, MUX_output_v, dim1, bw_x, bw_y);
        std::cout << "MUX_output_v[" << 0 << "] =" << MUX_output_v[0] << std::endl;
    }
    else
    {
        MUX_sel[0] = msb[0];
        std::cout << "MUX_sel[" << 0 << "] = " << static_cast<int>(MUX_sel[0]) << std::endl;
        aux->multiplexer(MUX_sel, inB, MUX_output_v, dim1, bw_x, bw_y);
        std::cout << "MUX_output_v[" << 0 << "] =" << MUX_output_v[0] << std::endl;
    }

    //////////step14

    ///////////输出时间和通信
    uint64_t comm_end = iopack->get_comm();
    cout << "Tptal Bytes Sent: " << (comm_end - comm_start) << "bytes" << endl;

    auto time_end = chrono::high_resolution_clock::now();
    cout << "Total time: "
         << chrono::duration_cast<chrono::milliseconds>(time_end - time_start).count()
         << " ms" << endl;
    ///////////////////////////////check
    if (party == ALICE)
    {
        iopack->io->send_data(MUX_output_u, dim * sizeof(uint64_t));
        iopack->io->send_data(MUX_output_v, dim * sizeof(uint64_t));
    }
    else
    {
        uint64_t *MUX_rec_u = new uint64_t[dim];
        uint64_t *MUX_rec_v = new uint64_t[dim];
        iopack->io->recv_data(MUX_rec_u, dim * sizeof(uint64_t));
        iopack->io->recv_data(MUX_rec_v, dim * sizeof(uint64_t));

        // uint64_t result = ((( MUX_output_u[0]+ MUX_rec_u[0]) >> 12 +  MUX_rec_v[0] + MUX_output_v[0]) & mask_bwC) >> 24;
            // 第一步：计算 MUX_output_u[0] + MUX_rec_u[0]
    uint64_t sum_u = MUX_output_u[0] + MUX_rec_u[0]- b_bob[0];
    sum_u = std::fmod(sum_u, static_cast<long double>(mask_bwC + 1));
    std::cout << "Step 1 - sum_u (MUX_output_u[0] + MUX_rec_u[0]): " << sum_u << std::endl;

    // 第二步：将 sum_u 转换为浮点数并除以 2**12
    long double sum_u_div = static_cast<long double>(sum_u) / static_cast<long double>(1ULL << 12);
    std::cout << "Step 2 - sum_u_div (sum_u / 2^12): " << sum_u_div << std::endl;

    // 第三步：继续计算 MUX_rec_v[0] 和 MUX_output_v[0] 的和，并加到 sum_u_div 上
    long double final_sum = sum_u_div + static_cast<long double>(b_bob[0]) ;
    std::cout << "Step 3 - final_sum (sum_u_div + MUX_rec_v[0] + MUX_output_v[0]): " << final_sum << std::endl;
    
    // 第四步：应用掩码
    final_sum = std::fmod(final_sum, static_cast<long double>(mask_bwC + 1)); // mask_bwC + 1 to include the full range of the mask
    std::cout << "Step 4 - final_sum after masking: " << final_sum << std::endl;

    // 第五步：用浮点数除以 2**24
    long double result = final_sum / static_cast<long double>(1ULL << 12);
    std::cout << "Step 5 - result (final_sum / 2^24): " << result << std::endl;
        // The result is automatically modulo 2^64 because of uint64_ts
        std::cout << "The input is: " << (inA[0]+inB[0]) << std::endl;
        std::cout << "The input turn to float is: " <<static_cast<double>(inA[0] + inB[0]) / static_cast<double>(1ULL << 12) << std::endl;
        std::cout << "The float_result is: " << result << std::endl;
        // calculate_GELU(inA[0]+inB[0]);
        std::cout << "The result should be calculate_GELU = " << calculate_GELU(inA[0]+inB[0]) << std::endl;
    }


    ////////////////////////
    delete[] inA_;
    delete[] inB_;
    delete[] inA;
    delete[] inB;
    delete[] outax;
    delete prod;
}
