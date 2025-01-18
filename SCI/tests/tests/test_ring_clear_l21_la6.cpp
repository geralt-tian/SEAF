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

bool signed_B = true;           // 表示矩阵B是否为有符号数
bool accumulate = true;         // 决定是否累加结果
bool precomputed_MSBs = false;  // 决定是否预计算最高有效位
MultMode mode = MultMode::None; // 乘法模式

// uint64_t la = 14;//la=5 f=5,la=14,f=12
uint64_t lb = 10;
// uint64_t f = 12;
uint64_t la = 6; // la=5 f=5,la=14,f=12
uint64_t f = 11;
uint64_t h = f + 2;
uint64_t Tk = f - 1;
uint64_t alpha = 3.5 * pow(2, f);

uint64_t mask_l_Tk = (bwL == 64 ? -1 : ((1ULL << (bwL - Tk)) - 1));
uint64_t mask_lah1 = ((la + h + 1) == 64 ? -1 : ((1ULL << (la + h + 1)) - 1));
uint64_t mask_lla = ((la + bwL) == 64 ? -1 : ((1ULL << (la + bwL)) - 1));
uint64_t s = 6;
uint64_t mask_s = ((s) == 64 ? -1 : ((1ULL << (s)) - 1));
// s = 5(低精度)，s = 6(高)， s = 7 与 s = 6 误差相差不大
Truncation *trunc_oracle;
AuxProtocols *aux;

double calculate_GELU(uint64_t value)
{
    //     // 定义 2^37 和 2^12 的浮点值
    const uint64_t sign_bit_mask = 1ULL << (bwL - 1); // 第 37 位的掩码
    const double pow_2_21 = static_cast<double>(1ULL << bwL);
    const double pow_2_12 = static_cast<double>(1ULL << f);

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

void select_share(uint8_t *sel,uint64_t *x, uint64_t *y, uint64_t *output,int32_t dim, int32_t h)
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

int main(int argc, char **argv)
{
    ArgMapping amap;
    int dim = 1;
    uint8_t acc = 1;
    uint64_t init_input = 0;
    uint64_t step_size = 2;
    uint8_t correct = 1;
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

    uint64_t comm_start = iopack->get_comm();
    auto time_start = chrono::high_resolution_clock::now();

    prod = new LinearOT(party, iopack, otpack);

    uint64_t *inA = new uint64_t[dim];
    uint64_t *inB = new uint64_t[dim];

    uint64_t *outax = new uint64_t[dim];
    for (int i = 0; i < dim; i++)
    {
        inA[i] = 0 + i * step_size;
        inB[i] = init_input + i * step_size;
    }
    uint64_t *inA_h = new uint64_t[dim];
    uint64_t *inB_h = new uint64_t[dim];

    std::cout << "\n=========STEP3 use DRelu to learn [[b]]^B===========" << std::endl;

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
    for (int i = 0; i < dim; i++)
    {
        std::cout << "wrap[" << i << "] = " << static_cast<int>(wrap[i]) << std::endl;
    }

    std::cout << "Drelu[" << 0 << "] = " << static_cast<int>(Drelu[0]) << std::endl;

    std::cout << "\n=========STEP4 use EMUX to learn [[|x|]]in L ring===========" << std::endl;
    uint64_t STEP4_comm_start = iopack->get_comm();
    aux = new AuxProtocols(party, iopack, otpack);
    uint64_t *EMUX_output_x = new uint64_t[dim];
    if (party == ALICE)
    {
        aux->multiplexerabs(Drelu, inA, EMUX_output_x, dim, bwL, bwL);
    }
    else
    {
        aux->multiplexerabs(Drelu, inB, EMUX_output_x, dim, bwL, bwL);
    }
    uint64_t STEP4_comm_end = iopack->get_comm();
    std::cout << "\n=========STEP6 extract the lower h bits===========" << std::endl;
    std::cout << "inB[" << 0 << "] = " << inB[0] << std::endl;
    assign_lower_h_bits(dim, inA, inB, inA_h, inB_h, h);

    //////////////////////////////////////////////////////// general版本：直接截取，不用截断协议；高精度版本：使用截断协议
    // step6 check
    std::cout << "\n=========STEP5 get mid s bit for LUT===========" << std::endl;
    uint64_t STEP5_comm_start = iopack->get_comm();
    trunc_oracle = new Truncation(party, iopack, otpack);
    uint64_t *outtrunc = new uint64_t[dim];
    // if(acc==2){
    uint64_t *arith_wrap = new uint64_t[dim];

    prod->aux->B2A(wrap, arith_wrap, dim, s);

    if (party == ALICE)
    {
        for (int i = 0; i < dim; i++)
        {
            outtrunc[i] = ((inA_h[i] >> (h - s)) + arith_wrap[i]) & mask_s;
            // std::cout << "inA_h[" << i << "] = " << inA_h[i] << std::endl;
            // outtrunc[i] = ((inA_h[i] >> (h - s)) ) & mask_s;
        }
    }
    else
    {
        for (int i = 0; i < dim; i++)
        {
            outtrunc[i] = ((inB_h[i] >> (h - s)) + arith_wrap[i]) & mask_s;
        }
    }

    // prod->aux->B2A(wrap, inA_h, dim, h - s);
    //         if (party == sci::ALICE)
    // {
    //     trunc_oracle->truncate_and_reduce(dim, inA_h, outtrunc, h - s, h); // shift=h-s,hypothesis s=7  truncate就是为了分组，截断后7位，为了前s位可以映射到对应的table
    // }
    // else
    // {
    //     trunc_oracle->truncate_and_reduce(dim, inB_h, outtrunc, h - s, h);
    // }
    // }
    // else{
    //     if(party==ALICE){
    //         for(int i=0;i<dim;i++){
    //             outtrunc[i] = (inA_h[i] >> (h - s)) ;
    //         }
    //     }
    //     else{
    //         for(int i=0;i<dim;i++){
    //             outtrunc[i] = (inA_h[i] >> (h - s)) ;
    //         }
    //     }
    // }
    uint64_t STEP5_comm_end = iopack->get_comm();

    std::cout << std::dec << "outtrunc = " << outtrunc[0] << std::endl;
    // step7 check

    std::cout << "\n=========STEP6 LookUp Table   ===========" << std::endl;
    // 重跑一个有256个的
    // std::vector<std::vector<uint64_t>> data = {{137438953471, 0}, {137438953470, 0}, {137438953470, 0}, {137438953470, 0}, {137438953470, 0}, {137438953469, 0}, {137438953469, 0}, {137438953468, 0}, {137438953468, 0}, {137438953467, 0}, {137438953467, 0}, {137438953466, 1}, {137438953465, 1}, {137438953465, 1}, {137438953464, 2}, {137438953463, 2}, {137438953462, 3}, {137438953460, 3}, {137438953459, 4}, {137438953458, 5}, {137438953456, 6}, {137438953454, 7}, {137438953453, 8}, {137438953450, 10}, {137438953448, 11}, {137438953446, 13}, {137438953443, 15}, {137438953440, 18}, {137438953437, 20}, {137438953434, 23}, {137438953431, 27}, {137438953427, 30}, {137438953423, 34}, {137438953418, 39}, {137438953414, 44}, {137438953409, 49}, {137438953404, 55}, {137438953398, 61}, {137438953392, 68}, {137438953386, 76}, {137438953379, 84}, {137438953372, 93}, {137438953365, 103}, {137438953357, 114}, {137438953349, 125}, {137438953340, 137}, {137438953331, 150}, {137438953322, 163}, {137438953312, 178}, {137438953302, 193}, {137438953292, 210}, {137438953281, 227}, {137438953269, 245}, {137438953258, 265}, {137438953246, 285}, {137438953234, 306}, {137438953221, 328}, {137438953208, 351}, {137438953195, 375}, {137438953182, 399}, {137438953168, 424}, {137438953155, 450}, {137438953141, 477}, {137438953127, 504}, {137438953113, 532}, {137438953099, 560}, {137438953086, 588}, {137438953072, 616}, {137438953059, 644}, {137438953046, 672}, {137438953033, 700}, {137438953021, 727}, {137438953010, 753}, {137438952999, 778}, {137438952988, 801}, {137438952979, 823}, {137438952970, 844}, {137438952963, 862}, {137438952956, 878}, {137438952951, 891}, {137438952947, 900}, {137438952945, 907}, {137438952944, 910}, {137438952944, 908}, {137438952947, 902}, {137438952951, 891}, {137438952957, 874}, {137438952965, 852}, {137438952975, 824}, {137438952988, 789}, {137438953003, 747}, {137438953020, 697}, {137438953040, 640}, {137438953063, 574}, {137438953088, 500}, {137438953116, 416}, {137438953147, 324}, {137438953181, 221}, {137438953218, 108}, {137438953258, 137438953457}, {137438953301, 137438953323}, {137438953347, 137438953178}, {137438953396, 137438953021}, {137438953448, 137438952853}, {31, 137438952673}, {90, 137438952482}, {151, 137438952278}, {216, 137438952062}, {283, 137438951834}, {354, 137438951594}, {427, 137438951342}, {503, 137438951078}, {582, 137438950803}, {663, 137438950515}, {747, 137438950216}, {834, 137438949906}, {922, 137438949586}, {1013, 137438949255}, {1105, 137438948913}, {1199, 137438948563}, {1295, 137438948203}, {1393, 137438947835}, {1491, 137438947459}, {1591, 137438947076}, {1692, 137438946686}, {1793, 137438946290}, {1895, 137438945889}, {1997, 137438945484}, {2099, 137438945075}, {2201, 137438944664}, {2303, 137438944250}, {2404, 137438943836}, {2505, 137438943421}, {2605, 137438943006}, {2703, 137438942593}, {2801, 137438942182}, {2897, 137438941775}, {2991, 137438941371}, {3083, 137438940972}, {3174, 137438940578}, {3262, 137438940191}, {3349, 137438939811}, {3433, 137438939439}, {3514, 137438939075}, {3593, 137438938720}, {3669, 137438938376}, {3742, 137438938041}, {3813, 137438937717}, {3880, 137438937405}, {3945, 137438937105}, {4006, 137438936816}, {4065, 137438936541}, {4120, 137438936278}, {4172, 137438936028}, {4221, 137438935792}, {4267, 137438935569}, {4310, 137438935360}, {4350, 137438935164}, {4387, 137438934982}, {4421, 137438934814}, {4452, 137438934659}, {4480, 137438934518}, {4505, 137438934390}, {4528, 137438934275}, {4548, 137438934172}, {4565, 137438934083}, {4580, 137438934005}, {4593, 137438933939}, {4603, 137438933885}, {4611, 137438933842}, {4617, 137438933809}, {4621, 137438933787}, {4624, 137438933774}, {4624, 137438933771}, {4623, 137438933777}, {4621, 137438933791}, {4617, 137438933812}, {4612, 137438933841}, {4605, 137438933877}, {4598, 137438933920}, {4589, 137438933968}, {4580, 137438934021}, {4569, 137438934080}, {4558, 137438934142}, {4547, 137438934209}, {4535, 137438934279}, {4522, 137438934352}, {4509, 137438934428}, {4496, 137438934506}, {4482, 137438934585}, {4469, 137438934667}, {4455, 137438934749}, {4441, 137438934832}, {4427, 137438934915}, {4413, 137438934998}, {4400, 137438935082}, {4386, 137438935164}, {4373, 137438935246}, {4360, 137438935328}, {4347, 137438935408}, {4334, 137438935486}, {4322, 137438935563}, {4310, 137438935639}, {4299, 137438935713}, {4287, 137438935785}, {4276, 137438935854}, {4266, 137438935922}, {4256, 137438935988}, {4246, 137438936051}, {4237, 137438936112}, {4228, 137438936171}, {4219, 137438936227}, {4211, 137438936281}, {4203, 137438936333}, {4196, 137438936383}, {4189, 137438936430}, {4182, 137438936475}, {4176, 137438936517}, {4170, 137438936558}, {4164, 137438936596}, {4159, 137438936632}, {4154, 137438936666}, {4150, 137438936698}, {4145, 137438936729}, {4141, 137438936757}, {4137, 137438936784}, {4134, 137438936808}, {4131, 137438936832}, {4128, 137438936853}, {4125, 137438936873}, {4122, 137438936892}, {4120, 137438936909}, {4118, 137438936925}, {4115, 137438936940}, {4114, 137438936954}, {4112, 137438936967}, {4110, 137438936978}, {4109, 137438936989}, {4108, 137438936999}, {4106, 137438937008}, {4105, 137438937016}, {4104, 137438937023}, {4103, 137438937030}, {4103, 137438937036}, {4102, 137438937042}, {4101, 137438937047}, {4101, 137438937051}, {4100, 137438937055}, {4100, 137438937059}, {4099, 137438937062}, {4099, 137438937065}, {4098, 137438937068}, {4098, 137438937070}, {4098, 137438937073}, {4098, 137438937074}, {4097, 137438937076}};
    // 128:

    // la=14 lb=14
    //  std::vector<std::vector<uint64_t>> data = {{0, 0}, {204, 16380}, {407, 16367}, {610, 16345}, {812, 16314}, {1013, 16275}, {1211, 16228}, {1407, 16172}, {1600, 16109}, {1790, 16038}, {1977, 15959}, {2160, 15874}, {2339, 15782}, {2514, 15684}, {2684, 15579}, {2850, 15470}, {3010, 15355}, {3165, 15236}, {3315, 15112}, {3459, 14985}, {3597, 14855}, {3729, 14722}, {3855, 14587}, {3975, 14450}, {4089, 14312}, {4197, 14173}, {4298, 14033}, {4394, 13894}, {4483, 13755}, {4566, 13617}, {4642, 13481}, {4713, 13346}, {4778, 13213}, {4837, 13083}, {4890, 12955}, {4938, 12830}, {4981, 12709}, {5018, 12590}, {5050, 12476}, {5078, 12365}, {5100, 12258}, {5119, 12155}, {5133, 12056}, {5143, 11962}, {5149, 11871}, {5151, 11785}, {5151, 11704}, {5147, 11626}, {5140, 11553}, {5130, 11483}, {5118, 11418}, {5104, 11356}, {5087, 11299}, {5069, 11245}, {5049, 11194}, {5027, 11147}, {5004, 11102}, {4980, 11061}, {4955, 11023}, {4929, 10987}, {4903, 10953}, {4876, 10922}, {4849, 10892}, {4821, 10864}, {4794, 10838}, {4766, 10813}, {4739, 10789}, {4711, 10767}, {4684, 10745}, {4658, 10723}, {4631, 10702}, {4606, 10682}, {4580, 10661}, {4556, 10641}, {4532, 10620}, {4509, 10599}, {4486, 10577}, {4464, 10555}, {4443, 10532}, {4423, 10509}, {4404, 10484}, {4385, 10459}, {4367, 10433}, {4350, 10406}, {4334, 10377}, {4318, 10348}, {4303, 10317}, {4289, 10286}, {4276, 10253}, {4263, 10218}, {4251, 10183}, {4240, 10146}, {4230, 10109}, {4220, 10070}, {4210, 10030}, {4202, 9988}, {4193, 9946}, {4186, 9902}, {4179, 9857}, {4172, 9812}, {4166, 9765}, {4160, 9717}, {4154, 9668}, {4149, 9619}, {4145, 9568}, {4141, 9516}, {4137, 9464}, {4133, 9411}, {4130, 9357}, {4126, 9303}, {4124, 9248}, {4121, 9192}, {4119, 9135}, {4116, 9079}, {4114, 9021}, {4113, 8963}, {4111, 8905}, {4109, 8846}, {4108, 8786}, {4107, 8727}, {4106, 8667}, {4105, 8606}, {4104, 8545}, {4103, 8484}, {4102, 8423}, {4101, 8362}, {4101, 8300}, {4100, 8238}};
    // la=13 lb=14
    // std::vector<std::vector<uint64_t>> data = {{125,16383}, {308,16377}, {514,16364}, {716,16345}, {915,16320}, {1113,16289}, {1315,16251}, {1507,16209}, {1691,16163}, {1880,16110}, {2069,16051}, {2255,15987}, {2423,15924}, {2598,15853}, {2763,15781}, {2940,15698}, {3080,15628}, {3242,15542}, {3393,15457}, {3526,15378}, {3659,15295}, {3786,15212}, {3914,15124}, {4031,15040}, {4147,14953}, {4252,14871}, {4349,14792}, {4438,14717}, {4526,14640}, {4600,14573}, {4679,14499}, {4740,14440}, {4815,14365}, {4873,14305}, {4919,14256}, {4961,14210}, {5001,14165}, {5033,14128}, {5065,14090}, {5084,14067}, {5116,14027}, {5123,14018}, {5136,14001}, {5145,13989}, {5151,13981}, {5151,13981}, {5149,13984}, {5144,13991}, {5135,14005}, {5125,14020}, {5111,14042}, {5096,14066}, {5078,14095}, {5059,14126}, {5038,14162}, {5016,14200}, {4993,14240}, {4968,14285}, {4943,14330}, {4917,14378}, {4890,14428}, {4863,14480}, {4836,14532}, {4808,14587}, {4780,14643}, {4753,14698}, {4725,14756}, {4698,14812}, {4671,14870}, {4645,14926}, {4619,14983}, {4593,15040}, {4569,15094}, {4544,15152}, {4521,15205}, {4498,15259}, {4476,15311}, {4454,15364}, {4434,15413}, {4414,15462}, {4395,15509}, {4376,15557}, {4359,15601}, {4342,15645}, {4326,15687}, {4311,15727}, {4297,15765}, {4283,15803}, {4270,15838}, {4258,15872}, {4246,15906}, {4235,15937}, {4225,15966}, {4216,15992}, {4206,16021}, {4198,16045}, {4190,16069}, {4183,16090}, {4176,16111}, {4169,16133}, {4163,16152}, {4158,16168}, {4152,16187}, {4148,16200}, {4143,16216}, {4139,16229}, {4135,16242}, {4132,16252}, {4129,16262}, {4126,16273}, {4123,16283}, {4120,16293}, {4118,16300}, {4116,16307}, {4114,16315}, {4112,16322}, {4111,16325}, {4109,16333}, {4108,16336}, {4107,16340}, {4106,16344}, {4105,16348}, {4104,16351}, {4103,16355}, {4102,16359}, {4102,16359}, {4101,16363}, {4100,16367}};
    // la=5
    //  std::vector<std::vector<uint64_t>> data = {{0, 0}, {12, 28}, {23, 15}, {2, 25}, {12, 26}, {21, 19}, {27, 4}, {31, 12}, {0, 13}, {30, 6}, {25, 23}, {16, 2}, {3, 6}, {18, 4}, {28, 27}, {2, 14}, {2, 27}, {29, 4}, {19, 8}, {3, 9}, {13, 7}, {17, 2}, {15, 27}, {7, 18}, {25, 8}, {5, 29}, {10, 17}, {10, 6}, {3, 27}, {22, 17}, {2, 9}, {9, 2}, {10, 29}, {5, 27}, {26, 27}, {10, 30}, {21, 5}, {26, 14}, {26, 28}, {22, 13}, {12, 2}, {31, 27}, {13, 24}, {23, 26}, {29, 31}, {31, 9}, {31, 24}, {27, 10}, {20, 1}, {10, 27}, {30, 26}, {16, 28}, {31, 3}, {13, 13}, {25, 26}, {3, 11}, {12, 30}, {20, 21}, {27, 15}, {1, 11}, {7, 9}, {12, 10}, {17, 12}, {21, 16}, {26, 22}, {30, 29}, {3, 5}, {7, 15}, {12, 25}, {18, 3}, {23, 14}, {30, 26}, {4, 5}, {12, 17}, {20, 28}, {29, 7}, {6, 17}, {16, 27}, {27, 4}, {7, 13}, {20, 20}, {1, 27}, {15, 1}, {30, 6}, {14, 9}, {30, 12}, {15, 13}, {1, 14}, {20, 13}, {7, 10}, {27, 7}, {16, 2}, {6, 29}, {28, 22}, {18, 14}, {10, 4}, {1, 26}, {26, 14}, {19, 1}, {12, 20}, {6, 5}, {0, 21}, {26, 4}, {21, 19}, {17, 0}, {13, 12}, {9, 24}, {5, 3}, {2, 13}, {30, 23}, {28, 0}, {25, 8}, {23, 15}, {20, 23}, {18, 29}, {17, 3}, {15, 9}, {13, 14}, {12, 18}, {11, 23}, {10, 27}, {9, 30}, {8, 1}, {7, 4}, {6, 7}, {5, 10}, {5, 12}, {4, 14}};
    // la=6  lb=13
    //  std::vector<std::vector<uint64_t>> data = {{0,0}, {1,16382}, {2,16374}, {3,16362}, {4,16344}, {4,16344}, {5,16321}, {6,16293}, {7,16260}, {7,16260}, {8,16221}, {9,16177}, {9,16177}, {10,16127}, {11,16070}, {11,16071}, {12,16008}, {13,15939}, {13,15939}, {14,15863}, {14,15864}, {15,15780}, {15,15780}, {16,15688}, {16,15688}, {17,15588}, {17,15587}, {17,15587}, {18,15475}, {18,15475}, {18,15475}, {19,15351}, {19,15350}, {19,15349}, {19,15350}, {19,15351}, {20,15207}, {20,15205}, {20,15204}, {20,15203}, {20,15203}, {20,15203}, {20,15203}, {20,15204}, {20,15204}, {20,15204}, {20,15205}, {20,15205}, {20,15206}, {20,15206}, {20,15206}, {20,15206}, {20,15205}, {20,15204}, {20,15203}, {20,15202}, {20,15200}, {19,15428}, {19,15429}, {19,15430}, {19,15431}, {19,15431}, {19,15431}, {19,15430}, {19,15429}, {19,15427}, {18,15692}, {18,15693}, {18,15694}, {18,15695}, {18,15696}, {18,15695}, {18,15695}, {18,15694}, {18,15693}, {18,15692}, {17,15996}, {17,15997}, {17,15999}, {17,16000}, {17,16001}, {17,16001}, {17,16002}, {17,16002}, {17,16001}, {17,16001}, {17,16000}, {17,15999}, {17,15998}, {17,15996}, {17,15995}, {17,15993}, {17,15991}, {16,16363}, {16,16365}, {16,16367}, {16,16368}, {16,16370}, {16,16371}, {16,16372}, {16,16373}, {16,16374}, {16,16375}, {16,16376}, {16,16377}, {16,16377}, {16,16378}, {16,16379}, {16,16379}, {16,16380}, {16,16380}, {16,16380}, {16,16381}, {16,16381}, {16,16381}, {16,16382}, {16,16382}, {16,16382}, {16,16382}, {16,16383}, {16,16383}, {16,16383}, {16,16383}, {16,16383}, {16,16383}, {16,16383}, {16,16383}, {16,16383}};
    // la=6 lb=10 size=128
    // std::vector<std::vector<uint64_t>> data = {{0, 0}, {1, 0}, {2, 1023}, {3, 1021}, {4, 1019}, {4, 1019}, {5, 1016}, {6, 1013}, {7, 1009}, {7, 1009}, {8, 1004}, {9, 998}, {9, 998}, {10, 992}, {11, 985}, {11, 985}, {12, 977}, {13, 968}, {13, 968}, {14, 959}, {14, 959}, {15, 948}, {15, 949}, {16, 937}, {16, 937}, {17, 924}, {17, 924}, {17, 924}, {18, 910}, {18, 910}, {18, 910}, {19, 895}, {19, 895}, {19, 895}, {19, 895}, {19, 895}, {20, 877}, {20, 877}, {20, 876}, {20, 876}, {20, 876}, {20, 876}, {20, 876}, {20, 876}, {20, 876}, {20, 877}, {20, 877}, {20, 877}, {20, 877}, {20, 877}, {20, 877}, {20, 877}, {20, 877}, {20, 877}, {20, 876}, {20, 876}, {20, 876}, {19, 904}, {19, 905}, {19, 905}, {19, 905}, {19, 905}, {19, 905}, {19, 905}, {19, 905}, {19, 904}, {18, 937}, {18, 938}, {18, 938}, {18, 938}, {18, 938}, {18, 938}, {18, 938}, {18, 938}, {18, 938}, {18, 937}, {17, 975}, {17, 976}, {17, 976}, {17, 976}, {17, 976}, {17, 976}, {17, 976}, {17, 976}, {17, 976}, {17, 976}, {17, 976}, {17, 976}, {17, 976}, {17, 976}, {17, 975}, {17, 975}, {17, 975}, {16, 1021}, {16, 1022}, {16, 1022}, {16, 1022}, {16, 1022}, {16, 1022}, {16, 1023}, {16, 1023}, {16, 1023}, {16, 1023}, {16, 1023}, {16, 1023}, {16, 1023}, {16, 1023}, {16, 1023}, {16, 1023}, {16, 1023}, {16, 0}, {16, 0}, {16, 0}, {16, 0}, {16, 0}, {16, 0}, {16, 0}, {16, 0}, {16, 0}, {16, 0}, {16, 0}, {16, 0}, {16, 0}, {16, 0}, {16, 0}, {16, 0}, {16, 0}, {16, 0}};
    // la=6 lb=10 size=64
    std::vector<std::vector<uint64_t>> data = {{1, 0}, {2, 1023}, {4, 1019}, {6, 1013}, {7, 1009}, {8, 1004}, {10, 992}, {11, 985}, {12, 977}, {14, 959}, {15, 949}, {16, 937}, {16, 937}, {17, 924}, {18, 910}, {18, 911}, {19, 895}, {19, 895}, {20, 877}, {20, 876}, {20, 876}, {20, 876}, {20, 877}, {20, 877}, {20, 877}, {20, 877}, {20, 877}, {20, 876}, {19, 904}, {19, 905}, {19, 905}, {19, 905}, {19, 905}, {18, 938}, {18, 938}, {18, 938}, {18, 938}, {18, 938}, {17, 976}, {17, 976}, {17, 976}, {17, 976}, {17, 976}, {17, 976}, {17, 976}, {17, 975}, {16, 1021}, {16, 1022}, {16, 1022}, {16, 1022}, {16, 1023}, {16, 1023}, {16, 1023}, {16, 1023}, {16, 1023}, {16, 0}, {16, 0}, {16, 0}, {16, 0}, {16, 0}, {16, 0}, {16, 0}, {16, 0}, {16, 0}};
    // la=6 lb=10 size=32
    // std::vector<std::vector<uint64_t>> data = {{2,1023}, {5,1017}, {8,1004}, {10,992}, {13,969}, {15,949}, {17,925}, {18,911}, {19,895}, {20,877}, {20,876}, {20,877}, {20,877}, {20,876}, {19,904}, {19,905}, {19,904}, {18,938}, {18,938}, {17,976}, {17,976}, {17,976}, {17,975}, {16,1021}, {16,1022}, {16,1023}, {16,1023}, {16,1023}, {16,0}, {16,0}, {16,0}, {16,0}};
    // la=6 lb=10 size=16
    // std::vector<std::vector<uint64_t>> data = {{3,1023}, {9,1000}, {14,960}, {18,912}, {19,896}, {20,877}, {20,876}, {19,904}, {18,937}, {18,937}, {17,976}, {17,975}, {16,1022}, {16,1023}, {16,0}, {16,0}};
    // la=6 lb=10 avgULP
    //  std::vector<std::vector<uint64_t>> data = {{0,0}, {1,0}, {2,1023}, {3,1021}, {4,1019}, {4,1019}, {5,1016}, {6,1013}, {7,1009}, {7,1009}, {8,1004}, {9,998}, {9,998}, {10,992}, {11,985}, {11,985}, {12,977}, {13,968}, {13,968}, {14,959}, {14,959}, {15,948}, {15,949}, {16,937}, {16,937}, {17,924}, {17,924}, {17,924}, {18,910}, {18,910}, {18,910}, {19,895}, {19,895}, {19,895}, {19,895}, {19,895}, {20,877}, {20,877}, {20,876}, {20,876}, {20,876}, {20,876}, {20,876}, {20,876}, {20,876}, {20,877}, {20,877}, {20,877}, {20,877}, {20,877}, {20,877}, {20,877}, {20,877}, {20,877}, {20,876}, {20,876}, {20,876}, {19,904}, {19,905}, {19,905}, {19,905}, {19,905}, {19,905}, {19,905}, {19,905}, {19,904}, {18,937}, {18,938}, {18,938}, {18,938}, {18,938}, {18,938}, {18,938}, {18,938}, {18,938}, {18,937}, {17,975}, {17,976}, {17,976}, {17,976}, {17,976}, {17,976}, {17,976}, {17,976}, {17,976}, {17,976}, {17,976}, {17,976}, {17,976}, {17,976}, {17,975}, {17,975}, {17,975}, {16,1021}, {16,1022}, {16,1022}, {16,1022}, {16,1022}, {16,1022}, {16,1023}, {16,1023}, {16,1023}, {16,1023}, {16,1023}, {16,1023}, {16,1023}, {16,1023}, {16,1023}, {16,1023}, {16,1023}, {16,0}, {16,0}, {16,0}, {16,0}, {16,0}, {16,0}, {16,0}, {16,0}, {16,0}, {16,0}, {16,0}, {16,0}, {16,0}, {16,0}, {16,0}, {16,0}, {16,0}, {16,0}};

    // la=6 lb=10 maxULP
    //  std::vector<std::vector<uint64_t>> data = {{0,0}, {1,0}, {2,1023}, {3,1021}, {4,1019}, {4,1019}, {5,1016}, {6,1013}, {7,1009}, {7,1009}, {8,1004}, {9,998}, {9,998}, {10,992}, {11,985}, {11,985}, {12,977}, {13,968}, {13,968}, {14,959}, {14,959}, {15,948}, {15,949}, {16,937}, {16,937}, {17,924}, {17,924}, {17,924}, {18,910}, {18,910}, {18,910}, {19,895}, {19,895}, {19,895}, {19,895}, {19,895}, {20,877}, {20,877}, {20,876}, {20,876}, {20,876}, {20,876}, {20,876}, {20,876}, {20,876}, {20,877}, {20,877}, {20,877}, {20,877}, {20,877}, {20,877}, {20,877}, {20,877}, {20,877}, {20,876}, {20,876}, {20,876}, {19,904}, {19,905}, {19,905}, {19,905}, {19,905}, {19,905}, {19,905}, {19,905}, {19,904}, {18,937}, {18,938}, {18,938}, {18,938}, {18,938}, {18,938}, {18,938}, {18,938}, {18,938}, {18,937}, {17,975}, {17,976}, {17,976}, {17,976}, {17,976}, {17,976}, {17,976}, {17,976}, {17,976}, {17,976}, {17,976}, {17,976}, {17,976}, {17,976}, {17,975}, {17,975}, {17,975}, {16,1021}, {16,1022}, {16,1022}, {16,1022}, {16,1022}, {16,1022}, {16,1023}, {16,1023}, {16,1023}, {16,1023}, {16,1023}, {16,1023}, {16,1023}, {16,1023}, {16,1023}, {16,1023}, {16,1023}, {16,0}, {16,0}, {16,0}, {16,0}, {16,0}, {16,0}, {16,0}, {16,0}, {16,0}, {16,0}, {16,0}, {16,0}, {16,0}, {16,0}, {16,0}, {16,0}, {16,0}, {16,0}};
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
    // for (size_t i = 0; i < dim; i++)
    // {
    //     outtrunc1[i] = 0;
    // }

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
            std::cout << "(outtrunc[i] + outtrunc1[i])" << i << "] = " << (outtrunc[i] + outtrunc1[i]) << std::endl;
            std::cout << "((1ULL << s) - 1)" << ((1ULL << s) - 1) << std::endl;
        }

        for (int i = 0; i < dim; i++)
        {
            std::cout << "outtrunc[" << i << "] = " << outtrunc[i] << std::endl;
            std::cout << "outtrunc_a[" << i << "] = " << outtrunc_a[i] << std::endl;
        }
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
    if (party != ALICE)
        std::cout << "b_bob[" << 0 << "] = " << b_bob[0] << std::endl;

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
            std::cout << "inA_h[" << 0 << "] = " << inA_h[0] << std::endl;
            std::cout << "a_alice[" << 0 << "] = " << a_alice[0] << std::endl;
            // prod->hadamard_product_MSB(dim, a_alice, EMUX_output_x, outax, la, bwL, la + bwL, true, true, mode, msb1, msb2);
            prod->hadamard_product(dim, a_alice, EMUX_output_x, outax, la, bwL, la + bwL, true, true, mode, msb1, msb2);
        }
        else
        {
            std::cout << "inB_h[" << 0 << "] = " << inB_h[0] << std::endl;
            std::cout << "a_bob[" << 0 << "] = " << a_bob[0] << std::endl;
            // prod->hadamard_product_MSB(dim, a_bob, EMUX_output_x, outax, la, bwL, la + bwL, true, true, mode, msb1, msb2);
            prod->hadamard_product(dim, a_bob, EMUX_output_x, outax, la, bwL, la + bwL, true, true, mode, msb1, msb2);
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
    /////////////////////////
    // if (party == ALICE)
    // {
    //     iopack->io->send_data(outax, dim * sizeof(uint64_t));
    // }
    // else
    // {
    //     uint64_t *recv_outax = new uint64_t[dim];
    //     iopack->io->recv_data(recv_outax, dim * sizeof(uint64_t));

    //     for (int i = 0; i < dim; i++)
    //     {
    //         std::cout << "outax[" << i << "] =  " << (outax[i] & mask_lla) << std::endl;
    //         std::cout << "recv_outax[" << i << "] =  " << (recv_outax[i] & mask_lla) << std::endl;
    //         std::cout << "total outax[" << i << "] =  " << ((outax[i] + recv_outax[i]) & mask_lla) << std::endl;
    //     }
    //     // std::cout << "total outax =  " << ((outax[0] + recv_outax[0]) & mask_lah1) << std::endl;
    // }

    // }

    std::cout << "\n=========STEP8 ax truncate from l+la to l+1  ===========" << std::endl; // 跟协议对不上，这里直接得到了axl
    //////////////////////////////////////////////////////// general版本：直接截取，不用截断协议；高精度版本：使用截断协议
    uint64_t *mid_ax = new uint64_t[dim];
    uint64_t STEP8_comm_start = iopack->get_comm();
    if (acc == 2)
    {
        if (party == ALICE)
        {
            trunc_oracle->truncate(dim, outax, mid_ax, la - 1, bwL, true, msbA);
        }
        else
        {
            trunc_oracle->truncate(dim, outax, mid_ax, la - 1, bwL, true, msbB);
        }
        for (int i = 0; i < dim; i++)
        {
            outax[i] = mid_ax[i];
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

    uint64_t s_extend_comm_start = iopack->get_comm();
    if (party == ALICE)
    {
        ext->s_extend(dim, b_alice, b_SExt, lb, bwL, msbA);
    }
    else
    {
        ext->s_extend(dim, b_bob, b_SExt, lb, bwL, msbB);
    }
    uint64_t s_extend_comm_end = iopack->get_comm();
    
    std::cout << "b_SExt[" << 0 << "] = " << b_SExt[0] << std::endl;

    // if (party == ALICE)
    // {
    //     iopack->io->send_data(b_SExt, dim * sizeof(uint64_t));
    // }
    // else
    // {
    //     uint64_t *recv_b_SExt = new uint64_t[dim];
    //     iopack->io->recv_data(recv_b_SExt, dim * sizeof(uint64_t));
    //     for (int i = 0; i < dim; i++)
    //     {
    //         std::cout << "total b_SExt[" << i << "] =  " << ((b_SExt[i] + recv_b_SExt[i]) & mask_bwL) << std::endl;
    //     }
    // }

    std::cout << "\n=========STEP12 Caculate z=ax+b   ===========" << std::endl;
    uint64_t *z = new uint64_t[dim];

    std::cout << "b_SExt[" << 0 << "] = " << b_SExt[0] << std::endl;
    std::cout << "outax[" << 0 << "] = " << outax[0] << std::endl;

    for (int i = 0; i < dim; i++)
        z[i] = ((outax[i] + b_SExt[i] * static_cast<uint64_t>(std::pow(2, f - lb + 1))) & mask_bwL);

    std::cout << "z[" << 0 << "] = " << z[0] << std::endl;

    // if (party == ALICE)
    // {
    //     iopack->io->send_data(z, dim * sizeof(uint64_t));
    // }
    // else
    // {
    //     uint64_t *recv_z = new uint64_t[dim];
    //     iopack->io->recv_data(recv_z, dim * sizeof(uint64_t));
    //     std::cout << "total recv_z =  " << ((z[0] + recv_z[0]) & mask_bwL) << std::endl;
    // }

    std::cout << "\n=========STEP14 Drelu |x|-a  to learn b' ===========" << std::endl;
    // 去掉13，修改14
    uint8_t *Drelu_ = new uint8_t[dim];
    uint8_t *DreluMSB = new uint8_t[dim];
    uint64_t STEP14_comm_start = iopack->get_comm();
    if (party == ALICE)
    {
        for (int i = 0; i < dim; i++)
        {
            EMUX_output_x[i] = (EMUX_output_x[i] - alpha) & mask_bwL;
            std::cout << "EMUX_output_x[i] A =  " << EMUX_output_x[i] << std::endl;
            EMUX_output_x[i] = (EMUX_output_x[i] >> Tk) & mask_l_Tk;
            std::cout << "EMUX_output_x[i] A trun =  " << EMUX_output_x[i] << std::endl;
        }
        prod->aux->MSBsec(EMUX_output_x, DreluMSB, dim, bwL - Tk);
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
    std::cout << "\n=========STEP15 get x_half ===========" << std::endl;

    // online
    uint64_t *xhalf = new uint64_t[dim];
    if (acc == 2)
    {
        if (party == ALICE)
        {
            trunc_oracle->truncate(dim, inA, xhalf, 1, bwL, true, msbA);
        }
        else
        {
            trunc_oracle->truncate(dim, inB, xhalf, 1, bwL, true, msbB);
        }
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

    std::cout << "xhalf[" << 0 << "] =" << xhalf[0] << std::endl;

    std::cout << "\n=========STEP16 get delta = z-x_half ===========" << std::endl;

    uint64_t *delta = new uint64_t[dim];
    for (int i = 0; i < dim; i++)
    {
        delta[i] = (xhalf[i] - z[i]) & mask_bwL;
    }

    std::cout << "\n=========STEP17 |g|=delta_ + x_half ===========" << std::endl;
    uint64_t *delta_ = new uint64_t[dim];
    for (int i = 0; i < dim; i++)
    {
        delta_[i] = 0;
    }
    aux->multiplexer(Drelu_, delta, delta_, dim, bwL, bwL);
    // std::cout << "MUX_output_u[" << 0 << "] =" << MUX_output_u[0] << std::endl;
    uint64_t *MUX_output_g = new uint64_t[dim];
    for (int i = 0; i < dim; i++)
    {
        MUX_output_g[i] = (delta_[i] + z[i]) & mask_bwL;
        std::cout << "MUX_output_g[" << i << "] =" << MUX_output_g[i] << std::endl;
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
    std::cout << "\n=========STEP19 y = xhalf + u + v ===========" << std::endl;

    uint64_t *y = new uint64_t[dim];

    for (int i = 0; i < dim; i++)
    {
        y[i] = (xhalf[i] + MUX_output_g[i]) & mask_bwL;
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
        // std::cout << "total y = y0 + y1 =  " << ((y[0] + recv_y[0]) & mask_bwL) << ", real num: " << (double)decode_ring((y[0] + recv_y[0])&mask_bwL,37) / f_pow << std::endl;

        // std::cout << "ax +b =  " << (((inA[0] + inB[0]) * a_bob[0] + b_bob[0]) & mask_bwL) << std::endl;
        // std::cout << "ax +b  >> 12=  " << ((((inA[0] + inB[0]) * a_bob[0] + b_bob[0]) & mask_bwL) >> 12) << std::endl;
        // std::cout << "The result should be calculate_GELU = " << calculate_GELU(inA[0] + inB[0]) << std::endl;
        std::vector<double> x_values, y_values;
        std::vector<double> x_real, y_real;
        double *ULPs = new double[dim];
        double f_pow = pow(2, f);
        for (int i = 0; i < dim; i++)
        {
            std::cout << "total y = y0 + y1 =  " << ((y[i] + recv_y[i]) & mask_bwL) << ", real num: " << (double)decode_ring((y[i] + recv_y[i]) & mask_bwL, bwL) / f_pow << std::endl;

            // std::cout << "ax +b =  " << (((inA[i] + inB[i]) * a_bob[i] + b_bob[i]) & mask_bwL) << std::endl;
            // std::cout << "ax +b  >> 12=  " << ((((inA[i] + inB[i]) * a_bob[i] + b_bob[i]) & mask_bwL) >> 12) << std::endl;
            std::cout << "The result " << inA[i] + inB[i] << " should be calculate_GELU = " << calculate_GELU(inA[i] + inB[i]) << std::endl;
            ULPs[i] = abs((((double)decode_ring((y[i] + recv_y[i]) & mask_bwL, bwL) / f_pow) - calculate_GELU(inA[i] + inB[i])) / 0.000244140625);
            std::cout << "The ULP is = " << ULPs[i] << std::endl;

            x_values.push_back((inA[i] + inB[i]) / (uint64_t)f_pow);
            y_values.push_back((double)decode_ring((y[i] + recv_y[i]) & mask_bwL, bwL) / (uint64_t)f_pow);
            x_real.push_back((inA[i] + inB[i]) / (uint64_t)f_pow);
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
    uint64_t crossterm_comm_start = iopack->get_comm();
    uint64_t *outCC = new uint64_t[dim];
    trunc_oracle->cross_term(dim, inA, inB, outCC, la, bwL, la + bwL);
    uint64_t crossterm_comm_end = iopack->get_comm();
    std::cout << "crossterm_comm: " << crossterm_comm_end - crossterm_comm_start << std::endl;

    uint64_t crossterm_reverse_comm_start = iopack->get_comm();
    uint64_t *outCCC = new uint64_t[dim];
    trunc_oracle->cross_term_reverse(dim, inA, inB, outCCC, la, bwL, la + bwL);
    uint64_t crossterm_reverse_comm_end = iopack->get_comm();
    std::cout << "cross_term_reverse: " << crossterm_reverse_comm_end - crossterm_reverse_comm_start << std::endl;

    uint64_t unsigned_mul_reverse_comm_start = iopack->get_comm();
    uint64_t *outCCCC = new uint64_t[dim];
    trunc_oracle->unsigned_mul(dim, inA, inB, outCCCC, la, bwL, la + bwL);
    uint64_t unsigned_mul_reverse_comm_end = iopack->get_comm();
    std::cout << "unsigned_mul: " << unsigned_mul_reverse_comm_end - unsigned_mul_reverse_comm_start << std::endl;

    // uint8_t *tmp_msbA = new uint8_t[dim];

    // uint8_t *wA = new uint8_t[dim];
    // uint8_t *wA1 = new uint8_t[dim];
    // if (party == ALICE)
    // {
    //     aux->MSB_to_Wrap(outax, msb1, wA, dim, bwL);
    //     aux->knowMSB_to_Wrap(outax, msb1, wA1, dim, bwL);
    // }
    // else
    // {
    //     aux->MSB_to_Wrap(outax, msb2, wA, dim, bwL);
    //     aux->knowMSB_to_Wrap(outax, msb2, wA1, dim, bwL);
    // }

    // for (int i = 0; i < dim; i++)
    // {
    //     std::cout << "wA[" << i << "] = " << static_cast<int>(wA[i]) << std::endl;
    //     std::cout << "wA1[" << i << "] = " << static_cast<int>(wA1[i]) << std::endl;
    // }

    ///////////输出时间和通信

    cout << "STEP3 DRELU Bytes Sent: " << (STEP3_comm_end - STEP3_comm_start)/dim*8 << " bytes" << endl;
    cout << "STEP4 EMUX Bytes Sent: " << (STEP4_comm_end - STEP4_comm_start) /dim*8<< " bytes" << endl;
    cout << "STEP5 trun&reduce Bytes Sent: " << (STEP5_comm_end - STEP5_comm_start) /dim*8<< " bytes" << endl;
    cout << "STEP6 LUT*2 Bytes Sent: " << (STEP6_comm_end - STEP6_comm_start) /dim*8<< " bytes" << endl;
    cout << "STEP7 SMUL Bytes Sent: " << (STEP7_comm_end - STEP7_comm_start) /dim*8<< " bytes" << endl;
    cout << "STEP8 Trunc Bytes Sent: " << (STEP8_comm_end - STEP8_comm_start) /dim*8<< " bytes" << endl;
    std::cout << "s_extend_comm: " <<( s_extend_comm_end - s_extend_comm_start)/dim*8 << std::endl;
    cout << "STEP14 DRELUsec Bytes Sent: " << (STEP14_comm_end - STEP14_comm_start)/dim*8 << " bytes" << endl;
    // cout << "STEP3 Bytes Sent: " << (comm_end - comm_start) << "bytes" << endl;
    // cout << "STEP3 Bytes Sent: " << (comm_end - comm_start) << "bytes" << endl;
    // cout << "STEP3 Bytes Sent: " << (comm_end - comm_start) << "bytes" << endl;
    // cout << "STEP3 Bytes Sent: " << (comm_end - comm_start) << "bytes" << endl;
    // cout << "STEP3 Bytes Sent: " << (comm_end - comm_start) << "bytes" << endl;
    // uint64_t comm_end = iopack->get_comm();
    cout << "Total Bytes Sent: " << (comm_end - comm_start) /dim*8<< " bytes" << endl;

    auto time_end = chrono::high_resolution_clock::now();
    cout << "Total time: "
         << chrono::duration_cast<chrono::milliseconds>(time_end - time_start).count()
         << " ms" << endl;
    delete[] inA;
    delete[] inB;
    delete[] outax;
    delete prod;
}
