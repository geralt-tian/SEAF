#include "LinearOT/linear-ot.h"
#include "utils/emp-tool.h"
#include <iostream>

using namespace sci;
using namespace std;

int party, port = 32000;
string address = "127.0.0.1";
IOPack *iopack;
OTPack *otpack;
LinearOT *prod;

int dim = 200;
int bwA = 32;
int bwB = 32;
int bwC = 32;

uint64_t maskA = (bwA == 64 ? -1 : ((1ULL << bwA) - 1));
uint64_t maskB = (bwB == 64 ? -1 : ((1ULL << bwB) - 1));
uint64_t maskC = (bwC == 64 ? -1 : ((1ULL << bwC) - 1));



int main(int argc, char **argv)
{
    ArgMapping amap;
    amap.arg("r", party, "Role of party: ALICE = 1; BOB = 2");
    amap.arg("p", port, "Port Number");
    amap.arg("ip", address, "IP Address of server (ALICE)");

    amap.parse(argc, argv);

    iopack = new IOPack(party, port, address);
    otpack = new OTPack(iopack, party);

    prod = new LinearOT(party, iopack, otpack);

    PRG128 prg;

    uint64_t *inA = new uint64_t[dim];
    uint64_t *inB = new uint64_t[dim];
    uint64_t *neginB = new uint64_t[dim];
    uint64_t *in0 = new uint64_t[dim];
    uint64_t *outC = new uint64_t[dim];
    //   prg.random_data(inA, dim * sizeof(uint64_t));
    //   prg.random_data(inB, dim * sizeof(uint64_t));

    for (int i = 0; i < dim; i++)
    {
        // inA[i] &= maskA;
        inA[i] = 100;
        inB[i] = i;
        neginB[i] = -i;
        in0[i] = 0;
    }

    if(party==ALICE){
        prod->hadamard_product(dim, inA, in0, outC, bwA, bwB, bwC, true, true, MultMode::None);
    }
    else{
        prod->hadamard_product(dim, in0, neginB, outC, bwA, bwB, bwC, true, true, MultMode::None);
    }
    

    if (party == ALICE)
    {
        for (int i = 0; i < dim; i++)
        {
            outC[i] = (outC[i] + inA[i]*inA[i]) & maskC;
        }
    }
    else
    {
        for (int i = 0; i < dim; i++)
        {
            outC[i] = (outC[i] + inB[i]*inB[i]) & maskC;
        }
    }

    for (int i = 0; i < dim; i++)
    {
        std::cout << "inA[" << i << "] = " << inA[i] << std::endl;
        std::cout << "inB[" << i << "] = " << inB[i] << std::endl;
        std::cout << "outC[" << i << "] = " << outC[i] << std::endl;
    }
    // test_hadamard_product(inA, inB, false);
    // test_hadamard_product(inA, inB, true);

    delete[] inA;
    delete[] inB;
    delete prod;
}
