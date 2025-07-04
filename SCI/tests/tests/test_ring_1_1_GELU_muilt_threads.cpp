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

#include "Math/math-functions.h"
#include <fstream>
#include <iostream>
#include <thread>

using namespace sci;
using namespace std;

#define MAX_THREADS 1

int party, port = 32000;
int num_threads = 1;
string address = "127.0.0.1";

int dim = 1048576;
// int dim = 500;
int bw_x = 21;
int bw_y = 21;
int s_x = 12;
int s_y = 12;

uint64_t mask_x = (bw_x == 64 ? -1 : ((1ULL << bw_x) - 1));
uint64_t mask_y = (bw_y == 64 ? -1 : ((1ULL << bw_y) - 1));

IOPack *iopackArr[MAX_THREADS];
OTPack *otpackArr[MAX_THREADS];

uint64_t computeULPErr(double calc, double actual, int SCALE)
{
    int64_t calc_fixed = (double(calc) * (1ULL << SCALE));
    int64_t actual_fixed = (double(actual) * (1ULL << SCALE));
    uint64_t ulp_err = (calc_fixed - actual_fixed) > 0
                           ? (calc_fixed - actual_fixed)
                           : (actual_fixed - calc_fixed);
    return ulp_err;
}

double calculate_GELU(double value)
{
    return 0.5 * value + 0.5 * value * std::erf(value / 1.414);
}

void gelu_thread(int32_t tid, uint64_t *x, uint64_t *y, int32_t num_ops, int32_t bwL, int32_t la, int32_t lb, int32_t s, int32_t f)
{
    MathFunctions *math;
    if (tid & 1)
    {
        math = new MathFunctions(3 - party, iopackArr[tid], otpackArr[tid]);
    }
    else
    {
        math = new MathFunctions(party, iopackArr[tid], otpackArr[tid]);
    }
    math->gelu(num_ops, x, y, bwL, la, lb, s, f);

    delete math;
}

int main(int argc, char **argv)
{
    /************* Argument Parsing  ************/
    /********************************************/
    ArgMapping amap;
    amap.arg("r", party, "Role of party: ALICE = 1; BOB = 2");
    amap.arg("p", port, "Port Number");
    amap.arg("N", dim, "Number of GELU operations");
    amap.arg("nt", num_threads, "Number of threads");
    amap.arg("ip", address, "IP Address of server (ALICE)");

    amap.parse(argc, argv);

    assert(num_threads <= MAX_THREADS);
    // int32_t la = 4;
    // int32_t lb = 9;
    // int32_t la = 5;
    // int32_t lb = 11;
    int32_t la = 7;
    int32_t lb = 11;
    int32_t s = 6;
    int32_t f = 12;
    int32_t bwL = 21;
    /********** Setup IO and Base OTs ***********/
    /********************************************/
    for (int i = 0; i < num_threads; i++)
    {
        iopackArr[i] = new IOPack(party, port + i, address);
        if (i & 1)
        {
            otpackArr[i] = new OTPack(iopackArr[i], 3 - party);
        }
        else
        {
            otpackArr[i] = new OTPack(iopackArr[i], party);
        }
    }
    std::cout << "All Base OTs Done" << std::endl;

    /************ Generate Test Data ************/
    /********************************************/
    PRG128 prg;

    uint64_t *x = new uint64_t[dim];
    uint64_t *y = new uint64_t[dim];

    //   if (party == ALICE)
    // {
    //   for (int i = 0; i < dim; i++)
    //   x[i] = 0 ;
    // }
    // else
    // {
    //     for (int i = 0; i < dim; i++) {
    //   x[i] = (i - 16384) & mask_x;
    // }
    // }
    prg.random_data(x, dim * sizeof(uint64_t));

    for (int i = 0; i < dim; i++)
    {
        x[i] &= mask_x;
    }

    /************** Fork Threads ****************/
    /********************************************/
    uint64_t total_comm = 0;
    uint64_t thread_comm[num_threads];
    for (int i = 0; i < num_threads; i++)
    {
        thread_comm[i] = iopackArr[i]->get_comm();
    }
    auto start = clock_start();
    std::thread GELU_threads[num_threads];
    int chunk_size = dim / num_threads;

    for (int i = 0; i < num_threads; ++i)
    {
        int offset = i * chunk_size;
        int lnum_ops;
        if (i == (num_threads - 1))
        {
            lnum_ops = dim - offset;
        }
        else
        {
            lnum_ops = chunk_size;
        }

        GELU_threads[i] =
            std::thread(gelu_thread, i, x + offset, y + offset, lnum_ops, bwL, la, lb, s, f);
    }
    for (int i = 0; i < num_threads; ++i)
    {
        GELU_threads[i].join();
    }
    long long t = time_from(start);

    for (int i = 0; i < num_threads; i++)
    {
        thread_comm[i] = iopackArr[i]->get_comm() - thread_comm[i];
        total_comm += thread_comm[i];
    }

    /************** Verification ****************/
    /********************************************/
    if (party == ALICE)
    {
        iopackArr[0]->io->send_data(x, dim * sizeof(uint64_t));
        iopackArr[0]->io->send_data(y, dim * sizeof(uint64_t));
    }
    else
    { // party == BOB
        uint64_t *x0 = new uint64_t[dim];
        uint64_t *y0 = new uint64_t[dim];
        iopackArr[0]->io->recv_data(x0, dim * sizeof(uint64_t));
        iopackArr[0]->io->recv_data(y0, dim * sizeof(uint64_t));

        uint64_t total_err = 0;
        uint64_t max_ULP_err = 0;
        for (int i = 0; i < dim; i++)
        {
            double dbl_x = (signed_val(x0[i] + x[i], bw_x)) / double(1LL << s_x);
            double dbl_y = (signed_val(y0[i] + y[i], bw_y)) / double(1LL << s_y);
            double gelu_x = calculate_GELU(dbl_x);
            uint64_t err = computeULPErr(dbl_y, gelu_x, s_y);
            // cout << "ULP["<<i<<"] Error: " << dbl_x << "," << dbl_y << "," << gelu_x << ","
            //      << err << endl;
            total_err += err;
            max_ULP_err = std::max(max_ULP_err, err);
        }

        cerr << "Average ULP error: " << total_err / dim << endl;
        cerr << "Max ULP error: " << max_ULP_err << endl;
        cerr << "Number of tests: " << dim << endl;

        delete[] x0;
        delete[] y0;
    }

    cout << "Number of GELU/s:\t" << (double(dim) / t) * 1e6 << std::endl;
    cout << "GELU Time\t" << t / (1000.0) << " ms" << endl;
    cout << "GELU Bytes Sent\t" << total_comm << " bytes" << endl;

    /******************* Cleanup ****************/
    /********************************************/
    delete[] x;
    delete[] y;
    for (int i = 0; i < num_threads; i++)
    {
        delete iopackArr[i];
        delete otpackArr[i];
    }
}
