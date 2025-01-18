#include "BuildingBlocks/aux-protocols.h"
#include "utils/emp-tool.h"
#include <iostream>
using namespace sci;
using namespace std;

int party, port = 8000, dim = 1;
string address = "127.0.0.1";
IOPack *iopack;
OTPack *otpack;
AuxProtocols *aux;

// void AuxProtocols::multiplexer(uint8_t *sel, uint64_t *x, uint64_t *y,
//                                int32_t size, int32_t bw_x, int32_t bw_y)
// {
//   assert(bw_x <= 64 && bw_y <= 64 && bw_y <= bw_x);
//   uint64_t mask_x = (bw_x == 64 ? -1 : ((1ULL << bw_x) - 1));
//   uint64_t mask_y = (bw_y == 64 ? -1 : ((1ULL << bw_y) - 1));

//   uint64_t *corr_data = new uint64_t[size];
//   uint64_t *data_S = new uint64_t[size];
//   uint64_t *data_R = new uint64_t[size];

//   // y = (sel_0 \xor sel_1) * (x_0 + x_1)
//   // y = (sel_0 + sel_1 - 2*sel_0*sel_1)*x_0 + (sel_0 + sel_1 -
//   // 2*sel_0*sel_1)*x_1 y = [sel_0*x_0 + sel_1*(x_0 - 2*sel_0*x_0)]
//   //     + [sel_1*x_1 + sel_0*(x_1 - 2*sel_1*x_1)]
//   for (int i = 0; i < size; i++)
//   {
//     corr_data[i] = (x[i] * (1 - 2 * uint64_t(sel[i]))) & mask_y;
//   }
// #pragma omp parallel num_threads(2)
//   {
//     // uint64_t comm_start_mux = iopack->get_comm();
//     if (omp_get_thread_num() == 1)
//     {

//       if (party == sci::ALICE)
//       {
//         otpack->iknp_reversed->recv_cot(data_R, (bool *)sel, size, bw_y);
//       }
//       else
//       { // party == sci::BOB
//         otpack->iknp_reversed->send_cot(data_S, corr_data, size, bw_y);
//       }
//     }

//     else
//     {
//       if (party == sci::ALICE)
//       {
//         otpack->iknp_straight->send_cot(data_S, corr_data, size, bw_y);
//       }
//       else
//       { // party == sci::BOB
//         otpack->iknp_straight->recv_cot(data_R, (bool *)sel, size, bw_y);
//       }
//     }
//     // uint64_t comm_END_mux = iopack->get_comm();
//     // cout << "INNER MUX Bytes Sent: " << (comm_END_mux - comm_start_mux) <<"bytes"<< endl;
//   }

//   std::cout << "data_R[" << 0 << "] = " << data_R[0] << std::endl;
//   std::cout << "data_S[" << 0 << "] = " << data_S[0] << std::endl;
//   for (int i = 0; i < size; i++)
//   {
//     y[i] = ((x[i] * uint64_t(sel[i]) + data_R[i] - data_S[i]) & mask_y);
//   }

//   delete[] corr_data;
//   delete[] data_S;
//   delete[] data_R;
// }


void test_absmux() { //  根据选择信号 sel 来选择输入信号 x 或 y 之一作为输出
  int bw_x = 37, bw_y = 37;
  PRG128 prg;
  uint64_t mask_x = (bw_x == 64 ? -1 : ((1ULL << bw_x) - 1));
  uint64_t mask_y = (bw_y == 64 ? -1 : ((1ULL << bw_y) - 1));

  uint8_t *sel = new uint8_t[dim];
  uint64_t *x = new uint64_t[dim];
  uint64_t *y = new uint64_t[dim];

  // prg.random_data(sel, dim * sizeof(uint8_t));
  // prg.random_data(x, dim * sizeof(uint64_t));
    if (party==ALICE){
    sel[0] = 0;
    x[0] = 10000;
  }
  else {
    sel[0] = 1;
    x[0] = 6383;
  }
  for (int i = 0; i < dim; i++) {
    sel[i] = sel[i] & 1;
    x[i] = x[i] & mask_x;
  }

  aux->multiplexerabs(sel, x, y, dim, bw_x, bw_y);

  if (party == ALICE) {
    iopack->io->send_data(sel, dim * sizeof(uint8_t));
    iopack->io->send_data(x, dim * sizeof(uint64_t));
    iopack->io->send_data(y, dim * sizeof(uint64_t));
  } else {
    uint8_t *sel0 = new uint8_t[dim];
    uint64_t *x0 = new uint64_t[dim];
    uint64_t *y0 = new uint64_t[dim];
    iopack->io->recv_data(sel0, dim * sizeof(uint8_t));
    iopack->io->recv_data(x0, dim * sizeof(uint64_t));
    iopack->io->recv_data(y0, dim * sizeof(uint64_t));
    std::cout << "x[0] = " << (x[0]) << std::endl;
    std::cout << "x0[0] = " << (x0[0]) << std::endl;
    std::cout << "y[0] = " << (y[0]) << std::endl;
    std::cout << "y0[0] = " << (y0[0]) << std::endl;
    std::cout << "2 * (sel0[0] ^ sel[0]) - 1 = " << 2 * (sel0[0] ^ sel[0]) - 1 << std::endl;
    std::cout << "2 * (sel0[0] ^ sel[0]) - 1 = " << 2 * (sel0[0] ^ sel[0]) - 1 << std::endl;
    std::cout << "(uint64_t(2 * (sel0[0] ^ sel[0]) - 1) * ((x0[0] + x[0]) & mask_y))" << " = " << ((uint64_t(2 * (sel0[0] ^ sel[0]) - 1) * ((x0[0] + x[0]) & mask_y))& mask_y )<< std::endl;
    std::cout << "(y0[i] + y[i]) & mask_y)" << " = " << ((y0[0] + y[0]) & mask_y) << std::endl;
    for (int i = 0; i < dim; i++) {
      assert(((uint64_t(2 * (sel0[i] ^ sel[i]) - 1) * ((x0[0] + x[0]) & mask_y)) & mask_y) ==
             ((y0[i] + y[i]) & mask_y));
    }
    cout << "MUX Tests passed" << endl;

    delete[] sel0;
    delete[] x0;
    delete[] y0;
  }
  delete[] sel;
  delete[] x;
  delete[] y;
}

int main(int argc, char **argv)
{
  ArgMapping amap;
  amap.arg("r", party, "Role of party: ALICE = 1; BOB = 2");
  amap.arg("p", port, "Port Number");
  amap.arg("d", dim, "Size of vector");
  amap.arg("ip", address, "IP Address of server (ALICE)");
  amap.parse(argc, argv);

  iopack = new IOPack(party, port, "127.0.0.1");
  otpack = new OTPack(iopack, party);
  uint64_t num_rounds;

  aux = new AuxProtocols(party, iopack, otpack);

  uint64_t comm_start_msb = iopack->get_comm();

  test_absmux();
  // test_MSB_computation();
  uint64_t comm_end_msb = iopack->get_comm();
  cout << "MSB Bytes Sent: " << (comm_end_msb - comm_start_msb) << "bytes" << endl;
  // test_wrap_computation();
  // test_mux();
  // test_B2A();
  // test_lookup_table<uint8_t>();
  // test_lookup_table<uint64_t>();
  // test_MSB_to_Wrap();
  // test_AND();
  // test_digit_decomposition();
  // test_msnzb_one_hot();

  return 0;
}
