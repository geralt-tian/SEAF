/*
Authors: Mayank Rathee, Deevashwer Rathee
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

#include "BuildingBlocks/truncation.h"
#include "BuildingBlocks/value-extension.h"

using namespace std;
using namespace sci;

Truncation::Truncation(int party, IOPack *iopack, OTPack *otpack)
{
  this->party = party;
  this->iopack = iopack;
  this->otpack = otpack;
  this->aux = new AuxProtocols(party, iopack, otpack);
  this->mill = this->aux->mill;
  this->mill_eq = new MillionaireWithEquality(party, iopack, otpack);
  this->eq = new Equality(party, iopack, otpack);
  this->triple_gen = this->mill->triple_gen;
}

Truncation::~Truncation()
{
  delete this->aux;
  delete this->mill_eq;
  delete this->eq;
}

void Truncation::div_pow2(int32_t dim, uint64_t *inA, uint64_t *outB,
                          int32_t shift, int32_t bw, bool signed_arithmetic,
                          uint8_t *msb_x)
{
  if (signed_arithmetic == false)
  {
    truncate(dim, inA, outB, shift, bw, false, msb_x);
    return;
  }
  if (shift == 0)
  {
    memcpy(outB, inA, sizeof(uint64_t) * dim);
    return;
  }
  assert((bw - shift) > 0 && "Division shouldn't truncate the full bitwidth");
  assert(signed_arithmetic && (bw - shift - 1 >= 0));
  assert((msb_x == nullptr) && "Not yet implemented");
  assert(inA != outB);

  uint64_t mask_bw = (bw == 64 ? -1 : ((1ULL << bw) - 1));
  uint64_t mask_shift = (shift == 64 ? -1 : ((1ULL << shift) - 1));
  // mask_upper extracts the upper bw-shift-1 bits after the MSB
  uint64_t mask_upper =
      ((bw - shift - 1) == 64 ? -1 : ((1ULL << (bw - shift - 1)) - 1));

  uint64_t *inA_orig = new uint64_t[dim];

  if (party == ALICE)
  {
    for (int i = 0; i < dim; i++)
    {
      inA_orig[i] = inA[i];
      inA[i] = ((inA[i] + (1ULL << (bw - 1))) & mask_bw);
    }
  }

  uint64_t *inA_lower = new uint64_t[dim];
  uint64_t *inA_upper = new uint64_t[dim];
  uint8_t *wrap_lower = new uint8_t[dim];
  uint8_t *wrap_upper = new uint8_t[dim];
  uint8_t *msb_upper = new uint8_t[dim];
  uint8_t *zero_test_lower = new uint8_t[dim];
  uint8_t *eq_upper = new uint8_t[dim];
  uint8_t *and_upper = new uint8_t[dim];
  uint8_t *div_correction = new uint8_t[dim];
  for (int i = 0; i < dim; i++)
  {
    inA_lower[i] = inA[i] & mask_shift;
    inA_upper[i] = (inA[i] >> shift) & mask_upper;
    if (party == BOB)
    {
      inA_upper[i] = (mask_upper - inA_upper[i]) & mask_upper;
    }
  }

  this->aux->wrap_computation(inA_lower, wrap_lower, dim, shift);
  for (int i = 0; i < dim; i++)
  {
    if (party == BOB)
    {
      inA_lower[i] = (-1 * inA_lower[i]) & mask_shift;
    }
  }
  this->eq->check_equality(zero_test_lower, inA_lower, dim, shift);
  this->mill_eq->compare_with_eq(msb_upper, eq_upper, inA_upper, dim,
                                 (bw - shift - 1));
  this->aux->AND(wrap_lower, eq_upper, and_upper, dim);
  for (int i = 0; i < dim; i++)
  {
    msb_upper[i] = (msb_upper[i] ^ and_upper[i] ^ (inA[i] >> (bw - 1))) & 1;
  }
  this->aux->MSB_to_Wrap(inA, msb_upper, wrap_upper, dim, bw);
  // negate zero_test_lower and msb_upper
  // if signed_arithmetic == true, MSB of inA is flipped in the beginning
  // if MSB was 1, and the lower shift bits were not all 0, add 1 as
  // div_correction
  for (int i = 0; i < dim; i++)
  {
    if (party == ALICE)
    {
      zero_test_lower[i] ^= 1;
      msb_upper[i] ^= 1;
    }
  }
  this->aux->AND(zero_test_lower, msb_upper, div_correction, dim);

  uint64_t *arith_wrap_upper = new uint64_t[dim];
  uint64_t *arith_wrap_lower = new uint64_t[dim];
  uint64_t *arith_div_correction = new uint64_t[dim];
  this->aux->B2A(wrap_upper, arith_wrap_upper, dim, shift);
  this->aux->B2A(wrap_lower, arith_wrap_lower, dim, bw);
  this->aux->B2A(div_correction, arith_div_correction, dim, bw);

  for (int i = 0; i < dim; i++)
  {
    outB[i] =
        ((inA[i] >> shift) + arith_div_correction[i] + arith_wrap_lower[i] -
         (1ULL << (bw - shift)) * arith_wrap_upper[i]) &
        mask_bw;
  }

  if (signed_arithmetic && (party == ALICE))
  {
    for (int i = 0; i < dim; i++)
    {
      outB[i] = ((outB[i] - (1ULL << (bw - shift - 1))) & mask_bw);
      inA[i] = inA_orig[i];
    }
  }
  delete[] inA_orig;
  delete[] inA_lower;
  delete[] inA_upper;
  delete[] wrap_lower;
  delete[] wrap_upper;
  delete[] msb_upper;
  delete[] zero_test_lower;
  delete[] eq_upper;
  delete[] and_upper;
  delete[] div_correction;
  delete[] arith_wrap_lower;
  delete[] arith_wrap_upper;
  delete[] arith_div_correction;

  return;
}

void Truncation::truncate(int32_t dim, uint64_t *inA, uint64_t *outB,
                          int32_t shift, int32_t bw, bool signed_arithmetic,
                          uint8_t *msb_x)
{
  if (shift == 0)
  {
    memcpy(outB, inA, sizeof(uint64_t) * dim);
    return;
  }
  assert((bw - shift) > 0 && "Truncation shouldn't truncate the full bitwidth");
  assert((signed_arithmetic && (bw - shift - 1 >= 0)) || !signed_arithmetic);
  assert(inA != outB);

  uint64_t mask_bw = (bw == 64 ? -1 : ((1ULL << bw) - 1));
  uint64_t mask_shift = (shift == 64 ? -1 : ((1ULL << shift) - 1));
  uint64_t mask_upper =
      ((bw - shift) == 64 ? -1 : ((1ULL << (bw - shift)) - 1));

  uint64_t *inA_orig = new uint64_t[dim];

  if (signed_arithmetic && (party == ALICE))
  {
    for (int i = 0; i < dim; i++)
    {
      inA_orig[i] = inA[i];
      inA[i] = ((inA[i] + (1ULL << (bw - 1))) & mask_bw);
    }
  }

  uint64_t *inA_lower = new uint64_t[dim];
  uint64_t *inA_upper = new uint64_t[dim];
  uint8_t *wrap_lower = new uint8_t[dim];
  uint8_t *wrap_upper = new uint8_t[dim];
  uint8_t *eq_upper = new uint8_t[dim];
  uint8_t *and_upper = new uint8_t[dim];
  for (int i = 0; i < dim; i++)
  {
    inA_lower[i] = inA[i] & mask_shift;
    inA_upper[i] = (inA[i] >> shift) & mask_upper;
    if (party == BOB)
    {
      inA_upper[i] = (mask_upper - inA_upper[i]) & mask_upper;
    }
  }

  this->aux->wrap_computation(inA_lower, wrap_lower, dim, shift);
  if (msb_x == nullptr)
  {
    this->mill_eq->compare_with_eq(wrap_upper, eq_upper, inA_upper, dim,
                                   (bw - shift));
    this->aux->AND(wrap_lower, eq_upper, and_upper, dim);
    for (int i = 0; i < dim; i++)
    {
      wrap_upper[i] ^= and_upper[i];
    }
  }
  else
  {
    if (signed_arithmetic)
    {
      uint8_t *inv_msb_x = new uint8_t[dim];
      for (int i = 0; i < dim; i++)
      {
        inv_msb_x[i] = msb_x[i] ^ (party == ALICE ? 1 : 0);
      }
      this->aux->MSB_to_Wrap(inA, inv_msb_x, wrap_upper, dim, bw);
      delete[] inv_msb_x;
    }
    else
    {
      this->aux->MSB_to_Wrap(inA, msb_x, wrap_upper, dim, bw);
    }
  }

  uint64_t *arith_wrap_upper = new uint64_t[dim];
  uint64_t *arith_wrap_lower = new uint64_t[dim];
  this->aux->B2A(wrap_upper, arith_wrap_upper, dim, shift);
  this->aux->B2A(wrap_lower, arith_wrap_lower, dim, bw);

  for (int i = 0; i < dim; i++)
  {
    outB[i] = (((inA[i] >> shift) & mask_upper) + arith_wrap_lower[i] -
               (1ULL << (bw - shift)) * arith_wrap_upper[i]) &
              mask_bw;
  }

  if (signed_arithmetic && (party == ALICE))
  {
    for (int i = 0; i < dim; i++)
    {
      outB[i] = ((outB[i] - (1ULL << (bw - shift - 1))) & mask_bw);
      inA[i] = inA_orig[i];
    }
  }
  delete[] inA_orig;
  delete[] inA_lower;
  delete[] inA_upper;
  delete[] wrap_lower;
  delete[] wrap_upper;
  delete[] eq_upper;
  delete[] and_upper;
  delete[] arith_wrap_lower;
  delete[] arith_wrap_upper;

  return;
}

void Truncation::truncate_red_then_ext(int32_t dim, uint64_t *inA,
                                       uint64_t *outB, int32_t shift,
                                       int32_t bw, bool signed_arithmetic,
                                       uint8_t *msb_x)
{
  if (shift == 0)
  {
    memcpy(outB, inA, dim * sizeof(uint64_t));
    return;
  }
  uint64_t *tmpB = new uint64_t[dim];
  truncate_and_reduce(dim, inA, tmpB, shift, bw);
  XTProtocol xt(this->party, this->iopack, this->otpack);
  if (signed_arithmetic)
    xt.s_extend(dim, tmpB, outB, bw - shift, bw);
  else
    xt.z_extend(dim, tmpB, outB, bw - shift, bw);

  delete[] tmpB;
  return;
}

void Truncation::truncate_and_reduce_eight_bit_wrap(int32_t dim, uint64_t *inA, uint64_t *outB, uint64_t *eight_bit_wrap,
                                     int32_t shift, int32_t bw)
{
  if (shift == 0)
  {
    memcpy(outB, inA, sizeof(uint64_t) * dim);
    return;
  }
  assert((bw - shift) > 0 && "Truncation shouldn't truncate the full bitwidth");

  uint64_t mask_bw = (bw == 64 ? -1 : ((1ULL << bw) - 1));
  uint64_t mask_shift = (shift == 64 ? -1 : ((1ULL << shift) - 1));
  uint64_t mask_out = ((bw - shift) == 64 ? -1 : ((1ULL << (bw - shift)) - 1));

  uint64_t *inA_lower = new uint64_t[dim];
  uint8_t *wrap = new uint8_t[dim];
  for (int i = 0; i < dim; i++)
  {
    inA_lower[i] = inA[i] & mask_shift;
  }

  this->aux->wrap_computation_eight_bit_wrap(inA_lower, wrap,eight_bit_wrap, dim, shift);//lower h bits

  uint64_t *arith_wrap = new uint64_t[dim];
  this->aux->B2A(wrap, arith_wrap, dim, (bw - shift));

  for (int i = 0; i < dim; i++)
  {
    outB[i] = ((inA[i] >> shift) + arith_wrap[i]) & mask_out;
  }

  delete[] inA_lower;
  delete[] wrap;
  delete[] arith_wrap;

  return;
}

void Truncation::truncate_and_reduce(int32_t dim, uint64_t *inA, uint64_t *outB,
                                     int32_t shift, int32_t bw)
{
  if (shift == 0)
  {
    memcpy(outB, inA, sizeof(uint64_t) * dim);
    return;
  }
  assert((bw - shift) > 0 && "Truncation shouldn't truncate the full bitwidth");

  uint64_t mask_bw = (bw == 64 ? -1 : ((1ULL << bw) - 1));
  uint64_t mask_shift = (shift == 64 ? -1 : ((1ULL << shift) - 1));
  uint64_t mask_out = ((bw - shift) == 64 ? -1 : ((1ULL << (bw - shift)) - 1));

  uint64_t *inA_lower = new uint64_t[dim];
  uint8_t *wrap = new uint8_t[dim];
  for (int i = 0; i < dim; i++)
  {
    inA_lower[i] = inA[i] & mask_shift;
  }

  this->aux->wrap_computation(inA_lower, wrap, dim, shift);//lower h bits

  uint64_t *arith_wrap = new uint64_t[dim];
  this->aux->B2A(wrap, arith_wrap, dim, (bw - shift));

  for (int i = 0; i < dim; i++)
  {
    outB[i] = ((inA[i] >> shift) + arith_wrap[i]) & mask_out;
  }

  delete[] inA_lower;
  delete[] wrap;
  delete[] arith_wrap;

  return;
}


void Truncation::cross_term(int32_t dim, uint64_t *inA, uint64_t *inB, uint64_t *outC, int32_t bwA, int32_t bwB,
                            int32_t bwC) {
    uint64_t mask = (1ULL << bwC) - 1;
    if (party == sci::ALICE) {
        sci::PRG128 prg;
        for (int i = 0; i < bwB; i++) {
            auto *data0 = new uint64_t[dim];
            prg.random_data(data0, dim * sizeof(uint64_t));
            for (int j = 0; j < dim; j++) {
                data0[j] = data0[j] & ((1ULL << (bwC - i)) - 1);
            }
            otpack->iknp_straight->send_cot(data0, inA, dim, bwC - i);
            for (int j = 0; j < dim; j++) {
                outC[j] += (-data0[j] * (1ULL << i));
                outC[j] &= mask;
            }
            delete[] data0;
        }
    } else {
        bool choice[bwB][dim];
        for (int i = 0; i < dim; i++) {
            bool *temp = new bool[bwB];
            int64_to_bool(temp, inB[i], bwB);
            for (int j = 0; j < bwB; j++) {
                choice[j][i] = temp[j];
            }
            delete[] temp;
        }
        for (int i = 0; i < bwB; i++) {
            auto *data = new uint64_t[dim];
            bool *c = new bool[dim];
            for (int j = 0; j < dim; j++) {
                bool *temp = new bool[bwB];
                int64_to_bool(temp, inB[j], bwB);
                c[j] = temp[i];
                //c[j] = choice[i][j];
            }
            otpack->iknp_straight->recv_cot(data, c, dim, bwC - i);
            for (int j = 0; j < dim; j++) {
                outC[j] += (data[j] * (1ULL << i));
                outC[j] &= mask;
            }
            delete[] data;
            delete[] c;
        }
    }
}

void Truncation::cross_term_reverse(int32_t dim, uint64_t *inA, uint64_t *inB, uint64_t *outC, int32_t bwA, int32_t bwB,
                                    int32_t bwC)
{
  uint64_t mask = (1ULL << bwC) - 1;
  if (party == sci::ALICE)
  {
    bool choice[bwA][dim];
    for (int i = 0; i < dim; i++)
    {
      bool *temp = new bool[bwA];
      int64_to_bool(temp, inA[i], bwA);
      for (int j = 0; j < bwA; j++)
      {
        choice[j][i] = temp[j];
      }
      delete[] temp;
    }
    for (int i = 0; i < bwA; i++)
    {
      //            auto *data = new uint64_t[dim];
      //            bool *c = new bool[dim];
      //            for (int j = 0; j < dim; j++) {
      //                c[j] = choice[i][j];
      //            }

      auto *data = new uint64_t[dim];
      bool *c = new bool[dim];
      for (int j = 0; j < dim; j++)
      {
        bool *temp = new bool[bwA];
        int64_to_bool(temp, inA[j], bwA);
        c[j] = temp[i];
        // c[j] = choice[i][j];
      }
      otpack->iknp_reversed->recv_cot(data, c, dim, bwC - i);
      for (int j = 0; j < dim; j++)
      {
        outC[j] += ((data[j] * (1ULL << i)) & mask);
        outC[j] &= mask;
      }
      delete[] data;
      delete[] c;
    }
  }
  else
  {
    sci::PRG128 prg;
    for (int i = 0; i < bwA; i++)
    {
      auto *data0 = new uint64_t[dim];
      prg.random_data(data0, dim * sizeof(uint64_t));
      otpack->iknp_reversed->send_cot(data0, inB, dim, bwC - i);
      for (int j = 0; j < dim; j++)
      {
        outC[j] += ((-data0[j] * (1ULL << i)) & mask);
        outC[j] &= mask;
      }
      delete[] data0;
    }
  }
}




void Truncation::unsigned_mul(int32_t dim, uint64_t *inA, uint64_t *inB, uint64_t *outC, int32_t bwA, int32_t bwB,
                              int32_t bwC)
{
  auto *c = new uint64_t[dim]();
  auto *d = new uint64_t[dim];
  for (int i = 0; i < dim; i++)
  {
    c[i] = 0;
    d[i] = 0;
  }
  if (bwA <= bwB)
  {
    cross_term_reverse(dim, inA, inB, c, bwA, bwB, bwC);
    cross_term(dim, inB, inA, d, bwB, bwA, bwC);
  }
  else
  {
    cross_term(dim, inA, inB, c, bwA, bwB, bwC);
    cross_term_reverse(dim, inB, inA, d, bwB, bwA, bwC);
  }
  auto *wx = new uint8_t[dim];
  auto *wy = new uint8_t[dim];
  auto *MSBx = new uint8_t[dim];
  for (int i = 0; i < dim; i++)
  {
    MSBx[i] = 0;
  }
  auto *MSBy = new uint8_t[dim];
  for (int i = 0; i < dim; i++)
  {
    MSBy[i] = 0;
  }
  // (uint64_t *x, uint8_t *msb_x, uint8_t *wrap_x,int32_t size, int32_t bw_x)
  aux->knowMSB_to_Wrap(inA,MSBx, wx, dim, bwA);
  aux->knowMSB_to_Wrap(inB,MSBx, wy, dim, bwB);
  auto *h = new uint64_t[dim];
  auto *g = new uint64_t[dim];
  aux->multiplexer(wx, inB, h, dim, bwB, bwB);
  aux->multiplexer(wy, inA, g, dim, bwA, bwA);
  uint64_t mask = (1ULL << bwC) - 1;
  for (int i = 0; i < dim; i++)
  {
    outC[i] = (inA[i] * inB[i] + c[i] + d[i] - (g[i] * (1ULL << bwB)) - (h[i] * (1ULL << bwA))) & mask;
  }
  delete[] c;
  delete[] d;
}


// void Truncation::gp_unsigned_mul(int32_t dim, uint64_t *inA, uint64_t *inB, uint64_t *outC, int32_t bwA, int32_t bwB,
//                               int32_t bwC) {
//     auto *c = new uint64_t[dim];
//     auto *d = new uint64_t[dim];
//     for (int i = 0; i < dim; i++) {
//         c[i] = 0;
//         d[i] = 0;
//     }
//     if (bwA <= bwB) {
//         cross_term_reverse(dim, inA, inB, c, bwA, bwB, bwC);
//         cross_term(dim, inB, inA, d, bwB, bwA, bwC);
//     } else {
//         cross_term(dim, inA, inB, c, bwA, bwB, bwC);
//         cross_term_reverse(dim, inB, inA, d, bwB, bwA, bwC);
//     }
//     auto *m_x = new uint64_t[dim];
//     auto *m_y = new uint64_t[dim];
//     mw(dim, inA, m_x, bwA, 2);
//     mw(dim, inB, m_y, bwB, 2);
//     auto *g = new uint64_t[dim];
//     auto *h = new uint64_t[dim];
//     mux_3(dim, inA, m_y, g, bwA);
//     mux_3(dim, inB, m_x, h, bwB);
//     uint64_t mask = (1ULL << bwC) - 1;
//     for (int i = 0; i < dim; i++) {
//         outC[i] = (inA[i] * inB[i] + c[i] + d[i] - (g[i] * (1ULL << bwB)) - (h[i] * (1ULL << bwA))) & mask;
//     }
//     delete[] c;
//     delete[] d;
// }