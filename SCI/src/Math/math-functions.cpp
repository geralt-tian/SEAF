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

using namespace std;
using namespace sci;

#define KKOT_LIMIT 8
#define SQRT_LOOKUP_SCALE 2

MathFunctions::MathFunctions(int party, IOPack *iopack, OTPack *otpack)
{
  this->party = party;
  this->iopack = iopack;
  this->otpack = otpack;
  this->aux = new AuxProtocols(party, iopack, otpack);
  this->xt = new XTProtocol(party, iopack, otpack);
  this->trunc = new Truncation(party, iopack, otpack);
  this->mult = new LinearOT(party, iopack, otpack);
  this->eq = new Equality(party, iopack, otpack);
  this->mill_eq = new MillionaireWithEquality(party, iopack, otpack);
  this->ext = new XTProtocol(party, iopack, otpack);
  this->prod = new LinearOT(party, iopack, otpack);
}

MathFunctions::~MathFunctions()
{
  delete aux;
  delete xt;
  delete trunc;
  delete mult;
  delete eq;
  delete mill_eq;
  delete ext;
  delete prod;
}

// A0 \in (1/4, 1)
uint64_t lookup_A0(uint64_t index, int m)
{
  uint64_t k = 1ULL << m;
  double p = 1 + (double(index) / double(k));
  double A1 = 1.0 / (p * (p + 1.0 / double(k)));
  int32_t scale = m + 3;
  uint64_t mask = (1ULL << scale) - 1;
  uint64_t val = uint64_t(A1 * (1ULL << scale)) & mask;
  return val;
}

// A1 \in (1/2, 1)
uint64_t lookup_A1(uint64_t index, int m)
{
  uint64_t k = 1ULL << m;
  double p = 1 + (double(index) / double(k));
  double z = (p * (p + (1.0 / double(k))));
  double A1 = ((1.0 / double(k * 2)) + sqrt(z)) / z;
  int32_t scale = 2 * m + 2;
  uint64_t mask = (1ULL << scale) - 1;
  uint64_t val = uint64_t(A1 * (1ULL << scale)) & mask;
  return val;
}

void MathFunctions::reciprocal_approximation(int32_t dim, int32_t m,
                                             uint64_t *dn, uint64_t *out,
                                             int32_t bw_dn, int32_t bw_out,
                                             int32_t s_dn, int32_t s_out)
{
  assert(bw_out == m + s_dn + 4);
  assert(s_out == m + s_dn + 4);

  uint64_t s_dn_mask = (1ULL << s_dn) - 1;
  uint64_t m_mask = (1ULL << m) - 1;
  uint64_t s_min_m_mask = (1ULL << (s_dn - m)) - 1;

  uint64_t *tmp_1 = new uint64_t[dim];
  uint64_t *tmp_2 = new uint64_t[dim];

  for (int i = 0; i < dim; i++)
  {
    tmp_1[i] = dn[i] & s_dn_mask;
  }
  trunc->truncate_and_reduce(dim, tmp_1, tmp_2, s_dn - m, s_dn);

  int M = (1ULL << m);
  uint64_t c0_mask = (1ULL << (m + 4)) - 1;
  uint64_t c1_mask = (1ULL << (2 * m + 3)) - 1;
  uint64_t *c0 = new uint64_t[dim];
  uint64_t *c1 = new uint64_t[dim];
  if (party == ALICE)
  {
    uint64_t **spec;
    spec = new uint64_t *[dim];
    PRG128 prg;
    prg.random_data(c0, dim * sizeof(uint64_t));
    prg.random_data(c1, dim * sizeof(uint64_t));
    for (int i = 0; i < dim; i++)
    {
      spec[i] = new uint64_t[M];
      c0[i] &= c0_mask;
      c1[i] &= c1_mask;
      for (int j = 0; j < M; j++)
      {
        int idx = (tmp_2[i] + j) & m_mask;
        spec[i][j] = (lookup_A0(idx, m) - c0[i]) & c0_mask;
        spec[i][j] <<= (2 * m + 3);
        spec[i][j] |= (lookup_A1(idx, m) - c1[i]) & c1_mask;
      }
    }
    aux->lookup_table<uint64_t>(spec, nullptr, nullptr, dim, m, 3 * m + 7);

    for (int i = 0; i < dim; i++)
      delete[] spec[i];
    delete[] spec;
  }
  else
  {
    aux->lookup_table<uint64_t>(nullptr, tmp_2, c1, dim, m, 3 * m + 7);

    for (int i = 0; i < dim; i++)
    {
      c0[i] = (c1[i] >> (2 * m + 3)) & c0_mask;
      c1[i] = c1[i] & c1_mask;
    }
  }
  for (int i = 0; i < dim; i++)
  {
    tmp_1[i] = dn[i] & s_min_m_mask;
  }
  uint8_t *zero_shares = new uint8_t[dim];
  for (int i = 0; i < dim; i++)
  {
    zero_shares[i] = 0;
  }

  // Unsigned mult
  mult->hadamard_product(dim, c0, tmp_1, tmp_2, m + 4, s_dn - m, s_dn + 4,
                         false, false, MultMode::None, zero_shares, nullptr);

  xt->z_extend(dim, tmp_2, tmp_1, s_dn + 4, s_dn + m + 4, zero_shares);

  uint64_t out_mask = (1ULL << (s_dn + m + 4)) - 1;
  uint64_t scale_up = (1ULL << (s_dn - m + 1));
  for (int i = 0; i < dim; i++)
  {
    out[i] = ((c1[i] * scale_up) - tmp_1[i]) & out_mask;
  }

  delete[] tmp_1;
  delete[] tmp_2;
  delete[] c0;
  delete[] c1;
  delete[] zero_shares;
}

void MathFunctions::div(int32_t dim, uint64_t *nm, uint64_t *dn, uint64_t *out,
                        int32_t bw_nm, int32_t bw_dn, int32_t bw_out,
                        int32_t s_nm, int32_t s_dn, int32_t s_out,
                        bool signed_nm, bool compute_msnzb)
{
  assert(s_out <= s_dn);

  // out_precision = iters * (2*m + 2)
  int32_t m, iters;
  m = (s_out <= 18 ? ceil((s_out - 2) / 2.0)
                   : ceil((ceil(s_out / 2.0) - 2) / 2.0));
  iters = (s_out <= 18 ? 0 : 1);

  int32_t s_tmp_dn;
  int32_t bw_adjust;
  int32_t s_adjust;
  uint64_t *tmp_dn;
  uint64_t *adjust;
  if (compute_msnzb)
  {
    s_tmp_dn = bw_dn - 1;
    bw_adjust = bw_dn + 1;
    s_adjust = bw_dn - 1 - s_dn;
    uint64_t mask_adjust = (bw_adjust == 64 ? -1 : ((1ULL << bw_adjust) - 1));
    // MSB is always 0, thus, not including it
    uint8_t *msnzb_vector_bool = new uint8_t[dim * bw_dn];
    uint64_t *msnzb_vector = new uint64_t[dim * bw_dn];
    aux->msnzb_one_hot(dn, msnzb_vector_bool, bw_dn, dim);
    aux->B2A(msnzb_vector_bool, msnzb_vector, dim * bw_dn, bw_adjust);
    // adjust: bw = bw_dn, scale = bw_dn - 1 - s_dn
    adjust = new uint64_t[dim];
    for (int i = 0; i < dim; i++)
    {
      adjust[i] = 0;
      for (int j = 0; j < bw_dn; j++)
      {
        adjust[i] += (1ULL << (bw_dn - 1 - j)) * msnzb_vector[i * bw_dn + j];
      }
      adjust[i] &= mask_adjust;
    }
    // tmp_dn: bw = bw_dn, scale = bw_dn - 1
    tmp_dn = new uint64_t[dim];
    mult->hadamard_product(dim, dn, adjust, tmp_dn, bw_dn, bw_dn + 1, bw_dn + 1,
                           false, false, MultMode::None);

    delete[] msnzb_vector_bool;
    delete[] msnzb_vector;
  }
  else
  {
    // tmp_dn: bw = s_dn + 1, scale = s_dn
    s_tmp_dn = s_dn;
    tmp_dn = dn;
  }

  uint64_t *tmp_1 = new uint64_t[dim];
  uint64_t *tmp_2 = new uint64_t[dim];
  // tmp_1: bw = s_tmp_dn + m + 4, scale = s_tmp_dn + m + 3
  reciprocal_approximation(dim, m, tmp_dn, tmp_1, bw_dn, s_tmp_dn + m + 4,
                           s_tmp_dn, s_tmp_dn + m + 4);

  uint64_t *w0 = new uint64_t[dim];
  // w0: bw = s_out + 1, scale = s_out
  trunc->truncate_and_reduce(dim, tmp_1, w0, s_tmp_dn + m + 3 - s_out,
                             s_tmp_dn + m + 4);

  uint8_t *msb_nm = new uint8_t[dim];
  aux->MSB(nm, msb_nm, dim, bw_nm);
  uint8_t *zero_shares = new uint8_t[dim];
  for (int i = 0; i < dim; i++)
  {
    zero_shares[i] = 0;
  }

  // a0: bw = bw_out, scale = s_out
  uint64_t *a0 = new uint64_t[dim];
  // Mixed mult with w0 unsigned
  mult->hadamard_product(dim, nm, w0, tmp_1, bw_nm, s_out + 1, s_out + bw_nm,
                         signed_nm, false, MultMode::None, msb_nm, zero_shares);
  trunc->truncate_and_reduce(dim, tmp_1, tmp_2, s_nm, s_out + bw_nm);
  if ((bw_nm - s_nm) >= (bw_out - s_out))
  {
    aux->reduce(dim, tmp_2, a0, bw_nm - s_nm + s_out, bw_out);
  }
  else
  {
    if (signed_nm)
    {
      xt->s_extend(dim, tmp_2, a0, s_out + bw_nm - s_nm, bw_out, msb_nm);
    }
    else
    {
      xt->z_extend(dim, tmp_2, a0, s_out + bw_nm - s_nm, bw_out, nullptr);
    }
  }

  if (compute_msnzb)
  {
    int32_t bw_tmp1 =
        (bw_out + s_adjust < bw_adjust ? bw_adjust : bw_out + s_adjust);
    // tmp_1: bw = bw_tmp1, scale = s_out + s_adjust
    mult->hadamard_product(dim, a0, adjust, tmp_1, bw_out, bw_adjust, bw_tmp1,
                           signed_nm, false, MultMode::None,
                           (signed_nm ? msb_nm : nullptr), zero_shares);
    // a0: bw = bw_out, scale = s_out
    trunc->truncate_and_reduce(dim, tmp_1, a0, s_adjust, bw_out + s_adjust);
  }

  // tmp_1: bw = s_tmp_dn + 2, scale = s_tmp_dn
  uint64_t s_plus_2_mask = (1ULL << (s_tmp_dn + 2)) - 1;
  for (int i = 0; i < dim; i++)
  {
    tmp_1[i] = tmp_dn[i] & s_plus_2_mask;
  }

  if (iters > 0)
  {
    // d0: bw = s_out + 2, scale = s_out
    uint64_t *d0 = new uint64_t[dim];
    mult->hadamard_product(dim, w0, tmp_1, tmp_2, s_out + 1, s_tmp_dn + 2,
                           s_out + s_tmp_dn + 2, false, false, MultMode::None,
                           zero_shares, zero_shares);
    trunc->truncate_and_reduce(dim, tmp_2, d0, s_tmp_dn, s_out + s_tmp_dn + 2);

    // e0: bw = s_out + 2, scale = s_out
    // e0 = 1 - d0
    uint64_t *e0 = new uint64_t[dim];
    for (int i = 0; i < dim; i++)
    {
      e0[i] = (party == ALICE ? (1ULL << (s_out)) : 0) - d0[i];
    }

    uint64_t e_mask = (1ULL << (s_out + 2)) - 1;
    uint64_t *a_curr = new uint64_t[dim];
    uint64_t *e_curr = new uint64_t[dim];
    uint64_t *a_prev = a0;
    uint64_t *e_prev = e0;
    for (int i = 0; i < iters - 1; i++)
    {
      // tmp_1: bw = 2*s_out+2, scale: 2*s_out
      mult->hadamard_product(dim, e_prev, e_prev, tmp_1, s_out + 2, s_out + 2,
                             2 * s_out + 2, true, true, MultMode::None,
                             zero_shares, zero_shares);
      // e_curr: bw = s_out + 2, scale: s_out
      trunc->truncate_and_reduce(dim, tmp_1, e_curr, s_out, 2 * s_out + 2);
      // e_prev = 1 + e_prev
      for (int j = 0; j < dim; j++)
      {
        e_prev[j] =
            ((party == ALICE ? (1ULL << (s_out)) : 0) + e_prev[j]) & e_mask;
      }
      // tmp_1: bw = bw_out + s_out, scale: 2*s_out
      mult->hadamard_product(dim, a_prev, e_prev, tmp_1, bw_out, s_out + 2,
                             bw_out + s_out, signed_nm, true, MultMode::None,
                             (signed_nm ? msb_nm : nullptr), zero_shares);
      // a_curr: bw = bw_out, scale: s_out
      trunc->truncate_and_reduce(dim, tmp_1, a_curr, s_out, bw_out + s_out);
      memcpy(a_prev, a_curr, dim * sizeof(uint64_t));
      memcpy(e_prev, e_curr, dim * sizeof(uint64_t));
    }
    // e_prev = 1 + e_prev
    for (int j = 0; j < dim; j++)
    {
      e_prev[j] =
          ((party == ALICE ? (1ULL << (s_out)) : 0) + e_prev[j]) & e_mask;
    }
    // out: bw = bw_out, scale: s_out
    // Mixed mult with e_prev unsigned
    mult->hadamard_product(dim, a_prev, e_prev, tmp_1, bw_out, s_out + 2,
                           bw_out + s_out, signed_nm, false, MultMode::None,
                           (signed_nm ? msb_nm : nullptr), zero_shares);
    trunc->truncate_and_reduce(dim, tmp_1, out, s_out, bw_out + s_out);

    delete[] d0;
    delete[] e0;
    delete[] a_curr;
    delete[] e_curr;
  }
  else
  {
    memcpy(out, a0, dim * sizeof(uint64_t));
  }

  delete[] tmp_1;
  delete[] tmp_2;
  delete[] w0;
  delete[] a0;
  delete[] msb_nm;
  delete[] zero_shares;
  if (compute_msnzb)
  {
    delete[] tmp_dn;
    delete[] adjust;
  }
}

uint64_t lookup_neg_exp(uint64_t val_in, int32_t s_in, int32_t s_out)
{
  if (s_in < 0)
  {
    s_in *= -1;
    val_in *= (1ULL << (s_in));
    s_in = 0;
  }
  uint64_t res_val =
      exp(-1.0 * (val_in / double(1ULL << s_in))) * (1ULL << s_out);
  return res_val;
}

void MathFunctions::lookup_table_exp(int32_t dim, uint64_t *x, uint64_t *y,
                                     int32_t bw_x, int32_t bw_y, int32_t s_x,
                                     int32_t s_y)
{
  assert(bw_y >= (s_y + 2));

  int LUT_size = KKOT_LIMIT;

  uint64_t bw_x_mask = (bw_x == 64 ? -1 : (1ULL << bw_x) - 1);
  uint64_t LUT_out_mask = ((s_y + 2) == 64 ? -1 : (1ULL << (s_y + 2)) - 1);

  uint64_t *tmp_1 = new uint64_t[dim];
  for (int i = 0; i < dim; i++)
  {
    tmp_1[i] = (-1 * x[i]) & bw_x_mask;
  }
  int digit_size = LUT_size;
  int num_digits = ceil(double(bw_x) / digit_size);
  int last_digit_size = bw_x - (num_digits - 1) * digit_size;
  uint64_t *x_digits = new uint64_t[num_digits * dim];

  aux->digit_decomposition_sci(dim, tmp_1, x_digits, bw_x, digit_size);

  uint64_t digit_mask = (digit_size == 64 ? -1 : (1ULL << digit_size) - 1);
  uint64_t last_digit_mask =
      (last_digit_size == 64 ? -1 : (1ULL << last_digit_size) - 1);
  int N = (1ULL << digit_size);
  int last_N = (1ULL << last_digit_size);
  int N_digits = (digit_size == last_digit_size ? num_digits : num_digits - 1);
  uint64_t *digits_exp = new uint64_t[num_digits * dim];
  if (party == ALICE)
  {
    uint64_t **spec;
    spec = new uint64_t *[num_digits * dim];
    PRG128 prg;
    prg.random_data(digits_exp, num_digits * dim * sizeof(uint64_t));
    for (int i = 0; i < N_digits * dim; i++)
    {
      int digit_idx = i / dim;
      spec[i] = new uint64_t[N];
      digits_exp[i] &= LUT_out_mask;
      for (int j = 0; j < N; j++)
      {
        int idx = (x_digits[i] + j) & digit_mask;
        spec[i][j] = (lookup_neg_exp(idx, s_x - digit_size * digit_idx, s_y) -
                      digits_exp[i]) &
                     LUT_out_mask;
      }
    }
    aux->lookup_table<uint64_t>(spec, nullptr, nullptr, N_digits * dim,
                                digit_size, s_y + 2);

    if (digit_size != last_digit_size)
    {
      int offset = N_digits * dim;
      int digit_idx = N_digits;
      for (int i = offset; i < num_digits * dim; i++)
      {
        spec[i] = new uint64_t[last_N];
        digits_exp[i] &= LUT_out_mask;
        for (int j = 0; j < last_N; j++)
        {
          int idx = (x_digits[i] + j) & last_digit_mask;
          spec[i][j] = (lookup_neg_exp(idx, s_x - digit_size * digit_idx, s_y) -
                        digits_exp[i]) &
                       LUT_out_mask;
        }
      }
      aux->lookup_table<uint64_t>(spec + offset, nullptr, nullptr, dim,
                                  last_digit_size, s_y + 2);
    }

    for (int i = 0; i < num_digits * dim; i++)
      delete[] spec[i];
    delete[] spec;
  }
  else
  {
    aux->lookup_table<uint64_t>(nullptr, x_digits, digits_exp, N_digits * dim,
                                digit_size, s_y + 2);
    if (digit_size != last_digit_size)
    {
      int offset = N_digits * dim;
      aux->lookup_table<uint64_t>(nullptr, x_digits + offset,
                                  digits_exp + offset, dim, last_digit_size,
                                  s_y + 2);
    }
    for (int i = 0; i < num_digits * dim; i++)
    {
      digits_exp[i] &= LUT_out_mask;
    }
  }

  uint8_t *zero_shares = new uint8_t[dim];
  for (int i = 0; i < dim; i++)
  {
    zero_shares[i] = 0;
  }
  for (int i = 1; i < num_digits; i *= 2)
  {
    for (int j = 0; j < num_digits and j + i < num_digits; j += 2 * i)
    {
      mult->hadamard_product(dim, digits_exp + j * dim,
                             digits_exp + (j + i) * dim, digits_exp + j * dim,
                             s_y + 2, s_y + 2, 2 * s_y + 2, false, false,
                             MultMode::None, zero_shares, zero_shares);
      trunc->truncate_and_reduce(dim, digits_exp + j * dim,
                                 digits_exp + j * dim, s_y, 2 * s_y + 2);
    }
  }
  xt->z_extend(dim, digits_exp, y, s_y + 2, bw_y, zero_shares);

  delete[] x_digits;
  delete[] tmp_1;
  delete[] digits_exp;
  delete[] zero_shares;
}

void MathFunctions::sigmoid(int32_t dim, uint64_t *x, uint64_t *y, int32_t bw_x,
                            int32_t bw_y, int32_t s_x, int32_t s_y)
{
  uint64_t mask_x = (bw_x == 64 ? -1 : ((1ULL << bw_x) - 1));
  uint64_t mask_y = (bw_y == 64 ? -1 : ((1ULL << bw_y) - 1));
  uint64_t mask_exp_out = ((s_y + 2) == 64 ? -1 : ((1ULL << (s_y + 2)) - 1));
  uint64_t mask_den = ((s_y + 2) == 64 ? -1 : ((1ULL << (s_y + 2)) - 1));
  uint8_t *zero_shares = new uint8_t[dim];
  for (int i = 0; i < dim; i++)
  {
    zero_shares[i] = 0;
  }

  uint8_t *msb_x = new uint8_t[dim];
  aux->MSB(x, msb_x, dim, bw_x);

  // neg_x = -x + msb_x * (2x) (neg_x is always negative)
  uint64_t *tmp_1 = new uint64_t[dim];
  uint64_t *tmp_2 = new uint64_t[dim];
  for (int i = 0; i < dim; i++)
  {
    tmp_1[i] = (-1 * x[i]) & mask_x;
    tmp_2[i] = (2 * x[i]) & mask_x;
  }
  uint64_t *neg_x = new uint64_t[dim];
  aux->multiplexer(msb_x, tmp_2, neg_x, dim, bw_x, bw_x);
  for (int i = 0; i < dim; i++)
  {
    neg_x[i] = (neg_x[i] + tmp_1[i]) & mask_x;
  }

  // den = tmp_1 = 1 + exp_neg_x
  uint64_t *exp_neg_x = new uint64_t[dim];
  lookup_table_exp(dim, neg_x, exp_neg_x, bw_x, s_y + 2, s_x, s_y);
  for (int i = 0; i < dim; i++)
  {
    tmp_1[i] =
        ((party == ALICE ? (1ULL << s_y) : 0) + exp_neg_x[i]) & mask_exp_out;
  }
  // den can't be 2^{s_y+1}, so 1 is subtracted if msb_den is 1
  uint8_t *msb_den = new uint8_t[dim];
  aux->MSB(tmp_1, msb_den, dim, s_y + 2);
  aux->B2A(msb_den, tmp_2, dim, s_y + 2);
  // den = tmp_1 = den - msb_den
  // tmp_2 = 1 (all 1 vector)
  for (int i = 0; i < dim; i++)
  {
    tmp_1[i] = (tmp_1[i] - tmp_2[i]) & mask_den;
    tmp_2[i] = (party == ALICE ? 1 : 0);
  }
  uint64_t *sig_neg_x = new uint64_t[dim];
  // sig_neg_x = 1/(1 + exp_neg_x)
  div(dim, tmp_2, tmp_1, sig_neg_x, 2, s_y + 2, s_y + 2, 0, s_y, s_y, true,
      false);

  // tmp_2 = num = 1 + msb_x * (exp_neg_x - 1)
  for (int i = 0; i < dim; i++)
  {
    tmp_1[i] = (exp_neg_x[i] - (party == ALICE ? 1ULL << s_y : 0)) & mask_den;
  }
  aux->multiplexer(msb_x, tmp_1, tmp_2, dim, s_y + 2, s_y + 2);
  for (int i = 0; i < dim; i++)
  {
    tmp_2[i] = (tmp_2[i] + (party == ALICE ? 1ULL << s_y : 0)) & mask_den;
  }

  mult->hadamard_product(dim, tmp_2, sig_neg_x, tmp_1, s_y + 2, s_y + 2,
                         2 * s_y + 2, false, false, MultMode::None, zero_shares,
                         zero_shares);
  trunc->truncate_and_reduce(dim, tmp_1, tmp_2, s_y, 2 * s_y + 2);

  if (bw_y <= (s_y + 2))
  {
    for (int i = 0; i < dim; i++)
    {
      y[i] = tmp_2[i] & mask_y;
    }
  }
  else
  {
    xt->z_extend(dim, tmp_2, y, s_y + 2, bw_y, zero_shares);
  }

  delete[] msb_x;
  delete[] tmp_1;
  delete[] tmp_2;
  delete[] neg_x;
  delete[] exp_neg_x;
  delete[] msb_den;
  delete[] sig_neg_x;
}

void MathFunctions::tanh(int32_t dim, uint64_t *x, uint64_t *y, int32_t bw_x,
                         int32_t bw_y, int32_t s_x, int32_t s_y)
{
  uint64_t mask_x = (bw_x == 64 ? -1 : ((1ULL << bw_x) - 1));
  uint64_t mask_y = (bw_y == 64 ? -1 : ((1ULL << bw_y) - 1));
  uint64_t mask_exp_out = ((s_y + 3) == 64 ? -1 : ((1ULL << (s_y + 3)) - 1));
  uint64_t mask_den = ((s_y + 2) == 64 ? -1 : ((1ULL << (s_y + 2)) - 1));

  uint8_t *msb_x = new uint8_t[dim];
  aux->MSB(x, msb_x, dim, bw_x);

  // neg_x = -x + msb_x * (2x) (neg_x is always negative)
  uint64_t *tmp_1 = new uint64_t[dim];
  uint64_t *tmp_2 = new uint64_t[dim];
  for (int i = 0; i < dim; i++)
  {
    tmp_1[i] = (-1 * x[i]) & mask_x;
    tmp_2[i] = (2 * x[i]) & mask_x;
  }
  uint64_t *neg_x = new uint64_t[dim];
  aux->multiplexer(msb_x, tmp_2, neg_x, dim, bw_x, bw_x);
  for (int i = 0; i < dim; i++)
  {
    neg_x[i] = (neg_x[i] + tmp_1[i]) & mask_x;
  }

  uint64_t *exp_neg_2x = new uint64_t[dim];
  // scale of neg_x is reduced by 1 instead of mulitplying it with 2 to get
  // neg_2x
  lookup_table_exp(dim, neg_x, exp_neg_2x, bw_x, s_y + 2, s_x - 1, s_y);
  // den = tmp_1 = 1 + exp_neg_2x
  for (int i = 0; i < dim; i++)
  {
    tmp_1[i] =
        ((party == ALICE ? (1ULL << s_y) : 0) + exp_neg_2x[i]) & mask_exp_out;
  }
  // den can't be 2^{s_y+1}, so 1 is subtracted if msb_den is 1
  uint8_t *msb_den = new uint8_t[dim];
  aux->MSB(tmp_1, msb_den, dim, s_y + 2);
  aux->B2A(msb_den, tmp_2, dim, s_y + 2);
  // den = tmp_1 = den - msb_den
  // num = tmp_2 = 1 - exp_neg_2x
  for (int i = 0; i < dim; i++)
  {
    tmp_1[i] = (tmp_1[i] - tmp_2[i]) & mask_den;
    tmp_2[i] =
        ((party == ALICE ? (1ULL << s_y) : 0) - exp_neg_2x[i]) & mask_den;
  }
  uint64_t *tanh_neg_x = new uint64_t[dim];
  // tanh_neg_x = (1 - exp_neg_2x)/(1 + exp_neg_2x)
  div(dim, tmp_2, tmp_1, tanh_neg_x, s_y + 2, s_y + 2, s_y + 2, s_y, s_y, s_y,
      true, false);

  // tmp_2 = tanh_neg_x + msb_x * (-2 * tanh_neg_x)
  // tmp_1 = -2 * tanh_neg_x
  for (int i = 0; i < dim; i++)
  {
    tmp_1[i] = (-2 * tanh_neg_x[i]) & mask_exp_out;
  }
  aux->multiplexer(msb_x, tmp_1, tmp_2, dim, s_y + 2, s_y + 2);
  for (int i = 0; i < dim; i++)
  {
    tmp_2[i] = (tmp_2[i] + tanh_neg_x[i]) & mask_exp_out;
  }

  if (bw_y <= (s_y + 2))
  {
    for (int i = 0; i < dim; i++)
    {
      y[i] = tmp_2[i] & mask_y;
    }
  }
  else
  {
    xt->s_extend(dim, tmp_2, y, s_y + 2, bw_y, msb_x);
  }

  delete[] msb_x;
  delete[] tmp_1;
  delete[] tmp_2;
  delete[] neg_x;
  delete[] exp_neg_2x;
  delete[] msb_den;
  delete[] tanh_neg_x;
}

void MathFunctions::DReLU_Eq(uint64_t *inA, uint8_t *b, uint8_t *b_, int32_t dim, int32_t bwl)
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

void MathFunctions::select_share(uint8_t *sel, uint64_t *x, uint64_t *y, uint64_t *output, int32_t dim, int32_t h)
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

void MathFunctions::third_interval(uint64_t *input_data, uint8_t *res_drelu_cmp, uint8_t *res_drelu_eq, uint8_t *res_eq, int32_t dim, int32_t d, int32_t bwL)
{

  uint64_t *comp_eq_input = new uint64_t[dim];
  uint64_t *outtrunc = new uint64_t[dim];
  uint8_t *res_cmp = new uint8_t[dim];
  trunc->truncate_and_reduce(dim, input_data, outtrunc, d, bwL); // test comm
  DReLU_Eq(outtrunc, res_drelu_cmp, res_drelu_eq, dim, bwL - d);
  uint64_t mask_l_sub1 = ((bwL - d ) == 64) ? ~0ULL : (1ULL << (bwL - d )) - 1;
  if (party == ALICE)
  {
    for (int i = 0; i < dim; i++)
    {
      comp_eq_input[i] = (mask_l_sub1 +1- outtrunc[i]) & mask_l_sub1;
    }
  }
  else
  {
    for (int i = 0; i < dim; i++)
    {
      comp_eq_input[i] = outtrunc[i] & mask_l_sub1;
    }
  }
  eq->check_equality(res_eq, comp_eq_input, dim, bwL - d);
}

void MathFunctions::gelu(int32_t dim, uint64_t *x, uint64_t *y, int32_t bwL,
                         int32_t la, int32_t lb, int32_t s, int32_t f,uint8_t *x_sharp)
{
  uint64_t h = f + 2;
  uint64_t d = f + 2;
  uint64_t mask_h = (h == 64) ? ~0ULL : (1ULL << h) - 1;
  uint64_t mask_bwL = (bwL == 64 ? -1 : ((1ULL << bwL) - 1));
  std::vector<std::vector<uint64_t>> data = {{0, 0}, {1, 1023}, {2, 1019}, {3, 1013}, {3, 1013}, {4, 1004}, {5, 992}, {6, 977}, {6, 977}, {7, 959}, {7, 959}, {8, 937}, {8, 937}, {9, 911}, {9, 910}, {9, 911}, {9, 911}, {10, 877}, {10, 877}, {10, 876}, {10, 876}, {10, 876}, {10, 877}, {10, 877}, {10, 877}, {10, 877}, {10, 877}, {10, 876}, {10, 876}, {10, 875}, {10, 874}, {9, 936}, {9, 937}, {9, 938}, {9, 938}, {9, 938}, {9, 938}, {9, 938}, {9, 937}, {9, 936}, {9, 936}, {8, 1018}, {8, 1019}, {8, 1019}, {8, 1020}, {8, 1021}, {8, 1021}, {8, 1022}, {8, 1022}, {8, 1022}, {8, 1023}, {8, 1023}, {8, 1023}, {8, 1023}, {8, 1023}, {8, 0}, {8, 0}, {8, 0}, {8, 0}, {8, 0}, {8, 0}, {8, 0}, {8, 0}, {8, 0}};
  MultMode mode = MultMode::None;
  uint64_t *a_alice = new uint64_t[dim];
  uint64_t *b_alice = new uint64_t[dim];
  uint64_t *a_bob = new uint64_t[dim];
  uint64_t *b_bob = new uint64_t[dim];
  if (party == ALICE)
  {
    for (size_t i = 0; i < dim; i++)
    {
      a_bob[i] = 0;
      b_bob[i] = 0;
    }
  }

  uint8_t *Drelu = new uint8_t[dim];
  uint8_t *outb = new uint8_t[dim];
  uint8_t *outb_star = new uint8_t[dim];
  uint8_t *outb_sharp = new uint8_t[dim];

  third_interval(x, outb, outb_star, outb_sharp, dim, d, bwL);

  for (int i = 0; i < dim; i++)
  {
    Drelu[i] = outb[i];
    x_sharp[i] = outb_star[i];
  }
  uint64_t *EMUX_output_x = new uint64_t[dim];
  uint64_t *neg_x = new uint64_t[dim];
  for (int i = 0; i < dim; i++)
  {
    neg_x[i] = ((-x[i]) & mask_bwL); // 取反
  }
  if (party == ALICE)
  {
    for (int i = 0; i < dim; i++)
    {
      neg_x[i] = (neg_x[i] -1) & mask_bwL;
    }
  }
  select_share(outb, x, neg_x, EMUX_output_x, dim, bwL);
  uint64_t *EMUX_output_x1 = new uint64_t[dim];
  for (int i = 0; i < dim; i++)
  {
    EMUX_output_x1[i] = EMUX_output_x[i];
  }
  uint64_t *outtrunc = new uint64_t[dim];
  uint64_t *EMUX_output_x1_h = new uint64_t[dim];
  for (int i = 0; i < dim; i++)
  {
    // std::cout << "outtrunc[" << i << "] = " << outtrunc[i] << std::endl;
    EMUX_output_x1_h[i] = EMUX_output_x1[i] & mask_h;
  }
  trunc->truncate_and_reduce(dim, EMUX_output_x1_h, outtrunc, h - s, h);
  uint64_t N = 1ULL << s;
  uint64_t **spec_a = new uint64_t *[dim];
  uint64_t **spec_b = new uint64_t *[dim];
  if (party == ALICE)
    for (int i = 0; i < dim; i++)
    {
      spec_a[i] = new uint64_t[N];
      spec_b[i] = new uint64_t[N];
      for (int j = 0; j < N; j++)
      {
        spec_a[i][j] = data[j][0];
        spec_b[i][j] = data[j][1];
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
      // std::cout << "outtrunc_a[" << i << "] = " << outtrunc_a[i] << std::endl;
    }
  }

  if (party == ALICE)
  {
    aux->lookup_table<uint64_t>(spec_a, nullptr, nullptr, dim, s, la); // step 12 lut
  }
  else
  {                                                                      // party == BOB
    aux->lookup_table<uint64_t>(nullptr, outtrunc_a, a_bob, dim, s, la); // a_bob是查询到的斜率
  }
  if (party == ALICE)
  {
    aux->lookup_table<uint64_t>(spec_b, nullptr, nullptr, dim, s, lb); // step 12 lut
  }
  else
  {                                                                      // party == BOB
    aux->lookup_table<uint64_t>(nullptr, outtrunc_a, b_bob, dim, s, lb); // b_bob是查询到的截距  重要问题，这里的outtrunc应该是两边share加起来，代码里只有Bob的outtrunc check
  }

  uint8_t *msb1 = new uint8_t[dim];
  uint8_t *msb2 = new uint8_t[dim];
  for (int i = 0; i < dim; i++)
  {
    msb1[i] = 0;
    msb2[i] = 0;
  }
  uint64_t *outax = new uint64_t[dim];
  prod->hadamard_product(dim, a_bob, EMUX_output_x, outax, la, bwL, la + bwL, true, true, mode, msb1, msb2);

  for (int i = 0; i < dim; i++)
    outax[i] = (outax[i] >> (la - 1)) & mask_bwL;

  uint64_t *b_SExt = new uint64_t[dim];
  uint8_t *msb_b_extend = new uint8_t[dim];
  uint64_t mask_lb = (lb == 64 ? -1 : ((1ULL << lb) - 1));
  if (party == ALICE)
  {
    for (int i = 0; i < dim; i++)
    {
      msb_b_extend[i] = 1;
      b_bob[i] = (b_bob[i] + 10) & mask_lb;
    }
  }
  else
  {
    for (int i = 0; i < dim; i++)
    {
      msb_b_extend[i] = 1;
      b_bob[i] = (b_bob[i] - 10) & mask_lb;
    }
  }

  ext->s_extend_msb(dim, b_bob, b_SExt, lb, bwL, msb_b_extend);
  uint64_t *z = new uint64_t[dim];
  for (int i = 0; i < dim; i++)
  {
    z[i] = ((outax[i] + b_SExt[i] * static_cast<uint64_t>(std::pow(2, f - lb + 1))) & mask_bwL);
    // std::cout << "z[" << i << "] = " << z[i] << std::endl;
    // std::cout << "outax[" << i << "] = " << outax[i] << std::endl;
    // std::cout << "b_SExt[" << i << "] = " << b_SExt[i] << std::endl;
  }

  uint8_t *Drelu_ = new uint8_t[dim];
  uint8_t *DreluMSB = new uint8_t[dim];
  uint64_t *xhalf = new uint64_t[dim];
  uint64_t *abs_xhalf = new uint64_t[dim];
  uint64_t *bitMul_wrap = new uint64_t[dim];
  uint64_t *out_last_bitwrap = new uint64_t[dim];

  uint8_t *msb_zero = new uint8_t[dim];
  for (int i = 0; i < dim; i++)
  {
    msb_zero[i] = 0;
  }

  // aux->lastbit_MSB_to_Wrap_bitMul(dim, EMUX_output_x1, out_last_bitwrap, bwL);
  // aux->clear_MSB_to_Wrap_bitMul(dim, EMUX_output_x1, msb_zero, bitMul_wrap, bwL);
  // for (int i = 0; i < dim; i++)
  // {
  //   abs_xhalf[i] = ((EMUX_output_x1[i] >> 1) - bitMul_wrap[i] * (uint64_t)pow(2, bwL - 1) + out_last_bitwrap[i]) & mask_bwL;
  // }
  trunc->truncate(dim,EMUX_output_x1,abs_xhalf,1,bwL);

  uint64_t *neg_abs_xhalf = new uint64_t[dim];
  for (int i = 0; i < dim; i++)
  {
    neg_abs_xhalf[i] = (-abs_xhalf[i]+1048576) & mask_bwL;
  }
  select_share(Drelu, abs_xhalf, neg_abs_xhalf, xhalf, dim, bwL); // step 22 ss

  uint64_t *MUX_output_g = new uint64_t[dim];
  for (int i = 0; i < dim; i++)
  {
    Drelu_[i] = (outb_star[i] + outb_sharp[i]) & 1;
            if (party == ALICE)
        {
            Drelu_[i] = Drelu_[i] ^ 1;
        }
  }
  select_share(Drelu_, abs_xhalf, z, MUX_output_g, dim, bwL);
  for (int i = 0; i < dim; i++)
  {
    y[i] = (xhalf[i] + MUX_output_g[i]) & mask_bwL;
  }

  delete[] a_bob;
  delete[] b_bob;
  delete[] outb;
  delete[] outb_star;
  delete[] outb_sharp;
  delete[] EMUX_output_x;
  delete[] neg_x;
  delete[] EMUX_output_x1;
  delete[] outtrunc;
  delete[] EMUX_output_x1_h;
  delete[] outtrunc1;
  delete[] outtrunc_a;
  delete[] msb1;
  delete[] msb2;
  delete[] b_SExt;
  delete[] msb_b_extend;
  delete[] z;
  delete[] Drelu_;
  delete[] DreluMSB;
  delete[] xhalf;
  delete[] abs_xhalf;
  delete[] bitMul_wrap;
  delete[] out_last_bitwrap;
  delete[] neg_abs_xhalf;
  delete[] MUX_output_g;
}

uint64_t lookup_sqrt(int32_t index, int32_t m, int32_t exp_parity)
{
  int32_t k = 1 << m;
  double u = (1.0 + (double(index) / double(k))) * (1 << exp_parity);
  double Y = 1.0 / sqrt(u);
  int32_t scale = m + SQRT_LOOKUP_SCALE;
  uint64_t val = (Y * (1ULL << scale));
  return val;
}

void MathFunctions::sqrt(int32_t dim, uint64_t *x, uint64_t *y, int32_t bw_x,
                         int32_t bw_y, int32_t s_x, int32_t s_y, bool inverse)
{
  int32_t m, iters;
  if (s_y <= 14)
  {
    m = ceil(s_y / 2.0);
    iters = 1;
  }
  else
  {
    m = ceil((ceil(s_y / 2.0)) / 2.0);
    iters = 2;
  }
  assert(m <= KKOT_LIMIT);
  int32_t bw_adjust = bw_x - 1;
  uint64_t mask_adjust = (bw_adjust == 64 ? -1 : ((1ULL << bw_adjust) - 1));
  // MSB is always 0, thus, not including it
  uint8_t *msnzb_vector_bool = new uint8_t[dim * (bw_x - 1)];
  uint64_t *msnzb_vector = new uint64_t[dim * (bw_x - 1)];
  aux->msnzb_one_hot(x, msnzb_vector_bool, bw_x - 1, dim);
  aux->B2A(msnzb_vector_bool, msnzb_vector, dim * (bw_x - 1), bw_x - 1);
  uint64_t *adjust = new uint64_t[dim];
  uint8_t *exp_parity = new uint8_t[dim];
  for (int i = 0; i < dim; i++)
  {
    adjust[i] = 0;
    exp_parity[i] = 0;
    for (int j = 0; j < (bw_x - 1); j++)
    {
      adjust[i] += (1ULL << (bw_x - 2 - j)) * msnzb_vector[i * (bw_x - 1) + j];
      if (((j - s_x) & 1))
      {
        exp_parity[i] ^= msnzb_vector_bool[i * (bw_x - 1) + j];
      }
    }
    adjust[i] &= mask_adjust;
  }
  // adjusted_x: bw = bw_x - 1, scale = bw_x - 2
  uint64_t *adjusted_x = new uint64_t[dim];
  mult->hadamard_product(dim, x, adjust, adjusted_x, bw_x - 1, bw_x - 1,
                         bw_x - 1, false, false, MultMode::None);
  // Top m bits of adjusted_x excluding MSB, which is always 1
  // adjusted_x_m: bw = m, scale = m
  uint64_t *adjusted_x_m = new uint64_t[dim];
  trunc->truncate_and_reduce(dim, adjusted_x, adjusted_x_m, bw_x - m - 2,
                             bw_x - 2);

  // m + 1 bits will be input to the lookup table
  int32_t M = (1LL << (m + 1));
  uint64_t Y_mask = (1ULL << (m + SQRT_LOOKUP_SCALE + 1)) - 1;
  uint64_t m_mask = (1ULL << m) - 1;
  // Y: bw = m + SQRT_LOOKUP_SCALE + 1, scale = m + SQRT_LOOKUP_SCALE
  uint64_t *Y = new uint64_t[dim];
  if (party == ALICE)
  {
    uint64_t **spec;
    spec = new uint64_t *[dim];
    PRG128 prg;
    prg.random_data(Y, dim * sizeof(uint64_t));
    for (int i = 0; i < dim; i++)
    {
      spec[i] = new uint64_t[M];
      Y[i] &= Y_mask;
      for (int j = 0; j < M; j++)
      {
        // j = exp_parity || (adjusted_x_m) (LSB -> MSB)
        int32_t idx = (adjusted_x_m[i] + (j >> 1)) & m_mask;
        int32_t exp_parity_val = (exp_parity[i] ^ (j & 1));
        spec[i][j] = (lookup_sqrt(idx, m, exp_parity_val) - Y[i]) & Y_mask;
      }
    }
    aux->lookup_table<uint64_t>(spec, nullptr, nullptr, dim, m + 1,
                                m + SQRT_LOOKUP_SCALE + 1);

    for (int i = 0; i < dim; i++)
      delete[] spec[i];
    delete[] spec;
  }
  else
  {
    // lut_in = exp_parity || adjusted_x_m
    uint64_t *lut_in = new uint64_t[dim];
    for (int i = 0; i < dim; i++)
    {
      lut_in[i] = ((adjusted_x_m[i] & m_mask) << 1) | (exp_parity[i] & 1);
    }
    aux->lookup_table<uint64_t>(nullptr, lut_in, Y, dim, m + 1,
                                m + SQRT_LOOKUP_SCALE + 1);

    delete[] lut_in;
  }
  // X = (exp_parity ? 2 * adjusted_x : adjusted_x); scale = bw_x - 2
  // X: bw = bw_x
  uint64_t *X = new uint64_t[dim];
  uint64_t *tmp_1 = new uint64_t[dim];
  uint64_t mask_x = (bw_x == 64 ? -1 : ((1ULL << bw_x) - 1));
  xt->z_extend(dim, adjusted_x, X, bw_x - 1, bw_x);
  aux->multiplexer(exp_parity, X, tmp_1, dim, bw_x, bw_x);
  for (int i = 0; i < dim; i++)
  {
    X[i] = (X[i] + tmp_1[i]) & mask_x;
  }

  uint64_t *x_prev = new uint64_t[dim];
  if (inverse)
  {
    // x_prev : bw = m + SQRT_LOOKUP_SCALE + 1, scale = m + SQRT_LOOKUP_SCALE
    // x_prev \approx 0.5 < 1/sqrt(X) < 1
    memcpy(x_prev, Y, dim * sizeof(uint64_t));
  }
  else
  {
    // x_prev : bw = s_y + 1, scale = s_y
    // x_prev \approx 1 < sqrt(X) < 2
    mult->hadamard_product(dim, X, Y, tmp_1, bw_x, m + SQRT_LOOKUP_SCALE + 1,
                           bw_x - 1 + m + SQRT_LOOKUP_SCALE, false, false,
                           MultMode::None);
    assert((bw_x - 2 + m + SQRT_LOOKUP_SCALE) >= s_y);
    trunc->truncate_and_reduce(dim, tmp_1, x_prev,
                               bw_x - 2 + m + SQRT_LOOKUP_SCALE - s_y,
                               bw_x - 1 + m + SQRT_LOOKUP_SCALE);
  }
  // b_prev: bw = s_y + 2, scale = s_y
  uint64_t *b_prev = new uint64_t[dim];
  assert((bw_x - 2) >= s_y);
  trunc->truncate_and_reduce(dim, X, b_prev, bw_x - 2 - s_y, bw_x);
  // Y_prev: bw = m + SQRT_LOOKUP_SCALE + 1, scale = m + SQRT_LOOKUP_SCALE
  uint64_t *Y_prev = new uint64_t[dim];
  memcpy(Y_prev, Y, dim * sizeof(uint64_t));

  uint64_t b_mask = (1ULL << (s_y + 2)) - 1;
  uint64_t *x_curr = new uint64_t[dim];
  uint64_t *b_curr = new uint64_t[dim];
  uint64_t *Y_curr = new uint64_t[dim];
  uint64_t *Y_sq = new uint64_t[dim];
  for (int i = 0; i < iters; i++)
  {
    if (i == 0)
    {
      // Y_sq: bw = 2m + 2SQRT_LOOKUP_SCALE + 1, scale = 2m + 2SQRT_LOOKUP_SCALE
      mult->hadamard_product(
          dim, Y_prev, Y_prev, Y_sq, m + SQRT_LOOKUP_SCALE + 1,
          m + SQRT_LOOKUP_SCALE + 1, 2 * m + 2 * SQRT_LOOKUP_SCALE + 1, false,
          false, MultMode::None);
      // tmp_1: bw = 2m+2SQRT_LOOKUP_SCALE+s_y+2, scale =
      // 2m+2SQRT_LOOKUP_SCALE+s_y
      mult->hadamard_product(dim, Y_sq, b_prev, tmp_1,
                             2 * m + 2 * SQRT_LOOKUP_SCALE + 1, s_y + 2,
                             2 * m + 2 * SQRT_LOOKUP_SCALE + s_y + 2, false,
                             false, MultMode::None);
      // b_curr: bw = s_y + 2, scale = s_y
      trunc->truncate_and_reduce(dim, tmp_1, b_curr,
                                 2 * m + 2 * SQRT_LOOKUP_SCALE,
                                 2 * m + 2 * SQRT_LOOKUP_SCALE + s_y + 2);
    }
    else
    {
      // tmp_1: bw = 2*s_y + 3, scale = 2*s_y + 2
      mult->hadamard_product(dim, Y_prev, Y_prev, tmp_1, s_y + 2, s_y + 2,
                             2 * s_y + 3, false, false, MultMode::None);
      // Y_sq: bw = s_y + 1, scale = s_y
      trunc->truncate_and_reduce(dim, tmp_1, Y_sq, s_y + 2, 2 * s_y + 3);
      // tmp_1: bw = 2s_y + 2, scale = 2s_y
      mult->hadamard_product(dim, Y_sq, b_prev, tmp_1, s_y + 1, s_y + 2,
                             2 * s_y + 2, false, false, MultMode::None);
      // b_curr: bw = s_y + 2, scale = s_y
      trunc->truncate_and_reduce(dim, tmp_1, b_curr, s_y, 2 * s_y + 2);
    }
    for (int j = 0; j < dim; j++)
    {
      // Y_curr: bw = s_y + 2, scale = s_y + 1
      // Y_curr = (3 - b_curr)/2
      Y_curr[j] = ((party == ALICE ? (3ULL << s_y) : 0) - b_curr[j]) & b_mask;
    }
    if (inverse && (i == 0))
    {
      // tmp_1: bw = s_y+m+SQRT_LOOKUP_SCALE+2, scale =
      // s_y+m+SQRT_LOOKUP_SCALE+1
      mult->hadamard_product(
          dim, x_prev, Y_curr, tmp_1, m + SQRT_LOOKUP_SCALE + 1, s_y + 2,
          s_y + m + SQRT_LOOKUP_SCALE + 2, false, false, MultMode::None);
      // x_curr: bw = s_y + 1, scale = s_y
      trunc->truncate_and_reduce(dim, tmp_1, x_curr, m + SQRT_LOOKUP_SCALE + 1,
                                 s_y + m + SQRT_LOOKUP_SCALE + 2);
    }
    else
    {
      // tmp_1: bw = 2*s_y + 2, scale = 2s_y + 1
      mult->hadamard_product(dim, x_prev, Y_curr, tmp_1, s_y + 1, s_y + 2,
                             2 * s_y + 2, false, false, MultMode::None);
      // x_curr: bw = s_y + 1, scale = s_y
      trunc->truncate_and_reduce(dim, tmp_1, x_curr, s_y + 1, 2 * s_y + 2);
    }
    memcpy(x_prev, x_curr, dim * sizeof(uint64_t));
    memcpy(b_prev, b_curr, dim * sizeof(uint64_t));
    memcpy(Y_prev, Y_curr, dim * sizeof(uint64_t));
  }

  int32_t bw_sqrt_adjust = bw_x / 2;
  uint64_t mask_sqrt_adjust =
      (bw_sqrt_adjust == 64 ? -1 : ((1ULL << bw_sqrt_adjust) - 1));
  uint64_t *sqrt_adjust = new uint64_t[dim];
  int32_t sqrt_adjust_scale =
      (inverse ? floor((bw_x - 1 - s_x) / 2.0) : floor((s_x + 1) / 2.0));
  for (int i = 0; i < dim; i++)
  {
    sqrt_adjust[i] = 0;
    for (int j = 0; j < (bw_x - 1); j++)
    {
      if (inverse)
      {
        sqrt_adjust[i] +=
            (1ULL << int(floor((s_x - j + 1) / 2.0) + sqrt_adjust_scale)) *
            msnzb_vector[i * (bw_x - 1) + j];
      }
      else
      {
        sqrt_adjust[i] +=
            (1ULL << int(floor((j - s_x) / 2.0) + sqrt_adjust_scale)) *
            msnzb_vector[i * (bw_x - 1) + j];
      }
    }
    sqrt_adjust[i] &= mask_sqrt_adjust;
  }
  if (iters > 0 || (!inverse))
  {
    // tmp_1: bw = s_y + 1 + bw_sqrt_adjust, scale = s_y + sqrt_adjust_scale
    mult->hadamard_product(dim, x_prev, sqrt_adjust, tmp_1, s_y + 1,
                           bw_sqrt_adjust, s_y + 1 + bw_sqrt_adjust, false,
                           false, MultMode::None);
    // x_curr: bw = s_y + floor(bw_x/2) + 1 - ceil(s_x/2), scale = s_y
    trunc->truncate_and_reduce(dim, tmp_1, x_prev, sqrt_adjust_scale,
                               s_y + 1 + bw_sqrt_adjust);
    if (bw_y > (s_y + 1 + bw_sqrt_adjust - sqrt_adjust_scale))
    {
      xt->z_extend(dim, x_prev, y, s_y + 1 + bw_sqrt_adjust - sqrt_adjust_scale,
                   bw_y);
    }
    else
    {
      aux->reduce(dim, x_prev, y, s_y + 1 + bw_sqrt_adjust - sqrt_adjust_scale,
                  bw_y);
    }
  }
  else
  {
    // tmp_1: bw = m + SQRT_LOOKUP_SCALE + 1 + bw_sqrt_adjust,
    //        scale = m + SQRT_LOOKUP_SCALE + sqrt_adjust_scale
    mult->hadamard_product(dim, x_prev, sqrt_adjust, tmp_1,
                           m + SQRT_LOOKUP_SCALE + 1, bw_sqrt_adjust,
                           m + SQRT_LOOKUP_SCALE + 1 + bw_sqrt_adjust, false,
                           false, MultMode::None);
    // x_curr: bw = m + floor(bw_x/2) + 1 - ceil(s_x/2), scale = m
    // If iters == 0, we know s_y = m
    trunc->truncate_and_reduce(dim, tmp_1, x_prev,
                               sqrt_adjust_scale + SQRT_LOOKUP_SCALE,
                               m + SQRT_LOOKUP_SCALE + 1 + bw_sqrt_adjust);
    if (bw_y > (m + 1 + bw_sqrt_adjust - sqrt_adjust_scale))
    {
      xt->z_extend(dim, x_prev, y, m + 1 + bw_sqrt_adjust - sqrt_adjust_scale,
                   bw_y);
    }
    else
    {
      aux->reduce(dim, x_prev, y, m + 1 + bw_sqrt_adjust - sqrt_adjust_scale,
                  bw_y);
    }
  }

  delete[] msnzb_vector_bool;
  delete[] msnzb_vector;
  delete[] adjust;
  delete[] exp_parity;
  delete[] adjusted_x;
  delete[] X;
  delete[] tmp_1;
  delete[] x_prev;
  delete[] b_prev;
  delete[] Y_prev;
  delete[] x_curr;
  delete[] b_curr;
  delete[] Y_curr;
  delete[] Y_sq;
  delete[] sqrt_adjust;

  return;
}

void MathFunctions::ReLU(int32_t dim, uint64_t *x, uint64_t *y, int32_t bw_x,
                         uint64_t six)
{
  bool six_comparison = false;
  if (six != 0)
    six_comparison = true;
  int32_t size = (six_comparison ? 2 * dim : dim);

  uint64_t mask_x = (bw_x == 64 ? -1 : ((1ULL << bw_x) - 1));

  uint64_t *tmp = new uint64_t[size];
  uint8_t *tmp_msb = new uint8_t[size];
  memcpy(tmp, x, dim * sizeof(uint64_t));
  if (six_comparison)
  {
    for (int i = 0; i < dim; i++)
    {
      tmp[dim + i] = (party == ALICE ? x[i] - six : x[i]) & mask_x;
    }
  }
  aux->MSB(tmp, tmp_msb, size, bw_x);
  for (int i = 0; i < size; i++)
  {
    if (party == ALICE)
    {
      tmp_msb[i] = tmp_msb[i] ^ 1;
    }
  }
  if (six_comparison)
    aux->AND(tmp_msb, tmp_msb + dim, tmp_msb + dim, dim);

  aux->multiplexer(tmp_msb, tmp, tmp, size, bw_x, bw_x);

  memcpy(y, tmp, dim * sizeof(uint64_t));
  if (six_comparison)
  {
    for (int i = 0; i < dim; i++)
    {
      y[i] = (y[i] - tmp[i + dim]) & mask_x;
    }
  }

  delete[] tmp;
  delete[] tmp_msb;
}

void MathFunctions::DReLU(int32_t dim, uint64_t *x, uint8_t *y, int32_t bw_x,
                          uint64_t six)
{
  bool six_comparison = false;
  if (six != 0)
    six_comparison = true;
  int32_t size = (six_comparison ? 2 * dim : dim);

  uint64_t mask_x = (bw_x == 64 ? -1 : ((1ULL << bw_x) - 1));

  uint64_t *tmp = new uint64_t[size];
  uint8_t *tmp_msb = new uint8_t[size];
  memcpy(tmp, x, dim * sizeof(uint64_t));
  if (six_comparison)
  {
    for (int i = 0; i < dim; i++)
    {
      tmp[dim + i] = (party == ALICE ? x[i] - six : x[i]) & mask_x;
    }
  }
  aux->MSB(tmp, y, size, bw_x);
  for (int i = 0; i < size; i++)
  {
    if (party == ALICE)
    {
      y[i] = y[i] ^ 1;
    }
  }

  delete[] tmp;
  delete[] tmp_msb;
}