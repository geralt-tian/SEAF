import numpy as np
from transformers import glue_compute_metrics as compute_metrics
import warnings, argparse, tqdm, torch, copy

warnings.filterwarnings("ignore")
np.set_printoptions(threshold=np.inf)

p_large = 536903681
p_small = 557057
input_prune_shape1 = 64
input_prune_shape2 = 64

max_linear1 = np.zeros(12)
max_linear2 = np.zeros(12)
max_linear3 = np.zeros(12)
max_linear4 = np.zeros(12)
max_softmaxv = np.zeros(12)

global x_scale
global w_scale
global b_scale
global pruned_H1

global layer_norm_max


def automatic_scale(max_ln1, max_ln2, max_ln3, max_ln4, original=False):
    x_scale = np.ones((4, 12))
    w_scale = np.ones((4, 12))
    b_scale = np.ones((4, 12))

    for layer_num in range(12):
        x_scale[0, layer_num] = 5
        w_scale[0, layer_num] = 6
        b_scale[0, layer_num] = 11

        x_scale[1, layer_num] = 6
        w_scale[1, layer_num] = 6
        b_scale[1, layer_num] = 12

        x_scale[2, layer_num] = 5
        w_scale[2, layer_num] = 6
        b_scale[2, layer_num] = 11

        x_scale[3, layer_num] = 4
        w_scale[3, layer_num] = 5
        b_scale[3, layer_num] = 9

    x_scale[3, 9] = 4
    w_scale[3, 9] = 4
    b_scale[3, 9] = 8

    x_scale[3, 10] = 4
    w_scale[3, 10] = 4
    b_scale[3, 10] = 8

    if original:
        return x_scale, w_scale, b_scale

    budget = np.zeros((4, 12))
    for layer_num in range(12):
        budget[0, layer_num] = np.floor(
            np.log2(np.floor(p_large / 2 / max_ln1[layer_num]))
        )
        budget[1, layer_num] = np.floor(
            np.log2(np.floor(p_small / 2 / max_ln2[layer_num]))
        )
        budget[2, layer_num] = np.floor(
            np.log2(np.floor(p_small / 2 / max_ln3[layer_num]))
        )
        budget[3, layer_num] = np.floor(
            np.log2(np.floor(p_small / 2 / max_ln4[layer_num]))
        )

    for layer_num in range(12):
        for linear_num in range(4):
            coin = 0
            while budget[linear_num, layer_num] > 0:
                if coin == 0:
                    x_scale[linear_num, layer_num] += 1
                else:
                    w_scale[linear_num, layer_num] += 1
                b_scale[linear_num, layer_num] += 1
                budget[linear_num, layer_num] -= 1
                coin = (coin + 1) % 2
    return x_scale, w_scale, b_scale


def truncate(x, bits=12):
    return np.floor(x * 2 ** bits) / 2 ** bits


def softmax(matrix, current_layer=None):
    matrix = np.floor(matrix)
    max_values = np.max(matrix, axis=1)
    matrix = matrix - max_values[:, np.newaxis]
    # noise = np.random.normal(0, 1e-2, matrix.shape)
    # exp_res = exp_approx(matrix)
    exp_res, exp_l = exp_approx_int(matrix, current_layer=current_layer)
    recp = np.sum(exp_res, axis=1, keepdims=True)
    recp = recp / 2 ** 12
    recp = np.floor(1.0 / recp * 2 ** 12)
    return np.floor(exp_res * recp / 2 ** 12), exp_l


def exp_approx(x):
    # x <= 0
    x = truncate(x, bits=12)
    l = np.floor(x * truncate(1.0 / np.log(2), bits=12) * (-1))
    p = truncate(x + l * truncate(np.log(2), bits=12), bits=12)
    fp = truncate(0.3585) * truncate((p + truncate(1.353)) ** 2) + truncate(0.344)
    fp = truncate(fp)
    res = truncate(fp / 2 ** l)
    return res


def exp_approx_int(x, current_layer=None):
    x = np.floor(x)
    temp_l = np.floor(x * truncate(1.0 / np.log(2), bits=12) * (-1))
    l = np.floor(temp_l / 2 ** 12)
    p = x + np.floor(l * truncate(np.log(2), bits=12) * 2 ** 12)
    c1 = np.floor(0.3585 * 2 ** 12)
    c2 = np.floor(1.353 * 2 ** 12)
    c3 = np.floor(0.344 * 2 ** 12)

    # print(np.log2(np.abs(temp_l * 2**12).max()))

    fp = np.floor((c1 * np.floor(((p + c2) ** 2) / 2 ** 12)) / 2 ** 12) + c3
    fp = np.floor(fp * 2 ** (-l))
    return fp, l


def comp_and_swap(
    array: list[int],
    index1: int,
    index2: int,
    direction: int,
    softmax_v=None,
    H1=None,
    second_sort_flag=False,
) -> None:
    """Compare the value at given index1 and index2 of the array and swap them as per
    the given direction.

    The parameter direction indicates the sorting direction, ASCENDING(1) or
    DESCENDING(0); if (a[i] > a[j]) agrees with the direction, then a[i] and a[j] are
    interchanged.

    >>> arr = [12, 42, -21, 1]
    >>> comp_and_swap(arr, 1, 2, 1)
    >>> arr
    [12, -21, 42, 1]

    >>> comp_and_swap(arr, 1, 2, 0)
    >>> arr
    [12, 42, -21, 1]

    >>> comp_and_swap(arr, 0, 3, 1)
    >>> arr
    [1, 42, -21, 12]

    >>> comp_and_swap(arr, 0, 3, 0)
    >>> arr
    [12, 42, -21, 1]
    """
    # FIXME: the comparison and swap below should be replaced by mpc oblivious versions
    if (direction == 1 and array[index1] > array[index2]) or (
        direction == 0 and array[index1] < array[index2]
    ):
        array[index1], array[index2] = array[index2], array[index1]
        if second_sort_flag:
            softmax_v[index1], softmax_v[index2] = softmax_v[index2], softmax_v[index1]
            H1[index1], H1[index2] = H1[index2], H1[index1]


def bitonic_merge(
    array: list[int],
    low: int,
    length: int,
    direction: int,
    softmax_v=None,
    H1=None,
    second_sort_flag=False,
) -> None:
    """
    It recursively sorts a bitonic sequence in ascending order, if direction = 1, and in
    descending if direction = 0.
    The sequence to be sorted starts at index position low, the parameter length is the
    number of elements to be sorted.

    >>> arr = [12, 42, -21, 1]
    >>> bitonic_merge(arr, 0, 4, 1)
    >>> arr
    [-21, 1, 12, 42]

    >>> bitonic_merge(arr, 0, 4, 0)
    >>> arr
    [42, 12, 1, -21]
    """
    if length > 1:
        middle = int(length / 2)
        for i in range(low, low + middle):
            comp_and_swap(
                array, i, i + middle, direction, softmax_v, H1, second_sort_flag
            )
        bitonic_merge(array, low, middle, direction, softmax_v, H1, second_sort_flag)
        bitonic_merge(
            array, low + middle, middle, direction, softmax_v, H1, second_sort_flag
        )


def bitonic_sort(
    array: list[int],
    low: int,
    length: int,
    direction: int,
    softmax_v=None,
    H1=None,
    second_sort_flag=False,
) -> None:
    # second_sort_flag indicates whether this sorting is the second or not
    # If second_sort_flag = True,
    # we will swap the rows of softmax_v and H1 while swapping the array (array is the score to be sorted)
    """
    This function first produces a bitonic sequence by recursively sorting its two
    halves in opposite sorting orders, and then calls bitonic_merge to make them in the
    same order.

    >>> arr = [12, 34, 92, -23, 0, -121, -167, 145]
    >>> bitonic_sort(arr, 0, 8, 1)
    >>> arr
    [-167, -121, -23, 0, 12, 34, 92, 145]

    >>> bitonic_sort(arr, 0, 8, 0)
    >>> arr
    [145, 92, 34, 12, 0, -23, -121, -167]
    """
    if length > 1:
        middle = int(length / 2)
        bitonic_sort(array, low, middle, 1, softmax_v, H1, second_sort_flag)
        bitonic_sort(array, low + middle, middle, 0, softmax_v, H1, second_sort_flag)
        bitonic_merge(array, low, length, direction, softmax_v, H1, second_sort_flag)


def oblivious_prune(softmax_l, softmax_v=None, H1=None, current_layer=0):
    # The input is the l computed in softmax, the shape is 12 x 128 x 128
    # softmax * V and H1 will be swapped together with the score vector in the 2nd sorting
    assert softmax_l.shape == (12, 128, 128)

    # Sum the 12 different l matrix, and we have a 128 x 128 matrix temp_sum_l
    temp_sum_l = softmax_l.sum(axis=0)

    # Sum the temp_sum_l along the first dimension (The 128 rows of temp_sum_l are summed together)
    orig_scores = temp_sum_l.sum(axis=0)

    # sort the scores (ascending order)
    sorted_scores = orig_scores.tolist()
    bitonic_sort(sorted_scores, low=0, length=128, direction=1)

    # Get the median threshold
    score_threshold = sorted_scores[63]

    # Compare the unsorted scores with the median, and add the comparison results to indices
    indice_plus_scores = np.arange(0, 128)
    for i in range(128):
        # The comparison returns 128 if the score is greater than the threshold, otherwise returns 0
        # FIXME: this comparison should be MPC oblivious version
        if orig_scores[i] <= score_threshold:
            comp_result = 0
        else:
            comp_result = 128

        # Add the comparison result to the corresponding index (0~127)
        indice_plus_scores[i] += comp_result

    # convert array to list
    indice_plus_scores = indice_plus_scores.tolist()
    temp_pruned_softmax_v = list(softmax_v)
    temp_pruned_H1 = list(H1)

    # Sort the indices+scores and swap softmax*V and H1 at the same time
    bitonic_sort(
        indice_plus_scores,
        low=0,
        length=128,
        direction=1,
        softmax_v=temp_pruned_softmax_v,
        H1=temp_pruned_H1,
        second_sort_flag=True,
    )

    # Take the first 64 rows of softmax*V and H1
    temp_pruned_softmax_v = temp_pruned_softmax_v[:64]
    temp_pruned_H1 = temp_pruned_H1[:64]
    return np.array(temp_pruned_softmax_v), np.array(temp_pruned_H1)


def sort_online_prune(softmax_out, current_layer=0):
    assert softmax_out.shape[0] == 12

    scores = None
    for i in range(12):
        if scores is None:
            scores = softmax_out[i].sum(axis=0)
        else:
            scores += softmax_out[i].sum(axis=0)
    # prev_words = np.argsort(scores)[::-1]
    prev_words = np.argsort(scores)
    if current_layer == 0:
        prev_words = prev_words[:input_prune_shape1]
    elif current_layer == 6:
        prev_words = prev_words[:input_prune_shape2]
    prev_words = np.sort(prev_words)
    return prev_words


def layer_norm_fix(matrix, W, B, print_flag=False):
    matrix = np.floor(matrix * 2 ** 12)

    if print_flag:
        print(matrix)
        np.save("/home/qipang/mnt/d2/trash/compare/np.npy", matrix)

    matrix_sum = np.sum(matrix, axis=1, keepdims=True)
    array_size = matrix.shape[1]

    dn = np.floor(1.0 / array_size * 2 ** 24)

    # print(np.log2(np.abs(matrix_sum * dn).max()))

    avg = np.floor(matrix_sum * dn / (2 ** 24))

    # mean_ref = np.floor(np.mean(matrix, axis=1, keepdims=True))

    x_avg = matrix - avg
    # print(np.log2(np.abs(x_avg).max() / np.sqrt(768)))

    x_avg_square = x_avg ** 2

    # print(np.log2(np.abs(x_avg_square).max()))

    x_avg_square = np.floor(x_avg_square / (2 ** 12))

    x_avg_square_sum = np.sum(x_avg_square, axis=1, keepdims=True)

    x_avg_square_sum_avg = np.floor(x_avg_square_sum * dn / (2 ** 24))

    global layer_norm_max
    if (x_avg_square_sum_avg / (2 ** 12)).max() > layer_norm_max:
        layer_norm_max = (x_avg_square_sum_avg / (2 ** 12)).max()

    sigma = np.floor((1 / np.sqrt(x_avg_square_sum_avg / (2 ** 12))) * 2 ** 12)

    x_avg_sigma = np.floor(x_avg * sigma / (2 ** 12))

    ln_w = np.floor(x_avg_sigma * W / (2 ** 12))

    ln_w_b = ln_w + B

    return ln_w_b


def layer_norm(matrix, W, B):
    matrix = np.floor(matrix * 2 ** 12) / 2 ** 12
    mean = np.mean(matrix, axis=1, keepdims=True)
    variance = np.var(matrix, axis=1, keepdims=True)

    print(mean[:, 0] * 2 ** 12)

    inv_sqrt_x_var = np.floor(1.0 / np.sqrt(variance) * 2 ** 12) / 2 ** 12

    res = np.floor((matrix - mean) * inv_sqrt_x_var * 2 ** 12)
    res = np.floor(res * W / 2 ** 12) + B

    return res / 2 ** 12


def gelu(x):
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * np.power(x, 3))))


def approx_gelu(x, print_flag=False):
    # x = truncate(x)

    # print(np.floor(softmax_input[4] + input_mask * 2**12)[8])
    # if print_flag:
    #     print(x)
    #     np.save('/home/qipang/mnt/d2/trash/compare/np.npy', x)

    c1 = 0.14439048359960427
    c2 = 0.7077117131613893
    c3 = 4.5702822654246535
    c4 = 8.15444702051307
    c5 = 16.382265425072532

    c1 = np.floor(c1 * 2 ** 12)
    c2 = np.floor(c2 * 2 ** 12)
    c3 = np.floor(c3 * 2 ** 12)
    c4 = np.floor(c4 * 2 ** 12)
    c5 = np.floor(c5 * 2 ** 12)

    abs_x = np.abs(x)
    # y = truncate((truncate(c1 * abs_x, bits=12) - c2) * abs_x, bits=12) + c3
    # res = truncate((y + truncate(c1 * abs_x, bits=12) - c4) * y, bits=12) + c5 + 0.5 * x
    # res = truncate(res, bits=12)
    # res[x > 2.7] = x[x > 2.7]
    # res[x < -2.7] = 0

    temp_y = np.floor(c1 * abs_x / 2 ** 12) - c2
    y = np.floor(temp_y * abs_x / 2 ** 12) + c3
    temp_res = y + np.floor(c1 * abs_x / 2 ** 12) - c4
    temp_res = temp_res * y
    res = np.floor(temp_res / 2 ** 12) + c5 + x / 2

    res[x > np.floor(2.7 * 2 ** 12)] = x[x > np.floor(2.7 * 2 ** 12)]
    res[x < np.floor(-2.7 * 2 ** 12)] = 0
    return res


def linear1(H1, Wq, Wk, Wv, Bq, Bk, Bv, input_mask, layer_count):
    assert Wq.shape == (12, 768, 64) and Bq.shape == (12, 64)

    global pruned_H1

    H1_12_scale = copy.deepcopy(H1)

    H1_scale = 12 - x_scale[0, layer_count]
    H1 = np.floor(H1 / 2 ** H1_scale)

    if args.online_prune and layer_count >= 7:
        softmax_input = np.zeros((12, input_prune_shape2, input_prune_shape2))
        Q = np.zeros((12, input_prune_shape2, 64))
        K = np.zeros((12, input_prune_shape2, 64))
        V = np.zeros((12, input_prune_shape2, 64))
        softmax_output = np.zeros((12, input_prune_shape2, input_prune_shape2))
    elif args.online_prune and layer_count >= 1 and layer_count <= 6:
        softmax_input = np.zeros((12, input_prune_shape1, input_prune_shape1))
        Q = np.zeros((12, input_prune_shape1, 64))
        K = np.zeros((12, input_prune_shape1, 64))
        V = np.zeros((12, input_prune_shape1, 64))
        softmax_output = np.zeros((12, input_prune_shape1, input_prune_shape1))
        softmax_temp_l = np.zeros((12, input_prune_shape1, input_prune_shape1))
    else:
        softmax_input = np.zeros((12, 128, 128))
        Q = np.zeros((12, 128, 64))
        K = np.zeros((12, 128, 64))
        V = np.zeros((12, 128, 64))
        softmax_output = np.zeros((12, 128, 128))
        softmax_temp_l = np.zeros((12, 128, 128))
    for i in range(12):
        Q[i] = np.matmul(H1, Wq[i]) + Bq[i]
        K[i] = np.matmul(H1, Wk[i]) + Bk[i]
        # softmax_input[i] = np.matmul(Q[i], K[i].T)
        softmax_input[i] = np.matmul(Q[i], K[i].T)
        V[i] = np.matmul(H1, Wv[i]) + Bv[i]

    # HACK: rescale

    max_linear1[layer_count] = np.max(
        [max_linear1[layer_count], np.abs(softmax_input).max()]
    )

    if np.abs(softmax_input).max() > p_large / 2:
        print("linear 1 layer ", layer_count, np.abs(softmax_input).max())

    softmax_input_scale = b_scale[0, layer_count] + b_scale[0, layer_count]
    if layer_count == 2:
        softmax_input_scale -= 1
    softmax_input_scale = softmax_input_scale - 12

    softmax_input = np.floor(softmax_input / 2 ** softmax_input_scale)

    for i in range(12):
        # softmax_output[i] = torch.nn.Softmax()(torch.Tensor(softmax_input[i] + input_mask)).numpy()
        if args.online_prune and (layer_count == 0):
            softmax_output[i], softmax_temp_l[i] = softmax(
                softmax_input[i] + input_mask[: softmax_input[i].shape[0]] * 2 ** 12,
                current_layer=layer_count,
            )
        else:
            softmax_output[i], _ = softmax(
                softmax_input[i] + input_mask[: softmax_input[i].shape[0]] * 2 ** 12,
                current_layer=layer_count,
            )
            # softmax_output[i] = torch.nn.Softmax()(torch.Tensor(softmax_input[i] / 2**12 + input_mask)).numpy() * 2**12

    attention_res = []
    for i in range(12):
        attention_res.append(np.matmul(softmax_output[i], V[i]))

    attention_res = np.hstack(attention_res)
    if args.online_prune and layer_count == 0:
        attention_res, pruned_H1 = oblivious_prune(
            softmax_temp_l, attention_res, H1_12_scale, layer_count
        )

    max_softmaxv[layer_count] = np.max(
        [max_softmaxv[layer_count], np.abs(attention_res).max()]
    )
    if np.abs(attention_res).max() > p_large / 2:
        print("softmax v ", layer_count, np.abs(attention_res).max())

    # attention_res scale 23
    return attention_res


def linear2(H2, W_attO, B_attO, H1, layer_count):
    assert W_attO.shape == (768, 768) and B_attO.shape == (768,)
    global pruned_H1
    # HACK: rescale to 6
    H2_scale = x_scale[0, layer_count] + w_scale[0, layer_count] + 12
    H2_scale = H2_scale - x_scale[1, layer_count]
    H2 = np.floor(H2 / 2 ** H2_scale)
    res = np.matmul(H2, W_attO) + B_attO

    max_linear2[layer_count] = np.max([max_linear2[layer_count], np.abs(res).max()])

    if np.abs(res).max() > p_small / 2:
        print("linear 2 layer ", layer_count, np.abs(res).max())

    H1_scale = 12
    H1_scale = x_scale[1, layer_count] + w_scale[1, layer_count] - H1_scale

    if args.online_prune and (layer_count == 0):
        res = res + pruned_H1 * 2 ** H1_scale
    else:
        res = res + H1 * 2 ** H1_scale

    return res


def linear3(H4, W_inter, B_inter, layer_count):
    assert W_inter.shape == (768, 3072) and B_inter.shape == (3072,)
    H4 = np.floor(H4 * 2 ** x_scale[2, layer_count])
    res = np.matmul(H4, W_inter) + B_inter

    max_linear3[layer_count] = np.max([max_linear3[layer_count], np.abs(res).max()])

    if np.abs(res).max() > p_small / 2:
        print("linear 3 layer ", layer_count, np.abs(res).max())
    # res has scale 11
    return res


def linear4(H6, W_out, B_out, H4, layer_count):
    assert W_out.shape == (3072, 768) and B_out.shape == (768,)

    H6_scale = x_scale[3, layer_count]
    H6_scale = 12 - H6_scale
    H6 = np.floor(H6 / 2 ** H6_scale)
    H4_scale = b_scale[3, layer_count]
    H4 = np.floor(H4 * 2 ** 12)
    H4 = H4 / 2 ** (12 - H4_scale)
    res = np.matmul(H6, W_out) + B_out

    max_linear4[layer_count] = np.max([max_linear4[layer_count], np.abs(res).max()])

    if np.abs(res).max() > p_small / 2:
        print("linear 4 layer ", layer_count, np.abs(res).max())
    res = res + H4

    res /= 2 ** b_scale[3, layer_count]

    return res


def attention_layer(
    H1,
    Wq,
    Wk,
    Wv,
    Bq,
    Bk,
    Bv,
    W_attO,
    B_attO,
    W_inter,
    B_inter,
    W_out,
    B_out,
    W_layernorm_att,
    B_layernorm_att,
    W_layernorm_out,
    B_layernorm_out,
    input_mask,
    layer_count,
):
    res_linear1 = linear1(H1, Wq, Wk, Wv, Bq, Bk, Bv, input_mask, layer_count)
    res_linear2 = linear2(res_linear1, W_attO, B_attO, H1, layer_count)
    # HACK: rescale
    res_linear2_scale = x_scale[1, layer_count] + w_scale[1, layer_count]
    res_linear2 /= 2 ** res_linear2_scale
    if layer_count == 0:
        H4 = layer_norm_fix(
            res_linear2, W_layernorm_att, B_layernorm_att, print_flag=False
        )
    else:
        H4 = layer_norm_fix(res_linear2, W_layernorm_att, B_layernorm_att)

    H4 = H4 / 2 ** 12

    res_linear3 = linear3(H4, W_inter, B_inter, layer_count)
    # HACK: rescale
    res_linear3_scale = x_scale[2, layer_count] + w_scale[2, layer_count]
    res_linear3_scale = 12 - res_linear3_scale
    res_linear3 *= 2 ** res_linear3_scale
    res_linear3 = np.floor(res_linear3)

    if layer_count == 0:
        H6 = approx_gelu(res_linear3, print_flag=False)
    else:
        H6 = approx_gelu(res_linear3)

    H8 = linear4(H6, W_out, B_out, H4, layer_count)
    H9 = layer_norm_fix(H8, W_layernorm_out, B_layernorm_out)

    return H9


def approx_tanh(x):

    x_1 = np.abs(x)

    a = -0.013232131886235352
    b = 0.09948747962825866
    c = -0.20093640347818847
    d = -0.17616532856475706
    e = 1.0542492677156243
    f = -0.0024920889620412097

    a = np.floor(a * 2 ** 12)
    b = np.floor(b * 2 ** 12)
    c = np.floor(c * 2 ** 12)
    d = np.floor(d * 2 ** 12)
    e = np.floor(e * 2 ** 12)
    f = np.floor(f * 2 ** 12)

    x_2 = np.floor(x_1 * x_1 / (2 ** 12))
    x_3 = np.floor(x_2 * x_1 / (2 ** 12))
    x_4 = np.floor(x_2 * x_2 / (2 ** 12))
    x_5 = np.floor(x_4 * x_1 / (2 ** 12))

    x_5_a = np.floor(x_5 * a / (2 ** 12))
    x_4_b = np.floor(x_4 * b / (2 ** 12))
    x_3_c = np.floor(x_3 * c / (2 ** 12))
    x_2_d = np.floor(x_2 * d / (2 ** 12))
    x_1_e = np.floor(x_1 * e / (2 ** 12))

    poly = x_5_a + x_4_b + x_3_c + x_2_d + x_1_e + f

    res = poly / (2 ** 12)

    res[x < 0] = -res[x < 0]

    res[x > np.floor(2.855 * 2 ** 12)] = 1
    res[x < np.floor(-2.855 * 2 ** 12)] = -1

    return res


def pool_classify_layer(X, W_p, W_c, B_p, B_c):
    pool_linear = np.matmul(X, W_p) + B_p

    pool_linear = np.floor(pool_linear / 2 ** 12)

    pool_res = approx_tanh(pool_linear)
    pool_res = np.floor(pool_res * 2 ** 12)

    result = np.matmul(pool_res, W_c) + B_c
    # print(result)
    # np.save('/home/qipang/mnt/d2/trash/compare/np.npy', result)

    return result / 2 ** 24


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--task_name", type=str)
    parser.add_argument("--sparse", action="store_true")
    parser.add_argument("--online_prune", action="store_true")
    parser.add_argument("--sample_num", type=int)

    args = parser.parse_args()

    data_dir = "/home/qipang/mnt/d2/"
    if args.sparse:
        data_dir += "sparse/"
    elif args.online_prune:
        data_dir += "prune/"
    else:
        # data_dir += 'original/'
        data_dir += "quantize/"
    data_dir += args.task_name
    data_dir += "/weights_txt/"

    global x_scale
    global w_scale
    global b_scale

    # x_scale, w_scale, b_scale = automatic_scale(max_linear1, max_linear2, max_linear3, max_linear4, original=True)

    # np.savetxt(data_dir + 'x_scale.txt', x_scale, delimiter=',', fmt='%d')
    # np.savetxt(data_dir + 'w_scale.txt', w_scale, delimiter=',', fmt='%d')
    # np.savetxt(data_dir + 'b_scale.txt', b_scale, delimiter=',', fmt='%d')

    acc = 0
    total = 0

    pred = []
    label = []
    results = {}

    (
        Wq,
        Wk,
        Wv,
        Bq,
        Bk,
        Bv,
        W_attO,
        B_attO,
        W_inter,
        B_inter,
        W_out,
        B_out,
        W_layernorm_att,
        B_layernorm_att,
        W_layernorm_out,
        B_layernorm_out,
    ) = ([] for i in range(16))

    for layer_num in range(12):
        # print('layer num: ', layer_num)
        Wq.append(
            np.loadtxt(
                data_dir
                + "bert.encoder.layer."
                + str(layer_num)
                + ".attention.self.query.weight.txt",
                delimiter=",",
            )
            .astype(np.int64)
            .reshape((12, 768, 64))
        )
        Wk.append(
            np.loadtxt(
                data_dir
                + "bert.encoder.layer."
                + str(layer_num)
                + ".attention.self.key.weight.txt",
                delimiter=",",
            )
            .astype(np.int64)
            .reshape((12, 768, 64))
        )
        Wv.append(
            np.loadtxt(
                data_dir
                + "bert.encoder.layer."
                + str(layer_num)
                + ".attention.self.value.weight.txt",
                delimiter=",",
            )
            .astype(np.int64)
            .reshape((12, 768, 64))
        )

        Bq.append(
            np.loadtxt(
                data_dir
                + "bert.encoder.layer."
                + str(layer_num)
                + ".attention.self.query.bias.txt",
                delimiter=",",
            )
            .astype(np.int64)
            .reshape((12, 64))
        )
        Bk.append(
            np.loadtxt(
                data_dir
                + "bert.encoder.layer."
                + str(layer_num)
                + ".attention.self.key.bias.txt",
                delimiter=",",
            )
            .astype(np.int64)
            .reshape((12, 64))
        )
        Bv.append(
            np.loadtxt(
                data_dir
                + "bert.encoder.layer."
                + str(layer_num)
                + ".attention.self.value.bias.txt",
                delimiter=",",
            )
            .astype(np.int64)
            .reshape((12, 64))
        )

        W_attO.append(
            np.loadtxt(
                data_dir
                + "bert.encoder.layer."
                + str(layer_num)
                + ".attention.output.dense.weight.txt",
                delimiter=",",
            ).astype(np.int64)
        )
        B_attO.append(
            np.loadtxt(
                data_dir
                + "bert.encoder.layer."
                + str(layer_num)
                + ".attention.output.dense.bias.txt",
                delimiter=",",
            ).astype(np.int64)
        )

        W_inter.append(
            np.loadtxt(
                data_dir
                + "bert.encoder.layer."
                + str(layer_num)
                + ".intermediate.dense.weight.txt",
                delimiter=",",
            ).astype(np.int64)
        )
        B_inter.append(
            np.loadtxt(
                data_dir
                + "bert.encoder.layer."
                + str(layer_num)
                + ".intermediate.dense.bias.txt",
                delimiter=",",
            ).astype(np.int64)
        )

        W_out.append(
            np.loadtxt(
                data_dir
                + "bert.encoder.layer."
                + str(layer_num)
                + ".output.dense.weight.txt",
                delimiter=",",
            ).astype(np.int64)
        )
        B_out.append(
            np.loadtxt(
                data_dir
                + "bert.encoder.layer."
                + str(layer_num)
                + ".output.dense.bias.txt",
                delimiter=",",
            ).astype(np.int64)
        )

        W_layernorm_att.append(
            np.loadtxt(
                data_dir
                + "bert.encoder.layer."
                + str(layer_num)
                + ".attention.output.LayerNorm.weight.txt",
                delimiter=",",
            )
        )
        B_layernorm_att.append(
            np.loadtxt(
                data_dir
                + "bert.encoder.layer."
                + str(layer_num)
                + ".attention.output.LayerNorm.bias.txt",
                delimiter=",",
            )
        )

        W_layernorm_out.append(
            np.loadtxt(
                data_dir
                + "bert.encoder.layer."
                + str(layer_num)
                + ".output.LayerNorm.weight.txt",
                delimiter=",",
            )
        )
        B_layernorm_out.append(
            np.loadtxt(
                data_dir
                + "bert.encoder.layer."
                + str(layer_num)
                + ".output.LayerNorm.bias.txt",
                delimiter=",",
            )
        )

    W_pool = np.loadtxt(
        data_dir + "bert.pooler.dense.weight.txt", delimiter=","
    ).astype(np.int64)
    B_pool = np.loadtxt(data_dir + "bert.pooler.dense.bias.txt", delimiter=",").astype(
        np.int64
    )

    W_classify = np.loadtxt(data_dir + "classifier.weight.txt", delimiter=",").astype(
        np.int64
    )
    B_classify = np.loadtxt(data_dir + "classifier.bias.txt", delimiter=",").astype(
        np.int64
    )

    pbar = tqdm.tqdm(range(args.sample_num))
    labels = np.loadtxt(data_dir + "labels.txt", delimiter=",")

    # x_scale, w_scale, b_scale = automatic_scale(max_linear1, max_linear2, max_linear3, max_linear4, original=True)
    x_scale = np.loadtxt(data_dir + "x_scale.txt", delimiter=",")
    w_scale = np.loadtxt(data_dir + "w_scale.txt", delimiter=",")
    b_scale = np.loadtxt(data_dir + "b_scale.txt", delimiter=",")

    # print(x_scale, w_scale, b_scale)

    # x_scale[0, :] = 5
    # b_scale[0, :] = 11
    global layer_norm_max
    layer_norm_max = 0

    for data_sample in range(args.sample_num):
        # if data_sample + 1 != 5:
        #     continue
        pbar.update(1)
        total += 1

        H1 = np.loadtxt(
            data_dir + "inputs_" + str(data_sample) + "_data.txt", delimiter=","
        ).astype(np.int64)
        input_mask = np.loadtxt(
            data_dir + "inputs_" + str(data_sample) + "_mask.txt", delimiter=","
        ).astype(np.int64)
        ground_truth = labels[data_sample]

        for layer_num in range(12):
            H1 = attention_layer(
                H1,
                Wq[layer_num],
                Wk[layer_num],
                Wv[layer_num],
                Bq[layer_num],
                Bk[layer_num],
                Bv[layer_num],
                W_attO[layer_num],
                B_attO[layer_num],
                W_inter[layer_num],
                B_inter[layer_num],
                W_out[layer_num],
                B_out[layer_num],
                W_layernorm_att[layer_num],
                B_layernorm_att[layer_num],
                W_layernorm_out[layer_num],
                B_layernorm_out[layer_num],
                input_mask,
                layer_num,
            )

        res = pool_classify_layer(H1[0], W_pool, W_classify, B_pool, B_classify)

        if "sts-b" in data_dir:
            pred.append(res)
        else:
            pred.append(np.argmax(res))
        label.append(ground_truth)

        if (data_sample + 1) % 10 == 0:
            huggingface_eval = compute_metrics(
                args.task_name, np.array(pred), np.array(label)
            )
            results.update(huggingface_eval)
            pbar.set_postfix(results)

    huggingface_eval = compute_metrics(args.task_name, np.array(pred), np.array(label))
    results.update(huggingface_eval)

    if np.argmax(res) == ground_truth:
        acc += 1

    # print(pred)
    print(results)

    print("layer norm max: ", layer_norm_max)

    # print(max_linear1)
    # print(max_linear2)
    # print(max_linear3)
    # print(max_linear4)

    # x_scale, w_scale, b_scale = automatic_scale(max_linear1, max_linear2, max_linear3, max_linear4, original=False)

    # np.savetxt(data_dir + 'x_scale.txt', x_scale, delimiter=',', fmt='%d')
    # np.savetxt(data_dir + 'w_scale.txt', w_scale, delimiter=',', fmt='%d')
    # np.savetxt(data_dir + 'b_scale.txt', b_scale, delimiter=',', fmt='%d')
