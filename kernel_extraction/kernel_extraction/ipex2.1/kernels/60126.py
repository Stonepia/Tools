

# Original file: ./DALLE2_pytorch__22_inference_62.2/DALLE2_pytorch__22_inference_62.2.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/float32/gz/cgzvnp5eb6zn6bu3hdakl77lzqx46nklsmkmlwga6iigdohqsol5.py
# Source Nodes: [add, float_1, full_like, l__self___rel_pos_bias_relative_attention_bias, log, long, lt, max_1, min_1, mul, neg, rearrange_2, sub, truediv, truediv_1, where, zeros_like], Original ATen: [aten._to_copy, aten.add, aten.div, aten.embedding, aten.full_like, aten.log, aten.lt, aten.maximum, aten.minimum, aten.mul, aten.neg, aten.permute, aten.sub, aten.where, aten.zeros_like]
# add => add
# float_1 => convert_element_type
# full_like => full_default_1
# l__self___rel_pos_bias_relative_attention_bias => embedding
# log => log
# long => convert_element_type_1
# lt => lt
# max_1 => maximum
# min_1 => minimum
# mul => mul
# neg => neg
# rearrange_2 => permute
# sub => sub
# truediv => div
# truediv_1 => div_1
# where => where
# zeros_like => full_default
triton_poi_fused__to_copy_add_div_embedding_full_like_log_lt_maximum_minimum_mul_neg_permute_sub_where_zeros_like_0 = async_compile.triton('triton_poi_fused__to_copy_add_div_embedding_full_like_log_lt_maximum_minimum_mul_neg_permute_sub_where_zeros_like_0', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[1048576], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_div_embedding_full_like_log_lt_maximum_minimum_mul_neg_permute_sub_where_zeros_like_0', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def triton_poi_fused__to_copy_add_div_embedding_full_like_log_lt_maximum_minimum_mul_neg_permute_sub_where_zeros_like_0(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 542880
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 8) % 261
    x2 = (xindex // 2088)
    x0 = xindex % 8
    x4 = xindex
    tmp0 = tl.maximum(0, x2 + ((-1)*x1), tl.PropagateNan.NONE)
    tmp1 = tl.full([1], 16, tl.int64)
    tmp2 = tmp0 < tmp1
    tmp3 = tmp0.to(tl.float32)
    tmp4 = 16.0
    tmp5 = tmp3 / tmp4
    tmp6 = tl.log(tmp5)
    tmp7 = 2.0794415416798357
    tmp8 = tmp6 / tmp7
    tmp9 = tmp8 * tmp4
    tmp10 = tmp9.to(tl.int64)
    tmp11 = tmp10 + tmp1
    tmp12 = tl.full([1], 31, tl.int64)
    tmp13 = triton_helpers.minimum(tmp11, tmp12)
    tmp14 = tl.where(tmp2, tmp0, tmp13)
    tmp15 = tl.where(tmp14 < 0, tmp14 + 32, tmp14)
    # tl.device_assert((0 <= tmp15) & (tmp15 < 32), "index out of bounds: 0 <= tmp15 < 32")
    tmp16 = tl.load(in_ptr0 + (x0 + (8*tmp15)), xmask)
    tl.store(out_ptr0 + (x4), tmp16, xmask)
''')
