

# Original file: ./levit_128___60.0/levit_128___60.0_3.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/timm_models/float32/z6/cz6d6aul57emtd5oage5uf6cxwszlyh7y5hmezx24qqwjd2jjp7f.py
# Source Nodes: [getattr_getattr_l__mod___stages___0___blocks___0___mlp_act, getattr_getattr_l__mod___stages___0___blocks___0___mlp_ln1_bn], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardswish]
# getattr_getattr_l__mod___stages___0___blocks___0___mlp_act => add_20, clamp_max_4, clamp_min_4, div_5, mul_26
# getattr_getattr_l__mod___stages___0___blocks___0___mlp_ln1_bn => add_19, mul_24, mul_25, sub_7
triton_poi_fused__native_batch_norm_legit_no_training_hardswish_8 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_hardswish_8', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[8388608], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_hardswish_8', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]})
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_hardswish_8(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6422528
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 256
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = tmp2 * tmp3
    tmp6 = tmp4 * tmp5
    tmp8 = tmp6 + tmp7
    tmp9 = 3.0
    tmp10 = tmp8 + tmp9
    tmp11 = 0.0
    tmp12 = triton_helpers.maximum(tmp10, tmp11)
    tmp13 = 6.0
    tmp14 = triton_helpers.minimum(tmp12, tmp13)
    tmp15 = tmp8 * tmp14
    tmp16 = tmp15 / tmp13
    tl.store(in_out_ptr0 + (x2), tmp16, None)
''')
