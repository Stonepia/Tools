

# Original file: ./eca_botnext26ts_256___60.0/eca_botnext26ts_256___60.0_2.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/timm_models/float32/bb/cbbjtmsqrefvhsuku2xazib6meky7ccw5ffxslbzyuejr5tk4e44.py
# Source Nodes: [batch_norm_29, getattr_getattr_l__mod___stages___3_____1___post_attn_act], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
# batch_norm_29 => add_72, mul_121, mul_122, sub_32
# getattr_getattr_l__mod___stages___3_____1___post_attn_act => mul_123, sigmoid_30
triton_poi_fused__native_batch_norm_legit_no_training_silu_33 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_silu_33', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[4194304], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_silu_33', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]})
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_silu_33(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4194304
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 512
    x1 = (xindex // 512) % 64
    x2 = (xindex // 32768)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + ((128*x1) + (8192*((x1 + (64*x0)) // 8192)) + (32768*x2) + (x0 % 128)), None)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = tmp2 * tmp3
    tmp6 = tmp4 * tmp5
    tmp8 = tmp6 + tmp7
    tmp9 = tl.sigmoid(tmp8)
    tmp10 = tmp8 * tmp9
    tl.store(in_out_ptr0 + (x3), tmp10, None)
''')