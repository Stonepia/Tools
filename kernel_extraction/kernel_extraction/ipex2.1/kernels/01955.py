

# Original file: ./mobilenet_v3_large___60.0/mobilenet_v3_large___60.0_3.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/amp_fp16/zy/czyedg3qro5ruuablcioz7i4cgkzgplbhduyzbw2me3xrp7jyt6i.py
# Source Nodes: [getattr_l__self___features___7___block_1_1, getattr_l__self___features___7___block_1_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardswish]
# getattr_l__self___features___7___block_1_1 => add_48, mul_63, mul_64, sub_19
# getattr_l__self___features___7___block_1_2 => add_49, clamp_max_5, clamp_min_5, convert_element_type_104, div_5, mul_65
triton_poi_fused__native_batch_norm_legit_no_training_hardswish_5 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_hardswish_5', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[2097152], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp16', 6: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_hardswish_5', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]})
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_hardswish_5(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1505280
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 240
    tmp0 = tl.load(in_ptr0 + (x2), None).to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tmp1 - tmp2
    tmp5 = tmp3 * tmp4
    tmp7 = tmp5 * tmp6
    tmp9 = tmp7 + tmp8
    tmp10 = 3.0
    tmp11 = tmp9 + tmp10
    tmp12 = 0.0
    tmp13 = triton_helpers.maximum(tmp11, tmp12)
    tmp14 = 6.0
    tmp15 = triton_helpers.minimum(tmp13, tmp14)
    tmp16 = tmp9 * tmp15
    tmp17 = tmp16 / tmp14
    tmp18 = tmp17.to(tl.float32)
    tl.store(out_ptr1 + (x2), tmp18, None)
''')