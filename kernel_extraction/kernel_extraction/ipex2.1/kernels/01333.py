

# Original file: ./mobilenet_v2___60.0/mobilenet_v2___60.0_3.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/amp_fp16/db/cdb6ncdtiudz4zbi4lxqsjhac3bojkwmyyshxr7cagsaxidh6sae.py
# Source Nodes: [getattr_l__self___features___14___conv_1_1, getattr_l__self___features___14___conv_1_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh]
# getattr_l__self___features___14___conv_1_1 => add_89, mul_121, mul_122, sub_40
# getattr_l__self___features___14___conv_1_2 => clamp_max_27, clamp_min_27, convert_element_type_220
triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_10 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_10', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[524288], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_10', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]})
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_10(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 451584
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 576
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask).to(tl.float32)
    tmp2 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tmp1 - tmp2
    tmp5 = tmp3 * tmp4
    tmp7 = tmp5 * tmp6
    tmp9 = tmp7 + tmp8
    tmp10 = 0.0
    tmp11 = triton_helpers.maximum(tmp9, tmp10)
    tmp12 = 6.0
    tmp13 = triton_helpers.minimum(tmp11, tmp12)
    tmp14 = tmp13.to(tl.float32)
    tl.store(in_out_ptr0 + (x2), tmp14, xmask)
''')
