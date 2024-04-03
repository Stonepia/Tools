

# Original file: ./timm_efficientnet___60.0/timm_efficientnet___60.0_2.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/amp_fp16/2e/c2eugr27lup345t3csxikvyax4ravpzu4kabjbzjcwebf2brsoil.py
# Source Nodes: [batch_norm_34, getattr_getattr_l__self___blocks___5_____0___bn2_act, mul_11], Original ATen: [aten._native_batch_norm_legit_no_training, aten.mul, aten.silu]
# batch_norm_34 => add_75, mul_148, mul_149, sub_34
# getattr_getattr_l__self___blocks___5_____0___bn2_act => convert_element_type_254, mul_150, sigmoid_45
# mul_11 => mul_152
triton_poi_fused__native_batch_norm_legit_no_training_mul_silu_34 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_mul_silu_34', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[4194304], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp16', 5: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_mul_silu_34', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]})
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_mul_silu_34(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2107392
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x0 = xindex % 672
    x2 = (xindex // 32928)
    tmp0 = tl.load(in_out_ptr0 + (x3), None).to(tl.float32)
    tmp2 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x0 + (672*x2)), None, eviction_policy='evict_last').to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tmp1 - tmp2
    tmp5 = tmp3 * tmp4
    tmp7 = tmp5 * tmp6
    tmp8 = tmp7 + tmp2
    tmp9 = tl.sigmoid(tmp8)
    tmp10 = tmp8 * tmp9
    tmp11 = tmp10.to(tl.float32)
    tmp13 = tmp11 * tmp12
    tl.store(in_out_ptr0 + (x3), tmp13, None)
''')
