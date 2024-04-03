

# Original file: ./pytorch_CycleGAN_and_pix2pix___60.0/pytorch_CycleGAN_and_pix2pix___60.0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/float16/m5/cm5lohjims72cvqxv3vqn4taqraggx77gnpny73xme5q6p4uzhtu.py
# Source Nodes: [add_1, add_2, getattr_l__mod___model___11___conv_block_6, getattr_l__mod___model___12___conv_block_6], Original ATen: [aten._native_batch_norm_legit, aten.add]
# add_1 => add_8
# add_2 => add_11
# getattr_l__mod___model___11___conv_block_6 => add_7, convert_element_type_12, convert_element_type_13, mul_6, rsqrt_6, sub_6, var_mean_6
# getattr_l__mod___model___12___conv_block_6 => add_10, convert_element_type_16, convert_element_type_17, mul_8, rsqrt_8, sub_8, var_mean_8
triton_poi_fused__native_batch_norm_legit_add_12 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_add_12', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[1048576], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp32', 3: '*fp32', 4: '*fp16', 5: '*fp32', 6: '*fp32', 7: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_add_12', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=())]})
@triton.jit
def triton_poi_fused__native_batch_norm_legit_add_12(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1048576
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 256
    tmp0 = tl.load(in_out_ptr0 + (x2), None).to(tl.float32)
    tmp1 = tl.load(in_ptr0 + (x2), None).to(tl.float32)
    tmp3 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x2), None).to(tl.float32)
    tmp16 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp1.to(tl.float32)
    tmp4 = tmp2 - tmp3
    tmp6 = 4096.0
    tmp7 = tmp5 / tmp6
    tmp8 = 1e-05
    tmp9 = tmp7 + tmp8
    tmp10 = libdevice.rsqrt(tmp9)
    tmp11 = tmp4 * tmp10
    tmp12 = tmp11.to(tl.float32)
    tmp13 = tmp0 + tmp12
    tmp15 = tmp14.to(tl.float32)
    tmp17 = tmp15 - tmp16
    tmp19 = tmp18 / tmp6
    tmp20 = tmp19 + tmp8
    tmp21 = libdevice.rsqrt(tmp20)
    tmp22 = tmp17 * tmp21
    tmp23 = tmp22.to(tl.float32)
    tmp24 = tmp13 + tmp23
    tl.store(in_out_ptr0 + (x2), tmp24, None)
''')
