

# Original file: ./visformer_small___60.0/visformer_small___60.0_1.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/timm_models/bfloat16/jn/cjni7zk6oacgcx263osemzkj6tqfjgb6hax4rgf3lk7q55k6joew.py
# Source Nodes: [add_4, getattr_l__mod___stage1___4___norm2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
# add_4 => add_24
# getattr_l__mod___stage1___4___norm2 => add_26, convert_element_type_36, mul_43, mul_44, sub_6
triton_poi_fused__native_batch_norm_legit_no_training_add_6 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_6', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[33554432], filename=__file__, meta={'signature': {0: '*bf16', 1: '*bf16', 2: '*fp32', 3: '*fp32', 4: '*bf16', 5: '*bf16', 6: '*bf16', 7: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_6', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=())]})
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_6(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 19267584
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 192
    tmp0 = tl.load(in_ptr0 + (x2), None).to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (x2), None).to(tl.float32)
    tmp4 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp11 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp2 = tmp0 + tmp1
    tmp3 = tmp2.to(tl.float32)
    tmp5 = tmp3 - tmp4
    tmp7 = tmp5 * tmp6
    tmp9 = tmp8.to(tl.float32)
    tmp10 = tmp7 * tmp9
    tmp12 = tmp11.to(tl.float32)
    tmp13 = tmp10 + tmp12
    tmp14 = tmp13.to(tl.float32)
    tl.store(out_ptr0 + (x2), tmp14, None)
''')
