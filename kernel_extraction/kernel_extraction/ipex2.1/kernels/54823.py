

# Original file: ./AllenaiLongformerBase__22_forward_137.2/AllenaiLongformerBase__22_forward_137.2_1.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/huggingface/amp_bf16/lx/clxnjfqafuvkxvr23rzt663rp673hltmcsozjifiz55i6jnjejrn.py
# Source Nodes: [setitem_9], Original ATen: [aten.slice_scatter]
# setitem_9 => slice_scatter_34
triton_poi_fused_slice_scatter_18 = async_compile.triton('triton_poi_fused_slice_scatter_18', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[1048576], filename=__file__, meta={'signature': {0: '*bf16', 1: '*bf16', 2: '*bf16', 3: '*bf16', 4: '*bf16', 5: '*bf16', 6: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_slice_scatter_18', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]})
@triton.jit
def triton_poi_fused_slice_scatter_18(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 525312
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 513) % 256
    x2 = (xindex // 131328)
    x3 = xindex % 131328
    x4 = xindex
    x0 = xindex % 513
    tmp12 = tl.load(in_ptr2 + (x4), xmask).to(tl.float32)
    tmp0 = x1
    tmp1 = tl.full([1], 1, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.load(in_ptr0 + ((-513) + x3 + (130815*x2)), tmp2 & xmask, other=0.0).to(tl.float32)
    tmp4 = tl.where(tmp2, tmp3, 0.0)
    tmp5 = tl.full([1], 0, tl.int64)
    tmp6 = tmp5 >= tmp1
    tmp7 = tl.load(in_ptr1 + ((-131328) + x3 + (393984*x2)), tmp6 & xmask, other=0.0).to(tl.float32)
    tmp8 = tl.where(tmp6, tmp7, 0.0)
    tmp9 = tl.full([1], 0, tl.int32)
    tmp10 = tl.full([1], 3, tl.int32)
    tmp11 = tmp9 == tmp10
    tmp13 = tl.full([1], 3, tl.int64)
    tmp14 = tmp5 < tmp13
    tmp15 = x0
    tmp16 = tl.full([1], 256, tl.int64)
    tmp17 = tmp15 >= tmp16
    tmp18 = tmp17 & tmp14
    tmp19 = (((-256) + x0 + (513*x1) + (787968*x2)) // 512) % 513
    tmp20 = tl.full([1], 512, tl.int64)
    tmp21 = tmp19 < tmp20
    tmp22 = tmp21 & tmp18
    tmp23 = tl.load(in_ptr3 + ((256*((((-256) + x0 + (513*x1) + (787968*x2)) // 262656) % 3)) + (1024*((((-256) + x0 + (513*x1) + (787968*x2)) // 787968) % 4)) + ((((-256) + x0 + (513*x1) + (787968*x2)) // 512) % 513)), tmp22 & xmask, other=0.0).to(tl.float32)
    tmp24 = tl.load(in_ptr4 + ((256*((((-256) + x0 + (513*x1) + (787968*x2)) // 262656) % 3)) + (1024*((((-256) + x0 + (513*x1) + (787968*x2)) // 787968) % 4)) + (((-256) + x0 + (513*x1) + (787968*x2)) % 512)), tmp22 & xmask, other=0.0).to(tl.float32)
    tmp25 = tmp23 * tmp24
    tmp26 = tl.where(tmp22, tmp25, 0.0)
    tmp27 = tl.where(tmp18, tmp26, 0.0)
    tmp28 = 0.0
    tmp29 = tl.where(tmp17, tmp27, tmp28)
    tmp30 = tl.where(tmp14, tmp29, 0.0)
    tmp31 = tl.where(tmp14, tmp30, tmp28)
    tmp32 = tl.where(tmp11, tmp12, tmp31)
    tmp33 = tl.where(tmp6, tmp8, tmp32)
    tmp34 = tl.where(tmp2, tmp4, tmp33)
    tl.store(out_ptr0 + (x4), tmp34, xmask)
''')
