	.file	"/tmp/099d67b1-aeb1-4d66-b8ef-f0edf23744cc.TMP"
	.text
	.globl	_Z4sqrtf
	.align	16, 0x90
	.type	_Z4sqrtf,@function
_Z4sqrtf:                               # @_Z4sqrtf
# BB#0:
	sub	RSP, 8
	call	__ocl_svml_h8_sqrtf1
	add	RSP, 8
	ret
.Ltmp0:
	.size	_Z4sqrtf, .Ltmp0-_Z4sqrtf

	.globl	_Z4fabsf
	.align	16, 0x90
	.type	_Z4fabsf,@function
_Z4fabsf:                               # @_Z4fabsf
# BB#0:
                                        # kill: XMM0<def> XMM0<kill> XMM0<def>
	mov	EAX, 2147483647
	movd	XMM1, EAX
	andpd	XMM1, XMM0
	movapd	XMM0, XMM1
	ret
.Ltmp1:
	.size	_Z4fabsf, .Ltmp1-_Z4fabsf

	.globl	allOne
	.align	16, 0x90
	.type	allOne,@function
allOne:                                 # @allOne
# BB#0:                                 # %entry
	mov	AL, DIL
	ret
.Ltmp2:
	.size	allOne, .Ltmp2-allOne

	.globl	allZero
	.align	16, 0x90
	.type	allZero,@function
allZero:                                # @allZero
# BB#0:                                 # %entry
	mov	AL, DIL
	xor	%al, 1
	ret
.Ltmp3:
	.size	allZero, .Ltmp3-allZero

	.globl	allZero_v4
	.align	16, 0x90
	.type	allZero_v4,@function
allZero_v4:                             # @allZero_v4
# BB#0:                                 # %entry
	xor	CL, 1
	xor	DL, 1
	and	DL, CL
	xor	SIL, 1
	xor	DIL, 1
	and	DIL, SIL
	mov	AL, DIL
	and	AL, DL
	ret
.Ltmp4:
	.size	allZero_v4, .Ltmp4-allZero_v4

	.globl	allOne_v4
	.align	16, 0x90
	.type	allOne_v4,@function
allOne_v4:                              # @allOne_v4
# BB#0:                                 # %entry
	and	EDX, ECX
	and	EDI, ESI
	and	EDI, EDX
	mov	AL, DIL
	ret
.Ltmp5:
	.size	allOne_v4, .Ltmp5-allOne_v4

	.globl	_Z4sqrtDv4_f
	.align	16, 0x90
	.type	_Z4sqrtDv4_f,@function
_Z4sqrtDv4_f:                           # @_Z4sqrtDv4_f
# BB#0:
	sub	RSP, 8
	call	__ocl_svml_h8_sqrtf4
	add	RSP, 8
	ret
.Ltmp6:
	.size	_Z4sqrtDv4_f, .Ltmp6-_Z4sqrtDv4_f

	.section	.rodata.cst16,"aM",@progbits,16
	.align	16
.LCPI7_0:                               # constant pool <4 x i32>
	.long	2147483647              # 0x7fffffff
	.long	2147483647              # 0x7fffffff
	.long	2147483647              # 0x7fffffff
	.long	2147483647              # 0x7fffffff
	.text
	.globl	_Z4fabsDv4_f
	.align	16, 0x90
	.type	_Z4fabsDv4_f,@function
_Z4fabsDv4_f:                           # @_Z4fabsDv4_f
# BB#0:
	andps	XMM0, XMMWORD PTR [RIP + .LCPI7_0]
	ret
.Ltmp7:
	.size	_Z4fabsDv4_f, .Ltmp7-_Z4fabsDv4_f

	.globl	allZero_v4_i32
	.align	16, 0x90
	.type	allZero_v4_i32,@function
allZero_v4_i32:                         # @allZero_v4_i32
# BB#0:                                 # %entry
	ptest 	XMM0, XMM0
	sete	AL
	ret
.Ltmp8:
	.size	allZero_v4_i32, .Ltmp8-allZero_v4_i32

	.globl	allOne_v4_i32
	.align	16, 0x90
	.type	allOne_v4_i32,@function
allOne_v4_i32:                          # @allOne_v4_i32
# BB#0:                                 # %entry
	pcmpeqd	XMM1, XMM1
	ptest 	XMM0, XMM1
	setb	AL
	ret
.Ltmp9:
	.size	allOne_v4_i32, .Ltmp9-allOne_v4_i32

	.section	.rodata.cst4,"aM",@progbits,4
	.align	4
.LCPI10_0:                              # constant pool float
	.long	1065353216              # float 1.000000e+00
.LCPI10_1:                              # constant pool float
	.long	3204448256              # float -5.000000e-01
	.text
	.globl	adt_calc
	.align	16, 0x90
	.type	adt_calc,@function
adt_calc:                               # @adt_calc
# BB#0:
	push	R15
	push	R14
	push	R13
	push	R12
	push	RBX
	sub	RSP, 112
	movss	DWORD PTR [RSP + 44], XMM2 # 4-byte Spill
	mov	RBX, R9
	mov	R14, RCX
	mov	R15, RDX
	mov	R12, RSI
	mov	R13, RDI
	movss	XMM2, DWORD PTR [RIP + .LCPI10_0]
	divss	XMM2, DWORD PTR [R8]
	movss	XMM3, DWORD PTR [R8 + 8]
	mulss	XMM3, XMM2
	movss	DWORD PTR [RSP + 108], XMM3 # 4-byte Spill
	movaps	XMM4, XMM3
	mulss	XMM4, XMM4
	movss	XMM5, DWORD PTR [R8 + 4]
	mulss	XMM5, XMM2
	movaps	XMMWORD PTR [RSP + 80], XMM5 # 16-byte Spill
	movaps	XMM6, XMM5
	mulss	XMM6, XMM6
	addss	XMM6, XMM4
	mulss	XMM6, DWORD PTR [RIP + .LCPI10_1]
	mulss	XMM2, DWORD PTR [R8 + 12]
	addss	XMM2, XMM6
	mulss	XMM0, XMM1
	mulss	XMM0, XMM2
	call	__ocl_svml_h8_sqrtf1
	movss	DWORD PTR [RSP + 76], XMM0 # 4-byte Spill
	movss	XMM0, DWORD PTR [R12]
	movss	XMM1, DWORD PTR [R12 + 4]
	subss	XMM1, DWORD PTR [R13 + 4]
	movaps	XMM5, XMMWORD PTR [RSP + 80] # 16-byte Reload
	movaps	XMM2, XMM5
	mulss	XMM2, XMM1
	movaps	XMMWORD PTR [RSP + 16], XMM2 # 16-byte Spill
	subss	XMM0, DWORD PTR [R13]
	movss	XMM4, DWORD PTR [RSP + 108] # 4-byte Reload
	mulss	XMM4, XMM0
	movss	DWORD PTR [RSP + 48], XMM4 # 4-byte Spill
	mulss	XMM1, XMM1
	mulss	XMM0, XMM0
	addss	XMM0, XMM1
	call	__ocl_svml_h8_sqrtf1
	mulss	XMM0, DWORD PTR [RSP + 76] # 4-byte Folded Reload
	movapd	XMM2, XMMWORD PTR [RSP + 16] # 16-byte Reload
	subss	XMM2, DWORD PTR [RSP + 48] # 4-byte Folded Reload
	mov	EAX, 2147483647
	movd	XMM1, EAX
	movapd	XMMWORD PTR [RSP + 48], XMM1 # 16-byte Spill
	andpd	XMM2, XMM1
	addss	XMM2, XMM0
	movss	DWORD PTR [RBX], XMM2
	movss	XMM0, DWORD PTR [R15]
	movss	XMM1, DWORD PTR [R15 + 4]
	subss	XMM1, DWORD PTR [R12 + 4]
	movaps	XMM5, XMMWORD PTR [RSP + 80] # 16-byte Reload
	movaps	XMM2, XMM5
	mulss	XMM2, XMM1
	movaps	XMMWORD PTR [RSP + 16], XMM2 # 16-byte Spill
	subss	XMM0, DWORD PTR [R12]
	movss	XMM4, DWORD PTR [RSP + 108] # 4-byte Reload
	mulss	XMM4, XMM0
	movss	DWORD PTR [RSP + 12], XMM4 # 4-byte Spill
	mulss	XMM1, XMM1
	mulss	XMM0, XMM0
	addss	XMM0, XMM1
	call	__ocl_svml_h8_sqrtf1
	mulss	XMM0, DWORD PTR [RSP + 76] # 4-byte Folded Reload
	movaps	XMM2, XMMWORD PTR [RSP + 16] # 16-byte Reload
	subss	XMM2, DWORD PTR [RSP + 12] # 4-byte Folded Reload
	andps	XMM2, XMMWORD PTR [RSP + 48] # 16-byte Folded Reload
	addss	XMM2, XMM0
	addss	XMM2, DWORD PTR [RBX]
	movss	DWORD PTR [RBX], XMM2
	movss	XMM0, DWORD PTR [R14]
	movss	XMM1, DWORD PTR [R14 + 4]
	subss	XMM1, DWORD PTR [R15 + 4]
	movaps	XMM5, XMMWORD PTR [RSP + 80] # 16-byte Reload
	movaps	XMM2, XMM5
	mulss	XMM2, XMM1
	movaps	XMMWORD PTR [RSP + 16], XMM2 # 16-byte Spill
	subss	XMM0, DWORD PTR [R15]
	movss	XMM4, DWORD PTR [RSP + 108] # 4-byte Reload
	mulss	XMM4, XMM0
	movss	DWORD PTR [RSP + 12], XMM4 # 4-byte Spill
	mulss	XMM1, XMM1
	mulss	XMM0, XMM0
	addss	XMM0, XMM1
	call	__ocl_svml_h8_sqrtf1
	mulss	XMM0, DWORD PTR [RSP + 76] # 4-byte Folded Reload
	movaps	XMM2, XMMWORD PTR [RSP + 16] # 16-byte Reload
	subss	XMM2, DWORD PTR [RSP + 12] # 4-byte Folded Reload
	andps	XMM2, XMMWORD PTR [RSP + 48] # 16-byte Folded Reload
	addss	XMM2, XMM0
	addss	XMM2, DWORD PTR [RBX]
	movss	DWORD PTR [RBX], XMM2
	movss	XMM0, DWORD PTR [R13]
	movss	XMM1, DWORD PTR [R13 + 4]
	subss	XMM1, DWORD PTR [R14 + 4]
	movaps	XMM5, XMMWORD PTR [RSP + 80] # 16-byte Reload
	mulss	XMM5, XMM1
	movaps	XMMWORD PTR [RSP + 80], XMM5 # 16-byte Spill
	subss	XMM0, DWORD PTR [R14]
	movss	XMM3, DWORD PTR [RSP + 108] # 4-byte Reload
	mulss	XMM3, XMM0
	movss	DWORD PTR [RSP + 108], XMM3 # 4-byte Spill
	mulss	XMM1, XMM1
	mulss	XMM0, XMM0
	addss	XMM0, XMM1
	call	__ocl_svml_h8_sqrtf1
	mulss	XMM0, DWORD PTR [RSP + 76] # 4-byte Folded Reload
	movaps	XMM5, XMMWORD PTR [RSP + 80] # 16-byte Reload
	subss	XMM5, DWORD PTR [RSP + 108] # 4-byte Folded Reload
	andps	XMM5, XMMWORD PTR [RSP + 48] # 16-byte Folded Reload
	addss	XMM5, XMM0
	addss	XMM5, DWORD PTR [RBX]
	divss	XMM5, DWORD PTR [RSP + 44] # 4-byte Folded Reload
	movss	DWORD PTR [RBX], XMM5
	add	RSP, 112
	pop	RBX
	pop	R12
	pop	R13
	pop	R14
	pop	R15
	ret
.Ltmp10:
	.size	adt_calc, .Ltmp10-adt_calc

	.section	.rodata.cst4,"aM",@progbits,4
	.align	4
.LCPI11_0:                              # constant pool float
	.long	1065353216              # float 1.000000e+00
.LCPI11_1:                              # constant pool float
	.long	3204448256              # float -5.000000e-01
	.text
	.globl	op_opencl_adt_calc
	.align	16, 0x90
	.type	op_opencl_adt_calc,@function
op_opencl_adt_calc:                     # @op_opencl_adt_calc
# BB#0:                                 # %FirstBB
	push	RBP
	push	R15
	push	R14
	push	R13
	push	R12
	push	RBX
	sub	RSP, 168
	mov	EAX, DWORD PTR [RSP + 336]
	lea	ECX, DWORD PTR [RAX + 2*RAX]
	mov	DWORD PTR [RSP + 32], ECX # 4-byte Spill
	lea	EAX, DWORD PTR [RAX + RAX]
	mov	DWORD PTR [RSP + 28], EAX # 4-byte Spill
	mov	DWORD PTR [RSP + 4], 7  # 4-byte Folded Spill
	mov	QWORD PTR [RSP + 88], 0 # 8-byte Folded Spill
	movsxd	RAX, DWORD PTR [RSP + 280]
	mov	QWORD PTR [RSP + 16], RAX # 8-byte Spill
	movsxd	RAX, DWORD PTR [RSP + 328]
	mov	QWORD PTR [RSP + 8], RAX # 8-byte Spill
	mov	RBX, QWORD PTR [RSP + 376]
	jmp	.LBB11_1
	.align	16, 0x90
.LBB11_16:                              # %thenBB18
                                        #   in Loop: Header=BB11_1 Depth=1
	inc	QWORD PTR [RSP + 88]    # 8-byte Folded Spill
	cmp	DWORD PTR [RSP + 4], 1  # 4-byte Folded Reload
	je	.LBB11_12
.LBB11_1:                               # %SyncBB10.outer
                                        # =>This Loop Header: Depth=1
                                        #     Child Loop BB11_14 Depth 2
                                        #     Child Loop BB11_7 Depth 2
                                        #       Child Loop BB11_9 Depth 3
                                        #     Child Loop BB11_2 Depth 2
	mov	RAX, QWORD PTR [RSP + 88] # 8-byte Reload
	shl	RAX, 5
	add	RAX, QWORD PTR [RSP + 408]
	jmp	.LBB11_2
	.align	16, 0x90
.LBB11_18:                              # %thenBB
                                        #   in Loop: Header=BB11_2 Depth=2
	add	RAX, 32
	inc	QWORD PTR [RSP + 88]    # 8-byte Folded Spill
.LBB11_2:                               # %SyncBB10
                                        #   Parent Loop BB11_1 Depth=1
                                        # =>  This Inner Loop Header: Depth=2
	mov	RCX, QWORD PTR [RSP + 384]
	mov	RCX, QWORD PTR [RCX + 80]
	mov	RDX, QWORD PTR [RSP + 392]
	imul	RCX, QWORD PTR [RDX + 8]
	add	RCX, QWORD PTR [RDX]
	cmp	RCX, QWORD PTR [RSP + 8] # 8-byte Folded Reload
	jae	.LBB11_15
# BB#3:                                 #   in Loop: Header=BB11_2 Depth=2
	cmp	QWORD PTR [RAX], 0
	jne	.LBB11_5
# BB#4:                                 #   in Loop: Header=BB11_2 Depth=2
	mov	RCX, RDX
	mov	RDX, QWORD PTR [RCX]
	add	RDX, QWORD PTR [RSP + 16] # 8-byte Folded Reload
	mov	RSI, QWORD PTR [RSP + 384]
	mov	RSI, QWORD PTR [RSI + 80]
	imul	RSI, QWORD PTR [RCX + 8]
	add	RSI, RDX
	mov	RCX, QWORD PTR [RSP + 288]
	movsxd	RCX, DWORD PTR [RCX + 4*RSI]
	mov	RDX, QWORD PTR [RSP + 304]
	mov	EDX, DWORD PTR [RDX + 4*RCX]
	mov	DWORD PTR [RBX + 384], EDX
	mov	RDX, QWORD PTR [RSP + 296]
	mov	EDX, DWORD PTR [RDX + 4*RCX]
	mov	DWORD PTR [RBX + 512], EDX
	mov	RDX, QWORD PTR [RSP + 264]
	mov	EDX, DWORD PTR [RDX + 4*RCX]
	mov	DWORD PTR [RBX + 128], EDX
	mov	RDX, QWORD PTR [RSP + 272]
	movsxd	RCX, DWORD PTR [RDX + 4*RCX]
	mov	RDX, QWORD PTR [RSP + 232]
	lea	RCX, QWORD PTR [RDX + 4*RCX]
	mov	QWORD PTR [RBX], RCX
	mov	RCX, QWORD PTR [RSP + 344]
	mov	QWORD PTR [RBX + 256], RCX
.LBB11_5:                               #   in Loop: Header=BB11_2 Depth=2
	mov	RCX, QWORD PTR [RSP + 88] # 8-byte Reload
	cmp	RCX, QWORD PTR [RSP + 424]
	jb	.LBB11_18
# BB#6:                                 # %.SyncBB_crit_edge
                                        #   in Loop: Header=BB11_1 Depth=1
	mov	RAX, -1
	mov	RCX, QWORD PTR [RSP + 408]
	.align	16, 0x90
.LBB11_7:                               # %SyncBB
                                        #   Parent Loop BB11_1 Depth=1
                                        # =>  This Loop Header: Depth=2
                                        #       Child Loop BB11_9 Depth 3
	mov	EDX, DWORD PTR [RBX + 128]
	add	EDX, EDX
	mov	RSI, QWORD PTR [RCX]
	cmp	ESI, EDX
	jge	.LBB11_10
# BB#8:                                 # %SyncBB.bb.nph7_crit_edge
                                        #   in Loop: Header=BB11_7 Depth=2
	mov	RDX, RSI
	.align	16, 0x90
.LBB11_9:                               # %bb.nph7
                                        #   Parent Loop BB11_1 Depth=1
                                        #     Parent Loop BB11_7 Depth=2
                                        # =>    This Inner Loop Header: Depth=3
	mov	EDI, ESI
	shr	EDI, 31
	add	EDI, ESI
	mov	R8D, EDI
	and	R8D, -2
	mov	R9D, ESI
	sub	R9D, R8D
	sar	EDI
	movsxd	RDI, EDI
	mov	R8, QWORD PTR [RBX]
	mov	R10, QWORD PTR [RBX + 256]
	mov	EDI, DWORD PTR [R8 + 4*RDI]
	lea	EDI, DWORD PTR [R9 + 2*RDI]
	movsxd	RDI, EDI
	mov	R8, QWORD PTR [RSP + 224]
	movss	XMM0, DWORD PTR [R8 + 4*RDI]
	movsxd	RSI, ESI
	movss	DWORD PTR [R10 + 4*RSI], XMM0
	mov	ESI, EDX
	mov	RDX, QWORD PTR [RSP + 384]
	add	RSI, QWORD PTR [RDX + 56]
	mov	EDX, DWORD PTR [RBX + 128]
	add	EDX, EDX
	cmp	ESI, EDX
	mov	RDX, RSI
	jl	.LBB11_9
.LBB11_10:                              # %._crit_edge
                                        #   in Loop: Header=BB11_7 Depth=2
	add	RCX, 32
	inc	RAX
	cmp	RAX, QWORD PTR [RSP + 424]
	jb	.LBB11_7
# BB#11:                                # %elseBB12
                                        #   in Loop: Header=BB11_1 Depth=1
	mfence
	mov	DWORD PTR [RSP + 4], 1  # 4-byte Folded Spill
	mov	QWORD PTR [RSP + 88], 0 # 8-byte Folded Spill
.LBB11_12:                              # %SyncBB8
                                        #   in Loop: Header=BB11_1 Depth=1
	mov	RAX, QWORD PTR [RSP + 88] # 8-byte Reload
	shl	RAX, 5
	mov	RCX, QWORD PTR [RSP + 408]
	mov	RAX, QWORD PTR [RCX + RAX]
	cmp	EAX, DWORD PTR [RBX + 384]
	jge	.LBB11_15
# BB#13:                                # %SyncBB8.bb.nph_crit_edge
                                        #   in Loop: Header=BB11_1 Depth=1
	mov	QWORD PTR [RSP + 80], RAX # 8-byte Spill
	.align	16, 0x90
.LBB11_14:                              # %bb.nph
                                        #   Parent Loop BB11_1 Depth=1
                                        # =>  This Inner Loop Header: Depth=2
	mov	ECX, DWORD PTR [RBX + 512]
	lea	EDX, DWORD PTR [RCX + RAX]
	lea	ESI, DWORD PTR [4*RDX]
	movsxd	RSI, ESI
	movss	XMM0, DWORD PTR [RIP + .LCPI11_0]
	movaps	XMM1, XMM0
	mov	RDI, QWORD PTR [RSP + 248]
	divss	XMM1, DWORD PTR [RDI + 4*RSI]
	movss	XMM0, DWORD PTR [RDI + 4*RSI + 8]
	mulss	XMM0, XMM1
	movss	DWORD PTR [RSP + 164], XMM0 # 4-byte Spill
	movaps	XMM2, XMM0
	mulss	XMM2, XMM2
	movss	XMM3, DWORD PTR [RDI + 4*RSI + 4]
	mulss	XMM3, XMM1
	movaps	XMMWORD PTR [RSP + 144], XMM3 # 16-byte Spill
	movaps	XMM4, XMM3
	mulss	XMM4, XMM4
	addss	XMM4, XMM2
	mulss	XMM4, DWORD PTR [RIP + .LCPI11_1]
	mulss	XMM1, DWORD PTR [RDI + 4*RSI + 12]
	addss	XMM1, XMM4
	mov	RSI, QWORD PTR [RSP + 352]
	movss	XMM0, DWORD PTR [RSI]
	mov	RSI, QWORD PTR [RSP + 360]
	mulss	XMM0, DWORD PTR [RSI]
	mulss	XMM0, XMM1
	mov	ESI, DWORD PTR [RSP + 32] # 4-byte Reload
	lea	ESI, DWORD PTR [RAX + RSI]
	add	ESI, ECX
	movsxd	RSI, ESI
	mov	RDI, QWORD PTR [RSP + 240]
	movsx	ESI, WORD PTR [RDI + 2*RSI]
	mov	DWORD PTR [RSP + 96], ESI # 4-byte Spill
	mov	R8D, DWORD PTR [RSP + 28] # 4-byte Reload
	lea	R8D, DWORD PTR [RAX + R8]
	add	R8D, ECX
	movsxd	R8, R8D
	movsx	R14D, WORD PTR [RDI + 2*R8]
	movsxd	R15, EDX
	movsx	R12D, WORD PTR [RDI + 2*R15]
	add	EAX, DWORD PTR [RSP + 336]
	add	EAX, ECX
	movsxd	RAX, EAX
	movsx	R13D, WORD PTR [RDI + 2*RAX]
	mov	RAX, QWORD PTR [RSP + 368]
	movss	XMM1, DWORD PTR [RAX]
	movss	DWORD PTR [RSP + 76], XMM1 # 4-byte Spill
	mov	RBP, QWORD PTR [RBX + 256]
	call	__ocl_svml_h8_sqrtf1
	movss	DWORD PTR [RSP + 140], XMM0 # 4-byte Spill
	add	R13D, R13D
	movsxd	R13, R13D
	movss	XMM1, DWORD PTR [RBP + 4*R13 + 4]
	add	R12D, R12D
	movsxd	RAX, R12D
	mov	QWORD PTR [RSP + 40], RAX # 8-byte Spill
	subss	XMM1, DWORD PTR [RBP + 4*RAX + 4]
	movaps	XMM3, XMMWORD PTR [RSP + 144] # 16-byte Reload
	movaps	XMM2, XMM3
	mulss	XMM2, XMM1
	movaps	XMMWORD PTR [RSP + 48], XMM2 # 16-byte Spill
	movss	XMM0, DWORD PTR [RBP + 4*R13]
	subss	XMM0, DWORD PTR [RBP + 4*RAX]
	movss	XMM4, DWORD PTR [RSP + 164] # 4-byte Reload
	mulss	XMM4, XMM0
	movss	DWORD PTR [RSP + 112], XMM4 # 4-byte Spill
	mulss	XMM1, XMM1
	mulss	XMM0, XMM0
	addss	XMM0, XMM1
	call	__ocl_svml_h8_sqrtf1
	mulss	XMM0, DWORD PTR [RSP + 140] # 4-byte Folded Reload
	movapd	XMM2, XMMWORD PTR [RSP + 48] # 16-byte Reload
	subss	XMM2, DWORD PTR [RSP + 112] # 4-byte Folded Reload
	mov	ECX, 2147483647
	movd	XMM1, ECX
	movapd	XMMWORD PTR [RSP + 112], XMM1 # 16-byte Spill
	andpd	XMM2, XMM1
	addss	XMM2, XMM0
	mov	R12, QWORD PTR [RSP + 256]
	movss	DWORD PTR [R12 + 4*R15], XMM2
	add	R14D, R14D
	movsxd	R14, R14D
	movss	XMM1, DWORD PTR [RBP + 4*R14 + 4]
	subss	XMM1, DWORD PTR [RBP + 4*R13 + 4]
	movapd	XMM3, XMMWORD PTR [RSP + 144] # 16-byte Reload
	movaps	XMM2, XMM3
	mulss	XMM2, XMM1
	movapd	XMMWORD PTR [RSP + 48], XMM2 # 16-byte Spill
	movss	XMM0, DWORD PTR [RBP + 4*R14]
	subss	XMM0, DWORD PTR [RBP + 4*R13]
	movss	XMM4, DWORD PTR [RSP + 164] # 4-byte Reload
	mulss	XMM4, XMM0
	movss	DWORD PTR [RSP + 36], XMM4 # 4-byte Spill
	mulss	XMM1, XMM1
	mulss	XMM0, XMM0
	addss	XMM0, XMM1
	call	__ocl_svml_h8_sqrtf1
	mulss	XMM0, DWORD PTR [RSP + 140] # 4-byte Folded Reload
	movaps	XMM2, XMMWORD PTR [RSP + 48] # 16-byte Reload
	subss	XMM2, DWORD PTR [RSP + 36] # 4-byte Folded Reload
	andps	XMM2, XMMWORD PTR [RSP + 112] # 16-byte Folded Reload
	addss	XMM2, XMM0
	addss	XMM2, DWORD PTR [R12 + 4*R15]
	movss	DWORD PTR [R12 + 4*R15], XMM2
	mov	ESI, DWORD PTR [RSP + 96] # 4-byte Reload
	add	ESI, ESI
	mov	DWORD PTR [RSP + 96], ESI # 4-byte Spill
	movsxd	R13, ESI
	movss	XMM1, DWORD PTR [RBP + 4*R13 + 4]
	subss	XMM1, DWORD PTR [RBP + 4*R14 + 4]
	movaps	XMM3, XMMWORD PTR [RSP + 144] # 16-byte Reload
	movaps	XMM2, XMM3
	mulss	XMM2, XMM1
	movaps	XMMWORD PTR [RSP + 96], XMM2 # 16-byte Spill
	movss	XMM0, DWORD PTR [RBP + 4*R13]
	subss	XMM0, DWORD PTR [RBP + 4*R14]
	movss	XMM4, DWORD PTR [RSP + 164] # 4-byte Reload
	mulss	XMM4, XMM0
	movss	DWORD PTR [RSP + 48], XMM4 # 4-byte Spill
	mulss	XMM1, XMM1
	mulss	XMM0, XMM0
	addss	XMM0, XMM1
	call	__ocl_svml_h8_sqrtf1
	mulss	XMM0, DWORD PTR [RSP + 140] # 4-byte Folded Reload
	movaps	XMM2, XMMWORD PTR [RSP + 96] # 16-byte Reload
	subss	XMM2, DWORD PTR [RSP + 48] # 4-byte Folded Reload
	andps	XMM2, XMMWORD PTR [RSP + 112] # 16-byte Folded Reload
	addss	XMM2, XMM0
	addss	XMM2, DWORD PTR [R12 + 4*R15]
	movss	DWORD PTR [R12 + 4*R15], XMM2
	mov	RAX, QWORD PTR [RSP + 40] # 8-byte Reload
	movss	XMM1, DWORD PTR [RBP + 4*RAX + 4]
	subss	XMM1, DWORD PTR [RBP + 4*R13 + 4]
	movaps	XMM3, XMMWORD PTR [RSP + 144] # 16-byte Reload
	mulss	XMM3, XMM1
	movaps	XMMWORD PTR [RSP + 144], XMM3 # 16-byte Spill
	movss	XMM0, DWORD PTR [RBP + 4*RAX]
	subss	XMM0, DWORD PTR [RBP + 4*R13]
	movss	XMM2, DWORD PTR [RSP + 164] # 4-byte Reload
	mulss	XMM2, XMM0
	movss	DWORD PTR [RSP + 164], XMM2 # 4-byte Spill
	mulss	XMM1, XMM1
	mulss	XMM0, XMM0
	addss	XMM0, XMM1
	call	__ocl_svml_h8_sqrtf1
	mulss	XMM0, DWORD PTR [RSP + 140] # 4-byte Folded Reload
	movaps	XMM3, XMMWORD PTR [RSP + 144] # 16-byte Reload
	subss	XMM3, DWORD PTR [RSP + 164] # 4-byte Folded Reload
	andps	XMM3, XMMWORD PTR [RSP + 112] # 16-byte Folded Reload
	addss	XMM3, XMM0
	addss	XMM3, DWORD PTR [R12 + 4*R15]
	divss	XMM3, DWORD PTR [RSP + 76] # 4-byte Folded Reload
	movss	DWORD PTR [R12 + 4*R15], XMM3
	mov	RAX, QWORD PTR [RSP + 80] # 8-byte Reload
	mov	EAX, EAX
	mov	RCX, QWORD PTR [RSP + 384]
	add	RAX, QWORD PTR [RCX + 56]
	cmp	EAX, DWORD PTR [RBX + 384]
	mov	QWORD PTR [RSP + 80], RAX # 8-byte Spill
	jl	.LBB11_14
.LBB11_15:                              # %.loopexit
                                        #   in Loop: Header=BB11_1 Depth=1
	mov	RAX, QWORD PTR [RSP + 88] # 8-byte Reload
	cmp	RAX, QWORD PTR [RSP + 424]
	jb	.LBB11_16
# BB#17:                                # %SyncBB9
	add	RSP, 168
	pop	RBX
	pop	R12
	pop	R13
	pop	R14
	pop	R15
	pop	RBP
	ret
.Ltmp11:
	.size	op_opencl_adt_calc, .Ltmp11-op_opencl_adt_calc

	.section	.rodata.cst16,"aM",@progbits,16
	.align	16
.LCPI12_0:                              # constant pool <2 x i64>
	.quad	2                       # 0x2
	.quad	3                       # 0x3
.LCPI12_1:                              # constant pool <2 x i64>
	.quad	4294967295              # 0xffffffff
	.quad	4294967295              # 0xffffffff
.LCPI12_2:                              # constant pool <4 x float>
	.long	1065353216              # float 1.000000e+00
	.long	1065353216              # float 1.000000e+00
	.long	1065353216              # float 1.000000e+00
	.long	1065353216              # float 1.000000e+00
.LCPI12_3:                              # constant pool <2 x i64>
	.quad	1                       # 0x1
	.quad	1                       # 0x1
.LCPI12_4:                              # constant pool <2 x i64>
	.quad	2                       # 0x2
	.quad	2                       # 0x2
.LCPI12_5:                              # constant pool <2 x i64>
	.quad	3                       # 0x3
	.quad	3                       # 0x3
.LCPI12_6:                              # constant pool <4 x float>
	.long	1056964608              # float 5.000000e-01
	.long	1056964608              # float 5.000000e-01
	.long	1056964608              # float 5.000000e-01
	.long	1056964608              # float 5.000000e-01
.LCPI12_7:                              # constant pool <4 x i32>
	.long	2147483647              # 0x7fffffff
	.long	2147483647              # 0x7fffffff
	.long	2147483647              # 0x7fffffff
	.long	2147483647              # 0x7fffffff
.LCPI12_8:                              # constant pool <4 x i32>
	.long	4294967295              # 0xffffffff
	.long	4294967295              # 0xffffffff
	.long	4294967295              # 0xffffffff
	.long	4294967295              # 0xffffffff
	.text
	.globl	__Vectorized_.op_opencl_adt_calc
	.align	16, 0x90
	.type	__Vectorized_.op_opencl_adt_calc,@function
__Vectorized_.op_opencl_adt_calc:       # @__Vectorized_.op_opencl_adt_calc
# BB#0:                                 # %deload
	push	RBP
	push	R15
	push	R14
	push	R13
	push	R12
	push	RBX
	sub	RSP, 776
	movsxd	RAX, DWORD PTR [RSP + 888]
	movq	XMM0, RAX
	movlhps	XMM0, XMM0              # xmm0 = xmm0[0,0]
	movaps	XMMWORD PTR [RSP + 16], XMM0 # 16-byte Spill
	mov	EAX, DWORD PTR [RSP + 944]
	movd	XMM0, EAX
	pshufd	XMM0, XMM0, 0           # xmm0 = xmm0[0,0,0,0]
	movdqa	XMMWORD PTR [RSP], XMM0 # 16-byte Spill
	lea	ECX, DWORD PTR [RAX + 2*RAX]
	movd	XMM0, ECX
	pshufd	XMM0, XMM0, 0           # xmm0 = xmm0[0,0,0,0]
	movdqa	XMMWORD PTR [RSP + 128], XMM0 # 16-byte Spill
	add	EAX, EAX
	movd	XMM0, EAX
	pshufd	XMM0, XMM0, 0           # xmm0 = xmm0[0,0,0,0]
	movdqa	XMMWORD PTR [RSP + 112], XMM0 # 16-byte Spill
	mov	QWORD PTR [RSP + 104], 0 # 8-byte Folded Spill
	mov	DWORD PTR [RSP + 100], 6 # 4-byte Folded Spill
	movsxd	RAX, DWORD PTR [RSP + 936]
	mov	QWORD PTR [RSP + 48], RAX # 8-byte Spill
	jmp	.LBB12_1
	.align	16, 0x90
.LBB12_152:                             # %deload946.postload947.loopexit_crit_edge
                                        #   in Loop: Header=BB12_1 Depth=1
	mov	QWORD PTR [RSP + 104], 0 # 8-byte Folded Spill
	mov	DWORD PTR [RSP + 100], 4 # 4-byte Folded Spill
	jmp	.LBB12_111
.LBB12_150:                             # %thenBB2519
                                        #   in Loop: Header=BB12_113 Depth=3
	add	QWORD PTR [RSP + 680], 32 # 8-byte Folded Spill
	inc	QWORD PTR [RSP + 104]   # 8-byte Folded Spill
	cmp	DWORD PTR [RSP + 100], 6 # 4-byte Folded Reload
	jne	.LBB12_113
.LBB12_1:                               # %SyncBB.outer
                                        # =>This Loop Header: Depth=1
                                        #     Child Loop BB12_112 Depth 2
                                        #       Child Loop BB12_156 Depth 3
                                        #         Child Loop BB12_158 Depth 4
                                        #       Child Loop BB12_113 Depth 3
                                        #         Child Loop BB12_115 Depth 4
                                        #     Child Loop BB12_2 Depth 2
	mov	RAX, QWORD PTR [RSP + 104] # 8-byte Reload
	shl	RAX, 5
	add	RAX, QWORD PTR [RSP + 1016]
	mov	QWORD PTR [RSP + 632], RAX # 8-byte Spill
	jmp	.LBB12_2
	.align	16, 0x90
.LBB12_151:                             # %deload946
                                        #   in Loop: Header=BB12_2 Depth=2
	mov	RAX, QWORD PTR [RSP + 104] # 8-byte Reload
	cmp	RAX, QWORD PTR [RSP + 1032]
	jae	.LBB12_152
# BB#153:                               # %thenBB2512
                                        #   in Loop: Header=BB12_2 Depth=2
	add	QWORD PTR [RSP + 632], 32 # 8-byte Folded Spill
	inc	QWORD PTR [RSP + 104]   # 8-byte Folded Spill
.LBB12_2:                               # %SyncBB
                                        #   Parent Loop BB12_1 Depth=1
                                        # =>  This Inner Loop Header: Depth=2
	mov	RAX, QWORD PTR [RSP + 1000]
	movq	XMM0, QWORD PTR [RAX + 8]
	movlhps	XMM0, XMM0              # xmm0 = xmm0[0,0]
	mov	RCX, QWORD PTR [RSP + 992]
	movq	XMM1, QWORD PTR [RCX + 80]
	movlhps	XMM1, XMM1              # xmm1 = xmm1[0,0]
	movaps	XMM2, XMM1
	pmuludq	XMM2, XMM0
	movdqa	XMM3, XMM0
	psrlq	XMM3, 32
	pmuludq	XMM3, XMM1
	psllq	XMM3, 32
	paddq	XMM3, XMM2
	psrlq	XMM1, 32
	pmuludq	XMM1, XMM0
	psllq	XMM1, 32
	paddq	XMM1, XMM3
	movq	XMM0, QWORD PTR [RAX]
	movlhps	XMM0, XMM0              # xmm0 = xmm0[0,0]
	paddq	XMM0, XMM1
	pextrq	RAX, XMM0, 1
	mov	RCX, QWORD PTR [RSP + 48] # 8-byte Reload
	cmp	RAX, RCX
	mov	EAX, 0
	mov	EDX, -1
	cmovb	EAX, EDX
	movq	RSI, XMM0
	mov	QWORD PTR [RSP + 40], RSI # 8-byte Spill
	cmp	RSI, RCX
	mov	ECX, 0
	cmovb	ECX, EDX
	movd	XMM0, ECX
	pinsrd	XMM0, EAX, 1
	pinsrd	XMM0, ECX, 2
	pinsrd	XMM0, EAX, 3
	movdqa	XMMWORD PTR [RSP + 80], XMM0 # 16-byte Spill
	mov	EAX, 1
	movq	XMM1, RAX
	pslldq	XMM1, 8
	movdqa	XMMWORD PTR [RSP + 64], XMM1 # 16-byte Spill
	mov	RAX, QWORD PTR [RSP + 632] # 8-byte Reload
	movq	XMM2, QWORD PTR [RAX]
	movlhps	XMM2, XMM2              # xmm2 = xmm2[0,0]
	movaps	XMM3, XMM2
	paddq	XMM3, XMM1
	pextrq	RAX, XMM3, 1
	cmp	RAX, 1
	sbb	EAX, EAX
	movq	RCX, XMM3
	cmp	RCX, 1
	sbb	ECX, ECX
	movd	XMM1, ECX
	pinsrd	XMM1, EAX, 1
	paddq	XMM2, XMMWORD PTR [RIP + .LCPI12_0]
	movq	RAX, XMM2
	cmp	RAX, 1
	sbb	EAX, EAX
	pinsrd	XMM1, EAX, 2
	pextrq	RAX, XMM2, 1
	cmp	RAX, 1
	sbb	EAX, EAX
	pinsrd	XMM1, EAX, 3
	pand	XMM1, XMM0
	movd	EAX, XMM1
	test	EAX, EAX
	pextrd	ECX, XMM1, 3
	pextrd	EDX, XMM1, 2
	pextrd	ESI, XMM1, 1
	jns	.LBB12_4
# BB#3:                                 # %deload750
                                        #   in Loop: Header=BB12_2 Depth=2
	mov	RDI, QWORD PTR [RSP + 1000]
	mov	RDI, QWORD PTR [RDI]
.LBB12_4:                               # %postload751
                                        #   in Loop: Header=BB12_2 Depth=2
	test	ESI, ESI
	jns	.LBB12_6
# BB#5:                                 # %deload784
                                        #   in Loop: Header=BB12_2 Depth=2
	mov	R8, QWORD PTR [RSP + 1000]
	mov	R8, QWORD PTR [R8]
.LBB12_6:                               # %postload785
                                        #   in Loop: Header=BB12_2 Depth=2
	test	EDX, EDX
	jns	.LBB12_8
# BB#7:                                 # %deload818
                                        #   in Loop: Header=BB12_2 Depth=2
	mov	R9, QWORD PTR [RSP + 1000]
	mov	R9, QWORD PTR [R9]
.LBB12_8:                               # %postload819
                                        #   in Loop: Header=BB12_2 Depth=2
	test	ECX, ECX
	jns	.LBB12_10
# BB#9:                                 # %deload852
                                        #   in Loop: Header=BB12_2 Depth=2
	mov	R10, QWORD PTR [RSP + 1000]
	mov	R10, QWORD PTR [R10]
.LBB12_10:                              # %postload853
                                        #   in Loop: Header=BB12_2 Depth=2
	movq	XMM0, R10
	movq	XMM1, R9
	punpcklqdq	XMM1, XMM0      # xmm1 = xmm1[0],xmm0[0]
	movq	XMM0, R8
	movq	XMM2, RDI
	punpcklqdq	XMM2, XMM0      # xmm2 = xmm2[0],xmm0[0]
	test	EAX, EAX
	jns	.LBB12_12
# BB#11:                                # %deload753
                                        #   in Loop: Header=BB12_2 Depth=2
	mov	RDI, QWORD PTR [RSP + 1000]
	mov	RDI, QWORD PTR [RDI + 8]
.LBB12_12:                              # %postload754
                                        #   in Loop: Header=BB12_2 Depth=2
	test	ESI, ESI
	jns	.LBB12_14
# BB#13:                                # %deload787
                                        #   in Loop: Header=BB12_2 Depth=2
	mov	R8, QWORD PTR [RSP + 1000]
	mov	R8, QWORD PTR [R8 + 8]
.LBB12_14:                              # %postload788
                                        #   in Loop: Header=BB12_2 Depth=2
	test	EDX, EDX
	jns	.LBB12_16
# BB#15:                                # %deload821
                                        #   in Loop: Header=BB12_2 Depth=2
	mov	R9, QWORD PTR [RSP + 1000]
	mov	R9, QWORD PTR [R9 + 8]
.LBB12_16:                              # %postload822
                                        #   in Loop: Header=BB12_2 Depth=2
	test	ECX, ECX
	jns	.LBB12_18
# BB#17:                                # %deload855
                                        #   in Loop: Header=BB12_2 Depth=2
	mov	R10, QWORD PTR [RSP + 1000]
	mov	R10, QWORD PTR [R10 + 8]
.LBB12_18:                              # %postload856
                                        #   in Loop: Header=BB12_2 Depth=2
	movq	XMM0, R10
	movq	XMM3, R9
	punpcklqdq	XMM3, XMM0      # xmm3 = xmm3[0],xmm0[0]
	movq	XMM0, R8
	movq	XMM4, RDI
	punpcklqdq	XMM4, XMM0      # xmm4 = xmm4[0],xmm0[0]
	test	EAX, EAX
	jns	.LBB12_20
# BB#19:                                # %deload756
                                        #   in Loop: Header=BB12_2 Depth=2
	mov	RDI, QWORD PTR [RSP + 992]
	mov	RDI, QWORD PTR [RDI + 80]
.LBB12_20:                              # %postload757
                                        #   in Loop: Header=BB12_2 Depth=2
	test	ESI, ESI
	jns	.LBB12_22
# BB#21:                                # %deload790
                                        #   in Loop: Header=BB12_2 Depth=2
	mov	R8, QWORD PTR [RSP + 992]
	mov	R8, QWORD PTR [R8 + 80]
.LBB12_22:                              # %postload791
                                        #   in Loop: Header=BB12_2 Depth=2
	test	EDX, EDX
	jns	.LBB12_24
# BB#23:                                # %deload824
                                        #   in Loop: Header=BB12_2 Depth=2
	mov	R9, QWORD PTR [RSP + 992]
	mov	R9, QWORD PTR [R9 + 80]
.LBB12_24:                              # %postload825
                                        #   in Loop: Header=BB12_2 Depth=2
	test	ECX, ECX
	jns	.LBB12_26
# BB#25:                                # %deload858
                                        #   in Loop: Header=BB12_2 Depth=2
	mov	R10, QWORD PTR [RSP + 992]
	mov	R10, QWORD PTR [R10 + 80]
.LBB12_26:                              # %postload859
                                        #   in Loop: Header=BB12_2 Depth=2
	movq	XMM0, R10
	movq	XMM5, R9
	punpcklqdq	XMM5, XMM0      # xmm5 = xmm5[0],xmm0[0]
	movdqa	XMM0, XMM5
	pmuludq	XMM0, XMM3
	movdqa	XMM6, XMM3
	psrlq	XMM6, 32
	pmuludq	XMM6, XMM5
	psllq	XMM6, 32
	paddq	XMM6, XMM0
	psrlq	XMM5, 32
	pmuludq	XMM5, XMM3
	psllq	XMM5, 32
	paddq	XMM5, XMM6
	movdqa	XMM0, XMMWORD PTR [RSP + 16] # 16-byte Reload
	paddq	XMM1, XMM0
	paddq	XMM1, XMM5
	pextrq	R9, XMM1, 1
	movq	R10, XMM1
	movq	XMM1, R8
	movq	XMM3, RDI
	punpcklqdq	XMM3, XMM1      # xmm3 = xmm3[0],xmm1[0]
	movdqa	XMM1, XMM3
	pmuludq	XMM1, XMM4
	movdqa	XMM5, XMM4
	psrlq	XMM5, 32
	pmuludq	XMM5, XMM3
	psllq	XMM5, 32
	paddq	XMM5, XMM1
	psrlq	XMM3, 32
	pmuludq	XMM3, XMM4
	psllq	XMM3, 32
	paddq	XMM3, XMM5
	paddq	XMM2, XMM0
	paddq	XMM2, XMM3
	pextrq	RDI, XMM2, 1
	test	EAX, EAX
	jns	.LBB12_28
# BB#27:                                # %deload759
                                        #   in Loop: Header=BB12_2 Depth=2
	movq	R8, XMM2
	mov	R11, QWORD PTR [RSP + 896]
	mov	R8D, DWORD PTR [R11 + 4*R8]
	mov	DWORD PTR [RSP + 672], R8D # 4-byte Spill
.LBB12_28:                              # %postload760
                                        #   in Loop: Header=BB12_2 Depth=2
	test	ESI, ESI
	jns	.LBB12_30
# BB#29:                                # %deload793
                                        #   in Loop: Header=BB12_2 Depth=2
	mov	R8, QWORD PTR [RSP + 896]
	mov	EDI, DWORD PTR [R8 + 4*RDI]
.LBB12_30:                              # %postload794
                                        #   in Loop: Header=BB12_2 Depth=2
	test	EDX, EDX
	jns	.LBB12_32
# BB#31:                                # %deload827
                                        #   in Loop: Header=BB12_2 Depth=2
	mov	R8, QWORD PTR [RSP + 896]
	mov	R8D, DWORD PTR [R8 + 4*R10]
.LBB12_32:                              # %postload828
                                        #   in Loop: Header=BB12_2 Depth=2
	test	ECX, ECX
	jns	.LBB12_34
# BB#33:                                # %deload861
                                        #   in Loop: Header=BB12_2 Depth=2
	mov	R10, QWORD PTR [RSP + 896]
	mov	R9D, DWORD PTR [R10 + 4*R9]
.LBB12_34:                              # %postload862
                                        #   in Loop: Header=BB12_2 Depth=2
	test	EAX, EAX
	movsxd	R9, R9D
	mov	QWORD PTR [RSP + 680], R9 # 8-byte Spill
	movsxd	R8, R8D
	mov	QWORD PTR [RSP + 656], R8 # 8-byte Spill
	movsxd	RDI, EDI
	mov	QWORD PTR [RSP + 640], RDI # 8-byte Spill
	jns	.LBB12_36
# BB#35:                                # %deload762
                                        #   in Loop: Header=BB12_2 Depth=2
	movsxd	RDI, DWORD PTR [RSP + 672] # 4-byte Folded Reload
	mov	R8, QWORD PTR [RSP + 912]
	mov	EDI, DWORD PTR [R8 + 4*RDI]
.LBB12_36:                              # %postload763
                                        #   in Loop: Header=BB12_2 Depth=2
	test	ESI, ESI
	jns	.LBB12_38
# BB#37:                                # %deload796
                                        #   in Loop: Header=BB12_2 Depth=2
	mov	R9, QWORD PTR [RSP + 640] # 8-byte Reload
	mov	R8, QWORD PTR [RSP + 912]
	mov	R8D, DWORD PTR [R8 + 4*R9]
.LBB12_38:                              # %postload797
                                        #   in Loop: Header=BB12_2 Depth=2
	test	EDX, EDX
	jns	.LBB12_40
# BB#39:                                # %deload830
                                        #   in Loop: Header=BB12_2 Depth=2
	mov	R10, QWORD PTR [RSP + 656] # 8-byte Reload
	mov	R9, QWORD PTR [RSP + 912]
	mov	R9D, DWORD PTR [R9 + 4*R10]
.LBB12_40:                              # %postload831
                                        #   in Loop: Header=BB12_2 Depth=2
	test	ECX, ECX
	jns	.LBB12_42
# BB#41:                                # %deload864
                                        #   in Loop: Header=BB12_2 Depth=2
	mov	R11, QWORD PTR [RSP + 680] # 8-byte Reload
	mov	R10, QWORD PTR [RSP + 912]
	mov	R10D, DWORD PTR [R10 + 4*R11]
.LBB12_42:                              # %postload865
                                        #   in Loop: Header=BB12_2 Depth=2
	test	EAX, EAX
	jns	.LBB12_44
# BB#43:                                # %deload765
                                        #   in Loop: Header=BB12_2 Depth=2
	mov	R11, QWORD PTR [RSP + 984]
	mov	DWORD PTR [R11 + 384], EDI
.LBB12_44:                              # %postload766
                                        #   in Loop: Header=BB12_2 Depth=2
	test	ESI, ESI
	jns	.LBB12_46
# BB#45:                                # %deload799
                                        #   in Loop: Header=BB12_2 Depth=2
	mov	RDI, QWORD PTR [RSP + 984]
	mov	DWORD PTR [RDI + 384], R8D
.LBB12_46:                              # %postload800
                                        #   in Loop: Header=BB12_2 Depth=2
	test	EDX, EDX
	jns	.LBB12_48
# BB#47:                                # %deload833
                                        #   in Loop: Header=BB12_2 Depth=2
	mov	RDI, QWORD PTR [RSP + 984]
	mov	DWORD PTR [RDI + 384], R9D
.LBB12_48:                              # %postload834
                                        #   in Loop: Header=BB12_2 Depth=2
	test	ECX, ECX
	jns	.LBB12_50
# BB#49:                                # %deload867
                                        #   in Loop: Header=BB12_2 Depth=2
	mov	RDI, QWORD PTR [RSP + 984]
	mov	DWORD PTR [RDI + 384], R10D
.LBB12_50:                              # %postload868
                                        #   in Loop: Header=BB12_2 Depth=2
	test	EAX, EAX
	jns	.LBB12_52
# BB#51:                                # %deload767
                                        #   in Loop: Header=BB12_2 Depth=2
	movsxd	RDI, DWORD PTR [RSP + 672] # 4-byte Folded Reload
	mov	R8, QWORD PTR [RSP + 904]
	mov	EDI, DWORD PTR [R8 + 4*RDI]
.LBB12_52:                              # %postload768
                                        #   in Loop: Header=BB12_2 Depth=2
	test	ESI, ESI
	jns	.LBB12_54
# BB#53:                                # %deload801
                                        #   in Loop: Header=BB12_2 Depth=2
	mov	R9, QWORD PTR [RSP + 640] # 8-byte Reload
	mov	R8, QWORD PTR [RSP + 904]
	mov	R8D, DWORD PTR [R8 + 4*R9]
.LBB12_54:                              # %postload802
                                        #   in Loop: Header=BB12_2 Depth=2
	test	EDX, EDX
	jns	.LBB12_56
# BB#55:                                # %deload835
                                        #   in Loop: Header=BB12_2 Depth=2
	mov	R10, QWORD PTR [RSP + 656] # 8-byte Reload
	mov	R9, QWORD PTR [RSP + 904]
	mov	R9D, DWORD PTR [R9 + 4*R10]
.LBB12_56:                              # %postload836
                                        #   in Loop: Header=BB12_2 Depth=2
	test	ECX, ECX
	jns	.LBB12_58
# BB#57:                                # %deload869
                                        #   in Loop: Header=BB12_2 Depth=2
	mov	R11, QWORD PTR [RSP + 680] # 8-byte Reload
	mov	R10, QWORD PTR [RSP + 904]
	mov	R10D, DWORD PTR [R10 + 4*R11]
.LBB12_58:                              # %postload870
                                        #   in Loop: Header=BB12_2 Depth=2
	test	EAX, EAX
	jns	.LBB12_60
# BB#59:                                # %deload770
                                        #   in Loop: Header=BB12_2 Depth=2
	mov	R11, QWORD PTR [RSP + 984]
	mov	DWORD PTR [R11 + 512], EDI
.LBB12_60:                              # %postload771
                                        #   in Loop: Header=BB12_2 Depth=2
	test	ESI, ESI
	jns	.LBB12_62
# BB#61:                                # %deload804
                                        #   in Loop: Header=BB12_2 Depth=2
	mov	RDI, QWORD PTR [RSP + 984]
	mov	DWORD PTR [RDI + 512], R8D
.LBB12_62:                              # %postload805
                                        #   in Loop: Header=BB12_2 Depth=2
	test	EDX, EDX
	jns	.LBB12_64
# BB#63:                                # %deload838
                                        #   in Loop: Header=BB12_2 Depth=2
	mov	RDI, QWORD PTR [RSP + 984]
	mov	DWORD PTR [RDI + 512], R9D
.LBB12_64:                              # %postload839
                                        #   in Loop: Header=BB12_2 Depth=2
	test	ECX, ECX
	jns	.LBB12_66
# BB#65:                                # %deload872
                                        #   in Loop: Header=BB12_2 Depth=2
	mov	RDI, QWORD PTR [RSP + 984]
	mov	DWORD PTR [RDI + 512], R10D
.LBB12_66:                              # %postload873
                                        #   in Loop: Header=BB12_2 Depth=2
	test	EAX, EAX
	jns	.LBB12_68
# BB#67:                                # %deload772
                                        #   in Loop: Header=BB12_2 Depth=2
	movsxd	RDI, DWORD PTR [RSP + 672] # 4-byte Folded Reload
	mov	R8, QWORD PTR [RSP + 872]
	mov	EDI, DWORD PTR [R8 + 4*RDI]
.LBB12_68:                              # %postload773
                                        #   in Loop: Header=BB12_2 Depth=2
	test	ESI, ESI
	jns	.LBB12_70
# BB#69:                                # %deload806
                                        #   in Loop: Header=BB12_2 Depth=2
	mov	R9, QWORD PTR [RSP + 640] # 8-byte Reload
	mov	R8, QWORD PTR [RSP + 872]
	mov	R8D, DWORD PTR [R8 + 4*R9]
.LBB12_70:                              # %postload807
                                        #   in Loop: Header=BB12_2 Depth=2
	test	EDX, EDX
	jns	.LBB12_72
# BB#71:                                # %deload840
                                        #   in Loop: Header=BB12_2 Depth=2
	mov	R10, QWORD PTR [RSP + 656] # 8-byte Reload
	mov	R9, QWORD PTR [RSP + 872]
	mov	R9D, DWORD PTR [R9 + 4*R10]
.LBB12_72:                              # %postload841
                                        #   in Loop: Header=BB12_2 Depth=2
	test	ECX, ECX
	jns	.LBB12_74
# BB#73:                                # %deload874
                                        #   in Loop: Header=BB12_2 Depth=2
	mov	R11, QWORD PTR [RSP + 680] # 8-byte Reload
	mov	R10, QWORD PTR [RSP + 872]
	mov	R10D, DWORD PTR [R10 + 4*R11]
.LBB12_74:                              # %postload875
                                        #   in Loop: Header=BB12_2 Depth=2
	test	EAX, EAX
	jns	.LBB12_76
# BB#75:                                # %deload775
                                        #   in Loop: Header=BB12_2 Depth=2
	mov	R11, QWORD PTR [RSP + 984]
	mov	DWORD PTR [R11 + 128], EDI
.LBB12_76:                              # %postload776
                                        #   in Loop: Header=BB12_2 Depth=2
	test	ESI, ESI
	jns	.LBB12_78
# BB#77:                                # %deload809
                                        #   in Loop: Header=BB12_2 Depth=2
	mov	RDI, QWORD PTR [RSP + 984]
	mov	DWORD PTR [RDI + 128], R8D
.LBB12_78:                              # %postload810
                                        #   in Loop: Header=BB12_2 Depth=2
	test	EDX, EDX
	jns	.LBB12_80
# BB#79:                                # %deload843
                                        #   in Loop: Header=BB12_2 Depth=2
	mov	RDI, QWORD PTR [RSP + 984]
	mov	DWORD PTR [RDI + 128], R9D
.LBB12_80:                              # %postload844
                                        #   in Loop: Header=BB12_2 Depth=2
	test	ECX, ECX
	jns	.LBB12_82
# BB#81:                                # %deload877
                                        #   in Loop: Header=BB12_2 Depth=2
	mov	RDI, QWORD PTR [RSP + 984]
	mov	DWORD PTR [RDI + 128], R10D
.LBB12_82:                              # %postload878
                                        #   in Loop: Header=BB12_2 Depth=2
	test	EAX, EAX
	js	.LBB12_84
# BB#83:                                # %postload878.postload778_crit_edge
                                        #   in Loop: Header=BB12_2 Depth=2
	xor	EDI, EDI
	jmp	.LBB12_85
.LBB12_84:                              # %deload777
                                        #   in Loop: Header=BB12_2 Depth=2
	movsxd	RDI, DWORD PTR [RSP + 672] # 4-byte Folded Reload
	mov	R8, QWORD PTR [RSP + 880]
	movsxd	RDI, DWORD PTR [R8 + 4*RDI]
.LBB12_85:                              # %postload778
                                        #   in Loop: Header=BB12_2 Depth=2
	test	ESI, ESI
	js	.LBB12_87
# BB#86:                                # %postload778.postload812_crit_edge
                                        #   in Loop: Header=BB12_2 Depth=2
	xor	R8D, R8D
	jmp	.LBB12_88
.LBB12_87:                              # %deload811
                                        #   in Loop: Header=BB12_2 Depth=2
	mov	R9, QWORD PTR [RSP + 640] # 8-byte Reload
	mov	R8, QWORD PTR [RSP + 880]
	movsxd	R8, DWORD PTR [R8 + 4*R9]
.LBB12_88:                              # %postload812
                                        #   in Loop: Header=BB12_2 Depth=2
	test	EDX, EDX
	js	.LBB12_90
# BB#89:                                # %postload812.postload846_crit_edge
                                        #   in Loop: Header=BB12_2 Depth=2
	xor	R9D, R9D
	jmp	.LBB12_91
.LBB12_90:                              # %deload845
                                        #   in Loop: Header=BB12_2 Depth=2
	mov	R10, QWORD PTR [RSP + 656] # 8-byte Reload
	mov	R9, QWORD PTR [RSP + 880]
	movsxd	R9, DWORD PTR [R9 + 4*R10]
.LBB12_91:                              # %postload846
                                        #   in Loop: Header=BB12_2 Depth=2
	test	ECX, ECX
	js	.LBB12_93
# BB#92:                                # %postload846.postload880_crit_edge
                                        #   in Loop: Header=BB12_2 Depth=2
	xor	R10D, R10D
	jmp	.LBB12_94
.LBB12_93:                              # %deload879
                                        #   in Loop: Header=BB12_2 Depth=2
	mov	R11, QWORD PTR [RSP + 680] # 8-byte Reload
	mov	R10, QWORD PTR [RSP + 880]
	movsxd	R10, DWORD PTR [R10 + 4*R11]
.LBB12_94:                              # %postload880
                                        #   in Loop: Header=BB12_2 Depth=2
	test	EAX, EAX
	jns	.LBB12_96
# BB#95:                                # %deload780
                                        #   in Loop: Header=BB12_2 Depth=2
	mov	R11, QWORD PTR [RSP + 840]
	lea	RDI, QWORD PTR [R11 + 4*RDI]
	mov	R11, QWORD PTR [RSP + 984]
	mov	QWORD PTR [R11], RDI
.LBB12_96:                              # %postload781
                                        #   in Loop: Header=BB12_2 Depth=2
	test	ESI, ESI
	jns	.LBB12_98
# BB#97:                                # %deload814
                                        #   in Loop: Header=BB12_2 Depth=2
	mov	RDI, QWORD PTR [RSP + 840]
	lea	RDI, QWORD PTR [RDI + 4*R8]
	mov	R8, QWORD PTR [RSP + 984]
	mov	QWORD PTR [R8], RDI
.LBB12_98:                              # %postload815
                                        #   in Loop: Header=BB12_2 Depth=2
	test	EDX, EDX
	jns	.LBB12_100
# BB#99:                                # %deload848
                                        #   in Loop: Header=BB12_2 Depth=2
	mov	RDI, QWORD PTR [RSP + 840]
	lea	RDI, QWORD PTR [RDI + 4*R9]
	mov	R8, QWORD PTR [RSP + 984]
	mov	QWORD PTR [R8], RDI
.LBB12_100:                             # %postload849
                                        #   in Loop: Header=BB12_2 Depth=2
	test	ECX, ECX
	jns	.LBB12_102
# BB#101:                               # %deload882
                                        #   in Loop: Header=BB12_2 Depth=2
	mov	RDI, QWORD PTR [RSP + 840]
	lea	RDI, QWORD PTR [RDI + 4*R10]
	mov	R8, QWORD PTR [RSP + 984]
	mov	QWORD PTR [R8], RDI
.LBB12_102:                             # %postload883
                                        #   in Loop: Header=BB12_2 Depth=2
	test	EAX, EAX
	jns	.LBB12_104
# BB#103:                               # %deload782
                                        #   in Loop: Header=BB12_2 Depth=2
	mov	RDI, QWORD PTR [RSP + 984]
	mov	RAX, QWORD PTR [RSP + 952]
	mov	QWORD PTR [RDI + 256], RAX
.LBB12_104:                             # %postload783
                                        #   in Loop: Header=BB12_2 Depth=2
	test	ESI, ESI
	jns	.LBB12_106
# BB#105:                               # %deload816
                                        #   in Loop: Header=BB12_2 Depth=2
	mov	RSI, QWORD PTR [RSP + 984]
	mov	RAX, QWORD PTR [RSP + 952]
	mov	QWORD PTR [RSI + 256], RAX
.LBB12_106:                             # %postload817
                                        #   in Loop: Header=BB12_2 Depth=2
	test	EDX, EDX
	jns	.LBB12_108
# BB#107:                               # %deload850
                                        #   in Loop: Header=BB12_2 Depth=2
	mov	RDX, QWORD PTR [RSP + 984]
	mov	RAX, QWORD PTR [RSP + 952]
	mov	QWORD PTR [RDX + 256], RAX
.LBB12_108:                             # %postload851
                                        #   in Loop: Header=BB12_2 Depth=2
	test	ECX, ECX
	jns	.LBB12_110
# BB#109:                               # %deload884
                                        #   in Loop: Header=BB12_2 Depth=2
	mov	RCX, QWORD PTR [RSP + 984]
	mov	RAX, QWORD PTR [RSP + 952]
	mov	QWORD PTR [RCX + 256], RAX
.LBB12_110:                             # %postload885
                                        #   in Loop: Header=BB12_2 Depth=2
	mov	RAX, QWORD PTR [RSP + 40] # 8-byte Reload
	cmp	RAX, QWORD PTR [RSP + 48] # 8-byte Folded Reload
	jb	.LBB12_151
.LBB12_111:                             # %postload885.postload947.loopexit_crit_edge
                                        #   in Loop: Header=BB12_1 Depth=1
	movaps	XMM0, XMMWORD PTR [RSP] # 16-byte Reload
	movaps	XMMWORD PTR [RSP + 144], XMM0 # 16-byte Spill
	.align	16, 0x90
.LBB12_112:                             # %postload947.outer
                                        #   Parent Loop BB12_1 Depth=1
                                        # =>  This Loop Header: Depth=2
                                        #       Child Loop BB12_156 Depth 3
                                        #         Child Loop BB12_158 Depth 4
                                        #       Child Loop BB12_113 Depth 3
                                        #         Child Loop BB12_115 Depth 4
	mov	RAX, QWORD PTR [RSP + 104] # 8-byte Reload
	shl	RAX, 5
	add	RAX, QWORD PTR [RSP + 1016]
	mov	QWORD PTR [RSP + 680], RAX # 8-byte Spill
	.align	16, 0x90
.LBB12_113:                             # %postload947
                                        #   Parent Loop BB12_1 Depth=1
                                        #     Parent Loop BB12_112 Depth=2
                                        # =>    This Loop Header: Depth=3
                                        #         Child Loop BB12_115 Depth 4
	mov	RAX, QWORD PTR [RSP + 680] # 8-byte Reload
	movq	XMM0, QWORD PTR [RAX]
	movlhps	XMM0, XMM0              # xmm0 = xmm0[0,0]
	movaps	XMM1, XMM0
	paddq	XMM1, XMMWORD PTR [RSP + 64] # 16-byte Folded Reload
	pextrq	RAX, XMM1, 1
	movq	RCX, XMM1
	movd	XMM2, ECX
	pinsrd	XMM2, EAX, 1
	paddq	XMM0, XMMWORD PTR [RIP + .LCPI12_0]
	movq	RAX, XMM0
	pinsrd	XMM2, EAX, 2
	pextrq	RAX, XMM0, 1
	pinsrd	XMM2, EAX, 3
	mov	RAX, QWORD PTR [RSP + 984]
	movd	XMM3, DWORD PTR [RAX + 128]
	pshufd	XMM3, XMM3, 0           # xmm3 = xmm3[0,0,0,0]
	pslld	XMM3, 1
	pcmpgtd	XMM3, XMM2
	pand	XMM3, XMMWORD PTR [RSP + 80] # 16-byte Folded Reload
	ptest 	XMM3, XMM3
	je	.LBB12_148
# BB#114:                               # %bb.nph7.preheader
                                        #   in Loop: Header=BB12_113 Depth=3
	movdqa	XMM4, XMM3
	pxor	XMM4, XMMWORD PTR [.LCPI12_8]
	.align	16, 0x90
.LBB12_115:                             # %bb.nph7
                                        #   Parent Loop BB12_1 Depth=1
                                        #     Parent Loop BB12_112 Depth=2
                                        #       Parent Loop BB12_113 Depth=3
                                        # =>      This Inner Loop Header: Depth=4
	pextrd	EAX, XMM2, 1
	mov	ECX, EAX
	shr	ECX, 31
	add	ECX, EAX
	mov	EDX, ECX
	and	EDX, -2
	mov	ESI, EAX
	sub	ESI, EDX
	movd	EDX, XMM2
	mov	EDI, EDX
	shr	EDI, 31
	add	EDI, EDX
	mov	R8D, EDI
	and	R8D, -2
	mov	R9D, EDX
	sub	R9D, R8D
	movd	XMM5, R9D
	pinsrd	XMM5, ESI, 1
	pextrd	ESI, XMM2, 2
	mov	R8D, ESI
	shr	R8D, 31
	add	R8D, ESI
	mov	R9D, R8D
	and	R9D, -2
	mov	R10D, ESI
	sub	R10D, R9D
	pinsrd	XMM5, R10D, 2
	pextrd	R9D, XMM2, 3
	mov	R10D, R9D
	shr	R10D, 31
	add	R10D, R9D
	mov	R11D, R10D
	and	R11D, -2
	mov	EBX, R9D
	sub	EBX, R11D
	pinsrd	XMM5, EBX, 3
	sar	ECX
	sar	EDI
	movd	XMM2, EDI
	pinsrd	XMM2, ECX, 1
	sar	R8D
	pinsrd	XMM2, R8D, 2
	sar	R10D
	pinsrd	XMM2, R10D, 3
	pextrd	ECX, XMM2, 3
	movsxd	RCX, ECX
	pextrd	EDI, XMM2, 2
	movsxd	RDI, EDI
	pextrd	R8D, XMM2, 1
	movsxd	R8, R8D
	movd	R10D, XMM3
	test	R10D, R10D
	mov	R11, QWORD PTR [RSP + 984]
	mov	R11, QWORD PTR [R11]
	pextrd	EBX, XMM3, 3
	pextrd	R14D, XMM3, 2
	pextrd	R15D, XMM3, 1
	jns	.LBB12_117
# BB#116:                               # %deload733
                                        #   in Loop: Header=BB12_115 Depth=4
	mov	R12, QWORD PTR [RSP + 984]
	mov	R12, QWORD PTR [R12]
	movd	R13D, XMM2
	movsxd	R13, R13D
	mov	R12D, DWORD PTR [R12 + 4*R13]
.LBB12_117:                             # %postload734
                                        #   in Loop: Header=BB12_115 Depth=4
	test	R15D, R15D
	jns	.LBB12_119
# BB#118:                               # %deload889
                                        #   in Loop: Header=BB12_115 Depth=4
	mov	R8D, DWORD PTR [R11 + 4*R8]
.LBB12_119:                             # %postload890
                                        #   in Loop: Header=BB12_115 Depth=4
	test	R14D, R14D
	jns	.LBB12_121
# BB#120:                               # %deload909
                                        #   in Loop: Header=BB12_115 Depth=4
	mov	EDI, DWORD PTR [R11 + 4*RDI]
.LBB12_121:                             # %postload910
                                        #   in Loop: Header=BB12_115 Depth=4
	test	EBX, EBX
	jns	.LBB12_123
# BB#122:                               # %deload929
                                        #   in Loop: Header=BB12_115 Depth=4
	mov	ECX, DWORD PTR [R11 + 4*RCX]
.LBB12_123:                             # %postload930
                                        #   in Loop: Header=BB12_115 Depth=4
	movd	XMM2, R12D
	pinsrd	XMM2, R8D, 1
	pinsrd	XMM2, EDI, 2
	pinsrd	XMM2, ECX, 3
	pslld	XMM2, 1
	paddd	XMM2, XMM5
	pextrd	ECX, XMM2, 3
	movsxd	RCX, ECX
	pextrd	EDI, XMM2, 2
	movsxd	RDI, EDI
	pextrd	R8D, XMM2, 1
	movsxd	R8, R8D
	test	R10D, R10D
	jns	.LBB12_125
# BB#124:                               # %deload736
                                        #   in Loop: Header=BB12_115 Depth=4
	movd	R11D, XMM2
	movsxd	R11, R11D
	mov	R12, QWORD PTR [RSP + 832]
	movss	XMM2, DWORD PTR [R12 + 4*R11]
.LBB12_125:                             # %postload737
                                        #   in Loop: Header=BB12_115 Depth=4
	test	R15D, R15D
	jns	.LBB12_127
# BB#126:                               # %deload892
                                        #   in Loop: Header=BB12_115 Depth=4
	mov	R11, QWORD PTR [RSP + 832]
	movss	XMM5, DWORD PTR [R11 + 4*R8]
.LBB12_127:                             # %postload893
                                        #   in Loop: Header=BB12_115 Depth=4
	test	R14D, R14D
	jns	.LBB12_129
# BB#128:                               # %deload912
                                        #   in Loop: Header=BB12_115 Depth=4
	mov	R8, QWORD PTR [RSP + 832]
	movss	XMM6, DWORD PTR [R8 + 4*RDI]
.LBB12_129:                             # %postload913
                                        #   in Loop: Header=BB12_115 Depth=4
	test	EBX, EBX
	jns	.LBB12_131
# BB#130:                               # %deload932
                                        #   in Loop: Header=BB12_115 Depth=4
	mov	RDI, QWORD PTR [RSP + 832]
	movss	XMM7, DWORD PTR [RDI + 4*RCX]
.LBB12_131:                             # %postload933
                                        #   in Loop: Header=BB12_115 Depth=4
	test	R10D, R10D
	mov	RCX, QWORD PTR [RSP + 984]
	mov	RCX, QWORD PTR [RCX + 256]
	jns	.LBB12_133
# BB#132:                               # %deload742
                                        #   in Loop: Header=BB12_115 Depth=4
	mov	RDI, QWORD PTR [RSP + 984]
	mov	RDI, QWORD PTR [RDI + 256]
	movsxd	RDX, EDX
	movss	DWORD PTR [RDI + 4*RDX], XMM2
.LBB12_133:                             # %postload743
                                        #   in Loop: Header=BB12_115 Depth=4
	test	R15D, R15D
	jns	.LBB12_135
# BB#134:                               # %deload898
                                        #   in Loop: Header=BB12_115 Depth=4
	movsxd	RAX, EAX
	movss	DWORD PTR [RCX + 4*RAX], XMM5
.LBB12_135:                             # %postload899
                                        #   in Loop: Header=BB12_115 Depth=4
	test	R14D, R14D
	jns	.LBB12_137
# BB#136:                               # %deload918
                                        #   in Loop: Header=BB12_115 Depth=4
	movsxd	RAX, ESI
	movss	DWORD PTR [RCX + 4*RAX], XMM6
.LBB12_137:                             # %postload919
                                        #   in Loop: Header=BB12_115 Depth=4
	test	EBX, EBX
	jns	.LBB12_139
# BB#138:                               # %deload938
                                        #   in Loop: Header=BB12_115 Depth=4
	movsxd	RAX, R9D
	movss	DWORD PTR [RCX + 4*RAX], XMM7
.LBB12_139:                             # %postload939
                                        #   in Loop: Header=BB12_115 Depth=4
	test	R10D, R10D
	jns	.LBB12_141
# BB#140:                               # %deload744
                                        #   in Loop: Header=BB12_115 Depth=4
	mov	RAX, QWORD PTR [RSP + 992]
	mov	RAX, QWORD PTR [RAX + 56]
.LBB12_141:                             # %postload745
                                        #   in Loop: Header=BB12_115 Depth=4
	test	R15D, R15D
	jns	.LBB12_143
# BB#142:                               # %deload900
                                        #   in Loop: Header=BB12_115 Depth=4
	mov	RCX, QWORD PTR [RSP + 992]
	mov	RCX, QWORD PTR [RCX + 56]
.LBB12_143:                             # %postload901
                                        #   in Loop: Header=BB12_115 Depth=4
	test	R14D, R14D
	jns	.LBB12_145
# BB#144:                               # %deload920
                                        #   in Loop: Header=BB12_115 Depth=4
	mov	RDX, QWORD PTR [RSP + 992]
	mov	RDX, QWORD PTR [RDX + 56]
.LBB12_145:                             # %postload921
                                        #   in Loop: Header=BB12_115 Depth=4
	test	EBX, EBX
	jns	.LBB12_147
# BB#146:                               # %deload940
                                        #   in Loop: Header=BB12_115 Depth=4
	mov	RSI, QWORD PTR [RSP + 992]
	mov	RSI, QWORD PTR [RSI + 56]
.LBB12_147:                             # %postload941
                                        #   in Loop: Header=BB12_115 Depth=4
	movdqa	XMM5, XMMWORD PTR [RIP + .LCPI12_1]
	pand	XMM1, XMM5
	movq	XMM2, RCX
	movq	XMM6, RAX
	punpcklqdq	XMM6, XMM2      # xmm6 = xmm6[0],xmm2[0]
	paddq	XMM1, XMM6
	pextrq	RAX, XMM1, 1
	movq	RCX, XMM1
	movd	XMM2, ECX
	pinsrd	XMM2, EAX, 1
	pand	XMM0, XMM5
	movq	XMM5, RSI
	movq	XMM6, RDX
	punpcklqdq	XMM6, XMM5      # xmm6 = xmm6[0],xmm5[0]
	paddq	XMM0, XMM6
	movq	RAX, XMM0
	pinsrd	XMM2, EAX, 2
	pextrq	RAX, XMM0, 1
	pinsrd	XMM2, EAX, 3
	mov	RAX, QWORD PTR [RSP + 984]
	movd	XMM5, DWORD PTR [RAX + 128]
	pshufd	XMM5, XMM5, 0           # xmm5 = xmm5[0,0,0,0]
	pslld	XMM5, 1
	pcmpgtd	XMM5, XMM2
	movdqa	XMM6, XMM3
	pand	XMM6, XMM5
	pandn	XMM5, XMM3
	por	XMM4, XMM5
	por	XMM5, XMM4
	pcmpeqd	XMM3, XMM3
	ptest 	XMM5, XMM3
	movdqa	XMM3, XMM6
	jae	.LBB12_115
.LBB12_148:                             # %._crit_edge
                                        #   in Loop: Header=BB12_113 Depth=3
	mov	RAX, QWORD PTR [RSP + 40] # 8-byte Reload
	cmp	RAX, QWORD PTR [RSP + 48] # 8-byte Folded Reload
	jae	.LBB12_155
# BB#149:                               # %deload951
                                        #   in Loop: Header=BB12_113 Depth=3
	mov	RAX, QWORD PTR [RSP + 104] # 8-byte Reload
	cmp	RAX, QWORD PTR [RSP + 1032]
	jb	.LBB12_150
# BB#154:                               # %elseBB2520
                                        #   in Loop: Header=BB12_112 Depth=2
	mfence
	mov	QWORD PTR [RSP + 104], 0 # 8-byte Folded Spill
	mov	DWORD PTR [RSP + 100], 5 # 4-byte Folded Spill
.LBB12_155:                             # %postload952.preheader
                                        #   in Loop: Header=BB12_112 Depth=2
	mov	RAX, QWORD PTR [RSP + 104] # 8-byte Reload
	shl	RAX, 5
	add	RAX, QWORD PTR [RSP + 1016]
	mov	QWORD PTR [RSP + 56], RAX # 8-byte Spill
	jmp	.LBB12_156
	.align	16, 0x90
.LBB12_448:                             # %thenBB
                                        #   in Loop: Header=BB12_156 Depth=3
	add	QWORD PTR [RSP + 56], 32 # 8-byte Folded Spill
	inc	QWORD PTR [RSP + 104]   # 8-byte Folded Spill
	cmp	DWORD PTR [RSP + 100], 4 # 4-byte Folded Reload
	je	.LBB12_112
# BB#449:                               # %thenBB
                                        #   in Loop: Header=BB12_156 Depth=3
	cmp	DWORD PTR [RSP + 100], 6 # 4-byte Folded Reload
	je	.LBB12_1
.LBB12_156:                             # %postload952
                                        #   Parent Loop BB12_1 Depth=1
                                        #     Parent Loop BB12_112 Depth=2
                                        # =>    This Loop Header: Depth=3
                                        #         Child Loop BB12_158 Depth 4
	mov	RAX, QWORD PTR [RSP + 56] # 8-byte Reload
	movq	XMM0, QWORD PTR [RAX]
	movlhps	XMM0, XMM0              # xmm0 = xmm0[0,0]
	movaps	XMM1, XMM0
	paddq	XMM1, XMMWORD PTR [RSP + 64] # 16-byte Folded Reload
	movdqa	XMMWORD PTR [RSP + 256], XMM1 # 16-byte Spill
	pextrq	RAX, XMM1, 1
	movq	RCX, XMM1
	movd	XMM1, ECX
	pinsrd	XMM1, EAX, 1
	paddq	XMM0, XMMWORD PTR [RIP + .LCPI12_0]
	movdqa	XMMWORD PTR [RSP + 272], XMM0 # 16-byte Spill
	movq	RAX, XMM0
	pinsrd	XMM1, EAX, 2
	pextrq	RAX, XMM0, 1
	pinsrd	XMM1, EAX, 3
	mov	RAX, QWORD PTR [RSP + 984]
	movd	XMM0, DWORD PTR [RAX + 384]
	pshufd	XMM0, XMM0, 0           # xmm0 = xmm0[0,0,0,0]
	pcmpgtd	XMM0, XMM1
	pand	XMM0, XMMWORD PTR [RSP + 80] # 16-byte Folded Reload
	movdqa	XMMWORD PTR [RSP + 544], XMM0 # 16-byte Spill
	ptest 	XMM0, XMM0
	je	.LBB12_447
# BB#157:                               # %.preheader
                                        #   in Loop: Header=BB12_156 Depth=3
	pxor	XMM0, XMMWORD PTR [.LCPI12_8]
	movdqa	XMMWORD PTR [RSP + 240], XMM0 # 16-byte Spill
	.align	16, 0x90
.LBB12_158:                             #   Parent Loop BB12_1 Depth=1
                                        #     Parent Loop BB12_112 Depth=2
                                        #       Parent Loop BB12_156 Depth=3
                                        # =>      This Inner Loop Header: Depth=4
	movapd	XMM0, XMMWORD PTR [RSP + 544] # 16-byte Reload
	movd	EBX, XMM0
	test	EBX, EBX
	pextrd	R14D, XMM0, 3
	pextrd	R15D, XMM0, 2
	pextrd	R12D, XMM0, 1
	jns	.LBB12_160
# BB#159:                               # %deload974
                                        #   in Loop: Header=BB12_158 Depth=4
	mov	RAX, QWORD PTR [RSP + 984]
	mov	ECX, DWORD PTR [RAX + 512]
	mov	RAX, QWORD PTR [RAX + 256]
	mov	QWORD PTR [RSP + 680], RAX # 8-byte Spill
.LBB12_160:                             # %postload975
                                        #   in Loop: Header=BB12_158 Depth=4
	test	R12D, R12D
	jns	.LBB12_162
# BB#161:                               # %deload1082
                                        #   in Loop: Header=BB12_158 Depth=4
	mov	RAX, QWORD PTR [RSP + 984]
	mov	EDX, DWORD PTR [RAX + 512]
	mov	RAX, QWORD PTR [RAX + 256]
	mov	QWORD PTR [RSP + 672], RAX # 8-byte Spill
.LBB12_162:                             # %postload1083
                                        #   in Loop: Header=BB12_158 Depth=4
	test	R15D, R15D
	jns	.LBB12_164
# BB#163:                               # %deload1190
                                        #   in Loop: Header=BB12_158 Depth=4
	mov	RAX, QWORD PTR [RSP + 984]
	mov	ESI, DWORD PTR [RAX + 512]
	mov	R13, QWORD PTR [RAX + 256]
.LBB12_164:                             # %postload1191
                                        #   in Loop: Header=BB12_158 Depth=4
	test	R14D, R14D
	jns	.LBB12_166
# BB#165:                               # %deload1298
                                        #   in Loop: Header=BB12_158 Depth=4
	mov	RAX, QWORD PTR [RSP + 984]
	mov	EDI, DWORD PTR [RAX + 512]
	mov	RBP, QWORD PTR [RAX + 256]
.LBB12_166:                             # %postload1299
                                        #   in Loop: Header=BB12_158 Depth=4
	movd	XMM0, ECX
	pinsrd	XMM0, EDX, 1
	pinsrd	XMM0, ESI, 2
	pinsrd	XMM0, EDI, 3
	movdqa	XMM2, XMM0
	paddd	XMM2, XMM1
	pextrd	EAX, XMM2, 3
	movsxd	RAX, EAX
	mov	QWORD PTR [RSP + 632], RAX # 8-byte Spill
	pextrd	EAX, XMM2, 2
	movsxd	RAX, EAX
	mov	QWORD PTR [RSP + 624], RAX # 8-byte Spill
	pextrd	EAX, XMM2, 1
	movsxd	RAX, EAX
	mov	QWORD PTR [RSP + 616], RAX # 8-byte Spill
	test	EBX, EBX
	jns	.LBB12_168
# BB#167:                               # %deload978
                                        #   in Loop: Header=BB12_158 Depth=4
	movd	EAX, XMM2
	movsxd	RAX, EAX
	mov	RCX, QWORD PTR [RSP + 848]
	movzx	EAX, WORD PTR [RCX + 2*RAX]
.LBB12_168:                             # %postload979
                                        #   in Loop: Header=BB12_158 Depth=4
	test	R12D, R12D
	jns	.LBB12_170
# BB#169:                               # %deload1086
                                        #   in Loop: Header=BB12_158 Depth=4
	mov	RDX, QWORD PTR [RSP + 616] # 8-byte Reload
	mov	RCX, QWORD PTR [RSP + 848]
	movzx	ECX, WORD PTR [RCX + 2*RDX]
.LBB12_170:                             # %postload1087
                                        #   in Loop: Header=BB12_158 Depth=4
	test	R15D, R15D
	jns	.LBB12_172
# BB#171:                               # %deload1194
                                        #   in Loop: Header=BB12_158 Depth=4
	mov	RSI, QWORD PTR [RSP + 624] # 8-byte Reload
	mov	RDX, QWORD PTR [RSP + 848]
	movzx	EDX, WORD PTR [RDX + 2*RSI]
.LBB12_172:                             # %postload1195
                                        #   in Loop: Header=BB12_158 Depth=4
	test	R14D, R14D
	jns	.LBB12_174
# BB#173:                               # %deload1302
                                        #   in Loop: Header=BB12_158 Depth=4
	mov	RDI, QWORD PTR [RSP + 632] # 8-byte Reload
	mov	RSI, QWORD PTR [RSP + 848]
	movzx	ESI, WORD PTR [RSI + 2*RDI]
.LBB12_174:                             # %postload1303
                                        #   in Loop: Header=BB12_158 Depth=4
	movsx	ECX, CX
	movsx	EAX, AX
	movd	XMM3, EAX
	pinsrd	XMM3, ECX, 1
	movsx	EAX, DX
	pinsrd	XMM3, EAX, 2
	movsx	EAX, SI
	pinsrd	XMM3, EAX, 3
	pslld	XMM3, 1
	movdqa	XMMWORD PTR [RSP + 752], XMM3
	movsxd	RAX, DWORD PTR [RSP + 764]
	mov	QWORD PTR [RSP + 536], RAX # 8-byte Spill
	movq	XMM3, RAX
	movsxd	RAX, DWORD PTR [RSP + 760]
	mov	QWORD PTR [RSP + 528], RAX # 8-byte Spill
	movq	XMM4, RAX
	punpcklqdq	XMM4, XMM3      # xmm4 = xmm4[0],xmm3[0]
	movdqa	XMMWORD PTR [RSP + 592], XMM4 # 16-byte Spill
	movsxd	RAX, DWORD PTR [RSP + 756]
	mov	QWORD PTR [RSP + 520], RAX # 8-byte Spill
	movq	XMM3, RAX
	movsxd	RAX, DWORD PTR [RSP + 752]
	mov	QWORD PTR [RSP + 512], RAX # 8-byte Spill
	movq	XMM4, RAX
	punpcklqdq	XMM4, XMM3      # xmm4 = xmm4[0],xmm3[0]
	movdqa	XMMWORD PTR [RSP + 304], XMM4 # 16-byte Spill
	movdqa	XMM3, XMM1
	paddd	XMM3, XMMWORD PTR [RSP + 144] # 16-byte Folded Reload
	paddd	XMM3, XMM0
	pextrd	EAX, XMM3, 3
	movsxd	RAX, EAX
	pextrd	ECX, XMM3, 2
	movsxd	RCX, ECX
	pextrd	EDX, XMM3, 1
	movsxd	RDX, EDX
	test	EBX, EBX
	jns	.LBB12_176
# BB#175:                               # %deload981
                                        #   in Loop: Header=BB12_158 Depth=4
	movd	ESI, XMM3
	movsxd	RSI, ESI
	mov	RDI, QWORD PTR [RSP + 848]
	movzx	ESI, WORD PTR [RDI + 2*RSI]
.LBB12_176:                             # %postload982
                                        #   in Loop: Header=BB12_158 Depth=4
	test	R12D, R12D
	jns	.LBB12_178
# BB#177:                               # %deload1089
                                        #   in Loop: Header=BB12_158 Depth=4
	mov	RDI, QWORD PTR [RSP + 848]
	movzx	EDX, WORD PTR [RDI + 2*RDX]
.LBB12_178:                             # %postload1090
                                        #   in Loop: Header=BB12_158 Depth=4
	test	R15D, R15D
	jns	.LBB12_180
# BB#179:                               # %deload1197
                                        #   in Loop: Header=BB12_158 Depth=4
	mov	RDI, QWORD PTR [RSP + 848]
	movzx	ECX, WORD PTR [RDI + 2*RCX]
.LBB12_180:                             # %postload1198
                                        #   in Loop: Header=BB12_158 Depth=4
	test	R14D, R14D
	jns	.LBB12_182
# BB#181:                               # %deload1305
                                        #   in Loop: Header=BB12_158 Depth=4
	mov	RDI, QWORD PTR [RSP + 848]
	movzx	EAX, WORD PTR [RDI + 2*RAX]
.LBB12_182:                             # %postload1306
                                        #   in Loop: Header=BB12_158 Depth=4
	movsx	EDX, DX
	movsx	ESI, SI
	movd	XMM3, ESI
	pinsrd	XMM3, EDX, 1
	movsx	ECX, CX
	pinsrd	XMM3, ECX, 2
	movsx	EAX, AX
	pinsrd	XMM3, EAX, 3
	pslld	XMM3, 1
	movdqa	XMMWORD PTR [RSP + 736], XMM3
	movsxd	RAX, DWORD PTR [RSP + 748]
	mov	QWORD PTR [RSP + 496], RAX # 8-byte Spill
	movq	XMM3, RAX
	movsxd	RAX, DWORD PTR [RSP + 744]
	mov	QWORD PTR [RSP + 488], RAX # 8-byte Spill
	movq	XMM4, RAX
	punpcklqdq	XMM4, XMM3      # xmm4 = xmm4[0],xmm3[0]
	movdqa	XMMWORD PTR [RSP + 336], XMM4 # 16-byte Spill
	movsxd	RAX, DWORD PTR [RSP + 740]
	mov	QWORD PTR [RSP + 480], RAX # 8-byte Spill
	movq	XMM3, RAX
	movsxd	RAX, DWORD PTR [RSP + 736]
	mov	QWORD PTR [RSP + 472], RAX # 8-byte Spill
	movq	XMM4, RAX
	punpcklqdq	XMM4, XMM3      # xmm4 = xmm4[0],xmm3[0]
	movdqa	XMMWORD PTR [RSP + 368], XMM4 # 16-byte Spill
	movdqa	XMM3, XMM1
	paddd	XMM3, XMMWORD PTR [RSP + 112] # 16-byte Folded Reload
	paddd	XMM3, XMM0
	pextrd	EAX, XMM3, 3
	movsxd	RAX, EAX
	pextrd	ECX, XMM3, 2
	movsxd	RCX, ECX
	pextrd	EDX, XMM3, 1
	movsxd	RDX, EDX
	test	EBX, EBX
	jns	.LBB12_184
# BB#183:                               # %deload984
                                        #   in Loop: Header=BB12_158 Depth=4
	movd	ESI, XMM3
	movsxd	RSI, ESI
	mov	RDI, QWORD PTR [RSP + 848]
	movzx	ESI, WORD PTR [RDI + 2*RSI]
.LBB12_184:                             # %postload985
                                        #   in Loop: Header=BB12_158 Depth=4
	test	R12D, R12D
	jns	.LBB12_186
# BB#185:                               # %deload1092
                                        #   in Loop: Header=BB12_158 Depth=4
	mov	RDI, QWORD PTR [RSP + 848]
	movzx	EDX, WORD PTR [RDI + 2*RDX]
.LBB12_186:                             # %postload1093
                                        #   in Loop: Header=BB12_158 Depth=4
	test	R15D, R15D
	jns	.LBB12_188
# BB#187:                               # %deload1200
                                        #   in Loop: Header=BB12_158 Depth=4
	mov	RDI, QWORD PTR [RSP + 848]
	movzx	ECX, WORD PTR [RDI + 2*RCX]
.LBB12_188:                             # %postload1201
                                        #   in Loop: Header=BB12_158 Depth=4
	test	R14D, R14D
	jns	.LBB12_190
# BB#189:                               # %deload1308
                                        #   in Loop: Header=BB12_158 Depth=4
	mov	RDI, QWORD PTR [RSP + 848]
	movzx	EAX, WORD PTR [RDI + 2*RAX]
.LBB12_190:                             # %postload1309
                                        #   in Loop: Header=BB12_158 Depth=4
	movsx	EDX, DX
	movsx	ESI, SI
	movd	XMM3, ESI
	pinsrd	XMM3, EDX, 1
	movsx	ECX, CX
	pinsrd	XMM3, ECX, 2
	movsx	EAX, AX
	pinsrd	XMM3, EAX, 3
	pslld	XMM3, 1
	movdqa	XMMWORD PTR [RSP + 720], XMM3
	movsxd	RAX, DWORD PTR [RSP + 732]
	mov	QWORD PTR [RSP + 464], RAX # 8-byte Spill
	movq	XMM3, RAX
	movsxd	RAX, DWORD PTR [RSP + 728]
	mov	QWORD PTR [RSP + 456], RAX # 8-byte Spill
	movq	XMM4, RAX
	punpcklqdq	XMM4, XMM3      # xmm4 = xmm4[0],xmm3[0]
	movdqa	XMMWORD PTR [RSP + 224], XMM4 # 16-byte Spill
	movsxd	RAX, DWORD PTR [RSP + 724]
	mov	QWORD PTR [RSP + 448], RAX # 8-byte Spill
	movq	XMM3, RAX
	movsxd	RAX, DWORD PTR [RSP + 720]
	mov	QWORD PTR [RSP + 440], RAX # 8-byte Spill
	movq	XMM4, RAX
	punpcklqdq	XMM4, XMM3      # xmm4 = xmm4[0],xmm3[0]
	movdqa	XMMWORD PTR [RSP + 208], XMM4 # 16-byte Spill
	paddd	XMM1, XMMWORD PTR [RSP + 128] # 16-byte Folded Reload
	paddd	XMM1, XMM0
	pextrd	EAX, XMM1, 3
	movsxd	RAX, EAX
	pextrd	ECX, XMM1, 2
	movsxd	RCX, ECX
	pextrd	EDX, XMM1, 1
	movsxd	RDX, EDX
	test	EBX, EBX
	jns	.LBB12_192
# BB#191:                               # %deload987
                                        #   in Loop: Header=BB12_158 Depth=4
	movd	ESI, XMM1
	movsxd	RSI, ESI
	mov	RDI, QWORD PTR [RSP + 848]
	movzx	ESI, WORD PTR [RDI + 2*RSI]
.LBB12_192:                             # %postload988
                                        #   in Loop: Header=BB12_158 Depth=4
	test	R12D, R12D
	jns	.LBB12_194
# BB#193:                               # %deload1095
                                        #   in Loop: Header=BB12_158 Depth=4
	mov	RDI, QWORD PTR [RSP + 848]
	movzx	EDX, WORD PTR [RDI + 2*RDX]
.LBB12_194:                             # %postload1096
                                        #   in Loop: Header=BB12_158 Depth=4
	test	R15D, R15D
	jns	.LBB12_196
# BB#195:                               # %deload1203
                                        #   in Loop: Header=BB12_158 Depth=4
	mov	RDI, QWORD PTR [RSP + 848]
	movzx	ECX, WORD PTR [RDI + 2*RCX]
.LBB12_196:                             # %postload1204
                                        #   in Loop: Header=BB12_158 Depth=4
	test	R14D, R14D
	jns	.LBB12_198
# BB#197:                               # %deload1311
                                        #   in Loop: Header=BB12_158 Depth=4
	mov	RDI, QWORD PTR [RSP + 848]
	movzx	EAX, WORD PTR [RDI + 2*RAX]
.LBB12_198:                             # %postload1312
                                        #   in Loop: Header=BB12_158 Depth=4
	movdqa	XMM1, XMM2
	pslld	XMM1, 2
	movdqa	XMMWORD PTR [RSP + 704], XMM1
	movsx	EDX, DX
	movsx	ESI, SI
	movd	XMM1, ESI
	pinsrd	XMM1, EDX, 1
	movsx	ECX, CX
	pinsrd	XMM1, ECX, 2
	movsx	EAX, AX
	pinsrd	XMM1, EAX, 3
	pslld	XMM1, 1
	movdqa	XMMWORD PTR [RSP + 688], XMM1
	movsxd	RAX, DWORD PTR [RSP + 716]
	movq	XMM1, RAX
	movsxd	RCX, DWORD PTR [RSP + 712]
	movq	XMM3, RCX
	punpcklqdq	XMM3, XMM1      # xmm3 = xmm3[0],xmm1[0]
	movsxd	RDX, DWORD PTR [RSP + 708]
	movq	XMM1, RDX
	movsxd	RSI, DWORD PTR [RSP + 704]
	movq	XMM4, RSI
	punpcklqdq	XMM4, XMM1      # xmm4 = xmm4[0],xmm1[0]
	movsxd	RDI, DWORD PTR [RSP + 700]
	mov	QWORD PTR [RSP + 432], RDI # 8-byte Spill
	movq	XMM1, RDI
	movsxd	RDI, DWORD PTR [RSP + 696]
	mov	QWORD PTR [RSP + 424], RDI # 8-byte Spill
	movq	XMM0, RDI
	punpcklqdq	XMM0, XMM1      # xmm0 = xmm0[0],xmm1[0]
	movdqa	XMMWORD PTR [RSP + 192], XMM0 # 16-byte Spill
	movsxd	RDI, DWORD PTR [RSP + 692]
	mov	QWORD PTR [RSP + 416], RDI # 8-byte Spill
	movq	XMM1, RDI
	movsxd	RDI, DWORD PTR [RSP + 688]
	mov	QWORD PTR [RSP + 408], RDI # 8-byte Spill
	movq	XMM0, RDI
	punpcklqdq	XMM0, XMM1      # xmm0 = xmm0[0],xmm1[0]
	movdqa	XMMWORD PTR [RSP + 176], XMM0 # 16-byte Spill
	test	EBX, EBX
	movd	EDI, XMM2
	movsxd	RDI, EDI
	mov	QWORD PTR [RSP + 608], RDI # 8-byte Spill
	jns	.LBB12_200
# BB#199:                               # %deload990
                                        #   in Loop: Header=BB12_158 Depth=4
	mov	RDI, QWORD PTR [RSP + 960]
	movss	XMM0, DWORD PTR [RDI]
                                        # kill: XMM0<def> XMM0<kill> XMM0<def>
.LBB12_200:                             # %postload991
                                        #   in Loop: Header=BB12_158 Depth=4
	test	R12D, R12D
	jns	.LBB12_202
# BB#201:                               # %deload1098
                                        #   in Loop: Header=BB12_158 Depth=4
	mov	RDI, QWORD PTR [RSP + 960]
	movss	XMM1, DWORD PTR [RDI]
.LBB12_202:                             # %postload1099
                                        #   in Loop: Header=BB12_158 Depth=4
	test	R15D, R15D
	jns	.LBB12_204
# BB#203:                               # %deload1206
                                        #   in Loop: Header=BB12_158 Depth=4
	mov	RDI, QWORD PTR [RSP + 960]
	movss	XMM2, DWORD PTR [RDI]
.LBB12_204:                             # %postload1207
                                        #   in Loop: Header=BB12_158 Depth=4
	test	R14D, R14D
	jns	.LBB12_206
# BB#205:                               # %deload1314
                                        #   in Loop: Header=BB12_158 Depth=4
	mov	RDI, QWORD PTR [RSP + 960]
	movss	XMM5, DWORD PTR [RDI]
.LBB12_206:                             # %postload1315
                                        #   in Loop: Header=BB12_158 Depth=4
                                        # kill: XMM0<def> XMM0<kill> XMM0<def>
	insertps	XMM0, XMM1, 16  # xmm0 = xmm0[0],xmm1[0],xmm0[2,3]
	insertps	XMM0, XMM2, 32  # xmm0 = xmm0[0,1],xmm2[0],xmm0[3]
	insertps	XMM0, XMM5, 48  # xmm0 = xmm0[0,1,2],xmm5[0]
	test	EBX, EBX
	jns	.LBB12_208
# BB#207:                               # %deload993
                                        #   in Loop: Header=BB12_158 Depth=4
	mov	RDI, QWORD PTR [RSP + 968]
	movss	XMM1, DWORD PTR [RDI]
.LBB12_208:                             # %postload994
                                        #   in Loop: Header=BB12_158 Depth=4
	test	R12D, R12D
	jns	.LBB12_210
# BB#209:                               # %deload1101
                                        #   in Loop: Header=BB12_158 Depth=4
	mov	RDI, QWORD PTR [RSP + 968]
	movss	XMM2, DWORD PTR [RDI]
.LBB12_210:                             # %postload1102
                                        #   in Loop: Header=BB12_158 Depth=4
	test	R15D, R15D
	jns	.LBB12_212
# BB#211:                               # %deload1209
                                        #   in Loop: Header=BB12_158 Depth=4
	mov	RDI, QWORD PTR [RSP + 968]
	movss	XMM5, DWORD PTR [RDI]
.LBB12_212:                             # %postload1210
                                        #   in Loop: Header=BB12_158 Depth=4
	test	R14D, R14D
	jns	.LBB12_214
# BB#213:                               # %deload1317
                                        #   in Loop: Header=BB12_158 Depth=4
	mov	RDI, QWORD PTR [RSP + 968]
	movss	XMM6, DWORD PTR [RDI]
.LBB12_214:                             # %postload1318
                                        #   in Loop: Header=BB12_158 Depth=4
	insertps	XMM1, XMM2, 16  # xmm1 = xmm1[0],xmm2[0],xmm1[2,3]
	insertps	XMM1, XMM5, 32  # xmm1 = xmm1[0,1],xmm5[0],xmm1[3]
	insertps	XMM1, XMM6, 48  # xmm1 = xmm1[0,1,2],xmm6[0]
	test	EBX, EBX
	jns	.LBB12_216
# BB#215:                               # %deload996
                                        #   in Loop: Header=BB12_158 Depth=4
	mov	RDI, QWORD PTR [RSP + 976]
	movss	XMM2, DWORD PTR [RDI]
	movdqa	XMMWORD PTR [RSP + 560], XMM2 # 16-byte Spill
.LBB12_216:                             # %postload997
                                        #   in Loop: Header=BB12_158 Depth=4
	test	R12D, R12D
	jns	.LBB12_218
# BB#217:                               # %deload1104
                                        #   in Loop: Header=BB12_158 Depth=4
	mov	RDI, QWORD PTR [RSP + 976]
	movss	XMM2, DWORD PTR [RDI]
.LBB12_218:                             # %postload1105
                                        #   in Loop: Header=BB12_158 Depth=4
	test	R15D, R15D
	jns	.LBB12_220
# BB#219:                               # %deload1212
                                        #   in Loop: Header=BB12_158 Depth=4
	mov	RDI, QWORD PTR [RSP + 976]
	movss	XMM5, DWORD PTR [RDI]
.LBB12_220:                             # %postload1213
                                        #   in Loop: Header=BB12_158 Depth=4
	test	R14D, R14D
	jns	.LBB12_222
# BB#221:                               # %deload1320
                                        #   in Loop: Header=BB12_158 Depth=4
	mov	RDI, QWORD PTR [RSP + 976]
	movss	XMM6, DWORD PTR [RDI]
.LBB12_222:                             # %postload1321
                                        #   in Loop: Header=BB12_158 Depth=4
	movdqa	XMM7, XMMWORD PTR [RSP + 560] # 16-byte Reload
	insertps	XMM7, XMM2, 16  # xmm7 = xmm7[0],xmm2[0],xmm7[2,3]
	insertps	XMM7, XMM5, 32  # xmm7 = xmm7[0,1],xmm5[0],xmm7[3]
	insertps	XMM7, XMM6, 48  # xmm7 = xmm7[0,1,2],xmm6[0]
	movdqa	XMMWORD PTR [RSP + 560], XMM7 # 16-byte Spill
	test	EBX, EBX
	jns	.LBB12_224
# BB#223:                               # %deload999
                                        #   in Loop: Header=BB12_158 Depth=4
	mov	RDI, QWORD PTR [RSP + 856]
	movss	XMM2, DWORD PTR [RDI + 4*RSI]
.LBB12_224:                             # %postload1000
                                        #   in Loop: Header=BB12_158 Depth=4
	test	R12D, R12D
	jns	.LBB12_226
# BB#225:                               # %deload1107
                                        #   in Loop: Header=BB12_158 Depth=4
	mov	RSI, QWORD PTR [RSP + 856]
	movss	XMM5, DWORD PTR [RSI + 4*RDX]
.LBB12_226:                             # %postload1108
                                        #   in Loop: Header=BB12_158 Depth=4
	test	R15D, R15D
	jns	.LBB12_228
# BB#227:                               # %deload1215
                                        #   in Loop: Header=BB12_158 Depth=4
	mov	RDX, QWORD PTR [RSP + 856]
	movss	XMM6, DWORD PTR [RDX + 4*RCX]
.LBB12_228:                             # %postload1216
                                        #   in Loop: Header=BB12_158 Depth=4
	test	R14D, R14D
	jns	.LBB12_230
# BB#229:                               # %deload1323
                                        #   in Loop: Header=BB12_158 Depth=4
	mov	RCX, QWORD PTR [RSP + 856]
	movss	XMM7, DWORD PTR [RCX + 4*RAX]
.LBB12_230:                             # %postload1324
                                        #   in Loop: Header=BB12_158 Depth=4
	insertps	XMM2, XMM5, 16  # xmm2 = xmm2[0],xmm5[0],xmm2[2,3]
	insertps	XMM2, XMM6, 32  # xmm2 = xmm2[0,1],xmm6[0],xmm2[3]
	insertps	XMM2, XMM7, 48  # xmm2 = xmm2[0,1,2],xmm7[0]
	movdqa	XMM5, XMMWORD PTR [RIP + .LCPI12_2]
	divps	XMM5, XMM2
	movdqa	XMM2, XMM4
	movdqa	XMM6, XMMWORD PTR [RIP + .LCPI12_3]
	por	XMM2, XMM6
	pextrq	RAX, XMM2, 1
	movdqa	XMM7, XMM3
	por	XMM7, XMM6
	pextrq	RCX, XMM7, 1
	movq	RDX, XMM7
	test	EBX, EBX
	jns	.LBB12_232
# BB#231:                               # %deload1002
                                        #   in Loop: Header=BB12_158 Depth=4
	movq	RSI, XMM2
	mov	RDI, QWORD PTR [RSP + 856]
	movss	XMM2, DWORD PTR [RDI + 4*RSI]
	movdqa	XMMWORD PTR [RSP + 656], XMM2 # 16-byte Spill
.LBB12_232:                             # %postload1003
                                        #   in Loop: Header=BB12_158 Depth=4
	test	R12D, R12D
	jns	.LBB12_234
# BB#233:                               # %deload1110
                                        #   in Loop: Header=BB12_158 Depth=4
	mov	RSI, QWORD PTR [RSP + 856]
	movss	XMM2, DWORD PTR [RSI + 4*RAX]
.LBB12_234:                             # %postload1111
                                        #   in Loop: Header=BB12_158 Depth=4
	test	R15D, R15D
	jns	.LBB12_236
# BB#235:                               # %deload1218
                                        #   in Loop: Header=BB12_158 Depth=4
	mov	RAX, QWORD PTR [RSP + 856]
	movss	XMM6, DWORD PTR [RAX + 4*RDX]
.LBB12_236:                             # %postload1219
                                        #   in Loop: Header=BB12_158 Depth=4
	test	R14D, R14D
	jns	.LBB12_238
# BB#237:                               # %deload1326
                                        #   in Loop: Header=BB12_158 Depth=4
	mov	RAX, QWORD PTR [RSP + 856]
	movss	XMM7, DWORD PTR [RAX + 4*RCX]
.LBB12_238:                             # %postload1327
                                        #   in Loop: Header=BB12_158 Depth=4
	movdqa	XMM8, XMMWORD PTR [RSP + 656] # 16-byte Reload
	insertps	XMM8, XMM2, 16  # xmm8 = xmm8[0],xmm2[0],xmm8[2,3]
	insertps	XMM8, XMM6, 32  # xmm8 = xmm8[0,1],xmm6[0],xmm8[3]
	insertps	XMM8, XMM7, 48  # xmm8 = xmm8[0,1,2],xmm7[0]
	mulps	XMM8, XMM5
	movdqa	XMMWORD PTR [RSP + 656], XMM8 # 16-byte Spill
	movdqa	XMM2, XMM4
	movdqa	XMM6, XMMWORD PTR [RIP + .LCPI12_4]
	por	XMM2, XMM6
	pextrq	RAX, XMM2, 1
	movdqa	XMM7, XMM3
	por	XMM7, XMM6
	pextrq	RCX, XMM7, 1
	movq	RDX, XMM7
	test	EBX, EBX
	jns	.LBB12_240
# BB#239:                               # %deload1005
                                        #   in Loop: Header=BB12_158 Depth=4
	movq	RSI, XMM2
	mov	RDI, QWORD PTR [RSP + 856]
	movss	XMM2, DWORD PTR [RDI + 4*RSI]
	movdqa	XMMWORD PTR [RSP + 640], XMM2 # 16-byte Spill
.LBB12_240:                             # %postload1006
                                        #   in Loop: Header=BB12_158 Depth=4
	test	R12D, R12D
	jns	.LBB12_242
# BB#241:                               # %deload1113
                                        #   in Loop: Header=BB12_158 Depth=4
	mov	RSI, QWORD PTR [RSP + 856]
	movss	XMM2, DWORD PTR [RSI + 4*RAX]
.LBB12_242:                             # %postload1114
                                        #   in Loop: Header=BB12_158 Depth=4
	test	R15D, R15D
	jns	.LBB12_244
# BB#243:                               # %deload1221
                                        #   in Loop: Header=BB12_158 Depth=4
	mov	RAX, QWORD PTR [RSP + 856]
	movss	XMM6, DWORD PTR [RAX + 4*RDX]
.LBB12_244:                             # %postload1222
                                        #   in Loop: Header=BB12_158 Depth=4
	test	R14D, R14D
	jns	.LBB12_246
# BB#245:                               # %deload1329
                                        #   in Loop: Header=BB12_158 Depth=4
	mov	RAX, QWORD PTR [RSP + 856]
	movss	XMM7, DWORD PTR [RAX + 4*RCX]
.LBB12_246:                             # %postload1330
                                        #   in Loop: Header=BB12_158 Depth=4
	movdqa	XMM8, XMMWORD PTR [RSP + 640] # 16-byte Reload
	insertps	XMM8, XMM2, 16  # xmm8 = xmm8[0],xmm2[0],xmm8[2,3]
	insertps	XMM8, XMM6, 32  # xmm8 = xmm8[0,1],xmm6[0],xmm8[3]
	insertps	XMM8, XMM7, 48  # xmm8 = xmm8[0,1,2],xmm7[0]
	mulps	XMM8, XMM5
	movdqa	XMMWORD PTR [RSP + 640], XMM8 # 16-byte Spill
	mulps	XMM0, XMM1
	movaps	XMM1, XMMWORD PTR [RIP + .LCPI12_5]
	por	XMM4, XMM1
	pextrq	RAX, XMM4, 1
	por	XMM3, XMM1
	pextrq	RCX, XMM3, 1
	movq	RDX, XMM3
	test	EBX, EBX
	jns	.LBB12_248
# BB#247:                               # %deload1008
                                        #   in Loop: Header=BB12_158 Depth=4
	movq	RSI, XMM4
	mov	RDI, QWORD PTR [RSP + 856]
	movss	XMM1, DWORD PTR [RDI + 4*RSI]
.LBB12_248:                             # %postload1009
                                        #   in Loop: Header=BB12_158 Depth=4
	test	R12D, R12D
	jns	.LBB12_250
# BB#249:                               # %deload1116
                                        #   in Loop: Header=BB12_158 Depth=4
	mov	RSI, QWORD PTR [RSP + 856]
	movss	XMM2, DWORD PTR [RSI + 4*RAX]
.LBB12_250:                             # %postload1117
                                        #   in Loop: Header=BB12_158 Depth=4
	test	R15D, R15D
	jns	.LBB12_252
# BB#251:                               # %deload1224
                                        #   in Loop: Header=BB12_158 Depth=4
	mov	RAX, QWORD PTR [RSP + 856]
	movss	XMM3, DWORD PTR [RAX + 4*RDX]
.LBB12_252:                             # %postload1225
                                        #   in Loop: Header=BB12_158 Depth=4
	test	R14D, R14D
	jns	.LBB12_254
# BB#253:                               # %deload1332
                                        #   in Loop: Header=BB12_158 Depth=4
	mov	RAX, QWORD PTR [RSP + 856]
	movss	XMM4, DWORD PTR [RAX + 4*RCX]
.LBB12_254:                             # %postload1333
                                        #   in Loop: Header=BB12_158 Depth=4
	insertps	XMM1, XMM2, 16  # xmm1 = xmm1[0],xmm2[0],xmm1[2,3]
	insertps	XMM1, XMM3, 32  # xmm1 = xmm1[0,1],xmm3[0],xmm1[3]
	insertps	XMM1, XMM4, 48  # xmm1 = xmm1[0,1,2],xmm4[0]
	mulps	XMM5, XMM1
	movaps	XMM1, XMMWORD PTR [RSP + 640] # 16-byte Reload
	mulps	XMM1, XMM1
	movaps	XMM2, XMMWORD PTR [RSP + 656] # 16-byte Reload
	mulps	XMM2, XMM2
	addps	XMM2, XMM1
	mulps	XMM2, XMMWORD PTR [RIP + .LCPI12_6]
	subps	XMM5, XMM2
	mulps	XMM0, XMM5
	call	__ocl_svml_h8_sqrtf4
	test	EBX, EBX
	movaps	XMMWORD PTR [RSP + 576], XMM0 # 16-byte Spill
	jns	.LBB12_256
# BB#255:                               # %deload1011
                                        #   in Loop: Header=BB12_158 Depth=4
	mov	RCX, QWORD PTR [RSP + 472] # 8-byte Reload
	mov	RAX, QWORD PTR [RSP + 680] # 8-byte Reload
	movss	XMM0, DWORD PTR [RAX + 4*RCX]
                                        # kill: XMM0<def> XMM0<kill> XMM0<def>
.LBB12_256:                             # %postload1012
                                        #   in Loop: Header=BB12_158 Depth=4
	test	R12D, R12D
	jns	.LBB12_258
# BB#257:                               # %deload1119
                                        #   in Loop: Header=BB12_158 Depth=4
	mov	RCX, QWORD PTR [RSP + 480] # 8-byte Reload
	mov	RAX, QWORD PTR [RSP + 672] # 8-byte Reload
	movss	XMM1, DWORD PTR [RAX + 4*RCX]
.LBB12_258:                             # %postload1120
                                        #   in Loop: Header=BB12_158 Depth=4
	test	R15D, R15D
	jns	.LBB12_260
# BB#259:                               # %deload1227
                                        #   in Loop: Header=BB12_158 Depth=4
	mov	RAX, QWORD PTR [RSP + 488] # 8-byte Reload
	movss	XMM2, DWORD PTR [R13 + 4*RAX]
.LBB12_260:                             # %postload1228
                                        #   in Loop: Header=BB12_158 Depth=4
	test	R14D, R14D
	jns	.LBB12_262
# BB#261:                               # %deload1335
                                        #   in Loop: Header=BB12_158 Depth=4
	mov	RAX, QWORD PTR [RSP + 496] # 8-byte Reload
	movss	XMM3, DWORD PTR [RBP + 4*RAX]
.LBB12_262:                             # %postload1336
                                        #   in Loop: Header=BB12_158 Depth=4
                                        # kill: XMM0<def> XMM0<kill> XMM0<def>
	insertps	XMM0, XMM1, 16  # xmm0 = xmm0[0],xmm1[0],xmm0[2,3]
	insertps	XMM0, XMM2, 32  # xmm0 = xmm0[0,1],xmm2[0],xmm0[3]
	insertps	XMM0, XMM3, 48  # xmm0 = xmm0[0,1,2],xmm3[0]
	test	EBX, EBX
	jns	.LBB12_264
# BB#263:                               # %deload1014
                                        #   in Loop: Header=BB12_158 Depth=4
	mov	RCX, QWORD PTR [RSP + 512] # 8-byte Reload
	mov	RAX, QWORD PTR [RSP + 680] # 8-byte Reload
	movss	XMM1, DWORD PTR [RAX + 4*RCX]
.LBB12_264:                             # %postload1015
                                        #   in Loop: Header=BB12_158 Depth=4
	test	R12D, R12D
	jns	.LBB12_266
# BB#265:                               # %deload1122
                                        #   in Loop: Header=BB12_158 Depth=4
	mov	RCX, QWORD PTR [RSP + 520] # 8-byte Reload
	mov	RAX, QWORD PTR [RSP + 672] # 8-byte Reload
	movss	XMM2, DWORD PTR [RAX + 4*RCX]
.LBB12_266:                             # %postload1123
                                        #   in Loop: Header=BB12_158 Depth=4
	test	R15D, R15D
	jns	.LBB12_268
# BB#267:                               # %deload1230
                                        #   in Loop: Header=BB12_158 Depth=4
	mov	RAX, QWORD PTR [RSP + 528] # 8-byte Reload
	movss	XMM3, DWORD PTR [R13 + 4*RAX]
.LBB12_268:                             # %postload1231
                                        #   in Loop: Header=BB12_158 Depth=4
	test	R14D, R14D
	jns	.LBB12_270
# BB#269:                               # %deload1338
                                        #   in Loop: Header=BB12_158 Depth=4
	mov	RAX, QWORD PTR [RSP + 536] # 8-byte Reload
	movss	XMM4, DWORD PTR [RBP + 4*RAX]
.LBB12_270:                             # %postload1339
                                        #   in Loop: Header=BB12_158 Depth=4
	insertps	XMM1, XMM2, 16  # xmm1 = xmm1[0],xmm2[0],xmm1[2,3]
	insertps	XMM1, XMM3, 32  # xmm1 = xmm1[0,1],xmm3[0],xmm1[3]
	insertps	XMM1, XMM4, 48  # xmm1 = xmm1[0,1,2],xmm4[0]
	subps	XMM0, XMM1
	movaps	XMM2, XMMWORD PTR [RSP + 368] # 16-byte Reload
	movaps	XMM1, XMMWORD PTR [RIP + .LCPI12_3]
	orps	XMM2, XMM1
	pextrq	RAX, XMM2, 1
	mov	QWORD PTR [RSP + 384], RAX # 8-byte Spill
	movq	QWORD PTR [RSP + 368], XMM2 # 8-byte Folded Spill
	movaps	XMM2, XMMWORD PTR [RSP + 336] # 16-byte Reload
	orps	XMM2, XMM1
	pextrq	RAX, XMM2, 1
	mov	QWORD PTR [RSP + 360], RAX # 8-byte Spill
	movq	QWORD PTR [RSP + 336], XMM2 # 8-byte Folded Spill
	test	EBX, EBX
	jns	.LBB12_272
# BB#271:                               # %deload1017
                                        #   in Loop: Header=BB12_158 Depth=4
	mov	RCX, QWORD PTR [RSP + 368] # 8-byte Reload
	mov	RAX, QWORD PTR [RSP + 680] # 8-byte Reload
	movss	XMM1, DWORD PTR [RAX + 4*RCX]
.LBB12_272:                             # %postload1018
                                        #   in Loop: Header=BB12_158 Depth=4
	test	R12D, R12D
	jns	.LBB12_274
# BB#273:                               # %deload1125
                                        #   in Loop: Header=BB12_158 Depth=4
	mov	RCX, QWORD PTR [RSP + 384] # 8-byte Reload
	mov	RAX, QWORD PTR [RSP + 672] # 8-byte Reload
	movss	XMM2, DWORD PTR [RAX + 4*RCX]
.LBB12_274:                             # %postload1126
                                        #   in Loop: Header=BB12_158 Depth=4
	test	R15D, R15D
	jns	.LBB12_276
# BB#275:                               # %deload1233
                                        #   in Loop: Header=BB12_158 Depth=4
	mov	RAX, QWORD PTR [RSP + 336] # 8-byte Reload
	movss	XMM3, DWORD PTR [R13 + 4*RAX]
.LBB12_276:                             # %postload1234
                                        #   in Loop: Header=BB12_158 Depth=4
	test	R14D, R14D
	jns	.LBB12_278
# BB#277:                               # %deload1341
                                        #   in Loop: Header=BB12_158 Depth=4
	mov	RAX, QWORD PTR [RSP + 360] # 8-byte Reload
	movss	XMM4, DWORD PTR [RBP + 4*RAX]
.LBB12_278:                             # %postload1342
                                        #   in Loop: Header=BB12_158 Depth=4
	insertps	XMM1, XMM2, 16  # xmm1 = xmm1[0],xmm2[0],xmm1[2,3]
	insertps	XMM1, XMM3, 32  # xmm1 = xmm1[0,1],xmm3[0],xmm1[3]
	insertps	XMM1, XMM4, 48  # xmm1 = xmm1[0,1,2],xmm4[0]
	movdqa	XMM3, XMMWORD PTR [RSP + 304] # 16-byte Reload
	movdqa	XMM2, XMMWORD PTR [RIP + .LCPI12_3]
	por	XMM3, XMM2
	pextrq	RAX, XMM3, 1
	mov	QWORD PTR [RSP + 328], RAX # 8-byte Spill
	movq	QWORD PTR [RSP + 304], XMM3 # 8-byte Folded Spill
	movaps	XMM3, XMMWORD PTR [RSP + 592] # 16-byte Reload
	por	XMM3, XMM2
	pextrq	RAX, XMM3, 1
	mov	QWORD PTR [RSP + 296], RAX # 8-byte Spill
	movq	QWORD PTR [RSP + 288], XMM3 # 8-byte Folded Spill
	test	EBX, EBX
	jns	.LBB12_280
# BB#279:                               # %deload1020
                                        #   in Loop: Header=BB12_158 Depth=4
	mov	RCX, QWORD PTR [RSP + 304] # 8-byte Reload
	mov	RAX, QWORD PTR [RSP + 680] # 8-byte Reload
	movss	XMM2, DWORD PTR [RAX + 4*RCX]
.LBB12_280:                             # %postload1021
                                        #   in Loop: Header=BB12_158 Depth=4
	test	R12D, R12D
	jns	.LBB12_282
# BB#281:                               # %deload1128
                                        #   in Loop: Header=BB12_158 Depth=4
	mov	RCX, QWORD PTR [RSP + 328] # 8-byte Reload
	mov	RAX, QWORD PTR [RSP + 672] # 8-byte Reload
	movss	XMM3, DWORD PTR [RAX + 4*RCX]
.LBB12_282:                             # %postload1129
                                        #   in Loop: Header=BB12_158 Depth=4
	test	R15D, R15D
	jns	.LBB12_284
# BB#283:                               # %deload1236
                                        #   in Loop: Header=BB12_158 Depth=4
	mov	RAX, QWORD PTR [RSP + 288] # 8-byte Reload
	movss	XMM4, DWORD PTR [R13 + 4*RAX]
.LBB12_284:                             # %postload1237
                                        #   in Loop: Header=BB12_158 Depth=4
	test	R14D, R14D
	jns	.LBB12_286
# BB#285:                               # %deload1344
                                        #   in Loop: Header=BB12_158 Depth=4
	mov	RAX, QWORD PTR [RSP + 296] # 8-byte Reload
	movss	XMM5, DWORD PTR [RBP + 4*RAX]
.LBB12_286:                             # %postload1345
                                        #   in Loop: Header=BB12_158 Depth=4
	insertps	XMM2, XMM3, 16  # xmm2 = xmm2[0],xmm3[0],xmm2[2,3]
	insertps	XMM2, XMM4, 32  # xmm2 = xmm2[0,1],xmm4[0],xmm2[3]
	insertps	XMM2, XMM5, 48  # xmm2 = xmm2[0,1,2],xmm5[0]
	subps	XMM1, XMM2
	movaps	XMM2, XMMWORD PTR [RSP + 656] # 16-byte Reload
	mulps	XMM2, XMM1
	movaps	XMMWORD PTR [RSP + 592], XMM2 # 16-byte Spill
	mulps	XMM1, XMM1
	movaps	XMM3, XMMWORD PTR [RSP + 640] # 16-byte Reload
	mulps	XMM3, XMM0
	movaps	XMMWORD PTR [RSP + 160], XMM3 # 16-byte Spill
	mulps	XMM0, XMM0
	addps	XMM0, XMM1
	call	__ocl_svml_h8_sqrtf4
	mulps	XMM0, XMMWORD PTR [RSP + 576] # 16-byte Folded Reload
	movaps	XMM2, XMMWORD PTR [RSP + 592] # 16-byte Reload
	subps	XMM2, XMMWORD PTR [RSP + 160] # 16-byte Folded Reload
	andps	XMM2, XMMWORD PTR [RIP + .LCPI12_7]
	addps	XMM2, XMM0
	movaps	XMMWORD PTR [RSP + 592], XMM2 # 16-byte Spill
	test	EBX, EBX
	jns	.LBB12_288
# BB#287:                               # %deload1023
                                        #   in Loop: Header=BB12_158 Depth=4
	movaps	XMM1, XMM2
	mov	RCX, QWORD PTR [RSP + 608] # 8-byte Reload
	mov	RAX, QWORD PTR [RSP + 864]
	movss	DWORD PTR [RAX + 4*RCX], XMM1
.LBB12_288:                             # %postload1024
                                        #   in Loop: Header=BB12_158 Depth=4
	test	R12D, R12D
	jns	.LBB12_290
# BB#289:                               # %deload1131
                                        #   in Loop: Header=BB12_158 Depth=4
	pshufd	XMM1, XMMWORD PTR [RSP + 592], 1 # 16-byte Folded Reload
                                        # xmm1 = mem[1,0,0,0]
	mov	RCX, QWORD PTR [RSP + 616] # 8-byte Reload
	mov	RAX, QWORD PTR [RSP + 864]
	movss	DWORD PTR [RAX + 4*RCX], XMM1
.LBB12_290:                             # %postload1132
                                        #   in Loop: Header=BB12_158 Depth=4
	test	R15D, R15D
	jns	.LBB12_292
# BB#291:                               # %deload1239
                                        #   in Loop: Header=BB12_158 Depth=4
	movaps	XMM1, XMMWORD PTR [RSP + 592] # 16-byte Reload
	movhlps	XMM1, XMM1              # xmm1 = xmm1[1,1]
	mov	RCX, QWORD PTR [RSP + 624] # 8-byte Reload
	mov	RAX, QWORD PTR [RSP + 864]
	movss	DWORD PTR [RAX + 4*RCX], XMM1
.LBB12_292:                             # %postload1240
                                        #   in Loop: Header=BB12_158 Depth=4
	test	R14D, R14D
	jns	.LBB12_294
# BB#293:                               # %deload1347
                                        #   in Loop: Header=BB12_158 Depth=4
	pshufd	XMM1, XMMWORD PTR [RSP + 592], 3 # 16-byte Folded Reload
                                        # xmm1 = mem[3,0,0,0]
	mov	RCX, QWORD PTR [RSP + 632] # 8-byte Reload
	mov	RAX, QWORD PTR [RSP + 864]
	movss	DWORD PTR [RAX + 4*RCX], XMM1
.LBB12_294:                             # %postload1348
                                        #   in Loop: Header=BB12_158 Depth=4
	test	EBX, EBX
	jns	.LBB12_296
# BB#295:                               # %deload1025
                                        #   in Loop: Header=BB12_158 Depth=4
	mov	RCX, QWORD PTR [RSP + 440] # 8-byte Reload
	mov	RAX, QWORD PTR [RSP + 680] # 8-byte Reload
	movss	XMM0, DWORD PTR [RAX + 4*RCX]
                                        # kill: XMM0<def> XMM0<kill> XMM0<def>
.LBB12_296:                             # %postload1026
                                        #   in Loop: Header=BB12_158 Depth=4
	test	R12D, R12D
	jns	.LBB12_298
# BB#297:                               # %deload1133
                                        #   in Loop: Header=BB12_158 Depth=4
	mov	RCX, QWORD PTR [RSP + 448] # 8-byte Reload
	mov	RAX, QWORD PTR [RSP + 672] # 8-byte Reload
	movss	XMM1, DWORD PTR [RAX + 4*RCX]
.LBB12_298:                             # %postload1134
                                        #   in Loop: Header=BB12_158 Depth=4
	test	R15D, R15D
	jns	.LBB12_300
# BB#299:                               # %deload1241
                                        #   in Loop: Header=BB12_158 Depth=4
	mov	RAX, QWORD PTR [RSP + 456] # 8-byte Reload
	movss	XMM2, DWORD PTR [R13 + 4*RAX]
.LBB12_300:                             # %postload1242
                                        #   in Loop: Header=BB12_158 Depth=4
	test	R14D, R14D
	jns	.LBB12_302
# BB#301:                               # %deload1349
                                        #   in Loop: Header=BB12_158 Depth=4
	mov	RAX, QWORD PTR [RSP + 464] # 8-byte Reload
	movss	XMM3, DWORD PTR [RBP + 4*RAX]
.LBB12_302:                             # %postload1350
                                        #   in Loop: Header=BB12_158 Depth=4
                                        # kill: XMM0<def> XMM0<kill> XMM0<def>
	insertps	XMM0, XMM1, 16  # xmm0 = xmm0[0],xmm1[0],xmm0[2,3]
	insertps	XMM0, XMM2, 32  # xmm0 = xmm0[0,1],xmm2[0],xmm0[3]
	insertps	XMM0, XMM3, 48  # xmm0 = xmm0[0,1,2],xmm3[0]
	test	EBX, EBX
	jns	.LBB12_304
# BB#303:                               # %deload1028
                                        #   in Loop: Header=BB12_158 Depth=4
	mov	RCX, QWORD PTR [RSP + 472] # 8-byte Reload
	mov	RAX, QWORD PTR [RSP + 680] # 8-byte Reload
	movss	XMM1, DWORD PTR [RAX + 4*RCX]
.LBB12_304:                             # %postload1029
                                        #   in Loop: Header=BB12_158 Depth=4
	test	R12D, R12D
	jns	.LBB12_306
# BB#305:                               # %deload1136
                                        #   in Loop: Header=BB12_158 Depth=4
	mov	RCX, QWORD PTR [RSP + 480] # 8-byte Reload
	mov	RAX, QWORD PTR [RSP + 672] # 8-byte Reload
	movss	XMM2, DWORD PTR [RAX + 4*RCX]
.LBB12_306:                             # %postload1137
                                        #   in Loop: Header=BB12_158 Depth=4
	test	R15D, R15D
	jns	.LBB12_308
# BB#307:                               # %deload1244
                                        #   in Loop: Header=BB12_158 Depth=4
	mov	RAX, QWORD PTR [RSP + 488] # 8-byte Reload
	movss	XMM3, DWORD PTR [R13 + 4*RAX]
.LBB12_308:                             # %postload1245
                                        #   in Loop: Header=BB12_158 Depth=4
	test	R14D, R14D
	jns	.LBB12_310
# BB#309:                               # %deload1352
                                        #   in Loop: Header=BB12_158 Depth=4
	mov	RAX, QWORD PTR [RSP + 496] # 8-byte Reload
	movss	XMM4, DWORD PTR [RBP + 4*RAX]
.LBB12_310:                             # %postload1353
                                        #   in Loop: Header=BB12_158 Depth=4
	insertps	XMM1, XMM2, 16  # xmm1 = xmm1[0],xmm2[0],xmm1[2,3]
	insertps	XMM1, XMM3, 32  # xmm1 = xmm1[0,1],xmm3[0],xmm1[3]
	insertps	XMM1, XMM4, 48  # xmm1 = xmm1[0,1,2],xmm4[0]
	subps	XMM0, XMM1
	movaps	XMM2, XMMWORD PTR [RSP + 208] # 16-byte Reload
	movaps	XMM1, XMMWORD PTR [RIP + .LCPI12_3]
	orps	XMM2, XMM1
	pextrq	RAX, XMM2, 1
	mov	QWORD PTR [RSP + 496], RAX # 8-byte Spill
	movq	QWORD PTR [RSP + 488], XMM2 # 8-byte Folded Spill
	movaps	XMM2, XMMWORD PTR [RSP + 224] # 16-byte Reload
	orps	XMM2, XMM1
	pextrq	RAX, XMM2, 1
	mov	QWORD PTR [RSP + 480], RAX # 8-byte Spill
	movq	QWORD PTR [RSP + 472], XMM2 # 8-byte Folded Spill
	test	EBX, EBX
	jns	.LBB12_312
# BB#311:                               # %deload1031
                                        #   in Loop: Header=BB12_158 Depth=4
	mov	RCX, QWORD PTR [RSP + 488] # 8-byte Reload
	mov	RAX, QWORD PTR [RSP + 680] # 8-byte Reload
	movss	XMM1, DWORD PTR [RAX + 4*RCX]
.LBB12_312:                             # %postload1032
                                        #   in Loop: Header=BB12_158 Depth=4
	test	R12D, R12D
	jns	.LBB12_314
# BB#313:                               # %deload1139
                                        #   in Loop: Header=BB12_158 Depth=4
	mov	RCX, QWORD PTR [RSP + 496] # 8-byte Reload
	mov	RAX, QWORD PTR [RSP + 672] # 8-byte Reload
	movss	XMM2, DWORD PTR [RAX + 4*RCX]
.LBB12_314:                             # %postload1140
                                        #   in Loop: Header=BB12_158 Depth=4
	test	R15D, R15D
	jns	.LBB12_316
# BB#315:                               # %deload1247
                                        #   in Loop: Header=BB12_158 Depth=4
	mov	RAX, QWORD PTR [RSP + 472] # 8-byte Reload
	movss	XMM3, DWORD PTR [R13 + 4*RAX]
.LBB12_316:                             # %postload1248
                                        #   in Loop: Header=BB12_158 Depth=4
	test	R14D, R14D
	jns	.LBB12_318
# BB#317:                               # %deload1355
                                        #   in Loop: Header=BB12_158 Depth=4
	mov	RAX, QWORD PTR [RSP + 480] # 8-byte Reload
	movss	XMM4, DWORD PTR [RBP + 4*RAX]
.LBB12_318:                             # %postload1356
                                        #   in Loop: Header=BB12_158 Depth=4
	insertps	XMM1, XMM2, 16  # xmm1 = xmm1[0],xmm2[0],xmm1[2,3]
	insertps	XMM1, XMM3, 32  # xmm1 = xmm1[0,1],xmm3[0],xmm1[3]
	insertps	XMM1, XMM4, 48  # xmm1 = xmm1[0,1,2],xmm4[0]
	test	EBX, EBX
	jns	.LBB12_320
# BB#319:                               # %deload1034
                                        #   in Loop: Header=BB12_158 Depth=4
	mov	RCX, QWORD PTR [RSP + 368] # 8-byte Reload
	mov	RAX, QWORD PTR [RSP + 680] # 8-byte Reload
	movss	XMM2, DWORD PTR [RAX + 4*RCX]
.LBB12_320:                             # %postload1035
                                        #   in Loop: Header=BB12_158 Depth=4
	test	R12D, R12D
	jns	.LBB12_322
# BB#321:                               # %deload1142
                                        #   in Loop: Header=BB12_158 Depth=4
	mov	RCX, QWORD PTR [RSP + 384] # 8-byte Reload
	mov	RAX, QWORD PTR [RSP + 672] # 8-byte Reload
	movss	XMM3, DWORD PTR [RAX + 4*RCX]
.LBB12_322:                             # %postload1143
                                        #   in Loop: Header=BB12_158 Depth=4
	test	R15D, R15D
	jns	.LBB12_324
# BB#323:                               # %deload1250
                                        #   in Loop: Header=BB12_158 Depth=4
	mov	RAX, QWORD PTR [RSP + 336] # 8-byte Reload
	movss	XMM4, DWORD PTR [R13 + 4*RAX]
.LBB12_324:                             # %postload1251
                                        #   in Loop: Header=BB12_158 Depth=4
	test	R14D, R14D
	jns	.LBB12_326
# BB#325:                               # %deload1358
                                        #   in Loop: Header=BB12_158 Depth=4
	mov	RAX, QWORD PTR [RSP + 360] # 8-byte Reload
	movss	XMM5, DWORD PTR [RBP + 4*RAX]
.LBB12_326:                             # %postload1359
                                        #   in Loop: Header=BB12_158 Depth=4
	insertps	XMM2, XMM3, 16  # xmm2 = xmm2[0],xmm3[0],xmm2[2,3]
	insertps	XMM2, XMM4, 32  # xmm2 = xmm2[0,1],xmm4[0],xmm2[3]
	insertps	XMM2, XMM5, 48  # xmm2 = xmm2[0,1,2],xmm5[0]
	subps	XMM1, XMM2
	movaps	XMM2, XMMWORD PTR [RSP + 656] # 16-byte Reload
	mulps	XMM2, XMM1
	movaps	XMMWORD PTR [RSP + 592], XMM2 # 16-byte Spill
	mulps	XMM1, XMM1
	movaps	XMM3, XMMWORD PTR [RSP + 640] # 16-byte Reload
	mulps	XMM3, XMM0
	movaps	XMMWORD PTR [RSP + 384], XMM3 # 16-byte Spill
	mulps	XMM0, XMM0
	addps	XMM0, XMM1
	call	__ocl_svml_h8_sqrtf4
	mulps	XMM0, XMMWORD PTR [RSP + 576] # 16-byte Folded Reload
	movaps	XMM2, XMMWORD PTR [RSP + 592] # 16-byte Reload
	subps	XMM2, XMMWORD PTR [RSP + 384] # 16-byte Folded Reload
	andps	XMM2, XMMWORD PTR [RIP + .LCPI12_7]
	addps	XMM2, XMM0
	movaps	XMMWORD PTR [RSP + 592], XMM2 # 16-byte Spill
	test	EBX, EBX
	jns	.LBB12_328
# BB#327:                               # %deload1037
                                        #   in Loop: Header=BB12_158 Depth=4
	mov	RCX, QWORD PTR [RSP + 608] # 8-byte Reload
	mov	RAX, QWORD PTR [RSP + 864]
	movss	XMM1, DWORD PTR [RAX + 4*RCX]
.LBB12_328:                             # %postload1038
                                        #   in Loop: Header=BB12_158 Depth=4
	test	R12D, R12D
	jns	.LBB12_330
# BB#329:                               # %deload1145
                                        #   in Loop: Header=BB12_158 Depth=4
	mov	RCX, QWORD PTR [RSP + 616] # 8-byte Reload
	mov	RAX, QWORD PTR [RSP + 864]
	movss	XMM0, DWORD PTR [RAX + 4*RCX]
.LBB12_330:                             # %postload1146
                                        #   in Loop: Header=BB12_158 Depth=4
	test	R15D, R15D
	jns	.LBB12_332
# BB#331:                               # %deload1253
                                        #   in Loop: Header=BB12_158 Depth=4
	mov	RCX, QWORD PTR [RSP + 624] # 8-byte Reload
	mov	RAX, QWORD PTR [RSP + 864]
	movss	XMM2, DWORD PTR [RAX + 4*RCX]
.LBB12_332:                             # %postload1254
                                        #   in Loop: Header=BB12_158 Depth=4
	test	R14D, R14D
	jns	.LBB12_334
# BB#333:                               # %deload1361
                                        #   in Loop: Header=BB12_158 Depth=4
	mov	RCX, QWORD PTR [RSP + 632] # 8-byte Reload
	mov	RAX, QWORD PTR [RSP + 864]
	movss	XMM3, DWORD PTR [RAX + 4*RCX]
.LBB12_334:                             # %postload1362
                                        #   in Loop: Header=BB12_158 Depth=4
	insertps	XMM1, XMM0, 16  # xmm1 = xmm1[0],xmm0[0],xmm1[2,3]
	insertps	XMM1, XMM2, 32  # xmm1 = xmm1[0,1],xmm2[0],xmm1[3]
	insertps	XMM1, XMM3, 48  # xmm1 = xmm1[0,1,2],xmm3[0]
	addps	XMM1, XMMWORD PTR [RSP + 592] # 16-byte Folded Reload
	test	EBX, EBX
	jns	.LBB12_336
# BB#335:                               # %deload1040
                                        #   in Loop: Header=BB12_158 Depth=4
	mov	RCX, QWORD PTR [RSP + 608] # 8-byte Reload
	mov	RAX, QWORD PTR [RSP + 864]
	movss	DWORD PTR [RAX + 4*RCX], XMM1
.LBB12_336:                             # %postload1041
                                        #   in Loop: Header=BB12_158 Depth=4
	test	R12D, R12D
	jns	.LBB12_338
# BB#337:                               # %deload1148
                                        #   in Loop: Header=BB12_158 Depth=4
	pshufd	XMM0, XMM1, 1           # xmm0 = xmm1[1,0,0,0]
	mov	RCX, QWORD PTR [RSP + 616] # 8-byte Reload
	mov	RAX, QWORD PTR [RSP + 864]
	movss	DWORD PTR [RAX + 4*RCX], XMM0
.LBB12_338:                             # %postload1149
                                        #   in Loop: Header=BB12_158 Depth=4
	test	R15D, R15D
	jns	.LBB12_340
# BB#339:                               # %deload1256
                                        #   in Loop: Header=BB12_158 Depth=4
	movaps	XMM0, XMM1
	movhlps	XMM0, XMM0              # xmm0 = xmm0[1,1]
	mov	RCX, QWORD PTR [RSP + 624] # 8-byte Reload
	mov	RAX, QWORD PTR [RSP + 864]
	movss	DWORD PTR [RAX + 4*RCX], XMM0
.LBB12_340:                             # %postload1257
                                        #   in Loop: Header=BB12_158 Depth=4
	test	R14D, R14D
	jns	.LBB12_342
# BB#341:                               # %deload1364
                                        #   in Loop: Header=BB12_158 Depth=4
	pshufd	XMM1, XMM1, 3           # xmm1 = xmm1[3,0,0,0]
	mov	RCX, QWORD PTR [RSP + 632] # 8-byte Reload
	mov	RAX, QWORD PTR [RSP + 864]
	movss	DWORD PTR [RAX + 4*RCX], XMM1
.LBB12_342:                             # %postload1365
                                        #   in Loop: Header=BB12_158 Depth=4
	test	EBX, EBX
	jns	.LBB12_344
# BB#343:                               # %deload1042
                                        #   in Loop: Header=BB12_158 Depth=4
	mov	RCX, QWORD PTR [RSP + 408] # 8-byte Reload
	mov	RAX, QWORD PTR [RSP + 680] # 8-byte Reload
	movss	XMM0, DWORD PTR [RAX + 4*RCX]
                                        # kill: XMM0<def> XMM0<kill> XMM0<def>
.LBB12_344:                             # %postload1043
                                        #   in Loop: Header=BB12_158 Depth=4
	test	R12D, R12D
	jns	.LBB12_346
# BB#345:                               # %deload1150
                                        #   in Loop: Header=BB12_158 Depth=4
	mov	RCX, QWORD PTR [RSP + 416] # 8-byte Reload
	mov	RAX, QWORD PTR [RSP + 672] # 8-byte Reload
	movss	XMM1, DWORD PTR [RAX + 4*RCX]
.LBB12_346:                             # %postload1151
                                        #   in Loop: Header=BB12_158 Depth=4
	test	R15D, R15D
	jns	.LBB12_348
# BB#347:                               # %deload1258
                                        #   in Loop: Header=BB12_158 Depth=4
	mov	RAX, QWORD PTR [RSP + 424] # 8-byte Reload
	movss	XMM2, DWORD PTR [R13 + 4*RAX]
.LBB12_348:                             # %postload1259
                                        #   in Loop: Header=BB12_158 Depth=4
	test	R14D, R14D
	jns	.LBB12_350
# BB#349:                               # %deload1366
                                        #   in Loop: Header=BB12_158 Depth=4
	mov	RAX, QWORD PTR [RSP + 432] # 8-byte Reload
	movss	XMM3, DWORD PTR [RBP + 4*RAX]
.LBB12_350:                             # %postload1367
                                        #   in Loop: Header=BB12_158 Depth=4
                                        # kill: XMM0<def> XMM0<kill> XMM0<def>
	insertps	XMM0, XMM1, 16  # xmm0 = xmm0[0],xmm1[0],xmm0[2,3]
	insertps	XMM0, XMM2, 32  # xmm0 = xmm0[0,1],xmm2[0],xmm0[3]
	insertps	XMM0, XMM3, 48  # xmm0 = xmm0[0,1,2],xmm3[0]
	test	EBX, EBX
	jns	.LBB12_352
# BB#351:                               # %deload1045
                                        #   in Loop: Header=BB12_158 Depth=4
	mov	RCX, QWORD PTR [RSP + 440] # 8-byte Reload
	mov	RAX, QWORD PTR [RSP + 680] # 8-byte Reload
	movss	XMM1, DWORD PTR [RAX + 4*RCX]
.LBB12_352:                             # %postload1046
                                        #   in Loop: Header=BB12_158 Depth=4
	test	R12D, R12D
	jns	.LBB12_354
# BB#353:                               # %deload1153
                                        #   in Loop: Header=BB12_158 Depth=4
	mov	RCX, QWORD PTR [RSP + 448] # 8-byte Reload
	mov	RAX, QWORD PTR [RSP + 672] # 8-byte Reload
	movss	XMM2, DWORD PTR [RAX + 4*RCX]
.LBB12_354:                             # %postload1154
                                        #   in Loop: Header=BB12_158 Depth=4
	test	R15D, R15D
	jns	.LBB12_356
# BB#355:                               # %deload1261
                                        #   in Loop: Header=BB12_158 Depth=4
	mov	RAX, QWORD PTR [RSP + 456] # 8-byte Reload
	movss	XMM3, DWORD PTR [R13 + 4*RAX]
.LBB12_356:                             # %postload1262
                                        #   in Loop: Header=BB12_158 Depth=4
	test	R14D, R14D
	jns	.LBB12_358
# BB#357:                               # %deload1369
                                        #   in Loop: Header=BB12_158 Depth=4
	mov	RAX, QWORD PTR [RSP + 464] # 8-byte Reload
	movss	XMM4, DWORD PTR [RBP + 4*RAX]
.LBB12_358:                             # %postload1370
                                        #   in Loop: Header=BB12_158 Depth=4
	insertps	XMM1, XMM2, 16  # xmm1 = xmm1[0],xmm2[0],xmm1[2,3]
	insertps	XMM1, XMM3, 32  # xmm1 = xmm1[0,1],xmm3[0],xmm1[3]
	insertps	XMM1, XMM4, 48  # xmm1 = xmm1[0,1,2],xmm4[0]
	subps	XMM0, XMM1
	movaps	XMM2, XMMWORD PTR [RSP + 176] # 16-byte Reload
	movaps	XMM1, XMMWORD PTR [RIP + .LCPI12_3]
	orps	XMM2, XMM1
	pextrq	RAX, XMM2, 1
	mov	QWORD PTR [RSP + 464], RAX # 8-byte Spill
	movq	QWORD PTR [RSP + 456], XMM2 # 8-byte Folded Spill
	movaps	XMM2, XMMWORD PTR [RSP + 192] # 16-byte Reload
	orps	XMM2, XMM1
	pextrq	RAX, XMM2, 1
	mov	QWORD PTR [RSP + 448], RAX # 8-byte Spill
	movq	QWORD PTR [RSP + 440], XMM2 # 8-byte Folded Spill
	test	EBX, EBX
	jns	.LBB12_360
# BB#359:                               # %deload1048
                                        #   in Loop: Header=BB12_158 Depth=4
	mov	RCX, QWORD PTR [RSP + 456] # 8-byte Reload
	mov	RAX, QWORD PTR [RSP + 680] # 8-byte Reload
	movss	XMM1, DWORD PTR [RAX + 4*RCX]
.LBB12_360:                             # %postload1049
                                        #   in Loop: Header=BB12_158 Depth=4
	test	R12D, R12D
	jns	.LBB12_362
# BB#361:                               # %deload1156
                                        #   in Loop: Header=BB12_158 Depth=4
	mov	RCX, QWORD PTR [RSP + 464] # 8-byte Reload
	mov	RAX, QWORD PTR [RSP + 672] # 8-byte Reload
	movss	XMM2, DWORD PTR [RAX + 4*RCX]
.LBB12_362:                             # %postload1157
                                        #   in Loop: Header=BB12_158 Depth=4
	test	R15D, R15D
	jns	.LBB12_364
# BB#363:                               # %deload1264
                                        #   in Loop: Header=BB12_158 Depth=4
	mov	RAX, QWORD PTR [RSP + 440] # 8-byte Reload
	movss	XMM3, DWORD PTR [R13 + 4*RAX]
.LBB12_364:                             # %postload1265
                                        #   in Loop: Header=BB12_158 Depth=4
	test	R14D, R14D
	jns	.LBB12_366
# BB#365:                               # %deload1372
                                        #   in Loop: Header=BB12_158 Depth=4
	mov	RAX, QWORD PTR [RSP + 448] # 8-byte Reload
	movss	XMM4, DWORD PTR [RBP + 4*RAX]
.LBB12_366:                             # %postload1373
                                        #   in Loop: Header=BB12_158 Depth=4
	insertps	XMM1, XMM2, 16  # xmm1 = xmm1[0],xmm2[0],xmm1[2,3]
	insertps	XMM1, XMM3, 32  # xmm1 = xmm1[0,1],xmm3[0],xmm1[3]
	insertps	XMM1, XMM4, 48  # xmm1 = xmm1[0,1,2],xmm4[0]
	test	EBX, EBX
	jns	.LBB12_368
# BB#367:                               # %deload1051
                                        #   in Loop: Header=BB12_158 Depth=4
	mov	RCX, QWORD PTR [RSP + 488] # 8-byte Reload
	mov	RAX, QWORD PTR [RSP + 680] # 8-byte Reload
	movss	XMM2, DWORD PTR [RAX + 4*RCX]
.LBB12_368:                             # %postload1052
                                        #   in Loop: Header=BB12_158 Depth=4
	test	R12D, R12D
	jns	.LBB12_370
# BB#369:                               # %deload1159
                                        #   in Loop: Header=BB12_158 Depth=4
	mov	RCX, QWORD PTR [RSP + 496] # 8-byte Reload
	mov	RAX, QWORD PTR [RSP + 672] # 8-byte Reload
	movss	XMM3, DWORD PTR [RAX + 4*RCX]
.LBB12_370:                             # %postload1160
                                        #   in Loop: Header=BB12_158 Depth=4
	test	R15D, R15D
	jns	.LBB12_372
# BB#371:                               # %deload1267
                                        #   in Loop: Header=BB12_158 Depth=4
	mov	RAX, QWORD PTR [RSP + 472] # 8-byte Reload
	movss	XMM4, DWORD PTR [R13 + 4*RAX]
.LBB12_372:                             # %postload1268
                                        #   in Loop: Header=BB12_158 Depth=4
	test	R14D, R14D
	jns	.LBB12_374
# BB#373:                               # %deload1375
                                        #   in Loop: Header=BB12_158 Depth=4
	mov	RAX, QWORD PTR [RSP + 480] # 8-byte Reload
	movss	XMM5, DWORD PTR [RBP + 4*RAX]
.LBB12_374:                             # %postload1376
                                        #   in Loop: Header=BB12_158 Depth=4
	insertps	XMM2, XMM3, 16  # xmm2 = xmm2[0],xmm3[0],xmm2[2,3]
	insertps	XMM2, XMM4, 32  # xmm2 = xmm2[0,1],xmm4[0],xmm2[3]
	insertps	XMM2, XMM5, 48  # xmm2 = xmm2[0,1,2],xmm5[0]
	subps	XMM1, XMM2
	movaps	XMM2, XMMWORD PTR [RSP + 656] # 16-byte Reload
	mulps	XMM2, XMM1
	movaps	XMMWORD PTR [RSP + 592], XMM2 # 16-byte Spill
	mulps	XMM1, XMM1
	movaps	XMM3, XMMWORD PTR [RSP + 640] # 16-byte Reload
	mulps	XMM3, XMM0
	movaps	XMMWORD PTR [RSP + 496], XMM3 # 16-byte Spill
	mulps	XMM0, XMM0
	addps	XMM0, XMM1
	call	__ocl_svml_h8_sqrtf4
	mulps	XMM0, XMMWORD PTR [RSP + 576] # 16-byte Folded Reload
	movaps	XMM2, XMMWORD PTR [RSP + 592] # 16-byte Reload
	subps	XMM2, XMMWORD PTR [RSP + 496] # 16-byte Folded Reload
	andps	XMM2, XMMWORD PTR [RIP + .LCPI12_7]
	addps	XMM2, XMM0
	movaps	XMMWORD PTR [RSP + 592], XMM2 # 16-byte Spill
	test	EBX, EBX
	jns	.LBB12_376
# BB#375:                               # %deload1054
                                        #   in Loop: Header=BB12_158 Depth=4
	mov	RCX, QWORD PTR [RSP + 608] # 8-byte Reload
	mov	RAX, QWORD PTR [RSP + 864]
	movss	XMM1, DWORD PTR [RAX + 4*RCX]
.LBB12_376:                             # %postload1055
                                        #   in Loop: Header=BB12_158 Depth=4
	test	R12D, R12D
	jns	.LBB12_378
# BB#377:                               # %deload1162
                                        #   in Loop: Header=BB12_158 Depth=4
	mov	RCX, QWORD PTR [RSP + 616] # 8-byte Reload
	mov	RAX, QWORD PTR [RSP + 864]
	movss	XMM0, DWORD PTR [RAX + 4*RCX]
.LBB12_378:                             # %postload1163
                                        #   in Loop: Header=BB12_158 Depth=4
	test	R15D, R15D
	jns	.LBB12_380
# BB#379:                               # %deload1270
                                        #   in Loop: Header=BB12_158 Depth=4
	mov	RCX, QWORD PTR [RSP + 624] # 8-byte Reload
	mov	RAX, QWORD PTR [RSP + 864]
	movss	XMM2, DWORD PTR [RAX + 4*RCX]
.LBB12_380:                             # %postload1271
                                        #   in Loop: Header=BB12_158 Depth=4
	test	R14D, R14D
	jns	.LBB12_382
# BB#381:                               # %deload1378
                                        #   in Loop: Header=BB12_158 Depth=4
	mov	RCX, QWORD PTR [RSP + 632] # 8-byte Reload
	mov	RAX, QWORD PTR [RSP + 864]
	movss	XMM3, DWORD PTR [RAX + 4*RCX]
.LBB12_382:                             # %postload1379
                                        #   in Loop: Header=BB12_158 Depth=4
	insertps	XMM1, XMM0, 16  # xmm1 = xmm1[0],xmm0[0],xmm1[2,3]
	insertps	XMM1, XMM2, 32  # xmm1 = xmm1[0,1],xmm2[0],xmm1[3]
	insertps	XMM1, XMM3, 48  # xmm1 = xmm1[0,1,2],xmm3[0]
	addps	XMM1, XMMWORD PTR [RSP + 592] # 16-byte Folded Reload
	test	EBX, EBX
	jns	.LBB12_384
# BB#383:                               # %deload1057
                                        #   in Loop: Header=BB12_158 Depth=4
	mov	RCX, QWORD PTR [RSP + 608] # 8-byte Reload
	mov	RAX, QWORD PTR [RSP + 864]
	movss	DWORD PTR [RAX + 4*RCX], XMM1
.LBB12_384:                             # %postload1058
                                        #   in Loop: Header=BB12_158 Depth=4
	test	R12D, R12D
	jns	.LBB12_386
# BB#385:                               # %deload1165
                                        #   in Loop: Header=BB12_158 Depth=4
	pshufd	XMM0, XMM1, 1           # xmm0 = xmm1[1,0,0,0]
	mov	RCX, QWORD PTR [RSP + 616] # 8-byte Reload
	mov	RAX, QWORD PTR [RSP + 864]
	movss	DWORD PTR [RAX + 4*RCX], XMM0
.LBB12_386:                             # %postload1166
                                        #   in Loop: Header=BB12_158 Depth=4
	test	R15D, R15D
	jns	.LBB12_388
# BB#387:                               # %deload1273
                                        #   in Loop: Header=BB12_158 Depth=4
	movaps	XMM0, XMM1
	movhlps	XMM0, XMM0              # xmm0 = xmm0[1,1]
	mov	RCX, QWORD PTR [RSP + 624] # 8-byte Reload
	mov	RAX, QWORD PTR [RSP + 864]
	movss	DWORD PTR [RAX + 4*RCX], XMM0
.LBB12_388:                             # %postload1274
                                        #   in Loop: Header=BB12_158 Depth=4
	test	R14D, R14D
	jns	.LBB12_390
# BB#389:                               # %deload1381
                                        #   in Loop: Header=BB12_158 Depth=4
	pshufd	XMM1, XMM1, 3           # xmm1 = xmm1[3,0,0,0]
	mov	RCX, QWORD PTR [RSP + 632] # 8-byte Reload
	mov	RAX, QWORD PTR [RSP + 864]
	movss	DWORD PTR [RAX + 4*RCX], XMM1
.LBB12_390:                             # %postload1382
                                        #   in Loop: Header=BB12_158 Depth=4
	test	EBX, EBX
	jns	.LBB12_392
# BB#391:                               # %deload1059
                                        #   in Loop: Header=BB12_158 Depth=4
	mov	RCX, QWORD PTR [RSP + 512] # 8-byte Reload
	mov	RAX, QWORD PTR [RSP + 680] # 8-byte Reload
	movss	XMM0, DWORD PTR [RAX + 4*RCX]
                                        # kill: XMM0<def> XMM0<kill> XMM0<def>
.LBB12_392:                             # %postload1060
                                        #   in Loop: Header=BB12_158 Depth=4
	test	R12D, R12D
	jns	.LBB12_394
# BB#393:                               # %deload1167
                                        #   in Loop: Header=BB12_158 Depth=4
	mov	RCX, QWORD PTR [RSP + 520] # 8-byte Reload
	mov	RAX, QWORD PTR [RSP + 672] # 8-byte Reload
	movss	XMM1, DWORD PTR [RAX + 4*RCX]
.LBB12_394:                             # %postload1168
                                        #   in Loop: Header=BB12_158 Depth=4
	test	R15D, R15D
	jns	.LBB12_396
# BB#395:                               # %deload1275
                                        #   in Loop: Header=BB12_158 Depth=4
	mov	RAX, QWORD PTR [RSP + 528] # 8-byte Reload
	movss	XMM2, DWORD PTR [R13 + 4*RAX]
.LBB12_396:                             # %postload1276
                                        #   in Loop: Header=BB12_158 Depth=4
	test	R14D, R14D
	jns	.LBB12_398
# BB#397:                               # %deload1383
                                        #   in Loop: Header=BB12_158 Depth=4
	mov	RAX, QWORD PTR [RSP + 536] # 8-byte Reload
	movss	XMM3, DWORD PTR [RBP + 4*RAX]
.LBB12_398:                             # %postload1384
                                        #   in Loop: Header=BB12_158 Depth=4
                                        # kill: XMM0<def> XMM0<kill> XMM0<def>
	insertps	XMM0, XMM1, 16  # xmm0 = xmm0[0],xmm1[0],xmm0[2,3]
	insertps	XMM0, XMM2, 32  # xmm0 = xmm0[0,1],xmm2[0],xmm0[3]
	insertps	XMM0, XMM3, 48  # xmm0 = xmm0[0,1,2],xmm3[0]
	test	EBX, EBX
	jns	.LBB12_400
# BB#399:                               # %deload1062
                                        #   in Loop: Header=BB12_158 Depth=4
	mov	RCX, QWORD PTR [RSP + 408] # 8-byte Reload
	mov	RAX, QWORD PTR [RSP + 680] # 8-byte Reload
	movss	XMM1, DWORD PTR [RAX + 4*RCX]
.LBB12_400:                             # %postload1063
                                        #   in Loop: Header=BB12_158 Depth=4
	test	R12D, R12D
	jns	.LBB12_402
# BB#401:                               # %deload1170
                                        #   in Loop: Header=BB12_158 Depth=4
	mov	RCX, QWORD PTR [RSP + 416] # 8-byte Reload
	mov	RAX, QWORD PTR [RSP + 672] # 8-byte Reload
	movss	XMM2, DWORD PTR [RAX + 4*RCX]
.LBB12_402:                             # %postload1171
                                        #   in Loop: Header=BB12_158 Depth=4
	test	R15D, R15D
	jns	.LBB12_404
# BB#403:                               # %deload1278
                                        #   in Loop: Header=BB12_158 Depth=4
	mov	RAX, QWORD PTR [RSP + 424] # 8-byte Reload
	movss	XMM3, DWORD PTR [R13 + 4*RAX]
.LBB12_404:                             # %postload1279
                                        #   in Loop: Header=BB12_158 Depth=4
	test	R14D, R14D
	jns	.LBB12_406
# BB#405:                               # %deload1386
                                        #   in Loop: Header=BB12_158 Depth=4
	mov	RAX, QWORD PTR [RSP + 432] # 8-byte Reload
	movss	XMM4, DWORD PTR [RBP + 4*RAX]
.LBB12_406:                             # %postload1387
                                        #   in Loop: Header=BB12_158 Depth=4
	insertps	XMM1, XMM2, 16  # xmm1 = xmm1[0],xmm2[0],xmm1[2,3]
	insertps	XMM1, XMM3, 32  # xmm1 = xmm1[0,1],xmm3[0],xmm1[3]
	insertps	XMM1, XMM4, 48  # xmm1 = xmm1[0,1,2],xmm4[0]
	subps	XMM0, XMM1
	test	EBX, EBX
	jns	.LBB12_408
# BB#407:                               # %deload1065
                                        #   in Loop: Header=BB12_158 Depth=4
	mov	RCX, QWORD PTR [RSP + 304] # 8-byte Reload
	mov	RAX, QWORD PTR [RSP + 680] # 8-byte Reload
	movss	XMM1, DWORD PTR [RAX + 4*RCX]
.LBB12_408:                             # %postload1066
                                        #   in Loop: Header=BB12_158 Depth=4
	test	R12D, R12D
	jns	.LBB12_410
# BB#409:                               # %deload1173
                                        #   in Loop: Header=BB12_158 Depth=4
	mov	RCX, QWORD PTR [RSP + 328] # 8-byte Reload
	mov	RAX, QWORD PTR [RSP + 672] # 8-byte Reload
	movss	XMM2, DWORD PTR [RAX + 4*RCX]
.LBB12_410:                             # %postload1174
                                        #   in Loop: Header=BB12_158 Depth=4
	test	R15D, R15D
	jns	.LBB12_412
# BB#411:                               # %deload1281
                                        #   in Loop: Header=BB12_158 Depth=4
	mov	RAX, QWORD PTR [RSP + 288] # 8-byte Reload
	movss	XMM3, DWORD PTR [R13 + 4*RAX]
.LBB12_412:                             # %postload1282
                                        #   in Loop: Header=BB12_158 Depth=4
	test	R14D, R14D
	jns	.LBB12_414
# BB#413:                               # %deload1389
                                        #   in Loop: Header=BB12_158 Depth=4
	mov	RAX, QWORD PTR [RSP + 296] # 8-byte Reload
	movss	XMM4, DWORD PTR [RBP + 4*RAX]
.LBB12_414:                             # %postload1390
                                        #   in Loop: Header=BB12_158 Depth=4
	insertps	XMM1, XMM2, 16  # xmm1 = xmm1[0],xmm2[0],xmm1[2,3]
	insertps	XMM1, XMM3, 32  # xmm1 = xmm1[0,1],xmm3[0],xmm1[3]
	insertps	XMM1, XMM4, 48  # xmm1 = xmm1[0,1,2],xmm4[0]
	test	EBX, EBX
	jns	.LBB12_416
# BB#415:                               # %deload1068
                                        #   in Loop: Header=BB12_158 Depth=4
	mov	RCX, QWORD PTR [RSP + 456] # 8-byte Reload
	mov	RAX, QWORD PTR [RSP + 680] # 8-byte Reload
	movss	XMM2, DWORD PTR [RAX + 4*RCX]
.LBB12_416:                             # %postload1069
                                        #   in Loop: Header=BB12_158 Depth=4
	test	R12D, R12D
	jns	.LBB12_418
# BB#417:                               # %deload1176
                                        #   in Loop: Header=BB12_158 Depth=4
	mov	RCX, QWORD PTR [RSP + 464] # 8-byte Reload
	mov	RAX, QWORD PTR [RSP + 672] # 8-byte Reload
	movss	XMM3, DWORD PTR [RAX + 4*RCX]
.LBB12_418:                             # %postload1177
                                        #   in Loop: Header=BB12_158 Depth=4
	test	R15D, R15D
	jns	.LBB12_420
# BB#419:                               # %deload1284
                                        #   in Loop: Header=BB12_158 Depth=4
	mov	RAX, QWORD PTR [RSP + 440] # 8-byte Reload
	movss	XMM4, DWORD PTR [R13 + 4*RAX]
.LBB12_420:                             # %postload1285
                                        #   in Loop: Header=BB12_158 Depth=4
	test	R14D, R14D
	jns	.LBB12_422
# BB#421:                               # %deload1392
                                        #   in Loop: Header=BB12_158 Depth=4
	mov	RAX, QWORD PTR [RSP + 448] # 8-byte Reload
	movss	XMM5, DWORD PTR [RBP + 4*RAX]
.LBB12_422:                             # %postload1393
                                        #   in Loop: Header=BB12_158 Depth=4
	insertps	XMM2, XMM3, 16  # xmm2 = xmm2[0],xmm3[0],xmm2[2,3]
	insertps	XMM2, XMM4, 32  # xmm2 = xmm2[0,1],xmm4[0],xmm2[3]
	insertps	XMM2, XMM5, 48  # xmm2 = xmm2[0,1,2],xmm5[0]
	subps	XMM1, XMM2
	movaps	XMM2, XMMWORD PTR [RSP + 656] # 16-byte Reload
	mulps	XMM2, XMM1
	movaps	XMMWORD PTR [RSP + 656], XMM2 # 16-byte Spill
	mulps	XMM1, XMM1
	movaps	XMM3, XMMWORD PTR [RSP + 640] # 16-byte Reload
	mulps	XMM3, XMM0
	movaps	XMMWORD PTR [RSP + 640], XMM3 # 16-byte Spill
	mulps	XMM0, XMM0
	addps	XMM0, XMM1
	call	__ocl_svml_h8_sqrtf4
	movaps	XMM1, XMMWORD PTR [RSP + 576] # 16-byte Reload
	mulps	XMM1, XMM0
	movaps	XMM2, XMMWORD PTR [RSP + 656] # 16-byte Reload
	subps	XMM2, XMMWORD PTR [RSP + 640] # 16-byte Folded Reload
	andps	XMM2, XMMWORD PTR [RIP + .LCPI12_7]
	addps	XMM2, XMM1
	movaps	XMMWORD PTR [RSP + 656], XMM2 # 16-byte Spill
	test	EBX, EBX
	jns	.LBB12_424
# BB#423:                               # %deload1071
                                        #   in Loop: Header=BB12_158 Depth=4
	mov	RCX, QWORD PTR [RSP + 608] # 8-byte Reload
	mov	RAX, QWORD PTR [RSP + 864]
	movss	XMM1, DWORD PTR [RAX + 4*RCX]
.LBB12_424:                             # %postload1072
                                        #   in Loop: Header=BB12_158 Depth=4
	test	R12D, R12D
	jns	.LBB12_426
# BB#425:                               # %deload1179
                                        #   in Loop: Header=BB12_158 Depth=4
	mov	RCX, QWORD PTR [RSP + 616] # 8-byte Reload
	mov	RAX, QWORD PTR [RSP + 864]
	movss	XMM0, DWORD PTR [RAX + 4*RCX]
.LBB12_426:                             # %postload1180
                                        #   in Loop: Header=BB12_158 Depth=4
	test	R15D, R15D
	jns	.LBB12_428
# BB#427:                               # %deload1287
                                        #   in Loop: Header=BB12_158 Depth=4
	mov	RCX, QWORD PTR [RSP + 624] # 8-byte Reload
	mov	RAX, QWORD PTR [RSP + 864]
	movss	XMM2, DWORD PTR [RAX + 4*RCX]
.LBB12_428:                             # %postload1288
                                        #   in Loop: Header=BB12_158 Depth=4
	test	R14D, R14D
	jns	.LBB12_430
# BB#429:                               # %deload1395
                                        #   in Loop: Header=BB12_158 Depth=4
	mov	RCX, QWORD PTR [RSP + 632] # 8-byte Reload
	mov	RAX, QWORD PTR [RSP + 864]
	movss	XMM3, DWORD PTR [RAX + 4*RCX]
.LBB12_430:                             # %postload1396
                                        #   in Loop: Header=BB12_158 Depth=4
	insertps	XMM1, XMM0, 16  # xmm1 = xmm1[0],xmm0[0],xmm1[2,3]
	insertps	XMM1, XMM2, 32  # xmm1 = xmm1[0,1],xmm2[0],xmm1[3]
	insertps	XMM1, XMM3, 48  # xmm1 = xmm1[0,1,2],xmm3[0]
	addps	XMM1, XMMWORD PTR [RSP + 656] # 16-byte Folded Reload
	divps	XMM1, XMMWORD PTR [RSP + 560] # 16-byte Folded Reload
	test	EBX, EBX
	jns	.LBB12_432
# BB#431:                               # %deload1074
                                        #   in Loop: Header=BB12_158 Depth=4
	mov	RCX, QWORD PTR [RSP + 608] # 8-byte Reload
	mov	RAX, QWORD PTR [RSP + 864]
	movss	DWORD PTR [RAX + 4*RCX], XMM1
.LBB12_432:                             # %postload1075
                                        #   in Loop: Header=BB12_158 Depth=4
	test	R12D, R12D
	jns	.LBB12_434
# BB#433:                               # %deload1182
                                        #   in Loop: Header=BB12_158 Depth=4
	pshufd	XMM0, XMM1, 1           # xmm0 = xmm1[1,0,0,0]
	mov	RCX, QWORD PTR [RSP + 616] # 8-byte Reload
	mov	RAX, QWORD PTR [RSP + 864]
	movss	DWORD PTR [RAX + 4*RCX], XMM0
.LBB12_434:                             # %postload1183
                                        #   in Loop: Header=BB12_158 Depth=4
	test	R15D, R15D
	jns	.LBB12_436
# BB#435:                               # %deload1290
                                        #   in Loop: Header=BB12_158 Depth=4
	movaps	XMM0, XMM1
	movhlps	XMM0, XMM0              # xmm0 = xmm0[1,1]
	mov	RCX, QWORD PTR [RSP + 624] # 8-byte Reload
	mov	RAX, QWORD PTR [RSP + 864]
	movss	DWORD PTR [RAX + 4*RCX], XMM0
.LBB12_436:                             # %postload1291
                                        #   in Loop: Header=BB12_158 Depth=4
	test	R14D, R14D
	jns	.LBB12_438
# BB#437:                               # %deload1398
                                        #   in Loop: Header=BB12_158 Depth=4
	pshufd	XMM1, XMM1, 3           # xmm1 = xmm1[3,0,0,0]
	mov	RCX, QWORD PTR [RSP + 632] # 8-byte Reload
	mov	RAX, QWORD PTR [RSP + 864]
	movss	DWORD PTR [RAX + 4*RCX], XMM1
.LBB12_438:                             # %postload1399
                                        #   in Loop: Header=BB12_158 Depth=4
	test	EBX, EBX
	jns	.LBB12_440
# BB#439:                               # %deload1076
                                        #   in Loop: Header=BB12_158 Depth=4
	mov	RAX, QWORD PTR [RSP + 992]
	mov	RAX, QWORD PTR [RAX + 56]
.LBB12_440:                             # %postload1077
                                        #   in Loop: Header=BB12_158 Depth=4
	test	R12D, R12D
	jns	.LBB12_442
# BB#441:                               # %deload1184
                                        #   in Loop: Header=BB12_158 Depth=4
	mov	RCX, QWORD PTR [RSP + 992]
	mov	RCX, QWORD PTR [RCX + 56]
.LBB12_442:                             # %postload1185
                                        #   in Loop: Header=BB12_158 Depth=4
	test	R15D, R15D
	jns	.LBB12_444
# BB#443:                               # %deload1292
                                        #   in Loop: Header=BB12_158 Depth=4
	mov	RDX, QWORD PTR [RSP + 992]
	mov	RDX, QWORD PTR [RDX + 56]
.LBB12_444:                             # %postload1293
                                        #   in Loop: Header=BB12_158 Depth=4
	test	R14D, R14D
	jns	.LBB12_446
# BB#445:                               # %deload1400
                                        #   in Loop: Header=BB12_158 Depth=4
	mov	RSI, QWORD PTR [RSP + 992]
	mov	RSI, QWORD PTR [RSI + 56]
.LBB12_446:                             # %postload1401
                                        #   in Loop: Header=BB12_158 Depth=4
	movdqa	XMM1, XMMWORD PTR [RSP + 256] # 16-byte Reload
	movdqa	XMM0, XMMWORD PTR [RIP + .LCPI12_1]
	pand	XMM1, XMM0
	movq	XMM2, RCX
	movq	XMM3, RAX
	punpcklqdq	XMM3, XMM2      # xmm3 = xmm3[0],xmm2[0]
	paddq	XMM1, XMM3
	movdqa	XMMWORD PTR [RSP + 256], XMM1 # 16-byte Spill
	pextrq	RAX, XMM1, 1
	movq	RCX, XMM1
	movd	XMM1, ECX
	pinsrd	XMM1, EAX, 1
	movdqa	XMM2, XMMWORD PTR [RSP + 272] # 16-byte Reload
	pand	XMM2, XMM0
	movq	XMM0, RSI
	movq	XMM3, RDX
	punpcklqdq	XMM3, XMM0      # xmm3 = xmm3[0],xmm0[0]
	paddq	XMM2, XMM3
	movdqa	XMMWORD PTR [RSP + 272], XMM2 # 16-byte Spill
	movq	RAX, XMM2
	pinsrd	XMM1, EAX, 2
	pextrq	RAX, XMM2, 1
	pinsrd	XMM1, EAX, 3
	mov	RAX, QWORD PTR [RSP + 984]
	movd	XMM0, DWORD PTR [RAX + 384]
	pshufd	XMM0, XMM0, 0           # xmm0 = xmm0[0,0,0,0]
	pcmpgtd	XMM0, XMM1
	movdqa	XMM2, XMMWORD PTR [RSP + 544] # 16-byte Reload
	movdqa	XMM3, XMM2
	pand	XMM3, XMM0
	pandn	XMM0, XMM2
	movdqa	XMM2, XMMWORD PTR [RSP + 240] # 16-byte Reload
	por	XMM2, XMM0
	movdqa	XMMWORD PTR [RSP + 240], XMM2 # 16-byte Spill
	por	XMM0, XMM2
	pcmpeqd	XMM2, XMM2
	ptest 	XMM0, XMM2
	movdqa	XMMWORD PTR [RSP + 544], XMM3 # 16-byte Spill
	jae	.LBB12_158
.LBB12_447:                             # %.loopexit
                                        #   in Loop: Header=BB12_156 Depth=3
	mov	RAX, QWORD PTR [RSP + 104] # 8-byte Reload
	cmp	RAX, QWORD PTR [RSP + 1032]
	jb	.LBB12_448
# BB#450:                               # %SyncBB2509
	add	RSP, 776
	pop	RBX
	pop	R12
	pop	R13
	pop	R14
	pop	R15
	pop	RBP
	ret
.Ltmp12:
	.size	__Vectorized_.op_opencl_adt_calc, .Ltmp12-__Vectorized_.op_opencl_adt_calc

	.globl	local.avx256.pcmpeq.d
	.align	16, 0x90
	.type	local.avx256.pcmpeq.d,@function
local.avx256.pcmpeq.d:                  # @local.avx256.pcmpeq.d
.Leh_func_begin13:
# BB#0:                                 # %entry
	pcmpeqd	XMM0, XMM2
	pcmpeqd	XMM1, XMM3
	ret
.Ltmp13:
	.size	local.avx256.pcmpeq.d, .Ltmp13-local.avx256.pcmpeq.d
.Leh_func_end13:

	.globl	local.avx256.pcmpgt.d
	.align	16, 0x90
	.type	local.avx256.pcmpgt.d,@function
local.avx256.pcmpgt.d:                  # @local.avx256.pcmpgt.d
.Leh_func_begin14:
# BB#0:                                 # %entry
	pcmpgtd	XMM0, XMM2
	pcmpgtd	XMM1, XMM3
	ret
.Ltmp14:
	.size	local.avx256.pcmpgt.d, .Ltmp14-local.avx256.pcmpgt.d
.Leh_func_end14:

	.type	opencl_op_opencl_adt_calc_local_ind_arg0_map,@object # @opencl_op_opencl_adt_calc_local_ind_arg0_map
	.local	opencl_op_opencl_adt_calc_local_ind_arg0_map # @opencl_op_opencl_adt_calc_local_ind_arg0_map
	.comm	opencl_op_opencl_adt_calc_local_ind_arg0_map,8,8
	.type	opencl_op_opencl_adt_calc_local_ind_arg0_size,@object # @opencl_op_opencl_adt_calc_local_ind_arg0_size
	.local	opencl_op_opencl_adt_calc_local_ind_arg0_size # @opencl_op_opencl_adt_calc_local_ind_arg0_size
	.comm	opencl_op_opencl_adt_calc_local_ind_arg0_size,4,4
	.type	opencl_op_opencl_adt_calc_local_ind_arg0_s,@object # @opencl_op_opencl_adt_calc_local_ind_arg0_s
	.local	opencl_op_opencl_adt_calc_local_ind_arg0_s # @opencl_op_opencl_adt_calc_local_ind_arg0_s
	.comm	opencl_op_opencl_adt_calc_local_ind_arg0_s,8,8
	.type	opencl_op_opencl_adt_calc_local_nelem,@object # @opencl_op_opencl_adt_calc_local_nelem
	.local	opencl_op_opencl_adt_calc_local_nelem # @opencl_op_opencl_adt_calc_local_nelem
	.comm	opencl_op_opencl_adt_calc_local_nelem,4,4
	.type	opencl_op_opencl_adt_calc_local_offset_b,@object # @opencl_op_opencl_adt_calc_local_offset_b
	.local	opencl_op_opencl_adt_calc_local_offset_b # @opencl_op_opencl_adt_calc_local_offset_b
	.comm	opencl_op_opencl_adt_calc_local_offset_b,4,4
	.section	.eh_frame,"aw",@progbits
.LEH_frame0:
.Lsection_eh_frame0:
.Leh_frame_common0:
.Lset0 = .Leh_frame_common_end0-.Leh_frame_common_begin0 # Length of Common Information Entry
	.long	.Lset0
.Leh_frame_common_begin0:
	.long	0                       # CIE Identifier Tag
	.byte	1                       # DW_CIE_VERSION
	.asciz	 "zR"                   # CIE Augmentation
	.uleb128	1               # CIE Code Alignment Factor
	.sleb128	-8              # CIE Data Alignment Factor
	.byte	16                      # CIE Return Address Column
	.uleb128	1               # Augmentation Size
	.byte	3                       # FDE Encoding = udata4
	.byte	12                      # DW_CFA_def_cfa
	.uleb128	7               # Register
	.uleb128	8               # Offset
	.byte	144                     # DW_CFA_offset + Reg (16)
	.uleb128	1               # Offset
	.align	8
.Leh_frame_common_end0:
.Llocal.avx256.pcmpeq.d.eh = 0

.Llocal.avx256.pcmpgt.d.eh = 0


	.section	.note.GNU-stack,"",@progbits
