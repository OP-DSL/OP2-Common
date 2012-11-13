	.file	"/tmp/b86beb40-6900-4f37-bea6-b47f2fdc4683.TMP"
	.section	.rodata.cst4,"aM",@progbits,4
	.align	4
.LCPI0_0:                               # constant pool float
	.long	1065353216              # float 1.000000e+00
.LCPI0_1:                               # constant pool float
	.long	1056964608              # float 5.000000e-01
	.text
	.globl	op_opencl_res_calc
	.align	16, 0x90
	.type	op_opencl_res_calc,@function
op_opencl_res_calc:                     # @op_opencl_res_calc
# BB#0:                                 # %FirstBB
	push	RBP
	push	R15
	push	R14
	push	R13
	push	R12
	push	RBX
	mov	EAX, DWORD PTR [RSP + 176]
	lea	ECX, DWORD PTR [RAX + 2*RAX]
	mov	DWORD PTR [RSP - 72], ECX # 4-byte Spill
	lea	ECX, DWORD PTR [RAX + 4*RAX]
	mov	DWORD PTR [RSP - 76], ECX # 4-byte Spill
	imul	ECX, EAX, 7
	mov	DWORD PTR [RSP - 80], ECX # 4-byte Spill
	imul	ECX, EAX, 6
	mov	DWORD PTR [RSP - 84], ECX # 4-byte Spill
	movsxd	RCX, ECX
	mov	QWORD PTR [RSP - 8], RCX # 8-byte Spill
	lea	ECX, DWORD PTR [RAX + RAX]
	mov	DWORD PTR [RSP - 88], ECX # 4-byte Spill
	lea	EAX, DWORD PTR [4*RAX]
	mov	DWORD PTR [RSP - 92], EAX # 4-byte Spill
	movsxd	RAX, EAX
	mov	QWORD PTR [RSP - 16], RAX # 8-byte Spill
	movsxd	RAX, ECX
	mov	QWORD PTR [RSP - 24], RAX # 8-byte Spill
	mov	DWORD PTR [RSP - 44], 8 # 4-byte Folded Spill
	movsxd	RAX, DWORD PTR [RSP + 120]
	mov	QWORD PTR [RSP - 32], RAX # 8-byte Spill
	movsxd	RAX, DWORD PTR [RSP + 168]
	mov	QWORD PTR [RSP - 40], RAX # 8-byte Spill
	mov	RCX, QWORD PTR [RSP + 216]
	mov	RSI, QWORD PTR [RSP + 208]
	mov	RDI, QWORD PTR [RSP + 56]
	xor	R8D, R8D
	xor	EAX, EAX
	mov	R9, RAX
	jmp	.LBB0_1
	.align	16, 0x90
.LBB0_43:                               # %thenBB433
                                        #   in Loop: Header=BB0_1 Depth=1
	add	R9, 352
	inc	R8
	cmp	DWORD PTR [RSP - 44], 0 # 4-byte Folded Reload
	je	.LBB0_36
# BB#44:                                # %thenBB433
                                        #   in Loop: Header=BB0_1 Depth=1
	cmp	DWORD PTR [RSP - 44], 2 # 4-byte Folded Reload
	je	.LBB0_21
.LBB0_1:                                # %SyncBB416.outer
                                        # =>This Loop Header: Depth=1
                                        #     Child Loop BB0_41 Depth 2
                                        #     Child Loop BB0_7 Depth 2
                                        #       Child Loop BB0_18 Depth 3
                                        #       Child Loop BB0_15 Depth 3
                                        #       Child Loop BB0_12 Depth 3
                                        #       Child Loop BB0_9 Depth 3
                                        #     Child Loop BB0_2 Depth 2
	mov	R10, R8
	shl	R10, 5
	add	R10, QWORD PTR [RSP + 240]
	jmp	.LBB0_2
	.align	16, 0x90
.LBB0_46:                               # %thenBB426
                                        #   in Loop: Header=BB0_2 Depth=2
	add	R10, 32
	add	R9, 352
	inc	R8
.LBB0_2:                                # %SyncBB416
                                        #   Parent Loop BB0_1 Depth=1
                                        # =>  This Inner Loop Header: Depth=2
	mov	RAX, QWORD PTR [RCX + 80]
	mov	RDX, QWORD PTR [RSP + 224]
	imul	RAX, QWORD PTR [RDX + 8]
	add	RAX, QWORD PTR [RDX]
	cmp	RAX, QWORD PTR [RSP - 40] # 8-byte Folded Reload
	jae	.LBB0_42
# BB#3:                                 #   in Loop: Header=BB0_2 Depth=2
	cmp	QWORD PTR [R10], 0
	jne	.LBB0_5
# BB#4:                                 #   in Loop: Header=BB0_2 Depth=2
	mov	RAX, RDX
	mov	RDX, QWORD PTR [RAX]
	add	RDX, QWORD PTR [RSP - 32] # 8-byte Folded Reload
	mov	R11, QWORD PTR [RCX + 80]
	imul	R11, QWORD PTR [RAX + 8]
	add	R11, RDX
	mov	RAX, QWORD PTR [RSP + 128]
	movsxd	R11, DWORD PTR [RAX + 4*R11]
	mov	RAX, QWORD PTR [RSP + 144]
	mov	EAX, DWORD PTR [RAX + 4*R11]
	mov	DWORD PTR [RSI + 640], EAX
	mov	RAX, QWORD PTR [RSP + 136]
	mov	EAX, DWORD PTR [RAX + 4*R11]
	mov	DWORD PTR [RSI + 768], EAX
	mov	RBX, QWORD PTR [RCX + 56]
	test	RBX, RBX
	mov	R14, RBX
	mov	EAX, 1
	cmove	R14, RAX
	mov	EAX, DWORD PTR [RSI + 640]
	dec	EAX
	movsxd	RAX, EAX
	xor	EDX, EDX
	div	R14
	inc	EAX
	imul	EBX, EAX
	mov	DWORD PTR [RSI + 128], EBX
	mov	RAX, QWORD PTR [RSP + 152]
	mov	EAX, DWORD PTR [RAX + 4*R11]
	mov	DWORD PTR [RSI + 384], EAX
	lea	EAX, DWORD PTR [4*R11]
	movsxd	RAX, EAX
	mov	RDX, QWORD PTR [RSP + 104]
	mov	EBX, DWORD PTR [RDX + 4*RAX]
	mov	DWORD PTR [RSI + 1792], EBX
	lea	R14D, DWORD PTR [4*R11 + 1]
	movsxd	R14, R14D
	movsxd	R15, DWORD PTR [RDX + 4*R14]
	mov	DWORD PTR [RSI + 256], R15D
	lea	R12D, DWORD PTR [4*R11 + 2]
	movsxd	R12, R12D
	mov	R13D, DWORD PTR [RDX + 4*R12]
	mov	DWORD PTR [RSI + 1152], R13D
	lea	R11D, DWORD PTR [4*R11 + 3]
	movsxd	R11, R11D
	mov	EDX, DWORD PTR [RDX + 4*R11]
	mov	DWORD PTR [RSI + 1664], EDX
	mov	RDX, QWORD PTR [RSP + 112]
	movsxd	RAX, DWORD PTR [RDX + 4*RAX]
	mov	RBP, QWORD PTR [RSP + 88]
	lea	RAX, QWORD PTR [RBP + 4*RAX]
	mov	QWORD PTR [RSI + 1408], RAX
	movsxd	RAX, DWORD PTR [RDX + 4*R14]
	add	RAX, QWORD PTR [RSP - 24] # 8-byte Folded Reload
	lea	RAX, QWORD PTR [RBP + 4*RAX]
	mov	QWORD PTR [RSI], RAX
	movsxd	RAX, DWORD PTR [RDX + 4*R12]
	add	RAX, QWORD PTR [RSP - 16] # 8-byte Folded Reload
	lea	RAX, QWORD PTR [RBP + 4*RAX]
	mov	QWORD PTR [RSI + 512], RAX
	movsxd	RAX, DWORD PTR [RDX + 4*R11]
	add	RAX, QWORD PTR [RSP - 8] # 8-byte Folded Reload
	lea	RAX, QWORD PTR [RBP + 4*RAX]
	mov	QWORD PTR [RSI + 1280], RAX
	mov	RAX, QWORD PTR [RSP + 184]
	mov	QWORD PTR [RSI + 1536], RAX
	lea	RDX, QWORD PTR [8*RBX + 15]
	movsxd	RDX, EDX
	mov	R11, RDX
	and	R11, -16
	add	R11, RAX
	mov	QWORD PTR [RSI + 896], R11
	shl	R15, 4
	and	EDX, -16
	add	RDX, R15
	movsxd	RDX, EDX
	lea	R11, QWORD PTR [RAX + RDX]
	mov	QWORD PTR [RSI + 1024], R11
	shl	R13, 2
	add	R13D, 15
	and	R13D, -16
	add	EDX, R13D
	movsxd	RDX, EDX
	add	RDX, RAX
	mov	QWORD PTR [RSI + 1920], RDX
.LBB0_5:                                #   in Loop: Header=BB0_2 Depth=2
	cmp	R8, QWORD PTR [RSP + 256]
	jb	.LBB0_46
# BB#6:                                 # %.SyncBB415_crit_edge
                                        #   in Loop: Header=BB0_1 Depth=1
	mov	R8, -1
	mov	R9, QWORD PTR [RSP + 240]
	.align	16, 0x90
.LBB0_7:                                # %SyncBB415
                                        #   Parent Loop BB0_1 Depth=1
                                        # =>  This Loop Header: Depth=2
                                        #       Child Loop BB0_18 Depth 3
                                        #       Child Loop BB0_15 Depth 3
                                        #       Child Loop BB0_12 Depth 3
                                        #       Child Loop BB0_9 Depth 3
	mov	EAX, DWORD PTR [RSI + 1792]
	add	EAX, EAX
	mov	RDX, QWORD PTR [R9]
	cmp	EDX, EAX
	jge	.LBB0_10
# BB#8:                                 # %SyncBB415.bb.nph57_crit_edge
                                        #   in Loop: Header=BB0_7 Depth=2
	mov	RAX, RDX
	.align	16, 0x90
.LBB0_9:                                # %bb.nph57
                                        #   Parent Loop BB0_1 Depth=1
                                        #     Parent Loop BB0_7 Depth=2
                                        # =>    This Inner Loop Header: Depth=3
	mov	R10D, EDX
	shr	R10D, 31
	add	R10D, EDX
	mov	R11D, R10D
	and	R11D, -2
	mov	EBX, EDX
	sub	EBX, R11D
	sar	R10D
	movsxd	R10, R10D
	mov	R11, QWORD PTR [RSI + 1408]
	mov	R14, QWORD PTR [RSI + 1536]
	mov	R10D, DWORD PTR [R11 + 4*R10]
	lea	R10D, DWORD PTR [RBX + 2*R10]
	movsxd	R10, R10D
	movss	XMM0, DWORD PTR [RDI + 4*R10]
	movsxd	RDX, EDX
	movss	DWORD PTR [R14 + 4*RDX], XMM0
	mov	EDX, EAX
	add	RDX, QWORD PTR [RCX + 56]
	mov	EAX, DWORD PTR [RSI + 1792]
	add	EAX, EAX
	cmp	EDX, EAX
	mov	RAX, RDX
	jl	.LBB0_9
.LBB0_10:                               # %._crit_edge58
                                        #   in Loop: Header=BB0_7 Depth=2
	mov	EAX, DWORD PTR [RSI + 256]
	shl	EAX, 2
	mov	RDX, QWORD PTR [R9]
	cmp	EDX, EAX
	jge	.LBB0_13
# BB#11:                                # %._crit_edge58.bb.nph52_crit_edge
                                        #   in Loop: Header=BB0_7 Depth=2
	mov	RAX, RDX
	.align	16, 0x90
.LBB0_12:                               # %bb.nph52
                                        #   Parent Loop BB0_1 Depth=1
                                        #     Parent Loop BB0_7 Depth=2
                                        # =>    This Inner Loop Header: Depth=3
	mov	R10D, EDX
	sar	R10D, 31
	shr	R10D, 30
	add	R10D, EDX
	mov	R11D, R10D
	and	R11D, -4
	mov	EBX, EDX
	sub	EBX, R11D
	sar	R10D, 2
	movsxd	R10, R10D
	mov	R11, QWORD PTR [RSI]
	mov	R10D, DWORD PTR [R11 + 4*R10]
	lea	R10D, DWORD PTR [RBX + 4*R10]
	movsxd	R10, R10D
	mov	R11, QWORD PTR [RSP + 64]
	movss	XMM0, DWORD PTR [R11 + 4*R10]
	mov	R10, QWORD PTR [RSI + 896]
	movsxd	RDX, EDX
	movss	DWORD PTR [R10 + 4*RDX], XMM0
	mov	EDX, EAX
	add	RDX, QWORD PTR [RCX + 56]
	mov	EAX, DWORD PTR [RSI + 256]
	shl	EAX, 2
	cmp	EDX, EAX
	mov	RAX, RDX
	jl	.LBB0_12
.LBB0_13:                               # %._crit_edge53
                                        #   in Loop: Header=BB0_7 Depth=2
	mov	RAX, QWORD PTR [R9]
	cmp	EAX, DWORD PTR [RSI + 1152]
	jge	.LBB0_16
# BB#14:                                # %._crit_edge53.bb.nph47_crit_edge
                                        #   in Loop: Header=BB0_7 Depth=2
	mov	RDX, RAX
	.align	16, 0x90
.LBB0_15:                               # %bb.nph47
                                        #   Parent Loop BB0_1 Depth=1
                                        #     Parent Loop BB0_7 Depth=2
                                        # =>    This Inner Loop Header: Depth=3
	movsxd	RAX, EAX
	mov	R10, QWORD PTR [RSI + 512]
	mov	R11, QWORD PTR [RSI + 1024]
	movsxd	R10, DWORD PTR [R10 + 4*RAX]
	mov	RBX, QWORD PTR [RSP + 72]
	movss	XMM0, DWORD PTR [RBX + 4*R10]
	movss	DWORD PTR [R11 + 4*RAX], XMM0
	mov	EAX, EDX
	add	RAX, QWORD PTR [RCX + 56]
	cmp	EAX, DWORD PTR [RSI + 1152]
	mov	RDX, RAX
	jl	.LBB0_15
.LBB0_16:                               # %._crit_edge48
                                        #   in Loop: Header=BB0_7 Depth=2
	mov	EAX, DWORD PTR [RSI + 1664]
	shl	EAX, 2
	mov	RDX, QWORD PTR [R9]
	cmp	EDX, EAX
	jge	.LBB0_19
# BB#17:                                # %._crit_edge48.bb.nph42_crit_edge
                                        #   in Loop: Header=BB0_7 Depth=2
	mov	RAX, RDX
	.align	16, 0x90
.LBB0_18:                               # %bb.nph42
                                        #   Parent Loop BB0_1 Depth=1
                                        #     Parent Loop BB0_7 Depth=2
                                        # =>    This Inner Loop Header: Depth=3
	mov	R10, QWORD PTR [RSI + 1920]
	movsxd	RDX, EDX
	mov	DWORD PTR [R10 + 4*RDX], 0
	mov	EDX, EAX
	add	RDX, QWORD PTR [RCX + 56]
	mov	EAX, DWORD PTR [RSI + 1664]
	shl	EAX, 2
	cmp	EDX, EAX
	mov	RAX, RDX
	jl	.LBB0_18
.LBB0_19:                               # %._crit_edge43
                                        #   in Loop: Header=BB0_7 Depth=2
	add	R9, 32
	inc	R8
	cmp	R8, QWORD PTR [RSP + 256]
	jb	.LBB0_7
# BB#20:                                # %._crit_edge43.SyncBB414_crit_edge
                                        #   in Loop: Header=BB0_1 Depth=1
	mov	DWORD PTR [RSP - 44], 2 # 4-byte Folded Spill
	xor	R8D, R8D
	mov	R9, R8
.LBB0_21:                               # %SyncBB414
                                        #   in Loop: Header=BB0_1 Depth=1
	mov	RAX, R8
	shl	RAX, 5
	mov	RDX, QWORD PTR [RSP + 240]
	mov	RAX, QWORD PTR [RDX + RAX]
	mov	RDX, QWORD PTR [RSP + 264]
	mov	QWORD PTR [RDX + R9], RAX
	mov	DWORD PTR [RDX + R9 + 8], EAX
	cmp	EAX, DWORD PTR [RSI + 128]
	jge	.LBB0_39
# BB#22:                                # %bb.nph37
                                        #   in Loop: Header=BB0_1 Depth=1
	mov	RAX, RDX
	lea	RDX, QWORD PTR [RAX + R9 + 128]
	mov	QWORD PTR [RAX + R9 + 16], RDX
	lea	RDX, QWORD PTR [RAX + R9 + 144]
	mov	QWORD PTR [RAX + R9 + 24], RDX
	mov	EDX, DWORD PTR [RAX + R9 + 8]
	mov	RAX, QWORD PTR [RAX + R9]
	mov	R10D, DWORD PTR [RSP - 80] # 4-byte Reload
	mov	DWORD PTR [RSP - 48], R10D # 4-byte Spill
	mov	R10D, DWORD PTR [RSP - 84] # 4-byte Reload
	mov	DWORD PTR [RSP - 52], R10D # 4-byte Spill
	mov	R10D, DWORD PTR [RSP - 76] # 4-byte Reload
	mov	DWORD PTR [RSP - 56], R10D # 4-byte Spill
	mov	R10D, DWORD PTR [RSP - 92] # 4-byte Reload
	mov	DWORD PTR [RSP - 60], R10D # 4-byte Spill
	mov	R10D, DWORD PTR [RSP - 72] # 4-byte Reload
	mov	DWORD PTR [RSP - 64], R10D # 4-byte Spill
	mov	R10D, DWORD PTR [RSP - 88] # 4-byte Reload
	mov	DWORD PTR [RSP - 68], R10D # 4-byte Spill
.LBB0_23:                               #   in Loop: Header=BB0_1 Depth=1
	mov	RBX, QWORD PTR [RSP + 264]
	mov	QWORD PTR [R9 + RBX + 48], RAX
	mov	DWORD PTR [R9 + RBX + 40], R10D
	mov	DWORD PTR [R9 + RBX + 36], R11D
	mov	DWORD PTR [R9 + RBX + 32], EDX
	cmp	EDX, DWORD PTR [RSI + 640]
	jl	.LBB0_25
# BB#24:                                # %..thread_crit_edge
                                        #   in Loop: Header=BB0_1 Depth=1
	mov	EAX, -1
	jmp	.LBB0_29
.LBB0_25:                               #   in Loop: Header=BB0_1 Depth=1
	xorps	XMM0, XMM0
	mov	RAX, RBX
	movaps	XMMWORD PTR [R9 + RAX + 128], XMM0
	movaps	XMMWORD PTR [R9 + RAX + 144], XMM0
	mov	R10D, DWORD PTR [R9 + RAX + 32]
	mov	R11D, DWORD PTR [RSI + 768]
	lea	EDX, DWORD PTR [R11 + R10]
	movsxd	RDX, EDX
	mov	RBX, QWORD PTR [RSP + 96]
	movsx	EDX, WORD PTR [RBX + 2*RDX]
	add	EDX, EDX
	movsxd	RDX, EDX
	mov	R14, QWORD PTR [RSI + 1536]
	movss	XMM0, DWORD PTR [R14 + 4*RDX]
	mov	R15D, DWORD PTR [RSP + 176]
	lea	R15D, DWORD PTR [R10 + R15]
	add	R15D, R11D
	movsxd	R15, R15D
	movsx	R15D, WORD PTR [RBX + 2*R15]
	add	R15D, R15D
	movsxd	R15, R15D
	subss	XMM0, DWORD PTR [R14 + 4*R15]
	mov	R12D, DWORD PTR [RSP - 64] # 4-byte Reload
	lea	R12D, DWORD PTR [R10 + R12]
	add	R12D, R11D
	movsxd	R12, R12D
	movsx	R12D, WORD PTR [RBX + 2*R12]
	shl	R12D, 2
	movsxd	R12, R12D
	mov	R13, QWORD PTR [RSI + 896]
	movss	XMM1, DWORD PTR [R13 + 4*R12 + 8]
	movaps	XMM2, XMM1
	mulss	XMM2, XMM0
	movss	XMM3, DWORD PTR [R14 + 4*RDX + 4]
	subss	XMM3, DWORD PTR [R14 + 4*R15 + 4]
	movss	XMM4, DWORD PTR [R13 + 4*R12 + 4]
	movaps	XMM5, XMM4
	mulss	XMM5, XMM3
	subss	XMM5, XMM2
	movss	XMM2, DWORD PTR [R13 + 4*R12]
	movss	XMM6, DWORD PTR [RIP + .LCPI0_0]
	movaps	XMM7, XMM6
	divss	XMM7, XMM2
	mulss	XMM5, XMM7
	movaps	XMM8, XMM5
	mulss	XMM8, XMM2
	mov	EDX, DWORD PTR [RSP - 68] # 4-byte Reload
	lea	EDX, DWORD PTR [R10 + RDX]
	add	EDX, R11D
	movsxd	RDX, EDX
	movsx	EDX, WORD PTR [RBX + 2*RDX]
	shl	EDX, 2
	movsxd	RDX, EDX
	movss	XMM9, DWORD PTR [R13 + 4*RDX + 8]
	movaps	XMM10, XMM9
	mulss	XMM10, XMM0
	movss	XMM11, DWORD PTR [R13 + 4*RDX + 4]
	movaps	XMM12, XMM11
	mulss	XMM12, XMM3
	subss	XMM12, XMM10
	movss	XMM10, DWORD PTR [R13 + 4*RDX]
	divss	XMM6, XMM10
	mulss	XMM12, XMM6
	movaps	XMM13, XMM12
	mulss	XMM13, XMM10
	addss	XMM13, XMM8
	mulss	XMM13, DWORD PTR [RIP + .LCPI0_1]
	mov	R14D, DWORD PTR [RSP - 56] # 4-byte Reload
	lea	R14D, DWORD PTR [R10 + R14]
	add	R14D, R11D
	movsxd	R14, R14D
	movsx	R14, WORD PTR [RBX + 2*R14]
	add	R10D, DWORD PTR [RSP - 60] # 4-byte Folded Reload
	add	R10D, R11D
	movsxd	R10, R10D
	movsx	R10, WORD PTR [RBX + 2*R10]
	mov	R11, QWORD PTR [RSI + 1024]
	movss	XMM8, DWORD PTR [R11 + 4*R10]
	addss	XMM8, DWORD PTR [R11 + 4*R14]
	mulss	XMM8, DWORD PTR [RIP + .LCPI0_1]
	mov	R10, QWORD PTR [RSP + 200]
	mulss	XMM8, DWORD PTR [R10]
	subss	XMM10, XMM2
	mulss	XMM10, XMM8
	addss	XMM10, XMM13
	mov	R10, QWORD PTR [R9 + RAX + 16]
	movss	XMM2, DWORD PTR [R10]
	addss	XMM2, XMM10
	movss	XMM13, DWORD PTR [R13 + 4*R12 + 12]
	movss	XMM14, DWORD PTR [R13 + 4*RDX + 12]
	mov	R11, QWORD PTR [RSP + 192]
	movss	XMM15, DWORD PTR [R11]
	movss	DWORD PTR [R10], XMM2
	mov	R10, QWORD PTR [R9 + RAX + 24]
	movss	XMM2, DWORD PTR [R10]
	subss	XMM2, XMM10
	movss	DWORD PTR [R10], XMM2
	mulss	XMM9, XMM9
	mulss	XMM11, XMM11
	addss	XMM11, XMM9
	mulss	XMM6, DWORD PTR [RIP + .LCPI0_1]
	mulss	XMM6, XMM11
	subss	XMM14, XMM6
	mulss	XMM14, XMM15
	movaps	XMM2, XMM14
	mulss	XMM2, XMM3
	movss	XMM6, DWORD PTR [R13 + 4*RDX + 4]
	movaps	XMM9, XMM12
	mulss	XMM9, XMM6
	addss	XMM9, XMM2
	movss	XMM2, DWORD PTR [R13 + 4*R12 + 4]
	movaps	XMM10, XMM5
	mulss	XMM10, XMM2
	addss	XMM10, XMM9
	mulss	XMM1, XMM1
	mulss	XMM4, XMM4
	addss	XMM4, XMM1
	mulss	XMM7, DWORD PTR [RIP + .LCPI0_1]
	mulss	XMM7, XMM4
	subss	XMM13, XMM7
	mulss	XMM13, XMM15
	mulss	XMM3, XMM13
	addss	XMM3, XMM10
	mulss	XMM3, DWORD PTR [RIP + .LCPI0_1]
	subss	XMM6, XMM2
	mulss	XMM6, XMM8
	addss	XMM6, XMM3
	mov	R10, QWORD PTR [R9 + RAX + 16]
	movss	XMM1, DWORD PTR [R10 + 4]
	addss	XMM1, XMM6
	movss	DWORD PTR [R10 + 4], XMM1
	mov	R10, QWORD PTR [R9 + RAX + 24]
	movss	XMM1, DWORD PTR [R10 + 4]
	subss	XMM1, XMM6
	movss	DWORD PTR [R10 + 4], XMM1
	movaps	XMM1, XMM14
	mulss	XMM1, XMM0
	movss	XMM2, DWORD PTR [R13 + 4*RDX + 8]
	movaps	XMM3, XMM12
	mulss	XMM3, XMM2
	subss	XMM3, XMM1
	movss	XMM1, DWORD PTR [R13 + 4*R12 + 8]
	movaps	XMM4, XMM5
	mulss	XMM4, XMM1
	addss	XMM4, XMM3
	mulss	XMM0, XMM13
	subss	XMM4, XMM0
	mulss	XMM4, DWORD PTR [RIP + .LCPI0_1]
	subss	XMM2, XMM1
	mulss	XMM2, XMM8
	addss	XMM2, XMM4
	mov	R10, QWORD PTR [R9 + RAX + 16]
	movss	XMM0, DWORD PTR [R10 + 8]
	addss	XMM0, XMM2
	movss	DWORD PTR [R10 + 8], XMM0
	mov	R10, QWORD PTR [R9 + RAX + 24]
	movss	XMM0, DWORD PTR [R10 + 8]
	subss	XMM0, XMM2
	movss	DWORD PTR [R10 + 8], XMM0
	movss	XMM0, DWORD PTR [R13 + 4*R12 + 12]
	addss	XMM13, XMM0
	mulss	XMM13, XMM5
	movss	XMM1, DWORD PTR [R13 + 4*RDX + 12]
	addss	XMM14, XMM1
	mulss	XMM14, XMM12
	addss	XMM14, XMM13
	mulss	XMM14, DWORD PTR [RIP + .LCPI0_1]
	subss	XMM1, XMM0
	mulss	XMM1, XMM8
	addss	XMM1, XMM14
	mov	RDX, QWORD PTR [R9 + RAX + 16]
	movss	XMM0, DWORD PTR [RDX + 12]
	addss	XMM0, XMM1
	movss	DWORD PTR [RDX + 12], XMM0
	mov	RDX, QWORD PTR [R9 + RAX + 24]
	movss	XMM0, DWORD PTR [RDX + 12]
	subss	XMM0, XMM1
	movss	DWORD PTR [RDX + 12], XMM0
	mov	R11D, DWORD PTR [RSI + 768]
	mov	R10D, DWORD PTR [R9 + RAX + 32]
	add	R10D, R11D
	movsxd	RDX, R10D
	mov	R10, QWORD PTR [RSP + 160]
	mov	R10D, DWORD PTR [R10 + 4*RDX]
	mov	DWORD PTR [R9 + RAX + 56], R10D
	test	R10D, R10D
	jns	.LBB0_27
# BB#26:                                # %.phi-split-bb_crit_edge
                                        #   in Loop: Header=BB0_1 Depth=1
	mov	RAX, QWORD PTR [RSP + 264]
	mov	R11D, DWORD PTR [R9 + RAX + 36]
	mov	R10D, DWORD PTR [R9 + RAX + 40]
	jmp	.LBB0_28
.LBB0_27:                               #   in Loop: Header=BB0_1 Depth=1
	mov	R10D, DWORD PTR [R9 + RAX + 32]
	add	R10D, DWORD PTR [RSP - 52] # 4-byte Folded Reload
	add	R10D, R11D
	movsxd	RDX, R10D
	movsx	R10D, WORD PTR [RBX + 2*RDX]
	mov	DWORD PTR [R9 + RAX + 60], R10D
	mov	EDX, DWORD PTR [R9 + RAX + 32]
	add	EDX, DWORD PTR [RSP - 48] # 4-byte Folded Reload
	add	EDX, R11D
	movsxd	RDX, EDX
	movsx	R11D, WORD PTR [RBX + 2*RDX]
	mov	DWORD PTR [R9 + RAX + 64], R11D
.LBB0_28:                               # %phi-split-bb
                                        #   in Loop: Header=BB0_1 Depth=1
	mov	RAX, QWORD PTR [RSP + 264]
	mov	DWORD PTR [R9 + RAX + 72], R10D
	mov	DWORD PTR [R9 + RAX + 68], R11D
	mov	EAX, DWORD PTR [R9 + RAX + 56]
.LBB0_29:                               # %.thread
                                        #   in Loop: Header=BB0_1 Depth=1
	mov	RDX, QWORD PTR [RSP + 264]
	mov	DWORD PTR [R9 + RDX + 84], R11D
	mov	DWORD PTR [R9 + RDX + 80], R10D
	mov	DWORD PTR [R9 + RDX + 76], EAX
	cmp	DWORD PTR [RSI + 384], 0
	jle	.LBB0_37
# BB#30:                                # %bb.nph30
                                        #   in Loop: Header=BB0_1 Depth=1
	mov	RAX, RDX
	mov	R10D, DWORD PTR [R9 + RAX + 80]
	shl	R10D, 2
	mov	DWORD PTR [R9 + RAX + 88], R10D
	mov	R10D, DWORD PTR [R9 + RAX + 84]
	shl	R10D, 2
	mov	DWORD PTR [R9 + RAX + 92], R10D
	xor	R10D, R10D
.LBB0_31:                               #   in Loop: Header=BB0_1 Depth=1
	lea	R11D, DWORD PTR [R10 + 1]
	mov	RAX, QWORD PTR [RSP + 264]
	mov	DWORD PTR [R9 + RAX + 96], R11D
	cmp	DWORD PTR [R9 + RAX + 76], R10D
	jne	.LBB0_33
# BB#32:                                # %.loopexit26
                                        #   in Loop: Header=BB0_1 Depth=1
	mov	RDX, QWORD PTR [R9 + RAX + 16]
	movsxd	R10, DWORD PTR [R9 + RAX + 88]
	mov	R11, QWORD PTR [RSI + 1920]
	movss	XMM0, DWORD PTR [R11 + 4*R10]
	addss	XMM0, DWORD PTR [RDX]
	movss	DWORD PTR [R11 + 4*R10], XMM0
	mov	R10D, DWORD PTR [R9 + RAX + 88]
	or	R10D, 1
	movsxd	RDX, R10D
	mov	R10, QWORD PTR [RSI + 1920]
	movss	XMM0, DWORD PTR [R10 + 4*RDX]
	addss	XMM0, DWORD PTR [R9 + RAX + 132]
	movss	DWORD PTR [R10 + 4*RDX], XMM0
	mov	R10D, DWORD PTR [R9 + RAX + 88]
	or	R10D, 2
	movsxd	RDX, R10D
	mov	R10, QWORD PTR [RSI + 1920]
	movss	XMM0, DWORD PTR [R10 + 4*RDX]
	addss	XMM0, DWORD PTR [R9 + RAX + 136]
	movss	DWORD PTR [R10 + 4*RDX], XMM0
	mov	R10D, DWORD PTR [R9 + RAX + 88]
	or	R10D, 3
	movsxd	RDX, R10D
	mov	R10, QWORD PTR [RSI + 1920]
	movss	XMM0, DWORD PTR [R10 + 4*RDX]
	addss	XMM0, DWORD PTR [R9 + RAX + 140]
	movss	DWORD PTR [R10 + 4*RDX], XMM0
	mov	RDX, QWORD PTR [R9 + RAX + 24]
	movsxd	R10, DWORD PTR [R9 + RAX + 92]
	mov	R11, QWORD PTR [RSI + 1920]
	movss	XMM0, DWORD PTR [R11 + 4*R10]
	addss	XMM0, DWORD PTR [RDX]
	movss	DWORD PTR [R11 + 4*R10], XMM0
	mov	R10D, DWORD PTR [R9 + RAX + 92]
	or	R10D, 1
	movsxd	RDX, R10D
	mov	R10, QWORD PTR [RSI + 1920]
	movss	XMM0, DWORD PTR [R10 + 4*RDX]
	addss	XMM0, DWORD PTR [R9 + RAX + 148]
	movss	DWORD PTR [R10 + 4*RDX], XMM0
	mov	R10D, DWORD PTR [R9 + RAX + 92]
	or	R10D, 2
	movsxd	RDX, R10D
	mov	R10, QWORD PTR [RSI + 1920]
	movss	XMM0, DWORD PTR [R10 + 4*RDX]
	addss	XMM0, DWORD PTR [R9 + RAX + 152]
	movss	DWORD PTR [R10 + 4*RDX], XMM0
	mov	R10D, DWORD PTR [R9 + RAX + 92]
	or	R10D, 3
	movsxd	RDX, R10D
	mov	R10, QWORD PTR [RSI + 1920]
	movss	XMM0, DWORD PTR [R10 + 4*RDX]
	addss	XMM0, DWORD PTR [R9 + RAX + 156]
	movss	DWORD PTR [R10 + 4*RDX], XMM0
.LBB0_33:                               #   in Loop: Header=BB0_1 Depth=1
	cmp	R8, QWORD PTR [RSP + 256]
	jb	.LBB0_35
# BB#34:                                # %.SyncBB_crit_edge
                                        #   in Loop: Header=BB0_1 Depth=1
	mov	DWORD PTR [RSP - 44], 0 # 4-byte Folded Spill
	xor	R8D, R8D
	mov	R9, R8
	jmp	.LBB0_36
.LBB0_35:                               # %thenBB
                                        #   in Loop: Header=BB0_1 Depth=1
	add	R9, 352
	inc	R8
	cmp	DWORD PTR [RSP - 44], 2 # 4-byte Folded Reload
	je	.LBB0_21
.LBB0_36:                               # %SyncBB
                                        #   in Loop: Header=BB0_1 Depth=1
	mov	RAX, QWORD PTR [RSP + 264]
	mov	R10D, DWORD PTR [R9 + RAX + 96]
	cmp	R10D, DWORD PTR [RSI + 384]
	jl	.LBB0_31
.LBB0_37:                               # %._crit_edge31
                                        #   in Loop: Header=BB0_1 Depth=1
	mov	RAX, QWORD PTR [RSP + 264]
	mov	RDX, QWORD PTR [R9 + RAX + 48]
	mov	R10D, 4294967295
	and	RDX, R10
	add	RDX, QWORD PTR [RCX + 56]
	mov	QWORD PTR [R9 + RAX + 104], RDX
	mov	DWORD PTR [R9 + RAX + 112], EDX
	cmp	EDX, DWORD PTR [RSI + 128]
	mov	R10D, DWORD PTR [R9 + RAX + 80]
	mov	R11D, DWORD PTR [R9 + RAX + 84]
	jge	.LBB0_39
# BB#38:                                #   in Loop: Header=BB0_1 Depth=1
	mov	RAX, RDX
	jmp	.LBB0_23
.LBB0_39:                               # %._crit_edge38
                                        #   in Loop: Header=BB0_1 Depth=1
	mov	RAX, R8
	shl	RAX, 5
	mov	RDX, QWORD PTR [RSP + 240]
	mov	RAX, QWORD PTR [RDX + RAX]
	mov	EDX, DWORD PTR [RSI + 1664]
	shl	EDX, 2
	cmp	EAX, EDX
	jge	.LBB0_42
# BB#40:                                # %._crit_edge38.bb.nph_crit_edge
                                        #   in Loop: Header=BB0_1 Depth=1
	mov	RDX, RAX
	.align	16, 0x90
.LBB0_41:                               # %bb.nph
                                        #   Parent Loop BB0_1 Depth=1
                                        # =>  This Inner Loop Header: Depth=2
	mov	R10D, EAX
	sar	R10D, 31
	shr	R10D, 30
	add	R10D, EAX
	mov	R11D, R10D
	and	R11D, -4
	mov	EBX, EAX
	sub	EBX, R11D
	sar	R10D, 2
	movsxd	R10, R10D
	mov	R11, QWORD PTR [RSI + 1280]
	mov	R10D, DWORD PTR [R11 + 4*R10]
	lea	R10D, DWORD PTR [RBX + 4*R10]
	movsxd	R10, R10D
	mov	R11, QWORD PTR [RSP + 80]
	movss	XMM0, DWORD PTR [R11 + 4*R10]
	movsxd	RAX, EAX
	mov	RBX, QWORD PTR [RSI + 1920]
	addss	XMM0, DWORD PTR [RBX + 4*RAX]
	movss	DWORD PTR [R11 + 4*R10], XMM0
	mov	EAX, EDX
	add	RAX, QWORD PTR [RCX + 56]
	mov	EDX, DWORD PTR [RSI + 1664]
	shl	EDX, 2
	cmp	EAX, EDX
	mov	RDX, RAX
	jl	.LBB0_41
.LBB0_42:                               # %.loopexit
                                        #   in Loop: Header=BB0_1 Depth=1
	cmp	R8, QWORD PTR [RSP + 256]
	jb	.LBB0_43
# BB#45:                                # %SyncBB417
	pop	RBX
	pop	R12
	pop	R13
	pop	R14
	pop	R15
	pop	RBP
	ret
.Ltmp0:
	.size	op_opencl_res_calc, .Ltmp0-op_opencl_res_calc

	.section	.rodata.cst4,"aM",@progbits,4
	.align	4
.LCPI1_0:                               # constant pool float
	.long	1065353216              # float 1.000000e+00
.LCPI1_1:                               # constant pool float
	.long	1056964608              # float 5.000000e-01
.LCPI1_2:                               # constant pool float
	.long	0                       # float 0.000000e+00
	.text
	.globl	__Vectorized_.op_opencl_res_calc
	.align	16, 0x90
	.type	__Vectorized_.op_opencl_res_calc,@function
__Vectorized_.op_opencl_res_calc:       # @__Vectorized_.op_opencl_res_calc
# BB#0:                                 # %FirstBB
	push	RBP
	push	R15
	push	R14
	push	R13
	push	R12
	push	RBX
	sub	RSP, 8
	mov	QWORD PTR [RSP - 128], R9 # 8-byte Spill
	mov	QWORD PTR [RSP - 64], R8 # 8-byte Spill
	mov	QWORD PTR [RSP - 56], RCX # 8-byte Spill
	mov	QWORD PTR [RSP], RDX    # 8-byte Spill
	mov	EAX, DWORD PTR [RSP + 136]
	lea	ECX, DWORD PTR [RAX + 2*RAX]
	mov	DWORD PTR [RSP - 96], ECX # 4-byte Spill
	lea	ECX, DWORD PTR [RAX + 4*RAX]
	mov	DWORD PTR [RSP - 100], ECX # 4-byte Spill
	imul	ECX, EAX, 7
	mov	DWORD PTR [RSP - 104], ECX # 4-byte Spill
	imul	ECX, EAX, 6
	mov	DWORD PTR [RSP - 108], ECX # 4-byte Spill
	movsxd	RCX, ECX
	mov	QWORD PTR [RSP - 16], RCX # 8-byte Spill
	lea	ECX, DWORD PTR [RAX + RAX]
	mov	DWORD PTR [RSP - 112], ECX # 4-byte Spill
	lea	EAX, DWORD PTR [4*RAX]
	mov	DWORD PTR [RSP - 116], EAX # 4-byte Spill
	movsxd	RAX, EAX
	mov	QWORD PTR [RSP - 24], RAX # 8-byte Spill
	movsxd	RAX, ECX
	mov	QWORD PTR [RSP - 32], RAX # 8-byte Spill
	mov	DWORD PTR [RSP - 68], 9 # 4-byte Folded Spill
	movsxd	RAX, DWORD PTR [RSP + 80]
	mov	QWORD PTR [RSP - 40], RAX # 8-byte Spill
	movsxd	RAX, DWORD PTR [RSP + 128]
	mov	QWORD PTR [RSP - 48], RAX # 8-byte Spill
	mov	RCX, QWORD PTR [RSP + 176]
	mov	R8, QWORD PTR [RSP + 168]
	mov	QWORD PTR [RSP - 8], 0  # 8-byte Folded Spill
	xor	R9D, R9D
	mov	RBP, QWORD PTR [RSP - 64] # 8-byte Reload
	jmp	.LBB1_1
	.align	16, 0x90
.LBB1_43:                               # %thenBB489
                                        #   in Loop: Header=BB1_1 Depth=1
	add	R9, 352
	inc	QWORD PTR [RSP - 8]     # 8-byte Folded Spill
	cmp	DWORD PTR [RSP - 68], 4 # 4-byte Folded Reload
	je	.LBB1_21
# BB#44:                                # %thenBB489
                                        #   in Loop: Header=BB1_1 Depth=1
	cmp	DWORD PTR [RSP - 68], 5 # 4-byte Folded Reload
	je	.LBB1_36
.LBB1_1:                                # %SyncBB472.outer
                                        # =>This Loop Header: Depth=1
                                        #     Child Loop BB1_41 Depth 2
                                        #     Child Loop BB1_7 Depth 2
                                        #       Child Loop BB1_18 Depth 3
                                        #       Child Loop BB1_15 Depth 3
                                        #       Child Loop BB1_12 Depth 3
                                        #       Child Loop BB1_9 Depth 3
                                        #     Child Loop BB1_2 Depth 2
	mov	R10, QWORD PTR [RSP - 8] # 8-byte Reload
	shl	R10, 5
	add	R10, QWORD PTR [RSP + 200]
	jmp	.LBB1_2
	.align	16, 0x90
.LBB1_46:                               # %thenBB
                                        #   in Loop: Header=BB1_2 Depth=2
	add	R10, 32
	add	R9, 352
	inc	QWORD PTR [RSP - 8]     # 8-byte Folded Spill
.LBB1_2:                                # %SyncBB472
                                        #   Parent Loop BB1_1 Depth=1
                                        # =>  This Inner Loop Header: Depth=2
	mov	RAX, QWORD PTR [RCX + 80]
	mov	RDX, QWORD PTR [RSP + 184]
	imul	RAX, QWORD PTR [RDX + 8]
	add	RAX, QWORD PTR [RDX]
	cmp	RAX, QWORD PTR [RSP - 48] # 8-byte Folded Reload
	jae	.LBB1_42
# BB#3:                                 #   in Loop: Header=BB1_2 Depth=2
	cmp	QWORD PTR [R10], 0
	jne	.LBB1_5
# BB#4:                                 #   in Loop: Header=BB1_2 Depth=2
	mov	RAX, RDX
	mov	RDX, QWORD PTR [RAX]
	add	RDX, QWORD PTR [RSP - 40] # 8-byte Folded Reload
	mov	R11, QWORD PTR [RCX + 80]
	imul	R11, QWORD PTR [RAX + 8]
	add	R11, RDX
	mov	RAX, QWORD PTR [RSP + 88]
	movsxd	R11, DWORD PTR [RAX + 4*R11]
	mov	RAX, QWORD PTR [RSP + 104]
	mov	EAX, DWORD PTR [RAX + 4*R11]
	mov	DWORD PTR [R8 + 640], EAX
	mov	RAX, QWORD PTR [RSP + 96]
	mov	EAX, DWORD PTR [RAX + 4*R11]
	mov	DWORD PTR [R8 + 768], EAX
	mov	RBX, QWORD PTR [RCX + 56]
	test	RBX, RBX
	mov	R14, RBX
	mov	EAX, 1
	cmove	R14, RAX
	mov	EAX, DWORD PTR [R8 + 640]
	dec	EAX
	movsxd	RAX, EAX
	xor	EDX, EDX
	div	R14
	inc	EAX
	imul	EBX, EAX
	mov	DWORD PTR [R8 + 128], EBX
	mov	RAX, QWORD PTR [RSP + 112]
	mov	EAX, DWORD PTR [RAX + 4*R11]
	mov	DWORD PTR [R8 + 384], EAX
	lea	EAX, DWORD PTR [4*R11]
	movsxd	RAX, EAX
	mov	RDX, QWORD PTR [RSP + 64]
	mov	EBX, DWORD PTR [RDX + 4*RAX]
	mov	DWORD PTR [R8 + 1792], EBX
	lea	R14D, DWORD PTR [4*R11 + 1]
	movsxd	R14, R14D
	movsxd	R15, DWORD PTR [RDX + 4*R14]
	mov	DWORD PTR [R8 + 256], R15D
	lea	R12D, DWORD PTR [4*R11 + 2]
	movsxd	R12, R12D
	mov	R13D, DWORD PTR [RDX + 4*R12]
	mov	DWORD PTR [R8 + 1152], R13D
	lea	R11D, DWORD PTR [4*R11 + 3]
	movsxd	R11, R11D
	mov	EDX, DWORD PTR [RDX + 4*R11]
	mov	DWORD PTR [R8 + 1664], EDX
	mov	RDX, QWORD PTR [RSP + 72]
	movsxd	RAX, DWORD PTR [RDX + 4*RAX]
	lea	RAX, QWORD PTR [RBP + 4*RAX]
	mov	QWORD PTR [R8 + 1408], RAX
	movsxd	RAX, DWORD PTR [RDX + 4*R14]
	add	RAX, QWORD PTR [RSP - 32] # 8-byte Folded Reload
	lea	RAX, QWORD PTR [RBP + 4*RAX]
	mov	QWORD PTR [R8], RAX
	movsxd	RAX, DWORD PTR [RDX + 4*R12]
	add	RAX, QWORD PTR [RSP - 24] # 8-byte Folded Reload
	lea	RAX, QWORD PTR [RBP + 4*RAX]
	mov	QWORD PTR [R8 + 512], RAX
	movsxd	RAX, DWORD PTR [RDX + 4*R11]
	add	RAX, QWORD PTR [RSP - 16] # 8-byte Folded Reload
	lea	RAX, QWORD PTR [RBP + 4*RAX]
	mov	QWORD PTR [R8 + 1280], RAX
	mov	RAX, QWORD PTR [RSP + 144]
	mov	QWORD PTR [R8 + 1536], RAX
	lea	RDX, QWORD PTR [8*RBX + 15]
	movsxd	RDX, EDX
	mov	R11, RDX
	and	R11, -16
	add	R11, RAX
	mov	QWORD PTR [R8 + 896], R11
	shl	R15, 4
	and	EDX, -16
	add	RDX, R15
	movsxd	RDX, EDX
	lea	R11, QWORD PTR [RAX + RDX]
	mov	QWORD PTR [R8 + 1024], R11
	shl	R13, 2
	add	R13D, 15
	and	R13D, -16
	add	EDX, R13D
	movsxd	RDX, EDX
	add	RDX, RAX
	mov	QWORD PTR [R8 + 1920], RDX
.LBB1_5:                                #   in Loop: Header=BB1_2 Depth=2
	mov	RAX, QWORD PTR [RSP - 8] # 8-byte Reload
	cmp	RAX, QWORD PTR [RSP + 216]
	jb	.LBB1_46
# BB#6:                                 # %.SyncBB_crit_edge
                                        #   in Loop: Header=BB1_1 Depth=1
	mov	R9, -1
	mov	R10, QWORD PTR [RSP + 200]
	.align	16, 0x90
.LBB1_7:                                # %SyncBB
                                        #   Parent Loop BB1_1 Depth=1
                                        # =>  This Loop Header: Depth=2
                                        #       Child Loop BB1_18 Depth 3
                                        #       Child Loop BB1_15 Depth 3
                                        #       Child Loop BB1_12 Depth 3
                                        #       Child Loop BB1_9 Depth 3
	mov	EAX, DWORD PTR [R8 + 1792]
	add	EAX, EAX
	mov	RDX, QWORD PTR [R10]
	cmp	EDX, EAX
	jge	.LBB1_10
# BB#8:                                 # %SyncBB.bb.nph57_crit_edge
                                        #   in Loop: Header=BB1_7 Depth=2
	mov	RAX, RDX
	.align	16, 0x90
.LBB1_9:                                # %bb.nph57
                                        #   Parent Loop BB1_1 Depth=1
                                        #     Parent Loop BB1_7 Depth=2
                                        # =>    This Inner Loop Header: Depth=3
	mov	R11D, EDX
	shr	R11D, 31
	add	R11D, EDX
	mov	EBX, R11D
	and	EBX, -2
	mov	R14D, EDX
	sub	R14D, EBX
	sar	R11D
	movsxd	R11, R11D
	mov	RBX, QWORD PTR [R8 + 1408]
	mov	R15, QWORD PTR [R8 + 1536]
	mov	R11D, DWORD PTR [RBX + 4*R11]
	lea	R11D, DWORD PTR [R14 + 2*R11]
	movsxd	R11, R11D
	movss	XMM0, DWORD PTR [RDI + 4*R11]
	movsxd	RDX, EDX
	movss	DWORD PTR [R15 + 4*RDX], XMM0
	mov	EDX, EAX
	add	RDX, QWORD PTR [RCX + 56]
	mov	EAX, DWORD PTR [R8 + 1792]
	add	EAX, EAX
	cmp	EDX, EAX
	mov	RAX, RDX
	jl	.LBB1_9
.LBB1_10:                               # %._crit_edge58
                                        #   in Loop: Header=BB1_7 Depth=2
	mov	EAX, DWORD PTR [R8 + 256]
	shl	EAX, 2
	mov	RDX, QWORD PTR [R10]
	cmp	EDX, EAX
	jge	.LBB1_13
# BB#11:                                # %._crit_edge58.bb.nph52_crit_edge
                                        #   in Loop: Header=BB1_7 Depth=2
	mov	RAX, RDX
	.align	16, 0x90
.LBB1_12:                               # %bb.nph52
                                        #   Parent Loop BB1_1 Depth=1
                                        #     Parent Loop BB1_7 Depth=2
                                        # =>    This Inner Loop Header: Depth=3
	mov	R11D, EDX
	sar	R11D, 31
	shr	R11D, 30
	add	R11D, EDX
	mov	EBX, R11D
	and	EBX, -4
	mov	R14D, EDX
	sub	R14D, EBX
	sar	R11D, 2
	movsxd	R11, R11D
	mov	RBX, QWORD PTR [R8]
	mov	R11D, DWORD PTR [RBX + 4*R11]
	lea	R11D, DWORD PTR [R14 + 4*R11]
	movsxd	R11, R11D
	movss	XMM0, DWORD PTR [RSI + 4*R11]
	mov	R11, QWORD PTR [R8 + 896]
	movsxd	RDX, EDX
	movss	DWORD PTR [R11 + 4*RDX], XMM0
	mov	EDX, EAX
	add	RDX, QWORD PTR [RCX + 56]
	mov	EAX, DWORD PTR [R8 + 256]
	shl	EAX, 2
	cmp	EDX, EAX
	mov	RAX, RDX
	jl	.LBB1_12
.LBB1_13:                               # %._crit_edge53
                                        #   in Loop: Header=BB1_7 Depth=2
	mov	RAX, QWORD PTR [R10]
	cmp	EAX, DWORD PTR [R8 + 1152]
	jge	.LBB1_16
# BB#14:                                # %._crit_edge53.bb.nph47_crit_edge
                                        #   in Loop: Header=BB1_7 Depth=2
	mov	RDX, RAX
	.align	16, 0x90
.LBB1_15:                               # %bb.nph47
                                        #   Parent Loop BB1_1 Depth=1
                                        #     Parent Loop BB1_7 Depth=2
                                        # =>    This Inner Loop Header: Depth=3
	movsxd	RAX, EAX
	mov	R11, QWORD PTR [R8 + 512]
	mov	RBX, QWORD PTR [R8 + 1024]
	movsxd	R11, DWORD PTR [R11 + 4*RAX]
	mov	R14, QWORD PTR [RSP]    # 8-byte Reload
	movss	XMM0, DWORD PTR [R14 + 4*R11]
	movss	DWORD PTR [RBX + 4*RAX], XMM0
	mov	EAX, EDX
	add	RAX, QWORD PTR [RCX + 56]
	cmp	EAX, DWORD PTR [R8 + 1152]
	mov	RDX, RAX
	jl	.LBB1_15
.LBB1_16:                               # %._crit_edge48
                                        #   in Loop: Header=BB1_7 Depth=2
	mov	EAX, DWORD PTR [R8 + 1664]
	shl	EAX, 2
	mov	RDX, QWORD PTR [R10]
	cmp	EDX, EAX
	jge	.LBB1_19
# BB#17:                                # %._crit_edge48.bb.nph42_crit_edge
                                        #   in Loop: Header=BB1_7 Depth=2
	mov	RAX, RDX
	.align	16, 0x90
.LBB1_18:                               # %bb.nph42
                                        #   Parent Loop BB1_1 Depth=1
                                        #     Parent Loop BB1_7 Depth=2
                                        # =>    This Inner Loop Header: Depth=3
	mov	R11, QWORD PTR [R8 + 1920]
	movsxd	RDX, EDX
	mov	DWORD PTR [R11 + 4*RDX], 0
	mov	EDX, EAX
	add	RDX, QWORD PTR [RCX + 56]
	mov	EAX, DWORD PTR [R8 + 1664]
	shl	EAX, 2
	cmp	EDX, EAX
	mov	RAX, RDX
	jl	.LBB1_18
.LBB1_19:                               # %._crit_edge43
                                        #   in Loop: Header=BB1_7 Depth=2
	add	R10, 32
	inc	R9
	cmp	R9, QWORD PTR [RSP + 216]
	jb	.LBB1_7
# BB#20:                                # %._crit_edge43.SyncBB470_crit_edge
                                        #   in Loop: Header=BB1_1 Depth=1
	mov	DWORD PTR [RSP - 68], 4 # 4-byte Folded Spill
	xor	R9D, R9D
	mov	QWORD PTR [RSP - 8], R9 # 8-byte Spill
.LBB1_21:                               # %SyncBB470
                                        #   in Loop: Header=BB1_1 Depth=1
	mov	RAX, QWORD PTR [RSP - 8] # 8-byte Reload
	shl	RAX, 5
	mov	RDX, QWORD PTR [RSP + 200]
	mov	RAX, QWORD PTR [RDX + RAX]
	mov	RDX, QWORD PTR [RSP + 224]
	mov	QWORD PTR [R9 + RDX + 160], RAX
	mov	DWORD PTR [R9 + RDX + 168], EAX
	cmp	EAX, DWORD PTR [R8 + 128]
	jge	.LBB1_39
# BB#22:                                # %bb.nph37
                                        #   in Loop: Header=BB1_1 Depth=1
	mov	RAX, RDX
	mov	EDX, DWORD PTR [R9 + RAX + 168]
	mov	RAX, QWORD PTR [R9 + RAX + 160]
	mov	R10D, DWORD PTR [RSP - 104] # 4-byte Reload
	mov	DWORD PTR [RSP - 72], R10D # 4-byte Spill
	mov	R10D, DWORD PTR [RSP - 108] # 4-byte Reload
	mov	DWORD PTR [RSP - 76], R10D # 4-byte Spill
	mov	R10D, DWORD PTR [RSP - 100] # 4-byte Reload
	mov	DWORD PTR [RSP - 80], R10D # 4-byte Spill
	mov	R10D, DWORD PTR [RSP - 116] # 4-byte Reload
	mov	DWORD PTR [RSP - 84], R10D # 4-byte Spill
	mov	R10D, DWORD PTR [RSP - 96] # 4-byte Reload
	mov	DWORD PTR [RSP - 88], R10D # 4-byte Spill
	mov	R10D, DWORD PTR [RSP - 112] # 4-byte Reload
	mov	DWORD PTR [RSP - 92], R10D # 4-byte Spill
.LBB1_23:                               #   in Loop: Header=BB1_1 Depth=1
	mov	RBX, QWORD PTR [RSP + 224]
	mov	QWORD PTR [R9 + RBX + 216], RAX
	mov	DWORD PTR [R9 + RBX + 212], R10D
	mov	DWORD PTR [R9 + RBX + 208], R11D
	mov	DWORD PTR [R9 + RBX + 204], EDX
	movss	DWORD PTR [R9 + RBX + 200], XMM7
	movss	DWORD PTR [R9 + RBX + 196], XMM6
	movss	DWORD PTR [R9 + RBX + 192], XMM5
	movss	DWORD PTR [R9 + RBX + 188], XMM4
	movss	DWORD PTR [R9 + RBX + 184], XMM3
	movss	DWORD PTR [R9 + RBX + 180], XMM2
	movss	DWORD PTR [R9 + RBX + 176], XMM1
	movss	DWORD PTR [R9 + RBX + 172], XMM0
	cmp	EDX, DWORD PTR [R8 + 640]
	jl	.LBB1_25
# BB#24:                                # %..thread_crit_edge
                                        #   in Loop: Header=BB1_1 Depth=1
	mov	EAX, -1
	jmp	.LBB1_29
.LBB1_25:                               #   in Loop: Header=BB1_1 Depth=1
	mov	RAX, RBX
	mov	R10D, DWORD PTR [R9 + RAX + 204]
	mov	R11D, DWORD PTR [R8 + 768]
	lea	EDX, DWORD PTR [R11 + R10]
	movsxd	RDX, EDX
	mov	RBX, QWORD PTR [RSP - 128] # 8-byte Reload
	movsx	EDX, WORD PTR [RBX + 2*RDX]
	add	EDX, EDX
	movsxd	RDX, EDX
	mov	R14, QWORD PTR [R8 + 1536]
	movss	XMM0, DWORD PTR [R14 + 4*RDX]
	mov	R15D, DWORD PTR [RSP + 136]
	lea	R15D, DWORD PTR [R10 + R15]
	add	R15D, R11D
	movsxd	R15, R15D
	movsx	R15D, WORD PTR [RBX + 2*R15]
	add	R15D, R15D
	movsxd	R15, R15D
	subss	XMM0, DWORD PTR [R14 + 4*R15]
	mov	R12D, DWORD PTR [RSP - 88] # 4-byte Reload
	lea	R12D, DWORD PTR [R10 + R12]
	add	R12D, R11D
	movsxd	R12, R12D
	movsx	R12D, WORD PTR [RBX + 2*R12]
	shl	R12D, 2
	movsxd	R12, R12D
	mov	R13, QWORD PTR [R8 + 896]
	movss	XMM1, DWORD PTR [R13 + 4*R12 + 8]
	movaps	XMM2, XMM1
	mulss	XMM2, XMM0
	movss	XMM3, DWORD PTR [R14 + 4*RDX + 4]
	subss	XMM3, DWORD PTR [R14 + 4*R15 + 4]
	movss	XMM4, DWORD PTR [R13 + 4*R12 + 4]
	movaps	XMM5, XMM4
	mulss	XMM5, XMM3
	subss	XMM5, XMM2
	movss	XMM2, DWORD PTR [R13 + 4*R12]
	movss	XMM6, DWORD PTR [RIP + .LCPI1_0]
	movaps	XMM7, XMM6
	divss	XMM7, XMM2
	mulss	XMM5, XMM7
	movaps	XMM8, XMM5
	mulss	XMM8, XMM2
	mov	EDX, DWORD PTR [RSP - 92] # 4-byte Reload
	lea	EDX, DWORD PTR [R10 + RDX]
	add	EDX, R11D
	movsxd	RDX, EDX
	movsx	EDX, WORD PTR [RBX + 2*RDX]
	shl	EDX, 2
	movsxd	RDX, EDX
	movss	XMM9, DWORD PTR [R13 + 4*RDX + 8]
	movaps	XMM10, XMM9
	mulss	XMM10, XMM0
	movss	XMM11, DWORD PTR [R13 + 4*RDX + 4]
	movaps	XMM12, XMM11
	mulss	XMM12, XMM3
	subss	XMM12, XMM10
	movss	XMM10, DWORD PTR [R13 + 4*RDX]
	divss	XMM6, XMM10
	mulss	XMM12, XMM6
	movaps	XMM13, XMM12
	mulss	XMM13, XMM10
	addss	XMM13, XMM8
	mulss	XMM13, DWORD PTR [RIP + .LCPI1_1]
	mov	R14D, DWORD PTR [RSP - 80] # 4-byte Reload
	lea	R14D, DWORD PTR [R10 + R14]
	add	R14D, R11D
	movsxd	R14, R14D
	movsx	R14, WORD PTR [RBX + 2*R14]
	add	R10D, DWORD PTR [RSP - 84] # 4-byte Folded Reload
	add	R10D, R11D
	movsxd	R10, R10D
	movsx	R10, WORD PTR [RBX + 2*R10]
	mov	R11, QWORD PTR [R8 + 1024]
	movss	XMM8, DWORD PTR [R11 + 4*R10]
	addss	XMM8, DWORD PTR [R11 + 4*R14]
	mulss	XMM8, DWORD PTR [RIP + .LCPI1_1]
	mov	R10, QWORD PTR [RSP + 160]
	mulss	XMM8, DWORD PTR [R10]
	subss	XMM10, XMM2
	mulss	XMM10, XMM8
	addss	XMM10, XMM13
	pxor	XMM2, XMM2
	subss	XMM2, XMM10
	addss	XMM10, DWORD PTR [.LCPI1_2]
	movss	XMM13, DWORD PTR [R13 + 4*R12 + 12]
	movss	XMM14, DWORD PTR [R13 + 4*RDX + 12]
	mov	R10, QWORD PTR [RSP + 152]
	movss	XMM15, DWORD PTR [R10]
	movss	DWORD PTR [R9 + RAX + 224], XMM10
	movss	DWORD PTR [R9 + RAX + 228], XMM2
	mulss	XMM9, XMM9
	mulss	XMM11, XMM11
	addss	XMM11, XMM9
	mulss	XMM6, DWORD PTR [RIP + .LCPI1_1]
	mulss	XMM6, XMM11
	subss	XMM14, XMM6
	mulss	XMM14, XMM15
	movaps	XMM2, XMM14
	mulss	XMM2, XMM3
	movss	XMM6, DWORD PTR [R13 + 4*RDX + 4]
	movaps	XMM9, XMM12
	mulss	XMM9, XMM6
	addss	XMM9, XMM2
	movss	XMM2, DWORD PTR [R13 + 4*R12 + 4]
	movaps	XMM10, XMM5
	mulss	XMM10, XMM2
	addss	XMM10, XMM9
	mulss	XMM1, XMM1
	mulss	XMM4, XMM4
	addss	XMM4, XMM1
	mulss	XMM7, DWORD PTR [RIP + .LCPI1_1]
	mulss	XMM7, XMM4
	subss	XMM13, XMM7
	mulss	XMM13, XMM15
	mulss	XMM3, XMM13
	addss	XMM3, XMM10
	mulss	XMM3, DWORD PTR [RIP + .LCPI1_1]
	subss	XMM6, XMM2
	mulss	XMM6, XMM8
	addss	XMM6, XMM3
	pxor	XMM1, XMM1
	subss	XMM1, XMM6
	addss	XMM6, DWORD PTR [.LCPI1_2]
	movss	DWORD PTR [R9 + RAX + 232], XMM6
	movss	DWORD PTR [R9 + RAX + 236], XMM1
	movaps	XMM1, XMM14
	mulss	XMM1, XMM0
	movss	XMM2, DWORD PTR [R13 + 4*RDX + 8]
	movaps	XMM3, XMM12
	mulss	XMM3, XMM2
	subss	XMM3, XMM1
	movss	XMM1, DWORD PTR [R13 + 4*R12 + 8]
	movaps	XMM4, XMM5
	mulss	XMM4, XMM1
	addss	XMM4, XMM3
	mulss	XMM0, XMM13
	subss	XMM4, XMM0
	mulss	XMM4, DWORD PTR [RIP + .LCPI1_1]
	subss	XMM2, XMM1
	mulss	XMM2, XMM8
	addss	XMM2, XMM4
	pxor	XMM0, XMM0
	subss	XMM0, XMM2
	addss	XMM2, DWORD PTR [.LCPI1_2]
	movss	DWORD PTR [R9 + RAX + 240], XMM2
	movss	DWORD PTR [R9 + RAX + 244], XMM0
	movss	XMM0, DWORD PTR [R13 + 4*R12 + 12]
	addss	XMM13, XMM0
	mulss	XMM13, XMM5
	movss	XMM1, DWORD PTR [R13 + 4*RDX + 12]
	addss	XMM14, XMM1
	mulss	XMM14, XMM12
	addss	XMM14, XMM13
	mulss	XMM14, DWORD PTR [RIP + .LCPI1_1]
	subss	XMM1, XMM0
	mulss	XMM1, XMM8
	addss	XMM1, XMM14
	pxor	XMM0, XMM0
	subss	XMM0, XMM1
	addss	XMM1, DWORD PTR [.LCPI1_2]
	movss	DWORD PTR [R9 + RAX + 248], XMM1
	movss	DWORD PTR [R9 + RAX + 252], XMM0
	mov	R11D, DWORD PTR [R8 + 768]
	mov	R10D, DWORD PTR [R9 + RAX + 204]
	add	R10D, R11D
	movsxd	RDX, R10D
	mov	R10, QWORD PTR [RSP + 120]
	mov	R10D, DWORD PTR [R10 + 4*RDX]
	mov	DWORD PTR [R9 + RAX + 256], R10D
	test	R10D, R10D
	jns	.LBB1_27
# BB#26:                                # %.phi-split-bb_crit_edge
                                        #   in Loop: Header=BB1_1 Depth=1
	mov	RAX, QWORD PTR [RSP + 224]
	mov	R11D, DWORD PTR [R9 + RAX + 208]
	mov	R10D, DWORD PTR [R9 + RAX + 212]
	jmp	.LBB1_28
.LBB1_27:                               #   in Loop: Header=BB1_1 Depth=1
	mov	R10D, DWORD PTR [R9 + RAX + 204]
	add	R10D, DWORD PTR [RSP - 76] # 4-byte Folded Reload
	add	R10D, R11D
	movsxd	RDX, R10D
	movsx	R10D, WORD PTR [RBX + 2*RDX]
	mov	DWORD PTR [R9 + RAX + 260], R10D
	mov	EDX, DWORD PTR [R9 + RAX + 204]
	add	EDX, DWORD PTR [RSP - 72] # 4-byte Folded Reload
	add	EDX, R11D
	movsxd	RDX, EDX
	movsx	R11D, WORD PTR [RBX + 2*RDX]
	mov	DWORD PTR [R9 + RAX + 264], R11D
.LBB1_28:                               # %phi-split-bb
                                        #   in Loop: Header=BB1_1 Depth=1
	mov	RDX, QWORD PTR [RSP + 224]
	mov	DWORD PTR [R9 + RDX + 272], R10D
	mov	DWORD PTR [R9 + RDX + 268], R11D
	mov	EAX, DWORD PTR [R9 + RDX + 256]
	movss	XMM0, DWORD PTR [R9 + RDX + 252]
	movss	XMM4, DWORD PTR [R9 + RDX + 248]
	movss	XMM1, DWORD PTR [R9 + RDX + 244]
	movss	XMM5, DWORD PTR [R9 + RDX + 240]
	movss	XMM2, DWORD PTR [R9 + RDX + 236]
	movss	XMM6, DWORD PTR [R9 + RDX + 232]
	movss	XMM7, DWORD PTR [R9 + RDX + 224]
	movss	XMM3, DWORD PTR [R9 + RDX + 228]
.LBB1_29:                               # %.thread
                                        #   in Loop: Header=BB1_1 Depth=1
	mov	RDX, QWORD PTR [RSP + 224]
	mov	DWORD PTR [R9 + RDX + 316], R11D
	mov	DWORD PTR [R9 + RDX + 312], R10D
	mov	DWORD PTR [R9 + RDX + 308], EAX
	movss	DWORD PTR [R9 + RDX + 304], XMM7
	movss	DWORD PTR [R9 + RDX + 300], XMM6
	movss	DWORD PTR [R9 + RDX + 296], XMM5
	movss	DWORD PTR [R9 + RDX + 292], XMM4
	movss	DWORD PTR [R9 + RDX + 288], XMM3
	movss	DWORD PTR [R9 + RDX + 284], XMM2
	movss	DWORD PTR [R9 + RDX + 280], XMM1
	movss	DWORD PTR [R9 + RDX + 276], XMM0
	cmp	DWORD PTR [R8 + 384], 0
	jle	.LBB1_37
# BB#30:                                # %bb.nph30
                                        #   in Loop: Header=BB1_1 Depth=1
	mov	RAX, RDX
	mov	R10D, DWORD PTR [R9 + RAX + 312]
	shl	R10D, 2
	mov	DWORD PTR [R9 + RAX + 320], R10D
	mov	R10D, DWORD PTR [R9 + RAX + 316]
	shl	R10D, 2
	mov	DWORD PTR [R9 + RAX + 324], R10D
	xor	R10D, R10D
.LBB1_31:                               #   in Loop: Header=BB1_1 Depth=1
	lea	R11D, DWORD PTR [R10 + 1]
	mov	RAX, QWORD PTR [RSP + 224]
	mov	DWORD PTR [R9 + RAX + 328], R11D
	cmp	DWORD PTR [R9 + RAX + 308], R10D
	jne	.LBB1_33
# BB#32:                                # %.loopexit26
                                        #   in Loop: Header=BB1_1 Depth=1
	movsxd	RDX, DWORD PTR [R9 + RAX + 320]
	mov	R10, QWORD PTR [R8 + 1920]
	movss	XMM0, DWORD PTR [R10 + 4*RDX]
	addss	XMM0, DWORD PTR [R9 + RAX + 304]
	movss	DWORD PTR [R10 + 4*RDX], XMM0
	mov	R10D, DWORD PTR [R9 + RAX + 320]
	or	R10D, 1
	movsxd	RDX, R10D
	mov	R10, QWORD PTR [R8 + 1920]
	movss	XMM0, DWORD PTR [R10 + 4*RDX]
	addss	XMM0, DWORD PTR [R9 + RAX + 300]
	movss	DWORD PTR [R10 + 4*RDX], XMM0
	mov	R10D, DWORD PTR [R9 + RAX + 320]
	or	R10D, 2
	movsxd	RDX, R10D
	mov	R10, QWORD PTR [R8 + 1920]
	movss	XMM0, DWORD PTR [R10 + 4*RDX]
	addss	XMM0, DWORD PTR [R9 + RAX + 296]
	movss	DWORD PTR [R10 + 4*RDX], XMM0
	mov	R10D, DWORD PTR [R9 + RAX + 320]
	or	R10D, 3
	movsxd	RDX, R10D
	mov	R10, QWORD PTR [R8 + 1920]
	movss	XMM0, DWORD PTR [R10 + 4*RDX]
	addss	XMM0, DWORD PTR [R9 + RAX + 292]
	movss	DWORD PTR [R10 + 4*RDX], XMM0
	movsxd	RDX, DWORD PTR [R9 + RAX + 324]
	mov	R10, QWORD PTR [R8 + 1920]
	movss	XMM0, DWORD PTR [R10 + 4*RDX]
	addss	XMM0, DWORD PTR [R9 + RAX + 288]
	movss	DWORD PTR [R10 + 4*RDX], XMM0
	mov	R10D, DWORD PTR [R9 + RAX + 324]
	or	R10D, 1
	movsxd	RDX, R10D
	mov	R10, QWORD PTR [R8 + 1920]
	movss	XMM0, DWORD PTR [R10 + 4*RDX]
	addss	XMM0, DWORD PTR [R9 + RAX + 284]
	movss	DWORD PTR [R10 + 4*RDX], XMM0
	mov	R10D, DWORD PTR [R9 + RAX + 324]
	or	R10D, 2
	movsxd	RDX, R10D
	mov	R10, QWORD PTR [R8 + 1920]
	movss	XMM0, DWORD PTR [R10 + 4*RDX]
	addss	XMM0, DWORD PTR [R9 + RAX + 280]
	movss	DWORD PTR [R10 + 4*RDX], XMM0
	mov	R10D, DWORD PTR [R9 + RAX + 324]
	or	R10D, 3
	movsxd	RDX, R10D
	mov	R10, QWORD PTR [R8 + 1920]
	movss	XMM0, DWORD PTR [R10 + 4*RDX]
	addss	XMM0, DWORD PTR [R9 + RAX + 276]
	movss	DWORD PTR [R10 + 4*RDX], XMM0
.LBB1_33:                               #   in Loop: Header=BB1_1 Depth=1
	mov	RAX, QWORD PTR [RSP - 8] # 8-byte Reload
	cmp	RAX, QWORD PTR [RSP + 216]
	jb	.LBB1_35
# BB#34:                                # %.SyncBB471_crit_edge
                                        #   in Loop: Header=BB1_1 Depth=1
	mov	DWORD PTR [RSP - 68], 5 # 4-byte Folded Spill
	xor	EAX, EAX
	mov	QWORD PTR [RSP - 8], RAX # 8-byte Spill
	mov	R9, RAX
	jmp	.LBB1_36
.LBB1_35:                               # %thenBB482
                                        #   in Loop: Header=BB1_1 Depth=1
	add	R9, 352
	inc	QWORD PTR [RSP - 8]     # 8-byte Folded Spill
	cmp	DWORD PTR [RSP - 68], 4 # 4-byte Folded Reload
	je	.LBB1_21
.LBB1_36:                               # %SyncBB471
                                        #   in Loop: Header=BB1_1 Depth=1
	mov	RAX, QWORD PTR [RSP + 224]
	mov	R10D, DWORD PTR [R9 + RAX + 328]
	cmp	R10D, DWORD PTR [R8 + 384]
	jl	.LBB1_31
.LBB1_37:                               # %._crit_edge31
                                        #   in Loop: Header=BB1_1 Depth=1
	mov	RAX, QWORD PTR [RSP + 224]
	mov	RDX, QWORD PTR [R9 + RAX + 216]
	mov	R10D, 4294967295
	and	RDX, R10
	add	RDX, QWORD PTR [RCX + 56]
	mov	QWORD PTR [R9 + RAX + 336], RDX
	mov	DWORD PTR [R9 + RAX + 344], EDX
	cmp	EDX, DWORD PTR [R8 + 128]
	mov	R11D, DWORD PTR [R9 + RAX + 316]
	mov	R10D, DWORD PTR [R9 + RAX + 312]
	movss	XMM7, DWORD PTR [R9 + RAX + 304]
	movss	XMM6, DWORD PTR [R9 + RAX + 300]
	movss	XMM5, DWORD PTR [R9 + RAX + 296]
	movss	XMM4, DWORD PTR [R9 + RAX + 292]
	movss	XMM3, DWORD PTR [R9 + RAX + 288]
	movss	XMM2, DWORD PTR [R9 + RAX + 284]
	movss	XMM0, DWORD PTR [R9 + RAX + 276]
	movss	XMM1, DWORD PTR [R9 + RAX + 280]
	jge	.LBB1_39
# BB#38:                                #   in Loop: Header=BB1_1 Depth=1
	mov	RAX, RDX
	jmp	.LBB1_23
.LBB1_39:                               # %._crit_edge38
                                        #   in Loop: Header=BB1_1 Depth=1
	mov	RAX, QWORD PTR [RSP - 8] # 8-byte Reload
	shl	RAX, 5
	mov	R10, QWORD PTR [RSP + 200]
	mov	RAX, QWORD PTR [R10 + RAX]
	mov	EDX, DWORD PTR [R8 + 1664]
	shl	EDX, 2
	cmp	EAX, EDX
	jge	.LBB1_42
# BB#40:                                # %._crit_edge38.bb.nph_crit_edge
                                        #   in Loop: Header=BB1_1 Depth=1
	mov	RDX, RAX
	.align	16, 0x90
.LBB1_41:                               # %bb.nph
                                        #   Parent Loop BB1_1 Depth=1
                                        # =>  This Inner Loop Header: Depth=2
	mov	R10D, EAX
	sar	R10D, 31
	shr	R10D, 30
	add	R10D, EAX
	mov	R11D, R10D
	and	R11D, -4
	mov	EBX, EAX
	sub	EBX, R11D
	sar	R10D, 2
	movsxd	R10, R10D
	mov	R11, QWORD PTR [R8 + 1280]
	mov	R10D, DWORD PTR [R11 + 4*R10]
	lea	R10D, DWORD PTR [RBX + 4*R10]
	movsxd	R10, R10D
	mov	R11, QWORD PTR [RSP - 56] # 8-byte Reload
	movss	XMM0, DWORD PTR [R11 + 4*R10]
	movsxd	RAX, EAX
	mov	RBX, QWORD PTR [R8 + 1920]
	addss	XMM0, DWORD PTR [RBX + 4*RAX]
	movss	DWORD PTR [R11 + 4*R10], XMM0
	mov	EAX, EDX
	add	RAX, QWORD PTR [RCX + 56]
	mov	EDX, DWORD PTR [R8 + 1664]
	shl	EDX, 2
	cmp	EAX, EDX
	mov	RDX, RAX
	jl	.LBB1_41
.LBB1_42:                               # %.loopexit
                                        #   in Loop: Header=BB1_1 Depth=1
	mov	RAX, QWORD PTR [RSP - 8] # 8-byte Reload
	cmp	RAX, QWORD PTR [RSP + 216]
	jb	.LBB1_43
# BB#45:                                # %SyncBB473
	add	RSP, 8
	pop	RBX
	pop	R12
	pop	R13
	pop	R14
	pop	R15
	pop	RBP
	ret
.Ltmp1:
	.size	__Vectorized_.op_opencl_res_calc, .Ltmp1-__Vectorized_.op_opencl_res_calc

	.type	opencl_op_opencl_res_calc_local_ind_arg0_map,@object # @opencl_op_opencl_res_calc_local_ind_arg0_map
	.local	opencl_op_opencl_res_calc_local_ind_arg0_map # @opencl_op_opencl_res_calc_local_ind_arg0_map
	.comm	opencl_op_opencl_res_calc_local_ind_arg0_map,8,8
	.type	opencl_op_opencl_res_calc_local_ind_arg0_size,@object # @opencl_op_opencl_res_calc_local_ind_arg0_size
	.local	opencl_op_opencl_res_calc_local_ind_arg0_size # @opencl_op_opencl_res_calc_local_ind_arg0_size
	.comm	opencl_op_opencl_res_calc_local_ind_arg0_size,4,4
	.type	opencl_op_opencl_res_calc_local_ind_arg1_map,@object # @opencl_op_opencl_res_calc_local_ind_arg1_map
	.local	opencl_op_opencl_res_calc_local_ind_arg1_map # @opencl_op_opencl_res_calc_local_ind_arg1_map
	.comm	opencl_op_opencl_res_calc_local_ind_arg1_map,8,8
	.type	opencl_op_opencl_res_calc_local_ind_arg1_size,@object # @opencl_op_opencl_res_calc_local_ind_arg1_size
	.local	opencl_op_opencl_res_calc_local_ind_arg1_size # @opencl_op_opencl_res_calc_local_ind_arg1_size
	.comm	opencl_op_opencl_res_calc_local_ind_arg1_size,4,4
	.type	opencl_op_opencl_res_calc_local_ind_arg2_map,@object # @opencl_op_opencl_res_calc_local_ind_arg2_map
	.local	opencl_op_opencl_res_calc_local_ind_arg2_map # @opencl_op_opencl_res_calc_local_ind_arg2_map
	.comm	opencl_op_opencl_res_calc_local_ind_arg2_map,8,8
	.type	opencl_op_opencl_res_calc_local_ind_arg2_size,@object # @opencl_op_opencl_res_calc_local_ind_arg2_size
	.local	opencl_op_opencl_res_calc_local_ind_arg2_size # @opencl_op_opencl_res_calc_local_ind_arg2_size
	.comm	opencl_op_opencl_res_calc_local_ind_arg2_size,4,4
	.type	opencl_op_opencl_res_calc_local_ind_arg3_map,@object # @opencl_op_opencl_res_calc_local_ind_arg3_map
	.local	opencl_op_opencl_res_calc_local_ind_arg3_map # @opencl_op_opencl_res_calc_local_ind_arg3_map
	.comm	opencl_op_opencl_res_calc_local_ind_arg3_map,8,8
	.type	opencl_op_opencl_res_calc_local_ind_arg3_size,@object # @opencl_op_opencl_res_calc_local_ind_arg3_size
	.local	opencl_op_opencl_res_calc_local_ind_arg3_size # @opencl_op_opencl_res_calc_local_ind_arg3_size
	.comm	opencl_op_opencl_res_calc_local_ind_arg3_size,4,4
	.type	opencl_op_opencl_res_calc_local_ind_arg0_s,@object # @opencl_op_opencl_res_calc_local_ind_arg0_s
	.local	opencl_op_opencl_res_calc_local_ind_arg0_s # @opencl_op_opencl_res_calc_local_ind_arg0_s
	.comm	opencl_op_opencl_res_calc_local_ind_arg0_s,8,8
	.type	opencl_op_opencl_res_calc_local_ind_arg1_s,@object # @opencl_op_opencl_res_calc_local_ind_arg1_s
	.local	opencl_op_opencl_res_calc_local_ind_arg1_s # @opencl_op_opencl_res_calc_local_ind_arg1_s
	.comm	opencl_op_opencl_res_calc_local_ind_arg1_s,8,8
	.type	opencl_op_opencl_res_calc_local_ind_arg2_s,@object # @opencl_op_opencl_res_calc_local_ind_arg2_s
	.local	opencl_op_opencl_res_calc_local_ind_arg2_s # @opencl_op_opencl_res_calc_local_ind_arg2_s
	.comm	opencl_op_opencl_res_calc_local_ind_arg2_s,8,8
	.type	opencl_op_opencl_res_calc_local_ind_arg3_s,@object # @opencl_op_opencl_res_calc_local_ind_arg3_s
	.local	opencl_op_opencl_res_calc_local_ind_arg3_s # @opencl_op_opencl_res_calc_local_ind_arg3_s
	.comm	opencl_op_opencl_res_calc_local_ind_arg3_s,8,8
	.type	opencl_op_opencl_res_calc_local_nelems2,@object # @opencl_op_opencl_res_calc_local_nelems2
	.local	opencl_op_opencl_res_calc_local_nelems2 # @opencl_op_opencl_res_calc_local_nelems2
	.comm	opencl_op_opencl_res_calc_local_nelems2,4,4
	.type	opencl_op_opencl_res_calc_local_ncolor,@object # @opencl_op_opencl_res_calc_local_ncolor
	.local	opencl_op_opencl_res_calc_local_ncolor # @opencl_op_opencl_res_calc_local_ncolor
	.comm	opencl_op_opencl_res_calc_local_ncolor,4,4
	.type	opencl_op_opencl_res_calc_local_nelem,@object # @opencl_op_opencl_res_calc_local_nelem
	.local	opencl_op_opencl_res_calc_local_nelem # @opencl_op_opencl_res_calc_local_nelem
	.comm	opencl_op_opencl_res_calc_local_nelem,4,4
	.type	opencl_op_opencl_res_calc_local_offset_b,@object # @opencl_op_opencl_res_calc_local_offset_b
	.local	opencl_op_opencl_res_calc_local_offset_b # @opencl_op_opencl_res_calc_local_offset_b
	.comm	opencl_op_opencl_res_calc_local_offset_b,4,4

	.section	.note.GNU-stack,"",@progbits
