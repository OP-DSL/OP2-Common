	.file	"/tmp/541c7788-64dd-4df4-a65c-f89b6e9e16be.TMP"
	.section	.rodata.cst4,"aM",@progbits,4
	.align	4
.LCPI0_0:                               # constant pool float
	.long	1065353216              # float 1.000000e+00
.LCPI0_1:                               # constant pool float
	.long	3204448256              # float -5.000000e-01
.LCPI0_2:                               # constant pool float
	.long	1056964608              # float 5.000000e-01
	.section	.rodata.cst16,"aM",@progbits,16
	.align	16
.LCPI0_3:                               # constant pool <4 x i32>
	.zero	16
	.text
	.globl	op_opencl_bres_calc
	.align	16, 0x90
	.type	op_opencl_bres_calc,@function
op_opencl_bres_calc:                    # @op_opencl_bres_calc
# BB#0:                                 # %FirstBB
	push	RBP
	push	R15
	push	R14
	push	R13
	push	R12
	push	RBX
	mov	EAX, DWORD PTR [RSP + 184]
	lea	ECX, DWORD PTR [RAX + 2*RAX]
	mov	DWORD PTR [RSP - 60], ECX # 4-byte Spill
	lea	EDX, DWORD PTR [RAX + RAX]
	mov	DWORD PTR [RSP - 64], EDX # 4-byte Spill
	lea	EAX, DWORD PTR [4*RAX]
	mov	DWORD PTR [RSP - 68], EAX # 4-byte Spill
	movsxd	RAX, EAX
	mov	QWORD PTR [RSP - 8], RAX # 8-byte Spill
	movsxd	RAX, ECX
	mov	QWORD PTR [RSP - 16], RAX # 8-byte Spill
	movsxd	RAX, EDX
	mov	QWORD PTR [RSP - 24], RAX # 8-byte Spill
	mov	DWORD PTR [RSP - 44], 9 # 4-byte Folded Spill
	movsxd	RAX, DWORD PTR [RSP + 128]
	mov	QWORD PTR [RSP - 32], RAX # 8-byte Spill
	movsxd	RAX, DWORD PTR [RSP + 176]
	mov	QWORD PTR [RSP - 40], RAX # 8-byte Spill
	mov	RCX, QWORD PTR [RSP + 280]
	mov	RSI, QWORD PTR [RSP + 232]
	mov	RDI, QWORD PTR [RSP + 224]
	movss	XMM0, DWORD PTR [RIP + .LCPI0_0]
	movss	XMM1, DWORD PTR [RIP + .LCPI0_2]
	xor	R8D, R8D
	xor	EAX, EAX
	mov	R9, RAX
	jmp	.LBB0_1
	.align	16, 0x90
.LBB0_46:                               # %thenBB462
                                        #   in Loop: Header=BB0_1 Depth=1
	add	R9, 384
	inc	R8
	cmp	DWORD PTR [RSP - 44], 1 # 4-byte Folded Reload
	je	.LBB0_21
# BB#47:                                # %thenBB462
                                        #   in Loop: Header=BB0_1 Depth=1
	cmp	DWORD PTR [RSP - 44], 2 # 4-byte Folded Reload
	je	.LBB0_39
.LBB0_1:                                # %SyncBB445.outer
                                        # =>This Loop Header: Depth=1
                                        #     Child Loop BB0_44 Depth 2
                                        #     Child Loop BB0_7 Depth 2
                                        #       Child Loop BB0_18 Depth 3
                                        #       Child Loop BB0_15 Depth 3
                                        #       Child Loop BB0_12 Depth 3
                                        #       Child Loop BB0_9 Depth 3
                                        #     Child Loop BB0_2 Depth 2
	mov	R10, R8
	shl	R10, 5
	add	R10, QWORD PTR [RSP + 256]
	jmp	.LBB0_2
	.align	16, 0x90
.LBB0_49:                               # %thenBB
                                        #   in Loop: Header=BB0_2 Depth=2
	add	R10, 32
	add	R9, 384
	inc	R8
.LBB0_2:                                # %SyncBB445
                                        #   Parent Loop BB0_1 Depth=1
                                        # =>  This Inner Loop Header: Depth=2
	mov	RAX, QWORD PTR [RSI + 80]
	mov	RDX, QWORD PTR [RSP + 240]
	imul	RAX, QWORD PTR [RDX + 8]
	add	RAX, QWORD PTR [RDX]
	cmp	RAX, QWORD PTR [RSP - 40] # 8-byte Folded Reload
	jae	.LBB0_45
# BB#3:                                 #   in Loop: Header=BB0_2 Depth=2
	cmp	QWORD PTR [R10], 0
	jne	.LBB0_5
# BB#4:                                 #   in Loop: Header=BB0_2 Depth=2
	mov	RAX, RDX
	mov	RDX, QWORD PTR [RAX]
	add	RDX, QWORD PTR [RSP - 32] # 8-byte Folded Reload
	mov	R11, QWORD PTR [RSI + 80]
	imul	R11, QWORD PTR [RAX + 8]
	add	R11, RDX
	mov	RAX, QWORD PTR [RSP + 136]
	movsxd	R11, DWORD PTR [RAX + 4*R11]
	mov	RAX, QWORD PTR [RSP + 152]
	mov	EAX, DWORD PTR [RAX + 4*R11]
	mov	DWORD PTR [RDI + 1792], EAX
	mov	RAX, QWORD PTR [RSP + 144]
	mov	EAX, DWORD PTR [RAX + 4*R11]
	mov	DWORD PTR [RDI + 1920], EAX
	mov	RBX, QWORD PTR [RSI + 56]
	test	RBX, RBX
	mov	R14, RBX
	mov	EAX, 1
	cmove	R14, RAX
	mov	EAX, DWORD PTR [RDI + 1792]
	dec	EAX
	movsxd	RAX, EAX
	xor	EDX, EDX
	div	R14
	inc	EAX
	imul	EBX, EAX
	mov	DWORD PTR [RDI + 1408], EBX
	mov	RAX, QWORD PTR [RSP + 160]
	mov	EAX, DWORD PTR [RAX + 4*R11]
	mov	DWORD PTR [RDI + 1664], EAX
	lea	EAX, DWORD PTR [4*R11]
	movsxd	RAX, EAX
	mov	RDX, QWORD PTR [RSP + 112]
	mov	EBX, DWORD PTR [RDX + 4*RAX]
	mov	DWORD PTR [RDI + 512], EBX
	lea	R14D, DWORD PTR [4*R11 + 1]
	movsxd	R14, R14D
	movsxd	R15, DWORD PTR [RDX + 4*R14]
	mov	DWORD PTR [RDI + 768], R15D
	lea	R12D, DWORD PTR [4*R11 + 2]
	movsxd	R12, R12D
	mov	R13D, DWORD PTR [RDX + 4*R12]
	mov	DWORD PTR [RDI + 384], R13D
	lea	R11D, DWORD PTR [4*R11 + 3]
	movsxd	R11, R11D
	mov	EDX, DWORD PTR [RDX + 4*R11]
	mov	DWORD PTR [RDI + 1024], EDX
	mov	RDX, QWORD PTR [RSP + 120]
	movsxd	RAX, DWORD PTR [RDX + 4*RAX]
	mov	RBP, QWORD PTR [RSP + 88]
	lea	RAX, QWORD PTR [RBP + 4*RAX]
	mov	QWORD PTR [RDI + 1536], RAX
	movsxd	RAX, DWORD PTR [RDX + 4*R14]
	add	RAX, QWORD PTR [RSP - 24] # 8-byte Folded Reload
	lea	RAX, QWORD PTR [RBP + 4*RAX]
	mov	QWORD PTR [RDI + 896], RAX
	movsxd	RAX, DWORD PTR [RDX + 4*R12]
	add	RAX, QWORD PTR [RSP - 16] # 8-byte Folded Reload
	lea	RAX, QWORD PTR [RBP + 4*RAX]
	mov	QWORD PTR [RDI], RAX
	movsxd	RAX, DWORD PTR [RDX + 4*R11]
	add	RAX, QWORD PTR [RSP - 8] # 8-byte Folded Reload
	lea	RAX, QWORD PTR [RBP + 4*RAX]
	mov	QWORD PTR [RDI + 640], RAX
	mov	RAX, QWORD PTR [RSP + 192]
	mov	QWORD PTR [RDI + 1152], RAX
	lea	RDX, QWORD PTR [8*RBX + 15]
	movsxd	RDX, EDX
	mov	R11, RDX
	and	R11, -16
	add	R11, RAX
	mov	QWORD PTR [RDI + 128], R11
	shl	R15, 4
	and	EDX, -16
	add	RDX, R15
	movsxd	RDX, EDX
	lea	R11, QWORD PTR [RAX + RDX]
	mov	QWORD PTR [RDI + 256], R11
	shl	R13, 2
	add	R13D, 15
	and	R13D, -16
	add	EDX, R13D
	movsxd	RDX, EDX
	add	RDX, RAX
	mov	QWORD PTR [RDI + 1280], RDX
.LBB0_5:                                #   in Loop: Header=BB0_2 Depth=2
	cmp	R8, QWORD PTR [RSP + 272]
	jb	.LBB0_49
# BB#6:                                 # %.SyncBB_crit_edge
                                        #   in Loop: Header=BB0_1 Depth=1
	mov	R8, -1
	mov	R9, QWORD PTR [RSP + 256]
	.align	16, 0x90
.LBB0_7:                                # %SyncBB
                                        #   Parent Loop BB0_1 Depth=1
                                        # =>  This Loop Header: Depth=2
                                        #       Child Loop BB0_18 Depth 3
                                        #       Child Loop BB0_15 Depth 3
                                        #       Child Loop BB0_12 Depth 3
                                        #       Child Loop BB0_9 Depth 3
	mov	EAX, DWORD PTR [RDI + 512]
	add	EAX, EAX
	mov	RDX, QWORD PTR [R9]
	cmp	EDX, EAX
	jge	.LBB0_10
# BB#8:                                 # %SyncBB.bb.nph49_crit_edge
                                        #   in Loop: Header=BB0_7 Depth=2
	mov	RAX, RDX
	.align	16, 0x90
.LBB0_9:                                # %bb.nph49
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
	mov	R11, QWORD PTR [RDI + 1152]
	mov	R14, QWORD PTR [RDI + 1536]
	mov	R10D, DWORD PTR [R14 + 4*R10]
	lea	R10D, DWORD PTR [RBX + 2*R10]
	movsxd	R10, R10D
	mov	RBX, QWORD PTR [RSP + 56]
	movss	XMM2, DWORD PTR [RBX + 4*R10]
	movsxd	RDX, EDX
	movss	DWORD PTR [R11 + 4*RDX], XMM2
	mov	EDX, EAX
	add	RDX, QWORD PTR [RSI + 56]
	mov	EAX, DWORD PTR [RDI + 512]
	add	EAX, EAX
	cmp	EDX, EAX
	mov	RAX, RDX
	jl	.LBB0_9
.LBB0_10:                               # %._crit_edge50
                                        #   in Loop: Header=BB0_7 Depth=2
	mov	EAX, DWORD PTR [RDI + 768]
	shl	EAX, 2
	mov	RDX, QWORD PTR [R9]
	cmp	EDX, EAX
	jge	.LBB0_13
# BB#11:                                # %._crit_edge50.bb.nph44_crit_edge
                                        #   in Loop: Header=BB0_7 Depth=2
	mov	RAX, RDX
	.align	16, 0x90
.LBB0_12:                               # %bb.nph44
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
	mov	R11, QWORD PTR [RDI + 896]
	mov	R10D, DWORD PTR [R11 + 4*R10]
	lea	R10D, DWORD PTR [RBX + 4*R10]
	movsxd	R10, R10D
	mov	R11, QWORD PTR [RSP + 64]
	movss	XMM2, DWORD PTR [R11 + 4*R10]
	mov	R10, QWORD PTR [RDI + 128]
	movsxd	RDX, EDX
	movss	DWORD PTR [R10 + 4*RDX], XMM2
	mov	EDX, EAX
	add	RDX, QWORD PTR [RSI + 56]
	mov	EAX, DWORD PTR [RDI + 768]
	shl	EAX, 2
	cmp	EDX, EAX
	mov	RAX, RDX
	jl	.LBB0_12
.LBB0_13:                               # %._crit_edge45
                                        #   in Loop: Header=BB0_7 Depth=2
	mov	RAX, QWORD PTR [R9]
	cmp	EAX, DWORD PTR [RDI + 384]
	jge	.LBB0_16
# BB#14:                                # %._crit_edge45.bb.nph39_crit_edge
                                        #   in Loop: Header=BB0_7 Depth=2
	mov	RDX, RAX
	.align	16, 0x90
.LBB0_15:                               # %bb.nph39
                                        #   Parent Loop BB0_1 Depth=1
                                        #     Parent Loop BB0_7 Depth=2
                                        # =>    This Inner Loop Header: Depth=3
	movsxd	RAX, EAX
	mov	R10, QWORD PTR [RDI]
	mov	R11, QWORD PTR [RDI + 256]
	movsxd	R10, DWORD PTR [R10 + 4*RAX]
	mov	RBX, QWORD PTR [RSP + 72]
	movss	XMM2, DWORD PTR [RBX + 4*R10]
	movss	DWORD PTR [R11 + 4*RAX], XMM2
	mov	EAX, EDX
	add	RAX, QWORD PTR [RSI + 56]
	cmp	EAX, DWORD PTR [RDI + 384]
	mov	RDX, RAX
	jl	.LBB0_15
.LBB0_16:                               # %._crit_edge40
                                        #   in Loop: Header=BB0_7 Depth=2
	mov	EAX, DWORD PTR [RDI + 1024]
	shl	EAX, 2
	mov	RDX, QWORD PTR [R9]
	cmp	EDX, EAX
	jge	.LBB0_19
# BB#17:                                # %._crit_edge40.bb.nph34_crit_edge
                                        #   in Loop: Header=BB0_7 Depth=2
	mov	RAX, RDX
	.align	16, 0x90
.LBB0_18:                               # %bb.nph34
                                        #   Parent Loop BB0_1 Depth=1
                                        #     Parent Loop BB0_7 Depth=2
                                        # =>    This Inner Loop Header: Depth=3
	mov	R10, QWORD PTR [RDI + 1280]
	movsxd	RDX, EDX
	mov	DWORD PTR [R10 + 4*RDX], 0
	mov	EDX, EAX
	add	RDX, QWORD PTR [RSI + 56]
	mov	EAX, DWORD PTR [RDI + 1024]
	shl	EAX, 2
	cmp	EDX, EAX
	mov	RAX, RDX
	jl	.LBB0_18
.LBB0_19:                               # %._crit_edge35
                                        #   in Loop: Header=BB0_7 Depth=2
	add	R9, 32
	inc	R8
	cmp	R8, QWORD PTR [RSP + 272]
	jb	.LBB0_7
# BB#20:                                # %._crit_edge35.SyncBB443_crit_edge
                                        #   in Loop: Header=BB0_1 Depth=1
	mov	DWORD PTR [RSP - 44], 1 # 4-byte Folded Spill
	xor	R8D, R8D
	mov	R9, R8
.LBB0_21:                               # %SyncBB443
                                        #   in Loop: Header=BB0_1 Depth=1
	mov	RAX, R8
	shl	RAX, 5
	mov	RDX, QWORD PTR [RSP + 256]
	mov	RAX, QWORD PTR [RDX + RAX]
	mov	QWORD PTR [RCX + R9], RAX
	mov	DWORD PTR [RCX + R9 + 8], EAX
	cmp	EAX, DWORD PTR [RDI + 1408]
	jge	.LBB0_42
# BB#22:                                # %bb.nph29
                                        #   in Loop: Header=BB0_1 Depth=1
	lea	RAX, QWORD PTR [RCX + R9 + 160]
	mov	QWORD PTR [RCX + R9 + 16], RAX
	mov	EAX, DWORD PTR [RCX + R9 + 8]
	mov	RDX, QWORD PTR [RCX + R9]
	mov	R10D, DWORD PTR [RSP - 68] # 4-byte Reload
	mov	DWORD PTR [RSP - 48], R10D # 4-byte Spill
	mov	R10D, DWORD PTR [RSP - 60] # 4-byte Reload
	mov	DWORD PTR [RSP - 52], R10D # 4-byte Spill
	mov	R10D, DWORD PTR [RSP - 64] # 4-byte Reload
	mov	DWORD PTR [RSP - 56], R10D # 4-byte Spill
.LBB0_23:                               #   in Loop: Header=BB0_1 Depth=1
	mov	QWORD PTR [R9 + RCX + 32], RDX
	mov	DWORD PTR [R9 + RCX + 28], R10D
	mov	DWORD PTR [R9 + RCX + 24], EAX
	cmp	EAX, DWORD PTR [RDI + 1792]
	jl	.LBB0_25
# BB#24:                                # %..thread_crit_edge
                                        #   in Loop: Header=BB0_1 Depth=1
	mov	EAX, -1
	jmp	.LBB0_32
.LBB0_25:                               #   in Loop: Header=BB0_1 Depth=1
	pxor	XMM2, XMM2
	movaps	XMMWORD PTR [R9 + RCX + 160], XMM2
	mov	EAX, DWORD PTR [R9 + RCX + 24]
	mov	R10D, DWORD PTR [RSP - 52] # 4-byte Reload
	lea	R10D, DWORD PTR [RAX + R10]
	mov	EDX, DWORD PTR [RDI + 1920]
	add	R10D, EDX
	lea	R11D, DWORD PTR [RDX + RAX]
	mov	EBX, DWORD PTR [RSP - 56] # 4-byte Reload
	lea	EBX, DWORD PTR [RAX + RBX]
	add	EBX, EDX
	movsxd	RBX, EBX
	mov	R14, QWORD PTR [RSP + 96]
	movsx	EBX, WORD PTR [R14 + 2*RBX]
	add	EAX, DWORD PTR [RSP + 184]
	add	EAX, EDX
	movsxd	RAX, EAX
	movsx	EAX, WORD PTR [R14 + 2*RAX]
	movsxd	RDX, R11D
	movsx	R11D, WORD PTR [R14 + 2*RDX]
	movsxd	R10, R10D
	movsx	R10, WORD PTR [R14 + 2*R10]
	shl	R10, 2
	add	R10, QWORD PTR [RDI + 256]
	mov	R14, QWORD PTR [RDI + 128]
	mov	R15, QWORD PTR [RDI + 1152]
	mov	QWORD PTR [R9 + RCX + 40], R10
	add	R11D, R11D
	movsxd	R10, R11D
	movss	XMM2, DWORD PTR [R15 + 4*R10 + 4]
	add	EAX, EAX
	movsxd	RAX, EAX
	subss	XMM2, DWORD PTR [R15 + 4*RAX + 4]
	movss	XMM3, DWORD PTR [R15 + 4*R10]
	subss	XMM3, DWORD PTR [R15 + 4*RAX]
	shl	EBX, 2
	mov	RAX, QWORD PTR [RSP + 104]
	cmp	DWORD PTR [RAX + 4*RDX], 1
	mov	RAX, QWORD PTR [RSP + 208]
	movss	XMM4, DWORD PTR [RAX]
	mov	RAX, QWORD PTR [RSP + 200]
	movss	XMM5, DWORD PTR [RAX]
	movss	DWORD PTR [R9 + RCX + 48], XMM3
	movss	DWORD PTR [R9 + RCX + 52], XMM2
	movsxd	RAX, EBX
	movss	XMM2, DWORD PTR [R14 + 4*RAX]
	movss	DWORD PTR [R9 + RCX + 56], XMM2
	movaps	XMM3, XMM0
	divss	XMM3, XMM2
	movss	DWORD PTR [R9 + RCX + 60], XMM3
	lea	RDX, QWORD PTR [R14 + 4*RAX + 12]
	mov	QWORD PTR [R9 + RCX + 64], RDX
	movss	XMM2, DWORD PTR [R14 + 4*RAX + 12]
	lea	RDX, QWORD PTR [R14 + 4*RAX + 4]
	mov	QWORD PTR [R9 + RCX + 72], RDX
	movss	XMM6, DWORD PTR [R14 + 4*RAX + 4]
	movss	DWORD PTR [R9 + RCX + 80], XMM6
	lea	RDX, QWORD PTR [R14 + 4*RAX + 8]
	mov	QWORD PTR [R9 + RCX + 88], RDX
	movss	XMM7, DWORD PTR [R14 + 4*RAX + 8]
	movss	DWORD PTR [R9 + RCX + 96], XMM7
	mulss	XMM6, XMM6
	mulss	XMM7, XMM7
	addss	XMM7, XMM6
	mulss	XMM3, DWORD PTR [RIP + .LCPI0_1]
	mulss	XMM3, XMM7
	addss	XMM3, XMM2
	mulss	XMM3, XMM5
	movss	DWORD PTR [R9 + RCX + 100], XMM3
	movss	XMM2, DWORD PTR [R9 + RCX + 52]
	jne	.LBB0_27
# BB#26:                                #   in Loop: Header=BB0_1 Depth=1
	mulss	XMM2, DWORD PTR [R9 + RCX + 100]
	mov	RAX, QWORD PTR [R9 + RCX + 16]
	addss	XMM2, DWORD PTR [RAX + 4]
	movss	DWORD PTR [RAX + 4], XMM2
	movss	XMM2, DWORD PTR [R9 + RCX + 48]
	mulss	XMM2, DWORD PTR [R9 + RCX + 100]
	mov	RAX, QWORD PTR [R9 + RCX + 16]
	movss	XMM4, DWORD PTR [RAX + 8]
	subss	XMM4, XMM2
	movss	DWORD PTR [RAX + 8], XMM4
	jmp	.LBB0_28
.LBB0_27:                               #   in Loop: Header=BB0_1 Depth=1
	mov	RAX, QWORD PTR [RSP + 216]
	movss	XMM3, DWORD PTR [RAX]
	movss	XMM6, DWORD PTR [RAX + 4]
	movaps	XMM7, XMM0
	divss	XMM7, XMM3
	movss	XMM8, DWORD PTR [R9 + RCX + 52]
	mulss	XMM8, XMM6
	movss	XMM9, DWORD PTR [R9 + RCX + 48]
	movss	XMM10, DWORD PTR [RAX + 8]
	movaps	XMM11, XMM10
	mulss	XMM11, XMM9
	subss	XMM8, XMM11
	mulss	XMM8, XMM7
	movaps	XMM11, XMM8
	mulss	XMM11, XMM3
	mulss	XMM2, DWORD PTR [R9 + RCX + 80]
	mulss	XMM9, DWORD PTR [R9 + RCX + 96]
	subss	XMM2, XMM9
	mulss	XMM2, DWORD PTR [R9 + RCX + 60]
	movss	XMM9, DWORD PTR [R9 + RCX + 56]
	movaps	XMM12, XMM2
	mulss	XMM12, XMM9
	addss	XMM12, XMM11
	mulss	XMM12, XMM1
	mov	RDX, QWORD PTR [R9 + RCX + 16]
	mov	R10, QWORD PTR [R9 + RCX + 40]
	mulss	XMM4, DWORD PTR [R10]
	subss	XMM9, XMM3
	mulss	XMM9, XMM4
	addss	XMM9, XMM12
	addss	XMM9, DWORD PTR [RDX]
	movss	XMM3, DWORD PTR [RAX + 12]
	movss	DWORD PTR [RDX], XMM9
	movss	XMM9, DWORD PTR [RAX + 4]
	movaps	XMM11, XMM8
	mulss	XMM11, XMM9
	mov	RDX, QWORD PTR [R9 + RCX + 72]
	movss	XMM12, DWORD PTR [RDX]
	movaps	XMM13, XMM2
	mulss	XMM13, XMM12
	movss	XMM14, DWORD PTR [R9 + RCX + 52]
	movss	XMM15, DWORD PTR [R9 + RCX + 100]
	mulss	XMM15, XMM14
	addss	XMM15, XMM13
	addss	XMM15, XMM11
	mulss	XMM7, XMM1
	mulss	XMM10, XMM10
	mulss	XMM6, XMM6
	addss	XMM6, XMM10
	mulss	XMM6, XMM7
	subss	XMM3, XMM6
	mulss	XMM3, XMM5
	mulss	XMM14, XMM3
	addss	XMM14, XMM15
	mulss	XMM14, XMM1
	subss	XMM12, XMM9
	mulss	XMM12, XMM4
	addss	XMM12, XMM14
	mov	RDX, QWORD PTR [R9 + RCX + 16]
	addss	XMM12, DWORD PTR [RDX + 4]
	movss	DWORD PTR [RDX + 4], XMM12
	movss	XMM5, DWORD PTR [RAX + 8]
	movaps	XMM6, XMM8
	mulss	XMM6, XMM5
	mov	RDX, QWORD PTR [R9 + RCX + 88]
	movss	XMM7, DWORD PTR [RDX]
	movaps	XMM9, XMM2
	mulss	XMM9, XMM7
	movss	XMM10, DWORD PTR [R9 + RCX + 48]
	movss	XMM11, DWORD PTR [R9 + RCX + 100]
	mulss	XMM11, XMM10
	subss	XMM9, XMM11
	addss	XMM9, XMM6
	mulss	XMM10, XMM3
	subss	XMM9, XMM10
	mulss	XMM9, XMM1
	subss	XMM7, XMM5
	mulss	XMM7, XMM4
	addss	XMM7, XMM9
	mov	RDX, QWORD PTR [R9 + RCX + 16]
	addss	XMM7, DWORD PTR [RDX + 8]
	movss	DWORD PTR [RDX + 8], XMM7
	movss	XMM5, DWORD PTR [RAX + 12]
	addss	XMM3, XMM5
	mulss	XMM3, XMM8
	mov	RAX, QWORD PTR [R9 + RCX + 16]
	mov	RDX, QWORD PTR [R9 + RCX + 64]
	movss	XMM6, DWORD PTR [RDX]
	movss	XMM7, DWORD PTR [R9 + RCX + 100]
	addss	XMM7, XMM6
	mulss	XMM7, XMM2
	addss	XMM7, XMM3
	mulss	XMM7, XMM1
	subss	XMM6, XMM5
	mulss	XMM6, XMM4
	addss	XMM6, XMM7
	addss	XMM6, DWORD PTR [RAX + 12]
	movss	DWORD PTR [RAX + 12], XMM6
.LBB0_28:                               # %bres_calc.exit
                                        #   in Loop: Header=BB0_1 Depth=1
	mov	EAX, DWORD PTR [RDI + 1920]
	mov	R10D, DWORD PTR [R9 + RCX + 24]
	add	R10D, EAX
	movsxd	RDX, R10D
	mov	R10, QWORD PTR [RSP + 168]
	mov	R10D, DWORD PTR [R10 + 4*RDX]
	mov	DWORD PTR [R9 + RCX + 104], R10D
	test	R10D, R10D
	jns	.LBB0_30
# BB#29:                                # %bres_calc.exit.phi-split-bb_crit_edge
                                        #   in Loop: Header=BB0_1 Depth=1
	mov	R10D, DWORD PTR [R9 + RCX + 28]
	jmp	.LBB0_31
.LBB0_30:                               #   in Loop: Header=BB0_1 Depth=1
	mov	R10D, DWORD PTR [R9 + RCX + 24]
	add	R10D, DWORD PTR [RSP - 48] # 4-byte Folded Reload
	add	R10D, EAX
	movsxd	RAX, R10D
	mov	RDX, QWORD PTR [RSP + 96]
	movsx	R10D, WORD PTR [RDX + 2*RAX]
	mov	DWORD PTR [R9 + RCX + 108], R10D
.LBB0_31:                               # %phi-split-bb
                                        #   in Loop: Header=BB0_1 Depth=1
	mov	DWORD PTR [R9 + RCX + 112], R10D
	mov	EAX, DWORD PTR [R9 + RCX + 104]
.LBB0_32:                               # %.thread
                                        #   in Loop: Header=BB0_1 Depth=1
	mov	DWORD PTR [R9 + RCX + 120], R10D
	mov	DWORD PTR [R9 + RCX + 116], EAX
	cmp	DWORD PTR [RDI + 1664], 0
	jle	.LBB0_40
# BB#33:                                # %bb.nph23
                                        #   in Loop: Header=BB0_1 Depth=1
	mov	R10D, DWORD PTR [R9 + RCX + 120]
	shl	R10D, 2
	mov	DWORD PTR [R9 + RCX + 124], R10D
	xor	R10D, R10D
.LBB0_34:                               #   in Loop: Header=BB0_1 Depth=1
	lea	EAX, DWORD PTR [R10 + 1]
	mov	DWORD PTR [R9 + RCX + 128], EAX
	cmp	DWORD PTR [R9 + RCX + 116], R10D
	jne	.LBB0_36
# BB#35:                                # %.loopexit19
                                        #   in Loop: Header=BB0_1 Depth=1
	mov	RAX, QWORD PTR [R9 + RCX + 16]
	movsxd	RDX, DWORD PTR [R9 + RCX + 124]
	mov	R10, QWORD PTR [RDI + 1280]
	movss	XMM2, DWORD PTR [R10 + 4*RDX]
	addss	XMM2, DWORD PTR [RAX]
	movss	DWORD PTR [R10 + 4*RDX], XMM2
	mov	R10D, DWORD PTR [R9 + RCX + 124]
	or	R10D, 1
	movsxd	RAX, R10D
	mov	RDX, QWORD PTR [RDI + 1280]
	movss	XMM2, DWORD PTR [RDX + 4*RAX]
	addss	XMM2, DWORD PTR [R9 + RCX + 164]
	movss	DWORD PTR [RDX + 4*RAX], XMM2
	mov	R10D, DWORD PTR [R9 + RCX + 124]
	or	R10D, 2
	movsxd	RAX, R10D
	mov	RDX, QWORD PTR [RDI + 1280]
	movss	XMM2, DWORD PTR [RDX + 4*RAX]
	addss	XMM2, DWORD PTR [R9 + RCX + 168]
	movss	DWORD PTR [RDX + 4*RAX], XMM2
	mov	R10D, DWORD PTR [R9 + RCX + 124]
	or	R10D, 3
	movsxd	RAX, R10D
	mov	RDX, QWORD PTR [RDI + 1280]
	movss	XMM2, DWORD PTR [RDX + 4*RAX]
	addss	XMM2, DWORD PTR [R9 + RCX + 172]
	movss	DWORD PTR [RDX + 4*RAX], XMM2
.LBB0_36:                               #   in Loop: Header=BB0_1 Depth=1
	cmp	R8, QWORD PTR [RSP + 272]
	jb	.LBB0_38
# BB#37:                                # %.SyncBB444_crit_edge
                                        #   in Loop: Header=BB0_1 Depth=1
	mov	DWORD PTR [RSP - 44], 2 # 4-byte Folded Spill
	xor	R8D, R8D
	mov	R9, R8
	jmp	.LBB0_39
.LBB0_38:                               # %thenBB455
                                        #   in Loop: Header=BB0_1 Depth=1
	add	R9, 384
	inc	R8
	cmp	DWORD PTR [RSP - 44], 1 # 4-byte Folded Reload
	je	.LBB0_21
.LBB0_39:                               # %SyncBB444
                                        #   in Loop: Header=BB0_1 Depth=1
	mov	R10D, DWORD PTR [R9 + RCX + 128]
	cmp	R10D, DWORD PTR [RDI + 1664]
	jl	.LBB0_34
.LBB0_40:                               # %._crit_edge24
                                        #   in Loop: Header=BB0_1 Depth=1
	mov	RAX, QWORD PTR [R9 + RCX + 32]
	mov	EDX, 4294967295
	and	RAX, RDX
	add	RAX, QWORD PTR [RSI + 56]
	mov	QWORD PTR [R9 + RCX + 136], RAX
	mov	DWORD PTR [R9 + RCX + 144], EAX
	cmp	EAX, DWORD PTR [RDI + 1408]
	mov	R10D, DWORD PTR [R9 + RCX + 120]
	jge	.LBB0_42
# BB#41:                                #   in Loop: Header=BB0_1 Depth=1
	mov	RDX, RAX
	jmp	.LBB0_23
.LBB0_42:                               # %._crit_edge30
                                        #   in Loop: Header=BB0_1 Depth=1
	mov	RAX, R8
	shl	RAX, 5
	mov	RDX, QWORD PTR [RSP + 256]
	mov	RAX, QWORD PTR [RDX + RAX]
	mov	EDX, DWORD PTR [RDI + 1024]
	shl	EDX, 2
	cmp	EAX, EDX
	jge	.LBB0_45
# BB#43:                                # %._crit_edge30.bb.nph_crit_edge
                                        #   in Loop: Header=BB0_1 Depth=1
	mov	RDX, RAX
	.align	16, 0x90
.LBB0_44:                               # %bb.nph
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
	mov	R11, QWORD PTR [RDI + 640]
	mov	R10D, DWORD PTR [R11 + 4*R10]
	lea	R10D, DWORD PTR [RBX + 4*R10]
	movsxd	R10, R10D
	mov	R11, QWORD PTR [RSP + 80]
	movss	XMM2, DWORD PTR [R11 + 4*R10]
	movsxd	RAX, EAX
	mov	RBX, QWORD PTR [RDI + 1280]
	addss	XMM2, DWORD PTR [RBX + 4*RAX]
	movss	DWORD PTR [R11 + 4*R10], XMM2
	mov	EAX, EDX
	add	RAX, QWORD PTR [RSI + 56]
	mov	EDX, DWORD PTR [RDI + 1024]
	shl	EDX, 2
	cmp	EAX, EDX
	mov	RDX, RAX
	jl	.LBB0_44
.LBB0_45:                               # %.loopexit
                                        #   in Loop: Header=BB0_1 Depth=1
	cmp	R8, QWORD PTR [RSP + 272]
	jb	.LBB0_46
# BB#48:                                # %SyncBB446
	pop	RBX
	pop	R12
	pop	R13
	pop	R14
	pop	R15
	pop	RBP
	ret
.Ltmp0:
	.size	op_opencl_bres_calc, .Ltmp0-op_opencl_bres_calc

	.section	.rodata.cst4,"aM",@progbits,4
	.align	16
.LCPI1_0:                               # constant pool float
	.long	1065353216              # float 1.000000e+00
.LCPI1_1:                               # constant pool float
	.long	3204448256              # float -5.000000e-01
	.zero	8
.LCPI1_2:                               # constant pool <1 x float>
	.long	2147483648              # float -0.000000e+00
.LCPI1_3:                               # constant pool float
	.long	1056964608              # float 5.000000e-01
.LCPI1_4:                               # constant pool float
	.long	0                       # float 0.000000e+00
	.text
	.globl	__Vectorized_.op_opencl_bres_calc
	.align	16, 0x90
	.type	__Vectorized_.op_opencl_bres_calc,@function
__Vectorized_.op_opencl_bres_calc:      # @__Vectorized_.op_opencl_bres_calc
# BB#0:                                 # %FirstBB
	push	RBP
	push	R15
	push	R14
	push	R13
	push	R12
	push	RBX
	mov	QWORD PTR [RSP - 88], R9 # 8-byte Spill
	mov	QWORD PTR [RSP - 72], R8 # 8-byte Spill
	mov	QWORD PTR [RSP - 64], RCX # 8-byte Spill
	mov	QWORD PTR [RSP - 8], RDX # 8-byte Spill
	mov	EAX, DWORD PTR [RSP + 136]
	lea	ECX, DWORD PTR [RAX + 2*RAX]
	mov	DWORD PTR [RSP - 104], ECX # 4-byte Spill
	lea	EDX, DWORD PTR [RAX + RAX]
	mov	DWORD PTR [RSP - 108], EDX # 4-byte Spill
	lea	EAX, DWORD PTR [4*RAX]
	mov	DWORD PTR [RSP - 112], EAX # 4-byte Spill
	movsxd	RAX, EAX
	mov	QWORD PTR [RSP - 24], RAX # 8-byte Spill
	movsxd	RAX, ECX
	mov	QWORD PTR [RSP - 40], RAX # 8-byte Spill
	movsxd	RAX, EDX
	mov	QWORD PTR [RSP - 32], RAX # 8-byte Spill
	mov	DWORD PTR [RSP - 76], 8 # 4-byte Folded Spill
	movsxd	RAX, DWORD PTR [RSP + 80]
	mov	QWORD PTR [RSP - 48], RAX # 8-byte Spill
	movsxd	RAX, DWORD PTR [RSP + 128]
	mov	QWORD PTR [RSP - 56], RAX # 8-byte Spill
	mov	RCX, QWORD PTR [RSP + 184]
	mov	R8, QWORD PTR [RSP + 176]
	movss	XMM0, DWORD PTR [RIP + .LCPI1_3]
	mov	QWORD PTR [RSP - 16], 0 # 8-byte Folded Spill
	xor	R9D, R9D
	mov	RBP, QWORD PTR [RSP - 72] # 8-byte Reload
	jmp	.LBB1_1
	.align	16, 0x90
.LBB1_46:                               # %thenBB461
                                        #   in Loop: Header=BB1_1 Depth=1
	add	R9, 384
	inc	QWORD PTR [RSP - 16]    # 8-byte Folded Spill
	cmp	DWORD PTR [RSP - 76], 3 # 4-byte Folded Reload
	je	.LBB1_21
# BB#47:                                # %thenBB461
                                        #   in Loop: Header=BB1_1 Depth=1
	cmp	DWORD PTR [RSP - 76], 6 # 4-byte Folded Reload
	je	.LBB1_39
.LBB1_1:                                # %SyncBB456.outer
                                        # =>This Loop Header: Depth=1
                                        #     Child Loop BB1_44 Depth 2
                                        #     Child Loop BB1_7 Depth 2
                                        #       Child Loop BB1_18 Depth 3
                                        #       Child Loop BB1_15 Depth 3
                                        #       Child Loop BB1_12 Depth 3
                                        #       Child Loop BB1_9 Depth 3
                                        #     Child Loop BB1_2 Depth 2
	mov	R10, QWORD PTR [RSP - 16] # 8-byte Reload
	shl	R10, 5
	add	R10, QWORD PTR [RSP + 208]
	jmp	.LBB1_2
	.align	16, 0x90
.LBB1_49:                               # %thenBB468
                                        #   in Loop: Header=BB1_2 Depth=2
	add	R10, 32
	add	R9, 384
	inc	QWORD PTR [RSP - 16]    # 8-byte Folded Spill
.LBB1_2:                                # %SyncBB456
                                        #   Parent Loop BB1_1 Depth=1
                                        # =>  This Inner Loop Header: Depth=2
	mov	RAX, QWORD PTR [RCX + 80]
	mov	RDX, QWORD PTR [RSP + 192]
	imul	RAX, QWORD PTR [RDX + 8]
	add	RAX, QWORD PTR [RDX]
	cmp	RAX, QWORD PTR [RSP - 56] # 8-byte Folded Reload
	jae	.LBB1_45
# BB#3:                                 #   in Loop: Header=BB1_2 Depth=2
	cmp	QWORD PTR [R10], 0
	jne	.LBB1_5
# BB#4:                                 #   in Loop: Header=BB1_2 Depth=2
	mov	RAX, RDX
	mov	RDX, QWORD PTR [RAX]
	add	RDX, QWORD PTR [RSP - 48] # 8-byte Folded Reload
	mov	R11, QWORD PTR [RCX + 80]
	imul	R11, QWORD PTR [RAX + 8]
	add	R11, RDX
	mov	RAX, QWORD PTR [RSP + 88]
	movsxd	R11, DWORD PTR [RAX + 4*R11]
	mov	RAX, QWORD PTR [RSP + 104]
	mov	EAX, DWORD PTR [RAX + 4*R11]
	mov	DWORD PTR [R8 + 1792], EAX
	mov	RAX, QWORD PTR [RSP + 96]
	mov	EAX, DWORD PTR [RAX + 4*R11]
	mov	DWORD PTR [R8 + 1920], EAX
	mov	RBX, QWORD PTR [RCX + 56]
	test	RBX, RBX
	mov	R14, RBX
	mov	EAX, 1
	cmove	R14, RAX
	mov	EAX, DWORD PTR [R8 + 1792]
	dec	EAX
	movsxd	RAX, EAX
	xor	EDX, EDX
	div	R14
	inc	EAX
	imul	EBX, EAX
	mov	DWORD PTR [R8 + 1408], EBX
	mov	RAX, QWORD PTR [RSP + 112]
	mov	EAX, DWORD PTR [RAX + 4*R11]
	mov	DWORD PTR [R8 + 1664], EAX
	lea	EAX, DWORD PTR [4*R11]
	movsxd	RAX, EAX
	mov	RDX, QWORD PTR [RSP + 64]
	mov	EBX, DWORD PTR [RDX + 4*RAX]
	mov	DWORD PTR [R8 + 512], EBX
	lea	R14D, DWORD PTR [4*R11 + 1]
	movsxd	R14, R14D
	movsxd	R15, DWORD PTR [RDX + 4*R14]
	mov	DWORD PTR [R8 + 768], R15D
	lea	R12D, DWORD PTR [4*R11 + 2]
	movsxd	R12, R12D
	mov	R13D, DWORD PTR [RDX + 4*R12]
	mov	DWORD PTR [R8 + 384], R13D
	lea	R11D, DWORD PTR [4*R11 + 3]
	movsxd	R11, R11D
	mov	EDX, DWORD PTR [RDX + 4*R11]
	mov	DWORD PTR [R8 + 1024], EDX
	mov	RDX, QWORD PTR [RSP + 72]
	movsxd	RAX, DWORD PTR [RDX + 4*RAX]
	lea	RAX, QWORD PTR [RBP + 4*RAX]
	mov	QWORD PTR [R8 + 1536], RAX
	movsxd	RAX, DWORD PTR [RDX + 4*R14]
	add	RAX, QWORD PTR [RSP - 32] # 8-byte Folded Reload
	lea	RAX, QWORD PTR [RBP + 4*RAX]
	mov	QWORD PTR [R8 + 896], RAX
	movsxd	RAX, DWORD PTR [RDX + 4*R12]
	add	RAX, QWORD PTR [RSP - 40] # 8-byte Folded Reload
	lea	RAX, QWORD PTR [RBP + 4*RAX]
	mov	QWORD PTR [R8], RAX
	movsxd	RAX, DWORD PTR [RDX + 4*R11]
	add	RAX, QWORD PTR [RSP - 24] # 8-byte Folded Reload
	lea	RAX, QWORD PTR [RBP + 4*RAX]
	mov	QWORD PTR [R8 + 640], RAX
	mov	RAX, QWORD PTR [RSP + 144]
	mov	QWORD PTR [R8 + 1152], RAX
	lea	RDX, QWORD PTR [8*RBX + 15]
	movsxd	RDX, EDX
	mov	R11, RDX
	and	R11, -16
	add	R11, RAX
	mov	QWORD PTR [R8 + 128], R11
	shl	R15, 4
	and	EDX, -16
	add	RDX, R15
	movsxd	RDX, EDX
	lea	R11, QWORD PTR [RAX + RDX]
	mov	QWORD PTR [R8 + 256], R11
	shl	R13, 2
	add	R13D, 15
	and	R13D, -16
	add	EDX, R13D
	movsxd	RDX, EDX
	add	RDX, RAX
	mov	QWORD PTR [R8 + 1280], RDX
.LBB1_5:                                #   in Loop: Header=BB1_2 Depth=2
	mov	RAX, QWORD PTR [RSP - 16] # 8-byte Reload
	cmp	RAX, QWORD PTR [RSP + 224]
	jb	.LBB1_49
# BB#6:                                 # %.SyncBB458_crit_edge
                                        #   in Loop: Header=BB1_1 Depth=1
	mov	R9, -1
	mov	R10, QWORD PTR [RSP + 208]
	.align	16, 0x90
.LBB1_7:                                # %SyncBB458
                                        #   Parent Loop BB1_1 Depth=1
                                        # =>  This Loop Header: Depth=2
                                        #       Child Loop BB1_18 Depth 3
                                        #       Child Loop BB1_15 Depth 3
                                        #       Child Loop BB1_12 Depth 3
                                        #       Child Loop BB1_9 Depth 3
	mov	EAX, DWORD PTR [R8 + 512]
	add	EAX, EAX
	mov	RDX, QWORD PTR [R10]
	cmp	EDX, EAX
	jge	.LBB1_10
# BB#8:                                 # %SyncBB458.bb.nph49_crit_edge
                                        #   in Loop: Header=BB1_7 Depth=2
	mov	RAX, RDX
	.align	16, 0x90
.LBB1_9:                                # %bb.nph49
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
	mov	RBX, QWORD PTR [R8 + 1152]
	mov	R15, QWORD PTR [R8 + 1536]
	mov	R11D, DWORD PTR [R15 + 4*R11]
	lea	R11D, DWORD PTR [R14 + 2*R11]
	movsxd	R11, R11D
	movss	XMM1, DWORD PTR [RDI + 4*R11]
	movsxd	RDX, EDX
	movss	DWORD PTR [RBX + 4*RDX], XMM1
	mov	EDX, EAX
	add	RDX, QWORD PTR [RCX + 56]
	mov	EAX, DWORD PTR [R8 + 512]
	add	EAX, EAX
	cmp	EDX, EAX
	mov	RAX, RDX
	jl	.LBB1_9
.LBB1_10:                               # %._crit_edge50
                                        #   in Loop: Header=BB1_7 Depth=2
	mov	EAX, DWORD PTR [R8 + 768]
	shl	EAX, 2
	mov	RDX, QWORD PTR [R10]
	cmp	EDX, EAX
	jge	.LBB1_13
# BB#11:                                # %._crit_edge50.bb.nph44_crit_edge
                                        #   in Loop: Header=BB1_7 Depth=2
	mov	RAX, RDX
	.align	16, 0x90
.LBB1_12:                               # %bb.nph44
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
	mov	RBX, QWORD PTR [R8 + 896]
	mov	R11D, DWORD PTR [RBX + 4*R11]
	lea	R11D, DWORD PTR [R14 + 4*R11]
	movsxd	R11, R11D
	movss	XMM1, DWORD PTR [RSI + 4*R11]
	mov	R11, QWORD PTR [R8 + 128]
	movsxd	RDX, EDX
	movss	DWORD PTR [R11 + 4*RDX], XMM1
	mov	EDX, EAX
	add	RDX, QWORD PTR [RCX + 56]
	mov	EAX, DWORD PTR [R8 + 768]
	shl	EAX, 2
	cmp	EDX, EAX
	mov	RAX, RDX
	jl	.LBB1_12
.LBB1_13:                               # %._crit_edge45
                                        #   in Loop: Header=BB1_7 Depth=2
	mov	RAX, QWORD PTR [R10]
	cmp	EAX, DWORD PTR [R8 + 384]
	jge	.LBB1_16
# BB#14:                                # %._crit_edge45.bb.nph39_crit_edge
                                        #   in Loop: Header=BB1_7 Depth=2
	mov	RDX, RAX
	.align	16, 0x90
.LBB1_15:                               # %bb.nph39
                                        #   Parent Loop BB1_1 Depth=1
                                        #     Parent Loop BB1_7 Depth=2
                                        # =>    This Inner Loop Header: Depth=3
	movsxd	RAX, EAX
	mov	R11, QWORD PTR [R8]
	mov	RBX, QWORD PTR [R8 + 256]
	movsxd	R11, DWORD PTR [R11 + 4*RAX]
	mov	R14, QWORD PTR [RSP - 8] # 8-byte Reload
	movss	XMM1, DWORD PTR [R14 + 4*R11]
	movss	DWORD PTR [RBX + 4*RAX], XMM1
	mov	EAX, EDX
	add	RAX, QWORD PTR [RCX + 56]
	cmp	EAX, DWORD PTR [R8 + 384]
	mov	RDX, RAX
	jl	.LBB1_15
.LBB1_16:                               # %._crit_edge40
                                        #   in Loop: Header=BB1_7 Depth=2
	mov	EAX, DWORD PTR [R8 + 1024]
	shl	EAX, 2
	mov	RDX, QWORD PTR [R10]
	cmp	EDX, EAX
	jge	.LBB1_19
# BB#17:                                # %._crit_edge40.bb.nph34_crit_edge
                                        #   in Loop: Header=BB1_7 Depth=2
	mov	RAX, RDX
	.align	16, 0x90
.LBB1_18:                               # %bb.nph34
                                        #   Parent Loop BB1_1 Depth=1
                                        #     Parent Loop BB1_7 Depth=2
                                        # =>    This Inner Loop Header: Depth=3
	mov	R11, QWORD PTR [R8 + 1280]
	movsxd	RDX, EDX
	mov	DWORD PTR [R11 + 4*RDX], 0
	mov	EDX, EAX
	add	RDX, QWORD PTR [RCX + 56]
	mov	EAX, DWORD PTR [R8 + 1024]
	shl	EAX, 2
	cmp	EDX, EAX
	mov	RAX, RDX
	jl	.LBB1_18
.LBB1_19:                               # %._crit_edge35
                                        #   in Loop: Header=BB1_7 Depth=2
	add	R10, 32
	inc	R9
	cmp	R9, QWORD PTR [RSP + 224]
	jb	.LBB1_7
# BB#20:                                # %._crit_edge35.SyncBB_crit_edge
                                        #   in Loop: Header=BB1_1 Depth=1
	mov	DWORD PTR [RSP - 76], 3 # 4-byte Folded Spill
	xor	R9D, R9D
	mov	QWORD PTR [RSP - 16], R9 # 8-byte Spill
.LBB1_21:                               # %SyncBB
                                        #   in Loop: Header=BB1_1 Depth=1
	mov	RAX, QWORD PTR [RSP - 16] # 8-byte Reload
	shl	RAX, 5
	mov	RDX, QWORD PTR [RSP + 208]
	mov	RAX, QWORD PTR [RDX + RAX]
	mov	RDX, QWORD PTR [RSP + 232]
	mov	QWORD PTR [R9 + RDX + 176], RAX
	mov	DWORD PTR [R9 + RDX + 184], EAX
	cmp	EAX, DWORD PTR [R8 + 1408]
	jge	.LBB1_42
# BB#22:                                # %bb.nph29
                                        #   in Loop: Header=BB1_1 Depth=1
	mov	RAX, RDX
	mov	EDX, DWORD PTR [R9 + RAX + 184]
	mov	RAX, QWORD PTR [R9 + RAX + 176]
	mov	R10D, DWORD PTR [RSP - 112] # 4-byte Reload
	mov	DWORD PTR [RSP - 92], R10D # 4-byte Spill
	mov	R10D, DWORD PTR [RSP - 104] # 4-byte Reload
	mov	DWORD PTR [RSP - 96], R10D # 4-byte Spill
	mov	R10D, DWORD PTR [RSP - 108] # 4-byte Reload
	mov	DWORD PTR [RSP - 100], R10D # 4-byte Spill
.LBB1_23:                               #   in Loop: Header=BB1_1 Depth=1
	mov	R11, QWORD PTR [RSP + 232]
	mov	QWORD PTR [R9 + R11 + 216], RAX
	mov	DWORD PTR [R9 + R11 + 208], R10D
	mov	DWORD PTR [R9 + R11 + 204], EDX
	movss	DWORD PTR [R9 + R11 + 200], XMM4
	movss	DWORD PTR [R9 + R11 + 196], XMM3
	movss	DWORD PTR [R9 + R11 + 192], XMM2
	movss	DWORD PTR [R9 + R11 + 188], XMM1
	cmp	EDX, DWORD PTR [R8 + 1792]
	jl	.LBB1_25
# BB#24:                                # %..thread_crit_edge
                                        #   in Loop: Header=BB1_1 Depth=1
	mov	EAX, -1
	jmp	.LBB1_32
.LBB1_25:                               #   in Loop: Header=BB1_1 Depth=1
	mov	RAX, R11
	mov	R10D, DWORD PTR [R9 + RAX + 204]
	mov	EDX, DWORD PTR [RSP + 136]
	lea	EDX, DWORD PTR [R10 + RDX]
	mov	R11D, DWORD PTR [R8 + 1920]
	add	EDX, R11D
	lea	EBX, DWORD PTR [R11 + R10]
	movsxd	RBX, EBX
	mov	R14, QWORD PTR [RSP - 88] # 8-byte Reload
	movsx	R15D, WORD PTR [R14 + 2*RBX]
	add	R15D, R15D
	movsxd	R15, R15D
	mov	R12, QWORD PTR [R8 + 1152]
	movss	XMM1, DWORD PTR [R12 + 4*R15 + 4]
	movsxd	RDX, EDX
	movsx	EDX, WORD PTR [R14 + 2*RDX]
	add	EDX, EDX
	movsxd	RDX, EDX
	subss	XMM1, DWORD PTR [R12 + 4*RDX + 4]
	movss	XMM2, DWORD PTR [R12 + 4*R15]
	subss	XMM2, DWORD PTR [R12 + 4*RDX]
	add	R10D, DWORD PTR [RSP - 100] # 4-byte Folded Reload
	add	R10D, R11D
	movsxd	RDX, R10D
	movsx	R10D, WORD PTR [R14 + 2*RDX]
	shl	R10D, 2
	mov	RDX, QWORD PTR [RSP + 56]
	cmp	DWORD PTR [RDX + 4*RBX], 1
	mov	RDX, QWORD PTR [RSP + 152]
	movss	XMM3, DWORD PTR [RDX]
	mov	RDX, QWORD PTR [R8 + 128]
	movss	DWORD PTR [R9 + RAX + 224], XMM2
	movss	DWORD PTR [R9 + RAX + 228], XMM1
	movsxd	R10, R10D
	movss	XMM1, DWORD PTR [RDX + 4*R10]
	movss	DWORD PTR [R9 + RAX + 232], XMM1
	movss	XMM2, DWORD PTR [RIP + .LCPI1_0]
	divss	XMM2, XMM1
	movss	DWORD PTR [R9 + RAX + 236], XMM2
	lea	RBX, QWORD PTR [RDX + 4*R10 + 12]
	mov	QWORD PTR [R9 + RAX + 240], RBX
	movss	XMM1, DWORD PTR [RDX + 4*R10 + 12]
	lea	RBX, QWORD PTR [RDX + 4*R10 + 4]
	mov	QWORD PTR [R9 + RAX + 248], RBX
	movss	XMM4, DWORD PTR [RDX + 4*R10 + 4]
	movss	DWORD PTR [R9 + RAX + 256], XMM4
	lea	RBX, QWORD PTR [RDX + 4*R10 + 8]
	mov	QWORD PTR [R9 + RAX + 264], RBX
	movss	XMM5, DWORD PTR [RDX + 4*R10 + 8]
	movss	DWORD PTR [R9 + RAX + 272], XMM5
	mulss	XMM4, XMM4
	mulss	XMM5, XMM5
	addss	XMM5, XMM4
	mulss	XMM2, DWORD PTR [RIP + .LCPI1_1]
	mulss	XMM2, XMM5
	addss	XMM2, XMM1
	mulss	XMM2, XMM3
	movss	DWORD PTR [R9 + RAX + 276], XMM2
	jne	.LBB1_27
# BB#26:                                #   in Loop: Header=BB1_1 Depth=1
	movss	XMM1, DWORD PTR [R9 + RAX + 276]
	mulss	XMM1, DWORD PTR [R9 + RAX + 228]
	movss	DWORD PTR [R9 + RAX + 280], XMM1
	movss	XMM3, DWORD PTR [R9 + RAX + 276]
	xorps	XMM3, XMMWORD PTR [RIP + .LCPI1_2]
	mulss	XMM3, DWORD PTR [R9 + RAX + 224]
	movss	DWORD PTR [R9 + RAX + 284], XMM3
	pxor	XMM2, XMM2
	movaps	XMM4, XMM2
	jmp	.LBB1_28
.LBB1_27:                               #   in Loop: Header=BB1_1 Depth=1
	mov	RAX, QWORD PTR [RSP + 168]
	movss	XMM1, DWORD PTR [RAX]
	movss	XMM2, DWORD PTR [RAX + 4]
	movss	XMM4, DWORD PTR [RIP + .LCPI1_0]
	movaps	XMM5, XMM4
	divss	XMM5, XMM1
	mov	RDX, QWORD PTR [RSP + 232]
	movss	XMM4, DWORD PTR [R9 + RDX + 224]
	movss	XMM6, DWORD PTR [R9 + RDX + 228]
	movss	XMM7, DWORD PTR [RAX + 8]
	movaps	XMM8, XMM7
	mulss	XMM8, XMM4
	movaps	XMM9, XMM2
	mulss	XMM9, XMM6
	subss	XMM9, XMM8
	mulss	XMM9, XMM5
	movaps	XMM8, XMM9
	mulss	XMM8, XMM1
	mulss	XMM4, DWORD PTR [R9 + RDX + 272]
	mulss	XMM6, DWORD PTR [R9 + RDX + 256]
	subss	XMM6, XMM4
	mulss	XMM6, DWORD PTR [R9 + RDX + 236]
	movss	XMM4, DWORD PTR [R9 + RDX + 232]
	movaps	XMM10, XMM6
	mulss	XMM10, XMM4
	addss	XMM10, XMM8
	mulss	XMM10, XMM0
	mov	R10D, DWORD PTR [R9 + RDX + 204]
	add	R10D, DWORD PTR [RSP - 96] # 4-byte Folded Reload
	add	R10D, R11D
	movsxd	R10, R10D
	mov	R11, QWORD PTR [RSP - 88] # 8-byte Reload
	movsx	R10, WORD PTR [R11 + 2*R10]
	mov	R11, QWORD PTR [R8 + 256]
	movss	XMM8, DWORD PTR [R11 + 4*R10]
	mov	R10, QWORD PTR [RSP + 160]
	mulss	XMM8, DWORD PTR [R10]
	subss	XMM4, XMM1
	mulss	XMM4, XMM8
	addss	XMM4, XMM10
	addss	XMM4, DWORD PTR [.LCPI1_4]
	movss	XMM10, DWORD PTR [RAX + 12]
	movss	DWORD PTR [R9 + RDX + 288], XMM4
	movss	XMM11, DWORD PTR [RAX + 4]
	movaps	XMM12, XMM9
	mulss	XMM12, XMM11
	mov	R10, QWORD PTR [R9 + RDX + 248]
	movss	XMM1, DWORD PTR [R10]
	movaps	XMM13, XMM6
	mulss	XMM13, XMM1
	movss	XMM14, DWORD PTR [R9 + RDX + 228]
	movss	XMM15, DWORD PTR [R9 + RDX + 276]
	mulss	XMM15, XMM14
	addss	XMM15, XMM13
	addss	XMM15, XMM12
	mulss	XMM5, XMM0
	mulss	XMM7, XMM7
	mulss	XMM2, XMM2
	addss	XMM2, XMM7
	mulss	XMM2, XMM5
	subss	XMM10, XMM2
	mulss	XMM10, XMM3
	mulss	XMM14, XMM10
	addss	XMM14, XMM15
	mulss	XMM14, XMM0
	subss	XMM1, XMM11
	mulss	XMM1, XMM8
	addss	XMM1, XMM14
	movss	DWORD PTR [R9 + RDX + 292], XMM1
	movss	XMM2, DWORD PTR [RAX + 8]
	movaps	XMM5, XMM9
	mulss	XMM5, XMM2
	mov	R10, QWORD PTR [R9 + RDX + 264]
	movss	XMM3, DWORD PTR [R10]
	movaps	XMM7, XMM6
	mulss	XMM7, XMM3
	movss	XMM11, DWORD PTR [R9 + RDX + 224]
	movss	XMM12, DWORD PTR [R9 + RDX + 276]
	mulss	XMM12, XMM11
	subss	XMM7, XMM12
	addss	XMM7, XMM5
	mulss	XMM11, XMM10
	subss	XMM7, XMM11
	mulss	XMM7, XMM0
	subss	XMM3, XMM2
	mulss	XMM3, XMM8
	addss	XMM3, XMM7
	movss	DWORD PTR [R9 + RDX + 296], XMM3
	movss	XMM5, DWORD PTR [RAX + 12]
	addss	XMM10, XMM5
	mulss	XMM10, XMM9
	mov	RAX, QWORD PTR [R9 + RDX + 240]
	movss	XMM2, DWORD PTR [RAX]
	movss	XMM7, DWORD PTR [R9 + RDX + 276]
	addss	XMM7, XMM2
	mulss	XMM7, XMM6
	addss	XMM7, XMM10
	mulss	XMM7, XMM0
	subss	XMM2, XMM5
	mulss	XMM2, XMM8
	addss	XMM2, XMM7
	addss	XMM2, DWORD PTR [.LCPI1_4]
	movss	DWORD PTR [R9 + RDX + 300], XMM2
.LBB1_28:                               # %bres_calc.exit
                                        #   in Loop: Header=BB1_1 Depth=1
	mov	RAX, QWORD PTR [RSP + 232]
	movss	DWORD PTR [R9 + RAX + 308], XMM4
	movss	DWORD PTR [R9 + RAX + 304], XMM2
	pxor	XMM2, XMM2
	addss	XMM3, XMM2
	movss	DWORD PTR [R9 + RAX + 312], XMM3
	addss	XMM1, XMM2
	movss	DWORD PTR [R9 + RAX + 316], XMM1
	mov	R10D, DWORD PTR [R8 + 1920]
	mov	EDX, DWORD PTR [R9 + RAX + 204]
	add	EDX, R10D
	movsxd	RDX, EDX
	mov	R11, QWORD PTR [RSP + 120]
	mov	EDX, DWORD PTR [R11 + 4*RDX]
	mov	DWORD PTR [R9 + RAX + 320], EDX
	test	EDX, EDX
	jns	.LBB1_30
# BB#29:                                # %bres_calc.exit.phi-split-bb_crit_edge
                                        #   in Loop: Header=BB1_1 Depth=1
	mov	RAX, QWORD PTR [RSP + 232]
	mov	R10D, DWORD PTR [R9 + RAX + 208]
	jmp	.LBB1_31
.LBB1_30:                               #   in Loop: Header=BB1_1 Depth=1
	mov	EDX, DWORD PTR [R9 + RAX + 204]
	add	EDX, DWORD PTR [RSP - 92] # 4-byte Folded Reload
	add	EDX, R10D
	movsxd	RDX, EDX
	mov	R10, QWORD PTR [RSP - 88] # 8-byte Reload
	movsx	R10D, WORD PTR [R10 + 2*RDX]
	mov	DWORD PTR [R9 + RAX + 324], R10D
.LBB1_31:                               # %phi-split-bb
                                        #   in Loop: Header=BB1_1 Depth=1
	mov	RDX, QWORD PTR [RSP + 232]
	mov	DWORD PTR [R9 + RDX + 328], R10D
	mov	EAX, DWORD PTR [R9 + RDX + 320]
	movss	XMM3, DWORD PTR [R9 + RDX + 316]
	movss	XMM2, DWORD PTR [R9 + RDX + 312]
	movss	XMM1, DWORD PTR [R9 + RDX + 304]
	movss	XMM4, DWORD PTR [R9 + RDX + 308]
.LBB1_32:                               # %.thread
                                        #   in Loop: Header=BB1_1 Depth=1
	mov	RDX, QWORD PTR [RSP + 232]
	mov	DWORD PTR [R9 + RDX + 352], R10D
	mov	DWORD PTR [R9 + RDX + 348], EAX
	movss	DWORD PTR [R9 + RDX + 344], XMM4
	movss	DWORD PTR [R9 + RDX + 340], XMM3
	movss	DWORD PTR [R9 + RDX + 336], XMM2
	movss	DWORD PTR [R9 + RDX + 332], XMM1
	cmp	DWORD PTR [R8 + 1664], 0
	jle	.LBB1_40
# BB#33:                                # %bb.nph23
                                        #   in Loop: Header=BB1_1 Depth=1
	mov	RAX, RDX
	mov	R10D, DWORD PTR [R9 + RAX + 352]
	shl	R10D, 2
	mov	DWORD PTR [R9 + RAX + 356], R10D
	xor	R10D, R10D
.LBB1_34:                               #   in Loop: Header=BB1_1 Depth=1
	lea	EAX, DWORD PTR [R10 + 1]
	mov	RDX, QWORD PTR [RSP + 232]
	mov	DWORD PTR [R9 + RDX + 360], EAX
	cmp	DWORD PTR [R9 + RDX + 348], R10D
	jne	.LBB1_36
# BB#35:                                # %.loopexit19
                                        #   in Loop: Header=BB1_1 Depth=1
	mov	RAX, RDX
	movsxd	RDX, DWORD PTR [R9 + RAX + 356]
	mov	R10, QWORD PTR [R8 + 1280]
	movss	XMM1, DWORD PTR [R10 + 4*RDX]
	addss	XMM1, DWORD PTR [R9 + RAX + 344]
	movss	DWORD PTR [R10 + 4*RDX], XMM1
	mov	R10D, DWORD PTR [R9 + RAX + 356]
	or	R10D, 1
	movsxd	RDX, R10D
	mov	R10, QWORD PTR [R8 + 1280]
	movss	XMM1, DWORD PTR [R10 + 4*RDX]
	addss	XMM1, DWORD PTR [R9 + RAX + 340]
	movss	DWORD PTR [R10 + 4*RDX], XMM1
	mov	R10D, DWORD PTR [R9 + RAX + 356]
	or	R10D, 2
	movsxd	RDX, R10D
	mov	R10, QWORD PTR [R8 + 1280]
	movss	XMM1, DWORD PTR [R10 + 4*RDX]
	addss	XMM1, DWORD PTR [R9 + RAX + 336]
	movss	DWORD PTR [R10 + 4*RDX], XMM1
	mov	R10D, DWORD PTR [R9 + RAX + 356]
	or	R10D, 3
	movsxd	RDX, R10D
	mov	R10, QWORD PTR [R8 + 1280]
	movss	XMM1, DWORD PTR [R10 + 4*RDX]
	addss	XMM1, DWORD PTR [R9 + RAX + 332]
	movss	DWORD PTR [R10 + 4*RDX], XMM1
.LBB1_36:                               #   in Loop: Header=BB1_1 Depth=1
	mov	RAX, QWORD PTR [RSP - 16] # 8-byte Reload
	cmp	RAX, QWORD PTR [RSP + 224]
	jb	.LBB1_38
# BB#37:                                # %.SyncBB459_crit_edge
                                        #   in Loop: Header=BB1_1 Depth=1
	mov	DWORD PTR [RSP - 76], 6 # 4-byte Folded Spill
	xor	EAX, EAX
	mov	QWORD PTR [RSP - 16], RAX # 8-byte Spill
	mov	R9, RAX
	jmp	.LBB1_39
.LBB1_38:                               # %thenBB475
                                        #   in Loop: Header=BB1_1 Depth=1
	add	R9, 384
	inc	QWORD PTR [RSP - 16]    # 8-byte Folded Spill
	cmp	DWORD PTR [RSP - 76], 3 # 4-byte Folded Reload
	je	.LBB1_21
.LBB1_39:                               # %SyncBB459
                                        #   in Loop: Header=BB1_1 Depth=1
	mov	RAX, QWORD PTR [RSP + 232]
	mov	R10D, DWORD PTR [R9 + RAX + 360]
	cmp	R10D, DWORD PTR [R8 + 1664]
	jl	.LBB1_34
.LBB1_40:                               # %._crit_edge24
                                        #   in Loop: Header=BB1_1 Depth=1
	mov	RAX, QWORD PTR [RSP + 232]
	mov	RDX, QWORD PTR [R9 + RAX + 216]
	mov	R10D, 4294967295
	and	RDX, R10
	add	RDX, QWORD PTR [RCX + 56]
	mov	QWORD PTR [R9 + RAX + 368], RDX
	mov	DWORD PTR [R9 + RAX + 376], EDX
	cmp	EDX, DWORD PTR [R8 + 1408]
	mov	R10D, DWORD PTR [R9 + RAX + 352]
	movss	XMM4, DWORD PTR [R9 + RAX + 344]
	movss	XMM3, DWORD PTR [R9 + RAX + 340]
	movss	XMM1, DWORD PTR [R9 + RAX + 332]
	movss	XMM2, DWORD PTR [R9 + RAX + 336]
	jge	.LBB1_42
# BB#41:                                #   in Loop: Header=BB1_1 Depth=1
	mov	RAX, RDX
	jmp	.LBB1_23
.LBB1_42:                               # %._crit_edge30
                                        #   in Loop: Header=BB1_1 Depth=1
	mov	RAX, QWORD PTR [RSP - 16] # 8-byte Reload
	shl	RAX, 5
	mov	R10, QWORD PTR [RSP + 208]
	mov	RAX, QWORD PTR [R10 + RAX]
	mov	EDX, DWORD PTR [R8 + 1024]
	shl	EDX, 2
	cmp	EAX, EDX
	jge	.LBB1_45
# BB#43:                                # %._crit_edge30.bb.nph_crit_edge
                                        #   in Loop: Header=BB1_1 Depth=1
	mov	RDX, RAX
	.align	16, 0x90
.LBB1_44:                               # %bb.nph
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
	mov	R11, QWORD PTR [R8 + 640]
	mov	R10D, DWORD PTR [R11 + 4*R10]
	lea	R10D, DWORD PTR [RBX + 4*R10]
	movsxd	R10, R10D
	mov	R11, QWORD PTR [RSP - 64] # 8-byte Reload
	movss	XMM1, DWORD PTR [R11 + 4*R10]
	movsxd	RAX, EAX
	mov	RBX, QWORD PTR [R8 + 1280]
	addss	XMM1, DWORD PTR [RBX + 4*RAX]
	movss	DWORD PTR [R11 + 4*R10], XMM1
	mov	EAX, EDX
	add	RAX, QWORD PTR [RCX + 56]
	mov	EDX, DWORD PTR [R8 + 1024]
	shl	EDX, 2
	cmp	EAX, EDX
	mov	RDX, RAX
	jl	.LBB1_44
.LBB1_45:                               # %.loopexit
                                        #   in Loop: Header=BB1_1 Depth=1
	mov	RAX, QWORD PTR [RSP - 16] # 8-byte Reload
	cmp	RAX, QWORD PTR [RSP + 224]
	jb	.LBB1_46
# BB#48:                                # %SyncBB457
	pop	RBX
	pop	R12
	pop	R13
	pop	R14
	pop	R15
	pop	RBP
	ret
.Ltmp1:
	.size	__Vectorized_.op_opencl_bres_calc, .Ltmp1-__Vectorized_.op_opencl_bres_calc

	.type	opencl_op_opencl_bres_calc_local_ind_arg0_map,@object # @opencl_op_opencl_bres_calc_local_ind_arg0_map
	.local	opencl_op_opencl_bres_calc_local_ind_arg0_map # @opencl_op_opencl_bres_calc_local_ind_arg0_map
	.comm	opencl_op_opencl_bres_calc_local_ind_arg0_map,8,8
	.type	opencl_op_opencl_bres_calc_local_ind_arg0_size,@object # @opencl_op_opencl_bres_calc_local_ind_arg0_size
	.local	opencl_op_opencl_bres_calc_local_ind_arg0_size # @opencl_op_opencl_bres_calc_local_ind_arg0_size
	.comm	opencl_op_opencl_bres_calc_local_ind_arg0_size,4,4
	.type	opencl_op_opencl_bres_calc_local_ind_arg1_map,@object # @opencl_op_opencl_bres_calc_local_ind_arg1_map
	.local	opencl_op_opencl_bres_calc_local_ind_arg1_map # @opencl_op_opencl_bres_calc_local_ind_arg1_map
	.comm	opencl_op_opencl_bres_calc_local_ind_arg1_map,8,8
	.type	opencl_op_opencl_bres_calc_local_ind_arg1_size,@object # @opencl_op_opencl_bres_calc_local_ind_arg1_size
	.local	opencl_op_opencl_bres_calc_local_ind_arg1_size # @opencl_op_opencl_bres_calc_local_ind_arg1_size
	.comm	opencl_op_opencl_bres_calc_local_ind_arg1_size,4,4
	.type	opencl_op_opencl_bres_calc_local_ind_arg2_map,@object # @opencl_op_opencl_bres_calc_local_ind_arg2_map
	.local	opencl_op_opencl_bres_calc_local_ind_arg2_map # @opencl_op_opencl_bres_calc_local_ind_arg2_map
	.comm	opencl_op_opencl_bres_calc_local_ind_arg2_map,8,8
	.type	opencl_op_opencl_bres_calc_local_ind_arg2_size,@object # @opencl_op_opencl_bres_calc_local_ind_arg2_size
	.local	opencl_op_opencl_bres_calc_local_ind_arg2_size # @opencl_op_opencl_bres_calc_local_ind_arg2_size
	.comm	opencl_op_opencl_bres_calc_local_ind_arg2_size,4,4
	.type	opencl_op_opencl_bres_calc_local_ind_arg3_map,@object # @opencl_op_opencl_bres_calc_local_ind_arg3_map
	.local	opencl_op_opencl_bres_calc_local_ind_arg3_map # @opencl_op_opencl_bres_calc_local_ind_arg3_map
	.comm	opencl_op_opencl_bres_calc_local_ind_arg3_map,8,8
	.type	opencl_op_opencl_bres_calc_local_ind_arg3_size,@object # @opencl_op_opencl_bres_calc_local_ind_arg3_size
	.local	opencl_op_opencl_bres_calc_local_ind_arg3_size # @opencl_op_opencl_bres_calc_local_ind_arg3_size
	.comm	opencl_op_opencl_bres_calc_local_ind_arg3_size,4,4
	.type	opencl_op_opencl_bres_calc_local_ind_arg0_s,@object # @opencl_op_opencl_bres_calc_local_ind_arg0_s
	.local	opencl_op_opencl_bres_calc_local_ind_arg0_s # @opencl_op_opencl_bres_calc_local_ind_arg0_s
	.comm	opencl_op_opencl_bres_calc_local_ind_arg0_s,8,8
	.type	opencl_op_opencl_bres_calc_local_ind_arg1_s,@object # @opencl_op_opencl_bres_calc_local_ind_arg1_s
	.local	opencl_op_opencl_bres_calc_local_ind_arg1_s # @opencl_op_opencl_bres_calc_local_ind_arg1_s
	.comm	opencl_op_opencl_bres_calc_local_ind_arg1_s,8,8
	.type	opencl_op_opencl_bres_calc_local_ind_arg2_s,@object # @opencl_op_opencl_bres_calc_local_ind_arg2_s
	.local	opencl_op_opencl_bres_calc_local_ind_arg2_s # @opencl_op_opencl_bres_calc_local_ind_arg2_s
	.comm	opencl_op_opencl_bres_calc_local_ind_arg2_s,8,8
	.type	opencl_op_opencl_bres_calc_local_ind_arg3_s,@object # @opencl_op_opencl_bres_calc_local_ind_arg3_s
	.local	opencl_op_opencl_bres_calc_local_ind_arg3_s # @opencl_op_opencl_bres_calc_local_ind_arg3_s
	.comm	opencl_op_opencl_bres_calc_local_ind_arg3_s,8,8
	.type	opencl_op_opencl_bres_calc_local_nelems2,@object # @opencl_op_opencl_bres_calc_local_nelems2
	.local	opencl_op_opencl_bres_calc_local_nelems2 # @opencl_op_opencl_bres_calc_local_nelems2
	.comm	opencl_op_opencl_bres_calc_local_nelems2,4,4
	.type	opencl_op_opencl_bres_calc_local_ncolor,@object # @opencl_op_opencl_bres_calc_local_ncolor
	.local	opencl_op_opencl_bres_calc_local_ncolor # @opencl_op_opencl_bres_calc_local_ncolor
	.comm	opencl_op_opencl_bres_calc_local_ncolor,4,4
	.type	opencl_op_opencl_bres_calc_local_nelem,@object # @opencl_op_opencl_bres_calc_local_nelem
	.local	opencl_op_opencl_bres_calc_local_nelem # @opencl_op_opencl_bres_calc_local_nelem
	.comm	opencl_op_opencl_bres_calc_local_nelem,4,4
	.type	opencl_op_opencl_bres_calc_local_offset_b,@object # @opencl_op_opencl_bres_calc_local_offset_b
	.local	opencl_op_opencl_bres_calc_local_offset_b # @opencl_op_opencl_bres_calc_local_offset_b
	.comm	opencl_op_opencl_bres_calc_local_offset_b,4,4

	.section	.note.GNU-stack,"",@progbits
