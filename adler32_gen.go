//go:build ignore

// Command gen produces adler32_amd64.s with go-asmgen: a vectorised Adler-32
// inner sum. Adler-32 is s1 = 1 + Σbytes (mod 65521) and s2 = Σ running-s1
// (mod 65521); the caller (adler32.go) keeps every kernel call within nmax=5552
// bytes so the 32-bit accumulators never overflow before the modular reduction,
// and feeds whole 16-byte (SSE) / 32-byte (AVX2) blocks.
//
// Method (the classic zlib/Chromium SIMD Adler-32). Across the blocks of one
// chunk we keep, in vector registers, vs1 = the running byte sum and vs2 = the
// running second sum, plus a per-block carry of the running s1:
//
//	per 16-byte block v:
//	    vs2  += vs1 << 4              (each block adds 16*s1_running to s2)
//	    vs1  += PSADBW(v, 0)         (Σ of the block's 16 bytes)
//	    vs2  += PMADDWD(PMADDUBSW(v, {16,15,…,1}), {1,…})   (Σ weight_i*byte_i)
//
// PMADDUBSW multiplies the unsigned bytes by the signed weights 16..1 into 8
// 16-bit pairwise sums (max 16*255*2 = 8160 < 2^15, no overflow); PMADDWD then
// widens-and-pair-adds those into 4 lanes of 32-bit, accumulated into vs2.
// PSADBW sums the 16 bytes into two 64-bit lanes (added into vs1). At the end
// the four 32-bit lanes of vs1/vs2 are horizontally reduced and combined with
// the incoming scalars. The AVX2 path is the same over 32-byte blocks with two
// 128-bit lanes whose weights run 32..1; its lane sums are reduced the same way.
//
// Signature: adlerSSE(s1, s2 uint32, p []byte, n int) (uint32, uint32) where n
// is the number of whole 16-byte blocks; adlerAVX2 is the 32-byte analogue.
//
// Run: go run adler32_gen.go
package main

import (
	"fmt"
	"os"

	"github.com/go-asmgen/asmgen/abi"
	"github.com/go-asmgen/asmgen/amd64"
	"github.com/go-asmgen/asmgen/emit"
)

func rep(v []byte, times int) []byte {
	var b []byte
	for i := 0; i < times; i++ {
		b = append(b, v...)
	}
	return b
}

// weights returns the n-byte descending weight vector n, n-1, …, 1 used by
// PMADDUBSW so byte i of an n-byte block is multiplied by weight n-i.
func weights(n int) []byte {
	b := make([]byte, n)
	for i := 0; i < n; i++ {
		b[i] = byte(n - i)
	}
	return b
}

// ones16 is the all-ones 16-bit multiplier for PMADDWD: it widens 8 signed
// 16-bit values to 4 lanes of 32-bit while pair-adding them (×1).
func ones16(nbytes int) []byte {
	b := make([]byte, nbytes)
	for i := 0; i+1 < nbytes; i += 2 {
		b[i] = 1
		b[i+1] = 0
	}
	return b
}

func sig() abi.Signature {
	return abi.LayoutArgs(
		[]abi.Arg{
			abi.Scalar("s1", abi.Uint32), abi.Scalar("s2", abi.Uint32),
			abi.Slice("p"), abi.Scalar("n", abi.Int64),
		},
		[]abi.Arg{abi.Scalar("o1", abi.Uint32), abi.Scalar("o2", abi.Uint32)},
	)
}

func main() {
	f := emit.NewFile("amd64")
	genSSE(f)
	genAVX2(f)
	if err := os.WriteFile("adler32_amd64.s", []byte(f.String()), 0o644); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	fmt.Println("wrote adler32_amd64.s")
}

// sseBlock emits the per-16-byte-block body for adlerSSE using the given source
// register, in carry-deferred form: instead of the latency-bound vs2 += vs1<<4
// each block, accumulate the running s1 into vcsum (X7) before folding this
// block, and multiply vcsum by 16 once at the end. This removes the per-block
// PSLLL from the critical path and lets the independent unrolled bodies overlap.
func sseBlock(b *amd64.Builder, src string) *amd64.Builder {
	return b.
		// vcsum += running-s1 (the s1 in effect before this block; *16 at end).
		Raw("PADDD X0, X7").
		// vs1 += Σbytes via PSADBW (two 64-bit lane sums). MOVOU: the source is
		// not guaranteed 16-byte aligned (real hardware faults on MOVAPS).
		Raw("MOVOU %s, X5", src).Raw("PSADBW X2, X5").Raw("PADDD X5, X0").
		// vs2 += Σ weight_i*byte_i. PMADDUBSW(bytes, weights) -> 8 signed 16-bit
		// pairwise sums (each 0..8160, non-negative). Go's amd64 assembler has no
		// SSE PMADDWD, so widen the 8 words to 32-bit by zero-extending the low
		// and high halves (PUNPCKLWL/PUNPCKHWL against the zero X2) into vs2.
		// PMADDUBSW dst, src treats dst's bytes as UNSIGNED and src's as SIGNED,
		// so the bytes (0..255) are the dst and the weights (1..16, positive) src.
		Raw("MOVOU %s, X5", src).Raw("PMADDUBSW X6, X5").
		Raw("MOVO X5, X4").Raw("PUNPCKLWL X2, X4").Raw("PADDD X4, X1").
		Raw("PUNPCKHWL X2, X5").Raw("PADDD X5, X1")
}

// genSSE emits adlerSSE: 16 bytes per block, main loop unrolled 2x with a 1x
// remainder, carry-deferred (see sseBlock).
func genSSE(f *emit.File) {
	w := f.Data("wSSE", weights(16))

	b := amd64.NewFunc("adlerSSE", sig(), 0)
	b.LoadArg("s1", "AX").LoadArg("s2", "DX").
		LoadArg("p_base", "SI").LoadArg("n", "CX").
		// vs1 = {s1,0,0,0}, vs2 = {s2,0,0,0}; X2 = zero, X6 = weights, X7 = vcsum.
		Raw("MOVD AX, X0").
		Raw("MOVD DX, X1").
		Raw("PXOR X2, X2").
		Raw("PXOR X7, X7").
		Raw("MOVOU %s+0(SB), X6", w).
		// Main loop: 2 blocks (32 bytes) per iteration while at least 2 remain.
		Raw("MOVQ CX, BX").Raw("ANDQ $-2, BX").Raw("JZ tail").
		Label("loop2")
	sseBlock(b, "(SI)")
	sseBlock(b, "16(SI)").
		Raw("ADDQ $32, SI").Raw("SUBQ $2, BX").Raw("JNZ loop2").
		Label("tail").
		// Remainder: the trailing 0..1 block.
		Raw("ANDQ $1, CX").Raw("JZ done").
		Label("loop1")
	sseBlock(b, "(SI)").
		Raw("ADDQ $16, SI").Raw("DECQ CX").Raw("JNZ loop1").
		Label("done").
		// Fold the deferred carry: vs2 += 16 * vcsum (each block's running-s1
		// contributes 16*s1 to s2).
		Raw("PSLLL $4, X7").Raw("PADDD X7, X1").
		// Horizontal reduce vs1 (X0) and vs2 (X1): sum the 4 32-bit lanes.
		Raw("MOVO X0, X4").Raw("PSRLDQ $8, X4").Raw("PADDD X4, X0").
		Raw("MOVO X0, X4").Raw("PSRLDQ $4, X4").Raw("PADDD X4, X0").
		Raw("MOVD X0, AX").
		Raw("MOVO X1, X4").Raw("PSRLDQ $8, X4").Raw("PADDD X4, X1").
		Raw("MOVO X1, X4").Raw("PSRLDQ $4, X4").Raw("PADDD X4, X1").
		Raw("MOVD X1, DX").
		StoreRet("AX", "o1").StoreRet("DX", "o2").Ret()
	f.Add(b.Func())
}

// avx2Block emits the per-32-byte-block body for adlerAVX2 from source register
// src, in carry-deferred form (see sseBlock): accumulate the running s1 into
// vcsum (Y8) before folding this block and multiply vcsum by 32 once at the end,
// dropping the latency-bound per-block VPSLLD from the inner loop so the unrolled
// bodies overlap. scr is a scratch ymm (Y4/Y5) used for the SAD/MADD results.
func avx2Block(b *amd64.Builder, src, scr string) *amd64.Builder {
	return b.
		// vcsum += running-s1 (the s1 in effect before this block; *32 at end).
		Raw("VPADDD Y0, Y8, Y8").
		// vs1 += Σbytes (VPSADBW -> four 64-bit lane sums).
		Raw("VPSADBW Y2, %s, %s", src, scr).Raw("VPADDD %s, Y0, Y0", scr).
		// vs2 += Σ weight_i*byte_i. VPMADDUBSW's middle operand is the unsigned
		// one, so the bytes (src) go there and the signed weights (Y6) first.
		Raw("VPMADDUBSW Y6, %s, %s", src, scr).
		Raw("VPMADDWD Y7, %s, %s", scr, scr).Raw("VPADDD %s, Y1, Y1", scr)
}

// genAVX2 emits adlerAVX2: 32 bytes per block. The two 128-bit lanes carry
// weights 32..17 (low) and 16..1 (high) so a 32-byte block's byte i gets weight
// 32-i; vs1/vs2 are 8x32-bit ymm accumulators reduced at the end. The main loop
// is unrolled 4x (128 bytes/iter, four independent block bodies) with a 1x
// remainder; this matches the mhr3 unroll-and-overlap and hits the Zen3 vector
// ALU port ceiling (~18 bytes/cycle, modelled with llvm-mca).
func genAVX2(f *emit.File) {
	w := f.Data("wAVX2", weights(32))
	one := f.Data("oneAVX2", ones16(32))

	b := amd64.NewFunc("adlerAVX2", sig(), 0)
	b.LoadArg("s1", "AX").LoadArg("s2", "DX").
		LoadArg("p_base", "SI").LoadArg("n", "CX").
		// Y0 = vs1 = {s1,0,…}, Y1 = vs2 = {s2,0,…}; Y2 zero, Y6 weights, Y7 ones,
		// Y8 = vcsum (deferred 32*running-s1 carry).
		Raw("VPXOR Y0, Y0, Y0").
		Raw("VPXOR Y1, Y1, Y1").
		Raw("VPXOR Y2, Y2, Y2").
		Raw("VPXOR Y8, Y8, Y8").
		Raw("MOVD AX, X0").
		Raw("MOVD DX, X1").
		Raw("VMOVDQU %s+0(SB), Y6", w).
		Raw("VMOVDQU %s+0(SB), Y7", one).
		// Main loop: 4 blocks (128 bytes) per iteration while at least 4 remain.
		Raw("MOVQ CX, BX").Raw("ANDQ $-4, BX").Raw("JZ vtail").
		Label("vloop4").
		Raw("VMOVDQU 0(SI), Y3").
		Raw("VMOVDQU 32(SI), Y9").
		Raw("VMOVDQU 64(SI), Y10").
		Raw("VMOVDQU 96(SI), Y11")
	avx2Block(b, "Y3", "Y5")
	avx2Block(b, "Y9", "Y4")
	avx2Block(b, "Y10", "Y5")
	avx2Block(b, "Y11", "Y4").
		Raw("ADDQ $128, SI").Raw("SUBQ $4, BX").Raw("JNZ vloop4").
		Label("vtail").
		// Remainder: the trailing 0..3 blocks, one at a time.
		Raw("ANDQ $3, CX").Raw("JZ vfold").
		Label("vloop1").
		Raw("VMOVDQU (SI), Y3")
	avx2Block(b, "Y3", "Y5").
		Raw("ADDQ $32, SI").Raw("DECQ CX").Raw("JNZ vloop1").
		Label("vfold").
		// Fold the deferred carry: vs2 += 32 * vcsum.
		Raw("VPSLLD $5, Y8, Y8").Raw("VPADDD Y8, Y1, Y1").
		Label("vdone").
		// Reduce the 256-bit accumulators to a scalar: fold the high 128-bit
		// lane into the low one, then horizontally sum the 4 32-bit lanes.
		Raw("VEXTRACTI128 $1, Y0, X4").Raw("VPADDD X4, X0, X0").
		Raw("VEXTRACTI128 $1, Y1, X4").Raw("VPADDD X4, X1, X1").
		Raw("VZEROUPPER").
		Raw("MOVO X0, X4").Raw("PSRLDQ $8, X4").Raw("PADDD X4, X0").
		Raw("MOVO X0, X4").Raw("PSRLDQ $4, X4").Raw("PADDD X4, X0").
		Raw("MOVD X0, AX").
		Raw("MOVO X1, X4").Raw("PSRLDQ $8, X4").Raw("PADDD X4, X1").
		Raw("MOVO X1, X4").Raw("PSRLDQ $4, X4").Raw("PADDD X4, X1").
		Raw("MOVD X1, DX").
		StoreRet("AX", "o1").StoreRet("DX", "o2").Ret()
	f.Add(b.Func())
}
