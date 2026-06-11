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

// genSSE emits adlerSSE: 16 bytes per block.
func genSSE(f *emit.File) {
	w := f.Data("wSSE", weights(16))

	b := amd64.NewFunc("adlerSSE", sig(), 0)
	b.LoadArg("s1", "AX").LoadArg("s2", "DX").
		LoadArg("p_base", "SI").LoadArg("n", "CX").
		// vs1 = {s1,0,0,0}, vs2 = {s2,0,0,0}; X2 = zero, X6 = weights.
		Raw("MOVD AX, X0").
		Raw("MOVD DX, X1").
		Raw("PXOR X2, X2").
		Raw("MOVOU %s+0(SB), X6", w).
		Raw("TESTQ CX, CX").Raw("JZ done").
		Label("loop").
		Raw("MOVOU (SI), X3").              // 16 source bytes
		// vs2 += vs1 << 4 (each block adds 16*running-s1 to s2).
		Raw("MOVO X0, X4").Raw("PSLLL $4, X4").Raw("PADDD X4, X1").
		// vs1 += Σbytes via PSADBW (two 64-bit lane sums).
		Raw("MOVO X3, X5").Raw("PSADBW X2, X5").Raw("PADDD X5, X0").
		// vs2 += Σ weight_i*byte_i: PMADDUBSW(bytes, weights) -> 8 signed
		// 16-bit pairwise sums (each in 0..8160, so non-negative). Go's amd64
		// assembler has no SSE PMADDWD, so widen the 8 words to 32-bit by
		// zero-extending the low and high halves (PUNPCKLWD/PUNPCKHWD against
		// the zero register X2) and accumulate both into vs2.
		// PMADDUBSW dst, src treats dst's bytes as UNSIGNED and src's as
		// SIGNED, so the bytes (0..255) must be the dst and the weights (1..16,
		// positive) the src.
		Raw("MOVO X3, X5").Raw("PMADDUBSW X6, X5").
		Raw("MOVO X5, X4").Raw("PUNPCKLWL X2, X4").Raw("PADDD X4, X1").
		Raw("PUNPCKHWL X2, X5").Raw("PADDD X5, X1").
		Raw("ADDQ $16, SI").Raw("DECQ CX").Raw("JNZ loop").
		Label("done").
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

// genAVX2 emits adlerAVX2: 32 bytes per block. The two 128-bit lanes carry
// weights 32..17 (low) and 16..1 (high) so a 32-byte block's byte i gets weight
// 32-i; vs1/vs2 are 8x32-bit ymm accumulators reduced at the end.
func genAVX2(f *emit.File) {
	w := f.Data("wAVX2", weights(32))
	one := f.Data("oneAVX2", ones16(32))

	b := amd64.NewFunc("adlerAVX2", sig(), 0)
	b.LoadArg("s1", "AX").LoadArg("s2", "DX").
		LoadArg("p_base", "SI").LoadArg("n", "CX").
		// Y0 = vs1 = {s1,0,…}, Y1 = vs2 = {s2,0,…}; Y2 zero, Y6 weights, Y7 ones.
		Raw("VPXOR Y0, Y0, Y0").
		Raw("VPXOR Y1, Y1, Y1").
		Raw("VPXOR Y2, Y2, Y2").
		Raw("MOVD AX, X0").
		Raw("MOVD DX, X1").
		Raw("VMOVDQU %s+0(SB), Y6", w).
		Raw("VMOVDQU %s+0(SB), Y7", one).
		Raw("TESTQ CX, CX").Raw("JZ vdone").
		Label("vloop").
		Raw("VMOVDQU (SI), Y3").                       // 32 source bytes
		// vs2 += vs1 << 5 (each 32-byte block adds 32*running-s1 to s2; the
		// running s1 is the sum of all vs1 lanes, so the per-lane << 5 sums to
		// 32*s1 after the final horizontal reduction).
		Raw("VPSLLD $5, Y0, Y4").Raw("VPADDD Y4, Y1, Y1").
		// vs1 += Σbytes (VPSADBW -> four 64-bit lane sums).
		Raw("VPSADBW Y2, Y3, Y5").Raw("VPADDD Y5, Y0, Y0").
		// vs2 += Σ weight_i*byte_i. VPMADDUBSW's middle operand is the unsigned
		// one, so the bytes (Y3) go there and the signed weights (Y6) first.
		Raw("VPMADDUBSW Y6, Y3, Y5").
		Raw("VPMADDWD Y7, Y5, Y5").Raw("VPADDD Y5, Y1, Y1").
		Raw("ADDQ $32, SI").Raw("DECQ CX").Raw("JNZ vloop").
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
