//go:build ignore

// Command gen produces adler32_ppc64le.s with go-asmgen: a VSX/AltiVec
// Adler-32 inner sum for ppc64le. Adler-32 is s1 = 1 + Σbytes (mod 65521) and
// s2 = Σ running-s1 (mod 65521); the caller (adler32.go) keeps every kernel
// call within nmax=5552 bytes so the 32-bit accumulators never overflow before
// the modular reduction, and feeds whole 16-byte blocks.
//
// Go's ppc64 assembler does not expose the single-instruction multiply-sum
// (vmsumubm) or the byte horizontal sum (vsum4ubs), so the weighted sum is
// built from the widening even/odd byte multiplies VMULEUB / VMULOUB (unsigned
// byte*byte -> halfword) and accumulated, overflow-safe, in 4x32-bit word
// vector lanes that are horizontally reduced once at the end. This mirrors the
// amd64 SSE kernel (carry-deferred): instead of the latency-bound per-block
// vs2 += 16*s1, the running per-lane vs1 is accumulated into vcsum each block
// and multiplied by 16 once at the end. The per-lane sums are linear, so
// 16*Σ horizontal(vs1_before) = 16*horizontal(Σ vs1_before) = 16*horizontal
// (vcsum): exact, and equal to the scalar reference.
//
// Per 16-byte block v (V0), weights w = {16,15,…,1} (Vw), ones-bytes = 1 (Vob),
// ones-halfwords = 1 (Voh):
//
//	vcsum += vs1                                   (deferred 16*running-s1)
//	bytes  = VMULEUB(v,Vob)+VMULOUB(v,Vob)         8 halfwords, each <= 510
//	weigh  = VMULEUB(v,Vw) +VMULOUB(v,Vw)          8 halfwords, each <= 8160
//	vs1   += widen(bytes) to 4 words via VMULEUH/VMULOUH(.,Voh)+VADDUWM
//	vs2   += widen(weigh) likewise
//
// VMULEUB/VMULOUB are unsigned and widen byte*byte (<= 16*255=4080) to 16-bit,
// their pairwise sum (<= 8160) stays in 16-bit; widening those halfwords to
// 32-bit before accumulating means the word lanes (summed over <= 347 blocks)
// cannot overflow. All loads use LXVB16X so the in-register byte order matches
// memory regardless of the little-endian element numbering.
//
// Signature: adlerVSX(s1, s2 uint32, p []byte, n int) (uint32, uint32) where n
// is the number of whole 16-byte blocks. Run: go run adler32_ppc64le_gen.go
package main

import (
	"fmt"
	"os"

	"github.com/go-asmgen/asmgen/abi"
	"github.com/go-asmgen/asmgen/emit"
	"github.com/go-asmgen/asmgen/ppc64"
)

// weights returns the 16-byte descending weight vector 16,15,…,1 so byte i of a
// 16-byte block is multiplied by weight 16-i.
func weights() []byte {
	b := make([]byte, 16)
	for i := 0; i < 16; i++ {
		b[i] = byte(16 - i)
	}
	return b
}

// block emits the per-16-byte-block body for adlerVSX from the source pointer
// offset (a Plan 9 indexed address built by the caller). The 16 bytes are
// loaded into V0 via LXVB16X (VS32), then folded into the vs1/vs2/vcsum word
// accumulators as described in the file header.
func block(b *ppc64.Builder, idxReg string) *ppc64.Builder {
	return b.
		// load 16 source bytes into V0 (VS32); LXVB16X keeps memory byte order.
		Raw("LXVB16X (R5)(%s), VS32", idxReg).
		// vcsum += running per-lane vs1 (deferred 16*s1 carry).
		Raw("VADDUWM V10, V12, V12").
		// byte sum: even/odd byte products by 1, summed -> 8 halfwords (<=510).
		Raw("VMULEUB V0, V13, V2").Raw("VMULOUB V0, V13, V3").Raw("VADDUHM V2, V3, V2").
		// widen those halfwords to 4 words and add into vs1.
		Raw("VMULEUH V2, V15, V6").Raw("VMULOUH V2, V15, V7").Raw("VADDUWM V6, V7, V6").
		Raw("VADDUWM V10, V6, V10").
		// weighted sum: even/odd byte*weight, summed -> 8 halfwords (<=8160).
		Raw("VMULEUB V0, V14, V4").Raw("VMULOUB V0, V14, V5").Raw("VADDUHM V4, V5, V4").
		// widen to 4 words and add into vs2.
		Raw("VMULEUH V4, V15, V8").Raw("VMULOUH V4, V15, V9").Raw("VADDUWM V8, V9, V8").
		Raw("VADDUWM V11, V8, V11")
}

// reduce horizontally sums the 4 word lanes of vReg into the low word of vReg
// using two VSLDOI rotate-and-adds (8- then 4-byte octet shifts), leaving the
// total replicated; the caller extracts it with MFVSRWZ.
func reduce(b *ppc64.Builder, vReg, scratch string) *ppc64.Builder {
	return b.
		Raw("VSLDOI $8, %s, %s, %s", vReg, vReg, scratch).Raw("VADDUWM %s, %s, %s", vReg, scratch, vReg).
		Raw("VSLDOI $4, %s, %s, %s", vReg, vReg, scratch).Raw("VADDUWM %s, %s, %s", vReg, scratch, vReg)
}

func main() {
	f := emit.NewFile("ppc64le")

	w := f.Data("wVSX", weights())

	sig := abi.LayoutArgs(
		[]abi.Arg{
			abi.Scalar("s1", abi.Uint32), abi.Scalar("s2", abi.Uint32),
			abi.Slice("p"), abi.Scalar("n", abi.Int64),
		},
		[]abi.Arg{abi.Scalar("ret", abi.Uint32), abi.Scalar("ret1", abi.Uint32)},
	)

	// R3=s1, R4=s2, R5=ptr, R6=n(blocks), R7=offset, R8=scratch/weights addr.
	b := ppc64.NewFunc("adlerVSX", sig, 0)
	b.LoadArg("s1", "R3").LoadArg("s2", "R4").
		LoadArg("p_base", "R5").LoadArg("n", "R6").
		// Zero the word accumulators vs1=V10, vs2=V11, vcsum=V12.
		Raw("VSPLTISW $0, V10").Raw("VSPLTISW $0, V11").Raw("VSPLTISW $0, V12").
		// Seed vs1 low word = s1, vs2 low word = s2 via GPR -> VSR move + place.
		// MTVSRWZ puts the GPR in word element 1 (low word of doubleword 0); the
		// horizontal reduce sums all lanes so the exact lane is immaterial.
		Raw("MTVSRWZ R3, VS42").Raw("MTVSRWZ R4, VS43").
		// Vob = ones bytes (1), Voh = ones halfwords (1).
		Raw("VSPLTISB $1, V13").Raw("VSPLTISH $1, V15").
		// Load the descending weight vector into V14 (VS46).
		Raw("MOVD $%s(SB), R8", w).Raw("LXVB16X (R8)(R0), VS46").
		// offset = 0.
		Raw("MOVD $0, R7").
		Raw("CMP R6, $0").Raw("BEQ done").
		Label("loop")
	block(b, "R7").
		Raw("ADD $16, R7").Raw("ADD $-1, R6").Raw("CMP R6, $0").Raw("BNE loop").
		Label("done").
		// Fold the deferred carry: vs2 += 16 * vcsum (each block adds 16*s1).
		Raw("VSPLTISW $4, V0").Raw("VSLW V12, V0, V12").Raw("VADDUWM V11, V12, V11")
	reduce(b, "V10", "V0")
	reduce(b, "V11", "V0").
		Raw("MFVSRWZ V10, R3").Raw("MFVSRWZ V11, R4").
		Raw("MOVWZ R3, R3").Raw("MOVWZ R4, R4").
		StoreRet("R3", "ret").StoreRet("R4", "ret1").Ret()
	f.Add(b.Func())

	if err := os.WriteFile("adler32_ppc64le.s", []byte(f.String()), 0o644); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	fmt.Println("wrote adler32_ppc64le.s")
}
