//go:build ignore

// Command gen produces adler32_s390x.s with go-asmgen: a vector-facility
// Adler-32 inner sum for s390x (z/Architecture, BIG-ENDIAN). Adler-32 is
// s1 = 1 + Σbytes (mod 65521) and s2 = Σ running-s1 (mod 65521); the caller
// (adler32.go) keeps every kernel call within nmax=5552 bytes so the 32-bit
// accumulators never overflow before the modular reduction, and feeds whole
// 16-byte blocks.
//
// s390x has the multiply-sum building blocks the task calls for: VSUMB sums each
// group of four byte elements into a word (the byte sum), and the even/odd byte
// multiplies VMLEB / VMLOB give the weighted products; the final per-register
// horizontal reduction uses VSUMQF (sum four words into the low word). The
// kernel mirrors the amd64 SSE design (carry-deferred): the running per-lane vs1
// is accumulated into vcsum each block and multiplied by 16 once at the end,
// since 16*Σ horizontal(vs1_before) = 16*horizontal(vcsum) exactly. All sums are
// kept in 4x32-bit word lanes and reduced once, so the result equals the scalar
// reference.
//
// BIG-ENDIAN note: VL loads the 16 source bytes so byte i of memory becomes
// element i of the vector (element 0 leftmost). The descending weight vector is
// emitted as the bytes 16,15,…,1, so weight element i = 16-i lines up with
// source element i under the same load: byte i is multiplied by weight 16-i,
// matching the standard library's positional weighting. The byte sum (VSUMB)
// and the word reductions (VSUMQF) are position-independent, hence endian-
// neutral; only the weight alignment depends on lane order, and it is correct
// because both the bytes and the weights are loaded big-endian. The ascending-
// byte test (bytesAsc) is the positional check that proves it.
//
// Per 16-byte block v (V0), weights w = {16,15,…,1} (V18):
//
//	vcsum += vs1                                        (deferred 16*running-s1)
//	bytesum_words = VSUMB(v, 0)                          4 words (Σ of 4 bytes)
//	vs1 += bytesum_words
//	weigh_h = VAH(VMLEB(v,w), VMLOB(v,w))               8 halfwords, each <=8160
//	weigh_words = VAF(VUPLHH(weigh_h), VUPLLH(weigh_h)) 4 words
//	vs2 += weigh_words
//
// VMLEB/VMLOB are unsigned even/odd byte multiplies (byte*byte <= 16*255 = 4080,
// halfword); their VAH sum (<= 8160) stays in 16-bit; unpacking to 32-bit before
// accumulating keeps the word lanes (summed over <= 347 blocks) from overflowing.
//
// Signature: adlerVX(s1, s2 uint32, p []byte, n int) (uint32, uint32) where n is
// the number of whole 16-byte blocks. Run: go run adler32_s390x_gen.go
package main

import (
	"fmt"
	"os"

	"github.com/go-asmgen/asmgen/abi"
	"github.com/go-asmgen/asmgen/emit"
	"github.com/go-asmgen/asmgen/s390x"
)

// weights returns the 16-byte descending weight vector 16,15,…,1 so element i
// of the big-endian load is multiplied by weight 16-i.
func weights() []byte {
	b := make([]byte, 16)
	for i := 0; i < 16; i++ {
		b[i] = byte(16 - i)
	}
	return b
}

// block emits the per-16-byte-block body for adlerVX, loading the 16 source
// bytes from the address in R2 into V0 and folding them into the vs1/vs2/vcsum
// word accumulators (V20/V21/V22) as described in the file header.
func block(b *s390x.Builder) *s390x.Builder {
	return b.
		// load 16 source bytes into V0 (big-endian: byte i -> element i).
		Raw("VL (R2), V0").
		// vcsum += running per-lane vs1 (deferred 16*s1 carry).
		Raw("VAF V20, V22, V22").
		// byte sum: VSUMB sums each group of 4 bytes into a word (4 words).
		Raw("VSUMB V0, V23, V6").Raw("VAF V20, V6, V20").
		// weighted sum: even/odd byte*weight -> halfwords, summed (<=8160).
		Raw("VMLEB V0, V18, V2").Raw("VMLOB V0, V18, V3").Raw("VAH V2, V3, V2").
		// unpack the 8 halfwords to two sets of 4 words and add into vs2.
		Raw("VUPLHH V2, V4").Raw("VUPLLH V2, V5").Raw("VAF V4, V5, V4").
		Raw("VAF V21, V4, V21")
}

func main() {
	f := emit.NewFile("s390x")

	w := f.Data("wVX", weights())

	sig := abi.LayoutArgs(
		[]abi.Arg{
			abi.Scalar("s1", abi.Uint32), abi.Scalar("s2", abi.Uint32),
			abi.Slice("p"), abi.Scalar("n", abi.Int64),
		},
		[]abi.Arg{abi.Scalar("ret", abi.Uint32), abi.Scalar("ret1", abi.Uint32)},
	)

	// R1=s1, R3=s2, R2=ptr, R4=n(blocks), R5=weights addr.
	b := s390x.NewFunc("adlerVX", sig, 0)
	b.LoadArg("s1", "R1").LoadArg("s2", "R3").
		LoadArg("p_base", "R2").LoadArg("n", "R4").
		// Zero the word accumulators vs1=V20, vs2=V21, vcsum=V22 and the
		// all-zero helper V23 (the addend operand for VSUMB / VSUMQF).
		Raw("VZERO V20").Raw("VZERO V21").Raw("VZERO V22").Raw("VZERO V23").
		// Seed vs1 low word = s1, vs2 low word = s2 (element 3 is the low word in
		// big-endian; the horizontal reduce sums all lanes so the lane is
		// immaterial). VLVGF $idx, Rsrc, Vdst inserts the GPR word at element idx.
		Raw("VLVGF $3, R1, V20").Raw("VLVGF $3, R3, V21").
		// Load the descending weight vector into V18.
		Raw("MOVD $%s(SB), R5", w).Raw("VL (R5), V18").
		Raw("CMPBEQ R4, $0, done").
		Label("loop")
	block(b).
		Raw("MOVD $16, R5").Raw("ADD R5, R2").
		Raw("ADD $-1, R4").Raw("CMPBNE R4, $0, loop").
		Label("done").
		// Fold the deferred carry: vs2 += 16 * vcsum (each block adds 16*s1).
		Raw("VESLF $4, V22, V22").Raw("VAF V21, V22, V21").
		// Horizontal reduce vs1 and vs2: VSUMQF sums the 4 words into the low
		// word (element 3), then VLGVF extracts it to a GPR.
		Raw("VSUMQF V20, V23, V20").Raw("VLGVF $3, V20, R1").
		Raw("VSUMQF V21, V23, V21").Raw("VLGVF $3, V21, R3").
		StoreRet("R1", "ret").StoreRet("R3", "ret1").Ret()
	f.Add(b.Func())

	if err := os.WriteFile("adler32_s390x.s", []byte(f.String()), 0o644); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	fmt.Println("wrote adler32_s390x.s")
}
