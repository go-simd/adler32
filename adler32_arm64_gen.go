//go:build ignore

// Command gen produces adler32_arm64.s with go-asmgen: a NEON Adler-32 inner
// sum for arm64. The weighted sum at the core needs an INTEGER vector multiply.
// Go's arm64 assembler historically exposed only the polynomial VPMULL; the
// integer VMUL / VUMULL / VUMLAL mnemonics were upstreamed in Go 1.27, so the
// generated kernel is guarded //go:build go1.27 and the stable build falls back
// to scalar. This file is the concrete demonstration of that newly-available
// VUMULL widening multiply.
//
// Per 16-byte block v (running s1 = S before the block):
//
//	s2 += 16*S                         (cross-block carry)
//	bytesum = Σ v                      (VUADDLV over the 16 bytes)
//	widen v to 16-bit halves (VUXTL / VUXTL2); weighted = Σ weight_i*v_i via
//	VUMULL/VUMULL2 against the 16-bit weight vectors {16..9} and {8..1}, summed
//	to a scalar with VUADDLV; s2 += weighted; s1 += bytesum.
//
// The accumulation is done in GPRs per block (the multiply, not the reduction,
// is the point of the demo), so the result is plainly identical to scalar.
//
// Signature: adlerNEON(s1, s2 uint32, p []byte, n int) (uint32, uint32), n =
// number of whole 16-byte blocks. Run: go run adler32_arm64_gen.go
package main

import (
	"fmt"
	"os"
	"strings"

	"github.com/go-asmgen/asmgen/abi"
	"github.com/go-asmgen/asmgen/arm64"
	"github.com/go-asmgen/asmgen/emit"
)

// h16 packs eight little-endian 16-bit weights into 16 bytes.
func h16(w ...uint16) []byte {
	b := make([]byte, 0, 16)
	for _, v := range w {
		b = append(b, byte(v), byte(v>>8))
	}
	return b
}

func main() {
	f := emit.NewFile("arm64")
	// 16-bit weights for the low 8 bytes (16..9) and high 8 bytes (8..1).
	wlo := f.Data("wlo", h16(16, 15, 14, 13, 12, 11, 10, 9))
	whi := f.Data("whi", h16(8, 7, 6, 5, 4, 3, 2, 1))

	sig := abi.LayoutArgs(
		[]abi.Arg{
			abi.Scalar("s1", abi.Uint32), abi.Scalar("s2", abi.Uint32),
			abi.Slice("p"), abi.Scalar("n", abi.Int64),
		},
		[]abi.Arg{abi.Scalar("ret", abi.Uint32), abi.Scalar("ret1", abi.Uint32)},
	)

	b := arm64.NewFunc("adlerNEON", sig, 0)
	b.LoadArg("s1", "R0").LoadArg("s2", "R1").
		LoadArg("p_base", "R2").LoadArg("n", "R3").
		// Load the 16-bit weight vectors once.
		Raw("MOVD $%s(SB), R4", wlo).Raw("VLD1 (R4), [V6.H8]").
		Raw("MOVD $%s(SB), R4", whi).Raw("VLD1 (R4), [V7.H8]").
		Raw("CBZ R3, done").
		Label("loop").
		Raw("VLD1 (R2), [V3.B16]"). // 16 source bytes
		// s2 += 16*s1.
		Raw("LSL $4, R0, R5").Raw("ADD R5, R1, R1").
		// bytesum = Σ bytes -> R5; s1 += bytesum.
		Raw("VUADDLV V3.B16, V4").Raw("VMOV V4.S[0], R5").
		Raw("ADD R5, R0, R0").
		// Widen bytes to 16-bit halves.
		Raw("VUXTL V3.B8, V16.H8").Raw("VUXTL2 V3.B16, V17.H8").
		// weighted = Σ weight_i*byte_i.
		Raw("VUMULL V6.H4, V16.H4, V20.S4").
		Raw("VUMULL2 V6.H8, V16.H8, V21.S4").
		Raw("VUMULL V7.H4, V17.H4, V22.S4").
		Raw("VUMULL2 V7.H8, V17.H8, V23.S4").
		Raw("VADD V21.S4, V20.S4, V20.S4").
		Raw("VADD V23.S4, V22.S4, V22.S4").
		Raw("VADD V22.S4, V20.S4, V20.S4").
		Raw("VUADDLV V20.S4, V24").Raw("VMOV V24.D[0], R5").
		Raw("ADD R5, R1, R1").
		Raw("ADD $16, R2").Raw("SUB $1, R3").Raw("CBNZ R3, loop").
		Label("done").
		Raw("MOVWU R0, R0").Raw("MOVWU R1, R1").
		StoreRet("R0", "ret").StoreRet("R1", "ret1").Ret()
	f.Add(b.Func())

	// The integer NEON multiply used here (VUMULL) is only assemblable on Go
	// 1.27+, so the build constraint must match the //go:build go1.27 Go file;
	// emit writes a bare "arm64", which we narrow here.
	out := strings.Replace(f.String(), "//go:build arm64\n", "//go:build arm64 && go1.27\n", 1)
	if err := os.WriteFile("adler32_arm64.s", []byte(out), 0o644); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	fmt.Println("wrote adler32_arm64.s")
}
