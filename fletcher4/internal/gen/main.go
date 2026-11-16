// Command gen writes the fletcher4 kernels with go-asmgen:
//
//	kernel_amd64.s   sumAVX2 (4 lanes) and sumAVX512 (8 lanes)
//	kernel_arm64.s   sumNEON (4 lanes as two 2-lane sets)
//
// ZFS fletcher4 (module/zcommon/zfs_fletcher.c, fletcher_4_scalar_native) is, per
// little-endian 32-bit word w: a += w; b += a; c += b; d += c, all mod 2^64. A
// SIMD kernel keeps L independent lanes (lane j sees words j, j+L, j+2L, ...) in
// the four accumulators a, b, c, d and stores them; the Go side folds the lanes
// with the closed forms OpenZFS uses in its fini functions (combine in
// fletcher4.go). The kernels do no folding and no tail.
//
// Signature, both architectures:
//
//	func sumX(p *byte, nblocks int, out *[32]uint64)
//
// out receives the accumulators as out[g*L+j], g = a, b, c, d and j = lane.
//
// Run from this directory: go run .
package main

import (
	"fmt"
	"os"
	"strings"

	"github.com/go-asmgen/asmgen/abi"
	"github.com/go-asmgen/asmgen/amd64"
	"github.com/go-asmgen/asmgen/arm64"
	"github.com/go-asmgen/asmgen/emit"
)

func sig() abi.Signature {
	return abi.LayoutArgs([]abi.Arg{
		abi.Scalar("p", abi.Uint64), abi.Scalar("nblocks", abi.Int64), abi.Scalar("out", abi.Uint64),
	}, nil)
}

// avx2 block = 32 bytes = two 4-word groups, both folded into the same lanes in
// order.
func genAVX2(f *emit.File) {
	b := amd64.NewFunc("sumAVX2", sig(), 0)
	b.LoadArg("p", "SI").LoadArg("nblocks", "CX").LoadArg("out", "DI").
		Raw("VPXOR Y0, Y0, Y0").Raw("VPXOR Y1, Y1, Y1").Raw("VPXOR Y2, Y2, Y2").Raw("VPXOR Y3, Y3, Y3").
		Raw("TESTQ CX, CX").Raw("JZ done").
		Label("loop")
	for _, off := range []int{0, 16} {
		b.Raw("VPMOVZXDQ %d(SI), Y4", off).
			Raw("VPADDQ Y4, Y0, Y0").Raw("VPADDQ Y0, Y1, Y1").Raw("VPADDQ Y1, Y2, Y2").Raw("VPADDQ Y2, Y3, Y3")
	}
	b.Raw("ADDQ $32, SI").Raw("DECQ CX").Raw("JNZ loop").
		Label("done").
		Raw("VMOVDQU Y0, 0(DI)").Raw("VMOVDQU Y1, 32(DI)").Raw("VMOVDQU Y2, 64(DI)").Raw("VMOVDQU Y3, 96(DI)").
		Raw("VZEROUPPER").Ret()
	f.Add(b.Func())
}

// avx512 block = 64 bytes = two 8-word groups.
func genAVX512(f *emit.File) {
	b := amd64.NewFunc("sumAVX512", sig(), 0)
	b.LoadArg("p", "SI").LoadArg("nblocks", "CX").LoadArg("out", "DI").
		Raw("VPXORQ Z0, Z0, Z0").Raw("VPXORQ Z1, Z1, Z1").Raw("VPXORQ Z2, Z2, Z2").Raw("VPXORQ Z3, Z3, Z3").
		Raw("TESTQ CX, CX").Raw("JZ done").
		Label("loop")
	for _, off := range []int{0, 32} {
		b.Raw("VPMOVZXDQ %d(SI), Z4", off).
			Raw("VPADDQ Z4, Z0, Z0").Raw("VPADDQ Z0, Z1, Z1").Raw("VPADDQ Z1, Z2, Z2").Raw("VPADDQ Z2, Z3, Z3")
	}
	b.Raw("ADDQ $64, SI").Raw("DECQ CX").Raw("JNZ loop").
		Label("done").
		Raw("VMOVDQU64 Z0, 0(DI)").Raw("VMOVDQU64 Z1, 64(DI)").Raw("VMOVDQU64 Z2, 128(DI)").Raw("VMOVDQU64 Z3, 192(DI)").
		Raw("VZEROUPPER").Ret()
	f.Add(b.Func())
}

// neon block = 16 bytes = one 4-word group: set A takes words 0,1 (lanes 0,1),
// set B words 2,3 (lanes 2,3). The loop is unrolled to 32 bytes.
func genNEON(f *emit.File) {
	b := arm64.NewFunc("sumNEON", sig(), 0)
	b.LoadArg("p", "R0").LoadArg("nblocks", "R1").LoadArg("out", "R2")
	for r := 1; r <= 8; r++ {
		b.Raw("VEOR V%d.B16, V%d.B16, V%d.B16", r, r, r)
	}
	b.Raw("CBZ R1, done").Label("loop")
	for u := 0; u < 2; u++ {
		b.Raw("VLD1.P 16(R0), [V0.S4]").
			Raw("VUXTL V0.S2, V9.D2").Raw("VUXTL2 V0.S4, V10.D2").
			Raw("VADD V9.D2, V1.D2, V1.D2").Raw("VADD V1.D2, V2.D2, V2.D2").Raw("VADD V2.D2, V3.D2, V3.D2").Raw("VADD V3.D2, V4.D2, V4.D2").
			Raw("VADD V10.D2, V5.D2, V5.D2").Raw("VADD V5.D2, V6.D2, V6.D2").Raw("VADD V6.D2, V7.D2, V7.D2").Raw("VADD V7.D2, V8.D2, V8.D2")
	}
	b.Raw("SUB $1, R1").Raw("CBNZ R1, loop").Label("done")
	// out[g*4+j]: a (set A, set B), b, c, d.
	for _, r := range []int{1, 5, 2, 6, 3, 7, 4, 8} {
		b.Raw("VST1.P [V%d.D2], 16(R2)", r)
	}
	b.Ret()
	f.Add(b.Func())
}

func write(name, text string) {
	if err := os.WriteFile(name, []byte(text), 0o644); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	fmt.Println("wrote", name)
}

func main() {
	fa := emit.NewFile("amd64")
	genAVX2(fa)
	genAVX512(fa)
	write("../../kernel_amd64.s", fa.String())

	fn := emit.NewFile("arm64")
	genNEON(fn)
	// VUXTL/VUXTL2/VLD1.P are accepted by the stable assembler; nothing to narrow.
	write("../../kernel_arm64.s", strings.TrimSpace(fn.String())+"\n")
}
