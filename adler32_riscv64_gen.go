//go:build ignore

// Command gen produces adler32_riscv64.s with go-asmgen: an RVV (vector)
// Adler-32 inner sum. RISC-V's vector extension has integer vector multiply
// (VWMULU widening unsigned multiply) and vector reductions (VREDSUM), so the
// weighted sum runs in vectors. The loop is length-agnostic: VSETVLI grants a
// strip length vl each iteration.
//
// Per strip of vl bytes (running s1 = S before the strip):
//   s2 += vl*S                       (cross-strip carry, scalar)
//   bytesum = Σ strip                (VWREDSUMU: e8 -> e16 sum)
//   id = 0..vl-1 (VID); weight = vl - id (VRSUBVX), so byte j has weight vl-j
//   widen bytes e8 -> e16 (VZEXT), weighted = Σ weight_j*byte_j via VWMULU
//   (e16*e16 -> e32) then VREDSUM (e32); s2 += weighted; s1 += bytesum
// All accumulation into S/s2 is in GPRs, so the result is plainly identical to
// the scalar reference. VWMULU is unsigned and widens to 32-bit, so weight*byte
// (<= vl*255) never overflows for any VLEN.
//
// Signature: adlerRVV(s1, s2 uint32, p []byte, n int) (uint32, uint32) where n
// is the byte count of the region to fold. Run: go run adler32_riscv64_gen.go
package main

import (
	"fmt"
	"os"

	"github.com/go-asmgen/asmgen/abi"
	"github.com/go-asmgen/asmgen/emit"
	"github.com/go-asmgen/asmgen/riscv64"
)

func main() {
	f := emit.NewFile("riscv64")

	sig := abi.LayoutArgs(
		[]abi.Arg{
			abi.Scalar("s1", abi.Uint32), abi.Scalar("s2", abi.Uint32),
			abi.Slice("p"), abi.Scalar("n", abi.Int64),
		},
		[]abi.Arg{abi.Scalar("ret", abi.Uint32), abi.Scalar("ret1", abi.Uint32)},
	)

	// Registers: X5=s1(S), X6=s2, X7=ptr, X8=remaining n, X9=vl, X10=scratch.
	// A fixed vl per strip is reused across element widths by scaling LMUL so
	// the element COUNT stays constant: e8/m1, e16/m2 and e32/m4 share VLMAX,
	// so the granted vl (X9) is the same element count under all three.
	b := riscv64.NewFunc("adlerRVV", sig, 0)
	b.LoadArg("s1", "X5").LoadArg("s2", "X6").
		LoadArg("p_base", "X7").LoadArg("n", "X8").
		Label("loop").
		Raw("BEQZ X8, done").
		// Seed the widening-reduce accumulator V2[0]=0 at 16-bit width (the
		// widening reduce reads its accumulator as 2*SEW=16 bits). Using e16/m1
		// with vl>=1 guarantees byte0..1 of V2 are cleared.
		Raw("VSETVLI X0, E16, M1, TA, MA, X10").
		Raw("VMVSX X0, V2").
		// vl = vsetvli(n, e8, m1); X9 = granted vl (element count for this strip).
		Raw("VSETVLI X8, E8, M1, TA, MA, X9").
		// Load vl bytes into V1.
		Raw("VLE8V (X7), V1").
		// s2 += vl * S.
		Raw("MUL X9, X5, X10").Raw("ADD X10, X6, X6").
		// bytesum: widening unsigned reduce e8->e16 into V2 lane0. Go operand
		// order is "vs1(seed), vs2(data), vd": seed V2, reduce data V1 -> V2.
		Raw("VWREDSUMUVS V2, V1, V2").
		// e16/m2 keeps the same element count: read the reduction lane and
		// build the weights id=0..vl-1, weight = vl - id.
		Raw("VSETVLI X9, E16, M2, TA, MA, X10").
		Raw("VMVXS V2, X10").Raw("ADD X10, X5, X5"). // s1 += bytesum
		Raw("VIDV V4").                              // V4:V5 = 0..vl-1 (e16/m2)
		Raw("VRSUBVX X9, V4, V4").                   // V4 = vl - id
		// widen bytes e8->e16 into V6 (VZEXT reads the e8 V1 under e16 vl).
		Raw("VZEXTVF2 V1, V6"). // V6:V7 = bytes as 16-bit
		// products e16*e16 -> e32 (widening) in V8 group.
		Raw("VWMULUVV V4, V6, V8"). // V8..V11 = weight*byte (32-bit, e32/m4)
		// e32/m4 keeps the same element count: reduce the 32-bit products.
		Raw("VSETVLI X9, E32, M4, TA, MA, X10").
		Raw("VMVVX X0, V12").
		// vs1(seed)=V12, vs2(data)=V8, vd=V12.
		Raw("VREDSUMVS V12, V8, V12").
		Raw("VMVXS V12, X10").Raw("ADD X10, X6, X6"). // s2 += weighted
		// advance.
		Raw("ADD X9, X7, X7").Raw("SUB X9, X8, X8").
		Raw("JMP loop").
		Label("done").
		// truncate to 32-bit and return.
		Raw("MOVWU X5, X5").Raw("MOVWU X6, X6").
		StoreRet("X5", "ret").StoreRet("X6", "ret1").Ret()
	f.Add(b.Func())

	if err := os.WriteFile("adler32_riscv64.s", []byte(f.String()), 0o644); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	fmt.Println("wrote adler32_riscv64.s")
}
