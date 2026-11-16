package fletcher4

// kernel is one SIMD kernel; ok says whether this CPU can run it.
type kernel struct {
	name       string
	lanes      int
	blockBytes int
	run        func(p *byte, nblocks int, out *[32]uint64)
	ok         bool
}

// kernels lists every kernel this CPU can run, narrowest first.
func kernels() []kernel {
	var ks []kernel
	for _, k := range allKernels() {
		if k.ok {
			ks = append(ks, k)
		}
	}
	return ks
}

// The widest kernel the CPU can run is the one Sum uses.
func init() {
	simd.name = "scalar"
	for _, k := range allKernels() {
		if k.ok {
			simd.name, simd.lanes, simd.blockBytes, simd.run = k.name, k.lanes, k.blockBytes, k.run
		}
	}
}
