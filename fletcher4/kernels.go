package fletcher4

// kernel is one SIMD kernel, for tests.
type kernel struct {
	name       string
	lanes      int
	blockBytes int
	run        func(p *byte, nblocks int, out *[32]uint64)
}
