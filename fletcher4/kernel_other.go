//go:build !amd64 && !arm64

package fletcher4

// No SIMD kernel on this architecture: Sum uses the scalar loop.
func kernels() []kernel { return nil }
