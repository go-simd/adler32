# fletcher64

The **Fletcher-64** object checksum of Apple's APFS (`obj_phys_t.o_cksum`), the
one kernel every fleet APFS reader and writer calls.

```go
import "github.com/go-simd/adler32/fletcher64"

stored := fletcher64.Sum(block[8:]) // the o_cksum of a block; its first 8 bytes hold it
```

Per little-endian 32-bit word `w`: `s1 += w; s2 += s1`, both mod 2^32-1; the stored
value is `c2<<32 | c1` with `c1 = M - (s1+s2)%M`, `c2 = M - (s1+c1)%M`,
`M = 2^32-1` (Apple File System Reference, "Fletcher 64 Checksum Algorithm"). A
trailing partial word is ignored.

This is **not** ZFS fletcher4 (see [`fletcher4`](../fletcher4): four sums mod
2^64, no modular reduction); the two are different algorithms and are not merged.

## Kernel

Pure Go. The modular reduction is deferred to every 4096 bytes (`s1 < 2^42`,
`s2 < 2^52`, so nothing overflows), the hot loop is additions only and is
unrolled to 128 bytes per iteration (32 iterations per 4 KiB block, which also
keeps coverage-instrumented CI runs fast). `Sum` is `//go:norace`: it is a pure
function and instrumenting every word load made a race-detector suite 2.4x slower.
No SIMD kernel yet: the dependent `s2 += s1` chain is the work, and a lane-split
version needs the same closed-form fold as fletcher4.

## Verification

- known answers pinned to the go-filesystems/apfs v0.7.0 kernel's output;
- `Sum` against the straightforward reduce-after-every-word loop (the oracle),
  for every length 0..600, around the 4096-byte reduction run, and with all-ones
  words; a fuzz target (`FuzzSum`, corpus under `testdata/fuzz`) does the same;
- the defining property: appending the checksum makes both sums zero;
- zero allocations per call.

`go test -bench . ./fletcher64` runs the benchmark.
