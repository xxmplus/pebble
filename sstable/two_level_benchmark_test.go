// Copyright 2025 The LevelDB-Go and Pebble Authors. All rights reserved. Use
// of this source code is governed by a BSD-style license that can be found in
// the LICENSE file.

package sstable

import (
	"context"
	"fmt"
	"testing"

	"github.com/cockroachdb/pebble/bloom"
	"github.com/cockroachdb/pebble/internal/base"
	"github.com/cockroachdb/pebble/objstorage/objstorageprovider"
	"github.com/cockroachdb/pebble/sstable/block"
	"github.com/cockroachdb/pebble/vfs"
	"github.com/stretchr/testify/require"
)

var globalTwoLevelReader *Reader
var globalTwoLevelIterOpts IterOptions

func init() {
	// Setup global test data once - create a two-level index SSTable
	mem := vfs.NewMem()
	f, err := mem.Create("two_level_bench.sst", vfs.WriteCategoryUnspecified)
	if err != nil {
		panic(err)
	}

	// Use small index block size to force two-level indexing
	w := NewWriter(objstorageprovider.NewFileWritable(f), WriterOptions{
		BlockSize:      4096,
		IndexBlockSize: 512, // Small index block size forces two-level indexing
		FilterPolicy:   bloom.FilterPolicy(10),
		TableFormat:    TableFormatPebblev3,
		Comparer:       base.DefaultComparer,
		MergerName:     base.DefaultMerger.Name,
	})

	// Create enough keys to force multiple index blocks
	const numKeys = 50000
	for i := range numKeys {
		key := fmt.Appendf(nil, "key%08d", i)
		value := fmt.Sprintf("value%d", i)
		if err := w.Set(key, []byte(value)); err != nil {
			panic(err)
		}
	}
	if err := w.Close(); err != nil {
		panic(err)
	}

	f2, err := mem.Open("two_level_bench.sst")
	if err != nil {
		panic(err)
	}

	globalTwoLevelReader, err = newReader(f2, ReaderOptions{
		Comparer: base.DefaultComparer,
		Merger:   base.DefaultMerger,
	})
	if err != nil {
		panic(err)
	}

	// Verify we actually created a two-level index
	if !globalTwoLevelReader.Attributes.Has(AttributeTwoLevelIndex) {
		panic("Failed to create two-level index SSTable")
	}

	var stats base.InternalIteratorStats
	var bufferPool block.BufferPool
	bufferPool.Init(5)

	globalTwoLevelIterOpts = IterOptions{
		Transforms:           NoTransforms,
		FilterBlockSizeLimit: NeverUseFilterBlock,
		Env:                  ReadEnv{Block: block.ReadEnv{Stats: &stats, BufferPool: &bufferPool}},
		ReaderProvider:       MakeTrivialReaderProvider(globalTwoLevelReader),
	}
}

// simulateEagerTwoLevelLoading creates a two-level iterator and immediately loads the top-level index
// to simulate what eager loading would have done
func simulateEagerTwoLevelLoading(r *Reader, opts IterOptions) (*twoLevelIteratorRowBlocks, error) {
	iter, err := newRowBlockTwoLevelIterator(context.Background(), r, opts)
	if err != nil {
		return nil, err
	}

	// Force top-level index loading immediately (simulating eager loading)
	if err := iter.ensureTopLevelIndexLoaded(); err != nil {
		iter.Close()
		return nil, err
	}

	return iter, nil
}

// BenchmarkTwoLevelIteratorConstruction measures pure construction performance
// How to: go test -bench=BenchmarkTwoLevelIteratorConstruction -run=^$ ./sstable
func BenchmarkTwoLevelIteratorConstruction(b *testing.B) {
	b.Run("LazyLoading_ConstructOnly", func(b *testing.B) {
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			iter, err := newRowBlockTwoLevelIterator(context.Background(), globalTwoLevelReader, globalTwoLevelIterOpts)
			if err != nil {
				b.Fatal(err)
			}
			// Don't access top-level index - just construct and close
			iter.Close()
		}
	})

	b.Run("SimulatedEagerLoading_ConstructOnly", func(b *testing.B) {
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			iter, err := simulateEagerTwoLevelLoading(globalTwoLevelReader, globalTwoLevelIterOpts)
			if err != nil {
				b.Fatal(err)
			}
			// Top-level index was loaded during construction
			iter.Close()
		}
	})
}

// BenchmarkTwoLevelFirstAccessLatency compares the latency of first access operations
// How to: go test -bench=BenchmarkTwoLevelFirstAccessLatency -run=^$ ./sstable
func BenchmarkTwoLevelFirstAccessLatency(b *testing.B) {
	// Benchmark: First() call including top-level index loading time (lazy loading)
	b.Run("LazyLoading_First", func(b *testing.B) {
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			iter, err := newRowBlockTwoLevelIterator(context.Background(), globalTwoLevelReader, globalTwoLevelIterOpts)
			if err != nil {
				b.Fatal(err)
			}
			_ = iter.First() // Top-level index loaded here
			iter.Close()
		}
	})

	// Benchmark: First() call when top-level index is already loaded (simulated eager loading)
	b.Run("EagerLoading_First", func(b *testing.B) {
		// Create a single iterator and pre-load its top-level index once
		iter, err := newRowBlockTwoLevelIterator(context.Background(), globalTwoLevelReader, globalTwoLevelIterOpts)
		if err != nil {
			b.Fatal(err)
		}
		defer iter.Close()

		_ = iter.First() // Pre-load top-level index

		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			_ = iter.First() // Top-level index already loaded - this is what we're measuring
		}
	})

	// Benchmark: SeekGE() call including top-level index loading time (lazy loading)
	b.Run("LazyLoading_SeekGE", func(b *testing.B) {
		key := []byte("key00025000") // Middle key
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			iter, err := newRowBlockTwoLevelIterator(context.Background(), globalTwoLevelReader, globalTwoLevelIterOpts)
			if err != nil {
				b.Fatal(err)
			}
			_ = iter.SeekGE(key, base.SeekGEFlagsNone) // Top-level index loaded here
			iter.Close()
		}
	})

	// Benchmark: SeekGE() call when top-level index is already loaded (simulated eager loading)
	b.Run("EagerLoading_SeekGE", func(b *testing.B) {
		key := []byte("key00025000") // Middle key

		// Create a single iterator and pre-load its top-level index once
		iter, err := newRowBlockTwoLevelIterator(context.Background(), globalTwoLevelReader, globalTwoLevelIterOpts)
		if err != nil {
			b.Fatal(err)
		}
		defer iter.Close()

		_ = iter.First() // Pre-load top-level index

		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			_ = iter.SeekGE(key, base.SeekGEFlagsNone) // Top-level index already loaded - this is what we're measuring
		}
	})
}

// BenchmarkTwoLevelIndexLoadingOverhead measures the overhead of lazy loading
// by comparing subsequent calls on the same iterator
// How to: go test -bench=BenchmarkTwoLevelIndexLoadingOverhead -run=^$ ./sstable
func BenchmarkTwoLevelIndexLoadingOverhead(b *testing.B) {
	b.Run("LazyLoading_FirstCall", func(b *testing.B) {
		// Create fresh iterators for each call to measure first access (includes index loading)
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			iter, err := newRowBlockTwoLevelIterator(context.Background(), globalTwoLevelReader, globalTwoLevelIterOpts)
			if err != nil {
				b.Fatal(err)
			}

			// This triggers index loading + navigation
			b.StartTimer()
			_ = iter.First()
			b.StopTimer()

			iter.Close()
		}
	})

	b.Run("LazyLoading_SubsequentCall", func(b *testing.B) {
		// Create one iterator and call First() multiple times to measure subsequent access
		iter, err := newRowBlockTwoLevelIterator(context.Background(), globalTwoLevelReader, globalTwoLevelIterOpts)
		if err != nil {
			b.Fatal(err)
		}
		defer iter.Close()

		// Load index once
		_ = iter.First()

		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			_ = iter.First() // Index already loaded, just navigation
		}
	})

	b.Run("EagerLoading_FirstCall", func(b *testing.B) {
		// Create fresh eager iterators for each call
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			iter, err := simulateEagerTwoLevelLoading(globalTwoLevelReader, globalTwoLevelIterOpts)
			if err != nil {
				b.Fatal(err)
			}

			// Index already loaded, just navigation
			b.StartTimer()
			_ = iter.First()
			b.StopTimer()

			iter.Close()
		}
	})
}

// TestTwoLevelLazyLoadingBehavior verifies that lazy loading works as expected
// How to: go test -run TestTwoLevelLazyLoadingBehavior -v ./sstable
func TestTwoLevelLazyLoadingBehavior(t *testing.T) {
	t.Run("TopLevelIndexNotLoadedOnConstruction", func(t *testing.T) {
		iter, err := newRowBlockTwoLevelIterator(context.Background(), globalTwoLevelReader, globalTwoLevelIterOpts)
		require.NoError(t, err)
		defer iter.Close()

		// Top-level index should not be loaded yet
		require.False(t, iter.topLevelIndexLoaded, "Top-level index should not be loaded on construction")
	})

	t.Run("TopLevelIndexLoadedOnFirstAccess", func(t *testing.T) {
		iter, err := newRowBlockTwoLevelIterator(context.Background(), globalTwoLevelReader, globalTwoLevelIterOpts)
		require.NoError(t, err)
		defer iter.Close()

		// Top-level index should not be loaded yet
		require.False(t, iter.topLevelIndexLoaded, "Top-level index should not be loaded on construction")

		// Trigger top-level index loading
		_ = iter.First()

		// Top-level index should now be loaded
		require.True(t, iter.topLevelIndexLoaded, "Top-level index should be loaded after first access")
	})

	t.Run("TopLevelIndexStaysLoadedAfterPoolReuse", func(t *testing.T) {
		iter, err := newRowBlockTwoLevelIterator(context.Background(), globalTwoLevelReader, globalTwoLevelIterOpts)
		require.NoError(t, err)

		// Load the top-level index
		_ = iter.First()
		require.True(t, iter.topLevelIndexLoaded, "Top-level index should be loaded after first access")

		// Close and return to pool
		iter.Close()

		// Create new iterator (may reuse from pool)
		iter2, err := newRowBlockTwoLevelIterator(context.Background(), globalTwoLevelReader, globalTwoLevelIterOpts)
		require.NoError(t, err)
		defer iter2.Close()

		// Top-level index should not be loaded for new iterator (flag reset on pool reuse)
		require.False(t, iter2.topLevelIndexLoaded, "Top-level index should not be loaded on new iterator from pool")
	})

	t.Run("VerifyTwoLevelIndex", func(t *testing.T) {
		// Verify that our test SSTable actually has a two-level index
		require.True(t, globalTwoLevelReader.Attributes.Has(AttributeTwoLevelIndex), "Test SSTable should have two-level index")
	})
}
