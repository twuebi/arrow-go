// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package pqarrow_test

import (
	"bytes"
	"math"
	"strings"
	"testing"

	"github.com/apache/arrow-go/v18/arrow"
	"github.com/apache/arrow-go/v18/arrow/array"
	"github.com/apache/arrow-go/v18/arrow/memory"
	"github.com/apache/arrow-go/v18/parquet"
	"github.com/apache/arrow-go/v18/parquet/compress"
	"github.com/apache/arrow-go/v18/parquet/file"
	"github.com/apache/arrow-go/v18/parquet/pqarrow"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestFileWriterRowGroupNumRows(t *testing.T) {
	schema := arrow.NewSchema([]arrow.Field{
		{Name: "one", Nullable: true, Type: arrow.PrimitiveTypes.Float64},
		{Name: "two", Nullable: true, Type: arrow.PrimitiveTypes.Float64},
	}, nil)

	data := `[
		{"one": 1, "two": 2},
		{"one": 1, "two": null},
		{"one": null, "two": 2},
		{"one": null, "two": null}
	]`
	record, _, err := array.RecordFromJSON(memory.DefaultAllocator, schema, strings.NewReader(data))
	require.NoError(t, err)

	output := &bytes.Buffer{}
	writerProps := parquet.NewWriterProperties(parquet.WithMaxRowGroupLength(100))
	writer, err := pqarrow.NewFileWriter(schema, output, writerProps, pqarrow.DefaultWriterProps())
	require.NoError(t, err)

	require.NoError(t, writer.Write(record))
	numRows, err := writer.RowGroupNumRows()
	require.NoError(t, err)
	assert.Equal(t, 4, numRows)

	// Make sure that row group stats are up-to-date immediately after writing
	bytesWritten := writer.RowGroupTotalBytesWritten()
	require.NoError(t, writer.Close())
	require.Equal(t, bytesWritten, writer.RowGroupTotalBytesWritten())
}

func TestFileWriterNumRows(t *testing.T) {
	schema := arrow.NewSchema([]arrow.Field{
		{Name: "one", Nullable: true, Type: arrow.PrimitiveTypes.Float64},
		{Name: "two", Nullable: true, Type: arrow.PrimitiveTypes.Float64},
	}, nil)

	data := `[
		{"one": 1, "two": 2},
		{"one": 1, "two": null},
		{"one": null, "two": 2},
		{"one": null, "two": null}
	]`
	record, _, err := array.RecordFromJSON(memory.DefaultAllocator, schema, strings.NewReader(data))
	require.NoError(t, err)

	maxRowGroupLength := 2

	output := &bytes.Buffer{}
	writerProps := parquet.NewWriterProperties(parquet.WithMaxRowGroupLength(int64(maxRowGroupLength)))
	writer, err := pqarrow.NewFileWriter(schema, output, writerProps, pqarrow.DefaultWriterProps())
	require.NoError(t, err)

	require.NoError(t, writer.Write(record))
	rowGroupNumRows, err := writer.RowGroupNumRows()
	require.NoError(t, err)
	assert.Equal(t, maxRowGroupLength, rowGroupNumRows)

	require.NoError(t, writer.Close())
	assert.Equal(t, 4, writer.NumRows())
}

func TestFileWriterBuffered(t *testing.T) {
	schema := arrow.NewSchema([]arrow.Field{
		{Name: "one", Nullable: true, Type: arrow.PrimitiveTypes.Float64},
		{Name: "two", Nullable: true, Type: arrow.PrimitiveTypes.Float64},
	}, nil)

	data := `[
		{"one": 1, "two": 2},
		{"one": 1, "two": null},
		{"one": null, "two": 2},
		{"one": null, "two": null}
	]`

	alloc := memory.NewCheckedAllocator(memory.DefaultAllocator)
	defer alloc.AssertSize(t, 0)

	record, _, err := array.RecordFromJSON(alloc, schema, strings.NewReader(data))
	require.NoError(t, err)
	defer record.Release()

	output := &bytes.Buffer{}
	writer, err := pqarrow.NewFileWriter(
		schema,
		output,
		parquet.NewWriterProperties(
			parquet.WithAllocator(alloc),
			// Ensure enough space so we can close the writer with rows still buffered
			parquet.WithMaxRowGroupLength(math.MaxInt64),
		),
		pqarrow.NewArrowWriterProperties(
			pqarrow.WithAllocator(alloc),
		),
	)
	require.NoError(t, err)

	require.NoError(t, writer.WriteBuffered(record))

	require.NoError(t, writer.Close())
	assert.Equal(t, 4, writer.NumRows())
}

func TestFileWriterTotalBytes(t *testing.T) {
	schema := arrow.NewSchema([]arrow.Field{
		{Name: "one", Nullable: true, Type: arrow.PrimitiveTypes.Float64},
		{Name: "two", Nullable: true, Type: arrow.PrimitiveTypes.Float64},
	}, nil)

	data := `[
		{"one": 1, "two": 2},
		{"one": 3, "two": 4}
	]`
	record1, _, err := array.RecordFromJSON(memory.DefaultAllocator, schema, strings.NewReader(data))
	require.NoError(t, err)
	defer record1.Release()

	data2 := `[
		{"one": 5, "two": 6},
		{"one": 7, "two": 8}
	]`
	record2, _, err := array.RecordFromJSON(memory.DefaultAllocator, schema, strings.NewReader(data2))
	require.NoError(t, err)
	defer record2.Release()

	output := &bytes.Buffer{}
	writerProps := parquet.NewWriterProperties(parquet.WithMaxRowGroupLength(2))
	writer, err := pqarrow.NewFileWriter(schema, output, writerProps, pqarrow.DefaultWriterProps())
	require.NoError(t, err)

	// Write first record
	require.NoError(t, writer.Write(record1))

	// Write second record, which creates a new row group
	require.NoError(t, writer.Write(record2))

	// Close the writer and verify final bytes
	require.NoError(t, writer.Close())

	// Verify total bytes & compressed bytes are correct
	assert.Equal(t, int64(408), writer.TotalCompressedBytes())
	assert.Equal(t, int64(910), writer.TotalBytesWritten())
}

func TestFileWriterTotalBytesBuffered(t *testing.T) {
	schema := arrow.NewSchema([]arrow.Field{
		{Name: "one", Nullable: true, Type: arrow.PrimitiveTypes.Float64},
		{Name: "two", Nullable: true, Type: arrow.PrimitiveTypes.Float64},
	}, nil)

	data := `[
		{"one": 1, "two": 2},
		{"one": 3, "two": 4},
		{"one": 5, "two": 6},
		{"one": 7, "two": 8},
		{"one": 9, "two": 10}
	]`
	record, _, err := array.RecordFromJSON(memory.DefaultAllocator, schema, strings.NewReader(data))
	require.NoError(t, err)
	defer record.Release()

	output := &bytes.Buffer{}
	// Use a large max row group length to ensure both records go into the same row group
	writerProps := parquet.NewWriterProperties(parquet.WithMaxRowGroupLength(2))
	writer, err := pqarrow.NewFileWriter(schema, output, writerProps, pqarrow.DefaultWriterProps())
	require.NoError(t, err)

	// Write record using WriteBuffered
	require.NoError(t, writer.WriteBuffered(record))

	// Close the writer and verify final bytes
	require.NoError(t, writer.Close())

	// Verify total bytes & compressed bytes are correct
	assert.Equal(t, int64(596), writer.TotalCompressedBytes())
	assert.Equal(t, int64(1306), writer.TotalBytesWritten())
}

// buildInt64Record returns a single-column record batch of consecutive
// int64 values [0, n) for row-group-sizing tests.
func buildInt64Record(t *testing.T, mem memory.Allocator, n int64) (*arrow.Schema, arrow.RecordBatch) {
	t.Helper()
	schema := arrow.NewSchema([]arrow.Field{
		{Name: "v", Type: arrow.PrimitiveTypes.Int64},
	}, nil)

	bldr := array.NewInt64Builder(mem)
	defer bldr.Release()
	for i := int64(0); i < n; i++ {
		bldr.Append(i)
	}
	arr := bldr.NewArray()
	defer arr.Release()

	return schema, array.NewRecord(schema, []arrow.Array{arr}, n)
}

// readRowGroupMeta opens a parquet file from buf and returns (numRowGroups,
// perRowGroupUncompressedSize, totalRows).
func readRowGroupMeta(t *testing.T, buf []byte) (int, []int64, int64) {
	t.Helper()
	pf, err := file.NewParquetReader(bytes.NewReader(buf))
	require.NoError(t, err)
	defer pf.Close()

	md := pf.MetaData()
	n := md.NumRowGroups()
	sizes := make([]int64, n)
	var total int64
	for i := 0; i < n; i++ {
		rg := md.RowGroup(i)
		sizes[i] = rg.TotalByteSize()
		total += rg.NumRows()
	}
	return n, sizes, total
}

// TestFileWriterMaxRowGroupBytes verifies that configuring
// WithMaxRowGroupBytes causes the buffered writer to close and open new row
// groups once the accumulated uncompressed size crosses the threshold across
// a stream of WriteBuffered calls.
//
// The trigger operates between WriteBuffered calls rather than splitting a
// single record batch: there is no cheap way to know upfront how many input
// rows will push us past a target byte size (the relationship depends on the
// encoder and compressor), so WriteBuffered keeps one batch in one row group
// and relies on subsequent calls to roll over once the budget is exceeded.
func TestFileWriterMaxRowGroupBytes(t *testing.T) {
	alloc := memory.NewCheckedAllocator(memory.DefaultAllocator)
	defer alloc.AssertSize(t, 0)

	// Small per-call batch so the byte budget is crossed after several calls,
	// letting us observe multiple row group boundaries.
	const rowsPerCall = int64(500)
	const calls = 50
	schema, record := buildInt64Record(t, alloc, rowsPerCall)
	defer record.Release()

	// Keep compression off and pages small so that totalUncompressedBytes
	// advances predictably as we write. Without a small page size the
	// counter only increments at page-flush boundaries (default 1 MiB),
	// which would require ~8x more data to exercise.
	const maxBytes = int64(16 * 1024)
	output := &bytes.Buffer{}
	writerProps := parquet.NewWriterProperties(
		parquet.WithAllocator(alloc),
		parquet.WithMaxRowGroupLength(math.MaxInt64),
		parquet.WithMaxRowGroupBytes(maxBytes),
		parquet.WithDataPageSize(4*1024),
		parquet.WithCompression(compress.Codecs.Uncompressed),
	)
	writer, err := pqarrow.NewFileWriter(schema, output, writerProps,
		pqarrow.NewArrowWriterProperties(pqarrow.WithAllocator(alloc)))
	require.NoError(t, err)
	for i := 0; i < calls; i++ {
		require.NoError(t, writer.WriteBuffered(record))
	}
	require.NoError(t, writer.Close())

	nRG, sizes, total := readRowGroupMeta(t, output.Bytes())
	assert.Equal(t, rowsPerCall*calls, total, "all rows should be preserved")
	assert.Greater(t, nRG, 1, "expected bytes trigger to produce multiple row groups")

	// Every closed row group except the final tail must have crossed the
	// threshold — that's the trigger condition.
	for i := 0; i < nRG-1; i++ {
		assert.GreaterOrEqual(t, sizes[i], maxBytes,
			"row group %d size %d below threshold %d", i, sizes[i], maxBytes)
	}
}

// TestFileWriterMaxRowGroupBytesDefaultDisabled confirms that the historical
// behavior is preserved when WithMaxRowGroupBytes is not set (i.e. a single
// buffered write produces a single row group regardless of size, bounded only
// by MaxRowGroupLength).
func TestFileWriterMaxRowGroupBytesDefaultDisabled(t *testing.T) {
	alloc := memory.NewCheckedAllocator(memory.DefaultAllocator)
	defer alloc.AssertSize(t, 0)

	const numRows = int64(50_000)
	schema, record := buildInt64Record(t, alloc, numRows)
	defer record.Release()

	output := &bytes.Buffer{}
	writerProps := parquet.NewWriterProperties(
		parquet.WithAllocator(alloc),
		parquet.WithMaxRowGroupLength(math.MaxInt64),
		// MaxRowGroupBytes left at default (0 = unlimited).
		parquet.WithDataPageSize(4*1024),
		parquet.WithCompression(compress.Codecs.Uncompressed),
	)
	writer, err := pqarrow.NewFileWriter(schema, output, writerProps,
		pqarrow.NewArrowWriterProperties(pqarrow.WithAllocator(alloc)))
	require.NoError(t, err)
	require.NoError(t, writer.WriteBuffered(record))
	require.NoError(t, writer.Close())

	nRG, _, total := readRowGroupMeta(t, output.Bytes())
	assert.Equal(t, numRows, total)
	assert.Equal(t, 1, nRG, "default (0) must preserve single-row-group behavior")
}

// TestFileWriterMaxRowGroupBytesRowLimitWins verifies that when both
// MaxRowGroupLength and MaxRowGroupBytes are configured, whichever trigger
// fires first closes the row group (AND semantics).
func TestFileWriterMaxRowGroupBytesRowLimitWins(t *testing.T) {
	alloc := memory.NewCheckedAllocator(memory.DefaultAllocator)
	defer alloc.AssertSize(t, 0)

	const numRows = int64(50_000)
	schema, record := buildInt64Record(t, alloc, numRows)
	defer record.Release()

	// Set row limit very tight (1_000) and bytes very loose (1 GiB).
	// The row-limit pre-slicing should drive row-group boundaries alone;
	// the bytes trigger should stay inert.
	const rowLimit = int64(1_000)
	output := &bytes.Buffer{}
	writerProps := parquet.NewWriterProperties(
		parquet.WithAllocator(alloc),
		parquet.WithMaxRowGroupLength(rowLimit),
		parquet.WithMaxRowGroupBytes(1<<30),
		parquet.WithDataPageSize(4*1024),
		parquet.WithCompression(compress.Codecs.Uncompressed),
	)
	writer, err := pqarrow.NewFileWriter(schema, output, writerProps,
		pqarrow.NewArrowWriterProperties(pqarrow.WithAllocator(alloc)))
	require.NoError(t, err)
	require.NoError(t, writer.WriteBuffered(record))
	require.NoError(t, writer.Close())

	nRG, _, total := readRowGroupMeta(t, output.Bytes())
	assert.Equal(t, numRows, total)
	// With a 1 000-row cap and 50 000 rows, expect exactly 50 row groups.
	assert.Equal(t, int(numRows/rowLimit), nRG)
}

// TestFileWriterMaxRowGroupBytesAcrossCalls ensures that the bytes-based
// trigger accumulates state across successive WriteBuffered calls rather than
// resetting on each one.
func TestFileWriterMaxRowGroupBytesAcrossCalls(t *testing.T) {
	alloc := memory.NewCheckedAllocator(memory.DefaultAllocator)
	defer alloc.AssertSize(t, 0)

	schema, record := buildInt64Record(t, alloc, 5_000)
	defer record.Release()

	const maxBytes = int64(16 * 1024)
	output := &bytes.Buffer{}
	writerProps := parquet.NewWriterProperties(
		parquet.WithAllocator(alloc),
		parquet.WithMaxRowGroupLength(math.MaxInt64),
		parquet.WithMaxRowGroupBytes(maxBytes),
		parquet.WithDataPageSize(4*1024),
		parquet.WithCompression(compress.Codecs.Uncompressed),
	)
	writer, err := pqarrow.NewFileWriter(schema, output, writerProps,
		pqarrow.NewArrowWriterProperties(pqarrow.WithAllocator(alloc)))
	require.NoError(t, err)

	// Write the same record ten times. Each individual record at 5_000 rows
	// × 8 bytes ≈ 40 KiB uncompressed, which exceeds the 16 KiB threshold,
	// so every call should close a row group.
	const iterations = 10
	for i := 0; i < iterations; i++ {
		require.NoError(t, writer.WriteBuffered(record))
	}
	require.NoError(t, writer.Close())

	nRG, _, total := readRowGroupMeta(t, output.Bytes())
	assert.Equal(t, int64(iterations*5_000), total)
	assert.GreaterOrEqual(t, nRG, iterations,
		"expected at least one row group per WriteBuffered call")
}

func TestWriteOnClosedFileWriter(t *testing.T) {
	schema := arrow.NewSchema([]arrow.Field{
		{Name: "one", Nullable: true, Type: arrow.PrimitiveTypes.Float64},
	}, nil)

	output := &bytes.Buffer{}
	writer, err := pqarrow.NewFileWriter(schema, output, parquet.NewWriterProperties(), pqarrow.DefaultWriterProps())
	require.NoError(t, err)

	// Close the writer
	require.NoError(t, writer.Close())

	// Call each write method and ensure they all return an error stating the writer is already closed
	require.ErrorContains(t, writer.WriteBuffered(nil), "already closed")
	require.ErrorContains(t, writer.Write(nil), "already closed")
	require.ErrorContains(t, writer.WriteColumnChunked(nil, 0, 0), "already closed")
	require.ErrorContains(t, writer.WriteColumnData(nil), "already closed")
}
