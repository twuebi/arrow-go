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
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

package compress

import (
	"io"

	"github.com/DataDog/zstd"
)

type zstdCodec struct{}

func (zstdCodec) Decode(dst, src []byte) []byte {
	out, err := zstd.Decompress(dst, src)
	if err != nil {
		panic(err)
	}
	return out
}

func (z zstdCodec) Encode(dst, src []byte) []byte {
	return z.EncodeLevel(dst, src, zstd.DefaultCompression)
}

func (z zstdCodec) EncodeLevel(dst, src []byte, level int) []byte {
	out, err := zstd.CompressLevel(dst, src, level)
	if err != nil {
		panic(err)
	}
	return out
}

func (zstdCodec) CompressBound(len int64) int64 {
	return int64(zstd.CompressBound(int(len)))
}

func (zstdCodec) NewReader(r io.Reader) io.ReadCloser {
	return zstd.NewReader(r)
}

func (zstdCodec) NewWriter(w io.Writer) io.WriteCloser {
	return zstd.NewWriter(w)
}

func (zstdCodec) NewWriterLevel(w io.Writer, level int) (io.WriteCloser, error) {
	return zstd.NewWriterLevel(w, level), nil
}

func init() {
	RegisterCodec(Codecs.Zstd, zstdCodec{})
}
