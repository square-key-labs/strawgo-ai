package fish

import (
	"encoding/binary"
	"errors"
	"fmt"
	"math"
	"sort"
)

func pack(dst []byte, v any) ([]byte, error) {
	switch t := v.(type) {
	case nil:
		return append(dst, 0xc0), nil
	case bool:
		if t {
			return append(dst, 0xc3), nil
		}
		return append(dst, 0xc2), nil
	case int:
		return packInt(dst, int64(t)), nil
	case int64:
		return packInt(dst, t), nil
	case float64:
		dst = append(dst, 0xcb)
		return binary.BigEndian.AppendUint64(dst, math.Float64bits(t)), nil
	case string:
		n := len(t)
		switch {
		case n < 32:
			dst = append(dst, 0xa0|byte(n))
		case n < 1<<8:
			dst = append(dst, 0xd9, byte(n))
		case n < 1<<16:
			dst = binary.BigEndian.AppendUint16(append(dst, 0xda), uint16(n))
		default:
			dst = binary.BigEndian.AppendUint32(append(dst, 0xdb), uint32(n))
		}
		return append(dst, t...), nil
	case []byte:
		n := len(t)
		switch {
		case n < 1<<8:
			dst = append(dst, 0xc4, byte(n))
		case n < 1<<16:
			dst = binary.BigEndian.AppendUint16(append(dst, 0xc5), uint16(n))
		default:
			dst = binary.BigEndian.AppendUint32(append(dst, 0xc6), uint32(n))
		}
		return append(dst, t...), nil
	case map[string]any:
		n := len(t)
		switch {
		case n < 16:
			dst = append(dst, 0x80|byte(n))
		case n < 1<<16:
			dst = binary.BigEndian.AppendUint16(append(dst, 0xde), uint16(n))
		default:
			dst = binary.BigEndian.AppendUint32(append(dst, 0xdf), uint32(n))
		}
		keys := make([]string, 0, n)
		for k := range t {
			keys = append(keys, k)
		}
		sort.Strings(keys)
		var err error
		for _, k := range keys {
			if dst, err = pack(dst, k); err != nil {
				return nil, err
			}
			if dst, err = pack(dst, t[k]); err != nil {
				return nil, err
			}
		}
		return dst, nil
	}
	return nil, fmt.Errorf("msgpack: unsupported type %T", v)
}

func packInt(dst []byte, v int64) []byte {
	switch {
	case v >= 0 && v < 128:
		return append(dst, byte(v))
	case v < 0 && v >= -32:
		return append(dst, byte(v))
	default:
		return binary.BigEndian.AppendUint64(append(dst, 0xd3), uint64(v))
	}
}

var errShort = errors.New("msgpack: truncated")

type unpacker struct {
	b []byte
	i int
}

func unpack(b []byte) (any, error) {
	u := unpacker{b: b}
	return u.value(0)
}

func (u *unpacker) take(n int) ([]byte, error) {
	if n < 0 || u.i+n > len(u.b) {
		return nil, errShort
	}
	out := u.b[u.i : u.i+n]
	u.i += n
	return out, nil
}

func (u *unpacker) uint(n int) (uint64, error) {
	b, err := u.take(n)
	if err != nil {
		return 0, err
	}
	var v uint64
	for _, c := range b {
		v = v<<8 | uint64(c)
	}
	return v, nil
}

func (u *unpacker) sized(n int, f func(int) (any, error)) (any, error) {
	size, err := u.uint(n)
	if err != nil {
		return nil, err
	}
	if size > uint64(len(u.b)) {
		return nil, errShort
	}
	return f(int(size))
}

func (u *unpacker) str(n int) (any, error) {
	b, err := u.take(n)
	return string(b), err
}

func (u *unpacker) bin(n int) (any, error) {
	return u.take(n)
}

func (u *unpacker) value(depth int) (any, error) {
	if depth > 32 {
		return nil, errors.New("msgpack: too deep")
	}
	c, err := u.uint(1)
	if err != nil {
		return nil, err
	}
	switch {
	case c < 0x80:
		return int64(c), nil
	case c >= 0xe0:
		return int64(int8(c)), nil
	case c&0xe0 == 0xa0:
		return u.str(int(c & 0x1f))
	case c&0xf0 == 0x80:
		return u.mapN(int(c&0x0f), depth)
	case c&0xf0 == 0x90:
		return u.arrayN(int(c&0x0f), depth)
	}
	switch c {
	case 0xc0:
		return nil, nil
	case 0xc2:
		return false, nil
	case 0xc3:
		return true, nil
	case 0xc4, 0xc5, 0xc6:
		return u.sized(1<<(c-0xc4), u.bin)
	case 0xca:
		v, err := u.uint(4)
		return float64(math.Float32frombits(uint32(v))), err
	case 0xcb:
		v, err := u.uint(8)
		return math.Float64frombits(v), err
	case 0xcc, 0xcd, 0xce, 0xcf:
		v, err := u.uint(1 << (c - 0xcc))
		return int64(v), err
	case 0xd0:
		v, err := u.uint(1)
		return int64(int8(v)), err
	case 0xd1:
		v, err := u.uint(2)
		return int64(int16(v)), err
	case 0xd2:
		v, err := u.uint(4)
		return int64(int32(v)), err
	case 0xd3:
		v, err := u.uint(8)
		return int64(v), err
	case 0xd9, 0xda, 0xdb:
		return u.sized(1<<(c-0xd9), u.str)
	case 0xdc, 0xdd:
		return u.sized(2<<(c-0xdc), func(n int) (any, error) { return u.arrayN(n, depth) })
	case 0xde, 0xdf:
		return u.sized(2<<(c-0xde), func(n int) (any, error) { return u.mapN(n, depth) })
	}
	return nil, fmt.Errorf("msgpack: unsupported type byte 0x%x", c)
}

func (u *unpacker) arrayN(n int, depth int) (any, error) {
	if n > len(u.b) {
		return nil, errShort
	}
	out := make([]any, n)
	for i := range out {
		v, err := u.value(depth + 1)
		if err != nil {
			return nil, err
		}
		out[i] = v
	}
	return out, nil
}

func (u *unpacker) mapN(n int, depth int) (any, error) {
	if n > len(u.b) {
		return nil, errShort
	}
	out := make(map[string]any, n)
	for i := 0; i < n; i++ {
		k, err := u.value(depth + 1)
		if err != nil {
			return nil, err
		}
		v, err := u.value(depth + 1)
		if err != nil {
			return nil, err
		}
		if ks, ok := k.(string); ok {
			out[ks] = v
		}
	}
	return out, nil
}
