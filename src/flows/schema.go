package flows

import (
	"context"
	"fmt"
	"reflect"
	"strings"
)

func Typed[T any](name, description string, fn func(ctx context.Context, args T, m *Manager) (any, *Node, error)) Function {
	props, required := structSchema(reflect.TypeOf((*T)(nil)).Elem())
	return Function{
		Name:        name,
		Description: description,
		Properties:  props,
		Required:    required,
		Handler: func(ctx context.Context, args Args, m *Manager) (any, *Node, error) {
			v, err := Decode[T](args)
			if err != nil {
				return nil, nil, err
			}
			return fn(ctx, v, m)
		},
	}
}

func structSchema(t reflect.Type) (map[string]any, []string) {
	for t.Kind() == reflect.Pointer {
		t = t.Elem()
	}
	props := map[string]any{}
	var required []string
	if t.Kind() != reflect.Struct {
		return props, required
	}
	for i := 0; i < t.NumField(); i++ {
		field := t.Field(i)
		if !field.IsExported() {
			continue
		}
		name, omitempty, skip := jsonName(field)
		if skip {
			continue
		}
		if field.Anonymous && name == "" {
			inner, innerRequired := structSchema(field.Type)
			for k, v := range inner {
				props[k] = v
			}
			required = append(required, innerRequired...)
			continue
		}
		if name == "" {
			name = field.Name
		}
		schema := typeSchema(field.Type)
		if desc := field.Tag.Get("desc"); desc != "" {
			schema["description"] = desc
		}
		if enum := field.Tag.Get("enum"); enum != "" {
			values := strings.Split(enum, ",")
			list := make([]any, len(values))
			for i, v := range values {
				list[i] = strings.TrimSpace(v)
			}
			schema["enum"] = list
		}
		props[name] = schema
		if !omitempty && field.Type.Kind() != reflect.Pointer {
			required = append(required, name)
		}
	}
	return props, required
}

func jsonName(field reflect.StructField) (name string, omitempty, skip bool) {
	tag := field.Tag.Get("json")
	if tag == "-" {
		return "", false, true
	}
	parts := strings.Split(tag, ",")
	for _, opt := range parts[1:] {
		if opt == "omitempty" || opt == "omitzero" {
			omitempty = true
		}
	}
	return parts[0], omitempty, false
}

func typeSchema(t reflect.Type) map[string]any {
	for t.Kind() == reflect.Pointer {
		t = t.Elem()
	}
	switch t.Kind() {
	case reflect.String:
		return map[string]any{"type": "string"}
	case reflect.Bool:
		return map[string]any{"type": "boolean"}
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64,
		reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64:
		return map[string]any{"type": "integer"}
	case reflect.Float32, reflect.Float64:
		return map[string]any{"type": "number"}
	case reflect.Slice, reflect.Array:
		return map[string]any{"type": "array", "items": typeSchema(t.Elem())}
	case reflect.Map:
		return map[string]any{"type": "object", "additionalProperties": typeSchema(t.Elem())}
	case reflect.Struct:
		props, required := structSchema(t)
		out := map[string]any{"type": "object", "properties": props}
		if len(required) > 0 {
			out["required"] = required
		}
		return out
	case reflect.Interface:
		return map[string]any{}
	}
	panic(fmt.Sprintf("flows: unsupported argument type %s", t))
}
