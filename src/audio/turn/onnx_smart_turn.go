package turn

import (
	"context"
	"encoding/binary"
	"fmt"
	"time"
)

type OnnxSmartTurnConfig struct {
	SockPath string
	Params   *SmartTurnParams
	Breaker  BreakerConfig
}

type OnnxSmartTurn struct {
	*smartTurn
	client *OnnxTurnClient
	pcm    []byte
}

func NewOnnxSmartTurn(config OnnxSmartTurnConfig) (*OnnxSmartTurn, error) {
	client, err := NewOnnxTurnClient(config.SockPath)
	if err != nil {
		return nil, fmt.Errorf("failed to connect to onnx-worker at %s: %w", config.SockPath, err)
	}
	o := &OnnxSmartTurn{client: client}
	o.smartTurn = newSmartTurn("OnnxSmartTurn", config.Params, config.Breaker, o.predictOnnx)
	return o, nil
}

func (o *OnnxSmartTurn) predictOnnx(ctx context.Context, samples []int16) (prediction, error) {
	o.pcm = int16ToBytes(o.pcm[:0], samples)
	start := time.Now()
	pred, err := o.client.AnalyzeContext(ctx, o.pcm, ModelSampleRate, 0)
	if err != nil {
		return prediction{}, err
	}
	return prediction{
		complete:    pred.Probability > 0.5,
		probability: float64(pred.Probability),
		inferenceMs: float64(time.Since(start).Microseconds()) / 1000,
	}, nil
}

func (o *OnnxSmartTurn) Close() error {
	return o.client.Close()
}

func int16ToBytes(dst []byte, src []int16) []byte {
	n := 2 * len(src)
	if cap(dst) < n {
		dst = make([]byte, n)
	}
	dst = dst[:n]
	for i, s := range src {
		binary.LittleEndian.PutUint16(dst[2*i:], uint16(s))
	}
	return dst
}
