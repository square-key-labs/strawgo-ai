package transports

import (
	"context"
	"errors"
	"fmt"
	"net/http"

	"github.com/gorilla/websocket"
	"github.com/square-key-labs/strawgo-ai/src/serializers"
)

type ClientConfig struct {
	URL        string
	Header     http.Header
	Serializer serializers.FrameSerializer
	Params     TransportParams
	Dialer     *websocket.Dialer
}

func Dial(ctx context.Context, cfg ClientConfig) (*Session, error) {
	if cfg.Serializer == nil {
		return nil, errors.New("transports: ClientConfig.Serializer is required")
	}
	dialer := cfg.Dialer
	if dialer == nil {
		dialer = websocket.DefaultDialer
	}
	params := cfg.Params.withDefaults()
	ws, resp, err := dialer.DialContext(ctx, cfg.URL, cfg.Header)
	if resp != nil && resp.Body != nil {
		_ = resp.Body.Close()
	}
	if err != nil {
		return nil, fmt.Errorf("websocket dial: %w", err)
	}
	sess := newSession(newWSConn(ws, params), cfg.Serializer, params, serializers.CallInfo{}, nil)
	sess.start()
	return sess, nil
}
