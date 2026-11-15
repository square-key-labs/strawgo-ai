package webrtc

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/google/uuid"
	"github.com/pion/sdp/v3"
	pwebrtc "github.com/pion/webrtc/v4"
	"github.com/square-key-labs/strawgo-ai/src/logger"
)

type SessionHandler func(ctx context.Context, s *Session) error

type ConnectionMode int

const (
	MultipleConnections ConnectionMode = iota
	SingleConnection
)

var (
	ErrShuttingDown     = errors.New("webrtc: handler shutting down")
	ErrNotFound         = errors.New("webrtc: peer connection not found")
	ErrBadRequest       = errors.New("webrtc: bad request")
	ErrNoSupportedCodec = errors.New("webrtc: no supported audio codec offered")
)

type Config struct {
	ICEServers     []pwebrtc.ICEServer
	Handler        SessionHandler
	Params         Params
	Codecs         []Codec
	SettingEngine  *pwebrtc.SettingEngine
	ConnectionMode ConnectionMode
	GatherTimeout  time.Duration
}

type Handler struct {
	cfg    Config
	params Params
	log    *logger.Logger

	mu         sync.Mutex
	sessions   map[string]*Session
	iceServers []pwebrtc.ICEServer
	closed     bool

	baseCtx context.Context
	cancel  context.CancelFunc
	wg      sync.WaitGroup
}

func NewHandler(cfg Config) (*Handler, error) {
	if cfg.Handler == nil {
		return nil, errors.New("webrtc: Config.Handler is required")
	}
	if len(cfg.Codecs) == 0 {
		cfg.Codecs = DefaultCodecs()
	}
	if cfg.GatherTimeout <= 0 {
		cfg.GatherTimeout = DefaultGatherTimeout
	}
	ctx, cancel := context.WithCancel(context.Background())
	return &Handler{
		cfg:        cfg,
		params:     cfg.Params.withDefaults(),
		log:        logger.WithPrefix("WebRTCHandler"),
		sessions:   make(map[string]*Session),
		iceServers: cfg.ICEServers,
		baseCtx:    ctx,
		cancel:     cancel,
	}, nil
}

func (h *Handler) UpdateICEServers(servers []pwebrtc.ICEServer) {
	h.mu.Lock()
	h.iceServers = servers
	h.mu.Unlock()
}

func (h *Handler) Session(pcID string) *Session {
	h.mu.Lock()
	defer h.mu.Unlock()
	return h.sessions[pcID]
}

func (h *Handler) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	r.Body = http.MaxBytesReader(w, r.Body, maxBodyBytes)
	switch r.Method {
	case http.MethodPost:
		var req OfferRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			h.writeError(w, fmt.Errorf("%w: %v", ErrBadRequest, err))
			return
		}
		ans, err := h.HandleOffer(r.Context(), req, r)
		if err != nil {
			h.writeError(w, err)
			return
		}
		writeJSON(w, http.StatusOK, ans)
	case http.MethodPatch:
		var req PatchRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			h.writeError(w, fmt.Errorf("%w: %v", ErrBadRequest, err))
			return
		}
		if err := h.HandleCandidates(r.Context(), req); err != nil {
			h.writeError(w, err)
			return
		}
		writeJSON(w, http.StatusOK, map[string]string{"status": "success"})
	default:
		w.Header().Set("Allow", "POST, PATCH")
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
	}
}

func (h *Handler) writeError(w http.ResponseWriter, err error) {
	status := http.StatusInternalServerError
	switch {
	case errors.Is(err, ErrBadRequest), errors.Is(err, ErrNoSupportedCodec):
		status = http.StatusBadRequest
	case errors.Is(err, ErrNotFound), errors.Is(err, ErrSessionClosed):
		status = http.StatusNotFound
	case errors.Is(err, ErrShuttingDown):
		status = http.StatusServiceUnavailable
	}
	if status == http.StatusInternalServerError {
		h.log.Warn("request failed: %v", err)
	}
	http.Error(w, err.Error(), status)
}

func writeJSON(w http.ResponseWriter, status int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(v)
}

func (h *Handler) HandleOffer(ctx context.Context, req OfferRequest, r *http.Request) (Answer, error) {
	if req.Type != pwebrtc.SDPTypeOffer.String() {
		return Answer{}, fmt.Errorf("%w: type must be offer, got %q", ErrBadRequest, req.Type)
	}
	h.mu.Lock()
	if h.closed {
		h.mu.Unlock()
		return Answer{}, ErrShuttingDown
	}
	if err := h.checkSingle(req.PCID); err != nil {
		h.mu.Unlock()
		return Answer{}, err
	}
	sess := h.sessions[req.PCID]
	servers := h.iceServers
	h.mu.Unlock()

	if sess != nil {
		desc, err := sess.negotiate(ctx, req.SDP, req.RestartPC)
		if err != nil {
			return Answer{}, err
		}
		return Answer{SDP: desc.SDP, Type: desc.Type.String(), PCID: sess.id}, nil
	}

	codec, err := chooseCodec(req.SDP, h.cfg.Codecs)
	if err != nil {
		return Answer{}, err
	}
	data := req.RequestData
	if len(data) == 0 {
		data = req.RequestDataCamel
	}
	id := "pc-" + strings.ReplaceAll(uuid.NewString(), "-", "")
	sess, err = newSession(h, id, codec, servers, r, data)
	if err != nil {
		return Answer{}, err
	}
	desc, err := sess.negotiate(ctx, req.SDP, false)
	if err != nil {
		_ = sess.Close()
		sess.Wait()
		return Answer{}, err
	}
	h.mu.Lock()
	if h.closed {
		h.mu.Unlock()
		_ = sess.Close()
		sess.Wait()
		return Answer{}, ErrShuttingDown
	}
	h.sessions[id] = sess
	h.wg.Add(1)
	h.mu.Unlock()
	go h.run(sess)
	return Answer{SDP: desc.SDP, Type: desc.Type.String(), PCID: id}, nil
}

func (h *Handler) checkSingle(pcID string) error {
	if h.cfg.ConnectionMode != SingleConnection || len(h.sessions) == 0 {
		return nil
	}
	if pcID == "" {
		return fmt.Errorf("%w: cannot create new connection with existing connection active", ErrBadRequest)
	}
	if _, ok := h.sessions[pcID]; !ok {
		return fmt.Errorf("%w: pc_id mismatch with existing connection", ErrBadRequest)
	}
	return nil
}

func (h *Handler) HandleCandidates(ctx context.Context, req PatchRequest) error {
	sess := h.Session(req.PCID)
	if sess == nil {
		return ErrNotFound
	}
	return sess.addCandidates(ctx, req.Candidates)
}

func (h *Handler) run(sess *Session) {
	defer h.wg.Done()
	sess.log.Info("connected (codec=%s)", sess.codecName())
	if err := h.runHandler(sess); err != nil {
		sess.log.Warn("handler: %v", err)
	}
	_ = sess.Close()
	sess.Wait()
	h.mu.Lock()
	if h.sessions[sess.id] == sess {
		delete(h.sessions, sess.id)
	}
	h.mu.Unlock()
	sess.log.Info("closed")
}

func (h *Handler) runHandler(sess *Session) (err error) {
	defer func() {
		if r := recover(); r != nil {
			err = fmt.Errorf("session handler panic: %v", r)
		}
	}()
	return h.cfg.Handler(h.baseCtx, sess)
}

func (h *Handler) Shutdown(ctx context.Context) error {
	h.mu.Lock()
	h.closed = true
	sessions := make([]*Session, 0, len(h.sessions))
	for _, s := range h.sessions {
		sessions = append(sessions, s)
	}
	h.mu.Unlock()
	h.cancel()
	for _, s := range sessions {
		_ = s.Close()
	}
	done := make(chan struct{})
	go func() {
		h.wg.Wait()
		close(done)
	}()
	select {
	case <-done:
		return nil
	case <-ctx.Done():
		return ctx.Err()
	}
}

func chooseCodec(offer string, codecs []Codec) (Codec, error) {
	var sd sdp.SessionDescription
	if err := sd.UnmarshalString(offer); err != nil {
		return nil, fmt.Errorf("%w: invalid sdp: %v", ErrBadRequest, err)
	}
	var offered []sdp.Codec
	hasAudio := false
	for _, m := range sd.MediaDescriptions {
		if m.MediaName.Media != "audio" {
			continue
		}
		hasAudio = true
		byPT := sd.GetCodecMap()
		for _, f := range m.MediaName.Formats {
			pt, err := strconv.ParseUint(f, 10, 8)
			if err != nil {
				continue
			}
			if c, ok := byPT[uint8(pt)]; ok {
				offered = append(offered, c)
			}
		}
		break
	}
	if !hasAudio {
		return nil, nil
	}
	for _, c := range codecs {
		for _, o := range offered {
			if matchCodec(c, o.Name, o.ClockRate) {
				return c, nil
			}
		}
	}
	return nil, ErrNoSupportedCodec
}
