package transports

import "time"

// PlaybackKind categorizes a transport by how closely server send-complete
// approximates client playback-complete.
type PlaybackKind int

const (
	// PlaybackLocal: writes to a local audio sink (speaker). Send-complete is
	// effectively play-complete, so BotStoppedSpeakingFrame can fire immediately.
	PlaybackLocal PlaybackKind = iota

	// PlaybackNetworkBlind: networked transport that cannot signal playback
	// completion. A small drain pad is applied after send-complete to
	// approximate client-side jitter buffer and audio pipeline drain.
	PlaybackNetworkBlind
)

// PlaybackClassifier is an optional interface a transport may implement to
// declare its playback class. If unimplemented, PlaybackNetworkBlind is
// assumed as the conservative default.
type PlaybackClassifier interface {
	PlaybackKind() PlaybackKind
}

// DefaultDrainPad is the delay applied after send-complete for network
// transports that provide no playback ack. Covers typical client-side
// jitter buffer and audio pipeline latency over mobile telephony.
const DefaultDrainPad = 300 * time.Millisecond
