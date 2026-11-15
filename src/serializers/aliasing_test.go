package serializers

import (
	"bytes"
	"testing"

	"github.com/square-key-labs/strawgo-ai/src/frames"
)

func TestSerializedMessagesDoNotAliasAcrossCalls(t *testing.T) {
	cases := []struct {
		name string
		ser  FrameSerializer
		rate int
	}{
		{"twilio", NewTwilioFrameSerializer(TwilioConfig{StreamSID: "MZ"}), 8000},
		{"telnyx", NewTelnyxFrameSerializer(TelnyxConfig{}), 8000},
		{"plivo", NewPlivoFrameSerializer(PlivoConfig{StreamID: "s"}), 8000},
		{"exotel", NewExotelFrameSerializer(ExotelConfig{StreamSID: "s"}), 8000},
		{"rtvi", NewRTVIClientSerializer(), 16000},
		{"asterisk-ulaw", NewAsteriskFrameSerializer(AsteriskSerializerConfig{Codec: "ulaw", SampleRate: 8000}), 8000},
		{"jambonz-resampled", NewJambonzFrameSerializer(JambonzConfig{AudioInSampleRate: 8000, AudioOutSampleRate: 8000}), 16000},
		{"vonage-resampled", NewVonageFrameSerializer(VonageConfig{SampleRate: 8000}), 16000},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			first := tonePCM(320, tc.rate, 440, 8000)
			second := tonePCM(320, tc.rate, 1200, 3000)
			msgs, err := tc.ser.SerializeAudio(nil, first, tc.rate)
			if err != nil || len(msgs) != 1 {
				t.Fatalf("first serialize: %v (%d messages)", err, len(msgs))
			}
			want := bytes.Clone(msgs[0].Data)
			msgs, err = tc.ser.SerializeAudio(msgs, second, tc.rate)
			if err != nil || len(msgs) != 2 {
				t.Fatalf("second serialize: %v (%d messages)", err, len(msgs))
			}
			if !bytes.Equal(msgs[0].Data, want) {
				t.Fatal("serializing a second message changed the first")
			}
			if bytes.Equal(msgs[0].Data, msgs[1].Data) {
				t.Fatal("different audio produced identical messages")
			}
		})
	}

	t.Run("protobuf", func(t *testing.T) {
		s := NewProtobufFrameSerializer()
		msgs, err := s.Serialize(nil, frames.NewTextFrame("first message"))
		if err != nil || len(msgs) != 1 {
			t.Fatalf("first serialize: %v", err)
		}
		want := bytes.Clone(msgs[0].Data)
		if msgs, err = s.Serialize(msgs, frames.NewTextFrame("second")); err != nil || len(msgs) != 2 {
			t.Fatalf("second serialize: %v", err)
		}
		if !bytes.Equal(msgs[0].Data, want) {
			t.Fatal("serializing a second message changed the first")
		}
	})
}
