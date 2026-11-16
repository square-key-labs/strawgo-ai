package user_start_test

import (
	"testing"
	"time"

	"github.com/square-key-labs/strawgo-ai/src/frames"
	"github.com/square-key-labs/strawgo-ai/src/turns/user_start"
)

func TestMinWordsStartStrategyBotSpeaking(t *testing.T) {
	strategy := user_start.NewMinWordsUserTurnStartStrategy(3, true)
	strategy.ShouldStart(frames.NewBotStartedSpeakingFrame())

	if strategy.ShouldStart(frames.NewTextFrame("ignore these many words")) {
		t.Fatalf("expected non-transcription frame to not trigger")
	}
	if strategy.ShouldStart(frames.NewTranscriptionFrame("hello", true)) {
		t.Fatalf("expected one word to not trigger while bot speaks")
	}
	if !strategy.ConsumeAggregationReset() {
		t.Fatalf("expected rejected final transcript to request an aggregation reset")
	}
	if !strategy.ShouldStart(frames.NewTranscriptionFrame("there now", true)) {
		t.Fatalf("expected words of the same utterance to reach the threshold")
	}
	if !strategy.EnableInterruptions() {
		t.Fatalf("expected enable interruptions to be true")
	}
}

func TestMinWordsStartStrategyCountsCurrentUtteranceOnly(t *testing.T) {
	strategy := user_start.NewMinWordsUserTurnStartStrategy(3, true)
	strategy.ShouldStart(frames.NewBotStartedSpeakingFrame())

	strategy.ShouldStart(frames.NewUserStartedSpeakingFrame())
	if strategy.ShouldStart(frames.NewTranscriptionFrame("yeah", true)) {
		t.Fatalf("expected one word to not trigger")
	}
	strategy.ShouldStart(frames.NewUserStartedSpeakingFrame())
	if strategy.ShouldStart(frames.NewTranscriptionFrame("uh huh", true)) {
		t.Fatalf("expected words of a previous utterance not to count")
	}
	strategy.ShouldStart(frames.NewUserStartedSpeakingFrame())
	for i := 0; i < 3; i++ {
		if strategy.ShouldStart(frames.NewTranscriptionFrame("ok", false)) {
			t.Fatalf("expected repeated interim results not to accumulate")
		}
	}
	if !strategy.ShouldStart(frames.NewTranscriptionFrame("wait a second", false)) {
		t.Fatalf("expected interim with enough words to trigger")
	}
}

func TestMinWordsStartStrategyBotSilent(t *testing.T) {
	strategy := user_start.NewMinWordsUserTurnStartStrategy(0, false)

	if !strategy.ShouldStart(frames.NewTranscriptionFrame("one", false)) {
		t.Fatalf("expected a single word to trigger when the bot is silent")
	}
	if strategy.EnableInterruptions() {
		t.Fatalf("expected enable interruptions to be false")
	}
	strategy.ShouldStart(frames.NewBotStartedSpeakingFrame())
	strategy.Reset()
	if !strategy.ShouldStart(frames.NewTranscriptionFrame("one", true)) {
		t.Fatalf("expected reset to clear bot speaking state")
	}
}

func botSpeakingBackchannel(params user_start.BackchannelParams) *user_start.BackchannelUserTurnStartStrategy {
	s := user_start.NewBackchannelUserTurnStartStrategy(params, true)
	s.ShouldStart(frames.NewBotStartedSpeakingFrame())
	return s
}

func TestBackchannelStartStrategyDefaultWordsWhileBotSpeaks(t *testing.T) {
	for _, words := range [][]string{user_start.EnglishBackchannelWords(), user_start.HindiBackchannelWords()} {
		for _, w := range words {
			s := botSpeakingBackchannel(user_start.BackchannelParams{})
			if s.ShouldStart(frames.NewUserStartedSpeakingFrame()) {
				t.Fatalf("%q: speech start interrupted the bot", w)
			}
			if s.ShouldStart(frames.NewInterimTranscriptionFrame(w, "")) {
				t.Fatalf("%q: interim started a turn", w)
			}
			if s.ShouldStart(frames.NewTranscriptionFrame(w, true)) {
				t.Fatalf("%q: final started a turn", w)
			}
			if got, ok := s.TakeBackchannel(); !ok || got != w {
				t.Fatalf("%q: backchannel = %q, %v", w, got, ok)
			}
			if !s.ConsumeAggregationReset() {
				t.Fatalf("%q: suppressed transcript kept for the next turn", w)
			}
		}
	}
}

func TestBackchannelStartStrategyBotSilentAlwaysStarts(t *testing.T) {
	for _, w := range user_start.DefaultBackchannelWords() {
		s := user_start.NewBackchannelUserTurnStartStrategy(user_start.BackchannelParams{}, true)
		if !s.ShouldStart(frames.NewTranscriptionFrame(w, true)) {
			t.Fatalf("%q did not start a turn while the bot was silent", w)
		}
		if _, ok := s.TakeBackchannel(); ok {
			t.Fatalf("%q reported as a backchannel while the bot was silent", w)
		}
	}
	s := user_start.NewBackchannelUserTurnStartStrategy(user_start.BackchannelParams{}, true)
	s.ShouldStart(frames.NewBotStartedSpeakingFrame())
	s.ShouldStart(frames.NewBotStoppedSpeakingFrame())
	if !s.ShouldStart(frames.NewUserStartedSpeakingFrame()) {
		t.Fatal("speech start did not start a turn once the bot stopped")
	}
}

func TestBackchannelStartStrategyRealInterruptions(t *testing.T) {
	cases := []frames.Frame{
		frames.NewTranscriptionFrame("okay but what about tomorrow", true),
		frames.NewInterimTranscriptionFrame("okay but", ""),
		frames.NewTranscriptionFrame("stop", true),
		frames.NewTranscriptionFrame("haan ji theek hai bilkul sahi", true),
		frames.NewTranscriptionFrame("okay okay okay okay okay", true),
		frames.NewTranscriptionFrame("theek hoon", true),
		frames.NewTranscriptionFrame("ठीक हूँ", true),
		frames.NewTranscriptionFrame("सही", true),
		frames.NewTranscriptionFrame("हाँ नहीं", true),
	}
	for _, f := range cases {
		s := botSpeakingBackchannel(user_start.BackchannelParams{})
		if !s.ShouldStart(f) {
			t.Fatalf("%+v did not interrupt", f)
		}
		if _, ok := s.TakeBackchannel(); ok {
			t.Fatalf("%+v reported as a backchannel", f)
		}
	}
}

func TestBackchannelStartStrategyNormalisesTranscripts(t *testing.T) {
	for _, text := range []string{"Hmmm...", "Okaaay!", "OK.", "Uh-huh.", "Yeah, yeah.", "Mm-hmm", "हाँ।", "Theek hai, ji.", "Accha accha", "Right, got it.", "हाँ हाँ", "जी हाँ, ठीक है।", "Thik h", "हम्म, अच्छा"} {
		s := botSpeakingBackchannel(user_start.BackchannelParams{})
		if s.ShouldStart(frames.NewTranscriptionFrame(text, true)) {
			t.Fatalf("%q interrupted the bot", text)
		}
		if got, ok := s.TakeBackchannel(); !ok || got != text {
			t.Fatalf("%q: backchannel = %q, %v", text, got, ok)
		}
	}
}

func TestBackchannelStartStrategyInterimThenFinal(t *testing.T) {
	s := botSpeakingBackchannel(user_start.BackchannelParams{})
	s.ShouldStart(frames.NewUserStartedSpeakingFrame())
	for _, interim := range []string{"th", "theek", "theek hai"} {
		if s.ShouldStart(frames.NewInterimTranscriptionFrame(interim, "")) {
			t.Fatalf("interim %q started a turn", interim)
		}
	}
	if _, ok := s.TakeBackchannel(); ok {
		t.Fatal("an interim transcript was reported as a backchannel")
	}
	if s.ShouldStart(frames.NewTranscriptionFrame("theek hai", true)) {
		t.Fatal("final backchannel started a turn")
	}
	if _, ok := s.TakeBackchannel(); !ok {
		t.Fatal("final backchannel not reported")
	}

	s = botSpeakingBackchannel(user_start.BackchannelParams{})
	if s.ShouldStart(frames.NewInterimTranscriptionFrame("oka", "")) {
		t.Fatal("partial word started a turn")
	}
	if !s.ShouldStart(frames.NewInterimTranscriptionFrame("okay wait", "")) {
		t.Fatal("interim with a real word did not interrupt")
	}
}

func TestBackchannelStartStrategySafetyValves(t *testing.T) {
	t.Run("second utterance within the window", func(t *testing.T) {
		s := botSpeakingBackchannel(user_start.BackchannelParams{RepeatWindow: time.Second})
		s.ShouldStart(frames.NewUserStartedSpeakingFrame())
		s.ShouldStart(frames.NewTranscriptionFrame("okay", true))
		s.ShouldStart(frames.NewUserStoppedSpeakingFrame())
		if !s.ShouldStart(frames.NewUserStartedSpeakingFrame()) {
			t.Fatal("second utterance did not interrupt")
		}
	})
	t.Run("second final without VAD", func(t *testing.T) {
		s := botSpeakingBackchannel(user_start.BackchannelParams{RepeatWindow: time.Second})
		s.ShouldStart(frames.NewTranscriptionFrame("okay", true))
		if !s.ShouldStart(frames.NewTranscriptionFrame("okay", true)) {
			t.Fatal("repeated backchannel did not interrupt")
		}
	})
	t.Run("second utterance after the window", func(t *testing.T) {
		s := botSpeakingBackchannel(user_start.BackchannelParams{RepeatWindow: 10 * time.Millisecond})
		s.ShouldStart(frames.NewTranscriptionFrame("okay", true))
		time.Sleep(30 * time.Millisecond)
		if s.ShouldStart(frames.NewUserStartedSpeakingFrame()) {
			t.Fatal("utterance after the window interrupted")
		}
	})
	t.Run("speaking too long", func(t *testing.T) {
		s := botSpeakingBackchannel(user_start.BackchannelParams{MaxSpeechDuration: 20 * time.Millisecond})
		s.ShouldStart(frames.NewUserStartedSpeakingFrame())
		deadline, ok := s.NextDeadline()
		if !ok {
			t.Fatal("no deadline while the user speaks over the bot")
		}
		if s.ShouldStart(nil) {
			t.Fatal("interrupted before the deadline")
		}
		time.Sleep(time.Until(deadline) + 5*time.Millisecond)
		if !s.ShouldStart(nil) {
			t.Fatal("long speech did not interrupt")
		}
	})
	t.Run("bot stops while the user is still speaking", func(t *testing.T) {
		s := botSpeakingBackchannel(user_start.BackchannelParams{})
		s.ShouldStart(frames.NewUserStartedSpeakingFrame())
		if !s.ShouldStart(frames.NewBotStoppedSpeakingFrame()) {
			t.Fatal("undecided speech did not start a turn once the bot stopped")
		}
	})
}

func TestBackchannelStartStrategyCustomWords(t *testing.T) {
	s := botSpeakingBackchannel(user_start.BackchannelParams{Words: []string{"bas"}})
	if !s.ShouldStart(frames.NewTranscriptionFrame("okay", true)) {
		t.Fatal("a replaced word list still suppressed okay")
	}
	s = botSpeakingBackchannel(user_start.BackchannelParams{Words: append(user_start.DefaultBackchannelWords(), "bas")})
	if s.ShouldStart(frames.NewTranscriptionFrame("Bas.", true)) {
		t.Fatal("an added word interrupted")
	}
}

func TestBackchannelStartStrategyBotStopsAfterBackchannel(t *testing.T) {
	s := botSpeakingBackchannel(user_start.BackchannelParams{BotStopWindow: -1})
	s.ShouldStart(frames.NewUserStartedSpeakingFrame())
	s.ShouldStart(frames.NewTranscriptionFrame("okay", true))
	if s.ShouldStart(frames.NewBotStoppedSpeakingFrame()) {
		t.Fatal("an already suppressed backchannel started an empty turn when the bot stopped")
	}
}

func TestBackchannelStartStrategyAnswerJustBeforeBotStops(t *testing.T) {
	s := botSpeakingBackchannel(user_start.BackchannelParams{})
	s.ShouldStart(frames.NewTranscriptionFrame("Yes.", true))
	if !s.ShouldStart(frames.NewBotStoppedSpeakingFrame()) {
		t.Fatal("a one-word answer just before the bot stopped did not start a turn")
	}
	if got, ok := s.TakeRestoredText(); !ok || got != "Yes." {
		t.Fatalf("restored text = %q, %v", got, ok)
	}

	s = botSpeakingBackchannel(user_start.BackchannelParams{BotStopWindow: 10 * time.Millisecond})
	s.ShouldStart(frames.NewTranscriptionFrame("okay", true))
	time.Sleep(30 * time.Millisecond)
	if s.ShouldStart(frames.NewBotStoppedSpeakingFrame()) {
		t.Fatal("a backchannel long before the bot stopped started a turn")
	}
	if _, ok := s.TakeRestoredText(); ok {
		t.Fatal("restored text without a turn")
	}
}

func TestBackchannelStartStrategyKeepsAcknowledgementBeforeRealTurn(t *testing.T) {
	s := botSpeakingBackchannel(user_start.BackchannelParams{})
	s.ShouldStart(frames.NewTranscriptionFrame("Yes", true))
	if !s.ShouldStart(frames.NewTranscriptionFrame("I'd like to proceed", true)) {
		t.Fatal("continuation did not start a turn")
	}
	if got, ok := s.TakeRestoredText(); !ok || got != "Yes" {
		t.Fatalf("restored text = %q, %v", got, ok)
	}
}

func TestBackchannelStartStrategyIgnoresTranscriptsDuringTurn(t *testing.T) {
	s := botSpeakingBackchannel(user_start.BackchannelParams{})
	s.HandleUserTurnStarted()
	if s.ShouldStart(frames.NewTranscriptionFrame("okay", true)) {
		t.Fatal("started a turn during a turn")
	}
	if _, ok := s.TakeBackchannel(); ok {
		t.Fatal("reported a backchannel during a turn")
	}
	s.HandleUserTurnStopped()
	if s.ShouldStart(frames.NewUserStartedSpeakingFrame()) {
		t.Fatal("a transcript inside the turn armed the repeat window")
	}
}
