package user_start

type UserTurnStartStrategy interface {
	ShouldStart(frame any) bool
	EnableInterruptions() bool
	Reset()
}

type AggregationResetter interface {
	ConsumeAggregationReset() bool
}

type TranscriptRestorer interface {
	TakeRestoredText() (string, bool)
}
