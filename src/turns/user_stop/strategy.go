package user_stop

// UserTurnStopStrategy decides when a user turn should end.
type UserTurnStopStrategy interface {
	// ShouldStop returns true when the provided frame indicates a user turn end.
	ShouldStop(frame any) bool

	// Reset clears any internal strategy state.
	Reset()
}

// StopResult is the richer result returned by UserTurnStopStrategyV2 strategies.
// Mirrors pipecat's ProcessFrameResult.STOP semantics from PR #4064: a strategy
// can not only signal "stop the turn" but also "stop the turn AND don't run any
// later strategies in the chain".
type StopResult int

const (
	// StopResultContinue keeps evaluating subsequent strategies. Maps from
	// the bool-result world to ShouldStop()=false.
	StopResultContinue StopResult = iota

	// StopResultStop ends the turn. Subsequent strategies in the chain still
	// get evaluated. Maps from ShouldStop()=true.
	StopResultStop

	// StopResultStopShortCircuit ends the turn AND stops evaluating any
	// subsequent strategies in the chain. Useful when a confident "this is
	// the end" signal arrives and we don't want a more permissive strategy
	// to override it.
	StopResultStopShortCircuit
)

// UserTurnStopStrategyV2 is a non-breaking richer interface that lets a
// strategy short-circuit later strategies in the chain. Existing strategies
// implementing only the bool-returning UserTurnStopStrategy keep working —
// the aggregator iteration treats their true/false result as
// StopResultStop / StopResultContinue.
//
// Only implement UserTurnStopStrategyV2 if you actually need short-circuit
// semantics; otherwise the original interface is simpler.
type UserTurnStopStrategyV2 interface {
	UserTurnStopStrategy

	// ShouldStopV2 returns a richer StopResult. Implementations that
	// implement V2 should make ShouldStop forward to ShouldStopV2 mapped
	// to bool (StopResultContinue → false, anything else → true) so they
	// satisfy both interfaces consistently.
	ShouldStopV2(frame any) StopResult
}
