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
	// StopResultContinue keeps evaluating subsequent strategies. Equivalent
	// to a legacy bool ShouldStop()=false return.
	StopResultContinue StopResult = iota

	// StopResultStop ends the turn but lets later strategies in the chain
	// still vote (e.g. for telemetry / observability). This is the V2
	// equivalent of pipecat ProcessFrameResult-non-STOP "yes" votes.
	//
	// NOTE: legacy bool ShouldStop()=true does NOT map to StopResultStop in
	// strawgo. To preserve the historical strawgo behavior where the loop
	// stopped on the first true, the dispatcher treats a V1 bool true as
	// equivalent to StopResultStopShortCircuit. If you want Stop's
	// "let later strategies vote" semantics, implement V2 explicitly.
	StopResultStop

	// StopResultStopShortCircuit ends the turn AND stops evaluating any
	// subsequent strategies in the chain. Useful when a confident "this is
	// the end" signal arrives and we don't want a more permissive strategy
	// to override it. Legacy bool true maps here for backwards compat
	// (see StopResultStop's NOTE).
	StopResultStopShortCircuit
)

// UserTurnStopStrategyV2 is a non-breaking richer interface that lets a
// strategy distinguish "vote yes, keep asking later strategies" (Stop)
// from "vote yes, stop asking" (StopShortCircuit). Existing strategies
// implementing only the bool-returning UserTurnStopStrategy keep working;
// see StopResultStop for the precise mapping.
//
// Only implement UserTurnStopStrategyV2 if you actually need to distinguish
// Stop from ShortCircuit; otherwise the original interface is simpler.
type UserTurnStopStrategyV2 interface {
	UserTurnStopStrategy

	// ShouldStopV2 returns a richer StopResult. Implementations that
	// implement V2 should make ShouldStop forward to ShouldStopV2 mapped
	// to bool (StopResultContinue → false, anything else → true) so they
	// satisfy both interfaces consistently.
	ShouldStopV2(frame any) StopResult
}
