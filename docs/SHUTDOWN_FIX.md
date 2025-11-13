# Graceful Shutdown Fix

## Problem

When the pipeline received an `EndFrame` and began shutdown, users experienced errors like:

```
Error reading message: read tcp ...: use of closed network connection
```

The pipeline would not shut down cleanly and required Ctrl+C to exit.

## Root Cause

There was a **race condition** during cleanup:

1. `Cleanup()` was called on services (Deepgram STT, ElevenLabs TTS)
2. The context was cancelled via `s.cancel()`
3. The websocket connection was **immediately** closed via `s.conn.Close()`
4. However, the `receiveTranscriptions()` and `receiveAudio()` goroutines were still running
5. These goroutines check `ctx.Done()` but might already be in the `default` case
6. They attempt to read from the **already closed** websocket connection
7. This generates "use of closed network connection" errors
8. These errors were treated as real errors and propagated as `ErrorFrame`s

## Solution

### 1. Improved Cleanup Order

**Before:**
```go
func (s *STTService) Cleanup() error {
    if s.cancel != nil {
        s.cancel()
    }
    if s.conn != nil {
        s.conn.Close()  // ❌ Immediate close causes race
    }
    return nil
}
```

**After:**
```go
func (s *STTService) Cleanup() error {
    // Cancel context first to signal goroutines to stop
    if s.cancel != nil {
        s.cancel()
    }

    // Give goroutines a moment to see the context cancellation
    time.Sleep(50 * time.Millisecond)

    // Now close the connection
    if s.conn != nil {
        s.conn.Close()
        s.conn = nil
    }
    return nil
}
```

### 2. Graceful Error Handling

**Before:**
```go
_, message, err := s.conn.ReadMessage()
if err != nil {
    log.Printf("[DeepgramSTT] Error reading message: %v", err)
    s.PushFrame(frames.NewErrorFrame(err), frames.Upstream)  // ❌ Treats all errors as problems
    return
}
```

**After:**
```go
_, message, err := s.conn.ReadMessage()
if err != nil {
    // Check if this is a normal closure during shutdown
    if websocket.IsCloseError(err, websocket.CloseNormalClosure, websocket.CloseGoingAway) ||
        strings.Contains(err.Error(), "use of closed network connection") {
        log.Printf("[DeepgramSTT] Connection closed normally")
        return  // ✅ Exit gracefully without error
    }
    log.Printf("[DeepgramSTT] Error reading message: %v", err)
    s.PushFrame(frames.NewErrorFrame(err), frames.Upstream)
    return
}
```

## Changes Made

### Files Modified

1. **`src/services/deepgram/stt.go`**
   - Added `strings` import
   - Updated `Cleanup()` to delay connection close
   - Updated `receiveTranscriptions()` to handle normal closure gracefully

2. **`src/services/elevenlabs/tts.go`**
   - Updated `Cleanup()` to delay connection close
   - Updated `receiveAudio()` to handle normal closure gracefully

## How It Works Now

### Shutdown Sequence

```
EndFrame received
    ↓
Service.Cleanup() called
    ↓
1. Context cancelled (s.cancel())
    ↓
2. Wait 50ms for goroutines to see cancellation
    ↓
3. Close websocket connection (s.conn.Close())
    ↓
Goroutines see ctx.Done() and exit cleanly
OR
Goroutines see connection closed and recognize it as normal closure
    ↓
No error frames generated
    ↓
Pipeline shuts down cleanly ✅
```

## Testing

The fix ensures:

1. **Normal closure** during shutdown doesn't generate error frames
2. **Goroutines** have time to see context cancellation before connection closes
3. **Race condition** is minimized by the 50ms delay
4. **Real errors** (network failures, unexpected closes) are still caught and reported
5. **Clean exit** - no need for Ctrl+C

## Best Practices

When implementing similar cleanup patterns:

1. **Cancel context first** - Signal to goroutines to stop
2. **Add a small delay** - Give goroutines time to react (50-100ms is usually sufficient)
3. **Close resources last** - After goroutines have stopped
4. **Handle expected errors gracefully** - Distinguish between normal shutdown and real errors
5. **Check for specific error types** - Use `websocket.IsCloseError()` and string matching

## Alternative Solutions Considered

### 1. WaitGroup Approach
```go
func (s *STTService) Cleanup() error {
    if s.cancel != nil {
        s.cancel()
    }
    s.wg.Wait()  // Wait for all goroutines to finish
    if s.conn != nil {
        s.conn.Close()
    }
    return nil
}
```

**Why not used:** Requires adding WaitGroups to all services and managing them properly. The sleep approach is simpler and works well for cleanup scenarios.

### 2. Done Channel Approach
```go
done := make(chan struct{})
go func() {
    s.receiveTranscriptions()
    close(done)
}()

func Cleanup() {
    s.cancel()
    <-done  // Wait for goroutine to finish
    s.conn.Close()
}
```

**Why not used:** More complex, requires refactoring existing goroutine management. The current solution is simpler and sufficient.

## Future Improvements

1. **Proper WaitGroup Management** - For more robust shutdown sequencing
2. **Shutdown Timeout** - Add a maximum wait time in case goroutines hang
3. **Graceful Close Message** - Send a proper close frame to the websocket before closing
4. **Connection State Tracking** - Track connection state to avoid operations on closed connections

## Related Issues

- This fix resolves the "use of closed network connection" errors during shutdown
- It ensures clean exit without requiring Ctrl+C
- It prevents spurious error logs that made it look like failures occurred
