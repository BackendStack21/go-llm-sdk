package llm

import (
	"context"
	"errors"
	"fmt"
	"sync"
)

// ToolHandler executes one model-requested tool call and returns its result.
type ToolHandler func(ctx context.Context, call ToolCall) (string, error)

// ToolLoopOptions tunes RunTools.
type ToolLoopOptions struct {
	// MaxTurns is the number of chat calls allowed; <=0 means 8.
	MaxTurns int
	// Parallel runs one turn's handlers concurrently. Results are still
	// appended in call order.
	Parallel bool
	// OnTurn is called after each chat call (1-based), before handlers run.
	// Panics in it are not recovered.
	OnTurn func(turn int, res *ChatResult)
}

// ErrMaxToolTurns reports that the model still wanted tools after MaxTurns.
var ErrMaxToolTurns = errors.New("llm: max tool turns exceeded")

const defaultMaxToolTurns = 8

// RunTools drives a buffered tool loop: it calls the model, runs the
// requested tools through handlers, appends the results and repeats until
// the model stops asking for tools. It returns the final result and the
// complete transcript (req.Messages plus everything appended). req.Messages
// is copied; the caller's slice is never mutated.
//
// A handler error (or panic, or a missing handler) becomes an IsError tool
// result and the loop continues. If MaxTurns chat calls all end in tool
// calls, the last result is returned with ErrMaxToolTurns and a transcript
// without the final (unanswered) assistant turn.
func RunTools(ctx context.Context, cc *ChatClient, req ChatRequest, handlers map[string]ToolHandler, opts ToolLoopOptions) (*ChatResult, []Message, error) {
	if cc == nil {
		return nil, nil, &ConfigError{Msg: "RunTools: nil ChatClient"}
	}
	maxTurns := opts.MaxTurns
	if maxTurns <= 0 {
		maxTurns = defaultMaxToolTurns
	}
	req.Messages = append([]Message(nil), req.Messages...)
	var last *ChatResult
	for turn := 1; turn <= maxTurns; turn++ {
		if err := ctx.Err(); err != nil {
			return last, req.Messages, err
		}
		res, err := cc.Call(ctx, &req)
		if err != nil {
			return last, req.Messages, err
		}
		last = res
		if opts.OnTurn != nil {
			opts.OnTurn(turn, res)
		}
		if res.FinishReason != FinishToolCalls || len(res.ToolCalls) == 0 {
			req.Messages = append(req.Messages, res.AssistantMessage())
			return res, req.Messages, nil
		}
		if turn == maxTurns {
			return res, req.Messages, fmt.Errorf("%w (%d turns)", ErrMaxToolTurns, maxTurns)
		}
		results := runToolCalls(ctx, res.ToolCalls, handlers, opts.Parallel)
		req.Messages = append(req.Messages, res.AssistantMessage())
		req.Messages = append(req.Messages, results...)
		if err := ctx.Err(); err != nil {
			return last, req.Messages, err
		}
	}
	return last, req.Messages, nil // unreachable: the loop returns on its last turn
}

func runToolCalls(ctx context.Context, calls []ToolCall, handlers map[string]ToolHandler, parallel bool) []Message {
	out := make([]Message, len(calls))
	run := func(i int) { out[i] = runToolCall(ctx, calls[i], handlers[calls[i].Name]) }
	if !parallel {
		for i := range calls {
			run(i)
		}
		return out
	}
	var wg sync.WaitGroup
	for i := range calls {
		wg.Add(1)
		go func() {
			defer wg.Done()
			run(i)
		}()
	}
	wg.Wait()
	return out
}

func runToolCall(ctx context.Context, call ToolCall, h ToolHandler) (msg Message) {
	msg = Message{Role: RoleTool, ToolCallID: call.ID, ToolName: call.Name}
	if h == nil {
		msg.Content, msg.IsError = "unknown tool", true
		return msg
	}
	defer func() {
		if r := recover(); r != nil {
			msg.Content, msg.IsError = fmt.Sprintf("tool %s panicked: %v", call.Name, r), true
		}
	}()
	content, err := h(ctx, call)
	if err != nil {
		msg.Content, msg.IsError = err.Error(), true
		return msg
	}
	msg.Content = content
	return msg
}
