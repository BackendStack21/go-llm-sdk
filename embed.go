package llm

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	neturl "net/url"
	"sort"
	"strings"
	"time"
)

// ── Embeddings ───────────────────────────────────────────────────────────
//
// Embed turns texts into vectors. Two wire formats are supported:
//   - OpenAI-compatible: POST {base}/embeddings
//   - Gemini: POST {base}/v1beta/models/{model}:batchEmbedContents
//
// Anthropic has no embeddings endpoint: a ConfigError. Calls share the chat
// retry ladder, whole-call budget, custom headers and error taxonomy.

// EmbedRequest describes one embeddings call.
type EmbedRequest struct {
	Inputs     []string // required: one or more non-empty texts
	Dimensions int      // optional output dimensionality (0 = model default)
}

// EmbedResult carries one vector per input, in input order.
type EmbedResult struct {
	Embeddings [][]float64
	Model      string
	Usage      Usage // PromptTokens when the provider reports it
}

// Embed computes embeddings for req.Inputs with the named provider and
// model.
func (s *SDK) Embed(ctx context.Context, providerID, model string, req EmbedRequest) (*EmbedResult, error) {
	if strings.TrimSpace(model) == "" {
		return nil, &ConfigError{Msg: "embed request requires a model"}
	}
	if len(req.Inputs) == 0 {
		return nil, &ConfigError{Msg: "embed request requires at least one input"}
	}
	for i, in := range req.Inputs {
		if strings.TrimSpace(in) == "" {
			return nil, &ConfigError{Msg: fmt.Sprintf("embed input %d is empty", i)}
		}
	}
	if req.Dimensions < 0 {
		return nil, &ConfigError{Msg: "embed Dimensions must not be negative"}
	}
	p, err := s.usableProvider(providerID)
	if err != nil {
		return nil, err
	}
	if p.cfg.Format != FormatOpenAI && p.cfg.Format != FormatGemini {
		return nil, &ConfigError{Msg: "provider format " + string(p.cfg.Format) + " does not support embeddings"}
	}
	return s.newClient(p, s.timeout, nil).embed(ctx, model, req)
}

type oaEmbedRequest struct {
	Model          string   `json:"model"`
	Input          []string `json:"input"`
	Dimensions     int      `json:"dimensions,omitempty"`
	EncodingFormat string   `json:"encoding_format"`
}

type oaEmbedResponse struct {
	Data []struct {
		Index     int       `json:"index"`
		Embedding []float64 `json:"embedding"`
	} `json:"data"`
	Usage *struct {
		PromptTokens int `json:"prompt_tokens"`
	} `json:"usage"`
}

type gmEmbedItem struct {
	Model                string    `json:"model"`
	Content              gmContent `json:"content"`
	OutputDimensionality int       `json:"outputDimensionality,omitempty"`
}

type gmEmbedRequest struct {
	Requests []gmEmbedItem `json:"requests"`
}

type gmEmbedResponse struct {
	Embeddings []struct {
		Values []float64 `json:"values"`
	} `json:"embeddings"`
}

// buildEmbedRequest serializes the request for the provider's format.
func (pc *providerClient) buildEmbedRequest(model string, req EmbedRequest) ([]byte, string, error) {
	if pc.cfg.Format == FormatGemini {
		r := gmEmbedRequest{Requests: make([]gmEmbedItem, 0, len(req.Inputs))}
		for _, in := range req.Inputs {
			r.Requests = append(r.Requests, gmEmbedItem{
				Model:                "models/" + model,
				Content:              gmContent{Parts: []gmPart{{Text: in}}},
				OutputDimensionality: req.Dimensions,
			})
		}
		b, err := json.Marshal(r)
		return b, fmt.Sprintf("%s/v1beta/models/%s:batchEmbedContents", pc.base, neturl.PathEscape(model)), err
	}
	b, err := json.Marshal(oaEmbedRequest{Model: model, Input: req.Inputs, Dimensions: req.Dimensions, EncodingFormat: "float"})
	return b, pc.base + "/embeddings", err
}

// parseEmbedResponse decodes vectors in input order and checks the count.
func (pc *providerClient) parseEmbedResponse(data []byte, n int) ([][]float64, Usage, error) {
	if pc.cfg.Format == FormatGemini {
		var r gmEmbedResponse
		if err := json.Unmarshal(data, &r); err != nil {
			return nil, Usage{}, fmt.Errorf("llm: parse embeddings: %w", err)
		}
		if len(r.Embeddings) != n {
			return nil, Usage{}, fmt.Errorf("llm: embeddings: got %d vectors for %d inputs", len(r.Embeddings), n)
		}
		out := make([][]float64, n)
		for i, e := range r.Embeddings {
			out[i] = e.Values
		}
		return out, Usage{}, nil
	}
	var r oaEmbedResponse
	if err := json.Unmarshal(data, &r); err != nil {
		return nil, Usage{}, fmt.Errorf("llm: parse embeddings: %w", err)
	}
	if len(r.Data) != n {
		return nil, Usage{}, fmt.Errorf("llm: embeddings: got %d vectors for %d inputs", len(r.Data), n)
	}
	allZero := true
	for _, d := range r.Data {
		if d.Index != 0 {
			allZero = false
			break
		}
	}
	if allZero {
		// Gateways that omit index: trust response order.
		for i := range r.Data {
			r.Data[i].Index = i
		}
	}
	sort.SliceStable(r.Data, func(i, j int) bool { return r.Data[i].Index < r.Data[j].Index })
	out := make([][]float64, n)
	for i, d := range r.Data {
		if d.Index != i {
			return nil, Usage{}, fmt.Errorf("llm: embeddings: vector index %d out of range", d.Index)
		}
		out[i] = d.Embedding
	}
	var u Usage
	if r.Usage != nil {
		u.PromptTokens = r.Usage.PromptTokens
	}
	return out, u, nil
}

// Per-request input caps: Gemini batchEmbedContents takes at most 100
// requests, OpenAI /embeddings at most 2048 inputs. Larger calls are split
// into consecutive batches and concatenated in input order.
const (
	geminiEmbedBatch = 100
	openAIEmbedBatch = 2048
)

// embed runs the request in provider-sized batches, each with the shared
// retry ladder, all under one whole-call budget.
func (pc *providerClient) embed(ctx context.Context, model string, req EmbedRequest) (*EmbedResult, error) {
	model = strings.TrimPrefix(model, "models/") // Gemini ids as ListModels may echo them
	batch := openAIEmbedBatch
	if pc.cfg.Format == FormatGemini {
		batch = geminiEmbedBatch
	}
	ctx, cancel := context.WithTimeout(ctx, pc.requestTimeout())
	defer cancel()
	res := &EmbedResult{Model: model, Embeddings: make([][]float64, 0, len(req.Inputs))}
	for start := 0; start < len(req.Inputs); start += batch {
		part := req
		part.Inputs = req.Inputs[start:min(start+batch, len(req.Inputs))]
		body, url, err := pc.buildEmbedRequest(model, part)
		if err != nil {
			return nil, err
		}
		err = pc.withRetry(ctx, func() (time.Duration, error) {
			data, ra, err := pc.post(ctx, pc.buffered(), url, body)
			if err != nil {
				return ra, err
			}
			vecs, usage, perr := pc.parseEmbedResponse(data, len(part.Inputs))
			if perr != nil {
				// 2xx with an unusable body is a provider protocol failure:
				// typed, at the actual HTTP status, never retried.
				return 0, terminal(&APIError{Provider: pc.cfg.ID, Status: http.StatusOK, Message: perr.Error()})
			}
			res.Embeddings = append(res.Embeddings, vecs...)
			res.Usage.PromptTokens += usage.PromptTokens
			return 0, nil
		}, nil)
		if err != nil {
			return nil, err
		}
	}
	return res, nil
}
