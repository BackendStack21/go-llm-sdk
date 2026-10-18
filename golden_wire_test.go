package llm

// Golden wire bodies recorded from the pre-change code (feature m6b, AC-7):
// a request that uses none of the new fields (DocumentPart, ImageURLPart,
// TopK, Gemini generationConfig in Extra, EmbedRequest.TaskType) must keep
// marshalling to these exact bytes on every format. These tests PASS before
// any production change and must keep passing.

import (
	"encoding/json"
	"testing"
)

func goldenReq() *ChatRequest {
	return &ChatRequest{
		System: []SystemBlock{{Text: "be brief"}},
		Messages: []Message{
			{Role: RoleUser, Content: "weather in Paris?"},
			{Role: RoleAssistant, Content: "checking", ToolCalls: []ToolCall{{ID: "c1", Name: "wx", Arguments: `{"city":"Paris"}`}}},
			{Role: RoleTool, ToolCallID: "c1", ToolName: "wx", Content: "sunny"},
			{Role: RoleUser, Parts: []ContentPart{TextPart("and"), ImagePart("image/png", []byte{1, 2, 3}), TextPart("London?")}},
		},
		Tools:       []ToolDef{{Name: "wx", Description: "weather", Parameters: json.RawMessage(`{"type":"object","properties":{"city":{"type":"string"}}}`)}},
		MaxTokens:   256,
		Temperature: 0.5,
		TopP:        0.9,
		Stop:        []string{"END"},
	}
}

func goldenToolImgReq() *ChatRequest {
	r := goldenReq()
	r.Messages = []Message{
		{Role: RoleUser, Content: "look"},
		{Role: RoleAssistant, ToolCalls: []ToolCall{{ID: "c2", Name: "wx", Arguments: `{}`}}},
		{Role: RoleTool, ToolCallID: "c2", ToolName: "wx", Parts: []ContentPart{TextPart("chart"), ImagePart("image/png", []byte{9})}},
	}
	return r
}

var goldenWires = []struct {
	name   string
	format Format
	model  string
	req    func() *ChatRequest
	url    string
	body   string
}{
	{"openai", FormatOpenAI, "gpt-4o", goldenReq, "https://x/chat/completions", `{"model":"gpt-4o","messages":[{"role":"system","content":"be brief"},{"role":"user","content":"weather in Paris?"},{"role":"assistant","content":"checking","tool_calls":[{"id":"c1","type":"function","function":{"name":"wx","arguments":"{\"city\":\"Paris\"}"}}]},{"role":"tool","content":"sunny","tool_call_id":"c1"},{"role":"user","content":[{"type":"text","text":"and"},{"type":"image_url","image_url":{"url":"data:image/png;base64,AQID"}},{"type":"text","text":"London?"}]}],"tools":[{"type":"function","function":{"name":"wx","description":"weather","parameters":{"type":"object","properties":{"city":{"type":"string"}}}}}],"max_tokens":256,"temperature":0.5,"top_p":0.9,"stop":["END"]}`},
	{"toolimg-openai", FormatOpenAI, "gpt-4o", goldenToolImgReq, "https://x/chat/completions", `{"model":"gpt-4o","messages":[{"role":"system","content":"be brief"},{"role":"user","content":"look"},{"role":"assistant","content":"","tool_calls":[{"id":"c2","type":"function","function":{"name":"wx","arguments":"{}"}}]},{"role":"tool","content":"chart","tool_call_id":"c2"},{"role":"user","content":[{"type":"text","text":"Images returned by tool call c2:"},{"type":"image_url","image_url":{"url":"data:image/png;base64,CQ=="}}]}],"tools":[{"type":"function","function":{"name":"wx","description":"weather","parameters":{"type":"object","properties":{"city":{"type":"string"}}}}}],"max_tokens":256,"temperature":0.5,"top_p":0.9,"stop":["END"]}`},
	{"responses", FormatOpenAI, "gpt-5.6", goldenReq, "https://x/responses", `{"model":"gpt-5.6","instructions":"be brief","input":[{"role":"user","content":"weather in Paris?"},{"type":"function_call","call_id":"c1","name":"wx","arguments":"{\"city\":\"Paris\"}"},{"role":"assistant","content":"checking"},{"type":"function_call_output","call_id":"c1","output":"sunny"},{"role":"user","content":[{"type":"input_text","text":"and"},{"type":"input_image","image_url":"data:image/png;base64,AQID"},{"type":"input_text","text":"London?"}]}],"tools":[{"type":"function","name":"wx","description":"weather","parameters":{"type":"object","properties":{"city":{"type":"string"}}}}],"max_output_tokens":256,"store":false,"include":["reasoning.encrypted_content"],"reasoning":{"effort":"medium","summary":"auto"}}`},
	{"toolimg-responses", FormatOpenAI, "gpt-5.6", goldenToolImgReq, "https://x/responses", `{"model":"gpt-5.6","instructions":"be brief","input":[{"role":"user","content":"look"},{"type":"function_call","call_id":"c2","name":"wx","arguments":"{}"},{"type":"function_call_output","call_id":"c2","output":[{"type":"input_text","text":"chart"},{"type":"input_image","image_url":"data:image/png;base64,CQ=="}]}],"tools":[{"type":"function","name":"wx","description":"weather","parameters":{"type":"object","properties":{"city":{"type":"string"}}}}],"max_output_tokens":256,"store":false,"include":["reasoning.encrypted_content"],"reasoning":{"effort":"medium","summary":"auto"}}`},
	{"anthropic", FormatAnthropic, "claude", goldenReq, "https://x/v1/messages", `{"model":"claude","max_tokens":256,"messages":[{"role":"user","content":[{"type":"text","text":"weather in Paris?"}]},{"role":"assistant","content":[{"type":"text","text":"checking"},{"type":"tool_use","id":"c1","name":"wx","input":{"city":"Paris"}}]},{"role":"user","content":[{"type":"tool_result","tool_use_id":"c1","content":"sunny"}]},{"role":"user","content":[{"type":"text","text":"and"},{"type":"image","source":{"type":"base64","media_type":"image/png","data":"AQID"}},{"type":"text","text":"London?"}]}],"system":[{"type":"text","text":"be brief"}],"tools":[{"name":"wx","description":"weather","input_schema":{"type":"object","properties":{"city":{"type":"string"}}}}],"temperature":0.5,"top_p":0.9,"stop_sequences":["END"]}`},
	{"toolimg-anthropic", FormatAnthropic, "claude", goldenToolImgReq, "https://x/v1/messages", `{"model":"claude","max_tokens":256,"messages":[{"role":"user","content":[{"type":"text","text":"look"}]},{"role":"assistant","content":[{"type":"tool_use","id":"c2","name":"wx","input":{}}]},{"role":"user","content":[{"type":"tool_result","tool_use_id":"c2","content":[{"type":"text","text":"chart"},{"type":"image","source":{"type":"base64","media_type":"image/png","data":"CQ=="}}]}]}],"system":[{"type":"text","text":"be brief"}],"tools":[{"name":"wx","description":"weather","input_schema":{"type":"object","properties":{"city":{"type":"string"}}}}],"temperature":0.5,"top_p":0.9,"stop_sequences":["END"]}`},
	{"gemini", FormatGemini, "gemini", goldenReq, "https://x/v1beta/models/gemini:generateContent", `{"systemInstruction":{"parts":[{"text":"be brief"}]},"contents":[{"role":"user","parts":[{"text":"weather in Paris?"}]},{"role":"model","parts":[{"text":"checking"},{"functionCall":{"name":"wx","args":{"city":"Paris"}}}]},{"role":"user","parts":[{"functionResponse":{"name":"wx","response":{"result":"sunny"}}}]},{"role":"user","parts":[{"text":"and"},{"inlineData":{"mimeType":"image/png","data":"AQID"}},{"text":"London?"}]}],"tools":[{"functionDeclarations":[{"name":"wx","description":"weather","parametersJsonSchema":{"type":"object","properties":{"city":{"type":"string"}}}}]}],"generationConfig":{"maxOutputTokens":256,"temperature":0.5,"topP":0.9,"stopSequences":["END"]}}`},
	{"toolimg-gemini", FormatGemini, "gemini", goldenToolImgReq, "https://x/v1beta/models/gemini:generateContent", `{"systemInstruction":{"parts":[{"text":"be brief"}]},"contents":[{"role":"user","parts":[{"text":"look"}]},{"role":"model","parts":[{"functionCall":{"name":"wx","args":{}}}]},{"role":"user","parts":[{"functionResponse":{"name":"wx","response":{"result":"chart"}}},{"inlineData":{"mimeType":"image/png","data":"CQ=="}}]}],"tools":[{"functionDeclarations":[{"name":"wx","description":"weather","parametersJsonSchema":{"type":"object","properties":{"city":{"type":"string"}}}}]}],"generationConfig":{"maxOutputTokens":256,"temperature":0.5,"topP":0.9,"stopSequences":["END"]}}`},
}

func TestGoldenWireBodiesUnchanged(t *testing.T) {
	for _, c := range goldenWires {
		t.Run(c.name, func(t *testing.T) {
			pc := newProviderClient(ProviderConfig{ID: string(c.format), Format: c.format, BaseURL: "https://x"}, nil, nil)
			body, url, err := pc.buildChatRequest(c.req(), c.model, false)
			if err != nil {
				t.Fatal(err)
			}
			if url != c.url {
				t.Errorf("url = %s, want %s", url, c.url)
			}
			if string(body) != c.body {
				t.Errorf("wire drifted\n got: %s\nwant: %s", body, c.body)
			}
		})
	}
}

func TestGoldenGeminiEmbedBodyUnchanged(t *testing.T) {
	pc := newProviderClient(ProviderConfig{ID: "gemini", Format: FormatGemini, BaseURL: "https://x"}, nil, nil)
	body, _, err := pc.buildEmbedRequest("text-embedding-004", EmbedRequest{Inputs: []string{"a", "b"}, Dimensions: 8})
	if err != nil {
		t.Fatal(err)
	}
	want := `{"requests":[{"model":"models/text-embedding-004","content":{"parts":[{"text":"a"}]},"outputDimensionality":8},{"model":"models/text-embedding-004","content":{"parts":[{"text":"b"}]},"outputDimensionality":8}]}`
	if string(body) != want {
		t.Errorf("embed wire drifted\n got: %s\nwant: %s", body, want)
	}
}
